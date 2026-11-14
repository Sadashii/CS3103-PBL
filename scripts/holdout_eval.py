"""Out-of-sample check on every stored session after the training cutoff (days the models never saw).

Scores each day exactly as the dashboard's signal log does: the 3-class call on every closed 5m candle against
always-Neutral / always-Up / RSI rule / persistence, the activity call against persistence, the big-move flag, and
the two fixed session outlooks (10:00, 12:30) against the close.

Usage: uv run python scripts/holdout_eval.py     -> prints a table, writes docs/holdout_results.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpmind import bars as B, config, model as M, store, tickmodel as TM


def main():
    frames = {s: store.load(s) for s in [config.SYMBOL, *config.CONTEXT_SYMBOLS]}
    pred, tick = M.Predictor(), TM.TickModel()
    c3 = pred.c3
    K, tau = c3["horizon_bars"], c3["tau_bps"]
    cutoff = pd.Timestamp(pred.c3["cutoff"]).date()
    days = sorted(d for d in set(frames[config.SYMBOL].index.tz_convert(B.ET).date) if d > cutoff)
    cls = lambda x: "Up" if x > tau else "Down" if x < -tau else "Neutral"
    rows = []
    for day in days:
        end = pd.Timestamp(day, tz=B.ET).tz_convert("UTC") + pd.Timedelta(days=1)
        fr = {s: d[(d.index >= end - pd.Timedelta(days=80)) & (d.index < end)] for s, d in frames.items()}
        t = pred.day_table(fr, day)
        g = t[(t.call != "No call") & t.yk.notna()]
        if g.empty:
            continue
        act = g.yk.map(pred.realised)
        r = {"day": str(day), "candles": len(g), "model": (g.call == act).mean(),
             "always_neutral": (act == "Neutral").mean(), "always_up": (act == "Up").mean(),
             "rsi_rule": (np.where(g.rsi < 30, "Up", np.where(g.rsi > 70, "Down", "Neutral")) == act).mean(),
             "persistence": (g.pk.map(cls) == act).mean()}
        if "act" in g:
            aa = g.yk.map(pred.activity_of)
            r.update(activity=(g.act == aa).mean(), activity_persist=(g.pk.map(pred.activity_of) == aa).mean())
        big = g.yk.abs() > c3["tau_big_bps"]
        r.update(big_flags=int(g.big.sum()), big_flag_hit=float(big[g.big].mean()) if g.big.any() else np.nan, big_base=float(big.mean()))
        mv = g[g.call.isin(["Up", "Down"])]
        r["dir_hit"] = float(((mv.call == "Up") == (mv.yk > 0)).mean()) if len(mv) else np.nan
        r["net_bps"] = float((np.where(mv.call == "Up", 1, -1) * mv.yk - config.COST_BPS).sum())
        # session outlooks, frozen at 10:00 / 12:30 and graded at the close
        tr = tick.day_rows(fr, day)
        rth = bars_d_rth = fr[config.SYMBOL].tz_convert(B.ET).between_time("09:30", "15:59")
        rth = rth[rth.index.date == day]
        r["open"] = float(rth.open.iloc[0]) if len(rth) else np.nan
        bars_d = fr[config.SYMBOL]
        close = float(bars_d[bars_d.index <= pd.Timestamp(f"{day} 15:59", tz=B.ET).tz_convert("UTC")].close.iloc[-1])
        for name, hhmm in (("am", "09:59"), ("pm", "12:29")):
            o = tick.at(tr, pd.Timestamp(f"{day} {hhmm}", tz=B.ET).tz_convert("UTC"))
            if o:
                d = o["to_close"]
                r[f"{name}_in50"] = d["band50"]["lower"] <= close <= d["band50"]["upper"]
                r[f"{name}_in90"] = d["band90"]["lower"] <= close <= d["band90"]["upper"]
                # predicted direction for the rest of the day vs what happened
                r[f"{name}_price"] = float(o["price"])
                r[f"{name}_median"] = float(d["median_close"])
                r[f"{name}_p_up"] = float(d["p_close_above_now"])
                r[f"{name}_lo90"], r[f"{name}_hi90"] = float(d["band90"]["lower"]), float(d["band90"]["upper"])
                r[f"{name}_lean_ok"] = (close > o["price"]) == (d["p_close_above_now"] >= 0.5)
        r["close"] = close
        rows.append(r)
        print(f"{day}  model {r['model']:.3f}  neutral {r['always_neutral']:.3f}  up {r['always_up']:.3f}  rsi {r['rsi_rule']:.3f}  "
              f"persist {r['persistence']:.3f}  activity {r.get('activity', np.nan):.3f} vs {r.get('activity_persist', np.nan):.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = config.ROOT / "docs" / "holdout_results.csv"
    df.to_csv(out, index=False)
    w = df.candles
    avg = lambda c: float(np.average(df[c], weights=w))
    print(f"\n{len(df)} unseen sessions ({df.day.iloc[0]} .. {df.day.iloc[-1]}), {int(w.sum())} graded candles, {K * 5}-min horizon")
    for c in ["model", "always_neutral", "always_up", "rsi_rule", "persistence"]:
        beats = (df.model > df[c]).mean() if c != "model" else np.nan
        print(f"  {c:15s} {avg(c):.3f}" + ("" if c == "model" else f"   model beats it on {beats:.0%} of days"))
    if "activity" in df:
        print(f"  activity        {avg('activity'):.3f} vs persistence {avg('activity_persist'):.3f}, beats it on {(df.activity > df.activity_persist).mean():.0%} of days")
    print(f"  direction on Up/Down calls {np.nanmean(df.dir_hit):.3f}; net {df.net_bps.sum():.1f} bps total after {config.COST_BPS} bps/call")
    for n in ("am", "pm"):
        if f"{n}_in90" in df:
            print(f"  outlook {n}: direction right {df[f'{n}_lean_ok'].mean():.0%}, close in 50% band {df[f'{n}_in50'].mean():.0%}, in 90% band {df[f'{n}_in90'].mean():.0%} (n={df[f'{n}_in90'].notna().sum()})")
    print("wrote", out)


if __name__ == "__main__":
    main()
