"""Run the same next-minute 'confident call' pipeline on several stocks, one after another, and compare.
Per stock: LightGBM walk-forward 2022-26 (train only on earlier years), confidence cutoffs set on 2022-24 validation, scored on 2025-26.
Reports all-minute accuracy, top 10/5/1% accuracy and gross gain per call, the same with a one-minute entry delay, and a liquidity/cost proxy.
Usage: uv run python scripts/stock_compare.py BAC C WFC MS AMD XOM V JNJ HD      (symbols whose data is missing are skipped)"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jpmind import bars as B, config, features1m as F, store  # noqa: E402

CONTEXT = ["SPY", "XLF", "JPM", "BAC", "C", "WFC", "MS"]


def run(sym):
    try:
        target = store.load(sym)
    except Exception as e:
        return None, f"{sym}: no data ({type(e).__name__})"
    if len(target) < 500_000:
        return None, f"{sym}: only {len(target):,} bars, skipped"
    frames = {s: store.load(s) for s in CONTEXT if s != sym}
    frames[sym] = target
    peers = [s for s in ["JPM", "BAC", "C", "WFC", "MS"] if s != sym]
    df = F.build(frames, sym=sym, peers=peers).replace([np.inf, -np.inf], np.nan)
    cols = F.feature_names(df)
    et = df.index.tz_convert(B.ET)
    year, day = np.asarray(et.year), np.asarray(et.date)
    ok = df[cols].notna().all(axis=1).values & (day <= pd.Timestamp("2026-10-27").date())
    ok1 = ok & df.y1.notna().values & (df.y1 != 0).values
    y1, y2 = df.y1.values, df.y1.shift(-1).values
    yd = (y1 > 0).astype(int)
    P = np.full(len(df), np.nan)
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr = np.where(ok1 & (year < Y))[0][::3]
        te = np.where(ok1 & (year == Y))[0]
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1, random_state=0)
        P[te] = m.fit(df[cols].iloc[tr], yd[tr]).predict_proba(df[cols].iloc[te])[:, 1]
    val, test = ok1 & np.isin(year, [2022, 2023, 2024]), ok1 & (year >= 2025)
    conf, sgn = np.abs(P - .5), np.where(P > .5, 1, -1)
    px_med = float(df.close[test].median())
    spread_proxy = float(((df.close.rolling(1).max()).pipe(lambda c: 0 * c)).mean())            # placeholder, replaced below
    rng_bps = float(np.nanmedian(((frames[sym].high - frames[sym].low) / frames[sym].close * 1e4)[frames[sym].index >= "2025-01-01"]))
    dollar_vol = float((frames[sym].close * frames[sym].volume)[frames[sym].index >= "2025-01-01"].groupby(frames[sym].index[frames[sym].index >= "2025-01-01"].date).sum().median() / 1e6)
    out = {"sym": sym, "n_test": int(test.sum()), "base_up": float(yd[test].mean()), "acc_all": float(((P[test] > .5) == (y1[test] > 0)).mean()),
           "median_1m_range_bps": rng_bps, "median_IEX_dollar_vol_M_per_day": dollar_vol}
    best = None
    for pct in (10, 5, 1):
        th = np.quantile(conf[val], 1 - pct / 100)
        for k, m_ in (("val", val), ("test", test)):
            c = m_ & (conf >= th)
            out[f"{k}_acc_{pct}"] = float(((P[c] > .5) == (y1[c] > 0)).mean())
            out[f"{k}_gross_{pct}"] = float((sgn[c] * y1[c]).mean())
            out[f"{k}_n_{pct}"] = int(c.sum())
        c = test & (conf >= th) & np.isfinite(y2) & (y2 != 0)
        out[f"delay_gross_{pct}"] = float((sgn[c] * y2[c]).mean())
    return out, None


if __name__ == "__main__":
    syms = sys.argv[1:] or ["BAC", "C", "WFC", "MS"]
    rows = []
    for s in syms:
        r, msg = run(s)
        if r is None:
            print(msg, flush=True)
            continue
        rows.append(r)
        print(f"{s:5s} all-minute acc {r['acc_all']:.4f} (up-rate {r['base_up']:.3f}) | "
              + " | ".join(f"top{p}%: val {r[f'val_acc_{p}']:.3f} test {r[f'test_acc_{p}']:.3f} gross {r[f'test_gross_{p}']:+.2f}bps delay {r[f'delay_gross_{p}']:+.2f}" for p in (10, 5, 1))
              + f" | 1m range {r['median_1m_range_bps']:.1f}bps", flush=True)
    if rows:
        T = pd.DataFrame(rows).set_index("sym")
        T.to_csv(ROOT / "docs" / "stock_compare.csv")
        print("\nRanked by validation-chosen best slice (highest validation gross gain per call), then its test gross gain:")
        for s, r in T.iterrows():
            pb = max((10, 5, 1), key=lambda p: r[f"val_gross_{p}"])
            print(f"  {s:5s} best slice on validation: top {pb}% | test acc {r[f'test_acc_{pb}']:.4f} | test gross {r[f'test_gross_{pb}']:+.3f} bps | with 1-min delay {r[f'delay_gross_{pb}']:+.3f} bps")
