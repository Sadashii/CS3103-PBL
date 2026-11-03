"""Next-day / next-week realized-volatility forecast with a calibrated 90% band.
Walk-forward 2020-2026 (training labels purged by the horizon). Baselines: naive (yesterday) and HAR (Corsi 2009).
Pooled panel over JPM BAC C WFC GS MS; reports JPM and all-bank results.
Variant A uses only past information. Variant B adds calendar items that are scheduled in advance (FOMC tomorrow, earnings tomorrow)."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import bars as B, config, features, store  # noqa: E402

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
DAY = 86400 * 10**9


def daily_rv(sym):
    j = features.bars5(store.load(sym))
    et = j.index.tz_convert(B.ET)
    r = np.log(j.close).diff()
    d = np.asarray(et.date)
    r = r.where(d == pd.Series(d, index=j.index).shift(1).values)          # no overnight return inside RV
    g = (r ** 2).groupby(d)
    out = pd.DataFrame({"rv": g.sum(), "n": g.count()})
    out = out[out.n >= 60]
    hi, lo = j.high.groupby(d).max(), j.low.groupby(d).min()
    op, cl = j.open.groupby(d).first(), j.close.groupby(d).last()
    out["rng"] = np.log(hi / lo).reindex(out.index)
    out["gap"] = np.log(op / cl.shift(1)).reindex(out.index)
    out.index = pd.to_datetime(out.index)
    return out


R = {s: daily_rv(s) for s in BANKS + ["SPY"]}
days = R["JPM"].index
R = {s: r.reindex(days) for s, r in R.items()}
LV = pd.DataFrame({s: np.log(r.rv) for s, r in R.items()})
eod = pd.Series([pd.Timestamp(d).tz_localize(B.ET) + pd.Timedelta(hours=16) for d in days], index=days).dt.tz_convert("UTC")
close_ns = eod.astype("int64").values
ev = pd.read_parquet(config.DATA_DIR / "raw" / "events.parquet")
fomc = np.sort(ev[ev.kind == "fomc"].time.astype("int64").values)


def next_in_days(t, within_days=1.6):
    nx = np.searchsorted(t, close_ns, side="right")
    gap = np.where(nx < len(t), (t[np.clip(nx, 0, len(t) - 1)] - close_ns) / DAY, 999)
    return (gap <= within_days).astype(float)


def news_count(sym):
    f = config.DATA_DIR / f"news_fin_{sym}.parquet"
    if not f.exists():
        return np.zeros(len(days)), np.zeros(len(days))
    n = pd.read_parquet(f).sort_values("created_at")
    t = n.created_at.astype("int64").values
    hi = np.searchsorted(t, close_ns, side="right")
    c1 = hi - np.searchsorted(t, close_ns - DAY, side="right")
    c10 = hi - np.searchsorted(t, close_ns - 10 * DAY, side="right")
    return c1, c1 / (c10 / 10 + 0.5)


def build(sym):
    lv = LV[sym]
    X = pd.DataFrame(index=days)
    X["lv_d"] = lv
    X["lv_w"] = np.log(R[sym].rv.rolling(5).mean())
    X["lv_m"] = np.log(R[sym].rv.rolling(22).mean())
    X["lv_q"] = np.log(R[sym].rv.rolling(66).mean())
    X["lrng"] = np.log(R[sym].rng)
    X["agap"] = R[sym].gap.abs()
    X["spy_lv"] = LV["SPY"]
    X["spy_lv_w"] = np.log(R["SPY"].rv.rolling(5).mean())
    X["peer_lv"] = LV[[p for p in BANKS if p != sym]].mean(axis=1)
    X["dow"] = days.dayofweek
    X["news_1d"], X["news_rel"] = news_count(sym)
    evt = ev[ev.kind == f"{sym.lower()}_earnings"].time.astype("int64").values
    X["_earn_tom"] = next_in_days(np.sort(evt))
    X["_fomc_tom"] = next_in_days(fomc)
    X["sym"] = sym
    return X.replace([np.inf, -np.inf], np.nan)


panel = {s: build(s) for s in BANKS}
base = ["lv_d", "lv_w", "lv_m", "lv_q", "lrng", "agap", "spy_lv", "spy_lv_w", "peer_lv", "dow", "news_1d", "news_rel"]
cal = base + ["_earn_tom", "_fomc_tom"]
har = ["lv_d", "lv_w", "lv_m"]

print("horizon | model | JPM R2 | JPM QLIKE | ALL R2 | JPM 90% band coverage (conformal)   (R2 on log realized variance, walk-forward 2020-2026)")
for h, name in ((1, "next day"), (5, "next week")):
    tgt = {s: np.log(R[s].rv.rolling(h).mean().shift(-h)) for s in BANKS}
    store_p = {k: [] for k in ("naive", "har", "A", "B")}
    ys, syms, lo_hi = [], [], []
    for Y in range(2020, 2027):
        te_idx = np.where(days.year == Y)[0]
        if len(te_idx) < 20:
            continue
        tr_idx = np.arange(0, max(te_idx[0] - h, 0))
        tr = pd.concat([panel[s].iloc[tr_idx].assign(y=tgt[s].iloc[tr_idx]) for s in BANKS]).dropna(subset=cal + ["y"])
        te = pd.concat([panel[s].iloc[te_idx].assign(y=tgt[s].iloc[te_idx]) for s in BANKS]).dropna(subset=cal + ["y"])
        har_m = Ridge(alpha=1.0).fit(tr[har], tr.y)
        mk = lambda c: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7,
                                         subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1).fit(tr[c], tr.y)
        mA, mB = mk(base), mk(cal)
        store_p["naive"] += list(te.lv_d if h == 1 else te.lv_w)
        store_p["har"] += list(har_m.predict(te[har]))
        store_p["A"] += list(mA.predict(te[base]))
        store_p["B"] += list(mB.predict(te[cal]))
        ys += list(te.y)
        syms += list(te.sym)
        # split-conformal 90% band: fit on the first 80% of the training days, take residual quantiles on the last 20%
        # (later than the fit, earlier than the test year), then use the full-data model's point forecast
        cut = days[int(len(tr_idx) * 0.8)]
        fit, cal_ = tr[tr.index < cut], tr[tr.index >= cut]
        m80 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7,
                                subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1).fit(fit[base], fit.y)
        q = np.quantile(cal_.y.values - m80.predict(cal_[base]), [0.05, 0.95])
        lo_hi += [(p + q[0], p + q[1]) for p in mA.predict(te[base])]
    y, sy = np.array(ys), np.array(syms)
    m = sy == "JPM"
    for k in ("naive", "har", "A", "B"):
        p = np.array(store_p[k])
        r2 = lambda mask: 1 - ((y[mask] - p[mask]) ** 2).sum() / ((y[mask] - y[mask].mean()) ** 2).sum()
        ql = np.mean(np.exp(y[m]) / np.exp(p[m]) - (y[m] - p[m]) - 1)               # QLIKE (lower is better), JPM
        cov = ""
        if k == "A":
            lh = np.array(lo_hi)
            cov = f"{((y[m] >= lh[m, 0]) & (y[m] <= lh[m, 1])).mean():.3f}"
        label = {"naive": "naive (yesterday/last week)", "har": "HAR baseline", "A": "LightGBM, past info only", "B": "LightGBM + scheduled calendar"}[k]
        print(f"{name:10s}| {label:30s}| {r2(m):.3f} | {ql:.3f} | {r2(np.ones(len(y), bool)):.3f} | {cov}", flush=True)
