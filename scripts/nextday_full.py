"""Whole-next-day forecast made at today's close: (A) one Up/Down call for tomorrow, (B) an Up/Down call for every 5m candle
of tomorrow's session. Walk-forward by year 2020-26, purged, JPM is scored.
A: own pooled-bank LR / LGB, equal-weight and correlation-weighted peer averaging of the per-bank predictions.
B: LGB on today's daily features + slot of day -> sign of each 5m candle return tomorrow."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "scripts" / "daily_panel.py").read_text()
src = src[: src.index("print(len(cols)")]      # reuse the feature builder; skip its own experiment loop
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--no-news"] + sys.argv[1:]
exec(compile(src, "daily_panel_prefix", "exec"))  # defines days, C, panel, cols, BANKS, B, store

import lightgbm as lgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

YEARS = range(2020, 2027)
ys = {s: (C[s].shift(-1) - C[s] > 0).astype(float).where(C[s].shift(-1).notna()) for s in BANKS}
ret = C.diff()
corr = {s: ret["JPM"].rolling(252).corr(ret[s]) for s in BANKS}   # causal: uses returns up to today's close

# ---------------- A: one call for tomorrow ----------------
P = {k: {s: pd.Series(np.nan, index=days) for s in BANKS} for k in ("lr", "gb")}
for Y in YEARS:
    te_idx = np.where(days.year == Y)[0]
    tr_idx = np.arange(0, max(te_idx[0] - 1, 0))
    tr = pd.concat([panel[s].iloc[tr_idx].assign(y=ys[s].iloc[tr_idx]) for s in BANKS]).dropna(subset=["y"])
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.03, max_iter=1000)).fit(tr[cols], tr.y)
    gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7,
                            subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1).fit(tr[cols], tr.y)
    for s in BANKS:
        Xs = panel[s].iloc[te_idx].reindex(columns=cols)
        P["lr"][s].iloc[te_idx] = lr.predict_proba(Xs)[:, 1]
        P["gb"][s].iloc[te_idx] = gb.predict_proba(Xs)[:, 1]

ok = (days.year >= 2020) & ys["JPM"].notna().values
y = ys["JPM"][ok].astype(int).values
n = len(y)
print(f"A. next-day close-to-close direction, JPM, {n} test days (2020-26); 95% interval +/-{1.96 * 0.5 / np.sqrt(n):.3f}")
print(f"   always-up {y.mean():.3f}")
cand = {}
for k in ("lr", "gb"):
    own = P[k]["JPM"][ok].values
    pe = pd.DataFrame({s: P[k][s] for s in BANKS})[ok]
    w = pd.DataFrame({s: corr[s] for s in BANKS})[ok].clip(lower=0).fillna(0)
    cand[f"{k} own"] = own
    cand[f"{k} peers equal"] = pe.mean(axis=1).values
    cand[f"{k} peers corr-weighted"] = ((pe * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)).values
cand["lr+gb corr-weighted"] = (cand["lr peers corr-weighted"] + cand["gb peers corr-weighted"]) / 2
for name, p in cand.items():
    m = ~np.isnan(p)
    print(f"   {name:24s} acc {((p[m] > .5) == y[m]).mean():.3f}   mean P(up) {np.nanmean(p):.3f}")
pbest = cand["lr+gb corr-weighted"]
q = np.abs(pbest - 0.5)
for frac in (0.5, 0.2):
    t = np.nanquantile(q, 1 - frac)
    m = q >= t
    print(f"   most confident {int(frac * 100)}% of days (lr+gb corr-weighted): acc {((pbest[m] > .5) == y[m]).mean():.3f} on {m.sum()} days")
dayp = pd.Series(pbest, index=days[ok])

# ---------------- B: every 5m candle of tomorrow ----------------
d1 = store.load("JPM").copy()
d1.index = d1.index.tz_convert(B.ET)
d1 = d1.between_time("09:30", "15:59")
c5 = d1.close.resample("5min", offset="30min").last().dropna()
c5 = c5[c5.index.time >= pd.Timestamp("09:30").time()]
df = pd.DataFrame({"c": c5})
df["date"] = pd.to_datetime(df.index.date)
df["slot"] = df.groupby("date").cumcount()
df["r"] = np.log(df.c).diff()
df.loc[df.slot == 0, "r"] = np.nan           # first candle: gap from the previous close is not a 5m candle
df = df.dropna(subset=["r"])
df["up"] = (df.r > 0).astype(int)
prev = pd.Series(days[:-1], index=days[1:])  # the trading day whose close we forecast from
df["asof"] = df.date.map(prev)
df = df.dropna(subset=["asof"])
feat = panel["JPM"][cols]
Xb = feat.reindex(df["asof"].values).reset_index(drop=True)
Xb["slot"] = df.slot.values
Xb["dayp"] = dayp.reindex(df["asof"].values).values   # tomorrow's day-call probability (only exists from 2020; NaN earlier)
Xb = Xb.drop(columns="dayp")                       # kept simple: day features + slot only (no stacked dependency)
yb, yr = df.up.values, df.date.dt.year.values
pred = np.full(len(df), np.nan)
for Y in YEARS:
    tr, te = yr < Y, yr == Y
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1).fit(Xb[tr], yb[tr])
    pred[te] = gbm.predict_proba(Xb[te])[:, 1]
m = ~np.isnan(pred)
print(f"\nB. per-candle direction for the whole of tomorrow, JPM, {m.sum()} candles (2020-26)")
print(f"   always-up {yb[m].mean():.3f}   model {((pred[m] > .5) == yb[m]).mean():.3f}   "
      f"95% interval +/-{1.96 * 0.5 / np.sqrt(m.sum() / 78):.3f} (effective n = days*~1 independent path)")
tm = df[m].assign(p=pred[m])
sl = tm.groupby("slot").apply(lambda g: ((g.p > .5) == g.up).mean())
print("   accuracy by slot: min %.3f max %.3f; first 3 slots %s" % (sl.min(), sl.max(), np.round(sl.values[:3], 3)))
dayacc = tm.groupby("date").apply(lambda g: ((g.p > .5) == g.up).mean())
print(f"   per-day average candle accuracy {dayacc.mean():.3f} (std {dayacc.std():.3f} across days)")
