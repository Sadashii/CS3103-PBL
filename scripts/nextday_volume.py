"""Does predicted next-day volume improve the next-day per-candle range forecast? JPM, 5m candles, forecast made at today's close.
Stage 1: predict log volume of every 5m slot tomorrow (walk-forward). Stage 2: 5%/95% quantiles of every candle return tomorrow
with (A) no volume info, (A2) volume history only, (B) + predicted volume, (C) + actual volume (oracle ceiling, not usable live).
Reports pinball loss and 90% coverage on 2020-23 and 2024-26. Nothing is tuned."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "scripts" / "daily_panel.py").read_text()
src = src[: src.index("print(len(cols)")]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--no-news", "--no-fund"]
exec(compile(src, "daily_panel_prefix", "exec"))
import lightgbm as lgb  # noqa: E402

d = store.load("JPM").copy()
d.index = d.index.tz_convert(B.ET)
d = d.between_time("09:30", "15:59")
c5 = d.close.resample("5min", offset="30min").last().dropna()
v5 = d.volume.resample("5min", offset="30min").sum().reindex(c5.index)
f = pd.DataFrame({"c": c5, "v": v5})
f["date"], f["slot"] = pd.to_datetime(f.index.date), f.index.hour * 12 + f.index.minute // 5 - (9 * 12 + 6)
f = f[(f.slot >= 0) & (f.slot < 78)]
c = f.pivot(index="date", columns="slot", values="c").reindex(index=days, columns=range(78)).ffill(axis=1)
v = f.pivot(index="date", columns="slot", values="v").reindex(index=days, columns=range(78))
lv = np.log1p(v.fillna(0).values)
lc = np.log(c.values)
R = np.diff(lc, axis=1) * 1e4                       # candle returns, slots 1..77
T, S = len(days), 77
slots = np.arange(1, 78)
feat = panel["JPM"][[x for x in panel["JPM"].columns if x != "sym"]].copy()
m20 = pd.DataFrame(lv).rolling(20, min_periods=5).mean().values
tot = lv.sum(1)
a = np.arange(T - 1)                                 # as-of day; target day is a+1
t = a + 1
X = pd.DataFrame(np.repeat(feat.iloc[a].values, S, axis=0), columns=feat.columns)
X["slot"] = np.tile(slots, T - 1)
X["tdow"] = np.repeat(days[t].dayofweek.values, S)
VH = pd.DataFrame({"lagv": lv[a][:, slots].ravel(), "m20": m20[a][:, slots].ravel(), "lagtot": np.repeat(tot[a], S)})
ylv = lv[t][:, slots].ravel()
yr_ = R[t].ravel()
ok = (v.values[t][:, slots].ravel() > 0) & np.isfinite(yr_) & np.repeat(days[t].year.values >= 2017, S)
year = np.repeat(days[t].year.values, S)
X, VH, ylv, yr_, year = X[ok].reset_index(drop=True), VH[ok].reset_index(drop=True), ylv[ok], yr_[ok], year[ok]
print(f"{len(X)} candle rows, {X.slot.nunique()} slots", flush=True)

P = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
# ---- Stage 1: volume
XV = pd.concat([X, VH], axis=1)
pv = np.full(len(X), np.nan)
for Y in range(2018, 2027):
    tr, te = year < Y, year == Y
    if tr.sum() < 20000 or te.sum() == 0:
        continue
    pv[te] = lgb.LGBMRegressor(**P).fit(XV[tr], ylv[tr]).predict(XV[te])
print("\nStage 1: predict log volume of each slot tomorrow")
for nm, yrs in (("2018-23", range(2018, 2024)), ("2024-26", range(2024, 2027))):
    mk = np.isin(year, list(yrs)) & np.isfinite(pv)
    base = XV.m20.values[mk]
    var = ((ylv[mk] - ylv[mk].mean()) ** 2).mean()
    print(f"  {nm}: R2 model {1 - ((ylv[mk] - pv[mk]) ** 2).mean() / var:.3f}   R2 trailing-20d slot mean {1 - ((ylv[mk] - base) ** 2).mean() / var:.3f}")

# ---- Stage 2: 5%/95% candle-return quantiles
sets = {"A  no volume info": X, "A2 volume history": pd.concat([X, VH], axis=1),
        "B  + predicted volume": pd.concat([X, VH, pd.Series(pv, name="pv")], axis=1),
        "C  + ACTUAL volume (oracle)": pd.concat([X, VH, pd.Series(ylv, name="av")], axis=1)}
pin = lambda y, q, a_: np.mean(np.maximum(a_ * (y - q), (a_ - 1) * (y - q)))
res = {k: {"lo": np.full(len(X), np.nan), "hi": np.full(len(X), np.nan)} for k in sets}
for Y in range(2020, 2027):
    te = year == Y
    tr = (year >= 2018) & (year < Y) & np.isfinite(pv)
    for k, M in sets.items():
        for nm, al in (("lo", .05), ("hi", .95)):
            res[k][nm][te] = lgb.LGBMRegressor(objective="quantile", alpha=al, **P).fit(M[tr], yr_[tr]).predict(M[te])
print("\nStage 2: 90% band for every 5m candle tomorrow (pinball = mean of 5% and 95% quantile loss; lower is better)")
for nm, yrs in (("val 2020-23", range(2020, 2024)), ("test 2024-26", range(2024, 2027))):
    mk = np.isin(year, list(yrs))
    print(f"  {nm}: {mk.sum()} candles")
    for k in sets:
        lo, hi = res[k]["lo"][mk], res[k]["hi"][mk]
        y = yr_[mk]
        print(f"    {k:28s} pinball {(pin(y, lo, .05) + pin(y, hi, .95)) / 2:.3f}   coverage {((y >= lo) & (y <= hi)).mean():.3f}   mean width {np.mean(hi - lo):.1f} bps")
