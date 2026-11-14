"""Per-candle next-day direction (JPM, 5m, forecast at today's close) with longer history: (L) 10 lagged daily summaries,
(S) per-slot return history, (N) analog days = nearest past days by daily state and what happened at each slot the day after.
Walk-forward by year 2020-26, nothing tuned. CI uses the number of DAYS (candles within a day are dependent)."""
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
from sklearn.metrics import roc_auc_score  # noqa: E402

d = store.load("JPM").copy()
d.index = d.index.tz_convert(B.ET)
d = d.between_time("09:30", "15:59")
c5 = d.close.resample("5min", offset="30min").last().dropna()
v5 = d.volume.resample("5min", offset="30min").sum().reindex(c5.index)
f = pd.DataFrame({"c": c5, "v": v5})
f["date"], f["slot"] = pd.to_datetime(f.index.date), f.index.hour * 12 + f.index.minute // 5 - (9 * 12 + 6)
f = f[(f.slot >= 0) & (f.slot < 78)]
c = f.pivot(index="date", columns="slot", values="c").reindex(index=days, columns=range(78)).ffill(axis=1)
v = f.pivot(index="date", columns="slot", values="v").reindex(index=days, columns=range(78)).fillna(0)
lc = np.log(c.values)
R = np.diff(lc, axis=1) * 1e4                                    # T x 77
T, S = len(days), 77
feat = panel["JPM"][[x for x in panel["JPM"].columns if x != "sym"]]

# ---- daily summaries and their lags
dsum = pd.DataFrame({"dret": C["JPM"].diff() * 1e4, "idret": (lc[:, -1] - lc[:, 0]) * 1e4, "last60": (lc[:, -1] - lc[:, -13]) * 1e4,
                     "rng": (np.nanmax(lc, 1) - np.nanmin(lc, 1)) * 1e4, "vol": np.log1p(v.sum(1).values)}, index=days)
dsum["vol"] = dsum.vol - dsum.vol.rolling(60, min_periods=10).mean()
L = pd.concat([dsum.shift(k).add_suffix(f"_l{k}") for k in range(1, 11)], axis=1)   # lag k relative to the as-of day
# ---- per-slot history (as-of day inclusive)
Rf = pd.DataFrame(R)
sh20, sh60 = Rf.rolling(20, min_periods=10).mean().values, Rf.rolling(60, min_periods=20).mean().values
upf20 = (Rf > 0).astype(float).rolling(20, min_periods=10).mean().values
# ---- analog days
akeys = [k for k in ["ret_1", "ret_5", "ret_20", "id_1", "on_1", "range", "vol_10", "rel_vol", "close_pos", "dist_ma50", "XLF_ret_5"] if k in feat.columns]
Z = feat[akeys].copy()
mu, sd = Z.expanding(60).mean().shift(1), Z.expanding(60).std().shift(1)
Z = ((Z - mu) / sd).clip(-4, 4).fillna(0).values
K = 40
an_mean, an_up, an_cum = np.full((T, S), np.nan), np.full((T, S), np.nan), np.full((T, S), np.nan)
Rn = np.nan_to_num(R)
for a in range(150, T - 1):
    cand = np.arange(0, a)                                       # j <= a-1 : next day j+1 <= a is already known at the as-of close
    dist = ((Z[cand] - Z[a]) ** 2).sum(1)
    nn = cand[np.argsort(dist)[:K]]
    nx = Rn[nn + 1]
    an_mean[a], an_up[a] = nx.mean(0), (nx > 0).mean(0)
    an_cum[a] = np.cumsum(nx, 1).mean(0)

a_idx = np.arange(T - 1)
t_idx = a_idx + 1
base = pd.DataFrame(np.repeat(feat.iloc[a_idx].values, S, axis=0), columns=feat.columns)
base["slot"] = np.tile(np.arange(1, 78), T - 1)
base["tdow"] = np.repeat(days[t_idx].dayofweek.values, S)
Lx = pd.DataFrame(np.repeat(L.iloc[a_idx].values, S, axis=0), columns=L.columns)
Sx = pd.DataFrame({"sh20": sh20[a_idx].ravel(), "sh60": sh60[a_idx].ravel(), "upf20": upf20[a_idx].ravel(), "ysl": R[a_idx].ravel()})
Nx = pd.DataFrame({"an_mean": an_mean[a_idx].ravel(), "an_up": an_up[a_idx].ravel(), "an_cum": an_cum[a_idx].ravel()})
y = (R[t_idx] > 0).ravel().astype(int)
yret = R[t_idx].ravel()
year = np.repeat(days[t_idx].year.values, S)
dayid = np.repeat(t_idx, S)
ok = np.isfinite(yret) & (yret != 0) & (year >= 2017)
sets = {"base": [base], "base+L (10 daily lags)": [base, Lx], "base+S (slot history)": [base, Sx], "base+N (analog days)": [base, Nx],
        "all (L+S+N)": [base, Lx, Sx, Nx]}
Pm = dict(n_estimators=200, learning_rate=0.04, num_leaves=15, min_child_samples=800, subsample=0.7, subsample_freq=1,
          colsample_bytree=0.6, reg_lambda=10, verbose=-1)
print(f"{ok.sum()} candle rows; analog features: {akeys}; K={K}\n", flush=True)
print("model                        |  VAL 2020-23: always-up  acc   AUC   |  TEST 2024-26: always-up  acc   AUC   (acc 95% CI from #days)")
for name, parts in sets.items():
    M = pd.concat(parts, axis=1)[ok].reset_index(drop=True)
    yy, yr, dd = y[ok], year[ok], dayid[ok]
    pred = np.full(len(M), np.nan)
    for Y in range(2020, 2027):
        tr, te = (yr < Y), (yr == Y)
        pred[te] = lgb.LGBMClassifier(**Pm).fit(M[tr], yy[tr]).predict_proba(M[te])[:, 1]
    row = [name.ljust(28)]
    for yrs in ((2020, 2024), (2024, 2027)):
        mk = (yr >= yrs[0]) & (yr < yrs[1])
        hit = pd.Series(((pred[mk] > .5) == yy[mk]).astype(float)).groupby(dd[mk]).mean()
        ci = 1.96 * hit.std() / np.sqrt(len(hit))
        row.append(f"{yy[mk].mean():.3f}  {((pred[mk] > .5) == yy[mk]).mean():.3f}+/-{ci:.3f}  {roc_auc_score(yy[mk], pred[mk]):.3f}")
    print(" | ".join(row), flush=True)
