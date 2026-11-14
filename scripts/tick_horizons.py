"""Live (updates every closed 1m bar) direction model at horizons 1/5/15/30 min using the 1m features incl. signed-volume imbalance.
Walk-forward by year (train < test year, purged by h minutes via day boundary), test 2025-26, no tuning. Scored on non-overlapping
rows (every h-th minute) so the interval is honest. Also: signed-volume features removed, to measure what they add."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jpmind import bars as B, config, features1m as F  # noqa: E402

df = pd.read_parquet(config.DATA_DIR / "features_1m.parquet").replace([np.inf, -np.inf], np.nan)
cols = F.feature_names(df)
sv = [c for c in cols if c.startswith(("imb_", "cumimb_"))]
et = df.index.tz_convert(B.ET)
day, year = np.asarray(et.date), np.asarray(et.year)
ok = df[cols].notna().all(axis=1).values
c = df.close.astype(float)
g = c.groupby(day)
P = dict(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1)
print(f"{ok.sum():,} usable minutes; signed-volume features: {sv}", flush=True)
for h in (1, 5, 15, 30):
    fwd = np.log(g.shift(-h) / c) * 1e4
    lab = fwd.notna().values & ok & (fwd != 0).values
    y = (fwd > 0).astype(int).values
    nonov = (np.arange(len(df)) % h == 0)
    out = {}
    for name, use in (("all features", cols), ("no signed-volume", [x for x in cols if x not in sv])):
        pr, yy = [], []
        for Y in (2025, 2026):
            tr = lab & (year < Y) & (np.arange(len(df)) % 2 == 0)
            te = lab & (year == Y) & nonov
            if te.sum() == 0:
                continue
            m = lgb.LGBMClassifier(**P).fit(df.loc[tr, use], y[tr])
            pr.append(m.predict_proba(df.loc[te, use])[:, 1])
            yy.append(y[te])
        pr, yy = np.concatenate(pr), np.concatenate(yy)
        n = len(yy)
        q = np.abs(pr - 0.5)
        top = q >= np.quantile(q, 0.9)
        out[name] = (n, yy.mean(), ((pr > .5) == yy).mean(), roc_auc_score(yy, pr), ((pr[top] > .5) == yy[top]).mean(), 1.96 * 0.5 / np.sqrt(n))
    for name, (n, base, acc, auc, top, ci) in out.items():
        print(f"h={h:2d}min  {name:17s} n={n:6d}  always-up {base:.3f}  acc {acc:.3f} (+/-{ci:.3f})  AUC {auc:.3f}  top-10%-confidence acc {top:.3f}", flush=True)
