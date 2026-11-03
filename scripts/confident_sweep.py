"""Sweep the 'top X% most confident next-minute calls' from 10% down to 1%. The best slice is chosen on VALIDATION (2022-24) by gain per call
after costs; the 2025-26 test numbers for that slice are then reported once. Test-best is shown too, labelled as biased.
Usage: uv run python scripts/confident_sweep.py"""
import sys, warnings
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jpmind import config, bars as B, features1m as F  # noqa: E402

df = pd.read_parquet(config.DATA_DIR / "features_1m.parquet").replace([np.inf, -np.inf], np.nan)
cols = F.feature_names(df)
et = df.index.tz_convert(B.ET)
year, day = np.asarray(et.year), np.asarray(et.date)
ok = df[cols].notna().all(axis=1).values & (day <= pd.Timestamp("2026-10-27").date())
ok1 = ok & df.y1.notna().values & (df.y1 != 0).values
y1, y2 = df.y1.values, df.y1.shift(-1).values
ydir = (y1 > 0).astype(int)
P = np.full(len(df), np.nan)
for Y in (2022, 2023, 2024, 2025, 2026):
    tr = np.where(ok1 & (year < Y))[0][::3]
    te = np.where(ok1 & (year == Y))[0]
    m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1, random_state=0)
    P[te] = m.fit(df[cols].iloc[tr], ydir[tr]).predict_proba(df[cols].iloc[te])[:, 1]
val, test = ok1 & np.isin(year, [2022, 2023, 2024]), ok1 & (year >= 2025)
conf = np.abs(P - .5)
sgn = np.where(P > .5, 1, -1)
COSTS = (0, 0.5, 1.0, 2.0)
ndays = len(set(day[test]))
rows = []
for pct in range(10, 0, -1):
    th = np.quantile(conf[val], 1 - pct / 100)
    r = {"pct": pct}
    for k, m in (("val", val), ("test", test)):
        c = m & (conf >= th)
        g = sgn[c] * y1[c]
        r[f"{k}_n"], r[f"{k}_acc"], r[f"{k}_gross"] = int(c.sum()), float(((P[c] > .5) == (y1[c] > 0)).mean()), float(g.mean())
    c = test & (conf >= th) & np.isfinite(y2) & (y2 != 0)
    r["test_gross_delay"] = float((sgn[c] * y2[c]).mean())
    rows.append(r)
R = pd.DataFrame(rows)
print(f"{'top%':>4} | {'val n':>6} {'val acc':>7} {'val gross':>9} | {'test n':>6} {'test acc':>8} {'test gross':>10} {'net@0.5':>8} {'net@1':>7} {'net@2':>7} | {'gross w/ 1-min delay':>20}")
for _, r in R.iterrows():
    print(f"{int(r.pct):>4} | {int(r.val_n):6d} {r.val_acc:7.4f} {r.val_gross:+9.3f} | {int(r.test_n):6d} {r.test_acc:8.4f} {r.test_gross:+10.3f} "
          f"{r.test_gross - .5:+8.3f} {r.test_gross - 1:+7.3f} {r.test_gross - 2:+7.3f} | {r.test_gross_delay:+20.3f}")
b = R.loc[R.val_gross.idxmax()]
print(f"\nBest slice chosen on VALIDATION (highest gross gain per call): top {int(b.pct)}%")
print(f"  test: {int(b.test_n)} calls (~{b.test_n / ndays:.1f}/day), accuracy {b.test_acc:.4f}, gross {b.test_gross:+.3f} bps/call")
for cst in COSTS:
    net = b.test_gross - cst
    tot = (1 + net / 1e4) ** b.test_n * 100
    print(f"  cost {cst:3.1f} bps round trip: net {net:+.3f} bps/call -> 100 becomes {tot:,.2f} over {ndays} test days")
bt = R.loc[R.test_gross.idxmax()]
print(f"\nFor reference, best on TEST (biased, chosen after seeing test): top {int(bt.pct)}%: gross {bt.test_gross:+.3f}, net at 2 bps {bt.test_gross - 2:+.3f} bps/call")
print(f"Break-even cost for the validation-chosen slice: {b.test_gross:.2f} bps round trip; with a 1-minute entry delay: {b.test_gross_delay:.2f} bps")
