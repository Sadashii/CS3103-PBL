"""Robustness tests for the 'confident next-minute call' result. Rule fixed in advance: call only when |p-0.5| is in the top 10%
of the 2022-24 validation predictions (absolute cutoff), score on 2025-26. Nothing below is used to change the rule.
Usage: uv run python scripts/confident_calls_tests.py"""
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
slot = df["slot"].values
ok = df[cols].notna().all(axis=1).values & (day <= pd.Timestamp("2026-10-27").date())
ok1 = ok & df.y1.notna().values & (df.y1 != 0).values
y1 = df.y1.values
y2 = df.y1.shift(-1).values                                    # return of the NEXT minute after that (one-minute execution delay)
ydir = (y1 > 0).astype(int)


def oof(seed=0, shuffle=False, stride=3):
    P = np.full(len(df), np.nan)
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr = np.where(ok1 & (year < Y))[0][::stride]
        te = np.where(ok1 & (year == Y))[0]
        y = ydir[tr].copy()
        if shuffle:
            y = np.random.default_rng(seed).permutation(y)
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1, random_state=seed)
        P[te] = m.fit(df[cols].iloc[tr], y).predict_proba(df[cols].iloc[te])[:, 1]
    return P


val = ok1 & np.isin(year, [2022, 2023, 2024])
test = ok1 & (year >= 2025)


def report(P, label, y_eval=y1, mask_extra=None):
    conf = np.abs(P - .5)
    th = np.quantile(conf[val], 0.90)
    call = (conf >= th) & test & np.isfinite(y_eval) & (y_eval != 0)
    if mask_extra is not None:
        call &= mask_extra
    acc = ((P[call] > .5) == (y_eval[call] > 0)).mean()
    g = (np.where(P[call] > .5, 1, -1) * y_eval[call]).mean()
    print(f"{label:46s} calls {call.sum():6d} | acc {acc:.4f} | gross {g:+.3f} bps/call")
    return call, th


P = oof(0)
print("Rule: call when |p-0.5| >= top-10% cutoff of 2022-24 validation; scored on 2025-26.\n")
call, th = report(P, "1. headline (trade at the signal bar's close)")
print(f"   cutoff = {th:.4f}")

# 2. by year, SAME absolute cutoff
print("\n2. same absolute cutoff, by year:")
for Y in (2022, 2023, 2024, 2025, 2026):
    m = ok1 & (year == Y) & (np.abs(P - .5) >= th)
    print(f"   {Y}: calls {m.sum():6d} | acc {((P[m] > .5) == (y1[m] > 0)).mean():.4f}")

# 3. day-level bootstrap CI
d = pd.DataFrame({"day": day[call], "hit": ((P[call] > .5) == (y1[call] > 0)).astype(int)})
g = d.groupby("day").hit.agg(["sum", "count"])
rng = np.random.default_rng(0)
bs = [(lambda s: g["sum"].values[s].sum() / g["count"].values[s].sum())(rng.integers(0, len(g), len(g))) for _ in range(2000)]
print(f"\n3. day-block bootstrap 95% CI for the confident-call accuracy: {np.quantile(bs, .025):.4f} - {np.quantile(bs, .975):.4f}  ({len(g)} days)")

# 4. one-minute execution delay: signal at bar t, enter at bar t+1's close, exit at t+2's close
report(P, "4. ONE-MINUTE DELAY (enter after the next bar)", y_eval=y2)

# 5. time of day
print("\n5. time of day (confident calls only):")
for a, b, nm in ((0, 15, "09:30-09:45"), (15, 60, "09:45-10:30"), (60, 330, "10:30-15:00"), (330, 375, "15:00-15:45"), (375, 390, "15:45-16:00")):
    m = call & (slot >= a) & (slot < b)
    print(f"   {nm}: calls {m.sum():6d} | acc {((P[m] > .5) == (y1[m] > 0)).mean():.4f}")
report(P, "   excluding first and last 15 minutes", mask_extra=(slot >= 15) & (slot < 375))

# 6. other seed (same rule recomputed from that seed's own validation) and the shuffled-label placebo
P2 = oof(7)
report(P2, "6. different random seed (7)")
Ps = oof(3, shuffle=True)
report(Ps, "7. PLACEBO: shuffled training labels")

print("\n8. thinner slices, cutoff from 2022-24 validation, scored on 2025-26 (headline vs one-minute delay):")
for frac in (0.10, 0.05, 0.01):
    conf = np.abs(P - .5)
    th_f = np.quantile(conf[val], 1 - frac)
    for nm, ye in (("trade at signal close", y1), ("one-minute delay    ", y2)):
        c = (conf >= th_f) & test & np.isfinite(ye) & (ye != 0)
        a = ((P[c] > .5) == (ye[c] > 0)).mean()
        gr = (np.where(P[c] > .5, 1, -1) * ye[c]).mean()
        se = 1.96 * np.sqrt(.25 / c.sum())
        print(f"   top {frac:4.0%} {nm}: calls {c.sum():6d} | acc {a:.4f} (+/-{se:.3f}) | gross {gr:+.3f} bps/call")
