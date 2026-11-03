"""Full 1..10% confidence sweep per stock, gross PnL (before costs) compounded from 100 over 2025-26 test.
Slice picked on VALIDATION (2022-24) by gross gain per call; test numbers for it reported once. Both execution timings shown.
Usage: uv run python scripts/stock_sweep.py JPM BAC C WFC MS"""
import sys, warnings
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jpmind import bars as B, features1m as F, store  # noqa: E402
CONTEXT = ["SPY", "XLF", "JPM", "BAC", "C", "WFC", "MS"]

def probs(sym):
    frames = {s: store.load(s) for s in CONTEXT if s != sym}
    frames[sym] = store.load(sym)
    df = F.build(frames, sym=sym, peers=[s for s in ["JPM", "BAC", "C", "WFC", "MS"] if s != sym]).replace([np.inf, -np.inf], np.nan)
    cols = F.feature_names(df)
    et = df.index.tz_convert(B.ET)
    year, day = np.asarray(et.year), np.asarray(et.date)
    ok = df[cols].notna().all(axis=1).values & (day <= pd.Timestamp("2026-10-27").date())
    ok1 = ok & df.y1.notna().values & (df.y1 != 0).values
    y1, yd = df.y1.values, (df.y1.values > 0).astype(int)
    P = np.full(len(df), np.nan)
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr, te = np.where(ok1 & (year < Y))[0][::3], np.where(ok1 & (year == Y))[0]
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1, random_state=0)
        P[te] = m.fit(df[cols].iloc[tr], yd[tr]).predict_proba(df[cols].iloc[te])[:, 1]
    return P, y1, df.y1.shift(-1).values, ok1, year

summary = []
for sym in sys.argv[1:]:
    P, y1, y2, ok1, year = probs(sym)
    val, test = ok1 & np.isin(year, [2022, 2023, 2024]), ok1 & (year >= 2025)
    conf, sgn = np.abs(P - .5), np.where(P > .5, 1, -1)
    print(f"\n=== {sym} ===  pct | val gross | test acc | test gross | n | 100 -> (gross, trade at close) | gross with 1-min delay | 100 -> (delay)")
    best = None
    for pct in range(1, 11):
        th = np.quantile(conf[val], 1 - pct / 100)
        cv, ct = val & (conf >= th), test & (conf >= th)
        vg, tg = (sgn[cv] * y1[cv]).mean(), (sgn[ct] * y1[ct]).mean()
        cd = ct & np.isfinite(y2) & (y2 != 0)
        dg = (sgn[cd] * y2[cd]).mean()
        eq, eqd = 100 * np.prod(1 + sgn[ct] * y1[ct] / 1e4), 100 * np.prod(1 + sgn[cd] * y2[cd] / 1e4)
        acc = ((P[ct] > .5) == (y1[ct] > 0)).mean()
        print(f"  {pct:2d}% | {vg:+.3f} | {acc:.4f} | {tg:+.3f} | {ct.sum():6d} | {eq:8.2f} | {dg:+.3f} | {eqd:8.2f}")
        if best is None or vg > best[1]:
            best = (pct, vg, acc, tg, eq, dg, eqd)
    summary.append((sym,) + best)
    print(f"  -> chosen on validation: top {best[0]}%", flush=True)
print("\nSUMMARY (slice chosen on validation; test 2025-26; PnL before costs, 100 start):")
print("sym | slice | test acc | gross bps/call | 100 -> trade at close | gross with 1-min delay | 100 -> with delay")
for s, pct, vg, acc, tg, eq, dg, eqd in summary:
    print(f"{s:4s} | top {pct}% | {acc:.4f} | {tg:+.3f} | {eq:8.2f} | {dg:+.3f} | {eqd:8.2f}")
