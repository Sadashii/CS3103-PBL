"""Basic-indicator voting, built greedily. No fitted models: every indicator is a fixed-parameter up/down vote.

Procedure (per target):
  1. Candidate pool = classic indicators (MA cross, MACD, RSI, Bollinger %b, momentum, price vs MA, last-bar), each in a
     trend and a mean-reversion version, several fixed lookbacks.  Report the first four families one by one.
  2. Greedy: step 1 pick the single indicator with the best accuracy on SELECTION data (2016-2021); every next step tries each
     remaining indicator added to the vote (sum of +-1 votes, ties broken by the first pick) and keeps the best.
  3. Stop when validation (2022-2024) accuracy has not improved for 3 steps; the kept set is the one with the best validation accuracy.
  4. Score ONCE on untouched test data (2025-26).  Test data is never used for any choice.
Targets: direction of the next 30 minutes (5m bars, inside the session) and the next trading day (daily close-to-close)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ET = "America/New_York"
CUTOFF = pd.Timestamp("2026-10-27").date()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def pool(c):
    """name -> series of +1 (vote up) / -1 (vote down), using closes up to and including each bar only."""
    P = {}
    sgn = lambda x: np.where(x > 0, 1, -1)
    for f, s in ((5, 20), (10, 30), (20, 50), (50, 200)):
        v = sgn(c.rolling(f).mean() - c.rolling(s).mean())
        P[f"MA cross {f}/{s} trend"], P[f"MA cross {f}/{s} reversion"] = v, -v
    for a, b in ((12, 26), (5, 35)):
        v = sgn(ema(c, a) - ema(c, b))
        h = ema(c, a) - ema(c, b)
        hist = sgn(h - ema(h, 9))
        P[f"MACD {a}/{b} line trend"], P[f"MACD {a}/{b} line reversion"] = v, -v
        P[f"MACD {a}/{b} histogram trend"], P[f"MACD {a}/{b} histogram reversion"] = hist, -hist
    for n in (2, 7, 14, 21):
        v = sgn(rsi(c, n) - 50)
        P[f"RSI{n} >50 trend"], P[f"RSI{n} >50 reversion"] = v, -v
    for n in (10, 20):
        z = (c - c.rolling(n).mean()) / (c.rolling(n).std() + 1e-12)
        P[f"Bollinger{n} %b trend"], P[f"Bollinger{n} %b reversion"] = sgn(z), -sgn(z)
    for n in (1, 3, 6, 12, 24):
        v = sgn(c.diff(n))
        P[f"Momentum {n} trend"], P[f"Momentum {n} reversion"] = v, -v
    for n in (10, 20, 50):
        v = sgn(c - c.rolling(n).mean())
        P[f"Price vs SMA{n} trend"], P[f"Price vs SMA{n} reversion"] = v, -v
    return pd.DataFrame({k: pd.Series(v, index=c.index) for k, v in P.items()})


def acc(vote, y, m):
    return float(((vote[m] > 0) == y[m]).mean())


def run(title, c, y, valid, year, seed_prefixes):
    V = pool(c)
    # keep only rows where every indicator is defined and a label exists
    ok = valid & ~V.isna().any(axis=1).values & ~np.isnan(y)
    ytrue = (y > 0)
    sel, val, test = ok & (year <= 2021), ok & np.isin(year, [2022, 2023, 2024]), ok & (year >= 2025)
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"rows: select {sel.sum():,} | validation {val.sum():,} | test {test.sum():,}")
    base = {k: max(ytrue[m].mean(), 1 - ytrue[m].mean()) for k, m in (("select", sel), ("val", val), ("test", test))}
    up = {k: ytrue[m].mean() for k, m in (("select", sel), ("val", val), ("test", test))}
    print(f"always-up accuracy: select {up['select']:.4f} | validation {up['val']:.4f} | test {up['test']:.4f}   (test majority class {base['test']:.4f})")
    yb = ytrue
    A = {k: V[k].values.astype(float) for k in V.columns}
    sa = lambda k: acc(A[k], yb, sel)
    print("\nStep 0: the four most basic indicators, tested one by one (selection accuracy):")
    for p in seed_prefixes:
        best = max((k for k in A if k.startswith(p)), key=sa)
        print(f"  {p:16s} best variant: {best:34s} select {sa(best):.4f}")
    chosen, remaining, hist = [], list(A), []
    total = np.zeros(len(c))
    stale, best_val, best_set = 0, -1, []
    step = 0
    while remaining and stale < 3 and step < 15:
        step += 1
        scores = {}
        for k in remaining:
            s = total + A[k] + (0.5 * A[chosen[0]] if chosen else 0.0)
            scores[k] = acc(s, yb, sel)
        k = max(scores, key=scores.get)
        chosen.append(k)
        remaining.remove(k)
        total = total + A[k]
        s = total + 0.5 * A[chosen[0]]
        a_sel, a_val = acc(s, yb, sel), acc(s, yb, val)
        hist.append((step, k, a_sel, a_val))
        print(f"  step {step:2d} + {k:34s} select {a_sel:.4f} | validation {a_val:.4f}")
        if a_val > best_val + 1e-9:
            best_val, best_set, stale = a_val, list(chosen), 0
        else:
            stale += 1
    s = sum(A[k] for k in best_set) + 0.5 * A[best_set[0]]
    a_te = acc(s, yb, test)
    n = test.sum()
    print(f"\nFINAL indicator set ({len(best_set)}): {best_set}")
    print(f"  select {acc(s, yb, sel):.4f} | validation {acc(s, yb, val):.4f} | TEST (untouched) {a_te:.4f}   vs test always-up {up['test']:.4f}")
    print(f"  test rows {n:,}; 95% interval on accuracy about +/-{1.96 * np.sqrt(0.25 / n):.4f} (wider if labels overlap)")
    # the single best raw indicator on test, for context, chosen on selection only
    b1 = max(A, key=sa)
    print(f"  best single indicator chosen on selection: {b1}: test {acc(A[b1], yb, test):.4f}")
    # significance of how much selection accuracy fell on validation/test (selection bias)
    return a_te


def main():
    df = pd.read_parquet(ROOT / "data" / "features_5m.parquet", columns=["close", "y6"])
    et = df.index.tz_convert(ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    c = df["close"].astype(float)
    y6 = df["y6"].values
    valid = (day <= CUTOFF) & (y6 != 0)
    seeds = ["MA cross", "MACD", "RSI14", "Bollinger20"]
    run("TARGET A: direction of the next 30 minutes (5-minute bars, inside the session)", c, y6, valid, year, seeds)

    d = c.groupby(day).last()
    dyear = np.array([x.year for x in d.index])
    ret = np.log(d.shift(-1) / d).values
    dvalid = np.array([x <= CUTOFF for x in d.index]) & (ret != 0)
    run("TARGET B: direction of the next trading day (daily closes)", d, ret, dvalid, dyear, seeds)


if __name__ == "__main__":
    main()
