"""Features at bar t must not change when data after t is removed (no look-ahead)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config, features, store  # noqa: E402


def main():
    syms = [config.SYMBOL, "SPY", "XLF"] + features.PEERS
    end = pd.Timestamp("2026-11-03", tz="UTC")
    frames = {s: store.load(s)[lambda d: d.index >= end - pd.Timedelta(days=75)] for s in syms}
    full = features.build(frames)
    cols = [c for c in full.columns if not c.startswith("y")]
    rng = np.random.default_rng(0)
    valid = full.index[full.index >= full.index[900]]
    worst = 0.0
    for t in rng.choice(valid, 12, replace=False):
        t = pd.Timestamp(t)
        cut = {s: d[d.index < t + pd.Timedelta(minutes=5)] for s, d in frames.items()}   # data up to the close of bar t
        part = features.build(cut)
        a, b = full.loc[t, cols].astype(float), part.loc[t, cols].astype(float)
        diff = (a - b).abs().max()
        worst = max(worst, diff)
        assert diff < 1e-6, (t, (a - b).abs().sort_values().tail(3))
    print(f"causality OK on 12 random bars, max abs diff {worst:.2e}")


if __name__ == "__main__":
    main()
