"""1-minute features at bar t must not change when all later bars are deleted."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import features1m as F, store

syms = [F.SYM, "SPY", "XLF"] + F.PEERS
full_frames = {s: store.load(s) for s in syms}
# restrict to a manageable recent window for the test, but keep enough history for the rolling features
start = pd.Timestamp("2025-06-01", tz="UTC")
frames = {s: d[d.index >= start] for s, d in full_frames.items()}
full = F.build(frames)
cols = F.feature_names(full)
rng = np.random.default_rng(3)
cand = full.index[(full.index >= pd.Timestamp("2026-01-05", tz="UTC"))]
worst = 0.0
for t in rng.choice(cand, 10, replace=False):
    t = pd.Timestamp(t)
    cut = F.build({s: d[d.index <= t + pd.Timedelta(minutes=1)] for s, d in frames.items()})
    a, b = full.loc[t, cols].astype(float), cut.loc[t, cols].astype(float)
    d = (a - b).abs().fillna(0).max()
    worst = max(worst, d)
    assert d < 1e-4, (t, (a - b).abs().sort_values().tail(3))
print("1m causality OK on 10 random bars, max abs diff", worst)
