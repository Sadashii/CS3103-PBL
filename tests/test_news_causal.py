"""News/event features must not change when everything published after the bar's close is deleted."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import newsfeat

idx = pd.date_range("2022-01-03 14:30", "2026-10-29 20:55", freq="5min", tz="UTC")
idx = idx[(idx.hour >= 14) & (idx.hour < 21)]
KNOWN_AHEAD = ["days_to_fomc"]   # the Fed publishes its meeting calendar a year in advance, so this one may look forward
full = newsfeat.build(idx).drop(columns=KNOWN_AHEAD)
rng = np.random.default_rng(1)
worst = 0.0
for i in rng.choice(len(idx), 25, replace=False):
    t = idx[i]
    cut = newsfeat.build(idx[: i + 1], until=t + pd.Timedelta(minutes=5)).drop(columns=KNOWN_AHEAD)
    d = (full.iloc[i] - cut.iloc[i]).abs().max()
    worst = max(worst, d)
    assert d < 1e-9, (t, (full.iloc[i] - cut.iloc[i]).abs().sort_values().tail(3))
print("news/event causality OK, max abs diff", worst)
