"""Fit the next-day outlook model (training sessions <= cutoff) and evaluate it walk-forward 2020-2026,
with and without the overnight-return features. Usage: uv run python scripts/next_day.py [--cutoff 2026-10-27] [--no-save]"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config, nextday as N, store  # noqa: E402

cutoff = sys.argv[sys.argv.index("--cutoff") + 1] if "--cutoff" in sys.argv else "2026-10-27"
frames = {s: store.load(s) for s in N.BANKS}
J = N.daily_frame(frames["JPM"])
XJ = N.xfeat(J)
r_next = np.log(J.close.shift(-1) / J.close)
T = np.log(1.01)

print("variant | year | n | 90% cov | 50% cov | Brier |move|>1% (model / climatology) | Brier up (model / base rate) | corr(expected, realized)")
agg = {}
for overnight in (False, True):
    name = "with overnight" if overnight else "HAR only      "
    rows = []
    for Y in range(2020, 2027):
        first = J.index[J.index.year == Y][0]
        m = N.NextDay.fit(frames, first - pd.Timedelta(days=1), overnight=overnight)
        X = XJ[m.cfg["vcols"] + (N.DRIFT if overnight else [])]
        te = (J.index.year == Y) & X.notna().all(axis=1).values & r_next.notna().values
        sig, mu = m.params(XJ[te])
        r = r_next[te].values
        z = m.z
        lo, md, hi, q25, q75 = [sig * (np.quantile(z, p) + mu) for p in (0.05, 0.5, 0.95, 0.25, 0.75)]
        c90, c50 = ((r >= lo) & (r <= hi)).mean(), ((r >= q25) & (r <= q75)).mean()
        dist = sig[:, None] * (z[None, :] + mu[:, None])
        p_ev = ((dist > T) | (dist < -T)).mean(axis=1)
        p_up = (dist > 0).mean(axis=1)
        ev, up = (np.abs(r) > 0.01).astype(float), (r > 0).astype(float)
        exp_ret = dist.mean(axis=1)
        rows.append((len(r), c90, c50, ((p_ev - ev) ** 2).mean(), ((ev.mean() - ev) ** 2).mean(), ((p_up - up) ** 2).mean(),
                     ((up.mean() - up) ** 2).mean(), np.corrcoef(exp_ret, r)[0, 1]))
        print(f"{name}| {Y} | {len(r)} | {c90:.3f} | {c50:.3f} | {rows[-1][3]:.3f} / {rows[-1][4]:.3f} | {rows[-1][5]:.3f} / {rows[-1][6]:.3f} | {rows[-1][7]:+.3f}")
    a = np.array(rows)
    w = a[:, 0] / a[:, 0].sum()
    print(f"{name}| POOLED | {int(a[:, 0].sum())} | {w @ a[:, 1]:.3f} | {w @ a[:, 2]:.3f} | {w @ a[:, 3]:.4f} / {w @ a[:, 4]:.4f} | "
          f"{w @ a[:, 5]:.4f} / {w @ a[:, 6]:.4f} | {np.mean(a[:, 7]):+.3f}", flush=True)

if "--no-save" not in sys.argv:
    m = N.NextDay.fit(frames, cutoff, overnight=True)
    m.save(config.MODEL_DIR / "nextday.json")
    print("saved models/nextday.json, trained through", m.cutoff, "| z samples", len(m.z), "| vol coef", np.round(m.cfg["vcoef"], 3),
          "| drift coef (z per 1 sd)", np.round(m.cfg["drift"]["coef"], 4))
