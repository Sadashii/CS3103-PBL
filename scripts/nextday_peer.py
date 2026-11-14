"""Peer-weighted next-day direction for JPM. A pooled model scores every stock in the universe at today's close; JPM's call is a
correlation-weighted average of those scores. Settings are chosen on 2020-23 only (walk-forward); 2024-26 is the untouched report.
Writes models/nextday_peer.json (config + tomorrow's per-stock scores and the JPM call) for the dashboard."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "scripts" / "daily_panel.py").read_text()
src = src[: src.index("print(len(cols)")]
src = src.replace('D = {s: daily(s) for s in BANKS + ["SPY", "XLF"]}', 'D = {s: daily(s) for s in BANKS + ["SPY", "XLF"]}')
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--no-news", "--no-fund"]
exec(compile(src, "daily_panel_prefix", "exec"))

import lightgbm as lgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

UNIV = {"banks6": BANKS, "banks+etf": BANKS + ["XLF", "SPY"]}
for s in ("XLF", "SPY"):
    panel[s] = build(s)
cols = [c for c in panel["JPM"].columns if c != "sym"]
ys = {s: (C[s].shift(-1) - C[s] > 0).astype(float).where(C[s].shift(-1).notna()) for s in UNIV["banks+etf"]}
ret = C.diff()
fwd = {s: (C[s].shift(-1) - C[s]) for s in UNIV["banks+etf"]}
VAL, TEST = range(2020, 2024), range(2024, 2027)


def fit(kind, c, tr, sw=None):
    if kind == "lr":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=c, max_iter=1000)).fit(
            tr[cols], tr.y, logisticregression__sample_weight=sw)
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1).fit(tr[cols], tr.y, sample_weight=sw)


def scores(univ, kind, c, years, wt="none", roll=0):
    """Out-of-sample P(up) for every stock in the universe, walk-forward by year (purged by 1 day)."""
    P = pd.DataFrame(np.nan, index=days, columns=univ)
    for Y in years:
        te = np.where(days.year == Y)[0]
        tr_idx = np.arange(max(te[0] - 1 - roll, 0) if roll else 0, max(te[0] - 1, 0))
        tr = pd.concat([panel[s].iloc[tr_idx].reindex(columns=cols + ["sym"]).assign(y=ys[s].iloc[tr_idx], mv=fwd[s].iloc[tr_idx].abs()) for s in univ]).dropna(subset=["y"])
        m = fit(kind, c, tr, tr.mv.fillna(0).values + 1e-6 if wt == "abs" else None)
        for s in univ:
            P.iloc[te, P.columns.get_loc(s)] = m.predict_proba(panel[s].iloc[te].reindex(columns=cols))[:, 1]
    return P


def combine(P, win, power, signed):
    r = ret["JPM"]
    w = pd.DataFrame({s: r.rolling(win).corr(ret[s]) for s in P.columns})
    w = (w if signed else w.clip(lower=0)).fillna(0)
    w = np.sign(w) * w.abs() ** power
    return ((P - 0.5) * w).sum(axis=1) / w.abs().sum(axis=1).replace(0, np.nan) + 0.5


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def mask(years):
    return (days.year.isin(list(years))) & ys["JPM"].notna().values


grid = list(itertools.product(["banks+etf"], ["lr", "lr+gb"], [0.001, 0.003, 0.01], [60, 252], [1, 2], [False], ["abs"], [0]))
cache, res = {}, []
for u, kind, c, win, pw, sg, wt, roll in grid:
    key = (u, kind, c, wt, roll)
    if key not in cache:
        if kind == "lr+gb":
            cache[key] = (scores(UNIV[u], "lr", c, list(VAL) + list(TEST), wt, roll) + scores(UNIV[u], "gb", c, list(VAL) + list(TEST), wt, roll)) / 2
        else:
            cache[key] = scores(UNIV[u], "lr", c, list(VAL) + list(TEST), wt, roll)
    p = combine(cache[key], win, pw, sg)
    mv = mask(VAL) & p.notna().values
    res.append(dict(u=u, kind=kind, C=c, wt=wt, roll=roll, win=win, power=pw, signed=sg, val_ll=ll(p[mv].values, ys["JPM"][mv].values),
                    val_acc=((p[mv] > .5) == ys["JPM"][mv]).mean()))
R = pd.DataFrame(res).sort_values("val_ll")
print("top 5 settings by VALIDATION (2020-23) log-loss:\n", R.head(5).to_string(index=False))
best = R.iloc[0]
Pb = combine(cache[(best.u, best.kind, best.C, best.wt, int(best.roll))], int(best.win), best.power, bool(best.signed))
yv, yt = ys["JPM"][mask(VAL)].values, ys["JPM"][mask(TEST)].values
pv, pt = Pb[mask(VAL)].values, Pb[mask(TEST)].values
# decision threshold on P(up): chosen on validation accuracy only
ths = np.arange(0.46, 0.56, 0.005)
th = ths[np.argmax([((pv > t) == yv).mean() for t in ths])]
nt = len(yt)
print(f"\nchosen: {dict(best)}; decision threshold {th:.3f} (val acc {((pv > th) == yv).mean():.3f}, val always-up {yv.mean():.3f})")
print(f"TEST 2024-26, {nt} days (95% interval +/-{1.96 * 0.5 / np.sqrt(nt):.3f}):")
print(f"  always-up        {yt.mean():.3f}")
print(f"  model @0.5       {((pt > .5) == yt).mean():.3f}")
print(f"  model @tuned th  {((pt > th) == yt).mean():.3f}   (calls Up on {(pt > th).mean():.0%} of days)")
print(f"  JPM-only LR      {((cache[(best.u, best.kind, best.C, best.wt, int(best.roll))]['JPM'][mask(TEST)].values > .5) == yt).mean():.3f}")
for fr in (0.5, 0.25):
    q = np.abs(pt - th)
    mm = q >= np.quantile(q, 1 - fr)
    print(f"  most confident {int(fr * 100)}% ({mm.sum()} days): {((pt[mm] > th) == yt[mm]).mean():.3f}   (always-up on same days {yt[mm].mean():.3f})")

for nm, pp, yy in (("val 2020-23", pv, yv), ("test 2024-26", pt, yt)):
    print(f"  {nm}: P(up) range {pp.min():.3f}-{pp.max():.3f}; Up calls at 0.5: {(pp > .5).mean():.1%}; at {th:.3f}: {(pp > th).mean():.1%}; actual up days {yy.mean():.1%}")
RANK_Q = 0.40   # call Up when the score is above its trailing 40th percentile -> about 60% Up calls
thr = Pb.shift(1).rolling(252, min_periods=60).quantile(RANK_Q)
for nm, yrs in (("val 2020-23", VAL), ("test 2024-26", TEST)):
    mk = mask(yrs) & thr.notna().values
    up = (Pb[mk] > thr[mk]).values
    yy = ys["JPM"][mk].values
    print(f"  rank rule {nm}: Up calls {up.mean():.1%}, accuracy {(up == yy).mean():.3f} (always-up {yy.mean():.3f}, coin flip 0.500); "
          f"accuracy when Up {yy[up].mean():.3f} / when Down {1 - yy[~up].mean():.3f}")
# final fit on everything up to today -> tomorrow's call
last = len(days) - 1
univ = UNIV[best.u]
kinds = ["lr", "gb"] if best.kind == "lr+gb" else ["lr"]
lo = max(last - int(best.roll), 0) if best.roll else 0
tr = pd.concat([panel[s].iloc[lo:last].reindex(columns=cols + ["sym"]).assign(y=ys[s].iloc[lo:last], mv=fwd[s].iloc[lo:last].abs()) for s in univ]).dropna(subset=["y"])
ms = [fit(k, best.C, tr, tr.mv.fillna(0).values + 1e-6 if best.wt == "abs" else None) for k in kinds]
now = {s: float(np.mean([m.predict_proba(panel[s].iloc[[last]].reindex(columns=cols))[0, 1] for m in ms])) for s in univ}
w = pd.DataFrame({s: ret["JPM"].rolling(int(best.win)).corr(ret[s]) for s in univ}).iloc[last]
w = (w if best.signed else w.clip(lower=0)).fillna(0)
w = np.sign(w) * w.abs() ** best.power
pj = float(((pd.Series(now) - .5) * w).sum() / w.abs().sum() + .5)
out = dict(asof=str(days[last].date()), config={k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}, threshold=float(th),
           jpm_p_up=pj, call="Up" if pj > float(thr.iloc[-1]) else "Down", rank_threshold=float(thr.iloc[-1]), rank_rule="Up if score > trailing-252d 40th percentile (about 60% Up calls)", stock_p_up=now, weights={s: float(w[s]) for s in univ},
           test=dict(days=int(nt), acc=float(((pt > th) == yt).mean()), always_up=float(yt.mean())))
(ROOT / "models" / "nextday_peer.json").write_text(json.dumps(out, indent=1))
print("\ntomorrow's call from", out["asof"], ":", out["call"], f"P(up)={pj:.3f}", {s: round(v, 3) for s, v in now.items()})
