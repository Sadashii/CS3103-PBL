"""Multi-bank daily model: pooled panel (JPM BAC C WFC GS MS), features known at the 16:00 close of day d, label = direction of
the return from d's close to the close h trading days later. Walk-forward by year, training labels purged by h.
Reports JPM accuracy and the all-bank accuracy (6x the test sample) against the always-up baseline.
Flags: --no-news (skip news features)."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import bars as B, config, fundamentals, store  # noqa: E402

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
HOR = [(1, "1 day"), (5, "1 week"), (21, "1 month")]
FLAGS = set(sys.argv[1:])
DAY = 86400 * 10**9


def daily(sym):
    d = store.load(sym).copy()
    d.index = d.index.tz_convert(B.ET)
    d = d.between_time("09:30", "15:59")
    g = d.groupby(d.index.date)
    o = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(), "vol": g.volume.sum()})
    o.index = pd.to_datetime(o.index)
    return o


D = {s: daily(s) for s in BANKS + ["SPY", "XLF"]}
days = D["JPM"].index
D = {s: d.reindex(days) for s, d in D.items()}
C = pd.DataFrame({s: np.log(d.close) for s, d in D.items()})
eod = pd.Series([pd.Timestamp(d).tz_localize(B.ET) + pd.Timedelta(hours=16) for d in days], index=days).dt.tz_convert("UTC")
close_ns = eod.astype("int64").values
ev = pd.read_parquet(config.DATA_DIR / "raw" / "events.parquet")
fomc = np.sort(ev[ev.kind == "fomc"].time.astype("int64").values)


def news_feats(sym):
    f = config.DATA_DIR / f"news_fin_{sym}.parquet"
    if "--no-news" in FLAGS or not f.exists():
        return pd.DataFrame(index=days)
    n = pd.read_parquet(f).sort_values("created_at")
    t = n.created_at.astype("int64").values
    cum = np.concatenate([[0], np.cumsum(n.s.values)])
    out = {}
    for w, nm in ((1, "1d"), (3, "3d"), (10, "10d")):
        hi = np.searchsorted(t, close_ns, side="right")
        lo = np.searchsorted(t, close_ns - w * DAY, side="right")
        c = hi - lo
        out[f"news_n_{nm}"] = c
        out[f"news_s_{nm}"] = np.where(c > 0, (cum[hi] - cum[lo]) / np.maximum(c, 1), 0)
    out["news_surprise"] = out["news_n_1d"] / (out["news_n_10d"] / 10 + 0.5)
    return pd.DataFrame(out, index=days)


def since(t, cap):
    if not len(t):
        return np.full(len(days), float(cap))
    last = np.searchsorted(t, close_ns, side="right") - 1
    return np.where(last >= 0, np.minimum((close_ns - t[np.clip(last, 0, None)]) / DAY, cap), cap)


def build(sym):
    d, c = D[sym], C[sym]
    X = pd.DataFrame(index=days)
    r1 = c.diff() * 1e4
    for k in (1, 2, 3, 5, 10, 20, 60):
        X[f"ret_{k}"] = c.diff(k) * 1e4
    on = np.log(d.open / d.close.shift(1)) * 1e4          # overnight return
    ind = np.log(d.close / d.open) * 1e4                   # intraday return
    X["on_1"], X["id_1"] = on, ind
    for k in (5, 20):
        X[f"on_{k}"], X[f"id_{k}"] = on.rolling(k).sum(), ind.rolling(k).sum()
    X["on_id_spread_20"] = X.on_20 - X.id_20
    X["vol_10"], X["vol_60"] = r1.rolling(10).std(), r1.rolling(60).std()
    X["range"] = np.log(d.high / d.low) * 1e4
    X["close_pos"] = (d.close - d.low) / (d.high - d.low + 1e-9)
    X["rel_vol"] = d.vol / d.vol.rolling(20).mean()
    X["dist_ma50"] = (c - np.log(d.close.rolling(50).mean())) * 1e4
    X["dd_252"] = (c - c.rolling(252).max()) * 1e4
    for m in ("SPY", "XLF"):
        for k in (1, 5):
            X[f"{m}_ret_{k}"] = C[m].diff(k) * 1e4
    X["rel_xlf_5"] = X.ret_5 - X.XLF_ret_5
    peers = [p for p in BANKS if p != sym]
    pm = C[peers].mean(axis=1)
    for k in (1, 2, 5):
        X[f"peer_ret_{k}"] = pm.diff(k) * 1e4
    X["rel_peer_1"], X["rel_peer_5"] = X.ret_1 - X.peer_ret_1, X.ret_5 - X.peer_ret_5
    sp = c - pm                                            # pairs-style spread against the other banks
    X["pair_z60"] = (sp - sp.rolling(60).mean()) / sp.rolling(60).std()
    X["pair_z20"] = (sp - sp.rolling(20).mean()) / sp.rolling(20).std()
    sx = c - C["XLF"]
    X["xlf_z60"] = (sx - sx.rolling(60).mean()) / sx.rolling(60).std()
    pd1 = pm.diff(1)
    X["peer_lead_1"] = (pd1 - pd1.rolling(60).mean()) / pd1.rolling(60).std()
    X["dow"] = days.dayofweek
    X["days_since_earn"] = since(np.sort(ev[ev.kind == f"{sym.lower()}_earnings"].time.astype("int64").values), 90)
    X["days_since_fomc"] = since(fomc, 90)
    nx = np.searchsorted(fomc, close_ns, side="right")     # the FOMC calendar is published a year ahead
    X["days_to_fomc"] = np.where(nx < len(fomc), np.minimum((fomc[np.clip(nx, 0, len(fomc) - 1)] - close_ns) / DAY, 60), 60)
    X = X.join(news_feats(sym))
    if "--no-fund" not in FLAGS:
        X = X.join(fundamentals.build(sym, days))
    X["sym"] = sym
    return X.replace([np.inf, -np.inf], np.nan)


panel = {s: build(s) for s in BANKS}
cols = [c for c in panel["JPM"].columns if c != "sym"]
print(len(cols), "features;", len(days), "days x", len(BANKS), "banks;", "with news" if "news_n_1d" in cols else "no news", flush=True)
print("horizon | JPM n | JPM always-up | JPM LR | JPM LGB | ALL n | all always-up | all LR | all LGB   (walk-forward 2020-2026)")
for h, name in HOR:
    ys = {s: (C[s].shift(-h) - C[s] > 0).astype(float).where(C[s].shift(-h).notna()) for s in BANKS}
    res = {"lr": [], "gb": [], "y": [], "sym": []}
    for Y in range(2020, 2027):
        te_idx = np.where(days.year == Y)[0]
        if len(te_idx) < 20:
            continue
        tr_idx = np.arange(0, max(te_idx[0] - h, 0))          # purge: training labels end before the test year starts
        tr = pd.concat([panel[s].iloc[tr_idx].assign(y=ys[s].iloc[tr_idx]) for s in BANKS]).dropna(subset=["y"])
        te = pd.concat([panel[s].iloc[te_idx].assign(y=ys[s].iloc[te_idx]) for s in BANKS]).dropna(subset=["y"])
        if len(tr) < 1000:
            continue
        lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.03, max_iter=1000)).fit(tr[cols], tr.y)
        gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.7,
                                subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1).fit(tr[cols], tr.y)
        res["lr"] += list(lr.predict_proba(te[cols])[:, 1] > .5)
        res["gb"] += list(gb.predict_proba(te[cols])[:, 1] > .5)
        res["y"] += list(te.y.astype(int))
        res["sym"] += list(te.sym)
    y, sy = np.array(res["y"]), np.array(res["sym"])
    m, allm = sy == "JPM", np.ones(len(y), bool)
    acc = lambda k, mask: f"{(np.array(res[k])[mask] == y[mask]).mean():.3f}"
    print(f"{name:8s}| {m.sum():5d} | {y[m].mean():.3f} | {acc('lr', m)} | {acc('gb', m)} | {len(y):6d} | {y.mean():.3f} | "
          f"{acc('lr', allm)} | {acc('gb', allm)}   (95% interval about +/-{1.96 * 0.5 / np.sqrt(len(y) / h):.3f} all banks, "
          f"+/-{1.96 * 0.5 / np.sqrt(max(m.sum() / h, 1)):.3f} JPM)", flush=True)
