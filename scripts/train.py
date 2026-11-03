"""Train and evaluate the JPM 30-minute direction model and the rest-of-day quantile band.

Validation is an expanding-window walk-forward: for each test year Y in 2022..2026 the models are fit
on all data before Y (1 session embargo; every label resolves inside its own session, so no training
label overlaps the test period) and predict year Y. Those out-of-fold (OOF) predictions are used as:
  validation 2022-2024: calibration, confidence threshold, conformal width (nothing else)
  test       2025-2026: reported untouched
The final models are then fit on everything up to the training cutoff and saved to models/.

Usage: uv run python scripts/train.py [--cutoff 2026-10-27] [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import bars as B  # noqa: E402
from jpmind import config, features, store  # noqa: E402

FOLDS = [2022, 2023, 2024, 2025, 2026]
VAL_YEARS, TEST_YEARS = [2022, 2023, 2024], [2025, 2026]
QS = [0.05, 0.5, 0.95]
H = features.HORIZON
COSTS = [1.0, 2.0, 5.0]
LGB_CLS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.7,
               subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=-1)
LGB_Q = dict(n_estimators=120, learning_rate=0.05, num_leaves=15, min_child_samples=300, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=-1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_dataset(refresh):
    cache = config.DATA_DIR / "features_5m.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    syms = [config.SYMBOL, "SPY", "XLF"] + features.PEERS
    frames = {s: store.load(s) for s in syms}
    df = features.build(frames)
    df.to_parquet(cache)
    return df


def lr_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=300))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), acc=float(((p > 0.5) == y).mean()), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p)))


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def signal_stats(p, ret, thr, gate, med, cost):
    """Up if p >= thr, Down if p <= 1-thr, only when the predicted 30-min median move beats the gate."""
    sig = np.where(p >= thr, 1, np.where(p <= 1 - thr, -1, 0))
    if gate > 0:
        sig = np.where(np.abs(med) >= gate, sig, 0)
        sig = np.where(np.sign(med) == sig, sig, 0)        # the median move must agree with the direction
    m = sig != 0
    n = int(m.sum())
    if n == 0:
        return dict(n=0, acc=np.nan, net=np.nan, t=np.nan, total=0.0)
    net = sig[m] * ret[m] - cost
    return dict(n=n, acc=float((sig[m] * ret[m] > 0).mean()), net=float(net.mean()),
                t=float(net.mean() / (net.std(ddof=1) / np.sqrt(n) + 1e-12)) if n > 2 else np.nan, total=float(net.sum()))


def fit_cls(X, y):
    m = lgb.LGBMClassifier(**LGB_CLS)
    m.fit(X, y)
    return m


def fit_quantiles(X, Y, quick):
    out = {}
    params = dict(LGB_Q, n_estimators=40 if quick else LGB_Q["n_estimators"])
    for k in features.ANCHORS:
        ok = Y[f"y{k}"].notna().values
        for q in QS:
            out[(k, q)] = lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(X[ok], Y[f"y{k}"].values[ok])
    return out


def predict_quantiles(models, X):
    P = np.zeros((len(X), len(features.ANCHORS), len(QS)), dtype=np.float32)
    for i, k in enumerate(features.ANCHORS):
        for j, q in enumerate(QS):
            P[:, i, j] = models[(k, q)].predict(X)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-10-27")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff).date()
    config.MODEL_DIR.mkdir(exist_ok=True)

    df = load_dataset(args.refresh)
    cols = features.feature_names(df)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    df = df.replace([np.inf, -np.inf], np.nan)
    elig = (day <= cutoff) & df[cols].notna().all(axis=1).values & df[f"y{H}"].notna().values & (df[f"y{H}"] != 0).values
    log(f"{len(df):,} bars; {elig.sum():,} usable rows up to {cutoff}; {len(cols)} features")
    Xall = df[cols]
    ret = df[f"y{H}"].values
    ydir = (ret > 0).astype(int)
    ycols = [f"y{k}" for k in features.ANCHORS]


    # ---- leakage gates: stop before any model is saved if these fail ----
    ics = {c: abs(pd.Series(Xall.loc[elig, c].values).corr(pd.Series(ret[elig]), method="spearman")) for c in cols}
    worst = max(ics, key=ics.get)
    log(f"leak gate 1 (single-feature |IC| with the 30-min label): worst {worst} = {ics[worst]:.4f} (limit 0.06)")
    if ics[worst] > 0.06:
        raise SystemExit(f"LEAKAGE SUSPECTED: {worst} has |IC| {ics[worst]:.3f}")
    rng = np.random.default_rng(0)
    trp = elig & (year < 2025)
    tep = elig & (year == 2025)
    ysh = rng.permutation(ydir[trp])
    pl = lr_model().fit(Xall[trp], ysh)
    acc = ((pl.predict_proba(Xall[tep])[:, 1] > .5) == ydir[tep]).mean()
    base = max(ydir[tep].mean(), 1 - ydir[tep].mean())
    log(f"leak gate 2 (shuffled-label placebo): acc {acc:.4f}, majority-class {base:.4f} (must be <= {base + 0.015:.4f})")
    if acc > base + 0.015:
        raise SystemExit("LEAKAGE SUSPECTED: model trained on shuffled labels still predicts")

    oof = {m: np.full(len(df), np.nan) for m in ("lgb", "lr")}
    oofq = np.full((len(df), len(features.ANCHORS), len(QS)), np.nan, dtype=np.float32)
    for Y in FOLDS:
        start = pd.Timestamp(f"{Y}-01-01").date()
        te = elig & (year == Y)
        tr = elig & (day < start) & (day <= (pd.Timestamp(start) - pd.Timedelta(days=1)).date())
        t0 = time.time()
        m = fit_cls(Xall[tr], ydir[tr])
        oof["lgb"][te] = m.predict_proba(Xall[te])[:, 1]
        lr = lr_model().fit(Xall[tr], ydir[tr])
        oof["lr"][te] = lr.predict_proba(Xall[te])[:, 1]
        # quantile models train on rows that have all feature values; targets differ per anchor
        trq = (day < start) & df[cols].notna().all(axis=1).values
        qm = fit_quantiles(Xall[trq], df.loc[trq, ycols], args.quick)
        teq = te
        oofq[teq] = predict_quantiles(qm, Xall[teq])
        log(f"fold {Y}: train {tr.sum():,} test {te.sum():,}  lgb acc {((oof['lgb'][te] > .5) == ydir[te]).mean():.4f}  "
            f"lr acc {((oof['lr'][te] > .5) == ydir[te]).mean():.4f}  ({time.time() - t0:.0f}s)")
    oof["ens"] = (oof["lgb"] + oof["lr"]) / 2

    val = elig & np.isin(year, VAL_YEARS)
    test = elig & np.isin(year, TEST_YEARS)
    res = {"cutoff": str(cutoff), "features": cols, "n_val": int(val.sum()), "n_test": int(test.sum()), "models": {}}

    # baselines (no training) on the same rows
    base = {"always_up": np.ones(len(df)),
            "momentum_6": (df["ret_6"].values > 0).astype(float),
            "rsi_meanrev": (df["rsi_14"].values < 50).astype(float)}
    for name, p in base.items():
        res["models"][name] = {k: dict(acc=float(((p[msk] > .5) == ydir[msk]).mean()), n=int(msk.sum()))
                               for k, msk in (("val", val), ("test", test))}
    for name in ("lr", "lgb", "ens"):
        res["models"][name] = {k: metrics(ydir[msk], oof[name][msk]) for k, msk in (("val", val), ("test", test))}
    best = min(("lr", "lgb", "ens"), key=lambda n: res["models"][n]["val"]["logloss"])
    res["deployed_classifier"] = best
    log("validation logloss ->", {n: round(res['models'][n]['val']['logloss'], 5) for n in ('lr', 'lgb', 'ens')}, "deploy:", best)

    # calibration: isotonic fit on validation OOF only
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(oof[best][val], ydir[val])
    pc = np.full(len(df), np.nan)
    oos = elig & np.isfinite(oof[best])
    pc[oos] = iso.predict(oof[best][oos])
    res["calibration"] = {k: dict(ece_raw=ece(ydir[msk], oof[best][msk]), ece_cal=ece(ydir[msk], pc[msk]),
                                  brier_raw=float(brier_score_loss(ydir[msk], oof[best][msk])),
                                  brier_cal=float(brier_score_loss(ydir[msk], pc[msk])))
                          for k, msk in (("val", val), ("test", test))}

    # confidence threshold + median-move gate: chosen on validation ONLY
    i6 = features.ANCHORS.index(H)
    med = oofq[:, i6, 1]
    grid = [(t, g) for t in np.arange(0.505, 0.62, 0.005) for g in (0, 0.5, 1, 2, 3)]
    best_rule, best_score = None, -1e9
    for t, g in grid:
        s = signal_stats(pc[val], ret[val], t, g, med[val], COSTS[1])
        if s["n"] >= 150 and s["t"] > best_score:
            best_rule, best_score = (float(t), float(g)), s["t"]
    if best_rule is None:       # nothing qualifies: abstain almost always
        best_rule = (0.60, 0.0)
    thr, gate = best_rule
    res["rule"] = {"threshold": thr, "median_gate_bps": gate, "selected_on": "validation 2022-2024, max t-stat of net bps, n>=150"}

    # ---- nested walk-forward check of the rule SEARCH itself: for each later year, pick the rule using only earlier years ----
    nested_net, nested_rules = [], {}
    for Y in (2024, 2025, 2026):
        past = elig & (year < Y) & (year >= 2022) & np.isfinite(pc)
        cur = elig & (year == Y) & np.isfinite(pc)
        bs, br = -1e9, None
        for t, g in grid:
            st = signal_stats(pc[past], ret[past], t, g, med[past], COSTS[1])
            if st["n"] >= 100 and st["t"] > bs:
                bs, br = st["t"], (float(t), float(g))
        if br is None:
            continue
        nested_rules[Y] = br
        sig = np.where(pc[cur] >= br[0], 1, np.where(pc[cur] <= 1 - br[0], -1, 0))
        if br[1] > 0:
            sig = np.where(np.abs(med[cur]) >= br[1], sig, 0)
            sig = np.where(np.sign(med[cur]) == sig, sig, 0)
        mk = sig != 0
        nested_net += list(sig[mk] * ret[cur][mk] - COSTS[1])
    nn = np.array(nested_net)
    tstat = float(nn.mean() / (nn.std(ddof=1) / np.sqrt(len(nn)))) if len(nn) > 2 else float("nan")
    res["nested"] = {"rules": {str(k): v for k, v in nested_rules.items()}, "n": int(len(nn)),
                     "net_bps": float(nn.mean()) if len(nn) else None, "t": tstat, "required_t": 3.0}
    log("nested walk-forward of the rule search (2024-26, rule chosen on earlier years only):", res["nested"])
    # deployment gate: with ~115 rules searched, require t >= 3 on data the rule never saw; otherwise make NO calls
    if not (len(nn) >= 100 and tstat >= 3.0):
        thr, gate = 1.01, 0.0
        res["rule"] = {"threshold": thr, "median_gate_bps": gate,
                       "selected_on": "NO CALLS: no rule passed nested validation (t >= 3 on unseen years)"}
        log("DEPLOY: no calls (rule search failed nested validation)")
    res["signals"] = {}
    for k, msk in (("val", val), ("test", test)):
        res["signals"][k] = {f"cost_{int(c)}bps": signal_stats(pc[msk], ret[msk], thr, gate, med[msk], c) for c in COSTS}
        res["signals"][k]["coverage"] = res["signals"][k]["cost_2bps"]["n"] / int(msk.sum())
    res["signals"]["no_gate_test"] = signal_stats(pc[test], ret[test], thr, 0, med[test], COSTS[1])
    log("rule", res["rule"], "\n  val ", res["signals"]["val"]["cost_2bps"], "\n  test", res["signals"]["test"]["cost_2bps"])

    # conformalized quantile regression: widen/narrow [q05,q95] so ~90% of futures fall inside
    alpha = 0.10
    Qadj, cov = {}, {}
    cal_rows = (day <= cutoff) & df[cols].notna().all(axis=1).values & (year >= 2023)      # recent window
    for i, k in enumerate(features.ANCHORS):
        y = df[f"y{k}"].values
        ok_val = np.isin(year, VAL_YEARS) & ~np.isnan(oofq[:, i, 0]) & ~np.isnan(y) & elig
        ok_test = np.isin(year, TEST_YEARS) & ~np.isnan(oofq[:, i, 0]) & ~np.isnan(y) & elig
        ok_cal = cal_rows & ~np.isnan(oofq[:, i, 0]) & ~np.isnan(y)
        def scores(m):
            return np.maximum(oofq[m, i, 0] - y[m], y[m] - oofq[m, i, 2])
        n = ok_val.sum()
        q_val = float(np.quantile(scores(ok_val), min(1.0, (1 - alpha) * (1 + 1 / n))))
        n2 = ok_cal.sum()
        Qadj[k] = float(np.quantile(scores(ok_cal), min(1.0, (1 - alpha) * (1 + 1 / n2))))
        inside = lambda m, adj: float(((y[m] >= oofq[m, i, 0] - adj) & (y[m] <= oofq[m, i, 2] + adj)).mean())
        cov[k] = dict(raw_val=inside(ok_val, 0), raw_test=inside(ok_test, 0), cqr_test=inside(ok_test, q_val),
                      q_val_bps=q_val, deployed_q_bps=Qadj[k], n_test=int(ok_test.sum()))
    res["band"] = {"alpha": alpha, "coverage_by_horizon": cov,
                   "note": "cqr_test = widened by the width fitted on validation only; deployed_q uses 2023..cutoff OOF"}
    log("band coverage (test):", {k: (round(v['raw_test'], 3), round(v['cqr_test'], 3)) for k, v in cov.items()})

    # final models on everything up to the cutoff
    log("fitting final models")
    allr = elig
    final = {"lgb": fit_cls(Xall[allr], ydir[allr]), "lr": lr_model().fit(Xall[allr], ydir[allr])}
    final["lgb"].booster_.save_model(str(config.MODEL_DIR / "cls_lgb.txt"))
    joblib.dump(final["lr"], config.MODEL_DIR / "cls_lr.joblib")
    joblib.dump(iso, config.MODEL_DIR / "iso.joblib")
    trq = (day <= cutoff) & df[cols].notna().all(axis=1).values
    fq = fit_quantiles(Xall[trq], df.loc[trq, ycols], args.quick)
    for (k, q), m in fq.items():
        m.booster_.save_model(str(config.MODEL_DIR / f"q_{k}_{int(q * 100):02d}.txt"))
    # analog statistics from history up to the cutoff: what followed a 30-min move of this size
    pastret = df["ret_6"].values
    nxt = df[f"y{H}"].values
    sd = np.nanstd(pastret[elig])
    analog = {}
    for name, lo_, hi_ in (("up", 0.5 * sd, np.inf), ("flat", -0.5 * sd, 0.5 * sd), ("down", -np.inf, -0.5 * sd)):
        m = elig & (pastret >= lo_) & (pastret < hi_) & ~np.isnan(nxt)
        analog[name] = dict(n=int(m.sum()), next_up=float((nxt[m] > 0).mean()), next_mean_bps=float(nxt[m].mean()))
    cfg = dict(features=cols, anchors=features.ANCHORS, quantiles=QS, classifier=best, threshold=thr,
               median_gate_bps=gate, band_q_bps=Qadj, cutoff=str(cutoff), cost_bps=COSTS[1],
               horizon_bars=H, ret6_sd_bps=float(sd), analogs=analog, trained_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    (config.MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=1))
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    log("saved models/ and metrics.json")


if __name__ == "__main__":
    main()
