"""Train the two live (every-update) models on 1-minute features.

  A) next minute : calibrated P(up), median move and a 90% band for the next 1m bar's close
  B) today's close: quantiles (5,10,25,50,75,90,95%) of the return from now to the day's last close, with conformal width
Walk-forward by year (train < test year), validation 2022-24 (calibration + conformal only), untouched test 2025-26.
Usage: uv run python scripts/train_tick.py [--quick] [--refresh] [--cutoff 2026-10-27]"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import bars as B, config, features1m as F, store  # noqa: E402

QD = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Q1 = [0.05, 0.50, 0.95]
FOLDS = [2022, 2023, 2024, 2025, 2026]
VAL, TEST = [2022, 2023, 2024], [2025, 2026]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cls():
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=2000, subsample=0.5, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1)


def qreg(a, n):
    return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=n, learning_rate=0.06, num_leaves=15, min_child_samples=2000,
                             subsample=0.5, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0, verbose=-1, n_jobs=-1)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cutoff", default="2026-10-27")
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff).date()
    cache = config.DATA_DIR / "features_1m.parquet"
    if cache.exists() and not a.refresh:
        df = pd.read_parquet(cache)
    else:
        syms = [config.SYMBOL, "SPY", "XLF"] + F.PEERS
        df = F.build({s: store.load(s) for s in syms}).astype("float32")
        df.to_parquet(cache)
    cols = F.feature_names(df)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    df = df.replace([np.inf, -np.inf], np.nan)
    ok = df[cols].notna().all(axis=1).values & (day <= cutoff)
    ok1 = ok & df.y1.notna().values & (df.y1 != 0).values
    okD = ok & df.yD.notna().values
    X = df[cols]
    y1, yD = df.y1.values, df.yD.values
    ydir = (y1 > 0).astype(int)
    log(f"{len(df):,} 1m bars; {ok1.sum():,} usable for next-minute, {okD.sum():,} for today's close; {len(cols)} features")
    stride_c = 4 if a.quick else 2          # training-row thinning (consecutive minutes are highly correlated)
    stride_q = 8 if a.quick else 4
    nq = 60 if a.quick else 150

    # ---- leakage gate: single-feature |IC| with both labels ----
    for nm, y, m in (("next-minute", y1, ok1), ("to-close", yD, okD)):
        ic = {c: abs(pd.Series(X.loc[m, c].values[::7]).corr(pd.Series(y[m][::7]), method="spearman")) for c in cols}
        w = max(ic, key=ic.get)
        log(f"leak gate ({nm}): worst single-feature |IC| {w} = {ic[w]:.4f} (limit 0.12 for to-close, 0.05 for next-minute)")
        if ic[w] > (0.12 if nm == "to-close" else 0.05):
            raise SystemExit(f"LEAKAGE SUSPECTED: {w} {ic[w]:.3f}")

    oofA = np.full(len(df), np.nan)
    oofq1 = np.full((len(df), len(Q1)), np.nan, dtype="float32")
    oofqD = np.full((len(df), len(QD)), np.nan, dtype="float32")
    for Y in FOLDS:
        t0 = time.time()
        tr = np.where(ok1 & (year < Y))[0][::stride_c]
        te = ok & (year == Y)
        te1 = te & ok1
        m = cls().fit(X.iloc[tr], ydir[tr])
        oofA[te1] = m.predict_proba(X[te1])[:, 1]
        tr1 = np.where(ok1 & (year < Y))[0][::stride_q]
        for k, q in enumerate(Q1):
            oofq1[te, k] = qreg(q, nq).fit(X.iloc[tr1], y1[tr1]).predict(X[te])
        trD = np.where(okD & (year < Y))[0][::stride_q]
        for k, q in enumerate(QD):
            oofqD[te, k] = qreg(q, nq).fit(X.iloc[trD], yD[trD]).predict(X[te])
        acc = ((oofA[te1] > .5) == ydir[te1]).mean()
        log(f"fold {Y}: test {te.sum():,}  next-minute acc {acc:.4f} (up-rate {ydir[te1].mean():.4f})  ({time.time() - t0:.0f}s)")

    val = np.isin(year, VAL)
    test = np.isin(year, TEST)
    res = {"cutoff": str(cutoff), "features": cols}

    # ---- A: next minute ----
    mv, mt = ok1 & val & np.isfinite(oofA), ok1 & test & np.isfinite(oofA)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(oofA[mv], ydir[mv])
    pc = np.full(len(df), np.nan)
    pc[ok1 & np.isfinite(oofA)] = iso.predict(oofA[ok1 & np.isfinite(oofA)])
    base = {k: float(ydir[m_].mean()) for k, m_ in (("val", mv), ("test", mt))}
    A = {}
    for k, m_ in (("val", mv), ("test", mt)):
        p = pc[m_]
        A[k] = {"n": int(m_.sum()), "acc": float(((p > .5) == ydir[m_]).mean()), "always_up_acc": max(base[k], 1 - base[k]),
                "logloss": float(log_loss(ydir[m_], np.clip(p, 1e-4, 1 - 1e-4))),
                "logloss_base_rate": float(log_loss(ydir[m_], np.full(m_.sum(), base["val"]))),
                "acc_top10pct_confidence": float(((p[np.abs(p - .5) >= np.quantile(np.abs(p - .5), .9)] > .5) ==
                                                  ydir[m_][np.abs(p - .5) >= np.quantile(np.abs(p - .5), .9)]).mean())}
    res["next_minute"] = A
    log("A next-minute:", json.dumps(A))
    # next-minute 90% band, conformalised on validation
    lo, md, hi = oofq1[:, 0], oofq1[:, 1], oofq1[:, 2]
    mv1 = ok & val & ok1 & np.isfinite(lo)
    s = np.maximum(lo[mv1] - y1[mv1], y1[mv1] - hi[mv1])
    q1adj = float(np.quantile(s, 0.9 * (1 + 1 / len(s))))
    mt1 = ok & test & ok1 & np.isfinite(lo)
    res["next_minute_band"] = {"q_adj_bps": q1adj, "coverage_test": float(((y1[mt1] >= lo[mt1] - q1adj) & (y1[mt1] <= hi[mt1] + q1adj)).mean()),
                               "coverage_test_raw": float(((y1[mt1] >= lo[mt1]) & (y1[mt1] <= hi[mt1])).mean())}
    log("A band:", res["next_minute_band"])

    # ---- B: today's close ----
    mvD, mtD = okD & val & np.isfinite(oofqD[:, 0]), okD & test & np.isfinite(oofqD[:, 0])
    Q = np.sort(oofqD, axis=1)
    out = {}
    adj = {}
    for (i, j), nm in (((0, 6), "90"), ((2, 4), "50")):
        s = np.maximum(Q[mvD, i] - yD[mvD], yD[mvD] - Q[mvD, j])
        target = 0.9 if nm == "90" else 0.5
        adj[nm] = float(np.quantile(s, min(target * (1 + 1 / len(s)), 1)))
        for k, mm in (("val", mvD), ("test", mtD)):
            out[f"cov{nm}_{k}"] = float(((yD[mm] >= Q[mm, i] - adj[nm]) & (yD[mm] <= Q[mm, j] + adj[nm])).mean())
            out[f"cov{nm}_{k}_raw"] = float(((yD[mm] >= Q[mm, i]) & (yD[mm] <= Q[mm, j])).mean())
    # pinball vs an unconditional baseline: sqrt-time scaled empirical quantiles from the validation window
    sc = np.sqrt(df.left_min.values.clip(1))
    zq = np.quantile((yD[mvD] / (df.vol_60.values[mvD] * sc[mvD])), QD)
    pin_m = np.mean([pinball(yD[mtD], Q[mtD, k], q) for k, q in enumerate(QD)])
    pin_b = np.mean([pinball(yD[mtD], zq[k] * df.vol_60.values[mtD] * sc[mtD], q) for k, q in enumerate(QD)])
    out["pinball_model_test"], out["pinball_vol_scaled_baseline_test"] = float(pin_m), float(pin_b)
    # P(close above current price): CDF read off the quantile curve (linear between quantiles)
    pu = np.array([np.interp(0.0, Q[i], QD, left=0.02, right=0.98) for i in np.where(mtD)[0][::5]])
    up_true = (yD[mtD][::5] > 0).astype(float)
    out["brier_p_close_above_now_test"] = float(((1 - pu - up_true) ** 2).mean())
    out["brier_climatology_test"] = float(((up_true.mean() - up_true) ** 2).mean())
    out["adj_bps"] = adj
    res["to_close"] = out
    log("B to-close:", json.dumps(out))

    # ---- fit final models on everything up to the cutoff ----
    log("fitting final models")
    md_ = config.MODEL_DIR
    tr = np.where(ok1)[0][::stride_c]
    final = cls().fit(X.iloc[tr], ydir[tr])
    final.booster_.save_model(str(md_ / "tick_cls.txt"))
    joblib.dump(iso, md_ / "tick_iso.joblib")
    tr1 = np.where(ok1)[0][::stride_q]
    for k, q in enumerate(Q1):
        qreg(q, nq).fit(X.iloc[tr1], y1[tr1]).booster_.save_model(str(md_ / f"tick_q1_{int(q * 100):02d}.txt"))
    trD = np.where(okD)[0][::stride_q]
    for k, q in enumerate(QD):
        qreg(q, nq).fit(X.iloc[trD], yD[trD]).booster_.save_model(str(md_ / f"tick_qD_{int(q * 100):02d}.txt"))
    res.update({"q1_adj_bps": q1adj, "qD_adj_bps": adj, "QD": QD, "Q1": Q1, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(res, open(md_ / "tick_config.json", "w"), indent=1)
    log("saved models/tick_*")


if __name__ == "__main__":
    main()
