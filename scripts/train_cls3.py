"""Train and evaluate the 3-class 30-minute call (Up / Neutral / Down) and the "big move" warning.

Why 3 classes (TASK.md 6.1, O15): the instructor asked for multi-class. Our earlier work found skill in the SIZE
of the next move, not its direction, so a Neutral class ("the next 30 min will move less than tau") is the part
of the call we can actually predict, and it gives a real, gradeable call on every candle.

Labels, on the 30-min log return y6 (bps):
  Up      y6 >  +tau
  Neutral |y6| <= tau
  Down    y6 <  -tau
  tau = 33rd percentile of |y6| over 2016-2021 (fixed before any validation or test year, so classes are ~1/3
  each and a "move" is several times the 2 bps round-trip cost).
Big move: |y6| > tau_big, tau_big = 80th percentile of |y6| over 2016-2021 (top ~20% of half hours).

Validation is the same expanding walk-forward as scripts/train.py (fit before year Y, predict Y, Y = 2022..2026);
model choice on 2022-24 only, 2025-26 reported untouched. Because the live test is ONE trading day, we also score
every test day on its own and report how often the model beats each baseline within a single day.

Usage: uv run python scripts/train_cls3.py [--cutoff 2026-10-27] [--quick]
Writes models/cls3_lgb.txt, models/cls3_lr.joblib, models/big_lgb.txt, models/cls3.json, models/cls3_metrics.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jpmind import bars as B  # noqa: E402
from jpmind import config, features  # noqa: E402
from train import FOLDS, LGB_CLS, TEST_YEARS, VAL_YEARS, load_dataset, log  # noqa: E402

H = features.HORIZON
CLASSES = ["Down", "Neutral", "Up"]          # label codes 0, 1, 2
COST = 2.0


def lr3():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=500))


def fit_lgb3(X, y, quick):
    p = dict(LGB_CLS, n_estimators=80 if quick else LGB_CLS["n_estimators"])
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, **p).fit(X, y)


def fit_big(X, y, quick):
    p = dict(LGB_CLS, n_estimators=80 if quick else LGB_CLS["n_estimators"])
    return lgb.LGBMClassifier(**p).fit(X, y)


def ece_multi(y, P, bins=10):
    """Mean over classes of the one-vs-rest expected calibration error."""
    out = []
    for k in range(P.shape[1]):
        p, t = P[:, k], (y == k).astype(float)
        b = np.minimum((p * bins).astype(int), bins - 1)
        out.append(sum(abs(t[b == j].mean() - p[b == j].mean()) * (b == j).mean() for j in range(bins) if (b == j).any()))
    return float(np.mean(out))


def call_stats(pred, y, ret):
    """pred/y in {0,1,2}. Accuracy over all candles, plus how the Up/Down calls did as trades after costs."""
    acc = float((pred == y).mean())
    mv = pred != 1
    sign = np.where(pred == 2, 1, -1)[mv]
    net = sign * ret[mv] - COST
    n = int(mv.sum())
    return dict(n=int(len(y)), acc=acc, moves_called=n,
                dir_hit=float((sign * ret[mv] > 0).mean()) if n else None,
                net_bps_per_call=float(net.mean()) if n else None,
                t=float(net.mean() / (net.std(ddof=1) / np.sqrt(n))) if n > 2 else None,
                neutral_precision=float((y[pred == 1] == 1).mean()) if (pred == 1).any() else None,
                share=[float((pred == k).mean()) for k in range(3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-10-27")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bars", type=int, default=2, help="horizon in 5m bars (2 = 10 min, chosen in exp_horizon_direction.py)")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff).date()

    df = load_dataset(False).replace([np.inf, -np.inf], np.nan)
    cols = features.feature_names(df)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    K = args.bars
    lc = np.log(df["close"])
    ret = ((lc.shift(-K) - lc) * 1e4).where(df["bars_left"] >= K).values       # same rule as features.build: never crosses the close
    elig = (day <= cutoff) & df[cols].notna().all(axis=1).values & np.isfinite(ret)

    pre = elig & (year <= 2021)
    tau = float(np.quantile(np.abs(ret[pre]), 1 / 3))
    tau_big = float(np.quantile(np.abs(ret[pre]), 0.8))
    y = np.where(ret > tau, 2, np.where(ret < -tau, 0, 1))
    ybig = (np.abs(ret) > tau_big).astype(int)
    # activity call: how big the next 30-min move will be, whatever its direction (Quiet is the same event as Neutral)
    tau_hi = float(np.quantile(np.abs(ret[pre]), 2 / 3))
    act_of = lambda a: np.where(np.abs(a) > tau_hi, 2, np.where(np.abs(a) > tau, 1, 0))
    yact = act_of(ret)
    log(f"{elig.sum():,} rows; tau {tau:.2f} bps, tau_big {tau_big:.2f} bps; class shares (all) "
        f"{[round(float((y[elig] == k).mean()), 3) for k in range(3)]}")
    X = df[cols]

    # leakage gate: shuffled-label placebo must not beat the majority class
    rng = np.random.default_rng(0)
    trp, tep = elig & (year < 2025), elig & (year == 2025)
    pl = lr3().fit(X[trp], rng.permutation(y[trp]))
    acc = float((pl.predict(X[tep]) == y[tep]).mean())
    maj = max(float((y[tep] == k).mean()) for k in range(3))
    log(f"placebo acc {acc:.4f} vs majority {maj:.4f}")
    if acc > maj + 0.015:
        raise SystemExit("LEAKAGE SUSPECTED: shuffled-label model still predicts")

    oof = {m: np.full((len(df), 3), np.nan) for m in ("lgb", "lr")}
    oofb = np.full(len(df), np.nan)
    oofa = np.full((len(df), 3), np.nan)
    for Y in FOLDS:
        start = pd.Timestamp(f"{Y}-01-01").date()
        tr, te = elig & (day < start), elig & (year == Y)
        t0 = time.time()
        oof["lgb"][te] = fit_lgb3(X[tr], y[tr], args.quick).predict_proba(X[te])
        oof["lr"][te] = lr3().fit(X[tr], y[tr]).predict_proba(X[te])
        oofa[te] = fit_lgb3(X[tr], yact[tr], args.quick).predict_proba(X[te])
        oofb[te] = fit_big(X[tr], ybig[tr], args.quick).predict_proba(X[te])[:, 1]
        log(f"fold {Y}: train {tr.sum():,} test {te.sum():,}  lgb acc {(oof['lgb'][te].argmax(1) == y[te]).mean():.4f}  "
            f"lr acc {(oof['lr'][te].argmax(1) == y[te]).mean():.4f}  big AUC {roc_auc_score(ybig[te], oofb[te]):.3f}  ({time.time() - t0:.0f}s)")
    oof["ens"] = (oof["lgb"] + oof["lr"]) / 2

    val, test = elig & np.isin(year, VAL_YEARS), elig & np.isin(year, TEST_YEARS)
    res = {"cutoff": str(cutoff), "tau_bps": tau, "tau_big_bps": tau_big, "cost_bps": COST,
           "class_share_test": [float((y[test] == k).mean()) for k in range(3)], "models": {}, "baselines": {}}
    for m in ("lr", "lgb", "ens"):
        res["models"][m] = {k: dict(logloss=float(log_loss(y[s], oof[m][s], labels=[0, 1, 2])), ece=ece_multi(y[s], oof[m][s]),
                                    **call_stats(oof[m][s].argmax(1), y[s], ret[s])) for k, s in (("val", val), ("test", test))}
    best = min(("lr", "lgb", "ens"), key=lambda m: res["models"][m]["val"]["logloss"])
    res["deployed"] = best
    P = oof[best]
    log("val logloss", {m: round(res["models"][m]["val"]["logloss"], 5) for m in res["models"]}, "-> deploy", best)

    # baselines (no training on the label beyond tau)
    rsi = df["rsi_14"].values
    prev = np.nan_to_num(((lc - lc.shift(K)) * 1e4).values)                     # the move over the last K bars
    train_major = int(np.bincount(y[elig & (year <= 2021)]).argmax())
    bl = {"always_neutral": np.ones(len(df), int), "always_up": np.full(len(df), 2),
          "majority_2016_21": np.full(len(df), train_major),
          "rsi_rule": np.where(rsi < 30, 2, np.where(rsi > 70, 0, 1)),            # oversold -> Up, overbought -> Down
          "persistence": np.where(prev > tau, 2, np.where(prev < -tau, 0, 1))}     # last 30 min's class repeats
    for name, pr in bl.items():
        res["baselines"][name] = {k: call_stats(pr[s], y[s], ret[s]) for k, s in (("val", val), ("test", test))}

    # big-move warning
    res["big_move"] = {k: dict(auc=float(roc_auc_score(ybig[s], oofb[s])), base_rate=float(ybig[s].mean()),
                               rate_top_decile=float(ybig[s][oofb[s] >= np.quantile(oofb[s], 0.9)].mean()),
                               rate_bottom_decile=float(ybig[s][oofb[s] <= np.quantile(oofb[s], 0.1)].mean()))
                       for k, s in (("val", val), ("test", test))}
    # warning threshold: on validation, the lowest p where the big-move rate is at least 2x the base rate
    pv, yv = oofb[val], ybig[val]
    thr_big = None
    for t in np.quantile(pv, np.linspace(0.5, 0.98, 49)):
        if (pv >= t).sum() >= 500 and yv[pv >= t].mean() >= 2 * yv.mean():
            thr_big = float(t)
            break
    thr_big = thr_big if thr_big is not None else float(np.quantile(pv, 0.9))
    res["big_move"]["threshold"] = thr_big
    res["big_move"]["test_flagged"] = dict(share=float((oofb[test] >= thr_big).mean()),
                                           big_rate=float(ybig[test][oofb[test] >= thr_big].mean()))
    log("big move:", res["big_move"])

    # ---- activity call (Quiet / Normal / Active) ----
    preda = oofa.argmax(1)
    abl = {"always_normal": np.ones(len(df), int), "always_quiet": np.zeros(len(df), int), "persistence": act_of(prev)}
    res["activity"] = {"tau_hi_bps": tau_hi,
                       "model": {k: dict(acc=float((preda[s] == yact[s]).mean()), logloss=float(log_loss(yact[s], oofa[s], labels=[0, 1, 2])),
                                         ece=ece_multi(yact[s], oofa[s]),
                                         active_precision=float((yact[s][preda[s] == 2] == 2).mean()),
                                         quiet_precision=float((yact[s][preda[s] == 0] == 0).mean()),
                                         share=[float((preda[s] == k).mean()) for k in range(3)]) for k, s in (("val", val), ("test", test))},
                       "baselines": {b: {k: float((abl[b][s] == yact[s]).mean()) for k, s in (("val", val), ("test", test))} for b in abl}}
    ad = pd.DataFrame({str(d): {"model": float((preda[test & (day == d)] == yact[test & (day == d)]).mean()),
                                **{b: float((abl[b][test & (day == d)] == yact[test & (day == d)]).mean()) for b in abl}}
                       for d in sorted(set(day[test]))}).T
    res["activity"]["single_day"] = {"model_mean": float(ad.model.mean()), "model_p10": float(ad.model.quantile(.1)),
                                     "beats": {b: float((ad.model > ad[b]).mean()) for b in abl}}
    log("activity:", res["activity"])

    # ---- single-day test: the live exam replays ONE day, so score each 2025-26 day on its own ----
    pred = P.argmax(1)
    per_day = {}
    for d in sorted(set(day[test])):
        s = test & (day == d)
        per_day[str(d)] = {"model": float((pred[s] == y[s]).mean()), "n": int(s.sum()),
                           **{b: float((bl[b][s] == y[s]).mean()) for b in bl}}
    pdf = pd.DataFrame(per_day).T
    res["single_day"] = {"days": int(len(pdf)), "model_mean": float(pdf.model.mean()), "model_p10": float(pdf.model.quantile(.1)),
                         "model_p90": float(pdf.model.quantile(.9)),
                         "beats": {b: float((pdf.model > pdf[b]).mean()) for b in bl},
                         "ties": {b: float((pdf.model == pdf[b]).mean()) for b in bl},
                         "baseline_mean": {b: float(pdf[b].mean()) for b in bl}}
    log("single day:", res["single_day"])

    # final models on everything up to the cutoff
    log("fitting final models")
    fl = fit_lgb3(X[elig], y[elig], args.quick)
    fl.booster_.save_model(str(config.MODEL_DIR / "cls3_lgb.txt"))
    joblib.dump(lr3().fit(X[elig], y[elig]), config.MODEL_DIR / "cls3_lr.joblib")
    fit_lgb3(X[elig], yact[elig], args.quick).booster_.save_model(str(config.MODEL_DIR / "act_lgb.txt"))
    fit_big(X[elig], ybig[elig], args.quick).booster_.save_model(str(config.MODEL_DIR / "big_lgb.txt"))
    cfg = dict(features=cols, classes=CLASSES, kind=best, tau_bps=tau, tau_big_bps=tau_big, tau_hi_bps=tau_hi, activity=["Quiet", "Normal", "Active"], big_threshold=thr_big,
               horizon_bars=K, cutoff=str(cutoff), cost_bps=COST, trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               honesty=dict(test_acc=res["models"][best]["test"]["acc"],
                            test_always_neutral=res["baselines"]["always_neutral"]["test"]["acc"],
                            test_persistence=res["baselines"]["persistence"]["test"]["acc"],
                            test_rsi=res["baselines"]["rsi_rule"]["test"]["acc"],
                            act_test_acc=res["activity"]["model"]["test"]["acc"],
                            act_test_persistence=res["activity"]["baselines"]["persistence"]["test"],
                            act_single_day_beats_persistence=res["activity"]["single_day"]["beats"]["persistence"],
                            big_auc_test=res["big_move"]["test"]["auc"],
                            single_day_beats_neutral=res["single_day"]["beats"]["always_neutral"]))
    (config.MODEL_DIR / "cls3.json").write_text(json.dumps(cfg, indent=1))
    (config.MODEL_DIR / "cls3_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    pdf.to_csv(config.MODEL_DIR / "cls3_per_day.csv")
    log("saved")


if __name__ == "__main__":
    main()
