"""Experiment: do volatility-specific features improve the move-size models (activity 3-class, big-move)?
All new features use only current and past bars. Choices are made on validation 2022-24; test 2025-26 is printed for the record.
Usage: uv run python scripts/exp_magnitude.py"""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from jpmind import bars as B, features  # noqa: E402
from train import FOLDS, LGB_CLS, TEST_YEARS, VAL_YEARS, load_dataset  # noqa: E402


def vol_features(df):
    """Causal volatility features from columns already in the 5m table."""
    f = pd.DataFrame(index=df.index)
    et = df.index.tz_convert(B.ET)
    day = pd.Series(np.asarray(et.date), index=df.index)
    r2 = df["ret_1"] ** 2
    for k in (3, 6, 12, 24, 78, 390):
        f[f"rv_{k}"] = np.sqrt(r2.rolling(k).sum() / k)
    f["abs_ret6"] = df["ret_6"].abs()
    f["abs_ret6_l6"] = f["abs_ret6"].shift(6)
    f["abs_ret6_l12"] = f["abs_ret6"].shift(12)
    f["abs_ret6_prevday_slot"] = f["abs_ret6"].groupby(df["slot"]).shift(1)          # same time of day, previous session
    f["abs_ret6_slot_avg5"] = f["abs_ret6"].groupby(df["slot"]).transform(lambda s: s.shift(1).rolling(5).mean())
    drv = np.sqrt(r2.groupby(day).sum())                                             # daily realised vol
    for name, s in (("har_d", drv.shift(1)), ("har_w", drv.shift(1).rolling(5).mean()), ("har_m", drv.shift(1).rolling(22).mean())):
        f[name] = s.reindex(day.values).values
    f["rv6_vs_har"] = f["rv_6"] / (f["har_w"] / np.sqrt(78) + 1e-9)
    f["rv6_vs_slot"] = f["abs_ret6"] / (f["abs_ret6_slot_avg5"] + 1e-9)
    f["abs_spy6"], f["abs_peer6"], f["abs_gap"] = df["spy_ret_6"].abs(), df["peer_ret_6"].abs(), df["gap"].abs()
    return f


def main():
    df = load_dataset(False).replace([np.inf, -np.inf], np.nan)
    base = features.feature_names(df)
    vf = vol_features(df).replace([np.inf, -np.inf], np.nan)
    df = df.join(vf)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    ret = df["y6"].values
    elig = (day <= pd.Timestamp("2026-10-27").date()) & df[base].notna().all(axis=1).values & np.isfinite(ret)
    pre = elig & (year <= 2021)
    a = np.abs(ret)
    t1, t2, tb = (float(np.quantile(a[pre], q)) for q in (1 / 3, 2 / 3, 0.8))
    yact = np.where(a > t2, 2, np.where(a > t1, 1, 0)); ybig = (a > tb).astype(int)
    val, test = elig & np.isin(year, VAL_YEARS), elig & np.isin(year, TEST_YEARS)
    sets = {"base": base, "base+vol": base + list(vf.columns), "vol_only+time": list(vf.columns) + ["slot", "bars_left", "dow"]}
    for name, cols in sets.items():
        oa, ob = np.full((len(df), 3), np.nan), np.full(len(df), np.nan)
        for Y in FOLDS:
            tr, te = elig & (day < pd.Timestamp(f"{Y}-01-01").date()), elig & (year == Y)
            oa[te] = lgb.LGBMClassifier(objective="multiclass", num_class=3, **LGB_CLS).fit(df.loc[tr, cols], yact[tr]).predict_proba(df.loc[te, cols])
            ob[te] = lgb.LGBMClassifier(**LGB_CLS).fit(df.loc[tr, cols], ybig[tr]).predict_proba(df.loc[te, cols])[:, 1]
        for k, s in (("val", val), ("test", test)):
            pred = oa[s].argmax(1)
            dd = pd.DataFrame({"d": day[s], "ok": pred == yact[s]}).groupby("d").ok.mean()
            print(f"{name:14s} {k:4s} act acc {(pred == yact[s]).mean():.4f} logloss {log_loss(yact[s], oa[s], labels=[0, 1, 2]):.4f} "
                  f"big AUC {roc_auc_score(ybig[s], ob[s]):.4f}  daily acc p10 {dd.quantile(.1):.3f}", flush=True)


if __name__ == "__main__":
    main()
