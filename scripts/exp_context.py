"""Experiment: do (1) SPY pre-market reaction features (proxy for 8:30 ET macro releases) and (2) volatility / rates / regional-bank
ETFs (VIXY, TLT, IEF, KRE) improve the models? Walk-forward by year; compare on validation 2022-24; test printed for the record.
All features use bars that closed at or before the candle close (pre-market values are final before 9:30).
Usage: uv run python scripts/exp_context.py [k=2]   (k = horizon in 5m bars)"""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from jpmind import bars as B, features, store  # noqa: E402
from train import FOLDS, LGB_CLS, TEST_YEARS, VAL_YEARS, load_dataset  # noqa: E402
ETFS = ["VIXY", "TLT", "IEF", "KRE"]


def premarket(idx):
    """Per-session SPY pre-market features, mapped to every 5m row of that session."""
    d = store.load("SPY")[["close", "high", "low", "volume"]]
    et = d.index.tz_convert(B.ET)
    m = et.hour * 60 + et.minute
    date = pd.Series(np.asarray(et.date), index=d.index)
    pre = d[(m >= 240) & (m < 570)]
    pd_ = date[pre.index]
    rth_close = d[(m >= 570) & (m < 960)].close.groupby(date).last()
    g = pre.groupby(pd_)
    out = pd.DataFrame({"pm_last": g.close.last(), "pm_hi": g.high.max(), "pm_lo": g.low.min(), "pm_vol": g.volume.sum()})
    win = lambda a, b: pre[(m[(m >= 240) & (m < 570)] >= a) & (m[(m >= 240) & (m < 570)] < b)]
    w0, w1 = win(505, 510), win(510, 525)                      # just before 8:30, and 8:30-8:45
    p0, p1 = w0.close.groupby(date[w0.index]).last(), w1.close.groupby(date[w1.index]).last()
    f = pd.DataFrame(index=out.index)
    f["pm_ret"] = np.log(out.pm_last / rth_close.shift(1).reindex(out.index)) * 1e4
    f["pm_range"] = np.log(out.pm_hi / out.pm_lo) * 1e4
    f["pm_vol_rel"] = out.pm_vol / out.pm_vol.shift(1).rolling(20).mean()
    f["pm_0830_ret"] = (np.log(p1 / p0) * 1e4).reindex(out.index)
    f["pm_0830_abs"] = f["pm_0830_ret"].abs()
    f["pm_0830_abs_rel"] = f["pm_0830_abs"] / f["pm_0830_abs"].shift(1).rolling(60, min_periods=20).median()
    days = np.asarray(idx.tz_convert(B.ET).date)
    return pd.DataFrame(f.reindex(days).values, index=idx, columns=f.columns)


def etf_features(idx, jpm_ret6):
    f = pd.DataFrame(index=idx)
    days = pd.Series(np.asarray(idx.tz_convert(B.ET).date), index=idx)
    for s in ETFS:
        c = features.bars5(store.load(s)).close.reindex(idx).ffill(limit=3)
        lr = np.log(c) * 1e4
        n = s.lower()
        f[f"{n}_ret_1"], f[f"{n}_ret_6"], f[f"{n}_ret_24"] = lr.diff(1), lr.diff(6), lr.diff(24)
        f[f"{n}_ret_day"] = lr - lr.groupby(days).transform("first")
        f[f"{n}_vol_12"] = lr.diff(1).rolling(12).std()
    f["vixy_ret_390"] = (np.log(features.bars5(store.load("VIXY")).close.reindex(idx).ffill(limit=3)) * 1e4).diff(390)
    f["rel_kre_6"] = jpm_ret6 - f["kre_ret_6"]
    f["curve_6"] = f["tlt_ret_6"] - f["ief_ret_6"]
    return f


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    df = load_dataset(False).replace([np.inf, -np.inf], np.nan)
    base = features.feature_names(df)
    pm = premarket(df.index).replace([np.inf, -np.inf], np.nan)
    ef = etf_features(df.index, df["ret_6"]).replace([np.inf, -np.inf], np.nan)
    df = df.join(pm).join(ef)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    lc = np.log(df["close"])
    ret = ((lc.shift(-k) - lc) * 1e4).where(df["bars_left"] >= k).values
    elig = (day <= pd.Timestamp("2026-10-27").date()) & df[base].notna().all(axis=1).values & np.isfinite(ret)
    pre = elig & (year <= 2021)
    a = np.abs(ret)
    t1, t2, tb = (float(np.quantile(a[pre], q)) for q in (1 / 3, 2 / 3, 0.8))
    y3 = np.where(ret > t1, 2, np.where(ret < -t1, 0, 1))
    yact = np.where(a > t2, 2, np.where(a > t1, 1, 0)); ybig = (a > tb).astype(int); up = (ret > 0).astype(int)
    print(f"horizon {k * 5} min; coverage pm {pm.notna().all(axis=1)[elig].mean():.3f} etf {ef.notna().all(axis=1)[elig].mean():.3f}", flush=True)
    sets = {"base": base, "+premarket": base + list(pm.columns), "+etf": base + list(ef.columns), "+both": base + list(pm.columns) + list(ef.columns)}
    for name, cols in sets.items():
        o3, oa = np.full((len(df), 3), np.nan), np.full((len(df), 3), np.nan)
        ob, od = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for Y in FOLDS:
            tr, te = elig & (day < pd.Timestamp(f"{Y}-01-01").date()), elig & (year == Y)
            mc = lambda yy: lgb.LGBMClassifier(objective="multiclass", num_class=3, **LGB_CLS).fit(df.loc[tr, cols], yy[tr]).predict_proba(df.loc[te, cols])
            o3[te], oa[te] = mc(y3), mc(yact)
            ob[te] = lgb.LGBMClassifier(**LGB_CLS).fit(df.loc[tr, cols], ybig[tr]).predict_proba(df.loc[te, cols])[:, 1]
            trm = tr & (a > t1)
            od[te] = lgb.LGBMClassifier(**LGB_CLS).fit(df.loc[trm, cols], up[trm]).predict_proba(df.loc[te, cols])[:, 1]
        for nm, yrs in (("val", VAL_YEARS), ("test", TEST_YEARS)):
            s = elig & np.isin(year, yrs); mv = s & (a > t1)
            print(f"{name:11s} {nm:4s} 3-class {(o3[s].argmax(1) == y3[s]).mean():.4f}  activity {(oa[s].argmax(1) == yact[s]).mean():.4f}  "
                  f"big AUC {roc_auc_score(ybig[s], ob[s]):.4f}  direction(move rows) acc {((od[mv] > .5) == up[mv]).mean():.4f} AUC {roc_auc_score(up[mv], od[mv]):.4f}", flush=True)


if __name__ == "__main__":
    main()
