"""Two experiments on the cached 5m table (walk-forward by year, choices on validation 2022-24, test 2025-26 untouched).
A. Horizon: 3-class call (tercile Neutral band fixed on 2016-21) at 10/15/30/60/120 minutes vs naive baselines, incl. single-day stats.
B. Selective direction at 30 min: classifier trained on move rows only; is there a subset of candles (confidence margin x condition)
   where Up/Down calls make money after 2 bps? Rule picked on validation, reported on test, plus a nested check (t >= 3 required).
Usage: uv run python scripts/exp_horizon_direction.py"""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import log_loss
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from jpmind import bars as B, features  # noqa: E402
from train import FOLDS, LGB_CLS, TEST_YEARS, VAL_YEARS, load_dataset  # noqa: E402
P = dict(LGB_CLS, n_estimators=200)
COST = 2.0


def tstat(x):
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def main():
    df = load_dataset(False).replace([np.inf, -np.inf], np.nan)
    cols = features.feature_names(df)
    et = df.index.tz_convert(B.ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    ok = (day <= pd.Timestamp("2026-10-27").date()) & df[cols].notna().all(axis=1).values
    lc = np.log(df["close"])
    X = df[cols]
    print("== A. horizon ==", flush=True)
    for k in (2, 3, 6, 12, 24):
        ret = ((lc.shift(-k) - lc) * 1e4).where(df["bars_left"] >= k).values
        past = ((lc - lc.shift(k)) * 1e4).values
        elig = ok & np.isfinite(ret)
        tau = float(np.quantile(np.abs(ret[elig & (year <= 2021)]), 1 / 3))
        cls = lambda a: np.where(a > tau, 2, np.where(a < -tau, 0, 1))
        y = cls(ret)
        oof = np.full((len(df), 3), np.nan)
        for Y in FOLDS:
            tr, te = elig & (day < pd.Timestamp(f"{Y}-01-01").date()), elig & (year == Y)
            oof[te] = lgb.LGBMClassifier(objective="multiclass", num_class=3, **P).fit(X[tr], y[tr]).predict_proba(X[te])
        pred = oof.argmax(1)
        bl = {"neutral": np.ones(len(df), int), "up": np.full(len(df), 2), "persist": cls(np.nan_to_num(past)),
              "rsi": np.where(df["rsi_14"].values < 30, 2, np.where(df["rsi_14"].values > 70, 0, 1))}
        for name, yrs in (("val", VAL_YEARS), ("test", TEST_YEARS)):
            s = elig & np.isin(year, yrs)
            acc = float((pred[s] == y[s]).mean())
            b = {n: float((v[s] == y[s]).mean()) for n, v in bl.items()}
            d = pd.DataFrame({"d": day[s], "m": pred[s] == y[s], **{n: v[s] == y[s] for n, v in bl.items()}}).groupby("d").mean()
            mv = s & (pred != 1)
            sg = np.where(pred[mv] == 2, 1, -1)
            print(f"{k * 5:4d}min {name:4s} tau {tau:5.1f} acc {acc:.4f} best-baseline {max(b.values()):.4f} ({max(b, key=b.get)}) edge {acc - max(b.values()):+.4f} "
                  f"logloss {log_loss(y[s], oof[s], labels=[0, 1, 2]):.4f} | day: p10 {d.m.quantile(.1):.3f} beats neutral {(d.m > d.neutral).mean():.2f} "
                  f"persist {(d.m > d.persist).mean():.2f} all4 {((d.m > d.neutral) & (d.m > d.up) & (d.m > d.persist) & (d.m > d.rsi)).mean():.2f} | "
                  f"dir hit {(sg * ret[mv] > 0).mean():.4f} net {np.mean(sg * ret[mv] - COST):+.2f}bps", flush=True)

    print("== B. selective direction (30 min) ==", flush=True)
    ret = df["y6"].values
    elig = ok & np.isfinite(ret)
    pre = elig & (year <= 2021)
    tau, tb = float(np.quantile(np.abs(ret[pre]), 1 / 3)), float(np.quantile(np.abs(ret[pre]), 0.8))
    up, big = (ret > 0).astype(int), (np.abs(ret) > tb).astype(int)
    p, pb, pl = np.full(len(df), np.nan), np.full(len(df), np.nan), np.full(len(df), np.nan)
    rng = np.random.default_rng(0)
    for Y in FOLDS:
        tr, te = elig & (day < pd.Timestamp(f"{Y}-01-01").date()), elig & (year == Y)
        trm = tr & (np.abs(ret) > tau)
        p[te] = lgb.LGBMClassifier(**P).fit(X[trm], up[trm], sample_weight=np.minimum(np.abs(ret[trm]), 60)).predict_proba(X[te])[:, 1]
        pb[te] = lgb.LGBMClassifier(**P).fit(X[tr], big[tr]).predict_proba(X[te])[:, 1]
        pl[te] = lgb.LGBMClassifier(**P).fit(X[trm], rng.permutation(up[trm])).predict_proba(X[te])[:, 1]     # placebo
    slot, gap = df["slot"].values, np.abs(df["gap"].values)
    g80 = np.nanquantile(gap[pre], 0.8)
    conds = {"all": np.ones(len(df), bool), "first30": slot < 6, "last60": df["bars_left"].values <= 18, "midday": (slot >= 24) & (slot < 54),
             "big_pred": pb >= 0.30, "quiet_pred": pb < 0.12, "gap_day": gap > g80, "earn_today": df["earn_today"].values > 0,
             "fomc_today": df["fomc_today"].values > 0}
    margins = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)

    def stats(prob, mask, m):
        s = mask & (np.abs(prob - 0.5) >= m)
        if s.sum() < 2:
            return 0, np.nan, np.nan, np.nan, np.array([])
        sg = np.where(prob[s] > 0.5, 1, -1)
        net = sg * ret[s] - COST
        return int(s.sum()), float((sg * ret[s] > 0).mean()), float(net.mean()), tstat(net), net

    val, test = elig & np.isin(year, VAL_YEARS), elig & np.isin(year, TEST_YEARS)
    mvt = test & (np.abs(ret) > tau)
    print(f"direction acc on test move rows: {((p[mvt] > .5) == up[mvt]).mean():.4f} (placebo {((pl[mvt] > .5) == up[mvt]).mean():.4f}); "
          f"all test rows {((p[test] > .5) == up[test]).mean():.4f}", flush=True)
    rows = []
    for cn, cm in conds.items():
        for m in margins:
            nv, hv, bv, tv, _ = stats(p, val & cm, m)
            nt, ht, bt, tt, _ = stats(p, test & cm, m)
            rows.append((cn, m, nv, hv, bv, tv, nt, ht, bt, tt))
    r = pd.DataFrame(rows, columns="cond margin n_val hit_val net_val t_val n_test hit_test net_test t_test".split())
    print(r[r.n_val >= 300].sort_values("t_val", ascending=False).head(12).round(3).to_string(index=False), flush=True)
    print("gross (before cost) hit rate by margin, all candles, test:",
          {m: round(stats(p, test, m)[1], 4) for m in margins}, flush=True)
    nets = []
    for Y in (2024, 2025, 2026):
        pastm, cur = elig & (year >= 2022) & (year < Y), elig & (year == Y)
        best, bt_ = None, -1e9
        for cn, cm in conds.items():
            for m in margins:
                n, _, _, t, _ = stats(p, pastm & cm, m)
                if n >= 200 and t > bt_:
                    best, bt_ = (cn, m), t
        n, h, b, t, net = stats(p, cur & conds[best[0]], best[1])
        print(f"nested {Y}: rule {best} (past t {bt_:.2f}) -> n {n} hit {h:.4f} net {b:+.2f} t {t:.2f}", flush=True)
        nets += list(net)
    nets = np.array(nets)
    print(f"nested pooled: n {len(nets)} net {nets.mean():+.2f} bps t {tstat(nets):.2f} (edge only if t >= 3)", flush=True)


if __name__ == "__main__":
    main()
