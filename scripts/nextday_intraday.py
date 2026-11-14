"""Next-day direction with intraday-structure features (from 1m bars) added to the daily panel, peer-weighted for JPM.
Regime-neutral scoring: AUC, and accuracy of a rank rule that calls Up on the top half of scores (trailing 252d median) so the
base-rate drift cannot inflate it. Selection on 2020-23, report 2024-26."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "scripts" / "daily_panel.py").read_text()
src = src[: src.index("print(len(cols)")]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--no-news", "--no-fund"]
exec(compile(src, "daily_panel_prefix", "exec"))

import lightgbm as lgb  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

UNI = BANKS + ["XLF", "SPY"]
for s in ("XLF", "SPY"):
    panel[s] = build(s)


def intraday(sym):
    d = store.load(sym).copy()
    d.index = d.index.tz_convert(B.ET)
    d = d.between_time("09:30", "15:59")
    c5 = d.close.resample("5min", offset="30min").last().dropna()
    v5 = d.volume.resample("5min", offset="30min").sum().reindex(c5.index)
    f = pd.DataFrame({"c": c5, "v": v5})
    f["date"], f["slot"] = pd.to_datetime(f.index.date), f.index.hour * 12 + f.index.minute // 5 - (9 * 12 + 6)
    f = f[(f.slot >= 0) & (f.slot < 78)]
    c = f.pivot(index="date", columns="slot", values="c").reindex(index=days, columns=range(78)).ffill(axis=1)
    v = f.pivot(index="date", columns="slot", values="v").reindex(index=days, columns=range(78)).fillna(0)
    lc = np.log(c.values)
    r = np.diff(lc, axis=1)
    vv = v.values[:, 1:]
    out = pd.DataFrame(index=days)
    out["i_first30"] = (lc[:, 6] - lc[:, 0]) * 1e4
    out["i_last30"] = (lc[:, -1] - lc[:, -7]) * 1e4
    out["i_last60"] = (lc[:, -1] - lc[:, -13]) * 1e4
    out["i_mid"] = (lc[:, -13] - lc[:, 12]) * 1e4
    out["i_late_minus_early"] = out.i_last60 - (lc[:, 12] - lc[:, 0]) * 1e4
    up, dn = (np.where(r > 0, r ** 2, 0)).sum(1), (np.where(r < 0, r ** 2, 0)).sum(1)
    out["i_semivar_asym"] = (up - dn) / (up + dn + 1e-12)
    sd = r.std(1) + 1e-12
    out["i_skew"] = ((r - r.mean(1, keepdims=True)) ** 3).mean(1) / sd ** 3
    out["i_upfrac"] = (r > 0).mean(1)
    out["i_signed_vol"] = (np.sign(r) * vv).sum(1) / (vv.sum(1) + 1)
    out["i_last_hr_vol"] = v.values[:, -12:].sum(1) / (v.values.sum(1) + 1)
    vw = (c.values * v.values).sum(1) / (v.values.sum(1) + 1)
    out["i_vs_vwap"] = (lc[:, -1] - np.log(vw + 1e-9)) * 1e4
    cum = lc - lc[:, [0]]
    bad = np.isnan(cum).all(1)
    cz = np.where(np.isnan(cum), 0, cum)
    out["i_t_high"], out["i_t_low"] = np.where(bad, np.nan, cz.argmax(1) / 78), np.where(bad, np.nan, cz.argmin(1) / 78)
    out["i_eff"] = np.abs(lc[:, -1] - lc[:, 0]) / (np.abs(r).sum(1) + 1e-12)
    out["i_last_vs_range"] = (lc[:, -1] - np.nanmin(cz,1) - lc[:, 0]) / (np.nanmax(cz,1) - np.nanmin(cz,1) + 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


intra = {s: intraday(s) for s in UNI}
icols = list(intra["JPM"].columns)
base_cols = [c for c in panel["JPM"].columns if c != "sym"]
for s in UNI:
    panel[s] = panel[s].join(intra[s])
ys = {s: (C[s].shift(-1) - C[s] > 0).astype(float).where(C[s].shift(-1).notna()) for s in UNI}
ret = C.diff()
w60 = pd.DataFrame({s: ret["JPM"].rolling(60).corr(ret[s]) for s in UNI}).clip(lower=0).fillna(0)
VAL, TEST = range(2020, 2024), range(2024, 2027)


def fit(kind, c, tr, cols):
    if kind == "lr":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(tr[cols], tr.y)
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=150, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, verbose=-1).fit(tr[cols], tr.y)


def run(kind, c, cols, roll=0):
    P = pd.DataFrame(np.nan, index=days, columns=UNI)
    for Y in list(VAL) + list(TEST):
        te = np.where(days.year == Y)[0]
        tri = np.arange(max(te[0] - 1 - roll, 0) if roll else 0, te[0] - 1)
        tr = pd.concat([panel[s].iloc[tri].reindex(columns=cols).assign(y=ys[s].iloc[tri]) for s in UNI]).dropna(subset=["y"])
        m = fit(kind, c, tr, cols)
        for s in UNI:
            P.iloc[te, P.columns.get_loc(s)] = m.predict_proba(panel[s].iloc[te].reindex(columns=cols))[:, 1]
    peer = ((P - .5) * w60).sum(1) / w60.sum(1).replace(0, np.nan) + .5
    return P["JPM"], peer


def report(name, p):
    row = [name.ljust(26)]
    for yrs in (VAL, TEST):
        mk = days.year.isin(list(yrs)) & ys["JPM"].notna().values & p.notna().values
        y, q = ys["JPM"][mk].values, p[mk]
        med = p.shift(1).rolling(252, min_periods=60).median()[mk]
        up = (q > med).values
        ok = ~np.isnan(med.values)
        acc = (up[ok] == y[ok]).mean()
        bal = 0.5 * ((up[ok] & (y[ok] == 1)).sum() / max((y[ok] == 1).sum(), 1) + (~up[ok] & (y[ok] == 0)).sum() / max((y[ok] == 0).sum(), 1))
        row.append(f"AUC {roc_auc_score(y, q):.3f}  rank-acc {acc:.3f}  bal-acc {bal:.3f}  Up {up[ok].mean():.0%}")
    print(" | ".join(row), flush=True)


print("                           |  VAL 2020-23                                  |  TEST 2024-26 (+/-0.037)")
for roll in (250, 500, 750):
    for fs, cols in (("intraday only", icols), ("base+intraday", base_cols + icols)):
        for c in (0.003, 0.03):
            own, peer = run("lr", c, cols, roll)
            report(f"roll{roll} C={c} {fs} JPM", own)
            report(f"roll{roll} C={c} {fs} PEER", peer)
