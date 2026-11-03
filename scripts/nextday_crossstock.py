"""Next-day JPM direction from OTHER stocks and their correlation structure, walk-forward by year (2019-2026), no tuning after seeing results.
Variants (all use only data up to day t's close; label = JPM close(t+1) > close(t)):
  A  ridge/logistic on today's and last-5-day returns of the 7 other symbols (lead-lag)
  B  rolling-correlation-weighted peer signal: sum_j corr_j(past 250d) * today's standardised return of j
  C  PCA factors of the 8-symbol return matrix (fit on past data only) -> logistic
  D  residual (idiosyncratic) reversal: JPM return minus beta*market, past 250d beta
Calls are balanced (threshold = median of training predictions), so none can be 'always up'.
Usage: uv run python scripts/nextday_crossstock.py"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import nextday as N, store  # noqa: E402

SYMS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "SPY", "XLF"]
px = pd.DataFrame({s: N.daily_frame(store.load(s)).close for s in SYMS}).dropna()
r = np.log(px).diff()
z = r / r.rolling(60).std()
y = (px.JPM.shift(-1) > px.JPM).astype(float).where(px.JPM.shift(-1).notna())
others = [s for s in SYMS if s != "JPM"]
yr = px.index.year.values

feats = {}
A = pd.concat([z[others].add_prefix("z_"), r[others].rolling(5).sum().add_prefix("w5_")], axis=1)
feats["A lead-lag of 7 other symbols"] = A
# B: rolling correlation-weighted signal (correlation of JPM(t+1) with j(t) over the past 250 days)
cw = pd.DataFrame(index=px.index)
for s in others:
    cw[s] = r.JPM.shift(-1).rolling(250).corr(r[s]).shift(1)        # shift: only past pairs (t+1 outcomes already known by t)
cw = cw.fillna(0)
B = pd.DataFrame({"corr_weighted": (cw * z[others]).sum(axis=1), "own_z": z.JPM, "mkt_z": z.SPY})
feats["B correlation-weighted peer signal"] = B
# C: PCA factors fit on past returns only (expanding window refit yearly)
Cf = pd.DataFrame(index=px.index, columns=["pc1", "pc2", "pc3", "pc4"], dtype=float)
for Y in range(2018, 2027):
    past = z[yr < Y].dropna()
    if len(past) < 250:
        continue
    pca = PCA(4).fit(past)
    m = (yr == Y)
    Cf.loc[m] = pca.transform(z[m].fillna(0))
feats["C PCA factors (8 symbols)"] = Cf.astype(float)
# D: idiosyncratic residual reversal
beta = r.JPM.rolling(250).cov(r.SPY).shift(1) / r.SPY.rolling(250).var().shift(1)
resid = r.JPM - beta * r.SPY
D = pd.DataFrame({"resid1": resid * 1e4, "resid5": resid.rolling(5).sum() * 1e4, "resid20": resid.rolling(20).sum() * 1e4})
feats["D idiosyncratic residual"] = D

mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
print("variant | pooled acc 2019-26 | up-rate | years > 50% | years beating up-rate | acc 2025-26 | pooled acc on top-25% confident")
for name, X in feats.items():
    d = X.join(y.rename("y")).dropna()
    yy = d.index.year.values
    tot = []; conf_hit = []
    for Y in range(2019, 2027):
        tr, te = yy < Y, yy == Y
        if te.sum() < 50:
            continue
        m = mk().fit(d.drop(columns="y")[tr], d.y[tr])
        thr = np.median(m.predict_proba(d.drop(columns="y")[tr])[:, 1])
        p = m.predict_proba(d.drop(columns="y")[te])[:, 1]
        yt = d.y[te].values
        hit = ((p > thr) == (yt > 0))
        q = np.abs(p - thr)
        tot.append((te.sum(), hit.mean(), yt.mean(), Y))
        conf_hit.append(hit[q >= np.quantile(q, .75)])
    n = np.array([t[0] for t in tot]); a = np.array([t[1] for t in tot]); u = np.array([t[2] for t in tot]); ys = np.array([t[3] for t in tot])
    pooled = (n * a).sum() / n.sum()
    recent = (n * a)[ys >= 2025].sum() / n[ys >= 2025].sum()
    ch = np.concatenate(conf_hit).mean()
    print(f"{name:40s} | {pooled:.4f} | {(n * u).sum() / n.sum():.4f} | {int((a > .5).sum())}/{len(a)} | {int((a > u).sum())}/{len(a)} | {recent:.4f} | {ch:.4f}")
