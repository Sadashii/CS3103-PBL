"""Next-day direction test. Features use data up to today's 16:00 close; label = sign of tomorrow's close-to-close return."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import bars as B, config, store, features, newsfeat

def daily(sym):
    d = store.load(sym).copy(); d.index = d.index.tz_convert(B.ET)
    d = d.between_time("09:30", "15:59"); g = d.groupby(d.index.date)
    o = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(), "vol": g.volume.sum()})
    o.index = pd.to_datetime(o.index); return o

J = daily("JPM"); X = pd.DataFrame(index=J.index)
r = np.log(J.close).diff() * 1e4
for k in (1, 2, 3, 5, 10, 20, 60): X[f"ret_{k}"] = np.log(J.close).diff(k) * 1e4
X["gap"] = np.log(J.open / J.close.shift(1)) * 1e4
X["intraday"] = np.log(J.close / J.open) * 1e4
X["vol_10"], X["vol_60"] = r.rolling(10).std(), r.rolling(60).std()
X["range"] = np.log(J.high / J.low) * 1e4
X["close_pos"] = (J.close - J.low) / (J.high - J.low + 1e-9)
X["rel_vol"] = J.vol / J.vol.rolling(20).mean()
X["dist_ma50"] = np.log(J.close / J.close.rolling(50).mean()) * 1e4
X["dow"] = J.index.dayofweek
for s in ("SPY", "XLF", "BAC", "C", "WFC", "GS", "MS"):
    D = daily(s); X[f"{s}_ret_1"] = np.log(D.close).diff() * 1e4; X[f"{s}_ret_5"] = np.log(D.close).diff(5) * 1e4
X["rel_spy_5"] = X.ret_5 - X.SPY_ret_5
for k in (60, 120, 252): X[f"ret_{k}"] = np.log(J.close).diff(k) * 1e4
X["vol_252"] = r.rolling(252).std()
X["dd_252"] = np.log(J.close / J.close.rolling(252).max()) * 1e4
X = X.replace([np.inf, -np.inf], np.nan)
if "--news" in sys.argv:       # news/filing/event features as known at the 16:00 close (the last 5m bar starts 15:55)
    last = pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize(B.ET) + pd.Timedelta(hours=15, minutes=55) for d in X.index]).tz_convert("UTC")
    N = newsfeat.build(last); N.index = X.index
    N = N.loc[:, N.std() > 0]
    X = X.join(N); print("added", N.shape[1], "news/event features")
print(len(X), "days", X.index[0].date(), X.index[-1].date())
print("horizon | n test days | eff. indep. n | always-up | LR acc | LGB acc   (walk-forward 2020-2026, train purged by the horizon)")
for h, name in ((1, "1 day"), (5, "1 week"), (21, "1 month"), (252, "1 year")):
    fut = np.log(J.close.shift(-h) / J.close) * 1e4
    y = (fut > 0).astype(int); ok = fut.notna().values
    pr_lr, pr_gb, yy = [], [], []
    for Y in range(2020, 2027):
        te = ok & (X.index.year == Y)
        if te.sum() < 20: continue
        first = np.where(X.index.year == Y)[0][0]
        tr = ok & (np.arange(len(X)) < first - h)          # purge: training labels end before the test period starts
        if tr.sum() < 200: continue
        lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(X[tr], y[tr])
        gb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1).fit(X[tr], y[tr])
        pr_lr += list(lr.predict_proba(X[te])[:, 1] > .5); pr_gb += list(gb.predict_proba(X[te])[:, 1] > .5); yy += list(y[te])
    yy = np.array(yy); n = len(yy)
    print(f"{name:8s}| {n:5d} | {n // h:4d} | {max(yy.mean(), 1 - yy.mean()) * 0 + yy.mean():.3f} | {(np.array(pr_lr) == yy).mean():.3f} | {(np.array(pr_gb) == yy).mean():.3f}")
