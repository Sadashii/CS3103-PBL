"""Next-trading-day outlook for JPM from many past days.

  sigma_hat = HAR forecast of tomorrow's realized volatility (Ridge on log RV of today / last 5 / last 22 days, pooled over six banks),
              optionally extended with overnight-gap features (|overnight| today, 5-day and 22-day mean)
  mu_hat    = optional heavily shrunk drift from overnight / intraday returns of the last 1, 5, 20 days (in units of sigma_hat)
  z         = (close-to-close return of day t+1 - mu_hat(t)) / sigma_hat(t), collected over TRAINING days only (JPM)
  distribution of tomorrow's log return = sigma_hat * (z + mu_hat)   (empirical: fat tails and base-rate drift are kept)
  outputs: expected return, 90% / 50% bands, P(up), P(down), P(move beyond +-0.5 / 1 / 2%).

Everything for base day d uses data up to d's close only. The overnight return of day d (open_d / close_{d-1}) is known at d's open."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import bars as B
from . import config, features, store

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
THRESH = [0.005, 0.01, 0.02]
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
VOL_BASE = ["d", "w", "m"]
VOL_ON = ["aon_d", "aon_w", "aon_m"]
DRIFT = ["on_1", "on_5", "on_20", "id_1", "id_5", "id_20"]


def daily_frame(df1m):
    """One row per session: rv (sum of squared 5m returns inside the session), open, close, overnight and intraday log returns."""
    j = features.bars5(df1m)
    et = j.index.tz_convert(B.ET)
    d = np.asarray(et.date)
    r = np.log(j.close).diff()
    r = r.where(d == pd.Series(d, index=j.index).shift(1).values)          # never count the overnight jump inside RV
    g = (r ** 2).groupby(d)
    out = pd.DataFrame({"rv": g.sum(), "n": g.count(), "open": j.open.groupby(d).first(), "close": j.close.groupby(d).last()})
    out = out[out.n >= 60].drop(columns="n")
    out["on"] = np.log(out.open / out.close.shift(1)) * 1.0
    out["id"] = np.log(out.close / out.open)
    out.index = pd.to_datetime(out.index)
    return out


def xfeat(D):
    """All model inputs for each session, known at that session's close."""
    lv = np.log(D.rv)
    X = pd.DataFrame({"d": lv, "w": np.log(D.rv.rolling(5).mean()), "m": np.log(D.rv.rolling(22).mean())})
    a = D.on.abs()
    X["aon_d"], X["aon_w"], X["aon_m"] = np.log(a + 1e-5), np.log(a.rolling(5).mean() + 1e-5), np.log(a.rolling(22).mean() + 1e-5)
    for k in (1, 5, 20):
        X[f"on_{k}"] = D.on.rolling(k).sum() * 1e4
        X[f"id_{k}"] = D["id"].rolling(k).sum() * 1e4
    return X


class NextDay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.z = np.sort(np.asarray(cfg["z"]))

    # ---- training: uses sessions up to and including `cutoff` only ----
    @classmethod
    def fit(cls, frames, cutoff, overnight=True):
        cutoff = pd.Timestamp(cutoff)
        vcols = VOL_BASE + (VOL_ON if overnight else [])
        D = {s: daily_frame(frames[s]) for s in BANKS}
        D = {s: d[d.index <= cutoff] for s, d in D.items()}
        X = {s: xfeat(d) for s, d in D.items()}
        tr = []
        for s in BANKS:
            t = X[s][vcols].copy()
            t["y"] = np.log(D[s].rv).shift(-1)
            tr.append(t.dropna())
        tr = pd.concat(tr)
        vm = Ridge(alpha=1.0).fit(tr[vcols], tr.y)
        # drift model on z-units, pooled over banks, standardised, strongly shrunk
        pool = []
        for s in BANKS:
            x = X[s]
            sig = np.sqrt(np.exp(vm.intercept_ + x[vcols].fillna(0).values @ vm.coef_))
            z = np.log(D[s].close.shift(-1) / D[s].close) / sig
            t = x[DRIFT].copy()
            t["z"], t["sym"] = z, s
            pool.append(t.dropna())
        pool = pd.concat(pool)
        cfg = {"vcols": vcols, "vcoef": vm.coef_.tolist(), "vint": float(vm.intercept_), "cutoff": str(cutoff.date()),
               "drift": None}
        if overnight:
            mu, sd = pool[DRIFT].mean(), pool[DRIFT].std()
            dm = Ridge(alpha=3000.0).fit(((pool[DRIFT] - mu) / sd).values, pool.z.values)
            cfg["drift"] = {"cols": DRIFT, "mu": mu.tolist(), "sd": sd.tolist(), "coef": dm.coef_.tolist(), "int": float(dm.intercept_)}
        # residual distribution for JPM only
        j = pool[pool.sym == "JPM"]
        cfg["z"] = (j.z - cls._drift(cfg, j[DRIFT])).round(5).tolist() if overnight else j.z.round(5).tolist()
        return cls(cfg)

    @staticmethod
    def _drift(cfg, x):
        d = cfg["drift"]
        if d is None:
            return 0.0
        z = (x[d["cols"]].values - np.array(d["mu"])) / np.array(d["sd"])
        return z @ np.array(d["coef"]) + d["int"] * 0.0              # intercept stays in the empirical z, so only the slope part shifts

    def save(self, path):
        json.dump(self.cfg, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    @property
    def cutoff(self):
        return self.cfg["cutoff"]

    # ---- inference ----
    def params(self, X):
        """sigma_hat and mu_hat (z-units) for the rows of the feature frame X."""
        c = self.cfg
        sig = np.sqrt(np.exp(c["vint"] + X[c["vcols"]].values @ np.array(c["vcoef"])))
        mu = self._drift(c, X) if c["drift"] else np.zeros(len(X))
        return sig, mu

    def outlook(self, df1m_jpm, base_day, target_day=None):
        """df1m_jpm: JPM 1m bars; only sessions <= base_day are used."""
        base_day = pd.Timestamp(base_day)
        D = daily_frame(df1m_jpm)
        D = D[D.index <= base_day]
        if len(D) < 30 or D.index[-1] != base_day:
            return None
        X = xfeat(D).iloc[[-1]]
        sig, mu = self.params(X.fillna(0))
        s, m = float(sig[0]), float(np.atleast_1d(mu)[0])
        px = float(D.close.iloc[-1])
        ret = s * (self.z + m)                                    # empirical distribution of tomorrow's log return
        q = {f"q{int(p * 100):02d}": float(np.quantile(ret, p)) for p in QS}
        price = lambda r: round(px * float(np.exp(r)), 2)
        pr = {"p_up": float((ret > 0).mean()), "p_down": float((ret < 0).mean())}
        for t in THRESH:
            pr[f"p_above_{int(t * 1000)}"] = float((ret > np.log(1 + t)).mean())
            pr[f"p_below_{int(t * 1000)}"] = float((ret < np.log(1 - t)).mean())
        return {
            "base_day": str(base_day.date()), "target_day": str(pd.Timestamp(target_day).date()) if target_day is not None else None,
            "base_close": round(px, 2), "sigma_bps": round(s * 1e4, 1), "expected_return_bps": round(float(ret.mean()) * 1e4, 1),
            "expected_close": price(float(ret.mean())),
            "band90": {"lower": price(q["q05"]), "upper": price(q["q95"])}, "band50": {"lower": price(q["q25"]), "upper": price(q["q75"])},
            "median_close": price(q["q50"]), "probabilities": {k: round(v, 3) for k, v in pr.items()},
            "overnight_features": bool(self.cfg["drift"]), "history_days": int(len(D)), "trained_through": self.cutoff,
            "note": "No directional edge was found at any horizon; P(up) reflects the base rate, the band reflects forecast volatility.",
        }
