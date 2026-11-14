"""Inference: saved models -> per-candle signal, rest-of-day 90% band and scenarios for one session.

The features are causal (tests/test_causal.py), so predictions for every candle of a session can be
computed in one pass and then released to the UI only up to the replay cursor.
"""
import json
from math import erf, sqrt

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import bars as B
from . import config, features

BAR = pd.Timedelta(minutes=5)


def _phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _phi_inv(p):
    from scipy.stats import norm
    return float(norm.ppf(min(max(p, 1e-4), 1 - 1e-4)))


class Predictor:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.cols = self.cfg["features"]
        self.kind = self.cfg["classifier"]
        self.lgb = lgb.Booster(model_file=str(model_dir / "cls_lgb.txt"))
        self.lr = joblib.load(model_dir / "cls_lr.joblib")
        self.iso = joblib.load(model_dir / "iso.joblib")
        self.q = {(k, q): lgb.Booster(model_file=str(model_dir / f"q_{k}_{int(q * 100):02d}.txt"))
                  for k in self.cfg["anchors"] for q in self.cfg["quantiles"]}
        self.anchors = self.cfg["anchors"]
        self.qadj = {int(k): v for k, v in self.cfg["band_q_bps"].items()}
        # optional 3-class call (Up / Neutral / Down) + big-move warning, from scripts/train_cls3.py
        self.c3 = None
        if (model_dir / "cls3.json").exists():
            self.c3 = json.loads((model_dir / "cls3.json").read_text())
            self.c3_lgb = lgb.Booster(model_file=str(model_dir / "cls3_lgb.txt"))
            self.c3_lr = joblib.load(model_dir / "cls3_lr.joblib")
            self.big = lgb.Booster(model_file=str(model_dir / "big_lgb.txt"))
            self.act = lgb.Booster(model_file=str(model_dir / "act_lgb.txt")) if (model_dir / "act_lgb.txt").exists() else None

    @property
    def cutoff(self):
        return self.cfg["cutoff"]

    def _prob(self, X):
        parts = []
        if self.kind in ("lgb", "ens"):
            parts.append(self.lgb.predict(X))
        if self.kind in ("lr", "ens"):
            parts.append(self.lr.predict_proba(X)[:, 1])
        return self.iso.predict(np.mean(parts, axis=0))

    def _prob3(self, X):
        """Rows of P(Down), P(Neutral), P(Up)."""
        X = X[self.c3["features"]]
        parts = []
        if self.c3["kind"] in ("lgb", "ens"):
            parts.append(self.c3_lgb.predict(X))
        if self.c3["kind"] in ("lr", "ens"):
            parts.append(self.c3_lr.predict_proba(X))
        return np.mean(parts, axis=0)

    def day_table(self, frames, day):
        """Predictions for every closed 5m candle of `day`. frames: 1m bars up to the end of that day."""
        f = features.build(frames).replace([np.inf, -np.inf], np.nan)
        if self.c3 is not None:                       # label and past move at the 3-class horizon (may differ from the band horizon)
            kk = self.c3["horizon_bars"]
            lc = np.log(f["close"])
            f["_yk"] = ((lc.shift(-kk) - lc) * 1e4).where(f["bars_left"] >= kk)
            f["_pk"] = ((lc - lc.shift(kk)) * 1e4).fillna(0.0)
        et = f.index.tz_convert(B.ET)
        f = f[np.asarray(et.date) == day]
        X = f[self.cols]
        p = self._prob(X)
        P = np.stack([[self.q[(k, q)].predict(X) for q in self.cfg["quantiles"]] for k in self.anchors])  # A,Q,n
        P = np.transpose(P, (2, 0, 1)).astype(float)                                                      # n,A,Q
        thr, gate = self.cfg["threshold"], self.cfg["median_gate_bps"]
        i6 = self.anchors.index(self.cfg["horizon_bars"])
        if self.c3 is not None:
            P3 = self._prob3(X)
            pbig = self.big.predict(X[self.c3["features"]])
            PA = self.act.predict(X[self.c3["features"]]) if self.act is not None else None
        rows = []
        for n, (t, r) in enumerate(f.iterrows()):
            left = int(r["bars_left"])
            lo = P[n, :, 0] - np.array([self.qadj[k] for k in self.anchors])
            hi = P[n, :, 2] + np.array([self.qadj[k] for k in self.anchors])
            med = P[n, :, 1]
            lo, hi = np.minimum(lo, med), np.maximum(hi, med)
            m6 = med[i6]
            pu = float(p[n])
            d = "No signal"
            if left >= self.cfg["horizon_bars"]:
                if pu >= thr and (gate <= 0 or m6 >= gate):
                    d = "Up"
                elif pu <= 1 - thr and (gate <= 0 or m6 <= -gate):
                    d = "Down"
            rows.append(dict(time=t, close=float(r["close"]), left=left, p_up=pu, direction=d, lean="Up" if pu >= 0.5 else "Down",
                             confidence=max(pu, 1 - pu), med6_bps=float(m6), lo=lo, med=med, hi=hi,
                             y6=r.get(f"y{self.cfg['horizon_bars']}", np.nan), rsi=float(r["rsi_14"]), ret6=float(r["ret_6"])))
            if self.c3 is not None:
                k = int(np.argmax(P3[n]))
                rows[-1].update(p_down3=float(P3[n, 0]), p_neutral=float(P3[n, 1]), p_up3=float(P3[n, 2]),
                                call=self.c3["classes"][k] if left >= self.c3["horizon_bars"] else "No call",
                                yk=float(r["_yk"]), pk=float(r["_pk"]),
                                call_conf=float(P3[n, k]), p_big=float(pbig[n]), big=bool(pbig[n] >= self.c3["big_threshold"]))
                if PA is not None:
                    ka = int(np.argmax(PA[n]))
                    rows[-1].update(act=self.c3["activity"][ka], act_conf=float(PA[n, ka]), p_quiet=float(PA[n, 0]), p_active=float(PA[n, 2]))
        return pd.DataFrame(rows).set_index("time")

    def activity_of(self, move_bps):
        """Quiet / Normal / Active for a 30-min move of this size (thresholds fixed at training time)."""
        a = abs(move_bps)
        return "Active" if a > self.c3["tau_hi_bps"] else "Normal" if a > self.c3["tau_bps"] else "Quiet"

    def realised(self, y6):
        """Class the 30-min move actually fell in (uses the tau fixed at training time)."""
        tau = self.c3["tau_bps"]
        return "Up" if y6 > tau else "Down" if y6 < -tau else "Neutral"

    # ---- per-row products ----

    def band(self, row, tf_sec=300):
        """Lower/median/upper price at each future 5m candle until the close, with display times."""
        left = row["left"]
        if left < 1:
            return {"times": [], "upper": [], "median": [], "lower": []}
        a = np.array([k for k in self.anchors if k <= left] or [self.anchors[0]], dtype=float)
        idx = [self.anchors.index(int(k)) for k in a]
        xs = np.concatenate([[0.0], np.sqrt(a)])
        steps = np.arange(1, left + 1)
        out = {}
        for name, arr in (("lower", row["lo"]), ("median", row["med"]), ("upper", row["hi"])):
            ys = np.concatenate([[0.0], arr[idx]])
            v = np.interp(np.sqrt(steps), xs, ys)
            beyond = steps > a[-1]
            v[beyond] = ys[-1] * np.sqrt(steps[beyond] / a[-1])          # square-root-of-time beyond the last anchor
            out[name] = v
        out["lower"], out["upper"] = np.minimum(out["lower"], out["median"]), np.maximum(out["upper"], out["median"])
        t0 = B.to_display(row.name)
        px = lambda v: [round(float(row["close"] * np.exp(x / 1e4)), 2) for x in v]
        return {"times": [int(t0 + j * tf_sec) for j in steps], "upper": px(out["upper"]),
                "median": px(out["median"]), "lower": px(out["lower"])}

    def scenarios(self, row):
        left = row["left"]
        i6 = self.anchors.index(self.cfg["horizon_bars"])
        sigma = max((row["hi"][i6] - row["lo"][i6]) / (2 * 1.645), 1e-3)           # 30-min sigma, bps
        z = _phi_inv(row["p_up"])
        p_bull, p_bear = 1 - _phi(0.5 - z), _phi(-0.5 - z)
        probs = dict(bull=p_bull, base=max(0.0, 1 - p_bull - p_bear), bear=p_bear)
        band = self.band(row)
        last = row["close"]
        paths, an = {}, self.cfg["analogs"]
        for key, zz in (("bull", 0.67), ("base", 0.0), ("bear", -0.67)):
            m = np.array(band["median"])
            up = np.array(band["upper"]) - m
            dn = m - np.array(band["lower"])
            path = m + (zz * up / 1.645 if zz > 0 else zz * dn / 1.645)
            paths[key] = [round(float(x), 2) for x in path]
        a = {"bull": an["up"], "base": an["flat"], "bear": an["down"]}
        move = {"bull": f"+{0.5 * sigma:.0f}", "base": f"within +/-{0.5 * sigma:.0f}", "bear": f"-{0.5 * sigma:.0f}"}
        text = {k: (f"30-min move {move[k]} bps or more. Historically after a move like this the next 30 min went up "
                    f"{a[k]['next_up'] * 100:.0f}% of the time (mean {a[k]['next_mean_bps']:+.1f} bps, n={a[k]['n']:,})."
                    if k != "base" else
                    f"30-min move {move[k]} bps. Historically after a quiet half hour the next one went up "
                    f"{a[k]['next_up'] * 100:.0f}% (mean {a[k]['next_mean_bps']:+.1f} bps, n={a[k]['n']:,}).")
                for k in probs}
        return {k: {"path": paths[k], "probability": round(float(probs[k]), 3), "description": text[k]} for k in probs}
