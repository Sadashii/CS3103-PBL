"""Live 1-minute models: (A) the next minute and (B) the rest of today, refreshed on every closed 1m bar.

Features come from features1m.build (causal, truncation-tested). Predictions for a whole replay day are computed once and cached,
but a row is only released when its bar has closed at the cursor, so nothing from the future is ever shown."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import bars as B
from . import config, features1m as F


class TickModel:
    def __init__(self, d=None):
        d = d or config.MODEL_DIR
        self.cfg = json.load(open(d / "tick_config.json"))
        self.cols = self.cfg["features"]
        self.cls = lgb.Booster(model_file=str(d / "tick_cls.txt"))
        import joblib
        self.iso = joblib.load(d / "tick_iso.joblib")
        self.q1 = [lgb.Booster(model_file=str(d / f"tick_q1_{int(q * 100):02d}.txt")) for q in self.cfg["Q1"]]
        self.qD = [lgb.Booster(model_file=str(d / f"tick_qD_{int(q * 100):02d}.txt")) for q in self.cfg["QD"]]
        self._cache = {}

    def day_rows(self, frames, day):
        """Predictions for every closed 1m bar of `day` (index = bar open time, UTC)."""
        if day in self._cache:
            return self._cache[day]
        end = pd.Timestamp(day, tz=B.ET).tz_convert("UTC") + pd.Timedelta(days=1)
        fr = {s: d[(d.index >= end - pd.Timedelta(days=80)) & (d.index < end)] for s, d in frames.items()}
        f = F.build(fr)
        f = f[(f.index >= end - pd.Timedelta(days=1))].replace([np.inf, -np.inf], np.nan)
        X = f[self.cols]
        ok = X.notna().all(axis=1).values
        out = pd.DataFrame(index=f.index)
        out["close"] = f.close
        out["left_min"] = f.left_min
        out["ok"] = ok
        pu = np.full(len(f), np.nan)
        if ok.any():
            pu[ok] = self.iso.predict(self.cls.predict(X[ok]))
        out["p_up"] = pu
        a1 = self.cfg["q1_adj_bps"]
        q1 = np.full((len(f), 3), np.nan)
        qd = np.full((len(f), len(self.qD)), np.nan)
        if ok.any():
            q1[ok] = np.sort(np.column_stack([m.predict(X[ok]) for m in self.q1]), axis=1)
            qd[ok] = np.sort(np.column_stack([m.predict(X[ok]) for m in self.qD]), axis=1)
        out["m_lo"], out["m_med"], out["m_hi"] = q1[:, 0] - a1, q1[:, 1], q1[:, 2] + a1
        for k, q in enumerate(self.cfg["QD"]):
            out[f"d{int(q * 100):02d}"] = qd[:, k]
        self._cache[day] = out
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        return out

    def at(self, rows, cursor):
        """Forecast made when the bar opened at `cursor` closed, or None."""
        r = rows[rows.index <= cursor]
        if r.empty:
            return None
        t = r.index[-1]
        r = r.iloc[-1]
        if not r.ok or r.left_min < 1:        # nothing left to forecast once the last bar of the session has closed
            return None
        px = float(r.close)
        price = lambda bps: round(px * float(np.exp(bps / 1e4)), 2)
        a = self.cfg["qD_adj_bps"]
        QD = self.cfg["QD"]
        dq = np.array([r[f"d{int(q * 100):02d}"] for q in QD])
        # widen the outer/inner pairs by the validation conformal margin
        lo90, hi90 = dq[0] - a["90"], dq[-1] + a["90"]
        lo50, hi50 = dq[2] - a["50"], dq[4] + a["50"]
        p_above = float(1 - np.interp(0.0, dq, QD, left=0.02, right=0.98))      # P(close > current price)
        p = float(r.p_up)
        m = {"p_up": round(p, 4), "p_down": round(1 - p, 4), "median_move_bps": round(float(r.m_med), 2),
             "expected_price": price(float(r.m_med)), "band90": {"lower": price(float(r.m_lo)), "upper": price(float(r.m_hi))}}
        d = {"p_close_above_now": round(p_above, 3), "p_close_below_now": round(1 - p_above, 3),
             "median_close": price(float(dq[3])), "expected_close": price(float(dq[3])),
             "band90": {"lower": price(lo90), "upper": price(hi90)}, "band50": {"lower": price(lo50), "upper": price(hi50)},
             "minutes_left": int(r.left_min)}
        return {"price": round(px, 2), "timestamp": B.to_display(t) + 60, "next_minute": m, "to_close": d}
