"""Support / resistance levels for the replay cursor, built only from 1m bars that have closed by then.

Levels (all standard intraday reference prices that traders watch):
  PDH / PDL / PDC   prior session's regular-hours high, low, close
  P, R1, R2, S1, S2 classic floor pivots from the prior session: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
  PWH / PWL         high / low of the last 5 completed sessions
  PMH / PML         today's pre-market high / low (04:00-09:29 ET)
  ORH / ORL         opening range: high / low of 09:30-09:59, shown only once the 09:59 bar has closed
  HOD / LOD         today's high / low so far
  Swing             15-minute swing highs / lows (fractal: higher/lower than 2 bars each side) over the last 5 sessions,
                    clustered within 0.15%; strength = number of swings in the cluster

Nothing here is fitted, so there is no training data to leak. Causality comes from the inputs: `history` holds earlier days,
`today` holds only the 1m bars closed at the cursor.
"""
import numpy as np
import pandas as pd

from jpmind import bars as B

MERGE_BPS = 5.0          # levels closer than this are shown as one line (names joined)
SWING_CLUSTER = 0.0015   # 0.15%
SWING_SESSIONS = 5


def _et(df):
    out = df.copy()
    out.index = out.index.tz_convert(B.ET)
    return out


def _rth(df_et):
    return df_et.between_time("09:30", "15:59")


def _swings(rth_et, k=2):
    """15m fractal swing highs/lows -> clustered levels [(price, kind, strength)]."""
    if len(rth_et) < 2 * k + 1:
        return []
    g = rth_et.resample("15min", label="left", closed="left")
    m = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min()}).dropna()
    h, l = m["high"].values, m["low"].values
    pts = []
    for i in range(k, len(m) - k):
        win = slice(i - k, i + k + 1)
        if h[i] == h[win].max():
            pts.append(h[i])
        if l[i] == l[win].min():
            pts.append(l[i])
    pts.sort()
    clusters = []
    for p in pts:
        if clusters and p <= clusters[-1][-1] * (1 + SWING_CLUSTER):
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def compute(history, today, price=None):
    """history: 1m UTC bars of earlier days; today: today's 1m UTC bars closed at the cursor. Returns a JSON-able dict."""
    hist = _rth(_et(history))
    tod = _et(today)
    days = sorted(set(hist.index.date))
    if not days:
        return {"levels": [], "price": price}
    prev = hist[hist.index.date == days[-1]]
    week = hist[np.isin(hist.index.date, days[-SWING_SESSIONS:])]
    H, L, C = float(prev.high.max()), float(prev.low.min()), float(prev.close.iloc[-1])
    P = (H + L + C) / 3
    raw = [("PDH", H, "Prior day high"), ("PDL", L, "Prior day low"), ("PDC", C, "Prior day close"),
           ("P", P, "Floor pivot"), ("R1", 2 * P - L, "Pivot resistance 1"), ("S1", 2 * P - H, "Pivot support 1"),
           ("R2", P + (H - L), "Pivot resistance 2"), ("S2", P - (H - L), "Pivot support 2"),
           ("PWH", float(week.high.max()), "5-session high"), ("PWL", float(week.low.min()), "5-session low")]

    pre = tod.between_time("04:00", "09:29")
    if len(pre):
        raw += [("PMH", float(pre.high.max()), "Pre-market high"), ("PML", float(pre.low.min()), "Pre-market low")]
    rth_today = _rth(tod)
    opening = rth_today.between_time("09:30", "09:59")
    or_done = len(rth_today) and rth_today.index[-1].time() >= pd.Timestamp("09:59").time()
    if or_done and len(opening):
        raw += [("ORH", float(opening.high.max()), "Opening range high (09:30-10:00)"),
                ("ORL", float(opening.low.min()), "Opening range low (09:30-10:00)")]
    if len(rth_today):
        raw += [("HOD", float(rth_today.high.max()), "High of day so far"), ("LOD", float(rth_today.low.min()), "Low of day so far")]
        if price is None:
            price = float(rth_today.close.iloc[-1])
    if price is None:
        price = float(tod.close.iloc[-1]) if len(tod) else C

    for p, n in _swings(pd.concat([week, rth_today])):
        raw.append(("Swing", p, f"15m swing level ({n} touch{'es' if n > 1 else ''})"))

    # merge near-duplicates into one line so the chart stays readable
    raw.sort(key=lambda x: x[1])
    merged = []
    for name, p, desc in raw:
        if merged and abs(p / merged[-1]["price"] - 1) * 1e4 < MERGE_BPS:
            m = merged[-1]
            m["names"].append(name)
            m["desc"].append(desc)
            m["price"] = (m["price"] * (len(m["names"]) - 1) + p) / len(m["names"])
        else:
            merged.append({"names": [name], "desc": [desc], "price": p})
    out = []
    for m in merged:
        d = (m["price"] / price - 1) * 1e4
        names = [n for n in m["names"] if n != "Swing"] or ["Swing"]
        out.append({"label": "/".join(dict.fromkeys(names)), "price": round(m["price"], 2), "desc": "; ".join(dict.fromkeys(m["desc"])),
                    "side": "resistance" if d > 0 else "support", "dist_bps": round(d, 1),
                    "strength": len(m["names"]), "major": any(n in ("PDH", "PDL", "PDC", "ORH", "ORL", "PWH", "PWL") for n in m["names"])})
    sup = [x for x in out if x["side"] == "support"]
    res = [x for x in out if x["side"] == "resistance"]
    return {"price": round(price, 2), "levels": out,
            "nearest_support": sup[-1] if sup else None, "nearest_resistance": res[0] if res else None,
            "prev_day": {"high": round(H, 2), "low": round(L, 2), "close": round(C, 2), "day": str(days[-1])},
            "opening_range_done": bool(or_done)}
