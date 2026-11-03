"""Causal feature builder for the 5-minute JPM model.

One function builds features for training (whole history) and for serving (history up to the replay
cursor), so the two can never drift apart. Every feature at bar t uses only bars <= t (the bar that
just closed). `tests/test_causal.py` checks that features computed on data truncated at t equal the
features computed on the full series.

A row is indexed by the START time (UTC) of a closed 5m regular-hours bar; the signal is issued when
that bar closes, at its close price, and is graded 30 minutes later (6 bars on).
"""
import numpy as np
import pandas as pd

from . import bars as B
from . import config, newsfeat, store

SYM = config.SYMBOL
PEERS = ["BAC", "C", "WFC", "GS", "MS"]
HORIZON = 6                              # bars ahead for the graded signal (30 min)
ANCHORS = [1, 3, 6, 12, 24, 48]          # bars ahead for the rest-of-day quantile models
MIN_HISTORY = 400                        # bars before features are considered valid (~5 days)


def bars5(df):
    return B.resample(df, "5m", "rth")[["open", "high", "low", "close", "volume"]]


def _lr(c, k=1):
    return np.log(c).diff(k) * 1e4         # log-return in basis points


def _phi(z):
    from scipy.special import ndtr
    return ndtr(z)


def flow5(df):
    """Signed-volume ("delta") proxies from 1m bars, summed per 5m bar. Only SIP OHLCV is stored, so the
    buy/sell split is estimated three ways: tick rule, close-location value, and bulk volume classification.
    Everything at 1m bar t uses bars <= t only; 5m sums use the minutes inside that 5m bar."""
    et = df.copy()
    et.index = et.index.tz_convert(B.ET)
    et = et.between_time("09:30", "15:59")
    o, h, l, c, v = (et[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    dayv = np.asarray(et.index.date)
    prev = c.groupby(dayv).shift(1)
    tick = np.sign(c - prev).replace(0, np.nan).groupby(dayv).ffill().fillna(0)
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    sd = (c - o).rolling(390, min_periods=60).std()
    bvc = 2 * _phi(((c - o) / sd).fillna(0).values) - 1
    g = pd.DataFrame({"v": v, "d_tick": v * tick, "d_clv": v * clv.fillna(0), "d_bvc": v * bvc,
                      "tr": et["trades"].astype(float), "pv": v * et["vwap"].astype(float)})
    a = g.resample("5min", label="left", closed="left").sum()
    a = a[a.v > 0]
    a.index = a.index.tz_convert("UTC")
    return a


def build(frames):
    """frames: {symbol: 1m UTC frame}. Returns a DataFrame indexed by 5m bar start (UTC):
    feature columns, `close`, `bars_left`, and labels y1..y48 (bps, NaN if it would pass the close)."""
    j = bars5(frames[SYM])
    idx = j.index
    et = idx.tz_convert(B.ET)
    dayv = np.asarray(et.date)
    slot = pd.Series(((et.hour * 60 + et.minute) - 570) // 5, index=idx)
    # session close comes from the NYSE calendar (known in advance), never from the data itself
    cal = store.calendar()
    close_of = pd.Series(cal["close"].values, index=pd.to_datetime(cal["date"]).dt.date)
    close_t = pd.Series(pd.DatetimeIndex(close_of.reindex(dayv).values).tz_localize("UTC"), index=idx)
    bars_left = ((close_t - (idx + pd.Timedelta(minutes=5))) / pd.Timedelta(minutes=5)).round().astype(float)
    o, h, l, c, v = (j[k] for k in ("open", "high", "low", "close", "volume"))

    f = pd.DataFrame(index=idx)
    lr1 = _lr(c)
    for k in (1, 2, 3, 6, 12, 24):
        f[f"ret_{k}"] = _lr(c, k)
    f["vol_12"] = lr1.rolling(12).std()
    f["vol_78"] = lr1.rolling(78).std()
    f["vol_390"] = lr1.rolling(390).std()
    f["vol_ratio"] = f.vol_12 / f.vol_390
    f["range_1"] = (h - l) / c * 1e4
    f["close_pos"] = (c - l) / (h - l + 1e-9)

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-12))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c * 1e4
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - m20) / (2 * s20 + 1e-9)
    f["bb_width"] = 4 * s20 / m20 * 1e4

    tp = (h + l + c) / 3
    pv, vv = (tp * v).groupby(dayv).cumsum(), v.groupby(dayv).cumsum()
    f["vwap_dist"] = (c / (pv / vv) - 1) * 1e4

    # volume relative to the same time-of-day over the previous 20 sessions
    piv = pd.DataFrame({"d": dayv, "s": slot.values, "v": v.values}).pivot_table(index="d", columns="s", values="v")
    rm = piv.rolling(20, min_periods=5).mean().shift(1).stack()
    key = pd.MultiIndex.from_arrays([dayv, slot.values])
    f["rel_vol"] = v.values / rm.reindex(key).values

    day_open = o.groupby(dayv).transform("first")
    f["ret_day"] = np.log(c / day_open) * 1e4
    dclose = c.groupby(dayv).last()
    prev_close = pd.Series(dclose.shift(1).reindex(dayv).values, index=idx)
    f["gap"] = np.log(day_open / prev_close) * 1e4
    f["prev_day_ret"] = pd.Series(np.log(dclose).diff().shift(1).reindex(dayv).values, index=idx) * 1e4
    dvol = lr1.groupby(dayv).std()
    f["prev_day_vol"] = pd.Series(dvol.shift(1).reindex(dayv).values, index=idx)
    f["vol_vs_prev_day"] = f.vol_12 / f.prev_day_vol
    f["slot"] = slot
    f["bars_left"] = bars_left
    f["dow"] = et.dayofweek

    # market and peer context, all from the same stored feed, aligned on JPM's bars
    px = {s: bars5(frames[s]).close.reindex(idx).ffill(limit=3) for s in ["SPY", "XLF"] + PEERS}
    spy1 = _lr(px["SPY"])
    for k in (1, 3, 6):
        f[f"spy_ret_{k}"] = _lr(px["SPY"], k)
    for k in (3, 6):
        f[f"xlf_ret_{k}"] = _lr(px["XLF"], k)
    for k in (1, 3, 6):
        f[f"peer_ret_{k}"] = pd.concat([_lr(px[s], k) for s in PEERS], axis=1).mean(axis=1)
    f["rel_spy_6"] = f.ret_6 - f.spy_ret_6
    f["rel_peer_6"] = f.ret_6 - f.peer_ret_6
    f["corr_spy_36"] = lr1.rolling(36).corr(spy1)
    f["spy_vol_12"] = spy1.rolling(12).std()

    # ---- order-flow ("delta") features ----
    fl = flow5(frames[SYM]).reindex(idx)
    for nm in ("tick", "clv", "bvc"):
        d = fl[f"d_{nm}"]
        f[f"imb_{nm}_1"] = d / fl.v
        for k in (3, 6, 12):
            f[f"imb_{nm}_{k}"] = d.rolling(k).sum() / fl.v.rolling(k).sum()
        f[f"cumimb_{nm}"] = d.groupby(dayv).cumsum() / fl.v.groupby(dayv).cumsum()
    f["delta_div_6"] = f.imb_clv_6 * np.sign(f.ret_6) * -1          # flow against the price move
    f["avg_trade"] = fl.v / fl.tr
    f["avg_trade_rel"] = f.avg_trade / f.avg_trade.rolling(78).mean()
    f["trades_rel"] = fl.tr / fl.tr.rolling(78).mean()
    f["vwap5_dist"] = (fl.pv / fl.v / c - 1) * 1e4
    f["vol_accel"] = fl.v.rolling(3).sum() / fl.v.rolling(12).sum()
    spf = flow5(frames["SPY"]).reindex(idx)
    f["spy_imb_clv_6"] = spf.d_clv.rolling(6).sum() / spf.v.rolling(6).sum()
    f["spy_imb_bvc_6"] = spf.d_bvc.rolling(6).sum() / spf.v.rolling(6).sum()
    pe = pd.concat([flow5(frames[x]).reindex(idx).d_clv.rolling(6).sum() / flow5(frames[x]).reindex(idx).v.rolling(6).sum()
                    for x in PEERS], axis=1)
    f["peer_imb_clv_6"] = pe.mean(axis=1)

    # ---- intraday momentum and structure ----
    first30 = np.log(c.where(slot == 5).groupby(dayv).transform("first") / day_open) * 1e4     # open -> 10:00 close
    f["open30_ret"] = first30.where(slot >= 5, f.ret_day)                                         # before 10:00: return so far
    hi_d, lo_d = h.groupby(dayv).cummax(), l.groupby(dayv).cummin()
    f["dist_day_high"] = np.log(c / hi_d) * 1e4
    f["dist_day_low"] = np.log(c / lo_d) * 1e4
    f["day_range_pos"] = (c - lo_d) / (hi_d - lo_d + 1e-9)
    dh, dl = h.groupby(dayv).max(), l.groupby(dayv).min()
    f["dist_prev_high"] = np.log(c / pd.Series(dh.shift(1).reindex(dayv).values, index=idx)) * 1e4
    f["dist_prev_low"] = np.log(c / pd.Series(dl.shift(1).reindex(dayv).values, index=idx)) * 1e4
    f["ret_skew_78"] = lr1.rolling(78).skew()
    f["mom_390"] = _lr(c, 78)

    f = f.join(newsfeat.build(idx))
    f["close"] = c
    for k in sorted(set(ANCHORS + [HORIZON])):
        y = np.log(c.shift(-k) / c) * 1e4
        f[f"y{k}"] = y.where(bars_left >= k)
    f.iloc[:MIN_HISTORY] = np.nan
    return f


FEATURES = None


def feature_names(frame):
    return [c for c in frame.columns if not c.startswith("y") and c not in ("close",)]
