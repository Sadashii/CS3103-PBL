"""Causal 1-minute features for the live (every-update) models.

A row is indexed by the START time (UTC) of a closed 1m regular-hours bar; the forecast is made when that bar closes.
Labels (training only):  y1 = next 1m bar's close-to-close return (bps);  yD = return from this bar's close to the day's last close (bps).
Both are masked when they would cross the session close (the close time comes from the NYSE calendar, never from the data).
Every feature at row t uses bars <= t only (tests/test_causal_1m.py checks this by truncation)."""
import numpy as np
import pandas as pd

from . import bars as B
from . import config, store

SYM = config.SYMBOL
PEERS = ["BAC", "C", "WFC", "GS", "MS"]
MIN = pd.Timedelta(minutes=1)
MIN_HISTORY = 400


def rth(df):
    x = df.copy()
    et = x.index.tz_convert(B.ET)
    m = (et.hour * 60 + et.minute >= 570) & (et.hour * 60 + et.minute < 960)
    return x[m]


def _lr(c, k=1):
    return np.log(c).diff(k) * 1e4


def build(frames, sym=SYM, peers=None):
    peers = PEERS if peers is None else peers
    j = rth(frames[sym])
    idx = j.index
    et = idx.tz_convert(B.ET)
    dv = np.asarray(et.date)
    slot = pd.Series((et.hour * 60 + et.minute) - 570, index=idx)
    o, h, l, c, v, tr = (j[k].astype(float) for k in ("open", "high", "low", "close", "volume", "trades"))
    cal = store.calendar()
    close_of = pd.Series(cal["close"].values, index=pd.to_datetime(cal["date"]).dt.date)
    close_t = pd.Series(pd.DatetimeIndex(close_of.reindex(dv).values).tz_localize("UTC"), index=idx)
    left = ((close_t - (idx + MIN)) / MIN).round().astype(float)          # whole minutes still to trade after this bar

    day_open = o.groupby(dv).transform("first")
    f = pd.DataFrame(index=idx)
    for k in (1, 2, 3, 5, 10, 15, 30, 60):
        ref = c.groupby(dv).shift(k).fillna(day_open)                      # early in the day: measure from the open
        f[f"ret_{k}"] = np.log(c / ref) * 1e4
    lr1 = f.ret_1
    f["vol_15"], f["vol_60"], f["vol_390"] = lr1.rolling(15).std(), lr1.rolling(60).std(), lr1.rolling(390).std()
    f["vol_ratio"] = f.vol_15 / f.vol_390
    f["range_1"] = (h - l) / c * 1e4
    f["close_pos"] = (c - l) / (h - l + 1e-9)
    f["ret_prev_bar_abs"] = lr1.abs()

    prev = c.groupby(dv).shift(1)
    tick = np.sign(c - prev).replace(0, np.nan).groupby(dv).ffill().fillna(0)
    clv = (((c - l) - (h - c)) / (h - l).replace(0, np.nan)).fillna(0)
    for nm, d in (("tick", v * tick), ("clv", v * clv)):
        for k in (1, 5, 15):
            f[f"imb_{nm}_{k}"] = d.rolling(k).sum() / v.rolling(k).sum()
        f[f"cumimb_{nm}"] = d.groupby(dv).cumsum() / v.groupby(dv).cumsum()
    rv = v / v.groupby(slot.values).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())   # vs same minute, last 20 sessions
    f["rel_vol"] = rv
    f["trades_rel"] = tr / tr.rolling(60).mean()
    f["avg_trade_rel"] = (v / tr) / (v / tr).rolling(60).mean()

    tp = (h + l + c) / 3
    f["vwap_dist"] = (c / ((tp * v).groupby(dv).cumsum() / v.groupby(dv).cumsum()) - 1) * 1e4
    f["ret_day"] = np.log(c / day_open) * 1e4
    f["dist_day_high"] = np.log(c / h.groupby(dv).cummax()) * 1e4
    f["dist_day_low"] = np.log(c / l.groupby(dv).cummin()) * 1e4
    dclose = c.groupby(dv).last()
    prev_close = pd.Series(dclose.shift(1).reindex(dv).values, index=idx)
    f["gap"] = np.log(day_open / prev_close) * 1e4
    f["ret_vs_prev_close"] = np.log(c / prev_close) * 1e4
    f["prev_day_ret"] = pd.Series(np.log(dclose).diff().shift(1).reindex(dv).values, index=idx) * 1e4
    dvol = lr1.groupby(dv).std()
    f["prev_day_vol"] = pd.Series(dvol.shift(1).reindex(dv).values, index=idx)

    px = {s: rth(frames[s]).close.reindex(idx).ffill(limit=3) for s in ["SPY", "XLF"] + peers}
    for k in (1, 5, 15):
        f[f"spy_ret_{k}"] = _lr(px["SPY"], k)
    f["xlf_ret_5"] = _lr(px["XLF"], 5)
    f["peer_ret_5"] = pd.concat([_lr(px[s], 5) for s in peers], axis=1).mean(axis=1)
    f["peer_ret_1"] = pd.concat([_lr(px[s], 1) for s in peers], axis=1).mean(axis=1)
    f["rel_spy_5"] = f.ret_5 - f.spy_ret_5
    f["slot"] = slot
    f["left_min"] = left
    f["dow"] = et.dayofweek

    f["close"] = c
    y1 = np.log(c.shift(-1) / c) * 1e4
    f["y1"] = y1.where(left >= 1)
    f["yD"] = (np.log(c.groupby(dv).transform("last") / c) * 1e4).where(left >= 1)
    f.iloc[:MIN_HISTORY] = np.nan
    return f


def feature_names(frame):
    return [c for c in frame.columns if not c.startswith("y") and c not in ("close",)]
