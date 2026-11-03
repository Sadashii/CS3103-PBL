"""Timeframe resampling and display times.

Bars are stored with UTC timestamps. Bins and the regular session are defined in
US Eastern time (where the exchange lives), and the UI shows ET too.
"""
import pandas as pd

ET = "America/New_York"

TIMEFRAMES = {  # label -> pandas rule
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "60min", "4h": "240min", "1D": "1D", "1W": "W-MON",
}
DAILY = ("1D", "1W")


def resample(df, tf, session="rth"):
    """1-minute UTC bars -> OHLCV bars for tf, binned on the ET clock. Index stays UTC."""
    et = df.copy()
    et.index = et.index.tz_convert(ET)
    if session == "rth":
        et = et.between_time("09:30", "15:59")
    rule = TIMEFRAMES[tf]
    if rule != "1min" and len(et):
        kw = {"label": "left", "closed": "left"}
        if rule in ("60min", "240min") and session == "rth":
            kw["offset"] = "30min"   # hourly bars start at the 9:30 ET open
        g = et.resample(rule, **kw)
        et = pd.DataFrame({
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
            "close": g["close"].last(), "volume": g["volume"].sum(), "trades": g["trades"].sum(),
        }).dropna(subset=["open"])
    et.index = et.index.tz_convert("UTC")
    return et[["open", "high", "low", "close", "volume", "trades"]]


def merge(a, b):
    """Concatenate two resampled frames, combining a bin that both contain (e.g. the current week)."""
    if not len(a):
        return b
    if not len(b):
        return a
    if a.index[-1] == b.index[0]:
        x, y = a.iloc[-1], b.iloc[0]
        joined = pd.DataFrame([{
            "open": x.open, "high": max(x.high, y.high), "low": min(x.low, y.low), "close": y.close,
            "volume": x.volume + y.volume, "trades": x.trades + y.trades}], index=b.index[:1])
        return pd.concat([a.iloc[:-1], joined, b.iloc[1:]])
    return pd.concat([a, b])


def display_times(index, tf):
    """Chart time for each bar: ET wall clock as epoch seconds; D/W bars use their trading date."""
    local = index.tz_convert(ET).tz_localize(None)
    if tf in DAILY:
        local = local.normalize()
    # Index is datetime64[us] (microseconds), so divide by 10^6, not 10^9
    return (local.astype("int64") // 10**6).astype(int)


def to_display(ts):
    """UTC timestamp -> ET wall-clock epoch seconds (what the UI shows)."""
    return int(ts.tz_convert(ET).tz_localize(None).timestamp())


def from_display(seconds):
    """ET wall-clock epoch seconds from the UI -> UTC timestamp."""
    return pd.Timestamp(int(seconds), unit="s").tz_localize(ET).tz_convert("UTC")


def to_json(frame, tf):
    times = display_times(frame.index, tf)
    return [
        {"time": int(t), "open": o, "high": h, "low": l, "close": c, "volume": float(v)}
        for t, o, h, l, c, v in zip(times, frame["open"], frame["high"], frame["low"],
                                    frame["close"], frame["volume"])
    ]
