"""Local data store. Everything downstream (replay, features, model, server) reads only these files.

data/raw/<SYM>_<feed>_1min.parquet   1-minute bars, UTC index
data/raw/calendar.parquet            NYSE sessions: date, open, close (UTC)

Only scripts/fetch_data.py talks to Alpaca, to fill these files.
"""
import pandas as pd

from . import config

COLUMNS = ["open", "high", "low", "close", "volume", "trades", "vwap"]


def path(symbol, feed=config.FEED):
    return config.DATA_DIR / "raw" / f"{symbol}_{feed}_1min.parquet"


def calendar_path():
    return config.DATA_DIR / "raw" / "calendar.parquet"


def empty():
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"))


def load(symbol, feed=config.FEED):
    p = path(symbol, feed)
    return pd.read_parquet(p) if p.exists() else empty()


def save(symbol, df, feed=config.FEED):
    p = path(symbol, feed)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def calendar():
    return pd.read_parquet(calendar_path())


def last_completed_close(now=None):
    """Close time (UTC) of the most recent NYSE session that has finished.

    Days enter the store only once their regular session is over, so the replay never
    sees a partial day. 20 minutes of slack covers the free plan's 15-minute SIP delay.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    cal = calendar()
    return cal.loc[cal["close"] + pd.Timedelta(minutes=20) <= now, "close"].iloc[-1]
