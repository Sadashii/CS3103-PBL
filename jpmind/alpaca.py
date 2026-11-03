"""Alpaca REST client. Used only by scripts/fetch_data.py to fill the local store."""
import time

import pandas as pd
import requests

from . import config, store

HEADERS = {
    "APCA-API-KEY-ID": config.ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
}


def get_bars(symbol, start, end=None, feed=config.FEED, timeframe="1Min"):
    """All bars between start and end (ISO strings, UTC), following pagination."""
    params = {"timeframe": timeframe, "start": start, "feed": feed,
              "limit": 10000, "adjustment": "all", "sort": "asc"}
    if end:
        params["end"] = end
    rows = []
    while True:
        r = requests.get(f"{config.DATA_REST}/stocks/{symbol}/bars",
                         headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        js = r.json()
        rows.extend(js.get("bars") or [])
        token = js.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    return to_frame(rows)


def to_frame(rows):
    if not rows:
        return store.empty()
    df = pd.DataFrame(rows).rename(columns={
        "t": "time", "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "n": "trades", "vw": "vwap"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")[store.COLUMNS]


def calendar(start, end):
    """NYSE sessions between two dates (YYYY-MM-DD): date, open and close as UTC timestamps."""
    r = requests.get("https://paper-api.alpaca.markets/v2/calendar", headers=HEADERS,
                     params={"start": start, "end": end}, timeout=30)
    r.raise_for_status()
    cal = pd.DataFrame(r.json())
    for col in ("open", "close"):
        cal[col] = pd.to_datetime(cal["date"] + " " + cal[col]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    return cal[["date", "open", "close"]]
