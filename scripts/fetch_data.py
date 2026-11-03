"""The only place that talks to Alpaca: download bars and the NYSE calendar into data/raw/.

Run it after a US session has closed to add that day. It is incremental: each symbol continues
from its last stored bar and stops at the last completed session close (never a partial day).

Usage: uv run python scripts/fetch_data.py [--symbols JPM SPY XLF] [--feed iex|sip] [--start 2016-01-01]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import alpaca, config, store  # noqa: E402

FIRST = {"iex": "2020-07-27", "sip": "2016-01-01"}   # earliest data each feed has on the free plan


def fetch_calendar():
    years_ahead = (pd.Timestamp.now() + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    cal = alpaca.calendar("2016-01-01", years_ahead)
    cal.to_parquet(store.calendar_path())
    print(f"calendar: {len(cal):,} sessions, {cal.date.min()} .. {cal.date.max()}", flush=True)


def fetch(symbol, feed, start):
    old = store.load(symbol, feed)
    begin = old.index.max() + pd.Timedelta(minutes=1) if len(old) else pd.Timestamp(start, tz="UTC")
    end = store.last_completed_close()
    if begin >= end:
        print(f"{symbol}: up to date (last completed close {end})", flush=True)
        return
    print(f"{symbol}: fetching {begin} .. {end} ({feed})", flush=True)
    new = alpaca.get_bars(symbol, begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          end.strftime("%Y-%m-%dT%H:%M:%SZ"), feed=feed)
    new = new[new.index < end]
    df = pd.concat([old, new]) if len(old) else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    store.save(symbol, df, feed)
    print(f"{symbol}: +{len(new):,} bars, {len(df):,} total, {df.index.min()} .. {df.index.max()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=[config.SYMBOL, *config.CONTEXT_SYMBOLS])
    ap.add_argument("--feed", default=config.FEED, choices=["iex", "sip"])
    ap.add_argument("--start", default=None, help="first date for a symbol not yet stored")
    args = ap.parse_args()
    store.calendar_path().parent.mkdir(parents=True, exist_ok=True)
    fetch_calendar()
    for sym in args.symbols:
        fetch(sym, args.feed, args.start or FIRST[args.feed])


if __name__ == "__main__":
    main()
