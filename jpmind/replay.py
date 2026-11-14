"""ReplayService: serves a chosen complete US session as if it were live.

Everything before that session is static history. Within the session a cursor moves
one minute at a time (one bar per real minute at 1x). Consumers only ever see bars at
or before the cursor, so nothing downstream (chart or model) can look ahead.
"""
import asyncio
import time
from datetime import date

import pandas as pd

from . import bars as B
from . import config

MIN = pd.Timedelta(minutes=1)
SPEEDS = [1, 2, 5, 10, 30, 60]

# Training cutoff (D30): read from the deployed model, so retraining with a later --cutoff moves the replayable days with it
TRAINING_CUTOFF = pd.Timestamp(config.trained_cutoff()).date()

# Timeframe to seconds mapping for "closed candle" filtering
TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1D": 86400,
    "1W": 604800,
}


class ReplayService:
    def __init__(self, bars_1m):
        self.listeners = []          # async callables(event dict)
        self.speed = 1
        self.playing = False
        self._task = None
        self._hist_cache = {}
        self._all_days = None
        self._replayable_days = None
        self._bars_1m = bars_1m
        self._load_all_days()
        # Load the last replayable day by default
        self.load_day(self._replayable_days[-1])

    # ---------- day listing ----------

    def _load_all_days(self):
        """Find all complete trading days in the stored data (sessions that reached 15:59 ET)."""
        et = self._bars_1m.index.tz_convert(B.ET)
        closes = et[(et.hour == 15) & (et.minute == 59)]
        self._all_days = sorted(closes.normalize().unique().date)
        # Only days after training cutoff are replayable (D30, Z4)
        self._replayable_days = [d for d in self._all_days if d > TRAINING_CUTOFF]

    def list_days(self):
        """Return list of replayable days as ISO strings (most recent first)."""
        return [d.isoformat() for d in reversed(self._replayable_days)]

    # ---------- data ----------

    def load_day(self, day: date | str):
        """Load a specific trading day for replay."""
        if isinstance(day, str):
            day = date.fromisoformat(day)
        if day not in self._replayable_days:
            raise ValueError(f"Day {day} not in replayable days (must be after training cutoff {TRAINING_CUTOFF})")
        self.day = day
        start = pd.Timestamp(self.day, tz=B.ET).tz_convert("UTC")
        end = start + pd.Timedelta(days=1)
        self.history = self._bars_1m[self._bars_1m.index < start]
        self.day_bars = self._bars_1m[(self._bars_1m.index >= start) & (self._bars_1m.index < end)]
        if len(self.day_bars) == 0:
            raise ValueError(f"No bars for day {day}")
        self.open_time = pd.Timestamp(f"{self.day} 09:30", tz=B.ET).tz_convert("UTC")
        self.close_time = pd.Timestamp(f"{self.day} 15:59", tz=B.ET).tz_convert("UTC")
        # replay covers the regular session only: from just before the open to the last 1m bar of the session
        # (pre-market bars still count as known data, they are just not stepped through one by one)
        self.first = self.open_time - MIN
        self.last = min(self.close_time, self.day_bars.index.max())
        self._hist_cache.clear()
        self.cursor = self.open_time - MIN   # before the open: nothing of today visible yet
        self.playing = False
        if self._task and not self._task.done():
            self._task.cancel()

    def visible_day(self):
        return self.day_bars[self.day_bars.index <= self.cursor]

    def _visible_day_for_tf(self, tf):
        """Return 1m bars from the visible day that form COMPLETE tf candles only.

        For tf=5m at cursor=10:30, the last complete 5m bar ends at 10:30 (10:25-10:30).
        The 1m bars for it are up to 10:29. So we cut off at the end of the last complete tf bar.
        """
        vis = self.visible_day()
        if tf not in TF_SECONDS or tf == "1m":
            return vis
        tf_sec = TF_SECONDS[tf]
        # End of last complete tf bar in ET seconds
        open_et = B.to_display(self.open_time)
        cursor_et = B.to_display(self.cursor)
        # cursor is the open time of the last CLOSED 1m bar, so data is complete up to cursor + 1 min
        known_et = cursor_et + 60
        if known_et < open_et + tf_sec:
            # No complete tf bars yet
            return vis.iloc[:0]
        n_complete = (known_et - open_et) // tf_sec
        last_complete_end_et = open_et + n_complete * tf_sec
        # Convert back to UTC timestamp for filtering 1m bars
        last_complete_end_utc = B.from_display(last_complete_end_et)
        # Include 1m bars strictly before the end of the last complete tf bar
        # (the tf bar ending at T uses 1m bars from T-tf to T-1min)
        return vis[vis.index < last_complete_end_utc]

    def bars(self, tf, session="rth"):
        """Return merged history + visible day bars, with only CLOSED candles from the visible day.

        A candle is closed if its end time (start + tf duration) <= cursor.
        History bars are always complete (from prior days).
        """
        key = (tf, session)
        if key not in self._hist_cache:
            self._hist_cache[key] = B.resample(self.history, tf, session)
        hist = self._hist_cache[key]
        vis_1m = self._visible_day_for_tf(tf)
        vis = B.resample(vis_1m, tf, session)
        return B.merge(hist, vis)

    def bar_at_cursor(self):
        row = self.day_bars[self.day_bars.index == self.cursor]
        return None if row.empty else B.to_json(row, "1m")[0]

    # ---------- control ----------

    def state(self):
        et = B.to_display
        return {
            "day": str(self.day), "playing": self.playing, "speed": self.speed, "speeds": SPEEDS,
            "cursor": et(self.cursor), "first": et(self.first), "last": et(self.last),
            "open": et(self.open_time), "close": et(self.close_time + MIN),
            "ended": self.cursor >= self.last,
        }

    def seek(self, et_seconds):
        t = B.from_display(et_seconds).floor("min")
        self.cursor = min(max(t, self.first - MIN), self.last)

    def sync_to_clock(self):
        """Put the cursor at the bar matching the current ET time of day on the replayed session."""
        now = pd.Timestamp.now(tz=B.ET)
        tod = now - now.normalize()
        day = pd.Timestamp(self.day, tz=B.ET)
        self.seek(B.to_display((day + tod - MIN).tz_convert("UTC")))

    async def play(self):
        if self.cursor >= self.last:
            return
        self.playing = True
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())
        await self._emit({"type": "state", "state": self.state()})

    async def pause(self):
        self.playing = False
        await self._emit({"type": "state", "state": self.state()})

    async def set_speed(self, speed):
        self.speed = speed if speed in SPEEDS else 1
        await self._emit({"type": "state", "state": self.state()})

    async def reset(self, how, value=None):
        if how == "seek":
            self.seek(value)
        elif how == "sync":
            self.sync_to_clock()
        elif how == "open":
            self.cursor = self.open_time - MIN
        await self._emit({"type": "reset", "state": self.state()})

    async def load_day_async(self, day: date | str):
        """Load a new day and reset to its open."""
        self.load_day(day)
        await self._emit({"type": "day_loaded", "state": self.state()})

    # ---------- loop ----------

    async def _run(self):
        while self.playing and self.cursor < self.last:
            if self.speed == 1:
                await asyncio.sleep(60 - time.time() % 60)   # step on real minute boundaries
            else:
                await asyncio.sleep(60 / self.speed)
            if not self.playing:
                break
            self.cursor += MIN
            await self._emit({"type": "step", "bar": self.bar_at_cursor(), "state": self.state()})
        if self.cursor >= self.last:
            self.playing = False
            await self._emit({"type": "state", "state": self.state()})

    async def _emit(self, event):
        for fn in list(self.listeners):
            await fn(event)
