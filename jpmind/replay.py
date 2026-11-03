"""ReplayService: serves the last complete US session as if it were live.

Everything before that session is static history. Within the session a cursor moves
one minute at a time (one bar per real minute at 1x). Consumers only ever see bars at
or before the cursor, so nothing downstream (chart or model) can look ahead.
"""
import asyncio
import time

import pandas as pd

from . import bars as B

MIN = pd.Timedelta(minutes=1)
SPEEDS = [1, 2, 5, 10, 30, 60]


class ReplayService:
    def __init__(self, bars_1m):
        self.listeners = []          # async callables(event dict)
        self.speed = 1
        self.playing = False
        self._task = None
        self._hist_cache = {}
        self.load(bars_1m)

    # ---------- data ----------

    def load(self, bars_1m):
        et = bars_1m.index.tz_convert(B.ET)
        closes = et[(et.hour == 15) & (et.minute == 59)]          # sessions that reached the close
        self.day = closes.normalize().max().date()
        start = pd.Timestamp(self.day, tz=B.ET).tz_convert("UTC")
        end = start + pd.Timedelta(days=1)
        self.history = bars_1m[bars_1m.index < start]
        self.day_bars = bars_1m[(bars_1m.index >= start) & (bars_1m.index < end)]
        self.first, self.last = self.day_bars.index.min(), self.day_bars.index.max()
        self.open_time = pd.Timestamp(f"{self.day} 09:30", tz=B.ET).tz_convert("UTC")
        self.close_time = pd.Timestamp(f"{self.day} 15:59", tz=B.ET).tz_convert("UTC")
        self._hist_cache.clear()
        self.cursor = self.open_time - MIN   # before the open: nothing of today visible yet

    def visible_day(self):
        return self.day_bars[self.day_bars.index <= self.cursor]

    def bars(self, tf, session="rth"):
        key = (tf, session)
        if key not in self._hist_cache:
            self._hist_cache[key] = B.resample(self.history, tf, session)
        return B.merge(self._hist_cache[key], B.resample(self.visible_day(), tf, session))

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
