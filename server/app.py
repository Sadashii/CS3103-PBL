"""JPM chart server, fed by the ReplayService.

GET  /api/bars           history + replayed bars up to the cursor, any timeframe, ET times
GET  /api/replay         replay state
POST /api/replay         {action: play|pause|speed|seek|sync|open, value?}
GET  /api/replay/days    list replayable days (most recent first)
POST /api/replay/load    {day: "YYYY-MM-DD"} load a specific day
WS   /ws                 replay steps and log events
GET  /api/status         store and replay health
GET  /api/signal         stub signal endpoint (returns "No signal" until model exists)

Run: uv run uvicorn server.app:app --port 8000
"""
import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd

from jpmind import bars as B
from jpmind import config, model as M, nextday as ND, store, tickmodel as TM
from jpmind.replay import ReplayService

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STATIC = Path(__file__).parent / "static"

clients = set()
events = deque(maxlen=300)       # recent log lines, replayed to each new client
replay: ReplayService | None = None
frames = {}                     # all stored 1m frames (JPM + context symbols); only read, never sent to clients
tick_model = None               # 1m models: next minute + rest of today
nd_model = None                 # next-day outlook model (HAR volatility + empirical return distribution)
predictor = None                # trained models (None until scripts/train.py has run)
tables = {}                     # day -> per-candle predictions; released to clients only up to the replay cursor
BAR5 = pd.Timedelta(minutes=5)


def load_models():
    global predictor, nd_model
    try:
        nd_model = ND.NextDay.load(config.MODEL_DIR / "nextday.json")
    except Exception as e:
        nd_model = None
        note("info", f"no next-day model ({e})")
    global tick_model
    try:
        tick_model = TM.TickModel()
    except Exception as e:
        tick_model = None
        note("info", f"no 1m tick model ({e})")
    try:
        predictor = M.Predictor()
        note("info", f"model loaded: {predictor.kind}, trained up to {predictor.cutoff}")
    except Exception as e:      # no models yet: the signal endpoints stay in 'No signal' mode
        predictor = None
        note("info", f"no trained model ({e})")


def day_table(day):
    if predictor is None:
        return None
    if day not in tables:
        end = pd.Timestamp(day, tz=B.ET).tz_convert("UTC") + pd.Timedelta(days=1)
        fr = {s: d[(d.index >= end - pd.Timedelta(days=80)) & (d.index < end)] for s, d in frames.items()}
        tables[day] = predictor.day_table(fr, day)
        note("info", f"predictions ready for {day}")
    return tables[day]


def closed_rows():
    """Predictions for candles that have fully closed at the replay cursor (never later ones)."""
    t = day_table(replay.day)
    if t is None:
        return None
    return t[t.index <= replay.cursor - pd.Timedelta(minutes=4)]


def note(kind, msg):
    line = {"ts": time.time(), "kind": kind, "msg": msg}
    events.append(line)
    log.info(msg)
    return {"type": "event", **line}


async def send_all(payload):
    text = json.dumps(payload)
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)


async def on_replay(ev):
    await send_all(ev)
    st = ev["state"]
    if ev["type"] == "step":
        b = ev["bar"]
        msg = (f"bar closed  O {b['open']:.2f}  H {b['high']:.2f}  L {b['low']:.2f}  C {b['close']:.2f}  "
               f"V {int(b['volume']):,}") if b else "minute passed, no IEX trades"
        await send_all(note("bar" if b else "info", msg) | {"at": st["cursor"]})
    elif ev["type"] == "reset":
        await send_all(note("info", "replay moved") | {"at": st["cursor"]})
    elif ev["type"] == "state":
        what = "ended" if st["ended"] else ("playing" if st["playing"] else "paused")
        await send_all(note("info", f"replay {what} at {st['speed']}x") | {"at": st["cursor"]})
    elif ev["type"] == "day_loaded":
        await send_all(note("info", f"loaded replay day {st['day']}") | {"at": st["cursor"]})


@asynccontextmanager
async def lifespan(app):
    global replay
    data = store.load(config.SYMBOL)    # local only; run scripts/fetch_data.py to add new days
    frames[config.SYMBOL] = data
    for sym in config.CONTEXT_SYMBOLS:
        frames[sym] = store.load(sym)
    load_models()
    replay = ReplayService(data)
    replay.listeners.append(on_replay)
    s = replay.state()
    note("info", f"loaded {len(data):,} stored 1m bars; replaying US session {s['day']}")
    await asyncio.to_thread(day_table, replay.day)
    replay.sync_to_clock()
    await replay.play()
    yield
    await replay.pause()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/bars")
def get_bars(tf: str = "5m", session: str = "rth", before: int | None = None,
             limit: int = Query(2000, ge=1, le=20000)):
    if tf not in B.TIMEFRAMES:
        return {"error": f"tf must be one of {list(B.TIMEFRAMES)}"}
    frame = replay.bars(tf, session)
    if before is not None:
        frame = frame[B.display_times(frame.index, tf) < before]
    chunk = frame.iloc[-limit:]
    return {"symbol": config.SYMBOL, "tf": tf, "session": session, "bars": B.to_json(chunk, tf),
            "more": len(frame) > len(chunk)}


@app.get("/api/replay")
def get_replay():
    return replay.state()


@app.post("/api/replay")
async def control(action: str = Body(..., embed=True), value: int | None = Body(None, embed=True)):
    if action == "play":
        await replay.play()
    elif action == "pause":
        await replay.pause()
    elif action == "speed":
        await replay.set_speed(value)
    elif action in ("seek", "sync", "open"):
        await replay.reset(action, value)
    return replay.state()


@app.get("/api/replay/days")
def list_replay_days():
    """Return list of replayable days (ISO strings, most recent first)."""
    return {"days": replay.list_days()}


@app.post("/api/replay/load")
async def load_replay_day(payload: dict = Body(...)):
    """Load a specific day for replay. Body: {day: "YYYY-MM-DD"}."""
    day_str = payload.get("day")
    if not day_str:
        return {"error": "day is required"}
    try:
        day = date.fromisoformat(day_str)
        await asyncio.to_thread(day_table, day)
        await replay.load_day_async(day)
        return replay.state()
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/status")
def status():
    h, d = replay.history, replay.day_bars
    return {
        "symbol": config.SYMBOL, "source": f"Alpaca {config.FEED.upper()} (stored) via replay",
        "history_bars": len(h), "history_first": B.to_display(h.index.min()),
        "day_bars": len(d), "visible_day_bars": len(replay.visible_day()),
        "clients": len(clients), "replay": replay.state(),
    }


def _entry(t, r, last_start):
    """One signal-log line. It is graded once the candle 30 minutes later has closed."""
    graded = r.direction != "No signal" and t + 6 * BAR5 <= last_start and not np.isnan(r.y6)
    sign = 1 if r.direction == "Up" else -1
    out = {"time": B.to_display(t) + 300, "candle": B.to_display(t), "direction": r.direction,
           "confidence": round(float(r.confidence), 4), "p_up": round(float(r.p_up), 4), "price": round(r.close, 2),
           "status": "no signal" if r.direction == "No signal" else "open"}
    if graded:
        out["outcome_bps"] = round(float(sign * r.y6), 2)
        out["net_bps"] = round(float(sign * r.y6 - config.COST_BPS), 2)
        out["status"] = "win" if sign * r.y6 > 0 else "loss"
    return out


@app.get("/api/signal")
def get_signal(cursor: int | None = None, tf: str = "5m"):
    """Latest signal at the replay cursor: direction, calibrated confidence, 90% band to the close, scenarios."""
    st = replay.state()
    rows = closed_rows()
    base = {"horizon_min": 30, "timestamp": st["cursor"], "model": None}
    if rows is None or rows.empty:
        return {**base, "direction": "No signal", "confidence": 0.0, "price": None,
                "forecast_band": {"times": [], "upper": [], "median": [], "lower": []},
                "scenarios": {k: {"path": [], "probability": 0.0, "description": "Waiting for the first closed 5m candle"}
                              for k in ("bull", "base", "bear")},
                "note": "No trained model loaded or no closed candle yet." if rows is None else "Before the first closed candle."}
    r = rows.iloc[-1]
    return {**base, "direction": r.direction, "confidence": round(float(r.confidence), 4), "p_up": round(float(r.p_up), 4), "lean": r.lean,
            "price": round(r.close, 2), "timestamp": B.to_display(rows.index[-1]) + 300,
            "predicted_30m_move_bps": round(float(r.med6_bps), 2), "bars_to_close": int(r.left),
            "forecast_band": predictor.band(r), "scenarios": predictor.scenarios(r),
            "model": {"classifier": predictor.kind, "trained_through": predictor.cutoff,
                      "threshold": predictor.cfg["threshold"], "median_gate_bps": predictor.cfg["median_gate_bps"]},
            "note": "Signal is computed from candles closed at or before the cursor only."}


@app.get("/api/tick")
def get_tick():
    """Fast output, refreshed on every closed 1m bar: next-minute P(up)/P(down) and price band, plus the rest-of-today distribution."""
    if tick_model is None:
        return {"available": False}
    rows = tick_model.day_rows(frames, replay.day)
    out = tick_model.at(rows, replay.cursor)
    if out is None:
        return {"available": False, "note": "No fast forecast now: it needs about 400 closed 1m bars of history, and nothing is left to forecast after the close."}
    cfg = tick_model.cfg
    out.update({"available": True, "trained_through": cfg["cutoff"],
                "honesty": {"next_minute_test_acc": round(cfg["next_minute"]["test"]["acc"], 4),
                            "next_minute_always_up_acc": round(cfg["next_minute"]["test"]["always_up_acc"], 4),
                            "next_minute_band_coverage": round(cfg["next_minute_band"]["coverage_test"], 3),
                            "to_close_band_coverage": round(cfg["to_close"]["cov90_test"], 3)},
                "note": "Next-minute edge is tiny (about 1 point over always-up, well under trading costs); treat probabilities as near 50/50."})
    return out


@app.get("/api/backtest")
def get_backtest():
    """Static summary of the confident-call backtest (see scripts/stock_sweep.py), shown with its assumptions."""
    return json.loads((Path(__file__).parent / "backtest_jpm.json").read_text(encoding="utf-8"))


@app.get("/api/daily")
def get_daily(limit: int = Query(260, ge=20, le=3000)):
    """Daily candles up to the replay cursor (today's candle is partial while the session is open) plus a close forecast:
    during the session -> today's close (live, from the 1m model; the next-day model's view from yesterday's close is also given);
    after the close -> the next session's close (next-day model)."""
    frame = replay.bars("1D", "rth")
    bars = B.to_json(frame.iloc[-limit:], "1D")
    done = replay.cursor >= replay.close_time
    nd = get_nextday()
    tk = None if done else get_tick()
    day_t = lambda d: int(pd.Timestamp(d).timestamp())          # ET wall-clock midnight as epoch seconds (same as the candle times)
    vis = replay.visible_day()                                  # today's candle from the 1m bars closed at the cursor (partial while open)
    if len(vis):
        t = day_t(replay.day)
        bars = [b for b in bars if b["time"] != t] + [{"time": t, "open": float(vis.open.iloc[0]), "high": float(vis.high.max()),
                                                        "low": float(vis.low.min()), "close": float(vis.close.iloc[-1]),
                                                        "volume": float(vis.volume.sum())}]
    fc = None
    if nd.get("available"):
        fc = {"mode": "next_day" if done else "today_close",
              "label": "Next session's close" if done else "Today's close",
              "anchor_time": day_t(nd["base_day"]), "anchor_close": nd["base_close"], "target_time": day_t(nd["target_day"]),
              "model_from_prev_close": {"median": nd["median_close"], "expected": nd["expected_close"], "band90": nd["band90"],
                                        "band50": nd["band50"], "p_up": nd["probabilities"]["p_up"], "as_of": nd["as_of"]}}
        if tk and tk.get("available"):
            t = tk["to_close"]
            fc["live"] = {"median": t["median_close"], "band90": t["band90"], "band50": t["band50"], "p_above_now": t["p_close_above_now"],
                          "price_now": tk["price"], "minutes_left": t["minutes_left"]}
    return {"symbol": config.SYMBOL, "bars": bars, "forecast": fc, "in_progress": not done}


@app.get("/api/nextday")
def get_nextday():
    """Outlook for the next trading day made from all sessions completed at the replay cursor."""
    if nd_model is None:
        return {"available": False}
    done = replay.cursor >= replay.close_time                  # today's 15:59 bar has closed: today counts as history
    cal = pd.to_datetime(store.calendar()["date"]).sort_values().reset_index(drop=True)
    today = pd.Timestamp(replay.day)
    base = today if done else cal[cal < today].iloc[-1]
    nxt = cal[cal > base]
    target = nxt.iloc[0] if len(nxt) else None
    out = nd_model.outlook(frames[config.SYMBOL][frames[config.SYMBOL].index < pd.Timestamp(base.date(), tz=B.ET).tz_convert("UTC") + pd.Timedelta(days=1)],
                           base, target)
    if out is None:
        return {"available": False}
    out["available"] = True
    out["as_of"] = "after today's close" if done else "yesterday's close (today is still in progress)"
    return out


@app.get("/api/signals")
def get_signals(calls_only: bool = False):
    """Append-only log of signals up to the cursor, with grading and a running scorecard."""
    rows = closed_rows()
    if rows is None or rows.empty:
        return {"signals": [], "summary": {}}
    last_start = rows.index[-1]
    log_rows = [_entry(t, r, last_start) for t, r in rows.iterrows()]
    calls = [e for e in log_rows if e["status"] != "no signal"]
    graded = [e for e in calls if "outcome_bps" in e]
    wins = sum(e["status"] == "win" for e in graded)
    summary = {"candles": len(log_rows), "calls": len(calls), "graded": len(graded), "wins": wins,
               "accuracy": round(wins / len(graded), 4) if graded else None,
               "net_bps": round(sum(e["net_bps"] for e in graded), 2), "cost_bps": config.COST_BPS}
    return {"signals": calls if calls_only else log_rows, "summary": summary}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    await ws.send_text(json.dumps({"type": "hello", "state": replay.state(), "events": list(events)}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
