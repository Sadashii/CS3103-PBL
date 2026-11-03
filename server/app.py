"""JPM chart server, fed by the ReplayService.

GET  /api/bars     history + replayed bars up to the cursor, any timeframe, ET times
GET  /api/replay   replay state;  POST /api/replay {action: play|pause|speed|seek|sync|open}
WS   /ws           replay steps and log events
GET  /api/status   store and replay health

Run: uv run uvicorn server.app:app --port 8000
"""
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from jpmind import bars as B
from jpmind import config, store
from jpmind.replay import ReplayService

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STATIC = Path(__file__).parent / "static"

clients = set()
events = deque(maxlen=300)       # recent log lines, replayed to each new client
replay: ReplayService | None = None


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


@asynccontextmanager
async def lifespan(app):
    global replay
    data = store.load(config.SYMBOL)    # local only; run scripts/fetch_data.py to add new days
    replay = ReplayService(data)
    replay.listeners.append(on_replay)
    s = replay.state()
    note("info", f"loaded {len(data):,} stored 1m bars; replaying US session {s['day']}")
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


@app.get("/api/status")
def status():
    h, d = replay.history, replay.day_bars
    return {
        "symbol": config.SYMBOL, "source": f"Alpaca {config.FEED.upper()} (stored) via replay",
        "history_bars": len(h), "history_first": B.to_display(h.index.min()),
        "day_bars": len(d), "visible_day_bars": len(replay.visible_day()),
        "clients": len(clients), "replay": replay.state(),
    }


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
