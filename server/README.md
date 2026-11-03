# JPM Trading Indicator — Replay Server

FastAPI server that replays historical JPM 1-minute bars (SIP feed) as if they were live, with a WebSocket dashboard for visualization.

## Quick Start

```bash
# From project root
uv run uvicorn server.app:app --port 8000
```

Then open http://localhost:8000 in your browser.

## Requirements

- Python 3.11+
- Data files in `data/raw/` (run `scripts/fetch_data.py` to populate)
- `.env` with `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` (for fetching new data only; server runs offline)

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│  Browser UI     │◄──────────────────►│  FastAPI Server  │
│  (Lightweight   │  Replay steps,     │  - ReplayService │
│   Charts)       │  signals, events   │  - REST endpoints│
└─────────────────┘                    └────────┬─────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │  Local Parquet Store  │
                                    │  data/raw/*_sip_1min  │
                                    └───────────────────────┘
```

## Endpoints

### `GET /`
Serves the dashboard UI (`server/static/index.html`).

### `GET /api/bars`
Returns OHLCV bars up to the replay cursor.

**Query params:**
- `tf` (default `5m`): Timeframe — `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1D`, `1W`
- `session` (default `rth`): `rth` (regular trading hours 09:30–15:59 ET) or `all`
- `before` (optional): ET epoch seconds; only bars strictly before this time
- `limit` (default 2000, max 20000): Maximum bars to return (most recent first)

**Response:**
```json
{
  "symbol": "JPM",
  "tf": "5m",
  "session": "rth",
  "bars": [
    {"time": 1700000000, "open": 150.0, "high": 151.0, "low": 149.5, "close": 150.5, "volume": 123456}
  ],
  "more": true
}
```
All `time` values are ET wall-clock epoch seconds (not UTC).

### `GET /api/replay`
Current replay state.

**Response:**
```json
{
  "day": "2026-10-30",
  "playing": true,
  "speed": 1,
  "speeds": [1, 2, 5, 10, 30, 60],
  "cursor": 1700000000,
  "first": 1699992600,
  "last": 1700016000,
  "open": 1699992600,
  "close": 1700016000,
  "ended": false
}
```
All timestamps are ET epoch seconds.

### `GET /api/replay/days`
List of replayable days (ISO dates, most recent first). Only days after the training cutoff (2026-10-27) that have complete RTH sessions.

### `POST /api/replay`
Control the replay.

**Body:**
```json
{ "action": "play" | "pause" | "speed" | "seek" | "sync" | "open" | "load_day", "value": 123 }
```

| Action | Value | Description |
|--------|-------|-------------|
| `play` | — | Start/resume playback |
| `pause` | — | Pause playback |
| `speed` | `1`\|`2`\|`5`\|`10`\|`30`\|`60` | Set playback speed (bars per real minute) |
| `seek` | ET epoch seconds | Jump to specific minute |
| `sync` | — | Sync cursor to current ET clock time on the replayed day |
| `open` | — | Jump to session open (09:30 ET) |
| `load_day` | `"YYYY-MM-DD"` | Load a different replayable day |

### `GET /api/signal`
**Signal stub endpoint** — returns "No signal" until a trained model is plugged in.

**Query params:**
- `cursor` (optional, default: current replay cursor): ET epoch seconds
- `tf` (default `5m`): Timeframe for signal generation

**Response:**
```json
{
  "symbol": "JPM",
  "cursor": 1700000000,
  "tf": "5m",
  "direction": "No signal",
  "confidence": 0.0,
  "horizon_minutes": 30,
  "forecast_band": { "median": [], "upper": [], "lower": [] },
  "scenarios": { "bull": [], "base": [], "bear": [], "probabilities": {} },
  "model_version": "stub",
  "note": "Replace this stub with a trained model. The model must only use bars <= cursor."
}
```

**Model Integration Contract:**
When you plug in a real model, it must:
1. **Never see future data** — only bars with timestamp ≤ `cursor`
2. Return direction: `"Up"` | `"Down"` | `"No signal"`
3. Return confidence: `0.0–1.0`
4. Return `horizon_minutes` (default 30, per D17)
5. Return `forecast_band` with `median`, `upper` (95th pctl), `lower` (5th pctl) arrays aligned to future 5m steps to session close
6. Return `scenarios` with `bull`, `base`, `bear` paths and probabilities
7. Be fast enough for 1-bar-per-minute replay (CPU inference, no GPU required)

### `GET /api/status`
Health and debug info.

### `WS /ws`
WebSocket for real-time updates. Messages:
- `hello` — initial state on connect
- `state` — replay state change
- `step` — new bar arrived (`{bar, state}`)
- `reset` — cursor moved
- `day_loaded` — new day loaded
- `event` — log messages (bar closed, replay started, etc.)

## Timezone Handling

- All stored bars have UTC timestamps
- All API timestamps are **ET wall-clock epoch seconds** (what TradingView shows)
- Conversion helpers in `jpmind/bars.py`:
  - `to_display(utc_ts)` → ET epoch seconds
  - `from_display(et_seconds)` → UTC timestamp
  - `display_times(index, tf)` → array of ET epoch seconds for charting

## Replay Rules (Hard Caps)

- **Z3 (No repainting):** Model/chart only receives bars ≤ cursor. The `/api/bars` endpoint enforces this via the `before` parameter.
- **Z4 (Training cutoff):** Only days after 2026-10-27 are replayable (see `TRAINING_CUTOFF` in `jpmind/replay.py`).
- **D28:** Real timestamps shown in ET, labelled REPLAY.
- **D17:** Graded signal is 30-minute horizon on 5-minute candles (~72 signals/session).

## Adding New Data

```bash
# Fetch latest bars from Alpaca (requires .env keys)
uv run python scripts/fetch_data.py

# The server will pick up new days on restart
```

## Project Structure

```
server/
  app.py              # FastAPI application
  static/
    index.html        # Dashboard UI
    style.css         # Dark theme styles
    app.js            # Dashboard logic (WebSocket, charts, controls)
  README.md           # This file

jpmind/
  replay.py           # ReplayService (cursor, playback, day loading)
  bars.py             # Resampling, timezone conversion
  store.py            # Parquet I/O
  config.py           # Settings (symbol, feed, horizons, costs)
```

## Testing the Replay

```bash
# Start server
uv run uvicorn server.app:app --port 8000 &

# List replayable days
curl http://localhost:8000/api/replay/days

# Get current state
curl http://localhost:8000/api/replay

# Load a specific day
curl -X POST http://localhost:8000/api/replay \
  -H "Content-Type: application/json" \
  -d '{"action": "load_day", "value": "2026-10-30"}'

# Get 5m bars up to cursor
curl "http://localhost:8000/api/bars?tf=5m"

# Check signal stub
curl "http://localhost:8000/api/signal"

# Play
curl -X POST http://localhost:8000/api/replay \
  -H "Content-Type: application/json" \
  -d '{"action": "play"}'

# Seek to 10:00 ET
curl -X POST http://localhost:8000/api/replay \
  -H "Content-Type: application/json" \
  -d '{"action": "seek", "value": 1700000000}'
```

## Verifying No Future Data

```bash
# 1. Get current cursor
curl http://localhost:8000/api/replay | jq .cursor

# 2. Request bars with before=cursor
curl "http://localhost:8000/api/bars?tf=5m&before=<cursor>" | jq '.bars[-1].time'

# 3. Verify last bar time ≤ cursor (in ET seconds)
# 4. Play a few steps, repeat — cursor advances, bars never exceed cursor
```

## Model Integration Guide

1. Create a new module (e.g., `jpmind/model.py`) with a `predict(cursor_et_seconds, bars_5m_up_to_cursor)` function
2. In `server/app.py`, replace the `/api/signal` endpoint to call your model
3. Ensure your model loads weights at startup (not per-request)
4. Test that inference completes in <100ms on CPU

Example stub replacement:
```python
# In server/app.py
from jpmind.model import predict_signal

@app.get("/api/signal")
def get_signal(cursor: int | None = None, tf: str = "5m"):
    if cursor is None:
        cursor = replay.state()["cursor"]
    bars = replay.bars(tf, "rth")
    # Filter to bars <= cursor (defense in depth)
    bars = bars[B.display_times(bars.index, tf) <= cursor]
    return predict_signal(cursor, bars)
```

## License

Student project — CS3103 PBL.