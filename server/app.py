"""JPM chart server, fed by the ReplayService.

GET  /api/bars           history + replayed bars up to the cursor, any timeframe, ET times
GET  /api/replay         replay state
POST /api/replay         {action: play|pause|speed|seek|sync|open, value?}
GET  /api/replay/days    list replayable days (most recent first)
POST /api/replay/load    {day: "YYYY-MM-DD"} load a specific day
WS   /ws                 replay steps and log events
GET  /api/status         store and replay health
GET  /api/signal         latest signal: 3-class 30-min call, big-move warning, 90% band, scenarios
GET  /api/signals        append-only graded signal log with baselines on the same candles
GET  /api/levels         support / resistance levels known at the cursor
GET  /api/outlook        frozen rest-of-day outlooks made at 10:00 and 12:30 ET, graded at the close

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
from jpmind import config, levels as LV, model as M, nextday as ND, store, tickmodel as TM
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
    # during market hours, follow the clock (like a live chart); otherwise start at the open, paused
    now = pd.Timestamp.now(tz=B.ET)
    if pd.Timestamp("09:30").time() <= now.time() < pd.Timestamp("16:00").time():
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
             limit: int = Query(2000, ge=1, le=20000), partial: bool = False):
    """Closed candles up to the cursor. With partial=1 the candle still forming at the cursor is appended and flagged
    ("partial": true): it is display only, no signal is ever computed from it."""
    if tf not in B.TIMEFRAMES:
        return {"error": f"tf must be one of {list(B.TIMEFRAMES)}"}
    frame = replay.bars(tf, session)
    if before is not None:
        frame = frame[B.display_times(frame.index, tf) < before]
    chunk = frame.iloc[-limit:]
    out = B.to_json(chunk, tf)
    if partial and tf not in B.DAILY and before is None:
        vis = replay.visible_day()
        if session == "rth":
            vis = vis[(vis.index >= replay.open_time) & (vis.index <= replay.close_time)]
        forming = B.resample(vis, tf, session)
        if len(forming):
            last = B.to_json(forming.iloc[-1:], tf)[0]
            if not out or last["time"] > out[-1]["time"]:
                out.append(last | {"partial": True})
    return {"symbol": config.SYMBOL, "tf": tf, "session": session, "bars": out, "more": len(frame) > len(chunk)}


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


def _base_calls(r, tau):
    """Baseline calls for the same candle, so the log can show the model against them on the replayed day."""
    cls = lambda x: "Up" if x > tau else "Down" if x < -tau else "Neutral"
    return {"always_neutral": "Neutral", "always_up": "Up",
            "rsi_rule": "Up" if r.rsi < 30 else "Down" if r.rsi > 70 else "Neutral",
            "persistence": cls(r.pk)}


def _entry3(t, r, last_start):
    """One 3-class log line: the call made at the candle close, graded once the candle one horizon later has closed."""
    out = {"time": B.to_display(t) + 300, "candle": B.to_display(t), "direction": r.call, "confidence": round(float(r.call_conf), 4),
           "p_up": round(float(r.p_up3), 4), "p_neutral": round(float(r.p_neutral), 4), "p_down": round(float(r.p_down3), 4),
           "big_move": bool(r.big), "p_big": round(float(r.p_big), 4), "price": round(r.close, 2),
           "status": "no signal" if r.call == "No call" else "open"}
    has_act = "act" in r.index
    if has_act:
        out.update(activity=r.act, act_conf=round(float(r.act_conf), 4))
    if r.call != "No call" and t + predictor.c3["horizon_bars"] * BAR5 <= last_start and not np.isnan(r.yk):
        act = predictor.realised(r.yk)
        out.update(outcome_bps=round(float(r.yk), 2), actual=act, status="win" if act == r.call else "loss",
                   price_after=round(float(r.close * np.exp(r.yk / 1e4)), 2),
                   baselines={k: v == act for k, v in _base_calls(r, predictor.c3["tau_bps"]).items()})
        if has_act:
            aa = predictor.activity_of(r.yk)
            out.update(act_actual=aa, act_ok=aa == r.act, act_persist_ok=aa == predictor.activity_of(r.pk))
        if r.call in ("Up", "Down"):
            out["net_bps"] = round(float((1 if r.call == "Up" else -1) * r.yk - config.COST_BPS), 2)
    return out


@app.get("/api/signal")
def get_signal(cursor: int | None = None, tf: str = "5m"):
    """Latest signal at the replay cursor: direction, calibrated confidence, 90% band to the close, scenarios."""
    st = replay.state()
    rows = closed_rows()
    base = {"horizon_min": predictor.c3["horizon_bars"] * 5 if predictor is not None and predictor.c3 is not None else 30,
            "timestamp": st["cursor"], "model": None}
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
            "note": "Signal is computed from candles closed at or before the cursor only.",
            **({"call3": {"call": r.call, "confidence": round(float(r.call_conf), 4),
                          "probs": {"Up": round(float(r.p_up3), 4), "Neutral": round(float(r.p_neutral), 4), "Down": round(float(r.p_down3), 4)},
                          "horizon_min": predictor.c3["horizon_bars"] * 5,
                          "neutral_band_bps": round(predictor.c3["tau_bps"], 1),
                          "neutral_band_price": [round(r.close * np.exp(-predictor.c3["tau_bps"] / 1e4), 2),
                                                 round(r.close * np.exp(predictor.c3["tau_bps"] / 1e4), 2)],
                          "big_move": bool(r.big), "p_big": round(float(r.p_big), 4),
                          **({"activity": r.act, "act_conf": round(float(r.act_conf), 4), "p_quiet": round(float(r.p_quiet), 4),
                              "p_active": round(float(r.p_active), 4), "active_bps": round(predictor.c3["tau_hi_bps"], 1)} if "act" in r.index else {}),
                          "big_move_bps": round(predictor.c3["tau_big_bps"], 1),
                          "honesty": predictor.c3["honesty"]}} if predictor.c3 is not None else {})}


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


@app.get("/api/levels")
def get_levels():
    """Support / resistance at the cursor: prior day, floor pivots, 5-session range, pre-market, opening range, day high/low, swings."""
    out = LV.compute(replay.history.iloc[-60 * 24 * 12:], replay.visible_day())
    out["timestamp"] = replay.state()["cursor"]
    return out


OUTLOOK_TIMES = (("Morning", "09:59"), ("Midday", "12:29"))   # bar opening at this minute closes at 10:00 / 12:30


@app.get("/api/outlook")
def get_outlook():
    """Two fixed outlooks per day (rest of day), made once from the bars closed at 10:00 and 12:30 ET and never changed afterwards.
    Each is graded at the close: did the close land in the 50% / 90% band, and was the lean (P(close above) vs 50%) right."""
    if tick_model is None:
        return {"available": False, "outlooks": []}
    rows = tick_model.day_rows(frames, replay.day)
    done = replay.cursor >= replay.close_time
    vis = replay.visible_day()
    close = float(vis[vis.index <= replay.close_time].close.iloc[-1]) if done and len(vis) else None
    res = []
    for name, hhmm in OUTLOOK_TIMES:
        t = pd.Timestamp(f"{replay.day} {hhmm}", tz=B.ET).tz_convert("UTC")
        item = {"name": name, "made_at": B.to_display(t) + 60, "status": "pending"}
        if replay.cursor >= t:
            o = tick_model.at(rows, t)
            if o is not None:
                d = o["to_close"]
                lv = LV.compute(replay.history.iloc[-60 * 24 * 12:], replay.day_bars[replay.day_bars.index <= t])
                item.update(status="open", price=o["price"], minutes_left=d["minutes_left"], median_close=d["median_close"],
                            band50=d["band50"], band90=d["band90"], p_close_above=d["p_close_above_now"],
                            support=lv["nearest_support"], resistance=lv["nearest_resistance"])
                if close is not None:
                    lean_up = d["p_close_above_now"] >= 0.5
                    item.update(status="graded", actual_close=round(close, 2), move_bps=round((close / o["price"] - 1) * 1e4, 1),
                                in50=d["band50"]["lower"] <= close <= d["band50"]["upper"],
                                in90=d["band90"]["lower"] <= close <= d["band90"]["upper"],
                                lean_ok=(close > o["price"]) == lean_up)
        res.append(item)
    cfg = tick_model.cfg
    return {"available": True, "outlooks": res, "day": str(replay.day),
            "honesty": {"to_close_band_coverage": round(cfg["to_close"]["cov90_test"], 3)},
            "note": "Bands are the skill; the up/down lean has no measured skill (Brier 0.255 vs 0.250 climatology)."}


def _hold_row(r, cols):
    """One unseen day: 10-minute call accuracy vs the best simple rule, and each outlook's day call vs the actual close."""
    f = lambda k: None if k not in r or pd.isna(r[k]) else (bool(r[k]) if isinstance(r[k], (bool, np.bool_)) else float(r[k]))
    out = {"day": r["day"], "candles": int(r["candles"]), "model": float(r["model"]),
           "best_baseline": float(max(r[c] for c in cols)), "best_name": max(cols, key=lambda c: r[c]),
           "open": f("open"), "close": f("close")}
    for n in ("am", "pm"):
        out[n] = {k: f(f"{n}_{k}") for k in ("price", "median", "p_up", "lo90", "hi90", "in50", "in90", "lean_ok")}
    return out


@app.get("/api/results")
def get_results():
    """Numerical results for the Results tab, read from the files the training / evaluation scripts write
    (models/cls3_metrics.json, models/cls3_per_day.csv, models/tick_config.json, models/metrics.json, docs/holdout_results.csv)."""
    md, root = config.MODEL_DIR, config.ROOT
    rd = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    m3, tk, m2 = rd(md / "cls3_metrics.json"), rd(md / "tick_config.json"), rd(md / "metrics.json")
    c3 = rd(md / "cls3.json") or {}
    out = {"cutoff": c3.get("cutoff"), "horizon_min": c3.get("horizon_bars", 2) * 5, "trained_at": c3.get("trained_at")}
    if m3:
        dep = m3["deployed"]
        t = m3["models"][dep]["test"]
        out["test"] = {"period": "2025-01-02 to " + m3["cutoff"], "candles": t["n"], "model": dep, "acc": t["acc"], "ece": t["ece"],
                       "logloss": t["logloss"], "dir_hit": t["dir_hit"], "net_bps_per_call": t["net_bps_per_call"],
                       "moves_called": t["moves_called"], "cost_bps": m3["cost_bps"], "tau_bps": m3["tau_bps"],
                       "baselines": {k: v["test"]["acc"] for k, v in m3["baselines"].items() if k != "majority_2016_21"},
                       "other_models": {k: v["test"]["acc"] for k, v in m3["models"].items() if k != dep}}
        sd = m3["single_day"]
        out["single_day"] = {"days": sd["days"], "mean": sd["model_mean"], "p10": sd["model_p10"], "p90": sd["model_p90"],
                             "beats": {k: v for k, v in sd["beats"].items() if k != "majority_2016_21"}}
        a = m3["activity"]
        out["activity"] = {"acc": a["model"]["test"]["acc"], "persistence": a["baselines"]["persistence"]["test"],
                           "always_normal": a["baselines"]["always_normal"]["test"], "beats_persistence_days": a["single_day"]["beats"]["persistence"],
                           "active_bps": a["tau_hi_bps"]}
        b = m3["big_move"]
        out["big_move"] = {"auc": b["test"]["auc"], "base_rate": b["test"]["base_rate"], "top_decile": b["test"]["rate_top_decile"],
                           "bottom_decile": b["test"]["rate_bottom_decile"], "flag_share": b["test_flagged"]["share"],
                           "flag_hit": b["test_flagged"]["big_rate"], "bps": m3["tau_big_bps"]}
    per_day = md / "cls3_per_day.csv"
    if per_day.exists():
        pdf = pd.read_csv(per_day)
        out["daily_acc"] = [round(float(x), 4) for x in pdf["model"]]
    hold = root / "docs" / "holdout_results.csv"
    if hold.exists():
        h = pd.read_csv(hold)
        w = h["candles"]
        avg = lambda c: float(np.average(h[c], weights=w))
        cols = ["always_neutral", "always_up", "rsi_rule", "persistence"]
        out["holdout"] = {"days": len(h), "first": h["day"].iloc[0], "last": h["day"].iloc[-1], "candles": int(w.sum()), "acc": avg("model"),
                          "baselines": {c: avg(c) for c in cols}, "beats": {c: float((h["model"] > h[c]).mean()) for c in cols},
                          "activity": avg("activity"), "activity_persist": avg("activity_persist"),
                          "am_in90": float(h["am_in90"].mean()), "pm_in90": float(h["pm_in90"].mean()),
                          "am_in50": float(h["am_in50"].mean()), "pm_in50": float(h["pm_in50"].mean()),
                          "am_dir": float(h["am_lean_ok"].mean()) if "am_lean_ok" in h else None,
                          "pm_dir": float(h["pm_lean_ok"].mean()) if "pm_lean_ok" in h else None,
                          "per_day": [_hold_row(r, cols) for _, r in h.iterrows()]}
    if tk:
        out["tick"] = {"acc": tk["next_minute"]["test"]["acc"], "always_up": tk["next_minute"]["test"]["always_up_acc"],
                       "n": tk["next_minute"]["test"]["n"], "band_cov": tk["next_minute_band"]["coverage_test"],
                       "close_cov90": tk["to_close"]["cov90_test"], "close_cov50": tk["to_close"]["cov50_test"],
                       "pinball": tk["to_close"].get("pinball_model_test"), "pinball_base": tk["to_close"].get("pinball_vol_scaled_baseline_test")}
    if m2 and "nested" in m2:
        out["nested"] = {"t": m2["nested"]["t"], "net_bps": m2["nested"]["net_bps"], "n": m2["nested"]["n"], "required_t": m2["nested"]["required_t"]}
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
    if predictor.c3 is not None:
        log_rows = [_entry3(t, r, last_start) for t, r in rows.iterrows()]
        calls = [e for e in log_rows if e["status"] != "no signal"]
        graded = [e for e in calls if "actual" in e]
        moves = [e for e in graded if "net_bps" in e]
        n = len(graded)
        summary = {"scheme": "3-class", "candles": len(log_rows), "calls": len(calls), "graded": n,
                   "wins": sum(e["status"] == "win" for e in graded),
                   "accuracy": round(sum(e["status"] == "win" for e in graded) / n, 4) if n else None,
                   "baselines": {k: round(sum(e["baselines"][k] for e in graded) / n, 4) for k in graded[0]["baselines"]} if n else {},
                   "move_calls": len(moves), "net_bps": round(sum(e["net_bps"] for e in moves), 2),
                   "act_accuracy": round(sum(e.get("act_ok", False) for e in graded) / n, 4) if n and "act_ok" in graded[0] else None,
                   "act_persistence": round(sum(e.get("act_persist_ok", False) for e in graded) / n, 4) if n and "act_ok" in graded[0] else None,
                   "big_flags": sum(e["big_move"] for e in graded),
                   "big_hit": sum(e["big_move"] and abs(e["outcome_bps"]) > predictor.c3["tau_big_bps"] for e in graded),
                   "neutral_band_bps": round(predictor.c3["tau_bps"], 1), "cost_bps": config.COST_BPS,
                   "horizon_min": predictor.c3["horizon_bars"] * 5}
        return {"signals": calls if calls_only else log_rows, "summary": summary}
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
