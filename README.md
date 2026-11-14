# AI Trading Indicator for JPMorgan Chase (JPM)

CS3103 Project-Based Learning (Machine Learning), IIT Patna.
Tanishq Sangwan (2401AI42). Instructor: Dr. Rishav Singh.

Every five minutes of the trading day, the indicator tells a newcomer whether JPM is likely to go **Up**, stay **Flat**
or go **Down** over the next 10 minutes, how active the market is about to be, whether a big move is near, and where
the price may close. A replay dashboard plays any past day bar by bar and grades every call against what happened.

## Results (out of sample)

| | Ours | Best simple rule |
|---|---|---|
| 3-class accuracy, test 2025-01-02 … 2026-10-27 (34,538 candles) | **40.7 %** | 35.8 % (persistence) |
| 3-class accuracy, 13 unseen days (2026-10-28 … 11-13) | **43.2 %** | 36.8 % |
| Share of single test days won vs each rule | 70–80 % | |
| Calibration error (ECE) | 0.6 % | |
| Big-move warning, AUC | 0.785 | 0.5 |
| Activity level (Quiet / Normal / Active) | 46.3 % | 40.2 % |
| Rest-of-day 90 % band coverage | 93.6 % | |

Direction alone is **not** predictable from these inputs: Up/Down calls are right 49.6 % of the time, and ten model
families (boosting, MLP, GRU, CNN, Transformer, see `docs/model_zoo_results.md`) all stay near 50 %. The model's skill
is in the *size* of the next move, so treat it as a risk and activity guide, not a buy/sell signal.

## How it works

```
1-min bars (JPM, SPY, XLF, BAC, C, WFC, GS, MS) + news / earnings / FOMC / SEC filings
        │  5-min candles, only data at or before t
        ▼
103 causal features (volatility, market & peers, order flow, news & events, momentum, time, day range)
        │
        ├─ LightGBM 3-class call (Up / Flat / Down, next 10 min)
        ├─ activity model and big-move model
        └─ 1-min model: next-minute call + conformal rest-of-day bands
        ▼
FastAPI + WebSocket replay server  →  dashboard (chart layers, call, outlook, results tab)
```

Training is an expanding walk-forward by year (fit on years before Y, predict Y). The model is chosen on 2022–24 and
reported once on 2025–26. A shuffled-label placebo and the tests in `tests/` guard against look-ahead.

## Repository layout

| Path | What it holds |
|---|---|
| `jpmind/` | library: data store, bar resampling, feature builders (5 m and 1 m), news features, S/R levels, replay |
| `scripts/` | data download, training, evaluation, experiments and figures (each file starts with its usage) |
| `server/` | FastAPI app and the dashboard (`server/static/`) |
| `models/` | trained models and their metrics (`cls3_metrics.json`, `tick_config.json`, …) |
| `tests/` | causality / leakage tests |
| `docs/` | task notes, research notes, result tables, figures |
| `report/` | LaTeX project report (`report/README.md` explains how to build it) |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Data comes from the Alpaca market-data API. Create `.env` in the project root:

```
ALPACA_KEY_ID=...
ALPACA_SECRET_KEY=...
ALPACA_FEED=sip
```

The keys are only needed to download data; the dashboard runs offline on the stored data.

## Usage

```bash
# 1. download data (incremental; stores complete sessions only)
uv run python scripts/fetch_data.py --feed sip --start 2016-01-01
uv run python scripts/fetch_news.py JPM SPY
uv run python scripts/fetch_events.py

# 2. train and evaluate (cutoff = last training day)
uv run python scripts/train.py --cutoff 2026-10-27        # 5-min features + rest-of-day models
uv run python scripts/train_cls3.py --cutoff 2026-10-27   # Up / Flat / Down, activity, big move
uv run python scripts/train_tick.py --cutoff 2026-10-27   # next-minute call and bands
uv run python scripts/holdout_eval.py                     # score the days after the cutoff

# or fetch + retrain everything up to the day before a test day T
uv run python scripts/prepare_test_day.py --day 2026-11-13

# 3. dashboard
uv run uvicorn server.app:app --port 8000                 # open http://localhost:8000

# leakage tests
uv run python tests/test_causal.py

# report figures
uv run --with matplotlib python scripts/make_figures.py
```

In the dashboard, pick a day and press play. The **Replay** tab shows the chart with prediction layers and the current
call in plain language; the **Results** tab shows the model's test numbers and each day's predicted vs actual direction.
