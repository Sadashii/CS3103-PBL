# Handoff: JPM trading indicator (CS3103 PBL)

Last updated 2026-11-03. Design decisions are in `docs/TASK.md` (D1-D33).

## What this is
A replay-based dashboard for JPM: 5-minute candles, a graded 30-minute Up/Down/No-signal call with confidence, a rest-of-day
90% price band, scenarios, and a graded append-only signal log. Replay is allowed only for days after the training cutoff
(2026-10-27). Data: Alpaca SIP 1-minute bars since 2016 (JPM, SPY, XLF, BAC, C, WFC, GS, MS).

## Run it
```
cp .env.example .env            # add your Alpaca keys (never commit .env)
uv sync
uv run python scripts/fetch_data.py                 # 1m bars -> data/raw (incremental)
uv run python scripts/fetch_news.py JPM BAC C WFC GS MS   # Alpaca news
uv run python scripts/fetch_events.py               # SEC 8-K/10-K/Form 4 timestamps + FOMC dates
uv run python scripts/fetch_xbrl.py                 # SEC XBRL quarterly financials
uv run python scripts/train.py [--quick]            # features, leak gates, walk-forward, models/ (not in git)
uv run uvicorn server.app:app --port 8000           # dashboard at http://127.0.0.1:8000
```
FinBERT news scoring needs GPU torch, which does not support Python 3.14, so it uses a separate env:
`uv venv .venv-nlp --python 3.12`, install torch (cu124 index) + transformers + pandas + pyarrow + python-dotenv into it, then
`.venv-nlp\Scripts\python scripts/score_news.py`.

## Layout
- `jpmind/` features.py (causal builder, one function for training and serving), newsfeat.py, fundamentals.py, model.py
  (Predictor), replay.py, bars.py, store.py, config.py
- `server/` FastAPI app + static dashboard (single Lightweight Charts pane, overlays and the forecast on the main chart)
- `scripts/` data fetchers, `train.py`, experiments (`daily_test.py`, `daily_panel.py`, `vol_forecast.py`)
- `tests/` `test_causal.py` (feature truncation test), `test_news_causal.py`

## Leakage controls (all must stay green)
1. `tests/test_causal.py`: features on truncated data equal features on full data (max diff 0).
2. `tests/test_news_causal.py`: deleting items published after a bar's close changes nothing (except the FOMC calendar, which is published a year ahead).
3. Replay sweep: every bar shown at every cursor checked against data known at that time: 0 leaks over 365,352 checks.
4. `train.py` gates stop the run if any single feature has |IC| > 0.06 with the label or a shuffled-label placebo predicts.
5. Walk-forward by year, 1-session embargo, session length from the NYSE calendar (not inferred from data), isotonic calibration and thresholds fitted on validation years only (2022-24), 2025-26 is untouched test.
6. Nested check of the call-rule search; the rule is deployed only if t >= 3 on years it never saw, otherwise the model makes NO calls.

## Results (honest)
- 30-min direction: about 50-51.5% per candle on 2025-26; called signals lost money on test. No edge found.
- 1 day / 1 week / 1 month direction (daily, pooled six banks, walk-forward 2020-26): 51-54%, below always-up (53/57/60%). News (FinBERT), SEC filings and fundamentals, overnight/intraday split, peer lead-lag, pairs z-scores did not change this beyond noise.
- 90% price band: calibrated, about 87-90% coverage on test.
- Volatility: HAR baseline beat LightGBM (next-day log-RV R2 0.59 vs 0.47 on JPM); the LightGBM 90% band under-covers (0.84 next day, 0.75 next week). Use HAR for the next step.
- Unverified: the "+ scheduled calendar" volatility variant gave results identical to the base to 3 decimals; check that the `_earn_tom` / `_fomc_tom` columns are non-trivial.
- 90%+ accuracy is not a realistic target: with daily vol around 1.5%, even perfect knowledge of the drift gives about 51% daily accuracy (accuracy ~ Phi(Sharpe / sqrt(252))).

## State at handoff
- `models/` and `data/` are not in git (regenerate with the commands above).
- A full `train.py` run with the nested-validation gate was still running when this was written; if `models/metrics.json` has no `nested` key, rerun it.
- `docs/research/02-04` were written by free-model agents and their citations were NOT verified; 01 and 05 do not exist.
- GDELT and the final report were skipped. Paid data (tick/quote data, options history) was ruled out by the owner; Alpaca options history only starts in 2024.

## Suggested next steps
1. Switch the volatility forecast to HAR (+ a calibrated residual band) and show it on the dashboard.
2. Keep the call rule on "no calls" until a rule passes the nested check; log live predictions and grade them.
3. Write the report as a negative-result study: honest baselines, intervals, leakage controls.
