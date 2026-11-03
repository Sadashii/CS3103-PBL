# Task: AI-Powered Trading Indicator

Source: *AI_Trading_Indicator_Project_Brief.pdf* (student project brief, 27 Oct 2026), plus the
instructor's follow-up instructions. Sections 1–5 restate the brief; section 6 holds the
follow-up; sections 7–8 are our own working list of hard caps and decisions.

---

## 1. The problem

Classic indicators (RSI, MACD, Bollinger Bands, VWAP) apply fixed, hand-written formulas to past
prices. They describe what the market did; they don't learn from data or say what is likely next.

**Task:** build an indicator driven by a **trained model**. The model learns from historical
market data and predicts where the price is going. The indicator shows that prediction on a
**live chart**, updating with every new candle, so a trader can use it while trading.

### 1.1 What is needed
- A **trained model** that takes market data (OHLCV and anything else that helps) and outputs a
  prediction about the next price move.
- A **live indicator** powered by that model, working on the live market, either on TradingView
  (Pine Script) or in our own live chart web app.
- It must answer one question: **is the price more likely to go up or down over the chosen
  horizon** (seconds, 10 minutes, rest of the day...).

### 1.2 What is left to us
No restrictions on technology or method. Every choice is ours and is graded, and **every choice
must be justified with evidence in the report.**

| Choice | Options |
|---|---|
| Asset | One only: Bitcoin, gold, silver, NIFTY, or any single stock |
| Technology and model | Anything: classical ML, NNs, LSTMs, Transformers, RL, CV on chart images, NLP on news/social, statistical models, ensembles |
| Inputs and parameters | OHLCV, indicators, volume, time, other markets, sentiment, anything |
| Training history | 30 days, 100 days, 1 year or more |
| Timeframe and horizon | Seconds, minutes, hours or the full day |
| Indicator design | How it looks and behaves on the chart |

## 2. Delivery options

Pine Script cannot call a Python model or any outside server (no HTTP, no ML imports). Pine Seeds
no longer accepts new repositories and only supported daily data.

- **Option A: TradingView indicator (Pine Script).** The model is rebuilt inside Pine. Suits
  logistic regression, decision trees, small NNs. Every model input must be available on the
  live chart.
- **Option B: own live chart web app (recommended).** Model runs in Python on a small server,
  reads live candles from an API, a web page draws the chart and signals.
  - TradingView's open-source Lightweight Charts can be used for the chart.
  - Any model works; nothing has to be ported.
  - Simple stacks: FastAPI/Flask + HTML/JS, or Streamlit/Dash + Plotly.
  - **Build it for one asset only**, the same one the model is trained on.

## 3. Data

Each team collects its own dataset. **The indicator is tested on the live market, not on the
team's data.** Free sources listed: Binance (crypto, live WS), Angel One SmartAPI (Indian
stocks, live WS), yfinance (delayed), Finnhub (live WS on free tier, ~60 calls/min), Twelve Data
(delayed on free tier), Alpha Vantage (25 req/day), FRED (macro, no live). Free-tier limits
change often; check before relying on them.

Option A note: training data should come from the same or a very similar source as the
indicator's live feed, or the inputs won't match.

## 4. Expected output

A **trained model and a working live indicator for one asset.**

### 4.1 On every candle the indicator must show
- **Direction:** Up, Down, or no clear signal, for the stated horizon.
- **Confidence:** e.g. "Up, 68%".
- **A clear visual** readable in one second.
- **Stable signals:** a signal on a closed candle must never change afterwards (**no repainting**).

### 4.2 Submission
- Code repository with everything needed to reproduce the model and results.
- The trained model.
- Option B: a **link to the running web app** plus its code.
- A **report**: choices made, results, what did not work.
- A short **user guide**: how to read the indicator and when not to trust it.
- A **live demonstration** followed by questions.

### 4.3 Live testing
1. The final indicator is submitted before the test. **No changes during the test.**
2. It runs on a live chart of the asset for **5 trading days** (stocks) or 7 days (crypto).
3. **Every signal is checked** against what the price actually did over the stated horizon.
4. Results are compared with simple baselines (**always Up**, a **basic RSI rule**).

**An indicator that cannot run live, gives no signals, or repaints scores zero on the live test.**

### 4.4 Marking

| Area | Marks |
|---|---|
| Live-market performance (accuracy of confident signals, simulated profit after costs) | 40 |
| Model quality and honest validation | 20 |
| Indicator usefulness to a trader | 20 |
| Report and demonstration | 20 |
| **Total** | **100** |

## 5. Rules
1. One asset per team, declared at the start.
2. Any library, model, paper or open-source code is allowed, but it must be **cited** and the team
   must be able to explain it. A copied public indicator without our own trained model scores zero.
3. Paper trading only; no real money.
4. Every result in the report must be reproducible from the submitted code.

## 6. Instructor follow-up

- **No testing dataset will be provided.** The model is tested in the live market via a real-time
  stock market API or TradingView.
- **Arrange access to a suitable real-time market data API and keep the API details available**
  for future testing.
- Show **as much relevant information and as many useful logs as possible** in the indicator,
  while keeping the UI clean, intuitive and user-friendly.
- *Consider* **multi-scenario future price predictions**. E.g. "may drop 10 points over the next
  10 minutes", then explain what might happen if that comes true (subsequent moves) and
  alternative scenarios under different market conditions.
- Goal: informative, interactive, easy to understand, with clear predictions, supporting signals,
  confidence levels and possible outcomes.
- **Testing procedure (as relayed by the team):** the indicator will be run on the **previous
  day's session**, so graded predictions must be **intraday** and resolve within that session.
  *To confirm: on our replay app or TradingView's Bar Replay, and one day or several.*

---

## 7. Hard caps

Things we cannot trade off. "Brief" = from the PDF; "Instr." = follow-up; "Env" = imposed by our
data source or the market.

### Scoring killers (zero on the 40-mark live test)
| # | Cap | Source | How we meet it |
|---|---|---|---|
| Z1 | Must run live on the real market for the whole test | Brief 4.3 | LiveService on Alpaca IEX, auto-reconnect, REST gap fill, hosted always-on |
| Z2 | Must give signals | Brief 4.3 | A signal (Up / Down / No signal) on every closed candle |
| Z3 | Must never repaint | Brief 4.1, 4.3 | Predict only on closed bars; append-only signal log; UI draws from the log |
| Z4 | The test day must not be in the training data, and the replay must never show the model a bar after the cursor | Test procedure (section 6) | Training cutoff saved with the model; the replay day picker only offers days after it; features read only bars <= cursor |

### Must-haves
| # | Cap | Source |
|---|---|---|
| M1 | One asset, the same for training and indicator: **JPM** | Brief 1.2, 2.2, Rule 1 |
| M2 | Our own trained model; a copied indicator scores zero | Rule 2 |
| M3 | Every model input must be available live (no future data, nothing the live feed lacks) | Brief 2.1 note, 3 |
| M4 | Direction + confidence + stated horizon on every candle | Brief 4.1 |
| M5 | Readable in one second | Brief 4.1 |
| M6 | Frozen before the test; no changes during it | Brief 4.3 |
| M7 | Paper trading only | Rule 3 |
| M8 | All report results reproducible from the repo | Rule 4, 4.2 |
| M9 | Cite every library, paper and borrowed code (incl. Velora lessons) | Rule 2 |
| M10 | Every choice justified with evidence in the report | Brief 1.2 |
| M11 | Real-time API arranged; details kept available for testers | Instr. |
| M12 | Deliverables: repo, trained model, web app link + code, report, user guide, live demo | Brief 4.2 |
| M13 | Rich info and logs, but a clean UI | Instr. |

### Environment limits
| # | Limit | Consequence |
|---|---|---|
| E1 | Alpaca free plan: live data is **IEX only** (single exchange, a few % of volume) | Train on IEX too, so inputs match; use relative volume, not raw |
| E2 | **One** concurrent WebSocket connection per account | The server holds it and fans out; no second client may connect with the same keys |
| E3 | Historical REST: 200 calls/min; SIP history only older than 15 min | Fine for history; live must use IEX |
| E4 | IEX history starts **27 Jul 2020** and is essentially regular hours only (160 of 515k bars outside 9:30–16:00 ET). SIP (consolidated) history goes back to 2016 | Training since 2016 needs SIP data (see O3); signals during regular hours only |
| E5 | ~16% of minutes have no IEX trade (328 bars/day of 390); median ~10 trades/bar | Explicit, identical gap handling in training and live; no-trade minutes still get a frame |
| E6 | NYSE hours **20:00–02:30 IST** (EST, until 13 Mar 2027); 19:00–01:30 IST under EDT | Live test and demos happen at night in India; server must run unattended |
| E7 | US holidays inside the test window (e.g. Thanksgiving 26 Nov 2026 closed, 27 Nov early close 13:00 ET) | Use Alpaca's calendar/clock; show "market closed" instead of failing |
| E8 | Keys in `.env`, never committed; rotate the ones pasted in chat | `API_ACCESS.md` documents providers, endpoints and how to regenerate keys |

---

## 8. Decisions

### Made
| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Asset | **JPM** (JPMorgan Chase, NYSE) | Liquid large-cap; free real-time US data available |
| D2 | Delivery | **Option B**, own web app | Recommended; any model; no porting |
| D3 | Data source | **Alpaca free plan, download only**: `scripts/fetch_data.py` pulls bars + the NYSE calendar into local files. The app, replay and model never call Alpaca | Minimal dependency; runs offline at test time; reproducible from stored files |
| D4 | Velora | **Not reused**; lessons go in "what did not work" | No training script; model below the always-Up baseline; broken live path |
| D5 | Storage | Local parquet of 1m bars, incremental top-up. **A day is added only after that US session has closed** (store ends at the last completed close, per Alpaca's market calendar); no partial days | Replay days are always complete; nothing half-written; SIP's 15-min delay never matters |
| D6 | Dev, demo **and test** feed | **ReplayService**: history static, the chosen day replayed 1 bar/min (seekable, speed control). The examiners test on this engine | Matches the testing procedure; same interface as a live feed |
| D7 | Display time | **US Eastern (exchange) time everywhere**; the whole project works in the ET timezone | Team choice; matches the exchange, TradingView and the data |
| D8 | Chart library | TradingView Lightweight Charts | Looks like a trading platform; free |
| D9 | Price inputs | OHLCV + RSI, MACD, Bollinger Bands, VWAP (as ratios, not price levels) | Team choice; ratios avoid the Velora scaling flaw |
| D10 | Market context | Correlation with the S&P 500 (SPY) and peer banks: BAC, C, WFC, GS, MS (+ XLF sector ETF) | All on the same Alpaca feed, so available live (M3) |
| D11 | Sentiment | Explore **GDELT** (news tone/volume for JPMorgan and peers) | Free, updates every 15 min (usable live), history from 2015 |
| D12 | Training history | **Since 2016** | More regimes (2018 selloff, COVID, 2022 rates, 2023 bank crisis) |
| D13 | Horizons | **Intraday, resolving the same session** (tests replay the previous day). Daily / weekly / monthly forecasts become outlook panels, not graded | Testing procedure (section 6) |
| D14 | Compute | 8x A100 available | Not a constraint; data size is (see O7) |
| D15 | Indicator design | UI work deferred until the model exists | — |
| D16 | Where it runs | **Everything on the laptop (training too, see D25); the app runs only during demo/test.** No permanent deployment | Testers use our replay engine in our presence. Model must therefore run in real time on a laptop CPU (exported weights, no GPU needed) |
| D17 | Graded signal | **30-minute horizon on 5-minute candles**: Up / Down / No signal + confidence on every closed 5m candle (~72 per session) | Typical 30-min move ~16 bps, so a 2 bps cost takes ~13% (vs ~21% at 10 min); enough calls per test day that the score isn't luck |
| D18 | Rest-of-day forecast (instructor ask) | **Three lines from now to the close: upper and lower bound of a 90% band, plus the model's prediction (median path).** The band is the price range at each future 5m candle (a cone that widens toward the close). Re-forecast every 5m candle; each candle's lines are stored and never change | Bounds are model quantiles (5th/95th); coverage is validated (~90% of prices inside) and reported |
| D19 | Price feed | **SIP (consolidated) 1m bars since 2016** for training and replay | Matches TradingView prices and volume; gives the 2016 history; replayed days are always >15 min old, so the free plan serves them |
| D20 | Chart overlays | UI buttons to toggle **RSI, MACD, Bollinger Bands, VWAP** | Team choice; traders can check the model against familiar indicators |
| D21 | "No signal" rule | Call Up/Down only when **confidence >= threshold AND the predicted 30-min move beats costs**; otherwise "No signal". Threshold chosen on validation data only | Grading counts "accuracy of confident signals" and profit after costs |
| D22 | Cost assumption | **2 bps round-trip** headline; report also shows 1 and 5 bps | Realistic for JPM; sensitivity shows the result isn't an artefact of the cost choice |
| D23 | Macro inputs | **FRED**: 10y-2y yield curve, Fed funds rate, VIX. Model sees the **previous day's** values only | Sets the regime (calm vs stressed) and band width; kept only if validation improves |
| D24 | News sentiment | **GDELT filtered to the financial sector** (banking / markets / rates themes, plus JPMorgan and peers by name) -> headlines embedded into a **local vector DB** -> sentiment, narrative clusters and **spread** features. Measures how far and fast a story spreads, not whether it is true, since widely spread news (true or fake) can move prices | Every feature uses only articles published before the replay cursor (allowing for GDELT's ~15 min delay). History via BigQuery, size checked with a dry run first. Kept only if validation improves |
| D25 | Models and where they train | Compare in order: **baselines** (always Up, RSI rule, momentum) -> **logistic regression** -> **LightGBM** -> **GRU / small Transformer** on the laptop RTX 4060. Final = validation winner or an ensemble of the top two; it produces direction, confidence and the 5/50/95% lines. All training on the laptop (64 GB RAM, i7-13650HX, RTX 4060 8 GB) | ~210k 5m training samples since 2016: small enough for the laptop; one machine for training, replay and demo |
| D26 | Scenarios | **Full**: bull/base/bear paths, "if this move happens, what came next" from held-out history, regime-adjusted odds | Instructor's multi-scenario ask |
| D27 | Earnings / Fed days | **Flag in the UI; test as a feature**, kept only if validation improves | — |
| D28 | Replay dates | Real timestamps, shown in ET, labelled REPLAY | Follows D7 |
| D29 | Deadline | **~1 week**; long unattended runs (downloads, training) are fine | Build in priority order so a complete version exists at every stage |
| D30 | Training cutoff | **Train on data up to 1 week before the training run** (trained on 3 Nov 2026 -> last training session 27 Oct 2026). The cutoff is saved with the model; only days after it can be replayed for testing | Leaves unseen days for the examiners' replay test and for our own final out-of-sample check (Z4) |
| D31 | Prediction pipeline design | **Decided by the research step (section 9)** before any model code: model architecture, look-back, inputs | — |
| D32 | News stack | Sentiment **FinBERT** (`ProsusAI/finbert`); embeddings **`BAAI/bge-small-en-v1.5`** (384-d, fast on the 4060); vector DB **LanceDB** (embedded, file-based, works offline). All open-weight, run locally | Free, no server, laptop-friendly; FinBERT is the standard finance sentiment baseline |
| D33 | 90% band and scenario paths | **Conformalized quantile regression (CQR)**: the model predicts 5/50/95% quantiles per future 5m step to the close; a recent calibration window corrects the width so ~90% coverage holds. Bull/base/bear paths come from those quantiles | Plain quantile models are often over-confident; CQR gives a coverage guarantee we can show in the report |

### Open
| # | Decision | Options | Recommendation |
|---|---|---|---|
| O1 | *(decided, see D26-D29)* | | |
| O2 | *(decided, see D17)* | | |
| O3 | *(decided, see D19)* | | |
| O4 | *(decided, see D21)* | | |
| O5 | *(decided, see D22)* | | |
| O6 | *(decided, see D23)* | | |
| O7 | *(decided, see D25)* | | |
| O8 | *(decided, see D24)* | | |
| O9 | *(decided, see D26-D29)* | | |
| O10 | *(decided, see D16)* | | |
| O11 | *(decided, see D26-D29)* | | |
| O12 | *(dropped: no live feed at test time, see D3)* | | |
| O13 | *(decided, see D26-D29)* | | |
| O14 | Declare the asset to the instructor | — | Rule 1 says declare at the start: send "JPM" in writing now |


---

## 9. Next: research step (run in a fresh session)

Rule: **free models only** (OpenRouter / opencode ids priced $0; check the live free list and make a test
call first; ~1,000 free requests/day). No paid model unless the user asks and confirms that specific use.

Spawn 4–5 research sub-agents, one topic each. Each writes `docs/research/<topic>.md` with cited papers
(title, authors, year, link), techniques, evidence for or against, and a recommendation for **this**
project (JPM, 5m candles, 30-min graded horizon, rest-of-day 90% band, SIP 1m data since 2016, laptop RTX 4060):

1. **Intraday direction prediction**: 5–30 min stock forecasting; LightGBM vs GRU/LSTM/Transformer/TCN; which features work; label design (triple-barrier, cost-aware).
2. **Honest validation**: purged and embargoed walk-forward CV, calibration (isotonic / Platt), choosing confidence thresholds, backtest overfitting (deflated Sharpe, PBO).
3. **Uncertainty bands and scenarios**: quantile regression, conformal prediction (CQR, adaptive conformal for time series), scenario path generation, regime-conditioned odds.
4. **News sentiment and spread**: GDELT for finance, FinBERT, measuring story spread/virality, evidence that widely spread (incl. false) news moves prices.
5. **Cross-asset and regime signals**: lead–lag of JPM vs SPY, XLF and peer banks; regime detection (HMM, volatility states); macro (yield curve, VIX) effects on banks.

Then combine the reports into `docs/PIPELINE.md`: the chosen prediction pipeline (D31) with reasons,
which experiments the report must include, and the build order (see the week plan in the chat summary:
UI on replay -> features/labels -> baselines/logistic/LightGBM -> signal log + grading -> scenarios ->
sequence model -> GDELT -> docs).
