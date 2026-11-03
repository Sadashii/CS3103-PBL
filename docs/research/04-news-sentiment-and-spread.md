# Research Report: News Sentiment and Spread (Topic 4)

**Project:** AI Trading Indicator for JPM  
**Date:** 2026-11-03  
**Prepared for:** CS3103-PBL — Section 9, research sub-agent 4  

---

## 1. Summary

This report surveys the literature on using news sentiment and story spread/virality features for intraday stock prediction, with focus on a single large-cap US bank (JPM), 5-minute candles, a 30-minute forecast horizon, and the requirement that all features be computable in real time on a laptop CPU. Key findings:

- **GDELT GKG** provides a free, 15-minute-updated, globally comprehensive news feed with theme codes, tone scores, and entity tags, queryable via BigQuery (costs must be checked with a dry run). It covers 2015–present, matching our 2016+ training window.
- **FinBERT** (Araci 2019; Yang et al. 2020) remains the standard open-weight finance sentiment baseline. Newer LLM-based models (DeBERTa, RoBERTa, GPT-4) show marginally better classification accuracy but are heavier; FinBERT is laptop-friendly and well-understood.
- **Embedding + vector search** using **BAAI/bge-small-en-v1.5** (384-dim, MIT licence, fast on RTX 4060) and **LanceDB** (embedded, file-based, no server) enables clustering headlines into narrative themes and measuring story spread (how many sources, how fast, cross-lingual reach).
- **Evidence on spread/virality**: Vosoughi et al. (2018, *Science*) showed false news spreads faster and wider than true news; Anagnostopoulos et al. (2014) confirmed homophily/polarization drives virality. For finance, Zhang (2025) found *sentiment dispersion* and *article impact* (GDELT GKG fields) are key predictive features for macro assets. No paper directly tests intraday JPM prediction with spread features — this is our original contribution opportunity.
- **Intraday predictive value**: Mixed. Shah et al. (2018) got 70.6% directional accuracy on pharma stocks with daily news. Srinivas et al. (2023) used 500k NIFTY articles with LSTMs — unclear if sentiment added value over price-only. Siala et al. (2026) found sentiment features *slightly* help some DL models (LSTM, PatchTST) for next-day direction. Liao et al. (2026) got 7.1% MAE reduction using stock-name attention over news embeddings for multi-stock daily prediction. **No published work tests 5-min/30-min horizon on a single large-cap with spread features.**
- **Leakage/timing pitfalls**: GDELT updates every 15 min but ingestion delay varies (minutes to hours). Publication timestamp ≠ GDELT ingestion timestamp. Must use *publication date* from article metadata, not GDELT processing time, and enforce a conservative embargo (e.g., only use articles published ≥15 min before the 5-min candle close). BigQuery `DATEADD` filtering and local caching are essential.

**Recommendation for this project**: Build a **hybrid news stack** — GDELT GKG (finance themes + tone + entity counts) → FinBERT sentiment per article → bge-small embeddings → LanceDB vector search for narrative clusters → *spread features* (source count, cross-lingual reach, velocity, tone dispersion) → feed into LightGBM/GRU alongside price/macro features. Validate with purged walk-forward CV. Original contribution: **intraday spread-virality features for a single large-cap** (JPM) at 5-min resolution, with rigorous leakage control.

---

## 2. Table of Cited Sources

| # | Title | Authors | Year | Link | What It Shows | Caveats |
|---|-------|---------|------|------|---------------|---------|
| 1 | GDELT 2.0: Our Global World in Realtime | GDELT Project | 2015 | https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/ | GKG updates every 15 min; 2,300+ emotions/themes; BigQuery tables (Events, Mentions, GKG) updated every 15 min; 65-language translation; history from 1979 (v2.0 from Feb 2015) | BigQuery costs not stated; need dry run; 15-min latency means article publication time ≠ availability time |
| 2 | Comparison of Common Crawl News & GDELT | El Ouadi, Beskow | 2026 | https://arxiv.org/abs/2610.00587 | GDELT relies on broadcasts, prints, web news globally; Common Crawl focuses on news sites via web crawl; considerable source differences | Conference paper (SysCon 2024); not finance-specific |
| 3 | Visual and Predictive Analytics on Singapore News: Experiments on GDELT, Wikipedia, and ^STI | Phua, Feng, Ji, Soh | 2014 | https://arxiv.org/abs/1404.1996 | Decision tree on daily GDELT feature counts predicted Straits Times Index; feature engineering from row-level GKG to temporal resolution | Singapore market; daily horizon; pre-GDELT 2.0 (no GKG themes/tone as in v2.0) |
| 4 | Interpretable ML for Macro Alpha: A News Sentiment Case Study | Zhang | 2025 | https://arxiv.org/abs/2505.16136 | GDELT GKG + FinBERT → daily sentiment indices (mean tone, dispersion, event impact) → XGBoost → Sharpe 5.87 (EUR/USD), 4.65 (USD/JPY, ZN); SHAP: dispersion & impact key | Daily horizon; macro assets (FX, futures); not single-stock intraday; OOS 2017–Apr 2025 |
| 5 | FinBERT: Financial Sentiment Analysis with Pre-trained Language Models | Araci | 2019 | https://arxiv.org/abs/1908.10063 | FinBERT (BERT fine-tuned on financial corpus) beats generic BERT & SOTA on two finance sentiment datasets (FiQA, Financial PhraseBank); works with smaller training sets | MSc thesis; datasets are sentence-level, not full articles; no intraday trading evaluation |
| 6 | FinBERT: A Pretrained Language Model for Financial Communications | Yang, UY, Huang | 2020 | https://arxiv.org/abs/2006.08097 | FinBERT pretrained on large financial comms corpora; beats BERT on three financial sentiment tasks; code/models at github.com/yya518/FinBERT | Same FinBERT family as Araci 2019; no trading evaluation |
| 7 | Impact of LLMs News Sentiment Analysis on Stock Price Movement Prediction | Siala, Khanfir, Papadakis | 2026 | https://arxiv.org/abs/2602.00086 | DeBERTa (75%) > RoBERTa/FinBERT for sentiment classification; ensemble ≈80%; sentiment features *slightly* help LSTM, PatchTST, tPatchGNN classifiers & PatchTST/TimesNet regressors for next-day direction | Next-day horizon; multiple stocks; "slight" benefit; ICLR 2026 workshop paper |
| 8 | Financial Sentiment Analysis Using FinBERT with Application in Predicting Stock Movement | Jiang, Zeng | 2023/2025 | https://arxiv.org/abs/2306.02136 | FinBERT-LSTM beats BERT-LSTM, LSTM, ARIMA on stock news dataset; sentiment enhances market fluctuation anticipation | Dataset not described in abstract; revised 2025 with metric fixes; no intraday horizon |
| 9 | Sentiment-Driven Prediction of Financial Returns: A Bayesian-Enhanced FinBERT Approach | Cestari, Formentin | 2024 | https://arxiv.org/abs/2403.04427 | FinBERT on StockTwits tweets + Bayesian RFE → F1 >70% on SPY ETF; higher cumulative profit in backtest | Daily horizon; social media (StockTwits), not news; workshop paper (not peer-reviewed) |
| 10 | BERTopic-Driven Stock Market Predictions: Unraveling Sentiment Insights | Zhu, Yen | 2024 | https://arxiv.org/abs/2404.02053 | BERTopic on stock comments + sentiment + DL models → improved prediction; topics give insights into volatility/trends | Comments/social media, not news; no horizon specified |
| 11 | PreBit: A Multimodal Model with Twitter FinBERT Embeddings for Extreme Price Movement Prediction of Bitcoin | Zou, Herremans | 2022/2023 | https://arxiv.org/abs/2206.00648 | FinBERT sentence embeddings + CNN + candlestick/technical/correlated assets → multimodal ensemble predicts extreme Bitcoin moves; profitable strategy vs hold/MA | Bitcoin; daily/extreme moves; Twitter not news; Expert Systems w/ Applications 2023 |
| 12 | News Sentiment Embeddings for Stock Price Forecasting | Qayyum | 2025 | https://arxiv.org/abs/2507.01970 | WSJ headlines + OpenAI embeddings + PCA → 40%+ improvement vs no-headlines for SPY daily prediction; 390 models trained | Daily horizon; WSJ only; OpenAI embeddings (not local/open); preliminary results |
| 13 | Effects of Daily News Sentiment on Stock Price Forecasting | Srinivas, Gadela, Sabu, Das, Nath, Datla | 2023 | https://arxiv.org/abs/2308.08549 | ~500k NIFTY50 articles over 3.7 years; LSTM with/without sentiment scores from multiple libraries; comparison | Indian market; daily; conference paper; unclear if sentiment added value |
| 14 | Predicting the Effects of News Sentiments on the Stock Market | Shah, Isah, Zulkernine | 2018 | https://arxiv.org/abs/1812.04199 | Dictionary-based financial sentiment model → 70.59% directional accuracy on pharma stocks (short-term) | Small sample (pharma only); dictionary-based (not transformer); 4-page conference paper |
| 15 | Generalized Stock Price Prediction for Multiple Stocks Combined with News Fusion | Liao, Lee, Cheng, Chen, Lee, Wang | 2026 | https://arxiv.org/abs/2603.19286 | Stock-name embeddings + attention pooling (self/cross/position-aware) filter news relevance → 7.11% MAE reduction vs baseline; single model across stocks | Daily horizon; accepted to JISE; generalized multi-stock, not single-stock intraday |
| 16 | Viral Misinformation: The Role of Homophily and Polarization | Anagnostopoulos et al. | 2014 | https://arxiv.org/abs/1411.2893 | 1.2M Facebook users; engagement correlates with homophily; 4,709 false claims — virality driven by polarization, not hubs/influencers | Social media (Facebook), not finance news; false claims are satirical |
| 17 | The Spread of True and False News Online | Vosoughi, Roy, Aral | 2018 | https://www.science.org/doi/10.1126/science.aap9559 | False news spreads faster, wider, deeper than true news on Twitter (126k cascades, 3M users); top 1% false cascades reach 1k-100k people | Twitter; general news, not finance; paywalled (abstract only accessed) |
| 18 | BAAI/bge-small-en-v1.5 Model Card | BAAI | 2023+ | https://huggingface.co/BAAI/bge-small-en-v1.5 | 384-dim, 33.4M params, MIT licence, MTEB avg 62.17, fast on GPU/CPU; sentence-transformers, FlagEmbedding, ONNX support | No finance-specific evaluation; general-purpose embedding |
| 19 | LanceDB Documentation | LanceDB Team | 2024+ | https://lancedb.github.io/lancedb/ | Embedded vector DB, file-based, no server, Python/JS/Java/Rust; supports ANN search, filtering, metadata | Newer project; less battle-tested than Chroma/FAISS; but fits "laptop, offline" requirement |
| 20 | Reconstructing Diffusion Model for Virality Detection in News Spread Networks | Jain, Garg, Jain | 2019/2021 | Crossref DOI: 10.4018/ijkss.2019010102 | SI diffusion model on news spread networks (business, tech, entertainment, health); diffusion rate predicts virality | Book chapter; not finance; SI model may not capture finance news dynamics |

---

## 3. Techniques Compared

| Technique | Input | Output | Compute | Licence | Fit for This Project |
|-----------|-------|--------|---------|---------|---------------------|
| **GDELT GKG (BigQuery)** | Global news (65 langs, 15-min updates) | Theme codes, tone, entities, counts, quotes, amounts, imagery | Query cost (TB scanned); free tier 1 TB/mo | Free data; BigQuery pay-per-query | **Core source** — finance themes, tone, entity tags match JPM/peers; 2015+ history |
| **FinBERT (Araci 2019 / Yang 2020)** | Article text / headlines | 3-class sentiment (Pos/Neg/Neu) + scores | ~110M params; runs on RTX 4060 / CPU | Apache-2.0 (github.com/yya518/FinBERT) | **Standard baseline** — finance-domain, open-weight, laptop-friendly |
| **DeBERTa / RoBERTa / Ensemble (Siala 2026)** | Same | Better sentiment accuracy (75–80%) | Heavier; DeBERTa-large ~300M+ | Various (DeBERTa MIT) | Marginal gain vs cost; skip for v1 |
| **bge-small-en-v1.5 + LanceDB** | Headlines / lead paragraphs | 384-dim embeddings; ANN search, clustering | 33M params; <1s/1k on 4060; LanceDB embedded | MIT (model); Apache-2.0 (LanceDB) | **Embedding + vector search** — narrative clusters, spread measurement |
| **BERTopic (Grootendorst 2022)** | Embeddings | Topics + representative docs | Adds UMAP/HDBSCAN; moderate | MIT | Alternative to LanceDB clustering; more complex |
| **Dictionary/VADER/FinVADER** | Text | Lexicon-based scores | Negligible | Various | Baseline only; inferior to FinBERT |
| **Spread/Virality Features (Original)** | GDELT GKG counts, FinBERT tones, embedding clusters | Source count, cross-lingual reach, velocity, tone dispersion, cluster size/growth | Computed locally from cached GKG | — | **Our original contribution** — no paper does this at 5-min for single large-cap |

---

## 4. Evidence For / Against

### For News Sentiment Having Intraday Predictive Value
- **Zhang (2025)**: GDELT GKG tone dispersion & article impact are top SHAP features for *daily* macro assets (Sharpe 4.6–5.9). Proves GKG fields carry signal.
- **Siala et al. (2026)**: Sentiment features *slightly* improve LSTM/PatchTST for *next-day* direction across multiple stocks.
- **Liao et al. (2026)**: Stock-name attention over news embeddings reduces MAE 7.1% for *daily* multi-stock prediction.
- **Qayyum (2025)**: Headline embeddings improve SPY *daily* prediction by 40%+.
- **Shah et al. (2018)**: Even dictionary-based sentiment got 70.6% directional accuracy on pharma (short-term).

### Against / Caveats
- **Srinivas et al. (2023)**: 500k NIFTY articles, LSTM with/without sentiment — abstract doesn't claim clear win; Indian market, daily.
- **Siala et al. (2026)**: Benefit is "slight" and model-dependent; next-day not intraday.
- **No paper** tests 5-min candles, 30-min horizon, single large-cap, with spread/virality features.
- **GDELT latency**: 15-min update cycle + variable ingestion delay → risk of look-ahead if not careful.
- **Finance news vs social media**: Twitter/StockTwits (Cestari 2024, Zou 2022) may have different dynamics than GDELT's global news wire feed.

### For Spread/Virality Moving Prices (Including False News)
- **Vosoughi et al. (2018, *Science*)**: False news spreads faster/wider than true news (Twitter, 126k cascades). Mechanism: novelty + emotional reaction.
- **Anagnostopoulos et al. (2014)**: Virality driven by homophily/polarization, not influencers — 4,709 false claims on Facebook.
- **Zhang (2025)**: *Sentiment dispersion* (GKG tone variance) and *article impact* (GKG impact score) are key predictive features for macro assets.
- **Jain et al. (2019/2021)**: SI diffusion model on news spread networks predicts virality by domain (business, tech, etc.).

### Against Direct Finance Intraday Evidence
- None of the above are intraday, single-stock, or use GDELT spread metrics for finance.
- Finance-specific virality studies are absent — this is the gap.

---

## 5. Concrete Recommendation for This Project

### 5.1 Hybrid News Stack Architecture

```
GDELT GKG (BigQuery, 15-min) 
  → Filter: finance themes (banking, markets, rates) + entities (JPM, BAC, C, WFC, GS, MS, XLF)
  → Fetch article metadata (URL, publication datetime, source, language, tone, themes, counts)
  → Local cache (Parquet, partitioned by date)
  
Per-article processing (offline batch, incremental daily):
  → FinBERT (ProsusAI/finbert) → 3-class sentiment + confidence
  → bge-small-en-v1.5 (BAAI) → 384-dim embedding (title + lead para)
  → Store in LanceDB: {embedding, finbert_score, pub_time, source, lang, themes, tone, gkg_counts}

Per-5-min candle (real-time, laptop CPU):
  → Query LanceDB: articles with pub_time in [candle_close - lookback, candle_close - 15min_embargo]
  → Compute spread features per narrative cluster (ANN search, cos_sim > threshold):
      • source_count: unique sources covering cluster
      • cross_lingual_reach: unique languages
      • velocity: articles/min in last 30/60/120 min
      • tone_dispersion: std(FinBERT score) across cluster
      • cluster_size_growth: Δsize vs previous candle
      • max_impact: max GKG impact score in cluster
      • entity_co_mention: count of peer-bank co-mentions
  → Aggregate: weighted avg (by recency) + dispersion stats
  → Concatenate with price/macro/peer features → LightGBM / GRU
```

### 5.2 Original Contribution
**Intraday Spread-Virality Features for a Single Large-Cap (JPM) at 5-Minute Resolution**

No published work combines:
- GDELT GKG finance themes + tone + entity tags
- FinBERT sentiment per article
- Embedding-based narrative clustering (bge-small + LanceDB)
- Spread/virality metrics (source count, cross-lingual reach, velocity, tone dispersion, cluster growth)
- Rigorous leakage control (publication-time embargo, purged walk-forward CV)
- 30-minute horizon on 5-minute candles for a single liquid stock (JPM)

### 5.3 Experiments the Final Report Must Include
1. **Ablation**: Price-only vs +FinBERT sentiment vs +spread features vs +peer/macro → LightGBM & GRU
2. **Horizon sensitivity**: 5/10/15/30/60 min horizons
3. **Embargo sweep**: 5/10/15/30 min publication-time embargo → measure leakage
4. **Cluster threshold sweep**: cos_sim 0.6/0.7/0.8 for narrative grouping
5. **Lookback window**: 30/60/120/240 min article lookback
6. **Purged walk-forward CV**: 6-month train / 1-month test, 4-week embargo (per Section 9, Topic 2)
7. **Calibration**: Isotonic/Platt on validation → confidence thresholds for Up/Down/No-signal (D21)
8. **Cost-adjusted profit**: 1/2/5 bps round-trip (D22) on confident signals only
9. **Coverage test**: CQR 90% band (D33) — empirical coverage on test folds
10. **Regime analysis**: Performance split by VIX / yield curve / earnings days (D23, D27)

---

## 6. Open Risks

| Risk | Mitigation |
|------|------------|
| **GDELT BigQuery cost exceeds budget** | Dry run `SELECT ... FROM gdelt-bq.gdeltv2.gkg WHERE _PARTITIONDATE BETWEEN ...` with `LIMIT 1000` first; estimate TB scanned; cache locally daily; only query new partitions |
| **Ingestion delay >15 min causes leakage** | Use article `DocumentIdentifier` publication datetime (GKG field `DATE`), not GKG processing time; enforce 15-min embargo; log delay distribution |
| **FinBERT too slow for per-article batch** | Batch inference on RTX 4060 (110M params); ~1k articles/min; daily batch <5 min |
| **LanceDB ANN index rebuild cost** | Incremental `add`; rebuild HNSW weekly; 384-dim is small |
| **Spread features noisy at 5-min** | Aggregate over 30/60 min windows; use EMA; require min cluster size (e.g., ≥3 sources) |
| **No signal on low-news days** | "No signal" rule (D21) handles this; fallback to price-only model |
| **Peer bank news contaminates JPM signal** | Separate JPM-specific vs sector clusters; ablate peer features |
| **Earnings/Fed days dominate** | Flag days (D27); test as feature; consider separate regime model |
| **BigQuery schema changes** | Pin GKG codebook version; monitor GDELT blog; schema-on-read in local Parquet |

---

## 7. Licences of Key Dependencies

| Component | Licence | Commercial Use |
|-----------|---------|----------------|
| GDELT Data | Free for research/non-commercial; commercial terms on request | Check GDELT terms for live trading |
| FinBERT (ProsusAI/finbert) | Apache-2.0 | Yes |
| FinBERT (yya518/FinBERT) | Apache-2.0 | Yes |
| BAAI/bge-small-en-v1.5 | MIT | Yes |
| LanceDB | Apache-2.0 | Yes |
| LightGBM | MIT | Yes |
| PyTorch / Transformers | BSD-3 / Apache-2.0 | Yes |

---

*End of Report*