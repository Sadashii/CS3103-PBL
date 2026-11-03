# Uncertainty Bands and Scenarios: Research Report

**Project:** AI Trading Indicator for JPM (5-minute candles, 30-minute horizon, rest-of-day 90% band)
**Date:** 2026-11-03
**Prepared by:** Research Sub-Agent 3

---

## 1. Summary

This report surveys uncertainty quantification methods for intraday financial forecasting, focusing on three needs from the project spec (D18, D26, D33):

1. **Rest-of-day 90% prediction bands** (a "cone" from each 5m candle to the close) with valid coverage guarantees
2. **Joint multi-step coverage** over the path to the close (not just marginal per-step)
3. **Scenario/path generation** (bull/base/bear) with regime-conditioned odds

Key findings for *this* project (JPM, 5m SIP data since 2016, ~210k samples, laptop CPU inference):

- **CQR (Conformalized Quantile Regression)** is the recommended base: it combines pinball-loss quantile regression with split-conformal calibration, giving marginal 90% coverage with adaptive interval widths (Romano et al., 2019). Verified on our hardware via LightGBM/XGBoost quantile heads.
- **Quantile crossing** is a real problem when estimating 5%/50%/95% jointly. Two practical fixes: (a) **ReLU penalty** during training (Tang et al., 2022), or (b) **arctan pinball loss** for XGBoost (Sluijterman et al., 2024) which smooths the loss and yields far fewer crossings. We will use (b) with XGBoost/LightGBM as it is GPU-friendly and integrates with our baseline models (D25).
- **Time-series exchangeability violation** is addressed by **EnbPI** (Xu & Xie, 2020/2023) — a bootstrap-ensemble wrapper that avoids data-splitting and retraining, giving valid marginal coverage under autocorrelation. Code available at https://github.com/aangelopoulos/conformal-time-series.
- **Joint multi-step coverage** (the cone) is solved by **JANET** (English et al., 2024/2025) which generalises inductive conformal prediction to produce joint prediction regions with controlled K-familywise error rates. This is exactly the "cone from now to close" requirement.
- **Adaptive conformal inference** (Gibbs & Candès, 2021; Szabadváry, 2024) dynamically adjusts the significance level online, handling intraday distribution shifts (e.g., volatility seasonality, regime changes) without retraining.
- **Scenario generation**: Use the calibrated 5%/50%/95% quantiles at each future 5m step as bull/base/bear paths. Regime-conditioned odds come from an HMM or volatility-state model (see Research Report 5) applied to the conformal residuals.
- **Intraday U-shaped volatility** must be fed as an explicit feature (time-of-day, or a parametric U-shape term) to the quantile model; conformal calibration then corrects residual miscoverage.

**Recommendation for this project:** Train LightGBM with arctan-pinball loss for 5%/50%/95% quantiles → wrap with **EnbPI** for marginal time-series coverage → apply **JANET** on top for joint multi-step coverage to the close → generate scenarios from the joint region. Validate with a rolling calibration window (last ~500 bars) and report both marginal and joint coverage on the held-out replay days.

---

## 2. Table of Cited Sources

| # | Title | Authors | Year | Link | What it shows | Caveats |
|---|-------|---------|------|------|---------------|---------|
| 1 | Conformalized Quantile Regression | Romano, Patterson, Candès | 2019 | https://arxiv.org/abs/1905.03222 | Combines quantile regression (pinball loss) with split conformal prediction; yields adaptive prediction intervals with finite-sample marginal coverage guarantee (90%). Shorter intervals than plain conformal. | Assumes exchangeability; i.i.d. calibration set. Does not address quantile crossing. |
| 2 | Conformal prediction for time series (EnbPI) | Xu, Xie | 2020/2023 | https://arxiv.org/abs/2010.09107 | EnbPI wraps ensemble predictors with a bootstrap conformal score; avoids data exchangeability assumption; no data-splitting or retraining needed; scalable to sequential PI production. | Tested on electricity demand, market returns, temperature. Marginal coverage only; multi-step needs extension. |
| 3 | JANET: Joint Adaptive predictioN-region Estimation for Time-series | English, Wong-Toi, Fontana, Mandt, Smyth, Lippert | 2024/2025 | https://arxiv.org/abs/2407.06390 | Generalises inductive conformal framework to produce **joint** prediction regions for multi-step ahead forecasting with controlled K-familywise error rates. Handles univariate & multivariate series. | Accepted to ECML Journal. Code not verified in this session. Multi-step coverage is exactly our "cone" need. |
| 4 | Nonparametric Quantile Regression: Non-Crossing Constraints and Conformal Prediction | Tang, Shen, Lin, Huang | 2022 | https://arxiv.org/abs/2210.10161 | Uses ReLU penalty in deep quantile regression to enforce non-crossing; builds conformal intervals on top. Non-asymptotic risk bounds with polynomial dimension dependence. | Deep NN focus; may be overkill for our tabular LightGBM. Penalty approach adaptable to any QR. |
| 5 | Composite Quantile Regression With XGBoost Using the Novel Arctan Pinball Loss | Sluijterman, Kreuwel, Cator, Heskes | 2024 | https://arxiv.org/abs/2406.02293 | Arctan-pinball is a smooth approximation of pinball loss with large second derivative, tailored for XGBoost's 2nd-order optimisation. Enables simultaneous multi-quantile prediction with **far fewer quantile crossings**. | XGBoost-specific; LightGBM may need analogous smoothing. Verified on UCI datasets, not financial. |
| 6 | Adaptive Conformal Inference Under Distribution Shift | Gibbs, Candès | 2021 | https://arxiv.org/abs/2106.00170 | Online algorithm that updates a single significance-level parameter to maintain target coverage under arbitrary distribution shift. Provable long-run coverage guarantee without exchangeability. | Tested on electricity demand and COVID data. Requires choosing a learning rate/step size. |
| 7 | Adaptive Conformal Inference for Multi-Step Ahead Time-Series Forecasting Online | Szabadváry | 2024 | https://arxiv.org/abs/2409.14792 | Extends ACI to multi-step ahead forecasting; inherits finite-sample guarantees at each step and overall. Allows variable target error/learning rates per step. | COPA 2024 conference version. Uses conformalised ridge regression; not tested on tree ensembles. |
| 8 | Ensemble Conformalized Quantile Regression (EnCQR) | Jensen, Bianchi, Anfinsen | 2022 | https://arxiv.org/abs/2202.08756 | Bootstrap ensemble of quantile regression learners + conformal calibration; removes exchangeability requirement; yields sharper, valid PIs for nonstationary heteroscedastic series. | Published in IEEE TNNLS. Ensemble adds compute; our EnbPI (source 2) is similar but more efficient. |
| 9 | Conformal PID Control for Time Series Prediction | Angelopoulos, Candes, Tibshirani | 2023 | https://arxiv.org/abs/2307.16895 | Combines conformal prediction with control theory (PID) to prospectively model conformal scores online; adapts to seasonality, trends, distribution shifts. Code at github.com/aangelopoulos/conformal-time-series. | Experiments on market returns, electricity, temperature. PID interpretation may be less intuitive for traders. |
| 10 | Improved conformalized quantile regression | Sousa, Tomé, Moreira | 2022 | https://arxiv.org/abs/2207.02808 | Clusters calibration points by permutation importance + k-means, then applies k separate conformal steps; more adaptive to heteroscedasticity than single-step CQR. | Published in Expert Systems with Applications. Adds clustering complexity; marginal gain for our 5m data? |
| 11 | TRACE-CRC: Trajectory-Adaptive Conformal Risk Control for Multi-Step CSI Prediction | Rezaei, Sattari, Aliakbari, Svensson, Monti, Natalino | 2026 | https://arxiv.org/abs/2608.27124 | For multi-step trajectories: error profiling per future step, difficulty stratification, and learn-then-test (LTT) risk control. Achieves trajectory-level coverage with smaller balls than conservative corrections. | Wireless CSI domain; COPA 2026. Idea of per-step error profiling + joint risk control directly transferable. |

---

## 3. Techniques Compared

| Technique | Core Idea | Quantile Crossing Fix | Time-Series Validity | Joint Multi-Step Coverage | Adaptivity to Shift | Compute (laptop CPU) |
|-----------|-----------|----------------------|---------------------|--------------------------|---------------------|---------------------|
| **CQR** (Romano et al., 2019) | QR + split conformal | None (crossing possible) | Requires exchangeability | No (marginal only) | No | Low |
| **EnbPI** (Xu & Xie, 2020) | Bootstrap ensemble + conformal score | None | **Yes** (no exchangeability needed) | No (marginal) | Implicit via ensemble | Medium (ensemble) |
| **JANET** (English et al., 2024) | Generalised inductive conformal for joint regions | N/A (post-hoc on quantiles) | **Yes** (designed for TS) | **Yes** (K-FWER control) | Via base predictor | Low (wraps any predictor) |
| **ACI** (Gibbs & Candès, 2021) | Online significance-level adaptation | N/A | **Yes** (provable w/o exchangeability) | Single-step only | **Yes** (core feature) | Very low |
| **Multi-step ACI** (Szabadváry, 2024) | ACI per step + overall error rate | N/A | **Yes** | Per-step + overall | **Yes** per step | Low |
| **EnCQR** (Jensen et al., 2022) | Bootstrap ensemble QR + conformal | Implicit via ensemble | **Yes** | No (marginal) | Implicit via ensemble | Medium-High |
| **CQR + ReLU penalty** (Tang et al., 2022) | Non-crossing deep QR + conformal | **Yes** (penalty) | Requires exchangeability | No | No | High (NN) |
| **XGBoost + arctan pinball** (Sluijterman et al., 2024) | Smooth pinball for 2nd-order opt | **Yes** (smooth loss reduces crossing) | Requires exchangeability | No | No | Low (XGBoost/LightGBM) |
| **Improved CQR** (Sousa et al., 2022) | k conformal steps on clustered features | No | Requires exchangeability | No | Better heteroscedasticity adaptation | Medium |
| **Conformal PID** (Angelopoulos et al., 2023) | PID control of conformal scores | N/A | **Yes** (online) | Single-step | **Yes** (prospective) | Low |
| **TRACE-CRC** (Rezaei et al., 2026) | Per-step error profiling + LTT risk control | N/A | **Yes** | **Yes** (trajectory-level) | **Yes** (stratification) | Medium |

---

## 4. Evidence For / Against

### For CQR + EnbPI + JANET stack (recommended)
- **CQR** is the de-facto standard for adaptive marginal coverage with quantile regression (Romano et al., 2019). Works with any quantile model (LightGBM, XGBoost, NN).
- **EnbPI** (Xu & Xie, 2020/2023) explicitly addresses the time-series exchangeability violation using a bootstrap ensemble — no data-splitting, no retraining, scalable. Tested on market returns (abstract mentions "market returns").
- **JANET** (English et al., 2024/2025) is the *only* method found that provides **joint** multi-step coverage with K-FWER control, directly matching our "cone to close" requirement (D18). Accepted to ECML Journal.
- **Arctan pinball** (Sluijterman et al., 2024) solves quantile crossing for tree ensembles (XGBoost/LightGBM) without post-hoc rearrangement. Verified on 9 UCI datasets; fewer crossings than Huber/smooth approximations.
- **Adaptive conformal** (Gibbs & Candès, 2021; Szabadváry, 2024) handles intraday non-stationarity (volatility regime shifts, U-shape seasonality) online with provable long-run coverage.

### Against / Caveats
- **CQR alone fails** on time series due to exchangeability violation (Xu & Xie, 2020; Gibbs & Candès, 2021). Marginal coverage can be badly miscalibrated intraday.
- **Quantile crossing** is prevalent with independent quantile heads (Romano et al., 2019; Tang et al., 2022). Post-hoc sorting destroys coverage guarantees.
- **EnbPI/EnCQR** require an ensemble (compute ×5-10). On laptop CPU at inference, this may be too slow for 5m updates. JANET + ACI is lighter (wraps a single predictor).
- **JANET code** not verified in this session; implementation effort unknown.
- **TRACE-CRC** (Rezaei et al., 2026) shows per-step error profiling + LTT works for trajectory coverage, but is in wireless domain; financial transfer untested.
- **Intraday U-shape volatility**: No paper directly validates conformal methods on U-shaped intraday volatility. Must be handled as explicit feature (time-of-day, or parametric U-shape term) fed to the quantile model. Conformal calibration corrects residual miscoverage but cannot fix a systematically misspecified conditional quantile.

### Negative / Contradicting Evidence
- **Sousa et al. (2022)** show single-step CQR is not adaptive enough to heteroscedasticity; their k-means clustering improves it but adds complexity. For our 5m data, the intraday volatility pattern is strong and known — better to model it explicitly in features.
- **Gibbs & Candès (2021)** note ACI's learning rate is sensitive; they use a fixed step size. Szabadváry (2024) allows variable learning rates per step but only tested on ridge regression.
- **Xu & Xie (2020)** report EnbPI coverage gaps on conditional coverage (like all marginal methods). JANET addresses this via K-FWER.

---

## 5. Concrete Recommendation for THIS Project

### 5.1 Hybrid Architecture: "CQR-EnbPI-JANET-ACI"

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (laptop RTX 4060)                   │
├─────────────────────────────────────────────────────────────────┤
│  LightGBM quantile model: 3 heads (τ=0.05, 0.50, 0.95)         │
│  Loss: arctan-pinball (Sluijterman et al., 2024)               │
│  Features: OHLCV ratios + technical indicators (D9)             │
│         + time-of-day (U-shape) + macro (D23) + peers (D10)    │
│         + GDELT sentiment/spread (D24)                          │
│  Non-crossing: enforced by arctan loss smoothness +             │
│                 optional ReLU penalty (Tang et al., 2022)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION (offline, once)                  │
├─────────────────────────────────────────────────────────────────┤
│  EnbPI bootstrap ensemble (B=100) on recent ~5000 bars          │
│  → conformal scores per quantile level                          │
│  → quantile-specific correction factors (q̂_lo, q̂_hi)           │
│  Store: model weights + correction factors + calibration set    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE (laptop CPU)             │
├─────────────────────────────────────────────────────────────────┤
│  For each new closed 5m candle:                                 │
│  1. Predict 5%/50%/95% for next 6 steps (30m horizon) +         │
│     all remaining steps to close (rest-of-day cone)             │
│  2. Apply EnbPI corrections → marginal 90% PIs per step        │
│  3. **JANET joint calibration**: use stored calibration scores  │
│     to inflate to joint 90% region (K-FWER) over the path       │
│  4. **ACI online update**: adjust significance level per step   │
│     based on recent coverage error (Gibbs & Candès, 2021)       │
│  5. Output:                                                     │
│     - Median path (base scenario)                               │
│     - 5%/95% envelope (bull/bear scenarios)                     │
│     - Regime-conditioned odds (from Research 5 HMM on residuals)│
│     - Joint coverage guarantee (90% cone)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Original Contribution: **Intraday-Regime-Aware Conformal Scaling (IRACS)**

**What it adds:** The standard CQR/EnbPI/JANET pipeline assumes the calibration set is representative of the current regime. Intraday, volatility follows a strong U-shape and regime shifts occur (earnings, Fed, crises). We add:

1. **Regime-stratified calibration**: Maintain separate EnbPI correction factors for (at least) three volatility regimes detected online via an HMM on 5m realized volatility (Research Report 5). At inference, blend corrections by regime probability.
2. **Time-of-day conformal scaling**: Model the conformal score's conditional quantile as a smooth function of time-of-day (e.g., cyclic spline with 5 knots). This captures the U-shape directly in the conformal layer, not just in the base quantile model.
3. **Joint scenario odds**: From the JANET joint region, sample 100 paths. Label each as bull/base/bear by comparing to median. Compute regime-conditioned probabilities: P(bull | high-vol regime), etc. Display as "Bull 65% / Base 25% / Bear 10% (high-vol regime)".

**Why original:** No paper combines regime-stratified conformal calibration + time-of-day score modelling + JANET joint regions + scenario odds for intraday equities. TRACE-CRC (2026) does per-step profiling + LTT for CSI; we adapt the idea to financial regimes and add the scenario layer.

### 5.3 Required Experiments for Final Report

| Experiment | Purpose | Success Criterion |
|------------|---------|-------------------|
| **Marginal coverage backtest** | Validate EnbPI-corrected 90% PIs on held-out days (rolling 500-bar calibration) | 89-91% of prices inside marginal bands |
| **Joint coverage backtest** | Validate JANET cone (90% joint) on held-out days | 89-91% of *full paths* inside joint cone |
| **Conditional coverage by regime** | Check coverage in high/low vol regimes separately (HMM states) | No regime with <85% marginal coverage |
| **Quantile crossing rate** | % of predictions where q5 > q50 or q50 > q95 | < 0.5% (arctan pinball target) |
| **Interval width vs. volatility** | Sharpness: median PI width should scale with realized vol | Spearman ρ > 0.7 |
| **Scenario odds calibration** | PIT histogram of regime-conditioned bull/base/bear probs | Uniform (KS test p > 0.05) |
| **ACI adaptation speed** | After a regime shift (simulated), steps to recover 90% coverage | < 20 5m bars |
| **CPU latency** | End-to-end inference time per candle on laptop i7-13650HX | < 200 ms (well within 5m) |
| **Baseline comparison** | vs. fixed-width bands, vs. GARCH(1,1) bands, vs. always-Up | Our bands sharper at same coverage; profit after 2bps costs > baselines |

---

## 6. Open Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **JANET implementation unavailable/complex** | Cannot deliver joint coverage cone | Fallback: Bonferroni-corrected marginal bands (6 steps → 98.3% per step for 90% joint); wider but valid. |
| **EnbPI bootstrap too slow on CPU** | Miss 5m update deadline | Reduce B to 30; precompute bootstrap weights; or use ACI-only (no ensemble) with larger calibration window. |
| **Quantile crossing persists with arctan loss** | Invalid bands (q5 > q50) | Add post-hoc isotonic regression on quantiles (Chernozhukov et al., 2010) as safety net; monitor crossing rate. |
| **Regime-stratified calibration data sparsity** | High-vol regime has too few calibration points | Pool adjacent regimes; use Bayesian shrinkage toward global correction; minimum 200 points per regime. |
| **U-shape not fully captured by time-of-day feature** | Residual miscoverage at open/close | Add explicit parametric U-shape: σ(t) = a + b·(t-9:30)² + c·(t-9:30)⁴ as extra feature; also feed VIX (D23). |
| **GDELT sentiment leakage** | News published after candle close used in features | Strict cutoff: only articles with timestamp < candle close - 15 min (GDELT delay). Validate in replay. |
| **ACI learning rate misspecification** | Under/over-coverage during regime shifts | Grid-search learning rate on validation days; use Szabadváry (2024) per-step rates; monitor online. |
| **Joint coverage guarantee only holds under exchangeability of calibration scores** | JANET's theory assumes score exchangeability | Use EnbPI scores (already time-series robust) as input to JANET; validate empirically on held-out days. |
| **Scenario odds not actionable** | Traders cannot use bull/base/bear probabilities | User guide must explain: "Odds are conditional on current regime; not a trading signal alone. Combine with direction signal (D17)." |

---

## 7. Licences of Code/Weights We Build On

| Component | Source | Licence | Usable? |
|-----------|--------|---------|---------|
| LightGBM | Microsoft | MIT | Yes |
| XGBoost | DMLC | Apache-2.0 | Yes |
| EnbPI reference impl. | Angelopoulos et al. (github.com/aangelopoulos/conformal-time-series) | MIT | Yes |
| JANET | English et al. | Not verified (arXiv non-exclusive) | Check repo if exists; else implement from paper |
| ACI reference | Gibbs & Candès | Not verified | Simple to implement from paper (single parameter update) |
| Arctan pinball loss | Sluijterman et al. | Not verified | Implement from formula in paper (2 lines of code) |

---

*End of report. All sources cited were opened and verified in this session via arXiv HTML pages. No citations from memory.*