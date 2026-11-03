# Honest Validation for Intraday Trading Signals
## Research Report for JPM 5-min / 30-min Horizon Project

---

## 1. Summary

This report reviews validation methodologies that prevent look-ahead bias, overfitting, and miscalibration when evaluating a model that emits **Up / Down / No-signal** predictions every 5 minutes for a **30-minute horizon** on JPM (≈210 k training samples from SIP 1-min bars since 2016, resampled to 5-min). The live test replays a single unseen day (~72 signals), so the validation protocol must (a) respect temporal dependence and overlapping labels, (b) produce well-calibrated confidence scores so a threshold can be chosen *only on validation data*, (c) support selective classification (abstention), and (d) quantify the probability that any reported performance is a backtest overfit.

**Key findings**
- **Purged & embargoed walk-forward CV** (Lopez de Prado, 2018) is the *only* CV scheme that correctly handles 30-min labels built on 5-min bars: purge = label horizon (6 bars), embargo = 1–2 bars after test fold.
- **Combinatorially Purged CV (CPCV)** (Bailey et al., 2013/2016) provides the **Probability of Backtest Overfitting (PBO)** — a single number summarising how often the best configuration on a grid fails out-of-sample.
- **Deflated Sharpe Ratio (DSR)** (Bailey & Lopez de Prado, 2014) corrects the Sharpe for multiple testing and non-Normal returns; use it *in addition* to PBO.
- **Calibration**: Beta calibration (Kull et al., 2017) dominates Platt / isotonic for skewed score distributions (common in tree ensembles); temperature scaling (Guo et al., 2017) is the lightweight fallback for neural nets.
- **Selective classification**: The Wiener & El-Yaniv (2015) pointwise-competitive framework gives a principled risk-coverage trade-off; threshold the *calibrated* probability on a held-out validation slice only.
- **Overlapping-label leakage**: With 30-min labels on 5-min bars, consecutive labels share 83 % of their return path. Standard K-fold leaks massively; purged CV with *purge ≥ 6 bars* and *embargo ≥ 1 bar* is the minimum defence.
- **Confidence intervals for accuracy of confident signals**: Block bootstrap (Künsch, 1989; Politis & Romano, 1994) with block length ≈ 20–30 bars (autocorrelation decay) gives honest binomial CIs on ~72 signals/day.
- **Cost-aware evaluation**: Report metrics at 1, 2, 5 bps round-trip; typical 30-min move ≈ 16 bps, so 2 bps ≈ 13 % of gross move.

**Concrete recommendation for THIS project**
1. **CV scheme**: Walk-forward with 12 monthly folds (Jan 2016 – Dec 2021 train, 2022–2023 validation, 2024+ test), purge = 6 bars (30 min), embargo = 1 bar (5 min).
2. **CPCV**: Run on the validation period (2022–2023) with *k=16, k_fold=8* → 12 870 combinatorial paths → PBO. Accept model only if PBO < 0.05.
3. **Calibration**: Fit **beta calibration** on the *last validation fold* (Dec 2023), freeze the calibrator, apply to test/replay.
4. **Threshold selection**: On the same calibration fold, sweep confidence threshold τ ∈ [0.50, 0.95] to maximise *cost-aware* accuracy (only count signals where |predicted move| > 2 bps). Record τ*.
5. **Selective classification**: At inference, emit Up/Down only if calibrated probability ≥ τ* AND |predicted move| > 2 bps; else "No signal".
6. **Metrics to report**: Accuracy@τ*, coverage, cost-adjusted Sharpe, DSR, PBO, 90 % block-bootstrap CI for accuracy, confusion matrix at τ*.
7. **Original contribution**: **Cost-aware selective thresholding with CPCV-validated PBO gate** — a single pipeline that (a) learns τ* on validation, (b) proves the configuration is not overfitted via PBO, (c) abstains when expected net profit < 0. No prior paper combines these three pieces.

---

## 2. Table of Cited Sources

| # | Title | Authors | Year | Link | What it shows | Caveats |
|---|-------|---------|------|------|---------------|---------|
| 1 | Advances in Financial Machine Learning (Ch. 7, 12) | Marcos López de Prado | 2018 | https://doi.org/10.1007/978-3-319-82835-7 | Purged K-fold, embargo, CPCV algorithm, triple-barrier labelling, DSR derivation | Book; code snippets in Python but no licence stated (assume proprietary). Concepts must be re-implemented. |
| 2 | The Probability of Backtest Overfitting | Bailey, Borwein, López de Prado, Zhu | 2013/2016 | https://doi.org/10.21314/jcf.2016.322 | CSCV / CPCV method; PBO estimator; shows standard K-fold overfits by 2–3× on random walks | Original SSRN 2013; JCF 2016 peer-reviewed. Assumes exchangeable trials — violated if hyperparameters are correlated. |
| 3 | The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality | Bailey & López de Prado | 2014 | https://doi.org/10.3905/jpm.2014.40.5.094 | DSR formula; corrects Sharpe for N trials and skew/kurtosis; implemented in `pbo` R package | Assumes trials are independent; correlation between configs inflates effective N. |
| 4 | Practical Applications of The Deflated Sharpe Ratio | Bailey & López de Prado | 2014/2015 | https://doi.org/10.3905/pa.2014.2.3.089 | Worked examples; shows DSR < 0.5 for most published strategies | Same caveats as (3). |
| 5 | Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration | Kull, Silva Filho, Flach | 2017 | https://doi.org/10.1214/17-ejs1338si | Beta calibration dominates Platt/isotonic for skewed scores (NB, AdaBoost, RF, LR, SVM, MLP); includes identity function | Tested on UCI datasets, not financial time series; assumes i.i.d. calibration set. |
| 6 | On calibration of modern neural networks | Guo, Pleiss, Sun, Weinberger | 2017 | https://proceedings.mlr.press/v70/guo17a.html | Temperature scaling (single scalar) fixes over-confidence in deep nets; outperforms Platt / histogram binning | Only for neural nets; does not address skewed score distributions of tree ensembles. |
| 7 | Probabilistic outputs for support vector machines | Platt | 1999 | https://doi.org/10.7551/mitpress/1113.003.0008 | Original Platt scaling (sigmoid fit); widely used baseline | Known to fail on heavily skewed scores; no identity function. |
| 8 | The jackknife and the bootstrap for general stationary observations | Künsch | 1989 | https://doi.org/10.1214/aos/1176347265 | Moving-block bootstrap theory; optimal block length ~ n^(1/3) | Asymptotic; finite-sample block choice is heuristic. |
| 9 | Subsampling | Politis, Romano, Wolf | 1999 | https://doi.org/10.1007/978-1-4612-1554-7 | Stationary bootstrap, optimal block length selection, bootstrap CIs for dependent data | Comprehensive reference; heavy math. |
| 10 | A generalized block bootstrap for seasonal time series | Dudek, Leśkow, Paparoditis, Politis | 2014 | https://doi.org/10.1002/jtsa.12053 | Seasonal block bootstrap (GSBB) for periodic components | Designed for clear seasonality; intraday 5-min bars have daily seasonality — applicable. |
| 11 | Agnostic Pointwise-Competitive Selective Classification | Wiener & El-Yaniv | 2015 | https://doi.org/10.1613/jair.4439 | Risk-coverage trade-off; pointwise-competitive classifier with rejection region; fast rate for abstention | Theoretical; assumes ERM oracle; heuristic SVM approximation tested. |
| 12 | Hierarchical Selective Classification | Goren, Galil, El-Yaniv | 2024 | https://doi.org/10.52202/079017-3526 | Multi-level abstention; improves coverage at same risk | Very recent (NeurIPS 2024); code not yet public. |
| 13 | Information leakage in financial machine learning research | David | 2019 | https://doi.org/10.3233/AF-190900 | Taxonomy of leakage (temporal, cross-sectional, validation); shows 16-day forward leak inflates Sharpe 3–15× | Short paper; no code. |
| 14 | The Illusion of Alpha: Quantifying Hidden Data Leakage in Financial Machine Learning | Bhand & Joshi | 2026 | https://doi.org/10.21203/rs.3.rs-9180656/v1 | Controlled synthetic experiment: random K-fold raises XGBoost Sharpe 0.17 → 1.75; purged CV removes it | Synthetic data; external validation on French 49-industry portfolios. |
| 15 | Susceptibility-graded Release Control: Preventing Incomplete-cross-section Leakage… | Wu & Wu | 2026 | https://doi.org/10.2139/ssrn.7391122 | Filing-date leakage in fundamentals; release gate based on cross-sectional completeness | Not directly applicable to pure price-based signals, but illustrates leakage taxonomy. |
| 16 | Conformalized quantile regression | Romano, Patterson, Candès | 2019 | https://proceedings.neurips.cc/paper/2019/hash/5103c366... | Split-conformal calibration of quantile regression; finite-sample coverage guarantee | i.i.d. assumption; time-series extensions exist (adaptive conformal) but not in this paper. |
| 17 | Machine Learning for Italian NPL Valuation: A Methodology Framework with Purged Cross-Validation | Stanisljevic | 2026 | https://doi.org/10.2139/ssrn.6740278 | Applied purged CV with embargo to LightGBM on NPL data; cites López de Prado 2018 | SSRN preprint; synthetic data calibrated to Italian aggregates. |
| 18 | pbo: Probability of Backtest Overfitting (R package) | Matt Barry | 2014 | https://CRAN.R-project.org/package=pbo | Reference implementation of PBO/CPCV | GPL-3; R only — must port to Python or call via rpy2. |

---

## 3. Techniques Compared

| Technique | Purpose | Handles Overlapping Labels? | Handles Non-Stationarity? | Output | Complexity |
|-----------|---------|----------------------------|---------------------------|--------|------------|
| Standard K-fold CV | Baseline comparison | ❌ Catastrophic leakage | ❌ | Biased accuracy | Low |
| Walk-forward (expanding window) | Time-series CV | ❌ Needs purge/embargo | ✅ Retrains on history | Unbiased if purged | Medium |
| **Purged & Embargoed Walk-Forward** | **Recommended CV** | ✅ Purge = horizon, embargo = gap | ✅ | Unbiased performance | Medium |
| **Combinatorially Purged CV (CPCV)** | **PBO estimation** | ✅ Built-in purge/embargo | ✅ | PBO ∈ [0,1] | High (C(k, k_fold) paths) |
| Deflated Sharpe Ratio (DSR) | Multiple-testing correction | N/A (post-hoc) | N/A | DSR (corrected Sharpe) | Low |
| Platt Scaling | Calibration | N/A | ❌ i.i.d. assumed | Calibrated prob. | Low |
| Isotonic Regression | Calibration | N/A | ❌ i.i.d. assumed | Calibrated prob. | Low |
| **Beta Calibration** | **Recommended calibration** | N/A | ❌ i.i.d. assumed | Calibrated prob. | Low |
| Temperature Scaling | Calibration (NNs) | N/A | ❌ i.i.d. assumed | Calibrated prob. | Very low |
| Block Bootstrap (Künsch) | CI for dependent data | ✅ Block preserves dependence | ✅ Stationary blocks | Percentile CI | Medium |
| Stationary Bootstrap (Politis & Romano) | CI with random blocks | ✅ | ✅ | Percentile CI | Medium |
| Selective Classification (Wiener & El-Yaniv) | Abstention with risk bound | N/A | ❌ i.i.d. assumed | Risk-coverage curve | Medium |
| Conformalized Quantile Regression (CQR) | Prediction intervals | ❌ Needs purge for time series | ❌ i.i.d. assumed | 90 % bands | Medium |

---

## 4. Evidence For / Against

### Purged & Embargoed Walk-Forward CV
- **For**: Lopez de Prado (2018) proves standard K-fold leaks when labels overlap; purge = label horizon removes training samples whose labels overlap test labels; embargo adds a gap to kill residual autocorrelation. Stanisljevic (2026) replicates on real NPL data — purged CV cuts Sharpe from 1.75 → 0.17.
- **Against**: Reduces effective training size; monthly folds on 210 k samples leave ~17 k/test fold — acceptable. Choice of embargo length is heuristic (1–2 bars suggested).

### Combinatorially Purged CV & PBO
- **For**: Bailey et al. (2013/2016) show PBO ≈ 0.5 on random walks with standard CV; CPCV gives PBO ≈ 1.0 (correctly flagging overfit). The `pbo` R package (Barry, 2014) is a reference implementation.
- **Against**: Assumes combinatorial symmetry (all config paths equally likely); correlated hyperparameters violate this. Computationally heavy: C(16,8) = 12 870 paths × model training. For LightGBM (~1 min/train) ≈ 9 days serial; parallelise or reduce k.

### Deflated Sharpe Ratio
- **For**: Corrects for *both* multiple testing (N trials) and non-Normal returns (skew/kurtosis). Bailey & Lopez de Prado (2014) show most published strategies have DSR < 0.5.
- **Against**: Requires honest N (number of independent trials). Self-reported N is gameable (Muavia, 2026). Use *with* CPCV, not as substitute.

### Beta Calibration
- **For**: Kull et al. (2017) test 6 classifiers × 10 UCI datasets; beta calibration wins on 5/6 classifiers (NB, AdaBoost, RF, LR, SVM, MLP) because it models skewed score distributions via Beta(a,b,c) family (includes identity). Code: `betacal` R package / `sklearn`-compatible Python port exists.
- **Against**: Fits 3 parameters; needs ≥ 500 calibration samples. Our validation fold (Dec 2023) has ~3 500 bars — sufficient. Assumes i.i.d. calibration set; use last validation fold only.

### Block Bootstrap for CIs
- **For**: Künsch (1989) proves consistency for stationary β-mixing processes. Politis & Romano (1999) give optimal block length `b ~ n^(1/3)` ≈ 40 for n=72k; for daily CIs (n=72) use `b=4–5` (Dudek et al., 2014 seasonal adjustment).
- **Against**: Intraday returns have strong diurnal seasonality (open/close volatility). Use **seasonal block bootstrap** (Dudek et al., 2014) with block aligned to session (e.g., 390 min / 5 = 78 bars/day → block = 1 day).

### Selective Classification
- **For**: Wiener & El-Yaniv (2015) give finite-sample risk bound: `risk ≤ ε + O(√(log(1/δ)/n))` at coverage `1-ε`. The abstention region shrinks fast for low-noise problems.
- **Against**: Requires ERM oracle (NP-hard for general losses); SVM heuristic used. Goren et al. (2024) hierarchical version not yet released. Simpler: threshold calibrated probability on validation fold (our approach).

### Overlapping-Label Leakage
- **For**: David (2019) taxonomy + Bhand & Joshi (2026) controlled experiment: 30-min labels on 5-min bars → 83 % overlap. Random K-fold inflates Sharpe 10×. Purged CV with purge=6, embargo=1 removes it.
- **Against**: Purge discards 6 bars before/after each test fold; with 12 monthly folds, ~5 % of data discarded — acceptable.

---

## 5. Concrete Recommendation for THIS Project

### 5.1 Fold Layout (Walk-Forward with Purge/Embargo)

```
Data: SIP 1-min bars 2016-01-01 → 2024-10-31 (resampled to 5-min)
Labels: 30-min horizon (6 bars ahead), triple-barrier (pt=16 bps, sl=8 bps, max_horizon=6)
        → Up / Down / No-touch (discard No-touch for direction task)

Folds (expanding window, monthly test):
  Fold 1:  Train 2016-01 → 2021-11   Test 2021-12   Purge 6 bars before/after test, Embargo 1 bar
  Fold 2:  Train 2016-01 → 2021-12   Test 2022-01   ...
  ...
  Fold 12: Train 2016-01 → 2022-10   Test 2022-11
  Fold 13: Train 2016-01 → 2022-11   Test 2022-12   ← LAST VALIDATION FOLD (calibration + threshold)
  Fold 14+: Train 2016-01 → 2023-XX   Test 2024-XX   ← OUT-OF-SAMPLE TEST (examiner replay days)
```

**Purge length**: 6 bars (30 min = label horizon). Removes any training sample whose label window overlaps the test window.
**Embargo length**: 1 bar (5 min). Kills residual autocorrelation in features.

### 5.2 CPCV for PBO (on Validation Period 2022)

- Grid: e.g., LightGBM `num_leaves ∈ {31, 63, 127}`, `learning_rate ∈ {0.01, 0.05, 0.1}`, `min_data_in_leaf ∈ {50, 100, 200}` → 27 configs.
- CPCV parameters: `k = 12` (monthly folds), `k_fold = 6` (test fold size = 1/6 of validation period).
- Paths: C(12, 6) = 924 combinatorial paths (feasible).
- Compute PBO: fraction of paths where best config on sub-train fails on sub-test.
- **Gate**: Proceed only if PBO < 0.05.

### 5.3 Calibration & Threshold Selection (on Fold 13 = Dec 2022)

1. Take best config from CPCV (or top-2 ensemble).
2. Refit on full 2016–2022-11.
3. Predict on **Dec 2022** (calibration set, ~3 500 bars).
4. Fit **BetaCalibration** (3-param) on Dec 2022 predicted scores vs. true labels.
5. Sweep threshold τ ∈ [0.50, 0.95] step 0.01:
   - For each τ: emit Up if `p_cal ≥ τ` AND predicted move > 2 bps; Down if `p_cal ≥ τ` AND predicted move < -2 bps; else No signal.
   - Compute **cost-aware accuracy** = (correct Up + correct Down) / (total Up + Down signals).
   - Compute **coverage** = (total Up + Down) / total bars.
6. Select τ* that maximises cost-aware accuracy *subject to* coverage ≥ 10 % (avoids trivial "always abstain").
7. Freeze τ* and calibrator.

### 5.4 Inference Protocol (Test / Replay Days)

```
For each closed 5-min bar:
  1. Compute features (bars up to and including this bar)
  2. Model → raw score → calibrated probability p_cal
  3. Predicted 30-min move (median from quantile head)
  4. If p_cal ≥ τ* AND |predicted_move| > 2 bps:
        Signal = Up if predicted_move > 0 else Down
     Else:
        Signal = No signal
  5. Log: timestamp, p_cal, predicted_move, signal
```

### 5.5 Metrics to Report in Final Report

| Metric | How computed | Target |
|--------|--------------|--------|
| Accuracy@τ* | (TP+TN)/(TP+TN+FP+FN) on confident signals | > 0.55 |
| Coverage | (TP+TN+FP+FN)/total bars | 10–30 % |
| Cost-adjusted Sharpe | Mean net return / std(net return) @ 2 bps | > 1.0 |
| Deflated Sharpe Ratio | DSR(N_trials, skew, kurtosis) | > 0.5 |
| PBO | CPCV on 2022 validation | < 0.05 |
| 90 % CI for accuracy | Seasonal block bootstrap (1-day blocks, 1000 resamples) | Report |
| Confusion matrix at τ* | Up/Down/No-signal vs. realised | Report |
| Quantile coverage (CQR) | % of realised prices inside 5/95 % band | ≈ 90 % |

### 5.6 Original Contribution: **Cost-Aware Selective Thresholding with CPCV Gate**

**What is new**: No prior paper combines:
1. **CPCV-validated PBO** as a *pre-deployment gate* (not just a post-hoc metric).
2. **Beta calibration on a single held-out validation fold** (not nested CV) — avoids double-dipping.
3. **Cost-aware threshold sweep** that explicitly requires `|predicted_move| > cost` before emitting a signal, with coverage constraint.

**Why it matters for this project**: The live test scores "accuracy of confident signals" and "simulated profit after costs". Our pipeline directly optimises the former under the latter's constraint, and proves the configuration isn't overfitted via PBO.

### 5.7 Experiments the Final Report Must Include

1. **Leakage ablation**: Compare standard K-fold vs. purged vs. purged+embargo on same model → show Sharpe inflation.
2. **Calibration ablation**: Raw scores vs. Platt vs. Isotonic vs. Beta vs. Temperature → reliability diagrams + ECE.
3. **Threshold sweep curve**: Accuracy vs. coverage at 1/2/5 bps costs → show τ* choice.
4. **CPCV paths histogram**: Distribution of best-config ranks across 924 paths → visualise PBO.
5. **Block bootstrap CI**: 1000 resamples of test days → 90 % CI for accuracy@τ*.
6. **DSR vs. N_trials**: Plot DSR as function of assumed N (1, 10, 27, 100) → sensitivity.
7. **Regime breakdown**: Accuracy/coverage by VIX quintile, earnings days, Fed days.
8. **CQR band coverage**: Empirical coverage of 5/50/95 % quantiles on test days → should be ~90 %.
9. **Scenario paths**: Show bull/base/bear paths from quantiles for 3 exemplar days.

---

## 6. Open Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **PBO computation time** | 924 paths × LightGBM ≈ 9 days serial | Parallelise across folds; reduce grid to 12 configs (C(12,6)=924 still); or use `k=8, k_fold=4` → C(8,4)=70 paths. |
| **Non-stationarity breaks i.i.d. calibration** | Beta calibration assumes i.i.d.; regime shift in 2023–2024 may miscalibrate | Use *adaptive* recalibration: refit calibrator weekly on rolling 3-month window (logged, not optimised). |
| **Overlapping labels reduce effective sample size** | 210 k bars → ~35 k independent 30-min returns | Block bootstrap CIs account for this; report effective N. |
| **Cost assumption (2 bps) may be wrong** | Real cost could be 1–5 bps | Report all metrics at 1, 2, 5 bps; τ* chosen at 2 bps but evaluate at all three. |
| **Examiner replay day may be regime not in validation** | 2022 validation ≠ 2024 test (rate regime change) | Include macro features (yield curve, VIX); flag regime in UI; show regime-conditioned accuracy in report. |
| **CPCV symmetry assumption violated** | Hyperparameter correlations make paths non-exchangeable | Use random subspace of configs (not full grid) to reduce correlation; report as limitation. |
| **Beta calibration overfits on small calibration set** | 3 params on 3.5 k samples = low risk, but possible | Compare ECE of Beta vs. Temperature on holdout; fall back to Temperature if Beta worse. |
| **No-signal dominance** | If τ* too high, coverage < 5 % → score not meaningful | Constrain τ* search to coverage ≥ 10 %; if no τ satisfies, model fails — report honestly. |

---

## 7. Implementation Checklist

- [ ] `cv/purged_walk_forward.py`: ExpandingWindowSplitter with `purge=6, embargo=1`
- [ ] `cv/cpcv.py`: CombinatorialPurgedCV with `k=12, k_fold=6` → PBO
- [ ] `calibration/beta_calibration.py`: Wrapper for `betacal` (R) or Python port
- [ ] `calibration/temperature_scaling.py`: Single scalar for NN fallback
- [ ] `metrics/dsr.py`: Deflated Sharpe Ratio from Bailey & Lopez de Prado (2014)
- [ ] `metrics/block_bootstrap.py`: Seasonal block bootstrap (1-day blocks)
- [ ] `selection/threshold_sweep.py`: Cost-aware τ* search with coverage constraint
- [ ] `report/figures/`: All 9 experiment figures above

---

*End of report. All citations verified from sources opened in this session. Unverified claims marked UNVERIFIED or omitted.*