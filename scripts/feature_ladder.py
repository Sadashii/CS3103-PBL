"""Feature ladder: add one feature group at a time and check accuracy at every step (30-min direction, 5m bars).

Model is deliberately simple and fixed (small LightGBM, strong regularisation). For each step:
  fit on years < 2022  -> score VALIDATION 2022-24   (this decides whether a group is kept)
  fit on years < 2025  -> score TEST 2025-26         (reported only, never used to decide)
A group is kept only if validation log-loss improves by > 0.0001. Accuracy is compared with the always-up baseline.
Usage: uv run python scripts/feature_ladder.py"""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ET = "America/New_York"

GROUPS = [
    ("1 recent returns", ["ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24"]),
    ("2 volatility", ["vol_12", "vol_78", "vol_390", "vol_ratio", "range_1", "close_pos", "ret_skew_78", "vol_accel"]),
    ("3 classic indicators", ["rsi_14", "macd_hist", "bb_pctb", "bb_width"]),
    ("4 volume", ["rel_vol", "avg_trade", "avg_trade_rel", "trades_rel"]),
    ("5 time of day", ["slot", "bars_left", "dow"]),
    ("6 day context", ["ret_day", "gap", "prev_day_ret", "prev_day_vol", "vol_vs_prev_day", "open30_ret", "mom_390", "day_range_pos",
                       "dist_day_high", "dist_day_low", "dist_prev_high", "dist_prev_low", "vwap_dist", "vwap5_dist"]),
    ("7 order-flow delta", ["imb_tick_1", "imb_tick_3", "imb_tick_6", "imb_tick_12", "cumimb_tick", "imb_clv_1", "imb_clv_3", "imb_clv_6",
                            "imb_clv_12", "cumimb_clv", "imb_bvc_1", "imb_bvc_3", "imb_bvc_6", "imb_bvc_12", "cumimb_bvc", "delta_div_6"]),
    ("8 market (SPY / sector)", ["spy_ret_1", "spy_ret_3", "spy_ret_6", "xlf_ret_3", "xlf_ret_6", "rel_spy_6", "corr_spy_36", "spy_vol_12",
                                 "spy_imb_clv_6", "spy_imb_bvc_6"]),
    ("9 peer banks", ["peer_ret_1", "peer_ret_3", "peer_ret_6", "rel_peer_6", "peer_imb_clv_6"]),
    ("10 JPM news", ["jpm_news_30m", "jpm_sent_30m", "jpm_news_3h", "jpm_sent_3h", "jpm_news_1d", "jpm_sent_1d", "jpm_news_rel",
                     "jpm_mins_since_news", "jpm_kw_upgrade_1d", "jpm_kw_downgrade_1d", "jpm_kw_pt_1d", "jpm_kw_legal_1d", "jpm_kw_earn_1d",
                     "jpm_kw_div_1d"]),
    ("11 market news", ["mkt_news_30m", "mkt_sent_30m", "mkt_news_3h", "mkt_sent_3h", "mkt_news_1d", "mkt_sent_1d", "mkt_news_rel",
                        "mkt_mins_since_news"]),
    ("12 events and filings", ["mins_since_earn", "earn_today", "earn_within_3d", "mins_since_fomc", "fomc_today", "days_to_fomc", "peer_earn_3d",
                               "jpm_8k_1d", "jpm_insider_5d", "jpm_13dg_5d", "mins_since_periodic"]),
]


def model():
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.04, num_leaves=7, min_child_samples=500, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=20.0, verbose=-1, n_jobs=-1, random_state=0)


def score(X, y, tr, te):
    p = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(((p > .5) == y[te]).mean()), float(log_loss(y[te], np.clip(p, 1e-4, 1 - 1e-4))), float(roc_auc_score(y[te], p))


def main():
    df = pd.read_parquet(ROOT / "data" / "features_5m.parquet").replace([np.inf, -np.inf], np.nan)
    et = df.index.tz_convert(ET)
    day, year = np.asarray(et.date), np.asarray(et.year)
    y6 = df["y6"].values
    allc = [c for _, g in GROUPS for c in g]
    elig = (day <= pd.Timestamp("2026-10-27").date()) & ~np.isnan(y6) & (y6 != 0) & df[allc].notna().all(axis=1).values
    y = (y6 > 0).astype(int)
    A = lambda m: np.where(m)[0]
    tr_v, va = A(elig & (year < 2022)), A(elig & np.isin(year, [2022, 2023, 2024]))
    tr_t, te = A(elig & (year < 2025)), A(elig & (year >= 2025))
    base_v, base_t = max(y[va].mean(), 1 - y[va].mean()), max(y[te].mean(), 1 - y[te].mean())
    ll_v0 = log_loss(y[va], np.full(len(va), y[tr_v].mean()))
    print(f"rows: train<2022 {len(tr_v):,}  validation {len(va):,}  train<2025 {len(tr_t):,}  test {len(te):,}")
    print(f"baseline (always-up): validation {base_v:.4f}  test {base_t:.4f}; validation log-loss of a constant: {ll_v0:.5f}\n")
    print(f"{'step / group added':28s} {'#feat':>5s} | {'VAL acc':>8s} {'logloss':>8s} {'AUC':>6s} | {'TEST acc':>8s} {'AUC':>6s} | kept?")
    kept, cols, best_ll = [], [], ll_v0
    for name, g in GROUPS:
        trial = cols + g
        X = df[trial].values.astype(np.float32)
        av, lv, uv = score(X, y, tr_v, va)
        at, lt, ut = score(X, y, tr_t, te)
        keep = lv < best_ll - 1e-4
        print(f"{name:28s} {len(trial):5d} | {av:8.4f} {lv:8.5f} {uv:6.3f} | {at:8.4f} {ut:6.3f} | {'KEEP' if keep else 'drop'}", flush=True)
        if keep:
            cols, best_ll = trial, lv
            kept.append(name)
    if not cols:
        print(f"\nNo group improved validation log-loss over a constant, so the final model is always-up (test {base_t:.4f}).")
        return
    X = df[cols].values.astype(np.float32)
    av, lv, uv = score(X, y, tr_v, va)
    at, lt, ut = score(X, y, tr_t, te)
    print(f"\nFINAL kept groups: {kept}")
    print(f"validation acc {av:.4f} (baseline {base_v:.4f}) | TEST acc {at:.4f} (baseline {base_t:.4f}), AUC {ut:.4f}")
    print(f"test rows {len(te):,}: 95% interval on accuracy about +/-{1.96 * np.sqrt(0.25 / len(te)):.4f}, wider because 30-min labels overlap")


if __name__ == "__main__":
    main()
