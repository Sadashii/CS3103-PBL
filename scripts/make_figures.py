"""Figures for the poster and report, drawn from the project's own data and metric files.

Writes docs/figures/<name>.svg (text kept as text, for Figma) and <name>.png (300 dpi), plus docs/figures/stats.json
with the dataset numbers quoted in the report.

Usage: uv run --with matplotlib python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jpmind import bars as B, config, store

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# theme (matches the poster / report board)
INK, MUTED, GRID = "#24211d", "#6b665e", "#e3ddd2"
PAPER = "#f8f3ea"
MODEL, SECOND, BASE = "#0f8fb8", "#b07a00", "#b3a998"
BAND = "#2cc3e9"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arimo", "DejaVu Sans"], "font.size": 9, "svg.fonttype": "none",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    svg = (OUT / f"{name}.svg").read_text()
    (OUT / f"{name}.svg").write_text(svg.replace("Liberation Sans", "Arimo"))   # same metrics; Arimo exists in Figma
    plt.close(fig)
    print("wrote", name)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


m3 = json.loads((config.MODEL_DIR / "cls3_metrics.json").read_text())
tk = json.loads((config.MODEL_DIR / "tick_config.json").read_text())
hold = pd.read_csv(ROOT / "docs" / "holdout_results.csv")
dep = m3["deployed"]
NAMES = {"always_neutral": "Always Flat", "always_up": "Always Up", "rsi_rule": "RSI rule", "persistence": "Repeat last move"}


# ---------------------------------------------------------------- 1. accuracy vs baselines (test + unseen days)
def fig_accuracy():
    test = {"Model": m3["models"][dep]["test"]["acc"], **{NAMES[k]: m3["baselines"][k]["test"]["acc"] for k in NAMES}}
    w = hold.candles
    unseen = {"Model": np.average(hold.model, weights=w), **{NAMES[k]: np.average(hold[k], weights=w) for k in NAMES}}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (title, d) in zip(axes, [(f"Test period 2025-26 ({m3['models'][dep]['test']['n']:,} candles)", test),
                                     (f"Unseen days after cutoff ({int(w.sum())} candles)", unseen)]):
        names = list(d)[::-1]
        vals = [d[n] * 100 for n in names]
        ax.barh(names, vals, color=[MODEL if n == "Model" else BASE for n in names], height=0.62)
        ax.axvline(100 / 3, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(100 / 3 + 0.5, -0.62, "guessing 33.3%", color=MUTED, fontsize=7, va="center")
        for i, v in enumerate(vals):
            ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK, fontweight="bold" if names[i] == "Model" else "normal")
        ax.set_xlim(0, 50)
        ax.set_ylim(-0.95, len(names) - 0.5)
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_xlabel("accuracy, Up / Flat / Down, next 10 min (%)")
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_accuracy")


# ---------------------------------------------------------------- 2. single-day accuracy distribution
def fig_daily():
    pdf = pd.read_csv(config.MODEL_DIR / "cls3_per_day.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.arange(0.15, 0.75, 0.025)
    ax.hist(pdf["always_neutral"], bins=bins, color=BASE, alpha=0.9, label="Always Flat")
    ax.hist(pdf["model"], bins=bins, color=MODEL, alpha=0.75, label="Model")
    for col, c, lab in (("model", MODEL, "Model"), ("always_neutral", "#857b6b", "Always Flat")):
        mu = pdf[col].mean()
        ax.axvline(mu, color=c, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("accuracy on one trading day")
    ax.set_ylabel("number of days")
    ax.set_title(f"One-day accuracy, {len(pdf)} test days")
    ax.legend(fontsize=8, loc="upper right")
    hgrid(ax)
    save(fig, "fig_daily")
    return pdf


# ---------------------------------------------------------------- 3. predicted vs actual close, unseen days
def fig_close():
    h = hold.dropna(subset=["am_median"]).reset_index(drop=True)
    x = np.arange(len(h))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.vlines(x, h.am_lo90, h.am_hi90, color=BAND, linewidth=7, alpha=0.35, label="10:00 forecast, 90% range")
    ax.scatter(x, h.am_median, marker="_", s=160, color=MODEL, linewidths=2.2, label="10:00 expected close", zorder=3)
    ax.scatter(x, h.close, marker="o", s=26, color=INK, zorder=4, label="actual close")
    for i, r in h.iterrows():
        ok = r.am_lo90 <= r.close <= r.am_hi90
        ax.text(i, max(r.am_hi90, r.close) + 1.0, "✓" if ok else "✗", ha="center", fontsize=8.5, color=MODEL if ok else "#b3261e")
    ax.set_xticks(x, [d[5:] for d in h.day], fontsize=7.5)
    ax.set_ylabel("JPM price ($)")
    ax.set_title(f"Close forecast made at 10:00 vs actual close: inside the 90% range on {int(h.am_in90.sum())} of {len(h)} unseen days")
    ax.legend(fontsize=7.5, ncol=3, loc="lower left", bbox_to_anchor=(0, -0.3))
    hgrid(ax)
    save(fig, "fig_close")


# ---------------------------------------------------------------- 4. what the model looks at (gain by feature family)
FAMILIES = [
    ("Volatility & range", ("vol_", "range_", "bb_width", "ret_skew", "prev_day_vol", "vol_vs_prev", "vol_ratio")),
    ("Market & peer banks", ("spy_", "xlf_", "peer_", "rel_", "corr_")),
    ("Time of day / calendar", ("slot", "bars_left", "dow")),
    ("Volume & order flow", ("imb_", "cumimb", "delta_div", "avg_trade", "trades_rel", "rel_vol", "vol_accel")),
    ("Price momentum & indicators", ("ret_", "rsi", "macd", "bb_pctb", "vwap", "close_pos", "gap", "open30", "mom_", "prev_day_ret")),
    ("Position in day's range", ("dist_", "day_range")),
    ("News, earnings, Fed, filings", ("jpm_", "mkt_", "earn", "fomc", "days_to", "mins_since", "peer_earn")),
]


def fig_importance():
    b = lgb.Booster(model_file=str(config.MODEL_DIR / "cls3_lgb.txt"))
    g = pd.Series(b.feature_importance("gain"), index=b.feature_name())
    g = g / g.sum()
    fam = {}
    for f, v in g.items():
        name = next((n for n, pre in FAMILIES if any(f.startswith(p) or f == p for p in pre)), "Other")
        fam[name] = fam.get(name, 0) + v
    s = pd.Series(fam).sort_values()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.barh(s.index, s.values * 100, color=[MODEL if v == s.max() else BASE for v in s.values], height=0.62)
    for i, v in enumerate(s.values):
        ax.text(v * 100 + 0.8, i, f"{v * 100:.0f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, s.max() * 100 + 12)
    ax.set_xlabel("share of LightGBM split gain (%)")
    ax.set_title(f"What the model uses ({len(g)} features)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_importance")
    return {k: round(float(v), 4) for k, v in s.items()}, g.sort_values(ascending=False).head(10).round(4).to_dict()


# ---------------------------------------------------------------- 5. model zoo (30-min direction, earlier experiment)
def fig_zoo():
    rows = []
    for line in (ROOT / "docs" / "model_zoo_results.md").read_text().splitlines():
        p = line.split()
        if len(p) >= 10 and p[0] in ("logreg", "lgb", "xgb", "cat", "extra_trees_clf", "extra_trees_reg", "mlp", "gru", "cnn1d", "transformer"):
            rows.append((p[0], float(p[5])))
        if line.startswith("STACKED"):
            rows.append(("stacked", float(line.split()[6])))
    nice = {"logreg": "Logistic reg.", "lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "extra_trees_clf": "Extra Trees (clf)",
            "extra_trees_reg": "Extra Trees (reg)", "mlp": "MLP", "gru": "GRU", "cnn1d": "1-D CNN", "transformer": "Transformer", "stacked": "Stacked ensemble"}
    d = pd.Series({nice[k]: v for k, v in rows}).sort_values()
    base = 0.5093
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.barh(d.index, d.values * 100, color=BASE, height=0.62)
    ax.axvline(base * 100, color=SECOND, linewidth=1.4)
    ax.text(base * 100 + 0.05, len(d) - 0.4, f"majority class {base * 100:.1f}%", color=SECOND, fontsize=7.5)
    for i, v in enumerate(d.values):
        ax.text(v * 100 + 0.05, i, f"{v * 100:.1f}", va="center", fontsize=7.5)
    ax.set_xlim(49, 52.2)
    ax.set_ylim(-0.6, len(d) + 0.3)
    ax.set_xlabel("test accuracy, Up vs Down in 30 min (%)")
    ax.set_title("Direction only: no model family beats the base rate")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_zoo")


# ---------------------------------------------------------------- 6 + 7. EDA on training years only (2016-2024)
def fig_eda():
    syms = [config.SYMBOL, *config.CONTEXT_SYMBOLS]
    rets, raw = {}, {}
    for s in syms:
        d = store.load(s)
        d = d[d.index < pd.Timestamp("2025-01-01", tz="UTC")]
        b5 = B.resample(d, "5m", "rth")
        raw[s] = (len(d), d.index.min(), d.index.max())
        r = np.log(b5.close).diff() * 1e4
        day = b5.index.tz_convert(B.ET).date
        r[pd.Series(day, index=b5.index).ne(pd.Series(day, index=b5.index).shift())] = np.nan   # no overnight returns
        rets[s] = r
    R = pd.DataFrame(rets).dropna()
    # intraday volatility profile
    j = rets[config.SYMBOL].dropna()
    t = j.index.tz_convert(B.ET)
    prof = j.abs().groupby(t.hour * 60 + t.minute).mean()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(prof.index / 60, prof.values, width=4.2 / 60, color=MODEL, align="edge")
    ax.set_xticks([9.5, 11, 12.5, 14, 15.5], ["9:30", "11:00", "12:30", "14:00", "15:30"])
    ax.set_xlabel("time of day (ET), 5-minute candle")
    ax.set_ylabel("mean |return| (bps)")
    ax.set_title("JPM moves most at the open and close")
    hgrid(ax)
    save(fig, "fig_eda_intraday")
    # correlation
    C = R.corr()
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    im = ax.imshow(C.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", [PAPER, "#9fdcef", MODEL, "#0b5f78"]), vmin=0.5, vmax=1)
    ax.set_xticks(range(len(C)), C.columns, rotation=0, fontsize=7.5)
    ax.set_yticks(range(len(C)), C.index, fontsize=7.5)
    for i in range(len(C)):
        for k in range(len(C)):
            ax.text(k, i, f"{C.values[i, k]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if C.values[i, k] > 0.78 else INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Correlation of 5-min returns, 2016-24")
    save(fig, "fig_eda_corr")
    return {s: {"bars_1m": int(n), "first": str(a.date()), "last": str(b.date())} for s, (n, a, b) in raw.items()}, \
        C.loc[config.SYMBOL].round(3).to_dict(), int(len(R))


def main():
    fig_accuracy()
    pdf = fig_daily()
    fig_close()
    fam, top = fig_importance()
    fig_zoo()
    raw, corr, n5 = fig_eda()
    t = m3["models"][dep]["test"]
    stats = {"raw_1m_until_2024": raw, "jpm_corr_5m": corr, "rows_5m_aligned_2016_24": n5,
             "importance_family": fam, "importance_top10": top, "tau_bps": m3["tau_bps"], "test": t,
             "test_days": len(pdf), "model_mean_day": float(pdf.model.mean()),
             "class_share_test": m3["class_share_test"], "tick_next_minute": tk["next_minute"]["test"], "to_close": tk["to_close"]}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: stats[k] for k in ("jpm_corr_5m", "rows_5m_aligned_2016_24", "importance_family", "class_share_test")}, indent=1))
    print({s: v for s, v in raw.items()})


if __name__ == "__main__":
    main()
