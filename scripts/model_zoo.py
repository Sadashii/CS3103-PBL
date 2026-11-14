"""Model zoo + stacked result layer for the 30-minute direction target (5m features, data/features_5m.parquet).

Base models (same rows, same walk-forward folds, same leak-checked features):
  logreg, LightGBM, XGBoost, CatBoost, ExtraTrees classifier, ExtraTrees regressor (sign of predicted return),
  neural nets on GPU: MLP, GRU, 1D-CNN, Transformer (last 12 five-minute bars as a sequence).
Result layer (stacking): every model's out-of-fold probability -> (a) plain average, (b) logistic meta-learner on the logits,
  fitted ONLY on validation years 2022-24 and scored on the untouched test years 2025-26.
Leak control: walk-forward (train = days before the test year), labels resolve inside the session, and a shuffled-label placebo
  is run for every model on the 2025 fold (must not beat the base rate).
Usage: python scripts/model_zoo.py [--quick] [--only lgb,xgb,...]   (run with the GPU env: .venv-nlp\\Scripts\\python.exe)"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
ET = "America/New_York"
FOLDS = [2022, 2023, 2024, 2025, 2026]
VAL, TEST = [2022, 2023, 2024], [2025, 2026]
CUTOFF = pd.Timestamp("2026-10-27").date()
L = 12                                   # sequence length for the sequence networks
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", default="")
args = ap.parse_args()
Q = args.quick


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
df = pd.read_parquet(ROOT / "data" / "features_5m.parquet").replace([np.inf, -np.inf], np.nan)
cols = [c for c in df.columns if not c.startswith("y") and c != "close"]
et = df.index.tz_convert(ET)
day, year = np.asarray(et.date), np.asarray(et.year)
y6 = df["y6"].values
elig = (day <= CUTOFF) & df[cols].notna().all(axis=1).values & ~np.isnan(y6) & (y6 != 0)
X = df[cols].values.astype(np.float32)
ydir = (y6 > 0).astype(np.float32)
pos_all = np.arange(len(df))
log(f"{len(df):,} bars, {elig.sum():,} usable rows, {len(cols)} features")


# ------------------------------------------------------------------ models
def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def m_logreg(tr, te, yv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(X[tr], yv[tr])
    return m.predict_proba(X[te])[:, 1]


def m_lgb(tr, te, yv):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=100 if Q else 400, learning_rate=0.03 if not Q else 0.08, num_leaves=15, min_child_samples=300,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1, n_jobs=-1)
    return m.fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]


def m_xgb(tr, te, yv):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=100 if Q else 400, learning_rate=0.08 if Q else 0.03, max_depth=4, min_child_weight=50,
                          subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0, tree_method="hist", n_jobs=-1, verbosity=0)
    return m.fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]


def m_cat(tr, te, yv):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=150 if Q else 500, learning_rate=0.1 if Q else 0.05, depth=5, l2_leaf_reg=10, verbose=0,
                           thread_count=-1, random_seed=0)
    return m.fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]


def m_etc(tr, te, yv):
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=100 if Q else 300, min_samples_leaf=300, max_features=0.5, n_jobs=-1, random_state=0)
    return m.fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]


def m_etr(tr, te, yv):
    """ExtraTrees REGRESSOR used for classification: regress the (signed, clipped) return, then calibrate the score to a probability."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression
    n = len(tr)
    cut = int(n * 0.85)
    fit_idx, cal_idx = tr[:cut], tr[cut:]                          # tr is time-ordered: last 15% calibrates the score
    m = ExtraTreesRegressor(n_estimators=100 if Q else 300, min_samples_leaf=300, max_features=0.5, n_jobs=-1, random_state=0)
    m.fit(X[fit_idx], np.where(yv[fit_idx] > 0, 1.0, -1.0))
    s_cal = m.predict(X[cal_idx]).reshape(-1, 1)
    pl = LogisticRegression(C=10.0).fit(s_cal, yv[cal_idx])
    return pl.predict_proba(m.predict(X[te]).reshape(-1, 1))[:, 1]


def _nn(arch):
    import torch
    import torch.nn as nn
    d = X.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "mlp":
                self.f = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
            elif arch == "gru":
                self.r = nn.GRU(d, 64, batch_first=True)
                self.o = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))
            elif arch == "cnn":
                self.c = nn.Sequential(nn.Conv1d(d, 64, 3, padding=1), nn.GELU(), nn.Dropout(0.2), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
                self.o = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))
            else:
                self.i = nn.Linear(d, 64)
                self.p = nn.Parameter(torch.zeros(1, L, 64))
                self.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, dropout=0.2, batch_first=True), 2)
                self.o = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):                       # x: (B, L, d)
            if arch == "mlp":
                return self.f(x[:, -1]).squeeze(-1)
            if arch == "gru":
                return self.o(self.r(x)[1][-1]).squeeze(-1)
            if arch == "cnn":
                h = self.c(x.transpose(1, 2))
                return self.o(torch.cat([h.mean(-1), h[:, :, -1]], 1)).squeeze(-1)
            h = self.t(self.i(x) + self.p)
            return self.o(h.mean(1)).squeeze(-1)

    def run(tr, te, yv):
        torch.manual_seed(0)
        dev = "cuda"
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = torch.tensor(np.nan_to_num(np.clip((X - mu) / sd, -5, 5)), dtype=torch.float32, device=dev)
        win = torch.arange(-L + 1, 1, device=dev)

        def batch(idx):
            p = torch.tensor(idx, device=dev)[:, None] + win[None, :]
            return Z[p.clamp(min=0)]

        n = len(tr)
        cut = int(n * 0.85)
        fit_idx, hold_idx = tr[:cut], tr[cut:]               # time-ordered: last 15% of training rows is the early-stopping set
        y = torch.tensor(yv, dtype=torch.float32, device=dev)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad = 1e9, None, 0
        for ep in range(4 if Q else 12):
            net.train()
            perm = np.random.default_rng(ep).permutation(len(fit_idx))
            for i in range(0, len(perm), 1024):
                b = fit_idx[perm[i:i + 1024]]
                opt.zero_grad()
                lossf(net(batch(b)), y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                hl = float(np.mean([lossf(net(batch(hold_idx[i:i + 4096])), y[hold_idx[i:i + 4096]]).item()
                                    for i in range(0, len(hold_idx), 4096)]))
            if hl < best - 1e-5:
                best, bad = hl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        net.load_state_dict(best_state)
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(te), 4096):
                out.append(torch.sigmoid(net(batch(te[i:i + 4096]))).cpu().numpy())
        return np.concatenate(out)
    return run


MODELS = {"logreg": m_logreg, "lgb": m_lgb, "xgb": m_xgb, "cat": m_cat, "extra_trees_clf": m_etc, "extra_trees_reg": m_etr,
          "mlp": _nn("mlp"), "gru": _nn("gru"), "cnn1d": _nn("cnn"), "transformer": _nn("tf")}
names = [n for n in MODELS if not args.only or n in args.only.split(",")]
oof = {n: np.full(len(df), np.nan) for n in names}
placebo = {}
folds = [2025, 2026] if Q else FOLDS
for Y in folds:
    start = pd.Timestamp(f"{Y}-01-01").date()
    tr = np.where(elig & (day < start))[0]
    if Q:
        tr = tr[::4]
    te = np.where(elig & (year == Y))[0]
    for n in names:
        t0 = time.time()
        try:
            oof[n][te] = MODELS[n](tr, te, ydir)
        except Exception as e:
            log(f"  {n} FAILED on {Y}: {type(e).__name__}: {str(e)[:120]}")
            continue
        acc = ((oof[n][te] > .5) == ydir[te]).mean()
        extra = ""
        if Y == 2025:                                         # shuffled-label placebo: must not beat the base rate
            ysh = ydir.copy()
            ysh[tr] = np.random.default_rng(1).permutation(ydir[tr])
            pp = MODELS[n](tr, te, ysh)
            placebo[n] = float(((pp > .5) == ydir[te]).mean())
            extra = f"  placebo acc {placebo[n]:.4f}"
        log(f"fold {Y} {n:16s} acc {acc:.4f} ({time.time() - t0:.0f}s){extra}")

# ------------------------------------------------------------------ scoring and the stacked result layer
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

val = elig & np.isin(year, VAL if not Q else [2025])
test = elig & np.isin(year, TEST)
have = [n for n in names if np.isfinite(oof[n][test]).all() and np.isfinite(oof[n][val]).all()] if not Q else \
       [n for n in names if np.isfinite(oof[n][test]).any()]
if Q:
    val = test
P = np.column_stack([oof[n] for n in have])
Lg = logit(P)
use = val | test
avg = np.full(len(df), np.nan)
avg[use] = sig(Lg[use].mean(1))
meta = LogisticRegression(C=0.1, max_iter=500).fit(Lg[val], ydir[val])
stack = np.full(len(df), np.nan)
stack[use] = meta.predict_proba(Lg[use])[:, 1]
base_acc = {k: float(max(ydir[m].mean(), 1 - ydir[m].mean())) for k, m in (("val", val), ("test", test))}
rows = []
for n, p in list(zip(have, P.T)) + [("AVERAGE of all", avg), ("STACKED (meta-learner)", stack)]:
    r = {"model": n}
    for k, m in (("val", val), ("test", test)):
        pm = p[m]
        conf = np.abs(pm - 0.5)
        top = conf >= np.quantile(conf, 0.9)
        r[f"{k}_acc"] = float(((pm > .5) == ydir[m]).mean())
        r[f"{k}_auc"] = float(roc_auc_score(ydir[m], pm))
        r[f"{k}_logloss"] = float(log_loss(ydir[m], np.clip(pm, 1e-4, 1 - 1e-4)))
        r[f"{k}_top10_acc"] = float(((pm[top] > .5) == ydir[m][top]).mean())
    r["placebo_acc"] = placebo.get(n)
    rows.append(r)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\nbase rate (majority class): val %.4f  test %.4f   | rows: val %d test %d" % (base_acc["val"], base_acc["test"], val.sum(), test.sum()))
print(res.round(4).to_string(index=False))
ci = 1.96 * 0.5 / np.sqrt(test.sum())
print(f"\n95% interval on a 50% accuracy with {test.sum()} test rows: +/-{ci:.4f}  (overlapping 30-min labels, so the effective interval is wider)")
sel = res[~res.model.str.contains("AVERAGE|STACKED")].sort_values("val_logloss").iloc[0]
print(f"model picked on VALIDATION log-loss: {sel.model}  -> its test accuracy {sel.test_acc:.4f}")
out = {"base_rate": base_acc, "rows": rows, "stack_weights": dict(zip(have, meta.coef_[0].round(4).tolist())), "quick": Q}
json.dump(out, open(ROOT / "models" / ("zoo_metrics_quick.json" if Q else "zoo_metrics.json"), "w"), indent=1)
(ROOT / "docs").mkdir(exist_ok=True)
if not Q:
    (ROOT / "docs" / "model_zoo_results.md").write_text("# Model zoo results (30-min direction, 5m features)\n\n" + res.round(4).to_string(index=False) +
        f"\n\nBase rate (majority class): val {base_acc['val']:.4f}, test {base_acc['test']:.4f}. Rows: val {val.sum()}, test {test.sum()}.\n"
        f"Stack weights (logits): {out['stack_weights']}\n", encoding="utf-8")
log("done")
