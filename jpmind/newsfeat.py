"""News and filing/event features on the 5m grid. A feature at bar t uses only items PUBLISHED at or before the
bar's close (news `created_at`, SEC acceptance time, FOMC statement time). Nothing scheduled in the future is used
except the FOMC calendar, which the Fed publishes a year ahead."""
import re
import numpy as np
import pandas as pd

from . import config

RAW = config.DATA_DIR / "raw"
KW = {"upgrade": r"\bupgrade", "downgrade": r"\bdowngrade", "pt": r"price target|\bpt\b",
      "legal": r"lawsuit|\bsue|settle|fine[sd]?\b|probe|investigat|charges?\b|penalt|subpoena|regulator",
      "earn": r"earnings|\bq[1-4]\b|\beps\b|revenue|guidance", "div": r"dividend|buyback|repurchase"}
MIN = pd.Timedelta(minutes=1)


def _scored(sym):
    out = config.DATA_DIR / f"news_scored_{sym}.parquet"
    src = RAW / f"news_{sym}.parquet"
    if not src.exists():
        return pd.DataFrame(columns=["created_at", "sent"])
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return pd.read_parquet(out)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    n = pd.read_parquet(src)
    txt = (n.headline.fillna("") + ". " + n.summary.fillna("")).str.slice(0, 600)
    n["sent"] = [a.polarity_scores(t)["compound"] for t in txt]
    low = n.headline.fillna("").str.lower()
    for k, pat in KW.items():
        n[k] = low.str.contains(pat, regex=True).astype(float)
    n = n[["created_at", "sent"] + list(KW)].sort_values("created_at").reset_index(drop=True)
    n.to_parquet(out)
    return n


def _count(times, close, window):
    """events with time in (close - window, close]; times and close are int64 ns arrays, times sorted."""
    return np.searchsorted(times, close, side="right") - np.searchsorted(times, close - window, side="right")


def _sum(times, cum, close, window):
    hi = np.searchsorted(times, close, side="right")
    lo = np.searchsorted(times, close - window, side="right")
    return cum[hi] - cum[lo]


def build(idx, until=None):
    """idx: UTC 5m bar start times. Returns a DataFrame of news/event features aligned on idx."""
    close = (idx + 5 * MIN).asi8
    cut = pd.Timestamp(until).tz_convert("UTC") if until is not None else None   # test hook: drop everything published later
    f = pd.DataFrame(index=idx)
    W = {"30m": 30 * MIN.value, "3h": 180 * MIN.value, "1d": 1440 * MIN.value, "5d": 7200 * MIN.value}
    for sym, tag in (("JPM", "jpm"), ("SPY", "mkt")):
        n = _scored(sym)
        if cut is not None:
            n = n[n.created_at <= cut]
        t = n.created_at.astype("int64").values if len(n) else np.array([], dtype="int64")
        if len(n) and n.created_at.dt.tz is None:
            t = n.created_at.dt.tz_localize("UTC").astype("int64").values
        s = np.concatenate([[0.0], np.cumsum(n.sent.values)]) if len(n) else np.zeros(1)
        for w in ("30m", "3h", "1d"):
            c = _count(t, close, W[w])
            f[f"{tag}_news_{w}"] = c
            f[f"{tag}_sent_{w}"] = np.where(c > 0, _sum(t, s, close, W[w]) / np.maximum(c, 1), 0.0)
        c5 = _count(t, close, W["5d"])
        f[f"{tag}_news_rel"] = f[f"{tag}_news_1d"] / (c5 / 5 + 0.5)
        last = np.searchsorted(t, close, side="right") - 1
        gap = np.where(last >= 0, (close - t[np.clip(last, 0, max(len(t) - 1, 0))]) / MIN.value, 1e5) if len(t) else np.full(len(close), 1e5)
        f[f"{tag}_mins_since_news"] = np.minimum(gap, 1440)
        if tag == "jpm":
            for k in KW:
                cum = np.concatenate([[0.0], np.cumsum(n[k].values)]) if len(n) else np.zeros(1)
                f[f"jpm_kw_{k}_1d"] = _sum(t, cum, close, W["1d"])

    ev = pd.read_parquet(RAW / "events.parquet") if (RAW / "events.parquet").exists() else pd.DataFrame(columns=["kind", "time"])
    if cut is not None:
        ev = ev[ev.time <= cut]
    def times(kind_prefix=None, kind=None):
        e = ev[ev.kind == kind] if kind else ev[ev.kind.str.startswith(kind_prefix)]
        return np.sort(e.time.astype("int64").values)
    def since(t, cap_min):
        if not len(t):
            return np.full(len(close), float(cap_min))
        last = np.searchsorted(t, close, side="right") - 1
        return np.where(last >= 0, np.minimum((close - t[np.clip(last, 0, None)]) / MIN.value, cap_min), cap_min)
    t_e = times(kind="jpm_earnings")
    f["mins_since_earn"] = since(t_e, 60 * 1440)
    f["earn_today"] = (f.mins_since_earn < 1440).astype(float) * (f.mins_since_earn <= (idx.tz_convert("America/New_York").hour * 60 + idx.tz_convert("America/New_York").minute - 570 + 5 + 480))
    f["earn_within_3d"] = (f.mins_since_earn < 3 * 1440).astype(float)
    t_f = times(kind="fomc")
    f["mins_since_fomc"] = since(t_f, 60 * 1440)
    f["fomc_today"] = (f.mins_since_fomc < 600).astype(float)
    nxt = np.searchsorted(t_f, close, side="right")
    # next FOMC decision: calendar is published a year in advance, so this is legitimately known at the time
    f["days_to_fomc"] = np.where(nxt < len(t_f), np.minimum((t_f[np.clip(nxt, 0, len(t_f) - 1)] - close) / (1440 * MIN.value), 60), 60)
    pe = np.sort(np.concatenate([times(kind=f"{x}_earnings") for x in ("bac", "c", "wfc", "gs", "ms")]))
    f["peer_earn_3d"] = _count(pe, close, 3 * 1440 * MIN.value)
    f["jpm_8k_1d"] = _count(times(kind_prefix="jpm_8k_"), close, W["1d"])
    f["jpm_insider_5d"] = _count(times(kind="jpm_insider"), close, W["5d"])
    f["jpm_13dg_5d"] = _count(times(kind="jpm_13dg"), close, W["5d"])
    f["mins_since_periodic"] = since(times(kind="jpm_periodic"), 120 * 1440)
    return f.astype(float)
