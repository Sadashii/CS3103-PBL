"""Quarterly fundamentals from SEC XBRL as known on each trading day.
A figure becomes usable the day AFTER its first filing date (filings only carry a date, so this is the conservative reading),
and only the first-filed value of each quarter is used (later restatements are never seen early)."""
import numpy as np
import pandas as pd

from . import config

TAGS = {"EarningsPerShareDiluted": "eps", "NetIncomeLoss": "ni", "Revenues": "rev", "InterestIncomeExpenseNet": "nii",
        "NoninterestIncome": "nonii", "NoninterestExpense": "nonie", "ProvisionForLoanLeaseAndOtherLosses": "prov"}


def _quarters(sym):
    f = config.DATA_DIR / "raw" / f"xbrl_{sym}.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    out = {}
    for tag, nm in TAGS.items():
        x = d[d.tag == tag].copy()
        if x.empty:
            continue
        x["end"], x["filed"] = pd.to_datetime(x.end), pd.to_datetime(x.filed)
        x = x.sort_values(["end", "filed"]).drop_duplicates("end", keep="first").reset_index(drop=True)   # first-filed value only
        prev = x[["end", "val"]].rename(columns={"end": "pend", "val": "pval"})
        prev["pend"] = prev.pend + pd.Timedelta(days=365)
        x = pd.merge_asof(x.sort_values("end"), prev.sort_values("pend"), left_on="end", right_on="pend",
                          tolerance=pd.Timedelta(days=25), direction="nearest")
        x["yoy"] = (x.val - x.pval) / x.pval.abs().clip(lower=1e-9)
        x["yoy"] = x.yoy.clip(-3, 3)
        out[nm] = x[["end", "filed", "val", "yoy"]]
    return out


def build(sym, days):
    """days: tz-naive DatetimeIndex of trading dates. Returns a feature frame aligned on days."""
    q = _quarters(sym)
    F = pd.DataFrame(index=days)
    filed_cols = []
    for nm, x in q.items():
        x = x.sort_values("filed")
        x = x.assign(avail=(x.filed + pd.Timedelta(days=1)).astype("datetime64[ns]")).sort_values("avail")
        xx = pd.merge_asof(pd.DataFrame({"day": pd.DatetimeIndex(days).astype("datetime64[ns]")}), x,
                           left_on="day", right_on="avail", direction="backward")
        F[f"fund_{nm}_yoy"] = xx.yoy.values
        filed_cols.append(pd.Series(xx.filed.values, index=days))
    if filed_cols:
        latest = pd.concat(filed_cols, axis=1).max(axis=1)
        F["fund_age_days"] = (pd.Series(days, index=days) - latest).dt.days.clip(upper=200)
    return F
