"""Quarterly financials from SEC XBRL (company facts), free. Each fact keeps its `filed` date, i.e. when it became public.
Output data/raw/xbrl_<SYM>.parquet: end, start, filed, tag, val (3-month periods only)."""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config

H = {"User-Agent": "jpmind-research-script student-project"}
CIKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001", "WFC": "0000072971", "GS": "0000886982", "MS": "0000895421"}
TAGS = ["EarningsPerShareDiluted", "NetIncomeLoss", "Revenues", "InterestIncomeExpenseNet", "ProvisionForLoanLeaseAndOtherLosses",
        "ProvisionForCreditLosses", "NoninterestIncome", "NoninterestExpense"]

for sym, cik in CIKS.items():
    j = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=H, timeout=60).json()
    rows = []
    for tag in TAGS:
        f = j.get("facts", {}).get("us-gaap", {}).get(tag)
        if not f:
            continue
        for unit, items in f["units"].items():
            for it in items:
                if "start" not in it:
                    continue
                days = (pd.Timestamp(it["end"]) - pd.Timestamp(it["start"])).days
                if 80 <= days <= 100:                      # a single quarter
                    rows.append({"tag": tag, "start": it["start"], "end": it["end"], "filed": it["filed"], "val": it["val"], "form": it.get("form")})
    d = pd.DataFrame(rows)
    d.to_parquet(config.DATA_DIR / "raw" / f"xbrl_{sym}.parquet")
    print(sym, len(d), sorted(d.tag.unique()) if len(d) else "", flush=True)
    time.sleep(0.3)
