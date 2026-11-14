"""Event calendar: JPM earnings releases (SEC 8-K item 2.02, with the exact acceptance time) and FOMC decisions.
Output data/raw/events.parquet: columns kind, time (UTC, when the event became public)."""
import re, sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config

H = {"User-Agent": "jpmind-research-script student-project"}
rows = []

CIKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001", "WFC": "0000072971", "GS": "0000886982", "MS": "0000895421"}


def _ts(t):
    t = pd.Timestamp(t)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def sec():
    """Every filing is stamped with its SEC acceptance time, i.e. when it became public."""
    base = "https://data.sec.gov/submissions/"
    for sym, cik in CIKS.items():
        j = requests.get(f"{base}CIK{cik}.json", headers=H, timeout=30).json()["filings"]
        parts = [pd.DataFrame(j["recent"])]
        for f in j["files"]:
            if f["filingTo"] >= "2015-12-01":
                time.sleep(0.15)
                parts.append(pd.DataFrame(requests.get(base + f["name"], headers=H, timeout=30).json()))
        d = pd.concat(parts)
        d["items"] = d["items"].fillna("")
        for _, r in d.iterrows():
            t = _ts(r.acceptanceDateTime)
            if sym != "JPM":
                if r.form == "8-K" and "2.02" in r["items"]:
                    rows.append((f"{sym.lower()}_earnings", t))
                continue
            if r.form == "8-K":
                for it in [x for x in r["items"].split(",") if x]:
                    rows.append((("jpm_earnings" if it == "2.02" else f"jpm_8k_{it}"), t))
            elif r.form in ("10-K", "10-Q"):
                rows.append(("jpm_periodic", t))
            elif r.form in ("4", "4/A"):
                rows.append(("jpm_insider", t))
            elif r.form in ("SC 13G", "SC 13G/A", "SC 13D", "SC 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A"):
                rows.append(("jpm_13dg", t))
        print(sym, "filings scanned", len(d), flush=True)


MONTHS = "January February March April May June July August September October November December".split()

def fomc():
    cur = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    hist = [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in range(2016, 2021)]
    mon = "|".join(MONTHS + [f"{a[:3]}/{b[:3]}" for a, b in zip(MONTHS, MONTHS[1:])])
    for u in [cur] + hist:
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", requests.get(u, headers=H, timeout=30).text))
        # current page: "<Month> d-d[*] Statement" inside "<year> FOMC Meetings"; historical: "<Month> d-d Meeting - <year>"
        spans = [(y, t.split(f"{y} FOMC Meetings")[1].split("FOMC Meetings")[0]) for y in set(re.findall(r"(20\d\d) FOMC Meetings", t))]
        if "historical" in u:
            spans = [(u[-8:-4], t)]
        for y, seg in spans:
            pat = r"(%s)(?:\.)? (\d{1,2})(?:-(\d{1,2}))?\*? (?:Statement|Meeting)" % mon
            for m in re.finditer(pat, seg):
                name, d1, d2 = m.group(1), int(m.group(2)), m.group(3)
                first = name.split("/")[-1] if "/" in name else name
                mi = [x[:3] for x in MONTHS].index(first[:3]) + 1
                try:
                    ts = pd.Timestamp(year=int(y), month=mi, day=int(d2) if d2 else d1, hour=14, tz="America/New_York")
                except ValueError:
                    continue
                if int(y) <= 2026:
                    rows.append(("fomc", ts.tz_convert("UTC")))
        time.sleep(0.3)

if __name__ == "__main__":
    sec(); fomc()
    df = pd.DataFrame(rows, columns=["kind", "time"]).drop_duplicates().sort_values("time")
    df = df[df.time >= "2015-12-01"]
    df.to_parquet(config.DATA_DIR / "raw" / "events.parquet")
    print(df.kind.value_counts().to_string())
