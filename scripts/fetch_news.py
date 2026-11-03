"""Download timestamped news from the Alpaca news API (the same account/keys as the price data).
Stored: id, created_at (UTC, publish time), headline, summary, source, symbols. Incremental and resumable.
Usage: uv run python scripts/fetch_news.py [SYMBOL ...]   (default JPM SPY)"""
import sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jpmind import config

URL = "https://data.alpaca.markets/v1beta1/news"
H = {"APCA-API-KEY-ID": config.ALPACA_KEY_ID, "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY}

def fetch(sym, start="2015-12-01T00:00:00Z"):
    out = config.DATA_DIR / "raw" / f"news_{sym}.parquet"
    old = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    if len(old):
        start = (old.created_at.max() - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, tok = [], None
    while True:
        p = {"symbols": sym, "start": start, "limit": 50, "sort": "asc", "include_content": "false"}
        if tok: p["page_token"] = tok
        for attempt in range(6):
            try:
                r = requests.get(URL, headers=H, params=p, timeout=60)
                if r.status_code == 429: time.sleep(5 * (attempt + 1)); continue
                r.raise_for_status(); break
            except requests.RequestException as e:
                time.sleep(3 * (attempt + 1))
        else:
            break
        j = r.json()
        for n in j.get("news", []):
            rows.append({"id": n["id"], "created_at": n["created_at"], "headline": n["headline"], "summary": n.get("summary", ""),
                         "source": n.get("source", ""), "symbols": ",".join(n.get("symbols", []))})
        tok = j.get("next_page_token")
        if len(rows) % 2000 < 50: print(sym, len(rows), rows[-1]["created_at"] if rows else "", flush=True)
        if not tok: break
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if len(df):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.drop_duplicates("id").sort_values("created_at").reset_index(drop=True)
        df.to_parquet(out)
    print(sym, "total", len(df), df.created_at.min() if len(df) else "", df.created_at.max() if len(df) else "", flush=True)

if __name__ == "__main__":
    for s in (sys.argv[1:] or ["JPM", "SPY"]):
        fetch(s)
