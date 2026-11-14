"""Get ready to test on trading day T: download bars up to the latest completed close, then retrain every model on
sessions up to T-1 (the session before T). The dashboard then only offers days after T-1 for replay, so T is unseen.

Usage:
  uv run python scripts/prepare_test_day.py                  # T = latest stored session after fetching
  uv run python scripts/prepare_test_day.py --day 2026-11-13 # a specific T
  add --no-fetch to skip the download, --fast to skip the slow train.py (rest-of-day band / scenarios keep their old cutoff)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(*args):
    t = time.time()
    print(f"\n>>> {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"<<< done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="test day T (YYYY-MM-DD); default: latest stored session")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--fast", action="store_true", help="skip scripts/train.py")
    a = ap.parse_args()
    if not a.no_fetch:
        run("scripts/fetch_data.py")
    from jpmind import bars as B, config, store
    stored = sorted(set(store.load(config.SYMBOL).index.tz_convert(B.ET).date))
    cal = sorted(pd.to_datetime(store.calendar()["date"]).dt.date)
    T = pd.Timestamp(a.day).date() if a.day else stored[-1]
    prev = [d for d in cal if d < T]
    if not prev:
        sys.exit(f"no session before {T}")
    cutoff = str(prev[-1])
    print(f"test day T = {T}, training cutoff T-1 = {cutoff}" + ("" if T in stored else "  (T not downloaded yet: fetch after it closes)"))
    run("scripts/train_cls3.py", "--cutoff", cutoff)       # graded 3-class call, activity, big-move (~3 min)
    run("scripts/train_tick.py", "--cutoff", cutoff)       # next minute + rest of today, session outlooks
    run("scripts/next_day.py", "--cutoff", cutoff)         # next-day outlook
    if not a.fast:
        run("scripts/train.py", "--cutoff", cutoff)        # rest-of-day quantile band + scenarios
    print(f"\nReady. Start the dashboard (uv run uvicorn server.app:app --port 8000) and pick {T} in the Day menu.")


if __name__ == "__main__":
    main()
