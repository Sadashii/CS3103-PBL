"""Project-wide settings. Alpaca is used only to download data into the local store."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SYMBOL = "JPM"
CONTEXT_SYMBOLS = ["SPY", "XLF", "BAC", "C", "WFC", "GS", "MS"]   # S&P 500, bank sector, peer banks
FEED = os.environ.get("ALPACA_FEED", "sip")  # which stored feed to read: iex or sip

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

DATA_REST = "https://data.alpaca.markets/v2"

HORIZONS = [5, 10, 30]          # minutes ahead
MAIN_HORIZON = 10               # the horizon signals are graded on
QUANTILES = [0.1, 0.5, 0.9]
COST_BPS = 2.0                  # round-trip cost assumed per trade, in basis points

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
