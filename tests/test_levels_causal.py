"""Support/resistance levels at a cursor use only bars closed by then.

compute() only ever receives history + today's bars up to the cursor, so the checks are:
  1. every non-pivot level is a price that has already traded (inside the seen low..high range);
  2. HOD/LOD equal the high/low of today's regular-hours bars seen so far, not of the full day;
  3. the opening range does not exist before the 09:59 bar has closed, and is fixed afterwards.
"""
import pandas as pd
import pytest

from jpmind import bars as B, config, levels as LV, store

PIVOTS = {"P", "R1", "R2", "S1", "S2"}


@pytest.fixture(scope="module")
def day_data():
    d = store.load(config.SYMBOL)
    if d is None or not len(d):
        pytest.skip("no stored bars")
    day = sorted(set(d.index.tz_convert(B.ET).date))[-2]
    start = pd.Timestamp(day, tz=B.ET).tz_convert("UTC")
    hist = d[d.index < start].iloc[-60 * 24 * 12:]
    today = d[(d.index >= start) & (d.index < start + pd.Timedelta(days=1))]
    return day, hist, today


def _upto(day, today, hhmm):
    return today[today.index <= pd.Timestamp(f"{day} {hhmm}", tz=B.ET).tz_convert("UTC")]


@pytest.mark.parametrize("hhmm", ["09:35", "09:58", "09:59", "11:17", "14:02", "15:59"])
def test_levels_use_only_seen_prices(day_data, hhmm):
    day, hist, today = day_data
    seen = _upto(day, today, hhmm)
    out = LV.compute(hist, seen)
    rth_hist = B.resample(hist, "1m", "rth")
    lo = min(rth_hist.low.min(), seen.low.min())
    hi = max(rth_hist.high.max(), seen.high.max())
    for lv in out["levels"]:
        if not set(lv["label"].split("/")) <= PIVOTS:
            assert lo - 1e-6 <= lv["price"] <= hi + 1e-6, lv
    rth_seen = B.resample(seen, "1m", "rth")
    labels = {n: lv for lv in out["levels"] for n in lv["label"].split("/")}
    if "HOD" in labels and labels["HOD"]["strength"] == 1:
        assert labels["HOD"]["price"] == round(float(rth_seen.high.max()), 2)
    if "LOD" in labels and labels["LOD"]["strength"] == 1:
        assert labels["LOD"]["price"] == round(float(rth_seen.low.min()), 2)
    assert out["opening_range_done"] == (hhmm >= "09:59")
    if hhmm < "09:59":
        assert "ORH" not in labels and "ORL" not in labels


def test_opening_range_fixed_after_10(day_data):
    day, hist, today = day_data
    a = LV.compute(hist, _upto(day, today, "10:05"))
    b = LV.compute(hist, _upto(day, today, "15:59"))
    get = lambda o, n: next(l for l in o["levels"] if n in l["label"].split("/"))
    for n in ("ORH", "ORL"):
        if get(a, n)["strength"] == 1 and get(b, n)["strength"] == 1:
            assert get(a, n)["price"] == get(b, n)["price"]
