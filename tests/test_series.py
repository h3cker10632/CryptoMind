"""External series store: point-in-time loads with revisions, causal daily
alignment, the source parsers, and ingest history."""
import math

import numpy as np
import pytest

DAY = 86400


@pytest.fixture
def S(tmp_path, monkeypatch):
    from app.data import series
    monkeypatch.setattr(series, "STORE", str(tmp_path))
    return series


def test_ingest_revisions_and_as_of(S):
    t0 = 100 * DAY
    st = S.ingest("x", [(t0, 1.0, t0 + DAY), (t0 + DAY, 2.0, t0 + 2 * DAY)], now=t0 + 3 * DAY)
    assert st["new"] == 2
    assert S.ingest("x", [(t0, 1.0, t0 + DAY)], now=t0 + 4 * DAY)["unchanged"] == 1
    st = S.ingest("x", [(t0, 1.5, t0 + DAY)], now=t0 + 5 * DAY)        # revised later
    assert st["revised"] == 1
    assert S.load("x") == [(t0, 1.5), (t0 + DAY, 2.0)]
    assert S.load("x", as_of=t0 + 4 * DAY) == [(t0, 1.0), (t0 + DAY, 2.0)]
    assert S.load("x", as_of=t0 + 5 * DAY) == [(t0, 1.5), (t0 + DAY, 2.0)]
    assert S.load("x", as_of=t0 + DAY) == [(t0, 1.0)]                    # 2nd not yet known
    bad = S.ingest("x", [(t0, float("nan"), t0 + DAY), (t0, 1.0, t0 - 1),
                         (t0, 1.0, t0 + 100 * DAY)], now=t0 + 6 * DAY)
    assert bad["rejected"] == 3
    assert S.names() == ["x"] and S.last_ts("x") == t0 + DAY


def test_daily_array_is_causal_and_sees_revisions_only_when_made(S):
    d0 = 200
    S.ingest("y", [(d0 * DAY, 10.0, (d0 + 1) * DAY), ((d0 + 1) * DAY, 11.0, (d0 + 3) * DAY)],
             now=(d0 + 3) * DAY)
    S.ingest("y", [(d0 * DAY, 12.0, (d0 + 1) * DAY)], now=(d0 + 5) * DAY + 10)
    days = np.arange(d0 - 1, d0 + 20)
    a = S.daily_array("y", days, stale_days=7)
    get = lambda d: a[d - (d0 - 1)]
    assert math.isnan(get(d0 - 1))                 # nothing known yet
    assert get(d0) == 10.0                         # day d0's value known at its close
    assert get(d0 + 1) == 10.0                     # day d0+1's value not known until d0+3
    assert get(d0 + 2) == 11.0
    assert get(d0 + 4) == 11.0                     # latest point wins; revision of d0 irrelevant
    assert math.isnan(get(d0 + 9))                 # older than stale_days
    b = S.daily_array("y", np.array([d0]), stale_days=7)
    assert b[0] == 10.0                            # the revision (made at d0+5) is not leaked


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Client:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return _Resp(self.pages.pop(0))


def test_fear_greed_parser():
    from app.data import series_sources as SS
    c = _Client([{"name": "Fear and Greed Index",
                  "data": [{"value": "40", "value_classification": "Fear",
                            "timestamp": "1551157200"},
                           {"value": "bad", "timestamp": "1551243600"}]}])
    rows = SS.fear_greed(c)
    assert rows == [(1551157200, 40.0, 1551157200 + 3600)]
    assert c.calls[0][1]["limit"] == 0


def test_deribit_dvol_pages_with_continuation():
    from app.data import series_sources as SS
    p1 = {"jsonrpc": "2.0", "result": {"data": [[1700092800000, 50, 52, 49, 51.5]],
                                       "continuation": 1700006400000}}
    p2 = {"jsonrpc": "2.0", "result": {"data": [[1700006400000, 48, 50, 47, 49.0]],
                                       "continuation": None}}
    c = _Client([p1, p2])
    rows = SS.deribit_dvol(c, "BTC", start=1_699_000_000, end=1_700_200_000)
    assert [r[0] for r in rows] == [1700006400, 1700092800]
    assert rows[0][1] == 49.0 and rows[0][2] == 1700006400 + DAY
    assert c.calls[1][1]["end_timestamp"] == 1700006400000


def test_defillama_and_coinmetrics_parsers():
    from app.data import series_sources as SS
    rows = SS.parse_defillama_stablecoins([
        {"date": "1609459200", "totalCirculatingUSD": {"peggedUSD": 2.8e10, "peggedEUR": 1e8}},
        {"date": "x"}])
    assert rows == [(1609459200, 2.81e10, 1609459200 + DAY)]
    c = _Client([{"data": [{"asset": "btc", "time": "2015-01-01T00:00:00.000000000Z",
                            "AdrActCnt": "187767"}], "next_page_url": "https://next"},
                 {"data": [{"asset": "btc", "time": "2015-01-02T00:00:00.000000000Z",
                            "AdrActCnt": "190000"}]}])
    rows = SS.coinmetrics(c, "btc", "AdrActCnt")
    assert rows[0] == (1420070400, 187767.0, 1420070400 + 2 * DAY) and len(rows) == 2
    assert c.calls[1][0] == "https://next" and c.calls[1][1] is None


def test_ingested_signals_are_kept_as_uncapped_history(monkeypatch, tmp_path):
    from app.data import ingest, series
    monkeypatch.setattr(ingest, "PATH", str(tmp_path / "ext.jsonl"))
    monkeypatch.setattr(series, "STORE", str(tmp_path / "s"))
    monkeypatch.setattr(ingest, "MAX_ROWS", 3)
    for i in range(5):
        ingest.push({"asset": "BTC", "value": 0.1 * i, "kind": "news_lean",
                     "ts": 1_700_000_000 + i * 3600})
    assert len(ingest._read_all()) == 3                      # bounded live view
    assert len(series.load("ext:news_lean:BTC")) == 5        # full history
