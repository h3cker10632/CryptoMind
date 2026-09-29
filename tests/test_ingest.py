"""External-signal ingest seam + crawl4ai producer mapping — offline tests."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data import ingest


def _isolate():
    """Point the ingest store at a fresh temp file for each test."""
    fd, path = tempfile.mkstemp(prefix="ingest-test-", suffix=".jsonl")
    os.close(fd)
    os.remove(path)
    ingest.PATH = path
    return path


def test_normalize_row_bounds_and_symbol():
    r = ingest.normalize_row({"asset": "BTC-USD", "value": 5.0, "source": "x"})
    assert r["asset"] == "BTC" and r["value"] == 1.0        # clamped to [-1,1]
    assert ingest.normalize_row({"asset": "ETH"}) is None    # no value
    assert ingest.normalize_row({"value": 0.5}) is None      # no asset


def test_push_and_feature_freshness_weighted():
    _isolate()
    now = time.time()
    res = ingest.push([
        {"asset": "BTC", "value": 0.8, "source": "crawl4ai", "text": "bullish", "ts": now},
        {"asset": "BTC", "value": 0.2, "source": "crawl4ai", "ts": now - 60},
        {"asset": "ETH", "value": -0.5, "source": "crawl4ai", "ts": now},
    ])
    assert res["accepted"] == 3 and res["rejected"] == 0
    f = ingest.feature("BTC-USD")
    assert f is not None and 0.2 < f <= 0.8                  # weighted toward fresh
    assert ingest.feature("DOGE") is None                    # no data


def test_feature_ignores_stale_rows():
    _isolate()
    old = time.time() - (ingest.DEFAULT_MAX_AGE + 100)
    ingest.push([{"asset": "BTC", "value": 0.9, "source": "s", "ts": old}])
    assert ingest.feature("BTC") is None                     # too old to count


def test_latest_texts_and_status():
    _isolate()
    now = time.time()
    ingest.push([
        {"asset": "BTC", "value": 0.5, "source": "crawl4ai", "text": "etf inflows", "ts": now},
        {"asset": "BTC", "value": 0.3, "source": "maxun", "text": "old news", "ts": now - 10},
    ])
    texts = ingest.latest_texts("BTC", n=2)
    assert len(texts) == 2 and "etf inflows" in texts[0]
    st = ingest.status()
    assert st["total_rows"] == 2 and "BTC" in st["assets"]
    assert st["sources"]["crawl4ai"] == 1 and st["sources"]["maxun"] == 1


def test_push_rejects_garbage():
    _isolate()
    res = ingest.push([{"nope": 1}, "not a dict", {"asset": "BTC", "value": 0.1}])
    assert res["accepted"] == 1 and res["rejected"] == 2


def test_study_reports_not_enough_gracefully():
    _isolate()
    ingest.push([{"asset": "BTC", "value": 0.5, "source": "s"}])
    rep = ingest.study()
    # no matured price pairs -> honest not-ok, never a crash
    assert rep["ok"] is False and "error" in rep


def test_crawl4ai_producer_build_rows_is_pure():
    from tools.crawl4ai_signal.producer import build_rows, SOURCE
    rows = build_rows([("BTC", 0.7, "bullish news"),
                       ("ETH", None, "skipped"),        # None lean dropped
                       ("SOL", 3.0, "clamped high")])
    assert len(rows) == 2
    assert rows[0]["source"] == SOURCE and rows[0]["asset"] == "BTC"
    assert rows[1]["value"] == 1.0                      # clamped to 1.0
    # rows are ingestable as-is
    _isolate()
    assert ingest.push(rows)["accepted"] == 2


def test_html_strip_fallback():
    from tools.crawl4ai_signal.producer import _strip_html
    txt = _strip_html("<html><body><script>x=1</script><p>Hello  world</p></body></html>")
    assert "Hello world" in txt and "x=1" not in txt
