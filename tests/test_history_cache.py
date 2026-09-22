"""TTL cache on fetch_history: cuts the repeated 3x HTTP round-trips per report
and degrades gracefully on network failure."""
import os, sys, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import engine


def _candles(n=5, base=0):
    return [[base + i, 1, 2, 1, 1.5, 10] for i in range(n)]


def setup_function(_):
    engine.clear_history_cache()


def test_second_call_is_served_from_cache(monkeypatch):
    calls = {"n": 0}

    async def fake_raw(product, granularity=3600, chunks=3):
        calls["n"] += 1
        return _candles()

    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)
    # disable disk layer for a clean in-memory test
    monkeypatch.setattr(engine, "_disk_read", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_disk_write", lambda *a, **k: None)

    a = asyncio.run(engine.fetch_history("BTC-USD"))
    b = asyncio.run(engine.fetch_history("BTC-USD"))
    assert a == b
    assert calls["n"] == 1            # network hit only once


def test_use_cache_false_always_fetches(monkeypatch):
    calls = {"n": 0}

    async def fake_raw(product, granularity=3600, chunks=3):
        calls["n"] += 1
        return _candles()

    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)
    asyncio.run(engine.fetch_history("ETH-USD", use_cache=False))
    asyncio.run(engine.fetch_history("ETH-USD", use_cache=False))
    assert calls["n"] == 2


def test_expired_ttl_refetches(monkeypatch):
    calls = {"n": 0}

    async def fake_raw(product, granularity=3600, chunks=3):
        calls["n"] += 1
        return _candles()

    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)
    monkeypatch.setattr(engine, "_disk_read", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_disk_write", lambda *a, **k: None)

    asyncio.run(engine.fetch_history("SOL-USD", ttl=0))
    time.sleep(0.01)
    asyncio.run(engine.fetch_history("SOL-USD", ttl=0))
    assert calls["n"] == 2            # ttl=0 -> always stale -> refetch


def test_network_failure_serves_stale_memory(monkeypatch):
    state = {"fail": False}

    async def fake_raw(product, granularity=3600, chunks=3):
        if state["fail"]:
            raise RuntimeError("coinbase down")
        return _candles(base=100)

    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)
    monkeypatch.setattr(engine, "_disk_read", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_disk_write", lambda *a, **k: None)

    good = asyncio.run(engine.fetch_history("BTC-USD", ttl=0))
    state["fail"] = True
    # ttl=0 forces a refetch attempt; it fails -> stale memory copy returned
    served = asyncio.run(engine.fetch_history("BTC-USD", ttl=0))
    assert served == good


def test_network_failure_with_no_cache_raises(monkeypatch):
    async def fake_raw(product, granularity=3600, chunks=3):
        raise RuntimeError("coinbase down")

    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)
    monkeypatch.setattr(engine, "_disk_read", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_disk_write", lambda *a, **k: None)

    try:
        asyncio.run(engine.fetch_history("NEW-USD"))
        assert False, "expected the error to propagate with no cache to fall back on"
    except RuntimeError:
        pass


def test_disk_cache_round_trip(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def fake_raw(product, granularity=3600, chunks=3):
        calls["n"] += 1
        return _candles(base=7)

    monkeypatch.setattr(engine, "_DISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_fetch_history_raw", fake_raw)

    # first call: network + writes disk + memory
    asyncio.run(engine.fetch_history("XRP-USD"))
    # wipe memory so the next call must use disk
    engine.clear_history_cache()
    served = asyncio.run(engine.fetch_history("XRP-USD"))
    assert served == _candles(base=7)
    assert calls["n"] == 1            # disk hit, no second network call
