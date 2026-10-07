"""Market data store: closed bars only, invalid rows rejected, revisions kept
and replayable as-of, stable fingerprints, point-in-time universe."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

H = 3600


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from app.data import store as s
    monkeypatch.setattr(s, "STORE", str(tmp_path))
    return s


def _bars(t0, n, px=100.0, step=1.0):
    return [[t0 + i * H, px + i * step - 1, px + i * step + 1, px + i * step,
             px + i * step + 0.5, 10.0] for i in range(n)]


def test_still_forming_bar_is_never_stored(store):
    now = 1_000_000 * H + 1800                      # half way through a bar
    rows = _bars(1_000_000 * H - 5 * H, 6)          # last bar is the open one
    st = store.ingest_candles("BTC-USD", H, rows, now=now)
    assert st["new"] == 5
    data, _ = store.load_candles(H)
    assert data["BTC-USD"][-1][0] == 1_000_000 * H - H


def test_invalid_rows_rejected(store):
    rows = _bars(0, 4)
    rows[1][4] = 0.0                                # close <= 0
    rows[2][1], rows[2][2] = 110.0, 100.0           # low > high
    rows[3][5] = float("nan")
    st = store.ingest_candles("X-USD", H, rows, now=10 * H)
    assert st["new"] == 1 and st["rejected"] == 3


def test_revision_kept_and_as_of_replays_history(store):
    rows = _bars(0, 10)
    store.ingest_candles("BTC-USD", H, rows, now=100 * H)
    v1 = store.load_candles(H)[1]
    revised = [list(r) for r in rows]
    revised[4][4] = 999.0
    st = store.ingest_candles("BTC-USD", H, revised + _bars(10 * H, 2), now=200 * H)
    assert st == {**st, "new": 2, "revised": 1}
    now_data, v2 = store.load_candles(H)
    assert now_data["BTC-USD"][4][4] == 999.0 and len(now_data["BTC-USD"]) == 12
    then, v_then = store.load_candles(H, as_of=150 * H)      # before the revision
    assert then["BTC-USD"][4][4] == rows[4][4] and len(then["BTC-USD"]) == 10
    assert v_then == v1 != v2
    assert store.quality(H)["BTC-USD"]["revisions"] == 1


def test_identical_reingest_changes_nothing(store):
    rows = _bars(0, 20)
    store.ingest_candles("ETH-USD", H, rows, now=100 * H)
    v = store.load_candles(H)[1]
    st = store.ingest_candles("ETH-USD", H, rows, now=200 * H)
    assert st["new"] == 0 and st["revised"] == 0
    assert store.load_candles(H)[1] == v


def test_quality_flags_gaps_and_reverting_spikes(store):
    rows = _bars(0, 30)
    del rows[10:13]                                 # 3-bar gap
    rows[20][4] = rows[20][4] * 2                   # spike up ...
    rows[20][2] = rows[20][4] + 1
    store.ingest_candles("SOL-USD", H, rows, now=100 * H)
    q = store.quality(H)["SOL-USD"]
    assert q["missing_bars"] == 3 and q["longest_gap_bars"] == 3
    assert q["reverting_spikes"] == 1               # ... and straight back down


def test_universe_at_is_point_in_time_and_keeps_delisted(store):
    D = 86400
    def daily(n, vol, t0=0):
        return [[t0 + i * D, 9, 11, 10, 10, vol] for i in range(n)]
    store.ingest_candles("OLD-USD", D, daily(200, 1e6), now=10**9)        # delisted at day 200
    store.ingest_candles("NEW-USD", D, daily(100, 5e6, t0=300 * D), now=10**9)
    store.ingest_candles("USDT-USD", D, daily(400, 1e9), now=10**9)       # stablecoin
    assert store.universe_at(150 * D, n=5) == ["OLD-USD"]                 # NEW not listed yet
    assert store.universe_at(399 * D, n=5) == ["NEW-USD"]                 # OLD no longer trades
    assert "USDT-USD" not in store.universe_at(399 * D, n=5)


def test_replay_loader_reads_store_with_fingerprint(store, tmp_path):
    from app.backtest import replay as rp
    now = 2_000_000 * H
    store.ingest_candles("BTC-USD", H, _bars(now - 48 * H, 48), now=now)
    store.ingest_candles("ETH-USD", H, _bars(now - 48 * H, 48), now=now)
    data, ver = rp.load_history(H, days=1, products=["BTC-USD"], as_of=now, root=str(tmp_path))
    assert list(data) == ["BTC-USD"] and len(data["BTC-USD"]) == 24   # the last 24 closed hours
    assert ver == store.fingerprint(data)


def test_replay_loader_falls_back_to_legacy_cache(store, tmp_path):
    import json
    from app.backtest import replay as rp
    d = tmp_path / ".cache" / "history"
    d.mkdir(parents=True)
    (d / "BTC-USD_3600_3.json").write_text(json.dumps(_bars(0, 5)))
    data, ver = rp.load_history(H, root=str(tmp_path))
    assert len(data["BTC-USD"]) == 5 and ver


def test_candle_range_and_product_filters(store):
    store.ingest_candles("BTC-USD", H, _bars(0, 6), now=10 * H)
    store.ingest_candles("ETH-USD", H, _bars(0, 6), now=10 * H)
    data, version = store.load_candles(H, ["BTC-USD"], start=H, end=3 * H)
    assert list(data) == ["BTC-USD"]
    assert [r[0] for r in data["BTC-USD"]] == [H, 2 * H]
    assert version == store.fingerprint(data)
    assert store.products("candles", H) == ["BTC-USD", "ETH-USD"]
    assert store.last_ts("candles", H, "BTC-USD") == 5 * H
    assert store.last_ts("candles", H, "MISSING-USD") is None


def test_funding_revisions_and_venue_isolation(store):
    rows = [[0, 0.001], [H, -0.002]]
    assert store.ingest_funding("hyperliquid", "BTC", rows, now=2 * H)["new"] == 2
    assert store.ingest_funding("hyperliquid", "BTC", rows, now=3 * H)["revised"] == 0
    store.ingest_funding("hyperliquid", "BTC", [[H, 0.003]], now=4 * H)
    store.ingest_funding("deribit", "BTC", [[H, 0.004]], now=4 * H)
    old, _ = store.load_funding("hyperliquid", as_of=3 * H)
    current, _ = store.load_funding("hyperliquid")
    other, _ = store.load_funding("deribit")
    assert old["BTC"] == rows
    assert current["BTC"][-1] == [H, 0.003]
    assert other["BTC"] == [[H, 0.004]]


def test_invalid_and_future_funding(store):
    stats = store.ingest_funding(
        "deribit", "BTC", [[0, 0.001], [H, float("nan")], [3 * H, 0.002]],
        now=2 * H)
    assert stats["new"] == 1 and stats["rejected"] == 1


def test_exploration_loop_reaches_initial_wait(monkeypatch):
    import asyncio
    from app.orchestrator import Orchestrator

    class ReachedWait(Exception):
        pass

    async def initial_wait(seconds):
        assert seconds == 200
        raise ReachedWait

    monkeypatch.setattr(asyncio, "sleep", initial_wait)
    orchestrator = Orchestrator.__new__(Orchestrator)
    with pytest.raises(ReachedWait):
        asyncio.run(orchestrator.exploration_loop())
