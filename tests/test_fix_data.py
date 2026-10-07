"""Review fixes: daily_array vs load() on never-visible versions, same-second
ingest rows, and the replay bar cache keyed by the code actually loaded."""
import os

import pytest

DAY = 86400


@pytest.fixture
def S(tmp_path, monkeypatch):
    from app.data import series
    monkeypatch.setattr(series, "STORE", str(tmp_path))
    return series


def test_daily_array_skips_versions_superseded_before_known(S):
    d, now = 300, 300 * DAY + 100
    t = d * DAY
    # same instant: known_at == superseded_at for the two earlier values
    for v in (0.9, -0.5, 0.1):
        S.ingest("same", [(t, v, now)], now=now)
    # known_at > superseded_at: a future-known value revised early
    S.ingest("early", [(t, 1.0, now + 50)], now=now)
    S.ingest("early", [(t, 2.0, now + 10)], now=now + 10)
    for name in ("same", "early"):
        want = [v for _, v in S.load(name, as_of=(d + 1) * DAY)]
        assert list(S.daily_array(name, [d])) == want


def test_ingest_push_keeps_same_second_rows_as_their_mean(tmp_path, monkeypatch):
    from app.data import ingest, series
    monkeypatch.setattr(ingest, "PATH", str(tmp_path / "ext.jsonl"))
    monkeypatch.setattr(series, "STORE", str(tmp_path / "s"))
    monkeypatch.setattr(ingest.time, "time", lambda: 1_790_000_000.25)
    ingest.push([{"asset": "BTC", "value": v, "kind": "news_lean"} for v in (0.9, -0.5, 0.1)])
    assert series._versions("ext:news_lean:BTC") == [
        (1_790_000_000, pytest.approx(0.5 / 3), 1_790_000_000.25, float("inf"))]


@pytest.fixture
def replay_disk(tmp_path, monkeypatch):
    """run_replay with persist=True against a temp cache dir, the simulation
    stubbed out (only the load/save decisions are under test), and every
    BC.load / BC.save call recorded."""
    import random

    from app.backtest import bar_cache as BC
    from app.backtest import replay as rp
    monkeypatch.setattr(BC, "DIR", str(tmp_path))
    calls = []
    load, save = BC.load, BC.save
    monkeypatch.setattr(BC, "load", lambda *a: calls.append("load") or load(*a))
    monkeypatch.setattr(BC, "save", lambda *a: calls.append("save") or save(*a))
    monkeypatch.setattr(rp, "_simulate", lambda *a, **k: {"ok": True})
    r = random.Random(0)
    data = {p: [[1_700_000_000 + i * 3600, 99, 101, 100, 100 + r.random(), 1000]
                for i in range(200)]
            for p in ("BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD")}
    return rp, BC, data, calls, tmp_path


def test_replay_never_persists_bars_when_source_changed_since_load(replay_disk, monkeypatch):
    rp, BC, data, calls, d = replay_disk
    off = {"only": ("trend",), "gate": 0.3, "shorts": True, "veto_align": 0.8}
    key = BC.cache_key(off, 24, 300, False)
    monkeypatch.setattr(BC, "code_version", lambda: "edited-on-disk")
    assert BC.cache_key(off, 24, 300, False) == key        # keyed by the loaded code
    c = {}
    assert rp.run_replay(data, cache=c, maker=False, shorts=True, persist=True)["ok"]
    assert calls == [] and "disk" not in c and not list(d.iterdir())


def test_replay_skips_save_when_tunables_change_mid_run(replay_disk, monkeypatch):
    rp, _BC, data, calls, d = replay_disk
    from app import tunables
    before = tunables.values()

    def sim(*a, **k):
        monkeypatch.setattr(tunables, "values", lambda: {**before, "stop_atr_mult": -1})
        return {"ok": True}
    monkeypatch.setattr(rp, "_simulate", sim)
    assert rp.run_replay(data, cache={}, maker=False, shorts=True, persist=True)["ok"]
    assert calls == ["load"] and not list(d.iterdir())


def test_server_startup_hashes_the_replay_code_as_loaded():
    """The persisted replay is safe only if the server hashes the source at
    startup — before a `git pull` could change it under a running process."""
    import subprocess
    import sys
    code = "import sys, app.main; assert 'app.backtest.bar_cache' in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


def test_store_panel_ignores_a_newest_bar_only_some_coins_have(monkeypatch, tmp_path):
    """The core / exploration loops ingest their own coins' newest day hourly;
    until the full sync brings the rest, a universe-wide champion would see
    every other coin as delisted on that row."""
    from app.data import store
    from app.engine import panel as P
    monkeypatch.setattr(store, "STORE", str(tmp_path))
    D = 86400
    bar = lambda d: [d * D, 9.9, 10.1, 10.0, 10.0 + d, 1000.0]
    for c in ("A-USD", "B-USD", "C-USD"):
        store.ingest_candles(c, D, [bar(d) for d in range(30)], now=40 * D)
    store.ingest_candles("A-USD", D, [bar(30)], now=40 * D)          # core loop: its coin only
    assert P.from_store().days[-1] == 29
    assert P.from_store(["A-USD"]).days[-1] == 30                   # complete for its own coins
    for c in ("B-USD", "C-USD"):                                     # the full sync catches up
        store.ingest_candles(c, D, [bar(30)], now=40 * D)
    assert P.from_store().days[-1] == 30
