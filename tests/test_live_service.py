"""Slim live service: the hourly bot only trades on evidence, the core holds
exactly the champion's backtested weights from the data store, stale data
never trades, and the scorecard reports champion vs benchmarks."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def _settings(monkeypatch, **vals):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: vals[k] if k in vals else real(k))


def test_hourly_bot_gate_follows_the_replay_evidence(monkeypatch):
    import app.orchestrator as O
    o = O.Orchestrator()
    o.last_replay = {"verdict": "negative"}
    _settings(monkeypatch, hourly_bot_mode="auto")
    assert o.hourly_bot_active() is False
    o.last_replay = {"verdict": "positive in both halves"}
    assert o.hourly_bot_active() is True
    _settings(monkeypatch, hourly_bot_mode="off")
    assert o.hourly_bot_active() is False
    _settings(monkeypatch, hourly_bot_mode="on")
    o.last_replay = {"verdict": "negative"}
    assert o.hourly_bot_active() is True


def _daily(n, seed, t0):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(n):
        px *= math.exp(0.001 + r.gauss(0, 0.03))
        out.append([t0 + i * 86400, px * 0.98, px * 1.02, px, px, 1e6])
    return out


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    from app.data import store
    monkeypatch.setattr(store, "STORE", str(tmp_path / "store"))
    return store


def test_core_holds_the_champions_backtested_weights(tmp_store, monkeypatch):
    from app.strategies.core import CoreBook
    from app.engine import challengers as C, panel as P
    now = time.time()
    t0 = int(now // 86400 - 600) * 86400
    for i, c in enumerate(("BTC-USD", "ETH-USD")):
        tmp_store.ingest_candles(c, 86400, _daily(599, i, t0), now=now)
    _settings(monkeypatch, core_strategy="champion", core_allocation_pct=100)
    core = CoreBook()
    t = core.targets(100_000, {}, now)
    name, cfg = C.champion()
    W = C.weights(cfg, P.from_store(cfg["assets"]))
    want = dict(zip(["BTC-USD", "ETH-USD"], W[-1]))
    for c in ("BTC-USD", "ETH-USD"):
        assert abs(t[c] - 100_000 * want[c]) < 1e-6
    assert core.mode_used == ("champion", name)


def test_core_never_trades_on_stale_data(tmp_store, monkeypatch):
    from app.strategies.core import CoreBook
    now = time.time()
    t0 = int(now // 86400 - 610) * 86400
    for i, c in enumerate(("BTC-USD", "ETH-USD")):
        tmp_store.ingest_candles(c, 86400, _daily(600, i, t0), now=now)   # ends ~10 days ago
    _settings(monkeypatch, core_strategy="champion", core_allocation_pct=100)
    core = CoreBook()
    core.positions = {"BTC-USD": {"qty": 1.0, "entry": 100.0, "opened": 0}}

    class Broker:
        cash = 1e6
    acts = core.rebalance(Broker(), None, 100_000, {}, now=now)
    assert acts == [] and core.positions["BTC-USD"]["qty"] == 1.0


def test_scorecard_shape(monkeypatch):
    import app.orchestrator as O

    class M:
        tickers = {}
        def price(self, p):
            return 100.0
    monkeypatch.setattr(O, "market", M())
    o = O.Orchestrator()
    sc = o.scorecard()
    assert {"champion", "candidates", "live", "core", "gates", "data"} <= set(sc)
    assert sc["champion"]["name"]
    assert set(sc["gates"]) >= {"hourly_bot_mode", "hourly_bot_active", "learners"}


def test_core_buys_what_cash_affords_after_fees(monkeypatch):
    """100% allocation: targets sum to all of equity; fees on the first buy
    must not leave the second slot in cash forever."""
    from app.strategies.core import CoreBook
    from app.execution.paper import PaperBroker
    monkeypatch.setattr("app.strategies.core.db.log_trade", lambda *a, **k: None)
    b = PaperBroker()
    b.cash = 100_000.0
    c = CoreBook()
    assert c._buy(b, "BTC-USD", 50_000.0, 100.0)
    assert c._buy(b, "ETH-USD", 50_000.0, 10.0)        # was refused: cash 49,750 < 50,250
    assert "ETH-USD" in c.positions
    assert b.cash >= -1e-6 and b.cash < 1.0             # fully invested, never negative


def test_allocation_change_rebalances_once(monkeypatch):
    """100% -> 90% is inside the 20% drift band per coin, but it is an
    operator decision: the core trims to the new targets once, then rests."""
    from app.strategies.core import CoreBook
    from app.execution.paper import PaperBroker
    monkeypatch.setattr("app.strategies.core.db.log_trade", lambda *a, **k: None)

    class M:
        def price(self, p):
            return 100.0
    b = PaperBroker()
    b.cash = 0.0
    c = CoreBook()
    c.positions = {"BTC-USD": {"qty": 500.0, "entry": 100.0, "opened": 0},
                   "ETH-USD": {"qty": 500.0, "entry": 100.0, "opened": 0}}
    c.applied_pct = 1.0
    monkeypatch.setattr(c, "targets", lambda eq, d, now=None: {"BTC-USD": 45_000.0,
                                                                "ETH-USD": 45_000.0})
    monkeypatch.setattr(CoreBook, "_settings", staticmethod(lambda: (0.9, ["BTC-USD", "ETH-USD"], True)))
    acts = c.rebalance(b, M(), 100_000.0, {})
    assert len(acts) == 2 and all("trimmed" in a for a in acts)
    assert b.cash > 9_000                              # freed for Polymarket
    assert c.rebalance(b, M(), 100_000.0, {}) == []    # once, then the band rules again
