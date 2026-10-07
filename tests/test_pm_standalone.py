"""Polymarket on its own bankroll, separate from the main account."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import db
from app.portfolio import PaperPortfolio
from app.markets.polymarket.broker import PMBroker


@pytest.fixture(autouse=True)
def _reset_portfolio_tables():
    db.reset_paper_portfolio_for_tests()
    yield


def _market(cid="pm-cond-1"):
    return {"condition_id": cid, "token_ids": ["tokA", "tokB"],
            "outcomes": ["A", "B"], "question": "Q?", "category": "test",
            "tick_size": 0.01, "min_order_size": 5.0}


def test_split_refunds_shared_bets_once_and_starts_fresh():
    db.initialize_paper_portfolio(10_000.0, "unified-paper-v1")
    shared = PaperPortfolio()
    b = PMBroker()
    b.bind_portfolio(shared)                 # the pre-split state
    pos = b.open(_market(), 0, 0.40, 800.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pos is not None and shared.cash == pytest.approx(9_200.0)
    saved = {k: dict(v) for k, v in b.positions.items()}

    out = b.make_standalone(500.0, shared)
    assert out == {"migrated": True, "refunded": 800.0,
                   "dropped_positions": 1, "start_cash": 500.0}
    assert shared.cash == pytest.approx(10_000.0)     # main account made whole
    assert b.standalone and b._portfolio is None
    assert b.cash == 500.0 and b.positions == {} and b.realized_pnl == 0.0

    # second call: no-op
    assert b.make_standalone(500.0, shared) == {"migrated": False}
    # a crash before the flag was saved re-runs it: the refund must not repeat
    again = PMBroker()
    again.positions = saved
    again.make_standalone(500.0, shared)
    assert shared.cash == pytest.approx(10_000.0)


def test_split_without_shared_account_just_starts_fresh():
    b = PMBroker(start_cash=10_000.0)
    b.open(_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    out = b.make_standalone(500.0, PaperPortfolio())      # ledger not ready
    assert out["refunded"] == 0.0 and out["dropped_positions"] == 1
    assert b.cash == 500.0 and b.positions == {}


def test_own_bankroll_grows_and_never_touches_main_cash():
    db.initialize_paper_portfolio(10_000.0, "unified-paper-v1")
    shared = PaperPortfolio()
    b = PMBroker()
    b.make_standalone(500.0, shared)
    b.open(_market(), 0, 0.40, 100.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert b.cash == pytest.approx(400.0)
    t = b.resolve("tokA", won=True)                       # 250 shares pay $1
    assert t["pnl"] == pytest.approx(150.0)
    assert b.cash == pytest.approx(650.0)                 # it keeps its winnings
    assert shared.cash == 10_000.0                        # main account untouched


def test_bankroll_and_flag_survive_restart(monkeypatch, tmp_path):
    from app import persistence
    from app.markets.polymarket.broker import broker as pm
    monkeypatch.setattr(persistence, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(pm, "_portfolio", None)
    monkeypatch.setattr(pm, "standalone", True)
    monkeypatch.setattr(pm, "start_cash", 500.0)
    monkeypatch.setattr(pm, "_local_cash", 612.5)
    monkeypatch.setattr(pm, "positions", {})
    assert persistence.save() is True

    monkeypatch.setattr(pm, "standalone", False)
    monkeypatch.setattr(pm, "start_cash", 10_000.0)
    monkeypatch.setattr(pm, "_local_cash", 10_000.0)
    assert persistence.load() is True
    assert pm.standalone is True
    assert pm.start_cash == 500.0 and pm.cash == 612.5
    # already split: startup's call is a no-op and keeps the grown balance
    assert pm.make_standalone(500.0, PaperPortfolio())["migrated"] is False
    assert pm.cash == 612.5


def test_engine_reset_uses_pm_start_cash_setting(monkeypatch):
    from app import settings
    from app.markets.polymarket.engine import engine
    from app.markets.polymarket.broker import broker as pm
    monkeypatch.setattr(pm, "_portfolio", None)
    monkeypatch.setattr(pm, "positions", {})
    settings.update({"pm_start_cash": 750})
    try:
        out = engine.reset()
        assert out["start_cash"] == 750.0 and out["cash"] == 750.0
    finally:
        settings.update({"pm_start_cash": 500})
