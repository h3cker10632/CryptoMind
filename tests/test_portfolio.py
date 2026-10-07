"""Shared paper-capital ledger tests (Task 1: Durable Shared Cash Ledger)."""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import db
from app.portfolio import PaperPortfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_tables():
    db.reset_paper_portfolio_for_tests()
    yield


def test_ledger_account_created_once_with_exact_opening_cash():
    assert db.initialize_paper_portfolio(12_345.67, "unified-paper-v1") is True
    account = db.paper_portfolio_account()
    assert account["cash"] == 12_345.67
    assert account["opening_cash"] == 12_345.67
    assert account["migration_id"] == "unified-paper-v1"


def test_ledger_duplicate_initialize_is_noop():
    assert db.initialize_paper_portfolio(100_000.0, "unified-paper-v1") is True
    assert db.initialize_paper_portfolio(999_999.0, "unified-paper-v1") is False
    assert db.paper_portfolio_account()["cash"] == 100_000.0


def test_ledger_reserve_rejects_insufficient_cash():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    assert db.reserve_paper_cash("evt-1", "crypto", 150.0, "open BTC-USD") is False
    assert db.paper_portfolio_account()["cash"] == 100.0


def test_ledger_reserve_rejects_non_positive_amount():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    assert db.reserve_paper_cash("evt-1", "crypto", 0.0, "r") is False
    assert db.reserve_paper_cash("evt-2", "crypto", -10.0, "r") is False
    assert db.paper_portfolio_account()["cash"] == 100.0


def test_ledger_duplicate_event_id_is_noop():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    assert db.reserve_paper_cash("evt-dup", "crypto", 40.0, "open") is True
    assert db.reserve_paper_cash("evt-dup", "crypto", 40.0, "open-again") is False
    assert db.paper_portfolio_account()["cash"] == 60.0


def test_ledger_apply_event_idempotent():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    assert db.apply_paper_cash_event("evt-close-1", "crypto", 25.5, "close BTC-USD") is True
    assert db.paper_portfolio_account()["cash"] == 125.5
    assert db.apply_paper_cash_event("evt-close-1", "crypto", 25.5, "close BTC-USD") is False
    assert db.paper_portfolio_account()["cash"] == 125.5


def test_ledger_concurrent_reservations_cannot_overspend():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    barrier = threading.Barrier(3)
    results = []

    def reserve(event_id):
        barrier.wait()
        results.append(portfolio.reserve("crypto", event_id, 60.0, "concurrent-open"))

    workers = [threading.Thread(target=reserve, args=(f"evt-{i}",)) for i in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()

    assert sorted(results) == [False, True]
    assert portfolio.cash == 40.0


def test_portfolio_wrapper_rejects_unknown_sleeve():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    with pytest.raises(ValueError):
        portfolio.reserve("stocks", "evt-x", 10.0, "r")


def test_portfolio_wrapper_cash_reads_through_ledger():
    portfolio = PaperPortfolio()
    assert portfolio.cash == 0.0
    assert portfolio.ready is False
    db.initialize_paper_portfolio(500.0, "unified-paper-v1")
    assert portfolio.ready is True
    assert portfolio.cash == 500.0


# ---------------- Task 2: migration gate ----------------

def test_migration_unconfirmed_leaves_trading_disabled():
    from app.portfolio import bootstrap, portfolio

    assert bootstrap(87_654.32, legacy_pm_confirmed=False) is False
    assert portfolio.ready is False
    assert db.paper_portfolio_account() is None


def test_migration_confirmed_uses_restored_crypto_cash_exactly_once():
    from app.portfolio import bootstrap, portfolio

    assert bootstrap(87_654.32, legacy_pm_confirmed=True) is True
    account = db.paper_portfolio_account()
    assert account["cash"] == 87_654.32
    assert account["opening_cash"] == 87_654.32
    assert portfolio.cash == 87_654.32          # no +$10,000 legacy PM seed


def test_migration_duplicate_startup_does_not_reapply_opening_cash():
    from app.portfolio import bootstrap, portfolio

    assert bootstrap(1_000.0, legacy_pm_confirmed=True) is True
    # a later restart calls bootstrap again (legacy_pm_confirmed defaults to
    # False on an unattended restart) — must be a strict no-op either way
    assert bootstrap(5_000.0, legacy_pm_confirmed=False) is False
    assert bootstrap(5_000.0, legacy_pm_confirmed=True) is False
    assert portfolio.cash == 1_000.0


def test_migration_bootstrap_does_not_touch_risk_kill_halt_state():
    from app.portfolio import bootstrap
    from app.risk.manager import RiskManager

    fresh_risk = RiskManager()
    before = (fresh_risk.killed, fresh_risk.kill_reason, fresh_risk.halted_today,
             fresh_risk.halt_reason)
    bootstrap(2_000.0, legacy_pm_confirmed=True)
    after = (fresh_risk.killed, fresh_risk.kill_reason, fresh_risk.halted_today,
            fresh_risk.halt_reason)
    assert before == after


def test_portfolio_migration_prefix_is_authenticated():
    from app import security

    assert any(p.startswith("/api/portfolio/migration")
              for p in security._PROTECTED_PREFIXES)


def test_entries_enabled_requires_running_and_portfolio_ready(monkeypatch):
    from app.orchestrator import Orchestrator

    from app import settings
    real = settings.get                          # isolate from the hourly-bot gate
    monkeypatch.setattr(settings, "get", lambda k: "on" if k == "hourly_bot_mode" else real(k))
    orch = Orchestrator()
    orch.running = True
    monkeypatch.setattr(db, "paper_portfolio_account", lambda: None)
    assert orch._entries_enabled() is False

    monkeypatch.setattr(db, "paper_portfolio_account",
                        lambda: {"cash": 1_000.0, "opening_cash": 1_000.0,
                                 "migration_id": "unified-paper-v1"})
    assert orch._entries_enabled() is True

    orch.running = False
    assert orch._entries_enabled() is False


def test_tick_gates_new_entries_on_entries_enabled():
    """Static regression guard: the entries section must gate on
    `_entries_enabled()`, not bare `self.running`, so new orders stay paused
    until the shared-portfolio migration is confirmed."""
    import inspect
    import app.orchestrator as orch_mod

    src = inspect.getsource(orch_mod.Orchestrator.tick)
    assert "if self._entries_enabled() and not self._chop_blocks_entries():" in src
    assert "if self.running" not in src


# ---------------- Task 3: crypto routed through shared cash ----------------

def test_crypto_open_reserves_exact_notional_plus_fee():
    from app.execution.paper import PaperBroker
    from app.tunables import tv

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    broker = PaperBroker()
    broker.bind_portfolio(portfolio)

    notional = 10_000.0
    fee_rate = tv("fee_rate")
    pos = broker.open("BTC-USD", 1, notional, 100.0, 90.0, 130.0, "t")
    assert pos is not None
    assert pos["fees"] == pytest.approx(notional * fee_rate)
    assert portfolio.cash == pytest.approx(
        100_000.0 - notional - notional * fee_rate)


def test_crypto_long_close_settles_exact_proceeds():
    from app.execution.paper import PaperBroker
    from app.tunables import tv

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    broker = PaperBroker()
    broker.bind_portfolio(portfolio)
    fee_rate = tv("fee_rate")

    pos = broker.open("BTC-USD", 1, 10_000.0, 100.0, 90.0, 130.0, "t")
    assert pos is not None
    cash_after_open = portfolio.cash

    trade = broker.sell("BTC-USD", 120.0, "tp")
    assert trade is not None
    gross = trade["qty"] * trade["exit"]
    fee_close = gross * fee_rate
    assert portfolio.cash == pytest.approx(cash_after_open + gross - fee_close)
    assert portfolio.cash == pytest.approx(100_000.0 + broker.realized_pnl,
                                           abs=0.05)


def test_crypto_short_close_settles_exact_margin_and_move():
    from app.execution.paper import PaperBroker
    from app.tunables import tv

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    broker = PaperBroker()
    broker.bind_portfolio(portfolio)
    fee_rate = tv("fee_rate")

    pos = broker.open("BTC-USD", -1, 10_000.0, 100.0, 110.0, 70.0, "t")
    assert pos is not None
    cash_after_open = portfolio.cash

    trade = broker.sell("BTC-USD", 80.0, "tp")
    assert trade is not None
    fee_close = trade["qty"] * trade["exit"] * fee_rate
    move = trade["qty"] * (pos["entry"] - trade["exit"])
    assert portfolio.cash == pytest.approx(
        cash_after_open + pos["margin"] + move - fee_close)


def test_crypto_funding_accrues_once_per_event_id(monkeypatch):
    from app.execution.paper import PaperBroker
    import app.data.derivatives as deriv_mod
    import time as time_mod

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    broker = PaperBroker()
    broker.bind_portfolio(portfolio)

    monkeypatch.setattr(deriv_mod.derivatives, "features",
                        lambda product: {"funding_rate": 0.01})

    pos = broker.open("BTC-USD", -1, 10_000.0, 100.0, 110.0, 70.0, "t")
    assert pos is not None
    pos["_last_funding"] = 1_000.0

    # keep int(now) pinned at 1000 across both calls (and any internal db
    # timestamp reads) so both funding attempts share one event id
    counter = {"n": 0}

    def fake_time():
        counter["n"] += 1
        return 1_000.0 + counter["n"] * 0.01

    monkeypatch.setattr(time_mod, "time", fake_time)

    cash_before = portfolio.cash
    broker._accrue_funding("BTC-USD", pos, 100.0)
    cash_after_first = portfolio.cash
    assert cash_after_first != cash_before

    broker._accrue_funding("BTC-USD", pos, 100.0)
    assert portfolio.cash == cash_after_first     # duplicate event -> no-op


def test_bound_broker_cash_is_read_only():
    from app.execution.paper import PaperBroker

    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    broker = PaperBroker()
    broker.bind_portfolio(portfolio)
    with pytest.raises(RuntimeError):
        broker.cash = 999.0


def test_standalone_broker_cash_unaffected_by_binding_elsewhere():
    """An un-bound PaperBroker() (existing unit-test style) keeps working
    exactly like before -- direct cash assignment, no ledger involved."""
    from app.execution.paper import PaperBroker

    b = PaperBroker()
    b.cash = 100_000.0
    pos = b.open("BTC-USD", 1, 10_000.0, 100.0, 90.0, 130.0, "t")
    assert pos is not None
    assert b.cash < 100_000.0


def test_portfolio_reset_account_is_explicit_and_idempotent():
    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    assert portfolio.reset_account(50_000.0, "reset-evt-1") is True
    assert portfolio.cash == 50_000.0
    assert portfolio.reset_account(999.0, "reset-evt-1") is False
    assert portfolio.cash == 50_000.0


def test_reset_account_endpoint_refuses_with_open_crypto_positions(monkeypatch):
    from app import main

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1")
    monkeypatch.setattr(main.broker, "_portfolio", PaperPortfolio())
    monkeypatch.setattr(main.db, "log_event", lambda *a, **kw: None)
    monkeypatch.setattr(main.persistence, "save", lambda: True)
    monkeypatch.setattr(main.broker, "positions", {"BTC-USD": {"qty": 1}})

    result = main.reset_account()
    assert result["ok"] is False
    assert PaperPortfolio().cash == 100_000.0


def test_reset_account_endpoint_resets_shared_cash_once_flat(monkeypatch, tmp_path):
    from app import main
    from app.config import START_CASH
    from app.markets.polymarket.broker import broker as pm_broker
    from app.strategies.core import core

    db.initialize_paper_portfolio(5_000.0, "unified-paper-v1")
    monkeypatch.setattr(main.broker, "_portfolio", PaperPortfolio())
    monkeypatch.setattr(main.broker, "positions", {})
    monkeypatch.setattr(main.db, "log_event", lambda *a, **kw: None)
    monkeypatch.setattr(main.persistence, "save", lambda: True)
    # NEVER touch the real reports/ folder from a test
    monkeypatch.setattr(main.orch, "_replay_path", lambda name: str(tmp_path / name))
    from app.strategies.exploration import manager as explore
    monkeypatch.setattr(explore, "path", str(tmp_path / "exploration_state.json"))
    (tmp_path / "forward_test_baseline.json").write_text("{}")
    monkeypatch.setattr(core, "positions", {"BTC-USD": {"qty": 1.0, "entry": 1.0, "opened": 0}})

    # Polymarket runs its own bankroll: its open bets neither block nor are
    # touched by a main-account reset
    monkeypatch.setattr(pm_broker, "positions", {"tok": {"shares": 1.0}})
    result = main.reset_account()
    assert result["reset"] is True
    assert pm_broker.positions == {"tok": {"shares": 1.0}}
    assert main.broker.cash == START_CASH
    assert core.positions == {}                       # the core book is part of the account
    assert not (tmp_path / "forward_test_baseline.json").exists()   # fresh forward test


# ---------------- Task 4: Polymarket routed through shared cash ----------------

def _pm_market(condition_id="pm-cond-1"):
    return {"condition_id": condition_id, "token_ids": ["tokA", "tokB"],
            "outcomes": ["A", "B"], "question": "Q?", "category": "test",
            "tick_size": 0.01, "min_order_size": 5.0}


def test_crypto_and_polymarket_share_one_cash_pool():
    from app.execution.paper import PaperBroker
    from app.markets.polymarket.broker import PMBroker

    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    crypto_broker = PaperBroker()
    crypto_broker.bind_portfolio(portfolio)
    pm_broker = PMBroker()
    pm_broker.bind_portfolio(portfolio)

    crypto_pos = crypto_broker.open("BTC-USD", 1, 60.0, 100.0, 90.0, 130.0, "t")
    assert crypto_pos is not None
    remaining = portfolio.cash
    assert remaining < 100.0

    market = _pm_market()
    too_big = pm_broker.open(market, 0, 0.40, remaining + 1.0, 0.0, 0.0,
                             0.1, 0.9, "t")
    assert too_big is None            # not enough SHARED cash left

    fits = pm_broker.open(market, 0, 0.40, 5.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert fits is not None
    assert portfolio.cash < remaining


def test_crypto_and_polymarket_concurrent_opens_cannot_overspend():
    from app.execution.paper import PaperBroker
    from app.markets.polymarket.broker import PMBroker

    db.initialize_paper_portfolio(100.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    crypto_broker = PaperBroker()
    crypto_broker.bind_portfolio(portfolio)
    pm_broker = PMBroker()
    pm_broker.bind_portfolio(portfolio)
    market = _pm_market()

    barrier = threading.Barrier(3)
    results = []

    def open_crypto():
        barrier.wait()
        results.append(crypto_broker.open(
            "BTC-USD", 1, 60.0, 100.0, 90.0, 130.0, "t"))

    def open_pm():
        barrier.wait()
        results.append(pm_broker.open(
            market, 0, 0.40, 60.0, 0.0, 0.0, 0.1, 0.9, "t"))

    t1 = threading.Thread(target=open_crypto)
    t2 = threading.Thread(target=open_pm)
    t1.start()
    t2.start()
    barrier.wait()
    t1.join()
    t2.join()

    successes = [r for r in results if r is not None]
    assert len(successes) == 1
    assert portfolio.cash >= 0


def test_pm_resolution_settles_shared_cash_exactly_once_per_position():
    from app.markets.polymarket.broker import PMBroker

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    pm_broker = PMBroker()
    pm_broker.bind_portfolio(portfolio)
    market = _pm_market()

    pos = pm_broker.open(market, 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pos is not None
    cash_after_open = portfolio.cash

    trade = pm_broker.resolve(pos["token_id"], True)
    assert trade is not None
    cash_after_resolve = portfolio.cash
    assert cash_after_resolve > cash_after_open

    # replay against the same (already-settled) position -> no double credit
    pm_broker._finish(pos, 1.0, pos["shares"] * 1.0, "resolution-replay")
    assert portfolio.cash == cash_after_resolve


def test_pm_sleeve_reset_never_touches_shared_cash():
    from app.markets.polymarket.broker import PMBroker

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    pm_broker = PMBroker()
    pm_broker.bind_portfolio(portfolio)

    market = _pm_market()
    pos = pm_broker.open(market, 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pos is not None
    cash_before_reset = portfolio.cash

    pm_broker.reset(10_000.0)
    assert portfolio.cash == cash_before_reset   # untouched -- sleeve-local only
    assert pm_broker.positions == {}


# ---------------- Task 5: aggregate equity / exposure ----------------

class _CryptoMkt:
    def __init__(self, price):
        self._price = price

    def price(self, product):
        return self._price


def test_total_equity_counts_shared_cash_once_plus_crypto_not_polymarket(monkeypatch):
    from app.execution.paper import broker as crypto_broker
    from app.markets.polymarket.broker import broker as pm_broker

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})

    crypto_pos = crypto_broker.open("BTC-USD", 1, 100.0, 100.0, 90.0, 130.0, "t")
    assert crypto_pos is not None
    pm_pos = pm_broker.open(_pm_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pm_pos is not None

    eq = portfolio.total_equity(_CryptoMkt(110.0))
    expected = portfolio.cash + crypto_broker.position_value(crypto_pos, 110.0)
    assert eq == pytest.approx(expected)          # Polymarket's bet not counted


def test_total_equity_short_position_and_missing_price_fallback(monkeypatch):
    from app.execution.paper import broker as crypto_broker
    from app.markets.polymarket.broker import broker as pm_broker

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})

    short_pos = crypto_broker.open("BTC-USD", -1, 100.0, 100.0, 110.0, 70.0, "t")
    assert short_pos is not None
    pm_pos = pm_broker.open(_pm_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pm_pos is not None

    # no live price for either sleeve -> falls back to entry price
    eq = portfolio.total_equity(_CryptoMkt(None))
    expected = portfolio.cash + crypto_broker.position_value(short_pos, short_pos["entry"])
    assert eq == pytest.approx(expected)


def test_total_exposure_is_crypto_only_and_excludes_shadow(monkeypatch):
    from app.execution.paper import broker as crypto_broker
    from app.markets.polymarket.broker import broker as pm_broker

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})

    crypto_pos = crypto_broker.open("BTC-USD", 1, 100.0, 100.0, 90.0, 130.0, "t")
    assert crypto_pos is not None
    pm_pos = pm_broker.open(_pm_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pm_pos is not None

    exposure = portfolio.total_exposure(_CryptoMkt(110.0))
    expected = crypto_pos["qty"] * 110.0
    assert exposure == pytest.approx(expected)

    # static regression guard: aggregate methods must never import/reference
    # the shadow OMS mirror module (the docstring may still explain why not)
    import inspect
    from app.portfolio import PaperPortfolio as _PP
    assert "execution.shadow" not in inspect.getsource(_PP.total_equity)
    assert "execution.shadow" not in inspect.getsource(_PP.total_exposure)
    assert "ShadowBroker" not in inspect.getsource(_PP.total_equity)
    assert "ShadowBroker" not in inspect.getsource(_PP.total_exposure)


def test_orchestrator_equity_helpers_fall_back_to_crypto_only_before_migration():
    from app.orchestrator import Orchestrator
    from app.execution.paper import broker as crypto_broker

    orch = Orchestrator()
    assert crypto_broker._portfolio is None     # sanity: unbound by default
    mkt = _CryptoMkt(100.0)
    assert orch._total_equity(mkt) == crypto_broker.equity(mkt)
    assert orch._total_exposure(mkt) == crypto_broker.exposure(mkt)


def test_orchestrator_equity_helpers_aggregate_once_portfolio_ready(monkeypatch):
    from app.orchestrator import Orchestrator
    from app.execution.paper import broker as crypto_broker
    from app.markets.polymarket.broker import broker as pm_broker
    from app.markets.polymarket.engine import engine as pm_engine

    db.initialize_paper_portfolio(1_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})
    monkeypatch.setattr(pm_engine, "_mid_lookup", lambda token_id: 0.5)

    crypto_pos = crypto_broker.open("BTC-USD", 1, 100.0, 100.0, 90.0, 130.0, "t")
    assert crypto_pos is not None

    orch = Orchestrator()
    mkt = _CryptoMkt(110.0)
    expected_eq = portfolio.total_equity(mkt)
    expected_exp = portfolio.total_exposure(mkt)
    assert orch._total_equity(mkt) == pytest.approx(expected_eq)
    assert orch._total_exposure(mkt) == pytest.approx(expected_exp)


# ---------------- Task 7: whole-account safety regression ----------------

def test_migration_refused_applies_no_capital_event_and_leaves_trading_off():
    from app.portfolio import bootstrap

    assert bootstrap(50_000.0, legacy_pm_confirmed=False) is False
    assert db.paper_portfolio_account() is None
    portfolio = PaperPortfolio()
    assert portfolio.ready is False
    # no capital event could possibly exist: there is no account row yet
    assert portfolio.reserve("crypto", "evt-should-fail", 1.0, "r") is False


def test_restart_round_trip_preserves_shared_cash_positions_and_kill_state(
        monkeypatch, tmp_path):
    from app import persistence, settings as app_settings
    from app.execution.paper import broker as crypto_broker
    from app.markets.polymarket.broker import broker as pm_broker
    from app.risk.manager import risk

    monkeypatch.setattr(persistence, "STATE_PATH", str(tmp_path / "state.json"))
    app_settings.update({"carry_equity": True})

    db.initialize_paper_portfolio(10_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})
    monkeypatch.setattr(crypto_broker, "closed_trades", [])
    monkeypatch.setattr(pm_broker, "closed_trades", [])

    crypto_pos = crypto_broker.open("BTC-USD", 1, 1_000.0, 100.0, 90.0, 130.0, "t")
    assert crypto_pos is not None
    pm_pos = pm_broker.open(_pm_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pm_pos is not None

    risk.killed = True
    risk.kill_reason = "test kill before restart"
    cash_before_restart = portfolio.cash

    assert persistence.save() is True

    # --- simulate a real process restart: fresh/unbound brokers, in-memory
    # positions and risk flags wiped -- but the DB-backed ledger survives. ---
    monkeypatch.setattr(crypto_broker, "_portfolio", None)
    monkeypatch.setattr(pm_broker, "_portfolio", None)
    monkeypatch.setattr(crypto_broker, "positions", {})
    monkeypatch.setattr(pm_broker, "positions", {})
    risk.killed = False
    risk.kill_reason = ""

    assert persistence.load() is True
    # main.py's startup() re-binds immediately after load() once migrated
    monkeypatch.setattr(crypto_broker, "_portfolio", portfolio)
    monkeypatch.setattr(pm_broker, "_portfolio", portfolio)

    assert portfolio.cash == cash_before_restart
    assert risk.killed is True
    assert "test kill before restart" in risk.kill_reason
    assert "BTC-USD" in crypto_broker.positions
    assert pm_pos["token_id"] in pm_broker.positions
    # restored positions keep their ledger IDs (exactly-once close/settle)
    assert crypto_broker.positions["BTC-USD"].get("_ledger_id") == crypto_pos["_ledger_id"]


def test_pm_standalone_local_cash_survives_restart_and_kill(monkeypatch, tmp_path):
    """Pre-migration (unbound) Polymarket cash must not silently reset to
    PM_START_CASH on a restart/kill -- it has to round-trip through state.json
    exactly like positions/closed_trades/realized_pnl already do."""
    from app import persistence
    import importlib
    pm_broker_mod = importlib.import_module("app.markets.polymarket.broker")
    from app.markets.polymarket.broker import PMBroker

    monkeypatch.setattr(persistence, "STATE_PATH", str(tmp_path / "state.json"))

    pm_broker = PMBroker()
    monkeypatch.setattr(pm_broker_mod, "broker", pm_broker)
    pos = pm_broker.open(_pm_market(), 0, 0.40, 50.0, 0.0, 0.0, 0.1, 0.9, "t")
    assert pos is not None
    cash_before = pm_broker.cash
    assert cash_before != pm_broker.start_cash        # genuinely moved

    snap = persistence._capture_pm_broker()
    assert snap["local_cash"] == cash_before

    # simulate a restart: a brand-new, unbound broker (fresh PM_START_CASH)
    fresh_broker = PMBroker()
    monkeypatch.setattr(pm_broker_mod, "broker", fresh_broker)
    assert fresh_broker.cash != cash_before

    persistence._restore_pm_broker(snap)
    assert fresh_broker.cash == cash_before
    assert pos["token_id"] in fresh_broker.positions


def test_pm_bound_broker_ignores_restored_local_cash(monkeypatch, tmp_path):
    """Once bound to the shared ledger, a restored local_cash must NOT
    override the durable DB-backed cash (it's read-only once bound)."""
    from app import persistence
    import importlib
    pm_broker_mod = importlib.import_module("app.markets.polymarket.broker")
    from app.markets.polymarket.broker import PMBroker

    db.initialize_paper_portfolio(5_000.0, "unified-paper-v1")
    portfolio = PaperPortfolio()
    bound_broker = PMBroker()
    bound_broker.bind_portfolio(portfolio)
    monkeypatch.setattr(pm_broker_mod, "broker", bound_broker)

    persistence._restore_pm_broker({"version": 1, "local_cash": 999.0,
                                    "positions": {}, "closed_trades": [],
                                    "realized_pnl": 0.0})
    assert bound_broker.cash == portfolio.cash
    assert bound_broker._local_cash != 999.0

