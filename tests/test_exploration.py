"""Exploration sleeve: equal funding, trades once per bar, NAV unaffected by
money moving in/out, bench at -20% of capital, reinstatement on tracked
evidence, money follows 30-day results within 0.5x..2x, and Polymarket only
spends its own budget."""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

DAY = 86400


class Mkt:
    def __init__(self, px):
        self.px = dict(px)

    def price(self, p):
        return self.px.get(p)


@pytest.fixture()
def ex(tmp_path, monkeypatch):
    from app.strategies import exploration as E
    from app import settings
    real = settings.get
    vals = {"exploration_enabled": True, "exploration_allocation_pct": 50,
            "core_allocation_pct": 40, "exploration_bench_drawdown": 0.20,
            "exploration_reinstate_days": 30}
    monkeypatch.setattr(settings, "get", lambda k: vals[k] if k in vals else real(k))
    monkeypatch.setattr(E.db, "log_trade", lambda *a, **k: None)
    return E, E.Exploration(path=str(tmp_path / "explore.json"))


@pytest.fixture()
def broker():
    from app.execution.paper import PaperBroker
    b = PaperBroker()
    b.cash = 100_000.0
    return b


def test_funds_equal_slices_once(ex, broker):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0, "ETH-USD": 10.0})
    assert x.fund(100_000.0, m, broker) is True
    assert x.fund(100_000.0, m, broker) is False
    for name in E.MEMBERS:
        assert abs(x.member_equity(name, m, broker) - 10_000.0) < 1e-6
    assert abs(x.bot_equity(m, broker) - 10_000.0) < 1e-6


def test_engine_member_trades_once_per_bar_and_nav_tracks(ex, broker, monkeypatch):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0, "ETH-USD": 10.0})
    x.fund(100_000.0, m, broker)
    name = "majors_trend_daily_30d"
    monkeypatch.setattr(x, "target_weights", lambda n, now=None: ({"BTC-USD": 0.5, "ETH-USD": 0.5}, 1))
    acts = x.step(name, broker, m)
    assert len(acts) == 2
    assert x.step(name, broker, m) == []                  # same bar: no churn
    m.px["BTC-USD"] = 120.0                               # BTC +20%
    eq = x._mark(name, m, time.time(), broker)
    nav = x._m(name)["nav"]
    x._transfer(name, 5_000.0, m, broker)                 # money in...
    assert abs(x._m(name)["nav"] - nav) < 1e-9            # ...does not move NAV
    assert abs(x.member_equity(name, m, broker) - (eq + 5_000.0)) < 1e-6


def test_bench_at_minus_20_percent_and_money_withdrawn(ex, broker, monkeypatch):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0, "ETH-USD": 10.0})
    x.fund(100_000.0, m, broker)
    name = "majors_trend_hourly_5d"
    monkeypatch.setattr(x, "target_weights", lambda n, now=None: ({"BTC-USD": 1.0}, 1))
    x.step(name, broker, m)
    m.px["BTC-USD"] = 75.0                                # -25% on a fully invested book
    events = x.review(100_000.0, m, broker)
    assert any("BENCHED majors_trend_hourly_5d" in e for e in events)
    mm = x._m(name)
    assert mm["status"] == "benched" and not mm["positions"]
    assert abs(x.member_equity(name, m, broker)) < 1e-6   # its cash went back to the pool
    assert x.review(100_000.0, m, broker) == []           # once a day


def test_reinstated_only_with_positive_tracked_return(ex, broker, monkeypatch):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0, "ETH-USD": 10.0})
    x.fund(100_000.0, m, broker)
    name = "majors_trend_daily_30d"
    mm = x._m(name)
    mm["status"], mm["benched_at"] = "benched", time.time() - 31 * DAY
    monkeypatch.setattr(x, "_shadow_return", lambda n, since, now: -0.05)
    x.review(100_000.0, m, broker, now=time.time())
    assert mm["status"] == "benched"
    x.last_review_day = None
    monkeypatch.setattr(x, "_shadow_return", lambda n, since, now: 0.08)
    events = x.review(100_000.0, m, broker, now=time.time() + DAY)
    assert mm["status"] == "active" and any("REINSTATED" in e for e in events)
    assert x.member_equity(name, m, broker) > 0           # it got money back


def test_money_follows_30_day_results_within_limits(ex, broker):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0, "ETH-USD": 10.0})
    now = time.time()
    x.fund(100_000.0, m, broker, now=now - 40 * DAY)
    for name in E.MEMBERS:                                # fake 40 days of NAV history
        x._m(name)["history"] = [[now - 40 * DAY, 1.0], [now, 1.0]]
    x._m("majors_trend_daily_30d")["history"][-1][1] = 1.30   # +30% winner
    x._m("alt_momentum_daily")["history"][-1][1] = 0.85       # -15% loser
    x.review(100_000.0, m, broker, now=now)
    eq = {n: x.member_equity(n, m, broker) for n in E.MEMBERS}
    share = 50_000.0 / len(E.MEMBERS)
    assert eq["majors_trend_daily_30d"] > eq["majors_trend_hourly_20d"] > eq["alt_momentum_daily"]
    assert all(0.5 * share * 0.6 <= v <= 2 * share * 1.7 for v in eq.values())
    assert abs(sum(eq.values()) - 50_000.0) < 1e-6


def test_benched_hourly_bot_gets_no_sizing_equity(ex, broker):
    E, x = ex
    m = Mkt({"BTC-USD": 100.0})
    x.fund(100_000.0, m, broker)
    assert x.bot_equity(m, broker) > 0
    x._m("hourly_bot")["status"] = "benched"
    assert x.bot_equity(m, broker) == 0.0 and not x.is_active("hourly_bot")


def test_allocator_has_no_polymarket_share(ex):
    # Polymarket runs its own bankroll (pm_start_cash), not a slice of this one
    from app.strategies import allocator
    assert not hasattr(allocator, "polymarket_budget")
