"""Slow daily rules: configurable core average, UNIVERSE core, benchmark."""
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _hourly(days=260, drift=0.0, seed=0, t0=1_700_000_000 // 86400 * 86400):
    r = random.Random(seed)
    px, out = 100.0, []
    for i in range(days * 24):
        op = px
        px *= math.exp(drift + r.gauss(0, 0.004))
        out.append([t0 + i * 3600, min(op, px), max(op, px), op, px, 1.0])
    return out


def test_benchmark_trend_rule_sits_out_a_downtrend():
    from app.backtest.replay import slow_benchmarks
    U = {"BTC-USD": _hourly(drift=-0.0004, seed=1), "ETH-USD": _hourly(drift=-0.0004, seed=2),
         "SOL-USD": _hourly(drift=-0.0004, seed=3)}
    b = slow_benchmarks(U)
    assert b["buy_hold_equal_weight"]["return_pct"] < -10
    trend = b["per_coin_trend_100d"]["return_pct"]
    assert trend > b["buy_hold_equal_weight"]["return_pct"]      # cash beats a falling market
    core_key = next(k for k in b if k.startswith("btc_eth_core_"))
    assert {"first_half_pct", "second_half_pct", "max_drawdown_pct"} <= set(b[core_key])


def test_benchmark_needs_enough_history():
    from app.backtest.replay import slow_benchmarks
    U = {p: _hourly(days=60, seed=i) for i, p in enumerate(("BTC-USD", "ETH-USD", "SOL-USD"))}
    assert slow_benchmarks(U) is None


def _settings(monkeypatch, **vals):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: vals[k] if k in vals else real(k))


def _daily(n, rising):
    t0 = int(time.time() // 86400 - n - 1) * 86400
    return [[t0 + i * 86400, 0, 0, 0, 100 + (i if rising else -i) * 0.5, 1] for i in range(n)]


def test_core_sma_days_and_universe(monkeypatch):
    from app.strategies.core import CoreBook
    monkeypatch.setattr("app.config.PRODUCTS", ["BTC-USD", "ETH-USD", "SOL-USD"])
    _settings(monkeypatch, core_allocation_pct=60, core_assets="UNIVERSE",
              core_trend_filter=True, core_sma_days=100,
              core_selection="trend", core_sizing="equal",    # the original rule
              core_strategy="settings", core_hysteresis=0.0)
    c = CoreBook()
    _, assets, _ = c._settings()
    assert assets == ["BTC-USD", "ETH-USD", "SOL-USD"]
    assert c.sma_days() == 100
    # 80 days of history < 100-day average -> unknown -> stay in cash
    t = c.targets(100_000, {a: _daily(80, True) for a in assets})
    assert all(v == 0 for v in t.values())
    t = c.targets(100_000, {"BTC-USD": _daily(150, True), "ETH-USD": _daily(150, False),
                            "SOL-USD": _daily(150, True)})
    assert t["BTC-USD"] == t["SOL-USD"] == 20_000 and t["ETH-USD"] == 0


def test_bot_never_sizes_against_the_cores_idle_cash(monkeypatch):
    from app.strategies.core import CoreBook

    class M:
        def price(self, p):
            return 100.0
    c = CoreBook()
    _settings(monkeypatch, core_allocation_pct=80, core_assets="BTC-USD,ETH-USD")
    assert abs(c.bot_equity(100_000.0, M()) - 20_000.0) < 1e-6           # nothing held yet: still reserved
    c.positions = {"BTC-USD": {"qty": 400.0, "entry": 100.0, "opened": 0}}
    assert abs(c.bot_equity(100_000.0, M()) - 20_000.0) < 1e-6
    _settings(monkeypatch, core_allocation_pct=0, core_assets="BTC-USD,ETH-USD")
    assert abs(c.bot_equity(100_000.0, M()) - 60_000.0) < 1e-6           # off: account minus holdings
