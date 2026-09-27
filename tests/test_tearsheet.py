"""Tests for the honest performance tearsheet over real closed trades.

Pins the leakage-free semantics that make the tuning decisions trustworthy: net
figures on real trades, correct per-regime / per-exit-reason slicing, and — the
lesson from the Invo 500 — a JSON-safe payload even with degenerate samples.
"""
import json
import math
import time

from app.analytics.tearsheet import build_tearsheet


def _t(pnl, entry=100.0, qty=10.0, regime="trend", reason="take-profit",
       ago_h=1.0, hedge=False, fees=0.5):
    return {"pnl": pnl, "entry": entry, "qty": qty, "regime_at_entry": regime,
            "exit_reason": reason, "closed": time.time() - ago_h * 3600,
            "fees": fees, "hedge": hedge}


def _mixed():
    return [
        _t(150, regime="trend", reason="take-profit", ago_h=100),
        _t(-40, regime="range", reason="exit-advisor cut", ago_h=90),
        _t(-25, regime="range", reason="peak-giveback", ago_h=80),
        _t(200, entry=50, qty=20, regime="trend", reason="take-profit", ago_h=60),
        _t(-60, entry=50, qty=20, regime="range", reason="stop", ago_h=40),
        _t(30, entry=50, qty=20, regime="trend", reason="signal-flip", ago_h=20),
        _t(-15, entry=50, qty=20, regime="range", reason="exit-advisor cut", ago_h=10),
    ]


def test_empty_is_json_safe_and_zeroed():
    ts = build_tearsheet([], start_cash=100_000)
    assert ts["n_trades"] == 0
    assert ts["overall"]["n"] == 0
    json.dumps(ts, allow_nan=False)  # must not raise


def test_single_winner_profit_factor_inf_is_neutralized():
    # one win, zero losses -> profit_factor is inf -> must be neutralized to None
    ts = build_tearsheet([_t(50)], start_cash=100_000)
    json.dumps(ts, allow_nan=False)  # the Invo-500 lesson: no non-finite floats
    assert ts["overall"]["profit_factor"] is None
    assert ts["overall"]["n"] == 1


def test_overall_net_pnl_is_sum_of_real_pnls():
    trades = _mixed()
    ts = build_tearsheet(trades, start_cash=100_000)
    assert ts["overall"]["net_pnl"] == sum(t["pnl"] for t in trades)
    assert ts["overall"]["n"] == len(trades)
    # 3 wins of 7 -> win_rate 3/7 (reported rounded to 4dp)
    assert abs(ts["overall"]["win_rate"] - 3 / 7) < 1e-3


def test_per_trade_return_is_pnl_over_entry_notional():
    # pnl 150 on 100*10 = 1000 notional -> 0.15 expectancy
    ts = build_tearsheet([_t(150, entry=100, qty=10)], start_cash=100_000)
    assert abs(ts["overall"]["expectancy_pct"] - 0.15) < 1e-9


def test_regime_slicing_isolates_the_bleed():
    ts = build_tearsheet(_mixed(), start_cash=100_000)
    reg = ts["by_regime"]
    assert set(reg) == {"trend", "range"}
    assert reg["range"]["net_pnl"] < 0          # range is losing here
    assert reg["trend"]["net_pnl"] > 0          # trend is winning
    # groups are ordered worst-net-pnl first
    assert list(reg.keys())[0] == "range"


def test_exit_reason_slicing_flags_aggressive_exits():
    ts = build_tearsheet(_mixed(), start_cash=100_000)
    ex = ts["by_exit_reason"]
    assert "exit-advisor cut" in ex
    assert ex["exit-advisor cut"]["n"] == 2
    assert ex["exit-advisor cut"]["net_pnl"] == -55
    assert ex["take-profit"]["net_pnl"] > 0


def test_hedge_legs_can_be_excluded_and_are_counted():
    trades = _mixed() + [_t(5, regime="trend", reason="take-profit", hedge=True)]
    incl = build_tearsheet(trades, start_cash=100_000, exclude_hedge=False)
    excl = build_tearsheet(trades, start_cash=100_000, exclude_hedge=True)
    assert incl["n_trades"] == len(trades)
    assert excl["n_trades"] == len(trades) - 1
    assert excl["n_hedge_legs"] == 1


def test_missing_entry_notional_is_skipped_in_returns_not_in_pnl():
    # a trade with no entry/qty still counts toward net pnl but not the return series
    trades = [_t(100, entry=100, qty=10), {"pnl": 20, "exit_reason": "x",
                                           "regime_at_entry": "trend", "closed": time.time()}]
    ts = build_tearsheet(trades, start_cash=100_000)
    assert ts["overall"]["net_pnl"] == 120
    # expectancy is mean of the ONE reconstructable return (0.10), not diluted
    assert abs(ts["overall"]["expectancy_pct"] - 0.10) < 1e-9


def test_drawdown_and_calmar_present_on_a_drawdown_path():
    # win then big loss then recover -> a real drawdown must register
    trades = [_t(500, ago_h=30), _t(-800, ago_h=20), _t(600, ago_h=10)]
    ts = build_tearsheet(trades, start_cash=100_000)
    assert ts["overall"]["max_drawdown"] > 0
    json.dumps(ts, allow_nan=False)
