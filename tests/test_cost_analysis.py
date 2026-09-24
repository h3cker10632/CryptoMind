"""Cost decomposition on run_composite must be EXACT dollars, not estimates.

We charge fees on every fill and slippage on every entry/exit; the two forced
end-of-run closes use the raw close (no slippage). These tests pin that the
accumulators reconcile against the fee/slippage tunables and that the gross
reconstruction is arithmetically consistent with net + cost drag.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import composite
from app import tunables


def _make_candles(n, seed=0):
    rng = random.Random(seed)
    px, t, out = 100.0, 0, []
    for _ in range(n):
        t += 3600
        op = px
        px *= (1 + rng.gauss(0.0003, 0.012))
        cl = px
        hi = max(op, cl) * (1 + abs(rng.gauss(0, 0.004)))
        lo = min(op, cl) * (1 - abs(rng.gauss(0, 0.004)))
        out.append([t, lo, hi, op, cl, rng.uniform(80, 300)])
    return out


def test_cost_analysis_present_and_shaped():
    r = composite.run_composite(_make_candles(500, seed=11))
    ca = r["cost_analysis"]
    for k in ("fees_paid", "slippage_paid", "total_costs", "fees_pct",
              "slippage_pct", "total_cost_pct", "turnover",
              "gross_return_approx", "net_return"):
        assert k in ca, f"missing cost_analysis key: {k}"


def test_costs_positive_when_trading():
    r = composite.run_composite(_make_candles(500, seed=11))
    ca = r["cost_analysis"]
    assert r["n_trades"] > 0
    assert ca["fees_paid"] > 0
    assert ca["total_costs"] >= ca["fees_paid"]  # slippage is non-negative
    assert ca["turnover"] > 0


def test_internal_arithmetic_consistency():
    r = composite.run_composite(_make_candles(500, seed=11), start_cash=10_000.0)
    ca = r["cost_analysis"]
    assert abs(ca["total_costs"] - (ca["fees_paid"] + ca["slippage_paid"])) < 0.01
    assert abs(ca["fees_pct"] - ca["fees_paid"] / 10_000.0) < 1e-4
    assert abs(ca["gross_return_approx"] - (ca["net_return"] + ca["total_cost_pct"])) < 1e-4
    assert ca["net_return"] == r["total_return"]


def test_zero_slippage_zeroes_slippage_only():
    try:
        tunables.update({"slippage_bps": 0})
        r = composite.run_composite(_make_candles(500, seed=11))
        ca = r["cost_analysis"]
        assert ca["slippage_paid"] == 0.0
        assert ca["fees_paid"] > 0.0
    finally:
        tunables.reset()


def test_zero_costs_make_gross_equal_net():
    try:
        tunables.update({"slippage_bps": 0, "fee_rate": 0.0})
        r = composite.run_composite(_make_candles(500, seed=11))
        ca = r["cost_analysis"]
        assert ca["total_costs"] == 0.0
        assert ca["gross_return_approx"] == ca["net_return"]
    finally:
        tunables.reset()
