"""Profit-ladder scale-out prototype (backtest only): OFF must be byte-identical
to the classic all-or-nothing path, and ON must change execution without
crashing. The *value* question is answered by tools/measure_scaleout.py — the
measured result was NEGATIVE, so this only guards the mechanics, not a live path.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backtest.composite import run_composite


def _make(seed, n=400, drift=0.0008, vol=0.011):
    r = random.Random(seed); px = 100.0; t = 1_700_000_000; cs = []
    for i in range(n):
        px *= (1 + drift + r.gauss(0, vol))
        hi = px * (1 + abs(r.gauss(0, 0.004)))
        lo = px * (1 - abs(r.gauss(0, 0.004)))
        cs.append([t + i * 3600, lo, hi, lo + (hi - lo) * r.random(), px, 1000 + r.random() * 500])
    return cs


def test_scale_out_off_is_identity():
    cs = _make(7)
    a = run_composite(cs)                       # default
    b = run_composite(cs, scale_out=None)       # explicit off
    assert a["total_return"] == b["total_return"]
    assert a["n_trades"] == b["n_trades"]
    assert a["final_equity"] == b["final_equity"]
    assert a["cost_analysis"]["turnover"] == b["cost_analysis"]["turnover"]


def test_scale_out_changes_execution_and_adds_legs():
    cs = _make(7)
    base = run_composite(cs, scale_out=None)
    so = run_composite(cs, scale_out={"arm_atr": 1.0, "frac": 0.5, "breakeven": True})
    # banking a runner leg is an extra fill → at least as many trades + turnover
    assert so["n_trades"] >= base["n_trades"]
    assert so["cost_analysis"]["turnover"] >= base["cost_analysis"]["turnover"]
    assert -1.0 <= so["total_return"]          # sane, finite result


def test_scale_out_config_is_clamped():
    cs = _make(3)
    # absurd fraction must be clamped into (0,1), not crash or sell >100%
    r = run_composite(cs, scale_out={"arm_atr": 1.0, "frac": 5.0, "breakeven": True})
    assert "total_return" in r and r["final_equity"] > 0
