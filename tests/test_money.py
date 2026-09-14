"""Exact-decimal money math + venue precision/lot/min-notional."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import money


def test_round_price_to_tick():
    assert money.round_price("BTC-USD", 100.017) == 100.02
    assert money.round_price("DOGE-USD", 0.123456) == 0.12346


def test_round_qty_rounds_down_never_up():
    # step for LINK is 0.001 — 1.2349 must floor to 1.234, never 1.235
    assert money.round_qty("LINK-USD", 1.2349) == 1.234


def test_min_notional_enforced():
    assert money.meets_min_notional("BTC-USD", 0.001, 2000.0)      # $2 >= $1
    assert not money.meets_min_notional("BTC-USD", 0.0000001, 1.0)  # < $1


def test_register_instrument_updates_rules():
    money.register_instrument("XYZ-USD", "0.5", "0.01", "5")
    assert money.round_price("XYZ-USD", 10.2) == 10.0        # tick 0.5
    assert not money.meets_min_notional("XYZ-USD", 0.1, 10.0)  # $1 < $5 min
