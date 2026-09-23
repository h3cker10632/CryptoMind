"""SpreadFilter entry gate (freqtrade idea) in RiskManager.can_open."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.risk.manager import RiskManager
from app import tunables


class _Broker:
    def __init__(self):
        self.positions = {}
        self.closed_trades = []
    def equity(self, market):
        return 100_000.0
    def exposure(self, market):
        return 0.0


class _Mkt:
    healthy = True
    candles = {}
    def __init__(self, spread_bps=None):
        self.books = {} if spread_bps is None else {"NEW-USD": {"spread_bps": spread_bps}}
    def price(self, p):
        return 100.0


def _open(rm, market):
    return rm.can_open("NEW-USD", _Broker(), market, data_healthy=True)


def test_spread_gate_disabled_by_default_or_zero():
    tunables.update({"max_spread_bps": 0, "cooldown_sec": 0})
    rm = RiskManager()
    ok, _ = _open(rm, _Mkt(spread_bps=999))     # huge spread, but gate off
    assert ok


def test_spread_gate_blocks_wide_spread():
    tunables.update({"max_spread_bps": 20, "cooldown_sec": 0})
    rm = RiskManager()
    ok, why = _open(rm, _Mkt(spread_bps=50))     # 50 > 20
    assert not ok and "spread" in why.lower()


def test_spread_gate_allows_tight_spread():
    tunables.update({"max_spread_bps": 20, "cooldown_sec": 0})
    rm = RiskManager()
    ok, _ = _open(rm, _Mkt(spread_bps=5))        # 5 < 20
    assert ok


def test_spread_gate_no_book_does_not_block():
    tunables.update({"max_spread_bps": 20, "cooldown_sec": 0})
    rm = RiskManager()
    ok, _ = _open(rm, _Mkt(spread_bps=None))     # no book reading for the pair
    assert ok


def test_cleanup():
    tunables.update({"max_spread_bps": 0})       # restore default off
