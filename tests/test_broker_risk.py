"""Paper-broker PnL (long & short) + risk-gate + security tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.execution.paper import PaperBroker
from app import money
from app.risk.manager import _utc_day


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)
        self.candles = {}
    def price(self, p):
        return self._px.get(p)


def test_long_pnl_positive_on_up_move():
    b = PaperBroker()
    b.cash = 100_000.0
    pos = b.open("BTC-USD", 1, 10_000.0, 100.0, 90.0, 130.0, "t")
    assert pos is not None
    t = b.sell("BTC-USD", 120.0, "tp")
    assert t["pnl"] > 0           # bought ~100, sold ~120


def test_short_pnl_positive_on_down_move():
    b = PaperBroker()
    b.cash = 100_000.0
    pos = b.open("BTC-USD", -1, 10_000.0, 100.0, 110.0, 70.0, "t")
    assert pos is not None
    t = b.sell("BTC-USD", 80.0, "tp")
    assert t["pnl"] > 0           # shorted ~100, covered ~80


def test_open_respects_min_notional():
    b = PaperBroker()
    b.cash = 100_000.0
    # $0.50 notional is below the $1 min-notional -> rejected
    assert b.open("BTC-USD", 1, 0.5, 100.0, 90.0, 130.0, "t") is None


def test_short_liquidation_force_closes():
    b = PaperBroker()
    b.cash = 100_000.0
    b.open("BTC-USD", -1, 1_000.0, 100.0, 10_000.0, 50.0, "t")  # very wide stop
    # atr_lookup=None disables trailing so we isolate the liquidation path;
    # a +95% move against a short exhausts the margin (>90% of it) -> liquidate
    mkt = _Mkt({"BTC-USD": 195.0})
    b.manage(mkt, trail_atr_mult=2.5, atr_lookup=lambda p: None)
    assert "BTC-USD" not in b.positions
    assert any("LIQUIDATION" in t.get("exit_reason", "") for t in b.closed_trades)


def test_utc_day_is_stable_within_day():
    assert _utc_day() == _utc_day()
