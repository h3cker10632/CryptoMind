"""A 0 / NaN / wildly-off price tick must never stop out, mark or fill a
position (regression for PUMP-USD being "sold" at $0.00 for -$13.7k)."""
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.execution.paper import PaperBroker
from app.data.market import MarketData


class _Mkt:
    def __init__(self, px):
        self._px = dict(px)
        self.candles = {}

    def price(self, p):
        return self._px.get(p)


def _open_long(b):
    b.cash = 100_000.0
    pos = b.open("BTC-USD", 1, 10_000.0, 100.0, 95.0, 130.0, "t")
    assert pos is not None
    return pos


def test_zero_price_does_not_trigger_stop():
    b = PaperBroker()
    _open_long(b)
    n = len(b.closed_trades)
    b.manage(_Mkt({"BTC-USD": 0.0}), 2.5, lambda p: 1.0)
    assert "BTC-USD" in b.positions
    assert len(b.closed_trades) == n


def test_nan_price_does_not_trigger_stop():
    b = PaperBroker()
    _open_long(b)
    b.manage(_Mkt({"BTC-USD": float("nan")}), 2.5, lambda p: 1.0)
    assert "BTC-USD" in b.positions


def test_sell_refuses_invalid_price():
    b = PaperBroker()
    _open_long(b)
    assert b.sell("BTC-USD", 0.0, "stop-loss/trail") is None
    assert "BTC-USD" in b.positions
    # a real price still closes it
    t = b.sell("BTC-USD", 99.0, "stop-loss/trail")
    assert t is not None and t["pnl"] < 0


def test_open_refuses_invalid_price():
    b = PaperBroker()
    b.cash = 100_000.0
    assert b.open("BTC-USD", 1, 10_000.0, 0.0, 0.0, 1.0, "t") is None
    assert b.open("BTC-USD", 1, 10_000.0, float("nan"), 0.0, 1.0, "t") is None


def test_market_price_rejects_bad_ticks():
    m = MarketData()
    import time
    m.candles["X-USD"] = [[time.time() - 60, 99, 101, 100, 100.0, 1]]
    for bad in (0.0, -1.0, float("nan"), float("inf"), 1.0, 1000.0):
        m.tickers["X-USD"] = {"price": bad}
        assert m.price("X-USD") is None, bad
    m.tickers["X-USD"] = {"price": 104.0}
    assert m.price("X-USD") == 104.0
