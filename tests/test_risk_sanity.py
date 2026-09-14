"""Regression tests for the penny-asset / junk-price blowup (PUMP-USD).

Covers the four fixes:
  1. price-sanity gate rejects junk (<min_price) and spiked/collapsed prices
  2. sizing skips (notional=0) when the honest ATR target can't clear costs,
     instead of ballooning notional to the position cap
  3. the liquidity cap uses the product passed to size(), not a stale hack
  4. reset_kill() also clears the daily halt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.risk.manager import RiskManager


class _Mkt:
    """Minimal market stub: prices + closes + candles per product."""
    def __init__(self):
        self._px = {}
        self._closes = {}
        self.candles = {}

    def set(self, p, price, closes=None, candles=None):
        self._px[p] = price
        if closes is not None:
            self._closes[p] = closes
        if candles is not None:
            self.candles[p] = candles

    def price(self, p):
        return self._px.get(p)

    def closes(self, p):
        return self._closes.get(p, [])


_RISK_STATUS = {"effective_risk_scale": 1.0}


# ---------- Fix 1: price-sanity gate ----------

def test_price_sane_rejects_junk_penny_price():
    r = RiskManager()
    m = _Mkt()
    m.set("PUMP-USD", 0.0039, closes=[0.0039] * 60)   # sub-cent junk
    ok, why = r.price_sane("PUMP-USD", m)
    assert not ok and "min tradable" in why


def test_price_sane_rejects_spike():
    r = RiskManager()
    m = _Mkt()
    # median ~ $100 but live print is $500 -> 5x spike (> default 3x band)
    m.set("SPK-USD", 500.0, closes=[100.0] * 60)
    ok, why = r.price_sane("SPK-USD", m)
    assert not ok and "spiked" in why


def test_price_sane_rejects_collapse():
    r = RiskManager()
    m = _Mkt()
    m.set("CRSH-USD", 20.0, closes=[100.0] * 60)       # < median/3
    ok, why = r.price_sane("CRSH-USD", m)
    assert not ok and "collapsed" in why


def test_price_sane_allows_normal_asset():
    r = RiskManager()
    m = _Mkt()
    m.set("BTC-USD", 101.0, closes=[100.0] * 60)
    ok, why = r.price_sane("BTC-USD", m)
    assert ok and why == "ok"


# ---------- Fix 2: honest sizing / cost gate ----------

def test_size_skips_when_target_cannot_clear_costs():
    """Tiny ATR relative to price -> ATR take-profit < cost floor -> skip."""
    r = RiskManager()
    price = 100.0
    atr = 0.01                       # 1bp of price; way under the fee wall
    notional, stop, take = r.size(100_000.0, price, atr, 0.8, _RISK_STATUS,
                                  direction=1, product="X-USD")
    assert notional == 0


def test_size_does_not_balloon_to_position_cap_on_penny_atr():
    """Even a viable target must not exceed the position cap; and a tiny ATR
    that fails the cost gate returns 0 rather than a capped mega-notional."""
    r = RiskManager()
    # penny-like: low ATR fails cost gate -> 0 (the PUMP failure mode)
    n, _, _ = r.size(100_000.0, 0.0039, 0.00001, 0.9, _RISK_STATUS,
                     direction=1, product="PUMP-USD")
    assert n == 0


def test_size_normal_trade_is_bounded_by_cap():
    r = RiskManager()
    equity = 100_000.0
    price, atr = 100.0, 5.0          # 5% ATR: target clears costs easily
    n, stop, take = r.size(equity, price, atr, 0.8, _RISK_STATUS,
                           direction=1, product="BTC-USD")
    assert n > 0
    assert n <= equity * 0.20 * 1.5 + 1e-6     # <= max_position_pct * cap
    assert stop < price < take                 # long geometry


# ---------- Fix 3: liquidity cap uses the right product ----------

def test_liquidity_cap_uses_passed_product():
    r = RiskManager()
    from app.data import market as mkt_mod
    # give the sized product thin volume, another product huge volume
    mkt_mod.market.candles["THIN-USD"] = [
        [0, 0, 0, 0, 100.0, 1.0] for _ in range(300)]     # ~1 base unit/bar
    liq = r._liquidity_notional(100.0, product="THIN-USD")
    assert liq is not None
    # function sums the last 288 bars (≈24h): 288 * 1 base unit * $100
    assert abs(liq - 288 * 1.0 * 100.0) < 1e-6


# ---------- Fix 4: reset clears daily halt ----------

def test_reset_kill_clears_daily_halt():
    r = RiskManager()
    r.killed = True
    r.halted_today = True
    r.consecutive_losses = 5
    r.reset_kill()
    assert r.killed is False
    assert r.halted_today is False          # the key fix
    assert r.consecutive_losses == 0
