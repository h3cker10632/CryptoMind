"""Protections — time-boxed circuit breakers (freqtrade-inspired, own code).

Covered:
  * StoplossGuard: N losing stop-outs in a window -> GLOBAL lock; only stop/
    liquidation losses count (TP/other exits don't); lock auto-expires.
  * LowProfitPairs: a single coin net-negative over enough trades -> PER-PAIR
    lock; other coins stay tradable; below trade_limit does nothing.
  * MaxDrawdown: realized peak-to-trough drawdown over the window -> temporary
    GLOBAL lock (auto-recovering, distinct from the permanent kill switch).
  * lookback windows exclude old trades; disabling knobs (limit=0) is a no-op.
  * persistence round-trip of active locks.
  * integration through RiskManager.can_open.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.risk.protections import ProtectionManager
from app.risk.manager import RiskManager
from app import tunables


def _trade(product="BTC-USD", pnl=-10.0, reason="stop-loss/trail",
           closed=None, entry=100.0, qty=10.0):
    return {"product": product, "pnl": pnl, "exit_reason": reason,
            "closed": closed if closed is not None else time.time(),
            "entry": entry, "qty": qty}


def _reset_protection_tunables():
    # deterministic, generous windows for the unit tests
    tunables.update({
        "protect_stopguard_trades": 4,
        "protect_stopguard_lookback_sec": 3600,
        "protect_stopguard_lock_sec": 3600,
        "protect_lowprofit_trades": 3,
        "protect_lowprofit_lookback_sec": 86400,
        "protect_lowprofit_required": 0.0,
        "protect_lowprofit_lock_sec": 21600,
        "protect_maxdd_trades": 10,
        "protect_maxdd_lookback_sec": 43200,
        "protect_maxdd_fraction": 0.10,
        "protect_maxdd_lock_sec": 7200,
    })


def _only_stopguard():
    """Isolate StoplossGuard so same-coin loser fixtures don't also trip the
    per-pair LowProfitPairs / global MaxDrawdown protections."""
    _reset_protection_tunables()
    tunables.update({"protect_lowprofit_trades": 0, "protect_maxdd_trades": 0})


# ---------------- StoplossGuard ----------------

def test_stoploss_guard_locks_globally_after_cluster():
    _only_stopguard()
    pm = ProtectionManager()
    trades = [_trade(product=f"C{i}-USD") for i in range(4)]   # 4 stop-outs
    pm.evaluate(trades)
    locked, why = pm.is_locked("ANY-USD")
    assert locked and "StoplossGuard" in why
    # and it applies to a pair that never traded (global)
    assert pm.is_locked("NEVER-USD")[0]


def test_stoploss_guard_ignores_non_stop_exits():
    _only_stopguard()
    pm = ProtectionManager()
    # 4 losers but exited via take-profit / signal flip -> not stop-outs
    trades = [_trade(reason="take-profit", pnl=-5.0) for _ in range(4)]
    pm.evaluate(trades)
    assert not pm.is_locked("BTC-USD")[0]


def test_stoploss_guard_needs_negative_pnl():
    _only_stopguard()
    pm = ProtectionManager()
    # stop-reason but POSITIVE pnl (trailing stop locked a gain) -> not counted
    trades = [_trade(reason="stop-loss/trail", pnl=+5.0) for _ in range(4)]
    pm.evaluate(trades)
    assert not pm.is_locked("BTC-USD")[0]


def test_stoploss_guard_respects_lookback():
    _only_stopguard()
    tunables.update({"protect_stopguard_lookback_sec": 600})    # 10 min window
    pm = ProtectionManager()
    old = time.time() - 4000
    trades = [_trade(closed=old) for _ in range(4)]             # all too old
    pm.evaluate(trades)
    assert not pm.is_locked("BTC-USD")[0]


def test_stoploss_guard_lock_expires():
    _only_stopguard()
    # min lock is 300s (tunable floor); check it lifts after that elapses
    tunables.update({"protect_stopguard_lock_sec": 300})
    pm = ProtectionManager()
    pm.evaluate([_trade() for _ in range(4)])
    assert pm.is_locked("BTC-USD")[0]
    # simulate time passing beyond the lock via the now argument
    future = time.time() + 400
    assert not pm.is_locked("BTC-USD", now=future)[0]


def test_stoploss_guard_disabled_when_zero():
    _only_stopguard()
    tunables.update({"protect_stopguard_trades": 0})
    pm = ProtectionManager()
    pm.evaluate([_trade() for _ in range(20)])
    assert not pm.is_locked("BTC-USD")[0]


# ---------------- LowProfitPairs ----------------

def test_low_profit_pairs_locks_only_the_bad_coin():
    _reset_protection_tunables()
    # disable the others so only LowProfitPairs is under test
    tunables.update({"protect_stopguard_trades": 0, "protect_maxdd_trades": 0,
                     "protect_lowprofit_required": 0.0})
    pm = ProtectionManager()
    bad = [_trade(product="DOGE-USD", pnl=-20.0, reason="take-profit")
           for _ in range(3)]
    good = [_trade(product="BTC-USD", pnl=+30.0, reason="take-profit")
            for _ in range(3)]
    pm.evaluate(bad + good)
    assert pm.is_locked("DOGE-USD")[0]
    assert "LowProfitPairs" in pm.is_locked("DOGE-USD")[1]
    assert not pm.is_locked("BTC-USD")[0]        # profitable coin untouched


def test_low_profit_pairs_needs_min_trades():
    _reset_protection_tunables()
    tunables.update({"protect_stopguard_trades": 0, "protect_maxdd_trades": 0})
    pm = ProtectionManager()
    # only 2 trades for the coin, limit is 3 -> not judged yet
    pm.evaluate([_trade(product="DOGE-USD", pnl=-20.0, reason="take-profit")
                 for _ in range(2)])
    assert not pm.is_locked("DOGE-USD")[0]


def test_low_profit_pairs_profitable_not_locked():
    _reset_protection_tunables()
    tunables.update({"protect_stopguard_trades": 0, "protect_maxdd_trades": 0})
    pm = ProtectionManager()
    pm.evaluate([_trade(product="ETH-USD", pnl=+5.0, reason="take-profit")
                 for _ in range(5)])
    assert not pm.is_locked("ETH-USD")[0]


# ---------------- MaxDrawdown ----------------

def test_max_drawdown_temporary_global_lock():
    _reset_protection_tunables()
    tunables.update({"protect_stopguard_trades": 0, "protect_lowprofit_trades": 0,
                     "protect_maxdd_trades": 5, "protect_maxdd_fraction": 0.10})
    pm = ProtectionManager()
    t0 = time.time() - 100
    # PnL curve: climbs to +100 then a run of losses draws it down hard
    pnls = [50, 50, -40, -40, -40, -30]      # peak 100 -> trough -50
    trades = [_trade(product="BTC-USD", pnl=p, reason="take-profit",
                     closed=t0 + i) for i, p in enumerate(pnls)]
    pm.evaluate(trades)
    locked, why = pm.is_locked("ANY-USD")
    assert locked and "MaxDrawdown" in why


def test_max_drawdown_needs_min_trades():
    _reset_protection_tunables()
    tunables.update({"protect_stopguard_trades": 0, "protect_lowprofit_trades": 0,
                     "protect_maxdd_trades": 20})
    pm = ProtectionManager()
    trades = [_trade(pnl=-50.0, reason="take-profit") for _ in range(5)]
    pm.evaluate(trades)
    assert not pm.is_locked("BTC-USD")[0]      # below trade_limit


# ---------------- persistence ----------------

def test_persistence_roundtrip():
    _reset_protection_tunables()
    pm = ProtectionManager()
    pm.evaluate([_trade() for _ in range(4)])                  # global lock
    tunables.update({"protect_stopguard_trades": 0, "protect_maxdd_trades": 0})
    pm.evaluate([_trade(product="DOGE-USD", pnl=-20.0, reason="take-profit")
                 for _ in range(3)])                            # pair lock
    d = pm.to_dict()
    pm2 = ProtectionManager()
    assert pm2.load_dict(d)
    assert pm2.is_locked("ANYTHING")[0]                        # global survived
    assert pm2.is_locked("DOGE-USD")[0]
    assert pm2.load_dict(None) is False


# ---------------- RiskManager integration ----------------

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
    def price(self, p):
        return 100.0
    candles = {}


def test_can_open_blocked_by_protection():
    _reset_protection_tunables()
    rm = RiskManager()
    b = _Broker()
    m = _Mkt()
    # a fresh coin with no cooldown normally opens
    ok, _ = rm.can_open("NEW-USD", b, m, data_healthy=True)
    assert ok
    # engage a global stop-out cluster
    rm.protections.evaluate([_trade() for _ in range(4)])
    ok, why = rm.can_open("NEW-USD", b, m, data_healthy=True)
    assert not ok and "protection" in why.lower()
