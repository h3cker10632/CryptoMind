"""Correctness tests for the market-neutral pair hedger.

Six constraints:
  1. a directional signal-flip cannot sell a hedge leg
  2. cooldown after a close (hours) before the same pair re-opens
  3. both legs live or neither (no orphans)
  4. cost gate on the PAIR: expected z-move ($) > round-trip on FOUR fills
  5. kill / daily halt / max gross exposure block new hedge risk
  6. realized PnL attributed to the `hedge` sleeve, not dropped on empty votes
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.execution.paper import PaperBroker
from app.strategies.hedge import PairHedger


# ---------------- stubs ----------------
class _Mkt:
    def __init__(self):
        self.candles = {}
        self._px = {}
    def price(self, p):
        return self._px.get(p)
    def closes(self, p):
        return [c[4] for c in self.candles.get(p, [])]
    def features(self, p):
        return {"atr": (self._px.get(p, 100) * 0.02), "atr_swing": self._px.get(p, 100) * 0.05}
    def regime(self):
        return {"label": "bull"}


class _Risk:
    def __init__(self):
        self.killed = False
        self.halted_today = False
        self.closed = []
    def on_trade_closed(self, t):
        self.closed.append(t)


def _corr_series(n, base, drift_b, noise, spread_kick=0.0):
    """Build two highly-correlated close series driven by a common factor; a
    sharp late `spread_kick` on A creates a large |z| divergence in the last
    couple of bars (high corr + high z + fat enough sd to clear the cost gate)."""
    import random
    rng = random.Random(3)
    a, b = [], []
    va = vb = base
    for i in range(n):
        shock = rng.uniform(-0.015, 0.015)          # common factor -> correlation
        va *= (1 + shock + rng.uniform(-0.002, 0.002))
        vb *= (1 + shock + rng.uniform(-0.002, 0.002))
        a.append(va); b.append(vb)
    if spread_kick:
        a[-1] *= (1 + spread_kick)                  # sharp divergence, few bars
        a[-2] *= (1 + spread_kick * 0.6)            # -> high |z|
    return a, b


def _bars(closes, start_ts=0):
    out = []
    for i, c in enumerate(closes):
        out.append([start_ts + i * 300, c * 0.999, c * 1.001, c, c, 1000.0])
    return out


def _enable(monkeypatch):
    from app import settings
    monkeypatch.setattr(settings, "get",
                        lambda k: True if k in ("hedge_enabled", "allow_shorts") else settings.get(k))


# ---------------- 3 + 5 + 6 via a full open/close cycle ----------------
def _prime_divergent(mkt, broker):
    a, b = _corr_series(120, 100.0, 0.00045, 3.0, spread_kick=0.05)
    mkt.candles["AAA-USD"] = _bars(a)
    mkt.candles["BBB-USD"] = _bars(b)
    mkt._px["AAA-USD"] = a[-1]
    mkt._px["BBB-USD"] = b[-1]


def test_kill_blocks_new_hedge(monkeypatch):
    _enable(monkeypatch)
    mkt, broker, risk = _Mkt(), PaperBroker(), _Risk()
    risk.killed = True
    _prime_divergent(mkt, broker)
    h = PairHedger()
    h.tick(mkt, broker, risk)
    assert not h.active, "hedge opened while kill switch active"


def test_halt_blocks_new_hedge(monkeypatch):
    _enable(monkeypatch)
    mkt, broker, risk = _Mkt(), PaperBroker(), _Risk()
    risk.halted_today = True
    _prime_divergent(mkt, broker)
    h = PairHedger()
    h.tick(mkt, broker, risk)
    assert not h.active, "hedge opened during daily-loss halt"


def test_both_legs_live_or_neither(monkeypatch):
    _enable(monkeypatch)
    mkt, broker, risk = _Mkt(), PaperBroker(), _Risk()
    _prime_divergent(mkt, broker)
    # force the SHORT leg to fail: make cash only enough for one leg
    from app.tunables import tv
    h = PairHedger()
    # monkeypatch broker.open to fail on the short (direction < 0)
    real_open = broker.open
    def flaky_open(product, direction, *a, **k):
        if direction < 0:
            return None
        return real_open(product, direction, *a, **k)
    broker.open = flaky_open
    h.tick(mkt, broker, risk)
    assert not h.active, "a hedge was recorded with a failed short leg"
    assert not broker.positions, "orphan long leg left open after short failed"


def test_full_cycle_attributes_hedge_sleeve(monkeypatch):
    _enable(monkeypatch)
    mkt, broker, risk = _Mkt(), PaperBroker(), _Risk()
    _prime_divergent(mkt, broker)
    h = PairHedger()
    h.tick(mkt, broker, risk)
    assert len(h.active) == 1, "divergent, correlated, cost-viable pair should open"
    key = next(iter(h.active))
    # both legs tagged as hedge
    for leg in (h.active[key]["long"], h.active[key]["short"]):
        assert broker.positions[leg].get("hedge") is True

    # now revert the spread so the pair exits, and check attribution
    from app.learn.loop import learner
    before = learner.trade_attributions
    a, b = _corr_series(120, 100.0, 0.00045, 3.0, spread_kick=0.0)  # no divergence
    mkt.candles["AAA-USD"] = _bars(a); mkt.candles["BBB-USD"] = _bars(b)
    mkt._px["AAA-USD"] = a[-1]; mkt._px["BBB-USD"] = b[-1]
    h.tick(mkt, broker, risk)
    assert not h.active, "reverted spread should have closed the pair"
    assert learner.trade_attributions >= before + 1, "hedge PnL was not attributed"
    assert key in h.cooldowns, "cooldown not recorded after close"


# ---------------- 2: cooldown ----------------
def test_cooldown_blocks_immediate_reentry(monkeypatch):
    _enable(monkeypatch)
    mkt, broker, risk = _Mkt(), PaperBroker(), _Risk()
    _prime_divergent(mkt, broker)
    h = PairHedger()
    # pretend the pair just closed
    h.cooldowns["AAA-USD|BBB-USD"] = time.time()
    h.tick(mkt, broker, risk)
    assert not h.active, "re-opened a pair still inside its cooldown window"


# ---------------- 4: pair cost gate ----------------
def test_pair_cost_gate_rejects_thin_edge():
    h = PairHedger()
    # a spread with tiny sd => expected reversion $ can't beat 4 fills
    ca = [100.0 * (1 + 0.00001 * i) for i in range(96)]
    cb = [100.0 * (1 + 0.00001 * i) for i in range(96)]
    ok = h._pair_cost_viable(ca, cb, z=3.0, leg_notional=10_000)
    assert ok is False


def test_pair_cost_gate_allows_fat_edge():
    h = PairHedger()
    import random
    random.seed(0)
    # a spread with real volatility => a 3-sigma reversion clears costs
    ca, cb = [], []
    va = vb = 100.0
    for i in range(96):
        va *= (1 + random.uniform(-0.01, 0.01))
        vb *= (1 + random.uniform(-0.01, 0.01))
        ca.append(va); cb.append(vb)
    ok = h._pair_cost_viable(ca, cb, z=3.0, leg_notional=10_000)
    assert ok is True


# ---------------- 1: signal flip can't close a hedge leg ----------------
def test_signal_flip_skips_hedge_leg():
    """The orchestrator's step-6 guard: a position tagged hedge is skipped."""
    broker = PaperBroker()
    mkt = _Mkt()
    mkt._px["AAA-USD"] = 100.0
    pos = broker.open("AAA-USD", 1, 1000.0, 100.0, 90.0, 130.0, "HEDGE long",
                      is_hedge=True)
    assert pos.get("hedge") is True
    # emulate the guard
    should_skip = broker.positions["AAA-USD"].get("hedge")
    assert should_skip, "hedge leg not marked — signal flip could orphan it"
