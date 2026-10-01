"""Pattern-aware exit: exit_threat scores contrary reversals correctly, and the
orchestrator step tightens / cuts an open position accordingly."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.signals import patterns


def _bearish_report(strength=0.8):
    return {"features": {"pat_reversal": -0.9, "pat_divergence": -0.7,
                         "pat_candle": -0.5, "pat_structure": -1.0, "pat_sr": -0.5},
            "detected": [{"name": "Double top", "direction": "bearish", "strength": strength},
                         {"name": "Bearish RSI divergence", "direction": "bearish", "strength": 0.6}],
            "structure": "downtrend"}


def _bullish_report(strength=0.8):
    return {"features": {"pat_reversal": 0.9, "pat_divergence": 0.7, "pat_candle": 0.5,
                         "pat_structure": 1.0, "pat_sr": 0.5},
            "detected": [{"name": "Double bottom", "direction": "bullish", "strength": strength}],
            "structure": "uptrend"}


def test_bearish_pattern_threatens_long_not_short():
    t_long, name = patterns.exit_threat(1, _bearish_report())
    t_short, _ = patterns.exit_threat(-1, _bearish_report())
    assert t_long > 0.6
    assert name == "Double top"
    assert t_short == 0.0                 # a bearish pattern HELPS a short


def test_bullish_pattern_threatens_short_not_long():
    t_short, name = patterns.exit_threat(-1, _bullish_report())
    t_long, _ = patterns.exit_threat(1, _bullish_report())
    assert t_short > 0.6
    assert name == "Double bottom"
    assert t_long == 0.0


def test_threat_zero_without_side_or_report():
    assert patterns.exit_threat(0, _bearish_report()) == (0.0, "")
    assert patterns.exit_threat(1, None) == (0.0, "")


def test_threat_ignores_structure_and_sr_only():
    # structure/S&R negative but NO reversal/divergence/candle -> no threat
    rep = {"features": {"pat_structure": -1.0, "pat_sr": -1.0,
                        "pat_reversal": 0.0, "pat_divergence": 0.0, "pat_candle": 0.0},
           "detected": [{"name": "Range / no clear structure", "direction": "neutral", "strength": 0.3}]}
    threat, name = patterns.exit_threat(1, rep)
    assert threat == 0.0
    assert name == ""


# ---------------- orchestrator integration ----------------

def _fresh_broker_with_long():
    from app.execution.paper import PaperBroker
    b = PaperBroker()
    b.open("PEPE-USD", 1, 1000.0, 100.0, stop=95.0, take=115.0, reason="test")
    # make it old enough to pass the min-hold guard
    b.positions["PEPE-USD"]["opened"] = time.time() - 100_000
    return b


class _StubMarket:
    def __init__(self, price, report):
        self._price = price
        self._report = report

    def price(self, p):
        return self._price

    def features(self, p):
        return {"atr_swing": 2.0, "atr": 2.0, "patterns": self._report}

    def regime(self):
        return {"label": "sideways/normal", "trend": "sideways",
                "vol_state": "normal", "vol": 0.0}


def test_orchestrator_cuts_long_on_strong_reversal(monkeypatch):
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    # neutralise the learner/risk side-effects of a close
    monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
    monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
    mkt = _StubMarket(100.0, _bearish_report(strength=0.9))
    assert "PEPE-USD" in b.positions
    orch._manage_pattern_exit(mkt)
    assert "PEPE-USD" not in b.positions          # strong reversal -> cut


def test_orchestrator_tightens_stop_on_moderate_reversal(monkeypatch):
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    # a moderate threat (0.45 <= threat < 0.70): reversal present, below cut.
    rep = {"features": {"pat_reversal": -0.9, "pat_divergence": 0.0, "pat_candle": -0.3},
           "detected": [{"name": "Head & shoulders", "direction": "bearish", "strength": 0.6}],
           "structure": "range"}
    mkt = _StubMarket(100.0, rep)
    old_stop = b.positions["PEPE-USD"]["stop"]
    orch._manage_pattern_exit(mkt)
    assert "PEPE-USD" in b.positions               # not cut
    assert b.positions["PEPE-USD"]["stop"] > old_stop   # stop pulled up closer


# ---------------- exit-throttle integration (learned trigger bar) ----------------

def _moderate_bearish_report():
    # threat = 0.45*0.9 + 0.30*0 + 0.25*0.3 = 0.48 (tighten-only by default)
    return {"features": {"pat_reversal": -0.9, "pat_divergence": 0.0, "pat_candle": -0.3},
            "detected": [{"name": "Head & shoulders", "direction": "bearish", "strength": 0.6}],
            "structure": "range"}


def test_pattern_exit_throttled_harder_when_confirmed_harmful(monkeypatch):
    """A strong reversal that cuts at the default bar (threat ~0.74 > 0.70)
    must NOT cut once pattern_exit is confirmed net-harmful in this (regime,
    trend bucket) -- the effective cut bar is raised."""
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle
    from app import tunables

    tunables.update({"pattern_exit_cut": 0.70})
    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
    monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
    mkt = _StubMarket(100.0, _bearish_report())        # threat ~0.74

    exit_throttle.arms.clear()
    try:
        # regime label from _StubMarket.regime() == "sideways/normal", side=1,
        # trend "sideways" -> trend_bucket "neutral"
        for _ in range(40):
            exit_throttle.record("pattern_exit", "sideways/normal", "neutral", -0.02)
        orch._manage_pattern_exit(mkt)
        assert "PEPE-USD" in b.positions          # throttled -> no longer cuts
    finally:
        exit_throttle.arms.clear()
        tunables.update({"pattern_exit_cut": 0.70})


def test_pattern_exit_loosened_when_confirmed_good(monkeypatch):
    """A threat just BELOW the default cut bar (threat ~0.48 < 0.70) still
    doesn't cut by default, but DOES once pattern_exit is confirmed net-GOOD
    here (lower effective cut bar)."""
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle
    from app import tunables

    tunables.update({"pattern_exit_cut": 0.70})
    mkt = _StubMarket(100.0, _moderate_bearish_report())    # threat ~0.48

    exit_throttle.arms.clear()
    try:
        baseline_broker = _fresh_broker_with_long()
        monkeypatch.setattr(orch_mod, "broker", baseline_broker)
        monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
        monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
        orch._manage_pattern_exit(mkt)
        assert "PEPE-USD" in baseline_broker.positions   # default: not cut

        b = _fresh_broker_with_long()
        monkeypatch.setattr(orch_mod, "broker", b)
        for _ in range(40):
            exit_throttle.record("pattern_exit", "sideways/normal", "neutral", 0.02)
        orch._manage_pattern_exit(mkt)
        assert "PEPE-USD" not in b.positions       # loosened -> now cuts
    finally:
        exit_throttle.arms.clear()
        tunables.update({"pattern_exit_cut": 0.70})


def test_pattern_exit_disabled_toggle_bypasses_throttle(monkeypatch):
    """exit_throttle_enabled=False must behave EXACTLY like before (factor
    pinned at 1.0), regardless of any learned evidence."""
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle
    from app import settings, tunables

    tunables.update({"pattern_exit_cut": 0.70})
    settings.update({"exit_throttle_enabled": False})
    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
    monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
    mkt = _StubMarket(100.0, _bearish_report())        # threat ~0.74, would be throttled

    exit_throttle.arms.clear()
    try:
        for _ in range(40):
            exit_throttle.record("pattern_exit", "sideways/normal", "neutral", -0.02)
        orch._manage_pattern_exit(mkt)
        assert "PEPE-USD" not in b.positions      # disabled -> default behavior (cuts)
    finally:
        exit_throttle.arms.clear()
        settings.update({"exit_throttle_enabled": True})
        tunables.update({"pattern_exit_cut": 0.70})
