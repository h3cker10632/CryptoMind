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
