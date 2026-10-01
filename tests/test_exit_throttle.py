"""Exit Mechanism Throttle: learns whether pattern_exit/signal-flip are
actually earning their keep, and dials their trigger bar accordingly."""
import time

import pytest

from app.learn.exit_throttle import ExitThrottle


def test_cold_start_factor_is_exactly_one():
    """Zero evidence at every pooling level must mean NO behavior change."""
    et = ExitThrottle()
    assert et.throttle_factor("pattern_exit", "trending/normal", "with") == 1.0


def test_confirmed_good_mechanism_loosens_above_one():
    et = ExitThrottle()
    for _ in range(40):
        et.record("pattern_exit", "chop/normal", "against", 0.01)   # validated cuts
    f = et.throttle_factor("pattern_exit", "chop/normal", "against")
    assert f > 1.0
    assert f <= ExitThrottle.MAX_FACTOR + 1e-9


def test_confirmed_bad_mechanism_tightens_below_one():
    et = ExitThrottle()
    for _ in range(40):
        et.record("pattern_exit", "chop/normal", "against", -0.01)  # false alarms
    f = et.throttle_factor("pattern_exit", "chop/normal", "against")
    assert f < 1.0
    assert f >= 1.0 / ExitThrottle.MAX_FACTOR - 1e-9


def test_sparse_fine_bucket_pools_up_to_regime_level():
    """A data-starved (mechanism, regime, trend_bucket) bucket should inherit
    a strong signal that exists for OTHER trend buckets in the same regime,
    instead of sitting at the uninformed prior."""
    et = ExitThrottle()
    for _ in range(40):
        et.record("pattern_exit", "chop/normal", "with", -0.01)
    # "against" in the SAME regime has zero direct evidence
    f_sparse = et.throttle_factor("pattern_exit", "chop/normal", "against")
    f_cold = 1.0
    assert f_sparse < f_cold          # pulled toward the regime's bad signal


def test_sparse_regime_pools_up_to_mechanism_global_prior():
    """A brand-new regime with zero evidence should inherit the mechanism's
    global estimate rather than starting from nothing."""
    et = ExitThrottle()
    for _ in range(40):
        et.record("pattern_exit", "bull/normal", "with", 0.01)
    f_new_regime = et.throttle_factor("pattern_exit", "never-seen/vol", "with")
    assert f_new_regime > 1.0


def test_well_sampled_local_evidence_not_overridden_by_global_pool():
    """A regime with its OWN well-sampled, confidently opposite sign must not
    be swamped by a strong but different global signal."""
    et = ExitThrottle()
    for _ in range(50):
        et.record("pattern_exit", "bull/normal", "with", 0.02)     # globally good
    for _ in range(50):
        et.record("pattern_exit", "chop/normal", "with", -0.02)    # locally bad
    f_local = et.throttle_factor("pattern_exit", "chop/normal", "with")
    assert f_local < 1.0


def test_different_mechanisms_learn_independently():
    et = ExitThrottle()
    for _ in range(40):
        et.record("pattern_exit", "chop/normal", "with", -0.02)
        et.record("signal_flip", "chop/normal", "with", 0.02)
    assert et.throttle_factor("pattern_exit", "chop/normal", "with") < 1.0
    assert et.throttle_factor("signal_flip", "chop/normal", "with") > 1.0


def test_decay_forgets_stale_arms():
    et = ExitThrottle()
    for _ in range(5):
        et.record("pattern_exit", "chop/normal", "with", -0.02)
    for _ in range(400):
        et.decay(gamma=0.9, prune_below=0.5)
    assert ("pattern_exit", "chop/normal", "with") not in et.arms


def test_record_exit_and_score_pending_sign_for_long():
    """Long cut at 100; price kept FALLING to 95 afterward -> the cut was
    VALIDATED (edge > 0)."""
    et = ExitThrottle()
    now = time.time()
    et.record_exit("pattern_exit", "chop/normal", "with", "BTC-USD",
                   side=1, exit_price=100.0, ts=now - 99999)
    n = et.score_pending(lambda p, ts: 95.0, horizon_sec=1800, now=now)
    assert n == 1
    mu, _, cnt = et._posterior("pattern_exit", "chop/normal", "with")
    assert cnt == 1 and mu > 0


def test_record_exit_and_score_pending_sign_for_short():
    """Short cut at 100; price FELL to 95 afterward (which would have been
    GOOD for the short) -> the cut was a FALSE ALARM (edge < 0)."""
    et = ExitThrottle()
    now = time.time()
    et.record_exit("pattern_exit", "chop/normal", "with", "ETH-USD",
                   side=-1, exit_price=100.0, ts=now - 99999)
    n = et.score_pending(lambda p, ts: 95.0, horizon_sec=1800, now=now)
    assert n == 1
    mu, _, cnt = et._posterior("pattern_exit", "chop/normal", "with")
    assert cnt == 1 and mu < 0


def test_score_pending_skips_immature_entries():
    et = ExitThrottle()
    now = time.time()
    et.record_exit("pattern_exit", "chop/normal", "with", "BTC-USD",
                   side=1, exit_price=100.0, ts=now)       # just happened
    n = et.score_pending(lambda p, ts: 95.0, horizon_sec=1800, now=now)
    assert n == 0
    assert len(et.pending) == 1


def test_score_pending_skips_missing_price():
    et = ExitThrottle()
    now = time.time()
    et.record_exit("pattern_exit", "chop/normal", "with", "BTC-USD",
                   side=1, exit_price=100.0, ts=now - 99999)
    n = et.score_pending(lambda p, ts: None, horizon_sec=1800, now=now)
    assert n == 0
    assert len(et.pending) == 0         # dropped, not retried forever


def test_capture_restore_roundtrip():
    et = ExitThrottle()
    et.record("pattern_exit", "chop/normal", "with", -0.01)
    et.record_exit("signal_flip", "bull/normal", "against", "SOL-USD",
                   side=1, exit_price=42.0, ts=123.0)
    snap = et.capture()

    et2 = ExitThrottle()
    et2.restore(snap)
    assert et2.arms[("pattern_exit", "chop/normal", "with")][0] == pytest.approx(1)
    assert len(et2.pending) == 1
    assert et2.pending[0][4] == "SOL-USD"


def test_restore_tolerates_garbage():
    et = ExitThrottle()
    et.restore({"arms": "not a dict", "pending": None})
    assert et.arms == {} or isinstance(et.arms, dict)
    et.restore(None)
    et.restore({})


# ---------------- orchestrator integration: signal-flip exit ----------------

def _fresh_broker_with_long():
    from app.execution.paper import PaperBroker
    b = PaperBroker()
    b.open("BTC-USD", 1, 1000.0, 100.0, stop=90.0, take=130.0, reason="test")
    b.positions["BTC-USD"]["opened"] = time.time() - 100_000
    return b


class _StubMarket:
    def __init__(self, price):
        self._price = price

    def price(self, p):
        return self._price


def test_signal_flip_throttled_harder_when_confirmed_harmful(monkeypatch):
    """A confidence just above the default 0.5 bar (would flip by default)
    must NOT flip once signal_flip is confirmed net-harmful here."""
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle

    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
    monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
    regime = {"label": "sideways/normal", "trend": "sideways"}
    signals = {"BTC-USD": {"direction": -1, "confidence": 0.55}}
    mkt = _StubMarket(100.0)

    exit_throttle.arms.clear()
    try:
        for _ in range(40):
            exit_throttle.record("signal_flip", "sideways/normal", "neutral", -0.02)
        orch._manage_signal_flip_exits(mkt, regime, signals)
        assert "BTC-USD" in b.positions          # throttled -> no flip
    finally:
        exit_throttle.arms.clear()


def test_signal_flip_loosened_when_confirmed_good(monkeypatch):
    """A confidence just below the default 0.5 bar (wouldn't flip by default)
    DOES flip once signal_flip is confirmed net-GOOD here."""
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle

    regime = {"label": "sideways/normal", "trend": "sideways"}
    signals = {"BTC-USD": {"direction": -1, "confidence": 0.45}}
    mkt = _StubMarket(100.0)

    exit_throttle.arms.clear()
    try:
        baseline = _fresh_broker_with_long()
        monkeypatch.setattr(orch_mod, "broker", baseline)
        monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
        monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
        orch._manage_signal_flip_exits(mkt, regime, signals)
        assert "BTC-USD" in baseline.positions   # default: no flip

        b = _fresh_broker_with_long()
        monkeypatch.setattr(orch_mod, "broker", b)
        for _ in range(40):
            exit_throttle.record("signal_flip", "sideways/normal", "neutral", 0.02)
        orch._manage_signal_flip_exits(mkt, regime, signals)
        assert "BTC-USD" not in b.positions      # loosened -> now flips
    finally:
        exit_throttle.arms.clear()


def test_signal_flip_disabled_toggle_bypasses_throttle(monkeypatch):
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.learn.exit_throttle import exit_throttle
    from app import settings

    settings.update({"exit_throttle_enabled": False})
    b = _fresh_broker_with_long()
    monkeypatch.setattr(orch_mod, "broker", b)
    monkeypatch.setattr(orch_mod.risk, "on_trade_closed", lambda t: None)
    monkeypatch.setattr(orch_mod.learner, "on_trade_closed", lambda t: None)
    regime = {"label": "sideways/normal", "trend": "sideways"}
    signals = {"BTC-USD": {"direction": -1, "confidence": 0.55}}
    mkt = _StubMarket(100.0)

    exit_throttle.arms.clear()
    try:
        for _ in range(40):
            exit_throttle.record("signal_flip", "sideways/normal", "neutral", -0.02)
        orch._manage_signal_flip_exits(mkt, regime, signals)
        assert "BTC-USD" not in b.positions      # disabled -> default behavior (flips)
    finally:
        exit_throttle.arms.clear()
        settings.update({"exit_throttle_enabled": True})


def test_signal_flip_never_closes_hedge_leg(monkeypatch):
    import app.orchestrator as orch_mod
    from app.orchestrator import orch
    from app.execution.paper import PaperBroker

    b = PaperBroker()
    b.open("BTC-USD", 1, 1000.0, 100.0, 90.0, 130.0, "HEDGE long", is_hedge=True)
    b.positions["BTC-USD"]["opened"] = time.time() - 100_000
    monkeypatch.setattr(orch_mod, "broker", b)
    regime = {"label": "sideways/normal", "trend": "sideways"}
    signals = {"BTC-USD": {"direction": -1, "confidence": 0.99}}
    mkt = _StubMarket(100.0)

    orch._manage_signal_flip_exits(mkt, regime, signals)
    assert "BTC-USD" in b.positions


# ---------------- persistence round-trip ----------------

def test_persistence_capture_restore_exit_throttle():
    from app import persistence
    from app.learn.exit_throttle import exit_throttle

    exit_throttle.arms.clear()
    exit_throttle.record("pattern_exit", "bull/normal", "with", 0.01)
    snap = persistence._capture_exit_throttle()
    assert ("pattern_exit", "bull/normal", "with") in [
        tuple(k.split("||")) for k in snap["arms"]]

    exit_throttle.arms.clear()
    exit_throttle.restore(snap)
    assert ("pattern_exit", "bull/normal", "with") in exit_throttle.arms
    exit_throttle.arms.clear()

