"""Conformal prediction-set gate for the long-vs-short direction veto."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.learn.direction_conformal import DirectionConformalGate
from app.learn.direction import DirectionLearner


def _train(gate, n=400, seed=0):
    r = random.Random(seed)
    for _ in range(n):
        align = r.uniform(-1, 1)
        p_long = 0.5 + 0.45 * align       # align=+1 -> long favored 95%
        favored = 1 if r.random() < p_long else -1
        gate.observe(align, favored)
    return r


def test_abstains_before_calibration():
    g = DirectionConformalGate()
    assert not g.ready
    # abstains -> no veto, set is everything
    assert g.veto(-1, 0.9) == (False, "")
    assert g.prediction_set(0.9) == {1, -1}


def test_prediction_set_narrows_at_strong_alignment():
    g = DirectionConformalGate(alpha=0.10)
    _train(g)
    assert g.ready
    assert g.prediction_set(0.9) == {1}       # strong bull -> only long plausible
    assert g.prediction_set(-0.9) == {-1}     # strong bear -> only short
    assert g.prediction_set(0.0) == {1, -1}   # neutral -> ambiguous


def test_veto_excludes_wrong_side_only():
    g = DirectionConformalGate(alpha=0.10)
    _train(g)
    # short into a strongly bullish tape is vetoed; long is fine
    v_short, why = g.veto(-1, 0.9)
    assert v_short and "conformal" in why
    assert g.veto(+1, 0.9) == (False, "")
    # neutral tape vetoes nothing
    assert g.veto(-1, 0.0)[0] is False
    assert g.veto(+1, 0.0)[0] is False


def test_observe_trade_labels_favored_side():
    g = DirectionConformalGate()
    # a long that won -> long favored; a long that lost -> short favored
    g.observe_trade({"side": 1, "pnl": 10.0, "mtf_at_entry": 0.5})
    g.observe_trade({"side": 1, "pnl": -10.0, "mtf_at_entry": -0.5})
    g.observe_trade({"side": -1, "pnl": 10.0, "mtf_at_entry": -0.5})  # short won
    assert g.n_seen == 3
    # zero-pnl or missing align are ignored
    g.observe_trade({"side": 1, "pnl": 0.0, "mtf_at_entry": 0.5})
    g.observe_trade({"side": 1, "pnl": 10.0})       # no align
    assert g.n_seen == 3


def test_persistence_roundtrip():
    g = DirectionConformalGate(alpha=0.10)
    _train(g)
    d = g.to_dict()
    g2 = DirectionConformalGate()
    assert g2.load_dict(d)
    assert g2.n_seen == g.n_seen
    assert g2.prediction_set(0.9) == g.prediction_set(0.9)
    assert g2.load_dict(None) is False


def test_direction_learner_integration_and_fallback():
    """The learner uses the hard threshold while cold, the conformal gate once
    warm, and both paths increment the veto counter."""
    dl = DirectionLearner()
    # cold: hard-threshold fallback still vetoes a short vs strong bull align
    v, why = dl.veto(-1, 0.9)
    assert v and dl.n_conformal_vetoes == 0        # fell back to legacy path
    # warm the conformal gate with a clean bull-favored regime
    r = random.Random(1)
    for _ in range(400):
        align = r.uniform(-1, 1)
        favored = 1 if r.random() < (0.5 + 0.45 * align) else -1
        dl.conformal.observe(align, favored)
    v2, why2 = dl.veto(-1, 0.9)
    assert v2 and dl.n_conformal_vetoes >= 1 and "conformal" in why2


def test_learner_capture_restore_roundtrips_conformal():
    dl = DirectionLearner()
    r = random.Random(2)
    for _ in range(60):
        dl.conformal.observe(r.uniform(-1, 1), 1 if r.random() < 0.6 else -1)
    cap = dl.capture()
    dl2 = DirectionLearner()
    dl2.restore(cap)
    assert dl2.conformal.n_seen == dl.conformal.n_seen
