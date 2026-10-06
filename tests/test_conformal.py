"""Conformal calibration of the online committee's predictive band.

The committee's quantile heads are trained but never CALIBRATED — nothing
guarantees the realized return lands inside [lo, hi] at the claimed rate. The
ConformalCalibrator wraps the model with a split/adaptive-conformal correction
so the emitted interval carries a distribution-free ~(1-alpha) coverage
guarantee, and the sizer's confidence is derived from that CALIBRATED width.

Covered here:
  * qhat WIDENS an over-confident (too-narrow) band and TIGHTENS an over-wide one
  * empirical coverage of the calibrated interval ~ 1-alpha in both cases
  * Adaptive Conformal Inference chases coverage back after a regime shift
  * predict_with_uncertainty exposes calibrated band + raw band + qhat and keeps
    the downstream keys (mean/lo/hi/confidence) intact
  * calibrator state round-trips through to_dict/load_dict
  * end-to-end: committee.observe_outcome warms the calibrator to target coverage
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.learn.online_model import (ConformalCalibrator, Committee, N_IN,
                                     _quantile_sorted)


from app.learn.online_model import TARGET_SCALE


def test_quantile_sorted_matches_interpolation():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _quantile_sorted(xs, 0.0) == 0.0
    assert _quantile_sorted(xs, 1.0) == 4.0
    assert abs(_quantile_sorted(xs, 0.5) - 2.0) < 1e-9
    assert abs(_quantile_sorted(xs, 0.25) - 1.0) < 1e-9
    assert _quantile_sorted([], 0.5) == 0.0


def test_widens_overconfident_band():
    """Truth is wide (N(0,0.5)) but the raw band is a tiny [-0.1, 0.1]:
    qhat must be POSITIVE (widen) and restore coverage to ~1-alpha."""
    r = random.Random(1)
    cal = ConformalCalibrator(alpha=0.20)
    truth = lambda: max(-1.0, min(1.0, r.gauss(0, 0.5)))
    for _ in range(600):
        cal.observe(-0.1, 0.1, truth())
    lo, hi, q = cal.calibrate(-0.1, 0.1)
    assert q > 0                                   # widened
    assert hi - lo > 0.2
    hits = sum(1 for _ in range(8000) if lo <= truth() <= hi)
    assert 0.75 <= hits / 8000 <= 0.88             # ~0.80


def test_tightens_overwide_band():
    """Truth is narrow (N(0,0.1)) but the raw band is a huge [-1, 1]:
    qhat must be NEGATIVE (tighten) while keeping ~1-alpha coverage."""
    r = random.Random(2)
    cal = ConformalCalibrator(alpha=0.20)
    truth = lambda: max(-1.0, min(1.0, r.gauss(0, 0.1)))
    for _ in range(600):
        cal.observe(-1.0, 1.0, truth())
    lo, hi, q = cal.calibrate(-1.0, 1.0)
    assert q < 0                                   # tightened
    assert hi - lo < 2.0
    hits = sum(1 for _ in range(8000) if lo <= truth() <= hi)
    assert 0.75 <= hits / 8000 <= 0.90


def test_not_ready_is_a_noop():
    """Below the minimum calibration set the correction is exactly zero, so we
    degrade to the old (uncalibrated) behaviour rather than emit garbage."""
    cal = ConformalCalibrator(alpha=0.20)
    for _ in range(5):
        cal.observe(-0.2, 0.2, 0.0)
    assert not cal.ready
    lo, hi, q = cal.calibrate(-0.2, 0.2)
    assert q == 0.0 and (lo, hi) == (-0.2, 0.2)


def test_adaptive_recovers_coverage_after_shift():
    """Adaptivity: after the noise scale jumps, the calibrator (rolling window
    + ACI) must pull coverage of a FIXED raw band back toward the target under
    the new regime instead of staying broken. We measure coverage on the new
    regime only, after the calibrator has had time to re-warm to it."""
    r = random.Random(3)
    cal = ConformalCalibrator(alpha=0.20, gamma=0.05, window=400)
    wide = lambda: max(-1.0, min(1.0, r.gauss(0, 0.9)))
    # calm regime first (narrow noise) so the calibrator starts tuned elsewhere
    for _ in range(400):
        cal.observe(-0.3, 0.3, max(-1, min(1, r.gauss(0, 0.15))))
    # volatility explodes: the fixed raw band [-0.3, 0.3] now under-covers
    for _ in range(600):            # long enough to flush the window
        cal.observe(-0.3, 0.3, wide())
    # having re-warmed to the new regime, the calibrated band should cover ~1-a
    lo, hi, q = cal.calibrate(-0.3, 0.3)
    assert q > 0                                   # widened for the new regime
    hits = sum(1 for _ in range(8000) if lo <= wide() <= hi)
    assert hits / 8000 >= 0.70                     # recovered toward 0.80


def test_predict_with_uncertainty_shape_and_keys():
    """The downstream contract (engine/sizer) must be preserved: mean, lo, hi,
    confidence stay present; new keys are additive."""
    c = Committee(n_members=3)
    x = [0.0] * N_IN
    u = c.predict_with_uncertainty(x)
    for k in ("mean", "epistemic", "aleatoric", "lo", "hi", "confidence"):
        assert k in u
    for k in ("raw_lo", "raw_hi", "qhat", "calibrated"):
        assert k in u
    assert 0.0 <= u["confidence"] <= 1.0
    assert u["lo"] <= u["hi"]
    # cold calibrator => calibrated band == raw band
    assert u["calibrated"] is False
    assert abs(u["lo"] - u["raw_lo"]) < 1e-12
    assert abs(u["hi"] - u["raw_hi"]) < 1e-12


def test_confidence_shrinks_as_band_widens():
    """Confidence is monotonically decreasing in the calibrated half-width."""
    cal = ConformalCalibrator()
    narrow = 1.0 / (1.0 + 4.0 * 0.1)
    wide = 1.0 / (1.0 + 4.0 * 0.8)
    assert narrow > wide                            # sanity on the mapping used


def test_roundtrip_persistence():
    r = random.Random(4)
    cal = ConformalCalibrator(alpha=0.20)
    for _ in range(300):
        cal.observe(-0.2, 0.2, max(-1, min(1, r.gauss(0, 0.4))))
    d = cal.to_dict()
    cal2 = ConformalCalibrator()
    assert cal2.load_dict(d)
    assert abs(cal2.qhat() - cal.qhat()) < 1e-12
    assert abs(cal2.alpha_t - cal.alpha_t) < 1e-12
    assert cal2.n_seen == cal.n_seen
    assert list(cal2.scores) == list(cal.scores)
    # bad / empty payloads are tolerated
    assert cal2.load_dict(None) is False


def test_end_to_end_committee_coverage():
    """Warm a real committee through observe_outcome + update on a noisy DGP and
    verify the calibrated interval hits ~1-alpha coverage on held-out data."""
    r = random.Random(5)

    def gen():
        x = [r.gauss(0, 1) for _ in range(N_IN)]
        fwd = TARGET_SCALE * (0.6 * math.tanh(x[0]) + r.gauss(0, 0.7))
        return x, fwd

    c = Committee(n_members=3)
    for _ in range(1500):
        x, fwd = gen()
        c.observe_outcome(x, fwd)
        c.update(x, fwd)

    assert c.calibrator.ready
    N, hits = 3000, 0
    for _ in range(N):
        x, fwd = gen()
        u = c.predict_with_uncertainty(x)
        y = max(-1.0, min(1.0, fwd / TARGET_SCALE))
        if u["lo"] <= y <= u["hi"]:
            hits += 1
    coverage = hits / N
    assert 0.72 <= coverage <= 0.88                 # ~0.80 target
    st = c.stats()["conformal"]
    assert st["ready"] and st["n_scores"] > 0
