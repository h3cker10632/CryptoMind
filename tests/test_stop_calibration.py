"""Conformal calibration of the ATR stop multiple (risk manager).

The stop was a guessed constant (stop_atr_mult * ATR). The StopCalibrator makes
it the (1-alpha) conformal quantile of realized MAXIMUM ADVERSE EXCURSIONS in
ATR units, so only ~alpha of otherwise-surviving trades are stopped by noise —
a distribution-free coverage guarantee instead of a guess.

Covered here:
  * quantile helper correctness
  * CENSORING: stop/liquidation exits never enter the score set (their MAE is
    capped by the stop), only the realized stop-rate
  * qhat tracks the (1-alpha) quantile of the uncensored MAE distribution
  * cold-start is a no-op (returns the operator default) and the result is
    clamped to the tunable bounds
  * empirical coverage: a stop at qhat survives ~(1-alpha) of fresh excursions
  * persistence round-trips
  * end-to-end through RiskManager.on_trade_closed -> size(): the stop multiple
    moves toward the data and the take/stop RR ratio is preserved
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.risk.stop_calibrator import StopCalibrator, _quantile_sorted
from app.risk.manager import RiskManager
from app import tunables


def test_quantile_sorted():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _quantile_sorted(xs, 0.0) == 0.0
    assert _quantile_sorted(xs, 1.0) == 4.0
    assert abs(_quantile_sorted(xs, 0.5) - 2.0) < 1e-9
    assert _quantile_sorted([], 0.9) == 0.0


def test_censoring_excludes_stopped_trades():
    c = StopCalibrator(alpha=0.10)
    # 40 survivors with known excursions
    for _ in range(40):
        c.observe(1.5, stopped=False)
    # a flood of stop-outs must NOT pollute the score set
    for _ in range(40):
        c.observe(99.0, stopped=True)      # censored: excluded from scores
    assert c.n_seen == 40
    assert len(c.scores) == 40
    assert all(s == 1.5 for s in c.scores)
    # but they DO count toward the realized stop-rate
    assert abs(c.stop_rate() - 0.5) < 1e-9


def test_cold_start_is_noop_and_clamped():
    c = StopCalibrator()
    # below MIN_CALIB -> return the operator default untouched
    for _ in range(5):
        c.observe(1.0, stopped=False)
    assert c.qhat() is None
    assert c.stop_mult(2.0, 0.5, 6.0) == 2.0
    # once ready but with an extreme excursion, the result is clamped to bounds
    c = StopCalibrator(alpha=0.01)
    for _ in range(60):
        c.observe(100.0, stopped=False)
    assert c.stop_mult(2.0, 0.5, 6.0) == 6.0     # clamped to the tunable max


def test_qhat_tracks_uncensored_quantile():
    c = StopCalibrator(alpha=0.10)
    # excursions uniformly in [0, 3] ATR; the 0.90 quantile ~ 2.7
    vals = [i / 100.0 * 3.0 for i in range(101)]
    for v in vals:
        c.observe(v, stopped=False)
    q = c.qhat()
    assert 2.5 <= q <= 3.0


def test_empirical_coverage_of_calibrated_stop():
    """A stop set at qhat should contain ~(1-alpha) of fresh excursions drawn
    from the same distribution."""
    import random
    r = random.Random(0)
    alpha = 0.10
    c = StopCalibrator(alpha=alpha)
    draw = lambda: abs(r.gauss(0, 1.0))          # half-normal adverse excursion
    for _ in range(300):
        c.observe(draw(), stopped=False)
    q = c.qhat()
    hits = sum(1 for _ in range(20000) if draw() <= q)
    cov = hits / 20000
    assert 1 - alpha - 0.03 <= cov <= 1 - alpha + 0.03   # ~0.90


def test_roundtrip_persistence():
    c = StopCalibrator(alpha=0.10)
    for i in range(50):
        c.observe(1.0 + i * 0.01, stopped=(i % 5 == 0))
    d = c.to_dict()
    c2 = StopCalibrator()
    assert c2.load_dict(d)
    assert c2.qhat() == c.qhat()
    assert c2.n_seen == c.n_seen
    assert c2.n_total == c.n_total
    assert list(c2.scores) == list(c.scores)
    assert c2.load_dict(None) is False


def _rs():
    return {"effective_risk_scale": 1.0}


def test_size_uses_default_before_calibrated():
    tunables.update({"stop_atr_mult": 2.0, "take_profit_atr_mult": 3.0})
    rm = RiskManager()
    _, stop, take = rm.size(100_000.0, 100.0, 2.0, 0.8, _rs(),
                            direction=1, product="BTC-USD")
    # cold: stop distance = 2.0(mult) * 2.0(atr) = 4.0 -> stop at 96
    assert abs(stop - 96.0) < 1e-6
    assert abs(take - 106.0) < 1e-6              # 3.0 * 2.0 = 6.0 -> 106


def test_size_shifts_stop_after_calibration_and_preserves_rr():
    tunables.update({"stop_atr_mult": 2.0, "take_profit_atr_mult": 3.0})
    rm = RiskManager()
    # feed survivor trades whose adverse excursion is ~1.0 ATR (much tighter
    # than the guessed 2.0), so the calibrated stop should TIGHTEN toward ~1.0.
    for _ in range(60):
        rm.on_trade_closed({
            "product": "BTC-USD", "pnl": 5.0, "side": 1, "entry": 100.0,
            "atr_at_entry": 2.0, "mae_price": 98.0,       # 1.0 ATR adverse
            "exit_reason": "take-profit"})
    _, stop, take = rm.size(100_000.0, 100.0, 2.0, 0.8, _rs(),
                            direction=1, product="ETH-USD")
    stop_dist = 100.0 - stop
    take_dist = take - 100.0
    # calibrated multiple ~1.0 -> stop_dist ~ 1.0 * 2.0(atr) = ~2.0, well under
    # the old 4.0
    assert stop_dist < 3.5
    # RR geometry (take/stop = 3.0/2.0 = 1.5) preserved
    assert abs(take_dist / stop_dist - 1.5) < 1e-6


def test_stopped_trades_alone_do_not_calibrate():
    """Only stop-outs seen -> censored -> calibrator never becomes ready, so
    size() keeps using the operator default (no circular self-calibration)."""
    tunables.update({"stop_atr_mult": 2.0, "take_profit_atr_mult": 3.0})
    rm = RiskManager()
    for _ in range(80):
        rm.on_trade_closed({
            "product": "BTC-USD", "pnl": -5.0, "side": 1, "entry": 100.0,
            "atr_at_entry": 2.0, "mae_price": 96.0,
            "exit_reason": "stop-loss/trail"})
    assert not rm.stop_calibrator.ready
    _, stop, _ = rm.size(100_000.0, 100.0, 2.0, 0.8, _rs(),
                         direction=1, product="ETH-USD")
    assert abs(stop - 96.0) < 1e-6              # unchanged default
    # but the realized stop-rate is fully observed
    assert abs(rm.stop_calibrator.stop_rate() - 1.0) < 1e-9
