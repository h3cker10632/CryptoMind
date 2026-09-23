"""ML-model advisor (crypto_ml_lab round-trip) — app/learn/model_advisor.py.

These tests exercise the advisor's GOVERNANCE and shaping logic without requiring
the optional crypto_ml package or a real artifact. The exact-parity feature path
(which needs crypto_ml + pandas) is covered by an integration check that skips
cleanly when crypto_ml isn't importable.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app import settings
from app.learn.model_advisor import ModelAdvisor, _first_scalar


def _fresh(enabled=True):
    settings._settings = None
    settings.update({"model_advisor_enabled": enabled})
    return ModelAdvisor()


class _RetModel:
    def __init__(self, r0, risk=None):
        self._r0, self._risk = r0, risk

    def predict(self, X):
        out = {"returns": np.array([[self._r0, 0.0, 0.0]])}
        if self._risk is not None:
            out["risk"] = np.array([self._risk])
        return out


def test_inert_when_disabled():
    adv = _fresh(enabled=False)
    assert adv.lean("BTC-USD") == 0.0
    assert adv.enabled() is False
    assert adv.configured() is False


def test_inert_without_artifact():
    adv = _fresh(enabled=True)
    # no artifact on disk in the test env
    if not adv.artifact_present():
        assert adv.configured() is False
        assert adv.lean("BTC-USD") == 0.0


def test_set_lean_clamped_and_ttl():
    adv = _fresh(enabled=True)
    adv.set_lean("BTC-USD", 5.0)          # clamps to 1.0
    assert adv.lean("BTC-USD") == 1.0
    adv.set_lean("ETH-USD", -9.0)         # clamps to -1.0
    assert adv.lean("ETH-USD") == -1.0
    # expiry -> 0
    adv._leans["BTC-USD"]["ts"] = time.time() - adv._ttl() - 1
    assert adv.lean("BTC-USD") == 0.0


def test_lean_zero_when_disabled_even_if_cached():
    adv = _fresh(enabled=True)
    adv.set_lean("BTC-USD", 0.8)
    settings.update({"model_advisor_enabled": False})
    assert adv.lean("BTC-USD") == 0.0


def test_interpret_multitask_and_array():
    r0, rk = ModelAdvisor._interpret({"returns": np.array([[0.02, 0.01]]),
                                      "risk": np.array([0.3])})
    assert abs(r0 - 0.02) < 1e-9 and abs(rk - 0.3) < 1e-9
    r0b, rkb = ModelAdvisor._interpret(np.array([0.05, 0.01]))
    assert abs(r0b - 0.05) < 1e-9 and rkb is None
    assert ModelAdvisor._interpret("nonsense")[0] is None


def test_first_scalar_nested():
    assert _first_scalar([[0.7, 0.1]]) == 0.7
    assert _first_scalar(np.array([[0.3]])) == 0.3
    assert _first_scalar(0.42) == 0.42


def test_return_scale_and_tanh_shape():
    """With crypto_ml absent, _predict_lean returns a reason string, not a crash.
    With it present, a positive return yields a positive bounded lean."""
    adv = _fresh(enabled=True)
    adv.set_model(_RetModel(0.02, risk=0.1), meta={"features": None})
    now = time.time()
    candles = [[now - 300 * i, 100, 101, 100, 100.5, 5.0] for i in range(40)][::-1]
    lean, why = adv._predict_lean("BTC-USD", candles,
                                  book={"bid_depth": 5e5, "ask_depth": 4e5})
    try:
        import crypto_ml.features  # noqa: F401
        have_cml = True
    except Exception:
        have_cml = False
    if have_cml:
        assert lean is not None and -1.0 <= lean <= 1.0 and lean > 0
    else:
        assert lean is None and "crypto_ml" in why


def test_rug_veto_forces_zero():
    adv = _fresh(enabled=True)
    adv.set_model(_RetModel(0.05, risk=0.95), meta={"features": None})
    now = time.time()
    candles = [[now - 300 * i, 100, 101, 100, 100.5, 5.0] for i in range(40)][::-1]
    lean, why = adv._predict_lean("BTC-USD", candles,
                                  book={"bid_depth": 5e5, "ask_depth": 4e5})
    try:
        import crypto_ml.features  # noqa: F401
        # veto only reachable once features build; when present it must be 0
        assert lean == 0.0 and "veto" in why
    except Exception:
        assert lean is None  # inert without crypto_ml


def test_insufficient_history_returns_none():
    adv = _fresh(enabled=True)
    adv.set_model(_RetModel(0.01), meta={"features": None})
    lean, why = adv._predict_lean("BTC-USD", [[time.time(), 1, 1, 1, 1, 1]])
    assert lean is None


def test_stats_shape():
    adv = _fresh(enabled=True)
    s = adv.stats()
    for k in ("enabled", "configured", "artifact_present", "model_dir",
              "cached_leans", "last_error", "calls"):
        assert k in s


def test_engine_registers_model_strategy():
    from app.signals.engine import STRATEGIES, engine
    assert "model" in STRATEGIES
    assert "model" in engine.weights


def test_strat_model_reads_cache(monkeypatch):
    from app.signals import engine as eng
    from app.learn.model_advisor import advisor as global_adv
    settings._settings = None
    settings.update({"model_advisor_enabled": True})
    global_adv.set_lean("BTC-USD", 0.5)
    eng._ctx.product = "BTC-USD"
    try:
        assert eng.strat_model({}, (0.0, 0), {}) == 0.5
    finally:
        eng._ctx.product = ""
        global_adv._leans.clear()
        settings._settings = None


def test_learner_full_stats_has_model_advisor():
    from app.learn.loop import learner
    fs = learner.full_stats()
    assert "model_advisor" in fs
