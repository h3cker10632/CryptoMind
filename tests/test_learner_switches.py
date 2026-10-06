"""Learner evidence gate (app/learn/gate.py): a learner drives live decisions
only when its mode is "on", or "auto" with fresh ablation evidence that it
helps; gated-off learners' outputs are not applied."""
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _set(monkeypatch, **kv):
    from app import settings
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda k: kv[k] if k in kv else real(k))


def _report(tmp_path, monkeypatch, ran_at=None, **helps):
    from app.learn import gate
    rep = {"ok": True, "ran_at": time.time() if ran_at is None else ran_at,
           "variants": {n: {"helps_in_both_halves": h,
                            "delta_vs_baseline_pct_pts": {"first_half": 1, "second_half": 1}}
                        for n, h in helps.items()}}
    p = tmp_path / "abl.json"
    p.write_text(json.dumps(rep))
    monkeypatch.setattr(gate, "REPORT", str(p))
    gate._cache.update(mtime=None, report=None)


def test_defaults_are_auto_and_chop_on():
    from app import settings
    d = settings.DEFAULTS
    for k in ("bandit_mode", "direction_learner_mode", "exit_advisor_mode",
              "rl_risk_mode", "ml_vote_mode"):
        assert d[k] == "auto"
    assert d["chop_filter_mode"] == "on"


def test_auto_follows_evidence_and_modes_override(tmp_path, monkeypatch):
    from app.learn import gate
    _report(tmp_path, monkeypatch, bandit=True, direction=False)
    assert gate.active("bandit") is True
    assert gate.active("direction") is False
    assert gate.active("rl_risk") is False             # not in the report
    _set(monkeypatch, direction_learner_mode="on", bandit_mode="off")
    assert gate.active("direction") is True
    assert gate.active("bandit") is False


def test_missing_or_stale_evidence_means_off(tmp_path, monkeypatch):
    from app.learn import gate
    monkeypatch.setattr(gate, "REPORT", str(tmp_path / "nope.json"))
    gate._cache.update(mtime=None, report=None)
    assert gate.active("bandit") is False
    _report(tmp_path, monkeypatch, ran_at=time.time() - 30 * 86400, bandit=True)
    assert gate.active("bandit") is False
    assert "days old" in gate.evidence("bandit")[1]


def test_direction_learner_gated_off_applies_no_bias(monkeypatch):
    from app.learn.direction import DirectionLearner
    dl = DirectionLearner()
    dl.edge = {("bull/normal", 1): (30, 0.02), ("bull/normal", -1): (30, -0.02)}
    _set(monkeypatch, direction_learner_mode="on")
    assert dl.adjust("bull/normal", 0.1, 0.0)[0] != 0.1
    _set(monkeypatch, direction_learner_mode="off")
    assert dl.adjust("bull/normal", 0.1, 0.0) == (0.1, False)
    assert dl.veto(-1, 0.99)[0] is True                 # fixed HTF veto stays


def test_rl_gated_off_keeps_scale_at_one(monkeypatch):
    from app.risk.manager import RiskManager
    from app.learn import rl_risk
    monkeypatch.setattr(rl_risk.agent, "act", lambda *a, **k: 0.25)
    r = RiskManager()
    _set(monkeypatch, rl_risk_mode="off")
    assert r.update(100_000.0, {"trend": "bull", "vol_state": "normal"})["rl_scale"] == 1.0
    _set(monkeypatch, rl_risk_mode="on")
    assert r.update(100_000.0, {"trend": "bull", "vol_state": "normal"})["rl_scale"] == 0.25


def test_bandit_and_ml_vote_follow_gate(monkeypatch):
    from app.learn import loop
    _set(monkeypatch, bandit_mode="off", ml_vote_mode="off")
    assert loop._bandit_enabled() is False
    assert "ml" in loop.learner._silent_sleeves()
    _set(monkeypatch, bandit_mode="on")
    assert loop._bandit_enabled() is True


def test_ml_reset_needs_confident_evidence():
    from app.learn.online_model import confidently_broken
    # 20 scored predictions at 45%: noise, not broken (the old rule reset here)
    assert not confidently_broken({"directional_accuracy": 0.45, "acc_samples": 20,
                                   "n_updates": 5000})
    # 300 scored at 50%: a coin flip, not broken
    assert not confidently_broken({"directional_accuracy": 0.50, "acc_samples": 300,
                                   "n_updates": 5000})
    # 300 scored at 40%: reliably worse than a coin flip
    assert confidently_broken({"directional_accuracy": 0.40, "acc_samples": 300,
                               "n_updates": 5000})


def test_warm_start_only_when_replay_state_is_more_experienced(tmp_path, monkeypatch):
    from app.learn import gate
    from app.learn.direction import DirectionLearner, direction_learner
    monkeypatch.setattr(gate, "STATES", str(tmp_path / "states.json"))
    seasoned = DirectionLearner()
    for _ in range(50):
        seasoned.on_trade_closed({"regime_at_entry": "bull/normal", "side": 1,
                                  "qty": 1, "entry": 100, "pnl": 2})
    before = direction_learner.capture()
    try:
        direction_learner.n_updates = 0
        gate.save_states({"direction": seasoned.capture()})
        assert gate.warm_start("direction") is True
        assert direction_learner.n_updates == 50
        assert gate.warm_start("direction") is False    # live is no longer behind
    finally:
        direction_learner.restore(before)
