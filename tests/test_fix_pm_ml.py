"""Review fixes: Polymarket calibration side flips, the online model's trust in
its stats (the dashboard reads it), and the pipeline runner's stage exit code."""
import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Model:
    def __init__(self, q0):
        self.q0 = q0

    def predict(self, X):
        return np.array([self.q0])


def test_calibration_flip_signs_votes_and_confidence_for_the_side_bought():
    from app.markets.polymarket import calibration as Cal
    from app.markets.polymarket import signals
    m = {"prices": [0.40, 0.60], "outcomes": ["Yes", "No"],
         "mom_1h": -0.05, "mom_1d": -0.05, "last_trade_price": 0.41}
    sig = signals.evaluate(m, {}, 0.1, llm_lean=-0.2)
    assert sig["outcome_index"] == 1 and sig["votes"]["momentum"] > 0    # strategies lean No
    assert 0.5 < sig["agreement"] < 1

    out = Cal.adjust(m, sig, _Model(0.55))                 # calibration buys Yes instead
    assert out["outcome_index"] == 0 and out["outcome"] == "Yes"
    assert out["votes"] == {s: round(v, 3) for s, v in sig["leans"].items()}
    assert out["votes"]["momentum"] < 0 < out["votes"]["microstructure"]
    # confidence follows the calibrated edge, not the agreement of the strategies
    # that leaned the other way: a bigger calibrated edge is a more confident bet
    weak = Cal.adjust(m, sig, _Model(0.41))
    strong = Cal.adjust(m, sig, _Model(0.47))
    assert weak["outcome_index"] == strong["outcome_index"] == 0
    assert 0 < weak["confidence"] < strong["confidence"] <= 1
    from app.tunables import tv
    assert strong["confidence"] >= tv("pm_confidence_gate")  # a clear flip can be bet

    same = Cal.adjust(m, sig, _Model(0.30))                # calibration agrees: still No
    assert same["outcome_index"] == 1
    assert same["votes"] == sig["votes"] and same["confidence"] == sig["confidence"]


def test_online_model_stats_carry_the_trust_that_gates_the_vote():
    from app.learn.online_model import TinyMLP, trust
    m = TinyMLP()
    assert m.stats()["trust"] == 0.0
    for day in range(30):                                  # 80% right, 50% up, every day
        m.skill_clusters[day] = [20, 16, 10]
    st = m.stats()
    assert st["trust"] > 0 and st["trust"] == round(trust(st), 3)


@pytest.mark.parametrize("codes", [(-1, 0), (0, -9), (2, 0), (0, 2)])
def test_pipeline_stage_fails_when_any_step_fails(codes, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "run_pipeline_under_test", os.path.join(ROOT, "tools", "run_pipeline.py"))
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    exits = dict(zip(("data_sync.py", "series_sync.py"), codes))  # either step failing
    monkeypatch.setattr(rp, "ROOT", str(tmp_path))
    monkeypatch.setattr(rp, "_run", lambda ws, script, args, say, timeout: {
        "script": script, "args": list(args), "exit": exits[script], "seconds": 0.0, "tail": []})
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stages", "sync", "--json"])
    with pytest.raises(SystemExit) as e:
        rp.main()
    run = json.loads(capsys.readouterr().out)
    assert run["stages"]["sync"]["exit"] == next(c for c in codes if c) and run["ok"] is False
    assert e.value.code == 1
