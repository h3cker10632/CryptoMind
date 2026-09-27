"""Unit tests for the autonomous, metric-gated ML retraining pipeline.

crypto_ml is NOT installed in CI, so every test stubs the lab CLI runner and the
in-process dataset export. That isolates exactly the logic we own: the
data-driven trigger, the fail-closed gate, atomic promotion, and trigger-state
bookkeeping.
"""
import json
import os

import pytest

from app.learn.ml_trainer import MLTrainer


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    """A fresh trainer whose model/staging/state dirs live under tmp_path."""
    model_dir = str(tmp_path / "model_artifact")
    monkeypatch.setattr(MLTrainer, "_model_dir", staticmethod(lambda: model_dir))

    settings_store = {
        "ml_autotrain_enabled": True,
        "ml_autotrain_min_new_labels": 100,
        "ml_lab_cmd": "python -m crypto_ml.cli",
        "ml_backtest_metrics_file": "metrics.json",
        "ml_gate_min_return": 0.0,
        "ml_gate_max_drawdown": 0.25,
        "ml_gate_min_oos_sharpe": 0.5,
        "ml_gate_min_oos_trades": 20,
        "ml_gate_require_beats_baseline": False,
    }
    monkeypatch.setattr(MLTrainer, "_s",
                        staticmethod(lambda k, d=None: settings_store.get(k, d)))

    t = MLTrainer()
    t._settings = settings_store  # test handle
    return t


def _stub_export(t, monkeypatch, n_events=500, n_labeled=300):
    def fake_export(staging):
        os.makedirs(staging, exist_ok=True)
        ev = os.path.join(staging, "events.json")
        lb = os.path.join(staging, "labels.json")
        with open(ev, "w") as f:
            json.dump({"events": [1] * n_events}, f)
        with open(lb, "w") as f:
            json.dump({"labels": [1] * n_labeled}, f)
        return ev, lb, {"n_events": n_events, "n_labeled_signals": n_labeled}
    monkeypatch.setattr(t, "_export_dataset", fake_export)


def _stub_cli_success(t, monkeypatch, metrics):
    """CLI that 'succeeds' and, on the train step, writes model.joblib +
    metadata.json + a metrics file into the run dir."""
    def fake_cli(args, cwd, timeout=None):
        if args and args[0] == "train":
            run_dir = args[args.index("--output") + 1]
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "model.joblib"), "w") as f:
                f.write("MODEL")
            with open(os.path.join(run_dir, "metadata.json"), "w") as f:
                json.dump({"features": ["f1", "f2"], "horizons": [3600]}, f)
        if args and args[0] == "backtest":
            # backtest writes metrics beside the model (run dir)
            model_path = args[args.index("--model") + 1]
            run_dir = os.path.dirname(model_path)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)
        return 0, "ok", ""
    monkeypatch.setattr(t, "_run_cli", fake_cli)


# --------------------------------------------------------------------- trigger

def test_trigger_not_ready_blocks_scheduled_run(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 50))
    ok, reason = trainer.should_run()
    assert not ok
    assert "waiting_for_labels" in reason


def test_trigger_ready_when_enough_new_labels(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    ok, reason = trainer.should_run()
    assert ok and reason == "ready"


def test_disabled_never_runs(trainer, monkeypatch):
    trainer._settings["ml_autotrain_enabled"] = False
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 999))
    ok, reason = trainer.should_run()
    assert not ok and reason == "disabled"


# ------------------------------------------------------------ full pipeline

def test_promotes_when_gates_pass(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": 1.5, "max_drawdown": -0.08})

    rep = trainer.run_pipeline(force=False)
    assert rep["started"] and rep["promoted"] is True
    assert rep["gate"]["pass"] is True
    # artifact actually landed in the live model dir
    assert os.path.exists(os.path.join(trainer._model_dir(), "model.joblib"))
    assert os.path.exists(os.path.join(trainer._model_dir(), "metadata.json"))
    # trigger baseline advanced to the count captured at run start
    state = trainer._load_state()
    assert state["last_trained_label_count"] == 500


def test_negative_return_blocks_promotion(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": -0.2, "max_drawdown": -0.05})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "return" in rep["gate"]["reason"]
    assert not os.path.exists(os.path.join(trainer._model_dir(), "model.joblib"))


def test_excessive_drawdown_blocks_promotion(trainer, monkeypatch):
    # mirrors the real observed result: huge return but a 50%+ drawdown
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": 3934.5, "max_drawdown": -0.502})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "max_drawdown" in rep["gate"]["reason"]


def test_optional_sharpe_enforced_when_present(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": 1.0, "max_drawdown": -0.1, "oos_sharpe": 0.1})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "sharpe" in rep["gate"]["reason"]


def test_too_few_trades_blocks_when_present(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": 1.0, "max_drawdown": -0.1, "n_oos_trades": 5})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "trades" in rep["gate"]["reason"]


def test_not_beats_baseline_blocks_when_required(trainer, monkeypatch):
    trainer._settings["ml_gate_require_beats_baseline"] = True
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_success(trainer, monkeypatch,
                      {"total_return": 1.0, "max_drawdown": -0.1, "beats_baseline": False})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "beats_baseline" in rep["gate"]["reason"]


def test_missing_metrics_fails_closed(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)

    def cli_no_metrics(args, cwd, timeout=None):
        if args and args[0] == "train":
            run_dir = args[args.index("--output") + 1]
            os.makedirs(run_dir, exist_ok=True)
            open(os.path.join(run_dir, "model.joblib"), "w").write("M")
        # backtest writes NO metrics file and prints nothing parseable
        return 0, "ok", ""
    monkeypatch.setattr(trainer, "_run_cli", cli_no_metrics)

    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["gate"]["reason"] == "no_metrics"


def test_metrics_parsed_from_stdout_when_no_file(trainer, monkeypatch):
    """Real crypto_ml backtest prints JSON to stdout; if --output somehow wrote
    nothing, the stdout fallback still feeds the gate."""
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)

    def cli(args, cwd, timeout=None):
        if args and args[0] == "train":
            run_dir = args[args.index("--output") + 1]
            os.makedirs(run_dir, exist_ok=True)
            open(os.path.join(run_dir, "model.joblib"), "w").write("M")
            open(os.path.join(run_dir, "metadata.json"), "w").write('{"features":[]}')
        if args and args[0] == "backtest":
            return 0, '{\n  "total_return": 2.1,\n  "max_drawdown": -0.12\n}', ""
        return 0, "ok", ""
    monkeypatch.setattr(trainer, "_run_cli", cli)

    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is True
    assert rep["metrics"]["total_return"] == 2.1


def test_lab_not_installed_is_reported(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    monkeypatch.setattr(trainer, "_run_cli",
                        lambda args, cwd, timeout=None: (127, "", "not found"))
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["reason"] == "lab_not_installed"


def test_step_failure_aborts(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)

    def failing(args, cwd, timeout=None):
        if args and args[0] == "prepare":
            return 2, "", "boom"
        return 0, "ok", ""
    monkeypatch.setattr(trainer, "_run_cli", failing)
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["reason"] == "prepare_failed_rc2"


def test_no_events_aborts_before_cli(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch, n_events=0, n_labeled=0)
    # if CLI were called it'd raise; ensure it isn't
    monkeypatch.setattr(trainer, "_run_cli",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cli called")))
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["reason"] == "no_events_to_train_on"


# ------------------------------------------------------------ metric aliases

def test_metric_aliases_accepted(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    # use alternative key names for the required metrics
    _stub_cli_success(trainer, monkeypatch,
                      {"cum_return": 1.0, "max_dd": -0.05})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is True


# ------------------------------------------------------------ concurrency guard

def test_concurrent_run_blocked(trainer):
    trainer._running = True
    rep = trainer.run_pipeline(force=True)
    assert rep["started"] is False and rep["reason"] == "already_running"
