"""Unit tests for the autonomous, metric-gated ML retraining pipeline.

crypto_ml + pandas are NOT installed in CI, so tests stub the lab CLI runner, the
in-process export, and (for gate tests) the purged walk-forward validator. That
isolates exactly the logic we own: the data-driven trigger, the fail-closed gate,
atomic promotion, trigger-state bookkeeping, and the validation fallback policy.
The purged-validation adapter itself is exercised separately behind importorskip.
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
        "ml_gate_min_frac_folds_positive": 0.75,
        "ml_gate_allow_backtest_fallback": False,
        "ml_gate_min_oos_sharpe": 0.5,
        "ml_gate_min_oos_trades": 20,
        "ml_gate_require_beats_baseline": False,
        "ml_val_folds": 4,
        "ml_val_embargo": 2,
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


def _stub_cli_train_ok(t, monkeypatch):
    """CLI where validate/prepare/train succeed and train writes the artifact."""
    def cli(args, cwd, timeout=None):
        if args and args[0] == "train":
            run_dir = args[args.index("--output") + 1]
            os.makedirs(run_dir, exist_ok=True)
            open(os.path.join(run_dir, "model.joblib"), "w").write("M")
            open(os.path.join(run_dir, "metadata.json"), "w").write('{"features":[]}')
        return 0, "ok", ""
    monkeypatch.setattr(t, "_run_cli", cli)


def _stub_validation(t, monkeypatch, metrics):
    monkeypatch.setattr(t, "_run_purged_validation",
                        lambda staging, feats: metrics)


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


# ------------------------------------------ full pipeline: OOS gate promotion

def test_promotes_when_oos_gates_pass(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    _stub_validation(trainer, monkeypatch,
                     {"total_return": 1.5, "max_drawdown": -0.08,
                      "oos_frac_folds_positive": 1.0, "oos_n_folds": 4})

    rep = trainer.run_pipeline(force=False)
    assert rep["started"] and rep["promoted"] is True
    assert rep["gate"]["pass"] is True
    assert rep["validation"]["source"] == "purged_walkforward"
    assert os.path.exists(os.path.join(trainer._model_dir(), "model.joblib"))
    assert os.path.exists(os.path.join(trainer._model_dir(), "metadata.json"))
    assert trainer._load_state()["last_trained_label_count"] == 500


def test_negative_return_blocks_promotion(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    _stub_validation(trainer, monkeypatch,
                     {"total_return": -0.2, "max_drawdown": -0.05,
                      "oos_frac_folds_positive": 1.0})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "return" in rep["gate"]["reason"]
    assert not os.path.exists(os.path.join(trainer._model_dir(), "model.joblib"))


def test_excessive_drawdown_blocks_promotion(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    # mirrors the real observed shape: good return but a 50%+ drawdown
    _stub_validation(trainer, monkeypatch,
                     {"total_return": 3934.5, "max_drawdown": -0.502,
                      "oos_frac_folds_positive": 1.0})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "max_drawdown" in rep["gate"]["reason"]


def test_inconsistent_folds_block_promotion(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    # good median return + low drawdown, but only half the OOS folds were positive
    _stub_validation(trainer, monkeypatch,
                     {"total_return": 0.4, "max_drawdown": -0.05,
                      "oos_frac_folds_positive": 0.5, "oos_n_folds": 4})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "frac_folds_positive" in rep["gate"]["reason"]


def test_optional_sharpe_enforced_when_present(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    _stub_validation(trainer, monkeypatch,
                     {"total_return": 1.0, "max_drawdown": -0.1,
                      "oos_frac_folds_positive": 1.0, "oos_sharpe": 0.1})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert "sharpe" in rep["gate"]["reason"]


def test_metric_aliases_accepted(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    _stub_validation(trainer, monkeypatch,
                     {"cum_return": 1.0, "max_dd": -0.05,
                      "oos_frac_folds_positive": 1.0})
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is True


# ------------------------------------------ validation availability / fallback

def test_validation_unavailable_refuses_to_promote(trainer, monkeypatch):
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)
    _stub_cli_train_ok(trainer, monkeypatch)
    _stub_validation(trainer, monkeypatch, None)  # e.g. pandas/parquet missing
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["reason"] == "validation_unavailable_not_promoted"
    assert not os.path.exists(os.path.join(trainer._model_dir(), "model.joblib"))


def test_backtest_fallback_when_explicitly_allowed(trainer, monkeypatch):
    trainer._settings["ml_gate_allow_backtest_fallback"] = True
    monkeypatch.setattr(MLTrainer, "_labeled_count", classmethod(lambda cls: 500))
    _stub_export(trainer, monkeypatch)

    def cli(args, cwd, timeout=None):
        if args and args[0] == "train":
            run_dir = args[args.index("--output") + 1]
            os.makedirs(run_dir, exist_ok=True)
            open(os.path.join(run_dir, "model.joblib"), "w").write("M")
            open(os.path.join(run_dir, "metadata.json"), "w").write('{"features":[]}')
        if args and args[0] == "backtest":
            out = args[args.index("--output") + 1]
            json.dump({"total_return": 1.2, "max_drawdown": -0.1}, open(out, "w"))
        return 0, "ok", ""
    monkeypatch.setattr(trainer, "_run_cli", cli)
    _stub_validation(trainer, monkeypatch, None)  # force fallback path

    rep = trainer.run_pipeline(force=True)
    assert rep["validation"]["source"] == "raw_backtest_fallback"
    assert rep["promoted"] is True


# ------------------------------------------------------- CLI step failures

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
    monkeypatch.setattr(trainer, "_run_cli",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cli called")))
    rep = trainer.run_pipeline(force=True)
    assert rep["promoted"] is False
    assert rep["reason"] == "no_events_to_train_on"


def test_concurrent_run_blocked(trainer):
    trainer._running = True
    rep = trainer.run_pipeline(force=True)
    assert rep["started"] is False and rep["reason"] == "already_running"


# --------------------------------- purged walk-forward adapter (needs pandas)

def test_purged_validation_trains_per_fold_and_aggregates(trainer, monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"date": range(100), "f1": range(100), "y": range(100)})
    feats = str(tmp_path / "features.parquet")
    df.to_parquet(feats)
    staging = str(tmp_path / "staging")
    os.makedirs(staging, exist_ok=True)

    seen = {"train_inputs": [], "test_inputs": []}

    def cli(args, cwd, timeout=None):
        if args[0] == "train":
            seen["train_inputs"].append(args[args.index("--input") + 1])
            rd = args[args.index("--output") + 1]
            os.makedirs(rd, exist_ok=True)
            open(os.path.join(rd, "model.joblib"), "w").write("M")
        if args[0] == "backtest":
            seen["test_inputs"].append(args[args.index("--input") + 1])
            out = args[args.index("--output") + 1]
            json.dump({"total_return": 0.1, "max_drawdown": -0.05}, open(out, "w"))
        return 0, "ok", ""
    monkeypatch.setattr(trainer, "_run_cli", cli)

    res = trainer._run_purged_validation(staging, feats)
    assert res is not None
    assert res["oos_n_folds"] == 4
    assert res["oos_frac_folds_positive"] == 1.0
    assert abs(res["total_return"] - 0.1) < 1e-9
    # each fold trained on its own train slice and scored a separate test slice
    assert len(seen["train_inputs"]) == 4 and len(seen["test_inputs"]) == 4


def test_purged_validation_none_when_fold_fails(trainer, monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"date": range(100), "f1": range(100)})
    feats = str(tmp_path / "features.parquet")
    df.to_parquet(feats)
    staging = str(tmp_path / "staging")
    os.makedirs(staging, exist_ok=True)
    # train fails => can't validate => None (caller then refuses to promote)
    monkeypatch.setattr(trainer, "_run_cli",
                        lambda args, cwd, timeout=None: (1, "", "train boom"))
    assert trainer._run_purged_validation(staging, feats) is None
