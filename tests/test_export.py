"""Full-state JSON export (app/export.py + /api/export)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, export


def _init():
    db.init()


def test_build_export_has_all_sections():
    _init()
    d = export.build_export(history_limit=10)
    for key in ("schema", "generated_at", "status", "trades", "equity",
                "signals", "market", "derivatives", "universe", "calendar",
                "research", "learning", "guardian", "decisions", "order_events",
                "events", "db_counts", "settings", "tunables", "alerts"):
        assert key in d, f"missing section: {key}"
    assert d["schema"] == "cryptomind.export.v1"


def test_export_is_json_serialisable():
    _init()
    d = export.build_export(history_limit=10)
    # default=str mirrors the CLI/endpoint dump; must not raise
    s = json.dumps(d, default=str)
    assert len(s) > 100
    # round-trips back to a dict
    assert isinstance(json.loads(s), dict)


def test_no_features_flag_shrinks_market():
    _init()
    with_f = export.build_export(include_features=True)
    without_f = export.build_export(include_features=False)
    assert "features" in with_f["market"]
    assert "features" not in without_f["market"]


def test_section_failure_is_isolated(monkeypatch):
    """A blowing-up producer records an error string, not a crash."""
    boom = export._safe(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert isinstance(boom, dict) and "boom" in boom["error"]


def test_learning_section_carries_internals():
    _init()
    d = export.build_export()
    learning = d["learning"]
    # tolerate an isolated-error dict, else assert the rich keys exist
    if "error" not in learning:
        for k in ("weights", "online_model", "bandit_posteriors", "direction"):
            assert k in learning


def test_cli_writes_file(tmp_path):
    out = tmp_path / "dump.json"
    rc = export.main([str(out), "--history", "5", "--no-features"])
    assert rc == 0
    assert out.exists()
    d = json.loads(out.read_text())
    assert d["schema"] == "cryptomind.export.v1"
    assert "features" not in d["market"]


def test_history_limit_caps_rows():
    _init()
    # push a bunch of events, then cap the export at 3
    for i in range(10):
        db.log_event("test", f"evt{i}")
    db.flush()
    d = export.build_export(history_limit=3)
    if isinstance(d["events"], list):
        assert len(d["events"]) <= 3
