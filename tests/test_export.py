"""Full-state JSON export (app/export.py + /api/export)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, export


def _init():
    db.init()


ALL_SECTIONS = (
    "meta", "status", "trades", "analytics", "equity", "signals", "market",
    "per_product", "derivatives", "universe", "memes", "calendar", "research",
    "learning", "guardian", "stance", "risk", "hedge", "decisions",
    "order_events", "events", "db_trades", "signal_scores", "db_counts",
    "settings", "tunables", "alerts", "security",
)


def test_build_export_has_all_sections():
    _init()
    d = export.build_export(history_limit=10)
    for key in ALL_SECTIONS:
        assert key in d, f"missing section: {key}"
    assert d["schema"] == "cryptomind.export.v2"


def test_export_is_json_serialisable():
    _init()
    d = export.build_export(history_limit=10)
    s = json.dumps(d, default=str)
    assert len(s) > 100
    assert isinstance(json.loads(s), dict)


def test_no_features_flag_shrinks_market():
    _init()
    with_f = export.build_export(include_features=True)
    without_f = export.build_export(include_features=False)
    assert "features" in with_f["market"]
    assert "features" not in without_f["market"]
    assert "per_product" in with_f
    assert "per_product" not in without_f


def test_no_analytics_flag():
    _init()
    assert "analytics" in export.build_export(include_analytics=True)
    assert "analytics" not in export.build_export(include_analytics=False)


def test_section_failure_is_isolated():
    boom = export._safe(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert isinstance(boom, dict) and "boom" in boom["error"]


def test_learning_section_carries_internals():
    _init()
    d = export.build_export()
    learning = d["learning"]
    if "error" not in learning:
        for k in ("weights", "online_model", "bandit_posteriors", "direction"):
            assert k in learning


def test_meta_carries_config_and_products():
    _init()
    meta = export.build_export()["meta"]
    assert "products" in meta and isinstance(meta["products"], list)
    assert "config" in meta and "risk_per_trade" in meta["config"]
    assert "strategies" in meta


def test_meta_uses_effective_runtime_tunables(monkeypatch):
    _init()
    from app import tunables
    effective = {"risk_per_trade": 0.02, "fee_rate": 0.001,
                 "slippage_bps": 0}
    monkeypatch.setattr(tunables, "values", lambda: effective)

    config = export.build_export()["meta"]["config"]

    assert config["risk_per_trade"] == 0.02
    assert config["fee_rate"] == 0.001
    assert config["slippage_bps"] == 0


def test_security_omits_token_value():
    _init()
    sec = export.build_export()["security"]
    # only reports whether a token is configured, never the value
    assert "auth_token_configured" in sec
    assert not any("token" == k for k in sec if k not in
                   ("auth_token_configured",)) or "note" in sec
    # the actual token string must not leak anywhere obvious
    assert isinstance(sec["auth_token_configured"], bool)


def test_trade_analytics_math():
    now = time.time()
    closed = [
        {"product": "BTC-USD", "side": 1, "pnl": 100.0,
         "opened": now - 7200, "closed": now - 3600, "exit_reason": "take_profit"},
        {"product": "BTC-USD", "side": 1, "pnl": -50.0,
         "opened": now - 3600, "closed": now - 1800, "exit_reason": "stop_loss"},
        {"product": "ETH-USD", "side": -1, "pnl": 30.0,
         "opened": now - 900, "closed": now - 60, "exit_reason": "trail"},
    ]
    a = export._trade_analytics(closed)
    assert a["trades"] == 3
    assert a["wins"] == 2 and a["losses"] == 1
    assert a["total_pnl"] == 80.0
    assert a["gross_profit"] == 130.0
    assert a["profit_factor"] == round(130.0 / 50.0, 3)
    # per-product + per-exit breakdowns
    pp = export._per_product_trades(closed)
    assert pp["BTC-USD"]["trades"] == 2
    assert pp["ETH-USD"]["short"] == 1
    pe = export._per_exit_reason(closed)
    assert pe["take_profit"]["count"] == 1


def test_empty_analytics_is_safe():
    a = export._trade_analytics([])
    assert a["trades"] == 0


def test_cli_writes_file(tmp_path):
    out = tmp_path / "dump.json"
    rc = export.main([str(out), "--history", "5", "--no-features"])
    assert rc == 0
    assert out.exists()
    d = json.loads(out.read_text())
    assert d["schema"] == "cryptomind.export.v2"
    assert "features" not in d["market"]


def test_cli_compact(tmp_path):
    out = tmp_path / "dump.json"
    rc = export.main([str(out), "--compact", "--no-analytics"])
    assert rc == 0
    txt = out.read_text()
    assert "\n" not in txt.strip()  # minified
    assert "analytics" not in json.loads(txt)


def test_write_report_creates_timestamped_file(tmp_path):
    _init()
    path = export.write_report(directory=str(tmp_path), history_limit=10)
    assert os.path.exists(path)
    assert os.path.basename(path).startswith("cryptomind_report_")
    d = json.loads(open(path).read())
    assert d["schema"] == "cryptomind.export.v2"
    # atomic write leaves no .tmp behind
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))


def test_prune_reports_keeps_newest(tmp_path):
    for i in range(6):
        p = tmp_path / f"cryptomind_report_2020010{i}_000000.json"
        p.write_text("{}")
        time.sleep(0.01)
    removed = export.prune_reports(directory=str(tmp_path), keep=2)
    assert removed == 4
    left = [f for f in os.listdir(tmp_path) if f.startswith("cryptomind_report_")]
    assert len(left) == 2


def test_prune_reports_missing_dir_is_safe():
    assert export.prune_reports(directory="/nonexistent/x/y/z", keep=5) == 0


def test_auto_export_settings_exist_and_clamp():
    from app import settings
    settings._settings = None
    assert "auto_export_enabled" in settings.DEFAULTS
    assert settings.DEFAULTS["auto_export_interval_sec"] == 300
    # clamps to [30, 86400]
    settings.update({"auto_export_interval_sec": 5})
    assert settings.get("auto_export_interval_sec") == 30
    settings.update({"auto_export_interval_sec": 10 ** 9})
    assert settings.get("auto_export_interval_sec") == 86400
    settings.update({"auto_export_interval_sec": 450})
    assert settings.get("auto_export_interval_sec") == 450
    # garbage falls back to default
    settings.update({"auto_export_interval_sec": "nope"})
    assert settings.get("auto_export_interval_sec") == 300
    settings._settings = None


def test_history_limit_caps_rows():
    _init()
    for i in range(10):
        db.log_event("test", f"evt{i}")
    db.flush()
    d = export.build_export(history_limit=3)
    if isinstance(d["events"], list):
        assert len(d["events"]) <= 3
