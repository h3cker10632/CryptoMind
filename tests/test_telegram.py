"""Tests for the two-way Telegram command bot + rich trade notifications."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import alerts
from app.risk.manager import risk


def test_authorized_only_matches_configured_chat():
    alerts._state["telegram_chat_id"] = "555"
    assert alerts._authorized("555") is True
    assert alerts._authorized(555) is True          # int form
    assert alerts._authorized("111") is False
    assert alerts._authorized("") is False


def test_help_lists_key_commands():
    h = alerts.handle_command("/help")
    for c in ("/status", "/positions", "/resetkill", "/kill", "/set"):
        assert c in h


def test_unknown_command():
    r = alerts.handle_command("/nope")
    assert "Unknown command" in r


def test_command_strips_botname_suffix():
    # Telegram appends @BotName in groups
    r = alerts.handle_command("/status@CryptoMindBot")
    assert "status" in r.lower()


def test_set_and_get_tunable_roundtrip():
    alerts.handle_command("/set max_drawdown_kill 0.25")
    r = alerts.handle_command("/get max_drawdown_kill")
    assert "0.25" in r
    # restore default
    alerts.handle_command("/set max_drawdown_kill 0.15")


def test_set_rejects_unknown_tunable():
    r = alerts.handle_command("/set not_a_real_knob 5")
    assert "Unknown tunable" in r


def test_resetkill_clears_kill():
    risk.trip_kill("test kill")
    assert risk.killed is True
    r = alerts.handle_command("/resetkill")
    assert risk.killed is False
    assert "cleared" in r.lower() or "resumed" in r.lower()


def test_resume_blocked_while_killed():
    risk.trip_kill("test kill")
    r = alerts.handle_command("/resume")
    assert "resetkill" in r.lower()
    risk.reset_kill()


def test_shorts_and_mode_toggles():
    assert "DISABLED" in alerts.handle_command("/shorts off")
    assert "ENABLED" in alerts.handle_command("/shorts on")
    assert "aggressive" in alerts.handle_command("/mode aggressive")
    alerts.handle_command("/mode auto")


def test_notify_toggle_updates_state():
    alerts.handle_command("/notify off")
    assert alerts._state["push_trades"] is False
    alerts.handle_command("/notify on")
    assert alerts._state["push_trades"] is True


def test_trade_open_notification_contains_details():
    captured = {}
    orig = alerts._push_now
    alerts._push_now = lambda level, title, msg, raw=False: captured.setdefault("msg", msg)
    try:
        alerts._state["push_trades"] = True
        alerts.notify_trade_open({
            "product": "BTC-USD", "side": 1, "entry": 100.0, "qty": 1.0,
            "stop": 90.0, "take": 130.0, "reason": "test",
            "votes": {"trend": 0.5}})
        m = captured["msg"]
        assert "OPENED" in m and "BTC-USD" in m and "Stop" in m and "R:R" in m
    finally:
        alerts._push_now = orig


def test_trade_close_notification_contains_pnl():
    captured = {}
    orig = alerts._push_now
    alerts._push_now = lambda level, title, msg, raw=False: captured.setdefault("msg", msg)
    try:
        alerts._state["push_trades"] = True
        alerts.notify_trade_close({
            "product": "BTC-USD", "side": 1, "entry": 100.0, "exit": 120.0,
            "qty": 1.0, "pnl": 20.0, "opened": 0, "closed": 3600,
            "exit_reason": "take-profit"})
        m = captured["msg"]
        assert "CLOSED" in m and "PnL" in m and "%" in m
    finally:
        alerts._push_now = orig


def test_push_trades_off_suppresses_notification():
    captured = []
    orig = alerts._push_now
    alerts._push_now = lambda *a, **k: captured.append(1)
    try:
        alerts._state["push_trades"] = False
        alerts.notify_trade_open({"product": "X", "side": 1, "entry": 1.0,
                                  "qty": 1.0, "stop": 0.9, "take": 1.3})
        assert not captured          # suppressed
    finally:
        alerts._push_now = orig
        alerts._state["push_trades"] = True


def test_render_equity_png_empty_returns_message(tmp_path, monkeypatch):
    # with no equity history the renderer returns (None, message)
    import app.db as _db
    monkeypatch.setattr(_db, "equity_since", lambda *a, **k: [])
    png, msg = alerts.render_equity_png("1d")
    assert png is None and "history" in msg.lower()


def test_render_equity_png_produces_valid_png(monkeypatch):
    import app.db as _db, time
    now = time.time()
    fake = [{"ts": now - (60 - i) * 60, "equity": 100000 + i * 50}
            for i in range(60)]
    monkeypatch.setattr(_db, "equity_since", lambda *a, **k: fake)
    png, cap = alerts.render_equity_png("1d")
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"      # PNG magic bytes
    assert "Equity" in cap


def test_diag_reports_kill_block():
    from app.risk.manager import risk
    risk.trip_kill("test kill")
    out = alerts.handle_command("/diag")
    assert "KILLED" in out
    assert "kill switch active" in out.lower()
    risk.reset_kill()


def test_diag_lists_all_products():
    out = alerts.handle_command("/diag")
    from app.config import PRODUCTS
    for p in PRODUCTS:
        assert p in out


def test_diag_in_help():
    assert "/diag" in alerts.handle_command("/help")


# ------------------------------------------------------------------ /close
def _set_price(product, price):
    """Stub the market ticker so market.price(product) returns `price`."""
    from app.data.market import market
    market.tickers[product] = {"price": price}


def _open_test_position(product="BTC-USD", price=100.0, direction=1):
    """Open a position via the broker and stub the market price feed."""
    from app.execution.paper import broker
    broker.cash = 1_000_000.0
    broker.positions.pop(product, None)
    _set_price(product, price)
    broker.open(product, direction, 1000.0, price, price * 0.9, price * 1.3,
                "test setup")
    return broker


def test_close_in_help():
    assert "/close" in alerts.handle_command("/help")


def test_close_no_positions():
    from app.execution.paper import broker
    broker.positions.clear()
    out = alerts.handle_command("/close BTC")
    assert "No open position" in out


def test_close_no_arg_lists_open():
    _open_test_position("ETH-USD", 50.0)
    out = alerts.handle_command("/close")
    assert "ETH-USD" in out and "Usage" in out
    from app.execution.paper import broker
    broker.positions.clear()


def test_close_resolves_loose_token_and_flattens():
    from app.execution.paper import broker
    from app.data.market import market
    _open_test_position("BTC-USD", 100.0)
    # bump the price so there is a positive PnL
    _set_price("BTC-USD", 110.0)
    out = alerts.handle_command("/close btc")     # loose lowercase token
    assert "Closed" in out and "BTC-USD" in out
    assert "BTC-USD" not in broker.positions


def test_close_unknown_product_reports_open_set():
    _open_test_position("BTC-USD", 100.0)
    out = alerts.handle_command("/close DOGE")
    assert "No open position matching" in out and "BTC-USD" in out
    from app.execution.paper import broker
    broker.positions.clear()


def test_close_all_flattens_everything():
    from app.execution.paper import broker
    from app.data.market import market
    broker.positions.clear()
    _open_test_position("BTC-USD", 100.0)
    _open_test_position("ETH-USD", 50.0)
    _set_price("BTC-USD", 100.0); _set_price("ETH-USD", 50.0)
    out = alerts.handle_command("/close all")
    assert "Closed" in out
    assert not broker.positions


# ------------------------------------------------------------------ /brains
def test_brains_reports_learning_state():
    out = alerts.handle_command("/brains")
    assert "learning state" in out.lower()
    # the learning subsystems + durable store are all surfaced, including the
    # Phase 1/2 additions (concept-drift + GA portfolios).
    for section in ("Bandit", "Online model", "RL risk", "Concept drift",
                    "GA evolution", "Durable history"):
        assert section in out, section
    assert "Page-Hinkley" in out
    assert "committee" in out.lower() and "Members:" in out
    # tells the operator WHERE it's stored
    assert "cryptomind.db" in out and "state.json" in out


def test_brains_aliases_and_help():
    assert "/brains" in alerts.handle_command("/help")
    for alias in ("/brains", "/learning", "/learn"):
        assert "learning state" in alerts.handle_command(alias).lower()


def test_env_seeded_credentials_persist_to_disk(tmp_path, monkeypatch):
    """A bot token seeded from an env var must be written to alerts.json so it
    survives a restart WITHOUT the env var (the persistence-gap fix)."""
    import json
    conf = tmp_path / "alerts.json"
    monkeypatch.setattr(alerts, "CONF_PATH", str(conf), raising=False)
    # simulate env-seeded credentials with an empty/absent file
    monkeypatch.setitem(alerts._state, "telegram_bot_token", "111:ENVTOKEN")
    monkeypatch.setitem(alerts._state, "telegram_chat_id", "777")
    alerts._load_conf()
    assert conf.exists(), "credentials were never persisted to disk"
    saved = json.loads(conf.read_text())
    assert saved.get("telegram_bot_token") == "111:ENVTOKEN"
    assert saved.get("telegram_chat_id") == "777"
