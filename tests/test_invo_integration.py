"""Invo dashboard integration: settings masking + config-driven (no-code) mapper."""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh_settings(tmp_path, monkeypatch):
    import app.settings as st
    monkeypatch.setattr(st, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(st, "SECRETS_PATH", str(tmp_path / ".secrets.json"))
    st._settings = None
    return st


def test_secret_never_written_to_settings_json(tmp_path, monkeypatch):
    """Hard guarantee: a saved token lands in the git-ignored secrets file, and
    settings.json never contains it."""
    import json
    st = _fresh_settings(tmp_path, monkeypatch)
    st.update({"invo_token": "leak-me-if-you-can", "invo_api_base": "https://x"})
    settings_txt = (tmp_path / "settings.json").read_text()
    assert "leak-me-if-you-can" not in settings_txt
    assert "invo_token" not in json.loads(settings_txt)
    secrets = json.loads((tmp_path / ".secrets.json").read_text())
    assert secrets["invo_token"] == "leak-me-if-you-can"


def test_secret_is_masked_and_not_clobbered(tmp_path, monkeypatch):
    st = _fresh_settings(tmp_path, monkeypatch)
    st.update({"invo_token": "supersecret", "invo_api_base": "https://x.test/"})
    # public() masks the token but flags it as set
    pub = st.public()
    assert pub["invo_token"] != "supersecret"
    assert pub["invo_token_set"] is True
    assert pub["invo_api_base"] == "https://x.test/"       # stored verbatim (slash trimmed at poll time)
    # an empty / masked save must NOT wipe the stored secret
    st.update({"invo_token": ""})
    assert st.get("invo_token") == "supersecret"
    st.update({"invo_token": st.MASK})
    assert st.get("invo_token") == "supersecret"
    # a real new value overwrites
    st.update({"invo_token": "rotated"})
    assert st.get("invo_token") == "rotated"


def test_float_and_int_clamps(tmp_path, monkeypatch):
    st = _fresh_settings(tmp_path, monkeypatch)
    st.update({"invo_horizon_hours": 9999, "invo_top_n": 5000, "invo_rank_decay": -3})
    assert st.get("invo_horizon_hours") == 168.0
    assert st.get("invo_top_n") == 200
    assert st.get("invo_rank_decay") == 0.0


def test_config_mapper_no_code(tmp_path, monkeypatch):
    st = _fresh_settings(tmp_path, monkeypatch)
    st.update({
        "invo_map_list": "data", "invo_map_id": "uid",
        "invo_map_positions": "pos", "invo_map_asset": "sym",
        "invo_map_side": "dir", "invo_map_long_value": "buy",
        "invo_map_size": "usd", "invo_map_score": "wr",
    })
    import app.data.invo as invo
    importlib.reload(invo)   # pick up patched settings module
    raw = {"data": [
        {"uid": "alice", "wr": 0.8, "pos": [
            {"sym": "btc", "dir": "buy", "usd": 10000},
            {"sym": "sol", "dir": "sell", "usd": 5000}]},
        {"uid": "bob", "wr": 0.6, "pos": [
            {"sym": "eth", "dir": "buy", "usd": 2000}]},
    ]}
    traders = invo.config_mapper(raw, None, top_n=25)
    assert len(traders) == 2
    assert traders[0]["id"] == "alice" and traders[0]["rank"] == 1
    assert traders[0]["score"] == 0.8
    btc = [p for p in traders[0]["positions"] if p["asset"] == "BTC"][0]
    sol = [p for p in traders[0]["positions"] if p["asset"] == "SOL"][0]
    assert btc["direction"] == 1 and btc["notional"] == 10000
    assert sol["direction"] == -1        # "sell" != long_value "buy"


def test_config_mapper_inline_single_trade_feed(tmp_path, monkeypatch):
    """The fire_moves feed exposes ONE inline trade per item (under 'update')
    with a BOOLEAN directionLong, not a positions list. The mapper must treat a
    dict position field as a single position and read the boolean side."""
    st = _fresh_settings(tmp_path, monkeypatch)
    st.update({
        "invo_map_list": "items", "invo_map_id": "owner.id",
        "invo_map_score": "update.portfolio.winRate",
        "invo_map_positions": "update", "invo_map_asset": "name",
        "invo_map_side": "directionLong", "invo_map_long_value": "true",
        "invo_map_size": "entrySize", "invo_map_leverage": "leverage",
    })
    import app.data.invo as invo
    importlib.reload(invo)
    raw = {"items": [
        {"id": "p1", "owner": {"id": "kstn"},
         "update": {"name": "PUMP", "directionLong": False, "entrySize": 5.0,
                    "leverage": 7, "portfolio": {"winRate": 98.5}}},
        {"id": "p2", "owner": {"id": "xyz"},
         "update": {"name": "BTC", "directionLong": True, "entrySize": 12.3,
                    "leverage": 3, "portfolio": {"winRate": 61.0}}},
    ]}
    traders = invo.config_mapper(raw, None, top_n=25)
    assert len(traders) == 2
    assert traders[0]["id"] == "kstn" and traders[0]["score"] == 98.5
    p = traders[0]["positions"][0]
    assert p["asset"] == "PUMP" and p["direction"] == -1   # directionLong false = short
    assert p["notional"] == 5.0
    assert traders[1]["positions"][0]["direction"] == 1    # directionLong true = long


def test_dig_helper_dotted_paths():
    import app.data.invo as invo
    assert invo._dig({"a": {"b": {"c": 7}}}, "a.b.c") == 7
    assert invo._dig({"a": 1}, "x.y") is None
    assert invo._as_list({"results": [1, 2]}, "") == [1, 2]
    assert invo._as_list([9], "") == [9]
