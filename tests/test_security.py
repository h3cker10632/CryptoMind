"""Control-API auth guard tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import security


class _Req:
    def __init__(self, host, headers=None, query=None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}
        self.query_params = query or {}


def test_loopback_allowed_by_default():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "1"
    assert security.check(_Req("127.0.0.1")) is True


def test_remote_without_token_denied():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    assert security.check(_Req("203.0.113.9")) is False


def test_remote_with_correct_token_allowed():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    tok = security.token()
    assert security.check(_Req("203.0.113.9",
                              headers={"authorization": f"Bearer {tok}"})) is True


def test_wrong_token_denied():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    assert security.check(_Req("203.0.113.9",
                              headers={"authorization": "Bearer nope"})) is False


def test_settings_never_leak_secrets_and_read_false_strings_as_false(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import db, export, main, settings
    monkeypatch.setattr(settings, "SECRETS_PATH", str(tmp_path / ".secrets.json"))
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(settings, "_settings", None)   # restored after: no key leaks to other tests
    settings.update({"llm_api_key": "sk-test-secret", "allow_shorts": "false"})
    assert settings.get("allow_shorts") is False
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "1"
    r = TestClient(main.app, client=("127.0.0.1", 5000)).post(
        "/api/settings", json={"llm_api_key": "sk-test-secret2"})
    assert "sk-test-secret" not in r.text
    db.flush()
    assert "sk-test-secret" not in str(db.recent("events", 50))
    assert "sk-test-secret" not in str(export.build_export(history_limit=5)["settings"])
