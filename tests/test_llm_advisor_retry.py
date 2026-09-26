"""LLM advisor resilience: a 400 on json-mode retries without response_format,
and any HTTP error's real body is surfaced in last_error (so a 400 is actionable
instead of a vague status line)."""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from app.learn.llm_advisor import advisor


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=httpx.Request("POST", "http://x"), response=self)


class _Client:
    def __init__(self, responder):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return self._responder(json)


def _ok(content):
    return _Resp(200, {"choices": [{"message": {"content": content}}]})


def test_retries_without_json_mode_on_400(monkeypatch):
    calls = {"with_rf": 0, "without_rf": 0}

    def responder(payload):
        if "response_format" in payload:
            calls["with_rf"] += 1
            return _Resp(400, text="Invalid value at 'response_format'")
        calls["without_rf"] += 1
        return _ok('{"lean": 0.5, "why": "trend up"}')

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(responder))
    lean, why = asyncio.run(advisor._ask_model({"product": "BTC-USD"}))
    assert lean == 0.5 and why == "trend up"
    assert calls["with_rf"] == 1 and calls["without_rf"] == 1   # tried strict, then fell back
    assert advisor.last_error == ""                             # success clears the error


def test_persistent_400_surfaces_real_body(monkeypatch):
    def responder(payload):
        return _Resp(400, text="models/gemini-9 is not found")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(responder))
    lean, why = asyncio.run(advisor._ask_model({"product": "BTC-USD"}))
    assert lean is None
    assert "HTTP 400" in advisor.last_error
    assert "is not found" in advisor.last_error       # operator sees the ACTUAL reason


def test_success_parses_fenced_json(monkeypatch):
    def responder(payload):
        return _ok('```json\n{"lean": -0.7, "why": "breakdown"}\n```')
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(responder))
    lean, why = asyncio.run(advisor._ask_model({"product": "ETH-USD"}))
    assert lean == -0.7 and why == "breakdown"


def test_base_url_strips_accidental_call_path(monkeypatch):
    from app import settings as s
    # user pastes the FULL endpoint into the base URL field
    monkeypatch.setattr(s, "get", lambda k: {
        "llm_api_base": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "llm_model": "", "llm_api_key": "",
    }.get(k, ""))
    monkeypatch.delenv("CRYPTOMIND_LLM_BASE", raising=False)
    assert advisor._base_url() == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_404_error_names_model_and_url(monkeypatch):
    def responder(payload):
        return _Resp(404, text="")           # bare 404, empty body (wrong path signature)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(responder))
    lean, why = asyncio.run(advisor._ask_model({"product": "BTC-USD"}))
    assert lean is None
    assert "HTTP 404" in advisor.last_error
    assert "chat/completions" in advisor.last_error      # the actual URL is shown
    assert "not found" in advisor.last_error             # actionable hint
    assert "(empty body)" in advisor.last_error          # empty body made explicit


def test_anthropic_base_gets_v1(monkeypatch):
    from app import settings as s
    monkeypatch.delenv("CRYPTOMIND_LLM_BASE", raising=False)
    # bare host, and full-endpoint paste, both normalize to /v1
    monkeypatch.setattr(s, "get", lambda k: {"llm_api_base": "https://api.anthropic.com"}.get(k, ""))
    assert advisor._base_url() == "https://api.anthropic.com/v1"
    monkeypatch.setattr(s, "get", lambda k: {"llm_api_base": "https://api.anthropic.com/v1/chat/completions"}.get(k, ""))
    assert advisor._base_url() == "https://api.anthropic.com/v1"


def test_payload_includes_max_tokens(monkeypatch):
    seen = {}
    def responder(payload):
        seen.update(payload)
        return _ok('{"lean": 0.1, "why": "ok"}')
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(responder))
    asyncio.run(advisor._ask_model({"product": "BTC-USD"}))
    assert seen.get("max_tokens") == 512
