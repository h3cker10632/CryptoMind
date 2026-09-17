"""Tier-2 LLM advisor: opt-in, inert-by-default, one bandit-weighted vote.

All offline: no real network calls. We test the gating, cache TTL, defensive
parsing, and that the advisor participates as the `llm` strategy only when it
has a fresh opinion.
"""
import os, sys, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.learn.llm_advisor import LLMAdvisor
from app import settings, tunables
from app.signals.engine import STRATEGIES


def test_llm_registered_as_strategy():
    assert "llm" in STRATEGIES


def test_disabled_by_default_returns_zero():
    settings.update({"llm_advisor_enabled": False})
    a = LLMAdvisor()
    a.set_lean("BTC-USD", 0.9)
    assert a.lean("BTC-USD") == 0.0        # disabled → silent regardless of cache
    assert a.enabled() is False


def test_enabled_returns_cached_lean():
    settings.update({"llm_advisor_enabled": True})
    a = LLMAdvisor()
    a.set_lean("BTC-USD", 0.7, "uptrend")
    assert abs(a.lean("BTC-USD") - 0.7) < 1e-9
    settings.update({"llm_advisor_enabled": False})


def test_lean_expires_after_ttl():
    settings.update({"llm_advisor_enabled": True})
    tunables.update({"llm_lean_ttl_sec": 300})
    a = LLMAdvisor()
    a.set_lean("ETH-USD", 0.5)
    a._leans["ETH-USD"]["ts"] = time.time() - 9999    # stale
    assert a.lean("ETH-USD") == 0.0
    settings.update({"llm_advisor_enabled": False})


def test_lean_clamped():
    settings.update({"llm_advisor_enabled": True})
    a = LLMAdvisor()
    a.set_lean("SOL-USD", 5.0)             # out of range
    assert a.lean("SOL-USD") == 1.0
    a.set_lean("SOL-USD", -5.0)
    assert a.lean("SOL-USD") == -1.0
    settings.update({"llm_advisor_enabled": False})


def test_parse_tolerates_prose_and_fences():
    p = LLMAdvisor._parse
    assert p('```json\n{"lean": 0.4, "why": "x"}\n```')[0] == 0.4
    assert p('Sure! {"lean": -0.6, "why": "downtrend"} hope that helps')[0] == -0.6
    assert p("no json here") == (None, "")
    assert p('{"nope": 1}') == (None, "")
    assert p('{"lean": "bad"}') == (None, "")


def test_not_configured_without_key(monkeypatch):
    settings.update({"llm_advisor_enabled": True})
    monkeypatch.delenv("CRYPTOMIND_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = LLMAdvisor()
    assert a.configured() is False
    # refresh no-ops (returns 0) when unconfigured — never raises
    assert asyncio.run(a.refresh(None, None, ["BTC-USD"])) == 0
    settings.update({"llm_advisor_enabled": False})


def test_stats_shape():
    a = LLMAdvisor()
    a.set_lean("BTC-USD", 0.3)
    st = a.stats()
    assert set(["enabled", "configured", "cached_leans", "calls"]) <= set(st)


def test_telegram_llm_command_toggles_setting(monkeypatch):
    import app.alerts as alerts
    monkeypatch.delenv("CRYPTOMIND_LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings.update({"llm_advisor_enabled": False})
    # usage/status when no arg
    assert "Usage: /llm" in alerts._cmd_llm("")
    # turning on with no key warns it's inert but flips the setting
    msg = alerts._cmd_llm("on")
    assert settings.get("llm_advisor_enabled") is True
    assert "no API key" in msg
    # off again
    alerts._cmd_llm("off")
    assert settings.get("llm_advisor_enabled") is False
    # reachable through the dispatcher too
    assert "LLM advisor" in alerts.handle_command("/llm")
