"""LLM advisor for Polymarket — a prediction-market read from the same model the
crypto advisor uses, added as ONE more bandit-weighted strategy.

Prediction markets are natural-language questions ("Will X happen by date?"), so
an LLM's judgement is a genuinely relevant input here — arguably more so than on
a price chart. This advisor asks the model for a directional lean on the FIRST
listed outcome, caches it per market with a TTL, and hands it to the signal
engine as the `llm` strategy. The Polymarket bandit then weights it from realized
resolutions exactly like the four heuristics; the operator's `pm_llm_influence`
lever raises its baseline voice without overriding what the bandit has learned.

It reuses the crypto LLMAdvisor's credential / endpoint / parsing plumbing (key
resolution, base-URL normalization, provider-agnostic request, defensive JSON
parse) so there is exactly one place that knows how to talk to the model. Queries
are SYNCHRONOUS (the engine tick runs in a worker thread), bounded per cycle, and
fully optional: with the advisor disabled or no key, every lean is 0 and inert.
"""
from __future__ import annotations

import json
import time

import httpx

from ...learn.llm_advisor import LLMAdvisor, advisor as _crypto_advisor


class PMLLMAdvisor:
    def __init__(self):
        self._leans: dict[str, dict] = {}     # condition_id -> {lean, ts, why}
        self.calls = 0
        self.last_error = ""
        self.last_refresh = 0.0

    def configured(self) -> bool:
        # reuse the crypto advisor's enabled + key resolution exactly
        return _crypto_advisor.configured()

    def lean(self, condition_id: str) -> float:
        """Cached lean in [-1, 1], signed toward OUTCOME 0. 0.0 when missing or
        expired (TTL borrowed from the crypto advisor's llm_lean_ttl_sec)."""
        e = self._leans.get(condition_id)
        if not e:
            return 0.0
        from ...tunables import tv
        if time.time() - e["ts"] > float(tv("llm_lean_ttl_sec")):
            return 0.0
        return max(-1.0, min(1.0, e["lean"]))

    def why(self, condition_id: str) -> str:
        e = self._leans.get(condition_id)
        return e["why"] if e else ""

    def set_lean(self, condition_id, lean, why=""):
        self._leans[condition_id] = {"lean": max(-1.0, min(1.0, float(lean))),
                                     "ts": time.time(), "why": str(why)[:200]}

    def refresh(self, markets, max_queries: int) -> int:
        """Best-effort sync refresh for up to `max_queries` uncached/expired
        markets. Returns how many leans were updated. Any failure is swallowed
        (recorded on last_error) and leaves prior leans in place."""
        if not self.configured() or max_queries <= 0:
            return 0
        from ...tunables import tv
        ttl = float(tv("llm_lean_ttl_sec"))
        now = time.time()
        updated = 0
        for m in markets:
            if updated >= max_queries:
                break
            cid = m.get("condition_id")
            if not cid:
                continue
            e = self._leans.get(cid)
            if e and now - e["ts"] <= ttl:
                continue                      # still fresh
            lean, why = self._ask(m)
            if lean is not None:
                self.set_lean(cid, lean, why)
                updated += 1
        self.last_refresh = now
        return updated

    def _ask(self, m: dict):
        """One synchronous model query for a market. Returns (lean, why) with
        lean signed toward outcome 0, or (None, "") on any error."""
        self.calls += 1
        base = LLMAdvisor._base_url()
        model = LLMAdvisor._model()
        ctx = {
            "question": m.get("question", "")[:280],
            "outcomes": m.get("outcomes"),
            "market_prices": [round(p, 3) for p in m.get("prices", [])],
            "resolves_in_hours": (round(m["ttl_hours"], 1)
                                  if m.get("ttl_hours") is not None else None),
            "category": m.get("category"),
        }
        sys_prompt = (
            "You are a calibrated prediction-market forecaster. You are given a "
            "binary market: a question, its two OUTCOMES, and the market-implied "
            "probabilities (market_prices, index-aligned to outcomes, summing to "
            "~1). Estimate the TRUE probability of OUTCOME 0 (the first listed) "
            "and respond with ONLY a JSON object "
            '{"lean": <number -1..1>, "why": "<short reason>"} where lean is how '
            "MISPRICED outcome 0 looks: +1 = outcome 0 is badly UNDERvalued (buy "
            "it), -1 = badly OVERvalued (buy outcome 1), 0 = fairly priced / no "
            "edge. Be conservative; prefer values near 0 unless you have a real, "
            "defensible reason to disagree with the market."
        )
        payload = {
            "model": model, "temperature": 0.2, "max_tokens": 400,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(ctx)},
            ],
        }
        url = f"{base}/chat/completions"
        try:
            with httpx.Client(timeout=20) as c:
                r = c.post(url, headers={
                    "Authorization": f"Bearer {LLMAdvisor._api_key()}"},
                    json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                self.last_error = ""
                return LLMAdvisor._parse(content)      # tolerant {lean, why}
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:200]
            except Exception:
                body = ""
            self.last_error = f"HTTP {e.response.status_code} at {url}: {body}"
            return None, ""
        except Exception as e:                          # noqa: BLE001
            self.last_error = str(e)[:200]
            return None, ""

    def stats(self):
        return {
            "enabled": self.configured(),
            "calls": self.calls,
            "cached": len(self._leans),
            "last_error": self.last_error,
        }


llm_advisor = PMLLMAdvisor()
