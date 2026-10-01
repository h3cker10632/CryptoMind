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
parse) so there is exactly one place that knows how to talk to the model.
Evidence-conditioned queries use a bounded executor so their provider timeout
cannot delay the quote cycle. With the advisor disabled or no key, every lean is
0 and inert.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import time

import httpx

from ...learn.llm_advisor import LLMAdvisor, advisor as _crypto_advisor


class PMLLMAdvisor:
    def __init__(self):
        self._research_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="pm-research-llm")
        self._research_pending: dict[str, tuple[Future, tuple[str, ...]]] = {}
        self._leans: dict[str, dict] = {}     # condition_id -> {lean, ts, why}
        self._research_leans: dict[str, dict] = {}
        self.calls = 0
        self.research_calls = 0
        self.research_abstentions = 0
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

    @staticmethod
    def _parse_research_response(content, allowed_ids):
        """Parse bounded JSON and discard citations not present in supplied evidence."""
        try:
            result = json.loads(content)
            lean = float(result["lean"])
            if not isinstance(result, dict) or not isinstance(result.get("citations"), list):
                raise ValueError("missing structured fields")
            if not (-1.0 <= lean <= 1.0):
                raise ValueError("lean outside bounds")
            citations = list(dict.fromkeys(
                str(value) for value in result["citations"]
                if str(value) in allowed_ids))
            if not citations:
                return {"lean": 0.0, "citations": [], "why": "",
                        "abstain": True}
            return {"lean": lean, "citations": citations[:8],
                    "why": str(result.get("why") or "")[:200],
                    "abstain": False}
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return {"lean": 0.0, "citations": [], "why": "",
                    "abstain": True}

    def research_lean(self, market: dict, evidence: list[dict]) -> dict:
        """Return a research lean grounded only in fresh evidence supplied by the caller."""
        cid = str(market.get("condition_id") or "")
        rows = [row for row in evidence if row.get("id")]
        evidence_ids = tuple(sorted({str(row["id"]) for row in rows}))
        now = time.time()
        cached = self._research_leans.get(cid)
        if (cached and cached.get("evidence_ids") == evidence_ids
                and now - cached.get("ts", 0.0) <= 1800):
            return dict(cached["result"])
        if not rows or not self.configured():
            result = {"lean": 0.0, "citations": [], "why": "", "abstain": True}
        else:
            try:
                self.research_calls += 1
                content = self._request_research(market, rows)
                result = self._parse_research_response(content, set(evidence_ids))
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Research advisor: {exc}"[:200]
                result = {"lean": 0.0, "citations": [], "why": "",
                          "abstain": True}
        if result["abstain"]:
            self.research_abstentions += 1
        self._research_leans[cid] = {
            "evidence_ids": evidence_ids, "ts": now, "result": dict(result)}
        return result

    def refresh_research(self, markets, research_cache, max_queries: int) -> int:
        """Harvest completed leans and schedule bounded provider work without waiting."""
        completed = 0
        for cid, (future, evidence_ids) in list(self._research_pending.items()):
            if not future.done():
                continue
            del self._research_pending[cid]
            try:
                result = self._parse_research_response(
                    future.result(), set(evidence_ids))
                self.last_error = ""
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Research advisor: {exc}"[:200]
                result = {"lean": 0.0, "citations": [], "why": "",
                          "abstain": True}
            if result["abstain"]:
                self.research_abstentions += 1
            self._research_leans[cid] = {
                "evidence_ids": evidence_ids, "ts": time.time(),
                "result": dict(result),
            }
            completed += 1

        if not self.configured() or max_queries <= 0:
            return completed
        scheduled = 0
        for market in markets:
            cid = str(market.get("condition_id") or "")
            if not cid or cid in self._research_pending:
                continue
            evidence = research_cache.evidence(cid)
            cached = self._research_leans.get(cid)
            ids = tuple(sorted({str(row.get("id")) for row in evidence
                                if row.get("id")}))
            if (cached and cached.get("evidence_ids") == ids
                    and time.time() - cached.get("ts", 0.0) <= 1800):
                continue
            if not evidence:
                continue
            if scheduled >= int(max_queries) or len(self._research_pending) >= 4:
                break
            self._research_pending[cid] = (
                self._research_executor.submit(
                    self._request_research, dict(market), list(evidence)), ids)
            self.research_calls += 1
            scheduled += 1
        return completed

    def research_result(self, condition_id: str, evidence=None) -> dict:
        entry = self._research_leans.get(str(condition_id))
        if not entry or time.time() - entry.get("ts", 0.0) > 1800:
            return {"lean": 0.0, "citations": [], "why": "", "abstain": True}
        if evidence is not None:
            evidence_ids = tuple(sorted({str(row.get("id")) for row in evidence
                                         if row.get("id")}))
            if entry.get("evidence_ids") != evidence_ids:
                return {"lean": 0.0, "citations": [], "why": "", "abstain": True}
        return dict(entry["result"])

    def _request_research(self, market: dict, evidence: list[dict]) -> str:
        """Ask for a bounded evidence-conditioned lean; never treat model text as a source."""
        from ...tunables import tv
        ctx = {
            "question": str(market.get("question") or "")[:280],
            "outcomes": market.get("outcomes"),
            "market_prices": [round(p, 3) for p in market.get("prices", [])],
            "resolves_in_hours": (round(market["ttl_hours"], 1)
                                  if market.get("ttl_hours") is not None else None),
            "category": market.get("category"),
            "evidence": [{key: str(row.get(key) or "")[:500]
                          for key in ("id", "publisher", "url", "published_at",
                                      "title", "excerpt")}
                         for row in evidence[:8]],
        }
        sys_prompt = (
            "You are a conservative prediction-market research advisor. Use only "
            "the supplied market and evidence. Do not invent facts. Estimate how "
            "mispriced outcome 0 is and respond only as JSON: "
            '{"lean": number from -1 to 1, "citations": [evidence IDs], '
            '"why": "brief explanation"}. Positive leans favor outcome 0; negative '
            "leans favor outcome 1. Cite only supplied evidence IDs. Return zero "
            "and an empty citation list when evidence does not support a view."
        )
        payload = {
            "model": LLMAdvisor._model(), "temperature": 0.1, "max_tokens": 350,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(ctx)},
            ],
        }
        url = f"{LLMAdvisor._base_url()}/chat/completions"
        with httpx.Client(timeout=20) as client:
            response = client.post(url, headers={
                "Authorization": f"Bearer {LLMAdvisor._api_key()}"}, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

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
            "research_calls": self.research_calls,
            "research_abstentions": self.research_abstentions,
            "cached": len(self._leans),
            "research_cached": len(self._research_leans),
            "research_pending": len(self._research_pending),
            "last_error": self.last_error,
        }

    def close(self):
        self._research_executor.shutdown(wait=False, cancel_futures=True)


llm_advisor = PMLLMAdvisor()
