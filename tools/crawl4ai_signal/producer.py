"""crawl4ai -> LLM -> per-asset lean -> CryptoMind ingest.

Pipeline (all steps best-effort, nothing here can crash the trading core):

  1. fetch_text(url)      : get clean, LLM-ready text for a page. Uses crawl4ai's
                            AsyncWebCrawler (Apache-2.0, optional) when installed
                            — it renders JS and returns fit-markdown — else falls
                            back to a plain httpx GET + crude HTML->text strip.
  2. llm_lean(asset,text) : ask the LLM (reusing CryptoMind's own LLM-advisor
                            credential/endpoint plumbing) for a directional lean
                            in [-1, 1] + a one-line rationale.
  3. build_rows(results)  : PURE mapping of scored results -> ingest rows.
  4. run_once(sources)    : orchestrate 1->3 and push rows into app.data.ingest.

crawl4ai is NEVER imported by the trading core — only here, in this standalone
tool. See __init__.py for the required Crawl4AI attribution notice.
"""
from __future__ import annotations

import asyncio
import json
import re

SOURCE = "crawl4ai"


# ------------------------------- fetch --------------------------------------
def _strip_html(html: str) -> str:
    """Very small HTML->text fallback for when crawl4ai isn't installed."""
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_text(url: str, max_chars: int = 4000) -> str:
    """Return clean text for a URL. Prefers crawl4ai; falls back to httpx.

    Returns "" on any failure so a dead source is simply skipped.
    """
    # ---- preferred: crawl4ai (renders JS, returns clean markdown) ----
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore

        async def _crawl():
            async with AsyncWebCrawler() as crawler:
                res = await crawler.arun(url=url)
                md = getattr(res, "markdown", None)
                # newer versions expose fit_markdown for boilerplate-stripped text
                fit = getattr(md, "fit_markdown", None) if md else None
                return (fit or getattr(md, "raw_markdown", None)
                        or (md if isinstance(md, str) else "") or "")

        text = asyncio.run(_crawl())
        if text:
            return text[:max_chars]
    except Exception:
        pass
    # ---- fallback: plain fetch + strip ----
    try:
        import httpx
        r = httpx.get(url, timeout=20,
                      headers={"User-Agent": "CryptoMind-crawl4ai-signal/1.0"})
        r.raise_for_status()
        return _strip_html(r.text)[:max_chars]
    except Exception:
        return ""


# -------------------------------- LLM ---------------------------------------
def llm_lean(asset: str, text: str):
    """(lean in [-1,1], why) for an asset given crawled text. (None, "") on any
    failure. Reuses CryptoMind's LLM-advisor endpoint/credential plumbing."""
    if not text.strip():
        return None, ""
    try:
        import httpx
        from app.learn.llm_advisor import LLMAdvisor
        if not LLMAdvisor.enabled():
            return None, ""
        base = LLMAdvisor._base_url()
        model = LLMAdvisor._model()
        sys_prompt = (
            "You are a cautious crypto news analyst. Given recent web text about "
            f"the asset {asset}, respond with ONLY a JSON object "
            '{"lean": <number -1..1>, "why": "<short reason>"} where lean is the '
            "net directional implication for the asset's price over the next few "
            "hours (-1 strongly bearish … +1 strongly bullish, 0 = neutral/no "
            "clear signal). Be conservative; prefer 0 for stale or irrelevant "
            "text."
        )
        payload = {
            "model": model, "temperature": 0.2, "max_tokens": 300,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text[:6000]},
            ],
        }
        with httpx.Client(timeout=25) as c:
            r = c.post(f"{base}/chat/completions",
                       headers={"Authorization": f"Bearer {LLMAdvisor._api_key()}"},
                       json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return LLMAdvisor._parse(content)
    except Exception:
        return None, ""


# ------------------------------- mapping ------------------------------------
def build_rows(results) -> list[dict]:
    """PURE: map [(asset, lean, why), ...] -> ingest rows. Drops entries whose
    lean is None. This is the unit-tested boundary."""
    rows = []
    for asset, lean, why in results:
        if lean is None:
            continue
        try:
            val = max(-1.0, min(1.0, float(lean)))
        except (TypeError, ValueError):
            continue
        rows.append({"source": SOURCE, "asset": asset, "value": val,
                     "kind": "news_lean", "text": (why or "")[:300]})
    return rows


def _sources_from_settings() -> dict:
    try:
        from app import settings as app_settings
        raw = app_settings.get("crawl4ai_sources")
        if raw:
            d = json.loads(raw)
            if isinstance(d, dict):
                return d
    except Exception:
        pass
    return {}


def run_once(sources: dict | None = None, push: bool = True) -> dict:
    """Crawl each asset's URLs, score with the LLM, and (optionally) push the
    resulting rows into app.data.ingest. Returns a summary dict."""
    sources = sources if sources is not None else _sources_from_settings()
    if not sources:
        return {"ok": False, "error": "no crawl4ai_sources configured"}
    results = []
    for asset, urls in sources.items():
        if isinstance(urls, str):
            urls = [urls]
        blob = "\n\n".join(t for t in (fetch_text(u) for u in urls) if t)
        if not blob:
            continue
        lean, why = llm_lean(asset, blob)
        results.append((asset, lean, why))
    rows = build_rows(results)
    pushed = {"accepted": 0, "rejected": 0}
    if push and rows:
        from app.data import ingest
        pushed = ingest.push(rows)
    return {"ok": True, "scored": len(results), "rows": len(rows),
            "pushed": pushed}
