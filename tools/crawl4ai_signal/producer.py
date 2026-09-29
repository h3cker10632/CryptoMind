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
from urllib.parse import quote_plus

SOURCE = "crawl4ai"

# Symbol -> human search term, so a discovery query reads like a person would
# type it ("Bitcoin crypto news", not "BTC-USD"). Unmapped symbols fall back to
# the bare ticker base.
_ASSET_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "DOGE": "Dogecoin",
    "LINK": "Chainlink", "AVAX": "Avalanche", "XRP": "XRP", "ADA": "Cardano",
    "SHIB": "Shiba Inu", "PEPE": "Pepe coin", "BONK": "Bonk token",
    "WIF": "dogwifhat", "FLOKI": "Floki coin", "MATIC": "Polygon crypto",
}


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


# ---------------------------- source discovery ------------------------------
def asset_query(symbol: str) -> str:
    """A natural news-search query for a trading symbol ('BTC-USD' -> 'Bitcoin
    crypto')."""
    base = str(symbol).split("-")[0].split("/")[0].upper()
    return f"{_ASSET_NAMES.get(base, base)} crypto"


def discover_headlines(query: str, max_items: int = 6, within_days: int = 1):
    """Discover fresh article links for a query via Google News RSS — a public,
    key-less feed. Returns [(title, link, snippet), ...] (newest first), or [] on
    any failure. This is the "let the system find the websites" step: no source
    list is configured by hand.

    The RSS already carries headlines + snippets, so scoring works even when the
    article pages themselves can't be fetched (Google's redirect links can be
    JS-gated); crawling the links just ENRICHES an already-usable signal.
    """
    try:
        import httpx
        import xml.etree.ElementTree as ET
        q = quote_plus(f"{query} when:{max(1, int(within_days))}d")
        url = (f"https://news.google.com/rss/search?q={q}"
               "&hl=en-US&gl=US&ceid=US:en")
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "CryptoMind-crawl4ai-signal/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        out = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            if title and link:
                out.append((title, link, desc[:200]))
            if len(out) >= max_items:
                break
        return out
    except Exception:
        return []


def discover_sources(assets, max_urls: int = 4, within_days: int = 1) -> dict:
    """{asset: [urls]} auto-discovered from Google News RSS for each asset."""
    found = {}
    for a in assets:
        items = discover_headlines(asset_query(a), max_items=max_urls,
                                   within_days=within_days)
        if items:
            found[a] = [link for _t, link, _s in items]
    return found


def _score_asset(asset: str, urls, headline_blob: str = ""):
    """Build the text blob for one asset (discovered headlines + best-effort
    crawled article text) and return (asset, lean, why)."""
    crawled = "\n\n".join(t for t in (fetch_text(u) for u in (urls or [])[:4]) if t)
    blob = "\n\n".join(x for x in (headline_blob, crawled) if x).strip()
    if not blob:
        return None
    lean, why = llm_lean(asset, blob)
    return (str(asset).split("-")[0].upper(), lean, why)


def run_once(sources: dict | None = None, push: bool = True,
             autodiscover: bool | None = None, assets=None) -> dict:
    """Crawl/score sources, then (optionally) push rows into app.data.ingest.

    Sources are the union of:
      * MANUAL: the `crawl4ai_sources` setting (or the `sources` arg) — a
        {asset: [urls]} map you curated.
      * AUTO-DISCOVERED: when `crawl4ai_autodiscover` is on (or autodiscover=True),
        the producer finds fresh news links itself (Google News RSS) for each
        traded asset — no hand-curated list needed. Discovered headlines are
        scored even if the article page can't be fetched.

    Either or both can be active; manual URLs and discovered URLs for the same
    asset are merged.
    """
    from app import settings as app_settings
    manual = sources if sources is not None else _sources_from_settings()
    manual = manual or {}
    if autodiscover is None:
        try:
            autodiscover = bool(app_settings.get("crawl4ai_autodiscover"))
        except Exception:
            autodiscover = False

    # headlines discovered per asset (kept separate so they can seed the blob)
    discovered_headlines = {}
    if autodiscover:
        if assets is None:
            try:
                from app.config import PRODUCTS
                assets = list(PRODUCTS)
            except Exception:
                assets = []
        # discover for the traded universe plus any asset named in manual sources
        for a in list(assets) + [k for k in manual if k not in assets]:
            try:
                mx = int(app_settings.get("crawl4ai_max_urls"))
            except Exception:
                mx = 4
            items = discover_headlines(asset_query(a), max_items=mx)
            if items:
                discovered_headlines[a] = "\n".join(f"- {t}: {s}" for t, _l, s in items)
                manual.setdefault(a, [])
                manual[a] = list(manual[a]) + [l for _t, l, _s in items]

    if not manual:
        return {"ok": False, "error": "no sources: set crawl4ai_sources or "
                "enable crawl4ai_autodiscover"}

    results = []
    for asset, urls in manual.items():
        if isinstance(urls, str):
            urls = [urls]
        scored = _score_asset(asset, urls, discovered_headlines.get(asset, ""))
        if scored:
            results.append(scored)
    rows = build_rows(results)
    pushed = {"accepted": 0, "rejected": 0}
    if push and rows:
        from app.data import ingest
        pushed = ingest.push(rows)
    return {"ok": True, "scored": len(results), "rows": len(rows),
            "autodiscovered": len(discovered_headlines), "pushed": pushed}
