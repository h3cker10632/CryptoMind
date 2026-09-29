# crawl4ai signal producer

A **standalone** producer that turns crawled web content into a **measured**
external signal for CryptoMind. It crawls operator-configured sources, condenses
each into a per-asset directional lean with an LLM, and pushes the result into
CryptoMind's generic ingest seam (`app.data.ingest`).

## Why it's separate from the core

- `crawl4ai` (Apache-2.0) is an **optional** dependency used **only here**. The
  trading core never imports it.
- The signal it produces influences trades **only** through the measured,
  bandit-weighted LLM-advisor context (`llm_web_context`) — never a blind copy.
- Because CryptoMind ingests **output rows**, not producer code, you can swap in
  any other producer (even a copyleft one like Maxun) with zero license reach.

## Setup

```bash
pip install crawl4ai        # optional; falls back to httpx + HTML strip if absent
python -m playwright install chromium   # crawl4ai uses Playwright to render JS
```

### Where the sources come from — two ways

**A. Auto-discovery (let the system find the websites).** Turn on
`crawl4ai_autodiscover` (dashboard: *auto-find sources*). For each traded asset
the producer queries **Google News RSS** (public, no API key) for fresh articles,
scores the discovered headlines (+ article text when fetchable), and pushes the
result. No hand-curated list required. `crawl4ai_max_urls` caps articles per asset.

**B. Manual list.** Set `crawl4ai_sources`, a JSON map of symbol → URLs:

```json
{"BTC": ["https://example.com/bitcoin-news"], "ETH": ["https://example.com/eth"]}
```

Both can be on at once — manual URLs and auto-discovered URLs for the same asset
are merged.

> **Invo is NOT crawled.** Invo already has a first-class JSON-API integration
> (`app/data/invo.py`). Crawl4AI / Maxun are for **API-less** sources (news,
> sentiment). Maxun (AGPL) has no code hook by design — run it standalone and have
> it POST its output to `/api/ingest/push`, exactly like this crawler does.

Enable the pipeline in Settings (all available as dashboard toggles):
- `ingest_enabled = true`  (accept ingested rows)
- `crawl4ai_autodiscover = true` (optional: auto-find sources) **or** set `crawl4ai_sources`
- `llm_web_context = true` (let the LLM advisor see the crawled signal)
- LLM advisor enabled + keyed (`llm_advisor_enabled`, `llm_api_key`).

## Run

```bash
python -m tools.crawl4ai_signal.run           # once
python -m tools.crawl4ai_signal.run --loop 900   # every 15 min
```

## Measure before you trust

Do **not** assume the crawl adds edge. Run the study first:

```bash
curl -X POST "http://127.0.0.1:8000/api/ingest/study/run?horizon_hours=4"
```

It reports the rank-IC of the signal vs realized forward return (per asset +
pooled) from CryptoMind's own candles. Only lean on it once the study earns it.

---

## Attribution

This product includes software developed by UncleCode
(https://x.com/unclecode) as part of the Crawl4AI project
(https://github.com/unclecode/crawl4ai).
