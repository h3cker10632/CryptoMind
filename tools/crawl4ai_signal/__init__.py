"""Standalone crawl4ai signal producer for CryptoMind.

Crawls operator-configured web sources, condenses each into a per-asset
directional lean with an LLM, and pushes the result into CryptoMind's generic
ingest seam (app.data.ingest) as a MEASURED external signal.

This tool is deliberately SEPARATE from the trading core: crawl4ai (Apache-2.0)
is optional here, and the core never imports it. The signal it produces only
influences trades through the measured, bandit-weighted LLM-advisor context —
never a blind copy.

Attribution (per the Crawl4AI license):
    This product includes software developed by UncleCode
    (https://x.com/unclecode) as part of the Crawl4AI project
    (https://github.com/unclecode/crawl4ai).
"""
from .producer import build_rows, run_once, fetch_text

__all__ = ["build_rows", "run_once", "fetch_text"]
