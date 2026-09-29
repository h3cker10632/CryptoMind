"""CLI: run the crawl4ai signal producer once.

Usage:
    python -m tools.crawl4ai_signal.run          # use crawl4ai_sources setting
    python -m tools.crawl4ai_signal.run --loop 900   # repeat every 900s

Attribution (per the Crawl4AI Apache-2.0 license): this product includes
software developed by UncleCode (https://x.com/unclecode) as part of the
Crawl4AI project (https://github.com/unclecode/crawl4ai).
"""
import argparse
import time

from .producer import run_once


def main():
    ap = argparse.ArgumentParser(description="crawl4ai -> LLM -> CryptoMind ingest")
    ap.add_argument("--loop", type=int, default=0,
                    help="repeat every N seconds (0 = run once)")
    args = ap.parse_args()
    while True:
        print(run_once())
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
