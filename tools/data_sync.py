"""Keep the market data store (app/data/store.py) current.

    python tools/data_sync.py                 # incremental sync of everything
    python tools/data_sync.py --migrate       # one-off: import .cache/history JSON
    python tools/data_sync.py --quality       # data-quality report only
    python tools/data_sync.py --only daily    # daily | hourly | funding

Daily candles: every Coinbase USD pair, delisted included, back to listing
(up to --daily-years). Hourly candles: the current trading universe plus the
most liquid coins today (--hourly-top), --hourly-years back. Funding: the
same hourly set from Hyperliquid (since 2023), BTC/ETH from Deribit (2019).
Each sync re-fetches a few overlapping bars so a revised bar is caught and
kept as a revision rather than silently overwriting history.
"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OVERLAP_BARS = 3


def migrate(say=print):
    """Import the legacy JSON caches (keeps the longest file per coin)."""
    from app.data import store
    best = {}
    for f in glob.glob(os.path.join(ROOT, ".cache", "history", "*.json")):
        parts = os.path.basename(f)[:-5].split("_")
        if len(parts) != 3:
            continue
        p, gran = parts[0], int(parts[1])
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        if len(rows) > len(best.get((p, gran), [])):
            best[(p, gran)] = rows
    for (p, gran), rows in sorted(best.items()):
        st = store.ingest_candles(p, gran, rows, source="legacy-json")
        say(f"migrated {p} {gran}s: {st['new']} new, {st['rejected']} rejected")


async def _sync_candles(client, sem, product, gran, years, now, say):
    from app.data import sources, store
    last = store.last_ts("candles", gran, product)
    start = (last - OVERLAP_BARS * gran) if last else now - years * 365 * 86400
    async with sem:
        try:
            rows = await sources.coinbase_candles(client, product, gran, start, now)
        except Exception as e:
            say(f"{product} {gran}s: fetch failed ({e})")
            return None
    st = store.ingest_candles(product, gran, rows, now=now)
    if st["new"] or st["revised"] or st["rejected"]:
        say(f"{product} {gran}s: +{st['new']} new, {st['revised']} revised, "
            f"{st['rejected']} rejected ({st['rows']} rows)")
    return st


async def sync(only=None, daily_years=10, hourly_years=3, hourly_top=30, say=print):
    import httpx
    from app.data import sources, store
    now = time.time()
    totals = {"new": 0, "revised": 0, "rejected": 0}
    fetched = False
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "cryptomind-data"}) as c:
        sem = asyncio.Semaphore(4)
        if only in (None, "daily"):
            prods = [p for p in await sources.coinbase_usd_products(c)
                     if not p["stable"] and store.is_tradeable_asset(p["id"])]
            say(f"daily: {len(prods)} USD coins "
                f"({sum(1 for p in prods if p['status'] == 'delisted')} delisted)")
            res = await asyncio.gather(*[_sync_candles(c, sem, p["id"], 86400, daily_years,
                                                       now, say) for p in prods])
            fetched = any(st is not None for st in res)
            for st in res:
                for k in totals:
                    totals[k] += (st or {}).get(k, 0)
        hourly = hourly_universe(hourly_top)
        if only in (None, "hourly"):
            say(f"hourly: {len(hourly)} coins")
            res = await asyncio.gather(*[_sync_candles(c, sem, p, 3600, hourly_years, now, say)
                                         for p in hourly])
            for st in res:
                for k in totals:
                    totals[k] += (st or {}).get(k, 0)
        if only in (None, "funding"):
            await sync_funding(c, hourly, now, say)
    say(f"sync done: {totals}")
    if only is None:
        if not fetched:      # exit non-zero: the scheduler retries in 6 h, not every pass
            raise SystemExit("no daily candles fetched")
        # when the FULL sync last finished: the scheduler and the scorecard read
        # this, not ingest_log.jsonl, which the core / exploration loops append
        # to every hour (that kept the daily sync from ever coming due)
        os.makedirs(store.STORE, exist_ok=True)
        with open(os.path.join(store.STORE, "last_sync"), "w") as f:
            f.write(str(now))
    return totals


def hourly_universe(top=30):
    """Coins the bot trades or has traded (legacy cache / store) plus today's
    `top` most liquid by daily dollar volume."""
    from app.data import store
    have = set(store.products("candles", 3600))
    have |= {os.path.basename(f).split("_")[0]
             for f in glob.glob(os.path.join(ROOT, ".cache", "history", "*_3600_*.json"))}
    try:
        have |= set(store.universe_at(time.time() // 86400 * 86400 - 86400, n=top))
    except Exception:
        pass
    return sorted(p for p in have if store.is_tradeable_asset(p))


async def sync_funding(client, products, now, say):
    from app.data import sources, store
    coins = sorted({p.split("-")[0] for p in products})
    names = await sources.hyperliquid_names(client)
    missing = [c for c in coins if c not in names]
    if missing:
        say(f"funding: not listed on Hyperliquid, skipped: {', '.join(missing)}")
    for coin in (c for c in coins if c in names):
        last = store.last_ts("funding", "hyperliquid", coin)
        start = last - 3 * 3600 if last else 1_683_000_000      # Hyperliquid launch, May 2023
        try:
            rows = await sources.hyperliquid_funding(client, names[coin], start, now)
        except Exception as e:
            say(f"funding {coin}: failed ({e})")
            continue
        if rows:
            st = store.ingest_funding("hyperliquid", coin, rows, now=now)
            if st["new"] or st["revised"]:
                say(f"funding hyperliquid {coin}: +{st['new']} ({st['rows']} rows)")
    for coin in ("BTC", "ETH"):
        last = store.last_ts("funding", "deribit", coin)
        start = last - 3 * 3600 if last else 1_556_668_800        # May 2019
        try:
            rows = await sources.deribit_funding(client, coin, start, now)
        except Exception as e:
            say(f"funding deribit {coin}: failed ({e})")
            continue
        st = store.ingest_funding("deribit", coin, rows, now=now)
        say(f"funding deribit {coin}: +{st['new']} ({st['rows']} rows)")


def print_quality(say=print):
    from app.data import store
    for gran in (86400, 3600):
        q = store.quality(gran)
        if not q:
            continue
        bad = {p: r for p, r in q.items()
               if r["coverage_pct"] < 99 or r["reverting_spikes"] or r["revisions"]}
        say(f"\n{gran}s candles: {len(q)} coins, {sum(r['rows'] for r in q.values()):,} rows; "
            f"{len(bad)} with gaps (<99% coverage), reverting spikes, or revisions")
        for p, r in sorted(bad.items(), key=lambda kv: kv[1]["coverage_pct"])[:15]:
            say(f"  {p:14s} coverage {r['coverage_pct']:6.2f}%  missing {r['missing_bars']:5d} "
                f"(longest {r['longest_gap_bars']})  spikes {r['reverting_spikes']}  "
                f"revisions {r['revisions']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--only", choices=("daily", "hourly", "funding"))
    ap.add_argument("--daily-years", type=int, default=10)
    ap.add_argument("--hourly-years", type=int, default=3)
    ap.add_argument("--hourly-top", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    if a.migrate:
        migrate(say)
    if not a.quality and not a.migrate:
        asyncio.run(sync(a.only, a.daily_years, a.hourly_years, a.hourly_top, say))
    print_quality(say)


if __name__ == "__main__":
    main()
