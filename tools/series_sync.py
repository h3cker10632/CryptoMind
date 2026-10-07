"""Keep the external series store (app/data/series.py) current.

    python tools/series_sync.py                 # every series in the catalog
    python tools/series_sync.py --only fear_greed dvol_btc
    python tools/series_sync.py --dry-run       # fetch + parse, store nothing

Series: app/data/series_sources.py CATALOG (Fear & Greed, Deribit DVOL,
DefiLlama stablecoin supply, Coin Metrics community on-chain activity). All
key-free. Each sync re-fetches an overlap so revised values are caught and
kept as revisions. Summary: reports/series_sync_latest.json.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OVERLAP_DAYS = 10


def main():
    import httpx
    from app.data import series as S, series_sources as SS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = a.only or list(SS.CATALOG)
    summary = {}
    with httpx.Client(headers={"User-Agent": "CryptoMind/series-sync"},
                      follow_redirects=True) as client:
        for name in names:
            if name not in SS.CATALOG:
                print(f"{name}: unknown series")
                continue
            last = S.last_ts(name)
            start = (last - OVERLAP_DAYS * 86400) if last else None
            try:
                rows = SS.fetch(name, client, start=start)
            except Exception as e:
                summary[name] = {"error": str(e)[:200]}
                print(f"{name}: fetch failed ({e})")
                continue
            if a.dry_run:
                summary[name] = {"fetched": len(rows), "first": rows[0] if rows else None,
                                 "last": rows[-1] if rows else None}
            else:
                summary[name] = dict(S.ingest(name, rows, source=SS.CATALOG[name]["fetch"]),
                                     fetched=len(rows))
            print(f"{name}: {summary[name]}")
    if not a.dry_run:
        os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
        with open(os.path.join(ROOT, "reports", "series_sync_latest.json"), "w") as f:
            json.dump({"ran_at": time.time(), "series": summary}, f, indent=1)
    sys.exit(0 if all("error" not in v for v in summary.values()) else 1)


if __name__ == "__main__":
    main()
