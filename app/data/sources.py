"""Key-free download functions for the market data store (tools/data_sync.py).

Rebuilt from the interface tools/data_sync.py uses (the original file was
kept out of git by an old `.gitignore` rule and never committed):

  coinbase_usd_products(client)        every Coinbase USD pair, delisted included
  coinbase_candles(client, p, g, s, e) [time, low, high, open, close, volume] bars
  hyperliquid_names(client)            {coin: Hyperliquid perp name} (kPEPE -> PEPE)
  hyperliquid_funding(client, n, s, e) [[ts, hourly rate]] since May 2023
  deribit_funding(client, coin, s, e)  [[ts, hourly rate]] BTC/ETH since 2019

All take an `httpx.AsyncClient`, page through the provider's limits, retry
on HTTP 429 / 5xx with backoff, and return rows sorted by time with
duplicates removed. Funding rates are PER HOUR (what app/engine/carry.py
sums into daily carry). The store (app/data/store.py) validates and versions
whatever these return; still-forming bars are dropped there.
"""
from __future__ import annotations
import asyncio

COINBASE = "https://api.exchange.coinbase.com"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"
DERIBIT = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
HOUR = 3600
CANDLES_PER_REQUEST = 300           # Coinbase's limit
HL_ROWS_PER_REQUEST = 500           # Hyperliquid fundingHistory limit
DERIBIT_WINDOW = 30 * 86400         # request funding a month at a time
PACE_SEC = 0.12                     # stay well inside Coinbase's public 10 req/s

STABLE_BASES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "GUSD", "PYUSD", "UST",
                "USTC", "EURC", "PAX", "FDUSD", "USDE", "USDS"}


async def _request(client, method, url, tries=5, **kw):
    """JSON from one HTTP call; retries 429 / 5xx / transport errors with
    linear backoff, raises on anything else."""
    last = None
    for attempt in range(tries):
        try:
            r = await client.request(method, url, timeout=30, **kw)
        except Exception as e:                    # transport error: retry
            last = e
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = RuntimeError(f"HTTP {r.status_code} from {url}")
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise last or RuntimeError(f"no response from {url}")


def _hour(ts_ms):
    """Exchange funding stamps sit a few ms off the hour; snap to it."""
    return int(round(int(ts_ms) / 1000 / HOUR)) * HOUR


# ------------------------------------------------------------------ Coinbase
async def coinbase_usd_products(client):
    """[{"id", "status", "stable"}] for every USD-quoted Coinbase product,
    including delisted ones (status "delisted") — needed for a universe
    without survivorship bias."""
    data = await _request(client, "GET", f"{COINBASE}/products")
    out = []
    for p in data or []:
        if p.get("quote_currency") != "USD" or not p.get("id"):
            continue
        base = str(p.get("base_currency") or p["id"].split("-")[0]).upper()
        status = str(p.get("status") or "online").lower()
        out.append({"id": p["id"], "status": status,
                    "stable": bool(p.get("fx_stablecoin")) or base in STABLE_BASES})
    return sorted(out, key=lambda p: p["id"])


async def coinbase_candles(client, product, granularity, start, end, pace=PACE_SEC):
    """All bars of `product` at `granularity` seconds whose time is in
    [start, end], oldest first. Pages forward 300 bars at a time; windows
    before the coin listed come back empty and are skipped."""
    step = granularity * CANDLES_PER_REQUEST
    s = int(start) // granularity * granularity
    end = int(end)
    rows = {}
    first = True
    while s <= end:
        e = min(s + step - granularity, end)
        if not first and pace:
            await asyncio.sleep(pace)
        first = False
        batch = await _request(client, "GET", f"{COINBASE}/products/{product}/candles",
                               params={"granularity": granularity, "start": s, "end": e})
        for b in batch or []:
            try:
                t = int(b[0])
            except (TypeError, ValueError, IndexError):
                continue
            if s <= t <= end and len(b) >= 6:
                rows[t] = [t] + [float(x) for x in b[1:6]]
        s = e + granularity
    return [rows[t] for t in sorted(rows)]


# ------------------------------------------------------------------ Hyperliquid
async def hyperliquid_names(client):
    """{coin symbol: Hyperliquid perp name}. Hyperliquid lists some small-
    priced coins per 1000 units with a "k" prefix (kPEPE, kSHIB, kBONK); their
    funding RATE is unaffected by the contract size."""
    meta = await _request(client, "POST", HYPERLIQUID, json={"type": "meta"})
    names = {}
    for u in (meta or {}).get("universe", []):
        n = u.get("name")
        if not n:
            continue
        sym = n[1:] if len(n) > 1 and n[0] == "k" and n[1:].isupper() else n
        if sym not in names or n == sym:          # an exact listing wins
            names[sym] = n
    return names


async def hyperliquid_funding(client, name, start, end):
    """[[ts, hourly funding rate]] for Hyperliquid perp `name` in [start, end]."""
    t = int(start) * 1000
    end_ms = int(end) * 1000
    rows = {}
    while t <= end_ms:
        batch = await _request(client, "POST", HYPERLIQUID,
                               json={"type": "fundingHistory", "coin": name,
                                     "startTime": t, "endTime": end_ms})
        if not batch:
            break
        last = t
        for b in batch:
            try:
                ms = int(b["time"])
                rows[_hour(ms)] = [_hour(ms), float(b["fundingRate"])]
                last = max(last, ms)
            except (KeyError, TypeError, ValueError):
                continue
        if len(batch) < HL_ROWS_PER_REQUEST or last <= t:
            break
        t = last + 1
    return [rows[k] for k in sorted(rows) if start <= k <= end]


# ------------------------------------------------------------------ Deribit
async def deribit_funding(client, coin, start, end):
    """[[ts, hourly funding rate]] for Deribit's `coin`-PERPETUAL
    (`interest_1h`), a month per request."""
    rows = {}
    s = int(start)
    while s <= int(end):
        e = min(s + DERIBIT_WINDOW, int(end))
        data = await _request(client, "GET", DERIBIT,
                              params={"instrument_name": f"{coin}-PERPETUAL",
                                      "start_timestamp": s * 1000, "end_timestamp": e * 1000})
        for b in (data or {}).get("result") or []:
            try:
                ts = _hour(b["timestamp"])
                rows[ts] = [ts, float(b["interest_1h"])]
            except (KeyError, TypeError, ValueError):
                continue
        s = e + 1
    return [rows[k] for k in sorted(rows) if start <= k <= end]
