"""Derivatives Data Feed — OKX public API (key-free).

Per asset:
  funding_rate        current perp funding rate (8h). Extreme positive =
                      crowded longs (contrarian bearish), extreme negative =
                      crowded shorts (contrarian bullish / squeeze fuel).
  oi_usd              open interest in USD (latest hourly)
  oi_change_24h       % change in OI over ~24h. Rising OI + rising price =
                      trend confirmation; falling OI on rallies = short cover.
  long_short_ratio    long/short account ratio (>1 = more longs). Extreme
                      crowding is a contrarian signal.
  taker_buy_ratio     taker buy volume / total taker volume over the last
                      hours (aggression gauge, >0.5 = buyers lifting offers).
"""
import asyncio, time
import httpx
from ..config import PRODUCTS
from .. import db

BASE = "https://www.okx.com"
POLL_SEC = 120


def _ccy(product):
    return product.split("-")[0]          # BTC-USD -> BTC


class DerivativesData:
    def __init__(self):
        self.metrics = {}          # product -> dict of metrics
        self.no_swap = set()       # products with no OKX perp market
        self.last_update = 0.0
        self.healthy = False

    async def _get(self, client, path, params=None):
        r = await client.get(BASE + path, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        if j.get("code") not in ("0", 0):
            raise RuntimeError(f"OKX error {j.get('code')}: {j.get('msg')}")
        return j["data"]

    async def refresh_product(self, client, product):
        ccy = _ccy(product)
        inst = f"{ccy}-USDT-SWAP"
        m = {}

        # funding rate
        d = await self._get(client, "/api/v5/public/funding-rate",
                            {"instId": inst})
        m["funding_rate"] = float(d[0]["fundingRate"])

        # open interest history (hourly, USD) → level + 24h change
        d = await self._get(client, "/api/v5/rubik/stat/contracts/open-interest-volume",
                            {"ccy": ccy, "period": "1H"})
        # rows: [ts, oi_usd, vol_usd], newest first
        if d:
            oi_now = float(d[0][1])
            m["oi_usd"] = oi_now
            older = d[min(24, len(d) - 1)]
            oi_then = float(older[1])
            m["oi_change_24h"] = (oi_now / oi_then - 1) if oi_then else 0.0

        # long/short account ratio
        d = await self._get(client, "/api/v5/rubik/stat/contracts/long-short-account-ratio",
                            {"ccy": ccy, "period": "1H"})
        if d:
            m["long_short_ratio"] = float(d[0][1])

        # taker buy/sell volume (aggression) — average over last 4 hours
        d = await self._get(client, "/api/v5/rubik/stat/taker-volume",
                            {"ccy": ccy, "instType": "CONTRACTS", "period": "1H"})
        if d:
            rows = d[:4]
            buy = sum(float(r[1]) for r in rows)
            sell = sum(float(r[2]) for r in rows)
            tot = buy + sell
            m["taker_buy_ratio"] = buy / tot if tot else 0.5

        m["ts"] = time.time()
        self.metrics[product] = m

    async def run(self):
        async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as client:
            while True:
                ok = 0
                for p in list(PRODUCTS):          # dynamic universe
                    if p in self.no_swap:
                        ok += 1                   # counted healthy; no perp exists
                        continue
                    try:
                        await self.refresh_product(client, p)
                        ok += 1
                    except Exception as e:
                        msg = str(e)
                        if "51001" in msg or "instId" in msg.lower():
                            # coin has no OKX perp — skip permanently, not an error
                            self.no_swap.add(p)
                            db.log_event("data", f"{p}: no OKX perp market — "
                                                 f"derivatives features disabled for it")
                        else:
                            db.log_event("warn", f"Derivatives fetch failed {p}: {e}")
                    await asyncio.sleep(0.6)   # OKX rubik endpoints are rate-limited
                was = self.healthy
                self.healthy = ok >= len(PRODUCTS) // 2
                if self.healthy and not was:
                    db.log_event("data", f"Derivatives feed healthy ({ok}/{len(PRODUCTS)} assets, OKX)")
                self.last_update = time.time()
                await asyncio.sleep(POLL_SEC)

    # ---------- feature accessors ----------
    def features(self, product):
        """Normalized derivative features (all roughly in [-1, 1])."""
        m = self.metrics.get(product)
        if not m or time.time() - m.get("ts", 0) > 900:
            return None
        f = {}
        # funding: normalize vs typical ±0.01%/8h band; clip at ±5x
        fr = m.get("funding_rate", 0.0)
        f["funding_norm"] = max(-1.0, min(1.0, fr / 0.0005))
        f["funding_rate"] = fr
        # OI 24h change: ±10% band
        oic = m.get("oi_change_24h", 0.0)
        f["oi_change_norm"] = max(-1.0, min(1.0, oic / 0.10))
        f["oi_change_24h"] = oic
        f["oi_usd"] = m.get("oi_usd")
        # long/short crowding: 1.0 = balanced; band 0.5..2.5
        ls = m.get("long_short_ratio", 1.0)
        f["ls_ratio"] = ls
        f["ls_crowding"] = max(-1.0, min(1.0, (ls - 1.0) / 1.2))
        # taker aggression: 0.5 = balanced
        tb = m.get("taker_buy_ratio", 0.5)
        f["taker_buy_ratio"] = tb
        f["taker_aggression"] = max(-1.0, min(1.0, (tb - 0.5) * 8))
        return f


derivatives = DerivativesData()
