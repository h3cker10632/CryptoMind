"""Meme-coin awareness — a discovery source + classifier.

The operator wants the system to trade meme coins "to see how it plays out"
and, crucially, to LEARN how meme hype flocculates (clumps up) and fades so it
can track that lifecycle. Two jobs:

  1. SEED — a curated set of established, Coinbase-listed meme majors that are
     eligible to trade immediately (no waiting to trend). These get a standing
     heat bump and a `meme` source tag in the universe.

  2. DISCOVER — pull CoinGecko's `meme-token` category so brand-new memes that
     start pumping surface automatically (tagged `meme`), same as the trending /
     OI-growth sources already feed the universe.

Everything here is best-effort: if a network source is down, the curated seed
still works and the rest of the system is unaffected. Nothing here places
trades — it only *labels* coins so the risk manager can apply a tighter
blast-radius envelope to anything tagged as a meme (see risk/manager.py).
"""
import asyncio, time
import httpx
from .. import db

CG_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"

# Curated seed: established meme coins that trade as <SYM>-USD on Coinbase.
# These are only *eligible* — they still pass the normal liquidity / price-sanity
# / cost-viability gates before any trade. Kept deliberately to liquid majors so
# the seed itself never introduces junk.
SEED = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI"]

REFRESH_SEC = 1800                 # meme-category refresh every 30 min


class Memes:
    def __init__(self):
        # symbols known to be memes (seed ∪ discovered category members)
        self.known = set(SEED)
        self.seed = set(SEED)
        self.discovered = set()        # from the CoinGecko meme category
        self.last_update = 0.0
        self.enabled_cache = None

    # ---------------- classification ----------------
    @staticmethod
    def _sym(product):
        return product.split("-")[0].upper() if product else ""

    def is_meme(self, product):
        """True if this product is a meme coin (seed or discovered category)."""
        return self._sym(product) in self.known

    @staticmethod
    def enabled():
        try:
            from .. import settings
            return bool(settings.get("meme_trading_enabled"))
        except Exception:
            return False

    # ---------------- discovery ----------------
    async def refresh(self, client):
        """Fetch the CoinGecko meme-token category and fold its members into the
        known-meme set. Best-effort; leaves the seed intact on any failure."""
        try:
            r = await client.get(CG_MARKETS, params={
                "vs_currency": "usd", "category": "meme-token",
                "order": "market_cap_desc", "per_page": 100, "page": 1,
            }, timeout=20)
            r.raise_for_status()
            syms = {(c.get("symbol") or "").upper()
                    for c in r.json() if c.get("symbol")}
            syms.discard("")
            if syms:
                self.discovered = syms
                self.known = self.seed | self.discovered
                self.last_update = time.time()
        except Exception as e:
            db.log_event("warn", f"Meme category fetch failed: {e}")

    def seed_universe(self):
        """Give the seeded memes a standing heat bump + `meme` source tag so they
        enter the tradable universe promptly. Also tag any currently-known meme
        already surfaced by another source. Called from the universe refresh."""
        if not self.enabled():
            return
        try:
            from .universe import universe
        except Exception:
            return
        # standing eligibility for the curated seed
        for sym in self.seed:
            universe.mention_heat[sym] = max(universe.mention_heat.get(sym, 0), 2.0)
            universe.sources.setdefault(sym, set()).add("meme")
        # tag any already-heated coin that is a known meme (from the category)
        for sym in list(universe.mention_heat.keys()):
            if sym in self.known:
                universe.sources.setdefault(sym, set()).add("meme")

    async def run(self):
        await asyncio.sleep(25)
        async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as client:
            while True:
                if self.enabled():
                    try:
                        await self.refresh(client)
                    except Exception as e:
                        db.log_event("error", f"Meme refresh failed: {e}")
                await asyncio.sleep(REFRESH_SEC)

    # ---------------- reporting ----------------
    def stats(self):
        from ..config import PRODUCTS
        active = [p for p in PRODUCTS if self.is_meme(p)]
        return {
            "enabled": self.enabled(),
            "seed": sorted(self.seed),
            "discovered_category": sorted(self.discovered)[:40],
            "known_count": len(self.known),
            "active_in_universe": active,
            "last_update": self.last_update,
        }


memes = Memes()
