"""Dynamic Trading Universe — the system branches onto new coins it
discovers in its research data.

Discovery inputs:
  * CoinGecko trending coins (what the market is searching for)
  * Ticker/name mentions extracted from ingested news & social documents

Validation gates before a coin joins the tradeable universe:
  * Listed on Coinbase Exchange as <SYM>-USD, status online, not restricted
  * Sufficient 24h dollar volume (liquidity floor)
  * Universe cap respected (core coins are never evicted; discovered coins
    are ranked by mention-heat and rotated)

The config.PRODUCTS list is mutated IN PLACE so every module that imported
it sees the same live universe.
"""
import asyncio, time, re
import httpx
from ..config import PRODUCTS
from .. import db

CB = "https://api.exchange.coinbase.com"
CG_TRENDING = "https://api.coingecko.com/api/v3/search/trending"

CORE = list(PRODUCTS)              # original 6 — never removed
MAX_UNIVERSE = 14                  # core + up to 8 discovered
MIN_DOLLAR_VOL_24H = 3_000_000     # liquidity floor for discovered coins
REFRESH_SEC = 900                  # discovery cycle every 15 min

# words that look like tickers but aren't
TICKER_BLACKLIST = {
    "USD", "USDT", "USDC", "ETF", "CEO", "SEC", "AI", "US", "UK", "EU", "GDP",
    "API", "NFT", "DEFI", "FUD", "FOMO", "ATH", "TVL", "APR", "APY", "IPO",
    "NYSE", "CPI", "FED", "DOJ", "OK", "USA", "RSS", "URL", "PM", "AM", "THE",
    "FOR", "NEW", "ALL", "ONE", "TOP", "NOW", "BIG", "WHY", "HOW", "CAN",
}
DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{2,6})\b")
BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,6})\b")


class Universe:
    def __init__(self):
        self.cb_products = {}      # SYM -> {id, status} from Coinbase
        self.cb_stats = {}         # SYM -> 24h dollar volume
        self.mention_heat = {}     # SYM -> decayed mention counter
        self.trending = []         # latest CoinGecko trending symbols
        self.discovered = []       # currently-added discovered product ids
        self.rejected = {}         # SYM -> reason (for dashboard transparency)
        self.name_to_sym = {}      # lowercase coin name -> SYM
        self.oi_growth = {}        # SYM -> 24h OI % change (capital-flow source)
        self.sources = {}          # SYM -> set of source tags that surfaced it
        self.last_update = 0.0

    # -------------------- reference data --------------------
    async def _load_coinbase_products(self, client):
        r = await client.get(f"{CB}/products", timeout=20)
        r.raise_for_status()
        self.cb_products = {
            p["base_currency"]: {"id": p["id"], "status": p["status"]}
            for p in r.json()
            if p["quote_currency"] == "USD" and p["status"] == "online"
            and not p.get("trading_disabled")
        }

    async def _volume_ok(self, client, sym):
        """Check 24h dollar volume via product stats (cached)."""
        if sym in self.cb_stats and time.time() - self.cb_stats[sym][1] < 3600:
            return self.cb_stats[sym][0] >= MIN_DOLLAR_VOL_24H
        try:
            pid = self.cb_products[sym]["id"]
            r = await client.get(f"{CB}/products/{pid}/stats", timeout=15)
            s = r.json()
            dollar_vol = float(s.get("volume", 0)) * float(s.get("last", 0))
            self.cb_stats[sym] = (dollar_vol, time.time())
            return dollar_vol >= MIN_DOLLAR_VOL_24H
        except Exception:
            return False

    # -------------------- discovery inputs --------------------
    async def _fetch_trending(self, client):
        try:
            r = await client.get(CG_TRENDING, timeout=15)
            coins = r.json().get("coins", [])
            self.trending = []
            for c in coins:
                item = c.get("item", {})
                sym = (item.get("symbol") or "").upper()
                name = (item.get("name") or "").lower()
                if sym:
                    self.trending.append(sym)
                    if name:
                        self.name_to_sym[name] = sym
                    self.mention_heat[sym] = self.mention_heat.get(sym, 0) + 3.0
                    self.sources.setdefault(sym, set()).add("trending")
        except Exception as e:
            db.log_event("warn", f"CoinGecko trending failed: {e}")

    async def _fetch_oi_growth(self, client):
        """Capital-flow discovery source (NOFX 'OI-growth' idea): rank coins by
        24h open-interest change on OKX perps. A coin whose OI is expanding fast
        is attracting fresh capital — a candidate worth looking at even before
        it trends in the news. Bounded to a candidate set (core + trending +
        current discovered) so we don't hammer the rate-limited rubik endpoints.
        Symbols above the growth threshold get a heat boost AND an 'oi_growth'
        source tag; a coin surfaced by TWO independent sources (narrative +
        capital flow) is a stronger prior than either alone.
        """
        from ..tunables import tv
        core_syms = {p.split("-")[0] for p in CORE}
        scan = list(core_syms | set(self.trending) | {
            p.split("-")[0] for p in self.discovered})
        thresh = tv("oi_growth_threshold")
        self.oi_growth = {}
        for ccy in scan[:20]:                      # cap the scan breadth
            try:
                d = await self._get_oi_history(client, ccy)
            except Exception:
                continue
            if not d:
                continue
            g = d
            self.oi_growth[ccy] = round(g, 4)
            if g >= thresh:
                # boost proportional to how far past the threshold it is
                boost = 2.0 + min(4.0, (g - thresh) / max(thresh, 1e-6))
                self.mention_heat[ccy] = self.mention_heat.get(ccy, 0) + boost
                self.sources.setdefault(ccy, set()).add("oi_growth")

    async def _get_oi_history(self, client, ccy):
        """24h OI % change for one coin from OKX rubik (newest-first rows)."""
        r = await client.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
            params={"ccy": ccy, "period": "1H"}, timeout=15)
        j = r.json()
        if j.get("code") not in ("0", 0):
            return None
        rows = j.get("data") or []
        if len(rows) < 2:
            return None
        oi_now = float(rows[0][1])
        older = rows[min(24, len(rows) - 1)]
        oi_then = float(older[1])
        return (oi_now / oi_then - 1) if oi_then else None

    def ingest_documents(self, documents):
        """Extract coin mentions from research documents → mention heat."""
        for d in documents:
            title = d.get("title", "")
            syms = set(DOLLAR_TICKER_RE.findall(title))
            # bare uppercase tokens only count if they're known Coinbase symbols
            for tok in BARE_TICKER_RE.findall(title):
                if tok in self.cb_products and tok not in TICKER_BLACKLIST:
                    syms.add(tok)
            # coin-name matches (e.g. "solana" -> SOL)
            low = title.lower()
            for name, sym in self.name_to_sym.items():
                if len(name) > 3 and name in low:
                    syms.add(sym)
            for s in syms:
                if s not in TICKER_BLACKLIST:
                    self.mention_heat[s] = self.mention_heat.get(s, 0) + 1.0
                    self.sources.setdefault(s, set()).add("news")

    # -------------------- universe update --------------------
    async def refresh(self, client):
        await self._load_coinbase_products(client)
        await self._fetch_trending(client)
        # MEME source: give curated meme majors standing eligibility and tag any
        # known meme so the risk manager can apply its tighter envelope. Runs
        # before decay/selection so seeded memes compete for a universe slot.
        try:
            from .memes import memes
            memes.seed_universe()
        except Exception as e:
            db.log_event("warn", f"Meme seeding failed: {e}")
        # capital-flow source (OI growth) — best-effort; never blocks discovery
        try:
            await self._fetch_oi_growth(client)
        except Exception as e:
            db.log_event("warn", f"OI-growth discovery failed: {e}")

        # decay heat so stale narratives fade
        for s in list(self.mention_heat):
            self.mention_heat[s] *= 0.85
            if self.mention_heat[s] < 0.2:
                del self.mention_heat[s]
                self.sources.pop(s, None)
        # MULTI-SOURCE confirmation: a coin surfaced by two+ independent sources
        # (e.g. narrative heat AND capital inflow) gets a small extra prior —
        # exactly NOFX's "mixed" universe tagging. Applied after decay so it's a
        # standing bonus for corroborated names, not a runaway compounding boost.
        for s, tags in self.sources.items():
            if len(tags) >= 2 and s in self.mention_heat:
                self.mention_heat[s] += 0.5 * (len(tags) - 1)

        core_syms = {p.split("-")[0] for p in CORE}
        candidates = sorted(
            ((s, h) for s, h in self.mention_heat.items() if s not in core_syms),
            key=lambda kv: -kv[1])

        self.rejected = {}
        chosen = []
        slots = MAX_UNIVERSE - len(CORE)
        for sym, heat in candidates:
            if len(chosen) >= slots:
                break
            if heat < 1.5:
                continue                              # not hot enough yet
            if sym not in self.cb_products:
                self.rejected[sym] = "not listed on Coinbase USD"
                continue
            if not await self._volume_ok(client, sym):
                self.rejected[sym] = "24h volume below liquidity floor"
                continue
            chosen.append(self.cb_products[sym]["id"])

        # apply changes to the live universe (mutate PRODUCTS in place).
        # never prune a coin we currently hold a position in.
        try:
            from ..execution.paper import broker
            held = set(broker.positions.keys())
        except Exception:
            held = set()
        current = set(PRODUCTS)
        target = set(CORE) | set(chosen) | (held & current)
        added = sorted(target - current)
        removed = sorted(current - target)
        for pid in removed:
            PRODUCTS.remove(pid)
        for pid in added:
            PRODUCTS.append(pid)
        self.discovered = [p for p in PRODUCTS if p not in CORE]

        if added:
            def _tag(pid):
                sym = pid.split("-")[0]
                srcs = "+".join(sorted(self.sources.get(sym, {"news"})))
                return f"{sym}={self.mention_heat.get(sym, 0):.1f}[{srcs}]"
            db.log_event("discovery",
                         f"🔭 Universe expanded: +{', '.join(added)} "
                         f"(heat/source: {', '.join(_tag(p) for p in added)})")
        if removed:
            db.log_event("discovery", f"Universe pruned: -{', '.join(removed)} (heat faded)")
        self.last_update = time.time()

    async def run(self):
        await asyncio.sleep(20)
        async with httpx.AsyncClient(headers={"User-Agent": "CryptoMind/1.0"}) as client:
            while True:
                try:
                    await self.refresh(client)
                except Exception as e:
                    db.log_event("error", f"Universe refresh failed: {e}")
                await asyncio.sleep(REFRESH_SEC)

    def stats(self):
        return {
            "core": CORE,
            "discovered": self.discovered,
            "universe": list(PRODUCTS),
            "trending_coingecko": self.trending[:15],
            "mention_heat": {k: round(v, 2) for k, v in
                             sorted(self.mention_heat.items(), key=lambda kv: -kv[1])[:20]},
            "oi_growth": {k: v for k, v in
                          sorted(self.oi_growth.items(), key=lambda kv: -kv[1])[:15]},
            "sources": {k: sorted(v) for k, v in self.sources.items()},
            "rejected": self.rejected,
            "coinbase_listed": len(self.cb_products),
            "max_universe": MAX_UNIVERSE,
            "last_update": self.last_update,
        }


universe = Universe()
