"""Research & Data Ingestion v2 — self-expanding.

New capabilities:
  * Adaptive per-feed rate limiting: honors 429/Retry-After with exponential
    backoff per source; Reddit subs are combined into ONE multireddit request
    with a descriptive User-Agent (fixes the 429 problem).
  * Source discovery: maintains a pool of candidate feeds, periodically
    trials them, and PROMOTES sources that deliver fresh, parseable items —
    DEMOTES sources that repeatedly fail.
  * Coin-branching research: when the trading universe discovers a new coin,
    the engine automatically spins up Google News + Reddit searches for it,
    so sentiment/narrative coverage follows the universe.
"""
import asyncio, time, re, html, random
import httpx
from ..config import PRODUCTS
from .. import db

UA = "linux:cryptomind.research:v2.0 (paper-trading research; contact: ops@cryptomind.local)"

# ---------------- asset keyword map (grows dynamically) ----------------
ASSET_KEYWORDS = {
    "BTC-USD": ["bitcoin", "btc", "satoshi"],
    "ETH-USD": ["ethereum", " eth ", "ether ", "vitalik"],
    "SOL-USD": ["solana", " sol "],
    "DOGE-USD": ["dogecoin", "doge"],
    "LINK-USD": ["chainlink", " link "],
    "AVAX-USD": ["avalanche", "avax"],
}

ITEM_RE = re.compile(r"<(?:item|entry)[\s>](.*?)</(?:item|entry)>", re.S | re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def _clean(s):
    s = html.unescape(s)
    s = s.replace("<![CDATA[", "").replace("]]>", "")
    return TAG_RE.sub(" ", s).strip()


class Feed:
    """One source with its own health/backoff state."""

    def __init__(self, name, url, kind="rss", min_interval=300, candidate=False):
        self.name, self.url, self.kind = name, url, kind
        self.min_interval = min_interval
        self.candidate = candidate       # trial source, not yet promoted
        self.next_ok = 0.0               # earliest next fetch time
        self.backoff = 0.0
        self.failures = 0
        self.successes = 0
        self.last_items = 0
        self.last_status = None
        self.dead = False

    def on_success(self, n_items):
        self.successes += 1
        self.failures = 0
        self.backoff = 0.0
        self.last_items = n_items
        self.last_status = 200
        # jitter so feeds don't all fire at once
        self.next_ok = time.time() + self.min_interval * random.uniform(0.9, 1.2)

    def on_failure(self, status=None, retry_after=None):
        self.failures += 1
        self.last_status = status
        if retry_after:
            self.backoff = max(self.backoff, retry_after)
        else:
            # exponential: 2min, 4, 8, ... capped 30 min
            self.backoff = min(1800, max(120, (self.backoff or 60) * 2))
        self.next_ok = time.time() + self.backoff
        if self.failures >= 6 and self.candidate:
            self.dead = True   # demote unreliable candidate sources

    @property
    def health(self):
        if self.dead:
            return "demoted"
        if self.failures > 0:
            return f"backoff {int(self.backoff)}s"
        return "ok" if self.successes else "untested"


# ---------------- established + candidate source pools ----------------
def _default_feeds():
    return [
        Feed("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        Feed("CoinTelegraph", "https://cointelegraph.com/rss"),
        Feed("Decrypt", "https://decrypt.co/feed"),
        # ONE combined multireddit request instead of several (429 fix)
        Feed("Reddit multi", "https://www.reddit.com/r/CryptoCurrency+Bitcoin+ethereum+solana/hot/.rss?limit=40",
             min_interval=600),
        # ---- macro / regulatory context (crypto is macro-driven) ----
        Feed("Macro: Fed & rates",
             "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en",
             kind="macro", min_interval=1200),
        Feed("Macro: inflation & economy",
             "https://news.google.com/rss/search?q=inflation+CPI+economy+markets&hl=en-US&gl=US&ceid=US:en",
             kind="macro", min_interval=1200),
        Feed("Macro: crypto regulation",
             "https://news.google.com/rss/search?q=crypto+regulation+SEC+ETF&hl=en-US&gl=US&ceid=US:en",
             kind="macro", min_interval=1200),
    ]


def _candidate_feeds():
    return [
        Feed("The Block", "https://www.theblock.co/rss.xml", candidate=True),
        Feed("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/", candidate=True),
        Feed("CryptoSlate", "https://cryptoslate.com/feed/", candidate=True),
        Feed("NewsBTC", "https://www.newsbtc.com/feed/", candidate=True),
        Feed("U.Today", "https://u.today/rss", candidate=True),
        Feed("AMBCrypto", "https://ambcrypto.com/feed/", candidate=True),
        Feed("CryptoPotato", "https://cryptopotato.com/feed/", candidate=True),
        Feed("BeInCrypto", "https://beincrypto.com/feed/", candidate=True),
    ]


class ResearchEngine:
    def __init__(self):
        self.feeds = _default_feeds()
        self.candidates = _candidate_feeds()
        self.dynamic_feeds = {}      # product -> Feed (Google News per coin)
        self.documents = []
        self.fear_greed = None
        self.research_queue = []
        self.last_update = 0.0
        self.healthy = False

    # ---------------- keyword map growth ----------------
    def ensure_asset_keywords(self, product):
        if product in ASSET_KEYWORDS:
            return
        sym = product.split("-")[0]
        ASSET_KEYWORDS[product] = [f"${sym.lower()}", f" {sym.lower()} "]
        try:
            from .universe import universe
            for name, s in universe.name_to_sym.items():
                if s == sym and len(name) > 3:
                    ASSET_KEYWORDS[product].append(name)
        except Exception:
            pass

    def _tag_assets(self, text):
        low = " " + text.lower() + " "
        return [a for a, kws in ASSET_KEYWORDS.items()
                if a in PRODUCTS and any(k in low for k in kws)]

    # ---------------- coin-branching: follow the universe ----------------
    def sync_dynamic_feeds(self):
        """Create a Google News feed for every coin in the universe that
        lacks dedicated coverage; drop feeds for pruned coins."""
        from .universe import universe
        for product in list(PRODUCTS):
            if product in self.dynamic_feeds:
                continue
            sym = product.split("-")[0]
            name = next((n for n, s in universe.name_to_sym.items() if s == sym), sym)
            q = f"{name}+{sym}+crypto".replace(" ", "+")
            self.dynamic_feeds[product] = Feed(
                f"GoogleNews:{sym}",
                f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                min_interval=900)
            self.ensure_asset_keywords(product)
            db.log_event("discovery", f"📰 New research feed spun up for {sym} (Google News)")
        for product in list(self.dynamic_feeds):
            if product not in PRODUCTS:
                del self.dynamic_feeds[product]

    # ---------------- fetching with backoff ----------------
    async def _fetch_feed(self, client, feed):
        if feed.dead or time.time() < feed.next_ok:
            return []
        try:
            r = await client.get(feed.url, timeout=20, follow_redirects=True)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                feed.on_failure(429, retry_after=float(ra) if ra and ra.isdigit() else None)
                db.log_event("warn", f"Feed rate-limited: {feed.name} — "
                                     f"backing off {int(feed.backoff)}s")
                return []
            r.raise_for_status()
            docs = []
            for m in list(ITEM_RE.finditer(r.text))[:30]:
                tm = TITLE_RE.search(m.group(1))
                if not tm:
                    continue
                title = _clean(tm.group(1))
                if len(title) < 8:
                    continue
                docs.append({"source": feed.name, "title": title,
                             "ts": time.time(), "assets": self._tag_assets(title),
                             "kind": feed.kind})
            feed.on_success(len(docs))
            if feed.candidate and feed.successes >= 2 and len(docs) >= 3:
                feed.candidate = False
                self.feeds.append(feed)
                self.candidates.remove(feed)
                db.log_event("discovery", f"✅ Source promoted: {feed.name} "
                                          f"({len(docs)} items, reliable)")
            return docs
        except Exception as e:
            feed.on_failure()
            db.log_event("warn", f"Feed failed: {feed.name}: {e} — "
                                 f"retry in {int(feed.backoff)}s")
            return []

    async def _fetch_fear_greed(self, client):
        r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=15)
        d = r.json()["data"][0]
        self.fear_greed = {"value": int(d["value"]), "label": d["value_classification"]}

    # ---------------- self-directed research loop ----------------
    def _self_research(self):
        counts = {}
        for d in self.documents:
            for a in d["assets"]:
                counts[a] = counts.get(a, 0) + 1
        queue = []
        for a, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n >= 3:
                queue.append({"asset": a, "mentions": n,
                              "task": f"Narrative heating up on {a} ({n} mentions) — "
                                      f"cross-check momentum, order-book imbalance and sentiment."})
        # also surface universe discovery work
        try:
            from .universe import universe
            for sym, why in list(universe.rejected.items())[:3]:
                queue.append({"asset": sym, "mentions": 0,
                              "task": f"Watching {sym}: trending in research but {why}."})
        except Exception:
            pass
        self.research_queue = queue[:8]

    # ---------------- main loop ----------------
    async def run(self):
        async with httpx.AsyncClient(headers={"User-Agent": UA}) as client:
            while True:
                self.sync_dynamic_feeds()
                all_feeds = (self.feeds + self.candidates +
                             list(self.dynamic_feeds.values()))
                docs, ok = [], 0
                for feed in all_feeds:
                    got = await self._fetch_feed(client, feed)
                    if got:
                        docs += got
                        ok += 1
                    await asyncio.sleep(random.uniform(0.8, 1.6))
                try:
                    await self._fetch_fear_greed(client)
                except Exception as e:
                    db.log_event("warn", f"Fear&Greed failed: {e}")

                if docs:
                    # merge: keep newest doc per title
                    seen = set()
                    merged = []
                    for d in docs + self.documents:
                        key = d["title"][:80]
                        if key not in seen:
                            seen.add(key)
                            merged.append(d)
                    self.documents = merged[:250]
                    # feed coin discovery
                    try:
                        from .universe import universe
                        universe.ingest_documents(docs)
                    except Exception:
                        pass
                    self._self_research()
                    self.last_update = time.time()
                    self.healthy = ok >= 2
                    db.log_event("research",
                                 f"Ingested {len(docs)} docs from {ok} sources "
                                 f"({len(self.feeds)} promoted, "
                                 f"{sum(1 for c in self.candidates if not c.dead)} on trial, "
                                 f"{len(self.dynamic_feeds)} coin-specific)")
                await asyncio.sleep(60)   # scheduler tick; per-feed intervals gate real fetches

    def source_stats(self):
        def row(f):
            return {"name": f.name, "health": f.health, "items": f.last_items,
                    "successes": f.successes, "status": f.last_status,
                    "candidate": f.candidate}
        return {
            "promoted": [row(f) for f in self.feeds],
            "on_trial": [row(f) for f in self.candidates if not f.dead],
            "demoted": [row(f) for f in self.candidates if f.dead],
            "coin_specific": [row(f) for f in self.dynamic_feeds.values()],
        }


research = ResearchEngine()
