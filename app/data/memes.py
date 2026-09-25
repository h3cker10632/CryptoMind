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

# ---- hype-lifecycle thresholds (tuned for 5-minute candles) ----
HYPE_SAMPLE_SEC = 60               # min spacing between hype-velocity samples
MARKUP_MOM = 0.03                  # +3% over ~1h = a real markup leg
BLOWOFF_MOM = 0.10                 # +10% over ~1h = parabolic territory
BLOWOFF_ACCEL = 0.04              # momentum accelerating hard = climax risk
CLIMAX_VOL = 2.5                   # 2.5x baseline volume = blow-off volume
EXPAND_VOL = 1.3                   # 1.3x baseline = meaningful volume expansion
DECAY_MOM = 0.03                   # -3% over ~1h with fading interest = decay
DISTRIB_ACCEL = 0.02              # momentum rolling over near the highs
ACCUM_MOM = 0.015                  # near-flat price = still basing


class Memes:
    def __init__(self):
        # symbols known to be memes (seed ∪ discovered category members)
        self.known = set(SEED)
        self.seed = set(SEED)
        self.discovered = set()        # from the CoinGecko meme category
        self.last_update = 0.0
        self.enabled_cache = None
        # per-symbol hype history for velocity: SYM -> deque[(ts, heat)]
        self.hype_hist = {}
        # last computed lifecycle report per product (for reporting/dashboard)
        self.lifecycle = {}

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

    # ---------------- hype lifecycle ----------------
    @staticmethod
    def classify(closes, vols, hype_vel=0.0):
        """Classify where a meme sits in its hype lifecycle and return
        ``(phase, lean, detail)`` from price action + volume + hype velocity.

        Pure function (no I/O) so it is fully unit-testable. ``lean`` is in
        [-1, 1]: positive = ride the pump, negative = fade / stay away (the
        blow-off and distribution phases). Phases:
          dormant · accumulation · markup · blowoff · distribution · decay
        """
        detail = {"mom": 0.0, "accel": 0.0, "vol_exp": 1.0, "hype_vel": hype_vel}
        if not closes or len(closes) < 26:
            return "dormant", 0.0, detail
        c = closes
        mom = c[-1] / c[-13] - 1.0 if c[-13] else 0.0          # ~last hour
        mom_prev = c[-13] / c[-25] - 1.0 if c[-25] else 0.0    # prior hour
        accel = mom - mom_prev
        vol_exp = 1.0
        if vols and len(vols) >= 12:
            recent = sum(vols[-6:]) / 6.0
            prior = vols[-48:-6] if len(vols) >= 18 else vols[:-6]
            if prior:
                base = sum(prior) / len(prior)
                if base > 1e-9:
                    vol_exp = recent / base
        detail.update(mom=round(mom, 4), accel=round(accel, 4),
                      vol_exp=round(vol_exp, 2), hype_vel=round(hype_vel, 3))

        def _clip(x, lo=-1.0, hi=1.0):
            return lo if x < lo else hi if x > hi else x

        # blow-off top: parabolic price + volume climax → FADE, do not chase
        if mom > BLOWOFF_MOM and accel > BLOWOFF_ACCEL and vol_exp > CLIMAX_VOL:
            return "blowoff", -0.85, detail
        # markup: strong rise on expanding volume with interest still building
        if mom > MARKUP_MOM and vol_exp > EXPAND_VOL and (hype_vel >= 0 or accel > 0):
            lean = _clip(0.35 + mom * 4.0 + max(0.0, hype_vel) * 0.3)
            return "markup", lean, detail
        # distribution: rolling over near the highs on still-heavy volume
        if accel < -DISTRIB_ACCEL and mom > 0 and vol_exp > EXPAND_VOL:
            return "distribution", -0.55, detail
        # decay: price bleeding with fading volume / interest
        if mom < -DECAY_MOM and (vol_exp < 1.0 or hype_vel < 0):
            return "decay", -0.5, detail
        # accumulation: basing (near-flat) with volume + hype quietly building
        if abs(mom) < ACCUM_MOM and vol_exp > EXPAND_VOL and hype_vel > 0:
            return "accumulation", 0.2, detail
        return "dormant", 0.0, detail

    def _record_hype(self, sym, heat, now=None):
        import time as _t
        now = now or _t.time()
        dq = self.hype_hist.get(sym)
        if dq is None:
            from collections import deque
            dq = self.hype_hist[sym] = deque(maxlen=30)
        if not dq or now - dq[-1][0] >= HYPE_SAMPLE_SEC:
            dq.append((now, float(heat)))

    def _hype_velocity(self, sym):
        """Fractional change in mention-heat per sample, from the rolling
        history. 0.0 until at least two spaced samples exist."""
        dq = self.hype_hist.get(sym)
        if not dq or len(dq) < 2:
            return 0.0
        old_t, old_h = dq[0]
        new_t, new_h = dq[-1]
        base = old_h if old_h > 1e-6 else 1.0
        return (new_h - old_h) / base

    def lean(self, product, market):
        """Directional lifecycle lean in [-1, 1] for a meme product, or 0.0 when
        meme trading is off / the product is not a meme / history is thin. Reads
        candles from `market` and mention-heat velocity from the universe.
        Best-effort; never raises."""
        try:
            if not self.enabled() or not self.is_meme(product):
                return 0.0
            cs = getattr(market, "candles", {}).get(product, [])
            if len(cs) < 26:
                return 0.0
            closes = [c[4] for c in cs]
            vols = [c[5] for c in cs]
            sym = self._sym(product)
            try:
                from .universe import universe
                self._record_hype(sym, universe.mention_heat.get(sym, 0.0))
            except Exception:
                pass
            hv = self._hype_velocity(sym)
            phase, lean, detail = self.classify(closes, vols, hv)
            self.lifecycle[product] = {"phase": phase, "lean": round(lean, 3),
                                       **detail}
            return lean
        except Exception:
            return 0.0

    # ---------------- reporting ----------------
    def stats(self):
        from ..config import PRODUCTS
        active = [p for p in PRODUCTS if self.is_meme(p)]
        phases = {p: {"phase": r.get("phase"), "lean": r.get("lean")}
                  for p, r in self.lifecycle.items() if self.is_meme(p)}
        return {
            "enabled": self.enabled(),
            "seed": sorted(self.seed),
            "discovered_category": sorted(self.discovered)[:40],
            "known_count": len(self.known),
            "active_in_universe": active,
            "lifecycle": phases,
            "last_update": self.last_update,
        }


memes = Memes()
