"""Strategy Researcher — automated discovery of NEW strategy SHAPES.

The GA (app/learn/evolution.py) searches PARAMETERS of one fixed rule template
(EMA/RSI/breakout entries + ATR exits). This module is complementary: it searches
across rule STRUCTURES the GA cannot express — arbitrary AND-combinations of
predicates over the shared technical features — and it is honest about the
multiple testing that a wide search invites.

Design guarantees (measurement-first, same as the rest of the system):

  * Train/live parity for free. A candidate is scored by exactly the same feature
    dict the live engine uses: features come from app.data.features
    `features_from_ohlcv`, the single source of truth for both the live feed and
    the backtester. A promoted candidate is evaluated live in engine.strat_discovered
    via `vote()` — the SAME function used in the backtest. There is no second
    feature definition to drift.
  * A safe DSL, never eval. A candidate is data: a whitelist of feature names, two
    comparison ops and numeric thresholds. LLM-proposed specs are validated against
    that whitelist before they can run, so an LLM can propose ideas but never inject
    code or reference an unknown feature.
  * Honest out-of-sample gate. Every candidate is scored on a purged, embargoed
    walk-forward (reusing validation.walk_forward_folds), and only promoted if its
    pooled OOS returns clear a DEFLATED-Sharpe bar (app.backtest.stats) that PRICES
    the number of candidates tried this run — the false-strategy tax. Consistency
    across folds is required too, not just a good average.
  * Measured, never blind. A promoted candidate becomes ONE more bandit-weighted
    ensemble arm (`discovered`); its live weight is still learned from realized net
    PnL like every other sleeve. Discovery only earns it the right to be measured.

Candidate sources: a systematic sampler over the DSL (reproducible, no API cost)
and, optionally, LLM-proposed hypotheses (reuses the LLM-advisor plumbing). Both
face the identical OOS gate.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import threading
import time

from ..data.features import features_from_ohlcv, MIN_BARS
from ..backtest.stats import sharpe, deflated_sharpe_ratio
from .validation import walk_forward_folds

STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "discovered_strategies.json")

# ---------------- the safe rule DSL ----------------
# Whitelisted features (the ONLY names a clause may reference) mapped to the
# (lo, hi) range the systematic sampler draws thresholds from. These are derived
# from the shared feature core, so a clause means the same thing in backtest and
# live. `ema_cross` is a sign feature (compare against 0).
FEATURES = {
    "rsi":       (25.0, 75.0),    # RSI-14, 0..100
    "pos20":     (0.05, 0.95),    # position in the 20-bar range, 0..1
    "mom_1h":    (-0.02, 0.02),   # short momentum (fractional)
    "mom_4h":    (-0.04, 0.04),   # medium momentum (fractional)
    "macd_norm": (-0.8, 0.8),     # MACD 1-bar delta / ATR
    "vol_ratio": (0.7, 1.8),      # volume vs 20-bar average
    "ema_cross": (0.0, 0.0),      # +1 if ema12>ema26 else -1 (sign; thr fixed 0)
}
OPS = (">", "<")

# ---------------- honest promotion gate thresholds ----------------
DSR_MIN = 0.90               # deflated-Sharpe bar (prices the multiple testing)
MIN_FRAC_FOLDS_POSITIVE = 0.6
MIN_TRADES = 20              # enough OOS activity to trust the estimate
DEFAULT_FEE_BPS = 6.0        # round-turn cost charged on position turnover
TOP_K = 6                    # max promoted candidates kept per product
MAX_CLAUSES = 3


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def derive(f: dict) -> dict:
    """Map a raw feature dict (from features_from_ohlcv, or the live feed) to the
    DSL feature space. Used IDENTICALLY in backtest and live so a clause is
    threshold-for-threshold the same in both — this is the parity guarantee."""
    price = f.get("price", 0.0) or 0.0
    hi20 = f.get("hi20", price)
    lo20 = f.get("lo20", price)
    rng = (hi20 - lo20) or 1e-9
    atr = f.get("atr", 0.0) or 1e-9
    return {
        "rsi": f.get("rsi", 50.0),
        "pos20": _clip((price - lo20) / rng, 0.0, 1.0),
        "mom_1h": f.get("mom_1h", 0.0),
        "mom_4h": f.get("mom_4h", 0.0),
        "macd_norm": _clip(f.get("macd_delta", 0.0) / atr, -5.0, 5.0),
        "vol_ratio": f.get("vol_ratio", 1.0),
        "ema_cross": 1.0 if f.get("ema12", 0.0) > f.get("ema26", 0.0) else -1.0,
    }


def _clause_holds(clause: dict, d: dict) -> bool:
    val = d.get(clause["feat"])
    if val is None:
        return False
    thr = clause["thr"]
    return val > thr if clause["op"] == ">" else val < thr


def _vote_derived(candidate: dict, d: dict) -> float:
    """Directional vote in [-1,1] from an already-derived feature dict."""
    longs = candidate.get("long") or []
    if longs and all(_clause_holds(c, d) for c in longs):
        return _clip(float(candidate.get("conf", 0.5)))
    shorts = candidate.get("short") or []
    if shorts and all(_clause_holds(c, d) for c in shorts):
        return -_clip(float(candidate.get("conf", 0.5)))
    return 0.0


def vote(candidate: dict, f: dict) -> float:
    """Public: score a candidate on a LIVE feature dict. engine.strat_discovered
    calls this; it delegates to the same _vote_derived the backtest uses."""
    try:
        return _vote_derived(candidate, derive(f))
    except Exception:
        return 0.0


def validate_candidate(c) -> bool:
    """True if `c` is a well-formed, SAFE candidate spec (whitelisted features/ops,
    numeric thresholds, at least one directional clause set, sane confidence).
    This is the trust boundary for LLM-proposed specs — nothing else runs them."""
    if not isinstance(c, dict):
        return False
    try:
        conf = float(c.get("conf", 0.5))
    except (TypeError, ValueError):
        return False
    if not (0.0 < conf <= 1.0):
        return False
    has_side = False
    for side in ("long", "short"):
        clauses = c.get(side)
        if clauses is None:
            continue
        if not isinstance(clauses, list) or not (1 <= len(clauses) <= MAX_CLAUSES):
            return False
        for cl in clauses:
            if not isinstance(cl, dict):
                return False
            if cl.get("feat") not in FEATURES:
                return False
            if cl.get("op") not in OPS:
                return False
            try:
                float(cl["thr"])
            except (KeyError, TypeError, ValueError):
                return False
        has_side = True
    return has_side


def candidate_id(c: dict) -> str:
    """Stable canonical id for dedup/naming (independent of dict ordering)."""
    def norm(side):
        return sorted((cl["feat"], cl["op"], round(float(cl["thr"]), 4))
                      for cl in (c.get(side) or []))
    key = {"long": norm("long"), "short": norm("short"),
           "conf": round(float(c.get("conf", 0.5)), 3)}
    raw = json.dumps(key, sort_keys=True)
    import hashlib
    return "disc_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


# ---------------- feature precompute + backtest ----------------

def feature_series(candles):
    """Precompute the derived-feature dict at every bar ONCE (None during warmup),
    so evaluating many candidates over the same candles is cheap. Uses the shared
    feature core on each trailing window — no lookahead."""
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[1] for c in candles]
    vols = [c[5] for c in candles]
    out = [None] * len(candles)
    for i in range(len(candles)):
        if (i + 1) < MIN_BARS:
            continue
        f = features_from_ohlcv(closes[:i + 1], highs[:i + 1],
                                lows[:i + 1], vols[:i + 1])
        if f:
            out[i] = derive(f)
    return out


def _backtest_returns(candidate, feats, closes, fee_bps=DEFAULT_FEE_BPS):
    """Per-bar strategy returns (position * next-bar return - turnover cost) and
    the entry count. Long/flat/short from the candidate's vote sign."""
    rets = []
    pos = 0.0
    trades = 0
    fee = fee_bps * 1e-4
    for i in range(len(feats) - 1):
        d = feats[i]
        if d is None:
            newpos = 0.0
        else:
            v = _vote_derived(candidate, d)
            newpos = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
        turn = abs(newpos - pos)
        if closes[i] > 0:
            r = closes[i + 1] / closes[i] - 1.0
        else:
            r = 0.0
        rets.append(newpos * r - turn * fee)
        if turn > 0 and newpos != 0.0:
            trades += 1
        pos = newpos
    return rets, trades


def evaluate(candidate, feats, closes, folds=4, embargo=24,
             fee_bps=DEFAULT_FEE_BPS):
    """Purged walk-forward evaluation. Returns pooled out-of-sample metrics, or
    None if the candidate produced too little to judge."""
    rets, trades = _backtest_returns(candidate, feats, closes, fee_bps)
    n = len(rets)
    fold_idx = walk_forward_folds(n, n_folds=folds, embargo=embargo)
    oos_pool, fold_sharpes = [], []
    for (_ts0, _te0, vs, ve) in fold_idx:
        seg = rets[vs:ve]
        if len(seg) < 5:
            continue
        oos_pool.extend(seg)
        fold_sharpes.append(sharpe(seg))
    if len(fold_sharpes) < 2 or len(oos_pool) < 20:
        return None
    frac_pos = sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
    return {
        "oos_returns": oos_pool,
        "oos_sharpe": round(sharpe(oos_pool), 4),
        "full_sharpe": round(sharpe(rets), 4),
        "frac_folds_positive": round(frac_pos, 3),
        "n_folds": len(fold_sharpes),
        "trades": trades,
        "oos_mean_bps": round(statistics.mean(oos_pool) * 1e4, 3),
    }


def gate(metrics, n_trials, trial_sr_std):
    """(passed, dsr). Deflated Sharpe on pooled OOS returns prices the multiple
    testing (n_trials candidates tried this run); plus consistency + activity."""
    if not metrics:
        return False, 0.0
    dsr = deflated_sharpe_ratio(metrics["oos_returns"], n_trials=max(2, n_trials),
                                trial_sr_std=trial_sr_std)
    passed = (dsr >= DSR_MIN
              and metrics["oos_sharpe"] > 0
              and metrics["frac_folds_positive"] >= MIN_FRAC_FOLDS_POSITIVE
              and metrics["trades"] >= MIN_TRADES)
    return passed, round(dsr, 4)


# ---------------- candidate sources ----------------

def sample_systematic(rnd, n):
    """Reproducible random candidates drawn from the DSL space."""
    feats = list(FEATURES)
    out = []
    for _ in range(n):
        k = rnd.randint(1, min(2, MAX_CLAUSES))
        chosen = rnd.sample(feats, k)
        side = rnd.choice(["long", "short"])
        clauses = []
        for fname in chosen:
            lo, hi = FEATURES[fname]
            thr = 0.0 if fname == "ema_cross" else round(rnd.uniform(lo, hi), 4)
            clauses.append({"feat": fname, "op": rnd.choice(OPS), "thr": thr})
        out.append({side: clauses, "conf": round(rnd.uniform(0.4, 0.8), 2),
                    "meta": {"source": "systematic"}})
    return out


def propose_llm(n=8):
    """Ask the LLM advisor to propose candidate strategy specs in the DSL. Returns
    a list of VALIDATED candidates (invalid/garbage specs dropped). Reuses the
    LLM-advisor endpoint/credential plumbing; a no-key / disabled advisor yields
    []. Nothing here can promote a strategy — every proposal still faces the gate."""
    try:
        import httpx
        from .llm_advisor import LLMAdvisor
        if not LLMAdvisor.enabled() or not LLMAdvisor._api_key():
            return []
        feat_desc = ", ".join(f"{k} in [{lo},{hi}]" for k, (lo, hi) in FEATURES.items())
        sys_prompt = (
            "You are a quantitative crypto strategist proposing NEW rule-based "
            "trading strategies for later out-of-sample backtesting. Respond with "
            "ONLY a JSON array of strategy objects. Each object: "
            '{"long": [clauses], "short": [clauses], "conf": <0..1>} where each '
            'clause is {"feat": <name>, "op": ">"|"<", "thr": <number>}. Include '
            'at least one of "long"/"short". Allowed feats and typical ranges: '
            + feat_desc + ". ema_cross is +1/-1 (use thr 0). Keep 1-3 clauses per "
            "side. Propose DIVERSE, economically-plausible ideas (trend, mean-"
            "reversion, breakout, volume-confirmed, momentum). No prose, JSON only."
        )
        payload = {
            "model": LLMAdvisor._model(), "temperature": 0.7, "max_tokens": 900,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Propose {n} strategy candidates."},
            ],
        }
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{LLMAdvisor._base_url()}/chat/completions",
                       headers={"Authorization": f"Bearer {LLMAdvisor._api_key()}"},
                       json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        arr = _extract_json_array(content)
        out = []
        for c in arr:
            if isinstance(c, dict):
                c.setdefault("conf", 0.5)
                c["meta"] = {"source": "llm"}
                if validate_candidate(c):
                    out.append(c)
        return out
    except Exception:
        return []


def _extract_json_array(content):
    """Best-effort parse of a JSON array from an LLM response (tolerates fences)."""
    if not content:
        return []
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        pass
    i, j = s.find("["), s.rfind("]")
    if 0 <= i < j:
        try:
            v = json.loads(s[i:j + 1])
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


# ---------------- the researcher ----------------

class Researcher:
    def __init__(self, store_path=STORE_PATH):
        self.store_path = store_path
        self.portfolio = {}     # product -> [ {id, candidate, metrics, dsr, ts} ]
        self.last_run = {}      # product -> summary dict
        self._lock = threading.Lock()
        self._load()

    # ---- persistence ----
    def _load(self):
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            port = data.get("portfolio", {})
            # defensive: only keep entries whose candidate still validates
            clean = {}
            for prod, entries in port.items():
                kept = [e for e in entries
                        if isinstance(e, dict) and validate_candidate(e.get("candidate"))]
                if kept:
                    clean[prod] = kept
            self.portfolio = clean
            self.last_run = data.get("last_run", {})
        except Exception:
            self.portfolio, self.last_run = {}, {}

    def _save(self):
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.store_path))
            with os.fdopen(fd, "w") as f:
                json.dump({"portfolio": self.portfolio,
                           "last_run": self.last_run}, f)
            os.replace(tmp, self.store_path)
        except Exception:
            pass

    # ---- live-arm accessor (called by engine.strat_discovered) ----
    def portfolio_for(self, product):
        with self._lock:
            return [e["candidate"] for e in self.portfolio.get(product, [])]

    # ---- one discovery run for a product ----
    def run_once(self, product, candles=None, n_systematic=None, use_llm=None,
                 seed=None):
        """Search, validate out-of-sample, and (auto) promote for one product.

        Returns a summary dict. Only PROMOTES candidates that clear the honest
        gate; if this run finds none, the existing (previously validated)
        portfolio for the product is left untouched to avoid churn on an unlucky
        draw. Never raises: any failure returns an {ok: False} summary."""
        try:
            from . import researcher as _self  # noqa
            from .. import settings as app_settings
            if n_systematic is None:
                try:
                    n_systematic = int(app_settings.get("researcher_candidates"))
                except Exception:
                    n_systematic = 60
            if use_llm is None:
                use_llm = bool(app_settings.get("researcher_use_llm"))

            candles = candles if candles is not None else self._fetch_candles(product)
            if not candles or len(candles) < MIN_BARS + 60:
                return {"ok": False, "product": product,
                        "error": f"not enough candle history ({len(candles or [])})"}

            feats = feature_series(candles)
            closes = [c[4] for c in candles]

            rnd = random.Random(seed if seed is not None
                                else (hash(product) ^ int(time.time()) & 0xffffffff))
            candidates = sample_systematic(rnd, n_systematic)
            n_llm = 0
            if use_llm:
                llm_c = propose_llm(8)
                n_llm = len(llm_c)
                candidates += llm_c

            # dedup by canonical id
            seen, uniq = set(), []
            for c in candidates:
                cid = candidate_id(c)
                if cid not in seen:
                    seen.add(cid)
                    uniq.append((cid, c))

            scored = []
            for cid, c in uniq:
                m = evaluate(c, feats, closes)
                if m:
                    scored.append((cid, c, m))
            n_trials = max(2, len(scored))
            # multiple-testing dispersion: spread of the trials' full-sample Sharpes
            fulls = [m["full_sharpe"] for _cid, _c, m in scored]
            trial_sr_std = statistics.pstdev(fulls) if len(fulls) > 1 else None

            passers = []
            for cid, c, m in scored:
                ok, dsr = gate(m, n_trials, trial_sr_std)
                if ok:
                    m2 = {k: v for k, v in m.items() if k != "oos_returns"}
                    passers.append({"id": cid, "candidate": c, "metrics": m2,
                                    "dsr": dsr, "ts": time.time()})
            passers.sort(key=lambda e: (e["dsr"], e["metrics"]["oos_sharpe"]),
                         reverse=True)
            passers = passers[:TOP_K]

            summary = {
                "ok": True, "product": product, "ts": time.time(),
                "tested": len(uniq), "evaluated": len(scored),
                "n_systematic": n_systematic, "n_llm": n_llm,
                "promoted": len(passers),
                "trial_sr_std": round(trial_sr_std, 4) if trial_sr_std else None,
                "top": [{"id": e["id"], "dsr": e["dsr"],
                         "oos_sharpe": e["metrics"]["oos_sharpe"],
                         "trades": e["metrics"]["trades"],
                         "candidate": e["candidate"]} for e in passers[:3]],
            }
            with self._lock:
                if passers:                         # replace only on a fruitful run
                    self.portfolio[product] = passers
                self.last_run[product] = summary
                self._save()
            try:
                from .. import db
                if passers:
                    db.log_event("learn", f"Researcher promoted {len(passers)} "
                                 f"discovered strategy(ies) for {product} "
                                 f"(best DSR={passers[0]['dsr']}) from "
                                 f"{len(scored)} tested")
            except Exception:
                pass
            return summary
        except Exception as e:
            return {"ok": False, "product": product, "error": str(e)}

    def run_universe(self, products=None, **kw):
        """Run discovery across all configured products (or a subset)."""
        if products is None:
            from ..config import PRODUCTS
            products = list(PRODUCTS)
        results = [self.run_once(p, **kw) for p in products]
        return {"ok": True, "runs": results,
                "promoted_total": sum(r.get("promoted", 0) for r in results
                                      if r.get("ok"))}

    def _fetch_candles(self, product):
        """Fetch hourly history via the backtest engine's cached fetcher. Falls
        back to the live in-memory candles if the network fetch is unavailable."""
        try:
            import asyncio
            from ..backtest.engine import fetch_history
            return asyncio.run(fetch_history(product, granularity=3600, chunks=3))
        except Exception:
            try:
                from ..data.market import market
                return list(market.candles.get(product, []))
            except Exception:
                return []

    # ---- introspection / control ----
    def status(self):
        from .. import settings as app_settings
        with self._lock:
            port = {p: [{"id": e["id"], "dsr": e["dsr"],
                         "oos_sharpe": e["metrics"]["oos_sharpe"],
                         "oos_mean_bps": e["metrics"].get("oos_mean_bps"),
                         "trades": e["metrics"]["trades"],
                         "source": (e["candidate"].get("meta") or {}).get("source"),
                         "candidate": e["candidate"]}
                        for e in entries]
                    for p, entries in self.portfolio.items()}
            last = dict(self.last_run)
        return {
            "enabled": bool(app_settings.get("researcher_enabled")),
            "use_llm": bool(app_settings.get("researcher_use_llm")),
            "gate": {"dsr_min": DSR_MIN,
                     "min_frac_folds_positive": MIN_FRAC_FOLDS_POSITIVE,
                     "min_trades": MIN_TRADES, "top_k": TOP_K},
            "features": list(FEATURES),
            "promoted_products": len(port),
            "promoted_total": sum(len(v) for v in port.values()),
            "portfolio": port,
            "last_run": last,
        }

    def clear(self):
        with self._lock:
            self.portfolio, self.last_run = {}, {}
            self._save()
        return {"ok": True, "cleared": True}


researcher = Researcher()
