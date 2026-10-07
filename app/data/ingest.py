"""Generic external-signal ingest — a license-safe seam for ANY standalone
data producer (crawl4ai, Maxun, a cron curl, a notebook…).

Design goals, straight from the operator's constraints:

  * LICENSE-SAFE BY CONSTRUCTION. This module NEVER imports crawl4ai, Maxun, or
    any scraper. It only ingests their OUTPUT (structured rows). That means a
    copyleft producer (Maxun, AGPL) can be run as a separate service and its
    *data* consumed here with zero license reach into CryptoMind, and a
    permissive producer (crawl4ai, Apache-2.0) is treated identically. The
    boundary is data, not code.

  * MEASUREMENT-FIRST. Ingested rows do not silently steer trades. They land in
    a store, expose a per-asset feature + recent texts, and ship with a STUDY
    (rank-IC of the signal vs realized forward return from CryptoMind's own
    candles). A signal earns trading influence only via an explicit, gated path
    (e.g. feeding the LLM advisor's context, which the bandit already weights) —
    never blind copy. Same philosophy as the Invo integration.

Row schema (normalized): {source, ts, asset, value, kind, text, meta}
  - asset  : ticker symbol or product ("BTC" or "BTC-USD"); folded to symbol.
  - value  : float, ideally a directional lean in [-1, 1] (bounded on ingest).
  - kind   : free label ("news_lean", "sentiment", "headline_score", …).
  - text   : short human-readable rationale/headline (used for LLM context).
  - ts     : unix seconds (defaults to now).
"""
from __future__ import annotations

import json
import math
import os
import time

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "external_signals.jsonl")
MAX_ROWS = 5000                 # bounded live view (full history: app/data/series.py)
DEFAULT_MAX_AGE = 6 * 3600      # a signal older than 6h no longer feeds features


def _sym(asset: str) -> str:
    return str(asset or "").split("-")[0].strip().upper()


def _f(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def normalize_row(raw: dict) -> dict | None:
    """Coerce one incoming row into the stored schema, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    asset = _sym(raw.get("asset") or raw.get("symbol") or raw.get("product"))
    if not asset:
        return None
    val = _f(raw.get("value", raw.get("lean", raw.get("score"))), None)
    if val is None:
        return None
    val = max(-1.0, min(1.0, val))            # bound to a lean range on ingest
    ts = _f(raw.get("ts"), None) or time.time()
    return {
        "source": str(raw.get("source", "unknown"))[:64],
        "ts": float(ts),
        "asset": asset,
        "value": val,
        "kind": str(raw.get("kind", "signal"))[:32],
        "text": str(raw.get("text", ""))[:300],
        "meta": raw.get("meta") if isinstance(raw.get("meta"), dict) else {},
    }


def _read_all() -> list[dict]:
    out = []
    try:
        with open(PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return out


def _write_all(rows: list[dict]):
    rows = rows[-MAX_ROWS:]
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, PATH)


def push(rows) -> dict:
    """Ingest one row or a list of rows. Returns {accepted, rejected}."""
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return {"accepted": 0, "rejected": 0, "error": "rows must be a list"}
    norm = [normalize_row(r) for r in rows]
    good = [r for r in norm if r]
    if good:
        existing = _read_all()
        existing.extend(good)
        _write_all(existing)
        _to_history(good)
    return {"accepted": len(good), "rejected": len(rows) - len(good)}


def _to_history(rows):
    """Also keep every accepted row, uncapped, in the point-in-time series
    store (app/data/series.py) as `ext:<kind>:<asset>`, known from the moment
    it arrived. The jsonl above is a bounded live view; this is the history a
    signal must build up before it can be tested (tools/signal_screen.py)."""
    try:
        from . import series
        now = time.time()
        by = {}
        for r in rows:
            by.setdefault(f"ext:{r['kind']}:{r['asset']}", []).append(
                (int(min(r["ts"], now)), r["value"], now))
        for name, pts in by.items():
            series.ingest(name, pts, source="ingest", now=now)
    except Exception:
        pass                      # history is best-effort; the live path never fails on it


def feature(product: str, max_age_sec: float = DEFAULT_MAX_AGE) -> float | None:
    """Latest (freshness-weighted mean) signal value for an asset, in ~[-1, 1].

    Averages all fresh rows for the symbol, weighting more recent rows higher
    (linear decay to the age cutoff). None when there is no fresh signal.
    """
    sym = _sym(product)
    now = time.time()
    num = den = 0.0
    for r in _read_all():
        if r.get("asset") != sym:
            continue
        age = now - float(r.get("ts", 0))
        if age < 0 or age > max_age_sec:
            continue
        w = 1.0 - age / max_age_sec        # 1 at now → 0 at the cutoff
        num += w * float(r.get("value", 0.0))
        den += w
    if den <= 0:
        return None
    return max(-1.0, min(1.0, num / den))


def latest_texts(product: str, n: int = 3,
                 max_age_sec: float = DEFAULT_MAX_AGE) -> list[str]:
    """Most-recent non-empty rationale/headline texts for an asset — the payload
    injected into the LLM advisor's context so its vote reflects fresh crawl."""
    sym = _sym(product)
    now = time.time()
    rows = [r for r in _read_all()
            if r.get("asset") == sym and r.get("text")
            and 0 <= now - float(r.get("ts", 0)) <= max_age_sec]
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return [f"[{r['source']}] {r['text']}" for r in rows[:n]]


def status() -> dict:
    rows = _read_all()
    now = time.time()
    per_source, per_asset, last_ts = {}, {}, {}
    for r in rows:
        s, a = r.get("source", "?"), r.get("asset", "?")
        per_source[s] = per_source.get(s, 0) + 1
        per_asset[a] = per_asset.get(a, 0) + 1
        last_ts[s] = max(last_ts.get(s, 0), float(r.get("ts", 0)))
    fresh = sum(1 for r in rows
                if 0 <= now - float(r.get("ts", 0)) <= DEFAULT_MAX_AGE)
    return {
        "total_rows": len(rows),
        "fresh_rows": fresh,
        "sources": per_source,
        "assets": sorted(per_asset.keys()),
        "last_ts_by_source": {k: round(v, 0) for k, v in last_ts.items()},
        "path": PATH,
    }


def recent(product: str | None = None, n: int = 20) -> list[dict]:
    rows = _read_all()
    if product:
        sym = _sym(product)
        rows = [r for r in rows if r.get("asset") == sym]
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[:n]


def clear() -> dict:
    try:
        os.remove(PATH)
    except FileNotFoundError:
        pass
    return {"ok": True}


# ------------------------------- study --------------------------------------
def _forward_return_prices() -> dict:
    """{SYMBOL: [[ts, close], ...]} from CryptoMind's own live candles."""
    from .market import market
    out = {}
    for prod, candles in market.candles.items():
        sym = _sym(prod)
        rows = [[float(c[0]), float(c[4])] for c in candles if len(c) >= 5]
        if rows:
            out[sym] = rows
    return out


def _fwd_return(price_rows, ts, horizon_sec):
    """Return over `horizon_sec` starting at the close nearest-after `ts`."""
    p0 = p1 = None
    for t, c in price_rows:
        if p0 is None and t >= ts:
            p0 = c
            t0 = t
        if p0 is not None and t >= t0 + horizon_sec:
            p1 = c
            break
    if p0 and p1 and p0 > 0:
        return p1 / p0 - 1.0
    return None


def study(horizon_hours: float = 4.0) -> dict:
    """Rank-IC of the ingested signal vs realized forward return, per asset +
    pooled. Honest measurement BEFORE the signal is allowed to influence trades.

    Returns {ok, pooled:{ic,n}, per_asset:{sym:{ic,n}}, verdict}.
    """
    from tools.invo_signal.study import spearman   # reuse the pure-numpy IC
    prices = _forward_return_prices()
    if not prices:
        return {"ok": False, "error": "no candle history for forward returns yet"}
    horizon = horizon_hours * 3600
    rows = _read_all()
    pairs_by_asset: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        sym = r.get("asset")
        pr = prices.get(sym)
        if not pr:
            continue
        fwd = _fwd_return(pr, float(r.get("ts", 0)), horizon)
        if fwd is None:
            continue
        pairs_by_asset.setdefault(sym, []).append((float(r["value"]), fwd))

    per_asset, all_sig, all_fwd = {}, [], []
    for sym, pairs in pairs_by_asset.items():
        if len(pairs) < 5:
            continue
        sig = [p[0] for p in pairs]
        fwd = [p[1] for p in pairs]
        import numpy as np
        ic = spearman(np.array(sig), np.array(fwd))
        per_asset[sym] = {"ic": round(ic, 4), "n": len(pairs)}
        all_sig += sig
        all_fwd += fwd

    if len(all_sig) < 8:
        return {"ok": False, "error": f"not enough matured pairs yet "
                f"({len(all_sig)}); let the producer + candles accumulate"}
    import numpy as np
    pooled_ic = spearman(np.array(all_sig), np.array(all_fwd))
    a = abs(pooled_ic)
    verdict = ("promising — measurable predictive edge" if a >= 0.05
               else "weak — no clear edge yet" if a >= 0.02
               else "no edge — do not let it influence trades")
    return {"ok": True,
            "pooled": {"ic": round(pooled_ic, 4), "n": len(all_sig)},
            "per_asset": per_asset, "horizon_hours": horizon_hours,
            "verdict": verdict}
