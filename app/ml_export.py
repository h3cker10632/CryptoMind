"""Training-grade ML dataset emitter (Phase 1 of the crypto_ml_lab integration).

The full-state export (`app/export.py`) is a SNAPSHOT — great for diagnosis, but
its `signal_scores` are the newest rows, which are almost all still UNLABELLED
(CryptoMind fills `fwd_return` ~1h later, once the forward horizon elapses). An
offline learner therefore receives features with null targets and cannot train
honestly. See docs/crypto_ml_lab_integration.md.

This module emits a LABEL-COMPLETE dataset instead. It produces two artefacts:

  1. events  — a per-bar time series in the lab's ``{"events": [...]}`` contract
     (ts, symbol, price, volume, liquidity, buy/sell proxy, bot_signal, ...).
     crypto_ml_lab's ``prepare`` command derives its own leakage-safe causal
     features + forward labels from this, so we do NOT ship any lookahead here.

  2. labels  — the bot's OWN realized decisions as supervised truth: only signals
     with a realized ``fwd_return`` (scored=1), joined to the decision-time
     context we already log (votes, regime, composite, confidence, ml_confidence,
     size pre/post, action, gated reason) and, where available, the trade outcome
     (pnl, MAE, exit reason, hold time).

Access:
  * live:     GET /api/ml_dataset            (JSON; download=true → file)
  * offline:  python -m app.ml_export out.json
              python -m app.ml_export outdir/ --split   (events.json + labels.json)

Nothing here touches the trading/learning core; it only READS existing tables and
in-memory market state.
"""
import json
import time


SCHEMA = "cryptomind.ml_dataset.v1"


# ---------------------------------------------------------------- bar-level events
def _bar_events(market, derivatives, nlp):
    """Build the lab-shaped per-bar event list from in-memory candle history.

    Each 5-minute candle becomes one event per product. We attach ONLY fields
    known at (or before) that bar's close — no forward information. Order-book
    depth/imbalance and sentiment are current-snapshot values stamped on the most
    recent bar only (they are not historical), so older bars leave them null
    rather than back-fill a value that wasn't known then.
    """
    from .config import PRODUCTS, CANDLE_GRANULARITY
    events = []
    now = time.time()
    for p in PRODUCTS:
        candles = list(market.candles.get(p, []))
        if not candles:
            continue
        book = market.books.get(p, {})
        asset_sent = None
        try:
            asset_sent = nlp.asset_sentiment.get(p, {}).get("score")
        except Exception:
            pass
        last_i = len(candles) - 1
        for i, c in enumerate(candles):
            # candle = [ts, low, high, open, close, volume]
            try:
                cts, low, high, cop, close, vol = c[0], c[1], c[2], c[3], c[4], c[5]
            except (IndexError, TypeError):
                continue
            is_last = (i == last_i)
            ev = {
                "ts": _iso(cts),
                "symbol": p,
                "price": _num(close),
                "volume": _num(vol),
                "open": _num(cop),
                "high": _num(high),
                "low": _num(low),
                "venue": "coinbase",
                # book depth is a live snapshot → only meaningful on the last bar
                "liquidity": _num(book.get("bid_depth", 0) + book.get("ask_depth", 0))
                             if is_last and book else None,
                "book_imbalance": _num(book.get("imbalance")) if is_last and book else None,
                "spread_bps": _num(book.get("spread_bps")) if is_last and book else None,
                "sentiment": _num(asset_sent) if is_last else None,
            }
            events.append(ev)
    events.sort(key=lambda e: (e["symbol"], e["ts"]))
    return events


# ------------------------------------------------------- labelled decisions/signals
def _labeled_rows(db, limit):
    """Join realized-labelled signals to decision-time context + trade outcomes.

    Only signals with a non-null realized ``fwd_return`` are emitted — these are
    the sole rows valid as supervised truth. Decision context is matched to the
    nearest decision for the same product at or before the signal's timestamp
    (decisions and signals are both logged per cycle but not with a shared id).
    """
    signals = db.labeled_signals(limit=limit)
    decisions = db.all_decisions(limit=max(limit * 4, 50000))
    closed = _closed_trades(db, limit=max(limit * 2, 20000))

    # index decisions by product, ascending ts, for a nearest-at-or-before match
    by_prod = {}
    for d in decisions:
        by_prod.setdefault(d["product"], []).append(d)
    for lst in by_prod.values():
        lst.sort(key=lambda d: d["ts"])

    # index closed trades by product for outcome attachment (entry near signal ts)
    trades_by_prod = {}
    for t in closed:
        trades_by_prod.setdefault(t.get("product"), []).append(t)
    for lst in trades_by_prod.values():
        lst.sort(key=lambda t: t.get("opened", 0) or 0)

    rows = []
    for s in signals:
        ctx = _nearest_before(by_prod.get(s["product"], []), s["ts"])
        outcome = _match_trade(trades_by_prod.get(s["product"], []), s["ts"])
        row = {
            "ts": _iso(s["ts"]),
            "ts_epoch": s["ts"],
            "symbol": s["product"],
            "strategy": s["strategy"],
            "direction": s["direction"],
            "confidence": s["confidence"],
            "regime": s.get("regime"),
            # LABEL: realized forward return over the scoring horizon
            "y_fwd_return": s["fwd_return"],
            "y_direction_correct": int(
                (s["direction"] > 0 and s["fwd_return"] > 0) or
                (s["direction"] < 0 and s["fwd_return"] < 0)),
        }
        if ctx:
            row["context"] = {
                "cycle": ctx.get("cycle"),
                "composite": ctx.get("composite"),
                "ctx_confidence": ctx.get("confidence"),
                "ml_confidence": ctx.get("ml_confidence"),
                "action": ctx.get("action"),
                "reason": ctx.get("reason"),
                "size_pre": ctx.get("size_pre"),
                "size_post": ctx.get("size_post"),
                "votes": ctx.get("votes"),
                "ctx_lag_sec": round(s["ts"] - ctx["ts"], 1),
            }
        if outcome:
            row["outcome"] = outcome
        rows.append(row)
    return rows


def _closed_trades(db, limit):
    return db.recent("trades", limit)   # raw trade log; used only for context


def _match_trade(trades, sig_ts, window_sec=1800):
    """Find a closed trade whose entry is within `window_sec` of the signal."""
    best, best_dt = None, window_sec + 1
    for t in trades:
        op = t.get("opened") or t.get("ts") or 0
        dt = abs(op - sig_ts)
        if dt < best_dt:
            best, best_dt = t, dt
    if best is None:
        return None
    return {
        "pnl": best.get("pnl"),
        "exit_reason": best.get("exit_reason") or best.get("reason"),
        "hold_sec": round((best.get("closed", 0) - best.get("opened", 0)), 1)
                    if best.get("closed") and best.get("opened") else None,
        "mae_price": best.get("mae_price"),
        "entry": best.get("entry") or best.get("price"),
    }


def _nearest_before(rows, ts):
    """Last row with row['ts'] <= ts (rows sorted ascending)."""
    import bisect
    if not rows:
        return None
    keys = [r["ts"] for r in rows]
    i = bisect.bisect_right(keys, ts) - 1
    return rows[i] if i >= 0 else None


# --------------------------------------------------------------------- utilities
def _num(x):
    try:
        f = float(x)
        return f if f == f else None   # drop NaN
    except (TypeError, ValueError):
        return None


def _iso(epoch):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))
    except (TypeError, ValueError, OSError):
        return None


# ------------------------------------------------------------------- main builder
def build_ml_dataset(limit=100000):
    """Assemble the training-grade dataset dict (events + labelled rows + meta)."""
    from . import db
    from .config import PRODUCTS, CANDLE_GRANULARITY, SIGNAL_EVAL_HORIZON_SEC
    from .data.market import market
    from .data.derivatives import derivatives
    from .nlp.sentiment import nlp

    events = _safe(lambda: _bar_events(market, derivatives, nlp), [])
    labels = _safe(lambda: _labeled_rows(db, limit), [])

    n_labeled = len(labels) if isinstance(labels, list) else 0
    return {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": {
            "products": list(PRODUCTS),
            "bar_seconds": CANDLE_GRANULARITY,
            "label_horizon_sec": SIGNAL_EVAL_HORIZON_SEC,
            "n_events": len(events) if isinstance(events, list) else 0,
            "n_labeled_signals": n_labeled,
            "note": ("`events` is a leakage-free bar time series for the lab's "
                     "own feature/label derivation; `labels` are realized-outcome "
                     "signal rows (fwd_return present) joined to decision context. "
                     "Feed `events` to crypto_ml prepare; use `labels` for "
                     "supervised diagnostics / direct training."),
        },
        # lab contract: an object with an events list is accepted directly by io.load_json
        "events": events,
        "labels": labels,
    }


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"{type(e).__name__}: {e}", "_default": default} or default


# ------------------------------------------------------------------------- CLI
def _prime_from_snapshot():
    from . import db, persistence
    db.init()
    try:
        persistence.load()
    except Exception as e:  # pragma: no cover
        print(f"[ml_export] warning: could not load saved state: {e}")


def main(argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(
        description="Emit a training-grade ML dataset for crypto_ml_lab.")
    parser.add_argument("path", nargs="?", default="cryptomind_ml_dataset.json",
                        help="output file, or a directory when --split is used")
    parser.add_argument("--limit", type=int, default=100000,
                        help="max labelled signal rows (default 100000)")
    parser.add_argument("--split", action="store_true",
                        help="write events.json + labels.json into the path dir "
                             "(events.json is directly consumable by crypto_ml)")
    parser.add_argument("--compact", action="store_true", help="minified JSON")
    args = parser.parse_args(argv)

    _prime_from_snapshot()
    data = build_ml_dataset(limit=args.limit)
    dump = (lambda o, f: json.dump(o, f, separators=(",", ":"), default=str)) if args.compact \
        else (lambda o, f: json.dump(o, f, indent=2, default=str))

    if args.split:
        os.makedirs(args.path, exist_ok=True)
        ev = os.path.join(args.path, "events.json")
        lb = os.path.join(args.path, "labels.json")
        with open(ev, "w") as f:
            dump({"events": data["events"]}, f)
        with open(lb, "w") as f:
            dump({"schema": SCHEMA, "meta": data["meta"], "labels": data["labels"]}, f)
        print(f"[ml_export] wrote {ev} ({data['meta']['n_events']} events) and "
              f"{lb} ({data['meta']['n_labeled_signals']} labelled signals)")
    else:
        with open(args.path, "w") as f:
            dump(data, f)
        print(f"[ml_export] wrote {args.path} "
              f"({data['meta']['n_events']} events, "
              f"{data['meta']['n_labeled_signals']} labelled signals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
