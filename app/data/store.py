"""Persistent, versioned market history for replay and strategy engines."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, ".cache", "store")


@contextmanager
def _connection():
    os.makedirs(STORE, exist_ok=True)
    conn = sqlite3.connect(os.path.join(STORE, "history.sqlite3"), timeout=30)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
                series TEXT NOT NULL, product TEXT NOT NULL,
                ts INTEGER NOT NULL, observed REAL NOT NULL,
                payload TEXT NOT NULL, source TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS history_lookup
            ON history (kind, series, product, ts, observed)
        """)
        with conn:
            yield conn
    finally:
        conn.close()


def fingerprint(data):
    """Stable content hash, independent of ingestion time and dictionary order."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ingest(kind, series, product, rows, now, source, normalize):
    now = time.time() if now is None else float(now)
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    accepted = {}
    rejected = 0
    for row in rows:
        try:
            normalized = normalize(row, now)
        except (TypeError, ValueError, IndexError, OverflowError):
            rejected += 1
            continue
        if normalized is not None:
            accepted[int(normalized[0])] = normalized
    stats = {"new": 0, "revised": 0, "rejected": rejected, "rows": 0}
    with _connection() as conn:
        # Acquire the writer lock before reading so concurrent ingests cannot
        # mistake an already-stored bar for a new one.
        conn.execute("BEGIN IMMEDIATE")
        for ts, row in sorted(accepted.items()):
            previous = conn.execute("""
                SELECT payload, observed FROM history
                WHERE kind=? AND series=? AND product=? AND ts=?
                ORDER BY observed DESC, id DESC LIMIT 1
            """, (kind, str(series), product, ts)).fetchone()
            payload = json.dumps(row, separators=(",", ":"), allow_nan=False)
            if previous and previous[0] == payload:
                continue
            if previous and now < previous[1]:
                raise ValueError("ingestion time precedes the latest revision")
            conn.execute("""
                INSERT INTO history (kind, series, product, ts, observed, payload, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (kind, str(series), product, ts, now, payload, source))
            stats["revised" if previous else "new"] += 1
        stats["rows"] = conn.execute("""
            SELECT COUNT(DISTINCT ts) FROM history
            WHERE kind=? AND series=? AND product=?
        """, (kind, str(series), product)).fetchone()[0]
    with open(os.path.join(STORE, "ingest_log.jsonl"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"kind": kind, "series": series, "product": product,
                              "observed": now, "source": source, **stats}) + "\n")
    return stats


def ingest_candles(product, granularity, rows, now=None, source="coinbase"):
    """Store closed Coinbase-format [time, low, high, open, close, volume] bars."""
    if not isinstance(granularity, int) or granularity <= 0:
        raise ValueError("granularity must be a positive integer")

    def normalize(row, observed):
        if len(row) != 6:
            raise ValueError("a candle must have six fields")
        values = [float(v) for v in row]
        ts, low, high, opening, close, volume = values
        if (not all(math.isfinite(v) for v in values)
                or ts < 0 or ts != int(ts) or int(ts) % granularity
                or min(low, high, opening, close) <= 0 or low > high or volume < 0):
            raise ValueError("invalid candle")
        if ts + granularity > observed:
            return None
        values[0] = int(ts)
        return values

    return _ingest("candles", granularity, product, rows, now, source, normalize)


def ingest_funding(venue, product, rows, now=None, source=None):
    """Store timestamp/rate pairs without discarding historical revisions."""
    def normalize(row, observed):
        if len(row) != 2:
            raise ValueError("funding must have two fields")
        ts, rate = [float(v) for v in row]
        if not math.isfinite(ts) or not math.isfinite(rate) or ts < 0 or ts != int(ts):
            raise ValueError("invalid funding")
        return [int(ts), rate] if ts <= observed else None

    return _ingest("funding", venue, product, rows, now, source or venue, normalize)


def _load(kind, series, selected=None, start=None, end=None, as_of=None):
    cutoff = math.inf if as_of is None else float(as_of)
    wanted = None if selected is None else set(selected)
    result = {}
    with _connection() as conn:
        rows = conn.execute("""
            SELECT product, ts, payload FROM (
                SELECT product, ts, payload,
                    ROW_NUMBER() OVER (
                        PARTITION BY product, ts ORDER BY observed DESC, id DESC
                    ) AS revision
                FROM history WHERE kind=? AND series=? AND observed<=?
            ) WHERE revision=1 ORDER BY product, ts
        """, (kind, str(series), cutoff))
        for product, ts, payload in rows:
            if wanted is not None and product not in wanted:
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts >= end:
                continue
            if kind == "candles" and as_of is not None and ts + int(series) > cutoff:
                continue
            result.setdefault(product, []).append(json.loads(payload))
    return result


def load_candles(granularity=3600, products=None, start=None, end=None, as_of=None):
    """Latest known closed bars, optionally restricted to a historical snapshot."""
    data = _load("candles", granularity, products, start, end, as_of)
    return data, fingerprint(data)


def load_funding(venue, products=None, start=None, end=None, as_of=None):
    data = _load("funding", venue, products, start, end, as_of)
    return data, fingerprint(data)


def products(kind="candles", series=3600):
    with _connection() as conn:
        return [r[0] for r in conn.execute("""
            SELECT DISTINCT product FROM history
            WHERE kind=? AND series=? ORDER BY product
        """, (kind, str(series)))]


def last_ts(kind, series, product):
    with _connection() as conn:
        return conn.execute("""
            SELECT MAX(ts) FROM history WHERE kind=? AND series=? AND product=?
        """, (kind, str(series), product)).fetchone()[0]


def quality(granularity=3600):
    data, _ = load_candles(granularity)
    result = {}
    with _connection() as conn:
        for product, rows in data.items():
            gaps = [max(0, (b[0] - a[0]) // granularity - 1)
                    for a, b in zip(rows, rows[1:])]
            missing = sum(gaps)
            revisions = conn.execute("""
                SELECT COUNT(*) - COUNT(DISTINCT ts) FROM history
                WHERE kind='candles' AND series=? AND product=?
            """, (str(granularity), product)).fetchone()[0]
            spikes = sum(
                abs(b[4] / a[4] - 1) > 0.5 and abs(c[4] / a[4] - 1) < 0.1
                for a, b, c in zip(rows, rows[1:], rows[2:])
                if b[0] - a[0] == granularity and c[0] - b[0] == granularity)
            result[product] = {
                "rows": len(rows), "missing_bars": missing,
                "longest_gap_bars": max(gaps, default=0),
                "coverage_pct": 100 * len(rows) / (len(rows) + missing),
                "reverting_spikes": spikes, "revisions": revisions,
            }
    return result


def is_tradeable_asset(product):
    symbol = product.split("-")[0].upper()
    return symbol not in {
        "USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "GUSD", "PYUSD",
        "UST", "USTC", "EURC", "EUR", "USD", "PAX", "FDUSD", "USDE",
        "WBTC", "WETH", "CBETH", "STETH", "WSTETH", "RETH",
    }


def universe_at(ts, n=20, lookback=30, min_history=100, min_dollar_volume=1e6):
    """Liquid daily universe at a bar close, excluding future and delisted coins."""
    if n <= 0:
        return []
    day = int(ts) // 86400 * 86400
    data, _ = load_candles(86400, end=day + 86400)
    ranked = []
    for product, rows in data.items():
        if not is_tradeable_asset(product) or len(rows) < min_history or rows[-1][0] != day:
            continue
        recent = [r for r in rows if r[0] >= day - (lookback - 1) * 86400]
        dollar_volume = sum(r[4] * r[5] for r in recent) / lookback
        if dollar_volume >= min_dollar_volume:
            ranked.append((product, dollar_volume))
    return [p for p, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))[:n]]
