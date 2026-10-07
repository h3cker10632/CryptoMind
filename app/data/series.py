"""Point-in-time store for market-wide and external time series — Fear & Greed,
implied volatility, stablecoin supply, on-chain activity, ingested signals —
so an outside source can be TESTED on history before it is ever traded.

Same rules as the candle store (app/data/store.py), in its own SQLite file
next to it:

  * every value carries `known_at`: when it could first have been known (its
    publication time, or for a value first seen late — a revision, a live
    push — the time we received it);
  * a changed value never overwrites history: the old version moves to a
    revisions table (`superseded_at`), so `load(name, as_of=T)` returns the
    series exactly as it was known at T;
  * nothing is ever capped or dropped.

`daily_array(name, days)` aligns a series to a panel's day grid CAUSALLY: day
d gets the latest value known by the close of day d (forward-filled for at
most `stale_days`), so features built from it never see the future.
"""
from __future__ import annotations
import heapq
import math
import os
import sqlite3
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, ".cache", "store")
DAY = 86400
_lock = threading.Lock()


def _db():
    os.makedirs(STORE, exist_ok=True)
    c = sqlite3.connect(os.path.join(STORE, "series.sqlite3"), timeout=30)
    c.executescript("""
        CREATE TABLE IF NOT EXISTS points(
            name TEXT NOT NULL, ts INTEGER NOT NULL, value REAL NOT NULL,
            known_at REAL NOT NULL, source TEXT, PRIMARY KEY(name, ts));
        CREATE TABLE IF NOT EXISTS revisions(
            name TEXT NOT NULL, ts INTEGER NOT NULL, value REAL NOT NULL,
            known_at REAL NOT NULL, superseded_at REAL NOT NULL, source TEXT);
        CREATE INDEX IF NOT EXISTS rev_name ON revisions(name, ts);
        CREATE TABLE IF NOT EXISTS ingest_log(
            name TEXT, ran_at REAL, new INTEGER, revised INTEGER, rejected INTEGER,
            source TEXT);
    """)
    return c


def ingest(name, rows, source="", now=None):
    """Store [(ts, value, known_at)] for series `name`. Rejects non-finite
    values, values known before their own time or in the future. A changed
    value is a revision: the old version is kept, the new one counts as known
    no earlier than `now`. Returns {"new", "revised", "unchanged", "rejected"}."""
    now = now or time.time()
    st = {"new": 0, "revised": 0, "unchanged": 0, "rejected": 0}
    with _lock:
        c = _db()
        try:
            for r in rows:
                try:
                    ts, v, k = int(r[0]), float(r[1]), float(r[2])
                except (TypeError, ValueError, IndexError):
                    st["rejected"] += 1
                    continue
                if not math.isfinite(v) or k < ts or k > now + 60:
                    st["rejected"] += 1
                    continue
                old = c.execute("SELECT value, known_at, source FROM points WHERE name=? AND ts=?",
                                (name, ts)).fetchone()
                if old is None:
                    c.execute("INSERT INTO points VALUES (?,?,?,?,?)", (name, ts, v, k, source))
                    st["new"] += 1
                elif abs(old[0] - v) <= 1e-12 * max(1.0, abs(v)):
                    st["unchanged"] += 1
                else:
                    c.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?)",
                              (name, ts, old[0], old[1], now, old[2]))
                    c.execute("UPDATE points SET value=?, known_at=?, source=? "
                              "WHERE name=? AND ts=?", (v, max(k, now), source, name, ts))
                    st["revised"] += 1
            c.execute("INSERT INTO ingest_log VALUES (?,?,?,?,?,?)",
                      (name, now, st["new"], st["revised"], st["rejected"], source))
            c.commit()
        finally:
            c.close()
    return st


def _versions(name):
    """Every version of every point: (ts, value, known_at, superseded_at|inf)."""
    with _lock:
        c = _db()
        try:
            cur = c.execute("SELECT ts, value, known_at FROM points WHERE name=?",
                            (name,)).fetchall()
            rev = c.execute("SELECT ts, value, known_at, superseded_at FROM revisions "
                            "WHERE name=?", (name,)).fetchall()
        finally:
            c.close()
    return [(t, v, k, math.inf) for t, v, k in cur] + [tuple(r) for r in rev]


def load(name, as_of=None):
    """[(ts, value)] sorted by ts — as known at `as_of` (default: now)."""
    if as_of is None:
        return sorted((t, v) for t, v, k, s in _versions(name) if s == math.inf)
    best = {}
    for t, v, k, s in _versions(name):
        if k <= as_of < s:
            best[t] = v
    return sorted(best.items())


def last_ts(name):
    with _lock:
        c = _db()
        try:
            r = c.execute("SELECT MAX(ts) FROM points WHERE name=?", (name,)).fetchone()
        finally:
            c.close()
    return r[0] if r and r[0] is not None else None


def names():
    with _lock:
        c = _db()
        try:
            return [r[0] for r in c.execute("SELECT DISTINCT name FROM points ORDER BY name")]
        finally:
            c.close()


def daily_array(name, days, stale_days=7):
    """(T,) values for day indices `days` (ts // 86400): for day d, the latest
    point (by its own time) among versions KNOWN by the end of day d and not
    yet superseded then; NaN if none, or if that point is older than
    `stale_days`. Causal by construction."""
    days = np.asarray(days, dtype=np.int64)
    out = np.full(len(days), np.nan)
    events = []
    for t, v, k, s in _versions(name):
        if k >= s:
            continue                      # superseded before it was known: load() never sees it
        events.append((k, 1, t, v))
        if s != math.inf:
            events.append((s, 0, t, v))
    events.sort(key=lambda e: (e[0], e[1]))
    valid, heap, i = {}, [], 0
    for j, d in enumerate(days):
        cutoff = (int(d) + 1) * DAY
        while i < len(events) and events[i][0] <= cutoff:
            when, add, t, v = events[i]
            if add:
                valid[t] = v
                heapq.heappush(heap, -t)
            elif valid.get(t) == v:
                del valid[t]
            i += 1
        while heap and -heap[0] not in valid:
            heapq.heappop(heap)
        if heap:
            t = -heap[0]
            if t // DAY >= int(d) - stale_days:
                out[j] = valid[t]
    return out
