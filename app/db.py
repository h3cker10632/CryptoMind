"""SQLite persistence: audit log, trades, equity curve, scored signals."""
import sqlite3, json, time, threading
from .config import DB_PATH

_lock = threading.Lock()

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS events(
            ts REAL, kind TEXT, message TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS trades(
            ts REAL, product TEXT, side TEXT, qty REAL, price REAL,
            fee REAL, reason TEXT, pnl REAL);
        CREATE TABLE IF NOT EXISTS equity(
            ts REAL, equity REAL, cash REAL, exposure REAL);
        CREATE TABLE IF NOT EXISTS signal_scores(
            ts REAL, strategy TEXT, product TEXT, direction REAL,
            confidence REAL, fwd_return REAL, scored INTEGER DEFAULT 0,
            regime TEXT);
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts)")
        cols = {r[1] for r in c.execute("PRAGMA table_info(signal_scores)")}
        if "regime" not in cols:
            c.execute("ALTER TABLE signal_scores ADD COLUMN regime TEXT")

def log_event(kind, message, data=None):
    with _lock, _conn() as c:
        c.execute("INSERT INTO events VALUES(?,?,?,?)",
                  (time.time(), kind, message, json.dumps(data or {})))

def log_trade(product, side, qty, price, fee, reason, pnl=0.0):
    with _lock, _conn() as c:
        c.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?)",
                  (time.time(), product, side, qty, price, fee, reason, pnl))

def log_equity(equity, cash, exposure):
    with _lock, _conn() as c:
        c.execute("INSERT INTO equity VALUES(?,?,?,?)",
                  (time.time(), equity, cash, exposure))

def record_signal(strategy, product, direction, confidence, regime=None):
    with _lock, _conn() as c:
        c.execute("INSERT INTO signal_scores(ts,strategy,product,direction,confidence,fwd_return,scored,regime) "
                  "VALUES(?,?,?,?,?,NULL,0,?)",
                  (time.time(), strategy, product, direction, confidence, regime))

def unscored_signals(older_than_ts):
    with _lock, _conn() as c:
        rows = c.execute("SELECT rowid,* FROM signal_scores WHERE scored=0 AND ts<=?",
                         (older_than_ts,)).fetchall()
        return [dict(r) for r in rows]

def score_signal(rowid, fwd_return):
    with _lock, _conn() as c:
        c.execute("UPDATE signal_scores SET fwd_return=?, scored=1 WHERE rowid=?",
                  (fwd_return, rowid))

def abandon_signal(rowid):
    """Mark scored without a return — lookup never resolved. Does not train."""
    with _lock, _conn() as c:
        c.execute("UPDATE signal_scores SET scored=1 WHERE rowid=?", (rowid,))

def strategy_scores(lookback):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT strategy, direction, confidence, fwd_return FROM signal_scores "
            "WHERE scored=1 AND fwd_return IS NOT NULL ORDER BY ts DESC LIMIT ?",
            (lookback * 8,)).fetchall()
        return [dict(r) for r in rows]

def recent(table, limit=100):
    with _lock, _conn() as c:
        rows = c.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def equity_since(since_ts, max_points=800):
    """Equity samples at or after since_ts, decimated to max_points (keeps first+last).

    If the window has fewer than 2 samples (typical for 20s — one tick), prepend
    the last sample before the window so the chart still has a line.
    """
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT ts, equity FROM equity WHERE ts>=? ORDER BY ts ASC LIMIT 200000",
            (since_ts,)).fetchall()
        if len(rows) < 2:
            prev = c.execute(
                "SELECT ts, equity FROM equity WHERE ts<? ORDER BY ts DESC LIMIT 1",
                (since_ts,)).fetchone()
            if prev:
                rows = [prev] + list(rows)
    n = len(rows)
    if n == 0:
        return []
    if n <= max_points:
        return [{"ts": r["ts"], "equity": r["equity"]} for r in rows]
    out = []
    last_idx = -1
    for i in range(max_points):
        idx = int(i * (n - 1) / (max_points - 1))
        if idx == last_idx:
            continue
        r = rows[idx]
        out.append({"ts": r["ts"], "equity": r["equity"]})
        last_idx = idx
    return out
