"""SQLite persistence: audit log, trades, equity curve, scored signals, and
an append-only order/fill event log for reconciliation.

Hardening vs the original:
  * WAL journal mode + busy_timeout  → concurrent readers don't block writers
  * a background writer thread        → write calls never block the event loop
  * retention/rollup on the equity    → the table can't grow without bound
  * order_events table                → immutable audit trail for the OMS
"""
import math, sqlite3, json, time, threading, queue, atexit
from .config import DB_PATH

_lock = threading.Lock()
_write_q: "queue.Queue" = queue.Queue()
_writer_started = False
EQUITY_RETENTION_SEC = 90 * 86400          # keep 90 days of raw equity samples


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=30000")
    return c


def init():
    with _lock, _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
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
        CREATE TABLE IF NOT EXISTS order_events(
            ts REAL, intent_id TEXT, client_order_id TEXT, product TEXT,
            side TEXT, event TEXT, qty REAL, price REAL, status TEXT,
            venue TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS decisions(
            ts REAL, cycle INTEGER, product TEXT, direction REAL,
            composite REAL, confidence REAL, ml_confidence REAL,
            action TEXT, reason TEXT, size_pre REAL, size_post REAL,
            regime TEXT, votes TEXT);
        CREATE TABLE IF NOT EXISTS pm_forecasts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT NOT NULL,
            ttl_bucket TEXT NOT NULL,
            forecast_ts REAL NOT NULL,
            predicted_p0 REAL NOT NULL,
            market_p0 REAL NOT NULL,
            votes TEXT NOT NULL,
            features TEXT NOT NULL,
            evidence TEXT NOT NULL,
            resolved_outcome0 INTEGER,
            resolved_ts REAL,
            resolution_due_ts REAL,
            resolution_checked_ts REAL,
            resolution_next_poll_ts REAL NOT NULL DEFAULT 0,
            resolution_attempts INTEGER NOT NULL DEFAULT 0,
            UNIQUE(condition_id, ttl_bucket));
        CREATE TABLE IF NOT EXISTS paper_portfolio(
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL,
            opening_cash REAL NOT NULL,
            migration_id TEXT NOT NULL,
            created_ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS paper_capital_events(
            event_id TEXT PRIMARY KEY,
            sleeve TEXT NOT NULL,
            amount REAL NOT NULL,
            reference TEXT,
            ts REAL NOT NULL);
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_orderev_intent ON order_events(intent_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)")
        pm_cols = {r[1] for r in c.execute("PRAGMA table_info(pm_forecasts)")}
        for name, declaration in (
            ("resolution_due_ts", "REAL"),
            ("resolution_checked_ts", "REAL"),
            ("resolution_next_poll_ts", "REAL NOT NULL DEFAULT 0"),
            ("resolution_attempts", "INTEGER NOT NULL DEFAULT 0")):
            if name not in pm_cols:
                c.execute(f"ALTER TABLE pm_forecasts ADD COLUMN {name} {declaration}")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pm_forecasts_pollable "
              "ON pm_forecasts(resolved_outcome0, resolution_next_poll_ts, "
              "resolution_due_ts, forecast_ts)")
        cols = {r[1] for r in c.execute("PRAGMA table_info(signal_scores)")}
        if "regime" not in cols:
            c.execute("ALTER TABLE signal_scores ADD COLUMN regime TEXT")
    _start_writer()


# ---------------- background writer ----------------

def _writer_loop():
    while True:
        item = _write_q.get()
        if item is None:
            _write_q.task_done()
            return
        sql, params = item
        for attempt in range(3):
            try:
                with _lock, _conn() as c:
                    c.execute(sql, params)
                break
            except sqlite3.OperationalError:
                time.sleep(0.1 * (attempt + 1))
        _write_q.task_done()


def _start_writer():
    global _writer_started
    if _writer_started:
        return
    _writer_started = True
    t = threading.Thread(target=_writer_loop, daemon=True, name="db-writer")
    t.start()
    atexit.register(flush)


def _enqueue(sql, params):
    if not _writer_started:
        # fallback: synchronous (e.g. before init)
        with _lock, _conn() as c:
            c.execute(sql, params)
        return
    _write_q.put((sql, params))


def flush(timeout=5.0):
    """Block until queued writes drain (used on shutdown)."""
    try:
        _write_q.join()
    except Exception:
        pass


# ---------------- writes (async via queue) ----------------

def log_event(kind, message, data=None):
    _enqueue("INSERT INTO events VALUES(?,?,?,?)",
             (time.time(), kind, message, json.dumps(data or {})))


def log_trade(product, side, qty, price, fee, reason, pnl=0.0):
    _enqueue("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?)",
             (time.time(), product, side, qty, price, fee, reason, pnl))


def log_equity(equity, cash, exposure):
    _enqueue("INSERT INTO equity VALUES(?,?,?,?)",
             (time.time(), equity, cash, exposure))


def log_order_event(intent_id, client_order_id, product, side, event,
                    qty=0.0, price=0.0, status="", venue="paper", data=None):
    """Append-only OMS audit trail — every intent/submit/ack/fill/cancel."""
    _enqueue("INSERT INTO order_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
             (time.time(), intent_id, client_order_id, product, side, event,
              qty, price, status, venue, json.dumps(data or {})))


def log_decision(cycle, product, direction, composite, confidence, ml_confidence,
                 action, reason, size_pre=0.0, size_post=0.0, regime=None,
                 votes=None):
    """Append one CYCLE-LEVEL decision record — the "no position without a
    paper trail" audit row (NOFX idea). Captures, per considered candidate:
    the composite signal, confidence, the chosen action (enter/skip/exit/...),
    a human reason (e.g. the gate that blocked it), and the notional BEFORE and
    AFTER the risk cage clamped it. This is what lets the operator answer
    "why did nothing trade / why was it sized so small" from history alone."""
    _enqueue("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
             (time.time(), cycle, product, direction, composite, confidence,
              ml_confidence, action, reason, size_pre, size_post, regime,
              json.dumps(votes or {})))


def recent_decisions(limit=100, product=None, action=None):
    """Most-recent decision-audit rows (optionally filtered by product/action)."""
    q = "SELECT * FROM decisions"
    where, params = [], []
    if product:
        where.append("product=?"); params.append(product)
    if action:
        where.append("action=?"); params.append(action)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    with _lock, _conn() as c:
        rows = c.execute(q, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["votes"] = json.loads(d.get("votes") or "{}")
            except Exception:
                d["votes"] = {}
            out.append(d)
        return out


def prune_decisions(retention_sec=14 * 86400):
    """Keep ~2 weeks of decision audit rows (they accrue every cycle)."""
    cutoff = time.time() - retention_sec
    with _lock, _conn() as c:
        c.execute("DELETE FROM decisions WHERE ts < ?", (cutoff,))


def record_signal(strategy, product, direction, confidence, regime=None):
    _enqueue("INSERT INTO signal_scores(ts,strategy,product,direction,confidence,fwd_return,scored,regime) "
             "VALUES(?,?,?,?,?,NULL,0,?)",
             (time.time(), strategy, product, direction, confidence, regime))


# ---------------- writes that must be synchronous (return rows) ----------------

def unscored_signals(older_than_ts):
    with _lock, _conn() as c:
        rows = c.execute("SELECT rowid,* FROM signal_scores WHERE scored=0 AND ts<=?",
                         (older_than_ts,)).fetchall()
        return [dict(r) for r in rows]


def score_signal(rowid, fwd_return):
    _enqueue("UPDATE signal_scores SET fwd_return=?, scored=1 WHERE rowid=?",
             (fwd_return, rowid))


def abandon_signal(rowid):
    _enqueue("UPDATE signal_scores SET scored=1 WHERE rowid=?", (rowid,))


def record_pm_forecast(forecast):
    """Persist one point-in-time PM forecast, returning false for a duplicate key."""
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO pm_forecasts "
            "(condition_id, ttl_bucket, forecast_ts, predicted_p0, market_p0, "
            "votes, features, evidence, resolution_due_ts) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (str(forecast["condition_id"]), str(forecast["ttl_bucket"]),
             float(forecast["forecast_ts"]), float(forecast["predicted_p0"]),
             float(forecast["market_p0"]), json.dumps(forecast.get("votes") or {}),
             json.dumps(forecast.get("features") or {}),
             json.dumps(forecast.get("evidence") or []),
             forecast.get("resolution_due_ts")))
        return c.execute("SELECT changes()").fetchone()[0] == 1


def _pm_forecast_row(row):
    out = dict(row)
    for key, default in (("votes", {}), ("features", {}), ("evidence", [])):
        try:
            out[key] = json.loads(out.get(key) or "")
        except (TypeError, json.JSONDecodeError):
            out[key] = default
    return out


def pending_pm_forecasts(limit=200, now=None):
    """Read a small batch of due, retry-eligible unresolved forecasts."""
    now = time.time() if now is None else float(now)
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM pm_forecasts WHERE resolved_outcome0 IS NULL "
            "AND resolution_next_poll_ts<=? "
            "AND (resolution_due_ts IS NULL OR resolution_due_ts<=?) "
            "ORDER BY resolution_due_ts IS NULL ASC, resolution_due_ts ASC, "
            "forecast_ts ASC LIMIT ?",
            (now, now + 6 * 3600, max(1, min(200, int(limit))))).fetchall()
        return [_pm_forecast_row(row) for row in rows]


def mark_pm_resolution_attempt(forecast_id, checked_ts=None):
    """Persist one resolution poll before network I/O and return its attempt number."""
    checked_ts = time.time() if checked_ts is None else float(checked_ts)
    with _lock, _conn() as c:
        c.execute(
            "UPDATE pm_forecasts SET resolution_attempts=resolution_attempts+1, "
            "resolution_checked_ts=?, resolution_next_poll_ts=? + "
            "CASE WHEN resolution_attempts>=4 THEN 3600 "
            "WHEN resolution_attempts=3 THEN 960 "
            "WHEN resolution_attempts=2 THEN 480 "
            "WHEN resolution_attempts=1 THEN 240 ELSE 120 END "
            "WHERE id=? AND resolved_outcome0 IS NULL",
            (checked_ts, checked_ts, int(forecast_id)))
        row = c.execute(
            "SELECT resolution_attempts FROM pm_forecasts WHERE id=?",
            (int(forecast_id),)).fetchone()
        return int(row[0]) if row else 0


def resolved_pm_forecasts(after_id=0, limit=1000):
    """Read a labeled page for restart-safe, idempotent learner replay."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM pm_forecasts WHERE resolved_outcome0 IS NOT NULL AND id>? "
            "ORDER BY id ASC LIMIT ?",
            (int(after_id), max(1, int(limit)))).fetchall()
        return [_pm_forecast_row(row) for row in rows]


def label_pm_forecast(forecast_id, outcome0_won):
    """Label a forecast once; repeated labels return false without a mutation."""
    outcome = int(outcome0_won)
    if outcome not in (0, 1):
        raise ValueError("outcome0_won must be 0 or 1")
    with _lock, _conn() as c:
        cursor = c.execute(
            "UPDATE pm_forecasts SET resolved_outcome0=?, resolved_ts=? "
            "WHERE id=? AND resolved_outcome0 IS NULL",
            (outcome, time.time(), int(forecast_id)))
        return cursor.rowcount == 1


def pm_forecast_rows(limit=10000):
    """Read the newest bounded forecast set for reporting and audit."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM pm_forecasts ORDER BY forecast_ts DESC LIMIT ?",
            (max(1, int(limit)),)).fetchall()
        return [_pm_forecast_row(row) for row in rows]


def pm_forecast_count():
    """Count the full forecast ledger without materializing rows."""
    with _lock, _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM pm_forecasts").fetchone()
        return int(row[0]) if row else 0


# ---------------- shared paper-capital ledger ----------------

def initialize_paper_portfolio(opening_cash, migration_id):
    """Create the singleton paper-portfolio account once; any later call is a
    no-op (returns False) so a repeated startup/migration can never re-fund
    the account or disturb its current cash."""
    opening_cash = float(opening_cash)
    with _lock, _conn() as c:
        if c.execute("SELECT 1 FROM paper_portfolio WHERE id=1").fetchone():
            return False
        c.execute(
            "INSERT INTO paper_portfolio(id, cash, opening_cash, migration_id, "
            "created_ts) VALUES(1,?,?,?,?)",
            (opening_cash, opening_cash, str(migration_id), time.time()))
        return True


def paper_portfolio_account():
    """Read the singleton paper-capital account, or None before migration."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT cash, opening_cash, migration_id, created_ts "
            "FROM paper_portfolio WHERE id=1").fetchone()
        return dict(row) if row else None


def reserve_paper_cash(event_id, sleeve, amount, reference):
    """Atomically debit `amount` (> 0) from the shared account and record a
    unique capital event in the same transaction. Returns False without any
    mutation when the account is missing, the amount is non-finite/non-
    positive, cash is insufficient, or `event_id` was already recorded."""
    amount = float(amount)
    if not math.isfinite(amount) or amount <= 0:
        return False
    with _lock, _conn() as c:
        if c.execute("SELECT 1 FROM paper_capital_events WHERE event_id=?",
                     (event_id,)).fetchone():
            return False
        row = c.execute("SELECT cash FROM paper_portfolio WHERE id=1").fetchone()
        if row is None or row["cash"] + 1e-9 < amount:
            return False
        c.execute("UPDATE paper_portfolio SET cash=cash-? WHERE id=1", (amount,))
        c.execute(
            "INSERT INTO paper_capital_events(event_id, sleeve, amount, "
            "reference, ts) VALUES(?,?,?,?,?)",
            (str(event_id), str(sleeve), -amount, reference, time.time()))
        return True


def apply_paper_cash_event(event_id, sleeve, delta, reference):
    """Apply one signed settlement/fee/funding delta idempotently. Returns
    False without mutation when the account is missing, `delta` is non-finite,
    or `event_id` was already recorded."""
    delta = float(delta)
    if not math.isfinite(delta):
        return False
    with _lock, _conn() as c:
        if c.execute("SELECT 1 FROM paper_capital_events WHERE event_id=?",
                     (event_id,)).fetchone():
            return False
        if not c.execute("SELECT 1 FROM paper_portfolio WHERE id=1").fetchone():
            return False
        c.execute("UPDATE paper_portfolio SET cash=cash+? WHERE id=1", (delta,))
        c.execute(
            "INSERT INTO paper_capital_events(event_id, sleeve, amount, "
            "reference, ts) VALUES(?,?,?,?,?)",
            (str(event_id), str(sleeve), delta, reference, time.time()))
        return True


def reset_paper_portfolio_for_tests():
    """Test-only: clear the shared ledger so an isolated test run can recreate
    the singleton account without a fresh process/DB file."""
    with _lock, _conn() as c:
        c.execute("DELETE FROM paper_portfolio")
        c.execute("DELETE FROM paper_capital_events")


def reset_paper_portfolio_cash(opening_cash, event_id, reference="account reset"):
    """Explicit whole-account reset: set cash to an absolute value, once per
    `event_id`. The caller must confirm every paper sleeve is flat first --
    this only touches the shared cash balance, never positions."""
    opening_cash = float(opening_cash)
    if not math.isfinite(opening_cash) or opening_cash < 0:
        return False
    with _lock, _conn() as c:
        if c.execute("SELECT 1 FROM paper_capital_events WHERE event_id=?",
                     (event_id,)).fetchone():
            return False
        if not c.execute("SELECT 1 FROM paper_portfolio WHERE id=1").fetchone():
            return False
        c.execute("UPDATE paper_portfolio SET cash=? WHERE id=1", (opening_cash,))
        c.execute(
            "INSERT INTO paper_capital_events(event_id, sleeve, amount, "
            "reference, ts) VALUES(?,?,?,?,?)",
            (str(event_id), "account", opening_cash, reference, time.time()))
        return True


def strategy_scores(lookback):
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT strategy, direction, confidence, fwd_return FROM signal_scores "
            "WHERE scored=1 AND fwd_return IS NOT NULL ORDER BY ts DESC LIMIT ?",
            (lookback * 8,)).fetchall()
        return [dict(r) for r in rows]


def labeled_signals_count():
    """Cheap COUNT of supervised-ready rows (scored signals with a realized
    forward-return label). The auto-trainer's data-driven trigger uses this to
    decide whether enough NEW labels have matured, without materializing the
    whole dataset."""
    with _lock, _conn() as c:
        r = c.execute(
            "SELECT COUNT(*) AS n FROM signal_scores "
            "WHERE scored=1 AND fwd_return IS NOT NULL").fetchone()
        return int(r["n"]) if r else 0


def labeled_signals(limit=100000):
    """Scored signals that HAVE a realized forward-return label — the only rows
    fit for supervised learning. Unlike strategy_scores() this keeps ts/product/
    regime so an offline ML pipeline can join them to decision context and align
    them in time. Ordered oldest->newest for chronological (walk-forward) use.

    See labeled_signals_count() for a cheap COUNT used by the auto-trainer."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT ts, strategy, product, direction, confidence, fwd_return, "
            "regime FROM signal_scores WHERE scored=1 AND fwd_return IS NOT NULL "
            "ORDER BY ts ASC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def all_decisions(limit=200000, since_ts=None):
    """Full decision-audit rows (oldest->newest) for offline joining. Parses the
    votes JSON. Optionally bounded by a start timestamp."""
    q = "SELECT * FROM decisions"
    params = []
    if since_ts is not None:
        q += " WHERE ts >= ?"
        params.append(since_ts)
    q += " ORDER BY ts ASC LIMIT ?"
    params.append(limit)
    with _lock, _conn() as c:
        rows = c.execute(q, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["votes"] = json.loads(d.get("votes") or "{}")
            except Exception:
                d["votes"] = {}
            out.append(d)
        return out


def recent(table, limit=100):
    with _lock, _conn() as c:
        rows = c.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def learning_counts():
    """Durable-memory row counts used by the /brains report: how much history
    the learning stack has actually accumulated in SQLite."""
    out = {}
    with _lock, _conn() as c:
        for t in ("events", "trades", "equity", "signal_scores", "order_events",
                  "decisions"):
            try:
                out[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                out[t] = 0
        try:
            out["signals_scored"] = c.execute(
                "SELECT COUNT(*) FROM signal_scores WHERE scored=1").fetchone()[0]
            out["signals_pending"] = c.execute(
                "SELECT COUNT(*) FROM signal_scores WHERE scored=0").fetchone()[0]
        except Exception:
            out["signals_scored"] = out["signals_pending"] = 0
    return out


def order_events(intent_id=None, limit=200):
    with _lock, _conn() as c:
        if intent_id:
            rows = c.execute("SELECT * FROM order_events WHERE intent_id=? ORDER BY ts ASC",
                             (intent_id,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM order_events ORDER BY ts DESC LIMIT ?",
                             (limit,)).fetchall()
        return [dict(r) for r in rows]


def prune_signals(retention_sec=60 * 86400):
    """Drop SCORED/abandoned strategy signals older than the window. They were
    logged every 20s tick before learning v2 (1.35M rows); the learner only
    ever reads recent ones. Unscored rows are kept until they're scored."""
    cutoff = time.time() - retention_sec
    with _lock, _conn() as c:
        c.execute("DELETE FROM signal_scores WHERE scored=1 AND ts < ?", (cutoff,))


def prune_events(retention_sec=90 * 86400):
    """Keep ~3 months of the event log."""
    cutoff = time.time() - retention_sec
    with _lock, _conn() as c:
        c.execute("DELETE FROM events WHERE ts < ?", (cutoff,))


def prune_equity(retention_sec=EQUITY_RETENTION_SEC):
    """Delete equity samples older than the retention window (called rarely)."""
    cutoff = time.time() - retention_sec
    with _lock, _conn() as c:
        c.execute("DELETE FROM equity WHERE ts < ?", (cutoff,))


def equity_since(since_ts, max_points=800):
    """Equity samples at/after since_ts, decimated to max_points (keeps first+last)."""
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
