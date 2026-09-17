"""Decider-health Guardian — a safety layer INDEPENDENT of the learning stack,
inspired by NOFX's "model proposes, runtime disposes" posture.

Responsibilities (all deterministic, none learned):

  * SAFE MODE — a circuit breaker. When the decision loop repeatedly fails, or
    the market data feed is unhealthy / stale, new ENTRIES are blocked while
    open positions keep being managed (stops/exits still run). It clears itself
    automatically once health is restored for a short dwell.

  * ENTRY-RATE THROTTLE — a per-hour cap on NEW opens so a misbehaving signal
    stack can't churn the book (NOFX added this after a live over-trading
    incident: dozens of fills, sub-hour average hold).

  * LAUNCH PREFLIGHT — one gate that verifies the obvious preconditions before
    the loop is trusted to open anything (feed healthy + enough history, model
    input dimension matches the feature builder, risk not already killed). It
    turns a class of silent "nothing works" states into one logged reason.

This object holds no money and makes no predictions; it only says YES/NO to new
risk and explains why.
"""
import time
from collections import deque


class Guardian:
    def __init__(self):
        # safe mode
        self.consecutive_failures = 0
        self.safe_mode = False
        self.safe_reason = ""
        self.entered_safe_ts = None
        self._healthy_since = None
        # entry-rate throttle
        self._entries = deque()          # timestamps of recent opens
        # preflight
        self.preflight_ok = None
        self.preflight_report = {}

    # ---------------- tunable accessors ----------------
    @staticmethod
    def _fail_threshold():
        from .tunables import tv
        return int(tv("safe_mode_fail_threshold"))

    @staticmethod
    def _recover_sec():
        from .tunables import tv
        return float(tv("safe_mode_recover_sec"))

    @staticmethod
    def _max_entries_hr():
        from .tunables import tv
        return int(tv("max_entries_per_hour"))

    @staticmethod
    def _data_stale_sec():
        from .tunables import tv
        return float(tv("data_stale_sec"))

    # ---------------- cycle health ----------------
    def on_cycle(self, ok, market=None, error=""):
        """Call once per decision tick with whether it succeeded. Updates the
        safe-mode circuit breaker from BOTH loop failures and data health."""
        now = time.time()
        if ok:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # gather the reasons NEW entries should be suspended
        reasons = []
        if self.consecutive_failures >= self._fail_threshold():
            reasons.append(f"{self.consecutive_failures} consecutive tick failures")
        if error and self.consecutive_failures >= self._fail_threshold():
            reasons.append(f"last error: {str(error)[:80]}")
        if market is not None:
            if not getattr(market, "healthy", False):
                reasons.append("market data feed unhealthy")
            else:
                last = getattr(market, "last_update", 0) or 0
                if last and now - last > self._data_stale_sec():
                    reasons.append(f"market data stale ({int(now - last)}s)")

        if reasons:
            self._healthy_since = None
            if not self.safe_mode:
                self.entered_safe_ts = now
                from . import db
                db.log_event("risk", "🟡 SAFE MODE engaged — new entries "
                                     f"suspended: {'; '.join(reasons)}")
                try:
                    from .alerts import alert
                    alert("warning", "Safe mode engaged",
                          "New entries suspended (open risk still managed): "
                          + "; ".join(reasons))
                except Exception:
                    pass
            self.safe_mode = True
            self.safe_reason = "; ".join(reasons)
        else:
            # healthy tick — require a dwell before clearing so we don't flap
            if self._healthy_since is None:
                self._healthy_since = now
            if self.safe_mode and now - self._healthy_since >= self._recover_sec():
                self.safe_mode = False
                self.safe_reason = ""
                self.entered_safe_ts = None
                from . import db
                db.log_event("risk", "🟢 SAFE MODE cleared — health restored, "
                                     "new entries allowed again")
                try:
                    from .alerts import alert
                    alert("info", "Safe mode cleared",
                          "Decision loop healthy again; new entries resumed.")
                except Exception:
                    pass

    # ---------------- entry-rate throttle ----------------
    def note_entry(self):
        self._entries.append(time.time())
        self._trim()

    def _trim(self):
        cut = time.time() - 3600
        while self._entries and self._entries[0] < cut:
            self._entries.popleft()

    def entries_last_hour(self):
        self._trim()
        return len(self._entries)

    def can_enter(self):
        """Deterministic YES/NO on opening ANY new position this tick, with a
        reason. Independent of (and checked before) per-product risk gates."""
        if self.safe_mode:
            return False, f"safe mode ({self.safe_reason})"
        cap = self._max_entries_hr()
        if cap > 0 and self.entries_last_hour() >= cap:
            return False, f"entry-rate cap ({cap}/hour) reached"
        return True, "ok"

    # ---------------- launch preflight ----------------
    def preflight(self):
        """Verify start-up preconditions. Returns a report dict; also sets
        `preflight_ok`. Non-fatal issues (feeds still warming) are reported but
        don't hard-fail — they simply keep safe mode engaged until healthy."""
        from .data.market import market
        from .config import PRODUCTS
        from .learn.online_model import N_IN, build_x, FEAT_NAMES
        from .risk.manager import risk

        checks = []

        def add(name, ok, detail):
            checks.append({"check": name, "ok": bool(ok), "detail": detail})

        add("market_feed_healthy", getattr(market, "healthy", False),
            "Coinbase market feed reporting healthy" if getattr(market, "healthy", False)
            else "feed not healthy yet")

        btc_bars = len(market.candles.get("BTC-USD", []))
        add("history_sufficient", btc_bars >= 60,
            f"BTC-USD has {btc_bars} candles (need ≥60 for features)")

        # model input dimension matches the live feature builder
        dim_ok, dim_detail = True, f"feature builder emits {N_IN} inputs"
        try:
            f = market.features("BTC-USD")
            if f:
                x = build_x(f, 0.0, 0.0, None)
                dim_ok = (len(x) == N_IN == len(FEAT_NAMES))
                dim_detail = (f"build_x emits {len(x)}, N_IN={N_IN}, "
                              f"FEAT_NAMES={len(FEAT_NAMES)}")
        except Exception as e:
            dim_ok, dim_detail = False, f"build_x failed: {e}"
        add("model_input_dim", dim_ok, dim_detail)

        add("risk_not_killed", not risk.killed,
            "kill switch active" if risk.killed else "kill switch clear")

        add("universe_nonempty", len(PRODUCTS) > 0,
            f"{len(PRODUCTS)} products in the trading universe")

        ok = all(c["ok"] for c in checks)
        self.preflight_ok = ok
        self.preflight_report = {"ok": ok, "ts": time.time(), "checks": checks}
        from . import db
        if ok:
            db.log_event("system", "✅ Launch preflight passed — decider cleared "
                                    "to open positions")
        else:
            failed = [c["check"] for c in checks if not c["ok"]]
            db.log_event("system", "⏳ Launch preflight incomplete: "
                                   f"{', '.join(failed)} — new entries held "
                                   "until healthy")
        return self.preflight_report

    # ---------------- reporting ----------------
    def snapshot(self):
        return {
            "safe_mode": self.safe_mode,
            "safe_reason": self.safe_reason,
            "consecutive_failures": self.consecutive_failures,
            "entries_last_hour": self.entries_last_hour(),
            "max_entries_per_hour": self._max_entries_hr(),
            "preflight_ok": self.preflight_ok,
        }


guardian = Guardian()
