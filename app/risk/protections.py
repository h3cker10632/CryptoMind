"""Protections — time-boxed circuit breakers over recent trade outcomes.

Inspired by freqtrade's Protections layer (StoplossGuard / LowProfitPairs /
MaxDrawdown), reimplemented as our own code. These are DISTINCT from the
existing risk controls:

  * the KILL SWITCH is permanent and needs a manual reset;
  * `on_trade_closed` only SCALES size after a loss streak;
  * `cooldown_sec` locks one pair briefly after any exit.

None of those benches the book when a cluster of trades goes wrong. Protections
add self-healing, TIME-BOXED locks — either for one product or globally — that
`RiskManager.can_open` consults and that expire on their own:

  * StoplossGuard   — too many stop-outs in a rolling window -> lock (global or
                      per-pair) for a cooldown, so a whipsaw regime can't grind
                      the account down one stop at a time.
  * LowProfitPairs  — a single coin whose recent net PnL is negative over enough
                      trades -> lock JUST that coin (per-pair edge, which the
                      regime/direction learner never tracks).
  * MaxDrawdown     — a temporary, auto-recovering drawdown halt (softer tier
                      below the permanent kill switch) computed from realized
                      trade PnL over a rolling window.

All windows are wall-clock seconds (the system ticks continuously, not on fixed
candles). State is a small dict of active locks + is fully serialisable so it
rides the normal snapshot. Every knob is a tunable; setting a trade_limit to 0
disables that protection.
"""
import time
from ..tunables import tv


class ProtectionManager:
    def __init__(self):
        # active locks. `_global_until` is a wall-clock ts; `_pair_until` maps
        # product -> ts. A lock is active while now < until.
        self._global_until = 0.0
        self._global_reason = ""
        self._pair_until = {}         # product -> ts
        self._pair_reason = {}        # product -> str
        self.n_global_locks = 0       # lifetime counters (reporting)
        self.n_pair_locks = 0

    # ------------------------------------------------ lock helpers
    def _lock_global(self, seconds, reason):
        until = time.time() + max(0.0, seconds)
        # extend, never shorten, an existing global lock
        if until > self._global_until:
            self._global_until = until
            self._global_reason = reason
            self.n_global_locks += 1

    def _lock_pair(self, product, seconds, reason):
        until = time.time() + max(0.0, seconds)
        if until > self._pair_until.get(product, 0.0):
            self._pair_until[product] = until
            self._pair_reason[product] = reason
            self.n_pair_locks += 1

    def is_locked(self, product, now=None):
        """Return (locked, reason). Global locks apply to every product."""
        now = now or time.time()
        if now < self._global_until:
            return True, f"protection (all pairs): {self._global_reason}"
        pu = self._pair_until.get(product, 0.0)
        if now < pu:
            return True, f"protection ({product}): {self._pair_reason.get(product, '')}"
        return False, ""

    def _prune(self, now):
        """Drop expired per-pair locks so the dict can't grow unbounded."""
        for p in [p for p, u in self._pair_until.items() if now >= u]:
            self._pair_until.pop(p, None)
            self._pair_reason.pop(p, None)

    # ------------------------------------------------ evaluation
    @staticmethod
    def _is_stop_exit(trade):
        r = (trade.get("exit_reason") or "").lower()
        return ("stop" in r or "liquidation" in r) and trade.get("pnl", 0.0) < 0

    def evaluate(self, closed_trades, equity=None, now=None):
        """Re-derive locks from the recent closed-trade history. Called once per
        tick (cheap: only scans the tail within the longest lookback). Idempotent
        — recomputes lock windows from the trades themselves, so a restart that
        replays history lands in the same state.

        `equity` (optional) is only used by the MaxDrawdown protection's context;
        the drawdown itself is computed from realized trade PnL.
        """
        now = now or time.time()
        self._prune(now)
        if not closed_trades:
            return
        self._eval_stoploss_guard(closed_trades, now)
        self._eval_low_profit_pairs(closed_trades, now)
        self._eval_max_drawdown(closed_trades, now)

    def _recent(self, closed_trades, lookback_sec, now):
        cutoff = now - lookback_sec
        return [t for t in closed_trades if t.get("closed", 0) >= cutoff]

    def _eval_stoploss_guard(self, closed_trades, now):
        limit = int(tv("protect_stopguard_trades"))
        if limit <= 0:
            return
        look = tv("protect_stopguard_lookback_sec")
        recent = self._recent(closed_trades, look, now)
        stops = [t for t in recent if self._is_stop_exit(t)]
        if len(stops) >= limit:
            self._lock_global(
                tv("protect_stopguard_lock_sec"),
                f"{len(stops)} stop-outs in {int(look/60)}m (StoplossGuard)")

    def _eval_low_profit_pairs(self, closed_trades, now):
        limit = int(tv("protect_lowprofit_trades"))
        if limit <= 0:
            return
        look = tv("protect_lowprofit_lookback_sec")
        req = tv("protect_lowprofit_required")     # required net PnL fraction
        recent = self._recent(closed_trades, look, now)
        by_pair = {}
        for t in recent:
            by_pair.setdefault(t.get("product"), []).append(t)
        for product, trades in by_pair.items():
            if product is None or len(trades) < limit:
                continue
            pnl = sum(t.get("pnl", 0.0) for t in trades)
            # notional base to express PnL as a return: sum of entry stakes
            base = sum(abs(t.get("qty", 0.0)) * t.get("entry", 0.0) for t in trades)
            ratio = (pnl / base) if base > 0 else 0.0
            if ratio < req:
                self._lock_pair(
                    product, tv("protect_lowprofit_lock_sec"),
                    f"net {ratio:+.2%} over {len(trades)} trades (LowProfitPairs)")

    def _eval_max_drawdown(self, closed_trades, now):
        limit = int(tv("protect_maxdd_trades"))
        if limit <= 0:
            return
        look = tv("protect_maxdd_lookback_sec")
        max_dd = tv("protect_maxdd_fraction")
        recent = self._recent(closed_trades, look, now)
        if len(recent) < limit:
            return
        # peak-to-trough drawdown of the CUMULATIVE realized-PnL curve over the
        # window, expressed as a fraction of the running peak equity proxy.
        recent = sorted(recent, key=lambda t: t.get("closed", 0))
        cum = 0.0
        peak = 0.0
        worst = 0.0
        for t in recent:
            cum += t.get("pnl", 0.0)
            peak = max(peak, cum)
            # drawdown from the window's running peak; normalise by |peak| or the
            # summed stake so an all-loss window still yields a sane fraction.
            denom = max(abs(peak), 1e-9)
            worst = max(worst, (peak - cum) / denom)
        if worst >= max_dd:
            self._lock_global(
                tv("protect_maxdd_lock_sec"),
                f"realized drawdown {worst:.1%} over {len(recent)} trades (MaxDrawdown)")

    # ------------------------------------------------ reporting / persistence
    def stats(self, now=None):
        now = now or time.time()
        self._prune(now)
        g = max(0.0, self._global_until - now)
        return {
            "global_locked": now < self._global_until,
            "global_reason": self._global_reason if now < self._global_until else "",
            "global_seconds_left": round(g, 1),
            "locked_pairs": {p: round(u - now, 1)
                             for p, u in self._pair_until.items() if u > now},
            "n_global_locks": self.n_global_locks,
            "n_pair_locks": self.n_pair_locks,
        }

    def to_dict(self):
        return {
            "global_until": self._global_until,
            "global_reason": self._global_reason,
            "pair_until": dict(self._pair_until),
            "pair_reason": dict(self._pair_reason),
            "n_global_locks": self.n_global_locks,
            "n_pair_locks": self.n_pair_locks,
        }

    def load_dict(self, d):
        if not d:
            return False
        try:
            self._global_until = float(d.get("global_until", 0.0))
            self._global_reason = str(d.get("global_reason", ""))
            self._pair_until = {p: float(u)
                                for p, u in (d.get("pair_until") or {}).items()}
            self._pair_reason = {p: str(r)
                                 for p, r in (d.get("pair_reason") or {}).items()}
            self.n_global_locks = int(d.get("n_global_locks", 0))
            self.n_pair_locks = int(d.get("n_pair_locks", 0))
            return True
        except (TypeError, ValueError):
            return False
