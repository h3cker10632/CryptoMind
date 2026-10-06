"""Exit Advisor — learns when to CUT a position early vs STAY IN.

Motivation: a stop-loss only fires at a fixed price; it can't tell that the
move has clearly turned against you and is *going to keep going*. This advisor
adds a predictive, self-learning early-exit:

  * PREDICTIVE — each tick it forms an expected next-horizon return **in the
    position's own frame** (positive = the position keeps working, negative =
    it keeps going against you) from two sources: the online ML committee's
    forward-return prediction, and a learned value estimate for the current
    market state. If a losing position's expected continuation is confidently
    negative, it cuts the loss before it gets worse.

  * SELF-LEARNING — every consultation records the state + the mark price. A
    horizon later we look up what price actually did and update the state's
    value estimate with the realized position-frame return. Because the reward
    is the *counterfactual* move (what the position WOULD have done next),
    BOTH implied actions are scored from one observation: holding is good
    exactly when that return is positive, cutting is good exactly when it is
    negative. Over time the advisor learns, per state, whether staying in pays.

The advisor NEVER blocks the hard stop / take-profit / liquidation logic — it
only adds an earlier, smarter exit on top. Hedge legs are exempt (managed as a
market-neutral pair by the hedger).
"""
import time
from collections import deque


class ExitAdvisor:
    def __init__(self):
        # state -> [n, mean_return]  (EMA-ish running estimate of the next-horizon
        # position-frame return observed from that state)
        self.v = {}
        # decisions awaiting counterfactual scoring: (ts, product, side, price, state)
        self.pending = deque(maxlen=6000)
        self.n_updates = 0
        self.n_cuts = 0
        self.last_reason = ""

    # ---------------- tunables ----------------
    @staticmethod
    def _enabled():
        try:
            from .. import settings
            from .gate import active
            return bool(settings.get("exit_advisor_enabled")) and active("exit_advisor")
        except Exception:
            return False

    @staticmethod
    def _tv(key, default):
        try:
            from ..tunables import tv
            return tv(key)
        except Exception:
            return default

    # ---------------- state encoding ----------------
    @staticmethod
    def _state(side, unrealized_pct, ml_pos, regime, atr_pct):
        """Compact, learnable market state for a position.
          side_trend : is the position WITH or AGAINST the BTC regime trend
          pnl_b      : winning / small-loss / big-loss bucket (ATR-scaled)
          ml_b       : model's forward view in the position frame (adverse/flat/favorable)
          vol        : regime vol state
        """
        trend = regime.get("trend", "sideways")
        trend_num = 1 if trend == "bull" else -1 if trend == "bear" else 0
        st = side * trend_num
        side_trend = "with" if st > 0 else "against" if st < 0 else "neutral"
        # pnl buckets scaled by the asset's own volatility (ATR as % of price)
        unit = max(atr_pct, 0.002)
        if unrealized_pct >= 0:
            pnl_b = "win"
        elif unrealized_pct > -unit:
            pnl_b = "small_loss"
        else:
            pnl_b = "big_loss"
        ml_b = "fav" if ml_pos > 0.05 else "adv" if ml_pos < -0.05 else "flat"
        vol = regime.get("vol_state", "normal")
        return (side_trend, pnl_b, ml_b, vol)

    def value(self, state):
        rec = self.v.get(state)
        return rec[1] if rec else 0.0

    def confidence(self, state):
        """0..1 confidence in the value estimate — grows with observations."""
        rec = self.v.get(state)
        if not rec:
            return 0.0
        return min(1.0, rec[0] / 20.0)

    # ---------------- decision ----------------
    def decide(self, product, side, unrealized_pct, ml_pos, regime, atr_pct,
               ml_pos_hi=None):
        """Return (action, reason, expected_return). action is 'cut' or 'hold'.

        `ml_pos` is the ML committee's forward-return prediction already rotated
        into the POSITION frame (side * raw_pred), so >0 means the model expects
        the position to keep working and <0 means it expects it to keep losing.

        `ml_pos_hi`, when provided, is the OPTIMISTIC (upper) end of the model's
        CONFORMALLY-CALIBRATED forward-return interval, also in the position
        frame. It is only supplied once the calibrator has a coverage guarantee.
        We use it as a one-sided safety guard: never cut a loser while even the
        optimistic end of its calibrated interval is still above the ceiling —
        i.e. the band says a bounce is genuinely plausible, not just hoped for.
        """
        state = self._state(side, unrealized_pct, ml_pos, regime, atr_pct)
        # record this consultation for later counterfactual scoring (throttled
        # by the caller via the price sampler — here we always append; the
        # sampler cadence lives in the orchestrator hook).
        v = self.value(state)
        vc = self.confidence(state)
        # Blend the model's forward view with the learned state value. Weight the
        # learned value by how much evidence backs it; the model always counts.
        w_ml = self._tv("exit_ml_weight", 1.0)
        expected = w_ml * ml_pos + vc * v
        thresh = self._tv("exit_cut_threshold", 0.004)
        min_loss = self._tv("exit_min_loss_pct", 0.003)
        action, reason = "hold", "expected to keep working"
        # Only CUT a position that is actually losing beyond a small buffer, and
        # only when the blended expectation is confidently negative — so we cut
        # losers we expect to keep bleeding, not winners or noise.
        if unrealized_pct <= -min_loss and expected <= -thresh:
            # CONFORMAL GUARD: if a calibrated interval is available and its
            # optimistic edge still clears the ceiling, the band gives this loser
            # a real (coverage-backed) chance of recovering — hold instead of
            # cutting into what may be noise.
            ceiling = self._tv("exit_conformal_ceiling", 0.0)
            if ml_pos_hi is not None and ml_pos_hi > ceiling:
                reason = (f"conformal hold: calibrated upper return "
                          f"{ml_pos_hi:+.3%} > {ceiling:+.3%} (bounce plausible)")
                return "hold", reason, expected
            action = "cut"
            guard = "" if ml_pos_hi is None else f", hi={ml_pos_hi:+.3%}"
            reason = (f"expected further adverse move "
                      f"(E[r]={expected:+.3%}, ml={ml_pos:+.3%}, "
                      f"learned={v:+.3%}@{vc:.0%}{guard})")
        return action, reason, expected

    def record(self, product, side, price, unrealized_pct, ml_pos, regime, atr_pct,
               ts=None):
        """Log the decision context so the next horizon can score it. `ts`
        defaults to now (the replay ablation passes the bar time)."""
        state = self._state(side, unrealized_pct, ml_pos, regime, atr_pct)
        self.pending.append((time.time() if ts is None else ts, product, side, price, state))

    def note_cut(self):
        self.n_cuts += 1

    # ---------------- counterfactual learning ----------------
    def score_pending(self, price_lookup, now=None):
        """Score matured decisions using the realized next-horizon move.

        `price_lookup(product, ts) -> price|None`. For each pending sample whose
        horizon has elapsed, compute the position-frame return over the horizon
        and fold it into the state's value estimate. Returns #updated.
        """
        now = now or time.time()
        horizon = self._tv("exit_horizon_sec", 1800)
        alpha = 0.15
        updated = 0
        while self.pending and now - self.pending[0][0] >= horizon:
            ts, product, side, price, state = self.pending.popleft()
            future = price_lookup(product, ts + horizon)
            if not future or not price or price <= 0:
                continue
            # position-frame return over the horizon (+ = position kept working)
            r = side * (future / price - 1)
            n, mean = self.v.get(state, (0, 0.0))
            n += 1
            # EMA once we have some history, plain average while cold-starting
            a = alpha if n > 10 else 1.0 / n
            mean = mean + a * (r - mean)
            self.v[state] = (n, mean)
            self.n_updates += 1
            updated += 1
        return updated

    # ---------------- reporting / persistence ----------------
    def stats(self):
        # surface the states where staying in has historically LOST money
        worst = sorted(((k, v) for k, v in self.v.items() if v[0] >= 5),
                       key=lambda kv: kv[1][1])[:6]
        return {
            "enabled": self._enabled(),
            "states_learned": len(self.v),
            "updates": self.n_updates,
            "cuts": self.n_cuts,
            "pending": len(self.pending),
            "worst_hold_states": [
                {"state": "/".join(map(str, k)),
                 "n": v[0], "mean_next_return": round(v[1], 5)} for k, v in worst],
        }

    def capture(self):
        return {
            "v": {"||".join(map(str, k)): [n, m] for k, (n, m) in self.v.items()},
            "pending": [[ts, p, side, price, list(state)]
                        for (ts, p, side, price, state) in self.pending],
            "n_updates": self.n_updates,
            "n_cuts": self.n_cuts,
        }

    def restore(self, d):
        if not d:
            return
        try:
            self.v = {tuple(k.split("||")): (int(v[0]), float(v[1]))
                      for k, v in (d.get("v") or {}).items()}
            self.pending = deque(
                [(ts, p, side, price, tuple(state))
                 for ts, p, side, price, state in (d.get("pending") or [])],
                maxlen=self.pending.maxlen)
            self.n_updates = d.get("n_updates", 0)
            self.n_cuts = d.get("n_cuts", 0)
        except Exception:
            pass


exit_advisor = ExitAdvisor()
