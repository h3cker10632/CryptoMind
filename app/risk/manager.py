"""Adaptive Risk Manager — position sizing, portfolio limits, kill switches,
and regime-adaptive risk scaling."""
import time
from datetime import datetime, timezone
from ..tunables import tv
from .. import db

# The RL agent can dial CONVICTION risk down to this floor but never to zero —
# a sit-out only skips discretionary probes, never a cost-viable conviction fill.
RL_CONVICTION_FLOOR = 0.25


def _utc_day():
    """UTC calendar-day index — the daily loss limit resets at 00:00 UTC,
    not on a rolling 24h from first tick (bug fix)."""
    return int(datetime.now(timezone.utc).timestamp() // 86400)


class RiskManager:
    def __init__(self):
        self.peak_equity = 0.0        # TRUE high-water mark — never zeroed
        # After a manual kill-reset we must NOT instantly re-trip on the same
        # historical drawdown, but we must also keep telling the truth about the
        # real high-water mark. So the auto-kill re-arms against this separate
        # baseline (set to equity at reset time) while `peak_equity` stays the
        # genuine all-time peak that `drawdown` is reported from.
        self.kill_arm_peak = 0.0
        self.day_start_equity = None
        self.day_start_ts = time.time()
        self.day_index = _utc_day()
        self.killed = False
        self.kill_reason = ""        # human-readable why + when the switch tripped
        self.kill_ts = None
        self.halted_today = False
        self.halt_reason = ""
        self.cooldowns = {}          # product -> ts of last exit/entry
        self.risk_scale = 1.0        # adaptive multiplier
        self.consecutive_losses = 0
        # Was the interval just ending a REAL chance to trade? Set by the
        # orchestrator at the end of each tick (a fill fired, exposure was open,
        # or a signal cleared the cost gate). When False, flat equity reflects a
        # LOCKED DOOR (fee gate), not a wise sit-out, so the RL agent must not
        # learn from it. Defaults False so the very first tick isn't credited.
        self._tradable_next = False

    # ---------- adaptive updates ----------
    def on_trade_closed(self, trade):
        if trade["pnl"] <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.cooldowns[trade["product"]] = time.time()
        # tighten after loss streaks, relax after wins
        self.risk_scale = max(0.25, min(1.0, 1.0 - 0.2 * self.consecutive_losses))

    def update(self, equity, regime):
        from ..learn.rl_risk import agent as rl_agent
        # TRUE high-water mark — persists across kill-resets so drawdown never
        # lies (previously zeroed on reset, making DD read 0% at a loss).
        self.peak_equity = max(self.peak_equity, equity)
        # kill re-arm baseline — tracks peak since the last reset so a fresh
        # drawdown (not the already-acknowledged one) is what re-trips the kill.
        self.kill_arm_peak = max(self.kill_arm_peak, equity)
        # daily rollover — on UTC calendar-day boundary (not rolling 24h)
        today = _utc_day()
        if today != self.day_index:
            self.day_index = today
            self.day_start_ts = time.time()
            self.day_start_equity = equity
            self.halted_today = False
        if self.day_start_equity is None:
            self.day_start_equity = equity

        # `dd` is the TRUE drawdown from the all-time peak — this is what gets
        # reported everywhere (dashboard, stance, snapshot) and never resets.
        dd = 1 - equity / self.peak_equity if self.peak_equity else 0.0
        # `arm_dd` is the drawdown since the last kill-reset — the auto-kill
        # trips on THIS so a reset genuinely re-arms instead of instantly
        # re-tripping on a drawdown the operator already acknowledged.
        arm_dd = 1 - equity / self.kill_arm_peak if self.kill_arm_peak else 0.0
        day_loss = 1 - equity / self.day_start_equity if self.day_start_equity else 0.0

        if arm_dd >= tv("max_drawdown_kill") and not self.killed:
            self.trip_kill(
                f"Auto: max drawdown {arm_dd:.1%} breached the "
                f"{tv('max_drawdown_kill'):.0%} limit "
                f"(equity ${equity:,.0f}, peak ${self.kill_arm_peak:,.0f})")
            from ..alerts import alert
            alert("critical", "KILL SWITCH TRIPPED",
                  f"Max drawdown {dd:.1%} breached (limit {tv('max_drawdown_kill'):.0%}). "
                  f"Equity ${equity:,.0f}. Trading stopped until manual reset.")
        if day_loss >= tv("daily_loss_limit") and not self.halted_today:
            self.halted_today = True
            self.halt_reason = (f"Auto: daily loss {day_loss:.1%} hit the "
                                f"{tv('daily_loss_limit'):.0%} limit "
                                f"(equity ${equity:,.0f})")
            db.log_event("risk", f"Daily loss limit hit ({day_loss:.1%}) — trading halted for the day")
            from ..alerts import alert
            alert("critical", "DAILY LOSS HALT",
                  f"Daily loss {day_loss:.1%} hit the {tv('daily_loss_limit'):.0%} limit. "
                  f"Equity ${equity:,.0f}. No new entries until tomorrow.")

        # regime-adaptive scaling
        regime_scale = 0.5 if regime.get("vol_state") == "high-vol" else 1.0
        # RL agent chooses a risk multiplier and learns from equity outcomes.
        # It only learns from the PREVIOUS interval when that interval was a real
        # chance to trade (`_tradable_next`, set by the orchestrator last tick) —
        # otherwise flat equity is a locked door, not a good sit-out call.
        rl_scale = rl_agent.act(regime, dd, self.consecutive_losses, equity,
                                tradable=self._tradable_next)
        rl_sit_out = (rl_scale == 0.0)
        # CRITICAL CONTRACT: sit-out means "skip discretionary probes / extra
        # names", NOT "size a cost-viable conviction trade to zero". So the
        # scale that reaches size() is floored — the agent can dial risk DOWN
        # but can never freeze the account shut on a structurally profitable
        # signal (that decision belongs to the cost gate, not the Q-table).
        conviction_scale = max(RL_CONVICTION_FLOOR, rl_scale)
        effective = self.risk_scale * regime_scale * conviction_scale
        self._last_status = {"drawdown": dd, "day_loss": day_loss}
        return {"drawdown": dd, "day_loss": day_loss,
                "effective_risk_scale": round(min(1.25, effective), 3),
                "regime_scale": regime_scale,
                "rl_scale": rl_scale,               # raw agent choice (may be 0)
                "rl_sit_out": rl_sit_out,           # skip probes this tick
                "streak_scale": self.risk_scale}

    # ---------- gates & sizing ----------
    def can_open(self, product, broker, market, data_healthy):
        if self.killed:
            return False, "kill switch active"
        if self.halted_today:
            return False, "daily loss halt"
        if not data_healthy:
            return False, "data feed unhealthy"
        # macro-event blackout: no NEW risk around CPI/FOMC/NFP prints
        from ..data.calendar import calendar
        blackout, ev = calendar.blackout()
        if blackout:
            return False, f"macro blackout: {ev['title']}"
        if product in broker.positions:
            return False, "already in position"
        from .stance import stance
        max_pos = max(1, round(tv("max_open_positions") * stance.current()["max_pos_mult"]))
        if len(broker.positions) >= max_pos:
            return False, f"max open positions ({max_pos} in {stance.current()['label']} stance)"
        eq = broker.equity(market)
        if broker.exposure(market) / eq >= tv("max_gross_exposure"):
            return False, "max gross exposure"
        last = self.cooldowns.get(product, 0)
        if time.time() - last < tv("cooldown_sec"):
            return False, "cooldown"
        ok, why = self.price_sane(product, market)
        if not ok:
            return False, why
        return True, "ok"

    def price_sane(self, product, market):
        """Reject entries on coins whose price is junk or has just spiked /
        collapsed versus its own recent history.

        Two independent checks:
          1. Absolute floor — sub-`min_price` assets (e.g. penny/junk coins)
             have quote grids and spreads that make honest sizing impossible.
          2. Relative band — if the live price is more than `price_spike_mult`x
             its recent median (or below 1/mult of it), the quote is a spike or
             a stale/broken print; entering there is buying the top of a pump.
        This is exactly the gate PUMP-USD should have failed.
        """
        price = market.price(product)
        if price is None or price <= 0:
            return False, "no price"
        if price < tv("min_price"):
            return False, f"price ${price:.6f} below min tradable ${tv('min_price'):.4f}"
        closes = market.closes(product) if hasattr(market, "closes") else []
        if len(closes) >= 20:
            import statistics
            med = statistics.median(closes[-60:])
            if med > 0:
                mult = tv("price_spike_mult")
                if price > med * mult:
                    return False, f"price spiked {price/med:.1f}x above recent median"
                if price < med / mult:
                    return False, f"price collapsed to {price/med:.2f}x of recent median"
        return True, "ok"

    def funding_gate(self, product, direction):
        """Extreme funding blocks entries in the crowded direction:
        very positive funding → block longs (long-squeeze risk);
        very negative funding → block shorts (short-squeeze risk)."""
        from ..data.derivatives import derivatives
        d = derivatives.features(product)
        if not d:
            return True, "ok"
        fr = d["funding_rate"]
        if direction > 0 and fr > tv("funding_extreme"):
            return False, f"funding extreme ({fr:.4%}/8h) — long-squeeze risk"
        if direction < 0 and fr < -tv("funding_extreme"):
            return False, f"funding extreme ({fr:.4%}/8h) — short-squeeze risk"
        return True, "ok"

    def size(self, equity, price, atr, confidence, risk_status, direction=1,
             product=None, ml_confidence=1.0):
        """Volatility-adjusted sizing: risk a fixed fraction of equity to the stop.
        Includes a cost-viability gate: the take-profit distance must clear
        round-trip fees+slippage by a healthy multiple, otherwise the trade
        is structurally unprofitable and is rejected (notional=0).

        ml_confidence in [0,1] is the online committee's predictive confidence
        (1.0 = certain / not participating). It scales risk DOWN when the model
        is uncertain — wide quantile bands or members disagreeing — so an
        unsure model bets small instead of full size. Bounded below so it damps,
        never zeroes, a trade the rest of the stack still wants."""
        from .stance import stance
        st = stance.current()
        scale = risk_status["effective_risk_scale"] * st["risk_mult"]
        ml_mult = 0.4 + 0.6 * max(0.0, min(1.0, ml_confidence))   # 0.4x .. 1.0x
        risk_dollars = (equity * tv("risk_per_trade") * scale
                        * (0.5 + confidence / 2) * ml_mult)
        # ---- honest ATR-based stop & target ----
        # Size from the REAL volatility horizon, never a fee-floor-inflated one.
        stop_dist = tv("stop_atr_mult") * atr
        take_dist = tv("take_profit_atr_mult") * atr
        if stop_dist <= 0 or take_dist <= 0:
            return 0, 0, 0
        # ---- cost-viability gate ----
        # Round-trip cost (fees + slippage, both sides). If the honest ATR
        # take-profit can't clear it by `cost_multiple`, the trade is
        # structurally unprofitable at its natural horizon: SKIP it. We do NOT
        # widen the target to the cost floor — doing that quietly distorts the
        # stop and, on a low-ATR penny coin, balloons notional straight to the
        # position cap (exactly the PUMP-USD failure).
        round_trip = 2 * tv("fee_rate") + 2 * tv("slippage_bps") / 1e4
        min_take_dist = round_trip * tv("cost_multiple") * price
        if take_dist < min_take_dist:
            return 0, 0, 0        # unprofitable after costs -> no trade
        notional = risk_dollars / (stop_dist / price)
        notional = min(notional, equity * tv("max_position_pct") * min(1.5, st["risk_mult"]))
        # per-coin liquidity cap: never take more than `liq_cap_pct` of the
        # asset's ~24h traded dollar volume, so our own order can't move a thin
        # discovered coin's book (also keeps paper fills realistic).
        liq = self._liquidity_notional(price, product)
        if liq is not None:
            notional = min(notional, liq * tv("liq_cap_pct"))
        if direction > 0:
            stop, take = price - stop_dist, price + take_dist
        else:
            stop, take = price + stop_dist, price - take_dist
        return notional, stop, take

    def _liquidity_notional(self, price, product=None):
        """Rough 24h traded dollar volume for the product being sized.

        The product is passed in explicitly (previously this reached into a
        `_sizing_product` attribute / the signal engine's `_cur_product()`
        thread-local, which could resolve to the WRONG coin — so thin-coin
        caps were silently computed against another asset's volume). We fall
        back to the old hack only if no product was supplied.

        Uses the market feed's stored 5-min candles (288 bars ≈ 24h). Returns
        None if we can't estimate it (then no liquidity cap is applied).
        """
        from ..data.market import market
        p = product or getattr(self, "_sizing_product", None)
        if not p:
            from ..signals.engine import _cur_product
            p = _cur_product()
        cs = market.candles.get(p, [])
        if len(cs) < 12:
            return None
        recent = cs[-288:]
        base_vol = sum(c[5] for c in recent)      # base-asset volume
        return base_vol * price

    def trip_kill(self, reason):
        """Trip the kill switch and RECORD why + when, so the dashboard can
        always explain a KILLED state (auto-drawdown, operator, or a kill flag
        restored from a saved snapshot). Idempotent-ish: keeps the first
        reason if already killed without one."""
        self.killed = True
        self.kill_reason = reason
        self.kill_ts = time.time()
        db.log_event("risk", f"KILL SWITCH: {reason}")

    def reset_kill(self):
        self.killed = False
        self.kill_reason = ""
        self.kill_ts = None
        # DO NOT zero peak_equity — that would erase the true high-water mark
        # and make drawdown read 0% at a loss (drawdown amnesia). Instead,
        # re-arm the kill baseline to the CURRENT equity so the same
        # already-acknowledged drawdown does not instantly re-trip the switch.
        self.kill_arm_peak = 0.0
        # clearing the kill switch must also clear the daily loss halt and the
        # loss-streak throttle — otherwise the operator "resets" but new
        # entries stay blocked by `halted_today` until the next UTC day.
        self.halted_today = False
        self.halt_reason = ""
        self.consecutive_losses = 0
        self.risk_scale = 1.0
        db.log_event("risk", "Kill switch manually reset (daily halt cleared)")


risk = RiskManager()
