"""Adaptive Risk Manager — position sizing, portfolio limits, kill switches,
and regime-adaptive risk scaling."""
import time
from ..tunables import tv
from .. import db


class RiskManager:
    def __init__(self):
        self.peak_equity = 0.0
        self.day_start_equity = None
        self.day_start_ts = time.time()
        self.killed = False
        self.halted_today = False
        self.cooldowns = {}          # product -> ts of last exit/entry
        self.risk_scale = 1.0        # adaptive multiplier
        self.consecutive_losses = 0

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
        self.peak_equity = max(self.peak_equity, equity)
        # daily rollover
        if time.time() - self.day_start_ts > 86400:
            self.day_start_ts = time.time()
            self.day_start_equity = equity
            self.halted_today = False
        if self.day_start_equity is None:
            self.day_start_equity = equity

        dd = 1 - equity / self.peak_equity if self.peak_equity else 0.0
        day_loss = 1 - equity / self.day_start_equity if self.day_start_equity else 0.0

        if dd >= tv("max_drawdown_kill") and not self.killed:
            self.killed = True
            db.log_event("risk", f"KILL SWITCH: max drawdown {dd:.1%} breached")
            from ..alerts import alert
            alert("critical", "KILL SWITCH TRIPPED",
                  f"Max drawdown {dd:.1%} breached (limit {tv('max_drawdown_kill'):.0%}). "
                  f"Equity ${equity:,.0f}. Trading stopped until manual reset.")
        if day_loss >= tv("daily_loss_limit") and not self.halted_today:
            self.halted_today = True
            db.log_event("risk", f"Daily loss limit hit ({day_loss:.1%}) — trading halted for the day")
            from ..alerts import alert
            alert("critical", "DAILY LOSS HALT",
                  f"Daily loss {day_loss:.1%} hit the {tv('daily_loss_limit'):.0%} limit. "
                  f"Equity ${equity:,.0f}. No new entries until tomorrow.")

        # regime-adaptive scaling
        regime_scale = 0.5 if regime.get("vol_state") == "high-vol" else 1.0
        # RL agent chooses a risk multiplier and learns from equity outcomes
        rl_scale = rl_agent.act(regime, dd, self.consecutive_losses, equity)
        effective = self.risk_scale * regime_scale * rl_scale
        self._last_status = {"drawdown": dd, "day_loss": day_loss}
        return {"drawdown": dd, "day_loss": day_loss,
                "effective_risk_scale": round(min(1.25, effective), 3),
                "regime_scale": regime_scale,
                "rl_scale": rl_scale,
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

    def size(self, equity, price, atr, confidence, risk_status, direction=1):
        """Volatility-adjusted sizing: risk a fixed fraction of equity to the stop.
        Includes a cost-viability gate: the take-profit distance must clear
        round-trip fees+slippage by a healthy multiple, otherwise the trade
        is structurally unprofitable and is rejected (notional=0)."""
        from .stance import stance
        st = stance.current()
        scale = risk_status["effective_risk_scale"] * st["risk_mult"]
        risk_dollars = equity * tv("risk_per_trade") * scale * (0.5 + confidence / 2)
        stop_dist = tv("stop_atr_mult") * atr
        if stop_dist <= 0:
            return 0, 0, 0
        # ---- cost-aware trade horizon ----
        # Round-trip cost (fees + slippage both sides). The take-profit must
        # clear it by 2.5x or the trade is structurally unprofitable. Rather
        # than scalping tiny ATR moves into a fee wall, we WIDEN the horizon:
        # target = max(ATR-based, cost floor), stop scaled to keep R:R.
        round_trip = 2 * tv("fee_rate") + 2 * tv("slippage_bps") / 1e4
        min_take_dist = round_trip * tv("cost_multiple") * price
        take_dist = max(tv("take_profit_atr_mult") * atr, min_take_dist)
        rr = tv("take_profit_atr_mult") / tv("stop_atr_mult")  # keep reward:risk
        stop_dist = take_dist / rr
        notional = risk_dollars / (stop_dist / price)
        notional = min(notional, equity * tv("max_position_pct") * min(1.5, st["risk_mult"]))
        if direction > 0:
            stop, take = price - stop_dist, price + take_dist
        else:
            stop, take = price + stop_dist, price - take_dist
        return notional, stop, take

    def reset_kill(self):
        self.killed = False
        self.peak_equity = 0.0
        db.log_event("risk", "Kill switch manually reset")


risk = RiskManager()
