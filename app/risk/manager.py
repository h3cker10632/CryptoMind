"""Adaptive Risk Manager — position sizing, portfolio limits, kill switches,
and regime-adaptive risk scaling."""
import time
from datetime import datetime, timezone
from ..tunables import tv, TUNABLES
from .. import db
from .stop_calibrator import StopCalibrator
from .protections import ProtectionManager

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
        # What the drawdown / daily-loss limits measure. "active_ex_core" =
        # account equity WITHOUT the core holding's P&L: the core is built to
        # ride through BTC/ETH drawdowns (-50%+ in backtests) and has its own
        # guard (core tracking monitor), so its swings must not trip a kill
        # switch that only stops the other sleeves. None = not yet rebased.
        self.dd_basis = None
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
        # REPLAY BRAKE: whole-portfolio risk multiplier set by the automatic
        # strategy replay (orchestrator.replay_loop). 1.0 = no brake; drops to
        # `replay_brake_mult` while the replay is negative in BOTH halves of its
        # window. A risk control, not a forecast — see _replay_brake_for().
        self.replay_brake = 1.0
        self.replay_brake_reason = ""
        self.consecutive_losses = 0
        # CONFORMAL stop calibrator: learns the ATR stop multiple that contains
        # (1-alpha) of realized adverse excursions, so the stop sits just
        # outside normal noise instead of at a guessed constant.
        self.stop_calibrator = StopCalibrator()
        # PROTECTIONS: time-boxed, self-healing circuit breakers derived from
        # recent trade outcomes (StoplossGuard / LowProfitPairs / MaxDrawdown).
        self.protections = ProtectionManager()
        # Was the interval just ending a REAL chance to trade? Set by the
        # orchestrator at the end of each tick (a fill fired, exposure was open,
        # or a signal cleared the cost gate). When False, flat equity reflects a
        # LOCKED DOOR (fee gate), not a wise sit-out, so the RL agent must not
        # learn from it. Defaults False so the very first tick isn't credited.
        self._tradable_next = False

    def reset_account_baselines(self, equity):
        self.peak_equity = equity
        self.kill_arm_peak = equity
        self.day_start_equity = equity
        self.day_start_ts = time.time()
        self.day_index = _utc_day()

    def rebase(self, equity, basis):
        """Switch what the limits measure: re-baseline the peaks and the day
        start at `equity` so the change itself can't look like a drawdown."""
        self.peak_equity = equity
        self.kill_arm_peak = equity
        self.day_start_equity = equity
        self.dd_basis = basis
        db.log_event("risk", f"Drawdown limits now measure {basis} equity "
                             f"(re-baselined at ${equity:,.0f})")

    def reconcile_account_peak(self, saved_peak, opening_equity):
        self.peak_equity = max(float(saved_peak or 0.0), float(opening_equity))

    # ---------- adaptive updates ----------
    def on_trade_closed(self, trade):
        if trade["pnl"] <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.cooldowns[trade["product"]] = time.time()
        # tighten after loss streaks, relax after wins
        self.risk_scale = max(0.25, min(1.0, 1.0 - 0.2 * self.consecutive_losses))
        self._feed_stop_calibrator(trade)

    def _feed_stop_calibrator(self, trade):
        """Fold a closed trade's adverse excursion into the conformal stop
        calibrator. Stop/liquidation exits are CENSORED (their MAE is capped by
        the stop itself) so they only count toward the realized stop-rate, never
        the score set — see stop_calibrator.py."""
        try:
            atr = trade.get("atr_at_entry")
            entry = trade.get("entry")
            mae_price = trade.get("mae_price")
            side = trade.get("side", 1)
            reason = (trade.get("exit_reason") or "").lower()
            stopped = "stop" in reason or "liquidation" in reason
            mae_atr = None
            if atr and atr > 0 and entry and mae_price is not None:
                adverse = (entry - mae_price) if side > 0 else (mae_price - entry)
                mae_atr = max(0.0, adverse) / atr
            self.stop_calibrator.observe(mae_atr, stopped)
        except Exception:
            pass      # calibration must never break trade bookkeeping

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
                  f"Max drawdown {dd:.1%} breached (limit {tv('max_drawdown_kill'):.0%}) "
                  f"on account equity excluding the core holding (${equity:,.0f}). "
                  f"New entries stopped in the hourly bot and exploration sleeves until "
                  f"manual reset (Polymarket runs its own bankroll); open positions are still managed to "
                  f"their exits. The core keeps following its champion (own tracking "
                  f"monitor).")
        if day_loss >= tv("daily_loss_limit") and not self.halted_today:
            self.halted_today = True
            self.halt_reason = (f"Auto: daily loss {day_loss:.1%} hit the "
                                f"{tv('daily_loss_limit'):.0%} limit "
                                f"(equity ${equity:,.0f})")
            db.log_event("risk", f"Daily loss limit hit ({day_loss:.1%}) — trading halted for the day")
            from ..alerts import alert
            alert("critical", "DAILY LOSS HALT",
                  f"Daily loss {day_loss:.1%} hit the {tv('daily_loss_limit'):.0%} limit "
                  f"on account equity excluding the core holding (${equity:,.0f}). "
                  f"No new entries in the hourly bot and exploration sleeves "
                  f"until tomorrow (UTC); the core is unaffected.")

        # regime-adaptive scaling
        regime_scale = 0.5 if regime.get("vol_state") == "high-vol" else 1.0
        # RL agent chooses a risk multiplier and learns from equity outcomes.
        # It only learns from the PREVIOUS interval when that interval was a real
        # chance to trade (`_tradable_next`, set by the orchestrator last tick) —
        # otherwise flat equity is a locked door, not a good sit-out call.
        rl_scale = rl_agent.act(regime, dd, self.consecutive_losses, equity,
                                tradable=self._tradable_next)
        # Gated off (app/learn/gate.py): the agent keeps learning, but its
        # choice is not applied.
        from ..learn.gate import active as _learner_active
        if not _learner_active("rl_risk"):
            rl_scale = 1.0
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
                "streak_scale": self.risk_scale,
                "stop_calibration": self.stop_calibrator.stats(),
                "protections": self.protections.stats()}

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
        from ..data.memes import memes
        max_pos = max(1, round(tv("max_open_positions") * stance.current()["max_pos_mult"]))
        # memes have their own dedicated blast-radius caps below and never
        # compete with non-meme coins for the general position-count budget.
        if not memes.is_meme(product):
            held_non_meme = [q for q in broker.positions if not memes.is_meme(q)]
            if len(held_non_meme) >= max_pos:
                return False, f"max open positions ({max_pos} in {stance.current()['label']} stance)"
        eq = broker.equity(market)
        if broker.exposure(market) / eq >= tv("max_gross_exposure"):
            return False, "max gross exposure"
        last = self.cooldowns.get(product, 0)
        if time.time() - last < tv("cooldown_sec"):
            return False, "cooldown"
        # PROTECTIONS: self-healing circuit breakers over recent outcomes — a
        # global halt after a stop-out cluster / temporary drawdown, or a
        # per-coin lockout for a chronic under-performer.
        locked, why = self.protections.is_locked(product)
        if locked:
            return False, why
        ok, why = self.price_sane(product, market)
        if not ok:
            return False, why
        # ---- spread gate (freqtrade SpreadFilter idea) ----
        # A wide bid/ask spread is a round-trip cost paid up front; entering into
        # one hands the edge to the market maker. Skip the entry when the live
        # spread exceeds the cap (0 = disabled). Only gates when we actually have
        # a fresh book reading, so a missing book never blocks trading.
        max_spread = tv("max_spread_bps")
        if max_spread > 0:
            book = getattr(market, "books", {}).get(product)
            if book is not None:
                spread = book.get("spread_bps", 0.0)
                if spread > max_spread:
                    return False, f"spread too wide ({spread:.0f}bps > {max_spread}bps)"
        # ---- meme blast-radius caps ----
        # Memes are high-variance; keep them from taking over the book. Two
        # independent caps: a limit on CONCURRENT meme positions and a limit on
        # total meme exposure as a fraction of equity. Non-meme trading is
        # unaffected by these.
        if memes.is_meme(product):
            if not memes.enabled():
                return False, "meme trading disabled"
            held_memes = [q for q in broker.positions if memes.is_meme(q)]
            if len(held_memes) >= tv("meme_max_positions"):
                return False, f"max meme positions ({tv('meme_max_positions')})"
            meme_expo = sum(broker.position_notional(q, market)
                            if hasattr(broker, "position_notional")
                            else abs(broker.positions[q]["qty"]) * (market.price(q) or 0)
                            for q in held_memes)
            if eq > 0 and meme_expo / eq >= tv("meme_max_exposure"):
                return False, "max meme exposure"
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
        scale = (risk_status["effective_risk_scale"] * st["risk_mult"]
                 * getattr(self, "replay_brake", 1.0))
        ml_mult = 0.4 + 0.6 * max(0.0, min(1.0, ml_confidence))   # 0.4x .. 1.0x
        # ---- meme risk envelope ----
        # A meme position risks a fraction of the normal dollar risk and uses a
        # wider stop (memes gap hard, so a normal-width stop just donates the
        # spread on noise). Both are tunables; 1.0 / 1.0 disables the effect.
        from ..data.memes import memes
        is_meme = product is not None and memes.is_meme(product)
        meme_risk_mult = tv("meme_risk_factor") if is_meme else 1.0
        meme_stop_mult = tv("meme_stop_widen") if is_meme else 1.0
        risk_dollars = (equity * tv("risk_per_trade") * scale
                        * (0.5 + confidence / 2) * ml_mult * meme_risk_mult)
        # ---- honest ATR-based stop & target ----
        # Size from the REAL volatility horizon, never a fee-floor-inflated one.
        # CONFORMAL STOP: the multiple is the (1-alpha) quantile of realized
        # adverse excursions (ATR units) once enough uncensored trades exist —
        # a stop wide enough to survive ~(1-alpha) of normal noise — otherwise
        # the operator's tunable default. The take-profit multiple scales with
        # it so the risk:reward geometry the operator set is preserved.
        base_stop = tv("stop_atr_mult")
        spec = TUNABLES["stop_atr_mult"]
        stop_mult = self.stop_calibrator.stop_mult(base_stop, spec["min"], spec["max"])
        rr = tv("take_profit_atr_mult") / base_stop if base_stop else 1.5
        take_mult = stop_mult * rr
        stop_dist = stop_mult * atr * meme_stop_mult
        take_dist = take_mult * atr * meme_stop_mult
        if stop_dist <= 0 or take_dist <= 0:
            return 0, 0, 0
        # ---- cost-viability gate ----
        # Round-trip cost (fees + slippage, both sides). If the honest ATR
        # take-profit can't clear it by `cost_multiple`, the trade is
        # structurally unprofitable at its natural horizon: SKIP it. We do NOT
        # widen the target to the cost floor — doing that quietly distorts the
        # stop and, on a low-ATR penny coin, balloons notional straight to the
        # position cap (exactly the PUMP-USD failure).
        from ..execution.costs import round_trip_cost
        round_trip = round_trip_cost()
        min_take_dist = round_trip * tv("cost_multiple") * price
        if take_dist < min_take_dist:
            return 0, 0, 0        # unprofitable after costs -> no trade
        notional = risk_dollars / (stop_dist / price)
        pos_cap = equity * tv("max_position_pct") * min(1.5, st["risk_mult"])
        if is_meme:
            # never let the position-% fallback re-inflate a meme past its share
            pos_cap = min(pos_cap, equity * tv("max_position_pct") * meme_risk_mult)
        notional = min(notional, pos_cap)
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

        Uses the market feed's stored candles (the last 24h of native bars). Returns
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
        from ..config import BARS_PER_DAY
        recent = cs[-BARS_PER_DAY:]
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
