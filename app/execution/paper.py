"""Paper-trading Execution Engine (OMS) — realistic fills with fees and
slippage. Supports LONG and SHORT positions.

Shorts are margin-style (perp-like) and now model the two frictions the
critique flagged as missing:
  * FUNDING accrual — the live OKX funding rate is charged/credited on the
    short's notional each management tick (crowded shorts pay to stay short).
  * LIQUIDATION — if an adverse move consumes the reserved margin (minus a
    maintenance buffer), the position is force-closed, exactly as a venue
    would liquidate it. This stops shorts from being unrealistically riskless.
"""
import time
from ..config import START_CASH
from ..tunables import tv
from .. import db
from .. import money


class Position(dict):
    pass


class PaperBroker:
    def __init__(self):
        self.cash = START_CASH
        self.positions = {}      # product -> Position
        self.closed_trades = []
        self.realized_pnl = 0.0

    # ---------- accounting ----------
    def position_value(self, pos, px):
        if pos.get("side", 1) > 0:
            return pos["qty"] * px
        # short: reserved margin + unrealized pnl
        return pos["margin"] + pos["qty"] * (pos["entry"] - px)

    def equity(self, market):
        eq = self.cash
        for p, pos in self.positions.items():
            px = market.price(p) or pos["entry"]
            eq += self.position_value(pos, px)
        return eq

    def exposure(self, market):
        return sum(pos["qty"] * (market.price(p) or pos["entry"])
                   for p, pos in self.positions.items())

    # ---------- order handling ----------
    def _fill_price(self, price, side, product=None):
        slip = price * tv("slippage_bps") / 1e4
        px = price + slip if side == "buy" else price - slip
        # normalise to the venue tick size (Decimal) so paper matches live
        return money.round_price(product, px) if product else px

    def open(self, product, direction, notional, price, stop, take, reason,
             votes=None, regime_at_entry=None, is_hedge=False):
        """direction: +1 long, -1 short (margin-style).

        `votes` (per-strategy vote dict at entry) and `regime_at_entry` are
        stamped INTO the Position at creation — before it is stored or returned —
        so the trade-PnL attribution can always credit the right strategies even
        if the position is closed on the same tick or the process restarts
        mid-tick. Setting them after open() returned (the old path) risked a
        snapshot/close dropping them, which is why `trade_attributions` stayed 0.

        `is_hedge` marks a market-neutral pair-hedge leg. Hedge legs are managed
        as a PAIR by the hedger (both live or neither) and MUST NOT be closed by
        the directional signal-flip exit, so the flag is baked in here too.
        """
        fill = self._fill_price(price, "buy" if direction > 0 else "sell", product)
        fee = notional * tv("fee_rate")
        if notional + fee > self.cash:
            return None
        # enforce exchange lot/step + min-notional like a real venue would
        qty = money.round_qty(product, notional / fill)
        if qty <= 0 or not money.meets_min_notional(product, qty, fill):
            return None
        self.cash -= notional + fee     # long: spent; short: margin reserved
        pos = Position(
            product=product, side=direction, qty=qty, entry=fill,
            stop=stop, take=take, water=fill, opened=time.time(),
            reason=reason, fees=fee)
        # bake the learning-attribution fields in atomically at creation
        pos["votes"] = {k: round(v, 3) for k, v in (votes or {}).items()
                        if abs(v) > 0.05}
        if regime_at_entry is not None:
            pos["regime_at_entry"] = regime_at_entry
        if is_hedge:
            pos["hedge"] = True
        if direction < 0:
            pos["margin"] = notional
        self.positions[product] = pos
        db.log_trade(product, "buy" if direction > 0 else "short",
                     qty, fill, fee, reason)
        db.log_event("trade", f"OPEN {'LONG' if direction > 0 else 'SHORT'} "
                              f"{product} qty={qty:.6f} @ {fill:.2f} "
                              f"stop={stop:.2f} tp={take:.2f} ({reason})")
        return pos

    def buy(self, product, notional, price, stop, take, reason):
        return self.open(product, 1, notional, price, stop, take, reason)

    def sell(self, product, price, reason):
        """Close a position (long OR short). Name kept for compatibility."""
        pos = self.positions.pop(product, None)
        if not pos:
            return None
        side = pos.get("side", 1)
        if side > 0:
            fill = self._fill_price(price, "sell", product)
            gross = pos["qty"] * fill
            fee = gross * tv("fee_rate")
            self.cash += gross - fee
            pnl = gross - fee - pos["qty"] * pos["entry"] - pos["fees"]
        else:
            fill = self._fill_price(price, "buy", product)      # buy to cover
            fee = pos["qty"] * fill * tv("fee_rate")
            move = pos["qty"] * (pos["entry"] - fill)  # short gains on drop
            self.cash += pos["margin"] + move - fee
            pnl = move - fee - pos["fees"]
        self.realized_pnl += pnl
        trade = {**pos, "exit": fill, "closed": time.time(),
                 "pnl": pnl, "exit_reason": reason}
        self.closed_trades.append(trade)
        db.log_trade(product, "sell" if side > 0 else "cover",
                     pos["qty"], fill, fee, reason, pnl)
        db.log_event("trade", f"CLOSE {'LONG' if side > 0 else 'SHORT'} "
                              f"{product} @ {fill:.2f} pnl={pnl:+.2f} ({reason})")
        # rich per-trade notification (Telegram/webhook) with full details
        try:
            from ..alerts import notify_trade_close
            notify_trade_close(trade)
        except Exception:
            pass
        return trade

    # ---------- stops / targets / trailing ----------
    def manage(self, market, trail_atr_mult, atr_lookup):
        for p in list(self.positions.keys()):
            px = market.price(p)
            if px is None:
                continue
            pos = self.positions[p]
            side = pos.get("side", 1)
            # migrate pre-shorts positions ("high_water" -> "water")
            water = pos.get("water", pos.get("high_water", pos["entry"]))
            atr = atr_lookup(p)
            if side > 0:
                pos["water"] = max(water, px)
                if atr:
                    trail = pos["water"] - trail_atr_mult * atr
                    if trail > pos["stop"]:
                        pos["stop"] = trail
                if px <= pos["stop"]:
                    self.sell(p, px, "stop-loss/trail")
                elif px >= pos["take"]:
                    self.sell(p, px, "take-profit")
            else:
                pos["water"] = min(water, px)          # low-water for shorts
                # ---- funding accrual (crowded shorts pay to stay short) ----
                self._accrue_funding(p, pos, px)
                # ---- liquidation guard: adverse move eating the margin ----
                margin = pos.get("margin", pos["qty"] * pos["entry"])
                loss = pos["qty"] * (px - pos["entry"])       # >0 = losing short
                maint = margin * 0.10                          # 10% maintenance
                if loss >= margin - maint:
                    self.sell(p, px, "LIQUIDATION (margin exhausted)")
                    continue
                if atr:
                    trail = pos["water"] + trail_atr_mult * atr
                    if trail < pos["stop"]:
                        pos["stop"] = trail
                if px >= pos["stop"]:
                    self.sell(p, px, "stop-loss/trail")
                elif px <= pos["take"]:
                    self.sell(p, px, "take-profit")

    def _accrue_funding(self, product, pos, px):
        """Charge/credit perp funding on a short's notional (8h rate, prorated
        to the ~20s tick). Positive funding = shorts receive, longs pay."""
        try:
            from ..data.derivatives import derivatives
            d = derivatives.features(product)
        except Exception:
            d = None
        if not d:
            return
        rate_8h = d.get("funding_rate", 0.0)
        now = time.time()
        last = pos.get("_last_funding", pos.get("opened", now))
        elapsed = now - last
        if elapsed <= 0:
            return
        notional = pos["qty"] * px
        # short RECEIVES funding when rate>0; sign flips the cash effect
        pos["_last_funding"] = now
        self.cash += notional * rate_8h * (elapsed / (8 * 3600))

    def stats(self):
        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        n = len(self.closed_trades)
        return {
            "trades": n,
            "win_rate": round(len(wins) / n, 3) if n else None,
            "realized_pnl": round(self.realized_pnl, 2),
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl"] for t in self.closed_trades if t["pnl"] <= 0)
                              / max(1, n - len(wins)), 2),
        }


broker = PaperBroker()
