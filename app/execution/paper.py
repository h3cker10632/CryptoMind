"""Paper-trading Execution Engine (OMS) — realistic fills with fees and
slippage. Supports LONG and SHORT positions (shorts are margin-style, like
perps; funding costs not modeled in paper fills)."""
import time
from ..config import START_CASH
from ..tunables import tv
from .. import db


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
    def _fill_price(self, price, side):
        slip = price * tv("slippage_bps") / 1e4
        return price + slip if side == "buy" else price - slip

    def open(self, product, direction, notional, price, stop, take, reason):
        """direction: +1 long, -1 short (margin-style)."""
        fill = self._fill_price(price, "buy" if direction > 0 else "sell")
        fee = notional * tv("fee_rate")
        if notional + fee > self.cash:
            return None
        qty = notional / fill
        self.cash -= notional + fee     # long: spent; short: margin reserved
        pos = Position(
            product=product, side=direction, qty=qty, entry=fill,
            stop=stop, take=take, water=fill, opened=time.time(),
            reason=reason, fees=fee)
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
            fill = self._fill_price(price, "sell")
            gross = pos["qty"] * fill
            fee = gross * tv("fee_rate")
            self.cash += gross - fee
            pnl = gross - fee - pos["qty"] * pos["entry"] - pos["fees"]
        else:
            fill = self._fill_price(price, "buy")      # buy to cover
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
        from ..alerts import alert
        alert("info", f"Closed {product}",
              f"PnL {pnl:+,.2f} ({reason}) @ {fill:,.2f}")
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
                if atr:
                    trail = pos["water"] + trail_atr_mult * atr
                    if trail < pos["stop"]:
                        pos["stop"] = trail
                if px >= pos["stop"]:
                    self.sell(p, px, "stop-loss/trail")
                elif px <= pos["take"]:
                    self.sell(p, px, "take-profit")

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
