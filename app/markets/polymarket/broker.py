"""Paper-trading broker for Polymarket binary outcome shares.

A prediction-market position is NOT a leveraged perp — it is a fully-collateral-
ized long in ONE outcome token that pays exactly $1 if that outcome resolves YES
and $0 if it resolves NO. There is no margin, no liquidation, no funding, and no
short: to "bet against" outcome A you simply buy shares of the complementary
outcome B. So this broker is deliberately simpler than the crypto PaperBroker —
it models the three frictions that actually exist here:

  * SLIPPAGE — you buy at the ask (mid + slip) and sell at the bid (mid - slip),
    both in probability points (a "price" here is 0..1).
  * FEE      — Polymarket currently charges zero trading fees, but the rate is a
    tunable so the operator can stress the book at a hypothetical fee.
  * RESOLUTION — the honest exit. Shares held to resolution pay $1 or $0; there
    is no mark-to-market fantasy about the terminal value.

Positions are keyed by outcome token id (a market can, in principle, be held on
both legs, though the sizer won't do that). Every closed trade carries the same
attribution fields (`votes`, `regime`) the crypto book stamps, so the shared
learner can credit the strategies that voted for it.
"""
from __future__ import annotations

import time

PM_START_CASH = 10_000.0        # separate paper bankroll from the crypto book


class PMBroker:
    def __init__(self, start_cash: float = PM_START_CASH):
        self.start_cash = float(start_cash)
        self.cash = float(start_cash)
        self.positions: dict[str, dict] = {}     # token_id -> position
        self.closed_trades: list[dict] = []
        self.realized_pnl = 0.0

    # ------------------------------ accounting ------------------------------
    def market_value(self, pos: dict, mid: float) -> float:
        """Mark-to-market value of the shares at the current midpoint."""
        return pos["shares"] * max(0.0, min(1.0, mid))

    def equity(self, price_lookup) -> float:
        """cash + MTM of open shares. `price_lookup(token_id)->mid|None`."""
        eq = self.cash
        for tid, pos in self.positions.items():
            mid = price_lookup(tid)
            if mid is None:
                mid = pos["entry"]
            eq += self.market_value(pos, mid)
        return eq

    def exposure(self, price_lookup) -> float:
        return sum(self.market_value(p, price_lookup(t) or p["entry"])
                   for t, p in self.positions.items())

    # ------------------------------ orders ---------------------------------
    def open(self, market: dict, outcome_index: int, mid: float, stake: float,
             fee_rate: float, slippage: float, stop: float, take: float,
             reason: str, votes=None, regime=None, edge: float = 0.0):
        """Buy `stake` USDC of one outcome token at ask = mid + slippage.

        `stop`/`take` are absolute price levels (0..1) on THIS token's price for
        the early-exit manager. Returns the created position or None if the fill
        can't be honoured (insufficient cash or below the market's min order).
        """
        tid = market["token_ids"][outcome_index]
        if tid in self.positions:
            return None                      # already holding this leg
        tick = market.get("tick_size", 0.01) or 0.01
        ask = min(0.999, mid + slippage)
        ask = round(ask / tick) * tick
        ask = max(tick, min(1.0 - tick, ask))
        fee = stake * fee_rate
        if stake + fee > self.cash:
            return None
        min_order = max(1.0, float(market.get("min_order_size", 5.0)))
        if stake < min_order:
            return None
        shares = stake / ask
        self.cash -= stake + fee
        pos = {
            "condition_id": market["condition_id"],
            "token_id": tid,
            "outcome": market["outcomes"][outcome_index],
            "outcome_index": outcome_index,
            "question": market["question"],
            "category": market.get("category", "other"),
            "shares": shares,
            "entry": ask,
            "cost": stake + fee,
            "fees": fee,
            "stop": stop,
            "take": take,
            "opened": time.time(),
            "reason": reason,
            "edge_at_entry": round(edge, 4),
            "votes": {k: round(v, 3) for k, v in (votes or {}).items()
                      if abs(v) > 0.05},
            "regime_at_entry": regime,
            "water": ask,
        }
        self.positions[tid] = pos
        return pos

    def _finish(self, pos: dict, exit_price: float, proceeds: float,
                reason: str) -> dict:
        pnl = proceeds - pos["cost"]
        self.realized_pnl += pnl
        self.cash += proceeds
        trade = {**pos, "exit": exit_price, "closed": time.time(),
                 "pnl": pnl, "exit_reason": reason,
                 "return_pct": pnl / pos["cost"] if pos["cost"] else 0.0}
        self.closed_trades.append(trade)
        return trade

    def exit(self, token_id: str, mid: float, fee_rate: float, slippage: float,
             reason: str):
        """Sell shares back to the book BEFORE resolution at bid = mid - slip."""
        pos = self.positions.pop(token_id, None)
        if not pos:
            return None
        bid = max(0.001, mid - slippage)
        gross = pos["shares"] * bid
        fee = gross * fee_rate
        return self._finish(pos, bid, gross - fee, reason)

    def resolve(self, token_id: str, won: bool, reason="resolution"):
        """Settle a held position at resolution: $1/share if won else $0."""
        pos = self.positions.pop(token_id, None)
        if not pos:
            return None
        payoff = pos["shares"] * (1.0 if won else 0.0)
        return self._finish(pos, 1.0 if won else 0.0, payoff, reason)

    def manage(self, price_lookup, fee_rate: float, slippage: float):
        """Early take-profit / stop-loss on the token's live midpoint."""
        closed = []
        for tid in list(self.positions.keys()):
            mid = price_lookup(tid)
            if mid is None:
                continue
            pos = self.positions[tid]
            pos["water"] = max(pos.get("water", pos["entry"]), mid)
            if pos["take"] and mid >= pos["take"]:
                closed.append(self.exit(tid, mid, fee_rate, slippage,
                                        "take-profit"))
            elif pos["stop"] and mid <= pos["stop"]:
                closed.append(self.exit(tid, mid, fee_rate, slippage,
                                        "stop-loss"))
        return [t for t in closed if t]

    # ------------------------------ reporting ------------------------------
    def reset(self, start_cash: float | None = None):
        if start_cash is not None:
            self.start_cash = float(start_cash)
        self.cash = self.start_cash
        self.positions.clear()
        self.closed_trades.clear()
        self.realized_pnl = 0.0

    def stats(self):
        n = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        return {
            "trades": n,
            "open_positions": len(self.positions),
            "win_rate": round(len(wins) / n, 3) if n else None,
            "realized_pnl": round(self.realized_pnl, 2),
            "start_cash": round(self.start_cash, 2),
            "cash": round(self.cash, 2),
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2)
                       if wins else 0.0,
            "avg_loss": round(sum(t["pnl"] for t in self.closed_trades
                                  if t["pnl"] <= 0) / max(1, n - len(wins)), 2),
        }


broker = PMBroker()
