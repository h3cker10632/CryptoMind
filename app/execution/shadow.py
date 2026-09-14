"""ShadowVenue + ShadowBroker — realistic execution simulation, zero real risk.

The ShadowVenue implements the OMS `Venue` interface but fills against LIVE
prices with realistic microstructure effects: taker fee, slippage, and
occasional partial fills split across two executions (so the partial-fill /
VWAP / idempotency paths get exercised for real). It NEVER contacts an
exchange with credentials — it is a simulation that mirrors what a LiveVenue
would do, which is exactly what "shadow mode" means in the roadmap.

The ShadowBroker mirrors the paper account through the OMS so we can measure
live-vs-paper divergence (the roadmap's "most important discovery of Stage 3")
before a single real dollar is ever at risk.
"""
from __future__ import annotations
import random
import time
from .. import db
from .. import money
from ..tunables import tv
from .oms import OMS, Venue, FILLED, PARTIAL


class ShadowVenue(Venue):
    name = "shadow"

    def __init__(self, market, partial_prob=0.25, seed=17):
        self.market = market
        self.partial_prob = partial_prob
        self._rnd = random.Random(seed)
        self._open = {}          # client_order_id -> remaining qty
        self._exec_seq = 0

    def _exec_id(self):
        self._exec_seq += 1
        return f"shadow-{self._exec_seq}"

    def _fill_price(self, product, ref, side):
        slip = ref * tv("slippage_bps") / 1e4
        px = ref + slip if side == "buy" else ref - slip
        return money.round_price(product, px)

    def place(self, intent):
        ref = self.market.price(intent.product) or intent.price
        # enforce venue precision/min-notional like a real exchange would
        qty = money.round_qty(intent.product, intent.qty)
        if qty <= 0 or not money.meets_min_notional(intent.product, qty, ref):
            raise ValueError("below min-notional / lot size")
        fills = []
        # sometimes fill in two pieces to exercise partial-fill handling
        if self._rnd.random() < self.partial_prob:
            first = money.round_qty(intent.product, qty * self._rnd.uniform(0.3, 0.6))
            if first > 0:
                px = self._fill_price(intent.product, ref, intent.side)
                fills.append({"exec_id": self._exec_id(), "qty": first,
                              "price": px, "fee": first * px * tv("fee_rate")})
                self._open[intent.client_order_id] = qty - first
                return fills            # rest fills on the next reconcile
        px = self._fill_price(intent.product, ref, intent.side)
        fills.append({"exec_id": self._exec_id(), "qty": qty, "price": px,
                      "fee": qty * px * tv("fee_rate")})
        return fills

    def cancel(self, intent):
        self._open.pop(intent.client_order_id, None)
        return True

    def fetch_open(self):
        return {cid: {"remaining": q} for cid, q in self._open.items() if q > 1e-12}

    def fetch_fills(self, intent):
        """Fill any remaining shadow quantity on reconcile (VWAP path)."""
        rem = self._open.get(intent.client_order_id, 0.0)
        if rem <= 1e-12:
            return []
        ref = self.market.price(intent.product) or intent.price
        px = self._fill_price(intent.product, ref, intent.side)
        self._open.pop(intent.client_order_id, None)
        return [{"exec_id": self._exec_id(), "qty": rem, "price": px,
                 "fee": rem * px * tv("fee_rate")}]


class ShadowBroker:
    """A second, OMS-driven account run in parallel with the paper broker to
    quantify execution divergence. Positions are DERIVED FROM FILLS only."""

    def __init__(self, market):
        from ..config import START_CASH
        self.market = market
        self.oms = OMS(ShadowVenue(market))
        self.cash = START_CASH
        self.positions = {}      # product -> {side, qty, entry, intent_id}
        self.realized_pnl = 0.0
        self.divergence_bps = []   # per-trade |shadow_fill - paper_fill|/paper

    def mirror_open(self, product, side, notional, paper_fill_price):
        ref = self.market.price(product) or paper_fill_price
        qty = notional / ref if ref else 0.0
        intent = self.oms.submit(product, "buy" if side > 0 else "sell", qty,
                                 ref, reason="mirror", meta={"venue": "shadow"})
        if intent and intent.filled_qty > 0:
            self._record_divergence(intent.avg_fill_price, paper_fill_price)
            self.cash -= intent.filled_qty * intent.avg_fill_price + intent.fees
            self.positions[product] = {"side": side, "qty": intent.filled_qty,
                                       "entry": intent.avg_fill_price,
                                       "intent_id": intent.id}
        return intent

    def mirror_close(self, product, paper_fill_price):
        pos = self.positions.pop(product, None)
        if not pos:
            return None
        ref = self.market.price(product) or paper_fill_price
        intent = self.oms.submit(product, "sell" if pos["side"] > 0 else "buy",
                                 pos["qty"], ref, reason="mirror-close",
                                 meta={"venue": "shadow"})
        if intent and intent.filled_qty > 0:
            self._record_divergence(intent.avg_fill_price, paper_fill_price)
            gross = intent.filled_qty * intent.avg_fill_price
            if pos["side"] > 0:
                self.cash += gross - intent.fees
                pnl = gross - intent.fees - pos["qty"] * pos["entry"]
            else:
                self.cash += pos["qty"] * pos["entry"] - gross - intent.fees
                pnl = pos["qty"] * (pos["entry"] - intent.avg_fill_price) - intent.fees
            self.realized_pnl += pnl
        return intent

    def _record_divergence(self, shadow_px, paper_px):
        if paper_px:
            self.divergence_bps.append(abs(shadow_px - paper_px) / paper_px * 1e4)
            self.divergence_bps = self.divergence_bps[-200:]

    def reconcile(self):
        self.oms.reconcile()

    def snapshot(self):
        div = self.divergence_bps
        return {
            "enabled": True, "mode": "shadow",
            "realized_pnl": round(self.realized_pnl, 2),
            "n_positions": len(self.positions),
            "avg_divergence_bps": round(sum(div) / len(div), 2) if div else None,
            "max_divergence_bps": round(max(div), 2) if div else None,
            "n_divergence_samples": len(div),
            "oms": self.oms.stats(),
        }
