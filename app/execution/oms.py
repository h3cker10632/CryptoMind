"""Order Management System — the production execution machinery, safe by design.

This is the "boring 90%" the critique flagged as missing: durable trade
intents, client order IDs, an explicit order state machine, partial-fill
aggregation to a quantity-weighted average price, idempotent fill handling,
and a reconciliation model that treats the venue as the source of truth.

SAFETY: shipped in SHADOW mode only. `ShadowVenue` simulates fills against
live prices with realistic fees/slippage/partials but NEVER sends a real
order. A `LiveVenue` implementing the same `Venue` interface (place / cancel
/ fetch_open / fetch_fills, all keyed by client order id) is the single seam
where real trading would later plug in — behind human approval gates. It is
intentionally not implemented here.

Design principles (industry consensus):
  * write the intent to durable storage BEFORE touching the venue
  * every order carries a client-generated id → retries are idempotent
  * take truth from executions (fills), not from a status field
  * fail CLOSED on ambiguity (timeout / unconfirmed cancel) → reconcile first
  * positions are DERIVED from fills, never from "an order was submitted"
"""
from __future__ import annotations
import time
import uuid
import threading
from decimal import Decimal
from .. import db
from .. import money

# ---- order lifecycle states (state only moves forward to a terminal) ----
CREATED = "created"            # intent persisted, not yet sent
SUBMITTING = "submitting"      # sent, awaiting ack
OPEN = "open"                  # acked, working
PARTIAL = "partially_filled"
FILLED = "filled"
CANCELED = "canceled"
REJECTED = "rejected"
UNKNOWN = "unknown"           # ambiguous — must be reconciled before new orders
TERMINAL = {FILLED, CANCELED, REJECTED}


class TradeIntent:
    """Internal decision joined to the external order via a stable id."""

    def __init__(self, product, side, qty, price, order_type="market",
                 reason="", meta=None):
        self.id = uuid.uuid4().hex
        self.client_order_id = "cm-" + self.id[:20]
        self.product = product
        self.side = side                     # "buy" | "sell"
        self.qty = float(qty)
        self.price = float(price)
        self.order_type = order_type
        self.reason = reason
        self.meta = meta or {}
        self.status = CREATED
        self.exchange_order_id = None
        self.filled_qty = 0.0
        self.avg_fill_price = 0.0            # quantity-weighted (VWAP)
        self.fees = 0.0
        self._seen_exec_ids = set()          # idempotent fill dedupe
        self.created_at = time.time()
        self.updated_at = self.created_at

    # ---- fill aggregation (idempotent, order-insensitive) ----
    def apply_fill(self, exec_id, fill_qty, fill_price, fee=0.0):
        """Aggregate a fill into a VWAP. Duplicate exec_ids are no-ops, and a
        terminal order never moves backward."""
        if exec_id in self._seen_exec_ids or self.status in TERMINAL:
            return False
        self._seen_exec_ids.add(exec_id)
        new_filled = self.filled_qty + fill_qty
        if new_filled > 0:
            self.avg_fill_price = (
                self.avg_fill_price * self.filled_qty + fill_price * fill_qty
            ) / new_filled
        self.filled_qty = new_filled
        self.fees += fee
        # forward-only status transition
        if self.filled_qty + 1e-12 >= self.qty:
            self.status = FILLED
        elif self.filled_qty > 0:
            self.status = PARTIAL
        self.updated_at = time.time()
        db.log_order_event(self.id, self.client_order_id, self.product,
                           self.side, "fill", fill_qty, fill_price,
                           self.status, self.meta.get("venue", "shadow"),
                           {"exec_id": exec_id, "vwap": self.avg_fill_price,
                            "fee": fee})
        return True

    def to_dict(self):
        return {"id": self.id, "client_order_id": self.client_order_id,
                "product": self.product, "side": self.side, "qty": self.qty,
                "price": self.price, "status": self.status,
                "exchange_order_id": self.exchange_order_id,
                "filled_qty": round(self.filled_qty, 8),
                "avg_fill_price": round(self.avg_fill_price, 8),
                "fees": round(self.fees, 4), "reason": self.reason,
                "created_at": self.created_at, "updated_at": self.updated_at}


class Venue:
    """Interface a real exchange adapter must implement (CCXT/native SDK)."""
    name = "abstract"

    def place(self, intent: TradeIntent):        # -> list[fill dicts]
        raise NotImplementedError

    def cancel(self, intent: TradeIntent) -> bool:
        raise NotImplementedError

    def fetch_open(self):                          # -> {client_order_id: {...}}
        raise NotImplementedError

    def fetch_fills(self, intent: TradeIntent):    # -> list[fill dicts]
        raise NotImplementedError


class OMS:
    """Manages intents through their lifecycle against a Venue.

    Fails closed: if an intent is UNKNOWN (ambiguous submit/cancel), the OMS
    refuses new orders for that product until reconciliation resolves it.
    """

    def __init__(self, venue: Venue):
        self.venue = venue
        self.intents: dict[str, TradeIntent] = {}      # id -> intent
        self.by_client: dict[str, TradeIntent] = {}
        self._lock = threading.RLock()
        self.reconciliations = 0
        self.drift_events = []                          # reported, not hidden

    # ---- submission ----
    def submit(self, product, side, qty, price, reason="", meta=None):
        with self._lock:
            if self._blocked(product):
                db.log_event("oms", f"BLOCKED new {side} {product}: unresolved "
                                    f"order pending reconciliation (fail-closed)")
                return None
            intent = TradeIntent(product, side, qty, price, reason=reason, meta=meta)
            self.intents[intent.id] = intent
            self.by_client[intent.client_order_id] = intent
            # 1) durable intent BEFORE touching the venue
            db.log_order_event(intent.id, intent.client_order_id, product, side,
                               "intent", qty, price, CREATED, self.venue.name)
            intent.status = SUBMITTING
            try:
                fills = self.venue.place(intent)       # 2) send
                intent.exchange_order_id = intent.meta.get("exchange_order_id") \
                    or intent.client_order_id
                if intent.status == SUBMITTING:
                    intent.status = OPEN
                db.log_order_event(intent.id, intent.client_order_id, product,
                                   side, "ack", qty, price, intent.status,
                                   self.venue.name)
                for fdict in fills or []:
                    intent.apply_fill(fdict["exec_id"], fdict["qty"],
                                      fdict["price"], fdict.get("fee", 0.0))
            except TimeoutError:
                # 3) ambiguous: mark UNKNOWN and fail closed until reconciled
                intent.status = UNKNOWN
                db.log_order_event(intent.id, intent.client_order_id, product,
                                   side, "timeout", qty, price, UNKNOWN,
                                   self.venue.name)
                db.log_event("oms", f"AMBIGUOUS submit {product} — marked UNKNOWN, "
                                    f"reconcile before new orders")
            except Exception as e:                     # clear validation error
                intent.status = REJECTED
                db.log_order_event(intent.id, intent.client_order_id, product,
                                   side, "reject", qty, price, REJECTED,
                                   self.venue.name, {"error": str(e)[:200]})
            return intent

    def cancel(self, intent: TradeIntent):
        with self._lock:
            if intent.status in TERMINAL:
                return True
            try:
                ok = self.venue.cancel(intent)
                if ok:
                    if intent.filled_qty <= 0:
                        intent.status = CANCELED
                    db.log_order_event(intent.id, intent.client_order_id,
                                       intent.product, intent.side, "cancel",
                                       intent.qty, intent.price, intent.status,
                                       self.venue.name)
                    return True
                intent.status = UNKNOWN     # unconfirmed cancel = live exposure
            except Exception:
                intent.status = UNKNOWN
            db.log_event("oms", f"UNCONFIRMED cancel {intent.product} — treated "
                                f"as live exposure (fail-closed)")
            return False

    # ---- reconciliation: venue is the source of truth ----
    def reconcile(self):
        """Compare local intents with the venue; resolve UNKNOWNs; report drift."""
        with self._lock:
            self.reconciliations += 1
            try:
                venue_open = self.venue.fetch_open()
            except Exception as e:
                db.log_event("oms", f"reconcile: venue query failed: {e}")
                return
            for intent in list(self.intents.values()):
                if intent.status in TERMINAL:
                    continue
                try:
                    fills = self.venue.fetch_fills(intent)
                except Exception:
                    fills = []
                for fdict in fills or []:
                    intent.apply_fill(fdict["exec_id"], fdict["qty"],
                                      fdict["price"], fdict.get("fee", 0.0))
                still_open = intent.client_order_id in venue_open
                if intent.status == UNKNOWN:
                    # resolve: filled? still open? or truly never placed?
                    if intent.filled_qty > 0:
                        intent.status = (FILLED if intent.filled_qty + 1e-12
                                         >= intent.qty else PARTIAL)
                    elif still_open:
                        intent.status = OPEN
                    else:
                        intent.status = CANCELED     # never placed / gone
                    db.log_event("oms", f"reconciled {intent.product} "
                                        f"{intent.client_order_id} -> {intent.status}")
                # drift: local thinks open but venue doesn't know it
                if intent.status in (OPEN, PARTIAL) and not still_open \
                        and intent.filled_qty < intent.qty:
                    msg = (f"DRIFT {intent.product} {intent.client_order_id}: "
                           f"local={intent.status} but not on venue")
                    self.drift_events.append({"ts": time.time(), "msg": msg})
                    db.log_event("oms", msg)

    def _blocked(self, product):
        return any(i.product == product and i.status == UNKNOWN
                   for i in self.intents.values())

    # ---- derived views ----
    def open_intents(self):
        return [i for i in self.intents.values() if i.status not in TERMINAL]

    def stats(self):
        by_status = {}
        for i in self.intents.values():
            by_status[i.status] = by_status.get(i.status, 0) + 1
        return {"venue": self.venue.name, "mode": "shadow",
                "n_intents": len(self.intents), "by_status": by_status,
                "reconciliations": self.reconciliations,
                "drift_events": self.drift_events[-10:],
                "unresolved": [i.to_dict() for i in self.intents.values()
                               if i.status == UNKNOWN]}
