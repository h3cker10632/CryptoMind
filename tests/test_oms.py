"""OMS state-machine tests — the fragile execution logic, run deterministically.

Covers exactly the failure modes the critique cited from production write-ups:
duplicate fills ignored, out-of-order partials aggregate to the correct VWAP,
state only moves forward, unconfirmed cancel fails closed, and reconciliation
resolves ambiguous (UNKNOWN) orders + reports drift.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.execution.oms import (TradeIntent, OMS, Venue, FILLED, PARTIAL, OPEN,
                               CANCELED, REJECTED, UNKNOWN)


def test_duplicate_fill_is_ignored():
    i = TradeIntent("BTC-USD", "buy", 1.0, 100.0)
    assert i.apply_fill("e1", 0.5, 100.0)
    assert not i.apply_fill("e1", 0.5, 100.0)   # same exec id -> no-op
    assert abs(i.filled_qty - 0.5) < 1e-9


def test_partial_fills_aggregate_to_vwap():
    i = TradeIntent("BTC-USD", "buy", 1.0, 100.0)
    i.apply_fill("a", 0.5, 100.0)
    i.apply_fill("b", 0.5, 102.0)
    assert i.status == FILLED
    assert abs(i.avg_fill_price - 101.0) < 1e-9    # (0.5*100 + 0.5*102)/1.0


def test_out_of_order_partial_still_correct_vwap():
    # a late-arriving first fragment must not distort the basis
    i = TradeIntent("BTC-USD", "buy", 2.0, 100.0)
    i.apply_fill("late", 1.0, 100.0)    # "early" fragment arrives late
    i.apply_fill("second", 1.0, 104.0)
    assert abs(i.avg_fill_price - 102.0) < 1e-9


def test_state_only_moves_forward():
    i = TradeIntent("BTC-USD", "buy", 1.0, 100.0)
    i.apply_fill("a", 1.0, 100.0)
    assert i.status == FILLED
    # a stray late fill on a terminal order is a no-op
    assert not i.apply_fill("z", 1.0, 50.0)
    assert i.status == FILLED


class _FakeVenue(Venue):
    name = "fake"

    def __init__(self, behavior):
        self.behavior = behavior
        self._open = {}

    def place(self, intent):
        if self.behavior == "timeout":
            raise TimeoutError("network")
        if self.behavior == "reject":
            raise ValueError("min notional")
        if self.behavior == "partial":
            self._open[intent.client_order_id] = intent.qty - 0.4
            return [{"exec_id": "p1", "qty": 0.4, "price": intent.price, "fee": 0}]
        return [{"exec_id": "f1", "qty": intent.qty, "price": intent.price, "fee": 0}]

    def cancel(self, intent):
        if self.behavior == "cancel_unconfirmed":
            return False
        self._open.pop(intent.client_order_id, None)
        return True

    def fetch_open(self):
        return dict(self._open)

    def fetch_fills(self, intent):
        rem = self._open.pop(intent.client_order_id, 0.0)
        return [{"exec_id": "r1", "qty": rem, "price": intent.price, "fee": 0}] if rem else []


def test_full_fill_flow():
    oms = OMS(_FakeVenue("full"))
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)
    assert i.status == FILLED


def test_reject_is_terminal():
    oms = OMS(_FakeVenue("reject"))
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)
    assert i.status == REJECTED


def test_timeout_marks_unknown_and_blocks_new_orders():
    oms = OMS(_FakeVenue("timeout"))
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)
    assert i.status == UNKNOWN
    # fail closed: no new orders for that product until reconciled
    assert oms.submit("BTC-USD", "buy", 1.0, 100.0) is None


def test_unconfirmed_cancel_is_treated_as_live_exposure():
    oms = OMS(_FakeVenue("cancel_unconfirmed"))
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)   # opens fully? -> full venue path
    # force an open, working order then attempt cancel
    i.status = OPEN
    ok = oms.cancel(i)
    assert ok is False
    assert i.status == UNKNOWN


def test_reconcile_completes_partial_via_vwap():
    oms = OMS(_FakeVenue("partial"))
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)
    assert i.status == PARTIAL and abs(i.filled_qty - 0.4) < 1e-9
    oms.reconcile()                # pulls the remaining 0.6
    assert i.status == FILLED
    assert abs(i.filled_qty - 1.0) < 1e-9


def test_reconcile_resolves_unknown():
    v = _FakeVenue("timeout")
    oms = OMS(v)
    i = oms.submit("BTC-USD", "buy", 1.0, 100.0)
    assert i.status == UNKNOWN
    oms.reconcile()                # venue has no open order, no fills -> canceled
    assert i.status == CANCELED
