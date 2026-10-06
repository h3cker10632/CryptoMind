"""Single source of truth for what a trade costs.

Entries can be simulated as post-only LIMIT (maker) orders or MARKET (taker)
orders (setting `entry_order_type`). Exits that must happen NOW — stops,
trailing stops, signal flips, loss-cuts — are always taker. A take-profit can
rest on the book as a limit, so it is maker when entries are maker.

The risk manager's cost-viability gate, the learner's net-of-cost scoring and
the replay all read these helpers, so the system never disagrees with itself
about whether a trade can pay for itself.
"""
from ..tunables import tv


def maker_entries():
    try:
        from .. import settings
        return settings.get("entry_order_type") == "maker"
    except Exception:
        return False


def taker_side_cost():
    """Fee + slippage for one market-order fill (fraction of notional)."""
    return tv("fee_rate") + tv("slippage_bps") / 1e4


def maker_side_cost():
    """Fee for one resting limit fill (no slippage: you set the price)."""
    return tv("maker_fee_rate")


def entry_side_cost():
    return maker_side_cost() if maker_entries() else taker_side_cost()


def round_trip_cost():
    """Expected round trip for gating/learning: entry at its order type, exit
    assumed TAKER (the conservative case — most exits are stops)."""
    return entry_side_cost() + taker_side_cost()
