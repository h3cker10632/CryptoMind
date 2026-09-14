"""Exact decimal money math + per-instrument precision / lot / min-notional.

Live exchanges reject orders that violate tick size (price precision),
lot/step size (quantity precision) or the minimum notional. Floating-point
arithmetic drifts against those constraints and causes reconciliation
mismatches, so all *order-facing* quantities are normalised through here
using :class:`decimal.Decimal`.

The paper broker historically used floats; it keeps working (Decimals cast
back to float at the boundary) but every price/qty that would be sent to a
real venue is first rounded with :func:`round_price` / :func:`round_qty`
and validated with :func:`meets_min_notional`.
"""
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, getcontext

getcontext().prec = 28


# ---- per-product trading rules (sensible Coinbase-like defaults) ----
# tick  = price increment, step = base-size increment, min_notional in quote $.
_DEFAULT = {"tick": "0.01", "step": "0.00000001", "min_notional": "1"}
INSTRUMENTS = {
    "BTC-USD":  {"tick": "0.01",  "step": "0.00000001", "min_notional": "1"},
    "ETH-USD":  {"tick": "0.01",  "step": "0.00000001", "min_notional": "1"},
    "SOL-USD":  {"tick": "0.01",  "step": "0.00000001", "min_notional": "1"},
    "DOGE-USD": {"tick": "0.00001", "step": "0.1",       "min_notional": "1"},
    "LINK-USD": {"tick": "0.001", "step": "0.001",       "min_notional": "1"},
    "AVAX-USD": {"tick": "0.001", "step": "0.001",       "min_notional": "1"},
}


def D(x) -> Decimal:
    """Coerce anything to Decimal safely (via str to avoid float noise)."""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def spec(product: str) -> dict:
    return INSTRUMENTS.get(product, _DEFAULT)


def register_instrument(product: str, tick, step, min_notional) -> None:
    """Called by the live venue adapter once it has fetched real rules."""
    INSTRUMENTS[product] = {"tick": str(tick), "step": str(step),
                            "min_notional": str(min_notional)}


def _quantize(value: Decimal, increment: Decimal, rounding) -> Decimal:
    if increment <= 0:
        return value
    return (value / increment).to_integral_value(rounding=rounding) * increment


def round_price(product: str, price) -> float:
    tick = D(spec(product)["tick"])
    return float(_quantize(D(price), tick, ROUND_HALF_UP))


def round_qty(product: str, qty) -> float:
    step = D(spec(product)["step"])
    return float(_quantize(D(qty), step, ROUND_DOWN))   # never round UP size


def min_notional(product: str) -> float:
    return float(D(spec(product)["min_notional"]))


def meets_min_notional(product: str, qty, price) -> bool:
    return D(qty) * D(price) >= D(spec(product)["min_notional"])
