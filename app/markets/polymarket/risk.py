"""Position sizing + entry gates for Polymarket binary bets.

The crypto book sizes off ATR and a stop distance; a prediction market has no ATR
and a *known, bounded* payoff, so the right sizing primitive here is fractional
Kelly on the estimated edge:

    buy one share at price p that pays 1 (prob q) or 0 (prob 1-q)
        Kelly fraction  f* = (q - p) / (1 - p)          # for a YES buy

We stake `equity * kelly_fraction * f* * confidence_scale`, then clamp by a
per-market cap, available cash, and the venue's $5 minimum order. `kelly_fraction`
defaults to 0.25 (quarter-Kelly) — Kelly is famously over-aggressive on estimated
(not true) probabilities, and our probabilities are estimates, so we stay well
below full Kelly.

The cost-viability gate is the direct analogue of the crypto engine's gate that
we diagnosed earlier: refuse a bet whose edge doesn't clear its round-trip
friction by a safety multiple. Here friction is the bid/ask spread plus slippage
(Polymarket fees are currently zero), measured in probability points.
"""
from __future__ import annotations

from ...tunables import tv


def size(equity: float, cash: float, price: float, edge: float,
         confidence: float, market: dict):
    """Return (stake_usdc, reason). stake 0 with a reason means 'skipped'."""
    if edge <= 0:
        return 0.0, "no edge"
    min_edge = tv("pm_min_edge")
    if edge < min_edge:
        return 0.0, f"edge {edge:.3f} below min {min_edge:.3f}"

    conf_gate = tv("pm_confidence_gate")
    if confidence < conf_gate:
        return 0.0, f"confidence {confidence:.2f} below gate {conf_gate:.2f}"

    # cost-viability: edge must clear round-trip spread+slippage by a multiple
    spread = max(0.0, float(market.get("spread", 0.0)))
    slip = tv("pm_slippage")
    round_trip = spread + 2.0 * slip
    need = round_trip * tv("pm_cost_multiple")
    if edge < need:
        return 0.0, (f"edge {edge:.3f} below cost floor {need:.3f} "
                     f"(spread {spread:.3f} + slippage)")

    p = min(0.999, max(0.001, price))
    q = min(0.999, max(0.001, price + edge))
    kelly = (q - p) / (1.0 - p)
    if kelly <= 0:
        return 0.0, "kelly <= 0"
    conf_scale = 0.5 + confidence / 2.0
    stake = equity * tv("pm_kelly_fraction") * kelly * conf_scale

    cap = equity * tv("pm_max_position_pct")
    stake = min(stake, cap, cash)

    min_order = max(float(market.get("min_order_size", 5.0)), tv("pm_min_notional"))
    if stake < min_order:
        return 0.0, f"stake ${stake:.2f} below min ${min_order:.2f}"
    return round(stake, 2), "ok"


def exit_levels(price: float):
    """Absolute take-profit / stop-loss price levels for a freshly bought share.

    Expressed as moves in the token's own probability price. Held-to-resolution
    is always available; these are just the optional EARLY exits.
    """
    tp = min(0.999, price + tv("pm_take_profit"))
    sl = max(0.001, price - tv("pm_stop_loss"))
    return sl, tp
