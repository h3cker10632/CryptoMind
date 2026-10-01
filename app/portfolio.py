"""Shared paper-capital ledger used by every paper-trading sleeve (crypto,
memes, Polymarket). This is the sole source of simulated cash: sleeves keep
their own position/fill math but reserve/settle against this one account so
they can never independently overspend the same buying power.

No in-memory cash copy is kept here — `cash` always reads the durable SQLite
ledger (app.db), so concurrent sleeves/processes observe the same balance.
"""
from __future__ import annotations

from . import db

VALID_SLEEVES = {"crypto", "polymarket"}


class PaperPortfolio:
    @property
    def cash(self) -> float:
        account = db.paper_portfolio_account()
        return account["cash"] if account else 0.0

    @property
    def ready(self) -> bool:
        """True once the one-time migration has created the shared account."""
        return db.paper_portfolio_account() is not None

    def _check_sleeve(self, sleeve: str):
        if sleeve not in VALID_SLEEVES:
            raise ValueError(f"unknown sleeve: {sleeve!r}")

    def reserve(self, sleeve: str, event_id: str, amount: float,
                reference: str = "") -> bool:
        """Debit `amount` once, keyed by `event_id`. False = not applied."""
        self._check_sleeve(sleeve)
        if not event_id:
            raise ValueError("event_id is required")
        return db.reserve_paper_cash(event_id, sleeve, amount, reference)

    def apply(self, sleeve: str, event_id: str, delta: float,
              reference: str = "") -> bool:
        """Apply one signed settlement/fee/funding delta once, keyed by
        `event_id`. False = not applied (duplicate or missing account)."""
        self._check_sleeve(sleeve)
        if not event_id:
            raise ValueError("event_id is required")
        return db.apply_paper_cash_event(event_id, sleeve, delta, reference)

    def reset_account(self, opening_cash: float, event_id: str,
                       reference: str = "account reset") -> bool:
        """Explicit whole-account reset. The caller MUST verify every paper
        sleeve (crypto, Polymarket) is flat before calling this -- it only
        resets shared cash, never positions."""
        if not event_id:
            raise ValueError("event_id is required")
        return db.reset_paper_portfolio_cash(opening_cash, event_id, reference)

    def total_equity(self, crypto_market, pm_price_lookup) -> float:
        """Shared cash plus every sleeve's marked position value, counted
        exactly once. The shadow OMS mirror account is intentionally excluded
        -- it tracks execution-quality divergence, not real paper capital."""
        from .execution.paper import broker as crypto_broker
        from .markets.polymarket.broker import broker as pm_broker
        eq = self.cash
        for product, pos in crypto_broker.positions.items():
            price = crypto_market.price(product) or pos["entry"]
            eq += crypto_broker.position_value(pos, price)
        for token_id, pos in pm_broker.positions.items():
            mid = pm_price_lookup(token_id)
            if mid is None:
                mid = pos["entry"]
            eq += pm_broker.market_value(pos, mid)
        return eq

    def total_exposure(self, crypto_market, pm_price_lookup) -> float:
        """Combined committed notional across both sleeves. Shadow excluded."""
        from .execution.paper import broker as crypto_broker
        from .markets.polymarket.broker import broker as pm_broker
        return (crypto_broker.exposure(crypto_market)
                + pm_broker.exposure(pm_price_lookup))


def bootstrap(opening_cash: float, migration_id: str = "unified-paper-v1",
              legacy_pm_confirmed: bool = False) -> bool:
    """One-time cutover from the legacy crypto-only paper cash to the shared
    ledger. Carries `opening_cash` forward exactly once; a later call (e.g. a
    restart) is always a no-op and can never re-fund the account.

    Because the legacy Polymarket broker never persisted its positions, a
    missing snapshot is not proof the old book was flat. The shared account
    stays unavailable (``portfolio.ready`` False, so no sleeve can open a new
    position) until the operator explicitly passes `legacy_pm_confirmed=True`
    — see ``POST /api/portfolio/migration/confirm``.
    """
    if db.paper_portfolio_account() is not None:
        return False
    if not legacy_pm_confirmed:
        return False
    return db.initialize_paper_portfolio(opening_cash, migration_id)


portfolio = PaperPortfolio()
