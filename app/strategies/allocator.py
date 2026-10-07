"""Account budgets: every sleeve spends only its own share of the shared cash.

  core         `core_allocation_pct`        proven strategy (app/strategies/core.py)
  exploration  `exploration_allocation_pct` fast strategies (exploration.py)
  polymarket   `polymarket_allocation_pct`, at most what the other two leave
               (it used to get the whole remainder — with the defaults, core
               0% and exploration off, that was the ENTIRE account for an
               untested sleeve)

Without this, Polymarket sized bets off the WHOLE account and could spend any
cash in the shared pool — including cash an exploration strategy holds while
it waits to re-enter.
"""
from __future__ import annotations


def _pct(key, default):
    from .. import settings
    try:
        return max(0.0, min(100.0, float(settings.get(key)))) / 100.0
    except Exception:
        return default


def exploration_pct():
    from .. import settings
    try:
        if not settings.get("exploration_enabled"):
            return 0.0
    except Exception:
        return 0.0
    return _pct("exploration_allocation_pct", 0.0)


def core_pct():
    return _pct("core_allocation_pct", 0.0)


def polymarket_pct():
    left = max(0.0, 1.0 - core_pct() - exploration_pct())
    return min(left, _pct("polymarket_allocation_pct", 0.10))


def polymarket_budget(account_equity, pm_exposure, cash):
    """(equity to size bets against, cash Polymarket may spend now)."""
    budget = account_equity * polymarket_pct()
    return budget, max(0.0, min(cash, budget - pm_exposure))
