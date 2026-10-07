"""Main-account budgets: every sleeve spends only its own share of the shared
cash.

  core         `core_allocation_pct`        proven strategy (app/strategies/core.py)
  exploration  `exploration_allocation_pct` fast strategies (exploration.py)
  remainder    stays as unallocated cash

Polymarket is not in here: it runs its own bankroll (`pm_start_cash`).
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
