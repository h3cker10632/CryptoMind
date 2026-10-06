"""Point-in-time investable universe (no survivorship bias).

`liquid_mask(panel, n)` -> (T, N) bool: on day t, the `n` coins with the
highest average daily dollar volume over the previous `lookback` days, among
coins with >= `min_history` days of history, trading today, above a dollar
volume floor, and not stablecoins / wrapped duplicates. Uses only data up to
day t, and delisted coins are in it for as long as they traded.
"""
from __future__ import annotations
import numpy as np

from . import features as F


def liquid_mask(panel, n=20, lookback=30, min_history=100, min_dollar_volume=1e6):
    from ..data.store import is_tradeable_asset
    dv = F.rolling_mean(np.where(np.isnan(panel.close), np.nan, panel.close * panel.volume),
                        lookback)
    ok = (~np.isnan(panel.close)) & (F.history_count(panel.close) >= min_history)
    ok &= np.nan_to_num(dv) >= min_dollar_volume
    ok &= np.array([is_tradeable_asset(c) for c in panel.coins])[None, :]
    score = np.where(ok, np.nan_to_num(dv), -np.inf)
    mask = np.zeros_like(ok)
    k = min(n, panel.N)
    if k <= 0:
        return mask
    top = np.argpartition(-score, k - 1, axis=1)[:, :k]
    rows = np.arange(panel.T)[:, None]
    mask[rows, top] = True
    return mask & ok
