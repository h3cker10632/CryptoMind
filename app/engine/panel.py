"""Aligned bars x coins panel (numpy) from candles — daily by default; any
bar size works (`bar_sec`), e.g. hourly for the exploration sleeve. `days`
then holds bar indices (ts // bar_sec) and every window is in bars."""
from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np

DAY = 86400


@dataclass
class Panel:
    days: np.ndarray          # (T,) day index (unix_ts // 86400)
    coins: list               # (N,) product ids
    close: np.ndarray         # (T, N) float, NaN where the coin has no bar
    volume: np.ndarray        # (T, N)
    data_version: str = ""
    high: np.ndarray = None   # (T, N) daily high / low (cost model); optional
    low: np.ndarray = None
    bar_sec: int = DAY        # bar size; `days` holds ts // bar_sec

    @property
    def T(self):
        return len(self.days)

    @property
    def N(self):
        return len(self.coins)

    def returns(self):
        """(T, N) simple return from day t-1 to t; NaN if either bar missing."""
        r = np.full_like(self.close, np.nan)
        r[1:] = self.close[1:] / self.close[:-1] - 1
        return r

    def subset(self, coins):
        idx = [self.coins.index(c) for c in coins if c in self.coins]
        sub = (lambda a: None if a is None else a[:, idx])
        return Panel(self.days, [self.coins[i] for i in idx], self.close[:, idx],
                     self.volume[:, idx], self.data_version, sub(self.high), sub(self.low),
                     self.bar_sec)


def from_candles(candles_by_product, now=None, data_version="", bar_sec=DAY):
    """Panel over every bar from the first to the last CLOSED bar."""
    now = now or time.time()
    rows = {}
    for p, cs in candles_by_product.items():
        d = {}
        for r in cs or []:
            t, c, v = int(r[0]), float(r[4]), float(r[5])
            if c > 0 and t + bar_sec <= now:
                d[t // bar_sec] = (c, v, float(r[2]), float(r[1]))
        if d:
            rows[p] = d
    if not rows:
        return Panel(np.zeros(0, int), [], np.zeros((0, 0)), np.zeros((0, 0)), data_version,
                     bar_sec=bar_sec)
    lo = min(min(d) for d in rows.values())
    hi = max(max(d) for d in rows.values())
    days = np.arange(lo, hi + 1)
    coins = sorted(rows)
    close = np.full((len(days), len(coins)), np.nan)
    vol, high, low = (np.full_like(close, np.nan) for _ in range(3))
    for j, p in enumerate(coins):
        for k, (c, v, h, l) in rows[p].items():
            close[k - lo, j], vol[k - lo, j] = c, v
            high[k - lo, j], low[k - lo, j] = h, l
    return Panel(days, coins, close, vol, data_version, high, low, bar_sec)


def from_store(products=None, start=None, end=None, as_of=None, bar_sec=DAY):
    """Panel straight from the versioned data store (daily unless `bar_sec`).
    Trailing bars only some coins have reached are dropped (< 90% of the
    best coverage of the last 30 bars): the core and exploration loops ingest
    their own coins' newest bars hourly, the full sync every coin's once a
    day, and on such a partial row the missing coins look delisted."""
    from ..data import store
    data, ver = store.load_candles(bar_sec, products, start=start, end=end, as_of=as_of)
    p = from_candles(data, data_version=ver, bar_sec=bar_sec)
    n = np.isfinite(p.close).sum(axis=1)
    k = p.T
    while k and n[k - 1] < 0.9 * n[-30:].max():
        k -= 1
    if k < p.T:
        cut = (lambda a: None if a is None else a[:k])
        p = Panel(p.days[:k], p.coins, p.close[:k], p.volume[:k], ver,
                  cut(p.high), cut(p.low), bar_sec)
    return p
