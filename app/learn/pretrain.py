"""Warm-start the online model on downloaded history.

The online committee otherwise starts from nothing and sees one sample per
coin per hour (~500/day), so it needs months before its vote can be trusted.
This replays the hourly history the replay already downloads, builds the SAME
input vector live trading uses (`online_model.build_x`) from each coin's
features at each bar, labels it with the 24h-ahead return (the live label
horizon), and trains the committee in CHRONOLOGICAL order exactly like live:
a sample is predicted when it occurs but only TRAINED ON once its 24h label
has fully matured (a pending queue). Training on it immediately would leak
the future — the next sample's prediction would already have seen prices
from inside its own label window — and inflate accuracy.

Only price-derived inputs exist historically; the live-only inputs
(sentiment, order book, chart patterns) are masked in build_x anyway, and the
derivatives / advisor-lean inputs are simply 0 in history and live where
unavailable.
"""
from __future__ import annotations
import math

MAX_SAMPLES = 40_000          # ~2.5 min of pure-Python training, once


def pretrain_online(candles_by_product, horizon_sec=None, max_samples=MAX_SAMPLES,
                    committee=None, model=None):
    """Train the committee on history. Returns the number of samples used."""
    from ..backtest.replay import _clean, _HistMarket, MIN_BARS
    from ..config import CANDLE_HISTORY
    from .loop import ML_HORIZON_SEC
    from . import online_model as om
    committee = committee or om.committee
    model = model or om.model
    horizon = int((horizon_sec or ML_HORIZON_SEC) // 3600)
    C = _clean(candles_by_product)
    if len(C) < 3:
        return 0
    ts_all = sorted({int(r[0]) for rows in C.values() for r in rows})
    pos_of = {p: {int(r[0]): i for i, r in enumerate(rows)} for p, rows in C.items()}
    usable = ts_all[MIN_BARS: max(MIN_BARS, len(ts_all) - horizon)]
    if not usable:
        return 0
    per_bar = max(1, len(C))
    step = max(1, math.ceil(len(usable) * per_bar / max_samples))
    from collections import deque
    mkt = _HistMarket(C, pos_of, CANDLE_HISTORY)
    pending = deque()            # (label_ready_ts, x, fwd, pred)
    n = 0

    def _train_matured(now):
        nonlocal n
        while pending and pending[0][0] <= now and n < max_samples:
            ready, x, fwd, pred = pending.popleft()
            committee.observe_outcome(x, fwd)
            committee.update(x, fwd, pred_at_record=pred,
                             cluster=int(ready // (horizon * 3600)))
            n += 1

    for t in usable[::step]:
        _train_matured(t)
        if n >= max_samples:
            return n
        mkt.set_time(t)
        for p in sorted(mkt.candles):
            i = pos_of[p].get(t)
            if i is None or i + horizon >= len(C[p]):
                continue
            f = mkt.features(p)
            if not f:
                continue
            p0, p1 = C[p][i][4], C[p][i + horizon][4]
            if not (p0 and p1):
                continue
            fwd = p1 / p0 - 1
            x = om.build_x(f, 0.0, 0.0, None)
            pred = model.predict(x) if model.n_updates >= 10 else 0.0
            pending.append((t + horizon * 3600, x, fwd, pred))
    _train_matured(float("inf"))       # the tail: everything has matured by now
    return n
