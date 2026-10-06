"""Funding-rate carry research (delta-neutral: long spot + short perpetual).

While a perp's funding rate is positive, shorts are paid by longs every hour.
Holding spot long + perp short has ~no price exposure and earns the funding.
This module scores that as a strategy on the stored funding history:

  * daily carry return for a coin = sum of that day's hourly funding rates;
  * a coin is held while its trailing `lookback`-day average daily funding is
    above the cost hurdle `min_daily` (else flat), equal-weight across held
    coins, at most `max_coins`;
  * entering or exiting pays costs on BOTH legs: spot `spot_cost` + perp
    `perp_cost` per side, per unit of capital traded;
  * capital is split between the two legs (spot bought in full, perp margin
    posted), so carry is earned on half the capital: `capital_efficiency`.

Not modelled (conservative caveats): basis moves between spot and perp while
held (small for delta-neutral, but not zero), liquidation risk on the perp
leg in a sharp rally, and exchange counterparty risk. RESEARCH ONLY: the bot
trades Coinbase spot and has no perp venue; Binance/Bybit are blocked here.
"""
from __future__ import annotations
import math
from collections import defaultdict

import numpy as np


def daily_funding(rows):
    """{day: summed hourly funding} from [(ts, rate_1h)]."""
    d = defaultdict(float)
    for ts, r in rows:
        d[int(ts) // 86400] += float(r)
    return dict(d)


def carry_backtest(funding_by_coin, start_day=None, lookback=7, min_daily=None,
                   max_coins=5, spot_cost=0.006, perp_cost=0.0005, capital_efficiency=0.5):
    """(days, daily net returns, avg coins held) for the carry sleeve."""
    daily = {c: daily_funding(r) for c, r in funding_by_coin.items()}
    days = sorted({d for f in daily.values() for d in f})
    if start_day is not None:
        days = [d for d in days if d >= start_day - lookback]
    side_cost = spot_cost + perp_cost                    # one side, both legs
    if min_daily is None:
        # break even on a round trip within ~30 days of holding
        min_daily = 2 * side_cost / 30 / capital_efficiency
    held = set()
    out_days, rets, n_held = [], [], []
    for i, d in enumerate(days):
        if i < lookback or (start_day is not None and d < start_day):
            continue
        avg = {c: np.mean([f.get(days[j], 0.0) for j in range(i - lookback, i)])
               for c, f in daily.items()}
        want = sorted((c for c, a in avg.items() if a > min_daily), key=lambda c: -avg[c])
        new = set(want[:max_coins])
        w_old = 1 / max(1, len(held)) if held else 0.0
        w_new = 1 / max(1, len(new)) if new else 0.0
        turnover = sum(abs((w_new if c in new else 0) - (w_old if c in held else 0))
                       for c in held | new)
        cost = turnover * side_cost
        earn = sum(w_new * daily[c].get(d, 0.0) for c in new) * capital_efficiency
        rets.append(earn - cost)
        out_days.append(d)
        n_held.append(len(new))
        held = new
    return np.array(out_days), np.array(rets), float(np.mean(n_held)) if n_held else 0.0
