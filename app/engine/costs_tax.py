"""What the champion keeps after trading costs and taxes — next to simply
holding the same coins.

A trend rule switching in and out every couple of months turns nearly every
gain into a SHORT-term gain (in the US taxed like income), while holding
defers tax and gets the long-term rate. Fees compound the same way: ~5x
turnover a year at 0.6% per side is ~3%/yr. Both can decide whether the rule
beats holding, so the research report prices them:

  scenarios()      the champion's weights under several cost settings
                   (exchange taker / maker / low-fee venue / spot ETF with
                   next-weekday execution and a fund fee), pre- and after-tax,
                   plus after-tax buy-and-hold of the same coins.
  simulate_taxed() the drift simulation (backtest.simulate_drift) with FIFO
                   tax lots, short/long-term rates, yearly netting and loss
                   carry-forward. With zero tax rates it reproduces
                   simulate_drift exactly (tests/test_research_evidence.py).

Simplifications (an estimate, NOT tax advice): taxes are paid on Dec 31 from
the portfolio's cash; losses offset gains within the year (short against
long), unused losses carry forward without the $3k ordinary offset; an ETF
is modelled as the coin's own price (no premium / tracking error), US market
holidays are ignored.
"""
from __future__ import annotations
import datetime as dt

import numpy as np

from . import backtest as B

ETF_COST_SIDE = 0.0005           # ~spread + slippage on a liquid spot ETF
ETF_EXPENSE = 0.0025             # yearly fund fee
SCENARIO_COSTS = (("exchange taker (current)", None), ("exchange maker 0.4%", 0.004),
                  ("low-fee venue 0.1%", 0.001))


def _weekday(day):
    return (int(day) + 3) % 7               # day 0 = 1970-01-01, a Thursday; Mon = 0


def etf_execution(W, days):
    """Weights as an ETF account would hold them: a decision at day t's close
    is filled at the close of the next US weekday after t."""
    W = np.asarray(W, dtype=float)
    out = np.zeros_like(W)
    cur = np.zeros(W.shape[1])
    pending = []                             # (exec_day_index, weights)
    T = len(days)
    for t in range(T):
        while pending and pending[0][0] <= t:
            cur = pending.pop(0)[1]
        out[t] = cur
        e = t + 1
        while e < T and _weekday(days[e]) > 4:
            e += 1
        pending = [p for p in pending if p[0] < e]   # a newer decision replaces queued ones
        pending.append((e, W[t].copy()))
    return out


def _year(day):
    return (dt.date(1970, 1, 1) + dt.timedelta(days=int(day))).year


def _tax(st, lt, carry, st_rate, lt_rate):
    """US-style netting for one year. Returns (tax, new carry-forward loss)."""
    st -= carry                               # carried losses hit short-term first
    if st < 0 and lt > 0:
        lt, st = lt + st, 0.0
    elif lt < 0 and st > 0:
        st, lt = st + lt, 0.0
    if st < 0 and lt <= 0:
        return 0.0, -(st + lt)
    if lt < 0 and st <= 0:
        return 0.0, -(st + lt)
    return st_rate * max(st, 0.0) + lt_rate * max(lt, 0.0), 0.0


def simulate_taxed(W, close, days, cost_side=0.006, start=0, band_rel=0.0,
                   st_rate=0.0, lt_rate=0.0, liquidate=True):
    """Drift simulation with tax lots. Returns
    {"rets": daily after-tax returns (t = start .. T-2), "taxes": total paid
    (fraction of the starting 1.0), "final": ending equity, "liquidation":
    ending equity after selling everything and paying that year's tax}."""
    W = np.nan_to_num(np.asarray(W, dtype=float))
    close = np.asarray(close, dtype=float)
    T, N = W.shape
    px = close.copy()                         # forward-filled valuation prices
    for t in range(1, T):
        m = np.isnan(px[t])
        px[t, m] = px[t - 1, m]
    units = np.zeros(N)
    lots = [[] for _ in range(N)]             # [units, basis_per_unit, buy_day]
    cash, taxes, carry = 1.0, 0.0, 0.0
    st = lt = 0.0
    rets = []

    def sell(j, u, p, t):
        nonlocal st, lt
        proceeds = p * (1 - cost_side)
        while u > 1e-15 and lots[j]:
            lot = lots[j][0]
            take = min(u, lot[0])
            g = take * (proceeds - lot[1])
            if days[t] - lot[2] >= 365:
                lt += g
            else:
                st += g
            lot[0] -= take
            u -= take
            if lot[0] <= 1e-15:
                lots[j].pop(0)

    for t in range(start, T - 1):
        p = px[t]
        ok = np.isfinite(p) & (p > 0)
        hold_val = np.where(ok, units * np.where(ok, p, 0.0), 0.0)
        eq = cash + hold_val.sum()
        h = hold_val / eq if eq > 0 else np.zeros(N)
        tgt = W[t]
        diff = tgt - h
        band = band_rel * tgt
        trade = ((np.abs(diff) > band) | ((tgt == 0) & (h != 0)) | ((tgt > 0) & (h == 0))) & ok
        delta = np.where(trade, diff, 0.0)
        for j in np.flatnonzero(delta):
            du = delta[j] * eq / p[j]
            if du > 0:
                lots[j].append([du, p[j] * (1 + cost_side), days[t]])
                units[j] += du
            else:
                sell(j, -du, p[j], t)
                units[j] += du
            cash -= delta[j] * eq + cost_side * abs(delta[j]) * eq
        if _year(days[t + 1]) != _year(days[t]):          # Dec 31: pay the year's tax
            tx, carry = _tax(st, lt, carry, st_rate, lt_rate)
            cash -= tx
            taxes += tx
            st = lt = 0.0
        p1 = px[t + 1]
        ok1 = np.isfinite(p1) & (p1 > 0)
        eq1 = cash + np.where(ok1, units * np.where(ok1, p1, 0.0), 0.0).sum()
        rets.append(eq1 / eq - 1 if eq > 0 else 0.0)
    final = cash + np.nansum(units * px[T - 1])
    liq = final
    if liquidate:
        for j in np.flatnonzero(units > 0):
            sell(j, units[j], px[T - 1, j], T - 1)
            liq -= cost_side * units[j] * px[T - 1, j]
        tx, _ = _tax(st, lt, carry, st_rate, lt_rate)
        liq -= tx
    return {"rets": np.array(rets), "taxes": taxes, "final": final, "liquidation": liq}


def _cagr(total, n_days):
    yrs = n_days / 365
    return round((total ** (1 / yrs) - 1) * 100, 1) if total > 0 and yrs > 0 else None


def _summ(rets, extra=None):
    st = B.stats(rets)
    out = {"cagr_pct": _cagr(float(np.prod(1 + np.asarray(rets))), len(rets)),
           "sharpe": st["sharpe"], "max_drawdown_pct": st["max_drawdown_pct"]}
    out.update(extra or {})
    return out


def scenarios(W, panel, start=0, band_rel=0.2, base_cost=0.006, st_rate=None, lt_rate=None):
    """The champion's weights `W` under each cost setting, pre- and after-tax,
    and after-tax buy-and-hold of the coins it trades."""
    if st_rate is None or lt_rate is None:
        try:
            from .. import settings
            st_rate = float(settings.get("tax_short_term_rate"))
            lt_rate = float(settings.get("tax_long_term_rate"))
        except Exception:
            st_rate, lt_rate = 0.28, 0.19
    W = np.nan_to_num(np.asarray(W, dtype=float))
    R = panel.returns()
    used = np.flatnonzero(W[start:].sum(axis=0) > 0)
    n_days = panel.T - 1 - start
    out = {"assumptions": {"tax_short_term_rate": st_rate, "tax_long_term_rate": lt_rate,
                           "etf_cost_side": ETF_COST_SIDE, "etf_expense_ratio": ETF_EXPENSE,
                           "note": "estimate, not tax advice"},
           "scenarios": {}}
    sc = out["scenarios"]
    for label, c in SCENARIO_COSTS:
        cost = base_cost if c is None else c
        r, _, _ = B.simulate_drift(W, R, cost, start=start, band_rel=band_rel)
        sc[f"{label} pre-tax"] = _summ(r)
    We = etf_execution(W, panel.days)
    r, inv, _ = B.simulate_drift(We, R, ETF_COST_SIDE, start=start, band_rel=band_rel)
    sc["spot ETF pre-tax"] = _summ(r - ETF_EXPENSE / 365 * inv)
    tx = simulate_taxed(W, panel.close, panel.days, base_cost, start, band_rel, st_rate, lt_rate)
    sc["exchange taker after-tax"] = _summ(tx["rets"], {
        "taxes_paid_pct_of_start": round(tx["taxes"] * 100, 1),
        "liquidation_cagr_pct": _cagr(tx["liquidation"], n_days)})
    if len(used):
        Wh = np.zeros_like(W)
        Wh[start:, used] = 1.0 / len(used)
        hold = simulate_taxed(Wh, panel.close, panel.days, base_cost, start, 1e9,
                              st_rate, lt_rate)
        sc["hold same coins after-tax"] = _summ(hold["rets"], {
            "taxes_paid_pct_of_start": round(hold["taxes"] * 100, 1),
            "liquidation_cagr_pct": _cagr(hold["liquidation"], n_days)})
    return out
