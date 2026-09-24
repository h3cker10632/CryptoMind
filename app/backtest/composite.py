"""Composite backtester — validates the REAL system, not 3 toy rules.

The critique's key strategy finding: the old backtester simulated only three
hand-coded long-only rules, while the live system trades an 8-strategy
ensemble combined by weights, with risk sizing, cost gates, stops/targets and
(optionally) shorts. This backtester reconstructs per-bar technical features
exactly like `market.features()` and drives the *actual* strategy functions
from `signals.engine`, combined with the live weighting/confidence logic and
risk-manager sizing — so what you validate is (much closer to) what you ship.

It also reports:
  * a BTC (and equal-weight) buy-and-hold BENCHMARK over the same window
  * walk-forward across rolling windows (positive-OOS fraction, Sharpe decay)
  * Deflated Sharpe Ratio + Probability of Backtest Overfitting

Data-only strategies that need live feeds (`derivatives`, `ml`, sentiment
docs, order-book `microstructure`) are neutral in backtest — noted in the
report so results aren't over-claimed.
"""
from __future__ import annotations
import math
import statistics
from ..tunables import tv
from ..signals.engine import strat_trend, strat_meanrev, strat_breakout, _clip
from ..data.features import features_from_ohlcv
from . import stats as st
from .engine import fetch_history

# strategies that are computable purely from OHLCV history
HIST_STRATEGIES = {"trend": strat_trend, "meanrev": strat_meanrev,
                   "breakout": strat_breakout}


def _ema(xs, n):
    k = 2 / (n + 1)
    e = xs[-n]
    for x in xs[-n + 1:]:
        e = x * k + e * (1 - k)
    return e


def _features_at(candles, i):
    """The same OHLCV-derivable features market.features() produces, at
    historical bar i (no look-ahead: uses candles[:i+1]).

    Delegates to the SHARED core (app/data/features.features_from_ohlcv) so the
    backtest and the live feed can never drift apart — train/live parity is
    enforced by construction, not by keeping two hand-written copies in sync.
    The live-only extras (atr_swing, mtf_*, book imbalance/spread) are absent
    here on purpose: they need live feeds and are neutral in backtest.
    """
    cs = candles[:i + 1]
    if len(cs) < 60:
        return None
    closes = [c[4] for c in cs]
    highs = [c[2] for c in cs]
    lows = [c[1] for c in cs]
    vols = [c[5] for c in cs]
    return features_from_ohlcv(closes, highs, lows, vols)


def _regime(f):
    trend = "bull" if f["sma20"] > f["sma50"] * 1.002 else \
            "bear" if f["sma20"] < f["sma50"] * 0.998 else "sideways"
    vol_state = "high-vol" if f["volatility"] > 0.004 else "normal"
    return {"label": f"{trend}/{vol_state}", "trend": trend,
            "vol_state": vol_state, "vol": f["volatility"]}


def run_composite(candles, weights=None, allow_shorts=True, start_cash=10_000.0,
                  _use_fast_features=True):
    """Event-driven sim of the composite ensemble with risk sizing, cost gate,
    ATR stops/targets/trailing — long and (optionally) short.

    `_use_fast_features` (default True) precomputes all per-bar features in one
    pass (Numba kernel when available, pure-Python fallback) instead of the old
    O(N^2) per-bar `candles[:i+1]` recompute. Output is verified bit-identical
    to the slow path; the flag exists so tests can compare the two.
    """
    weights = weights or {k: 1.0 for k in HIST_STRATEGIES}
    fee, slip = tv("fee_rate"), tv("slippage_bps") / 1e4
    cost_mult, gate = tv("cost_multiple"), tv("min_confidence")
    stop_m, take_m, trail_m = tv("stop_atr_mult"), tv("take_profit_atr_mult"), tv("trail_atr_mult")

    fast_feats = None
    if _use_fast_features:
        try:
            from .fast_features import precompute
            fast_feats = precompute(candles)
        except Exception:
            fast_feats = None

    cash = start_cash
    side = 0; qty = 0.0; entry = stop = take = water = 0.0; margin = 0.0
    equity_curve, trades = [], []
    bars_total = bars_in_market = 0        # exposure / time-in-market
    # ---- EXACT cost accounting (dollars actually charged, not estimated) ----
    # Accumulated at every execution so the report can decompose net edge into
    # the fee and slippage drag it hides, plus turnover. Idea borrowed from
    # qanat's net-edge headline; numbers are measured on THIS run's real fills.
    fees_paid = 0.0            # $ paid in fees across all fills
    slippage_paid = 0.0        # $ lost to slippage (adverse fill vs reference px)
    traded_notional = 0.0      # $ transacted (entries + exits) → turnover

    for i in range(60, len(candles)):
        ts, lo, hi, op, cl, vol = candles[i]
        bars_total += 1
        if side != 0:
            bars_in_market += 1
        f = fast_feats[i] if fast_feats is not None else _features_at(candles, i)
        if not f:
            equity_curve.append((ts, cash)); continue
        atr = f["atr"]; regime = _regime(f)

        # ---- manage open position (intra-bar stop/target + trailing) ----
        if side != 0 and atr > 0:
            if side > 0:
                water = max(water, hi)
                stop = max(stop, water - trail_m * atr)
                if lo <= stop:
                    px = stop * (1 - slip); cash += qty * px * (1 - fee)
                    fees_paid += qty * px * fee; slippage_paid += qty * stop * slip
                    traded_notional += qty * px
                    trades.append(px / entry - 1); side = 0; qty = 0.0
                elif hi >= take:
                    px = take * (1 - slip); cash += qty * px * (1 - fee)
                    fees_paid += qty * px * fee; slippage_paid += qty * take * slip
                    traded_notional += qty * px
                    trades.append(px / entry - 1); side = 0; qty = 0.0
            else:
                water = min(water, lo)
                stop = min(stop, water + trail_m * atr)
                if hi >= stop:
                    px = stop * (1 + slip)
                    cash += margin + qty * (entry - px) - qty * px * fee
                    fees_paid += qty * px * fee; slippage_paid += qty * stop * slip
                    traded_notional += qty * px
                    trades.append((entry - px) / entry); side = 0; qty = 0.0; margin = 0.0
                elif lo <= take:
                    px = take * (1 + slip)
                    cash += margin + qty * (entry - px) - qty * px * fee
                    fees_paid += qty * px * fee; slippage_paid += qty * take * slip
                    traded_notional += qty * px
                    trades.append((entry - px) / entry); side = 0; qty = 0.0; margin = 0.0

        # ---- composite signal from the REAL strategy functions ----
        raw = {}
        for name, fn in HIST_STRATEGIES.items():
            try:
                raw[name] = _clip(fn(f, (0.0, 0), regime))
            except Exception:
                raw[name] = 0.0
        active = {n: s for n, s in raw.items() if abs(s) > 0.05}
        if active:
            wsum = sum(weights.get(n, 1.0) for n in active) or 1e-9
            composite = sum(weights.get(n, 1.0) * s for n, s in active.items()) / wsum
            agree = sum(1 for s in active.values() if s * composite > 0) / len(active)
            breadth = min(1.0, len(active) / 3)
            conf = min(1.0, abs(composite) * (0.4 + 0.6 * agree) * (0.6 + 0.4 * breadth))
        else:
            composite, conf = 0.0, 0.0
        direction = 1 if composite > 0 else -1

        # ---- entry ----
        if side == 0 and conf >= gate and atr > 0 and (direction > 0 or allow_shorts):
            round_trip = 2 * fee + 2 * slip
            take_dist = max(take_m * atr, round_trip * cost_mult * cl)
            stop_dist = take_dist / (take_m / stop_m)
            notional = cash * 0.95
            if direction > 0:
                px = cl * (1 + slip); qty = notional / px; cash -= notional * (1 + fee)
                fees_paid += notional * fee; slippage_paid += qty * cl * slip
                traded_notional += qty * px
                entry, side, water = px, 1, px
                stop, take = px - stop_dist, px + take_dist
            else:
                px = cl * (1 - slip); qty = notional / px
                margin = notional                  # reserve margin from cash
                cash -= margin + qty * px * fee
                fees_paid += qty * px * fee; slippage_paid += qty * cl * slip
                traded_notional += qty * px
                entry, side, water = px, -1, px
                stop, take = px + stop_dist, px - take_dist

        mtm = cash + (qty * cl if side > 0 else
                      (margin + qty * (entry - cl) if side < 0 else 0))
        equity_curve.append((ts, mtm))

    if side > 0:
        _fpx = candles[-1][4]
        cash += qty * _fpx * (1 - fee); trades.append(_fpx / entry - 1)
        fees_paid += qty * _fpx * fee; traded_notional += qty * _fpx
    elif side < 0:
        px = candles[-1][4]
        cash += margin + qty * (entry - px) - qty * px * fee
        trades.append((entry - px) / entry)
        fees_paid += qty * px * fee; traded_notional += qty * px

    eq = [e for _, e in equity_curve]
    rets = [b / a - 1 for a, b in zip(eq[:-1], eq[1:]) if a > 0]
    total = eq[-1] / start_cash - 1 if eq else 0.0
    PPY = 24 * 365                       # hourly bars per year
    wins = [t for t in trades if t > 0]
    # richer, pybroker-parity metrics (all pure functions over eq / trades)
    tq = st.trade_quality(trades)
    exposure = round(bars_in_market / bars_total, 4) if bars_total else 0.0
    # ---- cost decomposition: expose the drag that `net` hides ----
    total_costs = fees_paid + slippage_paid
    fees_pct = fees_paid / start_cash
    slip_pct = slippage_paid / start_cash
    cost_pct = total_costs / start_cash
    cost_analysis = {
        # exact dollars charged on THIS run's fills
        "fees_paid": round(fees_paid, 2),
        "slippage_paid": round(slippage_paid, 2),
        "total_costs": round(total_costs, 2),
        # drag as a fraction of starting capital (exact)
        "fees_pct": round(fees_pct, 4),
        "slippage_pct": round(slip_pct, 4),
        "total_cost_pct": round(cost_pct, 4),
        # turnover = total notional transacted (entries+exits) / starting capital
        "turnover": round(traded_notional / start_cash, 2) if start_cash else 0.0,
        "cost_per_trade_pct": round(cost_pct / len(trades), 5) if trades else None,
        # gross = net + cost drag. Additive reconstruction (ignores the fact that
        # a dollar paid early can't compound), so it's labelled _approx — the
        # exact, trustworthy figures above are the dollars and per-cent drags.
        "gross_return_approx": round(total + cost_pct, 4),
        "net_return": round(total, 4),
    }
    return {"total_return": round(total, 4),
            "cost_analysis": cost_analysis,
            "sharpe_annualized": round(st.annualized_sharpe(rets, PPY), 2),
            "sortino_annualized": round(st.sortino(rets) * math.sqrt(PPY), 2),
            "max_drawdown": round(st.max_drawdown(eq), 4),
            "max_drawdown_bars": st.max_drawdown_duration(eq),
            "annualized_return": (round(st.annualized_return(eq, PPY), 4)
                                  if st.annualized_return(eq, PPY) is not None else None),
            "calmar": (round(st.calmar(eq, PPY), 3)
                       if st.calmar(eq, PPY) is not None else None),
            "profit_factor": tq["profit_factor"],
            "expectancy": tq["expectancy"],
            "avg_win": tq["avg_win"], "avg_loss": tq["avg_loss"],
            "win_loss_ratio": tq["win_loss_ratio"],
            "exposure": exposure,
            "n_trades": len(trades),
            "win_rate": round(len(wins) / len(trades), 3) if trades else None,
            "final_equity": round(eq[-1], 2) if eq else start_cash,
            "returns": rets, "trades": trades,
            "equity_curve": [(t, round(e, 2)) for t, e in
                             equity_curve[::max(1, len(equity_curve) // 200)]]}


def buy_and_hold(candles, start_cash=10_000.0):
    """Benchmark: buy at first bar, hold to last (fees on both sides)."""
    fee = tv("fee_rate")
    first, last = candles[60][4], candles[-1][4]
    qty = (start_cash * (1 - fee)) / first
    final = qty * last * (1 - fee)
    eq = [(c[0], round(qty * c[4], 2)) for c in candles[60::max(1, (len(candles)-60)//200)]]
    rets = [candles[i][4] / candles[i-1][4] - 1 for i in range(61, len(candles))]
    sharpe = (statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(24 * 365)
              if len(rets) > 2 and statistics.stdev(rets) > 0 else 0.0)
    return {"total_return": round(final / start_cash - 1, 4),
            "sharpe_annualized": round(sharpe, 2),
            "final_equity": round(final, 2), "equity_curve": eq}


def walk_forward(candles, n_windows=5, allow_shorts=True):
    """Rolling walk-forward: positive-OOS fraction + IS→OOS Sharpe retention."""
    T = len(candles)
    if T < 400:
        return None
    win = T // (n_windows + 1)
    results = []
    for k in range(n_windows):
        is_slice = candles[:win * (k + 1)]
        oos_slice = candles[win * (k + 1) - 60: win * (k + 2)]
        if len(oos_slice) < 120:
            continue
        is_r = run_composite(is_slice, allow_shorts=allow_shorts)
        oos_r = run_composite(oos_slice, allow_shorts=allow_shorts)
        results.append({"window": k + 1,
                        "is_sharpe": is_r["sharpe_annualized"],
                        "oos_sharpe": oos_r["sharpe_annualized"],
                        "oos_return": oos_r["total_return"],
                        "oos_trades": oos_r["n_trades"]})
    if not results:
        return None
    pos = sum(1 for r in results if r["oos_return"] > 0)
    is_avg = statistics.fmean(r["is_sharpe"] for r in results) or 1e-9
    oos_avg = statistics.fmean(r["oos_sharpe"] for r in results)
    return {"windows": results,
            "positive_oos_fraction": round(pos / len(results), 3),
            "is_sharpe_avg": round(is_avg, 2), "oos_sharpe_avg": round(oos_avg, 2),
            "sharpe_retention": round(oos_avg / is_avg, 3) if is_avg else None,
            "pass": pos / len(results) >= 0.70 and (oos_avg / is_avg if is_avg else 0) > 0.5}


async def composite_report(product, weights=None, allow_shorts=True, n_trials=192,
                           seed=None):
    """Full validated report: composite vs buy-and-hold, walk-forward, DSR, PBO.

    `seed` makes the bootstrap-based figures reproducible. When None it derives a
    stable per-product seed from the global validation seed (config.VALIDATION_
    SEED), so re-running the report yields identical CIs — important because
    these numbers gate live promotion. Pass an int to override, or set
    VALIDATION_SEED<0 for nondeterministic runs.
    """
    from .seeding import derive as _derive
    boot_seed = seed if seed is not None else _derive("composite_report", product)
    candles = await fetch_history(product, granularity=3600, chunks=3)
    if len(candles) < 200:
        return {"error": f"insufficient history for {product}"}
    full = run_composite(candles, weights=weights, allow_shorts=allow_shorts)
    bench = buy_and_hold(candles)
    wf = walk_forward(candles, allow_shorts=allow_shorts)

    rets = full["returns"]
    dsr = st.deflated_sharpe_ratio(rets, n_trials=n_trials) if len(rets) > 3 else None
    psr = st.probabilistic_sharpe_ratio(rets) if len(rets) > 3 else None
    # PBO across the per-strategy return matrix (each single strat, long-only)
    matrix = []
    for name in HIST_STRATEGIES:
        r = run_composite(candles, weights={name: 1.0}, allow_shorts=allow_shorts)
        if len(r["returns"]) > 20:
            matrix.append(r["returns"])
    pbo = st.probability_of_backtest_overfitting(matrix) if len(matrix) >= 2 else None

    wins = sum(1 for t in full["trades"] if t > 0)
    win_ci = st.wilson_interval(wins, len(full["trades"]))
    exp_ci = (st.bootstrap_ci(full["trades"], seed=(boot_seed if boot_seed is not None else 13))
              if len(full["trades"]) > 1 else (None, None, None))

    beat_bench = full["total_return"] > bench["total_return"]
    return {
        "product": product, "bars": len(candles), "granularity": "1h",
        "allow_shorts": allow_shorts,
        "composite": {k: v for k, v in full.items()
                      if k not in ("returns", "trades", "equity_curve")},
        "equity_curve": full["equity_curve"],
        "benchmark_buy_hold": bench,
        "beat_benchmark": beat_bench,
        "walk_forward": wf,
        "validation": {
            "deflated_sharpe": round(dsr, 4) if dsr is not None else None,
            "probabilistic_sharpe": round(psr, 4) if psr is not None else None,
            "pbo": round(pbo, 4) if pbo is not None else None,
            "n_trials_assumed": n_trials,
            "win_rate_ci95": {"point": win_ci[0], "lo": win_ci[1], "hi": win_ci[2]},
            "expectancy_ci95": {"point": exp_ci[0], "lo": exp_ci[1], "hi": exp_ci[2]},
            "gates": {
                "dsr_gt_0_95": (dsr is not None and dsr > 0.95),
                "pbo_lt_0_30": (pbo is not None and pbo < 0.30),
                "walk_forward_pass": bool(wf and wf["pass"]),
                "beats_buy_and_hold": beat_bench,
            },
        },
        "note": "microstructure/derivatives/ml/sentiment strategies are neutral "
                "in backtest (they need live feeds); results reflect the "
                "OHLCV-computable ensemble only.",
    }
