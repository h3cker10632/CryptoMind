# PyBroker vs CryptoMind — where they're ahead, and what's worth stealing

**Analyzed:** [edtechre/pybroker](https://github.com/edtechre/pybroker) (v2.0.1, ~3.5k★,
1,259 commits) against CryptoMind's current tree (`app/backtest/`, `app/learn/`).
**Date:** 2026-09-21.

## TL;DR

PyBroker is a **research/backtesting framework** — a library you write strategies
*in*. CryptoMind is a **live, self-learning autonomous trader**. They overlap only
in one area: **backtesting & validation**. That's the fair comparison surface.

The surprising headline: **CryptoMind's *statistical* validation is already ahead of
PyBroker's.** We have Deflated Sharpe, PSR, PBO via CSCV, and purged walk-forward
with an embargo (López de Prado's toolkit); PyBroker has percentile/BCa bootstrap
CIs but none of DSR/PBO. Where PyBroker is genuinely better is **engineering
maturity**: raw speed (Numba JIT), metric breadth (35 metrics), caching,
reproducibility (seeded bootstrap), and clean separation of data/indicator/model
layers. Those are the things worth borrowing.

---

## Where PyBroker is genuinely better

### 1. Raw backtest speed — Numba JIT everywhere ⭐ biggest gap
PyBroker's core loop and indicators are `@njit`-compiled NumPy; it markets
"backtest at the speed of thought" and benchmarks against VectorBT. CryptoMind's
`app/backtest/engine.py` and `composite.py` are **pure-Python bar loops**
(`for i in range(60, len(candles))` with per-bar `statistics.*`). Only the GA
backtester (`app/learn/evolution.py`) is NumPy-vectorized — and even that has no
Numba. On the GA's hot path (pop 24 × 8 gens × 5 WF windows × N coins) this is the
single largest wall-clock cost in the system.

### 2. Metric breadth — 35 metrics vs our ~6
PyBroker returns Sharpe, **Sortino, Profit Factor, Calmar/MAR, Ulcer index,
max-DD duration, avg win/loss, largest win/loss, win/loss streaks, exposure/time-in-
market, annualized return, expectancy**, etc. CryptoMind's `run_composite` returns
only `total_return, sharpe_annualized, max_drawdown, n_trades, win_rate,
final_equity`. Several missing ones are *directly decision-relevant* for a live book:
- **Sortino** (downside-only deviation) — Sharpe punishes upside vol; wrong for a
  book that wants convexity.
- **Profit Factor** (gross win $ / gross loss $) — the single most intuitive
  edge-quality number, and trivial from data we already have (`full["trades"]`).
- **Calmar / MAR** (annual return ÷ max DD) — the number that actually maps to
  "can I stomach running this."
- **Max-drawdown *duration*** — we report DD depth but not how long underwater.
- **Exposure / time-in-market** — is the edge real or just beta from being long?

### 3. Reproducibility — seeded bootstrap end-to-end
PyBroker added a `seed=` arg to `backtest()`/`walkforward()` so bootstrap CIs are
**reproducible across runs**. CryptoMind's `bootstrap_ci` is seeded (`seed=13`) but
the GA (`app/learn/evolution.py`) seeds per-run and the composite report's DSR/PBO
path is deterministic only by accident. A single `StrategyConfig`-style seed knob
would make every validation number reproducible — important when a promotion gate
(`dsr > 0.95`) decides whether a genome goes live.

### 4. Data / indicator / model **caching**
PyBroker caches downloaded data, computed indicators, and trained models to disk,
keyed by content — so re-running a backtest is near-instant. CryptoMind
**re-fetches Coinbase history every `/api/backtest` call** (`fetch_history` does 3
HTTP round-trips, no cache) and recomputes every feature. A tiny TTL cache on
`fetch_history` + memoized per-bar features would cut backtest latency dramatically
and reduce rate-limit exposure.

### 5. BCa bootstrap (bias-corrected & accelerated)
PyBroker uses the **BCa** bootstrap for its CIs (corrects for skew — crypto returns
are very skewed). CryptoMind's `bootstrap_ci` is a plain **percentile** bootstrap,
which is biased on skewed samples. Upgrading that one function to BCa makes every
win-rate/expectancy CI honest on fat-tailed crypto data.

### 6. Clean layering: DataSource / Indicator / Model registries
PyBroker's `pybroker.indicator(...)` / `pybroker.model(...)` registries with a
uniform `train_fn(train_data, test_data, ticker)` contract keep concerns separate
and make walk-forward retraining a one-liner. CryptoMind computes features in *three
places* (`market.features()`, `composite._features_at()`, `evolution._precompute_
indicators()`) that must be kept in sync by hand — a real bug surface (the composite
docstring even flags this). Not worth a rewrite, but a shared `features_from_ohlcv()`
helper the three callers reuse would remove the drift risk.

---

## Where CryptoMind is already ahead of PyBroker

Worth stating so we don't "improve" backwards:

- **Anti-overfitting statistics.** We have Deflated Sharpe Ratio, Probabilistic
  Sharpe, and PBO via CSCV (`app/backtest/stats.py`). PyBroker has none of these —
  it stops at bootstrap CIs. Our promotion gate is more rigorous than anything in
  PyBroker.
- **Purged walk-forward with embargo** (`walk_forward_eval`, embargo=70) — prevents
  train/test leakage across adjacent windows. PyBroker's walk-forward has no
  embargo/purge.
- **Multi-objective selection** — NSGA-II front (return / −drawdown / Sharpe) +
  top-k champion *portfolio*. PyBroker leaves ranking to the user.
- **Live learning stack** — regime bandit, online NN committee, RL risk controller,
  drift detection, counterfactual credit. PyBroker is offline-only; it has no live
  learning loop at all.
- **Train/live parity** — the GA sizes positions with the *same* risk-manager policy
  the live book uses. PyBroker's `buy_shares` is decoupled from any live executor.

---

## Recommended adoptions (ranked by leverage ÷ effort)

| # | Item | Effort | Payoff | Notes |
|---|------|--------|--------|-------|
| 1 | **Metric expansion** (Sortino, Profit Factor, Calmar, DD-duration, exposure, avg win/loss) | S | High | Pure functions over `trades`/`equity_curve` we already compute. No new data. |
| 2 | **BCa bootstrap** upgrade to `stats.bootstrap_ci` | S | Med-High | One function; makes every CI honest on skewed crypto returns. |
| 3 | **TTL cache on `fetch_history`** (+ optional on-disk) | S | High | Kills the 3× HTTP round-trip per backtest; lowers rate-limit risk. |
| 4 | **Global validation seed** (`StrategyConfig`-style) | S | Med | Reproducible DSR/PBO/GA promotion decisions. |
| 5 | **Numba JIT the composite/GA hot loops** | M-L | High (speed) | Biggest wall-clock win; adds a heavy dep (numba) — gate behind a fallback like evolution.py already does for numpy. |
| 6 | **Shared `features_from_ohlcv()` helper** | M | Med | Removes the 3-way feature-drift bug surface. |

**Suggested first commit (this session if you want):** items **1 + 2** — they're
small, pure-Python, fully testable, add real decision value (Sortino/Profit
Factor/Calmar surfaced in `composite_report` and the dashboard), and carry zero new
dependencies or live-path risk.

## Explicitly NOT recommended
- **Adopting PyBroker as a dependency / rewriting on top of it.** It's AGPL-adjacent
  in spirit but more importantly it's an *offline research* tool — it has no live
  execution, no learning loop, no regime/RL layer. It would replace our strongest
  parts with a weaker offline model. Borrow ideas, not the framework.
- **Optuna hyperparameter search** (PyBroker leans on it). We already do
  evolutionary search with a multiple-testing-aware promotion gate; bolting on Optuna
  would duplicate that with *less* overfitting protection.
