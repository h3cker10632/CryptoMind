# Evolving/Learning stack

> **Status:** historical record of the hourly bot's learning stack. The bot and
> its learners are gated off by evidence (replay / learner ablation); the
> system's learning now runs through the research loop, the signal screen and
> the pooled ML lab — see `docs/IMPROVEMENT_PIPELINE.md`. The online model's
> vote is trusted only on clustered, baseline-adjusted skill.

The learning stack landed in three original phases (below), then grew three more
self-learning subsystems (see "Later additions"). Everything learns from realized,
after-cost PnL — no golden labels, no look-ahead.

---

## Phase 1 — quick wins (sample efficiency + non-stationarity)
*`app/learn/bandit.py, online_model.py, drift.py, loop.py, evolution.py, tunables.py, persistence.py`*

- **Bandit forgetting** — `decay(gamma)` exponentially ages every arm's
  effective sample size + variance each cycle (`bandit_decay_gamma`, ~7h
  half-life) and prunes idle arms. A strategy that stopped working sheds its
  stale reputation instead of coasting on old wins.
- **Prioritized experience replay** — replay memories are sampled by prediction
  error, not uniformly, so informative samples train more often.
- **Online feature standardization** — Welford running mean/var per feature;
  scaling self-calibrates as the universe/regime shifts.
- **Concept-drift (Page-Hinkley)** — watches the model's *prediction-error*
  stream (PSI only catches input drift); a detection boosts the LR to re-adapt.
- **GA warm-start** — seed each run's population with the standing champion +
  mutants and anneal the mutation rate high→low. Continued search, not a cold
  restart (was ~192 evals rediscovering the basics every run).

## Phase 2 — evolution deep-dive (robust, not curve-fit)
*`app/learn/evolution.py, signals/engine.py, persistence.py, alerts.py`*

- **NSGA-II multi-objective** — evolve the whole Pareto front over
  {return, −drawdown, per-trade Sharpe} (fast non-dominated sort + crowding),
  instead of one hidden Calmar risk-appetite.
- **Purged walk-forward validation** — promotion runs candidates through 5
  sequential OOS windows with an embargo gap; requires profit in ≥60% of them,
  a healthy trade count, bounded worst-window drawdown, and a deflated-Sharpe
  gate that prices the multiple-testing bias. No more lucky-tail champions.
- **Champion portfolio (top-k)** — promote up to 3 validated genomes/product;
  `strat_evolved()` averages their votes, diluting overfit outliers.
- **Train/live sizing parity** — `simulate()` now sizes like the live risk
  manager (risk a fixed fraction to the ATR stop, capped) instead of 95%-cash,
  so the GA optimises the *same* strategy that runs live.

## Phase 3 — model deep-dive (knows what it doesn't know)
*`app/learn/online_model.py, loop.py, signals/engine.py, risk/manager.py, orchestrator.py, persistence.py, alerts.py`*

- **Quantile heads (P10/P50/P90)** — pinball loss on the shared hidden layer →
  an aleatoric uncertainty band around each prediction.
- **Model committee** — 3 seeded TinyMLPs; their disagreement = epistemic
  uncertainty (high on unfamiliar/OOD inputs). Combined into a [0,1] confidence.
- **Uncertainty → vote → size** — `strat_ml()` damps its vote by confidence, and
  the signal carries `ml_confidence` into `risk.size(...)` which scales risk
  0.4×–1.0×. The book bets small when unsure, presses size only on tight
  agreement — never zeroing a trade the rest of the stack still wants.

Backward compatible: the committee's primary member aliases the old `model`;
legacy single-net snapshots load into it and seed the other members.

---

## Later additions (post-phases)

These shipped after the three phases above and are all live on `4afb09a`.

### Adaptive loss-cut exit + hold-vs-fold learning
*`app/learn/exit_advisor.py`, `orchestrator.py` (step 5b), `tunables.py`, `persistence.py`*

- **Predictive early exit** — after the hard stops run, the advisor forms an
  expected next-horizon return **in the position's own frame** (ML committee
  forward view blended with a learned per-state value). A losing position it
  confidently expects to keep bleeding is **cut early**, before the hard stop.
  Never overrides stop/target/liquidation; exempts hedge legs; honors `min_hold_sec`.
- **Counterfactual learning** — every consultation is recorded and scored a
  horizon later against what price *actually* did next (reusing the learner's
  `_price_at`). One observation scores BOTH implied actions, so it learns per
  market-state (with/against trend × loss-size × model-view × vol) whether
  staying in pays.
- Tunables: `exit_cut_threshold`, `exit_min_loss_pct`, `exit_ml_weight`,
  `exit_horizon_sec`. Setting `exit_advisor_enabled` (default **on**) + dashboard
  toggle. Reported in `/brains` and the status snapshot.

### Direction learner (long vs short)
*`app/learn/direction.py`, `signals/engine.py`, `learn/loop.py`*

- **Learned per-regime edge** — tracks realized net PnL per `(regime, direction)`
  and nudges the composite toward the side that has actually paid. A bounded
  tie-breaker (can flip a *marginal* call), never an override.
- **Multi-timeframe veto** — blocks entries fighting a strongly-aligned higher-
  timeframe trend (`mtf_align`), so it stops shorting into strong uptrends and
  vice-versa. Tunables: `direction_bias_gain`, `direction_bias_cap`, `mtf_veto_align`.

### NumPy-vectorized GA backtester
*`app/learn/evolution.py`, `requirements.txt`*

- `_precompute_indicators()` computes EMA/ATR/RSI/rolling-breakout **once** per
  genome with NumPy (cumsum rolling means + `sliding_window_view`); the
  sequential (path-dependent) trade loop just indexes into those arrays.
- **~5× faster** (`simulate()` 9.6→2.0 ms, one `evolve()` run 1.56→0.32 s),
  **numerically identical** to the old per-bar arithmetic (~1e-15), verified by
  parity tests running the NumPy path and a forced pure-Python fallback side by
  side. NumPy is optional — falls back to per-bar computation if absent.

Tests: `test_exit_and_direction.py`, `test_memes.py`, `test_evolution_numpy.py`.
