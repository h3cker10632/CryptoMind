# ⚡ CryptoMind — Evidence-Gated Crypto Paper-Trading System

A paper-trading system that only lets a strategy, a model or an outside data
source drive trades once tested history shows it helps — and keeps measuring
after that. Paper trading only (simulated $100k account); real order routing is
deliberately **not** enabled. No API keys are needed for core operation.

**Safety-first defaults:** kill switch, daily-loss halt, per-sleeve budgets,
stale-data guards ("never trade blind"), a full audit trail, and evidence gates
on every learner.

## What runs today

| Sleeve | What it does | Code |
|---|---|---|
| **Core** (opt-in: set `core_allocation_pct` > 0) | Holds the research loop's **champion** (default: BTC/ETH trend, 125-day average with a 2% buffer) — the same function the backtest ran, on the versioned data store. Stale data → hold. Its own **tracking monitor** alerts when holdings drift from the champion's targets. | `app/strategies/core.py`, `app/engine/` |
| **Research loop** (daily) | Backtests every candidate (built-in + queued) on the point-in-time universe with real costs, forward-tracks each from the day its config was frozen, and promotes only when it beats the champion in both backtest halves, clears the **deflated Sharpe** over every variant ever tried, and wins a **paired always-valid forward test**. Clear losers are retired early. Prices the champion under other costs and **after tax**. | `app/engine/challengers.py`, `evidence.py`, `costs_tax.py` |
| **Exploration** (opt-in) | Fast strategies on their own NAV-tracked books; benched at −20%; money follows 30-day results **shrunk by how much evidence they carry**. | `app/strategies/exploration.py` |
| **Polymarket** | Prediction-market sleeve on its **own $500 paper bankroll** (`pm_start_cash`), separate from the main account. Forecasts every tracked market; **bets only once its resolved forecasts beat the market price** (Brier, per market) — raw or via a learned, walk-forward **calibration**. | `app/markets/polymarket/` |
| **Hourly bot** (legacy) | The original hourly signal ensemble and learner stack. Its replay is negative, so `hourly_bot_mode` auto keeps new entries off; learners stay gated by the learner ablation. | `app/signals/`, `app/learn/` |

The kill switch and daily-loss halt measure the main account **excluding the
core's P&L** (the core is built to ride through drawdowns) and stop new entries
in the bot and exploration sleeves. Polymarket runs its own bankroll.

## How it learns (and how fast)

The bottleneck is statistical power, not compute, so the pipeline is built to
get more evidence per day and to reject bad ideas cheaply. Details, settings and
verification for every step: **[docs/IMPROVEMENT_PIPELINE.md](docs/IMPROVEMENT_PIPELINE.md)**.

1. **Data** — versioned candle/funding store with `as_of` loads
   (`app/data/store.py`), plus a point-in-time store for external series with
   history: Fear & Greed, DVOL, stablecoin supply, on-chain activity, and every
   ingested signal (same-second rows in one push stored as their mean)
   (`app/data/series.py`, `tools/series_sync.py`).
2. **Screen** — does a feature predict anything at all? Cross-coin rank IC /
   time-series correlation, Hansen-Hodrick t for the overlapping labels (no
   verdict below 20 non-overlapping observations), both halves, Holm-adjusted
   (`app/engine/screen.py`, `tools/signal_screen.py`).
3. **Model** — pooled walk-forward ML over every liquid coin's history
   (market-relative, vol-scaled, overlap-weighted labels; ridge / boosted trees),
   judged against a momentum baseline before it may become a candidate
   (`app/ml/`, `tools/ml_lab.py`).
4. **Candidate** — frozen configs queued without code changes
   (`tools/candidates.py`, `/api/research/candidates`); each one is a counted trial.
5. **Promote** — research loop gates (above); the promotion rule itself is
   backtested over history (`tools/promotion_backtest.py`).
6. **Trade & watch** — core tracking monitor, per-sleeve limits, scorecard
   (`GET /api/scorecard`) with forward tests, retirements and cost/tax scenarios.

Run every stage once, start to finish: `python tools/run_pipeline.py`
(`--synthetic` for an offline dry run on a synthetic market). Heavy jobs (data
sync, research loop, screen, ML lab, ablation) also run on their own
schedules as separate lower-priority processes (`nice` 10 on Linux, below
normal on Windows); the replay runs in a worker thread. The replay keeps its per-bar signal data on
disk and only computes new bars; the ablation trains its seeds in parallel.

## Module map

| Component | Implementation |
|---|---|
| Orchestrator | `app/orchestrator.py` — decision loop, core / exploration / replay loops, background jobs, scorecard |
| Data | `app/data/store.py` (candles, funding, revisions, point-in-time universe), `app/data/series.py` + `series_sources.py` (external series), `app/data/market.py` (live Coinbase feed), `app/data/ingest.py` (external-signal seam) |
| Portfolio engine | `app/engine/` — panel, causal features, strategies as pure functions, drift simulation, registry, challengers, evidence, screen, costs/taxes, promotion backtest |
| ML | `app/ml/` — pooled dataset, walk-forward models, evaluation, ML candidate strategies; `app/learn/online_model.py` (hourly bot's online net, trusted only on clustered skill) |
| Risk | `app/risk/manager.py` — sizing, kill switch / daily halt (ex-core basis), protections |
| Execution | `app/execution/` — paper broker, OMS + shadow venue, shared costs |
| Polymarket | `app/markets/polymarket/` — engine, forecast ledger, skill gate, calibration |
| Storage | `app/db.py` (SQLite audit), `app/persistence.py` (`state.json` snapshots) |
| Dashboard | `static/index.html` — portfolio-first layout, scorecard |

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The dashboard **Restart** button saves state and relaunches with current code (spawns a successor if you started uvicorn directly; under a supervisor it just exits). Optional wrappers that also auto-relaunch on crash: `bash run.sh` or `.\run.ps1`.

Open http://localhost:8000

The exploration and replay engines use `app/data/store.py` for closed-bar
history. It stores candle and funding revisions in `.cache/store/history.sqlite3`,
with ingestion summaries in `.cache/store/ingest_log.jsonl`. Replay can select
the revisions known at an `as_of` timestamp and returns a content fingerprint.
Keep this directory when restarting if you want to retain historical data.

## API

- `GET /api/status` — full system snapshot (equity, risk, regime, weights, exit-advisor, memes…)
- `GET /api/signals` · `/api/market` · `/api/derivatives` · `/api/research` · `/api/learning` · `/api/universe` · `/api/decisions` · `/api/shadow`
- `GET /api/trades` · `/api/equity` · `/api/events`
- `GET /api/backtest?product=BTC-USD&strategy=trend|meanrev|breakout`
- `POST /api/control/pause` · `/resume` · `/kill` · `/reset-kill` · `/save` · `/reset-account` · `/evolve`
- `GET /api/scorecard` — champion vs candidates (backtest, deflated Sharpe, forward test, retirements), costs / after-tax, live vs holding BTC
- `GET /api/research/candidates` · `POST /api/control/research/candidates` (`{name, config}`) · `POST /api/control/research/candidates/remove`
- `POST /api/settings` — toggle `allow_shorts`, `hedge_enabled`, `llm_advisor_enabled`, `exit_advisor_enabled`, `meme_trading_enabled`, trade stance…

## Legacy: the hourly bot's learning stack

> These learners serve the hourly bot, which is gated off by its replay; each
> learner is also gated by the learner ablation (`app/learn/gate.py`) and on
> 2026-10-02 none passed. The online net's vote is trusted only on clustered,
> baseline-adjusted skill (`online_model.trust`). Kept for reference.

Multiple learning algorithms run concurrently (`app/learn/`), all learning from
realized, after-cost PnL:

1. **Signal scoring** (`loop.py`) — every strategy signal is labeled with its
   realized 1-hour forward return, aligned to direction.
2. **Regime-conditioned Thompson sampling** (`bandit.py`) — each
   (market regime, strategy) arm keeps a Gaussian posterior over aligned
   returns; ensemble weights are Thompson-sampled per regime, so exploration
   vs exploitation is handled by Bayesian uncertainty, not a fixed schedule.
   Weights are EMA-smoothed with a 4% exploration floor.
3. **Online neural-net committee** (`online_model.py`) — an ensemble of three
   independently-seeded 30→16→1 tanh MLPs trained continually (SGD + AdaGrad) on
   live feature snapshots vs 24-hour forward returns, with **quantile heads
   (P10/P90)** for an aleatoric band and inter-member disagreement for epistemic
   uncertainty. Continual-learning safeguards: prioritized experience replay
   (4000 samples, error-weighted), online feature standardization (Welford), and
   honest held-out directional-accuracy tracking (predictions recorded *before*
   labels arrive). It votes as the `ml` strategy only once its skill over the
   always-majority baseline is > 2 clustered SE over ≥ 20 daily label windows
   (weight 0 until then), and its combined uncertainty scales position size
   (0.4×–1.0×) so the book bets small when unsure. Weights + replay persist across
   restart.
4. **Drift detection** (`drift.py`) — Population Stability Index over the
   feature stream; on drift (PSI > 0.25) the online model's learning rate is
   boosted ×3 for fast re-adaptation, then decays back.
5. **Q-learning risk controller** (`rl_risk.py`) — tabular RL agent with state
   (trend, vol, drawdown bucket, loss-streak bucket) and actions
   {0.25…1.25}× risk scale. Reward = equity log-return minus a drawdown
   penalty; epsilon-greedy with decay. Its chosen multiplier feeds directly
   into position sizing every tick.
6. **Genetic strategy evolution** (`evolution.py`) — a GA (population 24,
   8 generations, NSGA-II multi-objective selection over {return, −drawdown,
   per-trade Sharpe}) evolves trading-rule genomes against real history in a
   background thread. Promotion requires **purged walk-forward** validation
   (profit in ≥60% of OOS windows + a deflated-Sharpe gate), and up to 3
   validated genomes are averaged as a champion **portfolio** to dilute
   overfit outliers. The backtester is **NumPy-vectorized** (~5× faster,
   numerically identical, parity-tested; pure-Python fallback if NumPy is absent).
7. **Adaptive loss-cut exit advisor** (`exit_advisor.py`) — after the hard stops
   run, forms an expected next-horizon return in the position's own frame (ML
   committee view blended with a learned per-state value) and **cuts a losing
   position early** when it confidently expects the move to keep going against it.
   Learns **hold-vs-fold** per market state from the counterfactual: what price
   actually did next. Never overrides the hard stop/target/liquidation; exempts
   hedge legs. Toggle `exit_advisor_enabled` (default on).
8. **Direction learner** (`direction.py`) — tracks realized net PnL per
   (regime, direction) and nudges the composite toward the side that has actually
   paid (a bounded tie-breaker, never an override), plus a **multi-timeframe veto**
   that blocks entries fighting a strongly-aligned higher-timeframe trend (stops
   shorting into strong uptrends and vice-versa).

Inspect everything live: `GET /api/learning` returns bandit posteriors,
model accuracy, Q-values, PSI per feature, GA generation history, exit-advisor
and direction-learner stats; the dashboard's "Learning intelligence stack" panel
and the `/brains` Telegram command render it all.

## Meme-coin trading (opt-in sleeve)

A dedicated meme sleeve (`app/data/memes.py`): a **curated seed** of Coinbase-listed
meme majors (DOGE/SHIB/PEPE/BONK/WIF/FLOKI) is eligible immediately, and CoinGecko's
meme-token category is polled so **hot new memes auto-surface** and get tagged. Memes
trade under a **tighter risk envelope** — reduced dollar-risk and position cap
(`meme_risk_factor`), wider ATR stops/targets (`meme_stop_widen`), a concurrent-meme
cap (`meme_max_positions`) and a total meme-exposure cap (`meme_max_exposure`) — so
one meme candle can't run away with the book. Non-meme trading is unaffected. Toggle
`meme_trading_enabled` (dashboard, `/api/settings`); high variance by nature.

## Optional LLM advisor (Gemini, off by default)

An opt-in advisor (`app/learn/llm_advisor.py`) contributes **one** directional vote
that the Thompson bandit weights like any other strategy — never the driver. Defaults
to **Gemini** (`gemini-2.5-flash`); provide a key in `llm_key.txt` (gitignored) or via
`CRYPTOMIND_LLM_KEY` / `GEMINI_API_KEY`. With no key it is completely inert (no cost,
no effect). Enable via the dashboard toggle, `/api/settings`, or the `/llm on` Telegram
command.

## Extending to live trading (shadow plumbing built; real orders deliberately NOT enabled)

The production execution machinery now exists and runs in **shadow mode** — it
mirrors the paper account through a real OMS against live prices but **never
sends a real order**:

- **OMS** (`app/execution/oms.py`): durable trade intents, client order IDs
  (idempotent retries), an explicit order state machine, partial-fill VWAP
  aggregation, idempotent fills, **fail-closed** on ambiguity, and a
  reconciliation loop that treats the venue as the source of truth.
- **`Venue` interface** is the single seam. `ShadowVenue`/`ShadowBroker`
  (`app/execution/shadow.py`) implement it as a simulation and measure
  **live-vs-paper divergence** (`GET /api/shadow`). A `LiveVenue` against
  CCXT / an exchange SDK would plug in here — keep read-only vs trading keys
  separated, no withdrawal permission, IP-whitelisted — and progress paper →
  shadow → canary → live with human approval gates. It is intentionally
  unimplemented.

Money is normalised with `Decimal` to exchange tick/lot/min-notional
(`app/money.py`), market data has a **WebSocket** stream
(`app/data/ws_market.py`), and the control API now **requires a token** for
any state change (`app/security.py`). Automated trading involves substantial
risk of loss and jurisdiction-specific regulation — treat this as research
infrastructure first. See `CRITIQUE.md` and `CHANGES.md`.

## Security

State-changing endpoints (`/api/control/*`, settings, tunables, alerts) require
`Authorization: Bearer <token>`. The token is auto-generated in `api_token.txt`
(0600) on first run, or set `CRYPTOMIND_API_TOKEN`. Requests from localhost are
allowed without a token for local ops; set `CRYPTOMIND_ALLOW_LOOPBACK=0` (as the
Docker image does) to require it everywhere. Front the app with a TLS reverse
proxy in production.

## Validation

`GET /api/backtest/composite?product=BTC-USD` validates the **real ensemble**
(not just the three legacy single-strategy rules) against a **buy-and-hold
benchmark**, with rolling **walk-forward**, **Deflated Sharpe**, **PBO**, and
win-rate / expectancy confidence intervals. Promotion gates: DSR>0.95,
PBO<0.30, walk-forward pass, and beats buy-and-hold. `pytest -q` runs the suite.

## Notes on realism

- **Real retail costs modeled**: 50 bps taker fee + 10 bps slippage per side
  (~1.2% round trip — matches Coinbase Advanced/Kraken base tiers). A
  cost-viability gate rejects any trade whose honest take-profit can't clear
  round-trip cost by >= 2.5x, so structurally unprofitable trades are skipped
  (the target is never quietly widened to the cost floor).
- **Swing-horizon sizing**: stops/targets/trailing are sized off a
  higher-timeframe ATR (default 1h, aggregated from 5m bars via the
  `swing_atr_bars` tunable: 12 = 1h, 3 = 15m, 1 = native 5m) so a trade can
  clear costs at its natural horizon. Widening the horizon — not loosening
  fees — is what makes a signal cost-viable; it's a different strategy, not a
  looser 5m scalp.
- **Market-neutral pair hedge**: longs the laggard / shorts the leader when a
  correlated pair's spread diverges beyond `hedge_z_entry`. Managed as a unit:
  both legs live or neither (a failed second leg unwinds the first), the
  directional signal-flip exit can never close a hedge leg, a pair cost gate
  requires the expected reversion move to clear round-trip cost on all four
  fills (`hedge_cost_multiple`), a re-entry cooldown (`hedge_cooldown_hours`)
  stops churn, the kill switch / daily-loss halt / gross-exposure cap all block
  new hedge risk, and realized PnL is attributed to the `hedge` learning sleeve.
- **Alerting**: Telegram + webhook (Discord/Slack/ntfy) push for kill-switch,
  daily-loss halt, feed outages, loop stalls; configure via dashboard 🔔 or
  `POST /api/alerts/config`; test with `POST /api/alerts/test`.
- **Rich trade notifications**: every position OPEN (side, qty, notional,
  stop/TP, R:R, strategy votes, regime) and CLOSE (entry→exit, PnL $ and %,
  hold time, reason) is pushed to Telegram. Toggle with `push_trades` (dashboard
  button or `/notify on|off` in the bot).
- **Two-way Telegram command bot**: once a bot token + chat id are set, control
  and query the system from your phone. Commands are private to your chat id.
  `/status /positions /trades /pnl /balance /stats /brains /signals /decisions /diag /chart /risk /why`
  `/pause /resume /kill /resetkill /close <product>|all /shorts on|off /mode <stance> /llm on|off`
  `/set <tunable> <value> /get <tunable> /tunables /notify on|off /help`.
  Notably `/resetkill` clears the kill switch + daily halt remotely, and
  `/close BTC` (or `/close all`) flattens a live position from your phone.
- Backtests use intra-bar low/high for stop/target fills (no close-only cheating)
  and report in-sample vs out-of-sample separately.
- Kill switches: 15% max drawdown, 5% daily loss, data-feed health gate.
