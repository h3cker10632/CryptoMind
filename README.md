# ⚡ CryptoMind — Self-Learning Autonomous Crypto Trading System

A working implementation of the full architecture: research ingestion → NLP →
signal generation → adaptive risk → paper execution → backtesting → self-improvement
loop, with an orchestrator, audit log, and live dashboard.

**Safety-first defaults:** paper trading only (simulated $100k account), with
kill switch, daily loss halt, gross-exposure caps, and a full audit trail.
Directional shorts and market-neutral pair hedging are supported (runtime
toggles); real order routing is deliberately **not** enabled. No API keys are
required for core operation — everything runs on public, key-free data sources
(an optional Gemini LLM advisor is the only key-gated, off-by-default extra).

## Architecture → Module map

| Blueprint component | Implementation |
|---|---|
| Orchestrator / Control Plane | `app/orchestrator.py` — decision loop, health, human-in-the-loop pause/kill |
| Research & Data Ingestion | `app/data/research.py` — self-expanding source pool (news RSS + combined Reddit multireddit + auto-spawned Google News feeds per coin), adaptive 429 backoff, source promotion/demotion, Fear & Greed index, self-directed research queue |
| Dynamic Universe Discovery | `app/data/universe.py` — extracts coin mentions from research + CoinGecko trending, validates against Coinbase listing & liquidity floor, expands/prunes the tradeable universe (core 6 protected, cap 12, held positions never pruned) |
| Market Data & Order Book Feed | `app/data/market.py` — live Coinbase Exchange candles, tickers, L2 order-book depth/imbalance |
| Derivatives Feed | `app/data/derivatives.py` — OKX public API: perp funding rates, open-interest history, long/short account ratio, taker buy/sell aggression (per asset, key-free) |
| NLP / Signal Generation | `app/nlp/sentiment.py` — crypto sentiment lexicon, per-asset scores, narrative detection (drop-in interface for FinBERT/CryptoBERT) |
| Signal Engine | `app/signals/engine.py` — 5-strategy ensemble: trend, mean-reversion, breakout, sentiment, microstructure; outputs direction, confidence, edge, invalidation |
| Adaptive Risk Manager | `app/risk/manager.py` — vol-adjusted sizing, exposure/position caps, drawdown kill switch, daily loss halt, loss-streak risk scaling, regime scaling, cooldowns |
| Execution Engine (Paper) | `app/execution/paper.py` — OMS with fees, slippage, stops, take-profits, ATR trailing stops |
| Backtesting & Validation | `app/backtest/engine.py` — event-driven sim on ~950 real hourly bars, in-sample vs out-of-sample walk-forward split, overfitting flag |
| Self-Improvement Loop | `app/learn/loop.py` — bandit-style meta-learner: scores every signal against realized 1h forward returns, softmax-reweights the strategy ensemble (with exploration floor) |
| Storage / Audit | `app/db.py` — SQLite: events, trades, equity curve, scored signals |
| State Persistence | `app/persistence.py` — full system snapshot to `state.json` (~1/min, atomic): paper account & open positions, neural-net weights + replay buffer, Q-table, bandit posteriors, GA champion, discovered universe, research corpus + source-discovery memory (promotions/demotions/track records). Auto-restored on startup |
| Macro Context | Macro feeds (Fed/rates, inflation, crypto regulation via Google News) scored with a risk-on/risk-off lexicon; macro sentiment is 20% of market sentiment and feeds the ML model via sentiment features |
| Dashboard | `static/index.html` — live equity curve, signals, positions, weights, research feed, audit log, backtest runner |

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The dashboard **Restart** button saves state and relaunches with current code (spawns a successor if you started uvicorn directly; under a supervisor it just exits). Optional wrappers that also auto-relaunch on crash: `bash run.sh` or `.\run.ps1`.

Open http://localhost:8000

## API

- `GET /api/status` — full system snapshot (equity, risk, regime, weights, exit-advisor, memes…)
- `GET /api/signals` · `/api/market` · `/api/derivatives` · `/api/research` · `/api/learning` · `/api/universe` · `/api/decisions` · `/api/shadow`
- `GET /api/trades` · `/api/equity` · `/api/events`
- `GET /api/backtest?product=BTC-USD&strategy=trend|meanrev|breakout`
- `POST /api/control/pause` · `/resume` · `/kill` · `/reset-kill` · `/save` · `/reset-account` · `/evolve`
- `POST /api/settings` — toggle `allow_shorts`, `hedge_enabled`, `llm_advisor_enabled`, `exit_advisor_enabled`, `meme_trading_enabled`, trade stance…

## The learning intelligence stack

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
   independently-seeded 18→16→1 tanh MLPs trained continually (SGD + AdaGrad) on
   live feature snapshots vs 30-min forward returns, with **quantile heads
   (P10/P90)** for an aleatoric band and inter-member disagreement for epistemic
   uncertainty. Continual-learning safeguards: prioritized experience replay
   (4000 samples, error-weighted), online feature standardization (Welford), and
   honest held-out directional-accuracy tracking (predictions recorded *before*
   labels arrive). Once warmed up it joins the ensemble as the `ml` strategy,
   trust-weighted by accuracy, and its combined uncertainty scales position size
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
