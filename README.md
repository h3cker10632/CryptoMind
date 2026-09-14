# ⚡ CryptoMind — Self-Learning Autonomous Crypto Trading System

A working implementation of the full architecture: research ingestion → NLP →
signal generation → adaptive risk → paper execution → backtesting → self-improvement
loop, with an orchestrator, audit log, and live dashboard.

**Safety-first defaults:** paper trading only (simulated $100k account), spot
long-only, with kill switch, daily loss halt, and full audit trail. No API keys
required — everything runs on public, key-free data sources.

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

- `GET /api/status` — full system snapshot (equity, risk, regime, weights…)
- `GET /api/signals` · `/api/market` · `/api/derivatives` · `/api/research` · `/api/learning`
- `GET /api/trades` · `/api/equity` · `/api/events`
- `GET /api/backtest?product=BTC-USD&strategy=trend|meanrev|breakout`
- `POST /api/control/pause` · `/resume` · `/kill` · `/reset-kill` · `/save` · `/reset-account` · `/evolve`

## The learning intelligence stack (v2)

Six learning algorithms run concurrently (`app/learn/`):

1. **Signal scoring** (`loop.py`) — every strategy signal is labeled with its
   realized 1-hour forward return, aligned to direction.
2. **Regime-conditioned Thompson sampling** (`bandit.py`) — each
   (market regime, strategy) arm keeps a Gaussian posterior over aligned
   returns; ensemble weights are Thompson-sampled per regime, so exploration
   vs exploitation is handled by Bayesian uncertainty, not a fixed schedule.
   Weights are EMA-smoothed with a 4% exploration floor.
3. **Online neural network** (`online_model.py`) — a 13→16→1 tanh MLP trained
   continually (pure-Python SGD + AdaGrad) on live feature snapshots vs 30-min
   forward returns. Continual-learning safeguards: experience replay buffer
   (4000 samples, 6 replays per update) against catastrophic forgetting, and
   honest held-out directional-accuracy tracking (predictions are recorded
   *before* labels arrive). Once warmed up (40 updates) it joins the ensemble
   as the `ml` strategy, trust-weighted by its own accuracy.
4. **Drift detection** (`drift.py`) — Population Stability Index over the
   feature stream; on drift (PSI > 0.25) the online model's learning rate is
   boosted ×3 for fast re-adaptation, then decays back.
5. **Q-learning risk controller** (`rl_risk.py`) — tabular RL agent with state
   (trend, vol, drawdown bucket, loss-streak bucket) and actions
   {0.25…1.25}× risk scale. Reward = equity log-return minus a drawdown
   penalty; epsilon-greedy with decay. Its chosen multiplier feeds directly
   into position sizing every tick.
6. **Genetic strategy evolution** (`evolution.py`) — a GA (population 24,
   8 generations, elitism + tournament selection + crossover + mutation)
   evolves 7-gene trading rules against real hourly history in a background
   thread (hourly, or on demand via the dashboard). Fitness is Calmar-like
   (return/drawdown with overtrading penalties). A champion is **only
   promoted** into the live ensemble (`evolved` strategy) if it passes
   out-of-sample validation — genomes that shine in-sample but fail
   out-of-sample are rejected (anti-overfitting gate).

Inspect everything live: `GET /api/learning` returns bandit posteriors,
model accuracy, Q-values, PSI per feature, and GA generation history; the
dashboard's "Learning intelligence stack" panel renders it all.

## Extending to live trading (deliberately not enabled)

The `PaperBroker` interface (`buy/sell/equity/exposure`) is the seam: implement
the same interface against CCXT / an exchange SDK, keep read-only vs trading
keys separated, and progress paper → canary → live with human approval gates.
Automated trading involves substantial risk of loss and jurisdiction-specific
regulation — treat this system as research infrastructure first.

## Notes on realism

- **Real retail costs modeled**: 50 bps taker fee + 10 bps slippage per side
  (~1.2% round trip — matches Coinbase Advanced/Kraken base tiers). A
  cost-viability gate widens take-profit targets to >= 2.5x round-trip cost
  so structurally unprofitable scalps are never taken.
- **Alerting**: Telegram + webhook (Discord/Slack/ntfy) push for kill-switch,
  daily-loss halt, feed outages, loop stalls; configure via dashboard 🔔 or
  `POST /api/alerts/config`; test with `POST /api/alerts/test`.
- Backtests use intra-bar low/high for stop/target fills (no close-only cheating)
  and report in-sample vs out-of-sample separately.
- Kill switches: 15% max drawdown, 5% daily loss, data-feed health gate.
