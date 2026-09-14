# CryptoMind — Deep Code & Go-Live Critique

*Reviewer analysis of the repository at commit `897c768` ("Initial commit: CryptoMind paper-trading app"). ~5,650 LOC across 38 files. The app imports and runs cleanly under `uvicorn app.main:app`.*

---

## 0. TL;DR verdict

CryptoMind is an **unusually mature, self-aware research prototype**. It is *far* better engineered than the average "AI trading bot" repo: it models real fees, has a kill switch, persists state atomically, backtests in/out-of-sample, and — crucially — ships a `ROADMAP.md` that already tells you most of the hard truths (edge is unproven, fees are the boss fight, most systems don't beat buy-and-hold). Credit where due.

But the gap between "impressive paper simulator" and "safe to trade real money" is **enormous**, and it is almost entirely in the parts that are *not* the strategy. The single most important finding of this review matches the industry consensus: **a trading bot is ~10% strategy and ~90% the boring engineering — connectivity, idempotent orders, reconciliation, risk enforcement — that keeps it alive.** ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9)) That 90% is currently **missing or stubbed**, and the roadmap knows it.

A second, quieter finding is statistical: with a genetic algorithm running ~192 trials per coin repeatedly plus 8 concurrent strategies, the system is doing **massive multiple hypothesis testing**, and its overfitting defense (`in_sample > 0 > out_sample`) is far too weak to catch it. Academic work fully-costing ~50,000 crypto strategies found **nothing survives a Deflated Sharpe gate** — the best of 50k trials is "a coin flip once you count the trials." ([daru.finance](https://daru.finance/research-review/lopez-de-prado/backtest-overfitting)) CryptoMind needs that gate before it trusts its own learning.

The rest of this document is the detailed teardown: **what's good**, **what will break in production**, **concrete bugs**, **statistical validity**, **security**, and a **research-backed "what to add" list**.

---

## 1. What is genuinely good (keep this)

- **Cost realism is a first-class citizen.** 50 bps taker + 10 bps slippage (~1.2% round-trip), a cost-viability gate that widens targets to ≥2.5× round-trip cost, and fees exposed as live tunables. This is the correct instinct — most retail backtests die on fee assumptions. ([turbinefi](https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026))
- **Safety architecture exists.** Kill switch (15% DD), daily-loss halt (5%), macro blackout windows around CPI/FOMC, funding-extreme entry gate, data-health gate, cooldowns, exposure caps. The roadmap correctly names the kill switch "the most valuable feature in the system."
- **State persistence is done right.** Atomic temp-file + `os.replace`, versioned schema (`version: 3`), dimensional-safety on model restore (won't load a model whose feature count changed), and separation of `settings.json` (survives resets) from `state.json`.
- **Separation of concerns is mostly clean.** Data → NLP → signals → risk → execution → learning → orchestrator is a legible pipeline that roughly mirrors the recommended four-module production shape (data collector / strategy / order manager / risk). ([creditforstartups](https://creditforstartups.com/resources/polymarket-trading-bot))
- **The `PaperBroker` interface is a deliberate seam** for a future `LiveBroker`. Good foresight.
- **Honest self-assessment.** The ROADMAP's "Reality checks" section is more intellectually honest than most commercial products. This is the project's best feature.

---

## 2. Critical blockers before ANY live capital (P0)

These are not "nice to have." Each one, alone, can silently drain a live account.

### 2.1 There is no live execution layer — and it's the 90% that matters
`app/execution/paper.py` is a *simulator*: it assumes every order fills instantly at `price ± slippage`, in full, with no rejects. Live exchanges do none of that. The missing machinery, per the current production consensus:

- **Client order IDs + idempotency.** Networks time out *after* the exchange got your order but *before* you get the response. Retry naively → duplicate position. Every order needs a client-generated ID so retries dedupe. ([dev.to](https://dev.to/cvchelles/building-an-idempotent-crypto-order-pipeline-in-nodejs-9ip), [reddit r/algotrading](https://www.reddit.com/r/algotrading/comments/1qjazo0/how_do_you_all_deal_with_exchange_api_failures/))
- **Intent-before-order, durable.** Write the trade *intent* to disk with a unique id **before** calling the exchange. If you crash between call and DB write, you have a live order your system doesn't know about. ([dev.to](https://dev.to/cvchelles/building-an-idempotent-crypto-order-pipeline-in-nodejs-9ip))
- **Explicit partial fills.** A strategy built for full fills that receives partials now owns unwanted inventory. Accumulate fills into a quantity-weighted average price; take truth from executions, not from a single order-status field. ([lycore](https://www.lycore.com/blog/crypto-trading-bot-development/), [dev.to](https://dev.to/lungucristian1980hub/most-trading-bots-break-in-the-same-four-places-and-none-of-them-are-the-strategy-bil))
- **Reconciliation loop.** The exchange is the source of truth, always. On startup, on reconnect, and on a timer, pull open orders/positions/balances and compare — and on disagreement **report the drift, don't silently overwrite it**. ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9))
- **Fail-closed on ambiguity.** On a timeout you don't know if it filled — stop placing new orders and reconcile first. Treat "cancel unconfirmed" as *live exposure*, not "no fill." ([github PR review](https://github.com/gepappas98/polymarket-quant-bot-lite/pull/10))

**None of this exists yet.** The roadmap's P1 "LiveBroker (shadow mode) + reconciliation loop" is the single most important build item, and it is under-scoped as one line. It's a subsystem.

### 2.2 The API is completely unauthenticated — and it can move money
`app/main.py` exposes `POST /api/control/kill`, `/shutdown`, `/restart`, `/reset-account`, `/reset-kill`, and `/api/alerts/config` with **no auth, no CORS policy, no rate limiting**. Uvicorn binds `0.0.0.0`. The moment this runs on the "$5–10/mo VPS" the roadmap recommends, **anyone who finds the port can flatten your positions, wipe your account, disable the kill switch, or steal your Telegram token.** On a live system that's a direct path to loss. This is a P0 the roadmap does not currently mention.

### 2.3 Money math is `float`, not `Decimal`
Every cash/qty/PnL calculation uses binary floating point (`grep -rn Decimal app/` → none). For paper it's cosmetic; live it causes rounding drift against exchange tick/lot sizes and reconciliation mismatches. Exact decimal arithmetic for quantities and prices is a standard requirement — "a rounding bug is expensive compared to decimal.js/Decimal." ([dev.to](https://dev.to/cvchelles/building-an-idempotent-crypto-order-pipeline-in-nodejs-9ip)) You will also need per-symbol **precision, min-notional, and lot-size** handling (roadmap Stage 3 mentions minimums — pull it forward, it's a correctness issue not a scaling one).

### 2.4 30-second REST polling cannot manage live stops
`MARKET_POLL_SEC = 30` with sequential per-product REST calls. A stop can be blown through by 2–5% inside a 30s window in crypto. Production bots use **WebSocket** streams for market data *and* private order/fill updates; "a bot using HTTP polling misses fills, accumulates stale position data, and risks double-submitting orders... If yours doesn't use WebSockets, fix that before anything else." ([chainstack](https://chainstack.com/hyperliquid-trading-bots-2026/), [dev.to](https://dev.to/weston_carnes_d580b505e0c/exchange-api-integration-connecting-a-trading-system-without-losing-orders-13g9)) Roadmap has this as P1 — agree, but it's really a P0 for the *execution* path.

### 2.5 "Spot long-only" in the README, but shorts are on by default
README says *"spot long-only."* Reality: `ALLOW_SHORTS = True`, `settings.allow_shorts` defaults `True`, the paper broker models **margin-style shorts**, and the whole market-neutral hedge module *requires* shorting. **Coinbase spot cannot short.** So a large fraction of the live strategy set (shorts, the pair-hedge, short-capable evolved genomes) is **structurally un-runnable on the venue the data comes from** and would require a derivatives/margin venue with entirely different margin, funding, and liquidation mechanics that the paper broker does not model (no liquidation price, no maintenance margin, no funding payments on your own position). This is a fundamental paper-vs-live divergence, not a config detail.

---

## 3. High-priority issues (P1)

### 3.1 The backtest validates almost nothing the live system actually does
This is the most important *strategy-side* finding. `app/backtest/engine.py` simulates exactly **three** hand-coded long-only rules (`trend`, `meanrev`, `breakout`). But the live ensemble is **eight** strategies (`trend, meanrev, breakout, sentiment, microstructure, derivatives, ml, evolved`) **plus** the Thompson bandit re-weighting, the RL risk controller, the stance machine, exploration probes, and the market-neutral hedge. So:

- `sentiment`, `microstructure`, `derivatives`, `ml`, `evolved` — **never backtested at all.**
- The bandit/RL/stance meta-layer that decides *how* those combine — **never backtested.**
- Shorts and the hedge — **never backtested.**

The "in-sample vs out-of-sample" report therefore tells you very little about the system you'd actually deploy. **Priority: build an event-driven backtester that runs the real composite `SignalEngine.compute()` + risk sizing over history**, so the thing you validate is the thing you ship. (Industry rule: "backtest the exact code you run live." ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9)))

### 3.2 Overfitting defense is far too weak for the amount of searching done
The GA is population 24 × 8 generations ≈ **192 evaluations per coin**, re-run every 20 minutes, rotating the universe — thousands of trials over a day. The bandit continuously selects among 8 strategies. That is heavy multiple testing. The current guardrails:

- `walk_forward_degradation_flag = bool(in_sample>0 > out_sample)` — a single boolean.
- GA promotion gate: champion promoted if OOS is "positive-ish."

The literature is blunt here: "any perseverant researcher will always be able to find a backtest with a desired Sharpe ratio," and across ~50,000 fully-costed crypto strategies the **Deflated Sharpe Ratio of the best was ~0.03** (bar is 0.95) — i.e., indistinguishable from luck once trials are counted. ([daru.finance](https://daru.finance/research-review/lopez-de-prado/backtest-overfitting)) Recommended gates to add (see §7):
- **Deflated Sharpe Ratio (DSR)** on every backtest/GA champion, tracking the trial count. ([turbinefi](https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026), [fortraders](https://fortraders.com/blog/how-to-avoid-bias-in-backtesting))
- **Probability of Backtest Overfitting (PBO)** via combinatorial cross-validation; promote only if PBO < 0.30. ([paperswithbacktest](https://paperswithbacktest.com/course/backtesting-pitfalls-overfitting))
- **Walk-forward with purge + embargo** across ≥5 rolling windows; require positive OOS in ≥70% of windows and IS→OOS Sharpe retention >50–60%. ([vantixs](https://vantixs.com/blog/walk-forward-optimization-crypto), [darkbot](https://darkbot.io/blog/walk-forward-testing))

The roadmap lists "Deflated Sharpe / PBO" as **P2** — this review argues it's **P1**, because the GA is *already* promoting champions into live decisions on the strength of a weak gate.

### 3.3 ~40 days of hourly history proves nothing (roadmap agrees)
`fetch_history` pulls ~900 hourly bars (~37 days). Crypto has multi-month regimes; the Oct-2025 leverage/liquidity crash is the canonical "no backtest covering 2024 included it" example. ([turbinefi](https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026)) You need 1–2+ years and per-regime walk-forward before any statistic is meaningful. Roadmap P1 — agreed.

### 3.4 Blocking I/O on the async event loop
The decision loop is `async`, but inside a single `tick()` it calls **synchronous SQLite** (`db.log_*`, `db.log_equity`) and, every ~60s, a full **`persistence.save()`** that JSON-serializes the entire system state (model weights, replay buffer, research corpus) on the event-loop thread. As the universe and history grow, that save can stall the loop for hundreds of ms — during which stops aren't checked and the watchdog clock ticks. Backtests (`full_report`) also run in the request handler. **Move DB writes and snapshotting off the loop** (thread executor or a writer task/queue), and run backtests in a worker.

### 3.5 SQLite + a global lock is the wrong long-term store
`db.py` opens a fresh connection per call, guarded by one `threading.Lock`, writing an `equity` row every 20s (unbounded growth, no rollup/retention) and an event row per action. Fine for a demo; a bottleneck and an ops liability for a 24/7 live system with WebSocket-rate fills. Plan for WAL mode at minimum, retention/downsampling on the equity table, and ideally Postgres or a time-series store for the equity/trade log. Also: keep an **append-only order/fill event log** (immutable audit trail) — a hard requirement for reconciliation and, later, tax/regulatory review. ([appinventiv](https://appinventiv.com/blog/crypto-trading-bot-development/))

### 3.6 No automated tests — for a system whose hardest code is yet to be written
`find -iname '*test*'` → nothing (only the `backtest` package). The reconciliation/idempotency/partial-fill logic you must add is *exactly* the code that needs deterministic tests: "keep the fragile logic pure... run the nasty scenarios deterministically in CI." ([dev.to](https://dev.to/lungucristian1980hub/most-trading-bots-break-in-the-same-four-places-and-none-of-them-are-the-strategy-bil)) At minimum: unit tests for PnL math (long & short), risk gates, the fill/reconciliation state machine (duplicate fill ignored, out-of-order partials aggregate to correct VWAP, unconfirmed cancel fails closed), and persistence round-trips.

---

## 4. Concrete bugs, smells, and correctness nits (with file refs)

| # | Location | Issue |
|---|----------|-------|
| 1 | `main.py` `kill()` / `shutdown()` | On kill, `broker.sell(p, market.price(p) or broker.positions[p]["entry"], ...)` — if the price feed is down (`None`), it "sells" at the **entry price**, fabricating a flat P&L. Live, this must be a real market/exit order, and a dead feed during a kill is precisely when you can't assume entry price. |
| 2 | `risk/manager.py` `update()` | "Daily" rollover is `time.time() - day_start_ts > 86400` — a **rolling 24h from first tick**, not an exchange/UTC calendar day. The 5% "daily" loss limit resets at an arbitrary wall-clock offset, not midnight. |
| 3 | `orchestrator.py` `snapshot()` | The `unrealized_pnl` expression is convoluted (`... - (0 if broker.positions else 0)`) and mixes long entry-notional with short accounting; for shorts `qty*entry` is not the right basis. Likely reports wrong unrealized PnL when shorts are open. |
| 4 | `online_model.py` | Docstring says "13 → 16 → 1" but `N_IN = 17` and there are 17 `FEAT_NAMES`. Comment rot — harmless but signals drift between code and docs (also appears in README's module map: "13→16→1"). |
| 5 | `signals/engine.py` | Strategy dispatch uses **module-global mutable cells** `_CURRENT_PRODUCT` / `_MARKET_SENT` set per product before calling strategies. This is not reentrant — the moment anything runs `compute()` concurrently (or you parallelize per-product), signals cross-contaminate. |
| 6 | `execution/paper.py` shorts | Short "margin" reserves `notional` of cash but models **no liquidation, no maintenance margin, no borrow/funding cost**. A short can go arbitrarily against you with no margin call — unrealistic, and it flatters short performance the learner then trusts. |
| 7 | `data/market.py` `run()` | Feeds mutate `self.tickers` / `self.candles` while API handlers and the tick loop read them; `persistence.capture()` reads `research.documents` etc. Mostly safe under a single event loop, but `fetch_history`/GA thread + save create windows for "changed size during iteration" on shared dicts/lists. Snapshot-copy before serializing. |
| 8 | `backtest/engine.py` | Uses the **current** `FEE_RATE`/`STOP_ATR_MULT` constants, not `tv(...)` tunables, so backtests silently ignore live-tuned costs/risk the operator set in the dashboard (the GA sim *does* use `tv()`, so the two disagree). |
| 9 | `alerts.py` / `alerts.json` | Telegram bot token + chat id + webhook URL stored **plaintext** on disk and settable via the unauthenticated config endpoint (see §2.2). Secret exposure. |
| 10 | `data/*` polling | Aggressive key-free polling of Coinbase/OKX/CoinGecko/ForexFactory from a fixed VPS IP risks **IP bans / sustained 429s**. There's per-feed backoff for research RSS, but the market/derivatives/candle loops have thinner protection and no key rotation or fallback venue. |
| 11 | `orchestrator.py` `tick()` | `persistence.save()` every 3 ticks and `learner.run()` every 9 ticks both execute **inline** in the decision tick (see §3.4). A slow save delays stop management for every open position. |
| 12 | `db.equity_since` | `LIMIT 200000` guards the query, but nothing ever prunes the `equity` table; after weeks of 20s samples it's millions of rows and the "1m" (30-day) window scans a lot. Add retention/rollup. |
| 13 | `regime()` | Whole-system regime is derived from **BTC alone**. Fine as a proxy, but altcoins routinely decouple; the RL state and bandit conditioning inherit BTC's regime for every asset. |
| 14 | No **per-coin liquidity cap** | Discovered coins pass a 24h-dollar-volume floor once, but position size is never capped as a fraction of that volume, so at scale "your own orders move the book" (roadmap P2 acknowledges — but it's also a *realism* bug in paper: fills assume infinite depth). |

---

## 5. Statistical / ML validity concerns

- **Learning on noise ≈ expensive curve-fitting.** The online MLP labels on 30-min forward returns, the bandit on 1h aligned returns, over *minutes-to-days* of live data. With ±0.4% clipping and a 40-update warm-up, "warmed up" is nowhere near "statistically valid." The roadmap's ≥300-trades gate is the right discipline; the code should surface *confidence intervals* (e.g., Wilson interval on win rate, bootstrap on expectancy) so the dashboard shows how much of the equity curve is skill vs luck.
- **Lexicon sentiment is fragile.** `nlp/sentiment.py` is a bag-of-words polarity counter (`"sec": .5` bearish will misfire on "SEC approves…"; negation and sarcasm are invisible). It's honestly labeled a drop-in for FinBERT/CryptoBERT — do that upgrade before sentiment is trusted with size, or gate its weight hard.
- **Multiple-testing across the whole stack.** GA + 8 strategies + per-regime bandit arms = a lot of "effective trials." Track the trial count and apply DSR/PBO not just to the GA but conceptually to the ensemble selection (§3.2). ([darkbot](https://darkbot.io/blog/walk-forward-testing))
- **Benchmark gap.** The roadmap rightly says "did it beat holding BTC, risk-adjusted?" — but the code has **no benchmark line**. Add BTC (and an equal-weight basket) buy-and-hold to the equity chart and to every backtest report. If it doesn't beat buy-and-hold after costs, nothing else matters.
- **Look-ahead hygiene.** The backtest triggers on `cl` (bar close) and enters at `cl` same-bar — acceptable if you truly fill at that close, but the live path decides on a 30s-stale ticker; document and match the assumption, and prefer next-bar-open fills to be safe. ([fortraders](https://fortraders.com/blog/how-to-avoid-bias-in-backtesting))

---

## 6. Security & operations (must-fix for a 24/7 box)

1. **Authenticate the control API.** Token/JWT or mTLS on all `POST /api/control/*` and `/api/alerts/config`; bind to `127.0.0.1` and put a TLS reverse proxy (Caddy/nginx) with auth in front. Today it's wide open (§2.2).
2. **Secrets out of the repo/disk.** Env vars or a secrets manager for Telegram/webhook/exchange keys; never plaintext `alerts.json`; the `.gitignore` already excludes it but that's not the same as encryption.
3. **Exchange key hygiene (live).** Roadmap says it — trading-only keys, **no withdrawal permission**, IP-whitelisted to the VPS, separate read-only keys for reconciliation. ([chainstack](https://chainstack.com/hyperliquid-trading-bots-2026/))
4. **Supervision.** `run.sh` is a bash `while` loop; use `systemd` (or Docker `restart: always`) with resource limits, log rotation, and health checks. There's no `Dockerfile` — add one (the roadmap's P0 supervision item implies it).
5. **Chaos/failure drills.** Kill mid-order, drop the network, feed malformed API responses — assert it **halts + alerts** (fail safe), never orphans/duplicates (fail open). The roadmap Stage 2 says this; it needs tests to back it (§3.6). ([appinventiv](https://appinventiv.com/blog/crypto-trading-bot-development/))
6. **Observability.** Structured JSON logs, Prometheus metrics (loop latency, feed staleness, fill rate, live-vs-paper slippage, PnL), and an immutable order/fill audit log. "Log every decision, order, fill, and error with enough context to reconstruct 3am incidents." ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9))

---

## 7. What to ADD — research-backed, prioritized

Ordered so each item unblocks the next. This refines and re-sequences the repo's own "Priority build list."

### Tier 0 — before the box even runs live
1. **Auth + TLS + localhost bind on the API.** (§2.2) — hours of work, closes a total-compromise hole.
2. **`Dockerfile` + `docker-compose` with `restart: always`, or a `systemd` unit.** Log rotation, healthcheck. (roadmap P0)
3. **Alerting you'll actually see, verified.** (already built — just wire + test Telegram/webhook end to end).
4. **Benchmark overlay** (BTC & basket buy-and-hold) on equity chart + every backtest. Cheap, and it's the north-star metric the roadmap demands.

### Tier 1 — make validation trustworthy
5. **Backtest the REAL composite** (`SignalEngine.compute()` + risk sizing + stance), not 3 toy rules. (§3.1)
6. **DSR + PBO + purged walk-forward** as hard promotion gates for the GA and for any "this strategy works" claim. Thresholds: DSR>0.95, PBO<0.30, OOS-positive in ≥70% of ≥5 windows, IS→OOS Sharpe retention >50%. Track the trial count explicitly. (§3.2) ([paperswithbacktest](https://paperswithbacktest.com/course/backtesting-pitfalls-overfitting), [vantixs](https://vantixs.com/blog/walk-forward-optimization-crypto)) A ready reference implementation exists: [`esvhd/pypbo`](https://github.com/esvhd/pypbo).
7. **1–2 years of history** + per-regime walk-forward. (§3.3, roadmap P1)
8. **Statistical confidence surfacing** — Wilson interval on win rate, bootstrap CI on expectancy, sample-size counters on the dashboard so nobody mistakes 30 trades for proof.

### Tier 2 — the live execution subsystem (the real 90%)
9. **`LiveBroker` behind the existing `PaperBroker` interface**, built as a proper OMS:
   - Durable **trade-intent store** written *before* the exchange call; **client order IDs** for idempotent retries. ([dev.to](https://dev.to/cvchelles/building-an-idempotent-crypto-order-pipeline-in-nodejs-9ip))
   - Explicit **order state machine** (submitted→accepted→partial→filled/canceled/rejected), transitions **persisted, not inferred**. ([dev.to](https://dev.to/weston_carnes_d580b505e0c/exchange-api-integration-connecting-a-trading-system-without-losing-orders-13g9))
   - **Fill aggregation to VWAP**, idempotent on execution id; **positions derived from fills**, never from "an order was submitted." ([dev.to](https://dev.to/lungucristian1980hub/most-trading-bots-break-in-the-same-four-places-and-none-of-them-are-the-strategy-bil))
   - **Reconciliation loop** (startup + timer + post-reconnect) that treats the exchange as truth and **reports drift**. ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9))
   - **Fail-closed** on timeouts / unconfirmed cancels: reconcile before any new order. ([github PR](https://github.com/gepappas98/polymarket-quant-bot-lite/pull/10))
   - Use **CCXT** or the native exchange SDK; consider **testnet/sandbox** first (e.g., Hyperliquid/Binance testnets). ([chainstack](https://chainstack.com/hyperliquid-trading-bots-2026/))
10. **WebSocket feeds** for market data *and* private order/fill updates, with heartbeat, backoff-reconnect, re-subscribe, sequence-gap resnapshot, and **REST resync on reconnect**. (§2.4) ([dev.to](https://dev.to/weston_carnes_d580b505e0c/exchange-api-integration-connecting-a-trading-system-without-losing-orders-13g9))
11. **`Decimal` money + exchange precision/lot/min-notional** enforcement everywhere. (§2.3)
12. **Shadow mode + live-vs-paper divergence metric** — run paper and live in parallel on the same signals; the divergence number is "your most important discovery." (roadmap Stage 3)

### Tier 3 — quality of edge & scaling safety
13. **Real sentiment model** (CryptoBERT/FinBERT or an LLM classifier) replacing the lexicon, behind the same interface. (§5)
14. **Per-coin liquidity caps** (position ≤ 0.5–1% of 24h volume) enforced in *both* paper fills and live sizing. (§4 #14, roadmap P2)
15. **Async/off-loop persistence + DB upgrade** (WAL now; Postgres/time-series + retention/rollups later) and an **append-only order/fill audit log**. (§3.4, §3.5)
16. **Test suite + CI**, front-loaded on the reconciliation/idempotency/partial-fill state machine kept as **pure functions** so CI can fire every failure mode deterministically. (§3.6) ([dev.to](https://dev.to/lungucristian1980hub/most-trading-bots-break-in-the-same-four-places-and-none-of-them-are-the-strategy-bil))
17. **Multi-venue support** (redundancy + fee optimization) — last, per roadmap P3.

---

## 8. Suggested target architecture (live)

```
        WebSocket (market + user/fill stream)         REST (actions + reconciliation)
                 │  fast, unreliable                        │  authoritative, rate-limited
                 ▼                                          ▼
        ┌──────────────────┐                    ┌───────────────────────────┐
        │  Data Collector  │ in-mem book,       │   Reconciliation Worker   │ exchange = truth,
        │  (seq handling)  │ staleness stamps   │   (startup/timer/reconnect)│ reports drift
        └────────┬─────────┘                    └─────────────┬─────────────┘
                 ▼                                             ▲
        ┌──────────────────┐   pure intents      ┌────────────┴─────────────┐
        │ Strategy Engine  │────────────────────▶│      Order Manager       │ client-order-ids,
        │ (pure function)  │   buy/sell/reduce   │  intent store + OMS FSM   │ idempotent retries,
        └──────────────────┘                     │  partial-fill VWAP        │ fail-closed
                 ▲                                └────────────┬─────────────┘
                 │ vetoes                                      ▼
        ┌──────────────────┐                       ┌────────────────────────┐
        │  Risk Manager    │◀──────────────────────│  Positions (from fills)│
        │ caps/kill/halts  │  state-mismatch halt  └────────────────────────┘
        └──────────────────┘
                 │
                 ▼  append-only audit log · metrics · alerts (all of the above)
```
Key principle already partly present in CryptoMind: **keep the strategy a pure decision function**; everything that touches the exchange is separate, testable infrastructure. That separation is what lets you backtest the exact code you run live and swap strategies without touching the machinery that keeps you alive. ([dev.to](https://dev.to/weston_carnes_d580b505e0c/how-to-build-a-crypto-trading-bot-architecture-not-hype-21g9))

---

## 9. Quick-wins checklist (low effort, high value)

- [ ] Add auth + bind `127.0.0.1` + reverse proxy (closes the open-control-API hole). **P0**
- [ ] Fix the "daily" rollover to UTC calendar day. **1-line-ish**
- [ ] Fix kill/shutdown to not fabricate exit price when the feed is down.
- [ ] Make the backtester read `tv()` tunables so it matches the GA sim.
- [ ] Add a BTC buy-and-hold benchmark line to the equity chart + backtest report.
- [ ] Reconcile README ("spot long-only") with reality (shorts default-on) — pick one and make code+docs+venue agree.
- [ ] Fix the `13→16→1` doc rot (it's `17→16→1`).
- [ ] WAL mode on SQLite + a retention job on the `equity` table.
- [ ] Move `persistence.save()` and DB writes off the event loop.
- [ ] Add a `Dockerfile` + `systemd` unit and a `pytest` skeleton.

---

## 10. Bottom line

CryptoMind is a **genuinely strong research prototype with an honest roadmap** — the ideas, the safety scaffolding, and the cost-realism are ahead of most hobby projects. Its weaknesses are almost entirely in the **production-engineering 90%** (live execution, reconciliation, idempotency, auth, decimal math, WebSockets, tests) and in **statistical validation rigor** (DSR/PBO, backtesting the real ensemble, longer history, benchmark). The good news: the codebase's clean seams (`PaperBroker` interface, pure-ish strategy functions, tunables registry) make all of it *addable* without a rewrite.

Do the Tier-0 and Tier-1 work first — they're cheap and they make everything you learn afterward *trustworthy*. Then build the live execution subsystem as the serious, tested piece it deserves to be. And keep taking the roadmap's own reality checks seriously: **the statistically likely outcome is that it doesn't beat buy-and-hold BTC after costs — the entire staged-gate discipline exists to discover that cheaply on paper rather than expensively live.** ([daru.finance](https://daru.finance/research-review/lopez-de-prado/backtest-overfitting))
