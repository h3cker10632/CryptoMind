# CryptoMind — Hardening Changelog

## 2026-10-07 — Improvement pipeline: learn faster, stay honest

Full map: `docs/IMPROVEMENT_PIPELINE.md`.

| Change | Where |
|---|---|
| **Repo**: `.gitignore` `data/` also matched `app/data/`, so `store.py` / `sources.py` were never committed (fresh clones fail) — anchored to `/data/`; two tracked runtime snapshots untracked; CI on every branch. | `.gitignore`, `.github/workflows/ci.yml` |
| **Speed, exact parity**: fsum stdev (cold replay 7.9 s → 4.3 s, identical report); per-bar replay data cached on disk with per-bar input fingerprints (next-day replay 5.8 s → 0.9 s); ablation seeds in parallel; vol forecast fitted on the candidate's own coins (also fixes a backtest/live mismatch). | `app/data/features.py`, `app/backtest/bar_cache.py`, `ablation.py`, `challengers.py` |
| **Forward test**: paired (vol-matched) always-valid sequential test vs the champion replaces "30-day forward Sharpe > champion's" (SE ~3.5 — a coin flip); early promotion on evidence, early retirement of losers; false alarms ~3% when checked every 5 days, ~5% checked daily as the loop does (bound 10%). | `app/engine/evidence.py` |
| **Candidate queue** (CLI + API), **promotion-process backtest**, **signal screen** (cross-coin IC, NW t, halves, Holm), **cost & after-tax scenarios** (taker/maker/low-fee/spot ETF, FIFO tax lots) on the scorecard. | `challengers.py`, `promotion_backtest.py`, `screen.py`, `costs_tax.py`, `tools/` |
| **Per-sleeve risk**: kill switch / daily halt measure the account without the core's P&L and now stop exploration too; core tracking monitor; Polymarket bets only once its forecasts beat the market price (its 10% allocation was then replaced by its own $500 bankroll — entry below); exploration reallocates on shrunk evidence. | `risk/manager.py`, `strategies/`, `markets/polymarket/skill_gate.py` |
| **External data with history**: point-in-time series store (revisions, `as_of`, causal daily alignment); Fear & Greed, DVOL, stablecoin supply, Coin Metrics; ingested signals kept uncapped. Fetchers tested on documented response shapes (hosts blocked in the build environment). | `app/data/series.py`, `series_sources.py`, `tools/series_sync.py` |
| **ML**: online-model trust on clustered, baseline-adjusted skill (a zero-skill "always up" model used to get full trust ~40% of the time); pooled walk-forward cross-coin models (ridge / boosted trees, market-relative vol-scaled overlap-weighted labels) judged vs a momentum baseline; `ml_rank` / `trend_meta` challengers; Polymarket learned calibration used only when it wins out of sample. | `app/learn/online_model.py`, `app/ml/`, `tools/ml_lab.py`, `markets/polymarket/calibration.py` |
| Core: a universe-wide champion no longer disables the core. | `app/strategies/core.py` |
## 2026-10-06 — Polymarket on its own bankroll

Operator choice: Polymarket gets its own $500 paper bankroll and grows (or
shrinks) from there, fully separate from the main account.

| Change | Where |
|---|---|
| **Own bankroll**: the Polymarket broker no longer draws on the shared main-account cash. It starts with `pm_start_cash` (setting, default $500), keeps what it wins and loses, and sizes bets off its own equity. Auto-trade no longer waits for the main account's ledger migration. | `app/markets/polymarket/broker.py`, `engine.py`, `app/settings.py`, `app/main.py` |
| **One-time split at startup**: bets opened while Polymarket shared the main account are refunded at cost to that account (once per bet, idempotent across crashes) and dropped; Polymarket then starts fresh. The `standalone` flag and bankroll persist in `state.json`. | `PMBroker.make_standalone`, `app/persistence.py` |
| **Main account excludes Polymarket**: account equity, exposure, drawdown/kill switch and the main-account reset no longer count or touch Polymarket; `POST /api/polymarket/reset` restarts its bankroll at `pm_start_cash`. The allocator's Polymarket remainder is gone (core + exploration; the rest is unallocated cash). | `app/portfolio.py`, `app/orchestrator.py`, `app/strategies/allocator.py`, `POST /api/control/reset-account` |
| Dashboard: Polymarket tab shows its own equity, return since start and cash. | `static/index.html` |

## 2026-10-04 — Exploration sleeve (fast trading, live learning)

Operator choice: 50% of the account to fast strategies trading live paper in
parallel; a strategy down 20% of its slice is benched.

| Change | Where |
|---|---|
| **Allocator**: core 40% / exploration 50% / Polymarket the remainder (10%). Polymarket used to size bets off the WHOLE account and could spend any pooled cash (incl. cash a strategy holds while out of the market); it now sizes and spends only its own budget. | `app/strategies/allocator.py`, polymarket engine |
| **Exploration sleeve**: 5 members on their own books, measured by NAV per unit (money moving in/out never distorts returns): hourly bot (sized against its slice), BTC/ETH hourly trend 20-day and 5-day, BTC/ETH daily trend 30-day, alt momentum re-picked daily among the top-20 liquid coins. Engine members trade the same strategy functions the backtests run, on store bars, once per new closed bar, 10% drift band, taker costs, stale-data guard. Daily review: bench at −20% of capital (sell out, keep tracking by simulation); reinstate after 30 days if the tracked return since benching is positive (hourly bot: replay positive in both halves); reallocate by 30-day live NAV return, weight exp(5·r30) kept within 0.5x–2x of equal. | `app/strategies/exploration.py`, orchestrator `exploration_loop` |
| History at real fees, before going live (honest preview): BTC/ETH daily 30d trend +12%/yr (Sharpe 0.49, 5.3 y); BTC/ETH hourly 20d −11% over the last year (holding lost −29% the same year); hourly bot −29% (replay); alt momentum daily −44%/yr (−99% DD, 83x turnover); BTC/ETH hourly 5d −74%/yr (162x turnover). The bench rule caps each at ~−20% of its ~$10k. | |
| Faster learning: research loop daily (was weekly); challenger promotion after 30 forward days (was 90) with a stricter deflated-Sharpe bar 0.97 (was 0.95). | settings, `tools/research_loop.py` |
| Engine runs on any bar size (`Panel.bar_sec`), e.g. hourly. Account equity / exposure include the exploration books; the account reset clears them; dashboard lists EXPLORE holdings in Open positions and a per-strategy table (status, capital, equity, return, 30-day, trades) in the Scorecard. | `app/engine/panel.py`, orchestrator, `static/index.html` |

## 2026-10-04 — Rebuild step 5: slim, honest live service

| Change | Where |
|---|---|
| **Core follows the champion** (`core_strategy` "champion", default): the core loop ingests fresh closed daily bars into the store, builds the panel from it and holds the last row of the champion's own backtest function — backtest, research loop and live are one code path, and a promotion reaches live trading only on evidence. Missing / stale data (> 3 days) → no trades (never trade blind). Core may now use up to 100% of the account. | `app/strategies/core.py`, orchestrator `core_loop` |
| **Hourly bot evidence gate** (`hourly_bot_mode` off / on / auto, default auto = new entries only while its daily replay is positive in both halves; it is −75%/yr). Open positions are always managed to their exits. Learner ablation + legacy daily lab only run when they can matter. | `app/orchestrator.py` |
| Background work that only fed the hourly bot defaults to OFF (code kept, one setting each): LLM advisor + web context, model advisor, ML autotrain, news ingestion, strategy researcher + its LLM, crawl4ai producer. Polymarket left as is (separate sleeve, untested — the operator's call). | `app/settings.py` |
| **Scorecard** (`GET /api/scorecard` + dashboard card): what the core runs and why, every candidate's backtest / deflated Sharpe / forward days / promotability, the live result vs simply holding BTC and BTC+ETH since the forward baseline, gate states, data-store freshness. | `app/orchestrator.py`, `app/main.py`, `static/index.html` |
| **Account reset fixed**: it left the CORE book in place (a "fresh $100k" would have carried ~$46k of coins on top) and only checked the crypto sleeve for open positions though its own contract requires every sleeve flat; it now refuses while Polymarket holds positions, clears the core book, and restarts the forward-test baseline. | `POST /api/control/reset-account` |
| Removed `tools/fetch_daily_history.py` (superseded by `tools/data_sync.py`). | |

## 2026-10-04 — Rebuild step 4: ML for risk / costs / regime + challenger loop

Each model was scored as a challenger to the BTC/ETH trend champion
(2019-06 → 2026-10, drift simulation, registry-counted trials).

| Model | Result |
|---|---|
| **Volatility forecast** (HAR on Parkinson range variance, log space, refit monthly, walk-forward) | Forecasts next-week vol **20% better** than the naive 30-day estimate (log MSE 0.120 vs 0.150). As sizing it lowers drawdown (−39% at 0.4 target vs −55%) but also return, and does not beat the champion in both halves. Kept as a challenger / risk dial, not deployed. |
| **Cost model** (Corwin-Schultz spread + fee + sqrt impact) | **Rejected**: estimates BTC's half-spread at 55 bps (real: ~1) — on 24/7 daily crypto bars it reads intraday volatility as spread. Simulations stay on actual fees. |
| **Regime model** (gradient-boosted trees on BTC trend / vol / drawdown / Deribit funding, P(next 30 days up), walk-forward, 30-day embargo) | **Rejected**: 52.1% out-of-sample accuracy vs 55.0% for "always up"; scaling the champion by it cut Sharpe 1.01 → 0.70. |
| **Champion / challenger loop** | 6 frozen candidates backtested weekly on the store and logged; each forward-tracked from its registration day (data unseen when its config was frozen); promotion only with >= 90 forward days of higher Sharpe than the champion + a backtest beating it in both halves + deflated Sharpe >= 0.95 over every variant ever tried (51 so far). A changed config restarts its clock. Champion: `btc_eth_trend` (Sharpe 1.01, DSR 0.961). Weekly in a background process; promotions alerted. `app/engine/challengers.py`, `tools/research_loop.py` |

Also: all models verified causal (scrambling future prices / labels leaves past
outputs unchanged); panel carries daily high / low; drift simulation accepts
per-coin per-day costs.

## 2026-10-03 — Rebuild step 3: daily portfolio strategy (honest data)

All figures: realistic drift simulation (holdings drift, every trade incl.
re-balancing pays 0.6%/side, 20% no-trade band); every variant logged to the
registry (39 tried) so the deflated Sharpe counts them all.

| Finding / change | Detail |
|---|---|
| **Survivorship bias was flattering the core.** Point-in-time universe (top N by trailing dollar volume, delisted coins included), scored from 2021-07 (first date with enough coins): every trend / momentum / vol-target variant on the top 10/20/30 coins had Sharpe −0.55 to 0.31; best (top-10 momentum + vol-target) +4%/yr, −71% max DD, deflated Sharpe 0.43. Holding BTC: Sharpe 0.60, +19%/yr. Same rules on TODAY's 21 coins: up to 0.89 — the gap is hindsight. Equal-weight top-20 alts: −19%/yr. | `app/engine/universe.py` |
| Costs: those strategies traded 17–49x capital a year (10–60%/yr drag at 0.6%/side). Lower fees help (top-10 momentum: 0.31 → 0.51 Sharpe at 0.1%/side) but no fee level makes alt momentum beat holding BTC. | `backtest.simulate_drift` |
| **BTC/ETH trend holds up** (no survivorship question: top two throughout), 2017-06 → 2026-10: Sharpe 0.81–1.17 across EVERY setting (average 50–250 days, buffer 0–5%) vs 0.78 holding BTC; max DD −52% to −69% vs −77%. A plateau, not one lucky cell. Deflated Sharpe 0.995 after 39 variants; probability of backtest overfitting across the 21 settings 0.46 — i.e. *which* length wins is noise, so the plateau middle was taken. 2022–2026 half: ~ties holding BTC on Sharpe (0.78 vs 0.89) with smaller drawdowns — the edge is mostly protection. | |
| Trend buffer (`hysteresis`): switch in above average x 1.02, out below x 0.98 — turnover ~13x → ~5x a year, better Sharpe in most settings. No-trade band width barely matters (Sharpe 1.08–1.10 for 0–50%). | `app/engine/strategies.py` |
| **Carry (hedged funding income) — rejected.** BTC/ETH funding averaged 6.7%/yr; delta-neutral earns on half the capital (~3.4%/yr), below cash. Chasing high-funding coins loses after costs even at 0.1% fees (funding spikes mean-revert). Needs a perp venue anyway. | `app/engine/carry.py` |
| **Core deployed as BTC/ETH trend**: `core_assets` BTC-USD,ETH-USD, `core_sma_days` 125, `core_hysteresis` 0.02, selection trend / sizing equal (no longer "auto" — the daily lab's auto winners were survivorship-biased). Live core runs the engine with full-history replay (buffer state rebuilt from data); drift is now checked daily, as simulated (was weekly). | settings, `app/strategies/core.py` |

## 2026-10-03 — Rebuild step 2: one engine for backtests and live

| Change | Where |
|---|---|
| Portfolio engine: aligned days x coins panel from the store; causal features (row t only sees rows <= t); vectorized simulation (weights at t's close earn t+1, turnover pays cost per side); strategies are pure functions panel -> weights, and live trading takes the SAME function's row for today. | `app/engine/` |
| Verified: reproduces the daily lab's daily returns for all 6 trend/momentum x sizing variants + buy & hold to 1e-15; future prices can't change past weights; live weights == the backtest's row for the same day. On all 411 coins: panel 0.4 s and 6 variants 5 s vs 64 s for the lab's feature build alone. The live core now runs the engine for trend/momentum. | `tests/test_engine.py`, `app/strategies/core.py` |
| **Pick-day luck found and removed**: which weekday momentum re-picks on moved its Sharpe 1.03–1.37 and max drawdown −49% to −71% on identical data. Weekly picks are now on fixed calendar days (live == backtest without saved state) and split into 7 tranches, one per weekday (`core_tranches`). Tranched momentum + vol-target: Sharpe 1.22 (halves 1.00 / 1.43), max DD −64%. | `app/engine/strategies.py` |
| Experiment registry: every backtest logged with params, data version, metrics and daily returns; `n_trials(family)` counts every variant ever tried, feeding the deflated Sharpe. CLI: `tools/backtest.py`. | `app/engine/registry.py` |
| Deflated Sharpe units fixed in the engine: the shared helper defaults the trial spread to 0.5 *per period* (~9.5/yr on daily data → DSR 0 for any real strategy) and treats 1 trial as 2. (Other callers — GA gate, researcher, composite — flagged separately.) | `app/engine/backtest.py` |

## 2026-10-03 — Rebuild step 1: versioned market data store

Why: a re-download silently replaced history and moved a backtested year by 16
points; the old data was gone, so the cause could never be found. The JSON
cache also saved the still-forming bar as if it were final.

| Change | Where |
|---|---|
| Parquet store, one table per (dataset, coin). Each row has `known_at`; a changed bar moves to a revisions table (`superseded_at`) instead of being overwritten, so `load_*(as_of=T)` returns history exactly as known at T. Still-forming bars are never stored; invalid bars are rejected and counted. Every load returns a content fingerprint (`data_version`) that replay / ablation / daily-lab reports now record. | `app/data/store.py` |
| Data: daily candles for 411 Coinbase USD coins incl. 79 delisted (survivorship-bias fix), 440k rows back to listing; hourly for 34 coins, 3 years, 469k rows; hourly funding — Deribit BTC/ETH since 2019 (65k rows each), Hyperliquid 28 coins since May 2023. (Binance/Bybit block this location; OKX keeps ~3 months.) | `app/data/sources.py`, `tools/data_sync.py` |
| Incremental sync with a 3-bar overlap (revisions get caught), retries, Hyperliquid name mapping (kPEPE…); data-quality report (coverage, gaps, zero-volume, reverting spikes, revisions); point-in-time liquidity universe `universe_at(day)` using only data up to that day. The bot syncs daily (`data_sync_interval_sec`) in a background process. | store, orchestrator |
| Verified: identical loads → identical fingerprint; as-of the migration reproduces the legacy cache bar-for-bar; the first sync caught 21 revisions — exactly the 21 still-open bars the old cache had saved. Replay from the store: −75.0% vs −75.1% from the legacy cache. Backtest loaders read the store (legacy JSON as fallback). | `tests/test_store.py` |

## 2026-10-02 — Evidence-gated learning (learning v3)

Rule: nothing that learns drives live decisions until tested history shows it
helps in BOTH halves of its window. Learners keep learning while gated off.

| Change | Where |
|---|---|
| **Learner ablation**: replays the year with no learners, then each learner switched on from an empty state, learning causally; a learner "helps" only if it beats the baseline in both halves, and a sizing learner must also beat a FIXED size at its own average multiplier in every seed (betting smaller on a losing strategy "helps" with no skill). | `app/backtest/ablation.py`, `tools/learner_ablation.py`, replay `hooks` |
| Result (373 days, 21 coins; stochastic learners over 3 seeds, every seed must help): bandit, direction learner, exit advisor — no gain; online-ML vote helped in 1 of 3 seeds (+3.6 / −0.3 on average); RL risk's +30 pts was smaller bets (a fixed 0.31x size does as well). Chop filter helps in both halves → `chop_filter_mode` "on". Note: re-downloading one day of hourly data moved the baseline year from −85% to −69% (same code), so single-run edges of a few points are noise — hence the seed rule. | `reports/learner_ablation_latest.json` |
| **Learner evidence gate**: `bandit_mode`, `direction_learner_mode`, `exit_advisor_mode`, `rl_risk_mode`, `ml_vote_mode` = off / on / **auto** (default: on only while the latest ablation, ≤ 10 days old, says it helps). Re-run weekly in a low-priority subprocess; switches are alerted; a learner switching on is warm-started from its replay-trained state when that is more experienced. | `app/learn/gate.py`, orchestrator |
| **Online-model reset thrash**: reset fired at acc ≤ 50% after 20 scored predictions (noise) — 1,054 resets in a year. Now needs 500 updates, 200 scored predictions and accuracy 2 standard errors below a coin flip. | `online_model.confidently_broken` |
| **Daily lab** for the core (8 years of daily candles, ~6.7 years out-of-sample): selection trend / momentum / gradient-boosted **rank model** (walk-forward, refit monthly on finished labels only) × sizing equal / inverse-vol / **vol-target**. Judged on Sharpe in both halves vs the current rule; the rank model must also beat plain momentum. Live core runs the same `target_weights` function; `core_selection` / `core_sizing` = **auto** follow the weekly lab. | `app/backtest/daily_lab.py`, `tools/daily_lab.py`, `app/strategies/core.py` |
| First result: momentum top-5 + vol-target passes (Sharpe 1.22 vs 0.84; halves 1.03 / 1.41 vs 0.73 / 1.04; max DD −67% vs −83%). The rank model did NOT beat momentum (0.89). Over the longer history the current core rule had a −83% drawdown (May 2021 → Oct 2023) that the one-year replay never saw. Survivorship bias (today's coins) applies to all variants. | `reports/daily_lab_latest.json` |
| Direction conformal gate recomputed its O(n²) threshold on every veto (~1 s/tick live); now cached until a new point arrives. | `direction_conformal._qhat` |
| `combine()` shared by the live engine and the ablation, so re-mixed votes match live exactly. | `app/signals/engine.py` |

## 2026-10-01 — Profitability & learning fixes (learning v2)

Diagnosis from the live paper record (Sep 12 – Oct 1: -$36.5k on $100k, 14.5%
win rate, $20k in fees) and 1.35M scored signals. Root causes and fixes:

| Problem | Fix |
|---|---|
| **$0 price tick "sold" PUMP-USD at $0.00 (-$13.7k)** — `broker.manage` only checked `px is None` | `market.price()` rejects 0/NaN/inf/negative ticks and ticks >35% off the fresh candle; the paper broker refuses to open/close/mark at an invalid price (`tests/test_bad_price_guard.py`). |
| **Costs ~10x the edge.** Avg signal edge ~+5 bps/h vs ~120 bps round trip; median hold 30 min | Native bars 5m -> **1h** (`CANDLE_GRANULARITY=3600`), signal/ML label horizon 1h/30m -> **24h**, stops 2/3/2.5 -> **3/6/3 ATR**, min-hold & re-entry cooldown **12h**, max 2 entries/h, exploration probes off. |
| **Learner trained on gross returns.** Gross signal stream (+5 bps, 1.35M samples) outvoted ~450 real losing trades ~40:1 | Signal stream is now scored **net of round-trip cost** at the 24h horizon; signals and ML samples are recorded **once per bar** (was every 20s tick). |
| **Short trades credited the wrong strategies** (`aligned = net if vote > 0`) — a winning short punished the short-voters | Credit is side-aware (`vote * side > 0`) for real trades and counterfactual skips (`tests/test_learning_v2.py`). |
| **loss_lesson_mult = 5** biased the bandit against low-win-rate / high-payoff sleeves (trend, breakout) | Default 1.0 (unbiased expected net return). |
| **Bandit forgot everything in ~7h** (gamma 0.995) — sparse trade evidence never accumulated | gamma 0.9997 (~5-day half-life). |
| **Mean-reversion / reversal sleeves had negative edge even before costs**; sentiment negative; microstructure/pattern ~0 | Disabled via `config.DISABLED_STRATEGIES`. New slow sleeves: `trend_slow` (EMA24/96 + 72h momentum), `breakout_slow` (48h channel), `xsmom` (cross-sectional 72h momentum). |
| **Multi-timeframe veto never fired** — the "4h" fold needed 576 5m bars but only 300 were kept | MTF folds derived from the native granularity (1h/4h/12h). |
| **Pair hedge lost -$17.4k** (4 taker fills per pair, ~0 spread capture) | `hedge_enabled` default off; memes off; `trade_mode` auto (aggressive doubled risk). |
| Peak-giveback exit sold winners at ~+1% while losers ran to the stop | `trail_giveback_pct` default 0 (hard trailing stop still applies). |
| Online model predicted 30-min moves scaled to 0.4% | Label horizon 24h, target scale = round-trip cost (1.2%), +4 slow features (N_IN 30 -> fresh model). |
| Stale bandit beliefs carried across restarts | `persistence.LEARN_VERSION = 2` discards pre-fix bandit arms/weights once. |

**Validation.** `tools/replay_backtest.py` replays the real signal engine bar by
bar over the cached 1h history (35 days, 21 coins) with the paper broker's
rules. At the modelled 0.5%/side fee: **-6.1%** (max DD -14%) vs the live
record's -36.5% over 19 days. It does **not** show a robust edge: both halves
are negative at 0.5%/side, and equal-weight buy-and-hold was +56%. At
0.25%/side: +2.3%; at 0.1%/side: +7.7% (positive in both halves) — the venue
fee is the single biggest lever. Re-run the replay as more history accrues.

**Automatic replay (added same day).** `app/backtest/replay.py` drives the live
signal engine in a side-effect-free *offline* mode. The orchestrator's
`replay_loop` re-runs it over **every coin in the live universe** (fetching
hourly history for each, including newly discovered coins) on a schedule
(`replay_interval_sec`, default daily) **and immediately when a coin joins the
universe**; the alert then reports that coin's own replayed trades (or that its
history is too short to judge). Results: `reports/replay_latest.json`,
`reports/replay_history.jsonl`, `GET /api/backtest/replay`,
`POST /api/control/replay_now`, the `/replay` (and `/replay now`) Telegram
command, and an alert (warning when negative or down >5 pts vs the last run).
Toggle with the `replay_enabled` setting.

**Replay risk brake.** When a replay is negative in **both** halves of its
window (over >= `replay_brake_min_trades`, default 30, trades), new positions
are sized at `replay_brake_mult` (default 50%) of normal risk; the brake lifts
automatically on the first replay that isn't. It never picks or bans coins —
per-coin replay results had a *negative* rank correlation (-0.29) between the
two halves, so coin selection from them would have hurt. The replay itself
ignores the brake (it tests the strategy, not the braked book). Alerts fire
when the brake turns on/off; `/replay` shows its state. Disable with the
`replay_brake_enabled` setting.

**Efficiency round.**
- *Limit (maker) entries* — setting `entry_order_type` (default `maker`): a
  post-only limit at the signal price that fills only when price trades
  strictly THROUGH it within `maker_timeout_sec` (default 1h), maker fee
  (`maker_fee_rate`, default 0.4% — set yours), no slippage; unfilled orders
  are cancelled. Take-profits of limit-entered positions rest as limits
  (maker); stops/flips/loss-cuts stay taker. Working orders hold a position
  slot. `app/execution/costs.py` is now the one place the risk gate, the
  learner and the replay get round-trip cost from. Replay (35 days): -6.1% ->
  -3.1%, max DD -14% -> -9%, ~95% of limits filled.
- *Long history* — the replay downloads `replay_history_days` (default 365)
  of hourly candles per coin (paced, retries on HTTP 429). Per-bar signals are
  computed once and shared by every window / A/B variant (identical results,
  ~5x faster).
- *Chop filter* — pauses new entries while market-wide trendiness (median
  48-bar efficiency ratio) is below `chop_er_min` (0.12 ~= a random walk).
  `chop_filter_mode` off/on/**auto**: auto = on only while the daily replay
  shows the filter beating no-filter in BOTH halves. (On the 35 days it does
  not, so auto keeps it off.)
- *Waste* — reports every 4h (was 20 min; ~2.7 MB each) and 2 weeks kept;
  LLM/model advisor refresh hourly (leans live 2h); equity sampled once a
  minute; scored signals pruned after 60 days and events after 90; weekly
  component audit alert (which sleeves earn net of costs, which to consider
  switching off, which lack evidence) — it never switches anything off.
- *Forward test* — the first start records equity and every coin's price;
  the replay alert, `/replay` and the status API then show the LIVE paper
  return since then vs equal-weight buy & hold of those coins.
- *Optional core holding* (`app/strategies/core.py`) — OFF unless
  `core_allocation_pct` > 0: holds `core_assets` (default BTC-USD, ETH-USD)
  equally, each only while its daily close is above its 50-day average
  (`core_trend_filter`), weekly drift rebalance. Separate book, never
  managed/flattened/learned by the bot, counted in account equity but not in
  the bot's sizing equity, persisted in state.json.

**ML round.**
- *Learned trade filter* (`app/learn/trade_filter.py`, meta-labeling): judges
  whether an actionable signal is worth taking. Trained on the replay's
  history: every actionable signal, 15 clean direction-aligned features
  (shared verbatim between history and live), labelled with the NET return of
  the bot's own exits (stop / take-profit / trailing, costs included — "target
  or stop first"). Gradient-boosted trees (scikit-learn) with a numpy
  fallback. Walk-forward, leakage-free out-of-sample probabilities (a sample
  is trainable only once its trade RESOLVED before the test month) drive a
  replay A/B; `trade_filter_mode` auto = live gate on only while the filter
  beats no-filter in BOTH halves. Rule: take a signal only if P(net win) >=
  base win rate x `filter_strictness`. Retrained daily with the replay.
- *Warm start* (`app/learn/pretrain.py`): the online model is pre-trained on
  up to 40k historical samples in chronological order, training on each only
  after its 24h label has matured (as live does — training immediately leaked
  the future and inflated accuracy to a fake 65%).
- *Cleaner inputs*: order-book, sentiment and chart-pattern inputs are held at
  0 (no measured edge, absent in history) — same vector width, no reset.
- *Better grading*: the online model's accuracy only counts moves >= half the
  round-trip cost.

**First real one-year replay (Sep 2025 - Oct 2026, 18 coins, hourly).** The
hourly strategy: **-91.6%** (halves -79% / -71%), 1,456 trades averaging
-1.2% net — slightly negative even before costs. Every hourly signal family's
gross edge is far below the round-trip cost. The trade filter (out-of-sample)
and the chop filter each help in BOTH halves (-87.5% / -84.9%), so auto
turns them on, but neither creates an edge. Leak-free warm-start accuracy on
tradeable 24h moves: 50.7% vs 53.1% for always-down — no skill. Slow daily
rules over the same history, same costs: per-coin "hold while above the
100-day average" +75% (max DD -12%), BTC/ETH core with the 50-day filter +20%
(DD -22%). The daily replay alert now reports these slow rules next to the
bot (`slow_benchmarks`); the core holding gained `core_sma_days` and
`core_assets = UNIVERSE`.

**Operator decision (2026-10-01):** 80% of the account in the slow rule —
`core_allocation_pct=80`, `core_assets=UNIVERSE`, `core_sma_days=100` — and
the hourly bot on the remaining 20%. The bot now sizes against its fixed 20%
share (`core.bot_equity`), never the core's idle cash for coins below their
average.

---


Everything from `CRITIQUE.md` was addressed, **except** the recommendation to
turn shorts off (§2.5): per operator request, **shorts remain ON by default**
(`allow_shorts=True`), and the new live-execution machinery is built in
**shadow mode only — it never trades real money.**

Status legend: ✅ done · 🟡 done, opt-in/partial by design.

---

## Critical blockers (P0)

| Item | What was done |
|---|---|
| §2.1 No live execution layer | ✅ New OMS (`app/execution/oms.py`): durable trade **intents**, **client order IDs** (idempotent retries), explicit **state machine** (created→submitting→open→partial→filled/canceled/rejected/**unknown**), **partial-fill VWAP** aggregation, idempotent fill dedupe, **fail-closed** on timeout/unconfirmed-cancel, and a **reconciliation loop** that treats the venue as truth and **reports drift**. A `Venue` interface is the single live seam; `LiveVenue` is intentionally NOT implemented. |
| §2.1 Shadow trading | ✅ `ShadowVenue` + `ShadowBroker` (`app/execution/shadow.py`) mirror the paper account through the OMS against **live prices** with realistic fees/slippage/partials — **no real money** — and measure **live-vs-paper divergence** (roadmap's key Stage-3 metric), surfaced at `/api/shadow`. |
| §2.2 Unauthenticated control API | ✅ `app/security.py` middleware: every state-changing route (`/api/control/*`, settings, tunables, alert-config/test) needs `Authorization: Bearer <token>` (token in `api_token.txt`, 0600, or `CRYPTOMIND_API_TOKEN`). Loopback allowed for local ops; compose sets `CRYPTOMIND_ALLOW_LOOPBACK=0`. `/api/security` shows status. |
| §2.3 Float money math | ✅ `app/money.py` — `Decimal` price/qty rounding to **tick/lot** size + **min-notional** enforcement, wired into the paper broker and shadow venue at the fill boundary. Per-instrument rules registry. |
| §2.4 30s polling can't manage stops | ✅ `app/data/ws_market.py` — Coinbase **WebSocket** ticker stream with heartbeat, exponential-backoff reconnect, re-subscribe on universe change; REST stays authoritative for candles/book. Status at `/api/market.ws`. |
| §2.5 "spot long-only" vs shorts on | 🟡 Per request, **shorts kept ON**. README wording corrected; shorts now model **funding accrual + liquidation** (see §4.6) so they're no longer unrealistically riskless. |

## High priority (P1)

| Item | What was done |
|---|---|
| §3.1 Backtest only 3 toy rules | ✅ `app/backtest/composite.py` runs the **real ensemble** strategy functions + confidence/weighting + risk sizing + cost gate + ATR stops/targets/trailing (long & short), with a **buy-and-hold benchmark**. `/api/backtest/composite`. |
| §3.2 Weak overfitting defense | ✅ `app/backtest/stats.py` — **Deflated Sharpe (DSR)**, **Probabilistic Sharpe**, **PBO (CSCV)**, Wilson & bootstrap CIs. Composite report emits hard **gates** (DSR>0.95, PBO<0.30, walk-forward pass, beats buy&hold). GA promotion now uses a **DSR + OOS-trade-count** gate. |
| §3.3 ~40 days history | 🟡 Backtester pulls the max Coinbase allows (~950 hourly bars ≈ 40d) and now runs **rolling walk-forward** (positive-OOS fraction + Sharpe retention). Deeper multi-year history needs an external OHLCV source — hook is `fetch_history`. |
| §3.4 Blocking I/O on event loop | ✅ DB writes now go through a **background writer thread**; `persistence.save()` runs in an **executor**; reconciliation + backtests run off the hot path (`asyncio.to_thread`). |
| §3.5 SQLite + global lock | ✅ **WAL** mode + `busy_timeout`, async writer queue, **retention/prune** on the equity table, and an append-only **`order_events`** audit table. |
| §3.6 No tests | ✅ `tests/` with 29 tests (OMS state machine incl. duplicate/out-of-order/unknown/unconfirmed-cancel, money math, DSR/PSR/PBO, long/short PnL, liquidation, auth) + **GitHub Actions CI**. |

## Concrete bugs (all fixed)

1. ✅ Kill/shutdown no longer **fabricate an exit at entry price** when the feed is down — they strand + alert instead of faking a flat PnL.
2. ✅ Daily-loss rollover is now a **UTC calendar day**, not a rolling 24h.
3. ✅ `snapshot()` **unrealized PnL** rewritten correctly for longs **and** shorts.
4. ✅ Doc rot fixed: MLP is **17→16→1** (code + README).
5. ✅ Signal-engine per-product context moved to **thread-local** (reentrant; safe to parallelize).
6. ✅ Shorts now model **funding accrual** (live OKX rate) + **liquidation** on margin exhaustion.
7. ✅ Persistence **snapshot-copies** shared collections + a save lock (no "changed size during iteration").
8. ✅ Backtester reads **live `tv()` tunables** (matches the running system / GA sim).
9. ✅ Secrets: `api_token.txt` 0600 + gitignored; control API protects alert-config.
10. 🟡 Data polling backoff already present for research; market/deriv loops unchanged (documented risk).
11. ✅ `persistence.save()` off the event loop (#3.4).
12. ✅ Equity table **retention/prune** job.
13. 🟡 Regime still BTC-derived (documented; per-asset regime is a larger design change).
14. ✅ **Per-coin liquidity cap** (`liq_cap_pct` tunable) — position ≤ x% of ~24h dollar volume.

## Also added

- ✅ **Sentiment upgrade path**: optional CryptoBERT/FinBERT via `CRYPTOMIND_SENTIMENT_MODEL` (graceful fallback), plus **negation handling** in the lexicon.
- ✅ **Ops**: `Dockerfile` (non-root, healthcheck), `docker-compose.yml` (`restart: always`, localhost bind, secrets via env), `deploy/cryptomind.service` (systemd, hardened).
- ✅ **Dashboard**: shadow-execution & divergence panel, security panel, "Validate REAL ensemble" button with benchmark overlay + stat gates.
- ✅ **Statistical honesty**: win-rate Wilson intervals + bootstrap expectancy CIs on the composite report.

---

### How to run the new bits

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000     # api_token.txt is auto-created
# Dashboard → "Validate REAL ensemble" for DSR/PBO/walk-forward vs buy-and-hold
# /api/shadow shows the shadow OMS + live-vs-paper divergence (no real money)
docker compose up -d                                # always-on, restart:always
pytest -q                                           # 29 tests
```

### Deliberately NOT done
- **No `LiveVenue`.** Real order placement stays unimplemented by design — the OMS/`Venue` seam is ready, but going live requires exchange keys, human approval gates, and the roadmap's staged canary. Shadow mode gives you all the plumbing safely.
