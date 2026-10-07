# Improvement pipeline — learn faster, stay honest

Everything here follows one rule: **nothing drives trades until tested history
shows it helps, and evidence is used as efficiently as possible.** The limit on
how fast the system can learn is statistical power — how much real evidence
arrives per day — so each stage is about extracting more evidence, rejecting
bad ideas cheaply, and never mistaking noise for skill.

```
 data ──► screen ──► model ──► candidate ──► promote ──► trade & watch
 store    IC test    walk-     frozen        paired      core tracking,
 series   both       forward   config,       always-     per-sleeve limits,
 as-of    halves     vs        counted       valid       scorecard
                     baseline  trial         test
```

## 0. Run it start to finish

```bash
python tools/run_pipeline.py                 # sync real data, then every stage
python tools/run_pipeline.py --skip-sync     # every stage on the data already stored
python tools/run_pipeline.py --synthetic     # offline dry run on a synthetic market
```

`tools/run_pipeline.py` runs **sync → screen → ml → research → promotion**,
each as its own process (one failing doesn't stop the rest), and writes a
one-page summary to `reports/pipeline_latest.json`. The orchestrator also runs
every stage on its own schedule (table at the end).

`--synthetic` never touches this checkout: it copies `app/` and `tools/` into
a throwaway workspace, fills its store with a synthetic market
(`tools/make_synthetic_store.py`: regime-switching trends, listings and
delistings, hourly bars, funding, Fear & Greed / DVOL-like series) and runs
every stage there. `--plant-signal 0.003` adds a cross-coin effect (a
low-volatility premium momentum doesn't capture) so the whole
model → candidate → research → live chain can be watched working.
`tests/test_pipeline_e2e.py` does exactly that on every CI run: the ML lab
finds the planted signal (IC ≈ 0.14, t ≈ 10 vs a momentum baseline of 0.07),
queues `ml_rank_top20_h7`, the research loop scores it (no forward days yet,
so not promotable), and — made champion — its saved weights drive the core.
Synthetic numbers prove the plumbing, not an edge.

**Network.** The build environment used for this work blocks the market-data
hosts (Coinbase, Hyperliquid, Deribit, alternative.me, DefiLlama, Coin
Metrics), so the real-data run has to happen on your machine — or in a cloud
environment whose network access allows those domains.

**`app/data/sources.py`** (the downloads `tools/data_sync.py` uses: Coinbase
USD products incl. delisted, candles, Hyperliquid and Deribit hourly funding)
was never committed — an old `.gitignore` rule `data/` also matched
`app/data/` (now anchored to `/data/`). It has been rebuilt from the interface
`tools/data_sync.py` calls and tested against mocked APIs, including a full
`data_sync` run into the store (`tests/test_sources.py`). If you still have
the original on your machine, diff the two before replacing either.

## 1. Data — history first, point-in-time

| What | Where |
|---|---|
| Candles / funding with revisions and `as_of` loads | `app/data/store.py`, `app/data/sources.py`, `tools/data_sync.py` |
| External series with history: Fear & Greed (2018+), Deribit DVOL BTC/ETH (2021+), DefiLlama stablecoin supply, Coin Metrics community active addresses / tx count | `app/data/series_sources.py` |
| Point-in-time series store: `known_at` per value, revisions kept, `load(as_of=T)`, causal `daily_array()` for panels | `app/data/series.py` (`.cache/store/series.sqlite3`) |
| Every ingested external signal (crawl4ai etc.), uncapped, as `ext:<kind>:<asset>`, known from arrival | `app/data/ingest.py` → series store |

- `python tools/series_sync.py [--dry-run] [--only fear_greed dvol_btc]` —
  daily from the replay loop (`series_sync_interval_sec`).
- Publication lags are conservative (a daily value is never "known" before
  its day ends; Coin Metrics +2 days).
- **Not verified live here**: this build environment's network proxy blocks
  these hosts. The parsers follow each API's documented response format and
  are tested against those shapes (`tests/test_series.py`). Run
  `tools/series_sync.py --dry-run` once on your machine and check the first /
  last rows it prints.

## 2. Screen — reject cheaply

`app/engine/screen.py`, `python tools/signal_screen.py` (weekly):

- **Cross-sectional**: daily rank correlation (IC) between a feature and the
  next-h-day return across the point-in-time liquid universe (delisted coins
  included). Uses ~400 coins of history even when only BTC/ETH trade, and the
  market's direction drops out.
- **Time series**: a market-wide series (e.g. Fear & Greed) vs BTC's next-h-day
  return.
- Newey-West t with `h` lags (overlapping labels), both halves must agree in
  sign, Holm-adjusted across everything screened. Report:
  `reports/signal_screen_latest.json`.

## 3. Model — pooled, walk-forward, vs a baseline

`app/ml/`, `python tools/ml_lab.py [--with-series]` (weekly):

- **Dataset**: every (day, coin) of the liquid universe is a sample. Causal
  features z-scored across coins daily; market-wide features (BTC trend,
  breadth, optional external series as causal rolling z-scores).
- **Labels**: next-h-day return **minus the universe average** (the model is
  not asked to call the market), **divided by the coin's own vol × √h**;
  overlapping labels weighted so each day counts 1/h.
- **Models**: ridge, shallow regularized gradient-boosted trees, rank
  ensemble; logistic for the meta-model. Refit every 30 days on labels that
  ended before the refit day. Changing future prices leaves earlier
  predictions bit-identical (`tests/test_ml_pipeline.py`).
- **Gate**: out-of-sample IC with NW t ≥ 2, positive in both halves and above
  the 30-day-momentum baseline in both halves; calibration by decile. The
  trend meta-model must beat the base rate's Brier in both halves.
- **Candidates it can queue**: `ml_rank` (pooled ranker inside the trend
  filter, momentum-style sizing) and `trend_meta` (the champion's BTC/ETH
  trend sized by P(a coin in its trend gains over h days), trained on the
  top-30 universe — thousands of trend episodes instead of BTC/ETH's ~90).
- Live: walk-forward fits take minutes, so the research loop saves ML
  candidates' weights (`reports/ml_weights/`); an ML champion's core holds the
  newest saved row, older than 3 days → hold.

**Online model (hourly bot)** — its vote used to be trusted on accuracy vs a
coin flip over the last 300 predictions: ~14 hours of ~21 correlated coins on
overlapping 24h labels, i.e. mostly the market's direction that day (a
zero-skill "always up" model read >60% about 40% of the time). It is now
trusted only on **skill over the majority-direction baseline, with
cluster-robust standard errors per label window, ≥ 20 windows**
(`online_model.trust`, `tests/test_ml_skill.py`).

## 4. Candidates — frozen configs, counted trials

- Queue: `python tools/candidates.py add NAME '{json}'`, `list`, `remove`;
  `GET /api/research/candidates`, `POST /api/control/research/candidates`.
  Configs are validated data (`challengers.validate_config`).
- A queued config starts its forward clock at the next research-loop run;
  changing it restarts the clock. Every variant ever tried counts in the
  deflated Sharpe (`app/engine/registry.py`). Parameter neighbours of the
  champion are **not** auto-generated: the plateau analysis showed which
  length wins is noise (PBO 0.46) — they would only add trials.

## 5. Promote — paired, always-valid, and the rule itself backtested

`app/engine/evidence.py`, `challengers.run` (daily):

- Backtest beats the champion in both halves; deflated Sharpe ≥
  `research_min_dsr` (0.97) over all trials (every variant in the research
  families; an empty family no longer counts as a phantom trial).
- **Forward test**: challenger scaled to the champion's volatility, day-by-day
  difference tested with a mixture SPRT — valid however often it is checked
  (false alarms checked every 5 days for 2 years: 1.3%). `better` → eligible
  after `research_min_forward_days` (30); `worse` → **retired**; undecided →
  eligible only after `research_max_forward_days` (180) with a positive paired
  difference. Settings: `research_forward_alpha` (0.05), `research_forward_tau`.
- **Process backtest**: `python tools/promotion_backtest.py` replays the
  promotion rule over history with only data known at each date — following
  it vs never switching, holding BTC+ETH and the best candidate in hindsight
  (`reports/promotion_backtest_latest.json`, weekly).
- **Costs and taxes** in every research report and on the scorecard
  (`app/engine/costs_tax.py`): taker / maker / low-fee venue / spot ETF
  (next-weekday fills + fund fee), FIFO tax lots with short/long-term rates and
  yearly netting vs holding the same coins after tax. Rates are settings
  (`tax_short_term_rate` 0.28, `tax_long_term_rate` 0.19) — illustrative, not
  tax advice.

## 6. Trade & watch — per-sleeve risk

| Control | Setting(s) |
|---|---|
| Kill switch / daily halt measure the main account **without the core's P&L**; stop new entries in the bot and exploration (exits still managed); one-time re-baseline on upgrade | `max_drawdown_kill`, `daily_loss_limit` |
| Core tracking monitor: held vs champion weights daily, off-target streak + tracking error, alert once per episode | `core_tracking_alert_days` (2), `core_tracking_alert_te` (0.05) |
| Polymarket on its own bankroll, separate from the main account (it used to get "the remainder" = the whole account with defaults) | `pm_start_cash` (500) |
| Polymarket evidence gate: bets only once resolved forecasts beat the market price (Brier per market, t ≤ −2, ≥ 100 markets); learned calibration used only if its walk-forward record wins | `pm_trade_mode` (auto), `pm_gate_min_markets`, `pm_calibration_mode` (auto) |
| Exploration reallocation on 30-day results shrunk toward 0 by their evidence | `exploration_shrink_tau` (0.001) |

## Speed (exact-parity)

| Change | Effect | Check |
|---|---|---|
| `math.fsum` stdev instead of `statistics.stdev` (exact rationals) in the shared features and the daily lab | cold replay 7.9 s → 4.3 s per 1,300 bars × 21 coins; 23× per call | within 1 ulp; replay report byte-identical |
| Replay per-bar data on disk, fingerprinted per bar on exactly the candle rows it reads (`app/backtest/bar_cache.py`) | rolling next-day replay 5.8 s → 0.9 s on 1,900 bars (more on a year) | persisted == fresh after new / revised / dropped bars |
| Ablation online-model seeds in parallel processes | ~3× on ≥ 4 cores (≈96% of its time) | parallel == serial |
| Research loop fits the vol forecast on the candidate's coins | ~100× faster; also fixes a backtest/live mismatch | weights == live core's |

`tests/test_speed_parity.py`.

## Schedules (replay loop)

| Job | Interval setting | Default |
|---|---|---|
| Market data sync | `data_sync_interval_sec` | daily |
| External series sync | `series_sync_interval_sec` | daily |
| Research loop (champion / challengers) | `research_loop_interval_sec` | daily |
| Signal screen, promotion backtest, ML lab | `research_extras_interval_sec` | weekly |
| Learner ablation (only while the hourly bot is active) | `learner_ablation_interval_sec` | weekly |

## What to expect

Most screened features, models and candidates will fail their gates — that is
the system working, not failing. Realistic wins: volatility forecasts 20–30%
better than naive, cross-coin ranking skill a point or two of IC, cheaper
execution and taxes for the core. A "much more accurate" price-direction model
is not on the table for liquid markets.
