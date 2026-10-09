"""Operator settings — persisted separately from state.json so they survive
account resets and are available before state restore runs at startup."""
import json, os, tempfile

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")

DEFAULTS = {
    # If True: kill-server flattens all positions before shutdown.
    # If False: positions are kept, saved, and restored/re-managed on restart.
    "flatten_on_shutdown": True,
    # If True: account (cash, positions, PnL, trade history) carries over
    # across restarts. If False: every restart begins with a fresh $100k
    # paper account — learned intelligence (models, bandit, Q-table, GA
    # champion, universe) is ALWAYS restored either way.
    "carry_equity": True,
    # Keep new entries paused across process restarts until explicitly resumed.
    "trading_paused": False,
    # If True the system may open SHORT positions (margin-style paper).
    "allow_shorts": True,
    # Trading stance: "passive" | "auto" | "aggressive".
    # auto = system scores current conditions and adapts on its own.
    "trade_mode": "auto",
    # Market-neutral pair hedging (2.7-sigma relative-strength divergence,
    # long the laggard / short the leader in equal notional). OFF by default:
    # each pair pays FOUR taker fills (~2.4% round trip) to harvest a spread
    # that reverted only ~0.01% on average over 23-min holds — it lost
    # -$17.4k of paper capital, the largest strategy loss in the book.
    "hedge_enabled": False,
    # Optional LLM advisor sleeve. When True (and an API key is set via
    # CRYPTOMIND_LLM_KEY / OPENAI_API_KEY) the advisor contributes ONE
    # directional vote that the bandit weights like any other strategy — it is
    # never the driver. Inert with no key configured (no cost, no effect).
    "llm_advisor_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    # LLM advisor credentials/config, settable from the dashboard (no file edit).
    # llm_api_key is a SECRET (stored in .secrets.json, masked in the API, never
    # committed). Blank model/base fall back to the Gemini defaults in the
    # advisor. Env vars (CRYPTOMIND_LLM_KEY/GEMINI_API_KEY/OPENAI_API_KEY,
    # CRYPTOMIND_LLM_MODEL, CRYPTOMIND_LLM_BASE) still override these if set.
    "llm_api_key": "",                # SECRET
    "llm_model": "",                  # blank = advisor default (gemini-2.5-flash)
    "llm_api_base": "",               # blank = advisor default (Gemini OpenAI-compat)
    # Optional ML-model advisor sleeve. When True (and a validated crypto_ml_lab
    # artifact exists at model_artifact/ or $CRYPTOMIND_MODEL_DIR) the advisor
    # contributes ONE directional vote that the bandit weights like any other
    # strategy — never the driver. Inert with no artifact present (no effect).
    "model_advisor_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    # Autonomous, metric-gated ML retraining (closes the crypto_ml_lab loop
    # without operator input). When True AND crypto_ml is installed, the system
    # periodically checks whether enough NEW labeled rows have matured and, if
    # so, runs export->validate->prepare->train->backtest, then AUTO-PROMOTES the
    # fresh artifact into the model advisor ONLY IF the backtest gates pass
    # (OOS Sharpe / trade count / beats-baseline). Enabling it on a machine
    # without crypto_ml simply records "lab not installed" and promotes
    # nothing. Governance note: promotion is measurement-gated, never on vibes.
    "ml_autotrain_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    "ml_autotrain_min_new_labels": 200,  # retrain only after N new matured labels
    "ml_autotrain_check_sec": 3600,      # how often the loop checks the trigger
    "ml_lab_cmd": "python -m crypto_ml.cli",  # base command to invoke the lab
    "ml_backtest_metrics_file": "metrics.json",  # metrics JSON the backtest writes
    # Promotion gate. PRIMARY (required) = profitability + drawdown, the metrics
    # crypto_ml's backtest actually emits (total_return / max_drawdown).
    "ml_gate_min_return": 0.0,           # gate: min OOS total return (0 = must not lose)
    "ml_gate_max_drawdown": 0.25,        # gate: max |drawdown| allowed (0.25 = 25%)
    # Purged, embargoed walk-forward validation — the trustworthy gate. The
    # promoted model is trained on ALL data, but promotion is gated on genuine
    # out-of-sample performance estimated by training throwaway models per fold
    # and scoring only on held-out slices. Prevents promoting overfit models.
    "ml_val_folds": 4,                   # expanding walk-forward folds
    "ml_val_embargo": 24,                # rows dropped between train/test (purge)
    "ml_gate_min_frac_folds_positive": 0.75,  # gate: fraction of OOS folds profitable
    "ml_gate_allow_backtest_fallback": False,  # if val can't run, gate on in-sample backtest (unsafe)
    # OPTIONAL gates — only enforced if the backtest reports them.
    "ml_gate_min_oos_sharpe": 0.5,       # gate: min out-of-sample Sharpe (if present)
    "ml_gate_min_oos_trades": 20,        # gate: min OOS trade count (if present)
    "ml_gate_require_beats_baseline": False,  # gate: require a beats-baseline flag
    # Predictive, self-learning early loss-cut. When True, a losing position the
    # system confidently expects to keep moving against it is cut before the
    # hard stop. Learns hold-vs-cut per market state from realized outcomes.
    # Master switch; whether it actually acts is `exit_advisor_mode` below.
    "exit_advisor_enabled": True,
    # ---- learner evidence gate (app/learn/gate.py) ----
    # Each learner: "off", "on", or "auto" = drives live decisions only while
    # the weekly learner ablation (tools/learner_ablation.py) shows it beating
    # the plain strategy in BOTH halves of the replay (sizing learners must
    # also beat a fixed size of the same average). Gated-off learners keep
    # learning in the background. On 2026-10-02 none passed.
    "bandit_mode": "auto",              # Thompson strategy weights (off = equal)
    "direction_learner_mode": "auto",   # regime bias + conformal HTF veto
    "exit_advisor_mode": "auto",        # predictive loss-cut
    "rl_risk_mode": "auto",             # Q-learning risk scale (off = 1.0)
    "ml_vote_mode": "auto",             # online-model committee vote
    "learner_ablation_interval_sec": 604800,   # re-run the ablation weekly
    "data_sync_interval_sec": 86400,    # refresh the market data store daily
    "series_sync_interval_sec": 86400,  # external series store (tools/series_sync.py)
    "research_loop_interval_sec": 86400,    # champion / challenger loop daily
    # signal screen + promotion-process backtest (tools/signal_screen.py,
    # tools/promotion_backtest.py)
    "research_extras_interval_sec": 604800,
    "learner_gate_max_age_days": 10,    # older evidence counts as none
    # Learns whether the DISCRETIONARY early exits (pattern_exit, signal-flip)
    # are actually earning their keep per regime, and dials their trigger bar
    # up/down accordingly. Never touches stop-loss/take-profit/kill-switch.
    "exit_throttle_enabled": True,
    # ---- Polymarket prediction-market sleeve (standalone, paper only) ----
    # Master switch for the Polymarket engine loop. When False the sleeve is
    # completely dormant (no fetch, no trading); the read-only "peek" endpoint
    # still works so the operator can inspect markets before enabling it.
    "polymarket_enabled": True,
    # When True (and the engine is enabled) the engine may OPEN paper bets that
    # clear the edge/cost gate. When False it only evaluates + records
    # would-open decisions, so the operator can watch it think before it trades.
    "pm_auto_trade": True,
    # Evidence gate for Polymarket bets (app/markets/polymarket/skill_gate.py):
    # "auto" = bet only once the forecast ledger shows the bot's probabilities
    # beating the market price (Brier, per resolved market, t <= -2) over at
    # least `pm_gate_min_markets` markets; forecasts keep being recorded either
    # way. "on" / "off" override.
    "pm_trade_mode": "auto",
    # learned calibration of the bot's probabilities from resolved markets
    # (app/markets/polymarket/calibration.py): "auto" = used for betting only
    # when its walk-forward record beats the market price (and the raw bot).
    "pm_calibration_mode": "auto",
    "pm_gate_min_markets": 100,
    # Polymarket runs its OWN paper bankroll, separate from the main account:
    # it starts with this much, keeps what it wins / loses, and never touches
    # crypto cash. Changing it takes effect on the next Polymarket reset.
    "pm_start_cash": 500,
    # ---- External-signal ingest (crawl4ai / Maxun / any standalone producer) ----
    # Master switch for accepting ingested external rows via /api/ingest/push.
    # When False the endpoint rejects pushes; the store is still readable.
    "ingest_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    # When True, the LLM advisor injects the freshest crawled web signal +
    # headlines (from app.data.ingest) into its per-asset context, so its
    # bandit-weighted vote reflects current news. Measured, never a blind copy.
    # Requires the LLM advisor to be enabled + keyed to have effect.
    "llm_web_context": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    # Strategy Researcher: autonomous discovery of NEW strategy shapes (see
    # app/learn/researcher.py). When True the system periodically searches a safe
    # rule DSL, validates each candidate on a purged walk-forward + deflated-Sharpe
    # gate, and AUTO-PROMOTES passers into the bandit-weighted `discovered`
    # ensemble arm (its live weight is still learned from realized PnL).
    # researcher_use_llm additionally asks the LLM advisor to PROPOSE
    # candidate specs (still gated identically); needs the LLM advisor keyed.
    "researcher_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    "researcher_use_llm": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    "researcher_candidates": 60,      # systematic candidates sampled per run
    # crawl4ai producer config (used by the standalone tools/crawl4ai_signal):
    # JSON mapping asset symbol -> list of URLs to crawl. Blank = nothing to do.
    "crawl4ai_sources": "",
    # When True, the crawl4ai producer FINDS its own news sources per traded asset
    # (Google News RSS, key-less) instead of only using the hand-curated
    # crawl4ai_sources map. Discovered signals still enter as MEASURED features.
    "crawl4ai_autodiscover": True,
    "crawl4ai_max_urls": 4,           # max articles discovered per asset per run
    # Run the crawl4ai producer INSIDE the app on a schedule (no manual CLI). It
    # crawls/scores sources off the hot path and pushes rows into the ingest seam.
    # The core imports the producer lazily + defensively, so a missing crawl4ai
    # (Apache-2.0, optional) never affects trading.
    "crawl4ai_producer_enabled": False,   # rebuild step 5: only fed the hourly bot (gated off by evidence)
    "crawl4ai_interval_sec": 900,     # producer cadence (15 min default)
    # Real on-chain execution. Inert unless this is True AND a pm_wallet_key
    # secret is present AND py-clob-client is installed (see execution.py).
    "pm_live_enabled": False,
    # Polygon wallet private key for live CLOB order signing. SECRET — stored in
    # the git-ignored .secrets.json only, masked in the API, never committed.
    "pm_wallet_key": "",              # SECRET
    # Pattern-aware exits. When True, a CONFIRMED reversal chart pattern forming
    # against an open position (e.g. a double top / head-&-shoulders / bearish
    # divergence on a long) tightens that position's stop, and cuts it outright
    # when the pattern is strong — layered ON TOP of the hard stop / take-profit
    # / trailing / loss-cut advisor, never replacing them.
    "pattern_exit_enabled": True,
    # Meme-coin trading. When True, a curated seed of Coinbase-listed meme
    # majors becomes eligible immediately and CoinGecko's meme category is
    # polled so hot new memes surface automatically — all under a tighter risk
    # envelope (smaller size, wider stops, concurrent + total exposure caps).
    # High variance by nature. Enabled per the operator's request to "see how
    # it plays out"; toggle off any time on the dashboard.
    "meme_trading_enabled": False,
    # Automatic strategy replay (app/backtest/replay.py): re-backtest the LIVE
    # strategy over the whole trading universe on fresh hourly history every
    # `replay_interval_sec`, AND immediately whenever a new coin joins the
    # universe. Results are logged, saved to reports/replay_latest.json and
    # pushed as an alert. Informational only — it never changes trading.
    # Entry order type for the paper broker: "maker" = post-only LIMIT order at
    # the signal price that fills only if price trades THROUGH it within
    # `maker_timeout_sec` (maker fee, no slippage; unfilled orders are
    # cancelled — some trades are missed, which is the honest cost of the
    # cheaper fee). "taker" = immediate market order (taker fee + slippage).
    # Stops/flips/loss-cuts are always taker; take-profits are maker when
    # entries are.
    "entry_order_type": "maker",
    # Chop filter (pause new entries while the whole market is no more
    # directional than noise): "off", "on", or "auto" = ON only while the
    # daily replay shows the filter beats no-filter in BOTH halves of its
    # history window (the bot gathers the evidence; nobody hand-fits it).
    # ON: it was the only layer that helped in both halves of the learner
    # ablation (2026-10-02).
    "chop_filter_mode": "on",
    # Learned trade filter (app/learn/trade_filter.py): "off", "on", or "auto"
    # = on only while the daily replay's walk-forward A/B shows it beating
    # no-filter in BOTH halves.
    "trade_filter_mode": "auto",
    # Optional CORE holding (app/strategies/core.py): percent of account equity
    # held in `core_assets` (comma-separated), split equally, beside the
    # trading bot. 0 = OFF. With `core_trend_filter`, each asset is held only
    # while its daily close is above its 50-day average.
    "core_allocation_pct": 0,
    # "champion": the core holds the research loop's champion strategy
    # (app/engine/challengers.py) — backtested, forward-tracked, promoted only
    # on evidence. "settings": the core_* settings below.
    "core_strategy": "champion",
    # The hourly trading bot: "off", "on", or "auto" = new entries only while
    # its daily replay is positive in BOTH halves (it was -75%/yr on the
    # 2026-10 replay). Existing positions are always managed to their exits.
    "hourly_bot_mode": "auto",
    # ---- exploration sleeve (app/strategies/exploration.py) ----
    # fast strategies trading live paper side by side on their own slices;
    # money follows 30-day live results; a member down `bench_drawdown` of its
    # capital is benched (tracked, can come back after `reinstate_days`)
    "exploration_enabled": False,
    "exploration_allocation_pct": 50,
    "exploration_bench_drawdown": 0.20,
    "exploration_reinstate_days": 30,
    # reallocation shrinks each member's 30-day result toward zero by how much
    # evidence it carries: prior sd of the true daily return (0.001 = 0.1%/day).
    # Smaller = money moves only on stronger evidence.
    "exploration_shrink_tau": 0.001,
    # core tracking monitor (app/strategies/core.py): alert when the core has
    # been off its champion's target for this many daily checks, or its
    # holdings tracking error vs target (annualized, 30 days) exceeds this.
    "core_tracking_alert_days": 2,
    "core_tracking_alert_te": 0.05,
    # champion / challenger promotion bar (app/engine/challengers.py)
    "research_min_forward_days": 30,
    "research_min_dsr": 0.97,
    # paired, always-valid forward test vs the champion (app/engine/evidence.py):
    # promote early on 'better', retire on 'worse'; undecided candidates are
    # eligible only after `research_max_forward_days` with a positive paired
    # difference. alpha = error rate however often it is checked; tau = the
    # mixture prior's scale for the standardized daily difference.
    "research_max_forward_days": 180,
    "research_forward_alpha": 0.05,
    "research_forward_tau": 0.1,
    # cost / after-tax scenarios in the research report (app/engine/costs_tax.py).
    # Illustrative US-style rates, NOT tax advice: set your own, ask your CPA.
    "tax_short_term_rate": 0.28,
    "tax_long_term_rate": 0.19,
    # BTC/ETH: the only coins with no survivorship question (top two the whole
    # time). On the survivorship-free universe no altcoin rule beat holding BTC.
    "core_assets": "BTC-USD,ETH-USD",
    "core_trend_filter": True,
    # days in the core holding's trend average (50 = faster, 100 = steadier);
    # core_assets may be "UNIVERSE" to apply the rule to every tracked coin
    # 125 = middle of the plateau (Sharpe 0.9-1.17 for 50..200 days over
    # 2017-2026); the single best cell was not chosen (selection is noise:
    # probability of backtest overfitting 0.46 across settings)
    "core_sma_days": 125,
    # buffer around the average: switch in above avg x 1.02, out below
    # avg x 0.98 — cut turnover ~13x -> ~5x a year and improved most settings
    "core_hysteresis": 0.02,
    # Core selection / sizing (app/backtest/daily_lab.py): "auto" = the
    # variant the weekly daily lab found beating trend+equal on Sharpe in both
    # halves of years of out-of-sample daily history; else trend+equal.
    # NOT auto: the daily lab's winners (momentum, vol-target) were measured on
    # today's coins = survivorship bias; on the point-in-time universe incl.
    # delisted coins they did not beat holding BTC (rebuild step 3).
    "core_selection": "trend",          # trend | momentum | rank | auto
    "core_sizing": "equal",             # equal | inverse_vol | vol_target | auto
    "core_top_k": 5,                    # coins held by momentum / rank
    "core_vol_target": 0.5,             # annual volatility for vol_target sizing
    # weekly picks split into this many slices re-picked on different days
    # (removes pick-day luck: one weekday vs another moved Sharpe 1.03-1.37)
    "core_tranches": 7,
    "daily_lab_interval_sec": 604800,   # re-run the daily lab weekly
    "daily_lab_max_age_days": 14,
    "replay_enabled": True,
    # Let the replay put the whole-portfolio risk brake on (tunable
    # replay_brake_mult, default half size) while the strategy replays negative
    # in BOTH halves. Off = replay stays purely informational.
    "replay_brake_enabled": True,
    "replay_interval_sec": 86400,
    # How much hourly history the replay downloads per coin. ~40 days is one
    # market regime; a year covers rallies, sell-offs and chop. A coin listed
    # more recently simply contributes what exists.
    "replay_history_days": 365,
    # Periodic full-state export. When True, the system writes a timestamped
    # JSON report (the same document as GET /api/export) to the reports/ folder
    # every `auto_export_interval_sec` seconds. The interval is operator-tunable
    # from the dashboard slider and takes effect on the next cycle (no restart).
    "auto_export_enabled": True,
    "auto_export_interval_sec": 14400,  # default: every 4 hours (each report is ~2.7 MB)

    # ---------------- Invo copy-signal study (measurement only) ----------------
    # Master switch for the Invo positioning signal *study*. This NEVER wires the
    # signal into live trading — it only enables collecting snapshots and running
    # the edge study from the dashboard. The signal becomes a learner feature
    # only if/when the study earns it (separate, explicit step).
    "invo_enabled": False,
    "invo_api_base": "",              # e.g. https://app.invoapp.com
    "invo_token": "",                 # SECRET — bearer token for YOUR authorized session
    "invo_method": "GET",             # leaderboard request method: GET or POST (e.g. get_users is POST)
    "invo_body": "",                  # JSON body sent with a POST leaderboard request
    "invo_leaderboard_path": "",      # path returning ranked traders
    "invo_positions_tmpl": "",        # optional per-trader positions path, {id} placeholder
    "invo_positions_method": "GET",   # per-trader positions method: GET or POST
    "invo_positions_body": "",        # JSON body for a POST positions request; {id} placeholder
    "invo_top_n": 25,
    "invo_interval_sec": 300,
    # response->schema field map (dotted paths allowed, e.g. "data.items"):
    "invo_map_list": "",              # key holding the list of traders (blank = response is the list)
    "invo_map_id": "id",             # trader id field
    "invo_map_score": "",            # optional quality score field (e.g. winRate)
    "invo_map_positions": "",        # inline positions field (blank = use positions_tmpl call)
    "invo_map_asset": "coin",        # position symbol field
    "invo_map_side": "side",         # position direction field
    "invo_map_long_value": "long",   # value of the side field that means LONG
    "invo_map_size": "sizeUsd",      # position notional (USD) field
    "invo_map_leverage": "",         # optional leverage field
    # ---- optional auto token-refresh (so a short-lived JWT never stalls the
    # collector). You capture the refresh call ONCE; we mint fresh access tokens
    # on demand. Only a refresh token is stored — never your password. ----
    "invo_refresh_path": "",         # refresh endpoint (path or full URL); blank = disabled
    "invo_refresh_token": "",        # SECRET — long-lived refresh token
    "invo_refresh_body": "",         # JSON body template w/ {refresh_token}; blank = send it as a Bearer header
    "invo_token_path": "access_token",   # dotted path to the NEW access token in the refresh response
    "invo_refresh_rotates_path": "",     # optional dotted path to a rotated refresh token (if the API rotates it)
    # study parameters:
    "invo_horizon_hours": 4.0,
    "invo_rank_decay": 1.0,
    "invo_use_score": False,
}

STR_KEYS = {"trade_mode": {"passive", "auto", "aggressive"},
            "entry_order_type": {"maker", "taker"},
            "chop_filter_mode": {"off", "on", "auto"},
            "trade_filter_mode": {"off", "on", "auto"},
            "bandit_mode": {"off", "on", "auto"},
            "direction_learner_mode": {"off", "on", "auto"},
            "exit_advisor_mode": {"off", "on", "auto"},
            "rl_risk_mode": {"off", "on", "auto"},
            "ml_vote_mode": {"off", "on", "auto"},
            "core_selection": {"trend", "momentum", "rank", "auto"},
            "core_strategy": {"champion", "settings"},
            "hourly_bot_mode": {"off", "on", "auto"},
            "pm_trade_mode": {"off", "on", "auto"},
            "pm_calibration_mode": {"off", "auto"},
            "core_sizing": {"equal", "inverse_vol", "vol_target", "auto"},
            "invo_method": {"GET", "POST"},
            "invo_positions_method": {"GET", "POST"}}

# Free-text string settings (stored verbatim, trimmed).
TEXT_KEYS = {
    "llm_model", "llm_api_base", "invo_body",
    "invo_api_base", "invo_leaderboard_path", "invo_positions_tmpl",
    "invo_positions_body",
    "invo_map_list", "invo_map_id", "invo_map_score", "invo_map_positions",
    "invo_map_asset", "invo_map_side", "invo_map_long_value",
    "invo_map_size", "invo_map_leverage",
    "invo_refresh_path", "invo_refresh_body", "invo_token_path",
    "invo_refresh_rotates_path",
    "ml_lab_cmd", "ml_backtest_metrics_file",
    # crawl4ai producer: JSON mapping of asset symbol -> [urls to crawl], e.g.
    # {"BTC": ["https://…/news"], "ETH": ["https://…"]}. Consumed by the
    # standalone tools/crawl4ai_signal producer, not the core.
    "crawl4ai_sources",
    "core_assets",
}

# Secret settings: stored, but MASKED in the public payload and never clobbered
# by an empty save (only overwritten when a new non-empty value is supplied).
SECRET_KEYS = {"invo_token", "invo_refresh_token", "llm_api_key", "pm_wallet_key"}

# Float settings: (min, max) inclusive clamp.
FLOAT_KEYS = {
    "invo_horizon_hours": (0.25, 168.0),
    "invo_rank_decay": (0.0, 4.0),
    "ml_gate_min_oos_sharpe": (-10.0, 10.0),
    "ml_gate_min_return": (-1.0, 1000000.0),
    "ml_gate_max_drawdown": (0.0, 1.0),
    "ml_gate_min_frac_folds_positive": (0.0, 1.0),
    "core_vol_target": (0.1, 1.5),
    "core_hysteresis": (0.0, 0.2),
    "exploration_bench_drawdown": (0.05, 0.9),
    "exploration_shrink_tau": (0.0, 0.05),
    "core_tracking_alert_te": (0.0, 1.0),
    "research_min_dsr": (0.5, 0.999),
    "research_forward_alpha": (0.001, 0.2),
    "research_forward_tau": (0.01, 1.0),
    "tax_short_term_rate": (0.0, 0.6),
    "tax_long_term_rate": (0.0, 0.6),
}

# Integer settings: (min, max) inclusive clamp. Everything else is treated as
# a boolean toggle.
INT_KEYS = {
    "auto_export_interval_sec": (30, 86400),   # 30s .. 24h
    "invo_top_n": (1, 200),
    "invo_interval_sec": (30, 86400),
    "ml_autotrain_min_new_labels": (10, 1000000),
    "ml_autotrain_check_sec": (60, 86400),     # 1min .. 24h
    "ml_gate_min_oos_trades": (0, 1000000),
    "ml_val_folds": (2, 12),
    "ml_val_embargo": (0, 1000000),
    "researcher_candidates": (10, 500),
    "crawl4ai_max_urls": (1, 15),
    "crawl4ai_interval_sec": (60, 86400),    # 1min .. 24h
    "replay_interval_sec": (3600, 604800),   # 1h .. 7d
    "replay_history_days": (30, 1095),       # 1 month .. 3 years
    "core_allocation_pct": (0, 100),
    "core_sma_days": (10, 250),
    "learner_ablation_interval_sec": (86400, 2592000),   # 1d .. 30d
    "data_sync_interval_sec": (3600, 604800),
    "series_sync_interval_sec": (3600, 604800),
    "research_loop_interval_sec": (86400, 2592000),
    "research_extras_interval_sec": (86400, 2592000),
    "learner_gate_max_age_days": (1, 60),
    "core_top_k": (2, 15),
    "core_tranches": (1, 7),
    "exploration_allocation_pct": (0, 100),
    "exploration_reinstate_days": (1, 365),
    "pm_gate_min_markets": (10, 100000),
    "core_tracking_alert_days": (1, 30),
    "research_min_forward_days": (7, 365),
    "research_max_forward_days": (30, 1095),
    "daily_lab_interval_sec": (86400, 2592000),
    "daily_lab_max_age_days": (1, 60),
    "pm_start_cash": (5, 10_000_000),
}


# Secrets (e.g. the Invo bearer token) are stored SEPARATELY from settings.json —
# in a git-ignored file — so a tracked config file can never leak a credential.
SECRETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".secrets.json")

_settings = None


def _load_secrets() -> dict:
    try:
        with open(SECRETS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_secrets(d: dict):
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SECRETS_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.chmod(tmp, 0o600)          # owner-only, like api_token.txt
        os.replace(tmp, SECRETS_PATH)
    except Exception:
        pass


def load():
    global _settings
    if _settings is None:
        _settings = dict(DEFAULTS)
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH) as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k not in DEFAULTS or k in SECRET_KEYS:  # secrets never from here
                        continue
                    _settings[k] = _coerce(k, v)
        except Exception:
            pass
        # overlay secrets from the git-ignored store
        secrets = _load_secrets()
        for k in SECRET_KEYS:
            if k in secrets:
                _settings[k] = str(secrets[k])
    return _settings


def _coerce_int(key, v):
    """Clamp an integer setting to its [min, max] range; fall back to the
    default on garbage input."""
    lo, hi = INT_KEYS[key]
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return DEFAULTS[key]


def _coerce_float(key, v):
    lo, hi = FLOAT_KEYS[key]
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return DEFAULTS[key]


def _coerce(key, v):
    """Coerce one setting by its category. Returns None to signal 'ignore'."""
    if key in STR_KEYS:
        return v if v in STR_KEYS[key] else None
    if key in TEXT_KEYS:
        return str(v).strip()
    if key in SECRET_KEYS:
        return str(v)
    if key in FLOAT_KEYS:
        return _coerce_float(key, v)
    if key in INT_KEYS:
        return _coerce_int(key, v)
    if isinstance(v, str):                 # "false" / "0" / "off" -> False
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


MASK = "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"   # ••••••••


def public():
    """Settings for the API/UI: secrets masked, with a companion '<key>_set'
    boolean so the UI can show whether a secret is configured."""
    s = dict(load())
    for k in SECRET_KEYS:
        s[k + "_set"] = bool(s.get(k))
        s[k] = MASK if s.get(k) else ""
    # Non-secret computed flag: whether the ML model advisor actually has a
    # trained artifact to serve. Enabling the toggle alone is inert without one,
    # so the UI needs this to show "on (active)" vs "on (needs artifact)".
    try:
        from .learn.model_advisor import advisor as _model_advisor
        s["model_artifact_present"] = bool(_model_advisor.artifact_present())
    except Exception:
        s["model_artifact_present"] = False
    return s


def get(key):
    return load()[key]


def update(changes: dict):
    s = load()
    secret_dirty = False
    for k, v in changes.items():
        if k not in DEFAULTS:
            continue
        if k in SECRET_KEYS:
            # empty save / the mask never clobber a stored secret
            if not v or v == MASK:
                continue
            s[k] = str(v)
            secret_dirty = True
            continue
        coerced = _coerce(k, v)
        if coerced is not None:
            s[k] = coerced
    # persist secrets to the git-ignored store; strip them from settings.json
    if secret_dirty:
        _save_secrets({k: s[k] for k in SECRET_KEYS if s.get(k)})
    try:
        public_persist = {k: v for k, v in s.items() if k not in SECRET_KEYS}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SETTINGS_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(public_persist, f)
        os.replace(tmp, SETTINGS_PATH)
    except Exception:
        pass
    return s
