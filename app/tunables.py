"""Runtime-tunable parameters — every operational knob in one registry,
editable live from the dashboard Settings tab and persisted to tunables.json.

Design principle (from the research the operator supplied): optimize for
NET PROFIT AFTER COSTS, never raw accuracy. Cost parameters are therefore
first-class tunables so the operator can stress the system at higher fee
assumptions at any time.
"""
import json, os, tempfile

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tunables.json")

def T(default, lo, hi, step, group, label, desc, integer=False):
    return {"default": default, "min": lo, "max": hi, "step": step,
            "group": group, "label": label, "desc": desc, "int": integer}

TUNABLES = {
    # ---- risk ----
    "risk_per_trade":    T(0.0075, 0.001, 0.05, 0.0005, "risk", "Risk per trade",
                           "Fraction of equity risked to the stop on each full-size trade"),
    "max_position_pct":  T(0.20, 0.02, 0.50, 0.01, "risk", "Max position %",
                           "Cap on a single position as a fraction of equity"),
    "max_open_positions":T(4, 1, 12, 1, "risk", "Max open positions",
                           "Base cap on concurrent positions (stance scales it)", True),
    "max_gross_exposure":T(0.60, 0.10, 1.00, 0.05, "risk", "Max gross exposure",
                           "Total deployed capital cap as a fraction of equity"),
    "stop_atr_mult":     T(3.0, 0.5, 6.0, 0.1, "risk", "Stop ATR mult",
                           "Initial stop distance in ATR multiples"),
    "take_profit_atr_mult": T(6.0, 1.0, 10.0, 0.1, "risk", "Take-profit ATR mult",
                           "Target distance in ATR multiples (cost floor may widen it)"),
    "trail_atr_mult":    T(3.0, 0.5, 6.0, 0.1, "risk", "Trailing stop ATR mult",
                           "Trailing stop distance from the high/low-water mark"),
    "swing_atr_bars":    T(1, 1, 48, 1, "risk", "Swing ATR timeframe (x native bars)",
                           "Native candles (1h since the 1h switch) folded into one "
                           "ATR bar for sizing stops/targets: 1 = native 1h, 4 = 4h. Bigger = wider, "
                           "more cost-viable swing stops (a different strategy, "
                           "not a looser scalp).", True),
    "max_drawdown_kill": T(0.15, 0.03, 0.50, 0.01, "risk", "Kill-switch drawdown",
                           "Peak-to-trough drawdown that trips the kill switch"),
    "daily_loss_limit":  T(0.05, 0.01, 0.25, 0.005, "risk", "Daily loss halt",
                           "Daily loss that halts new entries until tomorrow"),
    "cooldown_sec":      T(43200, 0, 172800, 60, "risk", "Re-entry cooldown (s)",
                           "Seconds before re-entering a product after an exit", True),
    "max_spread_bps":    T(0, 0, 200, 1, "risk", "Max entry spread (bps)",
                           "Skip a NEW entry when the live bid/ask spread is wider "
                           "than this (basis points) — a wide spread eats the edge "
                           "before the trade even moves. 0 = disabled (freqtrade "
                           "SpreadFilter idea).", True),
    "min_hold_sec":      T(43200, 0, 172800, 30, "risk", "Minimum hold (s)",
                           "A position younger than this is exempt from signal-flip "
                           "and trailing-giveback exits (its hard stop/target still "
                           "apply) — stops same-tick churn", True),
    "max_entries_per_hour": T(2, 0, 120, 1, "risk", "Max entries / hour",
                           "Hard cap on NEW position opens per rolling hour "
                           "(0 = unlimited); anti-overtrading circuit", True),
    "maker_fee_rate":    T(0.004, 0.0, 0.02, 0.0001, "costs", "Maker fee rate",
                           "Fee per side for a resting LIMIT (maker) fill — used when "
                           "the `entry_order_type` setting is 'maker'. Set to your "
                           "exchange's current maker rate."),
    "maker_timeout_sec": T(3600, 60, 86400, 60, "costs", "Limit order timeout (s)",
                           "A limit entry that hasn't filled after this long is "
                           "cancelled (no trade, no fee)", True),
    "chop_er_min":       T(0.12, 0.0, 0.6, 0.01, "signals", "Chop filter threshold",
                           "Market-wide trendiness (median efficiency ratio across "
                           "coins) below which the chop filter pauses NEW entries. "
                           "0.12 ~= a random walk over 48 bars, i.e. 'no more "
                           "directional than noise'."),
    "chop_lookback_bars": T(48, 12, 240, 1, "signals", "Chop filter lookback (bars)",
                           "Bars over which market trendiness is measured", True),
    "filter_strictness": T(1.0, 0.5, 1.5, 0.05, "signals", "Trade filter strictness",
                           "The trade filter takes a signal only if its predicted "
                           "chance of a net-of-cost win is >= the historical base "
                           "win rate x this (1.0 = skip below-average signals)"),
    "replay_brake_mult": T(0.5, 0.0, 1.0, 0.05, "risk", "Replay brake",
                           "Position-risk multiplier applied to NEW entries while the "
                           "automatic strategy replay is negative in BOTH halves of "
                           "its window (over >= replay_brake_min_trades trades). "
                           "1.0 = brake off; 0 = stop opening new positions."),
    "replay_brake_min_trades": T(30, 0, 1000, 1, "risk", "Replay brake min trades",
                           "The replay must contain at least this many trades before "
                           "it can engage the brake (too few = noise)", True),
    "trail_giveback_pct": T(0.0, 0.0, 0.90, 0.05, "risk", "Peak-giveback exit",
                           "OFF by default: in the 1h replay it sold winners at ~+1% "
                           "net while losers ran to the stop (-15.8% vs -6.1% "
                           "with it off). Close a WINNING position once it gives back this "
                           "fraction of its peak unrealized gain (price-basis). "
                           "0 = disabled. Arms only after a real move (see arm %)."),
    "trail_giveback_arm_pct": T(0.03, 0.0, 0.10, 0.001, "risk", "Giveback arm move",
                           "Peak unrealized gain (as a fraction of entry price) "
                           "required before the peak-giveback exit can trigger — "
                           "prevents arming on noise"),
    # ---- protections (time-boxed circuit breakers) ----
    "protect_stopguard_trades": T(4, 0, 30, 1, "protections", "StoplossGuard trades",
                           "Lock ALL pairs when at least this many losing stop-outs "
                           "occur inside the lookback window. 0 disables.", True),
    "protect_stopguard_lookback_sec": T(3600, 300, 86400, 300, "protections",
                           "StoplossGuard lookback (s)",
                           "Rolling window scanned for clustered stop-outs", True),
    "protect_stopguard_lock_sec": T(3600, 300, 86400, 300, "protections",
                           "StoplossGuard lock (s)",
                           "How long trading is halted for ALL pairs once the "
                           "stop-out cluster trips the guard", True),
    "protect_lowprofit_trades": T(3, 0, 30, 1, "protections", "LowProfitPairs trades",
                           "Minimum trades for a single coin inside the window before "
                           "its net edge is judged. 0 disables.", True),
    "protect_lowprofit_lookback_sec": T(86400, 300, 604800, 300, "protections",
                           "LowProfitPairs lookback (s)",
                           "Rolling window over which a coin's net PnL is summed", True),
    "protect_lowprofit_required": T(0.0, -0.20, 0.20, 0.005, "protections",
                           "LowProfitPairs required net",
                           "Lock a coin whose net PnL over the window (as a fraction "
                           "of staked notional) is BELOW this. 0 = lock net-losers."),
    "protect_lowprofit_lock_sec": T(21600, 300, 604800, 300, "protections",
                           "LowProfitPairs lock (s)",
                           "How long the under-performing coin is locked out", True),
    "protect_maxdd_trades": T(10, 0, 100, 1, "protections", "MaxDrawdown trades",
                           "Minimum trades in the window before the temporary "
                           "drawdown halt can arm. 0 disables.", True),
    "protect_maxdd_lookback_sec": T(43200, 300, 604800, 300, "protections",
                           "MaxDrawdown lookback (s)",
                           "Rolling window whose realized-PnL curve is measured", True),
    "protect_maxdd_fraction": T(0.10, 0.02, 0.50, 0.01, "protections",
                           "MaxDrawdown fraction",
                           "Realized peak-to-trough drawdown over the window that "
                           "trips a TEMPORARY, auto-recovering halt (softer tier "
                           "below the permanent kill switch)"),
    "protect_maxdd_lock_sec": T(7200, 300, 86400, 300, "protections",
                           "MaxDrawdown lock (s)",
                           "How long trading is halted after the temporary "
                           "drawdown halt trips", True),
    # ---- decider-health guardian (safe mode / preflight) ----
    "safe_mode_fail_threshold": T(4, 1, 20, 1, "risk", "Safe-mode fail threshold",
                           "Consecutive decision-loop failures that engage safe "
                           "mode (new entries suspended, open risk still managed)", True),
    "safe_mode_recover_sec": T(120, 0, 3600, 10, "risk", "Safe-mode recover dwell (s)",
                           "How long health must hold before safe mode clears "
                           "itself (anti-flap)", True),
    "data_stale_sec":    T(180, 30, 3600, 10, "risk", "Data-stale threshold (s)",
                           "Market feed older than this counts as unhealthy and "
                           "engages safe mode", True),
    "min_notional":      T(50, 10, 5000, 10, "risk", "Min trade notional ($)",
                           "Trades smaller than this are skipped", True),
    "funding_extreme":   T(0.0008, 0.0001, 0.005, 0.0001, "risk", "Funding-rate block",
                           "Per-8h funding rate beyond which crowded-side entries are blocked"),
    "liq_cap_pct":       T(0.01, 0.001, 0.10, 0.001, "risk", "Per-coin liquidity cap",
                           "Max position as a fraction of the coin's ~24h dollar volume (thin-coin protection)"),
    "min_price":         T(0.01, 0.0, 5.0, 0.005, "risk", "Min tradable price ($)",
                           "Reject entries on assets priced below this (junk/penny-coin protection)"),
    "price_spike_mult":  T(3.0, 1.5, 20.0, 0.5, "risk", "Price-sanity band (x median)",
                           "Reject entries when price is above this multiple of, or below 1/x of, its recent median"),
    # ---- costs ----
    "fee_rate":          T(0.005, 0.0, 0.02, 0.0005, "costs", "Fee rate (per side)",
                           "Taker fee per fill; 0.005 = 0.5% (realistic retail)"),
    "slippage_bps":      T(10, 0, 100, 1, "costs", "Slippage (bps/side)",
                           "Simulated slippage per side in basis points", True),
    "cost_multiple":     T(2.5, 1.0, 6.0, 0.1, "costs", "Cost-viability multiple",
                           "Take-profit must clear round-trip costs by this multiple"),
    # ---- signals & exploration ----
    "min_confidence":    T(0.45, 0.10, 0.90, 0.01, "signals", "Confidence gate (neutral)",
                           "Base signal confidence needed for a full-size entry"),
    "explore_min_confidence": T(0.25, 0.05, 0.60, 0.01, "signals", "Probe floor",
                           "Minimum confidence for small exploration probes"),
    "explore_prob":      T(0.0, 0.0, 1.0, 0.01, "signals", "Probe probability",
                           "Chance per decision tick of firing one probe trade"),
    "explore_size_factor": T(0.4, 0.05, 1.0, 0.05, "signals", "Probe size factor",
                           "Probe size as a fraction of a normal position"),
    # ---- performance-weighted universe (freqtrade PerformanceFilter idea) ----
    "perf_filter_enabled": T(1, 0, 1, 1, "signals", "Performance filter",
                           "Weight discovered-coin universe ranking by each coin's "
                           "own realized trade performance (winners up, chronic "
                           "losers down). 0 = rank by heat only.", True),
    "perf_filter_lookback_sec": T(604800, 3600, 2592000, 3600, "signals",
                           "Performance lookback (s)",
                           "Rolling window of closed trades used to score a coin's "
                           "realized edge (default 7 days)", True),
    "perf_filter_min_trades": T(3, 1, 20, 1, "signals", "Performance min trades",
                           "Minimum closed trades for a coin before its realized "
                           "edge adjusts its universe ranking", True),
    # ---- learning ----
    "bandit_decay_gamma": T(0.9997, 0.90, 1.0, 0.001, "learning", "Bandit forgetting γ",
                           "Per-cycle decay on bandit n, variance, AND mean (idle "
                           "edge forgets toward 0; 1.0 = never forget; 0.9997 ≈ 5-day "
                           "half-life at ~3-min cycles, 0.995 ≈ 7h). Lower = adapts "
                           "faster to regime change but forgets the sparse trade "
                           "evidence before it can accumulate"),
    "evolve_every_sec":  T(1200, 120, 21600, 60, "learning", "GA cadence (s)",
                           "Seconds between genetic-evolution runs (universe rotates)", True),
    # ---- evolution (GA champion promotion gate) ----
    # These are the bars a challenger genome must clear on a purged walk-forward
    # to be promoted to live voting. They are strict on purpose (anti-overfit);
    # lower them to evolve more readily, at the cost of promoting on thinner /
    # less certain evidence. NOTE: a net-LOSING out-of-sample genome is never
    # promoted regardless of these — that floor is hardcoded.
    "ga_min_oos_trades": T(20, 4, 100, 1, "evolution", "Min OOS trades",
                           "A challenger must make at least this many trades across the "
                           "purged walk-forward windows to be judged (evidence floor). "
                           "Lower = evolve on thinner evidence.", True),
    "ga_min_frac_positive": T(0.60, 0.0, 1.0, 0.05, "evolution", "Min positive OOS windows",
                           "Fraction of out-of-sample windows the challenger must be "
                           "profitable in (0.6 = 3 of 5). Lower = accept less consistency"),
    "ga_dsr_min": T(0.90, 0.0, 0.999, 0.01, "evolution", "Deflated-Sharpe gate",
                           "Probability — after pricing in the GA's multiple-testing bias "
                           "(all pop×gens genomes) — that the challenger's true OOS Sharpe "
                           "is positive. 0.90 is strict/anti-overfit; lower to promote more "
                           "readily"),
    "ga_fallback_sharpe_min": T(0.5, 0.0, 3.0, 0.1, "evolution", "Fallback Sharpe gate",
                           "When too few trades exist to compute a deflated Sharpe, require "
                           "at least this pooled OOS Sharpe instead"),
    "ga_max_wf_drawdown": T(0.15, 0.02, 0.60, 0.01, "evolution", "Max OOS drawdown",
                           "Worst-window drawdown a challenger may show under the fallback "
                           "Sharpe gate"),
    "ga_history_chunks": T(3, 3, 24, 1, "evolution", "GA history depth (chunks)",
                           "Coinbase candle pages (300 hourly bars each) fetched per "
                           "evolution run. More = longer backtest, at the cost of more API "
                           "calls. MEASURED (2026-09): raising 3->8 did NOT reliably improve "
                           "the deflated Sharpe (SOL +, DOGE/UNI -) because a longer window "
                           "spans more regimes the single-timeframe TA can't fit uniformly. "
                           "Left at 3; raise to experiment. 8 ≈ 2400 bars ≈ 100 days.", True),
    "ga_granularity": T(3600, 900, 3600, 2700, "evolution", "GA candle granularity (sec)",
                           "Candle timeframe the GA trains on, in seconds (snapped to the "
                           "nearest Coinbase step). 3600 = 1h (default), 900 = 15m. Lower TF "
                           "packs ~4x more bars/trades into the same calendar window, which "
                           "directly relieves the walk-forward 'too_few_trades' floor that "
                           "blocks regime-filtered strategies. Pair a lower TF with a higher "
                           "ga_history_chunks so the calendar span stays reasonable.", True),
    "ga_cross_sectional": T(0, 0, 1, 1, "evolution", "Cross-sectional (universe) GA",
                           "Evolve & validate ONE genome POOLED across the whole product "
                           "basket instead of per-product. MEASURED (2026-09): the trend edge "
                           "is low-frequency (a few quality trades per product) so per-product "
                           "runs die on 'too_few_trades'; pooling meets the evidence floor by "
                           "BREADTH and validated the FIRST genomes to approach the DSR gate "
                           "(pooled DSR up to ~0.77-0.90 with the regime filter on). Pair with "
                           "ga_regime_filter=1. Promotes a universe portfolio applied to every "
                           "product.", True),
    "ga_universe_basket": T(6, 2, 20, 1, "evolution", "Universe basket size",
                           "How many products (from the configured universe) the "
                           "cross-sectional GA pools over. More = more independent evidence, "
                           "at the cost of more API calls and slower runs.", True),
    "ga_universe_every_n": T(3, 1, 20, 1, "evolution", "Universe run cadence (every Nth)",
                           "When cross-sectional is on, run a universe-pooled GA every Nth "
                           "auto-evolution cycle and per-product runs on the others, so both "
                           "run ALONGSIDE each other on a rotation. 1 = universe every cycle "
                           "(no per-product). Ignored when cross-sectional is off.", True),
    "ga_regime_filter": T(0, 0, 1, 1, "evolution", "Regime filter (efficiency ratio)",
                           "Let evolution add a Kaufman efficiency-ratio entry filter "
                           "(er_n / er_min genes): only trade when the trailing trend is "
                           "strong enough, skipping low-quality chop. MEASURED (2026-09): the "
                           "trend-following edge is heavily concentrated in trending regimes "
                           "(pooled TREND trades ~+1.7%/trade vs ~+0.3% in chop). Unlike the "
                           "HTF/vol gates it is a CONTINUOUS threshold that need not starve "
                           "trades. See ga_market_structure for the (default-off) HTF/vol "
                           "gates.", True),
    "ga_market_structure": T(0, 0, 1, 1, "evolution", "Market-structure genes",
                           "Enable the candle-derived higher-timeframe-trend and "
                           "volatility-regime genome genes. OFF by default: MEASURED "
                           "(2026-09) these gates starve the walk-forward of trades "
                           "(too_few_trades) and collapse the deflated Sharpe to ~0, because "
                           "train fitness rewards the fewer/cleaner trades they produce while "
                           "the OOS min-trades floor then rejects them. Turn on only to "
                           "experiment further.", True),
    "oi_growth_threshold": T(0.15, 0.02, 1.0, 0.01, "learning", "OI-growth discovery",
                           "24h open-interest growth (as a fraction) that flags a "
                           "coin as a capital-flow discovery candidate and boosts "
                           "its universe heat"),
    "trade_weight":      T(2.0, 0.5, 10.0, 0.5, "learning", "Trade attribution weight",
                           "Scale on closed-trade net PnL fed to the bandit "
                           "(the premium, real-money learning signal)"),
    "signal_learn_weight": T(0.15, 0.0, 1.0, 0.01, "learning", "Signal-stream weight",
                           "Scale on the NET-of-cost signal-scoring stream fed "
                           "to the bandit. This is abundant (100-1000x the trade "
                           "count) but lower quality (not a real fill), so "
                           "it's discounted well below trade_weight. 0 disables it — "
                           "restoring the old dashboard-only behaviour."),
    "signal_learn_clip":  T(0.04, 0.001, 0.10, 0.001, "learning", "Signal-stream clip",
                           "Clip each net-of-cost signal forward-return (24h horizon) to +/- this before "
                           "feeding the bandit, so one violent bar can't dominate the "
                           "abundant-but-noisy signal stream"),
    "loss_lesson_mult":  T(1.0, 1.0, 10.0, 0.5, "learning", "Loss lesson multiplier",
                           "A losing trade teaches N-times harder than a winner. Keep at 1.0: "
                           "anything higher biases the bandit against low-win-rate, "
                           "high-payoff sleeves (trend/breakout), which are exactly "
                           "the ones that clear costs"),
    "skip_learn_weight": T(0.5, 0.0, 2.0, 0.05, "learning", "Skip counterfactual weight",
                           "Scale on the NET-of-cost counterfactual return of an "
                           "ACTIONABLE conviction signal that was gated out (risk/"
                           "guardian/funding/notional). It teaches the bandit what "
                           "the trade WOULD have paid — a real, cost-aware lesson "
                           "from decisions the book was throttled out of, sitting "
                           "between the abundant gross signal stream and a real "
                           "fill in quality. 0 disables counterfactual credit."),
    # ---- predictive loss-cut exit advisor ----
    "exit_cut_threshold": T(0.004, 0.001, 0.05, 0.001, "learning", "Loss-cut threshold",
                           "Cut a losing position when its blended expected next-"
                           "horizon return (model + learned state value) is more "
                           "adverse than this fraction"),
    "exit_min_loss_pct": T(0.003, 0.0, 0.05, 0.001, "learning", "Loss-cut min loss",
                           "Only the predictive loss-cut can fire once a position "
                           "is at least this far underwater (buffer vs noise)"),
    "exit_ml_weight":    T(1.0, 0.0, 3.0, 0.1, "learning", "Loss-cut model weight",
                           "How strongly the ML forward view counts vs the learned "
                           "state value in the cut decision"),
    "exit_conformal_ceiling": T(0.0, -0.02, 0.02, 0.001, "learning",
                           "Loss-cut conformal ceiling",
                           "Once the online model's uncertainty band is conformally "
                           "CALIBRATED, only cut when even the optimistic (upper) end "
                           "of the position-frame forward-return interval is at/below "
                           "this. Blocks premature cuts of losers the calibrated band "
                           "still gives a real chance of bouncing. 0 = require the "
                           "whole interval non-positive; higher = cut more readily"),
    "exit_horizon_sec":  T(43200, 300, 172800, 60, "learning", "Loss-cut learn horizon (s)",
                           "Forward window used to score hold-vs-cut decisions "
                           "against what price actually did next", True),
    # ---- direction (long vs short) learner ----
    "direction_bias_gain": T(8.0, 0.0, 30.0, 0.5, "learning", "Direction bias gain",
                           "How strongly the learned per-regime directional edge "
                           "nudges the composite toward the side that has paid"),
    "direction_bias_cap": T(0.25, 0.0, 0.8, 0.05, "learning", "Direction bias cap",
                           "Maximum absolute nudge the direction learner may add "
                           "to a composite (keeps it a tie-breaker, not an override)"),
    "mtf_veto_align":    T(0.75, 0.34, 1.0, 0.01, "learning", "HTF direction veto",
                           "Block entries that fight the higher-timeframe trend when "
                           "|mtf_align| is at least this (1.0 disables the veto)"),
    # ---- meme coin risk envelope ----
    "meme_risk_factor":  T(0.5, 0.1, 1.0, 0.05, "meme", "Meme risk factor",
                           "Fraction of normal dollar-risk and position cap used for "
                           "meme coins (0.5 = half size; 1.0 = same as any coin)"),
    "meme_stop_widen":   T(1.5, 1.0, 3.0, 0.1, "meme", "Meme stop/target widen",
                           "Multiplier on the ATR stop AND target for memes — they "
                           "gap hard, so a normal-width stop just donates spread"),
    "meme_max_positions": T(2, 1, 6, 1, "meme", "Max concurrent memes",
                           "Cap on how many meme positions can be open at once", True),
    "meme_max_exposure": T(0.15, 0.02, 0.60, 0.01, "meme", "Max meme exposure",
                           "Total meme notional cap as a fraction of equity — the "
                           "blast radius if a meme trade goes wrong"),
    "meme_universe_slots": T(4, 0, 8, 1, "meme", "Dedicated meme universe slots",
                           "Discovered meme coins get this many universe slots IN "
                           "ADDITION to the normal non-meme discovered budget, so "
                           "memes never crowd out non-meme coin discovery", True),
    "meme_strategy_influence": T(1.5, 1.0, 4.0, 0.1, "meme",
                           "Meme research/decision priority",
                           "Multiplies the `meme` strategy's EFFECTIVE composite "
                           "weight on classified meme coins only (non-memes are "
                           "unaffected). The bandit's learned weight is still the "
                           "base — a losing meme arm can still fall to zero. Does "
                           "NOT change position size, stop/target, max concurrent "
                           "memes or max meme exposure (meme_risk_factor/"
                           "meme_stop_widen/meme_max_positions/meme_max_exposure)."),
    "pattern_exit_cut":  T(0.70, 0.30, 1.0, 0.05, "risk", "Pattern-exit cut threshold",
                           "Contrary-reversal threat (0-1) at/above which an open "
                           "position is CUT outright. Higher = only the clearest "
                           "reversals close a trade early."),
    "pattern_exit_tighten": T(0.45, 0.20, 1.0, 0.05, "risk", "Pattern-exit tighten threshold",
                           "Contrary-reversal threat (0-1) at/above which the open "
                           "position's stop is pulled in (but not closed). Must be "
                           "below the cut threshold."),
    "pattern_exit_tighten_atr": T(1.0, 0.3, 3.0, 0.1, "risk", "Pattern-exit stop distance (ATR)",
                           "When a reversal tightens a stop, place it this many swing-"
                           "ATR from the current price (smaller = tighter)."),
    "exit_throttle_horizon_sec": T(43200, 300, 172800, 60, "risk",
                           "Exit-throttle learn horizon (s)",
                           "Forward window used to score whether a pattern_exit/"
                           "signal-flip cut was validated (price kept moving against "
                           "the position) or a false alarm (price recovered)", True),
    "llm_refresh_sec":   T(3600, 60, 7200, 30, "learning", "LLM advisor cadence (s)",
                           "Seconds between LLM-advisor lean refreshes (only when "
                           "the advisor is enabled + a key is configured)", True),
    "llm_lean_ttl_sec":  T(7200, 300, 21600, 60, "learning", "LLM lean TTL (s)",
                           "A cached LLM lean expires (→ no vote) after this long, "
                           "so a stale opinion can't dominate the ensemble", True),
    "model_refresh_sec": T(3600, 30, 3600, 30, "learning", "Model advisor cadence (s)",
                           "Seconds between ML-model-advisor inference refreshes "
                           "(only when enabled + a validated artifact is loaded)", True),
    "model_lean_ttl_sec": T(7200, 120, 21600, 60, "learning", "Model lean TTL (s)",
                            "A cached model lean expires (→ no vote) after this "
                            "long, so a stale prediction can't dominate the ensemble",
                            True),
    "model_rug_veto":    T(0.65, 0.10, 0.99, 0.01, "learning", "Model rug/risk veto",
                           "The model vote is forced to 0 when its rug/risk "
                           "probability is at or above this threshold"),
    "model_return_scale": T(20.0, 1.0, 100.0, 1.0, "learning", "Model return→lean scale",
                            "The predicted short-horizon return is multiplied by "
                            "this before tanh() to shape it into a [-1,1] lean"),
    # ---- LLM advisor influence (operator lever; the bandit still learns) ----
    "llm_influence":     T(2.0, 1.0, 6.0, 0.1, "learning", "LLM vote influence",
                           "Operator boost on the LLM advisor's weight in the "
                           "composite (1.0 = purely as the bandit has learned it). "
                           "The bandit STILL down-weights a losing LLM from realized "
                           "PnL — this only raises its baseline voice, never a blind "
                           "override. Only matters when the LLM advisor is enabled."),
    "llm_meme_influence": T(2.5, 1.0, 8.0, 0.1, "learning", "LLM influence on memes",
                           "Separate, usually larger LLM boost applied on MEME coins, "
                           "whose moves are narrative/sentiment-driven where an LLM's "
                           "read is most relevant. Same governance as llm_influence."),
    "llm_weight_floor":  T(0.0, 0.0, 0.50, 0.01, "learning", "LLM weight floor",
                           "Guarantee the LLM arm at least this SHARE of the composite "
                           "weight whenever it has an opinion, so the bandit can't fully "
                           "silence it. 0 = no floor (purely bandit-governed)."),
    "llm_ml_feature":    T(1, 0, 1, 1, "learning", "LLM teaches the ML",
                           "When 1, the LLM (and model-advisor) leans are fed to the "
                           "online model as INPUT FEATURES, so the ML learns whether "
                           "the LLM's opinion is predictive. 0 = ML ignores them.", True),
    # ---- stance presets ----
    "stance_passive_risk": T(0.5, 0.1, 1.0, 0.05, "stance", "Passive risk mult",
                           "Position-size multiplier at full passive"),
    "stance_aggr_risk":  T(1.6, 1.0, 3.0, 0.1, "stance", "Aggressive risk mult",
                           "Position-size multiplier at full aggressive"),
    "stance_passive_gate": T(0.55, 0.30, 0.90, 0.01, "stance", "Passive conf gate",
                           "Confidence gate at full passive (pickier)"),
    "stance_aggr_gate":  T(0.35, 0.10, 0.60, 0.01, "stance", "Aggressive conf gate",
                           "Confidence gate at full aggressive (looser)"),
    # ---- market-neutral hedge ----
    "hedge_z_entry":     T(2.7, 1.0, 5.0, 0.1, "hedge", "Hedge entry z-score",
                           "Relative-strength divergence (std devs) required to open a pair hedge"),
    "hedge_z_exit":      T(0.75, 0.1, 2.0, 0.05, "hedge", "Hedge exit z-score",
                           "Close the pair when the spread reverts inside this z"),
    "hedge_notional_pct": T(0.08, 0.01, 0.25, 0.01, "hedge", "Hedge leg size",
                           "Each leg notional as a fraction of equity (net exposure ~0)"),
    "hedge_corr_min":    T(0.60, 0.20, 0.95, 0.05, "hedge", "Min pair correlation",
                           "Only hedge pairs whose returns correlate at least this much"),
    "hedge_max_age_hours": T(72, 1, 336, 1, "hedge", "Hedge max age (h)",
                           "Force-close a pair hedge older than this", True),
    "hedge_cooldown_hours": T(6, 0, 72, 1, "hedge", "Hedge re-entry cooldown (h)",
                           "Hours to wait before re-opening a pair after it closed "
                           "(prevents churn on a spread that keeps grazing the band)"),
    "hedge_cost_multiple": T(1.5, 1.0, 5.0, 0.1, "hedge", "Hedge pair cost multiple",
                           "Expected z-reversion move (in $) must clear round-trip "
                           "cost on ALL FOUR fills by this multiple, or skip the pair"),
    # ---- Polymarket prediction-market sleeve (paper) ----
    "pm_interval_sec":   T(20, 15, 3600, 5, "polymarket", "Engine cadence (s)",
                           "Seconds between Polymarket decision cycles", True),
    "pm_universe_size":  T(200, 5, 200, 5, "polymarket", "Universe size",
                           "How many top-liquidity markets to evaluate each cycle", True),
    "pm_min_liquidity":  T(5000, 0, 500000, 500, "polymarket", "Min market liquidity ($)",
                           "Skip markets with CLOB liquidity below this — thin books "
                           "can't be paper-filled honestly", True),
    "pm_kelly_fraction": T(0.25, 0.02, 1.0, 0.01, "polymarket", "Kelly fraction",
                           "Fraction of full Kelly to stake on estimated edge "
                           "(0.25 = quarter-Kelly; Kelly over-bets on estimates)"),
    "pm_max_position_pct": T(0.05, 0.005, 0.30, 0.005, "polymarket", "Max position %",
                           "Cap on a single bet as a fraction of the paper bankroll"),
    "pm_max_positions":  T(8, 1, 40, 1, "polymarket", "Max open bets",
                           "Cap on concurrent open Polymarket positions", True),
    "pm_min_edge":       T(0.03, 0.0, 0.30, 0.005, "polymarket", "Min edge",
                           "Estimated fair-minus-market probability edge required "
                           "to consider a bet (probability points)"),
    "pm_edge_scale":     T(0.06, 0.005, 0.25, 0.005, "polymarket", "Edge scale (max nudge)",
                           "Largest fair-probability adjustment a full-strength "
                           "signal lean may claim over the market price"),
    "pm_cost_multiple":  T(1.5, 1.0, 6.0, 0.1, "polymarket", "Cost-viability multiple",
                           "Edge must clear round-trip spread+slippage by this multiple"),
    "pm_confidence_gate": T(0.15, 0.0, 0.90, 0.01, "polymarket", "Confidence gate",
                           "Minimum signal confidence to open a bet"),
    "pm_fee_rate":       T(0.0, 0.0, 0.05, 0.001, "polymarket", "Fee rate (per side)",
                           "Polymarket currently charges 0 trading fees; raise to "
                           "stress the book at a hypothetical fee"),
    "pm_slippage":       T(0.005, 0.0, 0.10, 0.001, "polymarket", "Slippage (prob pts/side)",
                           "Simulated fill slippage in probability points per side"),
    "pm_min_notional":   T(5, 1, 1000, 1, "polymarket", "Min bet notional ($)",
                           "Skip bets smaller than this (Polymarket min order is $5)", True),
    "pm_take_profit":    T(0.15, 0.0, 0.50, 0.01, "polymarket", "Early take-profit (prob)",
                           "Exit early once the token price rises this far above "
                           "entry (0 = hold to resolution only)"),
    "pm_stop_loss":      T(0.20, 0.0, 0.50, 0.01, "polymarket", "Early stop-loss (prob)",
                           "Exit early once the token price falls this far below "
                           "entry (0 = hold to resolution only)"),
    "pm_llm_influence":  T(2.5, 1.0, 8.0, 0.1, "polymarket", "LLM vote influence",
                           "Operator boost on the LLM advisor's weight in the "
                           "Polymarket composite. Prediction markets are natural-"
                           "language questions where an LLM's read is especially "
                           "relevant, so this defaults higher than the crypto lever. "
                           "The bandit still learns the LLM's real trust from "
                           "resolutions; this only raises its baseline voice. Only "
                           "matters when the LLM advisor is enabled + keyed."),
    "pm_llm_max_queries": T(6, 0, 40, 1, "polymarket", "LLM queries / cycle",
                           "Max LLM calls per Polymarket cycle (bounded for cost); "
                           "leans are cached with the llm_lean_ttl_sec TTL. 0 = "
                           "never query (LLM sleeve off for Polymarket).", True),
    "pm_research_influence": T(1.5, 1.0, 8.0, 0.1, "polymarket",
                           "Research vote influence",
                           "Operator boost on the web-research advisor's weight "
                           "in the Polymarket composite (same lever as "
                           "pm_llm_influence, for the `research` arm). The "
                           "bandit still learns its real trust from resolutions; "
                           "this only raises its baseline voice."),
    # ---- Polymarket heuristic-edge sensitivities (app.markets.polymarket.
    # signals._leans) — how strongly each transparent prior argues, and when it
    # activates at all. These do NOT change what counts as a real edge (that's
    # pm_min_edge/pm_edge_scale/pm_cost_multiple); they shape the raw per-
    # strategy lean the bandit then learns to trust or distrust from actual
    # resolutions, exactly like the crypto side's strategy weights.
    "pm_momentum_gain": T(6.0, 1.0, 15.0, 0.5, "polymarket", "Momentum gain",
                           "Sensitivity of the `momentum` prior to blended 1h/1d "
                           "outcome-0 price drift (higher = a given drift claims "
                           "a stronger lean)"),
    "pm_mean_revert_gain": T(8.0, 1.0, 20.0, 0.5, "polymarket",
                           "Mean-reversion gain",
                           "Sensitivity of the `mean_revert` prior to a sharp 1h "
                           "move (fades it) once past pm_mean_revert_threshold"),
    "pm_mean_revert_threshold": T(0.05, 0.0, 0.20, 0.005, "polymarket",
                           "Mean-reversion threshold",
                           "Minimum |1h move| in outcome-0 price before the "
                           "mean-reversion fade activates at all"),
    "pm_longshot_gain": T(2.0, 0.5, 5.0, 0.1, "polymarket", "Longshot-fade gain",
                           "Sensitivity of the favourite-longshot-bias fade to "
                           "how far outcome 0's price sits from a coin-flip, "
                           "once past pm_longshot_threshold"),
    "pm_longshot_threshold": T(0.15, 0.0, 0.40, 0.01, "polymarket",
                           "Longshot-fade threshold",
                           "Minimum |price - 0.5| before the favourite-longshot "
                           "fade activates at all"),
    "pm_microstructure_gain": T(10.0, 1.0, 30.0, 0.5, "polymarket",
                           "Microstructure gain",
                           "Sensitivity of the order-flow prior to the gap "
                           "between the last trade price and the implied mid"),
}

_overrides = None


def _load():
    global _overrides
    if _overrides is None:
        _overrides = {}
        try:
            if os.path.exists(PATH):
                with open(PATH) as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k in TUNABLES:
                        _overrides[k] = _coerce(k, v)
        except Exception:
            pass
    return _overrides


def _coerce(k, v):
    m = TUNABLES[k]
    v = float(v)
    v = max(m["min"], min(m["max"], v))
    return int(round(v)) if m["int"] else v


def tv(key):
    """Current value of a tunable (override or default)."""
    o = _load()
    return o.get(key, TUNABLES[key]["default"])


def values():
    return {k: tv(k) for k in TUNABLES}


def update(changes: dict):
    o = _load()
    for k, v in changes.items():
        if k in TUNABLES:
            try:
                o[k] = _coerce(k, v)
            except (TypeError, ValueError):
                pass
    _save(o)
    return values()


def reset():
    global _overrides
    _overrides = {}
    _save(_overrides)
    return values()


def _save(o):
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(o, f)
        os.replace(tmp, PATH)
    except Exception:
        pass
