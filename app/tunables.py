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
    "stop_atr_mult":     T(2.0, 0.5, 6.0, 0.1, "risk", "Stop ATR mult",
                           "Initial stop distance in ATR multiples"),
    "take_profit_atr_mult": T(3.0, 1.0, 10.0, 0.1, "risk", "Take-profit ATR mult",
                           "Target distance in ATR multiples (cost floor may widen it)"),
    "trail_atr_mult":    T(2.5, 0.5, 6.0, 0.1, "risk", "Trailing stop ATR mult",
                           "Trailing stop distance from the high/low-water mark"),
    "swing_atr_bars":    T(12, 1, 48, 1, "risk", "Swing ATR timeframe (x5m bars)",
                           "5m candles folded into one ATR bar for sizing stops/"
                           "targets: 12=1h, 3=15m, 1=native 5m. Bigger = wider, "
                           "more cost-viable swing stops (a different strategy, "
                           "not a looser scalp).", True),
    "max_drawdown_kill": T(0.15, 0.03, 0.50, 0.01, "risk", "Kill-switch drawdown",
                           "Peak-to-trough drawdown that trips the kill switch"),
    "daily_loss_limit":  T(0.05, 0.01, 0.25, 0.005, "risk", "Daily loss halt",
                           "Daily loss that halts new entries until tomorrow"),
    "cooldown_sec":      T(900, 0, 7200, 60, "risk", "Re-entry cooldown (s)",
                           "Seconds before re-entering a product after an exit", True),
    "min_hold_sec":      T(300, 0, 14400, 30, "risk", "Minimum hold (s)",
                           "A position younger than this is exempt from signal-flip "
                           "and trailing-giveback exits (its hard stop/target still "
                           "apply) — stops same-tick churn", True),
    "max_entries_per_hour": T(12, 0, 120, 1, "risk", "Max entries / hour",
                           "Hard cap on NEW position opens per rolling hour "
                           "(0 = unlimited); anti-overtrading circuit", True),
    "trail_giveback_pct": T(0.35, 0.0, 0.90, 0.05, "risk", "Peak-giveback exit",
                           "Close a WINNING position once it gives back this "
                           "fraction of its peak unrealized gain (price-basis). "
                           "0 = disabled. Arms only after a real move (see arm %)."),
    "trail_giveback_arm_pct": T(0.010, 0.0, 0.10, 0.001, "risk", "Giveback arm move",
                           "Peak unrealized gain (as a fraction of entry price) "
                           "required before the peak-giveback exit can trigger — "
                           "prevents arming on noise"),
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
    "explore_prob":      T(0.12, 0.0, 1.0, 0.01, "signals", "Probe probability",
                           "Chance per decision tick of firing one probe trade"),
    "explore_size_factor": T(0.4, 0.05, 1.0, 0.05, "signals", "Probe size factor",
                           "Probe size as a fraction of a normal position"),
    # ---- learning ----
    "bandit_decay_gamma": T(0.995, 0.90, 1.0, 0.001, "learning", "Bandit forgetting γ",
                           "Per-cycle decay on bandit n, variance, AND mean (idle "
                           "edge forgets toward 0; 1.0 = never forget; 0.995 ≈ 7h "
                           "half-life). Lower = adapts faster to regime change"),
    "evolve_every_sec":  T(1200, 120, 21600, 60, "learning", "GA cadence (s)",
                           "Seconds between genetic-evolution runs (universe rotates)", True),
    "oi_growth_threshold": T(0.15, 0.02, 1.0, 0.01, "learning", "OI-growth discovery",
                           "24h open-interest growth (as a fraction) that flags a "
                           "coin as a capital-flow discovery candidate and boosts "
                           "its universe heat"),
    "trade_weight":      T(2.0, 0.5, 10.0, 0.5, "learning", "Trade attribution weight",
                           "Scale on closed-trade net PnL fed to the bandit "
                           "(the premium, real-money learning signal)"),
    "signal_learn_weight": T(0.15, 0.0, 1.0, 0.01, "learning", "Signal-stream weight",
                           "Scale on the GROSS directional signal-scoring stream fed "
                           "to the bandit. This is abundant (100-1000x the trade "
                           "count) but lower quality (pre-cost, not a real fill), so "
                           "it's discounted well below trade_weight. 0 disables it — "
                           "restoring the old dashboard-only behaviour."),
    "signal_learn_clip":  T(0.01, 0.001, 0.05, 0.001, "learning", "Signal-stream clip",
                           "Clip each gross signal forward-return to +/- this before "
                           "feeding the bandit, so one violent bar can't dominate the "
                           "abundant-but-noisy signal stream"),
    "loss_lesson_mult":  T(5.0, 1.0, 10.0, 0.5, "learning", "Loss lesson multiplier",
                           "A losing trade teaches N-times harder than a winner (DeepAlpha heuristic)"),
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
    "exit_horizon_sec":  T(1800, 300, 14400, 60, "learning", "Loss-cut learn horizon (s)",
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
    "llm_refresh_sec":   T(900, 60, 7200, 30, "learning", "LLM advisor cadence (s)",
                           "Seconds between LLM-advisor lean refreshes (only when "
                           "the advisor is enabled + a key is configured)", True),
    "llm_lean_ttl_sec":  T(3600, 300, 21600, 60, "learning", "LLM lean TTL (s)",
                           "A cached LLM lean expires (→ no vote) after this long, "
                           "so a stale opinion can't dominate the ensemble", True),
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
