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
                           "Per-cycle decay on bandit evidence (1.0 = never forget; "
                           "0.995 ≈ 7h half-life). Lower = adapts faster to regime change"),
    "evolve_every_sec":  T(1200, 120, 21600, 60, "learning", "GA cadence (s)",
                           "Seconds between genetic-evolution runs (universe rotates)", True),
    "trade_weight":      T(2.0, 0.5, 10.0, 0.5, "learning", "Trade-vs-signal weight",
                           "How much a real closed trade outweighs a scored signal"),
    "loss_lesson_mult":  T(5.0, 1.0, 10.0, 0.5, "learning", "Loss lesson multiplier",
                           "A losing trade teaches N-times harder than a winner (DeepAlpha heuristic)"),
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
