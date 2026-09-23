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
    # If True the system may open SHORT positions (margin-style paper).
    "allow_shorts": True,
    # Trading stance: "passive" | "auto" | "aggressive".
    # auto = system scores current conditions and adapts on its own.
    "trade_mode": "auto",
    # Market-neutral pair hedging (2.7-sigma relative-strength divergence,
    # long the laggard / short the leader in equal notional).
    "hedge_enabled": True,
    # Optional LLM advisor sleeve. When True (and an API key is set via
    # CRYPTOMIND_LLM_KEY / OPENAI_API_KEY) the advisor contributes ONE
    # directional vote that the bandit weights like any other strategy — it is
    # never the driver. Off by default: no key, no cost, no effect.
    "llm_advisor_enabled": False,
    # Optional ML-model advisor sleeve. When True (and a validated crypto_ml_lab
    # artifact exists at model_artifact/ or $CRYPTOMIND_MODEL_DIR) the advisor
    # contributes ONE directional vote that the bandit weights like any other
    # strategy — never the driver. Off by default: no artifact, no effect.
    "model_advisor_enabled": False,
    # Predictive, self-learning early loss-cut. When True, a losing position the
    # system confidently expects to keep moving against it is cut before the
    # hard stop. Learns hold-vs-cut per market state from realized outcomes.
    "exit_advisor_enabled": True,
    # Meme-coin trading. When True, a curated seed of Coinbase-listed meme
    # majors becomes eligible immediately and CoinGecko's meme category is
    # polled so hot new memes surface automatically — all under a tighter risk
    # envelope (smaller size, wider stops, concurrent + total exposure caps).
    # High variance by nature. Enabled per the operator's request to "see how
    # it plays out"; toggle off any time on the dashboard.
    "meme_trading_enabled": True,
    # Periodic full-state export. When True, the system writes a timestamped
    # JSON report (the same document as GET /api/export) to the reports/ folder
    # every `auto_export_interval_sec` seconds. The interval is operator-tunable
    # from the dashboard slider and takes effect on the next cycle (no restart).
    "auto_export_enabled": True,
    "auto_export_interval_sec": 300,   # default: every 5 minutes
}

STR_KEYS = {"trade_mode": {"passive", "auto", "aggressive"}}

# Integer settings: (min, max) inclusive clamp. Everything else is treated as
# a boolean toggle.
INT_KEYS = {
    "auto_export_interval_sec": (30, 86400),   # 30s .. 24h
}

_settings = None


def load():
    global _settings
    if _settings is None:
        _settings = dict(DEFAULTS)
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH) as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k not in DEFAULTS:
                        continue
                    if k in STR_KEYS:
                        if v in STR_KEYS[k]:
                            _settings[k] = v
                    elif k in INT_KEYS:
                        _settings[k] = _coerce_int(k, v)
                    else:
                        _settings[k] = bool(v)
        except Exception:
            pass
    return _settings


def _coerce_int(key, v):
    """Clamp an integer setting to its [min, max] range; fall back to the
    default on garbage input."""
    lo, hi = INT_KEYS[key]
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return DEFAULTS[key]


def get(key):
    return load()[key]


def update(changes: dict):
    s = load()
    for k, v in changes.items():
        if k not in DEFAULTS:
            continue
        if k in STR_KEYS:
            if v in STR_KEYS[k]:
                s[k] = v
        elif k in INT_KEYS:
            s[k] = _coerce_int(k, v)
        else:
            s[k] = bool(v)
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(SETTINGS_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(s, f)
        os.replace(tmp, SETTINGS_PATH)
    except Exception:
        pass
    return s
