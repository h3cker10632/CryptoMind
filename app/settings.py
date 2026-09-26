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
    # strategy — never the driver. Off by default: no artifact, no effect.
    "model_advisor_enabled": False,
    # Predictive, self-learning early loss-cut. When True, a losing position the
    # system confidently expects to keep moving against it is cut before the
    # hard stop. Learns hold-vs-cut per market state from realized outcomes.
    "exit_advisor_enabled": True,
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
    "meme_trading_enabled": True,
    # Periodic full-state export. When True, the system writes a timestamped
    # JSON report (the same document as GET /api/export) to the reports/ folder
    # every `auto_export_interval_sec` seconds. The interval is operator-tunable
    # from the dashboard slider and takes effect on the next cycle (no restart).
    "auto_export_enabled": True,
    "auto_export_interval_sec": 300,   # default: every 5 minutes

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
}

# Secret settings: stored, but MASKED in the public payload and never clobbered
# by an empty save (only overwritten when a new non-empty value is supplied).
SECRET_KEYS = {"invo_token", "invo_refresh_token", "llm_api_key"}

# Float settings: (min, max) inclusive clamp.
FLOAT_KEYS = {
    "invo_horizon_hours": (0.25, 168.0),
    "invo_rank_decay": (0.0, 4.0),
}

# Integer settings: (min, max) inclusive clamp. Everything else is treated as
# a boolean toggle.
INT_KEYS = {
    "auto_export_interval_sec": (30, 86400),   # 30s .. 24h
    "invo_top_n": (1, 200),
    "invo_interval_sec": (30, 86400),
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
    return bool(v)


MASK = "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"   # ••••••••


def public():
    """Settings for the API/UI: secrets masked, with a companion '<key>_set'
    boolean so the UI can show whether a secret is configured."""
    s = dict(load())
    for k in SECRET_KEYS:
        s[k + "_set"] = bool(s.get(k))
        s[k] = MASK if s.get(k) else ""
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
