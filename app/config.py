"""CryptoMind — global configuration."""
import os

# Universe traded (Coinbase Exchange product ids)
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "LINK-USD", "AVAX-USD"]

# Data
# 1-hour candles. Backtests on the cached history showed every 5m/30-min
# holding horizon loses to the ~1.2% round-trip cost; only multi-hour/
# multi-day holds clear it. 300 x 1h = 12.5 days of context (Coinbase
# returns at most 300 candles per request).
CANDLE_GRANULARITY = 3600           # 1-hour candles
CANDLE_HISTORY = 300                # bars kept in memory per product
BARS_PER_DAY = 86400 // CANDLE_GRANULARITY
# Higher timeframes (seconds) read by the multi-timeframe trend filter. They
# must fit in CANDLE_HISTORY native bars with >=12 folded bars each.
MTF_TIMEFRAMES = (3600, 4 * 3600, 12 * 3600)
MARKET_POLL_SEC = 30                # market refresh cadence
NEWS_POLL_SEC = 300                 # news/sentiment refresh cadence
TICK_SEC = 20                       # orchestrator decision cadence

# Paper account
START_CASH = 100_000.0
# REALISTIC retail costs (Coinbase Advanced base tier taker ~0.6%, Kraken
# ~0.4%; we model 0.5%). This is 5x the old flattering 10 bps — if the
# system can't show positive expectancy at these costs, it isn't ready.
FEE_RATE = 0.005                    # 50 bps taker fee per side
SLIPPAGE_BPS = 10                   # 10 bps simulated slippage per side

# Risk
RISK_PER_TRADE = 0.0075             # 0.75% of equity risked per trade
MAX_POSITION_PCT = 0.20             # max 20% of equity in one position
MAX_OPEN_POSITIONS = 4
MAX_GROSS_EXPOSURE = 0.60           # max 60% of equity deployed
STOP_ATR_MULT = 3.0
TAKE_PROFIT_ATR_MULT = 6.0
TRAIL_ATR_MULT = 3.0
MAX_DRAWDOWN_KILL = 0.15            # 15% peak-to-trough → kill switch
DAILY_LOSS_LIMIT = 0.05             # 5% daily loss → halt for the day
MIN_CONFIDENCE = 0.45               # signal confidence gate (full-size entries)
COOLDOWN_SEC = 12 * 3600            # per-product re-entry cooldown

# Shorts
ALLOW_SHORTS = True                 # runtime-toggleable via settings

# Exploration trading — small probing positions below the confidence gate so
# the learning stack gets real trade outcomes to learn from (paper mode).
EXPLORE_MIN_CONFIDENCE = 0.25       # floor for exploration candidates
EXPLORE_PROB = 0.0                  # probes paid full fees for noise; the
                                    # counterfactual skip-scorer learns for free
EXPLORE_SIZE_FACTOR = 0.4           # fraction of normal position size

# Self-improvement loop
ALLOC_LOOKBACK = 200                # scored signals kept per strategy
ALLOC_TEMPERATURE = 4.0             # softmax temperature (percent-return units)
# Forward-return horizon for scoring signals. Matches the strategy's natural
# holding period (median ~20h in backtest) so the learner is graded on the
# same exam the trades actually sit.
SIGNAL_EVAL_HORIZON_SEC = 24 * 3600

# Strategy sleeves switched OFF. Measured on 1.35M scored live signals + the
# hourly backtest: RSI mean-reversion and short-term reversal had a NEGATIVE
# edge even before costs; sentiment was negative; order-book microstructure is
# a seconds-scale signal with no edge at a multi-hour horizon; memes were
# operator-disabled. They stay in the code and can be re-enabled here.
DISABLED_STRATEGIES = frozenset({
    "meanrev", "sentiment", "microstructure", "meme",
    # superseded by the slow-horizon versions below (trend_slow /
    # breakout_slow), which use 24h/96h EMAs, 72h momentum and a 48h channel
    # instead of 12/26-bar EMAs and a 20-bar channel
    "trend", "breakout",
    # chart patterns measured ~0 bps of edge over 162k scored signals
    "pattern",
    # (the online-model `ml` vote is governed by the learner evidence gate,
    # app/learn/gate.py, not listed here)
})

# Reproducibility: a single base seed for every stochastic step in the
# VALIDATION path (GA search, bootstrap CIs, DSR/PBO). Fixing it makes a
# promotion decision (dsr>0.95, PBO<0.30, GA champion) reproducible run-to-run —
# so a champion that goes live can be re-derived and audited. Override with the
# CRYPTOMIND_VALIDATION_SEED env var; set to a NEGATIVE value for nondeterministic
# runs (fresh entropy each time). pybroker added the same seed knob to
# Strategy#backtest / #walkforward for exactly this reason.
VALIDATION_SEED = int(os.getenv("CRYPTOMIND_VALIDATION_SEED", "1337"))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cryptomind.db")
