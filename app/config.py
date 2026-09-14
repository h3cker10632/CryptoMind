"""CryptoMind — global configuration."""
import os

# Universe traded (Coinbase Exchange product ids)
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "LINK-USD", "AVAX-USD"]

# Data
CANDLE_GRANULARITY = 300            # 5-minute candles
CANDLE_HISTORY = 300                # bars kept in memory per product
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
STOP_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.0
TRAIL_ATR_MULT = 2.5
MAX_DRAWDOWN_KILL = 0.15            # 15% peak-to-trough → kill switch
DAILY_LOSS_LIMIT = 0.05             # 5% daily loss → halt for the day
MIN_CONFIDENCE = 0.45               # signal confidence gate (full-size entries)
COOLDOWN_SEC = 900                  # per-product re-entry cooldown

# Shorts
ALLOW_SHORTS = True                 # runtime-toggleable via settings

# Exploration trading — small probing positions below the confidence gate so
# the learning stack gets real trade outcomes to learn from (paper mode).
EXPLORE_MIN_CONFIDENCE = 0.25       # floor for exploration candidates
EXPLORE_PROB = 0.12                 # chance per decision tick to probe
EXPLORE_SIZE_FACTOR = 0.4           # fraction of normal position size

# Self-improvement loop
ALLOC_LOOKBACK = 200                # scored signals kept per strategy
ALLOC_TEMPERATURE = 4.0             # softmax temperature (percent-return units)
SIGNAL_EVAL_HORIZON_SEC = 3600      # forward-return horizon for scoring signals

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cryptomind.db")
