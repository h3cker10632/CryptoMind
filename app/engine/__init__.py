"""Portfolio engine: one code path for backtests and live trading.

  panel       aligned days x coins arrays from the data store
  features    causal (point-in-time) features, computed for all coins at once
  backtest    vectorized simulation + metrics (Sharpe, drawdown, halves, DSR)
  strategies  pure functions panel -> daily target weights; live trading takes
              the LAST row of the same function, so what is tested is traded
  registry    every backtest logged (params, data version, metrics, returns)
              so multiple-testing corrections count every variant ever tried
"""
