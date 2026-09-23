"""Performance-weighted universe ranking (freqtrade PerformanceFilter idea)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.performance import (performance_scores, rank_multiplier,
                                  performance_table)


def _t(product, pnl, closed, entry=100.0, qty=10.0):
    return {"product": product, "pnl": pnl, "closed": closed,
            "entry": entry, "qty": qty}


def test_scores_aggregate_by_symbol():
    now = time.time()
    trades = [_t("DOGE-USD", 50, now), _t("DOGE-USD", -20, now),
              _t("BTC-USD", 100, now)]
    s = performance_scores(trades, 86400, now, min_trades=1)
    assert s["DOGE"]["n"] == 2
    assert abs(s["DOGE"]["net"] - 30.0) < 1e-6
    assert s["BTC"]["n"] == 1


def test_lookback_excludes_old_trades():
    now = time.time()
    trades = [_t("OLD-USD", -500, now - 999_999), _t("NEW-USD", 10, now)]
    s = performance_scores(trades, 3600, now, min_trades=1)
    assert "OLD" not in s and "NEW" in s


def test_winner_boosted_loser_penalized_untested_neutral():
    now = time.time()
    trades = ([_t("WIN-USD", 60, now) for _ in range(3)] +
              [_t("LOSE-USD", -80, now) for _ in range(3)] +
              [_t("NEW-USD", 5, now)])                 # only 1 trade
    s = performance_scores(trades, 86400, now, min_trades=3)
    assert rank_multiplier(s, "WIN", 3) > 1.0
    assert rank_multiplier(s, "LOSE", 3) < 1.0
    assert rank_multiplier(s, "NEW", 3) == 1.0        # untested -> neutral
    # bounds
    assert rank_multiplier(s, "WIN", 3) <= 1.5
    assert rank_multiplier(s, "LOSE", 3) >= 0.3


def test_multiplier_saturates():
    now = time.time()
    # enormous winner still capped at max_boost
    trades = [_t("MOON-USD", 10_000, now) for _ in range(3)]
    s = performance_scores(trades, 86400, now, min_trades=3)
    assert rank_multiplier(s, "MOON", 3) == 1.5


def test_table_sort_order_winners_untested_losers():
    now = time.time()
    trades = ([_t("WIN-USD", 60, now) for _ in range(3)] +
              [_t("LOSE-USD", -80, now) for _ in range(3)] +
              [_t("NEW-USD", 5, now)])
    rows = performance_table(trades, 86400, now, min_trades=3)
    syms = [r["sym"] for r in rows]
    assert syms.index("WIN") < syms.index("NEW") < syms.index("LOSE")


def test_empty_history_safe():
    assert performance_scores([], 3600, time.time()) == {}
    assert rank_multiplier({}, "BTC") == 1.0
    assert performance_table(None, 3600, time.time()) == []
