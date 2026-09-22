"""Trade- & equity-quality metrics (pybroker parity): Sortino, Profit Factor,
Calmar, drawdown duration, exposure, win/loss shape."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backtest import stats as st


# ---------------- Sortino ----------------
def test_sortino_ignores_upside_vol():
    # Both series have the SAME mean and the SAME downside moves (one -0.05),
    # but 'spiky' adds a big harmless upside spike. Sharpe would punish the
    # spike's variance; Sortino must rate spiky >= smooth because only the
    # (identical) downside is penalised.
    smooth = [0.03, 0.03, -0.05, 0.03, 0.06]
    spiky = [0.00, 0.00, -0.05, 0.00, 0.15]   # same sum (0.10), same downside
    assert sum(smooth) == sum(spiky)
    assert st.sortino(spiky) >= st.sortino(smooth)


def test_sortino_zero_when_no_downside():
    assert st.sortino([0.01, 0.02, 0.03]) == 0.0


# ---------------- Profit Factor ----------------
def test_profit_factor_basic():
    # wins sum 0.30, losses sum 0.10 -> PF 3.0
    assert abs(st.profit_factor([0.1, 0.2, -0.1]) - 3.0) < 1e-9


def test_profit_factor_none_and_inf():
    assert st.profit_factor([]) is None
    assert st.profit_factor([0.1, 0.2]) == math.inf     # no losses
    assert st.profit_factor([-0.1, -0.2]) == 0.0        # no wins -> 0 gross win


# ---------------- win/loss shape ----------------
def test_win_loss_stats():
    s = st.win_loss_stats([0.10, 0.30, -0.05, -0.15])
    assert abs(s["avg_win"] - 0.20) < 1e-9
    assert abs(s["avg_loss"] - 0.10) < 1e-9        # magnitude, positive
    assert abs(s["win_loss_ratio"] - 2.0) < 1e-9
    assert s["win_rate"] == 0.5
    assert s["n"] == 4


def test_expectancy_is_mean():
    assert abs(st.expectancy([0.1, -0.1, 0.2]) - (0.2 / 3)) < 1e-9


# ---------------- drawdown ----------------
def test_max_drawdown_depth():
    eq = [100, 120, 90, 110]        # peak 120 -> trough 90 = 25%
    assert abs(st.max_drawdown(eq) - 0.25) < 1e-9


def test_max_drawdown_duration_bars():
    # peak at idx1(120); underwater idx2,3,4 then new high idx5 -> 3 bars
    eq = [100, 120, 110, 100, 115, 130]
    assert st.max_drawdown_duration(eq) == 3


def test_drawdown_duration_zero_when_monotonic():
    assert st.max_drawdown_duration([1, 2, 3, 4]) == 0


# ---------------- annualized / calmar ----------------
def test_annualized_return_doubling_in_half_year():
    ppy = 24 * 365
    n = ppy // 2                    # half a year of hourly bars
    eq = [100.0] + [None] * 0
    # build a curve that exactly doubles over n steps
    r = 2 ** (1 / n)
    eq = [100.0 * (r ** i) for i in range(n + 1)]
    ann = st.annualized_return(eq, ppy)
    # doubled in half a year -> ~4x annualized (2^2 - 1 = 3.0)
    assert abs(ann - 3.0) < 0.05


def test_calmar_none_without_drawdown():
    eq = [100, 101, 102, 103]      # monotonic -> no drawdown
    assert st.calmar(eq, 24 * 365) is None


def test_calmar_positive_with_growth_and_drawdown():
    # a realistic-length rising curve with a mid drawdown (short curves make
    # annualization overflow / meaningless, so use a full quarter of hourly bars)
    ppy = 24 * 365
    n = ppy // 4
    r = 2 ** (1 / n)               # doubles over the quarter
    eq = [100.0 * (r ** i) for i in range(n + 1)]
    eq[n // 2] *= 0.85             # carve a ~15% drawdown mid-curve
    c = st.calmar(eq, ppy)
    assert c is not None and c > 0


# ---------------- annualized sharpe scaling ----------------
def test_annualized_sharpe_scales_by_sqrt_periods():
    rets = [0.001, -0.0005, 0.0012, -0.0003, 0.0008] * 20
    per_obs = st.sharpe(rets)
    ann = st.annualized_sharpe(rets, 24 * 365)
    assert abs(ann - per_obs * math.sqrt(24 * 365)) < 1e-9


# ---------------- bundle ----------------
def test_trade_quality_bundle_keys():
    tq = st.trade_quality([0.1, -0.05, 0.2, -0.1])
    for k in ("profit_factor", "expectancy", "avg_win", "avg_loss",
              "win_loss_ratio", "win_rate", "n"):
        assert k in tq
    tq0 = st.trade_quality([])
    assert tq0["profit_factor"] is None and tq0["n"] == 0
