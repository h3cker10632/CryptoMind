"""Polymarket sleeve — pure-logic tests (no network).

Covers the honest primitives: binary-share accounting (buy/sell/resolve payoff),
fractional-Kelly sizing + the edge/cost-viability gate, the heuristic signal
leans, and the shared learner's attribution. The live Gamma/CLOB client is
exercised separately and is NOT hit here so the suite stays offline + fast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import tunables as T
from app.markets.polymarket.broker import PMBroker
from app.markets.polymarket import signals, risk
from app.markets.polymarket.learner import PMLearner, regime_of


def _market(p0=0.40, spread=0.01, mom1h=0.0, mom1d=0.0, last=None, ttl=48.0,
            liq=50000.0):
    return {
        "condition_id": "0xcond", "question": "Team A vs Team B",
        "slug": "team-a-vs-b", "category": "nfl",
        "outcomes": ["A", "B"],
        "token_ids": ["tokA", "tokB"],
        "prices": [p0, round(1 - p0, 4)],
        "best_bid": p0 - spread / 2, "best_ask": p0 + spread / 2,
        "spread": spread, "last_trade_price": last if last is not None else p0,
        "liquidity": liq, "volume_24h": liq, "tick_size": 0.01,
        "min_order_size": 5.0, "mom_1h": mom1h, "mom_1d": mom1d,
        "ttl_hours": ttl, "end_date": "", "active": True, "closed": False,
        "accepting_orders": True, "fee_type": "zero_fees",
        "enable_order_book": True,
    }


# ------------------------------ broker ---------------------------------
def test_resolve_winner_pays_one_dollar_per_share():
    b = PMBroker(start_cash=1000.0)
    m = _market(p0=0.40)
    pos = b.open(m, 0, mid=0.40, stake=100.0, fee_rate=0.0, slippage=0.0,
                 stop=0.0, take=0.0, reason="t")
    assert pos is not None
    # 100 USDC / 0.40 = 250 shares; resolves YES -> 250 * $1 = $250
    t = b.resolve("tokA", won=True)
    assert round(t["pnl"], 2) == 150.0          # 250 payoff - 100 cost
    assert b.cash == 1000.0 - 100.0 + 250.0


def test_resolve_loser_loses_stake():
    b = PMBroker(start_cash=1000.0)
    m = _market(p0=0.40)
    b.open(m, 0, mid=0.40, stake=100.0, fee_rate=0.0, slippage=0.0,
           stop=0.0, take=0.0, reason="t")
    t = b.resolve("tokA", won=False)
    assert round(t["pnl"], 2) == -100.0
    assert b.cash == 900.0


def test_early_exit_on_price_rise_is_profit():
    b = PMBroker(start_cash=1000.0)
    m = _market(p0=0.40)
    b.open(m, 0, mid=0.40, stake=100.0, fee_rate=0.0, slippage=0.0,
           stop=0.30, take=0.55, reason="t")
    t = b.exit("tokA", mid=0.50, fee_rate=0.0, slippage=0.0, reason="tp")
    assert t["pnl"] > 0                          # bought 0.40, sold 0.50


def test_insufficient_cash_and_min_order_rejected():
    b = PMBroker(start_cash=10.0)
    m = _market(p0=0.40)
    assert b.open(m, 0, 0.40, 100.0, 0.0, 0.0, 0.0, 0.0, "t") is None  # cash
    b2 = PMBroker(start_cash=1000.0)
    assert b2.open(m, 0, 0.40, 3.0, 0.0, 0.0, 0.0, 0.0, "t") is None   # <$5


def test_manage_triggers_take_and_stop():
    b = PMBroker(start_cash=1000.0)
    m = _market(p0=0.40)
    b.open(m, 0, 0.40, 100.0, 0.0, 0.0, stop=0.30, take=0.55, reason="t")
    closed = b.manage(lambda tid: 0.60, fee_rate=0.0, slippage=0.0)
    assert len(closed) == 1 and closed[0]["exit_reason"] == "take-profit"


# ------------------------------ sizing ---------------------------------
def test_size_zero_when_no_edge():
    stake, why = risk.size(1000.0, 1000.0, price=0.40, edge=0.0,
                           confidence=0.9, market=_market())
    assert stake == 0.0 and "edge" in why


def test_size_blocked_by_cost_floor():
    T.update({"pm_min_edge": 0.0, "pm_cost_multiple": 2.0, "pm_slippage": 0.02,
              "pm_confidence_gate": 0.0})
    # round-trip = spread(0.05) + 2*slip(0.02) = 0.09; need = 0.18 > edge 0.05
    stake, why = risk.size(1000.0, 1000.0, price=0.40, edge=0.05,
                           confidence=0.9, market=_market(spread=0.05))
    assert stake == 0.0 and "cost floor" in why


def test_size_positive_with_real_edge():
    T.reset()
    T.update({"pm_min_edge": 0.02, "pm_cost_multiple": 1.0, "pm_slippage": 0.005,
              "pm_confidence_gate": 0.0, "pm_kelly_fraction": 0.25,
              "pm_max_position_pct": 0.50, "pm_min_notional": 5})
    stake, why = risk.size(1000.0, 1000.0, price=0.40, edge=0.10,
                           confidence=0.8, market=_market(spread=0.01))
    assert stake > 0 and why == "ok"
    # quarter-Kelly on a real edge must stay well under the position cap
    assert stake <= 1000.0 * 0.50
    T.reset()


# ------------------------------ signals --------------------------------
def test_signal_leans_toward_underpriced_side_and_bounds_edge():
    T.reset()
    # strong upward momentum on outcome 0 -> lean toward outcome 0
    sig = signals.evaluate(_market(p0=0.40, mom1h=0.06, mom1d=0.08),
                           weights={}, edge_scale=0.06)
    assert sig["outcome_index"] in (0, 1)
    assert abs(sig["edge"]) <= 0.06 + 1e-9      # bounded by edge_scale
    assert 0.0 <= sig["confidence"] <= 1.0


def test_signal_no_lean_is_flat():
    sig = signals.evaluate(_market(p0=0.50, mom1h=0.0, mom1d=0.0, last=0.50),
                           weights={}, edge_scale=0.06)
    assert sig["outcome_index"] is None or sig["edge"] == 0.0


# ------------------------------ learner --------------------------------
def test_regime_bucketing():
    assert regime_of(5) == "<1d"
    assert regime_of(100) == "1-7d"
    assert regime_of(None) == "unknown"


def test_trade_teacher_attributes_and_online_model_learns():
    lr = PMLearner()
    assert lr.stats()["starved"] is True
    trade = {"return_pct": 0.5, "regime_at_entry": "<1d",
             "votes": {"momentum": 0.8, "microstructure": 0.4}}
    lr.on_trade_closed(trade)
    st = lr.stats()
    assert st["starved"] is False
    assert st["n_trades_learned"] == 1
    assert "momentum" in st["attributions"]
    # signal teacher + online model update on a resolution
    m = _market(p0=0.40, mom1d=0.05)
    lr.score_resolution(m, {"momentum": 0.5}, outcome0_won=1)
    assert lr.stats()["online_model"]["n_updates"] == 1


def test_online_model_predicts_probability():
    lr = PMLearner()
    p = lr.online.predict(_market())
    assert 0.0 <= p <= 1.0


def test_llm_is_a_strategy_and_influence_moves_the_lean():
    assert "llm" in signals.STRATEGIES
    m = _market(p0=0.50, mom1h=0.0, mom1d=0.0, last=0.50)  # heuristics ~flat
    base = signals.evaluate(m, weights={}, edge_scale=0.10, llm_lean=0.9,
                            llm_influence=1.0)
    boosted = signals.evaluate(m, weights={}, edge_scale=0.10, llm_lean=0.9,
                               llm_influence=5.0)
    # a bullish LLM lean on outcome 0 should push the edge more when boosted
    assert boosted["net_lean"] >= base["net_lean"] > 0
    assert boosted["votes"]["llm"] != 0.0


def test_llm_lean_feeds_the_pm_online_model():
    lr = PMLearner()
    m = _market()
    w_off = lr.online.features(m)
    w_on = lr.online.features({**m, "llm_lean": 0.6})
    assert len(w_off) == 6 and w_off[-1] == 0.0
    assert w_on[-1] == 0.6
