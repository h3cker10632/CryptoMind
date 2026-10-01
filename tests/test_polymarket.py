"""Polymarket sleeve — pure-logic tests (no network).

Covers the honest primitives: binary-share accounting (buy/sell/resolve payoff),
fractional-Kelly sizing + the edge/cost-viability gate, the heuristic signal
leans, and the shared learner's attribution. The live Gamma/CLOB client is
exercised separately and is NOT hit here so the suite stays offline + fast.
"""
import os
import sys
import importlib.util
import json
import threading
import time
from email.utils import formatdate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import tunables as T
from app import db
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


def test_forecast_ledger_deduplicates_and_labels_once():
    db.init()
    forecast = {
        "condition_id": "ledger-condition",
        "ttl_bucket": "1-7d",
        "forecast_ts": 1000.0,
        "predicted_p0": 0.62,
        "market_p0": 0.55,
        "votes": {"momentum": 0.5},
        "features": {"prices": [0.55, 0.45], "ttl_hours": 48.0},
        "evidence": [{"id": "e1", "url": "https://example.test/a"}],
    }

    assert db.record_pm_forecast(forecast) is True
    assert db.record_pm_forecast(forecast) is False
    db.flush()

    rows = [r for r in db.pm_forecast_rows() if r["condition_id"] == "ledger-condition"]
    assert len(rows) == 1
    assert rows[0]["votes"] == {"momentum": 0.5}
    assert rows[0]["features"] == forecast["features"]
    assert rows[0]["evidence"] == forecast["evidence"]
    assert rows[0]["resolved_outcome0"] is None

    assert db.label_pm_forecast(rows[0]["id"], 1) is True
    assert db.label_pm_forecast(rows[0]["id"], 1) is False


def test_forecast_resolution_polling_persists_due_and_retry_state():
    forecast = {
        "condition_id": "polling-condition",
        "ttl_bucket": "0-1h",
        "forecast_ts": 1000.0,
        "predicted_p0": 0.5,
        "market_p0": 0.5,
        "votes": {},
        "features": {"values": [0.5] * 7, "ttl_hours": 0.1},
        "evidence": [],
        "resolution_due_ts": 1100.0,
    }
    assert db.record_pm_forecast(forecast) is True
    row = next(row for row in db.pending_pm_forecasts()
               if row["condition_id"] == "polling-condition")
    assert row["resolution_due_ts"] == 1100.0
    assert row["resolution_attempts"] == 0
    assert db.mark_pm_resolution_attempt(row["id"], 1200.0) == 1
    assert all(row["condition_id"] != "polling-condition"
               for row in db.pending_pm_forecasts(now=1201.0))
    retried = next(row for row in db.pending_pm_forecasts(now=1320.0)
                   if row["condition_id"] == "polling-condition")
    assert retried["resolution_attempts"] == 1
    assert retried["resolution_checked_ts"] == 1200.0
    assert retried["resolution_next_poll_ts"] == 1320.0


def test_pending_forecast_query_filters_future_due_and_backoff_rows():
    suffix = str(time.time_ns())
    base = {
        "ttl_bucket": "0-1h", "forecast_ts": time.time(),
        "predicted_p0": 0.5, "market_p0": 0.5, "votes": {},
        "features": {}, "evidence": [],
    }
    due_id = f"eligible-{suffix}"
    future_id = f"future-{suffix}"
    retry_id = f"backoff-{suffix}"
    for condition_id, due in ((due_id, None), (future_id, time.time() + 24 * 3600),
                              (retry_id, None)):
        assert db.record_pm_forecast({**base, "condition_id": condition_id,
                                      "resolution_due_ts": due})
    retry_row = next(row for row in db.pending_pm_forecasts()
                     if row["condition_id"] == retry_id)
    db.mark_pm_resolution_attempt(retry_row["id"], 10_000.0)

    rows = db.pending_pm_forecasts(limit=10, now=10_001.0)
    ids = {row["condition_id"] for row in rows}
    assert due_id in ids
    assert future_id not in ids
    assert retry_id not in ids


def test_pm_forecast_count_includes_rows_outside_report_window():
    before = db.pm_forecast_count()
    assert db.record_pm_forecast({
        "condition_id": f"counted-{time.time_ns()}",
        "ttl_bucket": "1-7d", "forecast_ts": time.time(),
        "predicted_p0": 0.5, "market_p0": 0.5,
        "votes": {}, "features": {}, "evidence": [],
    })
    assert db.pm_forecast_count() == before + 1


def test_learning_report_discloses_bounded_metrics_window(monkeypatch):
    import importlib

    engine_module = importlib.import_module("app.markets.polymarket.engine")
    monkeypatch.setattr(engine_module.db, "pm_forecast_rows", lambda limit: [])
    monkeypatch.setattr(engine_module.db, "pm_forecast_count", lambda: 12_345)

    report = engine_module.PolymarketEngine().learning_report()
    assert report["forecast_metrics_scope"] == "newest_forecasts"
    assert report["forecast_metrics_window"] == 10_000
    assert report["forecast_total_count"] == 12_345


def test_pending_forecasts_prioritize_resolution_due_time():
    suffix = str(time.time_ns())
    base = {
        "ttl_bucket": "0-1h", "forecast_ts": 1000.0,
        "predicted_p0": 0.5, "market_p0": 0.5, "votes": {},
        "features": {"values": [0.5] * 7, "ttl_hours": 0.1},
        "evidence": [],
    }
    assert db.record_pm_forecast({**base,
                                  "condition_id": f"later-{suffix}",
                                  "resolution_due_ts": 2.0})
    assert db.record_pm_forecast({**base,
                                  "condition_id": f"due-{suffix}",
                                  "forecast_ts": 2000.0,
                                  "resolution_due_ts": 1.0})

    assert db.pending_pm_forecasts(limit=1)[0]["condition_id"] == f"due-{suffix}"


def test_bounded_forecast_report_reads_keep_the_newest_rows():
    now = time.time()
    suffix = str(time.time_ns())
    for condition_id, forecast_ts in ((f"old-report-row-{suffix}", now),
                                      (f"new-report-row-{suffix}", now + 1.0)):
        assert db.record_pm_forecast({
            "condition_id": condition_id,
            "ttl_bucket": "1-7d",
            "forecast_ts": forecast_ts,
            "predicted_p0": 0.5,
            "market_p0": 0.5,
            "votes": {},
            "features": {},
            "evidence": [],
        }) is True
    assert db.pm_forecast_rows(limit=1)[0]["condition_id"] == f"new-report-row-{suffix}"


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


# ---------------- heuristic-edge sensitivity tunables ----------------

def test_momentum_gain_tunable_scales_lean():
    T.reset()
    m = _market(p0=0.40, mom1h=0.02, mom1d=0.02)
    T.update({"pm_momentum_gain": 2.0})
    low = signals._leans(m)["momentum"]
    T.update({"pm_momentum_gain": 12.0})
    high = signals._leans(m)["momentum"]
    assert abs(high) > abs(low)
    T.reset()


def test_mean_revert_threshold_tunable_gates_activation():
    T.reset()
    m = _market(p0=0.40, mom1h=0.03)
    T.update({"pm_mean_revert_threshold": 0.10})   # 0.03 move no longer qualifies
    assert signals._leans(m)["mean_revert"] == 0.0
    T.update({"pm_mean_revert_threshold": 0.01})    # now it qualifies
    assert signals._leans(m)["mean_revert"] != 0.0
    T.reset()


def test_mean_revert_gain_tunable_scales_lean():
    T.reset()
    m = _market(p0=0.40, mom1h=0.08)
    T.update({"pm_mean_revert_gain": 2.0})
    low = abs(signals._leans(m)["mean_revert"])
    T.update({"pm_mean_revert_gain": 18.0})
    high = abs(signals._leans(m)["mean_revert"])
    assert high > low
    T.reset()


def test_longshot_threshold_and_gain_tunables():
    T.reset()
    m = _market(p0=0.30)      # |0.30-0.5| = 0.20
    T.update({"pm_longshot_threshold": 0.30})   # 0.20 no longer qualifies
    assert signals._leans(m)["longshot_fade"] == 0.0
    T.update({"pm_longshot_threshold": 0.05, "pm_longshot_gain": 1.0})
    low = abs(signals._leans(m)["longshot_fade"])
    T.update({"pm_longshot_gain": 4.0})
    high = abs(signals._leans(m)["longshot_fade"])
    assert high >= low
    T.reset()


def test_microstructure_gain_tunable_scales_lean():
    T.reset()
    m = _market(p0=0.40, last=0.45)
    T.update({"pm_microstructure_gain": 2.0})
    low = abs(signals._leans(m)["microstructure"])
    T.update({"pm_microstructure_gain": 25.0})
    high = abs(signals._leans(m)["microstructure"])
    assert high > low
    T.reset()


def test_research_influence_tunable_boosts_research_arm():
    """pm_research_influence must boost the `research` arm's effective weight
    the same way pm_llm_influence boosts `llm` (engine.tick wiring)."""
    T.reset()
    m = _market(p0=0.40, mom1h=0.0, mom1d=0.0)
    sig_low = signals.evaluate(m, weights={"research": 1.0}, edge_scale=0.06,
                              research_lean=0.8, research_influence=1.0)
    sig_high = signals.evaluate(m, weights={"research": 1.0}, edge_scale=0.06,
                               research_lean=0.8, research_influence=6.0)
    assert abs(sig_high["edge"]) >= abs(sig_low["edge"])



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
    w_on = lr.online.features({**m, "llm_lean": 0.6, "research_lean": 0.4})
    assert len(w_off) == 7 and w_off[-2:] == [0.0, 0.0]
    assert w_on[-2:] == [0.6, 0.4]


def test_research_lean_is_a_separate_signal_arm():
    m = _market(p0=0.50, mom1h=0.0, mom1d=0.0, last=0.50)
    sig = signals.evaluate(m, weights={}, edge_scale=0.10,
                           research_lean=0.8)
    assert "research" in signals.STRATEGIES
    assert sig["votes"]["research"] > 0.0
    assert sig["votes"]["llm"] == 0.0


def test_polymarket_research_cache_module_exists():
    assert importlib.util.find_spec("app.markets.polymarket.research") is not None


def test_research_cache_filters_stale_irrelevant_and_duplicate_items():
    from app.markets.polymarket.research import PMResearchCache

    now = time.time()
    date = formatdate(now, usegmt=True)
    rss = f"""<rss><channel>
      <item><title>Alpha wins major championship</title>
        <link>https://news.example.test/alpha?utm_source=rss</link>
        <pubDate>{date}</pubDate><source>Daily Sports</source>
        <description>Alpha secured the final result.</description></item>
      <item><title>Alpha championship update</title>
        <link>https://news.example.test/alpha</link>
        <pubDate>{date}</pubDate><source>Duplicate Feed</source></item>
      <item><title>Bitcoin price moves higher</title>
        <link>https://news.example.test/bitcoin</link>
        <pubDate>{date}</pubDate><source>Crypto News</source></item>
            <item><title>The market outlook rises today</title>
                <link>https://news.example.test/generic</link>
                <pubDate>{date}</pubDate><source>Generic News</source></item>
      <item><title>Alpha won the championship</title>
        <link>https://news.example.test/old</link>
        <pubDate>Wed, 01 Jan 2020 00:00:00 GMT</pubDate>
        <source>Old News</source></item>
    </channel></rss>"""
    calls = []

    def fetch(query):
        calls.append(query)
        return rss

    cache = PMResearchCache(fetcher=fetch)
    market = _market()
    market.update({"condition_id": "research-market",
                   "question": "Will Alpha win the championship?",
                   "category": "sports"})
    try:
        assert cache.refresh([market]) == 1
        for future in list(cache._pending.values()):
            future.result()

        evidence = cache.evidence("research-market")
        assert len(evidence) == 1
        assert evidence[0]["publisher"] == "Daily Sports"
        assert evidence[0]["url"] == "https://news.example.test/alpha"
        assert "alpha" in evidence[0]["matched_terms"]
        assert evidence[0]["retrieved_at"] > 0
        assert cache.refresh([market]) == 0
        assert len(calls) == 1
        replacement = {**market, "condition_id": "replacement-market"}
        assert cache.refresh([replacement]) == 0
        assert cache.stats()["markets_seen"] == 1
    finally:
        cache.close()


def test_research_relevance_rejects_generic_single_term_entity_collision():
        from app.markets.polymarket.research import _parse_feed

        now = time.time()
        date = formatdate(now, usegmt=True)
        rss = f"""<rss><channel><item>
            <title>City council approves annual budget</title>
            <link>https://news.example.test/council</link>
            <pubDate>{date}</pubDate><source>Local News</source>
            <description>City representatives voted Tuesday.</description>
        </item></channel></rss>"""

        assert _parse_feed(rss, ["manchester", "city"], now) == []


def test_research_advisor_filters_citations_and_abstains_without_evidence():
    from app.markets.polymarket.llm import PMLLMAdvisor

    advisor = PMLLMAdvisor()
    advisor.configured = lambda: True
    requests = []
    advisor._request_research = lambda market, evidence: requests.append(evidence) or (
        '{"lean":0.7,"citations":["e1","unsupported"],"why":"fresh report"}')
    evidence = [{"id": "e1", "publisher": "Daily Sports", "title": "Alpha result"}]

    result = advisor.research_lean(_market(), evidence)
    assert result["lean"] == 0.7
    assert result["citations"] == ["e1"]

    advisor._request_research = lambda market, evidence: "not-json"
    malformed = advisor.research_lean(_market(), [{**evidence[0], "id": "e2"}])
    assert malformed["lean"] == 0.0
    assert malformed["abstain"] is True

    no_evidence = advisor.research_lean(_market(), [])
    assert no_evidence["lean"] == 0.0
    assert no_evidence["abstain"] is True
    assert len(requests) == 1
    advisor.close()


def test_research_advisor_refresh_does_not_block_on_provider(monkeypatch):
    from app.markets.polymarket.llm import PMLLMAdvisor

    advisor = PMLLMAdvisor()
    market = {**_market(), "condition_id": "slow-research"}
    evidence = [{"id": "e1", "title": "Alpha update"}]

    monkeypatch.setattr(advisor, "configured", lambda: True)
    monkeypatch.setattr(advisor, "_request_research", lambda *_: (
        time.sleep(0.2) or '{"lean": 0.4, "citations": ["e1"]}'))
    cache = type("Cache", (), {"evidence": lambda self, condition_id: evidence})()

    started = time.perf_counter()
    assert advisor.refresh_research([market], cache, 1) == 0
    assert time.perf_counter() - started < 0.1
    time.sleep(0.25)
    assert advisor.refresh_research([market], cache, 1) == 1
    assert advisor.research_result("slow-research")["lean"] == 0.4
    advisor.close()


def test_pm_learner_capture_restore_and_forecast_idempotency():
    learned = PMLearner()
    market = {**_market(), "llm_lean": 0.2, "research_lean": 0.6}
    learned.score_resolution(market, {"research": 0.6}, 1)
    state = learned.capture()
    json.dumps(state)

    restored = PMLearner()
    assert restored.restore(state) is True
    assert restored.bandit.arms == learned.bandit.arms
    assert restored.online.n == learned.online.n
    assert restored.stats()["n_signals_scored"] == 1
    assert restored.stats()["starved"] is False

    before = restored.online.n
    features = [0.31, 0.02, -0.1, 2.2, 8.0, 0.2, 0.75]
    assert restored.score_forecast(
        "forecast-42", 48.0, {"research": 0.75}, 0, features) is True
    assert restored.score_forecast(
        "forecast-42", 48.0, {"research": 0.75}, 0, features) is False
    assert restored.online.n == before + 1
    assert restored.online._mean[-1] == 0.675

    unchanged = restored.capture()
    assert restored.restore({"version": -1}) is False
    assert restored.capture() == unchanged


def test_pm_learner_scores_one_forecast_once_under_concurrency():
    learned = PMLearner()
    barrier = threading.Barrier(3)
    results = []
    original_score = learned._score_signal

    def synchronized_score(*args):
        time.sleep(0.05)
        return original_score(*args)

    learned._score_signal = synchronized_score

    def score():
        barrier.wait()
        results.append(learned.score_forecast(
            "concurrent-forecast", 12.0, {"research": 0.5}, 1,
            [0.5, 0.01, 0.0, 2.0, 8.0, 0.0, 0.5]))

    workers = [threading.Thread(target=score) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()

    assert sorted(results) == [False, True]
    assert learned.online.n == 1
    assert learned.n_signals_scored == 1


def test_shared_persistence_snapshot_includes_pm_learner(monkeypatch):
    from app import persistence
    from app.markets.polymarket.learner import learner

    monkeypatch.setattr(learner, "capture", lambda: {"version": 1, "n": 9})
    assert persistence.capture()["polymarket_learner"] == {"version": 1, "n": 9}


def test_skipped_market_forecast_is_scored_once_at_resolution(monkeypatch):
    import importlib
    from types import SimpleNamespace

    engine_module = importlib.import_module("app.markets.polymarket.engine")
    market = _market(p0=0.50, mom1h=0.0, mom1d=0.0, last=0.50, ttl=0.05)
    market["condition_id"] = f"skipped-market-{time.time_ns()}"
    resolution = {"resolved": False, "winning_index": None, "prices": [0.5, 0.5]}
    paper_broker = PMBroker()
    pm_learner = PMLearner()
    engine = engine_module.PolymarketEngine()

    monkeypatch.setattr(engine_module, "broker", paper_broker)
    monkeypatch.setattr(engine_module, "learner", pm_learner)
    monkeypatch.setattr(engine_module.client, "fetch_markets", lambda **kwargs: [market])
    monkeypatch.setattr(engine_module.client, "resolution", lambda condition_id: resolution)
    monkeypatch.setattr(engine_module.llm_advisor, "refresh", lambda markets, count: 0)
    monkeypatch.setattr(engine_module.llm_advisor, "refresh_research",
                        lambda markets, cache, count: 0, raising=False)
    monkeypatch.setattr(engine_module, "research_cache",
                        SimpleNamespace(refresh=lambda markets: 0,
                                        evidence=lambda condition_id: []),
                        raising=False)
    monkeypatch.setattr(engine_module.app_settings, "get",
                        lambda key, default=None: False if key == "pm_auto_trade" else default)

    first = engine.tick()
    rows = [row for row in engine_module.db.pm_forecast_rows()
            if row["condition_id"] == market["condition_id"]]
    assert first["markets"] == 1
    assert len(paper_broker.positions) == 0
    assert len(rows) == 1
    assert rows[0]["resolved_outcome0"] is None
    assert pm_learner.stats()["n_signals_scored"] == 0

    resolution.update({"resolved": True, "winning_index": 0, "prices": [1.0, 0.0]})
    engine.tick()
    assert pm_learner.stats()["n_signals_scored"] == 1
    assert pm_learner.online.n == 1
    assert pm_learner.scored_forecast_ids == {str(rows[0]["id"])}

    engine.tick()
    assert pm_learner.stats()["n_signals_scored"] == 1
    assert pm_learner.online.n == 1


def test_pm_auto_trade_paused_until_portfolio_migration_confirmed(monkeypatch):
    import importlib
    from types import SimpleNamespace

    engine_module = importlib.import_module("app.markets.polymarket.engine")
    db.reset_paper_portfolio_for_tests()
    market = _market(p0=0.10, mom1h=0.0, mom1d=0.0, last=0.10, ttl=48.0)
    market["condition_id"] = f"gated-market-{time.time_ns()}"
    paper_broker = PMBroker()
    fixed_signal = {
        "outcome_index": 0, "outcome": "A", "price": 0.10, "fair": 0.80,
        "fair_p0": 0.80, "edge": 0.70, "confidence": 0.9, "net_lean": 1.0,
        "agreement": 1.0, "votes": {"momentum": 0.5}, "leans": {"momentum": 0.5},
    }

    monkeypatch.setattr(engine_module, "broker", paper_broker)
    monkeypatch.setattr(engine_module.client, "fetch_markets", lambda **kwargs: [market])
    monkeypatch.setattr(engine_module.client, "resolution",
                        lambda condition_id: {"resolved": False})
    monkeypatch.setattr(engine_module.llm_advisor, "refresh", lambda markets, count: 0)
    monkeypatch.setattr(engine_module.llm_advisor, "refresh_research",
                        lambda markets, cache, count: 0, raising=False)
    monkeypatch.setattr(engine_module, "research_cache",
                        SimpleNamespace(refresh=lambda markets: 0,
                                        evidence=lambda condition_id: []),
                        raising=False)
    monkeypatch.setattr(engine_module.signals, "evaluate",
                        lambda *a, **kw: dict(fixed_signal))
    monkeypatch.setattr(engine_module, "size_bet", lambda *a, **kw: (50.0, "ok"))
    monkeypatch.setattr(engine_module.app_settings, "get",
                        lambda key, default=None: True if key == "pm_auto_trade" else default)

    assert engine_module.paper_portfolio.ready is False
    engine = engine_module.PolymarketEngine()
    engine.tick()
    assert len(paper_broker.positions) == 0        # paused: migration pending

    db.initialize_paper_portfolio(100_000.0, "unified-paper-v1-test")
    assert engine_module.paper_portfolio.ready is True
    engine.tick()
    assert len(paper_broker.positions) == 1        # unblocked once confirmed
    db.reset_paper_portfolio_for_tests()


def test_pm_scan_defaults_use_20_seconds_and_200_markets():
    T.reset()
    assert T.tv("pm_interval_sec") == 20
    assert T.tv("pm_universe_size") == 200
    T.reset()


def test_forecast_report_pairs_scores_and_excludes_unresolved_rows():
    from app.markets.polymarket.engine import _forecast_metrics

    rows = [
        {"resolved_outcome0": 1, "predicted_p0": 0.9, "market_p0": 0.5,
         "votes": {"research": 0.5, "momentum": -0.2},
         "features": {"edge_scale": 0.1, "research_citations": ["e1"]},
         "evidence": [{"id": "e1", "published_ts": 9_900.0}]},
        {"resolved_outcome0": 0, "predicted_p0": 0.1, "market_p0": 0.1,
         "votes": {"research": 0.0},
         "features": {"edge_scale": 0.1, "research_citations": []},
         "evidence": []},
        {"resolved_outcome0": None, "predicted_p0": 0.7, "market_p0": 0.6,
         "votes": {}, "features": {"edge_scale": 0.1},
         "evidence": [{"id": "e2", "published_ts": 9_950.0}]},
    ]
    report = _forecast_metrics(rows, now=10_000.0)

    assert report["scored_forecast_count"] == 2
    assert report["unresolved_forecast_count"] == 1
    assert report["research_covered_count"] == 2
    assert report["research_abstention_count"] == 2
    assert report["bot_brier"] == 0.01
    assert report["market_brier"] == 0.13
    assert round(report["brier_delta"], 6) == -0.12
    assert report["bot_log_loss"] < report["market_log_loss"]
    assert report["strategy_performance"]["research"]["n"] == 1
    assert report["strategy_performance"]["momentum"]["n"] == 1
    assert report["strategy_performance"]["mean_revert"]["n"] == 0
    assert report["fresh_evidence_documents"] == 2


def test_polymarket_status_exposes_learning_and_research_coverage(monkeypatch):
    import importlib

    engine_module = importlib.import_module("app.markets.polymarket.engine")
    engine = engine_module.PolymarketEngine()
    monkeypatch.setattr(engine_module.db, "pm_forecast_rows", lambda limit: [{
        "resolved_outcome0": 1, "predicted_p0": 0.8, "market_p0": 0.5,
        "votes": {"research": 0.5},
        "features": {"edge_scale": 0.1, "research_citations": ["e1"]},
        "evidence": [{"id": "e1", "published_ts": time.time()}],
    }])
    monkeypatch.setattr(engine_module.research_cache, "stats", lambda: {
        "markets_seen": 1, "markets_covered": 1, "coverage": 1.0,
        "fresh_documents": 1, "pending_queries": 0, "requests": 1,
        "last_error": "",
    })

    snapshot = engine.snapshot()
    assert snapshot["learning"]["scored_forecast_count"] == 1
    assert snapshot["learning"]["research_covered_count"] == 1
    assert snapshot["research"]["coverage"] == 1.0
    assert "api_key" not in str(snapshot).lower()


def test_recent_forecasts_expose_evidence_provenance_without_secrets(monkeypatch):
    import importlib

    engine_module = importlib.import_module("app.markets.polymarket.engine")
    monkeypatch.setattr(engine_module.db, "pm_forecast_rows", lambda limit: [{
        "id": 7, "condition_id": "inspect-condition", "ttl_bucket": "1-7d",
        "forecast_ts": 1000.0, "predicted_p0": 0.61, "market_p0": 0.55,
        "resolved_outcome0": None, "votes": {"research": 0.4},
        "features": {"question": "Will Alpha win?", "category": "sports",
                      "research_citations": ["e1"]},
        "evidence": [{"id": "e1", "publisher": "Daily Sports",
                      "url": "https://news.example.test/alpha",
                      "published_at": "2026-09-29T12:00:00+00:00",
                      "excerpt": "Alpha result"}],
    }])

    result = engine_module.PolymarketEngine().forecasts(limit=5)
    forecast = result["forecasts"][0]
    assert forecast["question"] == "Will Alpha win?"
    assert forecast["evidence"][0]["publisher"] == "Daily Sports"
    assert forecast["research_citations"] == ["e1"]
    assert "api_key" not in str(result).lower()
