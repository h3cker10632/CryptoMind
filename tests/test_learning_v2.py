"""Learning v2 regressions:
  * SHORT trades credit the strategies that voted short (was sign-inverted)
  * the signal stream teaches the bandit NET of round-trip cost
  * disabled sleeves never vote and are silent for allocation
  * signals are recorded once per native bar, not every tick
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _learner():
    from app.learn.loop import Learner
    L = Learner()
    L.bandit.arms = {}
    return L


def _trade(side, pnl, votes):
    return {"product": "BTC-USD", "side": side, "qty": 1.0, "entry": 100.0,
            "pnl": pnl, "votes": votes, "regime_at_entry": "bear/normal"}


def test_winning_short_rewards_short_voters(monkeypatch):
    monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
    L = _learner()
    # trend_slow voted SHORT (-0.8), xsmom voted LONG (+0.4); the short WON
    L.on_trade_closed(_trade(-1, +3.0, {"trend_slow": -0.8, "xsmom": 0.4}))
    _, m_short_voter, _ = L.bandit.arms[("bear/normal", "trend_slow")]
    _, m_long_voter, _ = L.bandit.arms[("bear/normal", "xsmom")]
    assert m_short_voter > 0          # was punished before the fix
    assert m_long_voter < 0           # was rewarded before the fix


def test_losing_long_punishes_long_voters(monkeypatch):
    monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
    L = _learner()
    L.on_trade_closed(_trade(1, -2.0, {"trend_slow": 0.8, "xsmom": -0.4}))
    assert L.bandit.arms[("bear/normal", "trend_slow")][1] < 0
    assert L.bandit.arms[("bear/normal", "xsmom")][1] > 0


def test_skip_counterfactual_short_credit_is_side_aware(monkeypatch):
    import time
    from app import tunables
    monkeypatch.setattr("app.learn.loop.db.log_event", lambda *a, **k: None)
    tunables.update({"skip_learn_weight": 0.5})
    L = _learner()
    L.pending_skips.append((time.time() - 10 ** 6, "BTC-USD", -1, 100.0,
                            "bear/normal", {"trend_slow": -0.9}))
    # price fell 10% -> the skipped short would have won, even after costs
    monkeypatch.setattr(L, "_price_at", lambda *a, **k: 90.0)

    class M:
        def price(self, p):
            return 90.0
    import app.learn.loop as loop
    monkeypatch.setattr(loop, "LOOKUP_GIVE_UP_SEC", 10 ** 7)
    assert L._score_skips(M()) == 1
    assert L.bandit.arms[("bear/normal", "trend_slow")][1] > 0


def test_signal_stream_is_net_of_cost(monkeypatch):
    """A signal that was directionally right but moved LESS than the round-trip
    cost must teach the bandit a NEGATIVE lesson."""
    from app import tunables
    from app.learn.evolution import evolution
    L = _learner()
    tunables.update({"signal_learn_weight": 0.5, "signal_learn_clip": 0.05})
    orig_c, orig_p = dict(evolution.champions), dict(evolution.champion_portfolios)
    try:
        evolution.champions, evolution.champion_portfolios = {}, {}
        monkeypatch.setattr("app.learn.loop.db.unscored_signals",
                            lambda older: [{"rowid": 1, "product": "BTC-USD",
                                            "ts": 0, "direction": 1,
                                            "strategy": "trend_slow",
                                            "confidence": 1.0, "regime": "bull"}])
        for name in ("score_signal", "strategy_scores", "log_event", "abandon_signal"):
            monkeypatch.setattr(f"app.learn.loop.db.{name}",
                                (lambda *a, **k: []) if name == "strategy_scores"
                                else (lambda *a, **k: None))
        # +0.5% move: right direction, but well under the ~1.2% round trip
        monkeypatch.setattr(L, "_price_at",
                            lambda p, ts, m=None: 100.0 if ts == 0 else 100.5)
        for name in ("maybe_evolve", "_check_drift"):
            monkeypatch.setattr(L, name, lambda *a, **k: None)
        monkeypatch.setattr(L, "_train_online_model", lambda *a, **k: 0)
        monkeypatch.setattr(L, "_maybe_reset_broken_ml", lambda: False)

        class M:
            tickers = {}
            def price(self, p):
                return 100.5
        L.run(M(), {"label": "bull"})
        assert L.bandit.arms[("bull", "trend_slow")][1] < 0
    finally:
        tunables.update({"signal_learn_weight": 0.15, "signal_learn_clip": 0.04})
        evolution.champions, evolution.champion_portfolios = orig_c, orig_p


def test_disabled_sleeves_are_silent():
    from app.config import DISABLED_STRATEGIES
    L = _learner()
    silent = L._silent_sleeves()
    assert DISABLED_STRATEGIES <= silent
    assert "meanrev" in silent and "trend_slow" not in silent


def test_engine_records_once_per_bar_and_skips_disabled(monkeypatch):
    import time
    from app.signals import engine as eng
    from app.data.features import features_from_ohlcv

    recorded = []
    monkeypatch.setattr(eng.db, "record_signal",
                        lambda name, p, d, c, r: recorded.append((name, p)))
    monkeypatch.setattr(eng, "PRODUCTS", ["BTC-USD"])
    # strong uptrend: every slow sleeve fires
    now = time.time()
    cs = [[now - (200 - i) * 3600, 100 + i, 101 + i, 100 + i, 100.5 + i, 10.0]
          for i in range(200)]

    class Mkt:
        candles = {"BTC-USD": cs}
        def regime(self):
            return {"label": "bull/normal", "trend": "bull"}
        def features(self, p):
            closes = [c[4] for c in cs]
            f = features_from_ohlcv(closes, [c[2] for c in cs],
                                    [c[1] for c in cs], [c[5] for c in cs])
            f.update(atr_swing=f["atr"], mtf_align=1.0, xs_mom=2.0)
            return f

    class NLP:
        market_sentiment = 0.0
        def asset_score(self, p):
            return (0.0, 0)

    e = eng.SignalEngine()
    e.compute(Mkt(), NLP())
    first = list(recorded)
    e.compute(Mkt(), NLP())                       # same bar -> no new records
    assert recorded == first
    names = {n for n, _ in first}
    assert {"trend_slow", "xsmom"} <= names
    assert not names & eng.DISABLED_STRATEGIES
    assert all(e.per_strategy["BTC-USD"][n] == 0.0 for n in eng.DISABLED_STRATEGIES)
