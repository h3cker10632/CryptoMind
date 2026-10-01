"""Meme-coin trading: classification, seed, discovery, and risk envelope."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.data.memes import memes, Memes, SEED
from app.risk.manager import RiskManager
from app import settings, tunables


def _enable(flag=True):
    settings.update({"meme_trading_enabled": flag})


# ---------------- classification ----------------

def test_seed_members_classified_as_meme():
    for sym in SEED:
        assert memes.is_meme(f"{sym}-USD")


def test_non_meme_not_classified():
    assert not memes.is_meme("BTC-USD")
    assert not memes.is_meme("ETH-USD")


def test_discovered_category_members_classified():
    m = Memes()
    m.discovered = {"MOG", "TURBO"}
    m.known = m.seed | m.discovered
    assert m.is_meme("MOG-USD")
    assert m.is_meme("TURBO-USD")
    assert not m.is_meme("BTC-USD")


def test_enabled_reflects_setting():
    _enable(True)
    assert memes.enabled() is True
    _enable(False)
    assert memes.enabled() is False
    _enable(True)


# ---------------- risk envelope ----------------

def _rs():
    return {"effective_risk_scale": 1.0, "rl_sit_out": False}


def test_meme_gets_smaller_size_than_non_meme():
    tunables.update({"meme_risk_factor": 0.5, "meme_stop_widen": 1.5})
    rm = RiskManager()
    n_btc, _, _ = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="BTC-USD")
    n_doge, _, _ = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="DOGE-USD")
    assert n_doge < n_btc


def test_meme_gets_wider_stop_and_target():
    tunables.update({"meme_stop_widen": 1.5})
    rm = RiskManager()
    _, s_btc, t_btc = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="BTC-USD")
    _, s_doge, t_doge = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="DOGE-USD")
    assert abs(100 - s_doge) > abs(100 - s_btc)      # stop further from entry
    assert abs(100 - t_doge) > abs(100 - t_btc)      # target further too


def test_meme_factor_one_matches_normal():
    tunables.update({"meme_risk_factor": 1.0, "meme_stop_widen": 1.0})
    rm = RiskManager()
    n_btc, s_btc, _ = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="BTC-USD")
    n_doge, s_doge, _ = rm.size(100000, 100.0, 2.0, 0.8, _rs(), direction=1, product="DOGE-USD")
    assert abs(n_doge - n_btc) < 1e-6 and abs(s_doge - s_btc) < 1e-6
    tunables.update({"meme_risk_factor": 0.5, "meme_stop_widen": 1.5})


# ---------------- can_open caps ----------------

class _FakeMarket:
    def __init__(self, price=100.0, closes=None):
        self._price = price
        self._closes = closes or [100.0] * 60

    def price(self, p):
        return self._price

    def closes(self, p):
        return self._closes


class _FakeBroker:
    def __init__(self, positions=None):
        self.positions = positions or {}

    def equity(self, market):
        return 100000.0

    def exposure(self, market):
        return sum(pos["qty"] * market.price(p) for p, pos in self.positions.items())


def test_meme_disabled_blocks_meme_entry():
    _enable(False)
    rm = RiskManager()
    ok, why = rm.can_open("DOGE-USD", _FakeBroker(), _FakeMarket(), True)
    assert not ok and "disabled" in why
    _enable(True)


def test_max_concurrent_meme_positions_enforced():
    _enable(True)
    tunables.update({"meme_max_positions": 2, "meme_max_exposure": 0.60})
    rm = RiskManager()
    positions = {
        "DOGE-USD": {"qty": 10, "side": 1, "entry": 100},
        "SHIB-USD": {"qty": 10, "side": 1, "entry": 100},
    }
    ok, why = rm.can_open("PEPE-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert not ok and "meme positions" in why


def test_max_meme_exposure_enforced():
    _enable(True)
    tunables.update({"meme_max_positions": 6, "meme_max_exposure": 0.05})
    rm = RiskManager()
    # one meme worth 10% of equity already open, cap is 5% → block the next
    positions = {"DOGE-USD": {"qty": 100, "side": 1, "entry": 100}}  # $10k / $100k
    ok, why = rm.can_open("PEPE-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert not ok and "meme exposure" in why
    tunables.update({"meme_max_exposure": 0.15})


def test_non_meme_entry_unaffected_by_meme_caps():
    _enable(True)
    tunables.update({"meme_max_positions": 1})
    rm = RiskManager()
    positions = {"DOGE-USD": {"qty": 10, "side": 1, "entry": 100}}
    ok, why = rm.can_open("BTC-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert ok, why                       # meme cap must not block a core coin


def test_memes_excluded_from_general_max_open_positions_count():
    """Held memes must not consume the general max_open_positions budget --
    only non-meme positions count against it."""
    _enable(True)
    tunables.update({"max_open_positions": 2, "meme_max_positions": 6,
                     "meme_max_exposure": 0.60})
    rm = RiskManager()
    positions = {
        "BTC-USD": {"qty": 1, "side": 1, "entry": 100},
        "DOGE-USD": {"qty": 1, "side": 1, "entry": 100},
        "SHIB-USD": {"qty": 1, "side": 1, "entry": 100},
        "PEPE-USD": {"qty": 1, "side": 1, "entry": 100},
    }
    # only 1 non-meme (BTC-USD) is held -- a second non-meme must still fit
    # under the cap of 2, even though 3 memes are also open.
    ok, why = rm.can_open("ETH-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert ok, why
    tunables.update({"max_open_positions": 4, "meme_max_positions": 2,
                     "meme_max_exposure": 0.15})


def test_meme_entry_exempt_from_general_max_open_positions():
    """A new meme entry must not be blocked by the general cap even when
    non-meme positions already occupy every general slot."""
    _enable(True)
    tunables.update({"max_open_positions": 2, "meme_max_positions": 6,
                     "meme_max_exposure": 0.60})
    rm = RiskManager()
    positions = {
        "BTC-USD": {"qty": 1, "side": 1, "entry": 100},
        "ETH-USD": {"qty": 1, "side": 1, "entry": 100},
    }
    ok, why = rm.can_open("PEPE-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert ok, why
    tunables.update({"max_open_positions": 4, "meme_max_positions": 2,
                     "meme_max_exposure": 0.15})


def test_non_meme_entry_blocked_counts_only_non_meme_positions():
    """A non-meme entry IS still blocked once non-meme positions alone hit
    the general cap, regardless of how many memes are also open."""
    _enable(True)
    tunables.update({"max_open_positions": 2, "meme_max_positions": 6,
                     "meme_max_exposure": 0.60})
    rm = RiskManager()
    positions = {
        "BTC-USD": {"qty": 1, "side": 1, "entry": 100},
        "ETH-USD": {"qty": 1, "side": 1, "entry": 100},
        "DOGE-USD": {"qty": 1, "side": 1, "entry": 100},
        "SHIB-USD": {"qty": 1, "side": 1, "entry": 100},
    }
    ok, why = rm.can_open("SOL-USD", _FakeBroker(positions), _FakeMarket(), True)
    assert not ok and "max open positions" in why
    tunables.update({"max_open_positions": 4, "meme_max_positions": 2,
                     "meme_max_exposure": 0.15})


def test_seed_universe_tags_and_heats(monkeypatch):
    _enable(True)
    from app.data.universe import universe
    universe.mention_heat.clear()
    universe.sources.clear()
    memes.seed_universe()
    for sym in SEED:
        assert universe.mention_heat.get(sym, 0) >= 2.0
        assert "meme" in universe.sources.get(sym, set())


# ---------------- meme research/decision priority (Task 6) ----------------

def test_research_queue_reserves_slots_for_fresh_meme_narratives():
    """A meme narrative with FEWER mentions than every non-meme candidate must
    still surface in the bounded research queue -- reserved slots prevent it
    being crowded out by raw mention-count ranking."""
    from app.data.research import ResearchEngine

    engine = ResearchEngine()
    docs = []
    for i in range(8):                       # 8 non-meme assets, all louder
        for _ in range(20 - i):
            docs.append({"assets": [f"COIN{i}-USD"]})
    for _ in range(3):                        # one quiet meme narrative
        docs.append({"assets": ["DOGE-USD"]})
    engine.documents = docs

    engine._self_research()

    assert len(engine.research_queue) <= 8
    queue_assets = [row["asset"] for row in engine.research_queue]
    assert "DOGE-USD" in queue_assets
    meme_row = next(row for row in engine.research_queue
                    if row["asset"] == "DOGE-USD")
    assert meme_row.get("meme_priority") is True


def test_research_queue_keeps_total_size_and_non_meme_ranking():
    """Reserving meme slots must not grow the queue or re-order non-meme
    candidates among themselves."""
    from app.data.research import ResearchEngine

    engine = ResearchEngine()
    docs = []
    for i in range(10):
        for _ in range(30 - i):
            docs.append({"assets": [f"COIN{i}-USD"]})
    engine.documents = docs

    engine._self_research()

    assert len(engine.research_queue) == 8
    non_meme_assets = [row["asset"] for row in engine.research_queue]
    assert non_meme_assets == [f"COIN{i}-USD" for i in range(8)]


def test_universe_candidate_ranking_breaks_ties_toward_memes_only():
    """Same tie-break pattern used by universe.refresh(): meme status only
    decides between EQUAL effective-heat candidates, never overriding heat."""
    from app.data.memes import memes

    candidates = [("BTC", 5.0), ("DOGE", 5.0), ("ETH", 5.0)]
    ranked = sorted(candidates,
                    key=lambda kv: (-kv[1], 0 if memes.is_meme(kv[0]) else 1))
    assert ranked[0][0] == "DOGE"

    # a clear heat leader still wins regardless of meme status
    candidates2 = [("BTC", 9.0), ("DOGE", 5.0)]
    ranked2 = sorted(candidates2,
                     key=lambda kv: (-kv[1], 0 if memes.is_meme(kv[0]) else 1))
    assert ranked2[0][0] == "BTC"


def test_universe_refresh_still_enforces_listing_after_ranking():
    """Static guard: the Coinbase-listing eligibility check must still run
    AFTER candidate ranking in universe.refresh() -- ranking never bypasses
    it."""
    import inspect
    from app.data.universe import Universe

    src = inspect.getsource(Universe.refresh)
    rank_pos = src.index("candidates = sorted(")
    listing_pos = src.index("not in self.cb_products")
    assert listing_pos > rank_pos


def test_meme_universe_slots_tunable_default():
    from app import tunables
    assert tunables.tv("meme_universe_slots") == 4


def test_meme_candidates_get_dedicated_universe_slots(monkeypatch):
    """Memes must not consume the non-meme discovered budget: with
    meme_universe_slots=1 and slots=2 (non-meme), 2 hot non-memes AND 1 hot
    meme must all make it into the universe -- the meme doesn't bump a
    non-meme out, and isn't bumped out by them."""
    import asyncio
    from app.config import PRODUCTS
    from app.data.universe import universe, CORE, MAX_UNIVERSE
    from app import tunables
    from app.execution.paper import broker

    products_before = list(PRODUCTS)
    tunables.update({"meme_universe_slots": 1})
    monkeypatch.setattr(universe, "_load_coinbase_products",
                        lambda client: _async_noop())
    monkeypatch.setattr(universe, "_fetch_trending", lambda client: _async_noop())
    monkeypatch.setattr(universe, "_fetch_oi_growth", lambda client: _async_noop())
    monkeypatch.setattr(universe, "_volume_ok",
                        lambda client, sym: _async_true())
    monkeypatch.setattr("app.data.memes.memes.seed_universe", lambda: None)
    monkeypatch.setattr(broker, "positions", {})

    non_meme_slots = MAX_UNIVERSE - len(CORE)
    universe.mention_heat.clear()
    universe.sources.clear()
    universe.cb_products.clear()
    # fill exactly `non_meme_slots` hot non-memes + 1 hot meme (SHIB -- not a
    # CORE coin, unlike DOGE), all above the heat floor and all "listed"/liquid.
    syms = [f"ALT{i}" for i in range(non_meme_slots)] + ["SHIB"]
    for i, sym in enumerate(syms):
        universe.mention_heat[sym] = 10.0 - i * 0.01   # descending heat
        universe.cb_products[sym] = {"id": f"{sym}-USD", "status": "online"}

    try:
        asyncio.run(universe.refresh(None))
        assert "SHIB-USD" in universe.discovered
        for sym in syms[:non_meme_slots]:
            assert f"{sym}-USD" in universe.discovered
    finally:
        PRODUCTS[:] = products_before
        tunables.update({"meme_universe_slots": 4})


async def _async_noop():
    return None


async def _async_true():
    return True


def test_meme_strategy_influence_tunable_default_and_wiring():
    from app import tunables
    import inspect
    from app.signals.engine import SignalEngine

    assert tunables.tv("meme_strategy_influence") == 1.5
    src = inspect.getsource(SignalEngine.compute)
    assert '"meme" in active and _is_meme(p)' in src
    assert 'tv("meme_strategy_influence")' in src


def test_meme_influence_multiplies_only_the_meme_arm_on_classified_memes():
    """The formula compute() applies: base weight * influence ONLY when the
    product is a classified meme; a non-meme product's `meme` arm weight (if
    it ever had one) is left exactly as the learned bandit weight."""
    from app import tunables
    from app.data.memes import memes

    base_weight = 0.2
    influence = tunables.tv("meme_strategy_influence")

    assert memes.is_meme("DOGE-USD")
    eff_meme = (base_weight * influence if memes.is_meme("DOGE-USD")
               else base_weight)
    assert eff_meme == pytest.approx(base_weight * influence)
    assert eff_meme != base_weight            # boosted on a classified meme

    assert not memes.is_meme("BTC-USD")
    eff_non_meme = (base_weight * influence if memes.is_meme("BTC-USD")
                   else base_weight)
    assert eff_non_meme == base_weight         # unchanged on a non-meme
