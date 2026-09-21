"""Meme-coin trading: classification, seed, discovery, and risk envelope."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_seed_universe_tags_and_heats(monkeypatch):
    _enable(True)
    from app.data.universe import universe
    universe.mention_heat.clear()
    universe.sources.clear()
    memes.seed_universe()
    for sym in SEED:
        assert universe.mention_heat.get(sym, 0) >= 2.0
        assert "meme" in universe.sources.get(sym, set())
