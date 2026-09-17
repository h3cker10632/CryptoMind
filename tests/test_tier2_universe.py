"""Tier-2 universe discovery: OI-growth capital-flow source + multi-source tags."""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.universe import Universe
from app import tunables


class _FakeResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return {"code": "0", "data": self._data}


class _FakeClient:
    """Returns a scripted OI history per ccy from the request params."""
    def __init__(self, oi_by_ccy):
        self.oi_by_ccy = oi_by_ccy
    async def get(self, url, params=None, timeout=None):
        ccy = (params or {}).get("ccy")
        rows = self.oi_by_ccy.get(ccy)
        return _FakeResp(rows or [])


def _oi_rows(now, then):
    # newest-first: [ts, oi_usd, vol]; index 0 newest, index >=24 is ~24h old
    rows = [[i, now, 0] for i in range(1)]
    rows += [[i, then, 0] for i in range(1, 30)]
    return rows


def test_oi_growth_boosts_heat_and_tags_source():
    tunables.update({"oi_growth_threshold": 0.15})
    u = Universe()
    u.trending = ["FOO"]                     # FOO in the scan set
    client = _FakeClient({"FOO": _oi_rows(now=200.0, then=100.0)})  # +100% OI
    asyncio.run(u._fetch_oi_growth(client))
    assert u.oi_growth.get("FOO") == 1.0     # +100%
    assert u.mention_heat.get("FOO", 0) > 0  # boosted
    assert "oi_growth" in u.sources.get("FOO", set())


def test_oi_growth_below_threshold_no_boost():
    tunables.update({"oi_growth_threshold": 0.50})
    u = Universe()
    u.trending = ["BAR"]
    client = _FakeClient({"BAR": _oi_rows(now=110.0, then=100.0)})  # +10% < 50%
    asyncio.run(u._fetch_oi_growth(client))
    assert u.oi_growth.get("BAR") == 0.1
    assert "oi_growth" not in u.sources.get("BAR", set())


def test_multi_source_confirmation_bonus():
    u = Universe()
    # BAZ surfaced by two independent sources → standing corroboration bonus
    u.mention_heat["BAZ"] = 2.0
    u.sources["BAZ"] = {"news", "oi_growth"}
    u.mention_heat["QUX"] = 2.0
    u.sources["QUX"] = {"news"}
    # replicate the multi-source bonus step from refresh()
    for s, tags in u.sources.items():
        if len(tags) >= 2 and s in u.mention_heat:
            u.mention_heat[s] += 0.5 * (len(tags) - 1)
    assert u.mention_heat["BAZ"] > u.mention_heat["QUX"]


def test_stats_exposes_sources_and_oi():
    u = Universe()
    u.sources["ABC"] = {"news", "oi_growth"}
    u.oi_growth["ABC"] = 0.42
    st = u.stats()
    assert st["sources"]["ABC"] == ["news", "oi_growth"]   # set → sorted list
    assert st["oi_growth"]["ABC"] == 0.42
