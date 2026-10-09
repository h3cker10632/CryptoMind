"""Price feed: one failing coin must not take the feed down."""
import asyncio


def test_one_failing_coin_does_not_take_the_feed_down(monkeypatch):
    """A 429 / 404 on one coin skips that coin for the pass (its stale price
    is dropped); the feed only fails when most coins do."""
    import time as _t
    from app.data import market as M
    monkeypatch.setattr(M, "PRODUCTS", ["BTC-USD", "ETH-USD", "AERO-USD"])
    monkeypatch.setattr(M.db, "log_event", lambda *a: None)

    async def nosleep(_):
        return None
    monkeypatch.setattr(M.asyncio, "sleep", nosleep)
    bad = {"AERO-USD"}

    async def refresh(self, client, p):
        if p in bad:
            raise RuntimeError("Client error '429 Too Many Requests'")
        self.tickers[p] = {"price": 1.0, "ts": _t.time()}
    monkeypatch.setattr(M.MarketData, "refresh_product", refresh)
    m = M.MarketData()
    m.tickers["AERO-USD"] = {"price": 1.0, "ts": _t.time() - 3600}   # stale
    asyncio.run(m.refresh_all(None))                                  # no raise
    assert "AERO-USD" not in m.tickers and m.price("BTC-USD") == 1.0
    bad |= {"ETH-USD"}
    import pytest
    with pytest.raises(RuntimeError):                                 # most coins down
        asyncio.run(m.refresh_all(None))
