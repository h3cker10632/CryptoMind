"""app/data/sources.py against mocked HTTP: paging, retries, filtering and
the row formats the store expects (tools/data_sync.py's interface)."""
import asyncio
import json

import httpx

D, H = 86400, 3600


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro):
    return asyncio.run(coro)


def test_coinbase_products_keep_delisted_and_flag_stablecoins():
    from app.data import sources as S
    products = [
        {"id": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD", "status": "online"},
        {"id": "XRP-USD", "base_currency": "XRP", "quote_currency": "USD", "status": "delisted"},
        {"id": "USDT-USD", "base_currency": "USDT", "quote_currency": "USD", "status": "online"},
        {"id": "EURC-USD", "base_currency": "EURC", "quote_currency": "USD",
         "status": "online", "fx_stablecoin": True},
        {"id": "BTC-EUR", "base_currency": "BTC", "quote_currency": "EUR", "status": "online"},
    ]

    async def go():
        async with _client(lambda req: httpx.Response(200, json=products)) as c:
            return await S.coinbase_usd_products(c)
    out = {p["id"]: p for p in _run(go())}
    assert set(out) == {"BTC-USD", "XRP-USD", "USDT-USD", "EURC-USD"}
    assert out["XRP-USD"]["status"] == "delisted" and not out["XRP-USD"]["stable"]
    assert out["USDT-USD"]["stable"] and out["EURC-USD"]["stable"]


def test_coinbase_candles_page_300_bars_and_retry_rate_limits(monkeypatch):
    from app.data import sources as S
    t0 = 1_600_000_000 // D * D
    listed = t0 + 100 * D                               # first 100 days: not listed yet
    calls = {"n": 0, "429": 0}

    def handler(req):
        calls["n"] += 1
        if calls["429"] == 0:                            # first call: rate limited
            calls["429"] += 1
            return httpx.Response(429)
        q = req.url.params
        s, e, g = int(q["start"]), int(q["end"]), int(q["granularity"])
        assert (e - s) // g + 1 <= 300                   # never more than Coinbase allows
        bars = [[t, 9.0, 11.0, 10.0, 10.5, 5.0] for t in range(max(s, listed), e + 1, g)]
        return httpx.Response(200, json=list(reversed(bars)))    # newest first, like Coinbase

    async def nosleep(*_):
        return None
    monkeypatch.setattr(S.asyncio, "sleep", nosleep)

    async def go():
        async with _client(handler) as c:
            return await S.coinbase_candles(c, "BTC-USD", D, t0, t0 + 999 * D)
    rows = _run(go())
    ts = [r[0] for r in rows]
    assert ts == list(range(listed, t0 + 1000 * D, D))   # ascending, deduped, complete
    assert rows[0] == [listed, 9.0, 11.0, 10.0, 10.5, 5.0]
    assert calls["n"] == 1 + 4                           # 1000 days = 4 pages, + the retried 429


def test_hyperliquid_names_map_k_prefixed_contracts():
    from app.data import sources as S
    meta = {"universe": [{"name": "BTC"}, {"name": "kPEPE"}, {"name": "kSHIB"}, {"name": "ETH"}]}

    async def go():
        async with _client(lambda req: httpx.Response(200, json=meta)) as c:
            return await S.hyperliquid_names(c)
    assert _run(go()) == {"BTC": "BTC", "PEPE": "kPEPE", "SHIB": "kSHIB", "ETH": "ETH"}


def test_hyperliquid_funding_pages_by_500_rows():
    from app.data import sources as S
    start = 1_700_000_000 // H * H
    n_rows = 1200
    seen = []

    def handler(req):
        body = json.loads(req.content)
        assert body["type"] == "fundingHistory" and body["coin"] == "kPEPE"
        t = body["startTime"]
        seen.append(t)
        out = []
        for k in range(n_rows):
            ms = (start + k * H) * 1000 + 37             # a few ms past the hour, as live
            if ms >= t and len(out) < 500:
                out.append({"coin": "kPEPE", "fundingRate": "0.0000125", "premium": "0",
                            "time": ms})
        return httpx.Response(200, json=out)

    async def go():
        async with _client(handler) as c:
            return await S.hyperliquid_funding(c, "kPEPE", start, start + n_rows * H)
    rows = _run(go())
    assert len(rows) == n_rows and len(seen) == 3
    assert rows[0] == [start, 0.0000125] and rows[-1][0] == start + (n_rows - 1) * H


def test_deribit_funding_uses_hourly_rate_month_by_month():
    from app.data import sources as S
    start = 1_556_668_800
    end = start + 75 * D
    windows = []

    def handler(req):
        q = req.url.params
        assert q["instrument_name"] == "BTC-PERPETUAL"
        s, e = int(q["start_timestamp"]) // 1000, int(q["end_timestamp"]) // 1000
        windows.append((s, e))
        res = [{"timestamp": (t * 1000) - 5, "interest_1h": 1e-5, "interest_8h": 8e-5}
               for t in range((s + H - 1) // H * H, e + 1, H)]
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": res})

    async def go():
        async with _client(handler) as c:
            return await S.deribit_funding(c, "BTC", start, end)
    rows = _run(go())
    assert len(windows) == 3                              # 75 days in 30-day requests
    assert all(r[1] == 1e-5 for r in rows)               # the HOURLY rate, not 8h
    assert [r[0] for r in rows] == sorted({r[0] for r in rows})
    assert rows[0][0] == start and rows[-1][0] <= end


def test_data_sync_runs_end_to_end_on_mocked_apis(tmp_path, monkeypatch):
    """tools/data_sync.sync: products -> daily candles into the store,
    hourly candles, Hyperliquid + Deribit funding."""
    import importlib.util
    import os
    from app.data import store
    monkeypatch.setattr(store, "STORE", str(tmp_path))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("data_sync",
                                                  os.path.join(root, "tools", "data_sync.py"))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    now = 1_760_000_000 // D * D
    monkeypatch.setattr(ds.time, "time", lambda: now)
    monkeypatch.setattr(ds, "hourly_universe", lambda top=30: ["BTC-USD"])

    def handler(req):
        url = str(req.url)
        if url.endswith("/products"):
            return httpx.Response(200, json=[
                {"id": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD",
                 "status": "online"},
                {"id": "USDC-USD", "base_currency": "USDC", "quote_currency": "USD",
                 "status": "online"}])
        if "/candles" in url:
            q = req.url.params
            s, e, g = int(q["start"]), int(q["end"]), int(q["granularity"])
            lo = now - 30 * g
            return httpx.Response(200, json=[[t, 99.0, 101.0, 100.0, 100.5, 7.0]
                                             for t in range(max(s, lo), e + 1, g)][::-1])
        if "hyperliquid" in url:
            body = json.loads(req.content)
            if body["type"] == "meta":
                return httpx.Response(200, json={"universe": [{"name": "BTC"}]})
            return httpx.Response(200, json=[{"coin": "BTC", "fundingRate": "0.00001",
                                              "time": (now - 5 * H) * 1000}])
        if "deribit" in url:
            return httpx.Response(200, json={"result": [
                {"timestamp": (now - 2 * H) * 1000, "interest_1h": 2e-5}]})
        return httpx.Response(404)

    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)
    monkeypatch.setattr(httpx, "AsyncClient", Client)

    async def nosleep(*_):
        return None
    from app.data import sources
    monkeypatch.setattr(sources.asyncio, "sleep", nosleep)
    ds.asyncio.run(ds.sync(daily_years=1, hourly_years=1, say=lambda m: None))
    daily, _ = store.load_candles(D)
    assert set(daily) == {"BTC-USD"} and len(daily["BTC-USD"]) == 30   # stablecoin skipped
    hourly, _ = store.load_candles(H)
    assert len(hourly["BTC-USD"]) == 30
    hl, _ = store.load_funding("hyperliquid")
    db, _ = store.load_funding("deribit")
    assert hl["BTC"] == [[now - 5 * H, 0.00001]]
    assert db["BTC"] == [[now - 2 * H, 2e-5]] and db["ETH"] == [[now - 2 * H, 2e-5]]
    assert (tmp_path / "last_sync").read_text() == str(now)   # the scheduler's stamp
