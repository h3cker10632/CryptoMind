"""Tests for crawl4ai producer auto source-discovery (tools/crawl4ai_signal).

The producer is a standalone, license-safe tool: the trading core never imports
it. These tests cover the "let the system find the websites" path — query
building, RSS parsing, and the auto run_once merge — with the network and LLM
stubbed so nothing external is hit.
"""
from tools.crawl4ai_signal import producer as P


SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>news</title>
  <item>
    <title>Bitcoin ETF inflows hit record</title>
    <link>https://news.google.com/rss/articles/AAA</link>
    <description>&lt;a href="x"&gt;Spot ETFs saw strong net inflows today.&lt;/a&gt;</description>
  </item>
  <item>
    <title>Analysts eye BTC breakout</title>
    <link>https://news.google.com/rss/articles/BBB</link>
    <description>Momentum building above resistance.</description>
  </item>
</channel></rss>"""


def test_asset_query_mapping():
    assert P.asset_query("BTC-USD") == "Bitcoin crypto"
    assert P.asset_query("ETH-USD") == "Ethereum crypto"
    # unknown ticker falls back to the bare base
    assert P.asset_query("XYZ-USD") == "XYZ crypto"


def test_discover_headlines_parses_rss(monkeypatch):
    class _Resp:
        text = SAMPLE_RSS
        def raise_for_status(self): pass

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    items = P.discover_headlines("Bitcoin crypto", max_items=6)
    assert len(items) == 2
    titles = [t for t, _l, _s in items]
    assert "Bitcoin ETF inflows hit record" in titles
    # links extracted
    assert items[0][1].startswith("https://news.google.com/rss/articles/")
    # description HTML stripped into a snippet
    assert "<a" not in items[0][2]
    assert "inflows" in items[0][2].lower()


def test_discover_headlines_graceful_on_error(monkeypatch):
    import httpx
    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(httpx, "get", _boom)
    assert P.discover_headlines("Bitcoin crypto") == []


def test_discover_sources_shape(monkeypatch):
    monkeypatch.setattr(P, "discover_headlines",
                        lambda q, **k: [("t1", "https://a/1", "s1"),
                                        ("t2", "https://a/2", "s2")])
    src = P.discover_sources(["BTC-USD", "ETH-USD"], max_urls=2)
    assert set(src) == {"BTC-USD", "ETH-USD"}
    assert src["BTC-USD"] == ["https://a/1", "https://a/2"]


def test_run_once_autodiscover(monkeypatch):
    """Auto mode: no manual sources, the producer discovers its own, scores the
    discovered headlines with the (stubbed) LLM, and pushes rows."""
    # discovery returns headlines; article fetch returns nothing (redirect-gated)
    monkeypatch.setattr(P, "discover_headlines",
                        lambda q, **k: [("Bullish headline", "https://x/1", "up")])
    monkeypatch.setattr(P, "fetch_text", lambda u, **k: "")
    monkeypatch.setattr(P, "llm_lean", lambda asset, text: (0.6, "bullish news"))
    pushed = {}
    def _push(rows):
        pushed["rows"] = rows
        return {"accepted": len(rows), "rejected": 0}
    import app.data.ingest as ingest
    monkeypatch.setattr(ingest, "push", _push)

    out = P.run_once(sources={}, push=True, autodiscover=True,
                     assets=["BTC-USD", "ETH-USD"])
    assert out["ok"] is True
    assert out["autodiscovered"] == 2          # both assets got headlines
    assert out["scored"] == 2
    # headline-only blob was scored even though article fetch returned ""
    assert pushed["rows"] and pushed["rows"][0]["value"] == 0.6
    assert pushed["rows"][0]["source"] == "crawl4ai"


def test_run_once_no_sources_no_auto():
    out = P.run_once(sources={}, push=False, autodiscover=False)
    assert out["ok"] is False
    assert "no sources" in out["error"]
