"""Polymarket resolution: Gamma's condition lookup returns [] for many closed
markets, so resolution must fall back to the CLOB market's winner flags."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.markets.polymarket import client as C

CID = "0xcb5c6a56aaf7823c3148bb0f041fc6962ea4cdf29cffc7947819ff89253b39e8"


def _client(gamma, clob):
    cl = C.PolymarketClient() if hasattr(C, "PolymarketClient") else C.client.__class__()

    def fake_get(base, path, params=None):
        if base == C.GAMMA_BASE:
            return gamma
        if base == C.CLOB_BASE and path == f"/markets/{CID}":
            if isinstance(clob, Exception):
                raise clob
            return clob
        raise AssertionError((base, path))
    cl._get = fake_get
    return cl


def _clob(closed, w0, w1, p0=0.5, p1=0.5):
    return {"condition_id": CID, "closed": closed, "active": True,
            "tokens": [{"outcome": "Atlanta Dream", "price": p0, "winner": w0},
                       {"outcome": "Washington Mystics", "price": p1, "winner": w1}]}


def test_falls_back_to_clob_winner_when_gamma_is_empty():
    r = _client([], _clob(True, True, False, 1, 0)).resolution(CID)
    assert r["resolved"] and r["winning_index"] == 0 and r["prices"] == [1.0, 0.0]
    r = _client([], _clob(True, False, True, 0, 1)).resolution(CID)
    assert r["resolved"] and r["winning_index"] == 1


def test_open_or_ambiguous_clob_market_is_not_resolved():
    assert not _client([], _clob(False, False, False)).resolution(CID)["resolved"]
    assert not _client([], _clob(True, False, False)).resolution(CID)["resolved"]   # closed, no winner yet
    assert not _client([], _clob(True, True, True)).resolution(CID)["resolved"]


def test_no_data_anywhere_returns_none():
    assert _client([], RuntimeError("down")).resolution(CID) is None
