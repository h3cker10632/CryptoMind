"""Invo auto token-refresh: a 401 triggers one refresh + retry, the new access
token is swapped in and persisted, and both refresh styles (JSON body / Bearer
header) work."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from tools.invo_signal.poller import InvoPoller, PollerConfig


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._p = payload or {}
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, get_seq, post_resp=None):
        self.get_seq = list(get_seq)
        self.post_resp = post_resp
        self.gets = []      # Authorization headers seen on GET
        self.posts = []     # full POST calls seen
    def get(self, url, params=None, timeout=None, headers=None):
        self.gets.append((headers or {}).get("Authorization"))
        return self.get_seq.pop(0)
    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers or {}})
        return self.post_resp


def test_refresh_on_401_body_style_and_persist():
    cfg = PollerConfig(base_url="https://x", token="OLD", leaderboard_path="/lb",
                       refresh_path="/auth/refresh", refresh_token="RT",
                       refresh_body='{"refresh_token":"{refresh_token}"}',
                       token_json_path="access_token", max_retries=3)
    seen = {}
    p = InvoPoller(cfg, on_new_token=lambda a, r: seen.update(access=a, refresh=r))
    client = _Client(get_seq=[_Resp(401), _Resp(200, {"ok": 1})],
                     post_resp=_Resp(200, {"access_token": "NEW"}))
    out = p._get(client, "/lb")
    assert out == {"ok": 1}
    assert cfg.token == "NEW"                          # token swapped in
    assert client.gets[0].endswith("OLD")             # 1st try old token
    assert client.gets[1].endswith("NEW")             # retry new token
    assert client.posts[0]["json"] == {"refresh_token": "RT"}   # template substituted
    assert seen["access"] == "NEW"                     # persisted via callback


def test_refresh_bearer_header_style_dotted_path():
    cfg = PollerConfig(base_url="https://x", token="OLD", leaderboard_path="/lb",
                       refresh_path="/auth/refresh", refresh_token="RT",
                       refresh_body="", token_json_path="data.token", max_retries=2)
    p = InvoPoller(cfg)
    client = _Client(get_seq=[_Resp(403), _Resp(200, {"ok": 2})],
                     post_resp=_Resp(200, {"data": {"token": "NEW2"}}))
    out = p._get(client, "/lb")
    assert out == {"ok": 2}
    assert cfg.token == "NEW2"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer RT"  # no body → Bearer
    assert client.posts[0]["json"] is None


def test_refresh_rotates_refresh_token():
    cfg = PollerConfig(base_url="https://x", token="OLD", leaderboard_path="/lb",
                       refresh_path="/auth/refresh", refresh_token="RT1",
                       refresh_body='{"refresh_token":"{refresh_token}"}',
                       token_json_path="access_token",
                       refresh_rotates_path="refresh_token", max_retries=2)
    seen = {}
    p = InvoPoller(cfg, on_new_token=lambda a, r: seen.update(access=a, refresh=r))
    client = _Client(get_seq=[_Resp(401), _Resp(200, {"ok": 3})],
                     post_resp=_Resp(200, {"access_token": "NEW", "refresh_token": "RT2"}))
    p._get(client, "/lb")
    assert cfg.token == "NEW" and cfg.refresh_token == "RT2"   # both rotated
    assert seen == {"access": "NEW", "refresh": "RT2"}


def test_no_refresh_configured_still_raises_on_401():
    cfg = PollerConfig(base_url="https://x", token="OLD", leaderboard_path="/lb", max_retries=1)
    p = InvoPoller(cfg)
    client = _Client(get_seq=[_Resp(401)])
    with pytest.raises(RuntimeError):
        p._get(client, "/lb")
    assert client.posts == []                          # never attempted a refresh
