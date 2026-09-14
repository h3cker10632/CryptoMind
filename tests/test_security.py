"""Control-API auth guard tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import security


class _Req:
    def __init__(self, host, headers=None, query=None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}
        self.query_params = query or {}


def test_loopback_allowed_by_default():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "1"
    assert security.check(_Req("127.0.0.1")) is True


def test_remote_without_token_denied():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    assert security.check(_Req("203.0.113.9")) is False


def test_remote_with_correct_token_allowed():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    tok = security.token()
    assert security.check(_Req("203.0.113.9",
                              headers={"authorization": f"Bearer {tok}"})) is True


def test_wrong_token_denied():
    os.environ["CRYPTOMIND_ALLOW_LOOPBACK"] = "0"
    assert security.check(_Req("203.0.113.9",
                              headers={"authorization": "Bearer nope"})) is False
