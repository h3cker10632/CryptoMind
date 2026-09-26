"""Authorized Invo positioning poller — produces snapshot JSON for the study.

WHAT THIS IS
------------
A small, standalone scheduler that periodically calls an Invo endpoint *you*
configure, maps the response into the `InvoSnapshot` schema (see collector.py),
and appends it to an output file the edge study reads. It handles the boring,
reusable parts — bearer auth, timeouts, retry with backoff, polite rate limiting,
atomic append, graceful shutdown — so all you supply is (a) your authorized
access and (b) how to read your account's response shape.

WHAT THIS IS NOT
----------------
It ships NO reverse-engineered endpoints and NO credentials. The response→schema
`mapper` is a stub that raises until you implement it, and endpoint paths come
from env/args, not hardcoded knowledge of Invo's private API. Whether your access
is within Invo's Terms of Service is your decision. This tool is fully decoupled
from app/ and never touches live trading.

CONFIG (env vars; nothing secret is ever logged)
------------------------------------------------
  INVO_API_BASE        e.g. https://app.invoapp.com            (required)
  INVO_TOKEN           bearer token for YOUR authorized session (required)
  INVO_LEADERBOARD     path returning the ranked traders        (required)
  INVO_POSITIONS_TMPL  optional path template per trader, e.g. "/api/traders/{id}/positions"
  INVO_TOP_N           how many top traders to track (default 25)
  INVO_INTERVAL_SEC    seconds between snapshots (default 300)
  INVO_OUT             output file (default ./invo_snapshots.json)

RUN
---
  python -m tools.invo_signal.poller                 # loops forever
  python -m tools.invo_signal.poller --once          # single snapshot (smoke test)
  python -m tools.invo_signal.poller --max-iters 10  # bounded run
"""
from __future__ import annotations
import os, sys, json, time, signal, argparse, tempfile
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


@dataclass
class PollerConfig:
    base_url: str
    token: str
    leaderboard_path: str
    positions_tmpl: Optional[str] = None
    positions_method: str = "GET"     # per-trader positions request method
    positions_body: str = ""          # JSON body template for POST positions ({id})
    top_n: int = 25
    interval_sec: float = 300.0
    out_path: str = "invo_snapshots.json"
    timeout: float = 15.0
    max_retries: int = 3
    # ---- optional auto token-refresh ----
    refresh_path: str = ""            # refresh endpoint (path or full URL); blank = disabled
    refresh_token: str = ""           # long-lived refresh token
    refresh_body: str = ""            # JSON body template w/ {refresh_token}; blank = Bearer header
    token_json_path: str = "access_token"   # dotted path to the new access token in the response
    refresh_rotates_path: str = ""    # optional dotted path to a rotated refresh token
    # ---- leaderboard request shape ----
    method: str = "GET"               # GET or POST (some endpoints, e.g. get_users, are POST)
    body: str = ""                    # JSON body sent with a POST leaderboard request

    @staticmethod
    def from_env() -> "PollerConfig":
        def req(k):
            v = os.environ.get(k)
            if not v:
                sys.exit(f"[poller] missing required env var {k} — see module docstring")
            return v
        return PollerConfig(
            base_url=req("INVO_API_BASE").rstrip("/"),
            token=req("INVO_TOKEN"),
            leaderboard_path=req("INVO_LEADERBOARD"),
            positions_tmpl=os.environ.get("INVO_POSITIONS_TMPL"),
            top_n=int(os.environ.get("INVO_TOP_N", "25")),
            interval_sec=float(os.environ.get("INVO_INTERVAL_SEC", "300")),
            out_path=os.environ.get("INVO_OUT", "invo_snapshots.json"),
            method=os.environ.get("INVO_METHOD", "GET"),
            body=os.environ.get("INVO_BODY", ""),
            refresh_path=os.environ.get("INVO_REFRESH_PATH", ""),
            refresh_token=os.environ.get("INVO_REFRESH_TOKEN", ""),
            refresh_body=os.environ.get("INVO_REFRESH_BODY", ""),
            token_json_path=os.environ.get("INVO_TOKEN_PATH", "access_token"),
            refresh_rotates_path=os.environ.get("INVO_REFRESH_ROTATES_PATH", ""),
        )


# --------------------------------------------------------------------------
# The ONE piece you implement: map your account's JSON into the snapshot schema.
# It receives the raw leaderboard payload and (if you set INVO_POSITIONS_TMPL)
# a callable to fetch a given trader's positions. It must return a list of
# trader dicts exactly as collector.InvoSnapshot.from_dict expects:
#   [{"id","rank","score","positions":[{"asset","direction","notional","leverage"}]}]
# --------------------------------------------------------------------------
def default_mapper(leaderboard_raw: Any,
                   fetch_positions: Optional[Callable[[str], Any]],
                   top_n: int) -> List[Dict[str, Any]]:  # pragma: no cover - stub
    raise NotImplementedError(
        "Implement the response→schema mapper for YOUR authorized Invo response.\n"
        "Example skeleton (adapt field names to what your endpoint returns):\n\n"
        "    traders = []\n"
        "    for i, row in enumerate(leaderboard_raw['data'][:top_n], start=1):\n"
        "        tid = row['id']\n"
        "        pos_raw = fetch_positions(tid) if fetch_positions else row.get('positions', [])\n"
        "        positions = [{\n"
        "            'asset': p['coin'], 'direction': 1 if p['side']=='long' else -1,\n"
        "            'notional': float(p['sizeUsd']), 'leverage': float(p.get('lev',1))\n"
        "        } for p in pos_raw]\n"
        "        traders.append({'id': tid, 'rank': i,\n"
        "                        'score': float(row.get('winRate', 0.0)), 'positions': positions})\n"
        "    return traders\n")


def _dig_path(obj: Any, path: str):
    """Fetch a possibly-dotted key path from a nested dict; None if missing."""
    if not path:
        return None
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


class InvoPoller:
    def __init__(self, cfg: PollerConfig,
                 mapper: Callable[..., List[Dict[str, Any]]] = default_mapper,
                 on_new_token: Optional[Callable[[str, Optional[str]], None]] = None):
        if httpx is None:
            sys.exit("[poller] httpx not installed — pip install httpx")
        self.cfg = cfg
        self.mapper = mapper
        # called with (new_access_token, new_refresh_token_or_None) after a
        # successful refresh, so the caller can persist it (e.g. to settings).
        self.on_new_token = on_new_token
        self._stop = False

    def _refresh_enabled(self) -> bool:
        return bool(self.cfg.refresh_path and self.cfg.refresh_token)

    # ---- auth: mint a fresh access token from the refresh token ----
    def _refresh(self, client) -> bool:
        """POST to the configured refresh endpoint and swap in a new access
        token. Returns True on success; raises on a hard failure."""
        cfg = self.cfg
        url = (cfg.refresh_path if cfg.refresh_path.startswith("http")
               else cfg.base_url + cfg.refresh_path)
        headers = {"User-Agent": "invo-signal-study/1.0"}
        body = None
        if cfg.refresh_body:
            tmpl = cfg.refresh_body.replace("{refresh_token}", cfg.refresh_token)
            try:
                body = json.loads(tmpl)
            except Exception:
                body = {"refresh_token": cfg.refresh_token}
        else:
            # no body template → present the refresh token as a Bearer header
            headers["Authorization"] = f"Bearer {cfg.refresh_token}"
        r = client.post(url, json=body, headers=headers, timeout=cfg.timeout)
        r.raise_for_status()
        data = r.json()
        new_access = _dig_path(data, cfg.token_json_path or "access_token")
        if not new_access:
            raise RuntimeError(
                f"refresh response had no access token at '{cfg.token_json_path}'")
        self.cfg.token = str(new_access)
        new_refresh = (_dig_path(data, cfg.refresh_rotates_path)
                       if cfg.refresh_rotates_path else None)
        if new_refresh:
            self.cfg.refresh_token = str(new_refresh)
        if self.on_new_token:
            try:
                self.on_new_token(self.cfg.token,
                                  str(new_refresh) if new_refresh else None)
            except Exception:                               # persistence is best-effort
                pass
        return True

    # ---- transport (reusable; auth + retry + backoff + auto-refresh) ----
    def _request(self, client, path: str, method: str = "GET",
                 json_body=None, params=None):
        url = path if path.startswith("http") else self.cfg.base_url + path
        method = (method or "GET").upper()
        last = None
        did_refresh = False
        for attempt in range(self.cfg.max_retries):
            try:
                headers = {"Authorization": f"Bearer {self.cfg.token}",
                           "User-Agent": "invo-signal-study/1.0"}
                if method == "POST":
                    r = client.post(url, json=json_body, params=params,
                                    timeout=self.cfg.timeout, headers=headers)
                else:
                    r = client.get(url, params=params,
                                   timeout=self.cfg.timeout, headers=headers)
                if r.status_code == 429:                    # rate limited — back off
                    time.sleep(2 ** attempt)
                    continue
                # expired/invalid token → refresh ONCE, then retry immediately
                if r.status_code in (401, 403) and self._refresh_enabled() and not did_refresh:
                    did_refresh = True
                    self._refresh(client)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:                          # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{method} {path} failed after {self.cfg.max_retries} tries: {last}")

    def _get(self, client, path: str, params=None):
        return self._request(client, path, "GET", params=params)

    def _body_dict(self):
        """Parse the configured JSON body string for a POST leaderboard request."""
        if not self.cfg.body:
            return None
        try:
            return json.loads(self.cfg.body)
        except Exception:
            return None

    def _positions_body_dict(self, tid):
        """Parse the per-trader positions POST body, substituting the {id}."""
        if not self.cfg.positions_body:
            return None
        try:
            return json.loads(self.cfg.positions_body.replace("{id}", str(tid)))
        except Exception:
            return None

    def fetch_positions_for(self, client, tid):
        """Fetch one trader's positions using the configured method (GET/POST)."""
        path = self.cfg.positions_tmpl.format(id=tid)
        if (self.cfg.positions_method or "GET").upper() == "POST":
            return self._request(client, path, "POST",
                                 json_body=self._positions_body_dict(tid))
        return self._request(client, path, "GET")

    def fetch_leaderboard(self, client):
        """Fetch the leaderboard using the configured method + body (GET or POST)."""
        return self._request(client, self.cfg.leaderboard_path,
                             self.cfg.method, json_body=self._body_dict())

    def fetch_once(self) -> Dict[str, Any]:
        with httpx.Client() as client:
            lb = self.fetch_leaderboard(client)
            fetch_positions = None
            if self.cfg.positions_tmpl:
                def fetch_positions(tid, _c=client):
                    return self.fetch_positions_for(_c, tid)
            traders = self.mapper(lb, fetch_positions, self.cfg.top_n)
        return {"ts": time.time(), "traders": traders}

    def peek(self) -> None:
        """Fetch the raw leaderboard (and one trader's positions if a template is
        set) and pretty-print it — so you can see field names before writing the
        mapper. No mapper required; nothing is written."""
        with httpx.Client() as client:
            lb = self._get(client, self.cfg.leaderboard_path)
            print("=== RAW LEADERBOARD (first 2000 chars) ===")
            print(json.dumps(lb, indent=2)[:2000])
            if self.cfg.positions_tmpl:
                # try to grab an id from common shapes to demo the positions call
                rows = lb.get("data", lb) if isinstance(lb, dict) else lb
                tid = None
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    tid = rows[0].get("id") or rows[0].get("userId") or rows[0].get("trader_id")
                if tid is not None:
                    pos = self._get(client, self.cfg.positions_tmpl.format(id=tid))
                    print(f"\n=== RAW POSITIONS for trader {tid} (first 2000 chars) ===")
                    print(json.dumps(pos, indent=2)[:2000])
                else:
                    print("\n[peek] couldn't auto-detect a trader id field — "
                          "inspect the leaderboard above for the id key.")

    # ---- persistence (atomic append to a JSON list) ----
    def append_snapshot(self, snap: Dict[str, Any]):
        data = []
        if os.path.exists(self.cfg.out_path):
            try:
                with open(self.cfg.out_path) as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    data = data.get("snapshots", [])
            except Exception:
                data = []
        data.append(snap)
        d = os.path.dirname(os.path.abspath(self.cfg.out_path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.cfg.out_path)     # atomic
        return len(data)

    def run(self, once: bool = False, max_iters: Optional[int] = None):
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        i = 0
        while not self._stop:
            i += 1
            try:
                snap = self.fetch_once()
                total = self.append_snapshot(snap)
                print(f"[poller] snapshot {i}: {len(snap['traders'])} traders "
                      f"-> {self.cfg.out_path} ({total} total)")
            except NotImplementedError as e:
                sys.exit(f"[poller] mapper not implemented:\n{e}")
            except Exception as e:                          # noqa: BLE001
                print(f"[poller] snapshot {i} failed: {e}", file=sys.stderr)
            if once or (max_iters and i >= max_iters):
                break
            for _ in range(int(self.cfg.interval_sec)):
                if self._stop:
                    break
                time.sleep(1)
        print("[poller] stopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Authorized Invo positioning poller")
    ap.add_argument("--once", action="store_true", help="take one snapshot and exit")
    ap.add_argument("--peek", action="store_true",
                    help="print the raw API response (no mapper needed) to design your mapper")
    ap.add_argument("--max-iters", type=int, default=None)
    a = ap.parse_args()
    poller = InvoPoller(PollerConfig.from_env())
    if a.peek:
        poller.peek()
    else:
        poller.run(once=a.once, max_iters=a.max_iters)
