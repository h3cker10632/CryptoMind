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
    top_n: int = 25
    interval_sec: float = 300.0
    out_path: str = "invo_snapshots.json"
    timeout: float = 15.0
    max_retries: int = 3

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


class InvoPoller:
    def __init__(self, cfg: PollerConfig,
                 mapper: Callable[..., List[Dict[str, Any]]] = default_mapper):
        if httpx is None:
            sys.exit("[poller] httpx not installed — pip install httpx")
        self.cfg = cfg
        self.mapper = mapper
        self._stop = False

    # ---- transport (reusable; auth + retry + backoff) ----
    def _get(self, client, path: str, params=None):
        url = path if path.startswith("http") else self.cfg.base_url + path
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = client.get(url, params=params, timeout=self.cfg.timeout,
                               headers={"Authorization": f"Bearer {self.cfg.token}",
                                        "User-Agent": "invo-signal-study/1.0"})
                if r.status_code == 429:                    # rate limited — back off
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:                          # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"GET {path} failed after {self.cfg.max_retries} tries: {last}")

    def fetch_once(self) -> Dict[str, Any]:
        with httpx.Client() as client:
            lb = self._get(client, self.cfg.leaderboard_path)
            fetch_positions = None
            if self.cfg.positions_tmpl:
                def fetch_positions(tid, _c=client):
                    return self._get(_c, self.cfg.positions_tmpl.format(id=tid))
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
