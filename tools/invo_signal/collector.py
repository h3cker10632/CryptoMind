"""Data contract for Invo positioning snapshots + how to obtain them.

IMPORTANT — legal / ToS boundary
--------------------------------
This module does NOT ship a scraper or a reverse-engineered API client for
app.invoapp.com. Scraping behind Invo's login, or hitting their private API to
bypass the paid copy mechanism, likely violates their Terms of Service. That is
your call to make with your own authorized access.

What this module DOES provide is a clean data *contract*: if you obtain snapshots
through a means you're comfortable with (a manual export, an authorized/official
API session, etc.), drop them in as JSON matching `InvoSnapshot` and the rest of
the harness will evaluate whether the signal is worth anything. The live-fetch
adapter below is an intentionally empty stub you fill in with your own
authorized session — it raises by default.

Snapshot JSON schema (a list of these, one per capture time):
{
  "ts": 1789435352.0,               # unix seconds the snapshot was taken
  "traders": [
    {
      "id": "some_trader",
      "rank": 1,                     # 1 = best; used for rank weighting
      "score": 0.83,                 # optional quality score (e.g. win-rate)
      "positions": [
        {"asset": "BTC", "direction": 1, "notional": 12000.0, "leverage": 5},
        {"asset": "SOL", "direction": -1, "notional": 4000.0, "leverage": 3}
      ]
    }
  ]
}
direction: +1 long, -1 short. notional in quote currency (USD). asset is the
bare symbol (BTC, SOL) — matched to your traded universe in signal.py.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Position:
    asset: str
    direction: int
    notional: float
    leverage: float = 1.0


@dataclass
class Trader:
    id: str
    rank: int
    score: float
    positions: List[Position] = field(default_factory=list)


@dataclass
class InvoSnapshot:
    ts: float
    traders: List[Trader] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InvoSnapshot":
        traders = []
        for t in d.get("traders", []):
            pos = [Position(str(p["asset"]).upper(),
                            int(p.get("direction", 0)),
                            float(p.get("notional", 0.0)),
                            float(p.get("leverage", 1.0)))
                   for p in t.get("positions", [])]
            traders.append(Trader(str(t.get("id", "")),
                                  int(t.get("rank", 9999)),
                                  float(t.get("score", 0.0)),
                                  pos))
        return InvoSnapshot(float(d["ts"]), traders)


def load_snapshots(path: str) -> List[InvoSnapshot]:
    """Read a JSON file (a list of snapshot dicts) into InvoSnapshot objects."""
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):          # allow {"snapshots": [...]}
        raw = raw.get("snapshots", [])
    snaps = [InvoSnapshot.from_dict(d) for d in raw]
    snaps.sort(key=lambda s: s.ts)
    return snaps


def load_prices(path: str) -> Dict[str, list]:
    """Read a JSON price file: {"BTC": [[ts, close], ...], ...}. Sorted by ts."""
    with open(path) as fh:
        px = json.load(fh)
    return {a.upper(): sorted([[float(t), float(c)] for t, c in rows])
            for a, rows in px.items()}


def load_baseline(path: str) -> Dict[str, list]:
    """OPTIONAL existing-feature baseline for the orthogonality test.
    {"BTC": [[ts, funding_norm, oi_change, long_short], ...], ...}
    These mirror the features CryptoMind already has (derivatives.features)."""
    with open(path) as fh:
        b = json.load(fh)
    return {a.upper(): sorted([[float(v) for v in row] for row in rows])
            for a, rows in b.items()}


class LiveInvoSource:
    """Adapter stub for pulling snapshots from an authorized live session.

    Left unimplemented on purpose — fill `fetch()` in with YOUR authorized
    access if and only if you've decided that's within Invo's ToS. The harness
    never calls this automatically.
    """
    def fetch(self) -> InvoSnapshot:  # pragma: no cover - deliberate stub
        raise NotImplementedError(
            "No authorized Invo source configured. Provide collected snapshots "
            "as JSON (see collector docstring) instead, or implement fetch() "
            "with your own authorized/ToS-compliant access.")
