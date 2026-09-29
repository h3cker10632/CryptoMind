"""Real (on-chain) Polymarket execution — a DISABLED-BY-DEFAULT seam.

The operator chose "paper now, wire real so it can be switched on later." This
module is that switch, and it is intentionally inert until two independent gates
are both satisfied, so no code path can ever accidentally spend real USDC:

  1. settings `pm_live_enabled` is True, AND
  2. a funded Polygon wallet private key is present in the git-ignored
     `.secrets.json` under `pm_wallet_key`, AND the `py-clob-client` dependency
     is installed.

Even then, `place_order` performs the credential/dependency preflight and raises
a clear, structured error rather than silently submitting — turning it live is a
deliberate, reviewed follow-up (build the EIP-712 order via py_clob_client,
confirm on a tiny stake first). The paper engine NEVER calls this unless the
operator explicitly flips live mode; `status()` is safe to call anytime and is
what the dashboard shows.
"""
from __future__ import annotations

from ... import settings as app_settings


def _wallet_key() -> str:
    return str(app_settings._load_secrets().get("pm_wallet_key", ""))


def _clob_client_available() -> bool:
    try:
        import py_clob_client  # noqa: F401
        return True
    except Exception:                                # noqa: BLE001
        return False


def status() -> dict:
    """Read-only readiness report for the dashboard. Never touches the chain."""
    enabled = bool(app_settings.get("pm_live_enabled"))
    has_key = bool(_wallet_key())
    has_dep = _clob_client_available()
    ready = enabled and has_key and has_dep
    blockers = []
    if not enabled:
        blockers.append("pm_live_enabled is off (paper mode)")
    if not has_key:
        blockers.append("no pm_wallet_key in .secrets.json")
    if not has_dep:
        blockers.append("py-clob-client not installed")
    return {"mode": "live" if ready else "paper",
            "live_enabled": enabled, "wallet_configured": has_key,
            "clob_client_installed": has_dep, "ready": ready,
            "blockers": blockers}


class PolymarketLiveExecutor:
    """Order-placement seam. Inert until fully configured."""

    def enabled(self) -> bool:
        return status()["ready"]

    def place_order(self, token_id: str, side: str, price: float,
                    size_usdc: float):
        st = status()
        if not st["ready"]:
            raise RuntimeError(
                "Polymarket live execution is disabled: "
                + "; ".join(st["blockers"])
                + ". Running paper only.")
        # --- real path, intentionally not auto-enabled ---
        # Build & sign an EIP-712 CLOB order with py_clob_client and POST it.
        # Left as an explicit, reviewed follow-up so no untested code can spend
        # real funds. See docs.polymarket.com/#clob-api.
        raise NotImplementedError(
            "Live order signing is wired but not yet reviewed for real capital. "
            "Complete the py-clob-client order build in execution.py before use.")


executor = PolymarketLiveExecutor()
