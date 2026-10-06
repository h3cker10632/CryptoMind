"""Paper-trading Execution Engine (OMS) — realistic fills with fees and
slippage. Supports LONG and SHORT positions.

Shorts are margin-style (perp-like) and now model the two frictions the
critique flagged as missing:
  * FUNDING accrual — the live OKX funding rate is charged/credited on the
    short's notional each management tick (crowded shorts pay to stay short).
  * LIQUIDATION — if an adverse move consumes the reserved margin (minus a
    maintenance buffer), the position is force-closed, exactly as a venue
    would liquidate it. This stops shorts from being unrealistically riskless.
"""
import math
import time
import uuid
from ..config import START_CASH
from ..tunables import tv
from .. import db
from .. import money


class Position(dict):
    pass


def _valid_price(px):
    """A fill/mark price is usable only if it is a finite, strictly positive
    number. A 0 / NaN / negative print (a broken ticker tick) must NEVER mark,
    stop out or fill a position — that is exactly how PUMP-USD was "sold" at
    $0.00 for a -$13.7k paper loss."""
    try:
        return px is not None and math.isfinite(px) and px > 0
    except TypeError:
        return False


class PaperBroker:
    def __init__(self):
        self._portfolio = None       # set via bind_portfolio(); None = local cash
        self._local_cash = START_CASH
        self.positions = {}      # product -> Position
        self.closed_trades = []
        self.realized_pnl = 0.0

    def bind_portfolio(self, portfolio):
        """Route this broker's cash through the shared paper-capital ledger
        instead of a private balance. Call once, after restoring any legacy
        cash into the ledger's opening balance, before background loops start.
        Standalone brokers (unit tests) stay on local cash by never calling
        this."""
        self._portfolio = portfolio

    @property
    def cash(self):
        if self._portfolio is not None:
            return self._portfolio.cash
        return self._local_cash

    @cash.setter
    def cash(self, value):
        if self._portfolio is not None:
            raise RuntimeError(
                "cash is read-only once bound to the shared paper portfolio; "
                "use reserve/settle events (see app.portfolio) instead")
        self._local_cash = float(value)

    def _reserve(self, event_id, amount, reference):
        """Debit `amount` once. Routes through the shared ledger when bound;
        otherwise mutates the private balance exactly like before."""
        if self._portfolio is not None:
            return self._portfolio.reserve("crypto", event_id, amount, reference)
        if amount > self._local_cash + 1e-9:
            return False
        self._local_cash -= amount
        return True

    def _settle(self, event_id, delta, reference):
        """Apply one signed settlement delta once (close proceeds, funding)."""
        if self._portfolio is not None:
            return self._portfolio.apply("crypto", event_id, delta, reference)
        self._local_cash += delta
        return True

    # ---------- accounting ----------
    def position_value(self, pos, px):
        if pos.get("side", 1) > 0:
            return pos["qty"] * px
        # short: reserved margin + unrealized pnl
        return pos["margin"] + pos["qty"] * (pos["entry"] - px)

    def equity(self, market):
        eq = self.cash
        for p, pos in self.positions.items():
            px = market.price(p) or pos["entry"]
            eq += self.position_value(pos, px)
        return eq

    def exposure(self, market):
        return sum(pos["qty"] * (market.price(p) or pos["entry"])
                   for p, pos in self.positions.items())

    # ---------- order handling ----------
    def _fill_price(self, price, side, product=None):
        slip = price * tv("slippage_bps") / 1e4
        px = price + slip if side == "buy" else price - slip
        # normalise to the venue tick size (Decimal) so paper matches live
        return money.round_price(product, px) if product else px

    def open(self, product, direction, notional, price, stop, take, reason,
             votes=None, regime_at_entry=None, is_hedge=False, atr_at_entry=None,
             mtf_at_entry=None, maker=False):
        """direction: +1 long, -1 short (margin-style).

        `votes` (per-strategy vote dict at entry) and `regime_at_entry` are
        stamped INTO the Position at creation — before it is stored or returned —
        so the trade-PnL attribution can always credit the right strategies even
        if the position is closed on the same tick or the process restarts
        mid-tick. Setting them after open() returned (the old path) risked a
        snapshot/close dropping them, which is why `trade_attributions` stayed 0.

        `is_hedge` marks a market-neutral pair-hedge leg. Hedge legs are managed
        as a PAIR by the hedger (both live or neither) and MUST NOT be closed by
        the directional signal-flip exit, so the flag is baked in here too.

        `atr_at_entry` is stamped so the conformal stop calibrator can normalise
        the trade's realized adverse excursion into ATR units at close.

        `maker=True` is a filled LIMIT order: it fills exactly at `price` (the
        limit) with no slippage and pays the maker fee.
        """
        if not _valid_price(price):
            return None
        if maker:
            fill = money.round_price(product, price) if product else price
            fee = notional * tv("maker_fee_rate")
        else:
            fill = self._fill_price(price, "buy" if direction > 0 else "sell", product)
            fee = notional * tv("fee_rate")
        if notional + fee > self.cash:
            return None
        # enforce exchange lot/step + min-notional like a real venue would
        qty = money.round_qty(product, notional / fill)
        if qty <= 0 or not money.meets_min_notional(product, qty, fill):
            return None
        ledger_id = uuid.uuid4().hex
        if not self._reserve(f"crypto-open-{ledger_id}", notional + fee,
                             f"open {product}"):
            return None          # lost a race to another concurrent order
        pos = Position(
            product=product, side=direction, qty=qty, entry=fill,
            stop=stop, take=take, water=fill, opened=time.time(),
            reason=reason, fees=fee)
        pos["_ledger_id"] = ledger_id
        if maker:
            pos["maker_entry"] = True
        # MAE tracking: worst adverse price seen since entry, seeded at entry.
        # `mae_price` is the extreme AGAINST the position (min for long, max for
        # short); the stop calibrator reads it at close.
        pos["mae_price"] = fill
        if atr_at_entry:
            pos["atr_at_entry"] = atr_at_entry
        # multi-timeframe alignment at entry, for the conformal direction gate's
        # counterfactual labelling at close.
        if mtf_at_entry is not None:
            pos["mtf_at_entry"] = mtf_at_entry
        # bake the learning-attribution fields in atomically at creation
        pos["votes"] = {k: round(v, 3) for k, v in (votes or {}).items()
                        if abs(v) > 0.05}
        if regime_at_entry is not None:
            pos["regime_at_entry"] = regime_at_entry
        if is_hedge:
            pos["hedge"] = True
        # tag meme positions so PnL attribution / dashboards can separate the
        # meme sleeve's performance from the core book's.
        try:
            from ..data.memes import memes
            if memes.is_meme(product):
                pos["meme"] = True
        except Exception:
            pass
        if direction < 0:
            pos["margin"] = notional
        self.positions[product] = pos
        db.log_trade(product, "buy" if direction > 0 else "short",
                     qty, fill, fee, reason)
        db.log_event("trade", f"OPEN {'LONG' if direction > 0 else 'SHORT'} "
                              f"{product} qty={qty:.6f} @ {fill:.2f} "
                              f"stop={stop:.2f} tp={take:.2f} ({reason})")
        return pos

    def buy(self, product, notional, price, stop, take, reason):
        return self.open(product, 1, notional, price, stop, take, reason)

    def sell(self, product, price, reason, maker=False):
        """Close a position (long OR short). Name kept for compatibility.
        `maker=True`: a resting limit (e.g. take-profit) filled exactly at
        `price` with the maker fee and no slippage."""
        if not _valid_price(price):
            # refuse to close on a junk print; the position stays open and is
            # re-evaluated next tick against a sane price
            db.log_event("error", f"refused close of {product} at invalid "
                                  f"price {price!r} ({reason})")
            return None
        pos = self.positions.pop(product, None)
        if not pos:
            return None
        close_id = f"crypto-close-{pos.get('_ledger_id') or uuid.uuid4().hex}"
        side = pos.get("side", 1)
        fee_rate = tv("maker_fee_rate") if maker else tv("fee_rate")
        if side > 0:
            fill = (money.round_price(product, price) if maker and product else
                    price if maker else self._fill_price(price, "sell", product))
            gross = pos["qty"] * fill
            fee = gross * fee_rate
            self._settle(close_id, gross - fee, f"close {product}")
            pnl = gross - fee - pos["qty"] * pos["entry"] - pos["fees"]
        else:
            fill = (money.round_price(product, price) if maker and product else
                    price if maker else self._fill_price(price, "buy", product))
            fee = pos["qty"] * fill * fee_rate
            move = pos["qty"] * (pos["entry"] - fill)  # short gains on drop
            self._settle(close_id, pos["margin"] + move - fee, f"close {product}")
            pnl = move - fee - pos["fees"]
        self.realized_pnl += pnl
        trade = {**pos, "exit": fill, "closed": time.time(),
                 "pnl": pnl, "exit_reason": reason}
        self.closed_trades.append(trade)
        db.log_trade(product, "sell" if side > 0 else "cover",
                     pos["qty"], fill, fee, reason, pnl)
        db.log_event("trade", f"CLOSE {'LONG' if side > 0 else 'SHORT'} "
                              f"{product} @ {fill:.2f} pnl={pnl:+.2f} ({reason})")
        # rich per-trade notification (Telegram/webhook) with full details
        try:
            from ..alerts import notify_trade_close
            notify_trade_close(trade)
        except Exception:
            pass
        return trade

    # ---------- stops / targets / trailing ----------
    def manage(self, market, trail_atr_mult, atr_lookup):
        for p in list(self.positions.keys()):
            px = market.price(p)
            if not _valid_price(px):
                continue
            pos = self.positions[p]
            side = pos.get("side", 1)
            # migrate pre-shorts positions ("high_water" -> "water")
            water = pos.get("water", pos.get("high_water", pos["entry"]))
            atr = atr_lookup(p)
            # MAE: track the worst ADVERSE excursion since entry (min px for a
            # long, max px for a short) so the stop calibrator can measure how
            # far this trade actually dipped against us, in ATR units, at close.
            mae = pos.get("mae_price", pos["entry"])
            pos["mae_price"] = min(mae, px) if side > 0 else max(mae, px)
            if side > 0:
                pos["water"] = max(water, px)
                if atr:
                    trail = pos["water"] - trail_atr_mult * atr
                    if trail > pos["stop"]:
                        pos["stop"] = trail
                if px <= pos["stop"]:
                    self.sell(p, px, "stop-loss/trail")
                elif px >= pos["take"]:
                    self._take_profit(p, pos, px)
                elif self._giveback_hit(pos, px):
                    self.sell(p, px, "peak-giveback")
            else:
                pos["water"] = min(water, px)          # low-water for shorts
                # ---- funding accrual (crowded shorts pay to stay short) ----
                self._accrue_funding(p, pos, px)
                # ---- liquidation guard: adverse move eating the margin ----
                margin = pos.get("margin", pos["qty"] * pos["entry"])
                loss = pos["qty"] * (px - pos["entry"])       # >0 = losing short
                maint = margin * 0.10                          # 10% maintenance
                if loss >= margin - maint:
                    self.sell(p, px, "LIQUIDATION (margin exhausted)")
                    continue
                if atr:
                    trail = pos["water"] + trail_atr_mult * atr
                    if trail < pos["stop"]:
                        pos["stop"] = trail
                if px >= pos["stop"]:
                    self.sell(p, px, "stop-loss/trail")
                elif px <= pos["take"]:
                    self._take_profit(p, pos, px)
                elif self._giveback_hit(pos, px):
                    self.sell(p, px, "peak-giveback")

    def _take_profit(self, p, pos, px):
        """A position entered by limit order keeps its take-profit as a RESTING
        limit: it fills exactly at the target with the maker fee. Market-entered
        positions take profit at market (taker)."""
        if pos.get("maker_entry") and _valid_price(pos.get("take")):
            return self.sell(p, pos["take"], "take-profit", maker=True)
        return self.sell(p, px, "take-profit")

    def _giveback_hit(self, pos, px):
        """Peak-giveback trailing exit (NOFX idea): lock in a WINNER that hands
        back too much of its best unrealized gain. Computed on a PRICE basis —
        never a leverage/margin-multiplied basis (NOFX's live bug silently
        halved the trigger distance) — so a short and a long behave identically.

        Arms only after the position's peak gain has exceeded `trail_giveback_arm_pct`
        of entry price, so noise around break-even can't trip it. Disabled when
        `trail_giveback_pct` == 0. Hedge legs are exempt (managed as a pair).
        """
        if pos.get("hedge"):
            return False
        gb = tv("trail_giveback_pct")
        if gb <= 0:
            return False
        # min-hold: give a fresh position room to work before giveback can fire
        if time.time() - pos.get("opened", 0) < tv("min_hold_sec"):
            return False
        entry = pos["entry"]
        side = pos.get("side", 1)
        water = pos.get("water", entry)
        # peak & current gain per unit, price-basis, positive = in profit
        peak_gain = (water - entry) if side > 0 else (entry - water)
        cur_gain = (px - entry) * side
        if peak_gain <= 0:
            return False
        if peak_gain / entry < tv("trail_giveback_arm_pct"):
            return False                       # not a big enough move yet
        if cur_gain <= 0:
            return False                       # never give-back exit into a loss
        return (peak_gain - cur_gain) >= gb * peak_gain

    def _accrue_funding(self, product, pos, px):
        """Charge/credit perp funding on a short's notional (8h rate, prorated
        to the ~20s tick). Positive funding = shorts receive, longs pay."""
        try:
            from ..data.derivatives import derivatives
            d = derivatives.features(product)
        except Exception:
            d = None
        if not d:
            return
        rate_8h = d.get("funding_rate", 0.0)
        now = time.time()
        last = pos.get("_last_funding", pos.get("opened", now))
        elapsed = now - last
        if elapsed <= 0:
            return
        notional = pos["qty"] * px
        # short RECEIVES funding when rate>0; sign flips the cash effect
        pos["_last_funding"] = now
        ledger_id = pos.get("_ledger_id") or product
        event_id = f"crypto-funding-{ledger_id}-{int(now)}"
        self._settle(event_id, notional * rate_8h * (elapsed / (8 * 3600)),
                    f"funding {product}")

    def stats(self):
        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        n = len(self.closed_trades)
        return {
            "trades": n,
            "win_rate": round(len(wins) / n, 3) if n else None,
            "realized_pnl": round(self.realized_pnl, 2),
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl"] for t in self.closed_trades if t["pnl"] <= 0)
                              / max(1, n - len(wins)), 2),
        }


broker = PaperBroker()
