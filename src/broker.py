"""Thin wrapper over the Alpaca trading API (paper account).

Exposes just what the daily runner needs: account equity, current positions as
weights, and the ability to move toward a set of target weights via market
orders. All order sizing is done in notional dollars so we don't have to fetch
live quotes ourselves.
"""
from __future__ import annotations

import logging

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .config import AlpacaCredentials

log = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, creds: AlpacaCredentials):
        self.client = TradingClient(
            creds.api_key, creds.secret_key, paper=creds.paper
        )
        self.paper = creds.paper

    # --- account state -----------------------------------------------------
    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def positions(self) -> pd.DataFrame:
        """Current positions with market value and unrealised P&L percentage."""
        rows = []
        for p in self.client.get_all_positions():
            rows.append(
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry": float(p.avg_entry_price),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["ticker", "qty", "market_value", "avg_entry", "unrealized_plpc"]
            ).set_index("ticker")
        return pd.DataFrame(rows).set_index("ticker")

    def current_weights(self) -> pd.Series:
        """Position market values as a fraction of total account equity."""
        pos = self.positions()
        eq = self.equity()
        if pos.empty or eq <= 0:
            return pd.Series(dtype=float)
        return (pos["market_value"] / eq).rename("weight")

    # --- order execution ---------------------------------------------------
    def submit_notional(self, ticker: str, side: OrderSide, notional: float) -> None:
        """Submit a fractional market order for a dollar amount."""
        if notional < 1:  # Alpaca rejects sub-$1 notional orders
            return
        order = MarketOrderRequest(
            symbol=ticker,
            notional=round(notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        self.client.submit_order(order)
        log.info("Submitted %s $%.2f of %s", side.value, notional, ticker)

    def close_position(self, ticker: str) -> None:
        try:
            self.client.close_position(ticker)
            log.info("Closed position %s", ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to close %s: %s", ticker, e)

    def rebalance_to(self, target_weights: pd.Series, band: float) -> None:
        """Issue orders to move current holdings toward `target_weights`.

        Sells/exits run before buys so freed-up cash is available for purchases.
        """
        eq = self.equity()
        current = self.current_weights()
        all_names = target_weights.index.union(current.index)

        deltas = {}
        for name in all_names:
            tgt = float(target_weights.get(name, 0.0))
            cur = float(current.get(name, 0.0))
            d = tgt - cur
            # Always allow full exits; otherwise respect the no-trade band.
            if abs(d) >= band or tgt == 0.0:
                deltas[name] = d

        # Exits and trims first.
        for name, d in sorted(deltas.items(), key=lambda kv: kv[1]):
            if d >= 0:
                continue
            if target_weights.get(name, 0.0) == 0.0:
                self.close_position(name)
            else:
                self.submit_notional(name, OrderSide.SELL, abs(d) * eq)

        # Then buys / adds.
        for name, d in deltas.items():
            if d > 0:
                self.submit_notional(name, OrderSide.BUY, d * eq)

    def apply_stop_losses(self, stop_loss: float) -> list[str]:
        """Close any position down more than `stop_loss` from its entry.

        Returns the list of tickers stopped out.
        """
        stopped = []
        pos = self.positions()
        for ticker, row in pos.iterrows():
            if row["unrealized_plpc"] <= -abs(stop_loss):
                self.close_position(ticker)
                stopped.append(ticker)
        return stopped
