"""Reusable fill-price and transaction-cost helpers for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from bot.config import GameRulesConfig


TradeSide = Literal["BUY", "SELL"]
OrderType = Literal["market", "stop"]


def apply_slippage(price: float, *, side: TradeSide, slippage_bps: int) -> float:
    """Return a slippage-adjusted fill price for the requested side."""

    _validate_finite_positive(price, name="price")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative.")

    slippage_fraction = slippage_bps / 10_000.0
    if side == "BUY":
        return price * (1.0 + slippage_fraction)
    if side == "SELL":
        return price * (1.0 - slippage_fraction)
    raise ValueError(f"Unsupported trade side: {side}")


@dataclass(frozen=True)
class TransactionCostModel:
    """Explicit commission and slippage assumptions used by the backtest engine."""

    market_order_commission: float
    stop_order_commission: float
    slippage_bps: int

    @classmethod
    def from_game_rules(cls, game_rules: GameRulesConfig) -> "TransactionCostModel":
        """Build a cost model from the shared repo config."""

        return cls(
            market_order_commission=game_rules.commissions.market_order,
            stop_order_commission=game_rules.commissions.stop_limit_order,
            slippage_bps=game_rules.execution.slippage_bps,
        )

    def __post_init__(self) -> None:
        _validate_finite_non_negative(
            self.market_order_commission,
            name="market_order_commission",
        )
        _validate_finite_non_negative(
            self.stop_order_commission,
            name="stop_order_commission",
        )
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative.")

    def commission(self, *, order_type: OrderType) -> float:
        """Return the configured commission for an order type."""

        if order_type == "market":
            return self.market_order_commission
        if order_type == "stop":
            return self.stop_order_commission
        raise ValueError(f"Unsupported order_type: {order_type}")

    def market_fill_price(self, reference_price: float, *, side: TradeSide) -> float:
        """Return a market fill price with adverse slippage applied."""

        return apply_slippage(
            reference_price,
            side=side,
            slippage_bps=self.slippage_bps,
        )

    def long_stop_fill_price(
        self,
        *,
        stop_price: float,
        bar_open: float,
        bar_low: float,
    ) -> float | None:
        """Return the simulated long stop fill price or ``None`` if not triggered."""

        _validate_finite_positive(stop_price, name="stop_price")
        _validate_finite_positive(bar_open, name="bar_open")
        _validate_finite_positive(bar_low, name="bar_low")

        if bar_open <= stop_price:
            raw_fill = bar_open
        elif bar_low <= stop_price:
            raw_fill = stop_price
        else:
            return None

        return apply_slippage(raw_fill, side="SELL", slippage_bps=self.slippage_bps)


def _validate_finite_non_negative(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_finite_positive(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
