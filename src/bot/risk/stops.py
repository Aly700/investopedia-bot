"""Reusable stop-price helpers for ATR-based risk management."""

from __future__ import annotations

from math import isfinite
from typing import Literal


PositionSide = Literal["BUY", "SELL"]


def atr_stop_distance(atr_value: float, atr_multiple: float) -> float:
    """Return the ATR-based stop distance."""

    _validate_finite_positive(atr_value, name="atr_value")
    _validate_finite_positive(atr_multiple, name="atr_multiple")
    return atr_value * atr_multiple


def initial_stop_price(
    entry_price: float,
    atr_value: float,
    atr_multiple: float,
    *,
    side: PositionSide = "BUY",
) -> float:
    """Return an ATR-based initial stop price for a long or short position."""

    _validate_finite_positive(entry_price, name="entry_price")
    distance = atr_stop_distance(atr_value, atr_multiple)
    if side == "BUY":
        return entry_price - distance
    return entry_price + distance


def trailing_stop_reference(
    reference_price: float,
    atr_value: float,
    atr_multiple: float,
    *,
    side: PositionSide = "BUY",
) -> float:
    """Return the trailing stop implied by a reference price and ATR buffer."""

    _validate_finite_positive(reference_price, name="reference_price")
    distance = atr_stop_distance(atr_value, atr_multiple)
    if side == "BUY":
        return reference_price - distance
    return reference_price + distance


def stop_distance(entry_price: float, stop_price: float) -> float:
    """Return the absolute distance between entry and stop price."""

    _validate_finite(entry_price, name="entry_price")
    _validate_finite(stop_price, name="stop_price")
    return abs(entry_price - stop_price)


def _validate_finite(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_finite_positive(value: float, *, name: str) -> None:
    _validate_finite(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
