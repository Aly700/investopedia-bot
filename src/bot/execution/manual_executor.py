"""Helpers for generating manual order sheets from offline signals."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class ManualOrderError(ValueError):
    """Raised when manual order input is missing or malformed."""


VALID_ORDER_TYPES = {"MARKET", "LIMIT", "STOP_LIMIT"}
VALID_SIDES = {"BUY", "SELL"}
OUTPUT_COLUMNS = (
    "as_of_date",
    "generated_at_utc",
    "symbol",
    "side",
    "quantity",
    "order_type",
    "limit_price",
    "stop_price",
    "time_in_force",
    "strategy_name",
    "thesis",
)


@dataclass(frozen=True)
class ManualOrder:
    """A normalized order ticket that can be entered manually."""

    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"
    strategy_name: str | None = None
    thesis: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, row_number: int) -> "ManualOrder":
        """Create an order from an input row and validate required fields."""

        symbol = _required_text(data, "symbol", row_number).upper()
        side = _normalized_choice(data, "side", row_number, valid_values=VALID_SIDES)
        quantity = _positive_int(data, "quantity", row_number)
        order_type = _normalized_choice(
            data,
            "order_type",
            row_number,
            valid_values=VALID_ORDER_TYPES,
            default="MARKET",
        )
        limit_price = _optional_positive_float(data, "limit_price")
        stop_price = _optional_positive_float(data, "stop_price")
        time_in_force = _optional_text(data, "time_in_force") or "DAY"
        strategy_name = _optional_text(data, "strategy_name")
        thesis = _optional_text(data, "thesis")

        if order_type == "LIMIT" and limit_price is None:
            raise ManualOrderError(f"Row {row_number}: LIMIT orders require 'limit_price'.")
        if order_type == "STOP_LIMIT" and (limit_price is None or stop_price is None):
            raise ManualOrderError(
                f"Row {row_number}: STOP_LIMIT orders require both 'limit_price' and 'stop_price'."
            )

        return cls(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force.upper(),
            strategy_name=strategy_name,
            thesis=thesis,
        )

    def to_record(self, *, as_of_date: date, generated_at_utc: str) -> dict[str, str]:
        """Serialize the order into a CSV-friendly record."""

        return {
            "as_of_date": as_of_date.isoformat(),
            "generated_at_utc": generated_at_utc,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "order_type": self.order_type,
            "limit_price": _format_optional_float(self.limit_price),
            "stop_price": _format_optional_float(self.stop_price),
            "time_in_force": self.time_in_force,
            "strategy_name": self.strategy_name or "",
            "thesis": self.thesis or "",
        }


def load_orders_from_csv(input_path: Path) -> list[ManualOrder]:
    """Load and validate manual orders from a CSV file."""

    resolved_input_path = input_path.resolve()
    if not resolved_input_path.exists():
        raise ManualOrderError(f"Input file does not exist: {resolved_input_path}")

    with resolved_input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManualOrderError(f"Input CSV is missing a header row: {resolved_input_path}")

        orders: list[ManualOrder] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            orders.append(ManualOrder.from_mapping(row, row_number=row_number))

    if not orders:
        raise ManualOrderError(f"No orders found in input CSV: {resolved_input_path}")
    return orders


def write_manual_order_sheet(
    orders: Sequence[ManualOrder],
    output_path: Path,
    *,
    as_of_date: date,
) -> Path:
    """Write a normalized manual order sheet to disk."""

    if not orders:
        raise ManualOrderError("At least one order is required to write an order sheet.")

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for order in orders:
            writer.writerow(order.to_record(as_of_date=as_of_date, generated_at_utc=generated_at_utc))

    return resolved_output_path


def _required_text(data: Mapping[str, Any], key: str, row_number: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManualOrderError(f"Row {row_number}: '{key}' is required.")
    return value.strip()


def _optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalized_choice(
    data: Mapping[str, Any],
    key: str,
    row_number: int,
    *,
    valid_values: set[str],
    default: str | None = None,
) -> str:
    raw_value = data.get(key, default)
    if not isinstance(raw_value, str) or not raw_value.strip():
        if default is not None:
            return default
        raise ManualOrderError(f"Row {row_number}: '{key}' is required.")

    cleaned_value = raw_value.strip().upper()
    if cleaned_value not in valid_values:
        expected_values = ", ".join(sorted(valid_values))
        raise ManualOrderError(
            f"Row {row_number}: invalid '{key}' value '{raw_value}'. Expected one of: "
            f"{expected_values}."
        )
    return cleaned_value


def _positive_int(data: Mapping[str, Any], key: str, row_number: int) -> int:
    raw_value = data.get(key)
    if raw_value is None:
        raise ManualOrderError(f"Row {row_number}: '{key}' is required.")

    try:
        value = int(str(raw_value).strip())
    except ValueError as exc:
        raise ManualOrderError(f"Row {row_number}: '{key}' must be an integer.") from exc

    if value <= 0:
        raise ManualOrderError(f"Row {row_number}: '{key}' must be greater than zero.")
    return value


def _optional_positive_float(data: Mapping[str, Any], key: str) -> float | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None

    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ManualOrderError(f"'{key}' must be numeric.") from exc

    if value <= 0:
        raise ManualOrderError(f"'{key}' must be greater than zero when provided.")
    return value


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")
