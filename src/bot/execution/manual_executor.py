"""Helpers for generating manual order sheets from offline signals."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.execution.interface import CandidateExecutor, ExecutionBatch, ExecutionOrder
from bot.risk.portfolio_rules import RiskAssessedCandidate
from bot.strategy.breakout_momentum import build_breakout_rationale


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
EXECUTION_SHEET_COLUMNS = (
    "date",
    "generated_at_utc",
    "symbol",
    "action",
    "quantity",
    "intended_order_type",
    "entry_price_hint",
    "stop_level",
    "time_in_force",
    "strategy_name",
    "rationale",
    "risk_budget",
    "per_share_risk",
    "notional_value",
    "metadata_json",
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


class ManualExecutor(CandidateExecutor):
    """Manual executor that renders approved candidates into a daily order sheet."""

    executor_name = "manual"

    def __init__(
        self,
        *,
        intended_order_type: str = "MARKET",
        time_in_force: str = "DAY",
    ) -> None:
        normalized_order_type = intended_order_type.strip().upper()
        if normalized_order_type not in VALID_ORDER_TYPES:
            expected = ", ".join(sorted(VALID_ORDER_TYPES))
            raise ManualOrderError(
                f"Unsupported intended_order_type '{intended_order_type}'. Expected one of: {expected}."
            )
        self.intended_order_type = normalized_order_type
        self.time_in_force = time_in_force.strip().upper() or "DAY"

    def build_execution_batch(
        self,
        candidates: Sequence[RiskAssessedCandidate],
        *,
        as_of_date: date,
    ) -> ExecutionBatch:
        """Render approved risk candidates into a manual execution batch."""

        generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        orders = tuple(
            _candidate_to_execution_order(
                candidate,
                as_of_date=as_of_date,
                generated_at_utc=generated_at_utc,
                intended_order_type=self.intended_order_type,
                time_in_force=self.time_in_force,
            )
            for candidate in candidates
            if candidate.approved and candidate.sizing.shares > 0
        )
        return ExecutionBatch(
            executor_name=self.executor_name,
            as_of_date=as_of_date,
            generated_at_utc=generated_at_utc,
            orders=orders,
        )


def write_execution_batch(
    batch: ExecutionBatch,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write an execution batch to CSV or JSON."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(resolved_output_path, output_format)

    if resolved_format == "json":
        resolved_output_path.write_text(
            json.dumps(batch.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return resolved_output_path

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXECUTION_SHEET_COLUMNS)
        writer.writeheader()
        for order in batch.orders:
            writer.writerow(
                _execution_order_record(
                    order,
                    generated_at_utc=batch.generated_at_utc,
                )
            )
    return resolved_output_path


def _candidate_to_execution_order(
    candidate: RiskAssessedCandidate,
    *,
    as_of_date: date,
    generated_at_utc: str,
    intended_order_type: str,
    time_in_force: str,
) -> ExecutionOrder:
    return ExecutionOrder(
        date=as_of_date,
        symbol=candidate.signal.symbol,
        action=candidate.signal.side,
        quantity=candidate.sizing.shares,
        intended_order_type=intended_order_type,
        entry_price_hint=candidate.entry_price,
        stop_level=candidate.stop_price,
        rationale=_build_rationale(candidate),
        time_in_force=time_in_force,
        strategy_name=candidate.signal.strategy_name,
        metadata={
            "signal_date": candidate.signal.date.isoformat(),
            "generated_at_utc": generated_at_utc,
            "adjusted_risk_per_trade": candidate.adjusted_risk_per_trade,
            "risk_budget": candidate.sizing.risk_budget,
            "per_share_risk": candidate.sizing.per_share_risk,
            "notional_value": candidate.sizing.notional_value,
            "capped_by_notional": candidate.sizing.capped_by_notional,
            "signal_metadata": dict(candidate.signal.metadata),
            "existing_position": (
                candidate.existing_position.to_dict()
                if candidate.existing_position is not None
                else None
            ),
        },
    )


def _build_rationale(candidate: RiskAssessedCandidate) -> str:
    return build_breakout_rationale(
        candidate.signal.entry_reason,
        candidate.signal.metadata,
    )


def _execution_order_record(
    order: ExecutionOrder,
    *,
    generated_at_utc: str,
) -> dict[str, str]:
    metadata = dict(order.metadata)
    risk_budget = metadata.get("risk_budget")
    per_share_risk = metadata.get("per_share_risk")
    notional_value = metadata.get("notional_value")

    return {
        "date": order.date.isoformat(),
        "generated_at_utc": generated_at_utc,
        "symbol": order.symbol,
        "action": order.action,
        "quantity": str(order.quantity),
        "intended_order_type": order.intended_order_type,
        "entry_price_hint": _format_optional_float(order.entry_price_hint),
        "stop_level": _format_optional_float(order.stop_level),
        "time_in_force": order.time_in_force,
        "strategy_name": order.strategy_name or "",
        "rationale": order.rationale,
        "risk_budget": _format_optional_float(_optional_float(risk_budget)),
        "per_share_risk": _format_optional_float(_optional_float(per_share_risk)),
        "notional_value": _format_optional_float(_optional_float(notional_value)),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_output_format(output_path: Path, output_format: str | None) -> str:
    if output_format is not None:
        normalized = output_format.strip().lower()
    else:
        normalized = output_path.suffix.lower().lstrip(".") or "csv"

    if normalized not in {"csv", "json"}:
        raise ManualOrderError("output_format must be either 'csv' or 'json'.")
    return normalized
