"""Pending-order state, reservation, and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.data.portfolio_heat import PortfolioHoldingExposure
from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    load_json_mapping_file,
    write_json_file,
)
from bot.execution.interface import ExecutionOrder
from bot.logging_utils import get_logger
from bot.risk.portfolio_rules import ExistingPosition


LOGGER = get_logger(__name__)

PENDING_ORDER_STATE_SCHEMA_VERSION = 1
PENDING_ORDER_ACTIVE_STATUSES = ("pending", "partially_filled")
PENDING_ORDER_FINAL_STATUSES = (
    "filled",
    "cancelled",
    "expired",
    "replaced",
)
PENDING_ORDER_STATUSES = PENDING_ORDER_ACTIVE_STATUSES + PENDING_ORDER_FINAL_STATUSES
PENDING_ORDER_SIDES = ("BUY", "SELL")
PENDING_ORDER_TYPES = ("MARKET", "LIMIT", "STOP_LIMIT")
ORDER_LIFECYCLE_EVENT_TYPES = (
    "created",
    "partially_filled",
    "filled",
    "cancelled",
    "expired",
    "replaced",
)
_PENDING_ORDER_STATE_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "pending_orders.json",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "pending_orders.json"),
    ),
)


@dataclass(frozen=True)
class PendingOrderRecord:
    """One persisted pending-order snapshot."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    created_at: str
    status: str
    requested_quantity: int
    filled_quantity: int = 0
    requested_price: float | None = None
    entry_hint: float | None = None
    stop_price: float | None = None
    reserved_notional: float = 0.0
    reserved_slot: bool = False
    preset_name: str | None = None
    trade_id: str | None = None
    linked_decision_id: str | None = None
    sector_name: str | None = None
    industry_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.side not in PENDING_ORDER_SIDES:
            raise ValueError(
                f"side must be one of {PENDING_ORDER_SIDES}, got '{self.side}'."
            )
        if self.order_type not in PENDING_ORDER_TYPES:
            raise ValueError(
                f"order_type must be one of {PENDING_ORDER_TYPES}, got '{self.order_type}'."
            )
        if self.status not in PENDING_ORDER_STATUSES:
            raise ValueError(
                f"status must be one of {PENDING_ORDER_STATUSES}, got '{self.status}'."
            )
        coerce_iso8601_datetime_text(self.created_at, "created_at")
        if self.requested_quantity <= 0:
            raise ValueError("requested_quantity must be greater than zero.")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("filled_quantity cannot exceed requested_quantity.")
        if self.requested_price is not None and self.requested_price <= 0:
            raise ValueError("requested_price must be positive when provided.")
        if self.entry_hint is not None and self.entry_hint <= 0:
            raise ValueError("entry_hint must be positive when provided.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided.")
        if self.reserved_notional < 0:
            raise ValueError("reserved_notional cannot be negative.")
        if self.status in PENDING_ORDER_ACTIVE_STATUSES and self.remaining_quantity <= 0:
            raise ValueError(
                "Active pending orders must have a remaining unfilled quantity."
            )
        if self.status in PENDING_ORDER_FINAL_STATUSES:
            if self.reserved_notional != 0:
                raise ValueError("Finalized orders cannot retain reserved_notional.")
            if self.reserved_slot:
                raise ValueError("Finalized orders cannot retain reserved_slot.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())

    @property
    def remaining_quantity(self) -> int:
        return max(self.requested_quantity - self.filled_quantity, 0)

    @property
    def active(self) -> bool:
        return self.status in PENDING_ORDER_ACTIVE_STATUSES and self.remaining_quantity > 0

    @property
    def reserves_capacity(self) -> bool:
        return self.active and self.side == "BUY"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "created_at": self.created_at,
            "status": self.status,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "reserved_notional": self.reserved_notional,
            "reserved_slot": self.reserved_slot,
        }
        if self.requested_price is not None:
            payload["requested_price"] = self.requested_price
        if self.entry_hint is not None:
            payload["entry_hint"] = self.entry_hint
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.preset_name is not None:
            payload["preset_name"] = self.preset_name
        if self.trade_id is not None:
            payload["trade_id"] = self.trade_id
        if self.linked_decision_id is not None:
            payload["linked_decision_id"] = self.linked_decision_id
        if self.sector_name is not None:
            payload["sector_name"] = self.sector_name
        if self.industry_name is not None:
            payload["industry_name"] = self.industry_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(
        cls,
        order_id: str,
        data: Mapping[str, Any],
    ) -> "PendingOrderRecord":
        raw_metadata = data.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided.")
        return cls(
            order_id=order_id.strip(),
            symbol=_mapping_text(data, "symbol").upper(),
            side=_mapping_text(data, "side").upper(),
            order_type=_mapping_text(data, "order_type").upper(),
            created_at=coerce_iso8601_datetime_text(data.get("created_at"), "created_at"),
            status=_mapping_text(data, "status"),
            requested_quantity=_mapping_positive_int(data, "requested_quantity"),
            filled_quantity=_mapping_non_negative_int(data, "filled_quantity"),
            requested_price=_mapping_optional_positive_float(data, "requested_price"),
            entry_hint=_mapping_optional_positive_float(data, "entry_hint"),
            stop_price=_mapping_optional_positive_float(data, "stop_price"),
            reserved_notional=_mapping_non_negative_float(data, "reserved_notional"),
            reserved_slot=_mapping_bool(data, "reserved_slot"),
            preset_name=_mapping_optional_text(data, "preset_name"),
            trade_id=_mapping_optional_text(data, "trade_id"),
            linked_decision_id=_mapping_optional_text(data, "linked_decision_id"),
            sector_name=_mapping_optional_text(data, "sector_name"),
            industry_name=_mapping_optional_text(data, "industry_name"),
            metadata=dict(raw_metadata),
        )


@dataclass(frozen=True)
class OrderLifecycleEvent:
    """One typed order lifecycle transition."""

    event_type: str
    occurred_at: str
    order_id: str
    symbol: str
    status: str
    filled_quantity: int
    remaining_quantity: int
    reserved_notional: float
    reserved_slot: bool
    trade_id: str | None = None
    linked_decision_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in ORDER_LIFECYCLE_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {ORDER_LIFECYCLE_EVENT_TYPES}, got '{self.event_type}'."
            )
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.status not in PENDING_ORDER_STATUSES:
            raise ValueError(
                f"status must be one of {PENDING_ORDER_STATUSES}, got '{self.status}'."
            )
        coerce_iso8601_datetime_text(self.occurred_at, "occurred_at")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        if self.remaining_quantity < 0:
            raise ValueError("remaining_quantity cannot be negative.")
        if self.reserved_notional < 0:
            raise ValueError("reserved_notional cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "reserved_notional": self.reserved_notional,
            "reserved_slot": self.reserved_slot,
        }
        if self.trade_id is not None:
            payload["trade_id"] = self.trade_id
        if self.linked_decision_id is not None:
            payload["linked_decision_id"] = self.linked_decision_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OrderReservationState:
    """Compact reservation/capacity view used by execution workflows."""

    active_order_count: int
    reserved_notional: float
    pending_fill_notional: float
    reserved_slot_count: int
    current_position_notional: float
    available_capital: float
    available_slots: int
    pending_symbols: tuple[str, ...] = ()
    pending_order_count_by_sector: dict[str, int] = field(default_factory=dict)
    reserved_notional_by_sector: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.active_order_count < 0:
            raise ValueError("active_order_count cannot be negative.")
        if self.reserved_slot_count < 0:
            raise ValueError("reserved_slot_count cannot be negative.")
        if self.current_position_notional < 0:
            raise ValueError("current_position_notional cannot be negative.")
        if self.reserved_notional < 0:
            raise ValueError("reserved_notional cannot be negative.")
        if self.pending_fill_notional < 0:
            raise ValueError("pending_fill_notional cannot be negative.")
        if self.available_capital < 0:
            raise ValueError("available_capital cannot be negative.")
        if self.available_slots < 0:
            raise ValueError("available_slots cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_order_count": self.active_order_count,
            "reserved_notional": self.reserved_notional,
            "pending_fill_notional": self.pending_fill_notional,
            "reserved_slot_count": self.reserved_slot_count,
            "current_position_notional": self.current_position_notional,
            "available_capital": self.available_capital,
            "available_slots": self.available_slots,
            "pending_symbols": list(self.pending_symbols),
            "pending_order_count_by_sector": dict(self.pending_order_count_by_sector),
            "reserved_notional_by_sector": dict(self.reserved_notional_by_sector),
        }


@dataclass(frozen=True)
class PendingOrderState:
    """Persisted snapshot of current pending-order state."""

    orders: dict[str, PendingOrderRecord]
    updated_at: str | None = None
    schema_version: int = PENDING_ORDER_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.updated_at is not None:
            coerce_iso8601_datetime_text(self.updated_at, "updated_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "orders": {
                order_id: record.to_dict()
                for order_id, record in sorted(self.orders.items())
            },
        }

    def active_orders(self) -> tuple[PendingOrderRecord, ...]:
        active = [record for record in self.orders.values() if record.active]
        return tuple(
            sorted(active, key=lambda record: (record.created_at, record.order_id))
        )


@dataclass(frozen=True)
class PendingOrderUpdateResult:
    """Structured result from one lifecycle update."""

    state: PendingOrderState
    orders: tuple[PendingOrderRecord, ...]
    events: tuple[OrderLifecycleEvent, ...] = ()
    position_update: ExistingPosition | None = None


@dataclass(frozen=True)
class PendingOrderStateStore:
    """Project-scoped loader/saver for pending-order state."""

    project_root: Path

    @property
    def path(self) -> Path:
        return default_pending_order_state_path(self.project_root)

    def load(self) -> PendingOrderState:
        return load_pending_order_state(self.path)

    def save(self, state: PendingOrderState) -> Path:
        return write_pending_order_state(state, self.path)

    def update(self, record: PendingOrderRecord) -> Path:
        return self.save(upsert_pending_order_state(self.load(), record))


def default_pending_order_state_path(project_root: Path) -> Path:
    """Return the preferred repo-relative state path for pending orders."""

    return _PENDING_ORDER_STATE_PATH.resolve(project_root)


def load_pending_order_state(state_path: Path) -> PendingOrderState:
    """Load pending-order state, degrading gracefully on missing/corrupt files."""

    return load_json_mapping_file(
        state_path,
        artifact_label="pending order state",
        logger=LOGGER,
        empty_value=lambda: PendingOrderState(orders={}),
        build=lambda payload, _resolved_path: _build_pending_order_state(payload),
    )


def write_pending_order_state(
    state: PendingOrderState,
    state_path: Path,
) -> Path:
    """Persist pending-order state atomically."""

    return write_json_file(state_path, state.to_dict())


def upsert_pending_order_state(
    state: PendingOrderState,
    record: PendingOrderRecord,
    *,
    updated_at: str | None = None,
) -> PendingOrderState:
    """Return a new state snapshot with one order inserted or replaced by order_id."""

    updated_orders = dict(state.orders)
    updated_orders[record.order_id] = record
    _validate_active_buy_order_uniqueness(updated_orders)
    return PendingOrderState(
        orders=updated_orders,
        updated_at=updated_at or record.created_at,
        schema_version=state.schema_version,
    )


def build_pending_order_record_from_execution_order(
    order: ExecutionOrder,
    *,
    order_id: str,
    created_at: str,
    trade_id: str | None = None,
    linked_decision_id: str | None = None,
) -> PendingOrderRecord:
    """Build a pending-order record from an execution intent."""

    signal_metadata = order.metadata.get("signal_metadata")
    signal_metadata_mapping = (
        signal_metadata if isinstance(signal_metadata, Mapping) else {}
    )
    requested_price = _optional_positive_float(order.entry_price_hint)
    estimated_entry_hint = _execution_order_entry_hint(
        order,
        signal_metadata_mapping=signal_metadata_mapping,
    )
    resolved_entry_hint = requested_price or estimated_entry_hint
    reserved_notional = (
        _execution_order_reserved_notional(
            order,
            requested_price=requested_price,
            estimated_entry_hint=resolved_entry_hint,
        )
        if order.action == "BUY"
        else 0.0
    )
    return PendingOrderRecord(
        order_id=order_id,
        symbol=order.symbol,
        side=order.action,
        order_type=order.intended_order_type,
        created_at=created_at,
        status="pending",
        requested_quantity=order.quantity,
        filled_quantity=0,
        requested_price=requested_price,
        entry_hint=resolved_entry_hint,
        stop_price=order.stop_level,
        reserved_notional=reserved_notional,
        reserved_slot=order.action == "BUY",
        preset_name=_optional_text(signal_metadata_mapping.get("preset_name"))
        or order.strategy_name,
        trade_id=trade_id,
        linked_decision_id=linked_decision_id
        or _extract_linked_decision_id_from_metadata(order.metadata),
        sector_name=_optional_text(signal_metadata_mapping.get("sector_name")),
        industry_name=_optional_text(signal_metadata_mapping.get("industry_name")),
        metadata=dict(order.metadata),
    )


def create_pending_order(
    state: PendingOrderState,
    record: PendingOrderRecord,
) -> PendingOrderUpdateResult:
    """Create a new pending order idempotently when the payload already matches."""

    existing = state.orders.get(record.order_id)
    if existing is not None:
        if existing == record:
            return PendingOrderUpdateResult(state=state, orders=(existing,))
        raise ValueError(f"order_id '{record.order_id}' already exists.")
    updated_state = upsert_pending_order_state(state, record)
    return PendingOrderUpdateResult(
        state=updated_state,
        orders=(record,),
        events=(_lifecycle_event("created", record, occurred_at=record.created_at),),
    )


def mark_pending_order_partially_filled(
    state: PendingOrderState,
    order_id: str,
    *,
    occurred_at: str,
    filled_quantity: int,
) -> PendingOrderUpdateResult:
    """Advance an order into partial-fill state and reduce its reservation."""

    existing = _require_order(state, order_id)
    if not existing.active:
        raise ValueError("Only active pending orders can be partially filled.")
    if filled_quantity <= 0:
        raise ValueError("filled_quantity must be greater than zero.")
    if filled_quantity >= existing.requested_quantity:
        raise ValueError(
            "filled_quantity must be less than requested_quantity for a partial fill."
        )
    if filled_quantity < existing.filled_quantity:
        raise ValueError("filled_quantity cannot move backward.")
    updated = _with_fill_progress(
        existing,
        status="partially_filled",
        filled_quantity=filled_quantity,
    )
    if updated == existing:
        return PendingOrderUpdateResult(state=state, orders=(existing,))
    updated_state = upsert_pending_order_state(state, updated, updated_at=occurred_at)
    return PendingOrderUpdateResult(
        state=updated_state,
        orders=(updated,),
        events=(_lifecycle_event("partially_filled", updated, occurred_at=occurred_at),),
    )


def mark_pending_order_filled(
    state: PendingOrderState,
    order_id: str,
    *,
    occurred_at: str,
    fill_price: float | None = None,
) -> PendingOrderUpdateResult:
    """Mark a pending order as filled and return the linked position update when possible."""

    existing = _require_order(state, order_id)
    if existing.status == "filled":
        return PendingOrderUpdateResult(state=state, orders=(existing,))
    if not existing.active:
        raise ValueError("Only active pending orders can be marked filled.")
    updated = replace(
        existing,
        status="filled",
        filled_quantity=existing.requested_quantity,
        reserved_notional=0.0,
        reserved_slot=False,
    )
    if updated == existing:
        return PendingOrderUpdateResult(state=state, orders=(existing,))
    updated_state = upsert_pending_order_state(state, updated, updated_at=occurred_at)
    return PendingOrderUpdateResult(
        state=updated_state,
        orders=(updated,),
        events=(_lifecycle_event("filled", updated, occurred_at=occurred_at),),
        position_update=_position_update_from_filled_order(updated, fill_price=fill_price),
    )


def mark_pending_order_cancelled(
    state: PendingOrderState,
    order_id: str,
    *,
    occurred_at: str,
) -> PendingOrderUpdateResult:
    """Mark an order cancelled and release its reservation."""

    return _finalize_pending_order(
        state,
        order_id,
        occurred_at=occurred_at,
        status="cancelled",
        event_type="cancelled",
    )


def mark_pending_order_expired(
    state: PendingOrderState,
    order_id: str,
    *,
    occurred_at: str,
) -> PendingOrderUpdateResult:
    """Mark an order expired and release its reservation."""

    return _finalize_pending_order(
        state,
        order_id,
        occurred_at=occurred_at,
        status="expired",
        event_type="expired",
    )


def replace_pending_order(
    state: PendingOrderState,
    order_id: str,
    *,
    replacement: PendingOrderRecord,
    occurred_at: str,
) -> PendingOrderUpdateResult:
    """Mark one order replaced and create its replacement record."""

    existing = _require_order(state, order_id)
    if not existing.active:
        raise ValueError("Only active pending orders can be replaced.")
    if replacement.order_id == order_id:
        raise ValueError("replacement.order_id must differ from the existing order_id.")
    replaced_order = replace(
        existing,
        status="replaced",
        reserved_notional=0.0,
        reserved_slot=False,
    )
    updated_state = upsert_pending_order_state(state, replaced_order, updated_at=occurred_at)
    created_result = create_pending_order(updated_state, replacement)
    return PendingOrderUpdateResult(
        state=PendingOrderState(
            orders=created_result.state.orders,
            updated_at=occurred_at,
            schema_version=created_result.state.schema_version,
        ),
        orders=(replaced_order, replacement),
        events=(
            _lifecycle_event("replaced", replaced_order, occurred_at=occurred_at),
            _lifecycle_event("created", replacement, occurred_at=occurred_at),
        ),
    )


def build_order_reservation_state(
    pending_order_state: PendingOrderState,
    *,
    current_equity: float,
    current_positions: Sequence[ExistingPosition],
    max_concurrent_positions: int,
) -> OrderReservationState:
    """Return the reservation/capacity view after active pending buy orders."""

    current_position_notional = sum(
        float(position.shares) * float(position.average_entry_price)
        for position in current_positions
    )
    active_orders = [
        record
        for record in pending_order_state.active_orders()
        if record.reserves_capacity
    ]
    reserved_notional = sum(record.reserved_notional for record in active_orders)
    pending_fill_notional = sum(
        _pending_fill_notional(record, current_positions=current_positions)
        for record in active_orders
    )
    reserved_slot_count = sum(
        1
        for record in active_orders
        if _reserves_unmatched_slot(record, current_positions=current_positions)
    )
    pending_symbols = tuple(sorted({record.symbol for record in active_orders}))
    pending_order_count_by_sector: dict[str, int] = {}
    reserved_notional_by_sector: dict[str, float] = {}
    for record in active_orders:
        sector_name = record.sector_name.strip() if record.sector_name else "Unknown"
        pending_order_count_by_sector[sector_name] = (
            pending_order_count_by_sector.get(sector_name, 0) + 1
        )
        reserved_notional_by_sector[sector_name] = (
            reserved_notional_by_sector.get(sector_name, 0.0) + record.reserved_notional
        )

    available_capital = max(
        float(current_equity)
        - current_position_notional
        - reserved_notional
        - pending_fill_notional,
        0.0,
    )
    available_slots = max(
        int(max_concurrent_positions) - len(current_positions) - reserved_slot_count,
        0,
    )
    return OrderReservationState(
        active_order_count=len(active_orders),
        reserved_notional=reserved_notional,
        pending_fill_notional=pending_fill_notional,
        reserved_slot_count=reserved_slot_count,
        current_position_notional=current_position_notional,
        available_capital=available_capital,
        available_slots=available_slots,
        pending_symbols=pending_symbols,
        pending_order_count_by_sector=dict(sorted(pending_order_count_by_sector.items())),
        reserved_notional_by_sector=dict(sorted(reserved_notional_by_sector.items())),
    )


def build_pending_order_holding_exposures(
    pending_order_state: PendingOrderState,
    *,
    current_positions: Sequence[ExistingPosition] = (),
) -> tuple[PortfolioHoldingExposure, ...]:
    """Return pending-buy reservations as portfolio exposures for heat checks."""

    exposures = [
        PortfolioHoldingExposure(
            symbol=f"{record.symbol}:{record.order_id}",
            approximate_notional=(
                record.reserved_notional
                + _pending_fill_notional(record, current_positions=current_positions)
            ),
            sector_name=record.sector_name,
            industry_name=record.industry_name,
        )
        for record in pending_order_state.active_orders()
        if record.reserves_capacity
        and (
            record.reserved_notional
            + _pending_fill_notional(record, current_positions=current_positions)
        )
        > 0
    ]
    return tuple(sorted(exposures, key=lambda exposure: exposure.symbol))


def _build_pending_order_state(payload: Mapping[str, Any]) -> PendingOrderState:
    raw_orders = payload.get("orders", {})
    if raw_orders is None:
        raw_orders = {}
    if not isinstance(raw_orders, Mapping):
        raise ValueError("orders must be a mapping.")
    return PendingOrderState(
        orders={
            order_id.strip(): PendingOrderRecord.from_mapping(order_id, order_payload)
            for order_id, order_payload in sorted(raw_orders.items())
            if isinstance(order_id, str) and isinstance(order_payload, Mapping)
        },
        updated_at=_mapping_optional_text(payload, "updated_at"),
        schema_version=_mapping_optional_positive_int(payload, "schema_version")
        or PENDING_ORDER_STATE_SCHEMA_VERSION,
    )


def _finalize_pending_order(
    state: PendingOrderState,
    order_id: str,
    *,
    occurred_at: str,
    status: str,
    event_type: str,
) -> PendingOrderUpdateResult:
    existing = _require_order(state, order_id)
    if existing.status == status:
        return PendingOrderUpdateResult(state=state, orders=(existing,))
    if not existing.active:
        raise ValueError(f"Only active pending orders can be marked {status}.")
    updated = replace(
        existing,
        status=status,
        reserved_notional=0.0,
        reserved_slot=False,
    )
    if updated == existing:
        return PendingOrderUpdateResult(state=state, orders=(existing,))
    updated_state = upsert_pending_order_state(state, updated, updated_at=occurred_at)
    return PendingOrderUpdateResult(
        state=updated_state,
        orders=(updated,),
        events=(_lifecycle_event(event_type, updated, occurred_at=occurred_at),),
    )


def _with_fill_progress(
    record: PendingOrderRecord,
    *,
    status: str,
    filled_quantity: int,
) -> PendingOrderRecord:
    reservation_price = _reservation_price(record)
    remaining_quantity = max(record.requested_quantity - filled_quantity, 0)
    return replace(
        record,
        status=status,
        filled_quantity=filled_quantity,
        reserved_notional=remaining_quantity * reservation_price
        if record.side == "BUY"
        else 0.0,
        reserved_slot=record.side == "BUY" and remaining_quantity > 0,
    )


def _position_update_from_filled_order(
    record: PendingOrderRecord,
    *,
    fill_price: float | None,
) -> ExistingPosition | None:
    if record.side != "BUY" or record.filled_quantity <= 0:
        return None
    resolved_fill_price = fill_price or record.requested_price or record.entry_hint
    if resolved_fill_price is None or resolved_fill_price <= 0:
        return None
    metadata = dict(record.metadata)
    metadata["order_id"] = record.order_id
    if record.trade_id is not None:
        metadata["trade_id"] = record.trade_id
    if record.linked_decision_id is not None:
        metadata["linked_decision_id"] = record.linked_decision_id
    return ExistingPosition(
        symbol=record.symbol,
        shares=record.filled_quantity,
        average_entry_price=resolved_fill_price,
        current_stop=record.stop_price,
        preset_name=record.preset_name,
        source="pending_order_fill",
        metadata=metadata,
    )


def _reservation_price(record: PendingOrderRecord) -> float:
    if record.remaining_quantity > 0 and record.reserved_notional > 0:
        return record.reserved_notional / float(record.remaining_quantity)
    if record.requested_price is not None:
        return record.requested_price
    if record.entry_hint is not None:
        return record.entry_hint
    return 0.0


def _execution_order_reserved_notional(
    order: ExecutionOrder,
    *,
    requested_price: float | None,
    estimated_entry_hint: float | None,
) -> float:
    if requested_price is not None:
        return float(order.quantity) * requested_price
    metadata_notional = _optional_positive_float(order.metadata.get("notional_value"))
    if metadata_notional is not None:
        return metadata_notional
    if estimated_entry_hint is not None:
        return float(order.quantity) * estimated_entry_hint
    return 0.0


def _execution_order_entry_hint(
    order: ExecutionOrder,
    *,
    signal_metadata_mapping: Mapping[str, Any],
) -> float | None:
    if order.entry_price_hint is not None and order.entry_price_hint > 0:
        return float(order.entry_price_hint)
    metadata_notional = _optional_positive_float(order.metadata.get("notional_value"))
    if metadata_notional is not None and order.quantity > 0:
        return metadata_notional / float(order.quantity)
    for key in ("last_known_price", "last_price", "latest_close", "close"):
        value = _optional_positive_float(order.metadata.get(key))
        if value is not None:
            return value
    for key in ("last_known_price", "last_price", "latest_close", "close"):
        value = _optional_positive_float(signal_metadata_mapping.get(key))
        if value is not None:
            return value
    return None


def _pending_fill_notional(
    record: PendingOrderRecord,
    *,
    current_positions: Sequence[ExistingPosition],
) -> float:
    if record.side != "BUY" or record.filled_quantity <= 0:
        return 0.0
    reservation_price = _reservation_price(record)
    if reservation_price <= 0:
        return 0.0
    absorbed_quantity = min(
        _absorbed_filled_quantity(record, current_positions=current_positions),
        record.filled_quantity,
    )
    outstanding_quantity = max(record.filled_quantity - absorbed_quantity, 0)
    return outstanding_quantity * reservation_price


def _reserves_unmatched_slot(
    record: PendingOrderRecord,
    *,
    current_positions: Sequence[ExistingPosition],
) -> bool:
    if not record.reserved_slot:
        return False
    return _absorbed_filled_quantity(record, current_positions=current_positions) == 0


def _absorbed_filled_quantity(
    record: PendingOrderRecord,
    *,
    current_positions: Sequence[ExistingPosition],
) -> int:
    absorbed_quantity = 0
    for position in current_positions:
        if position.symbol.strip().upper() != record.symbol:
            continue
        if not _position_matches_pending_order(position, record):
            continue
        absorbed_quantity += int(round(float(position.shares)))
        if absorbed_quantity >= record.filled_quantity:
            return record.filled_quantity
    return absorbed_quantity


def _position_matches_pending_order(
    position: ExistingPosition,
    record: PendingOrderRecord,
) -> bool:
    metadata = position.metadata
    if not isinstance(metadata, Mapping):
        return False
    order_id = _optional_text(metadata.get("order_id"))
    if order_id == record.order_id:
        return True
    trade_id = _optional_text(metadata.get("trade_id"))
    if record.trade_id is not None and trade_id == record.trade_id:
        return True
    decision_id = _extract_linked_decision_id_from_metadata(metadata)
    return record.linked_decision_id is not None and decision_id == record.linked_decision_id


def _extract_linked_decision_id_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    for key in ("linked_decision_id", "decision_id"):
        value = _optional_text(metadata.get(key))
        if value is not None:
            return value
    signal_metadata = metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        value = _optional_text(signal_metadata.get("decision_id"))
        if value is not None:
            return value
    execution_metadata = metadata.get("execution_metadata")
    if isinstance(execution_metadata, Mapping):
        for key in ("linked_decision_id", "decision_id"):
            value = _optional_text(execution_metadata.get(key))
            if value is not None:
                return value
        nested_signal_metadata = execution_metadata.get("signal_metadata")
        if isinstance(nested_signal_metadata, Mapping):
            value = _optional_text(nested_signal_metadata.get("decision_id"))
            if value is not None:
                return value
    return None


def _validate_active_buy_order_uniqueness(
    orders: Mapping[str, PendingOrderRecord],
) -> None:
    active_symbols: dict[str, str] = {}
    for order_id, record in sorted(orders.items()):
        if not record.reserves_capacity:
            continue
        existing_order_id = active_symbols.get(record.symbol)
        if existing_order_id is not None and existing_order_id != order_id:
            raise ValueError(
                "Only one active capacity-reserving buy order is allowed per symbol."
            )
        active_symbols[record.symbol] = order_id


def _lifecycle_event(
    event_type: str,
    record: PendingOrderRecord,
    *,
    occurred_at: str,
) -> OrderLifecycleEvent:
    return OrderLifecycleEvent(
        event_type=event_type,
        occurred_at=occurred_at,
        order_id=record.order_id,
        symbol=record.symbol,
        status=record.status,
        filled_quantity=record.filled_quantity,
        remaining_quantity=record.remaining_quantity,
        reserved_notional=record.reserved_notional,
        reserved_slot=record.reserved_slot,
        trade_id=record.trade_id,
        linked_decision_id=record.linked_decision_id,
    )


def _require_order(state: PendingOrderState, order_id: str) -> PendingOrderRecord:
    order = state.orders.get(order_id)
    if order is None:
        raise KeyError(f"Unknown pending order_id '{order_id}'.")
    return order


def _mapping_text(data: Mapping[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _mapping_optional_text(data: Mapping[str, Any], field_name: str) -> str | None:
    value = data.get(field_name)
    return _optional_text(value)


def _mapping_bool(data: Mapping[str, Any], field_name: str) -> bool:
    value = data.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _mapping_positive_int(data: Mapping[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if coerced <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return coerced


def _mapping_optional_positive_int(data: Mapping[str, Any], field_name: str) -> int | None:
    value = data.get(field_name)
    if value is None:
        return None
    return _mapping_positive_int({field_name: value}, field_name)


def _mapping_non_negative_int(data: Mapping[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if coerced < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return coerced


def _mapping_non_negative_float(data: Mapping[str, Any], field_name: str) -> float:
    value = data.get(field_name)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number.")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if coerced < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return coerced


def _mapping_optional_positive_float(data: Mapping[str, Any], field_name: str) -> float | None:
    value = data.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number.")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if coerced <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return coerced


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if coerced <= 0:
        return None
    return coerced
