"""Reconcile normalized broker truth into local pending-order state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from bot.broker.execution import (
    BrokerExecutionAdapter,
    BrokerExecutionError,
    BrokerOrderUpdate,
    broker_order_request_from_pending_order,
)
from bot.data.pending_orders import (
    OrderLifecycleEvent,
    PendingOrderRecord,
    PendingOrderState,
    PendingOrderUpdateResult,
    create_pending_order,
    mark_pending_order_cancelled,
    mark_pending_order_expired,
    mark_pending_order_filled,
    mark_pending_order_partially_filled,
    upsert_pending_order_state,
)
from bot.risk.portfolio_rules import ExistingPosition


@dataclass(frozen=True)
class BrokerReconciliationResult:
    """Structured result from reconciling broker truth into local state."""

    state: PendingOrderState
    matched_updates: tuple[BrokerOrderUpdate, ...] = ()
    unmatched_updates: tuple[BrokerOrderUpdate, ...] = ()
    warnings: tuple[str, ...] = ()
    lifecycle_events: tuple[OrderLifecycleEvent, ...] = ()
    position_updates: tuple[ExistingPosition, ...] = ()


@dataclass(frozen=True)
class BrokerOrderSubmissionResult:
    """Structured result from submitting staged pending orders."""

    state: PendingOrderState
    submitted_updates: tuple[BrokerOrderUpdate, ...] = ()
    skipped_order_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    lifecycle_events: tuple[OrderLifecycleEvent, ...] = ()
    position_updates: tuple[ExistingPosition, ...] = ()


def submit_pending_orders_via_broker(
    state: PendingOrderState,
    *,
    adapter: BrokerExecutionAdapter,
    order_ids: Sequence[str] | None = None,
) -> BrokerOrderSubmissionResult:
    """Submit staged local pending orders and reconcile the broker response."""

    requested_ids = {order_id.strip() for order_id in (order_ids or ()) if order_id.strip()}
    candidate_orders = [
        record
        for record in state.active_orders()
        if not requested_ids or record.order_id in requested_ids
    ]
    current_state = state
    submitted_updates: list[BrokerOrderUpdate] = []
    skipped_order_ids: list[str] = []
    warnings: list[str] = []
    lifecycle_events: list[OrderLifecycleEvent] = []
    position_updates: list[ExistingPosition] = []

    for record in candidate_orders:
        if record.broker_order_id is not None and record.broker_status not in {
            "cancelled",
            "expired",
            "replaced",
            "rejected",
        }:
            skipped_order_ids.append(record.order_id)
            continue
        try:
            update = adapter.place_order(
                broker_order_request_from_pending_order(
                    record,
                    client_order_id=record.order_id,
                )
            )
        except BrokerExecutionError as exc:
            warnings.append(
                "Broker submit failed for local pending order "
                f"'{record.order_id}': {exc}"
            )
            continue
        submitted_updates.append(update)
        reconciliation = reconcile_broker_order_updates(current_state, (update,))
        current_state = reconciliation.state
        warnings.extend(reconciliation.warnings)
        lifecycle_events.extend(reconciliation.lifecycle_events)
        position_updates.extend(reconciliation.position_updates)

    missing_requested_ids = sorted(requested_ids - {record.order_id for record in candidate_orders})
    warnings.extend(
        f"Pending order '{order_id}' was not found in active local state."
        for order_id in missing_requested_ids
    )
    return BrokerOrderSubmissionResult(
        state=current_state,
        submitted_updates=tuple(submitted_updates),
        skipped_order_ids=tuple(sorted(skipped_order_ids)),
        warnings=tuple(warnings),
        lifecycle_events=tuple(lifecycle_events),
        position_updates=tuple(position_updates),
    )


def reconcile_broker_order_updates(
    state: PendingOrderState,
    updates: Sequence[BrokerOrderUpdate],
) -> BrokerReconciliationResult:
    """Apply broker order updates to local pending-order state."""

    current_state = state
    matched_updates: list[BrokerOrderUpdate] = []
    unmatched_updates: list[BrokerOrderUpdate] = []
    warnings: list[str] = []
    lifecycle_events: list[OrderLifecycleEvent] = []
    position_updates: list[ExistingPosition] = []

    ordered_updates = sorted(
        updates,
        key=lambda update: (
            update.updated_at or update.submitted_at or "",
            update.broker_order_id,
        ),
    )
    for update in ordered_updates:
        matched_record = _find_matching_pending_order(current_state, update)
        if matched_record is None:
            unmatched_updates.append(update)
            warnings.append(
                "Broker update could not be matched to a local pending order: "
                f"{update.symbol} {update.broker_order_id}."
            )
            continue
        if matched_record.symbol != update.symbol:
            unmatched_updates.append(update)
            warnings.append(
                "Broker update symbol does not match the local pending order: "
                f"{update.broker_order_id} -> {update.symbol}."
            )
            continue
        apply_result = _apply_broker_update(
            current_state,
            matched_record=matched_record,
            update=update,
        )
        current_state = apply_result.state
        matched_updates.append(update)
        warnings.extend(apply_result.warnings)
        lifecycle_events.extend(apply_result.lifecycle_events)
        position_updates.extend(apply_result.position_updates)

    return BrokerReconciliationResult(
        state=current_state,
        matched_updates=tuple(matched_updates),
        unmatched_updates=tuple(unmatched_updates),
        warnings=tuple(warnings),
        lifecycle_events=tuple(lifecycle_events),
        position_updates=tuple(position_updates),
    )


@dataclass(frozen=True)
class _ApplyBrokerUpdateResult:
    state: PendingOrderState
    warnings: tuple[str, ...] = ()
    lifecycle_events: tuple[OrderLifecycleEvent, ...] = ()
    position_updates: tuple[ExistingPosition, ...] = ()


def _apply_broker_update(
    state: PendingOrderState,
    *,
    matched_record: PendingOrderRecord,
    update: BrokerOrderUpdate,
) -> _ApplyBrokerUpdateResult:
    occurred_at = update.updated_at or update.submitted_at or matched_record.created_at
    current_state = state
    warnings: list[str] = []
    merged_record = _merge_pending_order_record_with_broker_update(
        matched_record,
        update,
    )
    if merged_record != matched_record:
        current_state = upsert_pending_order_state(
            current_state,
            merged_record,
            updated_at=occurred_at,
        )

    current_record = current_state.orders[matched_record.order_id]
    if update.status in {
        "accepted",
        "pending_cancel",
        "pending_replace",
        "done_for_day",
        "stopped",
        "suspended",
        "unknown",
    }:
        return _ApplyBrokerUpdateResult(state=current_state, warnings=tuple(warnings))

    if update.status == "partially_filled":
        if update.filled_quantity < current_record.filled_quantity:
            warnings.append(
                "Ignoring broker partial-fill update because filled quantity moved backward "
                f"for local order '{current_record.order_id}'."
            )
            return _ApplyBrokerUpdateResult(state=current_state, warnings=tuple(warnings))
        result = mark_pending_order_partially_filled(
            current_state,
            current_record.order_id,
            occurred_at=occurred_at,
            filled_quantity=update.filled_quantity,
        )
        return _finalize_lifecycle_result(result, update=update, occurred_at=occurred_at)

    if update.status == "filled":
        result = mark_pending_order_filled(
            current_state,
            current_record.order_id,
            occurred_at=occurred_at,
            fill_price=update.average_fill_price,
        )
        return _finalize_lifecycle_result(result, update=update, occurred_at=occurred_at)

    if update.status == "cancelled":
        result = mark_pending_order_cancelled(
            current_state,
            current_record.order_id,
            occurred_at=occurred_at,
        )
        return _finalize_lifecycle_result(result, update=update, occurred_at=occurred_at)

    if update.status == "expired":
        result = mark_pending_order_expired(
            current_state,
            current_record.order_id,
            occurred_at=occurred_at,
        )
        return _finalize_lifecycle_result(result, update=update, occurred_at=occurred_at)

    if update.status == "rejected":
        warnings.append(
            f"Broker rejected local pending order '{current_record.order_id}'."
        )
        result = mark_pending_order_cancelled(
            current_state,
            current_record.order_id,
            occurred_at=occurred_at,
        )
        return _finalize_lifecycle_result(result, update=update, occurred_at=occurred_at)

    if update.status == "replaced":
        replaced_record = replace(
            current_record,
            status="replaced",
            reserved_notional=0.0,
            reserved_slot=False,
        )
        replaced_record = _merge_pending_order_record_with_broker_update(
            replaced_record,
            update,
        )
        updated_state = upsert_pending_order_state(
            current_state,
            replaced_record,
            updated_at=occurred_at,
        )
        return _ApplyBrokerUpdateResult(
            state=updated_state,
            warnings=(
                "Broker reported a replaced order without a local replacement record. "
                f"Local order '{current_record.order_id}' was finalized as replaced.",
            ),
            lifecycle_events=(
                OrderLifecycleEvent(
                    event_type="replaced",
                    occurred_at=occurred_at,
                    order_id=replaced_record.order_id,
                    symbol=replaced_record.symbol,
                    status=replaced_record.status,
                    filled_quantity=replaced_record.filled_quantity,
                    remaining_quantity=replaced_record.remaining_quantity,
                    reserved_notional=replaced_record.reserved_notional,
                    reserved_slot=replaced_record.reserved_slot,
                    trade_id=replaced_record.trade_id,
                    linked_decision_id=replaced_record.linked_decision_id,
                ),
            ),
        )

    return _ApplyBrokerUpdateResult(state=current_state, warnings=tuple(warnings))


def _finalize_lifecycle_result(
    result: PendingOrderUpdateResult,
    *,
    update: BrokerOrderUpdate,
    occurred_at: str,
) -> _ApplyBrokerUpdateResult:
    current_state = result.state
    if result.orders:
        merged_record = _merge_pending_order_record_with_broker_update(result.orders[0], update)
        if merged_record != result.orders[0]:
            current_state = upsert_pending_order_state(
                current_state,
                merged_record,
                updated_at=occurred_at,
            )
    position_updates = (
        (result.position_update,)
        if result.position_update is not None
        else ()
    )
    return _ApplyBrokerUpdateResult(
        state=current_state,
        lifecycle_events=result.events,
        position_updates=position_updates,
    )


def _merge_pending_order_record_with_broker_update(
    record: PendingOrderRecord,
    update: BrokerOrderUpdate,
) -> PendingOrderRecord:
    broker_metadata = dict(record.metadata.get("broker_metadata", {}))
    broker_metadata.update(update.metadata)
    updated_metadata = dict(record.metadata)
    if broker_metadata:
        updated_metadata["broker_metadata"] = broker_metadata
    return replace(
        record,
        requested_quantity=update.quantity,
        requested_price=(
            update.limit_price
            if update.order_type in {"LIMIT", "STOP_LIMIT"}
            else record.requested_price
        ),
        stop_price=(
            update.stop_price
            if update.order_type == "STOP_LIMIT"
            else record.stop_price
        ),
        order_type=update.order_type,
        broker_name=update.broker_name,
        broker_order_id=update.broker_order_id,
        client_order_id=update.client_order_id or record.client_order_id or record.order_id,
        broker_status=update.status,
        submitted_at=update.submitted_at or record.submitted_at,
        last_broker_update_at=(
            update.updated_at
            or update.submitted_at
            or record.last_broker_update_at
        ),
        average_fill_price=update.average_fill_price or record.average_fill_price,
        metadata=updated_metadata,
    )


def _find_matching_pending_order(
    state: PendingOrderState,
    update: BrokerOrderUpdate,
) -> PendingOrderRecord | None:
    if update.broker_order_id:
        for record in state.orders.values():
            if record.broker_order_id == update.broker_order_id:
                return record
    client_order_id = update.client_order_id
    if client_order_id is not None:
        record = state.orders.get(client_order_id)
        if record is not None:
            return record
        for existing in state.orders.values():
            if existing.client_order_id == client_order_id:
                return existing
    for record in state.active_orders():
        if record.symbol != update.symbol:
            continue
        if record.trade_id is not None:
            broker_trade_id = update.metadata.get("trade_id")
            if isinstance(broker_trade_id, str) and broker_trade_id.strip() == record.trade_id:
                return record
        if record.linked_decision_id is not None:
            broker_decision_id = update.metadata.get("linked_decision_id")
            if (
                isinstance(broker_decision_id, str)
                and broker_decision_id.strip() == record.linked_decision_id
            ):
                return record
    return None
