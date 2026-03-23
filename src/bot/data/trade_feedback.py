"""Compact trade-decision logging and post-trade outcome helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from bot.logging_utils import get_logger


LOGGER = get_logger(__name__)

TRADE_FEEDBACK_SCHEMA_VERSION = 1
DEFAULT_TRADE_OUTCOME_WINDOWS = (1, 5, 10)


@dataclass(frozen=True)
class TradeOutcomeSnapshot:
    """Compact forward/outcome metrics for one executed trade."""

    as_of_date: date
    entry_date: date
    sessions_since_entry: int
    current_return_pct: float
    max_favorable_excursion_pct: float | None = None
    max_adverse_excursion_pct: float | None = None
    stop_hit: bool | None = None
    forward_return_1d: float | None = None
    forward_return_5d: float | None = None
    forward_return_10d: float | None = None
    above_entry_after_1d: bool | None = None
    above_entry_after_5d: bool | None = None
    above_entry_after_10d: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly outcome payload."""

        return {
            "as_of_date": self.as_of_date.isoformat(),
            "entry_date": self.entry_date.isoformat(),
            "sessions_since_entry": self.sessions_since_entry,
            "current_return_pct": self.current_return_pct,
            "max_favorable_excursion_pct": self.max_favorable_excursion_pct,
            "max_adverse_excursion_pct": self.max_adverse_excursion_pct,
            "stop_hit": self.stop_hit,
            "forward_return_1d": self.forward_return_1d,
            "forward_return_5d": self.forward_return_5d,
            "forward_return_10d": self.forward_return_10d,
            "above_entry_after_1d": self.above_entry_after_1d,
            "above_entry_after_5d": self.above_entry_after_5d,
            "above_entry_after_10d": self.above_entry_after_10d,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TradeOutcomeSnapshot":
        """Build an outcome snapshot from stored JSON."""

        return cls(
            as_of_date=_mapping_date(data, "as_of_date"),
            entry_date=_mapping_date(data, "entry_date"),
            sessions_since_entry=_mapping_non_negative_int(data, "sessions_since_entry"),
            current_return_pct=_mapping_float(data, "current_return_pct"),
            max_favorable_excursion_pct=_mapping_optional_float(
                data,
                "max_favorable_excursion_pct",
            ),
            max_adverse_excursion_pct=_mapping_optional_float(
                data,
                "max_adverse_excursion_pct",
            ),
            stop_hit=_mapping_optional_bool(data, "stop_hit"),
            forward_return_1d=_mapping_optional_float(data, "forward_return_1d"),
            forward_return_5d=_mapping_optional_float(data, "forward_return_5d"),
            forward_return_10d=_mapping_optional_float(data, "forward_return_10d"),
            above_entry_after_1d=_mapping_optional_bool(data, "above_entry_after_1d"),
            above_entry_after_5d=_mapping_optional_bool(data, "above_entry_after_5d"),
            above_entry_after_10d=_mapping_optional_bool(data, "above_entry_after_10d"),
        )


@dataclass(frozen=True)
class TradeFeedbackEvent:
    """One persisted decision, execution, or outcome event."""

    event_type: str
    workflow: str
    symbol: str
    as_of_date: date | None = None
    timestamp_utc: str | None = None
    decision_id: str | None = None
    trade_id: str | None = None
    linked_decision_id: str | None = None
    preset_name: str | None = None
    parameter_id: str | None = None
    strategy_name: str | None = None
    suggested_action: str | None = None
    approved: bool | None = None
    rejection_reasons: tuple[str, ...] = ()
    queue_rank: int | None = None
    priority_bucket: str | None = None
    actionable_now: bool | None = None
    quantity: int | None = None
    entry_price_hint: float | None = None
    stop_price: float | None = None
    notional_value: float | None = None
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    outcome_snapshot: TradeOutcomeSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = TRADE_FEEDBACK_SCHEMA_VERSION
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type cannot be empty.")
        if not self.workflow.strip():
            raise ValueError("workflow cannot be empty.")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.queue_rank is not None and self.queue_rank <= 0:
            raise ValueError("queue_rank must be greater than zero when provided.")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be greater than zero when provided.")
        object.__setattr__(self, "event_id", build_trade_feedback_event_id(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-friendly event payload."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "workflow": self.workflow,
            "symbol": self.symbol,
        }
        if self.as_of_date is not None:
            payload["as_of_date"] = self.as_of_date.isoformat()
        if self.timestamp_utc is not None:
            payload["timestamp_utc"] = self.timestamp_utc
        if self.decision_id is not None:
            payload["decision_id"] = self.decision_id
        if self.trade_id is not None:
            payload["trade_id"] = self.trade_id
        if self.linked_decision_id is not None:
            payload["linked_decision_id"] = self.linked_decision_id
        if self.preset_name is not None:
            payload["preset_name"] = self.preset_name
        if self.parameter_id is not None:
            payload["parameter_id"] = self.parameter_id
        if self.strategy_name is not None:
            payload["strategy_name"] = self.strategy_name
        if self.suggested_action is not None:
            payload["suggested_action"] = self.suggested_action
        if self.approved is not None:
            payload["approved"] = self.approved
        if self.rejection_reasons:
            payload["rejection_reasons"] = list(self.rejection_reasons)
        if self.queue_rank is not None:
            payload["queue_rank"] = self.queue_rank
        if self.priority_bucket is not None:
            payload["priority_bucket"] = self.priority_bucket
        if self.actionable_now is not None:
            payload["actionable_now"] = self.actionable_now
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        if self.entry_price_hint is not None:
            payload["entry_price_hint"] = self.entry_price_hint
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        if self.notional_value is not None:
            payload["notional_value"] = self.notional_value
        if self.feature_snapshot:
            payload["feature_snapshot"] = dict(self.feature_snapshot)
        if self.outcome_snapshot is not None:
            payload["outcome_snapshot"] = self.outcome_snapshot.to_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TradeFeedbackEvent":
        """Build an event from stored JSON."""

        raw_feature_snapshot = data.get("feature_snapshot", {})
        if raw_feature_snapshot is None:
            raw_feature_snapshot = {}
        if not isinstance(raw_feature_snapshot, Mapping):
            raise ValueError("feature_snapshot must be a mapping when provided.")
        raw_metadata = data.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided.")
        raw_outcome_snapshot = data.get("outcome_snapshot")
        if raw_outcome_snapshot is not None and not isinstance(raw_outcome_snapshot, Mapping):
            raise ValueError("outcome_snapshot must be a mapping when provided.")

        return cls(
            event_type=_mapping_text(data, "event_type"),
            workflow=_mapping_text(data, "workflow"),
            symbol=_mapping_text(data, "symbol").upper(),
            as_of_date=_mapping_optional_date(data, "as_of_date"),
            timestamp_utc=_mapping_optional_text(data, "timestamp_utc"),
            decision_id=_mapping_optional_text(data, "decision_id"),
            trade_id=_mapping_optional_text(data, "trade_id"),
            linked_decision_id=_mapping_optional_text(data, "linked_decision_id"),
            preset_name=_mapping_optional_text(data, "preset_name"),
            parameter_id=_mapping_optional_text(data, "parameter_id"),
            strategy_name=_mapping_optional_text(data, "strategy_name"),
            suggested_action=_mapping_optional_text(data, "suggested_action"),
            approved=_mapping_optional_bool(data, "approved"),
            rejection_reasons=_mapping_text_sequence(data, "rejection_reasons"),
            queue_rank=_mapping_optional_positive_int(data, "queue_rank"),
            priority_bucket=_mapping_optional_text(data, "priority_bucket"),
            actionable_now=_mapping_optional_bool(data, "actionable_now"),
            quantity=_mapping_optional_positive_int(data, "quantity"),
            entry_price_hint=_mapping_optional_float(data, "entry_price_hint"),
            stop_price=_mapping_optional_float(data, "stop_price"),
            notional_value=_mapping_optional_float(data, "notional_value"),
            feature_snapshot=dict(raw_feature_snapshot),
            outcome_snapshot=(
                TradeOutcomeSnapshot.from_mapping(raw_outcome_snapshot)
                if isinstance(raw_outcome_snapshot, Mapping)
                else None
            ),
            metadata=dict(raw_metadata),
            schema_version=_mapping_optional_positive_int(data, "schema_version")
            or TRADE_FEEDBACK_SCHEMA_VERSION,
        )


def default_trade_feedback_log_path(project_root: Path) -> Path:
    """Return the repo-relative JSONL path for persisted feedback events."""

    return project_root / "data" / "processed" / "state" / "trade_decision_log.jsonl"


def build_trade_decision_id(
    *,
    workflow: str,
    as_of_date: date,
    symbol: str,
    signal_date: date | None = None,
    preset_name: str | None = None,
    parameter_id: str | None = None,
    strategy_name: str | None = None,
    review_timestamp: str | None = None,
) -> str:
    """Return a deterministic decision identifier for one workflow decision."""

    return _stable_id(
        "decision",
        {
            "workflow": workflow,
            "as_of_date": as_of_date.isoformat(),
            "symbol": symbol.strip().upper(),
            "signal_date": signal_date.isoformat() if signal_date is not None else None,
            "preset_name": _clean_text(preset_name),
            "parameter_id": _clean_text(parameter_id),
            "strategy_name": _clean_text(strategy_name),
            "review_timestamp": _clean_text(review_timestamp),
        },
    )


def build_trade_id(
    *,
    symbol: str,
    entry_date: date | None,
    average_entry_price: float | None,
    decision_id: str | None = None,
) -> str:
    """Return a deterministic trade identifier for one executed position."""

    return _stable_id(
        "trade",
        {
            "symbol": symbol.strip().upper(),
            "entry_date": entry_date.isoformat() if entry_date is not None else None,
            "average_entry_price": average_entry_price,
            "decision_id": _clean_text(decision_id),
        },
    )


def build_trade_feedback_event_id(event: TradeFeedbackEvent) -> str:
    """Return the stable append/dedup identifier for one event."""

    identity_payload = {
        "event_type": event.event_type,
        "workflow": event.workflow,
        "symbol": event.symbol,
        "as_of_date": event.as_of_date.isoformat() if event.as_of_date is not None else None,
        "decision_id": event.decision_id,
        "trade_id": event.trade_id,
        "linked_decision_id": event.linked_decision_id,
        "approved": event.approved,
    }
    if event.event_type in {"stop_updated", "stop_raised"}:
        identity_payload["stop_price"] = event.stop_price
    return _stable_id("event", identity_payload)


def load_trade_feedback_events(log_path: Path) -> tuple[TradeFeedbackEvent, ...]:
    """Load JSONL feedback events, skipping malformed lines."""

    resolved_path = log_path.resolve()
    if not resolved_path.exists():
        return ()

    events: list[TradeFeedbackEvent] = []
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if not isinstance(payload, Mapping):
                        raise ValueError("event line must decode to a mapping.")
                    events.append(TradeFeedbackEvent.from_mapping(payload))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    LOGGER.warning(
                        "Ignoring malformed trade feedback line %s in %s: %s",
                        line_number,
                        resolved_path,
                        exc,
                    )
    except OSError as exc:
        LOGGER.warning(
            "Ignoring trade feedback log at %s because it could not be read: %s",
            resolved_path,
            exc,
        )
        return ()
    return tuple(events)


def append_trade_feedback_events(
    events: Sequence[TradeFeedbackEvent],
    log_path: Path,
) -> Path:
    """Append events to the JSONL log while deduplicating by stable event id."""

    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    existing_event_ids = {
        event.event_id
        for event in load_trade_feedback_events(resolved_path)
    }
    ordered_events = sorted(
        events,
        key=lambda event: (
            event.as_of_date.isoformat() if event.as_of_date is not None else "",
            event.symbol,
            event.workflow,
            event.event_type,
            event.decision_id or "",
            event.trade_id or "",
        ),
    )
    with resolved_path.open("a", encoding="utf-8") as handle:
        for event in ordered_events:
            if event.event_id in existing_event_ids:
                continue
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")
            existing_event_ids.add(event.event_id)
    return resolved_path


def summarize_trade_feedback_events(
    events: Sequence[TradeFeedbackEvent],
) -> dict[str, Any]:
    """Return compact rollup metrics from persisted feedback events."""

    approved_decision_ids = {
        event.decision_id
        for event in events
        if event.event_type == "decision"
        and event.approved is True
        and event.decision_id is not None
    }
    executed_decision_ids = {
        event.linked_decision_id or event.decision_id
        for event in events
        if event.event_type == "executed"
        and (event.linked_decision_id is not None or event.decision_id is not None)
    }
    return {
        "decision_count": sum(event.event_type == "decision" for event in events),
        "executed_count": sum(event.event_type == "executed" for event in events),
        "closed_count": sum(event.event_type == "closed" for event in events),
        "outcome_count": sum(event.event_type == "outcome" for event in events),
        "approved_decision_count": len(approved_decision_ids),
        "executed_approved_decision_count": len(
            approved_decision_ids & executed_decision_ids
        ),
        "approved_execution_rate": (
            len(approved_decision_ids & executed_decision_ids) / len(approved_decision_ids)
            if approved_decision_ids
            else None
        ),
    }


def compute_trade_outcome_snapshot(
    price_frame: pd.DataFrame,
    *,
    entry_date: date,
    entry_price: float,
    stop_price: float | None = None,
    as_of_date: date | None = None,
    windows: Sequence[int] = DEFAULT_TRADE_OUTCOME_WINDOWS,
) -> TradeOutcomeSnapshot | None:
    """Compute lightweight post-entry outcome metrics from daily bars."""

    if entry_price <= 0:
        raise ValueError("entry_price must be greater than zero.")

    prepared = price_frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared["high"] = pd.to_numeric(prepared.get("high"), errors="coerce")
    prepared["low"] = pd.to_numeric(prepared.get("low"), errors="coerce")
    prepared = prepared.dropna(subset=["date", "close"]).sort_values("date", kind="stable")
    if as_of_date is not None:
        prepared = prepared.loc[prepared["date"].dt.date <= as_of_date]
    if prepared.empty:
        return None

    entry_mask = prepared["date"].dt.date >= entry_date
    if not bool(entry_mask.any()):
        return None
    prepared = prepared.loc[entry_mask].reset_index(drop=True)
    if prepared.empty:
        return None

    latest_date = prepared.iloc[-1]["date"].date()
    latest_close = float(prepared.iloc[-1]["close"])
    high_reference = prepared["high"].fillna(prepared["close"])
    low_reference = prepared["low"].fillna(prepared["close"])
    forward_returns: dict[int, float | None] = {}
    above_entry_flags: dict[int, bool | None] = {}
    for window in windows:
        target_index = window
        if target_index >= len(prepared):
            forward_returns[window] = None
            above_entry_flags[window] = None
            continue
        target_close = float(prepared.iloc[target_index]["close"])
        forward_returns[window] = (target_close / entry_price) - 1.0
        above_entry_flags[window] = target_close >= entry_price

    return TradeOutcomeSnapshot(
        as_of_date=latest_date,
        entry_date=entry_date,
        sessions_since_entry=max(len(prepared) - 1, 0),
        current_return_pct=(latest_close / entry_price) - 1.0,
        max_favorable_excursion_pct=(float(high_reference.max()) / entry_price) - 1.0,
        max_adverse_excursion_pct=(float(low_reference.min()) / entry_price) - 1.0,
        stop_hit=(
            bool((low_reference <= stop_price).any())
            if stop_price is not None
            else None
        ),
        forward_return_1d=forward_returns.get(1),
        forward_return_5d=forward_returns.get(5),
        forward_return_10d=forward_returns.get(10),
        above_entry_after_1d=above_entry_flags.get(1),
        above_entry_after_5d=above_entry_flags.get(5),
        above_entry_after_10d=above_entry_flags.get(10),
    )


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_text(data: Mapping[str, Any], key: str) -> str:
    value = _clean_text(data.get(key))
    if value is None:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _mapping_optional_text(data: Mapping[str, Any], key: str) -> str | None:
    return _clean_text(data.get(key))


def _mapping_date(data: Mapping[str, Any], key: str) -> date:
    value = _mapping_optional_date(data, key)
    if value is None:
        raise ValueError(f"{key} must be an ISO date string.")
    return value


def _mapping_optional_date(data: Mapping[str, Any], key: str) -> date | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{key} must be an ISO date string when provided.")
    try:
        return date.fromisoformat(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be an ISO date string when provided.") from exc


def _mapping_float(data: Mapping[str, Any], key: str) -> float:
    value = _mapping_optional_float(data, key)
    if value is None:
        raise ValueError(f"{key} must be numeric.")
    return value


def _mapping_optional_float(data: Mapping[str, Any], key: str) -> float | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be numeric when provided.")
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            return float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric when provided.") from exc
    raise ValueError(f"{key} must be numeric when provided.")


def _mapping_optional_bool(data: Mapping[str, Any], key: str) -> bool | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, bool):
        raise ValueError(f"{key} must be a boolean when provided.")
    return raw_value


def _mapping_non_negative_int(data: Mapping[str, Any], key: str) -> int:
    raw_value = data.get(key)
    if raw_value is None:
        raise ValueError(f"{key} must be a non-negative integer.")
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be a non-negative integer.")
    if isinstance(raw_value, int):
        if raw_value < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be a non-negative integer.") from exc
        if parsed < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
        return parsed
    raise ValueError(f"{key} must be a non-negative integer.")


def _mapping_optional_positive_int(data: Mapping[str, Any], key: str) -> int | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be a positive integer when provided.")
    if isinstance(raw_value, int):
        if raw_value <= 0:
            raise ValueError(f"{key} must be a positive integer when provided.")
        return raw_value
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be a positive integer when provided.") from exc
        if parsed <= 0:
            raise ValueError(f"{key} must be a positive integer when provided.")
        return parsed
    raise ValueError(f"{key} must be a positive integer when provided.")


def _mapping_text_sequence(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw_value = data.get(key)
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of strings when provided.")
    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} must contain only non-empty strings.")
        values.append(item.strip())
    return tuple(values)
