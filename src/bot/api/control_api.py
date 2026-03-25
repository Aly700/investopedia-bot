"""Typed operator control API and lightweight HTTP server."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from bot.broker import (
    BrokerExecutionAdapter,
    BrokerExecutionError,
    BrokerOrderUpdate,
    BrokerReplaceOrderRequest,
    create_broker_execution_adapter,
    reconcile_broker_order_updates,
    submit_pending_orders_via_broker,
)
from bot.data.pending_orders import (
    PendingOrderRecord,
    PendingOrderState,
    PendingOrderStateStore,
    replace_pending_order,
    upsert_pending_order_state,
)
from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    coerce_positive_int,
    load_json_mapping_file,
    write_json_file,
)
from bot.data.trade_feedback import (
    TradeFeedbackEvent,
    TradeFeedbackLogStore,
)
from bot.logging_utils import get_logger


LOGGER = get_logger(__name__)

OPERATOR_CONTROL_API_VERSION = 1
OPERATOR_CONTROL_HTTP_ENDPOINTS = (
    "/control",
    "/control/safety",
    "/control/execution/policy",
    "/control/execution/pause",
    "/control/execution/resume",
    "/control/execution/mode",
    "/control/execution/live-confirmation",
    "/control/execution/broker-trading",
    "/control/orders/submit",
    "/control/orders/cancel",
    "/control/orders/replace",
    "/control/broker/sync",
)
OPERATOR_CONTROL_RESULT_STATUSES = (
    "completed",
    "rejected",
    "failed",
    "no_op",
    "needs_confirmation",
)
OPERATOR_EXECUTION_MODES = ("paper", "live")
OPERATOR_CONTROL_STATE_SCHEMA_VERSION = 1
_OPERATOR_CONTROL_STATE_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "operator_control_state.json",
    ),
)
_OPERATOR_CONTROL_AUDIT_LOG_PATH = (
    "data",
    "processed",
    "state",
    "logs",
    "operator_control_audit.jsonl",
)


@dataclass(frozen=True)
class OperatorControlState:
    """Persisted operator-side execution guardrails."""

    schema_version: int = OPERATOR_CONTROL_STATE_SCHEMA_VERSION
    updated_at_utc: str | None = None
    execution_submission_enabled: bool = True
    execution_mode: str = "paper"
    live_actions_require_confirmation: bool = True
    broker_trading_enabled: bool = False
    paused_reason: str | None = None
    last_pause_at_utc: str | None = None
    last_resume_at_utc: str | None = None

    def __post_init__(self) -> None:
        coerce_positive_int(self.schema_version, "schema_version")
        if self.updated_at_utc is not None:
            coerce_iso8601_datetime_text(self.updated_at_utc, "updated_at_utc")
        if self.execution_mode not in OPERATOR_EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be one of "
                f"{OPERATOR_EXECUTION_MODES}, got '{self.execution_mode}'."
            )
        if self.last_pause_at_utc is not None:
            coerce_iso8601_datetime_text(self.last_pause_at_utc, "last_pause_at_utc")
        if self.last_resume_at_utc is not None:
            coerce_iso8601_datetime_text(self.last_resume_at_utc, "last_resume_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at_utc": self.updated_at_utc,
            "execution_submission_enabled": self.execution_submission_enabled,
            "execution_mode": self.execution_mode,
            "live_actions_require_confirmation": self.live_actions_require_confirmation,
            "broker_trading_enabled": self.broker_trading_enabled,
            "paused_reason": self.paused_reason,
            "last_pause_at_utc": self.last_pause_at_utc,
            "last_resume_at_utc": self.last_resume_at_utc,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorControlState":
        return cls(
            schema_version=coerce_positive_int(payload.get("schema_version"), "schema_version"),
            updated_at_utc=_optional_text(payload.get("updated_at_utc")),
            execution_submission_enabled=_mapping_optional_bool(
                payload,
                "execution_submission_enabled",
                default=True,
            ),
            execution_mode=_optional_text(payload.get("execution_mode")) or "paper",
            live_actions_require_confirmation=_mapping_optional_bool(
                payload,
                "live_actions_require_confirmation",
                default=True,
            ),
            broker_trading_enabled=_mapping_optional_bool(
                payload,
                "broker_trading_enabled",
                default=False,
            ),
            paused_reason=_optional_text(payload.get("paused_reason")),
            last_pause_at_utc=_optional_text(payload.get("last_pause_at_utc")),
            last_resume_at_utc=_optional_text(payload.get("last_resume_at_utc")),
        )


@dataclass(frozen=True)
class _OperatorControlStateLoadResult:
    state: OperatorControlState
    warnings: tuple[str, ...] = ()
    degraded: bool = False


class BrokerAdapterFactory(Protocol):
    def __call__(
        self,
        broker_name: str,
        *,
        env_file: Path | None = None,
        target: str | None = None,
    ) -> BrokerExecutionAdapter: ...


@dataclass(frozen=True)
class OperatorControlStateStore:
    project_root: Path

    @property
    def path(self) -> Path:
        return default_operator_control_state_path(self.project_root)

    def load(self) -> OperatorControlState:
        return load_operator_control_state(self.path)

    def save(self, state: OperatorControlState) -> Path:
        return write_json_file(self.path, state.to_dict())


def default_operator_control_state_path(project_root: Path) -> Path:
    return _OPERATOR_CONTROL_STATE_PATH.resolve(project_root)


def load_operator_control_state(path: Path) -> OperatorControlState:
    return _load_operator_control_state_result(path).state


def _load_operator_control_state_result(path: Path) -> _OperatorControlStateLoadResult:
    resolved_path = path.resolve()
    if not resolved_path.exists():
        return _OperatorControlStateLoadResult(state=OperatorControlState())
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("operator control state payload must be a mapping.")
        return _OperatorControlStateLoadResult(
            state=OperatorControlState.from_mapping(payload)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "Ignoring operator control state at %s because it could not be loaded: %s",
            resolved_path,
            exc,
        )
        return _OperatorControlStateLoadResult(
            state=OperatorControlState(
                execution_submission_enabled=False,
                paused_reason=(
                    "Operator control state is unreadable; execution submissions are "
                    "paused for safety."
                ),
            ),
            warnings=(
                f"Operator control state at {resolved_path} could not be loaded: {exc}",
            ),
            degraded=True,
        )


@dataclass(frozen=True)
class OperatorControlAuditRecord:
    """One persisted operator control action record."""

    command_name: str
    requested_at_utc: str
    status: str
    message: str
    operator_id: str | None = None
    error_code: str | None = None
    warnings: tuple[str, ...] = ()
    command_payload: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] = field(default_factory=dict)
    audit_record_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in OPERATOR_CONTROL_RESULT_STATUSES:
            raise ValueError(
                "status must be one of "
                f"{OPERATOR_CONTROL_RESULT_STATUSES}, got '{self.status}'."
            )
        if not self.command_name.strip():
            raise ValueError("command_name cannot be empty.")
        if not self.message.strip():
            raise ValueError("message cannot be empty.")
        coerce_iso8601_datetime_text(self.requested_at_utc, "requested_at_utc")
        object.__setattr__(self, "audit_record_id", _operator_control_audit_record_id(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_record_id": self.audit_record_id,
            "command_name": self.command_name,
            "requested_at_utc": self.requested_at_utc,
            "status": self.status,
            "message": self.message,
            "operator_id": self.operator_id,
            "error_code": self.error_code,
            "warnings": list(self.warnings),
            "command_payload": dict(self.command_payload),
            "result_payload": dict(self.result_payload),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorControlAuditRecord":
        raw_warnings = payload.get("warnings", ())
        if not isinstance(raw_warnings, Sequence) or isinstance(raw_warnings, (str, bytes)):
            raise ValueError("warnings must be a sequence.")
        raw_command_payload = payload.get("command_payload", {})
        raw_result_payload = payload.get("result_payload", {})
        if not isinstance(raw_command_payload, Mapping):
            raise ValueError("command_payload must be a mapping.")
        if not isinstance(raw_result_payload, Mapping):
            raise ValueError("result_payload must be a mapping.")
        return cls(
            command_name=_required_text(payload.get("command_name"), "command_name"),
            requested_at_utc=coerce_iso8601_datetime_text(
                payload.get("requested_at_utc"),
                "requested_at_utc",
            ),
            status=_required_text(payload.get("status"), "status"),
            message=_required_text(payload.get("message"), "message"),
            operator_id=_optional_text(payload.get("operator_id")),
            error_code=_optional_text(payload.get("error_code")),
            warnings=tuple(
                item.strip()
                for item in raw_warnings
                if isinstance(item, str) and item.strip()
            ),
            command_payload=dict(raw_command_payload),
            result_payload=dict(raw_result_payload),
        )


def default_operator_control_audit_log_path(project_root: Path) -> Path:
    return project_root.resolve().joinpath(*_OPERATOR_CONTROL_AUDIT_LOG_PATH)


def load_operator_control_audit_records(
    log_path: Path,
) -> tuple[OperatorControlAuditRecord, ...]:
    resolved_path = log_path.resolve()
    if not resolved_path.exists():
        return ()
    records: list[OperatorControlAuditRecord] = []
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if not isinstance(payload, Mapping):
                        raise ValueError("audit line must decode to a mapping.")
                    records.append(OperatorControlAuditRecord.from_mapping(payload))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    LOGGER.warning(
                        "Ignoring malformed operator control audit line %s in %s: %s",
                        line_number,
                        resolved_path,
                        exc,
                    )
    except OSError as exc:
        LOGGER.warning(
            "Ignoring operator control audit log at %s because it could not be read: %s",
            resolved_path,
            exc,
        )
        return ()
    return tuple(records)


def append_operator_control_audit_record(
    record: OperatorControlAuditRecord,
    log_path: Path,
) -> Path:
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True))
        handle.write("\n")
    return resolved_path


@dataclass(frozen=True)
class SubmitPendingOrderCommand:
    broker: str
    order_id: str
    confirm_live_action: bool = False
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("broker cannot be empty.")
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "broker": self.broker,
            "order_id": self.order_id,
            "confirm_live_action": self.confirm_live_action,
        }
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SubmitPendingOrderCommand":
        return cls(
            broker=_required_text(payload.get("broker"), "broker"),
            order_id=_required_text(payload.get("order_id"), "order_id"),
            confirm_live_action=_mapping_optional_bool(
                payload,
                "confirm_live_action",
                default=False,
            ),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class CancelPendingOrderCommand:
    broker: str
    order_id: str
    confirm_live_action: bool = False
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("broker cannot be empty.")
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "broker": self.broker,
            "order_id": self.order_id,
            "confirm_live_action": self.confirm_live_action,
        }
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CancelPendingOrderCommand":
        return cls(
            broker=_required_text(payload.get("broker"), "broker"),
            order_id=_required_text(payload.get("order_id"), "order_id"),
            confirm_live_action=_mapping_optional_bool(
                payload,
                "confirm_live_action",
                default=False,
            ),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class ReplacePendingOrderCommand:
    broker: str
    order_id: str
    quantity: int | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    confirm_live_action: bool = False
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("broker cannot be empty.")
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")
        if self.quantity is None and self.limit_price is None and self.stop_price is None:
            raise ValueError(
                "At least one of quantity, limit_price, or stop_price must be provided."
            )
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be greater than zero when provided.")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "broker": self.broker,
            "order_id": self.order_id,
        }
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        if self.limit_price is not None:
            payload["limit_price"] = self.limit_price
        if self.stop_price is not None:
            payload["stop_price"] = self.stop_price
        payload["confirm_live_action"] = self.confirm_live_action
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReplacePendingOrderCommand":
        return cls(
            broker=_required_text(payload.get("broker"), "broker"),
            order_id=_required_text(payload.get("order_id"), "order_id"),
            quantity=_mapping_optional_positive_int(payload, "quantity"),
            limit_price=_mapping_optional_positive_float(payload, "limit_price"),
            stop_price=_mapping_optional_positive_float(payload, "stop_price"),
            confirm_live_action=_mapping_optional_bool(
                payload,
                "confirm_live_action",
                default=False,
            ),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class ForceBrokerSyncCommand:
    broker: str
    order_ids: tuple[str, ...] = ()
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("broker cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "broker": self.broker,
            "order_ids": list(self.order_ids),
        }
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ForceBrokerSyncCommand":
        return cls(
            broker=_required_text(payload.get("broker"), "broker"),
            order_ids=_mapping_text_sequence(payload, "order_ids"),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class PauseExecutionCommand:
    reason: str | None = None
    operator_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PauseExecutionCommand":
        return cls(
            reason=_optional_text(payload.get("reason")),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class ResumeExecutionCommand:
    reason: str | None = None
    operator_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResumeExecutionCommand":
        return cls(
            reason=_optional_text(payload.get("reason")),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class SetExecutionModeCommand:
    execution_mode: str
    reason: str | None = None
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in OPERATOR_EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be one of "
                f"{OPERATOR_EXECUTION_MODES}, got '{self.execution_mode}'."
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"execution_mode": self.execution_mode}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SetExecutionModeCommand":
        return cls(
            execution_mode=_required_text(payload.get("execution_mode"), "execution_mode"),
            reason=_optional_text(payload.get("reason")),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class SetLiveActionConfirmationCommand:
    required: bool
    reason: str | None = None
    operator_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"required": self.required}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "SetLiveActionConfirmationCommand":
        return cls(
            required=_mapping_required_bool(payload, "required"),
            reason=_optional_text(payload.get("reason")),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class SetBrokerTradingEnabledCommand:
    enabled: bool
    reason: str | None = None
    operator_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"enabled": self.enabled}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.operator_id is not None:
            payload["operator_id"] = self.operator_id
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "SetBrokerTradingEnabledCommand":
        return cls(
            enabled=_mapping_required_bool(payload, "enabled"),
            reason=_optional_text(payload.get("reason")),
            operator_id=_optional_text(payload.get("operator_id")),
        )


@dataclass(frozen=True)
class OperatorCommandResult:
    command_name: str
    ok: bool
    status: str
    message: str
    requested_at_utc: str
    completed_at_utc: str
    error_code: str | None = None
    warnings: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    safety_state: OperatorControlState | None = None
    audit_record_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in OPERATOR_CONTROL_RESULT_STATUSES:
            raise ValueError(
                "status must be one of "
                f"{OPERATOR_CONTROL_RESULT_STATUSES}, got '{self.status}'."
            )
        coerce_iso8601_datetime_text(self.requested_at_utc, "requested_at_utc")
        coerce_iso8601_datetime_text(self.completed_at_utc, "completed_at_utc")
        if not self.command_name.strip():
            raise ValueError("command_name cannot be empty.")
        if not self.message.strip():
            raise ValueError("message cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_name": self.command_name,
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "requested_at_utc": self.requested_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "error_code": self.error_code,
            "warnings": list(self.warnings),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "data": dict(self.data),
            "safety_state": (
                self.safety_state.to_dict() if self.safety_state is not None else None
            ),
            "audit_record_id": self.audit_record_id,
        }


class OperatorControlService:
    """Typed orchestration layer for operator-side backend mutations."""

    def __init__(
        self,
        *,
        project_root: Path,
        env_file: Path | None = None,
        broker_adapter_factory: BrokerAdapterFactory | None = None,
        now_utc: Callable[[], datetime] | None = None,
    ) -> None:
        self.project_root = project_root
        self.env_file = env_file
        self._broker_adapter_factory = broker_adapter_factory or create_broker_execution_adapter
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self._safety_store = OperatorControlStateStore(project_root)
        self._pending_order_store = PendingOrderStateStore(project_root)
        self._trade_feedback_store = TradeFeedbackLogStore(project_root)

    @property
    def audit_log_path(self) -> Path:
        return default_operator_control_audit_log_path(self.project_root)

    def safety_state(self) -> OperatorControlState:
        return _load_operator_control_state_result(self._safety_store.path).state

    def pause_execution(self, command: PauseExecutionCommand) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="pause_execution",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=load_result.state,
                warnings=load_result.warnings,
            )
        current_state = load_result.state
        if not current_state.execution_submission_enabled:
            return self._finalize_result(
                command_name="pause_execution",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="no_op",
                ok=True,
                message="Execution submissions are already paused.",
                warnings=load_result.warnings,
                safety_state=current_state,
            )
        updated_state = replace(
            current_state,
            updated_at_utc=requested_at,
            execution_submission_enabled=False,
            paused_reason=command.reason,
            last_pause_at_utc=requested_at,
        )
        state_path = self._safety_store.save(updated_state)
        return self._finalize_result(
            command_name="pause_execution",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message="Execution submissions paused.",
            warnings=load_result.warnings,
            safety_state=updated_state,
            artifact_paths={"operator_control_state": str(state_path.resolve())},
        )

    def resume_execution(self, command: ResumeExecutionCommand) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="resume_execution",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=load_result.state,
                warnings=load_result.warnings,
            )
        current_state = load_result.state
        if current_state.execution_submission_enabled:
            return self._finalize_result(
                command_name="resume_execution",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="no_op",
                ok=True,
                message="Execution submissions are already enabled.",
                safety_state=current_state,
            )
        updated_state = replace(
            current_state,
            updated_at_utc=requested_at,
            execution_submission_enabled=True,
            paused_reason=None,
            last_resume_at_utc=requested_at,
        )
        state_path = self._safety_store.save(updated_state)
        return self._finalize_result(
            command_name="resume_execution",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message="Execution submissions resumed.",
            safety_state=updated_state,
            artifact_paths={"operator_control_state": str(state_path.resolve())},
        )

    def set_execution_mode(
        self,
        command: SetExecutionModeCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="set_execution_mode",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=load_result.state,
                warnings=load_result.warnings,
            )
        current_state = load_result.state
        if current_state.execution_mode == command.execution_mode:
            return self._finalize_result(
                command_name="set_execution_mode",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="no_op",
                ok=True,
                message=(
                    f"Execution mode is already '{command.execution_mode}'."
                ),
                safety_state=current_state,
            )
        updated_state = replace(
            current_state,
            updated_at_utc=requested_at,
            execution_mode=command.execution_mode,
        )
        state_path = self._safety_store.save(updated_state)
        return self._finalize_result(
            command_name="set_execution_mode",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message=f"Execution mode set to '{command.execution_mode}'.",
            safety_state=updated_state,
            artifact_paths={"operator_control_state": str(state_path.resolve())},
        )

    def set_live_action_confirmation(
        self,
        command: SetLiveActionConfirmationCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="set_live_action_confirmation",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=load_result.state,
                warnings=load_result.warnings,
            )
        current_state = load_result.state
        if current_state.live_actions_require_confirmation == command.required:
            return self._finalize_result(
                command_name="set_live_action_confirmation",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="no_op",
                ok=True,
                message=(
                    "Live action confirmation is already "
                    f"{'enabled' if command.required else 'disabled'}."
                ),
                safety_state=current_state,
            )
        updated_state = replace(
            current_state,
            updated_at_utc=requested_at,
            live_actions_require_confirmation=command.required,
        )
        state_path = self._safety_store.save(updated_state)
        return self._finalize_result(
            command_name="set_live_action_confirmation",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message=(
                "Live action confirmation "
                f"{'enabled' if command.required else 'disabled'}."
            ),
            safety_state=updated_state,
            artifact_paths={"operator_control_state": str(state_path.resolve())},
        )

    def set_broker_trading_enabled(
        self,
        command: SetBrokerTradingEnabledCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="set_broker_trading_enabled",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=load_result.state,
                warnings=load_result.warnings,
            )
        current_state = load_result.state
        if current_state.broker_trading_enabled == command.enabled:
            return self._finalize_result(
                command_name="set_broker_trading_enabled",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="no_op",
                ok=True,
                message=(
                    "Broker trading is already "
                    f"{'enabled' if command.enabled else 'disabled'}."
                ),
                safety_state=current_state,
            )
        updated_state = replace(
            current_state,
            updated_at_utc=requested_at,
            broker_trading_enabled=command.enabled,
        )
        state_path = self._safety_store.save(updated_state)
        return self._finalize_result(
            command_name="set_broker_trading_enabled",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message=(
                "Broker trading "
                f"{'enabled' if command.enabled else 'disabled'}."
            ),
            safety_state=updated_state,
            artifact_paths={"operator_control_state": str(state_path.resolve())},
        )

    def submit_pending_order(
        self,
        command: SubmitPendingOrderCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        safety_state = load_result.state
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=safety_state,
                warnings=load_result.warnings,
            )
        starting_state = self._pending_order_store.load()
        record = starting_state.orders.get(command.order_id)
        if record is None:
            return self._finalize_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Unknown pending order '{command.order_id}'.",
                error_code="pending_order_not_found",
                safety_state=safety_state,
            )
        if not record.active:
            return self._finalize_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' is not active.",
                error_code="pending_order_not_active",
                safety_state=safety_state,
            )
        if record.broker_order_id is not None and (record.broker_status or "") not in {
            "cancelled",
            "expired",
            "replaced",
            "rejected",
        }:
            return self._finalize_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' is already linked to broker order '{record.broker_order_id}'.",
                error_code="pending_order_already_submitted",
                safety_state=safety_state,
            )
        policy_rejection = self._reject_if_broker_mutation_not_allowed(
            command_name="submit_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            safety_state=safety_state,
            mutation_kind="submit",
            confirm_live_action=command.confirm_live_action,
            action_label="submission",
        )
        if policy_rejection is not None:
            return policy_rejection
        try:
            adapter = self._create_broker_adapter_for_policy(
                command.broker,
                safety_state=safety_state,
            )
        except (BrokerExecutionError, TypeError) as exc:
            return self._finalize_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message=f"Failed to initialize broker adapter: {exc}",
                error_code="broker_unavailable",
                safety_state=safety_state,
            )
        submission_result = submit_pending_orders_via_broker(
            starting_state,
            adapter=adapter,
            order_ids=(command.order_id,),
        )
        pending_order_state_path = _save_pending_order_state_if_changed(
            self._pending_order_store,
            previous_state=starting_state,
            updated_state=submission_result.state,
        )
        feedback_events = _broker_submission_feedback_events(
            workflow="control-submit-pending-order",
            state=submission_result.state,
            submitted_updates=submission_result.submitted_updates,
            lifecycle_events=submission_result.lifecycle_events,
        )
        trade_feedback_log_path = _append_trade_feedback_events_safe(
            self._trade_feedback_store,
            feedback_events,
        )
        if not submission_result.submitted_updates:
            return self._finalize_result(
                command_name="submit_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message=f"Broker submission failed for pending order '{command.order_id}'.",
                error_code="broker_submit_failed",
                warnings=submission_result.warnings,
                safety_state=safety_state,
                artifact_paths=_artifact_paths(
                    pending_order_state=pending_order_state_path,
                    trade_feedback_log=trade_feedback_log_path,
                ),
                data={
                    "submitted_count": 0,
                    "submitted_updates": [],
                },
            )
        return self._finalize_result(
            command_name="submit_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message=f"Submitted pending order '{command.order_id}' to broker '{command.broker}'.",
            warnings=submission_result.warnings,
            safety_state=safety_state,
            artifact_paths=_artifact_paths(
                pending_order_state=pending_order_state_path,
                trade_feedback_log=trade_feedback_log_path,
            ),
            data={
                "submitted_count": len(submission_result.submitted_updates),
                "submitted_updates": [
                    update.to_dict() for update in submission_result.submitted_updates
                ],
            },
        )

    def cancel_pending_order(
        self,
        command: CancelPendingOrderCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        safety_state = load_result.state
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="cancel_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=safety_state,
                warnings=load_result.warnings,
            )
        starting_state = self._pending_order_store.load()
        record = starting_state.orders.get(command.order_id)
        if record is None:
            return self._finalize_result(
                command_name="cancel_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Unknown pending order '{command.order_id}'.",
                error_code="pending_order_not_found",
                safety_state=safety_state,
            )
        if not record.active:
            return self._finalize_result(
                command_name="cancel_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' is not active.",
                error_code="pending_order_not_active",
                safety_state=safety_state,
            )
        if record.broker_order_id is None:
            return self._finalize_result(
                command_name="cancel_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' does not have a broker_order_id yet.",
                error_code="pending_order_missing_broker_id",
                safety_state=safety_state,
            )
        policy_rejection = self._reject_if_broker_mutation_not_allowed(
            command_name="cancel_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            safety_state=safety_state,
            mutation_kind="cancel",
            confirm_live_action=command.confirm_live_action,
            action_label="cancel",
        )
        if policy_rejection is not None:
            return policy_rejection
        try:
            adapter = self._create_broker_adapter_for_policy(
                command.broker,
                safety_state=safety_state,
            )
            update = adapter.cancel_order(record.broker_order_id)
        except (BrokerExecutionError, TypeError) as exc:
            return self._finalize_result(
                command_name="cancel_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message=f"Broker cancel failed for pending order '{command.order_id}': {exc}",
                error_code="broker_cancel_failed",
                safety_state=safety_state,
            )
        warnings: list[str] = []
        if update is None:
            occurred_at = requested_at
            warnings.append(
                f"Broker cancel request for '{command.order_id}' succeeded but no broker order snapshot was returned."
            )
            updated_state = _mark_pending_order_broker_cancel_pending(
                starting_state,
                order_id=command.order_id,
                occurred_at=occurred_at,
            )
            lifecycle_events: tuple[Any, ...] = ()
            matched_updates: tuple[BrokerOrderUpdate, ...] = ()
        else:
            reconciliation = reconcile_broker_order_updates(starting_state, (update,))
            updated_state = reconciliation.state
            warnings.extend(reconciliation.warnings)
            lifecycle_events = reconciliation.lifecycle_events
            matched_updates = reconciliation.matched_updates
        pending_order_state_path = _save_pending_order_state_if_changed(
            self._pending_order_store,
            previous_state=starting_state,
            updated_state=updated_state,
        )
        feedback_events = _broker_reconciliation_feedback_events(
            workflow="control-cancel-pending-order",
            state=updated_state,
            lifecycle_events=lifecycle_events,
            matched_updates=matched_updates,
        )
        trade_feedback_log_path = _append_trade_feedback_events_safe(
            self._trade_feedback_store,
            feedback_events,
        )
        return self._finalize_result(
            command_name="cancel_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="needs_confirmation" if update is None else "completed",
            ok=True,
            message=(
                f"Cancel requested for pending order '{command.order_id}'; awaiting broker confirmation."
                if update is None
                else f"Processed broker cancel for pending order '{command.order_id}'."
            ),
            warnings=tuple(warnings),
            safety_state=safety_state,
            artifact_paths=_artifact_paths(
                pending_order_state=pending_order_state_path,
                trade_feedback_log=trade_feedback_log_path,
            ),
            data={
                "broker_update": update.to_dict() if update is not None else None,
                "cancel_confirmed": update is not None,
                "pending_order_status": updated_state.orders[command.order_id].status,
                "broker_status": updated_state.orders[command.order_id].broker_status,
                "reservation_active": updated_state.orders[command.order_id].reserves_capacity,
                "transitional_state": (
                    updated_state.orders[command.order_id].broker_status
                    if update is None
                    else None
                ),
            },
        )

    def replace_pending_order(
        self,
        command: ReplacePendingOrderCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        safety_state = load_result.state
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=safety_state,
                warnings=load_result.warnings,
            )
        starting_state = self._pending_order_store.load()
        existing = starting_state.orders.get(command.order_id)
        if existing is None:
            return self._finalize_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Unknown pending order '{command.order_id}'.",
                error_code="pending_order_not_found",
                safety_state=safety_state,
            )
        if not existing.active:
            return self._finalize_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' is not active.",
                error_code="pending_order_not_active",
                safety_state=safety_state,
            )
        if existing.broker_status == "pending_cancel":
            return self._finalize_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' is awaiting broker cancel confirmation.",
                error_code="pending_order_cancel_in_progress",
                safety_state=safety_state,
            )
        if existing.broker_order_id is None:
            return self._finalize_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message=f"Pending order '{command.order_id}' does not have a broker_order_id yet.",
                error_code="pending_order_missing_broker_id",
                safety_state=safety_state,
            )
        policy_rejection = self._reject_if_broker_mutation_not_allowed(
            command_name="replace_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            safety_state=safety_state,
            mutation_kind="replace",
            confirm_live_action=command.confirm_live_action,
            action_label="replacement",
        )
        if policy_rejection is not None:
            return policy_rejection
        replacement_local_order_id = _replacement_pending_order_id(
            existing,
            quantity=command.quantity,
            limit_price=command.limit_price,
            stop_price=command.stop_price,
        )
        try:
            adapter = self._create_broker_adapter_for_policy(
                command.broker,
                safety_state=safety_state,
            )
            replacement_update = adapter.replace_order(
                existing.broker_order_id,
                BrokerReplaceOrderRequest(
                    quantity=command.quantity,
                    time_in_force="DAY",
                    client_order_id=replacement_local_order_id,
                    limit_price=command.limit_price,
                    stop_price=command.stop_price,
                ),
            )
        except (BrokerExecutionError, TypeError, ValueError) as exc:
            return self._finalize_result(
                command_name="replace_pending_order",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message=f"Broker replace failed for pending order '{command.order_id}': {exc}",
                error_code="broker_replace_failed",
                safety_state=safety_state,
            )
        occurred_at = (
            replacement_update.updated_at
            or replacement_update.submitted_at
            or requested_at
        )
        replacement_record = _build_replacement_pending_order_record(
            existing,
            local_order_id=replacement_local_order_id,
            occurred_at=occurred_at,
            quantity=command.quantity,
            limit_price=command.limit_price,
            stop_price=command.stop_price,
        )
        replace_result = replace_pending_order(
            starting_state,
            existing.order_id,
            replacement=replacement_record,
            occurred_at=occurred_at,
        )
        reconciliation = reconcile_broker_order_updates(
            replace_result.state,
            (replacement_update,),
        )
        updated_state = reconciliation.state
        pending_order_state_path = _save_pending_order_state_if_changed(
            self._pending_order_store,
            previous_state=starting_state,
            updated_state=updated_state,
        )
        feedback_events = _broker_replacement_feedback_events(
            state=updated_state,
            replacement_update=replacement_update,
            replacement_order_id=replacement_local_order_id,
        )
        trade_feedback_log_path = _append_trade_feedback_events_safe(
            self._trade_feedback_store,
            feedback_events,
        )
        return self._finalize_result(
            command_name="replace_pending_order",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed",
            ok=True,
            message=f"Replaced pending order '{command.order_id}' with '{replacement_local_order_id}'.",
            warnings=reconciliation.warnings,
            safety_state=safety_state,
            artifact_paths=_artifact_paths(
                pending_order_state=pending_order_state_path,
                trade_feedback_log=trade_feedback_log_path,
            ),
            data={
                "replacement_order_id": replacement_local_order_id,
                "broker_update": replacement_update.to_dict(),
            },
        )

    def force_broker_sync(
        self,
        command: ForceBrokerSyncCommand,
    ) -> OperatorCommandResult:
        requested_at = _utc_now_isoformat(self._now_utc())
        load_result = self._load_safety_state_result()
        safety_state = load_result.state
        if load_result.degraded:
            return self._control_state_unavailable_result(
                command_name="force_broker_sync",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                safety_state=safety_state,
                warnings=load_result.warnings,
            )
        starting_state = self._pending_order_store.load()
        selected_orders = _select_pending_orders_for_broker_sync(
            starting_state,
            order_ids=command.order_ids,
        )
        if command.order_ids and not selected_orders:
            return self._finalize_result(
                command_name="force_broker_sync",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="rejected",
                ok=False,
                message="None of the requested order_ids were found in active local state.",
                error_code="pending_order_not_found",
                safety_state=safety_state,
            )
        try:
            adapter = self._create_broker_adapter_for_policy(
                command.broker,
                safety_state=safety_state,
            )
        except (BrokerExecutionError, TypeError) as exc:
            return self._finalize_result(
                command_name="force_broker_sync",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message=f"Failed to initialize broker adapter: {exc}",
                error_code="broker_unavailable",
                safety_state=safety_state,
            )
        broker_updates, broker_warnings = _load_broker_updates_for_pending_orders(
            adapter,
            selected_orders,
            use_open_order_scan=not bool(command.order_ids),
        )
        reconciliation = reconcile_broker_order_updates(starting_state, broker_updates)
        updated_state = reconciliation.state
        pending_order_state_path = _save_pending_order_state_if_changed(
            self._pending_order_store,
            previous_state=starting_state,
            updated_state=updated_state,
        )
        warnings = tuple(list(broker_warnings) + list(reconciliation.warnings))
        feedback_events = _broker_reconciliation_feedback_events(
            workflow="control-force-broker-sync",
            state=updated_state,
            lifecycle_events=reconciliation.lifecycle_events,
            matched_updates=reconciliation.matched_updates,
        )
        trade_feedback_log_path = _append_trade_feedback_events_safe(
            self._trade_feedback_store,
            feedback_events,
        )
        if not broker_updates and broker_warnings:
            return self._finalize_result(
                command_name="force_broker_sync",
                command_payload=command.to_dict(),
                operator_id=command.operator_id,
                requested_at_utc=requested_at,
                status="failed",
                ok=False,
                message="Broker sync could not load updates.",
                error_code="broker_sync_failed",
                warnings=warnings,
                safety_state=safety_state,
                artifact_paths=_artifact_paths(
                    pending_order_state=pending_order_state_path,
                    trade_feedback_log=trade_feedback_log_path,
                ),
                data={
                    "matched_update_count": len(reconciliation.matched_updates),
                    "unmatched_update_count": len(reconciliation.unmatched_updates),
                },
            )
        return self._finalize_result(
            command_name="force_broker_sync",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="completed" if broker_updates or selected_orders else "no_op",
            ok=True,
            message=(
                "Broker sync completed."
                if broker_updates or selected_orders
                else "No active pending orders were eligible for broker sync."
            ),
            warnings=warnings,
            safety_state=safety_state,
            artifact_paths=_artifact_paths(
                pending_order_state=pending_order_state_path,
                trade_feedback_log=trade_feedback_log_path,
            ),
            data={
                "selected_order_count": len(selected_orders),
                "matched_update_count": len(reconciliation.matched_updates),
                "unmatched_update_count": len(reconciliation.unmatched_updates),
                "matched_updates": [update.to_dict() for update in reconciliation.matched_updates],
                "unmatched_updates": [
                    update.to_dict() for update in reconciliation.unmatched_updates
                ],
            },
        )

    def control_index(self) -> dict[str, Any]:
        return {
            "service": "investopedia-bot-operator-control-api",
            "version": OPERATOR_CONTROL_API_VERSION,
            "project_root": str(self.project_root.resolve()),
            "endpoints": list(OPERATOR_CONTROL_HTTP_ENDPOINTS),
        }

    def safety(self) -> dict[str, Any]:
        return self._policy_payload(endpoint="control_safety")

    def execution_policy(self) -> dict[str, Any]:
        return self._policy_payload(endpoint="control_execution_policy")

    def _policy_payload(self, *, endpoint: str) -> dict[str, Any]:
        load_result = self._load_safety_state_result()
        state = load_result.state
        return {
            "endpoint": endpoint,
            "available": True,
            "not_available_reason": None,
            "artifact_paths": {
                "operator_control_state": str(self._safety_store.path.resolve()),
                "operator_control_audit_log": str(self.audit_log_path.resolve()),
            },
            "warnings": list(load_result.warnings),
            "data": {
                "control_state_readable": not load_result.degraded,
                "safety_state": state.to_dict(),
                "policy_summary": _execution_policy_summary(state),
            },
        }

    def _load_safety_state_result(self) -> _OperatorControlStateLoadResult:
        return _load_operator_control_state_result(self._safety_store.path)

    def _create_broker_adapter_for_policy(
        self,
        broker_name: str,
        *,
        safety_state: OperatorControlState,
    ) -> BrokerExecutionAdapter:
        return _call_broker_adapter_factory(
            self._broker_adapter_factory,
            broker_name,
            env_file=self.env_file,
            target=safety_state.execution_mode,
        )

    def _reject_if_broker_mutation_not_allowed(
        self,
        *,
        command_name: str,
        command_payload: Mapping[str, Any],
        operator_id: str | None,
        requested_at_utc: str,
        safety_state: OperatorControlState,
        mutation_kind: str,
        confirm_live_action: bool,
        action_label: str,
    ) -> OperatorCommandResult | None:
        if _pause_blocks_broker_mutation(mutation_kind) and (
            not safety_state.execution_submission_enabled
        ):
            return self._finalize_result(
                command_name=command_name,
                command_payload=command_payload,
                operator_id=operator_id,
                requested_at_utc=requested_at_utc,
                status="rejected",
                ok=False,
                message="Execution submissions are paused.",
                error_code="execution_paused",
                safety_state=safety_state,
            )
        if safety_state.execution_mode != "live":
            return None
        if not safety_state.broker_trading_enabled:
            return self._finalize_result(
                command_name=command_name,
                command_payload=command_payload,
                operator_id=operator_id,
                requested_at_utc=requested_at_utc,
                status="rejected",
                ok=False,
                message="Live broker trading is disabled by execution policy.",
                error_code="live_broker_trading_disabled",
                safety_state=safety_state,
            )
        # This is an operator intent assertion carried by the caller UI/CLI,
        # not a stronger proof of confirmation.
        if safety_state.live_actions_require_confirmation and not confirm_live_action:
            return self._finalize_result(
                command_name=command_name,
                command_payload=command_payload,
                operator_id=operator_id,
                requested_at_utc=requested_at_utc,
                status="rejected",
                ok=False,
                message=(
                    f"Live {action_label} requires confirm_live_action=true "
                    "while live confirmation is enabled."
                ),
                error_code="live_confirmation_required",
                safety_state=safety_state,
            )
        return None

    def _control_state_unavailable_result(
        self,
        *,
        command_name: str,
        command_payload: Mapping[str, Any],
        operator_id: str | None,
        requested_at_utc: str,
        safety_state: OperatorControlState,
        warnings: Sequence[str],
    ) -> OperatorCommandResult:
        return self._finalize_result(
            command_name=command_name,
            command_payload=command_payload,
            operator_id=operator_id,
            requested_at_utc=requested_at_utc,
            status="failed",
            ok=False,
            message=(
                "Operator control safety state is unreadable; control mutations are "
                "blocked until the state file is repaired."
            ),
            error_code="control_state_unavailable",
            warnings=warnings,
            safety_state=safety_state,
            artifact_paths={
                "operator_control_state": str(self._safety_store.path.resolve())
            },
        )

    def _finalize_result(
        self,
        *,
        command_name: str,
        command_payload: Mapping[str, Any],
        operator_id: str | None,
        requested_at_utc: str,
        status: str,
        ok: bool,
        message: str,
        error_code: str | None = None,
        warnings: Sequence[str] = (),
        artifact_paths: Mapping[str, str] | None = None,
        data: Mapping[str, Any] | None = None,
        safety_state: OperatorControlState | None = None,
    ) -> OperatorCommandResult:
        completed_at_utc = _utc_now_isoformat(self._now_utc())
        result = OperatorCommandResult(
            command_name=command_name,
            ok=ok,
            status=status,
            message=message,
            requested_at_utc=requested_at_utc,
            completed_at_utc=completed_at_utc,
            error_code=error_code,
            warnings=tuple(warnings),
            artifact_paths=dict(artifact_paths or {}),
            data=dict(data or {}),
            safety_state=safety_state,
        )
        audit_record = OperatorControlAuditRecord(
            command_name=command_name,
            requested_at_utc=requested_at_utc,
            status=status,
            message=message,
            operator_id=operator_id,
            error_code=error_code,
            warnings=tuple(warnings),
            command_payload=dict(command_payload),
            result_payload=result.to_dict(),
        )
        audit_path = self.audit_log_path
        try:
            append_operator_control_audit_record(audit_record, audit_path)
            artifact_paths_with_audit = dict(result.artifact_paths)
            artifact_paths_with_audit["operator_control_audit_log"] = str(audit_path.resolve())
            return replace(
                result,
                artifact_paths=artifact_paths_with_audit,
                audit_record_id=audit_record.audit_record_id,
            )
        except OSError as exc:
            warning_message = (
                f"Failed to append operator control audit log at "
                f"{audit_path.resolve()}: {exc}"
            )
            LOGGER.warning(warning_message)
            return replace(
                result,
                warnings=result.warnings + (warning_message,),
            )


class OperatorControlHttpServer(ThreadingHTTPServer):
    """Small HTTP server for operator control commands."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        control_service: OperatorControlService,
    ) -> None:
        self.control_service = control_service
        super().__init__(server_address, OperatorControlRequestHandler)


class OperatorControlRequestHandler(BaseHTTPRequestHandler):
    """Serve typed operator control commands as JSON."""

    server: OperatorControlHttpServer

    def do_GET(self) -> None:
        status, payload = operator_control_response_for_request(
            control_service=self.server.control_service,
            method="GET",
            path=self.path,
            body=None,
        )
        self._write_json(status, payload)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "invalid_content_length",
                    "message": "Content-Length must be an integer.",
                },
            )
            return
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        status, payload = operator_control_response_for_request(
            control_service=self.server.control_service,
            method="POST",
            path=self.path,
            body=raw_body,
        )
        self._write_json(status, payload)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("operator-control %s", format % args)

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_operator_control_server(
    *,
    control_service: OperatorControlService,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> OperatorControlHttpServer:
    return OperatorControlHttpServer((host, port), control_service)


def serve_operator_control_api(
    *,
    control_service: OperatorControlService,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> None:
    server = create_operator_control_server(
        control_service=control_service,
        host=host,
        port=port,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def operator_control_response_for_request(
    *,
    control_service: OperatorControlService,
    method: str,
    path: str,
    body: bytes | None,
) -> tuple[HTTPStatus, dict[str, Any]]:
    normalized_path = urlparse(path).path.rstrip("/") or "/"
    if method == "GET":
        routes: dict[str, Callable[[], dict[str, Any]]] = {
            "/control": control_service.control_index,
            "/control/safety": control_service.safety,
            "/control/execution/policy": control_service.execution_policy,
        }
        handler = routes.get(normalized_path)
        if handler is None:
            return HTTPStatus.NOT_FOUND, {
                "error": "not_found",
                "path": normalized_path,
                "endpoints": list(OPERATOR_CONTROL_HTTP_ENDPOINTS),
            }
        try:
            return HTTPStatus.OK, handler()
        except Exception as exc:
            LOGGER.exception(
                "Unhandled operator control GET failure for %s: %s",
                normalized_path,
                exc,
            )
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "internal_error",
                "message": "Operator control request failed unexpectedly.",
            }

    if method != "POST":
        return HTTPStatus.METHOD_NOT_ALLOWED, {
            "error": "method_not_allowed",
            "method": method,
            "allowed_methods": ["GET", "POST"],
        }

    try:
        payload = _decode_json_mapping(body or b"")
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {
            "error": "invalid_json",
            "message": str(exc),
        }

    try:
        if normalized_path == "/control/execution/pause":
            result = control_service.pause_execution(
                PauseExecutionCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/execution/resume":
            result = control_service.resume_execution(
                ResumeExecutionCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/execution/mode":
            result = control_service.set_execution_mode(
                SetExecutionModeCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/execution/live-confirmation":
            result = control_service.set_live_action_confirmation(
                SetLiveActionConfirmationCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/execution/broker-trading":
            result = control_service.set_broker_trading_enabled(
                SetBrokerTradingEnabledCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/orders/submit":
            result = control_service.submit_pending_order(
                SubmitPendingOrderCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/orders/cancel":
            result = control_service.cancel_pending_order(
                CancelPendingOrderCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/orders/replace":
            result = control_service.replace_pending_order(
                ReplacePendingOrderCommand.from_mapping(payload)
            )
        elif normalized_path == "/control/broker/sync":
            result = control_service.force_broker_sync(
                ForceBrokerSyncCommand.from_mapping(payload)
            )
        else:
            return HTTPStatus.NOT_FOUND, {
                "error": "not_found",
                "path": normalized_path,
                "endpoints": list(OPERATOR_CONTROL_HTTP_ENDPOINTS),
            }
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {
            "error": "invalid_command",
            "message": str(exc),
        }
    except Exception as exc:
        LOGGER.exception(
            "Unhandled operator control POST failure for %s: %s",
            normalized_path,
            exc,
        )
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "error": "internal_error",
            "message": "Operator control request failed unexpectedly.",
        }

    return _operator_command_http_status(result), result.to_dict()


def _operator_command_http_status(result: OperatorCommandResult) -> HTTPStatus:
    if result.status in {"completed", "no_op", "needs_confirmation"}:
        return HTTPStatus.OK
    if result.status == "rejected":
        return HTTPStatus.CONFLICT
    if result.error_code is not None and result.error_code.startswith("broker_"):
        return HTTPStatus.BAD_GATEWAY
    return HTTPStatus.INTERNAL_SERVER_ERROR


def _save_pending_order_state_if_changed(
    store: PendingOrderStateStore,
    *,
    previous_state: PendingOrderState,
    updated_state: PendingOrderState,
) -> Path | None:
    if updated_state == previous_state:
        return None
    return store.save(updated_state)


def _mark_pending_order_broker_cancel_pending(
    state: PendingOrderState,
    *,
    order_id: str,
    occurred_at: str,
) -> PendingOrderState:
    record = state.orders.get(order_id)
    if record is None:
        raise ValueError(f"Unknown pending order_id '{order_id}'.")
    updated_record = replace(
        record,
        broker_status="pending_cancel",
        last_broker_update_at=occurred_at,
    )
    if updated_record == record:
        return state
    return upsert_pending_order_state(
        state,
        updated_record,
        updated_at=occurred_at,
    )


def _select_pending_orders_for_broker_sync(
    state: PendingOrderState,
    *,
    order_ids: Sequence[str] | None,
) -> tuple[PendingOrderRecord, ...]:
    requested_ids = {order_id.strip() for order_id in (order_ids or ()) if order_id.strip()}
    selected = [
        record
        for record in state.active_orders()
        if not requested_ids or record.order_id in requested_ids
    ]
    return tuple(selected)


def _broker_update_identity_key(update: BrokerOrderUpdate) -> tuple[str, str]:
    return (update.broker_order_id, update.client_order_id or "")


def _pending_order_matches_broker_update(
    record: PendingOrderRecord,
    update: BrokerOrderUpdate,
) -> bool:
    if record.broker_order_id is not None and record.broker_order_id == update.broker_order_id:
        return True
    candidate_client_order_id = record.client_order_id or record.order_id
    if update.client_order_id is not None and candidate_client_order_id == update.client_order_id:
        return True
    return False


def _load_broker_updates_for_pending_orders(
    adapter: BrokerExecutionAdapter,
    orders: Sequence[PendingOrderRecord],
    *,
    use_open_order_scan: bool = False,
) -> tuple[tuple[BrokerOrderUpdate, ...], tuple[str, ...]]:
    updates_by_identity: dict[tuple[str, str], BrokerOrderUpdate] = {}
    warnings: list[str] = []
    broker_open_updates: tuple[BrokerOrderUpdate, ...] = ()
    if use_open_order_scan:
        try:
            broker_open_updates = adapter.list_open_orders()
        except BrokerExecutionError as exc:
            warnings.append(f"Broker open-order scan unavailable: {exc}")
        else:
            for update in broker_open_updates:
                updates_by_identity[_broker_update_identity_key(update)] = update
    for record in orders:
        if use_open_order_scan and any(
            _pending_order_matches_broker_update(record, update)
            for update in broker_open_updates
        ):
            continue
        try:
            if record.broker_order_id is not None:
                update = adapter.get_order(record.broker_order_id)
            else:
                update = adapter.get_order(client_order_id=record.client_order_id or record.order_id)
            updates_by_identity[_broker_update_identity_key(update)] = update
        except BrokerExecutionError as exc:
            warnings.append(
                f"Broker update unavailable for local pending order '{record.order_id}': {exc}"
            )
    return tuple(updates_by_identity.values()), tuple(warnings)


def _replacement_pending_order_id(
    record: PendingOrderRecord,
    *,
    quantity: int | None,
    limit_price: float | None,
    stop_price: float | None,
) -> str:
    identity_payload = {
        "existing_order_id": record.order_id,
        "quantity": quantity,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "broker_order_id": record.broker_order_id,
    }
    serialized = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"order_{digest}"


def _build_replacement_pending_order_record(
    existing: PendingOrderRecord,
    *,
    local_order_id: str,
    occurred_at: str,
    quantity: int | None,
    limit_price: float | None,
    stop_price: float | None,
) -> PendingOrderRecord:
    requested_quantity = quantity if quantity is not None else existing.requested_quantity
    if requested_quantity <= existing.filled_quantity:
        raise ValueError(
            "Replacement quantity must exceed the already filled quantity for the order."
        )
    requested_price = (
        limit_price
        if existing.order_type in {"LIMIT", "STOP_LIMIT"} and limit_price is not None
        else existing.requested_price
    )
    resolved_entry_hint = requested_price or existing.entry_hint
    remaining_quantity = requested_quantity - existing.filled_quantity
    reservation_price = requested_price or existing.entry_hint or 0.0
    return PendingOrderRecord(
        order_id=local_order_id,
        symbol=existing.symbol,
        side=existing.side,
        order_type=existing.order_type,
        created_at=occurred_at,
        status="pending",
        requested_quantity=requested_quantity,
        filled_quantity=existing.filled_quantity,
        requested_price=requested_price,
        entry_hint=resolved_entry_hint,
        stop_price=stop_price if stop_price is not None else existing.stop_price,
        reserved_notional=(
            remaining_quantity * reservation_price
            if existing.side == "BUY"
            else 0.0
        ),
        reserved_slot=existing.side == "BUY" and remaining_quantity > 0,
        preset_name=existing.preset_name,
        trade_id=existing.trade_id,
        linked_decision_id=existing.linked_decision_id,
        sector_name=existing.sector_name,
        industry_name=existing.industry_name,
        client_order_id=local_order_id,
        metadata=dict(existing.metadata),
    )


def _append_trade_feedback_events_safe(
    store: TradeFeedbackLogStore,
    events: Sequence[TradeFeedbackEvent],
) -> Path | None:
    if not events:
        return None
    log_path = store.path.resolve()
    try:
        return store.append(events)
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Skipping trade feedback log update at %s: %s", log_path, exc)
        return None


def _broker_submission_feedback_events(
    *,
    workflow: str,
    state: PendingOrderState,
    submitted_updates: Sequence[BrokerOrderUpdate],
    lifecycle_events: Sequence[Any],
) -> tuple[TradeFeedbackEvent, ...]:
    events: list[TradeFeedbackEvent] = []
    for update in submitted_updates:
        record = _find_pending_order_record_for_broker_update(state, update)
        if record is None:
            continue
        timestamp_utc = (
            update.submitted_at
            or update.updated_at
            or record.last_broker_update_at
            or record.created_at
        )
        event = _broker_trade_feedback_event(
            event_type="broker_submitted",
            workflow=workflow,
            record=record,
            timestamp_utc=timestamp_utc,
        )
        if event is not None:
            events.append(event)
    events.extend(
        _broker_reconciliation_feedback_events(
            workflow=workflow,
            state=state,
            lifecycle_events=lifecycle_events,
            matched_updates=submitted_updates,
        )
    )
    return tuple(events)


def _broker_reconciliation_feedback_events(
    *,
    workflow: str,
    state: PendingOrderState,
    lifecycle_events: Sequence[Any],
    matched_updates: Sequence[BrokerOrderUpdate],
) -> tuple[TradeFeedbackEvent, ...]:
    updates_by_order_id: dict[str, BrokerOrderUpdate] = {}
    for update in matched_updates:
        record = _find_pending_order_record_for_broker_update(state, update)
        if record is not None:
            updates_by_order_id[record.order_id] = update
    events: list[TradeFeedbackEvent] = []
    for lifecycle_event in lifecycle_events:
        order_id = getattr(lifecycle_event, "order_id", None)
        if not isinstance(order_id, str):
            continue
        record = state.orders.get(order_id)
        if record is None:
            continue
        update = updates_by_order_id.get(order_id)
        event_type = _broker_feedback_event_type_for_lifecycle(
            getattr(lifecycle_event, "event_type", None)
        )
        if event_type is None:
            continue
        timestamp_utc = (
            record.last_broker_update_at
            or (update.updated_at if update is not None else None)
            or (update.submitted_at if update is not None else None)
            or getattr(lifecycle_event, "occurred_at", None)
            or record.created_at
        )
        event = _broker_trade_feedback_event(
            event_type=event_type,
            workflow=workflow,
            record=record,
            timestamp_utc=timestamp_utc,
        )
        if event is not None:
            events.append(event)
    return tuple(events)


def _broker_replacement_feedback_events(
    *,
    state: PendingOrderState,
    replacement_update: BrokerOrderUpdate,
    replacement_order_id: str,
) -> tuple[TradeFeedbackEvent, ...]:
    record = state.orders.get(replacement_order_id)
    if record is None:
        return ()
    timestamp_utc = (
        replacement_update.updated_at
        or replacement_update.submitted_at
        or record.last_broker_update_at
        or record.created_at
    )
    event = _broker_trade_feedback_event(
        event_type="broker_replaced",
        workflow="control-replace-pending-order",
        record=record,
        timestamp_utc=timestamp_utc,
    )
    return (event,) if event is not None else ()


def _find_pending_order_record_for_broker_update(
    state: PendingOrderState,
    update: BrokerOrderUpdate,
) -> PendingOrderRecord | None:
    for record in state.orders.values():
        if record.broker_order_id == update.broker_order_id:
            return record
    if update.client_order_id is not None:
        record = state.orders.get(update.client_order_id)
        if record is not None:
            return record
        for existing in state.orders.values():
            if existing.client_order_id == update.client_order_id:
                return existing
    return None


def _broker_feedback_event_type_for_lifecycle(lifecycle_event_type: object) -> str | None:
    if lifecycle_event_type == "filled":
        return "executed"
    if lifecycle_event_type == "cancelled":
        return "broker_cancelled"
    if lifecycle_event_type == "expired":
        return "broker_expired"
    if lifecycle_event_type == "replaced":
        return "broker_replaced"
    return None


def _broker_trade_feedback_event(
    *,
    event_type: str,
    workflow: str,
    record: PendingOrderRecord,
    timestamp_utc: str,
) -> TradeFeedbackEvent | None:
    as_of_date = _broker_event_date(timestamp_utc)
    if as_of_date is None:
        return None
    quantity = record.filled_quantity if event_type == "executed" else record.requested_quantity
    return TradeFeedbackEvent(
        event_type=event_type,
        workflow=workflow,
        symbol=record.symbol,
        as_of_date=as_of_date,
        timestamp_utc=timestamp_utc,
        decision_id=record.linked_decision_id,
        trade_id=record.trade_id,
        linked_decision_id=record.linked_decision_id,
        preset_name=record.preset_name,
        strategy_name=_extract_strategy_name_from_metadata(record.metadata),
        suggested_action=record.side,
        approved=True,
        quantity=quantity,
        entry_price_hint=(
            record.average_fill_price
            if event_type == "executed"
            else record.requested_price or record.entry_hint
        ),
        stop_price=record.stop_price,
        notional_value=(
            quantity
            * (
                record.average_fill_price
                if event_type == "executed"
                else record.requested_price or record.entry_hint or 0.0
            )
            if quantity is not None
            else None
        ),
        metadata={
            "order_id": record.order_id,
            "broker_name": record.broker_name,
            "broker_order_id": record.broker_order_id,
            "client_order_id": record.client_order_id,
            "broker_status": record.broker_status,
        },
    )


def _broker_event_date(timestamp_utc: str | None) -> date | None:
    cleaned = _optional_text(timestamp_utc)
    if cleaned is None:
        return None
    normalized = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _extract_strategy_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    direct = _optional_text(metadata.get("strategy_name"))
    if direct is not None:
        return direct
    signal_metadata = metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        direct = _optional_text(signal_metadata.get("strategy_name"))
        if direct is not None:
            return direct
    execution_metadata = metadata.get("execution_metadata")
    if isinstance(execution_metadata, Mapping):
        direct = _optional_text(execution_metadata.get("strategy_name"))
        if direct is not None:
            return direct
        nested_signal_metadata = execution_metadata.get("signal_metadata")
        if isinstance(nested_signal_metadata, Mapping):
            direct = _optional_text(nested_signal_metadata.get("strategy_name"))
            if direct is not None:
                return direct
    return None


def _operator_control_audit_record_id(record: OperatorControlAuditRecord) -> str:
    payload = {
        "command_name": record.command_name,
        "requested_at_utc": record.requested_at_utc,
        "status": record.status,
        "message": record.message,
        "operator_id": record.operator_id,
        "error_code": record.error_code,
        "command_payload": dict(sorted(record.command_payload.items())),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _artifact_paths(**paths: Path | str | None) -> dict[str, str]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else str(value)
        for key, value in sorted(paths.items())
        if value is not None
    }


def _execution_policy_summary(state: OperatorControlState) -> dict[str, Any]:
    return {
        "execution_mode": state.execution_mode,
        "execution_submission_enabled": state.execution_submission_enabled,
        "live_actions_require_confirmation": state.live_actions_require_confirmation,
        "broker_trading_enabled": state.broker_trading_enabled,
        "is_live_mode": state.execution_mode == "live",
        "live_broker_mutations_allowed": (
            state.execution_mode != "live" or state.broker_trading_enabled
        ),
    }


def _call_broker_adapter_factory(
    factory: BrokerAdapterFactory,
    broker_name: str,
    *,
    env_file: Path | None,
    target: str | None,
) -> BrokerExecutionAdapter:
    return factory(
        broker_name,
        env_file=env_file,
        target=target,
    )


def _pause_blocks_broker_mutation(mutation_kind: str) -> bool:
    # Pause is intentionally a submission stop, not a risk-reduction stop.
    return mutation_kind == "submit"


def _decode_json_mapping(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Request body must be valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Request body must decode to a JSON object.")
    return dict(payload)


def _utc_now_isoformat(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _required_text(value: object, field_name: str) -> str:
    cleaned = _optional_text(value)
    if cleaned is None:
        raise ValueError(f"{field_name} is required.")
    return cleaned


def _mapping_optional_positive_int(payload: Mapping[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if coerced <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return coerced


def _mapping_optional_bool(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: bool,
) -> bool:
    value = payload.get(field_name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _mapping_required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    if field_name not in payload:
        raise ValueError(f"{field_name} is required.")
    return _mapping_optional_bool(payload, field_name, default=False)


def _mapping_optional_positive_float(payload: Mapping[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not a boolean.")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if coerced <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return coerced


def _mapping_text_sequence(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    raw_value = payload.get(field_name)
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        return (cleaned,) if cleaned else ()
    if not isinstance(raw_value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    values = [
        item.strip()
        for item in raw_value
        if isinstance(item, str) and item.strip()
    ]
    return tuple(values)
