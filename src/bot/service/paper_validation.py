"""Paper-mode runtime validation helpers built on the local staging harness."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from bot.api.control_api import (
    OperatorCommandResult,
    OperatorControlService,
    OperatorControlState,
    OperatorControlStateStore,
    SetExecutionModeCommand,
    _OperatorControlStateLoadResult,
    default_operator_control_audit_log_path,
    load_operator_control_audit_records,
)
from bot.api.internal_api import InternalApiQueryService
from bot.config import AppConfig, load_app_config
from bot.data.pending_orders import PendingOrderStateStore
from bot.data.trade_feedback import (
    TradeFeedbackEvent,
    default_trade_feedback_log_path,
    load_trade_feedback_events,
)
from bot.logging_utils import get_logger
from bot.notifications.routing import NotificationDeliveryStateStore
from bot.service.live_market_service import LiveMarketServiceStatusStore
from bot.service.local_staging_runtime import (
    LocalStagingRuntime,
    LocalStagingRuntimeConfig,
    LocalStagingRuntimePaths,
    RuntimeSupervisor,
    create_local_staging_runtime,
)


LOGGER = get_logger(__name__)

PAPER_VALIDATION_PROFILE_NAME = "local-paper-validation"
_PAPER_VALIDATION_JOURNAL_FILENAME = "paper_validation_runtime_journal.jsonl"
_PAPER_VALIDATION_PROFILE_FILENAME = "paper_validation_profile.json"
_PAPER_VALIDATION_REVIEW_DIRNAME = "validation_review"
_PAPER_VALIDATION_RUNTIME_EVENT_TYPES = (
    "runtime_started",
    "runtime_stopped",
    "runtime_restart_requested",
    "runtime_restarted",
    "runtime_logs_rotated",
    "runtime_env_overridden",
    "runtime_transport_disconnected",
    "runtime_transport_reconnected",
    "runtime_safety_state_corrupted",
    "runtime_safety_state_deleted",
    "runtime_safety_state_reset",
    "runtime_review_written",
)


def default_local_paper_validation_root(project_root: Path) -> Path:
    """Return the default isolated root for paper-mode proving runs."""

    return project_root.resolve() / "data" / "local_paper_validation"


@dataclass(frozen=True)
class LocalPaperValidationProfile:
    """Filesystem layout for one isolated local paper-validation run."""

    project_root: Path
    validation_root: Path
    runtime_root: Path
    hot_state_dir: Path
    logs_dir: Path
    archive_dir: Path
    review_dir: Path
    journal_path: Path
    profile_path: Path
    name: str = PAPER_VALIDATION_PROFILE_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "project_root": str(self.project_root.resolve()),
            "validation_root": str(self.validation_root.resolve()),
            "runtime_root": str(self.runtime_root.resolve()),
            "hot_state_dir": str(self.hot_state_dir.resolve()),
            "logs_dir": str(self.logs_dir.resolve()),
            "archive_dir": str(self.archive_dir.resolve()),
            "review_dir": str(self.review_dir.resolve()),
            "journal_path": str(self.journal_path.resolve()),
            "profile_path": str(self.profile_path.resolve()),
        }

    def default_live_output_dir(self, *, as_of_date: date) -> Path:
        return (self.archive_dir / "live_market" / as_of_date.isoformat()).resolve()

    def default_review_output_dir(self, *, as_of_date: date) -> Path:
        return (self.review_dir / as_of_date.isoformat()).resolve()


def build_local_paper_validation_profile(
    *,
    project_root: Path,
    validation_root: Path | None = None,
) -> LocalPaperValidationProfile:
    """Resolve the isolated profile layout for a paper-validation run."""

    resolved_project_root = project_root.resolve()
    resolved_validation_root = (
        validation_root.resolve()
        if validation_root is not None
        else default_local_paper_validation_root(resolved_project_root)
    )
    runtime_root = resolved_validation_root / "runtime-root"
    hot_state_dir = resolved_validation_root / "hot-state"
    logs_dir = resolved_validation_root / "logs"
    archive_dir = resolved_validation_root / "archive"
    review_dir = archive_dir / _PAPER_VALIDATION_REVIEW_DIRNAME
    journal_path = archive_dir / _PAPER_VALIDATION_JOURNAL_FILENAME
    profile_path = resolved_validation_root / _PAPER_VALIDATION_PROFILE_FILENAME
    return LocalPaperValidationProfile(
        project_root=resolved_project_root,
        validation_root=resolved_validation_root,
        runtime_root=runtime_root.resolve(),
        hot_state_dir=hot_state_dir.resolve(),
        logs_dir=logs_dir.resolve(),
        archive_dir=archive_dir.resolve(),
        review_dir=review_dir.resolve(),
        journal_path=journal_path.resolve(),
        profile_path=profile_path.resolve(),
    )


def build_local_paper_validation_runtime_config(
    *,
    profile: LocalPaperValidationProfile,
    source_config_dir: Path,
    source_env_file: Path | None = None,
    internal_api_host: str = "127.0.0.1",
    internal_api_port: int = 8765,
    control_api_host: str = "127.0.0.1",
    control_api_port: int = 8766,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8780,
    dashboard_refresh_seconds: int = 5,
    active_log_filename: str = "local_paper_validation.log",
    log_rotate_max_bytes: int = 250_000,
    log_rotate_backup_count: int = 5,
) -> LocalStagingRuntimeConfig:
    """Build a local staging config locked to the paper-validation profile dirs."""

    return LocalStagingRuntimeConfig(
        source_config_dir=source_config_dir,
        runtime_root=profile.runtime_root,
        source_env_file=source_env_file,
        hot_state_dir=profile.hot_state_dir,
        logs_dir=profile.logs_dir,
        archive_dir=profile.archive_dir,
        internal_api_host=internal_api_host,
        internal_api_port=internal_api_port,
        control_api_host=control_api_host,
        control_api_port=control_api_port,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_refresh_seconds=dashboard_refresh_seconds,
        active_log_filename=active_log_filename,
        log_rotate_max_bytes=log_rotate_max_bytes,
        log_rotate_backup_count=log_rotate_backup_count,
    )


def paper_validation_default_control_state(
    *,
    updated_at_utc: str | None = None,
) -> OperatorControlState:
    """Return the forced paper-mode safety policy for validation runs."""

    return OperatorControlState(
        updated_at_utc=updated_at_utc,
        execution_submission_enabled=True,
        execution_mode="paper",
        live_actions_require_confirmation=True,
        broker_trading_enabled=False,
        paused_reason=None,
        last_pause_at_utc=None,
        last_resume_at_utc=None,
    )


def apply_paper_validation_safety_state(
    project_root: Path,
    *,
    updated_at_utc: str | None = None,
) -> Path:
    """Write the forced paper-mode control state into one runtime root."""

    store = OperatorControlStateStore(project_root)
    return store.save(
        paper_validation_default_control_state(updated_at_utc=updated_at_utc)
    )


def write_local_paper_validation_profile(
    profile: LocalPaperValidationProfile,
) -> Path:
    """Persist the resolved paper-validation profile for later review tooling."""

    profile.profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.profile_path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return profile.profile_path


@dataclass(frozen=True)
class PaperValidationRuntimeEvent:
    """One persisted local paper-validation lifecycle event."""

    event_type: str
    created_at_utc: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in _PAPER_VALIDATION_RUNTIME_EVENT_TYPES:
            raise ValueError(
                "event_type must be one of "
                f"{_PAPER_VALIDATION_RUNTIME_EVENT_TYPES}, got '{self.event_type}'."
            )
        if not self.created_at_utc.strip():
            raise ValueError("created_at_utc cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "created_at_utc": self.created_at_utc,
            "details": dict(self.details),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperValidationRuntimeEvent":
        raw_details = payload.get("details", {})
        if raw_details is None:
            raw_details = {}
        if not isinstance(raw_details, Mapping):
            raise ValueError("details must be a mapping.")
        raw_created_at = payload.get("created_at_utc")
        if not isinstance(raw_created_at, str) or not raw_created_at.strip():
            raise ValueError("created_at_utc must be a non-empty string.")
        return cls(
            event_type=str(payload.get("event_type")),
            created_at_utc=raw_created_at.strip(),
            details=dict(raw_details),
        )


@dataclass(frozen=True)
class PaperValidationRuntimeEventStore:
    """Append-only journal for local paper-validation operations."""

    path: Path

    def load(self) -> tuple[PaperValidationRuntimeEvent, ...]:
        resolved_path = self.path.resolve()
        if not resolved_path.exists():
            return ()
        events: list[PaperValidationRuntimeEvent] = []
        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                        if not isinstance(payload, Mapping):
                            raise ValueError("runtime journal line must decode to a mapping.")
                        events.append(PaperValidationRuntimeEvent.from_mapping(payload))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        LOGGER.warning(
                            "Ignoring malformed paper validation journal line %s in %s: %s",
                            line_number,
                            resolved_path,
                            exc,
                        )
        except OSError as exc:
            LOGGER.warning(
                "Ignoring paper validation journal at %s because it could not be read: %s",
                resolved_path,
                exc,
            )
            return ()
        return tuple(events)

    def append(self, event: PaperValidationRuntimeEvent) -> Path:
        resolved_path = self.path.resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")
        return resolved_path


class PaperValidationControlService(OperatorControlService):
    """Control service variant that refuses to leave paper mode."""

    def _load_safety_state_result(self) -> _OperatorControlStateLoadResult:
        load_result = super()._load_safety_state_result()
        if load_result.degraded or load_result.state.execution_mode == "paper":
            return load_result
        warning = (
            "Paper validation requires execution_mode='paper'; control mutations are "
            "blocked until the safety state is reset."
        )
        return _OperatorControlStateLoadResult(
            state=replace(
                load_result.state,
                execution_submission_enabled=False,
                paused_reason=(
                    "Paper validation requires paper mode; control mutations are "
                    "blocked until the safety state is reset."
                ),
            ),
            warnings=tuple(load_result.warnings) + (warning,),
            degraded=True,
        )

    def set_execution_mode(self, command: SetExecutionModeCommand) -> OperatorCommandResult:
        if command.execution_mode == "paper":
            return super().set_execution_mode(command)
        requested_at = _utc_now_isoformat(self._now_utc())
        raw_load_result = OperatorControlService._load_safety_state_result(self)
        return self._finalize_result(
            command_name="set_execution_mode",
            command_payload=command.to_dict(),
            operator_id=command.operator_id,
            requested_at_utc=requested_at,
            status="rejected",
            ok=False,
            message="Paper validation locks execution mode to 'paper'.",
            error_code="paper_validation_mode_locked",
            warnings=raw_load_result.warnings,
            safety_state=raw_load_result.state,
            artifact_paths={
                "operator_control_state": str(self._safety_store.path.resolve())
            },
        )


@dataclass
class LocalPaperValidationRuntime:
    """Thin proving-run wrapper over the generic local staging runtime."""

    profile: LocalPaperValidationProfile
    runtime: LocalStagingRuntime
    event_store: PaperValidationRuntimeEventStore

    @property
    def paths(self) -> LocalStagingRuntimePaths:
        return self.runtime.paths

    @property
    def control_service(self) -> OperatorControlService:
        return self.runtime.control_service

    @property
    def query_service(self) -> InternalApiQueryService:
        return self.runtime.query_service

    @property
    def internal_api_url(self) -> str:
        return self.runtime.internal_api_url

    @property
    def control_api_url(self) -> str:
        return self.runtime.control_api_url

    @property
    def dashboard_url(self) -> str:
        return self.runtime.dashboard_url

    @property
    def last_supervisor_result(self) -> object | None:
        return self.runtime.last_supervisor_result

    def configure_logging(
        self,
        *,
        level: str = "INFO",
        json_logs: bool = False,
    ) -> None:
        self.runtime.configure_logging(level=level, json_logs=json_logs)

    def start(self, *, max_iterations: int | None = None) -> None:
        self.runtime.start(max_iterations=max_iterations)
        self._record_event(
            "runtime_started",
            {
                "internal_api_url": self.internal_api_url,
                "control_api_url": self.control_api_url,
                "dashboard_url": self.dashboard_url,
                "max_iterations": max_iterations,
            },
        )

    def wait_for_stack(self, timeout: float | None = None) -> bool:
        return self.runtime.wait_for_stack(timeout=timeout)

    def wait_for_supervisor(self, timeout: float | None = None) -> bool:
        return self.runtime.wait_for_supervisor(timeout=timeout)

    def stop(self) -> None:
        was_started = getattr(self.runtime, "_started", False)
        self.runtime.stop()
        if was_started:
            self._record_event(
                "runtime_stopped",
                {
                    "last_supervisor_result": (
                        self.last_supervisor_result.to_dict()
                        if hasattr(self.last_supervisor_result, "to_dict")
                        else self.last_supervisor_result
                    ),
                },
            )

    def restart(self, *, max_iterations: int | None = None) -> "LocalPaperValidationRuntime":
        self._record_event(
            "runtime_restart_requested",
            {"max_iterations": max_iterations},
        )
        restarted_runtime = self.runtime.restart(max_iterations=max_iterations)
        restarted = LocalPaperValidationRuntime(
            profile=self.profile,
            runtime=restarted_runtime,
            event_store=self.event_store,
        )
        restarted._record_event(
            "runtime_restarted",
            {"max_iterations": max_iterations},
        )
        return restarted

    def rotate_logs(self) -> tuple[Path, ...]:
        archived_paths = self.runtime.rotate_logs()
        self._record_event(
            "runtime_logs_rotated",
            {
                "archived_count": len(archived_paths),
                "archived_paths": [str(path.resolve()) for path in archived_paths],
            },
        )
        return archived_paths

    def corrupt_safety_state(self, payload: str = "{malformed-json") -> Path:
        path = self.runtime.corrupt_safety_state(payload=payload)
        self._record_event(
            "runtime_safety_state_corrupted",
            {"path": str(path.resolve())},
        )
        return path

    def delete_safety_state(self) -> None:
        self.runtime.delete_safety_state()
        self._record_event(
            "runtime_safety_state_deleted",
            {"path": str(self.paths.operator_control_state_path.resolve())},
        )

    def reset_safety_state(self) -> Path:
        path = apply_paper_validation_safety_state(
            self.paths.runtime_root,
            updated_at_utc=_utc_now_isoformat(datetime.now(timezone.utc)),
        )
        self._record_event(
            "runtime_safety_state_reset",
            {"path": str(path.resolve())},
        )
        return path

    def write_runtime_env(self, content: str) -> Path:
        path = self.runtime.write_runtime_env(content)
        self._record_event(
            "runtime_env_overridden",
            {"path": str(path.resolve()), "keys": sorted(_env_keys_from_text(content))},
        )
        return path

    def disconnect_transport(self) -> None:
        self.runtime.disconnect_transport()
        self._record_event("runtime_transport_disconnected", {})

    def reconnect_transport(self) -> None:
        self.runtime.reconnect_transport()
        self._record_event("runtime_transport_reconnected", {})

    def write_review_summary(
        self,
        *,
        as_of_date: date | None = None,
        window_days: int = 1,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        summary = build_local_paper_validation_summary(
            self.profile,
            as_of_date=as_of_date,
            window_days=window_days,
        )
        paths = write_local_paper_validation_summary(
            summary,
            output_dir=output_dir
            or self.profile.default_review_output_dir(as_of_date=summary.as_of_date),
        )
        self._record_event(
            "runtime_review_written",
            {label: str(path.resolve()) for label, path in paths.items()},
        )
        return paths

    def _record_event(self, event_type: str, details: Mapping[str, Any]) -> None:
        try:
            self.event_store.append(
                PaperValidationRuntimeEvent(
                    event_type=event_type,
                    created_at_utc=_utc_now_isoformat(datetime.now(timezone.utc)),
                    details=dict(details),
                )
            )
        except OSError as exc:
            LOGGER.warning("Failed to persist paper validation runtime event: %s", exc)


def create_local_paper_validation_runtime(
    *,
    runtime_config: LocalStagingRuntimeConfig,
    validation_root: Path | None = None,
    reset_safety_state: bool = True,
    supervisor_factory: Callable[[AppConfig, LocalStagingRuntimePaths], RuntimeSupervisor],
    query_service_factory: Callable[
        [AppConfig, LocalStagingRuntimePaths], InternalApiQueryService
    ]
    | None = None,
    control_service_factory: Callable[
        [AppConfig, LocalStagingRuntimePaths], OperatorControlService
    ]
    | None = None,
    internal_api_server_factory: Callable[..., Any] | None = None,
    control_api_server_factory: Callable[..., Any] | None = None,
    dashboard_server_factory: Callable[..., Any] | None = None,
) -> LocalPaperValidationRuntime:
    """Create one isolated paper-validation runtime over the staging harness."""

    profile = build_local_paper_validation_profile(
        project_root=runtime_config.source_config_dir.resolve().parent,
        validation_root=validation_root or runtime_config.runtime_root.parent,
    )
    resolved_config = replace(
        runtime_config,
        runtime_root=profile.runtime_root,
        hot_state_dir=profile.hot_state_dir,
        logs_dir=profile.logs_dir,
        archive_dir=profile.archive_dir,
    )

    def default_control_service_factory(
        app_config: AppConfig,
        paths: LocalStagingRuntimePaths,
    ) -> OperatorControlService:
        return PaperValidationControlService(
            project_root=app_config.project_root,
            env_file=paths.env_file,
            fail_closed_on_missing_safety_state=True,
        )

    runtime = create_local_staging_runtime(
        resolved_config,
        supervisor_factory=supervisor_factory,
        query_service_factory=query_service_factory,
        control_service_factory=(
            control_service_factory or default_control_service_factory
        ),
        internal_api_server_factory=internal_api_server_factory,
        control_api_server_factory=control_api_server_factory,
        dashboard_server_factory=dashboard_server_factory,
    )
    write_local_paper_validation_profile(profile)
    if reset_safety_state:
        apply_paper_validation_safety_state(
            profile.runtime_root,
            updated_at_utc=_utc_now_isoformat(datetime.now(timezone.utc)),
        )
    return LocalPaperValidationRuntime(
        profile=profile,
        runtime=runtime,
        event_store=PaperValidationRuntimeEventStore(profile.journal_path),
    )


@dataclass(frozen=True)
class LocalPaperValidationSummary:
    """Compact operational review snapshot for one proving window."""

    generated_at_utc: str
    as_of_date: date
    window_days: int
    profile: dict[str, Any]
    health_checkpoint: dict[str, Any]
    safety: dict[str, Any]
    runtime_journal: dict[str, Any]
    control_actions: dict[str, Any]
    trade_feedback: dict[str, Any]
    notification_summary: dict[str, Any]
    pending_orders: dict[str, Any] | None
    market_state: dict[str, Any] | None
    log_summary: dict[str, Any]
    artifact_paths: dict[str, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "as_of_date": self.as_of_date.isoformat(),
            "window_days": self.window_days,
            "profile": dict(self.profile),
            "health_checkpoint": dict(self.health_checkpoint),
            "safety": dict(self.safety),
            "runtime_journal": dict(self.runtime_journal),
            "control_actions": dict(self.control_actions),
            "trade_feedback": dict(self.trade_feedback),
            "notification_summary": dict(self.notification_summary),
            "pending_orders": dict(self.pending_orders) if self.pending_orders is not None else None,
            "market_state": dict(self.market_state) if self.market_state is not None else None,
            "log_summary": dict(self.log_summary),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "warnings": list(self.warnings),
        }

    def to_brief(self) -> str:
        lines = [
            f"Local paper validation summary for {self.as_of_date.isoformat()}",
            "",
            "Health",
            (
                f"- Service state: {self.health_checkpoint['service_state']} | "
                f"connected={self.health_checkpoint['connected']} | "
                f"stale={self.health_checkpoint['stale']} | "
                f"last cycle={self.health_checkpoint['last_cycle_status'] or 'n/a'}"
            ),
            (
                f"- Last message: {self.health_checkpoint['last_message_at_utc'] or 'n/a'} | "
                f"last successful cycle: {self.health_checkpoint['last_successful_flush_at_utc'] or 'n/a'} | "
                f"reconnect attempts: {self.health_checkpoint['reconnect_attempt_count']}"
            ),
            "",
            "Safety",
            (
                f"- Paper guardrail active: {self.safety['paper_guardrail_active']} | "
                f"current mode: {self.safety['current_execution_mode']} | "
                f"paper-only intact: {self.safety['paper_only_intact']}"
            ),
            (
                f"- Live mode attempts: {self.safety['live_mode_attempt_count']} | "
                f"live mode successes: {self.safety['live_mode_success_count']}"
            ),
            "",
            "Actions",
            (
                f"- Control actions: {self.control_actions['window_action_count']} | "
                f"failed/rejected: {self.control_actions['failed_or_rejected_count']} | "
                f"broker sync drift signals: {self.control_actions['broker_sync_drift_count']}"
            ),
            (
                f"- Orders: decisions={self.trade_feedback['decision_count']} | "
                f"submitted={self.trade_feedback['broker_submitted_count']} | "
                f"cancelled={self.trade_feedback['broker_cancelled_count']} | "
                f"replaced={self.trade_feedback['broker_replaced_count']} | "
                f"filled={self.trade_feedback['executed_count']}"
            ),
            "",
            "Runtime",
            (
                f"- Starts={self.runtime_journal['start_count']} | "
                f"stops={self.runtime_journal['stop_count']} | "
                f"restarts={self.runtime_journal['restart_count']} | "
                f"manual disconnects={self.runtime_journal['manual_disconnect_count']}"
            ),
            "",
            "Warnings",
            (
                f"- Log warnings: {self.log_summary['warning_count']} | "
                f"errors: {self.log_summary['error_count']} | "
                f"disconnect warnings: {self.log_summary['disconnect_warning_count']}"
            ),
        ]
        repeated = self.log_summary.get("top_warning_messages", [])
        if repeated:
            lines.extend(["", "Repeated warnings"])
            lines.extend(
                f"- {item['count']}x {item['message']}"
                for item in repeated[:5]
            )
        failures = self.control_actions.get("recent_failures", [])
        if failures:
            lines.extend(["", "Recent failures"])
            lines.extend(
                f"- {item['requested_at_utc']} {item['command_name']} -> {item['status']} ({item['error_code'] or 'no_error_code'})"
                for item in failures[:5]
            )
        if self.warnings:
            lines.extend(["", "Summary warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def build_local_paper_validation_summary(
    profile: LocalPaperValidationProfile,
    *,
    as_of_date: date | None = None,
    window_days: int = 1,
) -> LocalPaperValidationSummary:
    """Aggregate the current proving-run artifacts into one review summary."""

    if window_days <= 0:
        raise ValueError("window_days must be greater than zero.")
    resolved_as_of_date = as_of_date or date.today()
    window_start = resolved_as_of_date - timedelta(days=window_days - 1)
    generated_at = datetime.now(timezone.utc)
    warnings: list[str] = []

    config = _load_runtime_config_safe(profile.runtime_root / "config", warnings)
    query_service = (
        InternalApiQueryService(project_root=profile.runtime_root, config=config)
        if config is not None
        else None
    )
    health_payload = query_service.health() if query_service is not None else None
    pending_orders_payload = (
        query_service.pending_orders() if query_service is not None else None
    )
    market_state_payload = (
        query_service.market_state() if query_service is not None else None
    )
    market_state_transitions_payload = (
        query_service.market_state_transitions() if query_service is not None else None
    )

    control_service = PaperValidationControlService(
        project_root=profile.runtime_root,
        fail_closed_on_missing_safety_state=True,
    )
    safety_payload = control_service.safety()
    safety_state = dict(safety_payload["data"]["safety_state"])

    control_records = load_operator_control_audit_records(
        default_operator_control_audit_log_path(profile.runtime_root)
    )
    feedback_events = load_trade_feedback_events(
        default_trade_feedback_log_path(profile.runtime_root)
    )
    journal_events = PaperValidationRuntimeEventStore(profile.journal_path).load()
    notification_state = NotificationDeliveryStateStore(profile.runtime_root).load()

    control_records_window = [
        record
        for record in control_records
        if _record_in_window(record.requested_at_utc, window_start, resolved_as_of_date)
    ]
    feedback_events_window = [
        event
        for event in feedback_events
        if _feedback_event_in_window(event, window_start, resolved_as_of_date)
    ]
    journal_events_window = [
        event
        for event in journal_events
        if _record_in_window(event.created_at_utc, window_start, resolved_as_of_date)
    ]

    control_counts_by_command = Counter(
        record.command_name for record in control_records_window
    )
    failed_control_records = [
        {
            "requested_at_utc": record.requested_at_utc,
            "command_name": record.command_name,
            "status": record.status,
            "error_code": record.error_code,
            "message": record.message,
        }
        for record in control_records_window
        if record.status in {"failed", "rejected"}
    ]
    failed_control_records.sort(key=lambda item: item["requested_at_utc"], reverse=True)
    live_mode_attempt_records = [
        record
        for record in control_records
        if record.command_name == "set_execution_mode"
        and record.command_payload.get("execution_mode") == "live"
    ]
    live_mode_success_count = sum(
        record.status == "completed" for record in live_mode_attempt_records
    )
    broker_sync_drift_count = 0
    for record in control_records_window:
        if record.command_name != "force_broker_sync":
            continue
        result_data = record.result_payload.get("data", {})
        if not isinstance(result_data, Mapping):
            result_data = {}
        unmatched_update_count = result_data.get("unmatched_update_count", 0)
        try:
            unmatched_update_count = int(unmatched_update_count)
        except (TypeError, ValueError):
            unmatched_update_count = 0
        if unmatched_update_count > 0 or bool(record.warnings):
            broker_sync_drift_count += 1

    feedback_counts_by_type = Counter(event.event_type for event in feedback_events_window)
    feedback_counts_by_workflow = Counter(event.workflow for event in feedback_events_window)
    action_counts_by_day = Counter(
        record.requested_at_utc[:10] for record in control_records_window
    )
    action_counts_by_day.update(
        _feedback_event_day(event)
        for event in feedback_events_window
        if _feedback_event_day(event) is not None
    )
    action_counts_by_day.update(
        event.created_at_utc[:10] for event in journal_events_window
    )

    runtime_journal_summary = _build_runtime_journal_summary(
        journal_events,
        journal_events_window=journal_events_window,
        as_of_timestamp=generated_at,
    )
    log_summary = _build_runtime_log_summary(
        profile=profile,
        window_start=window_start,
        as_of_date=resolved_as_of_date,
    )

    notification_summary = {
        "notifications_configured": _notifications_configured(profile.runtime_root / ".env"),
        "updated_at_utc": notification_state.updated_at_utc,
        "active_key_count": len(notification_state.active_keys),
        "delivered_once_key_count": len(notification_state.delivered_once_keys),
    }

    health_checkpoint = _build_health_checkpoint_payload(health_payload)
    recent_transitions = _recent_transition_summaries(market_state_transitions_payload)
    market_state_summary = None
    if market_state_payload is not None:
        market_state_data = market_state_payload.get("data")
        if isinstance(market_state_data, Mapping):
            snapshot = market_state_data.get("snapshot", {})
            alertable_states = market_state_data.get("current_alertable_states", ())
            top_priority_candidates = market_state_data.get("top_priority_candidates", ())
            market_state_summary = {
                "available": bool(market_state_payload.get("available")),
                "transition_count": market_state_data.get("transition_count"),
                "baseline_established": market_state_data.get("baseline_established"),
                "alertable_state_count": len(alertable_states) if isinstance(alertable_states, Sequence) else 0,
                "top_priority_candidate_count": (
                    len(top_priority_candidates)
                    if isinstance(top_priority_candidates, Sequence)
                    else 0
                ),
                "portfolio_path": (
                    snapshot.get("portfolio_path")
                    if isinstance(snapshot, Mapping)
                    else None
                ),
                "recent_transitions": recent_transitions,
            }

    pending_orders_summary = None
    if pending_orders_payload is not None:
        pending_data = pending_orders_payload.get("data")
        if isinstance(pending_data, Mapping):
            summary = pending_data.get("summary", {})
            pending_orders_summary = dict(summary) if isinstance(summary, Mapping) else None

    paper_only_intact = (
        safety_state.get("execution_mode") == "paper" and live_mode_success_count == 0
    )
    safety_summary = {
        "paper_guardrail_active": True,
        "current_execution_mode": safety_state.get("execution_mode"),
        "execution_submission_enabled": safety_state.get("execution_submission_enabled"),
        "live_actions_require_confirmation": safety_state.get(
            "live_actions_require_confirmation"
        ),
        "broker_trading_enabled": safety_state.get("broker_trading_enabled"),
        "control_state_readable": bool(safety_payload["data"]["control_state_readable"]),
        "paper_only_intact": paper_only_intact,
        "live_mode_attempt_count": len(live_mode_attempt_records),
        "live_mode_success_count": live_mode_success_count,
        "current_state": safety_state,
        "warnings": list(safety_payload.get("warnings", ())),
    }

    artifact_paths = {
        "paper_validation_profile": str(profile.profile_path.resolve()),
        "paper_validation_runtime_journal": str(profile.journal_path.resolve()),
        "runtime_root": str(profile.runtime_root.resolve()),
        "hot_state_dir": str(profile.hot_state_dir.resolve()),
        "logs_dir": str(profile.logs_dir.resolve()),
        "archive_dir": str(profile.archive_dir.resolve()),
        "operator_control_state": str(OperatorControlStateStore(profile.runtime_root).path.resolve()),
        "operator_control_audit_log": str(
            default_operator_control_audit_log_path(profile.runtime_root).resolve()
        ),
        "live_market_service_status": str(
            LiveMarketServiceStatusStore(profile.runtime_root).path.resolve()
        ),
        "notification_delivery_state": str(
            NotificationDeliveryStateStore(profile.runtime_root).path.resolve()
        ),
        "pending_order_state": str(PendingOrderStateStore(profile.runtime_root).path.resolve()),
        "trade_feedback_log": str(default_trade_feedback_log_path(profile.runtime_root).resolve()),
    }

    return LocalPaperValidationSummary(
        generated_at_utc=_utc_now_isoformat(generated_at),
        as_of_date=resolved_as_of_date,
        window_days=window_days,
        profile=profile.to_dict(),
        health_checkpoint=health_checkpoint,
        safety=safety_summary,
        runtime_journal=runtime_journal_summary,
        control_actions={
            "window_action_count": len(control_records_window),
            "counts_by_command": dict(sorted(control_counts_by_command.items())),
            "failed_or_rejected_count": len(failed_control_records),
            "recent_failures": failed_control_records[:10],
            "broker_sync_drift_count": broker_sync_drift_count,
            "action_counts_by_day": dict(sorted(action_counts_by_day.items())),
        },
        trade_feedback={
            "window_event_count": len(feedback_events_window),
            "event_counts_by_type": dict(sorted(feedback_counts_by_type.items())),
            "event_counts_by_workflow": dict(sorted(feedback_counts_by_workflow.items())),
            "decision_count": feedback_counts_by_type.get("decision", 0),
            "broker_submitted_count": feedback_counts_by_type.get("broker_submitted", 0),
            "broker_cancelled_count": feedback_counts_by_type.get("broker_cancelled", 0),
            "broker_replaced_count": feedback_counts_by_type.get("broker_replaced", 0),
            "broker_expired_count": feedback_counts_by_type.get("broker_expired", 0),
            "executed_count": feedback_counts_by_type.get("executed", 0),
        },
        notification_summary=notification_summary,
        pending_orders=pending_orders_summary,
        market_state=market_state_summary,
        log_summary=log_summary,
        artifact_paths=artifact_paths,
        warnings=tuple(warnings),
    )


def write_local_paper_validation_summary(
    summary: LocalPaperValidationSummary,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write one review snapshot as JSON plus human-readable helper outputs."""

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = resolved_output_dir / "paper_validation_summary.json"
    checkpoint_json = resolved_output_dir / "paper_validation_checkpoint.json"
    brief_text = resolved_output_dir / "paper_validation_brief.txt"
    summary_json.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_json.write_text(
        json.dumps(summary.health_checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    brief_text.write_text(summary.to_brief(), encoding="utf-8")
    return {
        "paper_validation_summary_json": summary_json.resolve(),
        "paper_validation_checkpoint_json": checkpoint_json.resolve(),
        "paper_validation_brief": brief_text.resolve(),
    }


def _load_runtime_config_safe(config_dir: Path, warnings: list[str]) -> AppConfig | None:
    try:
        return load_app_config(config_dir=config_dir)
    except Exception as exc:
        warnings.append(f"Runtime config could not be loaded from {config_dir.resolve()}: {exc}")
        return None


def _build_health_checkpoint_payload(
    health_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(health_payload, Mapping):
        return {
            "available": False,
            "service_state": "unknown",
            "connected": False,
            "stale": False,
            "reconnect_attempt_count": 0,
            "warning_count": 0,
            "last_message_at_utc": None,
            "last_successful_flush_at_utc": None,
            "last_cycle_status": None,
            "last_cycle_warning_count": 0,
            "last_warning": None,
            "last_error": None,
        }
    data = health_payload.get("data")
    status = data.get("status", {}) if isinstance(data, Mapping) else {}
    if not isinstance(status, Mapping):
        status = {}
    return {
        "available": bool(health_payload.get("available")),
        "service_state": status.get("service_state", "unknown"),
        "connected": bool(status.get("connected")),
        "stale": bool(status.get("stale")),
        "reconnect_attempt_count": _coerce_int(status.get("reconnect_attempt_count")),
        "warning_count": _coerce_int(status.get("warning_count")),
        "cycle_count": _coerce_int(status.get("cycle_count")),
        "last_message_at_utc": status.get("last_message_at_utc"),
        "last_connected_at_utc": status.get("last_connected_at_utc"),
        "last_successful_flush_at_utc": status.get("last_successful_flush_at_utc"),
        "last_cycle_status": status.get("last_cycle_status"),
        "last_cycle_warning_count": _coerce_int(status.get("last_cycle_warning_count")),
        "last_warning": status.get("last_warning"),
        "last_error": status.get("last_error"),
    }


def _build_runtime_journal_summary(
    all_events: Sequence[PaperValidationRuntimeEvent],
    *,
    journal_events_window: Sequence[PaperValidationRuntimeEvent],
    as_of_timestamp: datetime,
) -> dict[str, Any]:
    counts = Counter(event.event_type for event in all_events)
    window_counts = Counter(event.event_type for event in journal_events_window)
    uptime_seconds = _estimated_runtime_uptime_seconds(all_events, as_of_timestamp=as_of_timestamp)
    return {
        "event_count": len(all_events),
        "window_event_count": len(journal_events_window),
        "start_count": counts.get("runtime_started", 0),
        "stop_count": counts.get("runtime_stopped", 0),
        "restart_count": counts.get("runtime_restarted", 0),
        "manual_disconnect_count": counts.get("runtime_transport_disconnected", 0),
        "manual_reconnect_count": counts.get("runtime_transport_reconnected", 0),
        "window_counts": dict(sorted(window_counts.items())),
        "estimated_uptime_seconds": uptime_seconds,
    }


def _estimated_runtime_uptime_seconds(
    events: Sequence[PaperValidationRuntimeEvent],
    *,
    as_of_timestamp: datetime,
) -> float:
    active_start: datetime | None = None
    total = 0.0
    sorted_events = sorted(events, key=lambda event: event.created_at_utc)
    for event in sorted_events:
        occurred_at = _parse_iso_datetime(event.created_at_utc)
        if occurred_at is None:
            continue
        if event.event_type == "runtime_started":
            active_start = occurred_at
        elif event.event_type == "runtime_stopped" and active_start is not None:
            total += max((occurred_at - active_start).total_seconds(), 0.0)
            active_start = None
    if active_start is not None:
        total += max((as_of_timestamp - active_start).total_seconds(), 0.0)
    return round(total, 3)


def _build_runtime_log_summary(
    *,
    profile: LocalPaperValidationProfile,
    window_start: date,
    as_of_date: date,
) -> dict[str, Any]:
    records = list(
        _iter_runtime_log_records(
            logs_dir=profile.logs_dir,
            archived_logs_dir=profile.archive_dir / "logs",
        )
    )
    window_records = [
        record
        for record in records
        if record.log_date is None or window_start <= record.log_date <= as_of_date
    ]
    warning_records = [record for record in window_records if record.level == "WARNING"]
    error_records = [record for record in window_records if record.level in {"ERROR", "CRITICAL"}]
    top_warning_messages = [
        {"message": message, "count": count}
        for message, count in Counter(record.message for record in warning_records).most_common(5)
    ]
    top_error_messages = [
        {"message": message, "count": count}
        for message, count in Counter(record.message for record in error_records).most_common(5)
    ]
    return {
        "record_count": len(window_records),
        "warning_count": len(warning_records),
        "error_count": len(error_records),
        "disconnect_warning_count": sum(
            "websocket transport disconnected:" in record.message for record in warning_records
        ),
        "connect_failure_count": sum(
            "websocket connect failed:" in record.message for record in warning_records
        ),
        "notification_failure_count": sum(
            "Notification routing failed:" in record.message for record in warning_records
        ),
        "top_warning_messages": top_warning_messages,
        "top_error_messages": top_error_messages,
    }


@dataclass(frozen=True)
class _RuntimeLogRecord:
    log_date: date | None
    level: str
    message: str


def _iter_runtime_log_records(
    *,
    logs_dir: Path,
    archived_logs_dir: Path,
) -> Iterable[_RuntimeLogRecord]:
    candidate_paths = sorted(logs_dir.glob("*")) + sorted(archived_logs_dir.glob("*"))
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    record = _parse_runtime_log_line(raw_line)
                    if record is not None:
                        yield record
        except OSError as exc:
            LOGGER.warning("Failed to read runtime log %s: %s", path, exc)


def _parse_runtime_log_line(raw_line: str) -> _RuntimeLogRecord | None:
    stripped = raw_line.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, Mapping):
            return None
        timestamp = _clean_text(payload.get("timestamp"))
        level = (_clean_text(payload.get("level")) or "INFO").upper()
        message = _clean_text(payload.get("message")) or ""
        return _RuntimeLogRecord(
            log_date=_log_date_from_text(timestamp),
            level=level,
            message=message,
        )
    parts = [part.strip() for part in stripped.split("|", maxsplit=3)]
    if len(parts) != 4:
        return None
    return _RuntimeLogRecord(
        log_date=_log_date_from_text(parts[0]),
        level=parts[1].upper(),
        message=parts[3],
    )


def _log_date_from_text(value: str | None) -> date | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if "T" in cleaned:
        parsed = _parse_iso_datetime(cleaned)
        return parsed.date() if parsed is not None else None
    if " " in cleaned:
        cleaned = cleaned.replace(",", ".", 1)
        try:
            return datetime.fromisoformat(cleaned).date()
        except ValueError:
            return None
    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        return None


def _recent_transition_summaries(
    market_state_transitions_payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(market_state_transitions_payload, Mapping):
        return []
    data = market_state_transitions_payload.get("data")
    if not isinstance(data, Mapping):
        return []
    transitions = data.get("transitions", ())
    if not isinstance(transitions, Sequence):
        return []
    recent: list[dict[str, Any]] = []
    for item in transitions[:10]:
        if not isinstance(item, Mapping):
            continue
        recent.append(
            {
                "symbol": item.get("symbol"),
                "action": item.get("current_action"),
                "category": item.get("category"),
                "transition_type": item.get("transition_type"),
            }
        )
    return recent


def _notifications_configured(env_file: Path) -> bool:
    env = _read_env_file(env_file)
    return any(
        _clean_text(env.get(key)) is not None
        for key in ("NOTIFICATION_WEBHOOK_URL", "DISCORD_WEBHOOK_URL")
    )


def _read_env_file(path: Path) -> dict[str, str]:
    resolved_path = path.resolve()
    if not resolved_path.exists():
        return {}
    env: dict[str, str] = {}
    try:
        for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key:
                env[key] = value.strip()
    except OSError as exc:
        LOGGER.warning("Failed to read env file %s: %s", resolved_path, exc)
    return env


def _env_keys_from_text(content: str) -> tuple[str, ...]:
    keys: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _value = stripped.split("=", 1)
        key = key.strip()
        if key:
            keys.append(key)
    return tuple(sorted(set(keys)))


def _record_in_window(timestamp_text: str, window_start: date, as_of_date: date) -> bool:
    parsed = _parse_iso_datetime(timestamp_text)
    if parsed is None:
        return False
    record_date = parsed.date()
    return window_start <= record_date <= as_of_date


def _feedback_event_in_window(
    event: TradeFeedbackEvent,
    window_start: date,
    as_of_date: date,
) -> bool:
    event_date = _feedback_event_date(event)
    if event_date is None:
        return False
    return window_start <= event_date <= as_of_date


def _feedback_event_day(event: TradeFeedbackEvent) -> str | None:
    event_date = _feedback_event_date(event)
    return event_date.isoformat() if event_date is not None else None


def _feedback_event_date(event: TradeFeedbackEvent) -> date | None:
    if event.timestamp_utc is not None:
        parsed = _parse_iso_datetime(event.timestamp_utc)
        if parsed is not None:
            return parsed.date()
    return event.as_of_date


def _parse_iso_datetime(value: str | None) -> datetime | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now_isoformat(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat()
