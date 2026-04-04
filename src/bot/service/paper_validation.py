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
from bot.workflow_status import (
    build_workflow_input_overview,
    render_workflow_input_summary_lines,
    summarize_named_workflow_input_statuses,
    summarize_workflow_input_statuses,
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
        was_started = getattr(self.runtime, "_started", False)
        self._record_event(
            "runtime_restart_requested",
            {"max_iterations": max_iterations},
        )
        if was_started:
            self._record_event(
                "runtime_stopped",
                {
                    "reason": "restart_requested",
                    "last_supervisor_result": (
                        self.last_supervisor_result.to_dict()
                        if hasattr(self.last_supervisor_result, "to_dict")
                        else self.last_supervisor_result
                    ),
                },
            )
        restarted_runtime = self.runtime.restart(max_iterations=max_iterations)
        restarted = LocalPaperValidationRuntime(
            profile=self.profile,
            runtime=restarted_runtime,
            event_store=self.event_store,
        )
        restarted._record_event(
            "runtime_started",
            {
                "internal_api_url": restarted.internal_api_url,
                "control_api_url": restarted.control_api_url,
                "dashboard_url": restarted.dashboard_url,
                "max_iterations": max_iterations,
                "restart": True,
            },
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
    pending_orders: dict[str, Any]
    market_state: dict[str, Any]
    log_summary: dict[str, Any]
    paper_integrity: dict[str, Any]
    anomaly_flags: tuple[dict[str, Any], ...]
    change_summary: dict[str, Any]
    daily_review: dict[str, Any]
    operator_checklist: tuple[dict[str, Any], ...]
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
            "pending_orders": dict(self.pending_orders),
            "market_state": dict(self.market_state),
            "log_summary": dict(self.log_summary),
            "paper_integrity": dict(self.paper_integrity),
            "anomaly_flags": [dict(item) for item in self.anomaly_flags],
            "change_summary": dict(self.change_summary),
            "daily_review": dict(self.daily_review),
            "operator_checklist": [dict(item) for item in self.operator_checklist],
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
            (
                f"- Providers: stream={self.health_checkpoint['stream_provider'] or 'n/a'} | "
                f"historical={self.health_checkpoint['historical_provider'] or 'n/a'} | "
                f"reference={self.health_checkpoint['reference_provider'] or 'n/a'} | "
                f"earnings={self.health_checkpoint['earnings_provider'] or 'n/a'} | "
                f"execution={self.health_checkpoint['execution_broker'] or 'n/a'} | "
                f"broker updates={self.health_checkpoint['broker_update_stream_provider'] or 'n/a'}"
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
            (
                f"- Anomaly flags: {len(self.anomaly_flags)} | "
                f"paper integrity warnings: {len(self.paper_integrity.get('integrity_warnings', ()))}"
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

    def to_daily_review_text(self) -> str:
        lines = [
            f"Local paper validation daily review for {self.as_of_date.isoformat()}",
            "",
            "Service",
            (
                f"- Healthy={self.daily_review['service']['healthy']} | "
                f"feed_continuity_healthy={self.daily_review['service']['feed_continuity_healthy']} | "
                f"status_artifact_available={self.daily_review['service']['status_artifact_available']} | "
                f"state={self.health_checkpoint['service_state']} | "
                f"uptime_seconds={self.runtime_journal['estimated_uptime_seconds']} | "
                f"restarts={self.runtime_journal['restart_count']}"
            ),
            (
                f"- Last message: {self.health_checkpoint['last_message_at_utc'] or 'n/a'} | "
                f"last successful cycle: {self.health_checkpoint['last_successful_flush_at_utc'] or 'n/a'} | "
                f"last cycle status: {self.health_checkpoint['last_cycle_status'] or 'n/a'}"
            ),
            (
                f"- Last warning: {self.health_checkpoint['last_warning'] or 'n/a'} | "
                f"status updated: {self.health_checkpoint['status_updated_at_utc'] or 'n/a'}"
            ),
            (
                f"- Providers: stream={self.health_checkpoint['stream_provider'] or 'n/a'} | "
                f"historical={self.health_checkpoint['historical_provider'] or 'n/a'} | "
                f"reference={self.health_checkpoint['reference_provider'] or 'n/a'} | "
                f"earnings={self.health_checkpoint['earnings_provider'] or 'n/a'} | "
                f"execution={self.health_checkpoint['execution_broker'] or 'n/a'} | "
                f"broker updates={self.health_checkpoint['broker_update_stream_provider'] or 'n/a'}"
            ),
            "",
            "Connectivity",
            (
                f"- Reconnect attempts={self.health_checkpoint['reconnect_attempt_count']} | "
                f"manual disconnects={self.runtime_journal['manual_disconnect_count']} | "
                f"manual reconnects={self.runtime_journal['manual_reconnect_count']}"
            ),
            (
                f"- Disconnect warnings={self.log_summary['disconnect_warning_count']} | "
                f"stale feed warnings={self.log_summary['stale_feed_warning_count']} | "
                f"connect failures={self.log_summary['connect_failure_count']}"
            ),
            "",
            "Safety",
            (
                f"- Paper only intact={self.paper_integrity['paper_only_intact']} | "
                f"execution mode stayed paper={self.paper_integrity['execution_mode_stayed_paper']} | "
                f"broker target stayed paper={self.paper_integrity['broker_target_stayed_paper']}"
            ),
            (
                f"- Live trading enabled ever={self.paper_integrity['live_trading_enabled_ever']} | "
                f"control-state degradation count={self.paper_integrity['control_state_degradation_count']} | "
                f"window degradation count={self.paper_integrity['control_state_degradation_window_count']}"
            ),
            "",
            "Orders And Broker",
            (
                f"- Pending orders={self.pending_orders['active_order_count']} | "
                f"unusual states={self.pending_orders['unusual_state_count']} | "
                f"broker sync drift signals={self.control_actions['broker_sync_drift_count']}"
            ),
            (
                f"- Broker submitted={self.trade_feedback['broker_submitted_count']} | "
                f"cancelled={self.trade_feedback['broker_cancelled_count']} | "
                f"replaced={self.trade_feedback['broker_replaced_count']} | "
                f"filled={self.trade_feedback['executed_count']}"
            ),
            "",
            "Notifications",
            (
                f"- Configured={self.notification_summary['notifications_configured']} | "
                f"delivery failures={self.notification_summary['delivery_failure_count']} | "
                f"categories available={self.notification_summary['categories_available']}"
            ),
            (
                f"- Active suppression keys={self.notification_summary['active_key_count']} | "
                f"delivered once keys={self.notification_summary['delivered_once_key_count']}"
            ),
            "",
            "Attention",
        ]
        workflow_input_summaries = self.daily_review.get("workflow_input_summaries", {})
        if not isinstance(workflow_input_summaries, Mapping) or not workflow_input_summaries:
            review_inputs = self.daily_review.get("review_inputs", {})
            if isinstance(review_inputs, Mapping) and review_inputs:
                workflow_input_summaries = summarize_named_workflow_input_statuses(
                    review_inputs
                )
        workflow_input_lines = render_workflow_input_summary_lines(
            workflow_input_summaries
            if isinstance(workflow_input_summaries, Mapping)
            else None
        )
        if workflow_input_lines:
            lines[-2:-2] = [""] + workflow_input_lines
        if self.anomaly_flags:
            lines.extend(
                f"- [{flag['severity'].upper()}] {flag['message']}"
                for flag in self.anomaly_flags
            )
        else:
            lines.append("- No anomaly flags for this review window.")
        if self.paper_integrity.get("integrity_warnings"):
            lines.extend(["", "Integrity warnings"])
            lines.extend(
                f"- {warning}"
                for warning in self.paper_integrity["integrity_warnings"]
            )
        if self.paper_integrity.get("evidence_notes"):
            lines.extend(["", "Evidence notes"])
            lines.extend(
                f"- {note}"
                for note in self.paper_integrity["evidence_notes"]
            )
        return "\n".join(lines)

    def to_changes_text(self) -> str:
        lines = [
            f"What changed on {self.as_of_date.isoformat()}",
            "",
            "Actions",
        ]
        control_counts = self.change_summary.get("control_actions_by_command", {})
        if control_counts:
            lines.extend(
                f"- {command}: {count}"
                for command, count in sorted(control_counts.items())
            )
        else:
            lines.append("- No control actions recorded in this window.")
        broker_counts = self.change_summary.get("broker_events_by_type", {})
        if broker_counts:
            lines.extend(["", "Broker/order events"])
            lines.extend(
                f"- {event_type}: {count}"
                for event_type, count in sorted(broker_counts.items())
            )
        important_broker_events = self.change_summary.get("important_broker_events", [])
        if important_broker_events:
            lines.extend(["", "Recent broker/order activity"])
            lines.extend(
                (
                    f"- {item['timestamp_utc'] or item['as_of_date'] or 'n/a'} "
                    f"{item['event_type']} {item['symbol']} via {item['workflow']}"
                )
                for item in important_broker_events
            )
        repeated = self.change_summary.get("repeated_warnings", [])
        if repeated:
            lines.extend(["", "Repeated warnings"])
            lines.extend(
                f"- {item['count']}x {item['message']}"
                for item in repeated
            )
        failures = self.change_summary.get("failures", [])
        if failures:
            lines.extend(["", "Failures"])
            lines.extend(
                f"- {item['requested_at_utc']} {item['command_name']} -> {item['status']} ({item['error_code'] or 'no_error_code'})"
                for item in failures
            )
        transitions = self.change_summary.get("state_transitions", [])
        if transitions:
            lines.extend(["", "State transitions"])
            lines.extend(
                f"- {item['symbol'] or 'n/a'} {item['action'] or 'n/a'} ({item['transition_type'] or 'n/a'})"
                for item in transitions
            )
        return "\n".join(lines)

    def to_checklist_text(self) -> str:
        lines = [
            f"Paper validation operator checklist for {self.as_of_date.isoformat()}",
            "",
        ]
        lines.extend(
            f"[{item['status'].upper()}] {item['item']}: {item['detail']}"
            for item in self.operator_checklist
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class LocalPaperValidationSmokeResult:
    """Compact Day-1 bounded smoke-run verdict for the paper-validation stack."""

    generated_at_utc: str
    as_of_date: date
    max_iterations: int | None
    passed: bool
    startup_succeeded: bool
    runtime_stopped_cleanly: bool
    api_available: bool
    internal_api_available: bool
    control_api_available: bool
    dashboard_available: bool
    service_status_artifact_present: bool
    successful_cycle_count: int
    last_successful_cycle_at_utc: str | None
    review_artifacts_written: bool
    paper_only_intact: bool
    execution_mode_current: str | None
    execution_submission_enabled: bool | None
    broker_trading_enabled: bool | None
    warning_count: int
    anomaly_flag_count: int
    blocking_issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    probe_results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "as_of_date": self.as_of_date.isoformat(),
            "max_iterations": self.max_iterations,
            "passed": self.passed,
            "startup_succeeded": self.startup_succeeded,
            "runtime_stopped_cleanly": self.runtime_stopped_cleanly,
            "api_available": self.api_available,
            "internal_api_available": self.internal_api_available,
            "control_api_available": self.control_api_available,
            "dashboard_available": self.dashboard_available,
            "service_status_artifact_present": self.service_status_artifact_present,
            "successful_cycle_count": self.successful_cycle_count,
            "last_successful_cycle_at_utc": self.last_successful_cycle_at_utc,
            "review_artifacts_written": self.review_artifacts_written,
            "paper_only_intact": self.paper_only_intact,
            "execution_mode_current": self.execution_mode_current,
            "execution_submission_enabled": self.execution_submission_enabled,
            "broker_trading_enabled": self.broker_trading_enabled,
            "warning_count": self.warning_count,
            "anomaly_flag_count": self.anomaly_flag_count,
            "blocking_issues": list(self.blocking_issues),
            "warnings": list(self.warnings),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "probe_results": dict(self.probe_results),
        }

    def to_brief(self) -> str:
        lines = [
            f"Local paper validation smoke run for {self.as_of_date.isoformat()}",
            "",
            (
                f"- Passed={self.passed} | startup_succeeded={self.startup_succeeded} | "
                f"runtime_stopped_cleanly={self.runtime_stopped_cleanly}"
            ),
            (
                f"- API available={self.api_available} | "
                f"internal_api={self.internal_api_available} | "
                f"control_api={self.control_api_available} | "
                f"dashboard={self.dashboard_available}"
            ),
            (
                f"- Successful cycles={self.successful_cycle_count} | "
                f"last successful cycle={self.last_successful_cycle_at_utc or 'n/a'}"
            ),
            (
                f"- Review artifacts written={self.review_artifacts_written} | "
                f"service status artifact present={self.service_status_artifact_present}"
            ),
            (
                f"- Paper only intact={self.paper_only_intact} | "
                f"execution_mode={self.execution_mode_current or 'n/a'} | "
                f"execution_submission_enabled={self.execution_submission_enabled} | "
                f"broker_trading_enabled={self.broker_trading_enabled}"
            ),
            (
                f"- Warning count={self.warning_count} | "
                f"anomaly flags={self.anomaly_flag_count} | "
                f"blocking issues={len(self.blocking_issues)}"
            ),
        ]
        if self.blocking_issues:
            lines.extend(["", "Blocking issues"])
            lines.extend(f"- {issue}" for issue in self.blocking_issues)
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings[:10])
        return "\n".join(lines)


def build_local_paper_validation_smoke_result(
    profile: LocalPaperValidationProfile,
    *,
    as_of_date: date,
    websocket_url: str | None,
    stream_provider: str | None,
    max_iterations: int | None,
    summary: LocalPaperValidationSummary | None,
    review_paths: Mapping[str, Path],
    internal_api_probe: Mapping[str, Any],
    control_api_probe: Mapping[str, Any],
    dashboard_probe: Mapping[str, Any],
    startup_succeeded: bool,
    runtime_stopped_cleanly: bool,
    runtime_failure_message: str | None = None,
    review_failure_message: str | None = None,
    summary_failure_message: str | None = None,
) -> LocalPaperValidationSmokeResult:
    """Build a compact pass/fail verdict for one bounded Day-1 smoke run."""

    internal_api_available = bool(internal_api_probe.get("available"))
    control_api_available = bool(control_api_probe.get("available"))
    dashboard_available = bool(dashboard_probe.get("available"))
    api_available = internal_api_available and control_api_available
    required_review_labels = (
        "paper_validation_summary_json",
        "paper_validation_checkpoint_json",
        "paper_validation_brief",
    )
    review_artifacts_written = all(
        review_paths.get(label) is not None and review_paths[label].exists()
        for label in required_review_labels
    )
    service_status_path = (
        Path(summary.artifact_paths["live_market_service_status"])
        if summary is not None
        and summary.artifact_paths.get("live_market_service_status") is not None
        else LiveMarketServiceStatusStore(profile.runtime_root).path
    )
    service_status_artifact_present = service_status_path.exists()
    health_checkpoint = summary.health_checkpoint if summary is not None else {}
    safety_summary = summary.safety if summary is not None else {}
    paper_integrity = summary.paper_integrity if summary is not None else {}
    successful_cycle_count = _coerce_int(health_checkpoint.get("cycle_count"))
    last_successful_cycle_at_utc = health_checkpoint.get("last_successful_flush_at_utc")
    execution_mode_current = safety_summary.get("current_execution_mode")
    execution_submission_enabled = safety_summary.get("execution_submission_enabled")
    broker_trading_enabled = safety_summary.get("broker_trading_enabled")
    paper_only_intact = bool(paper_integrity.get("paper_only_intact"))
    blocking_issues: list[str] = []
    smoke_warnings: list[str] = []
    if runtime_failure_message:
        blocking_issues.append(runtime_failure_message)
    if not startup_succeeded:
        blocking_issues.append("Smoke runtime did not complete startup successfully.")
    if review_failure_message:
        blocking_issues.append(review_failure_message)
    if summary_failure_message:
        blocking_issues.append(summary_failure_message)
    if not internal_api_available:
        blocking_issues.append("Internal API health endpoint was not reachable during the smoke run.")
    if not control_api_available:
        blocking_issues.append("Operator control safety endpoint was not reachable during the smoke run.")
    if not dashboard_available:
        blocking_issues.append("Operator dashboard was not reachable during the smoke run.")
    if not service_status_artifact_present:
        blocking_issues.append("Live service status artifact was not written.")
    if not review_artifacts_written:
        blocking_issues.append(
            "Required paper-validation review artifacts were not written."
        )
    if successful_cycle_count < 1 or not last_successful_cycle_at_utc:
        if _alpaca_test_stream_smoke_cycle_gap_is_non_blocking(
            websocket_url=websocket_url,
            stream_provider=stream_provider,
            health_checkpoint=health_checkpoint,
        ):
            smoke_warnings.append(
                "No successful market cycle completed during the bounded Alpaca test-stream "
                "smoke run. Websocket transport health was proven, but the historical/"
                "recommendation path remains unvalidated."
            )
        else:
            blocking_issues.append(
                "No successful market cycle completed during the smoke run."
            )
    if execution_mode_current != "paper":
        blocking_issues.append(
            f"Execution mode drifted out of paper mode: current_execution_mode={execution_mode_current!r}."
        )
    if broker_trading_enabled is True:
        blocking_issues.append(
            "Broker trading was enabled during the smoke run."
        )
    if summary is None:
        blocking_issues.append(
            "Paper-validation summary was not available for the smoke run verdict."
        )
    elif not bool(safety_summary.get("paper_guardrail_active")):
        blocking_issues.append(
            "Paper-validation guardrail was not active in the smoke-run summary."
        )
    elif not bool(safety_summary.get("control_state_readable")):
        blocking_issues.append(
            "Final operator control state was unreadable at the end of the smoke run."
        )
    elif not paper_only_intact:
        blocking_issues.append(
            "Paper-only integrity did not hold for the smoke run."
        )
    if not runtime_stopped_cleanly:
        blocking_issues.append("Smoke runtime did not stop cleanly.")

    warning_messages = list(summary.warnings) if summary is not None else []
    warning_messages.extend(smoke_warnings)
    if summary is not None:
        warning_messages.extend(
            str(flag.get("message"))
            for flag in summary.anomaly_flags
            if str(flag.get("severity")) != "critical" and flag.get("message")
        )
        warning_messages.extend(
            str(note)
            for note in paper_integrity.get("evidence_notes", ())
            if isinstance(note, str)
        )
    deduped_warnings = tuple(_dedupe_texts(warning_messages))
    artifact_paths = {}
    if summary is not None:
        artifact_paths.update(summary.artifact_paths)
    artifact_paths.update(
        {
            label: str(path.resolve())
            for label, path in sorted(review_paths.items())
        }
    )
    return LocalPaperValidationSmokeResult(
        generated_at_utc=_utc_now_isoformat(datetime.now(timezone.utc)),
        as_of_date=as_of_date,
        max_iterations=max_iterations,
        passed=not blocking_issues,
        startup_succeeded=startup_succeeded,
        runtime_stopped_cleanly=runtime_stopped_cleanly,
        api_available=api_available,
        internal_api_available=internal_api_available,
        control_api_available=control_api_available,
        dashboard_available=dashboard_available,
        service_status_artifact_present=service_status_artifact_present,
        successful_cycle_count=successful_cycle_count,
        last_successful_cycle_at_utc=last_successful_cycle_at_utc,
        review_artifacts_written=review_artifacts_written,
        paper_only_intact=paper_only_intact,
        execution_mode_current=execution_mode_current,
        execution_submission_enabled=execution_submission_enabled,
        broker_trading_enabled=broker_trading_enabled,
        warning_count=(
            _coerce_int(summary.log_summary.get("warning_count"))
            if summary is not None
            else 0
        ),
        anomaly_flag_count=(len(summary.anomaly_flags) if summary is not None else 0),
        blocking_issues=tuple(blocking_issues),
        warnings=deduped_warnings,
        artifact_paths=artifact_paths,
        probe_results={
            "internal_api": dict(internal_api_probe),
            "control_api": dict(control_api_probe),
            "dashboard": dict(dashboard_probe),
        },
    )


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
    broker_records = [
        record
        for record in control_records
        if record.command_name
        in {
            "submit_pending_order",
            "cancel_pending_order",
            "replace_pending_order",
            "force_broker_sync",
        }
    ]
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
    important_broker_events = _important_broker_event_summaries(feedback_events_window)
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

    notification_summary = _build_notification_summary(
        env_file=profile.runtime_root / ".env",
        notification_state=notification_state,
        log_summary=log_summary,
    )

    health_checkpoint = _build_health_checkpoint_payload(health_payload)
    recent_transitions = _recent_transition_summaries(market_state_transitions_payload)
    market_state_summary = _build_market_state_summary(
        market_state_payload,
        recent_transitions=recent_transitions,
    )
    pending_orders_summary = _build_pending_orders_summary(pending_orders_payload)
    review_input_statuses = _load_review_input_statuses(
        profile,
        as_of_date=resolved_as_of_date,
    )
    workflow_input_summaries = _load_workflow_input_summaries(
        profile,
        as_of_date=resolved_as_of_date,
    )
    paper_integrity = _build_paper_integrity_summary(
        current_safety_state=safety_state,
        safety_payload=safety_payload,
        control_records=control_records,
        broker_records=broker_records,
        live_mode_attempt_records=live_mode_attempt_records,
        live_mode_success_count=live_mode_success_count,
        journal_events=journal_events,
        journal_events_window=journal_events_window,
        window_start=window_start,
        as_of_date=resolved_as_of_date,
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
        "paper_only_intact": paper_integrity["paper_only_intact"],
        "live_mode_attempt_count": len(live_mode_attempt_records),
        "live_mode_success_count": live_mode_success_count,
        "current_state": safety_state,
        "warnings": list(safety_payload.get("warnings", ())),
    }
    change_summary = _build_change_summary(
        as_of_date=resolved_as_of_date,
        control_counts_by_command=control_counts_by_command,
        failed_control_records=failed_control_records,
        recent_transitions=recent_transitions,
        feedback_events_window=feedback_events_window,
        feedback_counts_by_type=feedback_counts_by_type,
        top_warning_messages=log_summary["top_warning_messages"],
        pending_orders_summary=pending_orders_summary,
    )
    anomaly_flags = _build_anomaly_flags(
        health_checkpoint=health_checkpoint,
        runtime_journal_summary=runtime_journal_summary,
        control_actions_summary={
            "broker_sync_drift_count": broker_sync_drift_count,
            "failed_or_rejected_count": len(failed_control_records),
        },
        log_summary=log_summary,
        paper_integrity=paper_integrity,
    )
    daily_review = _build_daily_review(
        as_of_date=resolved_as_of_date,
        health_checkpoint=health_checkpoint,
        runtime_journal_summary=runtime_journal_summary,
        control_actions_summary={
            "window_action_count": len(control_records_window),
            "failed_or_rejected_count": len(failed_control_records),
            "broker_sync_drift_count": broker_sync_drift_count,
        },
        trade_feedback_summary={
            "broker_submitted_count": feedback_counts_by_type.get("broker_submitted", 0),
            "broker_cancelled_count": feedback_counts_by_type.get("broker_cancelled", 0),
            "broker_replaced_count": feedback_counts_by_type.get("broker_replaced", 0),
            "broker_expired_count": feedback_counts_by_type.get("broker_expired", 0),
            "executed_count": feedback_counts_by_type.get("executed", 0),
        },
        notification_summary=notification_summary,
        pending_orders_summary=pending_orders_summary,
        log_summary=log_summary,
        paper_integrity=paper_integrity,
        anomaly_flags=anomaly_flags,
        review_input_statuses=review_input_statuses,
        workflow_input_summaries=workflow_input_summaries,
    )
    operator_checklist = _build_operator_checklist(
        health_checkpoint=health_checkpoint,
        notification_summary=notification_summary,
        pending_orders_summary=pending_orders_summary,
        control_actions_summary={
            "failed_or_rejected_count": len(failed_control_records),
            "broker_sync_drift_count": broker_sync_drift_count,
        },
        paper_integrity=paper_integrity,
        anomaly_flags=anomaly_flags,
    )

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
        paper_integrity=paper_integrity,
        anomaly_flags=tuple(anomaly_flags),
        change_summary=change_summary,
        daily_review=daily_review,
        operator_checklist=tuple(operator_checklist),
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
    daily_review_json = resolved_output_dir / "paper_validation_daily_review.json"
    daily_review_text = resolved_output_dir / "paper_validation_daily_review.txt"
    changes_text = resolved_output_dir / "paper_validation_changes_today.txt"
    checklist_text = resolved_output_dir / "paper_validation_operator_checklist.txt"
    summary_json.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_json.write_text(
        json.dumps(summary.health_checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    brief_text.write_text(summary.to_brief(), encoding="utf-8")
    daily_review_json.write_text(
        json.dumps(
            {
                "generated_at_utc": summary.generated_at_utc,
                "as_of_date": summary.as_of_date.isoformat(),
                "window_days": summary.window_days,
                "daily_review": dict(summary.daily_review),
                "paper_integrity": dict(summary.paper_integrity),
                "anomaly_flags": [dict(item) for item in summary.anomaly_flags],
                "change_summary": dict(summary.change_summary),
                "operator_checklist": [dict(item) for item in summary.operator_checklist],
                "warnings": list(summary.warnings),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    daily_review_text.write_text(summary.to_daily_review_text(), encoding="utf-8")
    changes_text.write_text(summary.to_changes_text(), encoding="utf-8")
    checklist_text.write_text(summary.to_checklist_text(), encoding="utf-8")
    return {
        "paper_validation_summary_json": summary_json.resolve(),
        "paper_validation_checkpoint_json": checkpoint_json.resolve(),
        "paper_validation_brief": brief_text.resolve(),
        "paper_validation_daily_review_json": daily_review_json.resolve(),
        "paper_validation_daily_review": daily_review_text.resolve(),
        "paper_validation_changes_today": changes_text.resolve(),
        "paper_validation_operator_checklist": checklist_text.resolve(),
    }


def write_local_paper_validation_smoke_result(
    result: LocalPaperValidationSmokeResult,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write compact Day-1 smoke-run verdict artifacts alongside review outputs."""

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    result_json = resolved_output_dir / "smoke_run_result.json"
    brief_text = resolved_output_dir / "smoke_run_brief.txt"
    result_json.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    brief_text.write_text(result.to_brief(), encoding="utf-8")
    return {
        "smoke_run_result_json": result_json.resolve(),
        "smoke_run_brief": brief_text.resolve(),
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
            "status_updated_at_utc": None,
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
            "stream_provider": None,
            "historical_provider": None,
            "reference_provider": None,
            "earnings_provider": None,
            "execution_broker": None,
            "broker_update_stream_provider": None,
            "provider_roles": {},
            "degraded_provider_roles": [],
            "unavailable_provider_roles": [],
            "subscription_status": None,
            "last_subscription_message": None,
        }
    data = health_payload.get("data")
    status = data.get("status", {}) if isinstance(data, Mapping) else {}
    if not isinstance(status, Mapping):
        status = {}
    return {
        "available": bool(health_payload.get("available")),
        "status_updated_at_utc": data.get("updated_at_utc")
        if isinstance(data, Mapping)
        else None,
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
        "stream_provider": status.get("stream_provider"),
        "historical_provider": status.get("historical_provider"),
        "reference_provider": status.get("reference_provider"),
        "earnings_provider": status.get("earnings_provider"),
        "execution_broker": status.get("execution_broker"),
        "broker_update_stream_provider": status.get("broker_update_stream_provider"),
        "provider_roles": (
            dict(status.get("provider_roles"))
            if isinstance(status.get("provider_roles"), Mapping)
            else {}
        ),
        "degraded_provider_roles": [
            str(role)
            for role in status.get("degraded_provider_roles", ())
            if isinstance(role, str) and role.strip()
        ],
        "unavailable_provider_roles": [
            str(role)
            for role in status.get("unavailable_provider_roles", ())
            if isinstance(role, str) and role.strip()
        ],
        "subscription_status": status.get("subscription_status"),
        "last_subscription_message": status.get("last_subscription_message"),
    }


def _alpaca_test_stream_smoke_cycle_gap_is_non_blocking(
    *,
    websocket_url: str | None,
    stream_provider: str | None,
    health_checkpoint: Mapping[str, Any],
) -> bool:
    resolved_stream_provider = _clean_text(stream_provider)
    resolved_url = _clean_text(websocket_url)
    if resolved_stream_provider != "alpaca":
        return False
    if resolved_url is None or "/v2/test" not in resolved_url.lower():
        return False
    if not bool(health_checkpoint.get("available")):
        return False
    if not bool(health_checkpoint.get("connected")):
        return False
    if _clean_text(health_checkpoint.get("subscription_status")) != "acknowledged":
        return False
    if not _clean_text(health_checkpoint.get("last_message_at_utc")):
        return False
    return _clean_text(health_checkpoint.get("last_error")) is None


def _service_feed_continuity_healthy(
    health_checkpoint: Mapping[str, Any],
) -> bool:
    service_state = _clean_text(health_checkpoint.get("service_state")) or "unknown"
    return (
        bool(health_checkpoint.get("available"))
        and service_state == "connected"
        and bool(health_checkpoint.get("connected"))
        and not bool(health_checkpoint.get("stale"))
        and bool(health_checkpoint.get("last_message_at_utc"))
        and bool(health_checkpoint.get("last_successful_flush_at_utc"))
    )


def _service_connectivity_degraded(
    health_checkpoint: Mapping[str, Any],
) -> bool:
    if not bool(health_checkpoint.get("available")):
        return False
    service_state = _clean_text(health_checkpoint.get("service_state")) or "unknown"
    if service_state in {"retrying", "stale"}:
        return True
    return service_state == "connected" and (
        not bool(health_checkpoint.get("connected"))
        or bool(health_checkpoint.get("stale"))
    )


def _build_notification_summary(
    *,
    env_file: Path,
    notification_state: Any,
    log_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "notifications_configured": _notifications_configured(env_file),
        "updated_at_utc": getattr(notification_state, "updated_at_utc", None),
        "active_key_count": len(getattr(notification_state, "active_keys", ())),
        "delivered_once_key_count": len(
            getattr(notification_state, "delivered_once_keys", {})
        ),
        "delivery_failure_count": _coerce_int(
            log_summary.get("notification_failure_count")
        ),
        "categories_available": False,
        "important_categories": [],
        "warnings": [
            (
                "Notification event categories are not persisted in current runtime "
                "artifacts; only configuration, suppression-state counts, and log "
                "failures are available."
            )
        ],
    }


def _build_market_state_summary(
    market_state_payload: Mapping[str, Any] | None,
    *,
    recent_transitions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(market_state_payload, Mapping):
        return {
            "available": False,
            "transition_count": 0,
            "baseline_established": False,
            "alertable_state_count": 0,
            "top_priority_candidate_count": 0,
            "portfolio_path": None,
            "recent_transitions": [],
        }
    market_state_data = market_state_payload.get("data")
    if not isinstance(market_state_data, Mapping):
        return {
            "available": bool(market_state_payload.get("available")),
            "transition_count": 0,
            "baseline_established": False,
            "alertable_state_count": 0,
            "top_priority_candidate_count": 0,
            "portfolio_path": None,
            "recent_transitions": list(recent_transitions),
        }
    snapshot = market_state_data.get("snapshot", {})
    alertable_states = market_state_data.get("current_alertable_states", ())
    top_priority_candidates = market_state_data.get("top_priority_candidates", ())
    return {
        "available": bool(market_state_payload.get("available")),
        "transition_count": _coerce_int(market_state_data.get("transition_count")),
        "baseline_established": bool(market_state_data.get("baseline_established")),
        "alertable_state_count": (
            len(alertable_states) if isinstance(alertable_states, Sequence) else 0
        ),
        "top_priority_candidate_count": (
            len(top_priority_candidates)
            if isinstance(top_priority_candidates, Sequence)
            else 0
        ),
        "portfolio_path": snapshot.get("portfolio_path")
        if isinstance(snapshot, Mapping)
        else None,
        "recent_transitions": list(recent_transitions),
    }


def _load_review_input_statuses(
    profile: LocalPaperValidationProfile,
    *,
    as_of_date: date,
) -> dict[str, dict[str, Any]]:
    review_root = profile.default_live_output_dir(as_of_date=as_of_date)
    review_files = {
        "portfolio_review": review_root / "portfolio_review.json",
        "intraday_review": review_root / "portfolio_review_intraday.json",
    }
    results: dict[str, dict[str, Any]] = {}
    for workflow_name, path in review_files.items():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        raw_statuses = metadata.get("review_input_statuses")
        if not isinstance(raw_statuses, Mapping):
            continue
        results[workflow_name] = {
            str(key): dict(value)
            for key, value in raw_statuses.items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
    return results


def _load_workflow_input_summaries(
    profile: LocalPaperValidationProfile,
    *,
    as_of_date: date,
) -> dict[str, dict[str, Any]]:
    review_root = profile.default_live_output_dir(as_of_date=as_of_date)
    artifact_specs = {
        "daily_summary": ("daily_summary.json", "research_input_statuses"),
        "portfolio_review": ("portfolio_review.json", "review_input_statuses"),
        "intraday_review": ("portfolio_review_intraday.json", "review_input_statuses"),
    }
    results: dict[str, dict[str, Any]] = {}
    for workflow_name, (filename, raw_status_key) in artifact_specs.items():
        path = review_root / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        stored_summary = metadata.get("workflow_input_summary")
        if isinstance(stored_summary, Mapping):
            results[workflow_name] = dict(stored_summary)
            continue
        raw_statuses = metadata.get(raw_status_key)
        if not isinstance(raw_statuses, Mapping):
            continue
        results[workflow_name] = summarize_workflow_input_statuses(
            raw_statuses
        ).to_dict()
    return results


def _build_pending_orders_summary(
    pending_orders_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    empty_summary = {
        "available": False,
        "active_order_count": 0,
        "capacity_reserving_buy_order_count": 0,
        "reserved_notional": None,
        "pending_fill_notional": None,
        "reserved_slot_count": None,
        "available_slots": None,
        "position_context_available": False,
        "status_counts": {},
        "broker_status_counts": {},
        "pending_symbols": [],
        "unusual_state_count": 0,
        "unusual_states": [],
    }
    if not isinstance(pending_orders_payload, Mapping):
        return empty_summary
    pending_data = pending_orders_payload.get("data")
    if not isinstance(pending_data, Mapping):
        return {
            **empty_summary,
            "available": bool(pending_orders_payload.get("available")),
        }
    summary = pending_data.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    active_orders = pending_data.get("active_orders", ())
    if not isinstance(active_orders, Sequence):
        active_orders = ()
    status_counts = Counter()
    broker_status_counts = Counter()
    pending_cancel_count = 0
    partially_filled_count = 0
    broker_link_missing_count = 0
    for item in active_orders:
        if not isinstance(item, Mapping):
            continue
        status = _clean_text(item.get("status")) or "unknown"
        broker_status = _clean_text(item.get("broker_status"))
        status_counts[status] += 1
        if broker_status is not None:
            broker_status_counts[broker_status] += 1
        if broker_status == "pending_cancel":
            pending_cancel_count += 1
        if status == "partially_filled":
            partially_filled_count += 1
        if (
            _clean_text(item.get("submitted_at")) is not None
            and _clean_text(item.get("broker_order_id")) is None
        ):
            broker_link_missing_count += 1
    unusual_states: list[str] = []
    if not bool(summary.get("position_context_available")):
        unusual_states.append(
            "Pending-order position context is unavailable for capacity review."
        )
    if pending_cancel_count > 0:
        unusual_states.append(
            f"{pending_cancel_count} active pending order(s) are awaiting broker cancel confirmation."
        )
    if partially_filled_count > 0:
        unusual_states.append(
            f"{partially_filled_count} active pending order(s) are partially filled."
        )
    if broker_link_missing_count > 0:
        unusual_states.append(
            f"{broker_link_missing_count} active pending order(s) have submitted_at set without a broker_order_id."
        )
    return {
        "available": bool(pending_orders_payload.get("available")),
        "active_order_count": _coerce_int(summary.get("active_order_count")),
        "capacity_reserving_buy_order_count": _coerce_int(
            summary.get("capacity_reserving_buy_order_count")
        ),
        "reserved_notional": summary.get("reserved_notional"),
        "pending_fill_notional": summary.get("pending_fill_notional"),
        "reserved_slot_count": summary.get("reserved_slot_count"),
        "available_slots": summary.get("available_slots"),
        "position_context_available": bool(summary.get("position_context_available")),
        "status_counts": dict(sorted(status_counts.items())),
        "broker_status_counts": dict(sorted(broker_status_counts.items())),
        "pending_symbols": list(summary.get("pending_symbols", ())),
        "unusual_state_count": len(unusual_states),
        "unusual_states": unusual_states,
    }


def _build_paper_integrity_summary(
    *,
    current_safety_state: Mapping[str, Any],
    safety_payload: Mapping[str, Any],
    control_records: Sequence[Any],
    broker_records: Sequence[Any],
    live_mode_attempt_records: Sequence[Any],
    live_mode_success_count: int,
    journal_events: Sequence[PaperValidationRuntimeEvent],
    journal_events_window: Sequence[PaperValidationRuntimeEvent],
    window_start: date,
    as_of_date: date,
) -> dict[str, Any]:
    observed_safety_states = [
        state
        for state in (_record_safety_state(record) for record in control_records)
        if state is not None
    ]
    observed_execution_modes = {
        _clean_text(state.get("execution_mode"))
        for state in observed_safety_states
        if _clean_text(state.get("execution_mode")) is not None
    }
    non_paper_safety_records = [
        record
        for record in control_records
        if (_record_safety_state(record) or {}).get("execution_mode") != "paper"
        and _record_safety_state(record) is not None
    ]
    broker_target_non_paper_records = [
        record
        for record in broker_records
        if (_record_safety_state(record) or {}).get("execution_mode") != "paper"
        and _record_safety_state(record) is not None
    ]
    live_trading_enabled_ever = any(
        bool(state.get("broker_trading_enabled")) for state in observed_safety_states
    ) or bool(current_safety_state.get("broker_trading_enabled"))
    live_confirmation_enabled_ever = any(
        bool(state.get("live_actions_require_confirmation"))
        for state in observed_safety_states
    ) or bool(current_safety_state.get("live_actions_require_confirmation"))
    live_confirmation_disabled_ever = any(
        state.get("live_actions_require_confirmation") is False
        for state in observed_safety_states
    ) or current_safety_state.get("live_actions_require_confirmation") is False
    degradation_from_controls = sum(
        1
        for record in control_records
        if _is_control_state_degradation_record(record)
    )
    degradation_from_journal = sum(
        event.event_type in {"runtime_safety_state_corrupted", "runtime_safety_state_deleted"}
        for event in journal_events
    )
    degradation_window_count = sum(
        1
        for record in control_records
        if _is_control_state_degradation_record(record)
        and _record_in_window(record.requested_at_utc, window_start, as_of_date)
    )
    journal_window_degradation_count = sum(
        event.event_type in {"runtime_safety_state_corrupted", "runtime_safety_state_deleted"}
        for event in journal_events_window
    )
    execution_mode_stayed_paper = (
        current_safety_state.get("execution_mode") == "paper"
        and not non_paper_safety_records
        and live_mode_success_count == 0
    )
    broker_target_stayed_paper = (
        current_safety_state.get("execution_mode") == "paper"
        and not broker_target_non_paper_records
    )
    control_state_readable = bool(safety_payload["data"]["control_state_readable"])
    integrity_warnings: list[str] = []
    evidence_notes: list[str] = []
    if not control_state_readable:
        integrity_warnings.append(
            "Current operator control state is degraded or unreadable."
        )
    if not execution_mode_stayed_paper:
        integrity_warnings.append(
            "Observed execution-mode evidence shows a non-paper state or live-mode success."
        )
    if not broker_target_stayed_paper:
        integrity_warnings.append(
            "Observed broker-mutation evidence includes a non-paper execution target."
        )
    if live_trading_enabled_ever:
        integrity_warnings.append(
            "Live broker trading was enabled in observed safety-state history."
        )
    if degradation_from_controls + degradation_from_journal > 0:
        integrity_warnings.append(
            "Control-state degradation or recovery events occurred during the proving run."
        )
        if live_mode_success_count == 0:
            evidence_notes.append(
                "Observed control-state degradation or recovery activity occurred without any "
                "observed live-mode success evidence; review the windowed degradation count to "
                "separate resilience testing from paper-mode safety drift."
            )
    if not broker_records:
        evidence_notes.append(
            "No broker mutation or broker sync commands were audited in the current runtime history; "
            "broker-target integrity is based on current backend policy enforcement, not direct broker-action observations."
        )
    if not control_records:
        evidence_notes.append(
            "No control actions were audited in the current runtime history; execution-mode integrity is based on current state plus runtime guardrails."
        )
    paper_only_intact = (
        execution_mode_stayed_paper
        and broker_target_stayed_paper
        and not live_trading_enabled_ever
        and (degradation_from_controls + degradation_from_journal) == 0
        and control_state_readable
    )
    return {
        "paper_only_intact": paper_only_intact,
        "execution_mode_current": current_safety_state.get("execution_mode"),
        "execution_mode_stayed_paper": execution_mode_stayed_paper,
        "observed_execution_modes": sorted(
            mode for mode in observed_execution_modes if mode is not None
        ),
        "broker_target_stayed_paper": broker_target_stayed_paper,
        "broker_target_evidence_limited": not broker_records,
        "observed_broker_mutation_count": len(broker_records),
        "live_mode_attempt_count": len(live_mode_attempt_records),
        "live_mode_success_count": live_mode_success_count,
        "live_trading_enabled_ever": live_trading_enabled_ever,
        "live_confirmation_enabled_ever": live_confirmation_enabled_ever,
        "live_confirmation_disabled_ever": live_confirmation_disabled_ever,
        "control_state_readable": control_state_readable,
        "control_state_degradation_count": (
            degradation_from_controls + degradation_from_journal
        ),
        "control_state_degradation_window_count": (
            degradation_window_count + journal_window_degradation_count
        ),
        "integrity_warnings": integrity_warnings,
        "evidence_notes": evidence_notes,
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
        "log_rotation_count": counts.get("runtime_logs_rotated", 0),
        "env_override_count": counts.get("runtime_env_overridden", 0),
        "safety_state_corruption_count": counts.get("runtime_safety_state_corrupted", 0),
        "safety_state_deletion_count": counts.get("runtime_safety_state_deleted", 0),
        "safety_state_reset_count": counts.get("runtime_safety_state_reset", 0),
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
        "stale_feed_warning_count": sum(
            "websocket feed stale:" in record.message for record in warning_records
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


def _build_change_summary(
    *,
    as_of_date: date,
    control_counts_by_command: Mapping[str, int],
    failed_control_records: Sequence[dict[str, Any]],
    recent_transitions: Sequence[dict[str, Any]],
    feedback_events_window: Sequence[TradeFeedbackEvent],
    feedback_counts_by_type: Mapping[str, int],
    top_warning_messages: Sequence[dict[str, Any]],
    pending_orders_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "as_of_date": as_of_date.isoformat(),
        "control_actions_by_command": dict(sorted(control_counts_by_command.items())),
        "broker_events_by_type": {
            key: int(feedback_counts_by_type.get(key, 0))
            for key in (
                "broker_submitted",
                "broker_cancelled",
                "broker_replaced",
                "broker_expired",
                "executed",
            )
            if int(feedback_counts_by_type.get(key, 0)) > 0
        },
        "repeated_warnings": [dict(item) for item in top_warning_messages[:5]],
        "failures": [dict(item) for item in failed_control_records[:5]],
        "state_transitions": [dict(item) for item in recent_transitions[:5]],
        "important_broker_events": _important_broker_event_summaries(feedback_events_window),
        "pending_order_attention": list(pending_orders_summary.get("unusual_states", ())),
    }


def _build_daily_review(
    *,
    as_of_date: date,
    health_checkpoint: Mapping[str, Any],
    runtime_journal_summary: Mapping[str, Any],
    control_actions_summary: Mapping[str, Any],
    trade_feedback_summary: Mapping[str, Any],
    notification_summary: Mapping[str, Any],
    pending_orders_summary: Mapping[str, Any],
    log_summary: Mapping[str, Any],
    paper_integrity: Mapping[str, Any],
    anomaly_flags: Sequence[Mapping[str, Any]],
    review_input_statuses: Mapping[str, Any] | None = None,
    workflow_input_summaries: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feed_continuity_healthy = _service_feed_continuity_healthy(health_checkpoint)
    connectivity_degraded = _service_connectivity_degraded(health_checkpoint)
    resolved_workflow_input_summaries = (
        dict(workflow_input_summaries)
        if isinstance(workflow_input_summaries, Mapping)
        else {}
    )
    return {
        "as_of_date": as_of_date.isoformat(),
        "service": {
            "healthy": feed_continuity_healthy,
            "feed_continuity_healthy": feed_continuity_healthy,
            "connectivity_degraded": connectivity_degraded,
            "status_artifact_available": bool(health_checkpoint.get("available")),
            "service_state": health_checkpoint.get("service_state"),
            "status_updated_at_utc": health_checkpoint.get("status_updated_at_utc"),
            "estimated_uptime_seconds": runtime_journal_summary.get(
                "estimated_uptime_seconds"
            ),
            "restart_count": runtime_journal_summary.get("restart_count"),
            "last_message_at_utc": health_checkpoint.get("last_message_at_utc"),
            "last_successful_cycle_at_utc": health_checkpoint.get(
                "last_successful_flush_at_utc"
            ),
            "last_cycle_status": health_checkpoint.get("last_cycle_status"),
            "last_warning": health_checkpoint.get("last_warning"),
        },
        "connectivity": {
            "reconnect_attempt_count": health_checkpoint.get("reconnect_attempt_count"),
            "manual_disconnect_count": runtime_journal_summary.get(
                "manual_disconnect_count"
            ),
            "manual_reconnect_count": runtime_journal_summary.get(
                "manual_reconnect_count"
            ),
            "disconnect_warning_count": _coerce_int(
                log_summary.get("disconnect_warning_count")
            ),
            "stale_feed_warning_count": _coerce_int(
                log_summary.get("stale_feed_warning_count")
            ),
            "connect_failure_count": _coerce_int(
                log_summary.get("connect_failure_count")
            ),
        },
        "notifications": {
            "notifications_configured": notification_summary.get(
                "notifications_configured"
            ),
            "active_key_count": notification_summary.get("active_key_count"),
            "delivered_once_key_count": notification_summary.get(
                "delivered_once_key_count"
            ),
            "delivery_failure_count": notification_summary.get(
                "delivery_failure_count"
            ),
            "important_categories": list(
                notification_summary.get("important_categories", ())
            ),
            "categories_available": notification_summary.get("categories_available"),
        },
        "pending_orders": {
            "active_order_count": pending_orders_summary.get("active_order_count"),
            "unusual_state_count": pending_orders_summary.get("unusual_state_count"),
            "unusual_states": list(pending_orders_summary.get("unusual_states", ())),
        },
        "broker_activity": dict(trade_feedback_summary),
        "control": {
            "window_action_count": control_actions_summary.get("window_action_count"),
            "failed_or_rejected_count": control_actions_summary.get(
                "failed_or_rejected_count"
            ),
            "broker_sync_drift_count": control_actions_summary.get(
                "broker_sync_drift_count"
            ),
        },
        "paper_integrity": {
            "paper_only_intact": paper_integrity.get("paper_only_intact"),
            "execution_mode_stayed_paper": paper_integrity.get(
                "execution_mode_stayed_paper"
            ),
            "broker_target_stayed_paper": paper_integrity.get(
                "broker_target_stayed_paper"
            ),
            "live_trading_enabled_ever": paper_integrity.get(
                "live_trading_enabled_ever"
            ),
            "control_state_degradation_count": paper_integrity.get(
                "control_state_degradation_count"
            ),
            "control_state_degradation_window_count": paper_integrity.get(
                "control_state_degradation_window_count"
            ),
        },
        "review_inputs": (
            dict(review_input_statuses)
            if isinstance(review_input_statuses, Mapping)
            else {}
        ),
        "workflow_input_summaries": resolved_workflow_input_summaries,
        "workflow_input_overview": build_workflow_input_overview(
            resolved_workflow_input_summaries
        ).to_dict(),
        "anomaly_flag_count": len(anomaly_flags),
    }


def _build_operator_checklist(
    *,
    health_checkpoint: Mapping[str, Any],
    notification_summary: Mapping[str, Any],
    pending_orders_summary: Mapping[str, Any],
    control_actions_summary: Mapping[str, Any],
    paper_integrity: Mapping[str, Any],
    anomaly_flags: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    checklist: list[dict[str, str]] = []
    service_healthy = _service_feed_continuity_healthy(health_checkpoint)
    checklist.append(
        {
            "item": "Service health",
            "status": "pass" if service_healthy else "warn",
            "detail": (
                f"feed_continuity_healthy={service_healthy} "
                f"state={health_checkpoint.get('service_state')} "
                f"last_message_at_utc={health_checkpoint.get('last_message_at_utc') or 'n/a'} "
                f"last_successful_cycle={health_checkpoint.get('last_successful_flush_at_utc') or 'n/a'}"
            ),
        }
    )
    drift_count = _coerce_int(control_actions_summary.get("broker_sync_drift_count"))
    checklist.append(
        {
            "item": "Broker drift and reconciliation",
            "status": "pass" if drift_count == 0 else "warn",
            "detail": f"broker_sync_drift_signals={drift_count}",
        }
    )
    integrity_ok = bool(paper_integrity.get("paper_only_intact"))
    checklist.append(
        {
            "item": "Safety boundaries",
            "status": "pass" if integrity_ok else "fail",
            "detail": (
                f"paper_only_intact={paper_integrity.get('paper_only_intact')} "
                f"control_state_degradation_count={paper_integrity.get('control_state_degradation_count')} "
                f"control_state_degradation_window_count={paper_integrity.get('control_state_degradation_window_count')}"
            ),
        }
    )
    notifications_configured = bool(notification_summary.get("notifications_configured"))
    notification_failures = _coerce_int(notification_summary.get("delivery_failure_count"))
    checklist.append(
        {
            "item": "Alerts and notifications",
            "status": (
                "warn"
                if notification_failures > 0
                else ("pass" if notifications_configured else "needs_review")
            ),
            "detail": (
                f"configured={notifications_configured} "
                f"delivery_failures={notification_failures}"
            ),
        }
    )
    unusual_pending = _coerce_int(pending_orders_summary.get("unusual_state_count"))
    checklist.append(
        {
            "item": "Pending-order and broker state coherence",
            "status": "pass" if unusual_pending == 0 else "warn",
            "detail": (
                f"active_orders={pending_orders_summary.get('active_order_count')} "
                f"unusual_states={unusual_pending}"
            ),
        }
    )
    checklist.append(
        {
            "item": "Dashboard/API truth surfaces",
            "status": (
                "pass"
                if bool(health_checkpoint.get("available"))
                and bool(paper_integrity.get("control_state_readable"))
                else "warn"
            ),
            "detail": (
                f"health_available={health_checkpoint.get('available')} "
                f"control_state_readable={paper_integrity.get('control_state_readable')}"
            ),
        }
    )
    if anomaly_flags:
        checklist.append(
            {
                "item": "Follow-up required",
                "status": "warn",
                "detail": f"{len(anomaly_flags)} anomaly flag(s) need review.",
            }
        )
    return checklist


def _build_anomaly_flags(
    *,
    health_checkpoint: Mapping[str, Any],
    runtime_journal_summary: Mapping[str, Any],
    control_actions_summary: Mapping[str, Any],
    log_summary: Mapping[str, Any],
    paper_integrity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if _service_connectivity_degraded(health_checkpoint):
        flags.append(
            {
                "code": "feed_continuity_failed",
                "severity": "critical",
                "message": (
                    "Feed continuity is degraded: "
                    f"service_state={health_checkpoint.get('service_state')}, "
                    f"connected={health_checkpoint.get('connected')}, "
                    f"stale={health_checkpoint.get('stale')}, "
                    f"last_message_at_utc={health_checkpoint.get('last_message_at_utc') or 'n/a'}, "
                    f"last_successful_cycle_at_utc={health_checkpoint.get('last_successful_flush_at_utc') or 'n/a'}."
                ),
            }
        )
    reconnect_attempts = _coerce_int(health_checkpoint.get("reconnect_attempt_count"))
    disconnect_warnings = _coerce_int(log_summary.get("disconnect_warning_count"))
    connect_failures = _coerce_int(log_summary.get("connect_failure_count"))
    if disconnect_warnings >= 3 or connect_failures >= 2:
        flags.append(
            {
                "code": "repeated_reconnects",
                "severity": "warning",
                "message": (
                    f"Repeated connectivity churn detected: reconnect_attempts={reconnect_attempts}, "
                    f"disconnect_warnings={disconnect_warnings}, connect_failures={connect_failures}."
                ),
            }
        )
    drift_count = _coerce_int(control_actions_summary.get("broker_sync_drift_count"))
    if drift_count >= 2:
        flags.append(
            {
                "code": "repeated_broker_sync_warnings",
                "severity": "warning",
                "message": f"Broker sync reported drift or warnings {drift_count} times in the review window.",
            }
        )
    malformed_recovery_count = (
        _coerce_int(runtime_journal_summary.get("window_counts", {}).get("runtime_safety_state_corrupted"))
        + _coerce_int(runtime_journal_summary.get("window_counts", {}).get("runtime_safety_state_deleted"))
        + _coerce_int(runtime_journal_summary.get("window_counts", {}).get("runtime_safety_state_reset"))
    )
    if malformed_recovery_count >= 2:
        flags.append(
            {
                "code": "repeated_malformed_state_recoveries",
                "severity": "warning",
                "message": (
                    f"Safety-state corruption/deletion/reset activity occurred {malformed_recovery_count} times in the review window."
                ),
            }
        )
    notification_failures = _coerce_int(log_summary.get("notification_failure_count"))
    if notification_failures > 0:
        flags.append(
            {
                "code": "notification_delivery_failures",
                "severity": "warning",
                "message": f"Notification routing logged {notification_failures} delivery failure(s).",
            }
        )
    if not bool(paper_integrity.get("paper_only_intact")):
        flags.append(
            {
                "code": "paper_integrity_warning",
                "severity": "critical",
                "message": "Paper-only integrity did not hold for the observed review evidence.",
            }
        )
    restart_count = _coerce_int(
        runtime_journal_summary.get("window_counts", {}).get("runtime_restarted")
    )
    if restart_count >= 3:
        flags.append(
            {
                "code": "too_many_restarts",
                "severity": "warning",
                "message": f"The runtime restarted {restart_count} times in the review window.",
            }
        )
    return flags


def _important_broker_event_summaries(
    feedback_events: Sequence[TradeFeedbackEvent],
) -> list[dict[str, Any]]:
    important_types = {
        "broker_submitted",
        "broker_cancelled",
        "broker_replaced",
        "broker_expired",
        "executed",
    }
    summaries: list[dict[str, Any]] = []
    sorted_events = sorted(
        feedback_events,
        key=lambda event: (event.timestamp_utc or "", event.as_of_date or date.min),
        reverse=True,
    )
    for event in sorted_events:
        if event.event_type not in important_types:
            continue
        summaries.append(
            {
                "event_type": event.event_type,
                "workflow": event.workflow,
                "symbol": event.symbol,
                "timestamp_utc": event.timestamp_utc,
                "as_of_date": (
                    event.as_of_date.isoformat() if event.as_of_date is not None else None
                ),
            }
        )
        if len(summaries) >= 10:
            break
    return summaries


def _record_safety_state(record: Any) -> Mapping[str, Any] | None:
    raw_state = getattr(record, "result_payload", {}).get("safety_state")
    if not isinstance(raw_state, Mapping):
        return None
    return raw_state


def _is_control_state_degradation_record(record: Any) -> bool:
    if getattr(record, "error_code", None) == "control_state_unavailable":
        return True
    for warning in getattr(record, "warnings", ()):
        if not isinstance(warning, str):
            continue
        lowered = warning.lower()
        if "control state" in lowered and (
            "unavailable" in lowered or "unreadable" in lowered
        ):
            return True
    return False


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


def _dedupe_texts(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return tuple(deduped)


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now_isoformat(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat()
