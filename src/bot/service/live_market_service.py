"""Long-running live market service and supervision loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    coerce_positive_int,
    load_json_mapping_file,
    write_json_file,
)
from bot.ingestion.streaming import StreamingMarketDataEvent, WebsocketIngestionAdapter
from bot.logging_utils import get_logger
from bot.notifications import (
    NotificationEvent,
    NotificationRouter,
    build_service_state_notifications,
    build_service_warning_notification,
)
from bot.orchestration.live_runner import (
    LiveMarketCycleRequest,
    LiveMarketCycleResult,
    LiveMarketRunner,
)
LOGGER = get_logger(__name__)

LIVE_MARKET_SERVICE_STATUS_SCHEMA_VERSION = 1
_LIVE_MARKET_SERVICE_STATUS_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "live_market_service_status.json",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "live_market_service_status.json"),
    ),
)


@dataclass(frozen=True)
class ReconnectBackoffPolicy:
    """Bounded reconnect/backoff configuration."""

    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.initial_delay_seconds <= 0:
            raise ValueError("initial_delay_seconds must be greater than zero.")
        if self.multiplier < 1:
            raise ValueError("multiplier must be greater than or equal to one.")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be greater than or equal to initial_delay_seconds."
            )

    def delay_for_attempt(self, attempt_number: int) -> float:
        if attempt_number <= 0:
            return self.initial_delay_seconds
        delay = self.initial_delay_seconds * (self.multiplier ** (attempt_number - 1))
        return min(delay, self.max_delay_seconds)


@dataclass
class LiveMarketServiceStatus:
    """Operational state snapshot for the live market supervisor."""

    service_state: str = "starting"
    connected: bool = False
    stale: bool = False
    reconnect_attempt_count: int = 0
    cycle_count: int = 0
    warning_count: int = 0
    last_message_at_utc: str | None = None
    last_raw_message_at_utc: str | None = None
    last_accepted_message_at_utc: str | None = None
    last_connected_at_utc: str | None = None
    last_successful_flush_at_utc: str | None = None
    last_warning: str | None = None
    last_error: str | None = None
    last_cycle_status: str | None = None
    last_cycle_warning_count: int = 0
    raw_message_count: int = 0
    accepted_message_count: int = 0
    skipped_message_count: int = 0
    last_raw_message_type: str | None = None
    last_accepted_message_type: str | None = None
    last_dropped_message_reason: str | None = None
    stream_provider: str | None = None
    historical_provider: str | None = None
    reference_provider: str | None = None
    earnings_provider: str | None = None
    execution_broker: str | None = None
    broker_update_stream_provider: str | None = None
    provider_roles: dict[str, Any] = field(default_factory=dict)
    degraded_provider_roles: tuple[str, ...] = ()
    unavailable_provider_roles: tuple[str, ...] = ()
    subscription_status: str | None = None
    last_subscription_message: str | None = None
    last_subscription_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_state": self.service_state,
            "connected": self.connected,
            "stale": self.stale,
            "reconnect_attempt_count": self.reconnect_attempt_count,
            "cycle_count": self.cycle_count,
            "warning_count": self.warning_count,
            "last_message_at_utc": self.last_message_at_utc,
            "last_raw_message_at_utc": self.last_raw_message_at_utc,
            "last_accepted_message_at_utc": self.last_accepted_message_at_utc,
            "last_connected_at_utc": self.last_connected_at_utc,
            "last_successful_flush_at_utc": self.last_successful_flush_at_utc,
            "last_warning": self.last_warning,
            "last_error": self.last_error,
            "last_cycle_status": self.last_cycle_status,
            "last_cycle_warning_count": self.last_cycle_warning_count,
            "raw_message_count": self.raw_message_count,
            "accepted_message_count": self.accepted_message_count,
            "skipped_message_count": self.skipped_message_count,
            "last_raw_message_type": self.last_raw_message_type,
            "last_accepted_message_type": self.last_accepted_message_type,
            "last_dropped_message_reason": self.last_dropped_message_reason,
            "stream_provider": self.stream_provider,
            "historical_provider": self.historical_provider,
            "reference_provider": self.reference_provider,
            "earnings_provider": self.earnings_provider,
            "execution_broker": self.execution_broker,
            "broker_update_stream_provider": self.broker_update_stream_provider,
            "provider_roles": dict(self.provider_roles),
            "degraded_provider_roles": list(self.degraded_provider_roles),
            "unavailable_provider_roles": list(self.unavailable_provider_roles),
            "subscription_status": self.subscription_status,
            "last_subscription_message": self.last_subscription_message,
            "last_subscription_at_utc": self.last_subscription_at_utc,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LiveMarketServiceStatus":
        return cls(
            service_state=_optional_text(payload.get("service_state")) or "starting",
            connected=_optional_bool(payload.get("connected")) or False,
            stale=_optional_bool(payload.get("stale")) or False,
            reconnect_attempt_count=_non_negative_int(
                payload.get("reconnect_attempt_count"),
                "reconnect_attempt_count",
            ),
            cycle_count=_non_negative_int(payload.get("cycle_count"), "cycle_count"),
            warning_count=_non_negative_int(payload.get("warning_count"), "warning_count"),
            last_message_at_utc=_optional_iso8601_datetime(payload.get("last_message_at_utc")),
            last_raw_message_at_utc=_optional_iso8601_datetime(
                payload.get("last_raw_message_at_utc")
            ),
            last_accepted_message_at_utc=_optional_iso8601_datetime(
                payload.get("last_accepted_message_at_utc")
            ),
            last_connected_at_utc=_optional_iso8601_datetime(
                payload.get("last_connected_at_utc")
            ),
            last_successful_flush_at_utc=_optional_iso8601_datetime(
                payload.get("last_successful_flush_at_utc")
            ),
            last_warning=_optional_text(payload.get("last_warning")),
            last_error=_optional_text(payload.get("last_error")),
            last_cycle_status=_optional_text(payload.get("last_cycle_status")),
            last_cycle_warning_count=_non_negative_int(
                payload.get("last_cycle_warning_count"),
                "last_cycle_warning_count",
            ),
            raw_message_count=_non_negative_int(
                payload.get("raw_message_count"),
                "raw_message_count",
            ),
            accepted_message_count=_non_negative_int(
                payload.get("accepted_message_count"),
                "accepted_message_count",
            ),
            skipped_message_count=_non_negative_int(
                payload.get("skipped_message_count"),
                "skipped_message_count",
            ),
            last_raw_message_type=_optional_text(payload.get("last_raw_message_type")),
            last_accepted_message_type=_optional_text(
                payload.get("last_accepted_message_type")
            ),
            last_dropped_message_reason=_optional_text(
                payload.get("last_dropped_message_reason")
            ),
            stream_provider=_optional_text(payload.get("stream_provider")),
            historical_provider=_optional_text(payload.get("historical_provider")),
            reference_provider=_optional_text(payload.get("reference_provider")),
            earnings_provider=_optional_text(payload.get("earnings_provider")),
            execution_broker=_optional_text(payload.get("execution_broker")),
            broker_update_stream_provider=_optional_text(
                payload.get("broker_update_stream_provider")
            ),
            provider_roles=_string_mapping(payload.get("provider_roles")),
            degraded_provider_roles=_string_sequence(payload.get("degraded_provider_roles")),
            unavailable_provider_roles=_string_sequence(
                payload.get("unavailable_provider_roles")
            ),
            subscription_status=_optional_text(payload.get("subscription_status")),
            last_subscription_message=_optional_text(payload.get("last_subscription_message")),
            last_subscription_at_utc=_optional_iso8601_datetime(
                payload.get("last_subscription_at_utc")
            ),
        )


@dataclass(frozen=True)
class LiveMarketServiceStatusArtifact:
    """Persisted live-service status wrapper for external readers."""

    schema_version: int
    updated_at_utc: str | None
    status: LiveMarketServiceStatus

    def __post_init__(self) -> None:
        coerce_positive_int(self.schema_version, "schema_version")
        if self.updated_at_utc is not None:
            coerce_iso8601_datetime_text(self.updated_at_utc, "updated_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at_utc": self.updated_at_utc,
            "status": self.status.to_dict(),
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LiveMarketServiceStatusArtifact":
        raw_status = payload.get("status", {})
        if not isinstance(raw_status, dict):
            raise ValueError("status must be a mapping.")
        return cls(
            schema_version=coerce_positive_int(
                payload.get("schema_version"),
                "schema_version",
            ),
            updated_at_utc=_optional_iso8601_datetime(payload.get("updated_at_utc")),
            status=LiveMarketServiceStatus.from_mapping(raw_status),
        )


@dataclass(frozen=True)
class LiveMarketServiceStatusStore:
    """Project-scoped loader/saver for persisted live-service status."""

    project_root: Path

    @property
    def path(self) -> Path:
        return default_live_market_service_status_path(self.project_root)

    def load(self) -> LiveMarketServiceStatusArtifact:
        return load_live_market_service_status_artifact(self.path)

    def save(
        self,
        status: LiveMarketServiceStatus,
        *,
        updated_at_utc: str,
    ) -> Path:
        artifact = LiveMarketServiceStatusArtifact(
            schema_version=LIVE_MARKET_SERVICE_STATUS_SCHEMA_VERSION,
            updated_at_utc=updated_at_utc,
            status=status,
        )
        return write_json_file(self.path, artifact.to_dict())


def default_live_market_service_status_path(project_root: Path) -> Path:
    """Return the preferred repo-relative service-status artifact path."""

    return _LIVE_MARKET_SERVICE_STATUS_PATH.resolve(project_root)


def load_live_market_service_status_artifact(
    path: Path,
) -> LiveMarketServiceStatusArtifact:
    """Load live-service status, degrading gracefully on missing/corrupt files."""

    return load_json_mapping_file(
        path,
        artifact_label="live market service status",
        logger=LOGGER,
        empty_value=lambda: LiveMarketServiceStatusArtifact(
            schema_version=LIVE_MARKET_SERVICE_STATUS_SCHEMA_VERSION,
            updated_at_utc=None,
            status=LiveMarketServiceStatus(),
        ),
        build=lambda payload, _source_path: LiveMarketServiceStatusArtifact.from_mapping(
            dict(payload)
        ),
    )


@dataclass
class LiveMarketSupervisor:
    """Own the websocket transport, streaming adapter, and cycle orchestration loop."""

    runner: LiveMarketRunner
    adapter: WebsocketIngestionAdapter
    request: LiveMarketCycleRequest
    poll_interval_seconds: float = 1.0
    stale_after_seconds: float = 30.0
    max_messages_per_poll: int | None = None
    reconnect_backoff: ReconnectBackoffPolicy = field(default_factory=ReconnectBackoffPolicy)
    stream_provider: str | None = None
    historical_provider: str | None = None
    reference_provider: str | None = None
    earnings_provider: str | None = None
    execution_broker: str | None = None
    broker_update_stream_provider: str | None = None
    provider_roles: dict[str, Any] = field(default_factory=dict)
    degraded_provider_roles: Sequence[str] = ()
    unavailable_provider_roles: Sequence[str] = ()
    expect_subscription_ack: bool = False
    startup_warnings: Sequence[str] = ()
    sleep: Callable[[float], None] = time.sleep
    now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    flush_on_shutdown: bool = True
    handle_cycle_result: Callable[[LiveMarketCycleResult], None] | None = None
    status_store: LiveMarketServiceStatusStore | None = None
    notification_router: NotificationRouter | None = None
    status: LiveMarketServiceStatus = field(default_factory=LiveMarketServiceStatus, init=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _startup_warnings_emitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be greater than or equal to zero.")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be greater than zero.")
        self.status.stream_provider = _optional_text(self.stream_provider)
        self.status.historical_provider = _optional_text(self.historical_provider)
        self.status.reference_provider = _optional_text(self.reference_provider)
        self.status.earnings_provider = _optional_text(self.earnings_provider)
        self.status.execution_broker = _optional_text(self.execution_broker)
        self.status.broker_update_stream_provider = _optional_text(
            self.broker_update_stream_provider
        )
        self.status.provider_roles = dict(self.provider_roles)
        self.status.degraded_provider_roles = _string_sequence(
            self.degraded_provider_roles
        )
        self.status.unavailable_provider_roles = _string_sequence(
            self.unavailable_provider_roles
        )

    def stop(self) -> None:
        self._stop_requested = True

    def run_once(self) -> LiveMarketServiceStatus:
        """Run one service iteration."""

        now = self.now_utc()
        previous_service_state = self.status.service_state
        self._emit_startup_warnings()
        if not self.adapter.connected:
            self._attempt_connect(now)
            if not self.adapter.connected:
                self._refresh_liveness(now)
                self._notify_service_state_change(
                    previous_state=previous_service_state,
                    now=now,
                )
                self._persist_status(now)
                return self.status

        poll_result = self.adapter.poll_messages(
            max_messages=self.max_messages_per_poll,
            observed_at_utc=now,
        )
        if poll_result.warning_count:
            for warning in poll_result.warnings:
                self._record_warning(warning)
        self._apply_control_events(
            poll_result.control_events,
            observed_at_utc=now.astimezone(timezone.utc).isoformat(),
        )
        self.status.raw_message_count += poll_result.received_count
        self.status.accepted_message_count += poll_result.accepted_count
        self.status.skipped_message_count += poll_result.skipped_count
        if poll_result.last_raw_message_at_utc is not None:
            self.status.last_message_at_utc = poll_result.last_raw_message_at_utc
            self.status.last_raw_message_at_utc = poll_result.last_raw_message_at_utc
        if poll_result.last_raw_message_type is not None:
            self.status.last_raw_message_type = poll_result.last_raw_message_type
        if poll_result.accepted_count > 0:
            self.status.last_accepted_message_at_utc = now.astimezone(timezone.utc).isoformat()
        if poll_result.last_accepted_message_type is not None:
            self.status.last_accepted_message_type = poll_result.last_accepted_message_type
        if poll_result.received_count > 0:
            self.status.last_dropped_message_reason = poll_result.last_dropped_message_reason
        if not poll_result.connected:
            self.status.connected = False
            self.status.stale = False
            self.status.reconnect_attempt_count += 1
            self.status.service_state = "retrying"
            if poll_result.warning_count:
                self.status.last_error = poll_result.warnings[-1]
            self._refresh_liveness(now)
            self._notify_service_state_change(
                previous_state=previous_service_state,
                now=now,
            )
            self._persist_status(now)
            return self.status

        envelope = self.adapter.flush_ready_cycle_ingestion(self.request, now_utc=now)
        if envelope is not None:
            try:
                cycle_result = self.runner.run_cycle(self.request, ingestion=envelope)
            except Exception as exc:
                self.status.last_error = str(exc)
                self._record_warning(f"live market cycle failed: {exc}")
            else:
                self.status.cycle_count += 1
                self.status.last_successful_flush_at_utc = now.astimezone(timezone.utc).isoformat()
                self.status.last_cycle_status = cycle_result.status
                self.status.last_cycle_warning_count = cycle_result.warning_count
                if cycle_result.warning_count:
                    for warning in cycle_result.warnings:
                        self._record_warning(
                            f"cycle warning ({warning.stage}): {warning.message}"
                        )
                if self.handle_cycle_result is not None:
                    try:
                        self.handle_cycle_result(cycle_result)
                    except Exception as exc:
                        self.status.last_error = str(exc)
                        self._record_warning(f"cycle result handler failed: {exc}")

        if self._recover_from_stale_connection(now):
            self._notify_service_state_change(
                previous_state=previous_service_state,
                now=now,
            )
            self._persist_status(now)
            return self.status
        self._notify_service_state_change(
            previous_state=previous_service_state,
            now=now,
        )
        self._persist_status(now)
        return self.status

    def run(self, *, max_iterations: int | None = None) -> LiveMarketServiceStatus:
        """Run the long-lived service loop until stopped or interrupted."""

        iteration_count = 0
        self._persist_status(self.now_utc())
        try:
            while not self._stop_requested:
                self.run_once()
                iteration_count += 1
                if max_iterations is not None and iteration_count >= max_iterations:
                    break
                if self._stop_requested:
                    break
                self.sleep(self._sleep_delay_seconds())
        except KeyboardInterrupt:
            self._record_warning("KeyboardInterrupt received; stopping live market service.")
            self._stop_requested = True
        finally:
            self._shutdown()
        return self.status

    def _attempt_connect(self, now: datetime) -> None:
        try:
            self.adapter.connect()
        except OSError as exc:
            self.status.connected = False
            self.status.stale = False
            self.status.reconnect_attempt_count += 1
            self.status.service_state = "retrying"
            self.status.last_error = str(exc)
            self._record_warning(f"websocket connect failed: {exc}")
            return
        self.status.connected = True
        self.status.stale = False
        self.status.reconnect_attempt_count = 0
        self.status.last_error = None
        self.status.last_connected_at_utc = now.astimezone(timezone.utc).isoformat()
        self.status.service_state = "connected"
        if self.expect_subscription_ack:
            self.status.subscription_status = "pending"
            self.status.last_subscription_message = (
                "Awaiting Alpaca subscription acknowledgement."
            )
            self.status.last_subscription_at_utc = None

    def _refresh_liveness(self, now: datetime) -> None:
        self.status.connected = self.adapter.connected
        if not self.adapter.connected:
            if self.status.service_state != "stopped":
                self.status.service_state = (
                    "retrying"
                    if self.status.reconnect_attempt_count > 0
                    else "starting"
                )
            self.status.stale = False
            return
        reference_timestamp = (
            self.status.last_raw_message_at_utc
            or self.status.last_message_at_utc
            or self.status.last_connected_at_utc
        )
        if reference_timestamp is None:
            self.status.service_state = "connected"
            self.status.stale = False
            return
        reference = datetime.fromisoformat(reference_timestamp)
        if now - reference >= timedelta(seconds=self.stale_after_seconds):
            self.status.service_state = "stale"
            self.status.stale = True
            return
        self.status.service_state = "connected"
        self.status.stale = False

    def _recover_from_stale_connection(self, now: datetime) -> bool:
        self._refresh_liveness(now)
        if self.status.service_state != "stale" or not self.adapter.connected:
            return False
        reference_timestamp = (
            self.status.last_raw_message_at_utc
            or self.status.last_message_at_utc
            or self.status.last_connected_at_utc
        )
        stale_seconds = self.stale_after_seconds
        if reference_timestamp is not None:
            reference = datetime.fromisoformat(reference_timestamp)
            stale_seconds = max((now - reference).total_seconds(), 0.0)
        warning = (
            "websocket feed stale: no messages received for "
            f"{stale_seconds:.1f}s (threshold={self.stale_after_seconds:.1f}s); "
            "forcing reconnect."
        )
        diagnostic_suffix = _stale_feed_diagnostic_suffix(self.status)
        if diagnostic_suffix:
            warning = f"{warning} {diagnostic_suffix}"
        self._record_warning(warning)
        self.status.last_error = warning
        try:
            self.adapter.disconnect()
        except OSError as exc:
            self._record_warning(f"websocket disconnect failed: {exc}")
        self.status.connected = False
        self.status.stale = True
        self.status.reconnect_attempt_count += 1
        self.status.service_state = "retrying"
        return True

    def _sleep_delay_seconds(self) -> float:
        if self.status.service_state == "retrying":
            return self.reconnect_backoff.delay_for_attempt(
                max(self.status.reconnect_attempt_count, 1)
            )
        return self.poll_interval_seconds

    def _shutdown(self) -> None:
        if self.flush_on_shutdown:
            self._flush_on_shutdown()
        try:
            self.adapter.disconnect()
        except OSError as exc:
            self._record_warning(f"websocket disconnect failed: {exc}")
        self.status.connected = False
        self.status.stale = False
        self.status.service_state = "stopped"
        self._persist_status(self.now_utc())

    def _flush_on_shutdown(self) -> None:
        if not self.adapter.has_pending_updates(as_of_date=self.request.as_of_date):
            return
        try:
            envelope = self.adapter.build_cycle_ingestion(self.request)
        except ValueError:
            return
        except Exception as exc:
            self._record_warning(f"shutdown flush failed before cycle build: {exc}")
            return
        now = self.now_utc()
        try:
            cycle_result = self.runner.run_cycle(self.request, ingestion=envelope)
        except Exception as exc:
            self._record_warning(f"shutdown flush cycle failed: {exc}")
            self.status.last_error = str(exc)
            return
        self.status.cycle_count += 1
        self.status.last_successful_flush_at_utc = now.astimezone(timezone.utc).isoformat()
        self.status.last_cycle_status = cycle_result.status
        self.status.last_cycle_warning_count = cycle_result.warning_count
        if self.handle_cycle_result is not None:
            try:
                self.handle_cycle_result(cycle_result)
            except Exception as exc:
                self.status.last_error = str(exc)
                self._record_warning(f"cycle result handler failed: {exc}")

    def _record_warning(self, message: str) -> None:
        LOGGER.warning("%s", message)
        self.status.warning_count += 1
        self.status.last_warning = message
        if self.notification_router is None:
            return
        event = build_service_warning_notification(
            message,
            created_at=self.now_utc().astimezone(timezone.utc).isoformat(),
        )
        if event is not None:
            self._route_notifications((event,))

    def _persist_status(self, now: datetime) -> None:
        if self.status_store is None:
            return
        try:
            self.status_store.save(
                self.status,
                updated_at_utc=now.astimezone(timezone.utc).isoformat(),
            )
        except OSError as exc:
            LOGGER.warning("Failed to persist live market service status: %s", exc)

    def _notify_service_state_change(
        self,
        *,
        previous_state: str,
        now: datetime,
    ) -> None:
        if self.notification_router is None:
            return
        events = build_service_state_notifications(
            previous_state=previous_state,
            current_state=self.status.service_state,
            created_at=now.astimezone(timezone.utc).isoformat(),
        )
        if events:
            self._route_notifications(events)

    def _route_notifications(self, events: Sequence[NotificationEvent]) -> None:
        if self.notification_router is None or not events:
            return
        try:
            self.notification_router.route_events(events)
        except Exception as exc:
            LOGGER.warning("Notification routing failed: %s", exc)

    def _apply_control_events(
        self,
        control_events: Sequence[StreamingMarketDataEvent],
        *,
        observed_at_utc: str,
    ) -> None:
        for event in control_events:
            status_kind = _optional_text(event.metadata.get("status_kind"))
            if status_kind == "subscription":
                acknowledged = _optional_bool(
                    event.metadata.get("subscription_acknowledged")
                )
                self.status.subscription_status = (
                    "acknowledged" if acknowledged else "warning"
                )
                self.status.last_subscription_message = (
                    _optional_text(event.metadata.get("note"))
                    or _optional_text(event.metadata.get("warning"))
                )
                self.status.last_subscription_at_utc = observed_at_utc
                continue
            if status_kind == "error":
                self.status.subscription_status = "error"
                self.status.last_subscription_message = (
                    _optional_text(event.metadata.get("warning"))
                    or _optional_text(event.metadata.get("msg"))
                )
                self.status.last_subscription_at_utc = observed_at_utc

    def _emit_startup_warnings(self) -> None:
        if self._startup_warnings_emitted:
            return
        self._startup_warnings_emitted = True
        for warning in self.startup_warnings:
            cleaned = _optional_text(warning)
            if cleaned is not None:
                self._record_warning(cleaned)


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_iso8601_datetime(value: object) -> str | None:
    cleaned = _optional_text(value)
    if cleaned is None:
        return None
    return coerce_iso8601_datetime_text(cleaned, "datetime")


def _non_negative_int(value: object, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if coerced < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return coerced


def _string_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): nested_value
        for key, nested_value in value.items()
        if isinstance(key, str) and key.strip()
    }


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        cleaned = _optional_text(value)
        return (cleaned,) if cleaned is not None else ()
    if not isinstance(value, Sequence):
        return ()
    values: list[str] = []
    for item in value:
        cleaned = _optional_text(item)
        if cleaned is not None:
            values.append(cleaned)
    return tuple(values)


def _stale_feed_diagnostic_suffix(status: LiveMarketServiceStatus) -> str:
    details: list[str] = []
    if status.last_raw_message_type is not None:
        details.append(f"last_raw_message_type={status.last_raw_message_type}")
    if status.last_accepted_message_type is not None:
        details.append(f"last_accepted_message_type={status.last_accepted_message_type}")
    if status.last_dropped_message_reason is not None:
        details.append(f"last_dropped_message_reason={status.last_dropped_message_reason}")
    if not details:
        return ""
    return "Diagnostics: " + "; ".join(details) + "."


def run_live_market_service(
    supervisor: LiveMarketSupervisor,
    *,
    max_iterations: int | None = None,
) -> LiveMarketServiceStatus:
    """Convenience wrapper for the long-running live market service."""

    return supervisor.run(max_iterations=max_iterations)
