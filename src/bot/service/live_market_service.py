"""Long-running live market service and supervision loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Callable

from bot.ingestion.streaming import WebsocketIngestionAdapter
from bot.logging_utils import get_logger
from bot.orchestration.live_runner import (
    LiveMarketCycleRequest,
    LiveMarketCycleResult,
    LiveMarketRunner,
)


LOGGER = get_logger(__name__)


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
    last_connected_at_utc: str | None = None
    last_successful_flush_at_utc: str | None = None
    last_warning: str | None = None
    last_error: str | None = None
    last_cycle_status: str | None = None
    last_cycle_warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_state": self.service_state,
            "connected": self.connected,
            "stale": self.stale,
            "reconnect_attempt_count": self.reconnect_attempt_count,
            "cycle_count": self.cycle_count,
            "warning_count": self.warning_count,
            "last_message_at_utc": self.last_message_at_utc,
            "last_connected_at_utc": self.last_connected_at_utc,
            "last_successful_flush_at_utc": self.last_successful_flush_at_utc,
            "last_warning": self.last_warning,
            "last_error": self.last_error,
            "last_cycle_status": self.last_cycle_status,
            "last_cycle_warning_count": self.last_cycle_warning_count,
        }


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
    sleep: Callable[[float], None] = time.sleep
    now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    flush_on_shutdown: bool = True
    handle_cycle_result: Callable[[LiveMarketCycleResult], None] | None = None
    status: LiveMarketServiceStatus = field(default_factory=LiveMarketServiceStatus, init=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be greater than or equal to zero.")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be greater than zero.")

    def stop(self) -> None:
        self._stop_requested = True

    def run_once(self) -> LiveMarketServiceStatus:
        """Run one service iteration."""

        now = self.now_utc()
        if not self.adapter.connected:
            self._attempt_connect(now)
            if not self.adapter.connected:
                self._refresh_liveness(now)
                return self.status

        poll_result = self.adapter.poll_messages(
            max_messages=self.max_messages_per_poll,
            observed_at_utc=now,
        )
        if poll_result.warning_count:
            for warning in poll_result.warnings:
                self._record_warning(warning)
        if poll_result.accepted_count > 0:
            self.status.last_message_at_utc = now.astimezone(timezone.utc).isoformat()
        if not poll_result.connected:
            self.status.connected = False
            self.status.stale = False
            self.status.reconnect_attempt_count += 1
            self.status.service_state = "retrying"
            if poll_result.warning_count:
                self.status.last_error = poll_result.warnings[-1]
            self._refresh_liveness(now)
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

        self._refresh_liveness(now)
        return self.status

    def run(self, *, max_iterations: int | None = None) -> LiveMarketServiceStatus:
        """Run the long-lived service loop until stopped or interrupted."""

        iteration_count = 0
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
        reference_timestamp = self.status.last_message_at_utc or self.status.last_connected_at_utc
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


def run_live_market_service(
    supervisor: LiveMarketSupervisor,
    *,
    max_iterations: int | None = None,
) -> LiveMarketServiceStatus:
    """Convenience wrapper for the long-running live market service."""

    return supervisor.run(max_iterations=max_iterations)
