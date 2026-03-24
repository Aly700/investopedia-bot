"""Normalized notification routing, delivery, and suppression helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    coerce_positive_int,
    load_json_mapping_file,
    write_json_file,
)
from bot.data.trade_feedback import TradeFeedbackEvent
from bot.logging_utils import get_logger
from bot.state.market_state import MarketStateTransition


LOGGER = get_logger(__name__)

NOTIFICATION_EVENT_TYPES = (
    "state_transition",
    "broker_event",
    "execution_warning",
    "service_health_warning",
    "connectivity_warning",
    "capacity_block",
    "analytics_warning",
)
NOTIFICATION_SEVERITIES = ("info", "warning", "critical")
NOTIFICATION_SUPPRESSION_SCOPES = ("none", "once", "active")
_SEVERITY_RANK = {severity: index for index, severity in enumerate(NOTIFICATION_SEVERITIES)}

NOTIFICATION_DELIVERY_STATE_SCHEMA_VERSION = 1
_NOTIFICATION_DELIVERY_STATE_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "notification_delivery_state.json",
    ),
)


@dataclass(frozen=True)
class NotificationEvent:
    """One normalized event ready for external delivery."""

    event_type: str
    severity: str
    title: str
    body: str
    created_at: str
    symbol: str | None = None
    related_ids: dict[str, str] = field(default_factory=dict)
    dedupe_key: str | None = None
    suppression_scope: str = "none"
    clears_keys: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.event_type not in NOTIFICATION_EVENT_TYPES:
            raise ValueError(
                "event_type must be one of "
                f"{NOTIFICATION_EVENT_TYPES}, got '{self.event_type}'."
            )
        if self.severity not in NOTIFICATION_SEVERITIES:
            raise ValueError(
                "severity must be one of "
                f"{NOTIFICATION_SEVERITIES}, got '{self.severity}'."
            )
        if self.suppression_scope not in NOTIFICATION_SUPPRESSION_SCOPES:
            raise ValueError(
                "suppression_scope must be one of "
                f"{NOTIFICATION_SUPPRESSION_SCOPES}, got '{self.suppression_scope}'."
            )
        if not self.title.strip():
            raise ValueError("title cannot be empty.")
        if not self.body.strip():
            raise ValueError("body cannot be empty.")
        coerce_iso8601_datetime_text(self.created_at, "created_at")
        if self.suppression_scope != "none" and (
            self.dedupe_key is None or not self.dedupe_key.strip()
        ):
            raise ValueError(
                "dedupe_key is required when suppression_scope is not 'none'."
            )
        object.__setattr__(self, "event_id", build_notification_event_id(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "related_ids": dict(sorted(self.related_ids.items())),
            "dedupe_key": self.dedupe_key,
            "suppression_scope": self.suppression_scope,
            "clears_keys": list(self.clears_keys),
            "metadata": dict(self.metadata),
        }


def build_notification_event_id(event: NotificationEvent) -> str:
    payload = {
        "event_type": event.event_type,
        "severity": event.severity,
        "title": event.title,
        "body": event.body,
        "created_at": event.created_at,
        "symbol": event.symbol,
        "related_ids": dict(sorted(event.related_ids.items())),
        "dedupe_key": event.dedupe_key,
        "suppression_scope": event.suppression_scope,
        "clears_keys": list(event.clears_keys),
        "metadata": dict(sorted(event.metadata.items())),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NotificationDeliveryResult:
    """One per-channel delivery outcome."""

    event_id: str
    channel_name: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "channel_name": self.channel_name,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NotificationBatchResult:
    """Aggregate routing outcomes for one batch of events."""

    results: tuple[NotificationDeliveryResult, ...] = ()

    @property
    def delivered_count(self) -> int:
        return sum(result.status == "delivered" for result in self.results)

    @property
    def suppressed_count(self) -> int:
        return sum(result.status == "suppressed" for result in self.results)

    @property
    def failure_count(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered_count": self.delivered_count,
            "suppressed_count": self.suppressed_count,
            "failure_count": self.failure_count,
            "results": [result.to_dict() for result in self.results],
        }


class NotificationDeliveryError(RuntimeError):
    """Raised when a channel fails to deliver a notification."""


class NotificationChannel(ABC):
    """Interface for one outbound notification channel."""

    channel_name: str

    @abstractmethod
    def deliver(self, event: NotificationEvent) -> None:
        """Deliver one normalized notification event."""


@dataclass(frozen=True)
class NotificationRoute:
    """One routing rule from events to a channel."""

    channel: NotificationChannel
    event_types: tuple[str, ...] = ()
    severities: tuple[str, ...] = ()
    minimum_severity: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_severity is not None and self.minimum_severity not in NOTIFICATION_SEVERITIES:
            raise ValueError(
                "minimum_severity must be one of "
                f"{NOTIFICATION_SEVERITIES}, got '{self.minimum_severity}'."
            )

    def matches(self, event: NotificationEvent) -> bool:
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        if self.minimum_severity is not None and (
            _SEVERITY_RANK[event.severity] < _SEVERITY_RANK[self.minimum_severity]
        ):
            return False
        return True


@dataclass(frozen=True)
class NotificationDeliveryState:
    """Persisted suppression state for routed notifications."""

    schema_version: int = NOTIFICATION_DELIVERY_STATE_SCHEMA_VERSION
    updated_at_utc: str | None = None
    active_keys: tuple[str, ...] = ()
    delivered_once_keys: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_positive_int(self.schema_version, "schema_version")
        if self.updated_at_utc is not None:
            coerce_iso8601_datetime_text(self.updated_at_utc, "updated_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at_utc": self.updated_at_utc,
            "active_keys": list(self.active_keys),
            "delivered_once_keys": dict(sorted(self.delivered_once_keys.items())),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NotificationDeliveryState":
        raw_active_keys = payload.get("active_keys", ())
        if raw_active_keys is None:
            raw_active_keys = ()
        if not isinstance(raw_active_keys, Sequence) or isinstance(raw_active_keys, (str, bytes)):
            raise ValueError("active_keys must be a sequence.")
        raw_delivered_once_keys = payload.get("delivered_once_keys", {})
        if raw_delivered_once_keys is None:
            raw_delivered_once_keys = {}
        if not isinstance(raw_delivered_once_keys, Mapping):
            raise ValueError("delivered_once_keys must be a mapping.")
        delivered_once_keys: dict[str, str] = {}
        for key, value in raw_delivered_once_keys.items():
            if not isinstance(key, str) or not key.strip():
                continue
            delivered_once_keys[key.strip()] = coerce_iso8601_datetime_text(
                value,
                f"delivered_once_keys[{key!r}]",
            )
        return cls(
            schema_version=coerce_positive_int(payload.get("schema_version"), "schema_version"),
            updated_at_utc=_optional_text(payload.get("updated_at_utc")),
            active_keys=tuple(
                sorted(
                    {
                        key.strip()
                        for key in raw_active_keys
                        if isinstance(key, str) and key.strip()
                    }
                )
            ),
            delivered_once_keys=delivered_once_keys,
        )


@dataclass(frozen=True)
class NotificationDeliveryStateStore:
    """Project-scoped loader/saver for notification suppression state."""

    project_root: Path

    @property
    def path(self) -> Path:
        return default_notification_delivery_state_path(self.project_root)

    def load(self) -> NotificationDeliveryState:
        return load_notification_delivery_state(self.path)

    def save(self, state: NotificationDeliveryState) -> Path:
        return write_json_file(self.path, state.to_dict())


def default_notification_delivery_state_path(project_root: Path) -> Path:
    return _NOTIFICATION_DELIVERY_STATE_PATH.resolve(project_root)


def load_notification_delivery_state(path: Path) -> NotificationDeliveryState:
    return load_json_mapping_file(
        path,
        artifact_label="notification delivery state",
        logger=LOGGER,
        empty_value=NotificationDeliveryState,
        build=lambda payload, _source_path: NotificationDeliveryState.from_mapping(payload),
    )


class NotificationRouter:
    """Route normalized events to matching channels with suppression state."""

    def __init__(
        self,
        *,
        routes: Sequence[NotificationRoute],
        state_store: NotificationDeliveryStateStore | None = None,
    ) -> None:
        self.routes = tuple(routes)
        self.state_store = state_store

    def route_events(
        self,
        events: Sequence[NotificationEvent],
    ) -> NotificationBatchResult:
        if not events:
            return NotificationBatchResult()
        starting_state = (
            self.state_store.load()
            if self.state_store is not None
            else NotificationDeliveryState()
        )
        current_state = starting_state
        results: list[NotificationDeliveryResult] = []
        for event in events:
            should_deliver, suppress_reason = _should_deliver_notification_event(
                current_state,
                event,
            )
            if not should_deliver:
                results.append(
                    NotificationDeliveryResult(
                        event_id=event.event_id,
                        channel_name="router",
                        status="suppressed",
                        reason=suppress_reason,
                    )
                )
                continue
            matching_routes = [route for route in self.routes if route.matches(event)]
            if not matching_routes:
                results.append(
                    NotificationDeliveryResult(
                        event_id=event.event_id,
                        channel_name="router",
                        status="skipped",
                        reason="no matching routes",
                    )
                )
                continue
            delivered_any = False
            for route in matching_routes:
                try:
                    route.channel.deliver(event)
                except Exception as exc:
                    LOGGER.warning(
                        "Notification delivery failed for %s via %s: %s",
                        event.event_id,
                        route.channel.channel_name,
                        exc,
                    )
                    results.append(
                        NotificationDeliveryResult(
                            event_id=event.event_id,
                            channel_name=route.channel.channel_name,
                            status="failed",
                            reason=str(exc),
                        )
                    )
                    continue
                delivered_any = True
                results.append(
                    NotificationDeliveryResult(
                        event_id=event.event_id,
                        channel_name=route.channel.channel_name,
                        status="delivered",
                    )
                )
            if delivered_any:
                current_state = _advance_notification_delivery_state(current_state, event)
        if self.state_store is not None and current_state != starting_state:
            try:
                self.state_store.save(current_state)
            except OSError as exc:
                LOGGER.warning("Failed to persist notification delivery state: %s", exc)
        return NotificationBatchResult(results=tuple(results))


@dataclass(frozen=True)
class WebhookNotificationChannel(NotificationChannel):
    """Generic or Discord-style webhook delivery channel."""

    webhook_url: str
    payload_format: str = "generic"
    timeout_seconds: float = 10.0
    channel_name: str = "webhook"

    def __post_init__(self) -> None:
        if not self.webhook_url.strip():
            raise ValueError("webhook_url cannot be empty.")
        if self.payload_format not in {"generic", "discord"}:
            raise ValueError("payload_format must be 'generic' or 'discord'.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

    def build_payload(self, event: NotificationEvent) -> dict[str, Any]:
        if self.payload_format == "discord":
            return _discord_webhook_payload(event)
        return {"event": event.to_dict()}

    def deliver(self, event: NotificationEvent) -> None:
        payload = self.build_payload(event)
        request = Request(
            self.webhook_url,
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds):
                return
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore").strip()
            raise NotificationDeliveryError(
                f"Webhook delivery failed with HTTP {exc.code}: {message or exc.reason}"
            ) from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            raise NotificationDeliveryError(
                f"Webhook delivery failed: {reason}"
            ) from exc


def build_market_state_transition_notification(
    transition: MarketStateTransition,
    *,
    created_at: str,
) -> NotificationEvent:
    event_type = (
        "capacity_block"
        if transition.transition_type in {"SECTOR_BLOCK_ENTERED", "SECTOR_BLOCK_CLEARED"}
        else "state_transition"
    )
    severity = _market_state_transition_severity(transition)
    title_target = transition.symbol or transition.sector_name or "market state"
    return NotificationEvent(
        event_type=event_type,
        severity=severity,
        title=f"{transition.transition_type.replace('_', ' ').title()}: {title_target}",
        body=transition.message,
        created_at=created_at,
        symbol=transition.symbol,
        related_ids={
            key: value
            for key, value in {
                "symbol": transition.symbol,
                "sector_name": transition.sector_name,
            }.items()
            if value is not None
        },
        dedupe_key=(
            f"state-transition:{transition.transition_type}:"
            f"{transition.symbol or transition.sector_name or 'global'}:"
            f"{transition.previous_value or 'n/a'}:{transition.current_value or 'n/a'}:{created_at}"
        ),
        suppression_scope="once",
        metadata=transition.to_dict(),
    )


def build_trade_feedback_notification(
    event: TradeFeedbackEvent,
) -> NotificationEvent | None:
    event_type = event.event_type
    if event_type not in {
        "broker_submitted",
        "broker_cancelled",
        "broker_expired",
        "broker_replaced",
        "executed",
    }:
        return None
    severity = "warning" if event_type == "broker_expired" else "info"
    verb = {
        "broker_submitted": "Broker order submitted",
        "broker_cancelled": "Broker order cancelled",
        "broker_expired": "Broker order expired",
        "broker_replaced": "Broker order replaced",
        "executed": "Broker order filled",
    }[event_type]
    detail_parts: list[str] = []
    if event.quantity is not None:
        detail_parts.append(f"{event.quantity} shares")
    if event.entry_price_hint is not None:
        detail_parts.append(f"@ {event.entry_price_hint:.2f}")
    body = f"{verb} for {event.symbol}."
    if detail_parts:
        body += " " + " ".join(detail_parts)
    related_ids = {
        key: value
        for key, value in {
            "decision_id": event.decision_id,
            "trade_id": event.trade_id,
            "linked_decision_id": event.linked_decision_id,
            "order_id": _optional_text(event.metadata.get("order_id")),
            "broker_order_id": _optional_text(event.metadata.get("broker_order_id")),
            "client_order_id": _optional_text(event.metadata.get("client_order_id")),
        }.items()
        if value is not None
    }
    created_at = event.timestamp_utc or _utc_now_isoformat()
    return NotificationEvent(
        event_type="broker_event",
        severity=severity,
        title=verb,
        body=body,
        created_at=created_at,
        symbol=event.symbol,
        related_ids=related_ids,
        dedupe_key=f"broker-event:{event.event_id}",
        suppression_scope="once",
        metadata=event.to_dict(),
    )


def build_warning_notification(
    *,
    event_type: str,
    severity: str,
    title: str,
    body: str,
    created_at: str,
    symbol: str | None = None,
    related_ids: Mapping[str, str] | None = None,
    dedupe_key: str | None = None,
    suppression_scope: str = "once",
    clears_keys: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_type=event_type,
        severity=severity,
        title=title,
        body=body,
        created_at=created_at,
        symbol=symbol,
        related_ids=dict(related_ids or {}),
        dedupe_key=dedupe_key,
        suppression_scope=suppression_scope,
        clears_keys=tuple(clears_keys),
        metadata=dict(metadata or {}),
    )


def build_service_state_notifications(
    *,
    previous_state: str,
    current_state: str,
    created_at: str,
) -> tuple[NotificationEvent, ...]:
    if current_state == "retrying" and previous_state != "retrying":
        return (
            NotificationEvent(
                event_type="connectivity_warning",
                severity="warning",
                title="Live market service retrying",
                body="The live market service lost connectivity and is retrying.",
                created_at=created_at,
                dedupe_key="service:connectivity:retrying",
                suppression_scope="active",
            ),
        )
    if current_state == "stale" and previous_state != "stale":
        return (
            NotificationEvent(
                event_type="connectivity_warning",
                severity="warning",
                title="Live market service stale",
                body="The live market service is connected but no recent messages have been received.",
                created_at=created_at,
                dedupe_key="service:connectivity:stale",
                suppression_scope="active",
            ),
        )
    if current_state == "connected" and previous_state in {"retrying", "stale"}:
        return (
            NotificationEvent(
                event_type="connectivity_warning",
                severity="info",
                title="Live market connectivity recovered",
                body="The live market service reconnected and is receiving data again.",
                created_at=created_at,
                clears_keys=("service:connectivity:retrying", "service:connectivity:stale"),
                metadata={"previous_state": previous_state},
            ),
        )
    return ()


def build_service_warning_notification(
    message: str,
    *,
    created_at: str,
) -> NotificationEvent | None:
    normalized = message.strip()
    if not normalized:
        return None
    if normalized.startswith("websocket connect failed:"):
        return None
    if normalized.startswith("websocket transport disconnected:"):
        return None
    if normalized.startswith("KeyboardInterrupt received;"):
        return None
    severity = "critical" if normalized.startswith("live market cycle failed:") else "warning"
    event_type = (
        "connectivity_warning"
        if "websocket" in normalized.lower()
        else "service_health_warning"
    )
    title = (
        "Live market cycle failed"
        if normalized.startswith("live market cycle failed:")
        else "Live market service warning"
    )
    message_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return NotificationEvent(
        event_type=event_type,
        severity=severity,
        title=title,
        body=normalized,
        created_at=created_at,
        dedupe_key=f"service-warning:{message_hash}",
        suppression_scope="once",
    )


def _should_deliver_notification_event(
    state: NotificationDeliveryState,
    event: NotificationEvent,
) -> tuple[bool, str | None]:
    active_keys = set(state.active_keys)
    if event.clears_keys and not any(key in active_keys for key in event.clears_keys):
        return False, "no active state to clear"
    if event.suppression_scope == "none":
        return True, None
    assert event.dedupe_key is not None
    if event.suppression_scope == "once":
        if event.dedupe_key in state.delivered_once_keys:
            return False, "already delivered"
        return True, None
    if event.dedupe_key in active_keys:
        return False, "active notification already delivered"
    return True, None


def _advance_notification_delivery_state(
    state: NotificationDeliveryState,
    event: NotificationEvent,
) -> NotificationDeliveryState:
    active_keys = set(state.active_keys)
    delivered_once_keys = dict(state.delivered_once_keys)
    for cleared_key in event.clears_keys:
        active_keys.discard(cleared_key)
    if event.suppression_scope == "active" and event.dedupe_key is not None:
        active_keys.add(event.dedupe_key)
    elif event.suppression_scope == "once" and event.dedupe_key is not None:
        delivered_once_keys[event.dedupe_key] = event.created_at
    return NotificationDeliveryState(
        schema_version=state.schema_version,
        updated_at_utc=_utc_now_isoformat(),
        active_keys=tuple(sorted(active_keys)),
        delivered_once_keys=delivered_once_keys,
    )


def _market_state_transition_severity(transition: MarketStateTransition) -> str:
    if transition.transition_type in {
        "HOLD_TO_WATCH_CLOSELY",
        "WATCH_CLOSELY_TO_EXIT_CANDIDATE",
        "POSITION_ACTION_ESCALATED",
        "BREADTH_REGIME_CHANGED",
        "VOLATILITY_REGIME_CHANGED",
        "SECTOR_BLOCK_ENTERED",
    }:
        return "warning"
    return "info"


def _discord_webhook_payload(event: NotificationEvent) -> dict[str, Any]:
    color = {
        "info": 0x2D9CDB,
        "warning": 0xF2994A,
        "critical": 0xEB5757,
    }[event.severity]
    fields = [
        {"name": key.replace("_", " ").title(), "value": value, "inline": True}
        for key, value in sorted(event.related_ids.items())
    ]
    if event.symbol is not None:
        fields.insert(0, {"name": "Symbol", "value": event.symbol, "inline": True})
    return {
        "content": f"[{event.severity.upper()}] {event.title}",
        "embeds": [
            {
                "title": event.title,
                "description": event.body,
                "timestamp": event.created_at,
                "color": color,
                "fields": fields,
            }
        ],
    }


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
