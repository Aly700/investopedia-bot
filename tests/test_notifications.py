from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from bot.data.trade_feedback import TradeFeedbackEvent
from bot.notifications import (
    NotificationBatchResult,
    NotificationChannel,
    NotificationDeliveryResult,
    NotificationDeliveryStateStore,
    NotificationEvent,
    NotificationRoute,
    NotificationRouter,
    WebhookNotificationChannel,
    build_market_state_transition_notification,
    build_service_state_notifications,
    build_service_warning_notification,
    build_trade_feedback_notification,
    build_warning_notification,
)
from bot.state.market_state import MarketStateTransition


@dataclass
class RecordingChannel(NotificationChannel):
    channel_name: str = "recording"
    events: list[NotificationEvent] = field(default_factory=list)

    def deliver(self, event: NotificationEvent) -> None:
        self.events.append(event)


@dataclass
class FailingChannel(NotificationChannel):
    channel_name: str = "failing"

    def deliver(self, event: NotificationEvent) -> None:
        raise RuntimeError(f"{self.channel_name} is unavailable")


def _event(
    *,
    event_type: str = "state_transition",
    severity: str = "warning",
    title: str = "Example event",
    body: str = "Something changed.",
    created_at: str = "2024-01-05T15:00:00+00:00",
    dedupe_key: str | None = None,
    suppression_scope: str = "none",
    clears_keys: tuple[str, ...] = (),
) -> NotificationEvent:
    return NotificationEvent(
        event_type=event_type,
        severity=severity,
        title=title,
        body=body,
        created_at=created_at,
        dedupe_key=dedupe_key,
        suppression_scope=suppression_scope,
        clears_keys=clears_keys,
    )


def test_webhook_notification_channel_builds_generic_payload() -> None:
    event = _event()
    channel = WebhookNotificationChannel(
        webhook_url="https://example.test/webhook",
        payload_format="generic",
    )

    assert channel.build_payload(event) == {"event": event.to_dict()}


def test_webhook_notification_channel_builds_discord_payload() -> None:
    event = _event(
        event_type="broker_event",
        severity="warning",
        title="Broker order submitted",
        body="Broker order submitted for AAPL.",
    )
    channel = WebhookNotificationChannel(
        webhook_url="https://example.test/webhook",
        payload_format="discord",
    )

    payload = channel.build_payload(event)

    assert payload["content"] == "[WARNING] Broker order submitted"
    assert payload["embeds"][0]["title"] == "Broker order submitted"
    assert payload["embeds"][0]["description"] == "Broker order submitted for AAPL."


def test_build_service_warning_notification_filters_transport_and_shutdown_messages() -> None:
    assert (
        build_service_warning_notification(
            "websocket connect failed: offline",
            created_at="2024-01-05T15:00:00+00:00",
        )
        is None
    )
    assert (
        build_service_warning_notification(
            "websocket transport disconnected: socket closed",
            created_at="2024-01-05T15:00:00+00:00",
        )
        is None
    )
    assert (
        build_service_warning_notification(
            "KeyboardInterrupt received; stopping live market service.",
            created_at="2024-01-05T15:00:00+00:00",
        )
        is None
    )


def test_build_service_warning_notification_maps_cycle_failure_to_critical_service_event() -> None:
    event = build_service_warning_notification(
        "live market cycle failed: provider unavailable",
        created_at="2024-01-05T15:00:00+00:00",
    )

    assert event is not None
    assert event.event_type == "service_health_warning"
    assert event.severity == "critical"
    assert event.title == "Live market cycle failed"


def test_build_trade_feedback_notification_maps_expired_event_to_warning() -> None:
    event = build_trade_feedback_notification(
        TradeFeedbackEvent(
            event_type="broker_expired",
            workflow="sync-broker-orders",
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            timestamp_utc="2024-01-05T15:00:00+00:00",
            trade_id="trade_aapl",
        )
    )

    assert event is not None
    assert event.event_type == "broker_event"
    assert event.severity == "warning"
    assert event.title == "Broker order expired"


def test_build_trade_feedback_notification_returns_none_for_non_routable_event_type() -> None:
    event = build_trade_feedback_notification(
        TradeFeedbackEvent(
            event_type="candidate_reviewed",
            workflow="generate-orders",
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
        )
    )

    assert event is None


def test_build_market_state_transition_notification_maps_sector_blocks_to_capacity_block() -> None:
    sector_block = build_market_state_transition_notification(
        MarketStateTransition(
            transition_type="SECTOR_BLOCK_ENTERED",
            message="Technology is now sector-blocked.",
            sector_name="Technology",
        ),
        created_at="2024-01-05T15:00:00+00:00",
    )
    position_transition = build_market_state_transition_notification(
        MarketStateTransition(
            transition_type="HOLD_TO_WATCH_CLOSELY",
            message="AAPL moved from HOLD to WATCH CLOSELY.",
            symbol="AAPL",
        ),
        created_at="2024-01-05T15:00:00+00:00",
    )

    assert sector_block.event_type == "capacity_block"
    assert position_transition.event_type == "state_transition"


def test_build_service_state_notifications_normalizes_retrying_stale_and_recovered_transitions() -> None:
    retrying = build_service_state_notifications(
        previous_state="connected",
        current_state="retrying",
        created_at="2024-01-05T15:00:00+00:00",
    )
    stale = build_service_state_notifications(
        previous_state="connected",
        current_state="stale",
        created_at="2024-01-05T15:01:00+00:00",
    )
    recovered = build_service_state_notifications(
        previous_state="retrying",
        current_state="connected",
        created_at="2024-01-05T15:02:00+00:00",
    )

    assert retrying[0].event_type == "connectivity_warning"
    assert retrying[0].severity == "warning"
    assert retrying[0].dedupe_key == "service:connectivity:retrying"
    assert stale[0].title == "Live market service stale"
    assert recovered[0].severity == "info"
    assert recovered[0].clears_keys == (
        "service:connectivity:retrying",
        "service:connectivity:stale",
    )


def test_notification_router_routes_by_event_type_and_minimum_severity() -> None:
    channel = RecordingChannel()
    router = NotificationRouter(
        routes=(
            NotificationRoute(
                channel=channel,
                event_types=("broker_event",),
                minimum_severity="warning",
            ),
        )
    )

    result = router.route_events(
        (
            _event(
                event_type="broker_event",
                severity="info",
                title="Informational broker event",
            ),
            _event(
                event_type="broker_event",
                severity="warning",
                title="Warning broker event",
            ),
            _event(
                event_type="service_health_warning",
                severity="warning",
                title="Service warning",
            ),
        )
    )

    assert [event.title for event in channel.events] == ["Warning broker event"]
    assert result.delivered_count == 1
    assert sum(item.status == "skipped" for item in result.results) == 2


def test_notification_router_suppresses_repeated_once_events(tmp_path) -> None:
    channel = RecordingChannel()
    store = NotificationDeliveryStateStore(tmp_path)
    router = NotificationRouter(
        routes=(NotificationRoute(channel=channel),),
        state_store=store,
    )
    event = build_warning_notification(
        event_type="execution_warning",
        severity="warning",
        title="Submit warning",
        body="Broker submit failed.",
        created_at="2024-01-05T15:00:00+00:00",
        dedupe_key="execution-warning:test",
        suppression_scope="once",
    )

    first = router.route_events((event,))
    second = router.route_events((event,))

    assert first.delivered_count == 1
    assert second.suppressed_count == 1
    assert channel.events == [event]
    assert "execution-warning:test" in store.load().delivered_once_keys


def test_notification_router_allows_recovery_after_active_suppression(tmp_path) -> None:
    channel = RecordingChannel()
    store = NotificationDeliveryStateStore(tmp_path)
    router = NotificationRouter(
        routes=(NotificationRoute(channel=channel),),
        state_store=store,
    )
    retrying = build_warning_notification(
        event_type="connectivity_warning",
        severity="warning",
        title="Live market service retrying",
        body="The live market service lost connectivity and is retrying.",
        created_at="2024-01-05T15:00:00+00:00",
        dedupe_key="service:connectivity:retrying",
        suppression_scope="active",
    )
    recovered = build_warning_notification(
        event_type="connectivity_warning",
        severity="info",
        title="Live market connectivity recovered",
        body="The live market service reconnected and is receiving data again.",
        created_at="2024-01-05T15:02:00+00:00",
        clears_keys=("service:connectivity:retrying",),
        suppression_scope="none",
    )

    first = router.route_events((retrying,))
    second = router.route_events((retrying,))
    recovery = router.route_events((recovered,))
    third = router.route_events((retrying,))

    assert first.delivered_count == 1
    assert second.suppressed_count == 1
    assert recovery.delivered_count == 1
    assert third.delivered_count == 1
    assert [event.title for event in channel.events] == [
        "Live market service retrying",
        "Live market connectivity recovered",
        "Live market service retrying",
    ]


def test_notification_router_captures_channel_failures_without_raising() -> None:
    failing_channel = FailingChannel()
    recording_channel = RecordingChannel()
    router = NotificationRouter(
        routes=(
            NotificationRoute(channel=failing_channel),
            NotificationRoute(channel=recording_channel),
        )
    )
    event = _event(
        event_type="broker_event",
        severity="warning",
        title="Broker order expired",
    )

    result = router.route_events((event,))

    assert result.failure_count == 1
    assert result.delivered_count == 1
    assert recording_channel.events == [event]
    assert any(
        item.status == "failed" and item.channel_name == "failing"
        for item in result.results
    )
