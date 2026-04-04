from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bot.main as main_module
from bot.config import load_app_config
from bot.data.providers import DataProviderConfigurationError
from bot.execution.manual_executor import ManualExecutor
from bot.ingestion.streaming import WebsocketIngestionAdapter
from bot.ingestion.websocket_transport import (
    JsonWebsocketTransport,
    normalize_alpaca_stock_websocket_message,
)
from bot.main import build_parser
from bot.notifications import (
    NotificationChannel,
    NotificationDeliveryStateStore,
    NotificationEvent,
    NotificationRoute,
    NotificationRouter,
)
from bot.orchestration.live_runner import (
    LiveMarketCycleRequest,
    LiveMarketRunner,
    PersistedMarketStateArtifacts,
)
from bot.reporting.daily_report import build_daily_research_summary
from bot.service.live_market_service import (
    LiveMarketServiceStatusStore,
    LiveMarketSupervisor,
    ReconnectBackoffPolicy,
)
from bot.state.market_state import (
    MarketStateChangeSet,
    MarketStateSnapshot,
    MarketStateUpdateResult,
)


def _fake_research_provider_bundle(*, historical_provider: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        historical_bars_provider=(historical_provider if historical_provider is not None else object()),
        historical_bars_capabilities=SimpleNamespace(
            supports_daily_bars=True,
            supports_intraday_bars=True,
        ),
        reference_data_provider=object(),
        reference_data_capabilities=SimpleNamespace(supports_reference_universe=True),
        earnings_calendar_provider=object(),
        earnings_calendar_capabilities=SimpleNamespace(supports_earnings_calendar=True),
    )


class FakeWebsocketCloseError(Exception):
    pass


class FakeWebsocketHandshakeError(Exception):
    pass


def test_json_websocket_transport_replays_subscriptions_on_reconnect() -> None:
    first_connection = FakeWebsocketConnection(
        [json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"})]
    )
    second_connection = FakeWebsocketConnection(
        [json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"})]
    )
    factory = FakeConnectionFactory([first_connection, second_connection])
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=factory,
        subscription_messages=({"action": "subscribe", "symbols": ["AAPL"]},),
    )

    transport.connect()
    first_messages = transport.read_messages()
    transport.disconnect()
    transport.connect()
    second_messages = transport.read_messages()

    assert first_connection.sent_messages == [
        json.dumps({"action": "subscribe", "symbols": ["AAPL"]}, sort_keys=True)
    ]
    assert second_connection.sent_messages == [
        json.dumps({"action": "subscribe", "symbols": ["AAPL"]}, sort_keys=True)
    ]
    assert first_messages == (
        {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"},
    )
    assert second_messages == (
        {"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"},
    )


def test_json_websocket_transport_skips_malformed_payloads() -> None:
    connection = FakeWebsocketConnection(
        [
            "not-json",
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )

    transport.connect()
    messages = transport.read_messages()

    assert messages == (
        {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"},
    )
    assert transport.last_message_at_utc is not None


def test_json_websocket_transport_wraps_non_oserror_recv_failures() -> None:
    connection = FakeWebsocketConnection([FakeWebsocketCloseError("socket closed")])
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )

    transport.connect()

    with pytest.raises(OSError, match="socket closed"):
        transport.read_messages()

    assert transport.connected is False
    assert connection.closed is True


def test_json_websocket_transport_wraps_non_oserror_connect_failures() -> None:
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory(
            [FakeWebsocketHandshakeError("handshake failed")]
        ),
    )

    with pytest.raises(OSError, match="handshake failed"):
        transport.connect()

    assert transport.connected is False


def test_json_websocket_transport_closes_partial_connection_after_subscription_failure() -> None:
    connection = FakeWebsocketConnection(
        [],
        send_exception=FakeWebsocketHandshakeError("subscription failed"),
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
        subscription_messages=({"action": "subscribe", "symbols": ["AAPL"]},),
    )

    with pytest.raises(OSError, match="subscription failed"):
        transport.connect()

    assert transport.connected is False
    assert connection.closed is True


def test_json_websocket_transport_preserves_original_recv_error_when_close_fails() -> None:
    connection = FakeWebsocketConnection(
        [FakeWebsocketCloseError("socket closed")],
        close_exception=FakeWebsocketHandshakeError("close cleanup failed"),
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )

    transport.connect()

    with pytest.raises(OSError, match="socket closed"):
        transport.read_messages()

    assert transport.connected is False
    assert connection.closed is True


def test_live_market_supervisor_reconnects_after_transport_failure_and_runs_cycle(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([OSError("offline"), connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=3)

    assert clock.sleeps[0] == 1.0
    assert status.cycle_count == 1
    assert status.last_message_at_utc is not None
    assert status.last_successful_flush_at_utc is not None
    assert status.service_state == "stopped"


def test_live_market_supervisor_runs_cycle_from_quote_bar_updates(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps(
                {
                    "type": "quote_bar_batch_refresh",
                    "as_of_date": "2024-01-05",
                    "symbols": ["AAPL"],
                }
            ),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=2)

    assert status.cycle_count == 1
    assert status.last_successful_flush_at_utc is not None
    assert status.service_state == "stopped"


def test_live_market_supervisor_reconnects_after_non_oserror_recv_failure(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    failing_connection = FakeWebsocketConnection(
        [FakeWebsocketCloseError("socket closed")]
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([failing_connection, recovered_connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=4)

    assert status.cycle_count == 1
    assert status.warning_count >= 1
    assert status.last_successful_flush_at_utc is not None
    assert clock.sleeps[1] == 1.0


def test_live_market_supervisor_recovers_after_subscription_replay_failure(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    failing_connection = FakeWebsocketConnection(
        [],
        send_exception=FakeWebsocketHandshakeError("subscription failed"),
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([failing_connection, recovered_connection]),
        subscription_messages=({"action": "subscribe", "symbols": ["AAPL"]},),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=4)

    assert status.cycle_count == 1
    assert status.warning_count >= 1
    assert status.last_successful_flush_at_utc is not None
    assert failing_connection.closed is True
    assert recovered_connection.sent_messages == [
        json.dumps({"action": "subscribe", "symbols": ["AAPL"]}, sort_keys=True)
    ]


def test_live_market_supervisor_retries_after_close_failure_during_recv_cleanup(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    failing_connection = FakeWebsocketConnection(
        [FakeWebsocketCloseError("socket closed")],
        close_exception=FakeWebsocketHandshakeError("close cleanup failed"),
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([failing_connection, recovered_connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=4)

    assert status.cycle_count == 1
    assert status.warning_count >= 1
    assert status.last_successful_flush_at_utc is not None
    assert failing_connection.closed is True


def test_live_market_supervisor_applies_bounded_backoff(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory(
            [OSError("down"), OSError("down"), OSError("down"), OSError("down")]
        ),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        poll_interval_seconds=0.0,
        reconnect_backoff=ReconnectBackoffPolicy(
            initial_delay_seconds=1.0,
            multiplier=2.0,
            max_delay_seconds=4.0,
        ),
    )

    status = supervisor.run(max_iterations=4)

    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert status.reconnect_attempt_count == 4
    assert status.warning_count >= 4


def test_live_market_supervisor_applies_backoff_after_non_oserror_connect_failure(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory(
            [
                FakeWebsocketHandshakeError("auth failed"),
                FakeWebsocketHandshakeError("auth failed"),
                FakeWebsocketHandshakeError("auth failed"),
                FakeWebsocketHandshakeError("auth failed"),
            ]
        ),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        poll_interval_seconds=0.0,
        reconnect_backoff=ReconnectBackoffPolicy(
            initial_delay_seconds=1.0,
            multiplier=2.0,
            max_delay_seconds=4.0,
        ),
    )

    status = supervisor.run(max_iterations=4)

    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert status.reconnect_attempt_count == 4
    assert status.warning_count >= 4
    assert status.connected is False


def test_live_market_supervisor_detects_stale_connection_without_messages(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory(
            [
                FakeWebsocketConnection(
                    [
                        TimeoutError("idle"),
                        TimeoutError("idle"),
                        TimeoutError("idle"),
                    ]
                )
            ]
        ),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        stale_after_seconds=5.0,
    )

    supervisor.run_once()
    clock.advance(3)
    supervisor.run_once()
    clock.advance(3)
    status = supervisor.run_once()

    assert status.connected is False
    assert status.stale is True
    assert status.service_state == "retrying"
    assert status.last_raw_message_at_utc is None
    assert status.raw_message_count == 0
    assert status.last_warning is not None
    assert status.last_warning.startswith("websocket feed stale:")


def test_live_market_supervisor_uses_raw_provider_traffic_for_liveness_when_messages_are_dropped(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps({"T": "x", "S": "AAPL", "t": "2024-01-05T14:30:00Z"}),
            TimeoutError("idle"),
            json.dumps({"T": "x", "S": "AAPL", "t": "2024-01-05T14:31:00Z"}),
            TimeoutError("idle"),
            json.dumps({"T": "x", "S": "AAPL", "t": "2024-01-05T14:32:00Z"}),
            TimeoutError("idle"),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://stream.data.alpaca.markets/v2/sip",
        connection_factory=FakeConnectionFactory([connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        stale_after_seconds=5.0,
    )
    supervisor.adapter.message_normalizer = normalize_alpaca_stock_websocket_message

    first_status = supervisor.run_once()
    assert first_status.service_state == "connected"
    assert first_status.stale is False

    clock.advance(3)
    second_status = supervisor.run_once()
    assert second_status.service_state == "connected"
    assert second_status.stale is False

    clock.advance(3)
    third_status = supervisor.run_once()

    assert third_status.connected is True
    assert third_status.stale is False
    assert third_status.service_state == "connected"
    assert third_status.cycle_count == 0
    assert third_status.raw_message_count == 3
    assert third_status.accepted_message_count == 0
    assert third_status.last_raw_message_at_utc is not None
    assert third_status.last_message_at_utc == third_status.last_raw_message_at_utc
    assert third_status.last_raw_message_type == "x"
    assert third_status.last_accepted_message_at_utc is None
    assert third_status.last_accepted_message_type is None
    assert third_status.last_dropped_message_reason == (
        "stream message was dropped by the normalizer (raw_type=x)."
    )


def test_live_market_supervisor_reconnects_after_disconnect_during_long_running_session(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    first_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
            TimeoutError("idle"),
            TimeoutError("idle"),
            OSError("socket closed"),
        ]
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
            TimeoutError("idle"),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([first_connection, recovered_connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
        reconnect_backoff=ReconnectBackoffPolicy(
            initial_delay_seconds=1.0,
            multiplier=1.0,
            max_delay_seconds=1.0,
        ),
    )

    status = supervisor.run(max_iterations=5)

    assert status.cycle_count == 2
    assert status.warning_count >= 1
    assert status.last_successful_flush_at_utc is not None
    assert first_connection.closed is True
    assert recovered_connection.closed is True


def test_live_market_supervisor_forces_reconnect_after_stale_feed_and_resumes_cycles(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    stale_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
        ]
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
            TimeoutError("idle"),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([stale_connection, recovered_connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
        stale_after_seconds=5.0,
        reconnect_backoff=ReconnectBackoffPolicy(
            initial_delay_seconds=1.0,
            multiplier=1.0,
            max_delay_seconds=1.0,
        ),
    )

    status = supervisor.run(max_iterations=8)

    assert status.cycle_count == 2
    assert status.warning_count >= 1
    assert status.last_warning is not None
    assert "websocket feed stale:" in status.last_warning
    assert stale_connection.closed is True
    assert recovered_connection.closed is True


def test_live_market_supervisor_exposes_retrying_status_when_stale_feed_reconnect_fails(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    stale_connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
            TimeoutError("idle"),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([stale_connection, OSError("offline")]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
        stale_after_seconds=5.0,
        reconnect_backoff=ReconnectBackoffPolicy(
            initial_delay_seconds=1.0,
            multiplier=1.0,
            max_delay_seconds=1.0,
        ),
    )

    supervisor.run_once()
    for _ in range(4):
        clock.advance(1)
        supervisor.run_once()
    clock.advance(1)
    stale_status = supervisor.run_once()

    assert stale_status.service_state == "retrying"
    assert stale_status.connected is False
    assert stale_status.stale is True
    assert stale_status.last_warning is not None
    assert stale_status.last_warning.startswith("websocket feed stale:")

    clock.advance(1)
    failed_reconnect_status = supervisor.run_once()

    assert failed_reconnect_status.service_state == "retrying"
    assert failed_reconnect_status.connected is False
    assert failed_reconnect_status.last_error == "offline"
    assert failed_reconnect_status.last_warning == "websocket connect failed: offline"


def test_live_market_supervisor_gracefully_shutdowns_and_flushes_remaining_buffer(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=10,
        poll_interval_seconds=1.0,
    )

    def interrupting_sleep(_: float) -> None:
        raise KeyboardInterrupt

    supervisor.sleep = interrupting_sleep
    status = supervisor.run()

    assert status.cycle_count == 1
    assert status.service_state == "stopped"
    assert connection.closed is True


def test_live_market_supervisor_persists_status_artifact_when_store_is_configured(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    status_store = LiveMarketServiceStatusStore(tmp_path)
    supervisor = _build_supervisor(
        tmp_path,
        transport=JsonWebsocketTransport(
            url="wss://example.test/live",
            connection_factory=FakeConnectionFactory([connection]),
        ),
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )
    supervisor.status_store = status_store

    status = supervisor.run(max_iterations=3)
    persisted = status_store.load()

    assert status.cycle_count == 1
    assert persisted.updated_at_utc is not None
    assert persisted.status.cycle_count == 1
    assert persisted.status.service_state == "stopped"


def test_live_market_supervisor_malformed_messages_do_not_crash_service(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            "not-json",
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([connection]),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )

    status = supervisor.run(max_iterations=2)

    assert status.cycle_count == 1
    assert status.warning_count == 0
    assert status.last_cycle_status == "ok"


def test_live_market_supervisor_routes_connectivity_notifications(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps(
                {"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}
            ),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    transport = JsonWebsocketTransport(
        url="wss://example.test/live",
        connection_factory=FakeConnectionFactory([OSError("offline"), connection]),
    )
    channel = RecordingNotificationChannel()
    router = NotificationRouter(
        routes=(NotificationRoute(channel=channel),),
        state_store=NotificationDeliveryStateStore(tmp_path),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )
    supervisor.notification_router = router

    status = supervisor.run(max_iterations=4)

    assert status.cycle_count == 1
    assert [event.title for event in channel.events] == [
        "Live market service retrying",
        "Live market connectivity recovered",
    ]


def test_live_market_supervisor_reconnects_with_fresh_transport_after_alpaca_protocol_disconnect(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    failing_connection = FakeWebsocketConnection(
        [
            json.dumps({"T": "error", "code": 406, "msg": "connection limit exceeded"}),
        ]
    )
    recovered_connection = FakeWebsocketConnection(
        [
            json.dumps({"T": "success", "msg": "authenticated"}),
            json.dumps({"T": "subscription", "bars": ["AAPL"]}),
            json.dumps(
                {
                    "T": "b",
                    "S": "AAPL",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 10,
                    "vw": 100.2,
                    "n": 2,
                    "t": "2024-01-05T14:30:00Z",
                }
            ),
        ]
    )
    factory = FakeConnectionFactory([failing_connection, recovered_connection])
    transport = JsonWebsocketTransport(
        url="wss://stream.data.alpaca.markets/v2/sip",
        connection_factory=factory,
        subscription_messages=({"action": "subscribe", "bars": ["AAPL"]},),
    )
    supervisor = _build_supervisor(
        tmp_path,
        transport=transport,
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )
    supervisor.adapter.message_normalizer = normalize_alpaca_stock_websocket_message
    supervisor.stream_provider = "alpaca"
    supervisor.historical_provider = "polygon"
    supervisor.expect_subscription_ack = True

    status = supervisor.run(max_iterations=4)

    assert status.cycle_count == 1
    assert status.subscription_status == "acknowledged"
    assert status.last_subscription_message == "alpaca websocket subscription acknowledged: bars=1"
    assert status.raw_message_count == 4
    assert status.accepted_message_count == 4
    assert status.last_raw_message_type == "b"
    assert status.last_accepted_message_type == "quote_bar_batch_refresh"
    assert status.last_raw_message_at_utc is not None
    assert status.last_accepted_message_at_utc is not None
    assert failing_connection.closed is True
    assert recovered_connection.closed is True
    assert factory.call_count == 2


def test_live_market_supervisor_persists_provider_role_labels(
    tmp_path: Path,
) -> None:
    clock = FakeClock(datetime(2024, 1, 5, 14, 30, tzinfo=timezone.utc))
    connection = FakeWebsocketConnection(
        [
            json.dumps({"type": "candidate_universe_refresh_batch", "as_of_date": "2024-01-05"}),
            json.dumps({"type": "market_context_refresh_batch", "as_of_date": "2024-01-05"}),
        ]
    )
    status_store = LiveMarketServiceStatusStore(tmp_path)
    supervisor = _build_supervisor(
        tmp_path,
        transport=JsonWebsocketTransport(
            url="wss://stream.data.alpaca.markets/v2/sip",
            connection_factory=FakeConnectionFactory([connection]),
        ),
        clock=clock,
        flush_interval_seconds=1,
        poll_interval_seconds=1.0,
    )
    supervisor.status_store = status_store
    supervisor.stream_provider = "alpaca"
    supervisor.historical_provider = "polygon"
    supervisor.reference_provider = "polygon"
    supervisor.earnings_provider = "polygon"
    supervisor.execution_broker = "alpaca"
    supervisor.broker_update_stream_provider = None
    supervisor.provider_roles = {
        "stream_market_data": {
            "provider": "alpaca",
            "configured": True,
            "available": True,
        },
        "historical_bars": {
            "provider": "polygon",
            "configured": True,
            "available": True,
            "degraded": True,
        },
    }
    supervisor.degraded_provider_roles = ("historical_bars",)
    supervisor.unavailable_provider_roles = ()
    supervisor.status.stream_provider = "alpaca"
    supervisor.status.historical_provider = "polygon"
    supervisor.status.reference_provider = "polygon"
    supervisor.status.earnings_provider = "polygon"
    supervisor.status.execution_broker = "alpaca"
    supervisor.status.broker_update_stream_provider = None
    supervisor.status.provider_roles = dict(supervisor.provider_roles)
    supervisor.status.degraded_provider_roles = tuple(supervisor.degraded_provider_roles)
    supervisor.status.unavailable_provider_roles = tuple(
        supervisor.unavailable_provider_roles
    )

    supervisor.run(max_iterations=3)
    persisted = status_store.load()

    assert persisted.status.stream_provider == "alpaca"
    assert persisted.status.historical_provider == "polygon"
    assert persisted.status.reference_provider == "polygon"
    assert persisted.status.earnings_provider == "polygon"
    assert persisted.status.execution_broker == "alpaca"
    assert persisted.status.broker_update_stream_provider is None
    assert persisted.status.provider_roles["stream_market_data"]["provider"] == "alpaca"
    assert persisted.status.degraded_provider_roles == ("historical_bars",)
    assert persisted.status.unavailable_provider_roles == ()


def test_build_parser_accepts_run_live_market_service_command() -> None:
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--stream-provider",
            "alpaca",
            "--subscription-message",
            '{"action":"subscribe","symbols":["AAPL"]}',
            "--max-iterations",
            "1",
        ]
    )

    assert args.command == "run-live-market-service"
    assert args.websocket_url == "wss://example.test/live"
    assert args.stream_provider == "alpaca"
    assert args.subscription_message == [
        {"action": "subscribe", "symbols": ["AAPL"]}
    ]


def test_build_live_market_service_bundle_auto_configures_alpaca_stream_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\nAAPL\n", encoding="utf-8")
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        (
            "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
            "MSFT,5,100.0,95.0,standard_breakout,manual,{}\n"
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY_ID=test-key\nALPACA_SECRET_KEY=test-secret\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/sip",
            "--stream-provider",
            "alpaca",
            "--portfolio-file",
            str(portfolio_path),
        ]
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        data_sources=SimpleNamespace(provider="polygon"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_research_provider_bundle",
        lambda config, env_file: _fake_research_provider_bundle(),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=env_file,
        output_dir=tmp_path / "outputs",
    )

    transport = bundle.supervisor.adapter.transport
    assert isinstance(transport, JsonWebsocketTransport)
    assert transport.connection_headers == {
        "APCA-API-KEY-ID": "test-key",
        "APCA-API-SECRET-KEY": "test-secret",
    }
    assert transport.subscription_messages == (
        {"action": "subscribe", "bars": ["AAPL", "MSFT", "NVDA"]},
    )
    assert bundle.supervisor.stale_after_seconds == pytest.approx(90.0)
    assert bundle.supervisor.stream_provider == "alpaca"
    assert bundle.supervisor.historical_provider == "polygon"
    assert bundle.supervisor.reference_provider == "polygon"
    assert bundle.supervisor.earnings_provider == "polygon"
    assert bundle.supervisor.execution_broker is None
    assert bundle.supervisor.broker_update_stream_provider is None
    assert bundle.supervisor.adapter.provider_name == "alpaca"
    normalized = bundle.supervisor.adapter.message_normalizer(
        {
            "T": "b",
            "S": "AAPL",
            "o": 100.0,
            "h": 101.0,
            "l": 99.0,
            "c": 100.5,
            "v": 10,
            "vw": 100.2,
            "n": 2,
            "t": "2024-01-05T14:30:00Z",
        }
    )
    assert normalized is not None
    assert normalized["type"] == "quote_bar_batch_refresh"


def test_build_live_market_service_bundle_uses_configured_stream_role_when_cli_is_auto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY_ID=test-key\nALPACA_SECRET_KEY=test-secret\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://example.test/feed",
            "--stream-provider",
            "auto",
        ]
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        data_sources=SimpleNamespace(
            provider="polygon",
            roles=SimpleNamespace(stream_market_data="alpaca"),
        ),
    )
    monkeypatch.setattr(
        main_module,
        "_build_research_provider_bundle",
        lambda config, env_file: _fake_research_provider_bundle(),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=env_file,
        output_dir=tmp_path / "outputs",
    )

    assert bundle.supervisor.stream_provider == "alpaca"
    assert bundle.supervisor.adapter.provider_name == "alpaca"


def test_build_live_market_service_bundle_uses_alpaca_test_stream_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY_ID=test-key\nALPACA_SECRET_KEY=test-secret\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/test",
            "--stream-provider",
            "alpaca",
        ]
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        data_sources=SimpleNamespace(provider="polygon"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_research_provider_bundle",
        lambda config, env_file: _fake_research_provider_bundle(),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=env_file,
        output_dir=tmp_path / "outputs",
    )

    transport = bundle.supervisor.adapter.transport
    assert isinstance(transport, JsonWebsocketTransport)
    assert transport.subscription_messages == (
        {"action": "subscribe", "trades": ["FAKEPACA"]},
    )
    assert bundle.supervisor.stale_after_seconds == pytest.approx(30.0)
    normalized = bundle.supervisor.adapter.message_normalizer(
        {
            "T": "t",
            "S": "FAKEPACA",
            "p": 100.0,
            "s": 1,
            "t": "2024-01-05T14:30:00Z",
        }
    )
    assert normalized is not None
    assert normalized["type"] == "quote_bar_batch_refresh"


def test_build_live_market_service_bundle_allows_explicit_alpaca_auth_message_without_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/test",
            "--stream-provider",
            "alpaca",
            "--subscription-message",
            '{"action":"auth","key":"manual-key","secret":"manual-secret"}',
            "--subscription-message",
            '{"action":"subscribe","trades":["FAKEPACA"]}',
        ]
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        data_sources=SimpleNamespace(provider="polygon"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_research_provider_bundle",
        lambda config, env_file: _fake_research_provider_bundle(),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=None,
        output_dir=tmp_path / "outputs",
    )

    transport = bundle.supervisor.adapter.transport
    assert isinstance(transport, JsonWebsocketTransport)
    assert transport.connection_headers is None
    assert transport.subscription_messages == (
        {"action": "auth", "key": "manual-key", "secret": "manual-secret"},
        {"action": "subscribe", "trades": ["FAKEPACA"]},
    )


def test_build_live_market_service_bundle_warns_when_alpaca_subscription_lacks_bar_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/sip",
            "--stream-provider",
            "alpaca",
            "--subscription-message",
            '{"action":"auth","key":"manual-key","secret":"manual-secret"}',
            "--subscription-message",
            '{"action":"subscribe","quotes":["NVDA"]}',
        ]
    )
    config = SimpleNamespace(
        project_root=tmp_path,
        data_sources=SimpleNamespace(provider="polygon"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_research_provider_bundle",
        lambda config, env_file: _fake_research_provider_bundle(),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=None,
        output_dir=tmp_path / "outputs",
    )

    assert bundle.supervisor.stale_after_seconds == pytest.approx(30.0)
    assert bundle.supervisor.startup_warnings == (
        "Alpaca monitor-market cycles currently require `bars` subscriptions to drive "
        "daily-summary, candidate-queue, and portfolio refresh triggers. Quotes-only, "
        "trades-only, `dailyBars`, and `updatedBars` subscriptions will not refresh "
        "recommendation state in this runtime.",
    )


def test_build_live_market_service_bundle_supports_mixed_alpaca_historical_and_polygon_reference_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\nAAPL\n", encoding="utf-8")
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        (
            "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
            "MSFT,5,100.0,95.0,standard_breakout,manual,{}\n"
        ),
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        (
            "ALPACA_API_KEY_ID=alpaca-key\n"
            "ALPACA_SECRET_KEY=alpaca-secret\n"
            "POLYGON_API_KEY=polygon-key\n"
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/sip",
            "--stream-provider",
            "alpaca",
            "--portfolio-file",
            str(portfolio_path),
        ]
    )
    base_config = replace(load_app_config(), project_root=tmp_path)
    config = replace(
        base_config,
        data_sources=replace(
            base_config.data_sources,
            roles=base_config.data_sources.roles.__class__(
                historical_bars="alpaca",
                reference_data="polygon",
                earnings_calendar="polygon",
            ),
        ),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    bundle = main_module._build_live_market_service_bundle(
        args,
        config=config,
        env_file=env_file,
        output_dir=tmp_path / "outputs",
    )

    assert bundle.supervisor.stream_provider == "alpaca"
    assert bundle.supervisor.historical_provider == "alpaca"
    assert bundle.supervisor.reference_provider == "polygon"
    assert bundle.supervisor.earnings_provider == "polygon"
    assert bundle.supervisor.provider_roles["historical_bars"]["provider"] == "alpaca"
    assert bundle.supervisor.provider_roles["reference_data"]["provider"] == "polygon"
    assert "historical_bars" in bundle.supervisor.degraded_provider_roles


def test_build_live_market_service_bundle_fails_early_for_unsupported_reference_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("NVDA\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("POLYGON_API_KEY=polygon-key\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run-live-market-service",
            str(candidate_path),
            "--websocket-url",
            "wss://stream.data.alpaca.markets/v2/sip",
            "--stream-provider",
            "alpaca",
        ]
    )
    base_config = replace(load_app_config(), project_root=tmp_path)
    config = replace(
        base_config,
        data_sources=replace(
            base_config.data_sources,
            roles=base_config.data_sources.roles.__class__(
                historical_bars="polygon",
                reference_data="alpaca",
                earnings_calendar="polygon",
            ),
        ),
    )
    monkeypatch.setattr(main_module, "_build_notification_router", lambda **kwargs: None)

    with pytest.raises(
        DataProviderConfigurationError,
        match="Unsupported provider assignment for role 'reference_data': requested 'alpaca'",
    ):
        main_module._build_live_market_service_bundle(
            args,
            config=config,
            env_file=env_file,
            output_dir=tmp_path / "outputs",
        )


def _build_supervisor(
    tmp_path: Path,
    *,
    transport: JsonWebsocketTransport,
    clock: "FakeClock",
    flush_interval_seconds: int = 1,
    poll_interval_seconds: float = 0.0,
    stale_after_seconds: float = 30.0,
    reconnect_backoff: ReconnectBackoffPolicy | None = None,
) -> LiveMarketSupervisor:
    request = LiveMarketCycleRequest(
        workflow="monitor-market",
        as_of_date=date(2024, 1, 5),
        include_daily_summary=True,
        daily_summary_required=True,
    )
    adapter = WebsocketIngestionAdapter(
        transport_name=transport.transport_name,
        transport=transport,
        portfolio_path=None,
        load_current_positions=lambda: (),
        run_daily_summary=lambda snapshot, current_positions: {"summary": _daily_summary()},
        flush_interval_seconds=flush_interval_seconds,
    )
    state_artifacts = _state_artifacts(tmp_path)
    runner = LiveMarketRunner(
        persist_market_state=lambda as_of_date, workflow, summary, review, intraday, portfolio_path: state_artifacts,
    )
    return LiveMarketSupervisor(
        runner=runner,
        adapter=adapter,
        request=request,
        poll_interval_seconds=poll_interval_seconds,
        stale_after_seconds=stale_after_seconds,
        reconnect_backoff=(
            reconnect_backoff if reconnect_backoff is not None else ReconnectBackoffPolicy()
        ),
        sleep=clock.sleep,
        now_utc=clock.now_utc,
    )


class FakeClock:
    def __init__(self, current: datetime) -> None:
        self.current = current
        self.sleeps: list[float] = []

    def now_utc(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class FakeConnectionFactory:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def __call__(self, url: str):
        self.call_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingNotificationChannel(NotificationChannel):
    channel_name = "recording"

    def __init__(self) -> None:
        self.events: list[NotificationEvent] = []

    def deliver(self, event: NotificationEvent) -> None:
        self.events.append(event)


class FakeWebsocketConnection:
    def __init__(
        self,
        raw_messages: list[object],
        *,
        send_exception: Exception | None = None,
        close_exception: Exception | None = None,
        settimeout_exception: Exception | None = None,
    ) -> None:
        self.raw_messages = list(raw_messages)
        self.send_exception = send_exception
        self.close_exception = close_exception
        self.settimeout_exception = settimeout_exception
        self.sent_messages: list[str] = []
        self.closed = False
        self.timeout: float | None = None

    def send(self, payload: str) -> None:
        if self.send_exception is not None:
            raise self.send_exception
        self.sent_messages.append(payload)

    def recv(self) -> str:
        if not self.raw_messages:
            raise TimeoutError("no messages available")
        response = self.raw_messages.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)

    def close(self) -> None:
        self.closed = True
        if self.close_exception is not None:
            raise self.close_exception

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout
        if self.settimeout_exception is not None:
            raise self.settimeout_exception


def _daily_summary():
    execution_batch = ManualExecutor().build_execution_batch(
        [],
        as_of_date=date(2024, 1, 5),
    )
    return build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[],
        selected_presets=[],
        universe_symbols=[],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={},
        benchmark_symbol="SPY",
        preset_selection_source="test",
    )


def _state_artifacts(tmp_path: Path) -> PersistedMarketStateArtifacts:
    snapshot = MarketStateSnapshot(
        schema_version=1,
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
    )
    change_set = MarketStateChangeSet(
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
        baseline_established=False,
    )
    return PersistedMarketStateArtifacts(
        update_result=MarketStateUpdateResult(
            current_snapshot=snapshot,
            previous_snapshot=None,
            change_set=change_set,
            current_path=tmp_path / "current_market_state.json",
            previous_path=tmp_path / "previous_market_state.json",
        ),
        output_paths={
            "current_market_state_snapshot": str(tmp_path / "current_market_state.json"),
            "market_state_changes_json": str(tmp_path / "market_state_changes.json"),
            "market_state_changes_text": str(tmp_path / "market_state_changes.txt"),
        },
    )
