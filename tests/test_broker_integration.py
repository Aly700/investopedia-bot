from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import bot.main as main_module
from bot.broker.execution import (
    AlpacaBrokerCredentials,
    AlpacaBrokerExecutionAdapter,
    BrokerOrderRequest,
    BrokerRequestError,
    BrokerOrderUpdate,
    BrokerReplaceOrderRequest,
)
from bot.broker.reconciliation import (
    reconcile_broker_order_updates,
    submit_pending_orders_via_broker,
)
from bot.config import load_app_config
from bot.data.pending_orders import (
    PendingOrderRecord,
    PendingOrderState,
    PendingOrderStateStore,
    build_order_reservation_state,
    create_pending_order,
    load_pending_order_state,
)
from bot.data.trade_feedback import (
    default_trade_feedback_log_path,
    load_trade_feedback_events,
)
from bot.notifications import (
    NotificationBatchResult,
    NotificationDeliveryResult,
    NotificationEvent,
)


class FakeBrokerAdapter:
    def __init__(
        self,
        *,
        placed_update: BrokerOrderUpdate | None = None,
        placed_results: list[BrokerOrderUpdate | Exception] | None = None,
        sync_updates: dict[str, BrokerOrderUpdate] | None = None,
        cancel_update: BrokerOrderUpdate | None = None,
        replace_update: BrokerOrderUpdate | None = None,
        get_order_error: Exception | None = None,
    ) -> None:
        self.placed_update = placed_update
        self.placed_results = list(placed_results or ())
        self.sync_updates = dict(sync_updates or {})
        self.cancel_update = cancel_update
        self.replace_update = replace_update
        self.get_order_error = get_order_error
        self.placed_requests: list[BrokerOrderRequest] = []
        self.cancelled_order_ids: list[str] = []
        self.replace_requests: list[tuple[str, BrokerReplaceOrderRequest]] = []
        self.list_open_orders_calls = 0

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderUpdate:
        self.placed_requests.append(request)
        if self.placed_results:
            result = self.placed_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.placed_update is None:
            raise AssertionError("placed_update was not configured for FakeBrokerAdapter.")
        return self.placed_update

    def cancel_order(self, broker_order_id: str) -> BrokerOrderUpdate | None:
        self.cancelled_order_ids.append(broker_order_id)
        return self.cancel_update

    def replace_order(
        self,
        broker_order_id: str,
        request: BrokerReplaceOrderRequest,
    ) -> BrokerOrderUpdate:
        self.replace_requests.append((broker_order_id, request))
        if self.replace_update is None:
            raise AssertionError("replace_update was not configured for FakeBrokerAdapter.")
        if request.client_order_id is not None and (
            self.replace_update.client_order_id != request.client_order_id
        ):
            return replace(
                self.replace_update,
                client_order_id=request.client_order_id,
            )
        return self.replace_update

    def list_open_orders(self) -> tuple[BrokerOrderUpdate, ...]:
        self.list_open_orders_calls += 1
        return tuple(self.sync_updates.values())

    def get_order(
        self,
        broker_order_id: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> BrokerOrderUpdate:
        if self.get_order_error is not None:
            raise self.get_order_error
        key = broker_order_id or client_order_id
        if key is None:
            raise AssertionError("Expected broker_order_id or client_order_id.")
        try:
            return self.sync_updates[key]
        except KeyError as exc:
            raise AssertionError(f"No sync update configured for key '{key}'.") from exc

    def list_positions(self) -> tuple[object, ...]:
        return ()


class FakeNotificationRouter:
    def __init__(self) -> None:
        self.events: list[NotificationEvent] = []

    def route_events(
        self,
        events: tuple[NotificationEvent, ...] | list[NotificationEvent],
    ) -> NotificationBatchResult:
        event_batch = tuple(events)
        self.events.extend(event_batch)
        return NotificationBatchResult(
            results=tuple(
                NotificationDeliveryResult(
                    event_id=event.event_id,
                    channel_name="test-webhook",
                    status="delivered",
                )
                for event in event_batch
            )
        )


def test_alpaca_broker_adapter_normalizes_place_order_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AlpacaBrokerExecutionAdapter(
        credentials=AlpacaBrokerCredentials(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )
    )
    captured: dict[str, object] = {}

    def fake_request_json(
        path: str,
        *,
        method: str = "GET",
        params: dict[str, str] | None = None,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        captured["path"] = path
        captured["method"] = method
        captured["params"] = params
        captured["payload"] = payload
        return {
            "id": "alpaca-ord-1",
            "client_order_id": "order_local_1",
            "symbol": "AAPL",
            "side": "buy",
            "type": "limit",
            "qty": "10",
            "filled_qty": "0",
            "limit_price": "101.5",
            "status": "new",
            "submitted_at": "2024-01-05T15:00:00+00:00",
            "updated_at": "2024-01-05T15:00:01+00:00",
        }

    monkeypatch.setattr(adapter, "_request_json", fake_request_json)

    update = adapter.place_order(
        BrokerOrderRequest(
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=10,
            client_order_id="order_local_1",
            limit_price=101.5,
        )
    )

    assert captured["path"] == "/v2/orders"
    assert captured["method"] == "POST"
    assert captured["payload"] == {
        "client_order_id": "order_local_1",
        "limit_price": "101.5",
        "qty": "10",
        "side": "buy",
        "symbol": "AAPL",
        "time_in_force": "day",
        "type": "limit",
    }
    assert update.broker_name == "alpaca"
    assert update.status == "accepted"
    assert update.side == "BUY"
    assert update.order_type == "LIMIT"
    assert update.quantity == 10


def test_submit_pending_orders_via_broker_updates_state_with_broker_ids() -> None:
    state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
        )
    )
    adapter = FakeBrokerAdapter(
        placed_update=BrokerOrderUpdate(
            broker_name="alpaca",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=10,
            filled_quantity=0,
            status="accepted",
            limit_price=100.0,
            submitted_at="2024-01-05T15:01:00+00:00",
            updated_at="2024-01-05T15:01:01+00:00",
        )
    )

    result = submit_pending_orders_via_broker(state, adapter=adapter)

    record = result.state.orders["order_aapl_1"]
    assert result.skipped_order_ids == ()
    assert record.broker_order_id == "alpaca-ord-1"
    assert record.client_order_id == "order_aapl_1"
    assert record.broker_status == "accepted"


def test_submit_pending_orders_via_broker_preserves_earlier_success_when_later_submit_fails() -> None:
    state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
        ),
        PendingOrderRecord(
            order_id="order_msft_1",
            symbol="MSFT",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=5,
            requested_price=200.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
        ),
    )
    adapter = FakeBrokerAdapter(
        placed_results=[
            BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=0,
                status="accepted",
                limit_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:01:01+00:00",
            ),
            BrokerRequestError("broker temporarily unavailable"),
        ]
    )

    result = submit_pending_orders_via_broker(state, adapter=adapter)

    assert len(result.submitted_updates) == 1
    assert "broker temporarily unavailable" in result.warnings[0]
    assert result.state.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"
    assert result.state.orders["order_aapl_1"].broker_status == "accepted"
    assert result.state.orders["order_msft_1"].broker_order_id is None
    assert result.state.orders["order_msft_1"].broker_status is None


def test_reconcile_broker_partial_fill_preserves_capacity_across_reload(
    tmp_path: Path,
) -> None:
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
        )
    )

    result = reconcile_broker_order_updates(
        starting_state,
        (
            BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=5,
                status="partially_filled",
                limit_price=100.0,
                average_fill_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:05:00+00:00",
            ),
        ),
    )
    store = PendingOrderStateStore(tmp_path)
    store.save(result.state)
    reloaded_state = load_pending_order_state(store.path)
    reservation_state = build_order_reservation_state(
        reloaded_state,
        current_equity=2_000.0,
        current_positions=(),
        max_concurrent_positions=4,
    )

    assert reloaded_state.orders["order_aapl_1"].status == "partially_filled"
    assert reloaded_state.orders["order_aapl_1"].reserved_notional == pytest.approx(500.0)
    assert reservation_state.pending_fill_notional == pytest.approx(500.0)
    assert reservation_state.available_capital == pytest.approx(1_000.0)


def test_reconcile_broker_full_fill_marks_order_filled_and_emits_position_update() -> None:
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
        )
    )

    result = reconcile_broker_order_updates(
        starting_state,
        (
            BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=10,
                status="filled",
                limit_price=100.0,
                average_fill_price=100.5,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:10:00+00:00",
            ),
        ),
    )

    record = result.state.orders["order_aapl_1"]
    assert record.status == "filled"
    assert record.reserved_notional == pytest.approx(0.0)
    assert record.average_fill_price == pytest.approx(100.5)
    assert len(result.position_updates) == 1
    assert result.position_updates[0].metadata["order_id"] == "order_aapl_1"


def test_reconcile_broker_update_matches_by_client_order_id_when_broker_id_missing() -> None:
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
        )
    )

    result = reconcile_broker_order_updates(
        starting_state,
        (
            BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=0,
                status="accepted",
                limit_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:01:01+00:00",
            ),
        ),
    )

    assert result.warnings == ()
    assert result.state.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"


def test_handle_submit_broker_orders_persists_state_and_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
            preset_name="standard_breakout",
            metadata={"strategy_name": "breakout"},
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        placed_update=BrokerOrderUpdate(
            broker_name="alpaca",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=10,
            filled_quantity=0,
            status="accepted",
            limit_price=100.0,
            submitted_at="2024-01-05T15:01:00+00:00",
            updated_at="2024-01-05T15:01:01+00:00",
        )
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_submit_broker_orders(args) == 0

    persisted_state = store.load()
    events = load_trade_feedback_events(default_trade_feedback_log_path(tmp_path))
    assert persisted_state.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"
    assert events[-1].event_type == "broker_submitted"
    assert payloads[-1]["submitted_count"] == 1


def test_handle_submit_broker_orders_persists_partial_success_when_later_submit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
            preset_name="standard_breakout",
            metadata={"strategy_name": "breakout"},
        ),
        PendingOrderRecord(
            order_id="order_msft_1",
            symbol="MSFT",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=5,
            requested_price=200.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_msft",
            trade_id="trade_msft",
            preset_name="standard_breakout",
            metadata={"strategy_name": "breakout"},
        ),
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        placed_results=[
            BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=0,
                status="accepted",
                limit_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:01:01+00:00",
            ),
            BrokerRequestError("broker temporarily unavailable"),
        ]
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_submit_broker_orders(args) == 0

    persisted_state = store.load()
    events = load_trade_feedback_events(default_trade_feedback_log_path(tmp_path))
    assert persisted_state.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"
    assert persisted_state.orders["order_msft_1"].broker_order_id is None
    assert any(
        event.event_type == "broker_submitted" and event.symbol == "AAPL"
        for event in events
    )
    assert payloads[-1]["submitted_count"] == 1
    assert "broker temporarily unavailable" in payloads[-1]["warnings"][0]


def test_handle_submit_broker_orders_routes_notification_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        placed_update=BrokerOrderUpdate(
            broker_name="alpaca",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=10,
            filled_quantity=0,
            status="accepted",
            limit_price=100.0,
            submitted_at="2024-01-05T15:01:00+00:00",
            updated_at="2024-01-05T15:01:01+00:00",
        )
    )
    notification_router = FakeNotificationRouter()
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(
        main_module,
        "_create_broker_adapter",
        lambda broker, env_file=None: adapter,
    )
    monkeypatch.setattr(
        main_module,
        "_build_notification_router",
        lambda **kwargs: notification_router,
    )
    monkeypatch.setattr(
        main_module,
        "_print_structured",
        lambda payload, output_format: payloads.append(payload),
    )

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_submit_broker_orders(args) == 0

    assert [event.event_type for event in notification_router.events] == ["broker_event"]
    assert payloads[-1]["notification_delivery"]["delivered_count"] == 1


def test_handle_sync_broker_orders_reconciles_fill_into_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
            preset_name="standard_breakout",
            metadata={"strategy_name": "breakout"},
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        sync_updates={
            "alpaca-ord-1": BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=10,
                status="filled",
                limit_price=100.0,
                average_fill_price=100.5,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:10:00+00:00",
            )
        }
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_sync_broker_orders(args) == 0

    persisted_state = store.load()
    events = load_trade_feedback_events(default_trade_feedback_log_path(tmp_path))
    assert persisted_state.orders["order_aapl_1"].status == "filled"
    assert any(event.event_type == "executed" for event in events)
    assert payloads[-1]["matched_update_count"] == 1


def test_handle_sync_broker_orders_uses_open_order_scan_to_recover_missing_local_broker_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            client_order_id="order_aapl_1",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        sync_updates={
            "alpaca-ord-1": BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_aapl_1",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=0,
                status="accepted",
                limit_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:01:01+00:00",
            )
        }
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_sync_broker_orders(args) == 0

    persisted_state = store.load()
    assert adapter.list_open_orders_calls == 1
    assert persisted_state.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"
    assert payloads[-1]["matched_update_count"] == 1


def test_handle_sync_broker_orders_surfaces_unmatched_broker_open_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    store.save(PendingOrderState(orders={}))
    adapter = FakeBrokerAdapter(
        sync_updates={
            "alpaca-ord-1": BrokerOrderUpdate(
                broker_name="alpaca",
                broker_order_id="alpaca-ord-1",
                client_order_id="order_missing_local",
                symbol="AAPL",
                side="BUY",
                order_type="LIMIT",
                quantity=10,
                filled_quantity=0,
                status="accepted",
                limit_price=100.0,
                submitted_at="2024-01-05T15:01:00+00:00",
                updated_at="2024-01-05T15:01:01+00:00",
            )
        }
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_sync_broker_orders(args) == 0

    assert payloads[-1]["matched_update_count"] == 0
    assert payloads[-1]["unmatched_update_count"] == 1
    assert "could not be matched" in payloads[-1]["warnings"][0]


def test_handle_cancel_broker_order_marks_order_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        cancel_update=BrokerOrderUpdate(
            broker_name="alpaca",
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=10,
            filled_quantity=0,
            status="cancelled",
            limit_price=100.0,
            submitted_at="2024-01-05T15:01:00+00:00",
            updated_at="2024-01-05T15:04:00+00:00",
        )
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_id="order_aapl_1",
        format="json",
    )

    assert main_module._handle_cancel_broker_order(args) == 0

    persisted_state = store.load()
    assert persisted_state.orders["order_aapl_1"].status == "cancelled"
    assert adapter.cancelled_order_ids == ["alpaca-ord-1"]
    assert payloads[-1]["order_id"] == "order_aapl_1"


def test_handle_cancel_broker_order_marks_order_pending_cancel_when_snapshot_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(cancel_update=None)
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_id="order_aapl_1",
        format="json",
    )

    assert main_module._handle_cancel_broker_order(args) == 0

    persisted_state = store.load()
    record = persisted_state.orders["order_aapl_1"]
    assert record.status == "pending"
    assert record.broker_status == "pending_cancel"
    assert record.last_broker_update_at is not None
    assert record.reserved_notional == pytest.approx(1_000.0)
    assert "no broker order snapshot" in payloads[-1]["warnings"][0]


def test_handle_replace_broker_order_transfers_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
            linked_decision_id="decision_aapl",
            trade_id="trade_aapl",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        replace_update=BrokerOrderUpdate(
            broker_name="alpaca",
            broker_order_id="alpaca-ord-2",
            client_order_id="placeholder",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            quantity=12,
            filled_quantity=0,
            status="accepted",
            limit_price=101.0,
            submitted_at="2024-01-05T15:03:00+00:00",
            updated_at="2024-01-05T15:03:01+00:00",
        )
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_id="order_aapl_1",
        quantity=12,
        limit_price=101.0,
        stop_price=None,
        format="json",
    )

    assert main_module._handle_replace_broker_order(args) == 0

    persisted_state = store.load()
    active_orders = list(persisted_state.active_orders())
    assert persisted_state.orders["order_aapl_1"].status == "replaced"
    assert len(active_orders) == 1
    assert active_orders[0].requested_quantity == 12
    assert active_orders[0].broker_order_id == "alpaca-ord-2"
    assert adapter.replace_requests[0][1].time_in_force == "DAY"
    assert payloads[-1]["replaced_order_id"] == "order_aapl_1"


def test_handle_sync_broker_orders_warns_when_broker_update_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    store = PendingOrderStateStore(tmp_path)
    starting_state = _pending_order_state(
        PendingOrderRecord(
            order_id="order_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T15:00:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            broker_order_id="alpaca-ord-1",
            client_order_id="order_aapl_1",
            broker_status="accepted",
        )
    )
    store.save(starting_state)
    adapter = FakeBrokerAdapter(
        get_order_error=BrokerRequestError("broker temporarily unavailable"),
    )
    payloads: list[dict[str, object]] = []

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir=None: config)
    monkeypatch.setattr(main_module, "_create_broker_adapter", lambda broker, env_file=None: adapter)
    monkeypatch.setattr(main_module, "_print_structured", lambda payload, output_format: payloads.append(payload))

    args = SimpleNamespace(
        config_dir=config.config_dir,
        env_file=None,
        broker="alpaca",
        order_ids=None,
        format="json",
    )

    assert main_module._handle_sync_broker_orders(args) == 0
    assert "broker temporarily unavailable" in payloads[-1]["warnings"][0]
    assert store.load() == starting_state


def _pending_order_state(*records: PendingOrderRecord):
    state = PendingOrderState(orders={})
    for record in records:
        state = create_pending_order(state, record).state
    return state
