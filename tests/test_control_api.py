from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from http import HTTPStatus
import json
from pathlib import Path

import pytest

from bot.api.control_api import (
    CancelPendingOrderCommand,
    ForceBrokerSyncCommand,
    OperatorControlService,
    OperatorControlState,
    OperatorControlStateStore,
    PauseExecutionCommand,
    ReplacePendingOrderCommand,
    ResumeExecutionCommand,
    SetBrokerTradingEnabledCommand,
    SetExecutionModeCommand,
    SetLiveActionConfirmationCommand,
    SubmitPendingOrderCommand,
    default_operator_control_audit_log_path,
    load_operator_control_audit_records,
    operator_control_response_for_request,
)
from bot.broker.execution import BrokerOrderUpdate, BrokerReplaceOrderRequest, BrokerRequestError
from bot.data.pending_orders import (
    PendingOrderRecord,
    PendingOrderState,
    PendingOrderStateStore,
    create_pending_order,
)
from bot.data.trade_feedback import (
    default_trade_feedback_log_path,
    load_trade_feedback_events,
)
from bot.main import build_parser


class FakeBrokerAdapter:
    def __init__(
        self,
        *,
        placed_update: BrokerOrderUpdate | None = None,
        sync_update: BrokerOrderUpdate | None = None,
        cancel_update: BrokerOrderUpdate | None = None,
        replace_update: BrokerOrderUpdate | None = None,
        get_order_error: Exception | None = None,
    ) -> None:
        self.placed_update = placed_update
        self.sync_update = sync_update
        self.cancel_update = cancel_update
        self.replace_update = replace_update
        self.get_order_error = get_order_error
        self.place_order_calls = 0
        self.cancelled_order_ids: list[str] = []
        self.replace_requests: list[tuple[str, BrokerReplaceOrderRequest]] = []

    def place_order(self, request) -> BrokerOrderUpdate:
        self.place_order_calls += 1
        if self.placed_update is None:
            raise AssertionError("placed_update was not configured.")
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
            raise AssertionError("replace_update was not configured.")
        return replace(
            self.replace_update,
            client_order_id=request.client_order_id or self.replace_update.client_order_id,
        )

    def list_open_orders(self) -> tuple[BrokerOrderUpdate, ...]:
        return (self.sync_update,) if self.sync_update is not None else ()

    def get_order(
        self,
        broker_order_id: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> BrokerOrderUpdate:
        if self.get_order_error is not None:
            raise self.get_order_error
        if self.sync_update is None:
            raise AssertionError("sync_update was not configured.")
        return self.sync_update

    def list_positions(self) -> tuple[object, ...]:
        return ()


def test_operator_control_service_submit_pending_order_executes_and_writes_audit(
    tmp_path: Path,
) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file, target=None: adapter,
        now_utc=lambda: datetime(2024, 1, 5, 15, 1, tzinfo=timezone.utc),
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    persisted = store.load()
    events = load_trade_feedback_events(default_trade_feedback_log_path(tmp_path))
    audit_records = load_operator_control_audit_records(
        default_operator_control_audit_log_path(tmp_path)
    )

    assert result.ok is True
    assert result.status == "completed"
    assert result.data["submitted_count"] == 1
    assert result.safety_state is not None
    assert result.safety_state.execution_mode == "paper"
    assert persisted.orders["order_aapl_1"].broker_order_id == "alpaca-ord-1"
    assert events[-1].event_type == "broker_submitted"
    assert result.audit_record_id == audit_records[-1].audit_record_id
    assert "pending_order_state" in result.artifact_paths
    assert "trade_feedback_log" in result.artifact_paths
    assert "operator_control_audit_log" in result.artifact_paths


def test_operator_control_service_rejects_factory_without_target_support(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
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
        )
    )

    def keyword_only_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
    ) -> FakeBrokerAdapter:
        assert broker_name == "alpaca"
        assert env_file is None
        return adapter

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=keyword_only_factory,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "broker_unavailable"
    assert "unexpected keyword argument 'target'" in result.message


def test_operator_control_service_pause_and_resume_execution_updates_safety_state(
    tmp_path: Path,
) -> None:
    service = OperatorControlService(
        project_root=tmp_path,
        now_utc=lambda: datetime(2024, 1, 5, 15, 0, tzinfo=timezone.utc),
    )

    pause_result = service.pause_execution(
        PauseExecutionCommand(reason="maintenance window")
    )
    resume_result = service.resume_execution(
        ResumeExecutionCommand(reason="maintenance done")
    )

    persisted_state = OperatorControlStateStore(tmp_path).load()

    assert pause_result.ok is True
    assert pause_result.status == "completed"
    assert pause_result.safety_state is not None
    assert pause_result.safety_state.execution_submission_enabled is False
    assert resume_result.ok is True
    assert resume_result.status == "completed"
    assert persisted_state.execution_submission_enabled is True
    assert persisted_state.paused_reason is None


def test_operator_control_service_policy_changes_persist_and_reload(
    tmp_path: Path,
) -> None:
    service = OperatorControlService(
        project_root=tmp_path,
        now_utc=lambda: datetime(2024, 1, 5, 15, 0, tzinfo=timezone.utc),
    )

    mode_result = service.set_execution_mode(
        SetExecutionModeCommand(execution_mode="live")
    )
    confirmation_result = service.set_live_action_confirmation(
        SetLiveActionConfirmationCommand(required=False)
    )
    trading_result = service.set_broker_trading_enabled(
        SetBrokerTradingEnabledCommand(enabled=True)
    )

    persisted_state = OperatorControlStateStore(tmp_path).load()

    assert mode_result.ok is True
    assert mode_result.status == "completed"
    assert confirmation_result.ok is True
    assert trading_result.ok is True
    assert persisted_state.execution_mode == "live"
    assert persisted_state.live_actions_require_confirmation is False
    assert persisted_state.broker_trading_enabled is True


def test_operator_control_service_paper_mode_requests_paper_broker_target(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
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
        )
    )
    captured: dict[str, str | None] = {}

    def target_aware_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
        target: str | None = None,
    ) -> FakeBrokerAdapter:
        assert broker_name == "alpaca"
        assert env_file is None
        captured["target"] = target
        return adapter

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=target_aware_factory,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is True
    assert captured["target"] == "paper"


def test_operator_control_service_live_mode_requests_live_broker_target(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_mode="live",
            broker_trading_enabled=True,
            live_actions_require_confirmation=False,
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
        )
    )
    captured: dict[str, str | None] = {}

    def target_aware_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
        target: str | None = None,
    ) -> FakeBrokerAdapter:
        assert broker_name == "alpaca"
        assert env_file is None
        captured["target"] = target
        return adapter

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=target_aware_factory,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is True
    assert captured["target"] == "live"


def test_operator_control_service_rejects_env_target_mismatch_against_policy(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "ALPACA_API_KEY_ID=test-key",
                "ALPACA_SECRET_KEY=test-secret",
                "ALPACA_BROKER_BASE_URL=https://api.alpaca.markets",
            )
        ),
        encoding="utf-8",
    )
    service = OperatorControlService(
        project_root=tmp_path,
        env_file=env_file,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "broker_unavailable"
    assert "Execution policy requires the Alpaca paper target" in result.message


def test_operator_control_service_rejects_submit_when_paused(tmp_path: Path) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_submission_enabled=False,
            paused_reason="manual pause",
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file, target=None: adapter,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "rejected"
    assert result.error_code == "execution_paused"
    assert adapter.place_order_calls == 0


def test_operator_control_service_cancel_allowed_when_execution_submissions_paused(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="accepted",
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_submission_enabled=False,
            paused_reason="operator pause",
        )
    )
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.cancel_pending_order(
        CancelPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is True
    assert result.status == "completed"
    assert adapter.cancelled_order_ids == ["alpaca-ord-1"]


def test_operator_control_service_replace_allowed_when_execution_submissions_paused(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="accepted",
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_submission_enabled=False,
            paused_reason="operator pause",
        )
    )
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.replace_pending_order(
        ReplacePendingOrderCommand(
            broker="alpaca",
            order_id="order_aapl_1",
            quantity=12,
            limit_price=101.0,
        )
    )

    assert result.ok is True
    assert result.status == "completed"
    assert adapter.replace_requests[0][0] == "alpaca-ord-1"


def test_operator_control_service_rejects_live_submit_without_confirmation(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_mode="live",
            broker_trading_enabled=True,
            live_actions_require_confirmation=True,
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "rejected"
    assert result.error_code == "live_confirmation_required"
    assert adapter.place_order_calls == 0


def test_operator_control_service_rejects_live_submit_when_broker_trading_disabled(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )
    OperatorControlStateStore(tmp_path).save(
        OperatorControlState(
            updated_at_utc="2024-01-05T15:00:00+00:00",
            execution_mode="live",
            broker_trading_enabled=False,
            live_actions_require_confirmation=False,
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "rejected"
    assert result.error_code == "live_broker_trading_disabled"
    assert adapter.place_order_calls == 0


def test_operator_control_service_fails_closed_when_control_state_has_string_bool(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )
    OperatorControlStateStore(tmp_path).path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "execution_submission_enabled": "false",
            }
        ),
        encoding="utf-8",
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "control_state_unavailable"
    assert result.safety_state is not None
    assert result.safety_state.execution_submission_enabled is False
    assert adapter.place_order_calls == 0


def test_operator_control_service_pause_fails_closed_when_control_state_file_is_malformed(
    tmp_path: Path,
) -> None:
    state_path = OperatorControlStateStore(tmp_path).path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not-json", encoding="utf-8")
    service = OperatorControlService(project_root=tmp_path)

    result = service.pause_execution(PauseExecutionCommand(reason="maintenance"))

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "control_state_unavailable"


@pytest.mark.parametrize(
    ("command_builder", "expected_error_code"),
    [
        (
            lambda: SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1"),
            "control_state_unavailable",
        ),
        (
            lambda: CancelPendingOrderCommand(broker="alpaca", order_id="order_aapl_1"),
            "control_state_unavailable",
        ),
        (
            lambda: ReplacePendingOrderCommand(
                broker="alpaca",
                order_id="order_aapl_1",
                quantity=12,
            ),
            "control_state_unavailable",
        ),
    ],
)
def test_operator_control_service_fails_closed_when_control_state_file_is_malformed(
    tmp_path: Path,
    command_builder,
    expected_error_code: str,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="accepted",
            )
        )
    )
    OperatorControlStateStore(tmp_path).path.write_text("{not-json", encoding="utf-8")
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    command = command_builder()
    if isinstance(command, SubmitPendingOrderCommand):
        result = service.submit_pending_order(command)
        assert adapter.place_order_calls == 0
    elif isinstance(command, CancelPendingOrderCommand):
        result = service.cancel_pending_order(command)
        assert adapter.cancelled_order_ids == []
    else:
        result = service.replace_pending_order(command)
        assert adapter.replace_requests == []

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == expected_error_code
    assert result.safety_state is not None
    assert result.safety_state.execution_submission_enabled is False


def test_operator_control_service_rejects_cancel_and_replace_for_wrong_local_state(
    tmp_path: Path,
) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
            )
        )
    )
    service = OperatorControlService(project_root=tmp_path)

    cancel_result = service.cancel_pending_order(
        CancelPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )
    replace_result = service.replace_pending_order(
        ReplacePendingOrderCommand(
            broker="alpaca",
            order_id="order_aapl_1",
            quantity=12,
        )
    )

    assert cancel_result.ok is False
    assert cancel_result.status == "rejected"
    assert cancel_result.error_code == "pending_order_missing_broker_id"
    assert replace_result.ok is False
    assert replace_result.status == "rejected"
    assert replace_result.error_code == "pending_order_missing_broker_id"


def test_operator_control_service_rejects_submit_for_already_submitted_order(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="accepted",
            )
        )
    )
    service = OperatorControlService(project_root=tmp_path)

    result = service.submit_pending_order(
        SubmitPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    assert result.ok is False
    assert result.status == "rejected"
    assert result.error_code == "pending_order_already_submitted"


def test_operator_control_service_cancel_without_snapshot_returns_transitional_result(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="accepted",
            )
        )
    )
    adapter = FakeBrokerAdapter(cancel_update=None)
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file=None, target=None: adapter,
    )

    result = service.cancel_pending_order(
        CancelPendingOrderCommand(broker="alpaca", order_id="order_aapl_1")
    )

    persisted = PendingOrderStateStore(tmp_path).load().orders["order_aapl_1"]

    assert result.ok is True
    assert result.status == "needs_confirmation"
    assert result.data["cancel_confirmed"] is False
    assert result.data["transitional_state"] == "pending_cancel"
    assert result.data["reservation_active"] is True
    assert persisted.status == "pending"
    assert persisted.broker_status == "pending_cancel"
    assert persisted.reserves_capacity is True


def test_operator_control_service_rejects_replace_when_cancel_is_pending(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
                broker_status="pending_cancel",
            )
        )
    )
    service = OperatorControlService(project_root=tmp_path)

    result = service.replace_pending_order(
        ReplacePendingOrderCommand(
            broker="alpaca",
            order_id="order_aapl_1",
            quantity=12,
        )
    )

    assert result.ok is False
    assert result.status == "rejected"
    assert result.error_code == "pending_order_cancel_in_progress"


def test_operator_control_service_force_broker_sync_reconciles_updates(
    tmp_path: Path,
) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
    )
    adapter = FakeBrokerAdapter(
        sync_update=BrokerOrderUpdate(
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
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file, target=None: adapter,
    )

    result = service.force_broker_sync(ForceBrokerSyncCommand(broker="alpaca"))

    assert result.ok is True
    assert result.status == "completed"
    assert result.data["matched_update_count"] == 1
    assert store.load().orders["order_aapl_1"].status == "filled"


def test_operator_control_service_force_broker_sync_requires_target_aware_factory(
    tmp_path: Path,
) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
    )

    def keyword_only_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
    ) -> FakeBrokerAdapter:
        raise AssertionError(f"Factory should not be called without target support: {broker_name}")

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=keyword_only_factory,
    )

    result = service.force_broker_sync(ForceBrokerSyncCommand(broker="alpaca"))

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "broker_unavailable"
    assert "unexpected keyword argument 'target'" in result.message


def test_operator_control_service_returns_clear_failure_when_broker_sync_unavailable(
    tmp_path: Path,
) -> None:
    store = PendingOrderStateStore(tmp_path)
    store.save(
        _pending_order_state(
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
    )
    adapter = FakeBrokerAdapter(get_order_error=BrokerRequestError("broker unavailable"))
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file, target=None: adapter,
    )

    result = service.force_broker_sync(ForceBrokerSyncCommand(broker="alpaca"))

    assert result.ok is False
    assert result.status == "failed"
    assert result.error_code == "broker_sync_failed"
    assert "broker unavailable" in result.warnings[0]


def test_operator_control_response_for_request_returns_structured_submit_result(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
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
        )
    )
    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=lambda broker, env_file, target=None: adapter,
    )

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="POST",
        path="/control/orders/submit",
        body=json.dumps({"broker": "alpaca", "order_id": "order_aapl_1"}).encode("utf-8"),
    )

    assert status == HTTPStatus.OK
    assert payload["command_name"] == "submit_pending_order"
    assert payload["ok"] is True
    assert payload["status"] == "completed"


def test_operator_control_response_for_request_rejects_factory_without_target_support(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
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
        )
    )

    def keyword_only_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
    ) -> FakeBrokerAdapter:
        assert broker_name == "alpaca"
        assert env_file is None
        return adapter

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=keyword_only_factory,
    )

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="POST",
        path="/control/orders/submit",
        body=json.dumps({"broker": "alpaca", "order_id": "order_aapl_1"}).encode("utf-8"),
    )

    assert status == HTTPStatus.BAD_GATEWAY
    assert payload["ok"] is False
    assert payload["error_code"] == "broker_unavailable"
    assert "unexpected keyword argument 'target'" in payload["message"]


def test_operator_control_response_for_request_returns_structured_failure_for_factory_error(
    tmp_path: Path,
) -> None:
    PendingOrderStateStore(tmp_path).save(
        _pending_order_state(
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
            )
        )
    )

    def failing_keyword_only_factory(
        broker_name: str,
        *,
        env_file: Path | None = None,
    ):
        raise BrokerRequestError(f"broker unavailable for {broker_name}")

    service = OperatorControlService(
        project_root=tmp_path,
        broker_adapter_factory=failing_keyword_only_factory,
    )

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="POST",
        path="/control/orders/submit",
        body=json.dumps({"broker": "alpaca", "order_id": "order_aapl_1"}).encode("utf-8"),
    )

    assert status == HTTPStatus.BAD_GATEWAY
    assert payload["ok"] is False
    assert payload["error_code"] == "broker_unavailable"


def test_operator_control_response_for_request_rejects_invalid_payload() -> None:
    service = OperatorControlService(project_root=Path("/tmp"))

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="POST",
        path="/control/orders/replace",
        body=json.dumps({"broker": "alpaca"}).encode("utf-8"),
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert payload["error"] == "invalid_command"


def test_operator_control_response_for_request_returns_control_index() -> None:
    service = OperatorControlService(project_root=Path("/tmp"))

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="GET",
        path="/control",
        body=None,
    )

    assert status == HTTPStatus.OK
    assert payload["service"] == "investopedia-bot-operator-control-api"


def test_operator_control_response_for_request_returns_readable_safety_state_by_default(
    tmp_path: Path,
) -> None:
    service = OperatorControlService(project_root=tmp_path)

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="GET",
        path="/control/safety",
        body=None,
    )

    assert status == HTTPStatus.OK
    assert payload["available"] is True
    assert payload["data"]["control_state_readable"] is True
    assert payload["data"]["safety_state"]["execution_submission_enabled"] is True
    assert payload["data"]["safety_state"]["execution_mode"] == "paper"
    assert payload["data"]["safety_state"]["live_actions_require_confirmation"] is True
    assert payload["data"]["safety_state"]["broker_trading_enabled"] is False
    assert payload["data"]["policy_summary"]["is_live_mode"] is False


def test_operator_control_response_for_request_returns_execution_policy_endpoint(
    tmp_path: Path,
) -> None:
    service = OperatorControlService(project_root=tmp_path)

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="GET",
        path="/control/execution/policy",
        body=None,
    )

    assert status == HTTPStatus.OK
    assert payload["endpoint"] == "control_execution_policy"
    assert payload["data"]["policy_summary"]["execution_mode"] == "paper"


def test_operator_control_response_for_request_sets_execution_mode(
    tmp_path: Path,
) -> None:
    service = OperatorControlService(project_root=tmp_path)

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="POST",
        path="/control/execution/mode",
        body=json.dumps({"execution_mode": "live"}).encode("utf-8"),
    )

    assert status == HTTPStatus.OK
    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert payload["safety_state"]["execution_mode"] == "live"


def test_operator_control_response_for_request_returns_structured_safety_when_state_is_malformed(
    tmp_path: Path,
) -> None:
    state_path = OperatorControlStateStore(tmp_path).path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{broken-json", encoding="utf-8")
    service = OperatorControlService(project_root=tmp_path)

    status, payload = operator_control_response_for_request(
        control_service=service,
        method="GET",
        path="/control/safety",
        body=None,
    )

    assert status == HTTPStatus.OK
    assert payload["available"] is True
    assert payload["warnings"]
    assert payload["data"]["control_state_readable"] is False
    assert payload["data"]["safety_state"]["execution_submission_enabled"] is False


def test_build_parser_accepts_serve_operator_control_api_command() -> None:
    args = build_parser().parse_args(
        [
            "serve-operator-control-api",
            "--host",
            "127.0.0.1",
            "--port",
            "8766",
        ]
    )

    assert args.command == "serve-operator-control-api"
    assert args.port == 8766


def _pending_order_state(*records: PendingOrderRecord) -> PendingOrderState:
    state = PendingOrderState(orders={})
    for record in records:
        state = create_pending_order(state, record).state
    return state
