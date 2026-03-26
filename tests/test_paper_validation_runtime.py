from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import threading

import pytest

import bot.main as main_module
from bot.api.control_api import (
    OperatorControlAuditRecord,
    OperatorControlState,
    PauseExecutionCommand,
    SetExecutionModeCommand,
    append_operator_control_audit_record,
    default_operator_control_audit_log_path,
)
from bot.data.trade_feedback import TradeFeedbackEvent, TradeFeedbackLogStore
from bot.main import build_parser
from bot.service.live_market_service import LiveMarketServiceStatus, LiveMarketServiceStatusStore
from bot.service.paper_validation import (
    build_local_paper_validation_profile,
    build_local_paper_validation_runtime_config,
    build_local_paper_validation_summary,
    create_local_paper_validation_runtime,
    write_local_paper_validation_summary,
)


@dataclass
class FakeRuntimeStatus:
    service_state: str = "stopped"

    def to_dict(self) -> dict[str, str]:
        return {"service_state": self.service_state}


class FakeTransportAdapter:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class FakeRuntimeSupervisor:
    def __init__(self) -> None:
        self.adapter = FakeTransportAdapter()
        self.run_calls: list[int | None] = []
        self.stop_calls = 0

    def run(self, *, max_iterations: int | None = None) -> FakeRuntimeStatus:
        self.run_calls.append(max_iterations)
        return FakeRuntimeStatus()

    def stop(self) -> None:
        self.stop_calls += 1


class FakeServer:
    def __init__(self, *, host: str, port: int) -> None:
        self.server_address = (host, port or 10000)
        self.shutdown_calls = 0
        self.server_close_calls = 0
        self._shutdown_event = threading.Event()

    def serve_forever(self) -> None:
        self._shutdown_event.wait(timeout=2.0)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._shutdown_event.set()

    def server_close(self) -> None:
        self.server_close_calls += 1
        self._shutdown_event.set()


def test_build_parser_accepts_run_local_paper_validation_command() -> None:
    args = build_parser().parse_args(
        [
            "run-local-paper-validation",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
        ]
    )

    assert args.command == "run-local-paper-validation"
    assert args.websocket_url == "wss://example.test/live"


def test_build_parser_accepts_run_local_paper_validation_smoke_command() -> None:
    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
        ]
    )

    assert args.command == "run-local-paper-validation-smoke"
    assert args.websocket_url == "wss://example.test/live"
    assert args.max_iterations == 5


def test_build_local_paper_validation_profile_stays_isolated_from_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    profile = build_local_paper_validation_profile(project_root=project_root)
    runtime_config = build_local_paper_validation_runtime_config(
        profile=profile,
        source_config_dir=project_root / "config",
    )

    assert profile.validation_root == project_root / "data" / "local_paper_validation"
    assert profile.runtime_root != project_root
    assert profile.hot_state_dir.parent == profile.validation_root
    assert runtime_config.runtime_root == profile.runtime_root
    assert runtime_config.hot_state_dir == profile.hot_state_dir
    assert runtime_config.logs_dir == profile.logs_dir
    assert runtime_config.archive_dir == profile.archive_dir


def test_local_paper_validation_runtime_forces_paper_mode_and_locks_live_mode(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_stack(timeout=2.0) is True

        safety_state = runtime.control_service.safety_state()
        result = runtime.control_service.set_execution_mode(
            SetExecutionModeCommand(execution_mode="live")
        )

        assert safety_state.execution_mode == "paper"
        assert safety_state.execution_submission_enabled is True
        assert safety_state.broker_trading_enabled is False
        assert result.ok is False
        assert result.status == "rejected"
        assert result.error_code == "paper_validation_mode_locked"
    finally:
        runtime.stop()


def test_local_paper_validation_runtime_recovery_helpers_remain_usable_in_paper_mode(
    tmp_path: Path,
) -> None:
    runtime, supervisors = _create_runtime(tmp_path)

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_stack(timeout=2.0) is True

        runtime.write_runtime_env("BROKER_MODE=broken\n")
        runtime.disconnect_transport()
        runtime.reconnect_transport()
        runtime.corrupt_safety_state()

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_stack(timeout=2.0) is True

        failed_pause = runtime.control_service.pause_execution(
            PauseExecutionCommand(reason="expect fail closed")
        )
        reset_path = runtime.reset_safety_state()
        recovered_pause = runtime.control_service.pause_execution(
            PauseExecutionCommand(reason="after reset")
        )

        assert supervisors[0].adapter.disconnect_calls == 1
        assert supervisors[0].adapter.connect_calls == 1
        assert runtime.paths.env_file.read_text(encoding="utf-8") == "BROKER_MODE=broken\n"
        assert failed_pause.ok is False
        assert failed_pause.error_code == "control_state_unavailable"
        assert reset_path.exists()
        assert recovered_pause.ok is True
        assert recovered_pause.status == "completed"
    finally:
        runtime.stop()


def test_local_paper_validation_summary_writes_clean_review_outputs(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_stack(timeout=2.0) is True
        runtime.stop()

        LiveMarketServiceStatusStore(runtime.paths.runtime_root).save(
            LiveMarketServiceStatus(
                service_state="connected",
                connected=True,
                reconnect_attempt_count=2,
                warning_count=1,
                last_message_at_utc="2024-01-05T15:02:00+00:00",
                last_successful_flush_at_utc="2024-01-05T15:01:00+00:00",
                last_cycle_status="completed",
                last_cycle_warning_count=1,
                last_warning="websocket transport disconnected: simulated",
            ),
            updated_at_utc="2024-01-05T15:02:30+00:00",
        )
        TradeFeedbackLogStore(runtime.paths.runtime_root).append(
            (
                TradeFeedbackEvent(
                    event_type="broker_submitted",
                    workflow="control-submit-pending-order",
                    symbol="AAPL",
                    as_of_date=date(2024, 1, 5),
                    timestamp_utc="2024-01-05T15:01:00+00:00",
                    metadata={"order_id": "order_aapl_1"},
                ),
            )
        )
        runtime.control_service.set_execution_mode(
            SetExecutionModeCommand(execution_mode="live")
        )
        log_path = runtime.profile.logs_dir / "local_paper_validation.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    "2024-01-05 15:00:00,000 | WARNING  | bot.service.live_market_service | websocket transport disconnected: simulated",
                    "2024-01-05 15:01:00,000 | ERROR    | bot.service.live_market_service | cycle result handler failed: simulated",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = build_local_paper_validation_summary(
            runtime.profile,
            as_of_date=date(2024, 1, 5),
            window_days=1,
        )
        outputs = write_local_paper_validation_summary(
            summary,
            output_dir=tmp_path / "review",
        )

        assert summary.safety["paper_guardrail_active"] is True
        assert summary.safety["paper_only_intact"] is True
        assert summary.paper_integrity["execution_mode_stayed_paper"] is True
        assert summary.paper_integrity["broker_target_stayed_paper"] is True
        assert summary.paper_integrity["live_confirmation_enabled_ever"] is True
        assert summary.paper_integrity["live_trading_enabled_ever"] is False
        assert summary.trade_feedback["broker_submitted_count"] == 1
        assert summary.log_summary["warning_count"] == 1
        assert summary.log_summary["error_count"] == 1
        assert outputs["paper_validation_summary_json"].is_file()
        assert outputs["paper_validation_checkpoint_json"].is_file()
        assert outputs["paper_validation_brief"].is_file()
        assert outputs["paper_validation_daily_review_json"].is_file()
        assert outputs["paper_validation_daily_review"].is_file()
        assert outputs["paper_validation_changes_today"].is_file()
        assert outputs["paper_validation_operator_checklist"].is_file()

        payload = json.loads(
            outputs["paper_validation_summary_json"].read_text(encoding="utf-8")
        )
        assert payload["health_checkpoint"]["service_state"] == "connected"
        assert payload["paper_integrity"]["paper_only_intact"] is True
        assert payload["daily_review"]["service"]["healthy"] is True
        assert payload["daily_review"]["paper_integrity"]["broker_target_stayed_paper"] is True
        assert isinstance(payload["operator_checklist"], list)
        assert "Paper guardrail active" in outputs["paper_validation_brief"].read_text(
            encoding="utf-8"
        )
        assert "Safety" in outputs["paper_validation_daily_review"].read_text(
            encoding="utf-8"
        )
        assert "What changed" in outputs["paper_validation_changes_today"].read_text(
            encoding="utf-8"
        )
        assert "operator checklist" in outputs[
            "paper_validation_operator_checklist"
        ].read_text(encoding="utf-8").lower()
    finally:
        runtime.stop()


def test_local_paper_validation_summary_generates_anomaly_flags(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)
    now_utc = datetime.now(timezone.utc)
    review_date = now_utc.date()
    timestamp_utc = now_utc.isoformat()

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_stack(timeout=2.0) is True
        runtime.stop()

        LiveMarketServiceStatusStore(runtime.paths.runtime_root).save(
            LiveMarketServiceStatus(
                service_state="connected",
                connected=True,
                reconnect_attempt_count=4,
                warning_count=4,
                last_message_at_utc=timestamp_utc,
                last_successful_flush_at_utc=timestamp_utc,
                last_cycle_status="completed",
            ),
            updated_at_utc=timestamp_utc,
        )
        runtime.disconnect_transport()
        runtime.reconnect_transport()
        runtime.corrupt_safety_state()
        runtime.reset_safety_state()
        _append_control_audit_record(
            runtime.paths.runtime_root,
            command_name="force_broker_sync",
            requested_at_utc=timestamp_utc,
            status="completed",
            message="Broker sync completed with drift.",
            warnings=("Broker open-order scan unavailable: simulated",),
            safety_state=OperatorControlState(updated_at_utc=timestamp_utc),
            result_data={"unmatched_update_count": 1},
        )
        _append_control_audit_record(
            runtime.paths.runtime_root,
            command_name="force_broker_sync",
            requested_at_utc=timestamp_utc,
            status="completed",
            message="Broker sync completed with drift.",
            warnings=("Broker open-order scan unavailable: simulated",),
            safety_state=OperatorControlState(updated_at_utc=timestamp_utc),
            result_data={"unmatched_update_count": 2},
        )
        (runtime.profile.logs_dir / "local_paper_validation.log").write_text(
            "\n".join(
                [
                    f"{review_date.isoformat()} 10:00:00,000 | WARNING  | bot.service.live_market_service | websocket transport disconnected: simulated",
                    f"{review_date.isoformat()} 10:05:00,000 | WARNING  | bot.service.live_market_service | websocket transport disconnected: simulated",
                    f"{review_date.isoformat()} 10:10:00,000 | WARNING  | bot.service.live_market_service | websocket transport disconnected: simulated",
                    f"{review_date.isoformat()} 10:15:00,000 | WARNING  | bot.service.live_market_service | Notification routing failed: simulated",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = build_local_paper_validation_summary(
            runtime.profile,
            as_of_date=review_date,
            window_days=1,
        )

        flag_codes = {flag["code"] for flag in summary.anomaly_flags}
        assert "repeated_reconnects" in flag_codes
        assert "repeated_broker_sync_warnings" in flag_codes
        assert "repeated_malformed_state_recoveries" in flag_codes
        assert "notification_delivery_failures" in flag_codes
        assert "paper_integrity_warning" in flag_codes
    finally:
        runtime.stop()


def test_local_paper_validation_summary_ignores_cumulative_reconnect_attempts_without_window_churn(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)
    now_utc = datetime.now(timezone.utc)
    review_date = now_utc.date()
    timestamp_utc = now_utc.isoformat()

    try:
        LiveMarketServiceStatusStore(runtime.paths.runtime_root).save(
            LiveMarketServiceStatus(
                service_state="connected",
                connected=True,
                reconnect_attempt_count=5,
                warning_count=0,
                last_message_at_utc=timestamp_utc,
                last_successful_flush_at_utc=timestamp_utc,
                last_cycle_status="completed",
            ),
            updated_at_utc=timestamp_utc,
        )

        summary = build_local_paper_validation_summary(
            runtime.profile,
            as_of_date=review_date,
            window_days=1,
        )

        flag_codes = {flag["code"] for flag in summary.anomaly_flags}
        assert "repeated_reconnects" not in flag_codes
    finally:
        runtime.stop()


def test_local_paper_validation_summary_reports_paper_integrity_review_fields(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)
    now_utc = datetime.now(timezone.utc)
    timestamp_utc = now_utc.isoformat()

    try:
        _append_control_audit_record(
            runtime.paths.runtime_root,
            command_name="set_broker_trading_enabled",
            requested_at_utc=timestamp_utc,
            status="completed",
            message="Live broker trading enabled for test coverage.",
            safety_state=OperatorControlState(
                updated_at_utc=timestamp_utc,
                execution_mode="paper",
                live_actions_require_confirmation=True,
                broker_trading_enabled=True,
            ),
        )
        _append_control_audit_record(
            runtime.paths.runtime_root,
            command_name="force_broker_sync",
            requested_at_utc=timestamp_utc,
            status="completed",
            message="Broker sync completed.",
            safety_state=OperatorControlState(updated_at_utc=timestamp_utc),
            result_data={"unmatched_update_count": 0},
        )

        summary = build_local_paper_validation_summary(
            runtime.profile,
            as_of_date=now_utc.date(),
            window_days=1,
        )

        assert summary.paper_integrity["execution_mode_stayed_paper"] is True
        assert summary.paper_integrity["broker_target_stayed_paper"] is True
        assert summary.paper_integrity["live_confirmation_enabled_ever"] is True
        assert summary.paper_integrity["live_trading_enabled_ever"] is True
        assert summary.paper_integrity["control_state_degradation_count"] == 0
    finally:
        runtime.stop()


def test_local_paper_validation_summary_explains_windowed_control_state_degradation_without_live_mode_success(
    tmp_path: Path,
) -> None:
    runtime, _supervisors = _create_runtime(tmp_path)

    try:
        runtime.corrupt_safety_state()
        failed_pause = runtime.control_service.pause_execution(
            PauseExecutionCommand(reason="resilience test")
        )
        runtime.reset_safety_state()

        summary = build_local_paper_validation_summary(
            runtime.profile,
            as_of_date=datetime.now(timezone.utc).date(),
            window_days=1,
        )

        assert failed_pause.ok is False
        assert failed_pause.error_code == "control_state_unavailable"
        assert summary.paper_integrity["live_mode_success_count"] == 0
        assert summary.paper_integrity["control_state_degradation_window_count"] == 2
        assert any(
            "without any observed live-mode success evidence" in note
            for note in summary.paper_integrity["evidence_notes"]
        )
        assert (
            summary.daily_review["paper_integrity"]["control_state_degradation_window_count"]
            == 2
        )
        safety_check = next(
            item
            for item in summary.operator_checklist
            if item["item"] == "Safety boundaries"
        )
        assert "control_state_degradation_window_count=2" in safety_check["detail"]
        review_text = summary.to_daily_review_text()
        assert "Evidence notes" in review_text
        assert "without any observed live-mode success evidence" in review_text
    finally:
        runtime.stop()


def test_local_paper_validation_summary_handles_missing_artifacts_gracefully(
    tmp_path: Path,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=tmp_path / "repo",
        validation_root=tmp_path / "validation-root",
    )

    summary = build_local_paper_validation_summary(
        profile,
        as_of_date=date.today(),
        window_days=1,
    )
    outputs = write_local_paper_validation_summary(
        summary,
        output_dir=tmp_path / "review",
    )

    assert summary.health_checkpoint["available"] is False
    assert summary.pending_orders["available"] is False
    assert summary.market_state["available"] is False
    assert summary.paper_integrity["control_state_readable"] is False
    assert summary.anomaly_flags
    assert outputs["paper_validation_daily_review_json"].is_file()
    assert outputs["paper_validation_daily_review"].is_file()
    assert outputs["paper_validation_operator_checklist"].is_file()


def test_run_local_paper_validation_writes_review_artifacts_on_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    created_runtime: _FailingPaperValidationRuntime | None = None

    def create_runtime(**_kwargs) -> _FailingPaperValidationRuntime:
        nonlocal created_runtime
        created_runtime = _FailingPaperValidationRuntime(profile)
        return created_runtime

    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        create_runtime,
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    with pytest.raises(ValueError, match="simulated validation server failure"):
        main_module._handle_run_local_paper_validation(args)

    assert created_runtime is not None
    assert created_runtime.stop_calls == 1
    assert created_runtime.write_review_summary_calls == [
        (args.as_of, 1, None),
    ]
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    assert (review_output_dir / "paper_validation_summary.json").is_file()
    assert (review_output_dir / "paper_validation_checkpoint.json").is_file()
    assert (review_output_dir / "paper_validation_brief.txt").is_file()


def test_run_local_paper_validation_smoke_writes_passed_result_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    created_runtime: _SmokeTestPaperValidationRuntime | None = None
    service_status_path = _write_smoke_service_status(
        profile,
        service_state="connected",
        cycle_count=1,
    )

    def create_runtime(**_kwargs) -> _SmokeTestPaperValidationRuntime:
        nonlocal created_runtime
        created_runtime = _SmokeTestPaperValidationRuntime(profile)
        return created_runtime

    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        create_runtime,
    )
    monkeypatch.setattr(
        main_module,
        "_run_local_paper_validation_smoke_probes",
        lambda runtime: _successful_smoke_probes(runtime),
    )
    monkeypatch.setattr(
        main_module,
        "build_local_paper_validation_summary",
        lambda *_args, **kwargs: _FakeSmokeSummary(
            as_of_date=kwargs["as_of_date"],
            service_status_path=service_status_path,
            cycle_count=1,
        ),
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    exit_code = main_module._handle_run_local_paper_validation_smoke(args)

    assert exit_code == 0
    assert created_runtime is not None
    assert created_runtime.start_calls == [5]
    assert created_runtime.stop_calls == 1
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    smoke_result_path = review_output_dir / "smoke_run_result.json"
    smoke_brief_path = review_output_dir / "smoke_run_brief.txt"
    assert smoke_result_path.is_file()
    assert smoke_brief_path.is_file()
    payload = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["startup_succeeded"] is True
    assert payload["api_available"] is True
    assert payload["dashboard_available"] is True
    assert payload["paper_only_intact"] is True
    assert payload["review_artifacts_written"] is True
    assert payload["successful_cycle_count"] == 1


def test_run_local_paper_validation_smoke_failure_still_writes_result_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    created_runtime: _SmokeTestPaperValidationRuntime | None = None
    service_status_path = _write_smoke_service_status(
        profile,
        service_state="starting",
        cycle_count=0,
        last_successful_flush_at_utc=None,
        last_cycle_status=None,
    )

    def create_runtime(**_kwargs) -> _SmokeTestPaperValidationRuntime:
        nonlocal created_runtime
        created_runtime = _SmokeTestPaperValidationRuntime(
            profile,
            wait_error=RuntimeError("simulated smoke runtime failure"),
        )
        return created_runtime

    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        create_runtime,
    )
    monkeypatch.setattr(
        main_module,
        "_run_local_paper_validation_smoke_probes",
        lambda runtime: _successful_smoke_probes(runtime),
    )
    monkeypatch.setattr(
        main_module,
        "build_local_paper_validation_summary",
        lambda *_args, **kwargs: _FakeSmokeSummary(
            as_of_date=kwargs["as_of_date"],
            service_status_path=service_status_path,
            cycle_count=0,
            last_successful_cycle_at_utc=None,
        ),
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    exit_code = main_module._handle_run_local_paper_validation_smoke(args)

    assert exit_code == 1
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    smoke_result_path = review_output_dir / "smoke_run_result.json"
    assert smoke_result_path.is_file()
    payload = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any(
        "simulated smoke runtime failure" in issue
        for issue in payload["blocking_issues"]
    )


def test_run_local_paper_validation_smoke_fails_on_paper_only_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    _write_smoke_service_status(
        profile,
        service_state="connected",
        cycle_count=1,
    )

    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        lambda **_kwargs: _SmokeTestPaperValidationRuntime(profile),
    )
    monkeypatch.setattr(
        main_module,
        "_run_local_paper_validation_smoke_probes",
        lambda runtime: _successful_smoke_probes(runtime),
    )
    monkeypatch.setattr(
        main_module,
        "build_local_paper_validation_summary",
        lambda *_args, **kwargs: _FakeSmokeSummary(
            as_of_date=kwargs["as_of_date"],
            service_status_path=LiveMarketServiceStatusStore(profile.runtime_root).path,
            cycle_count=1,
            paper_only_intact=False,
            execution_mode_current="live",
            paper_guardrail_active=False,
        ),
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    exit_code = main_module._handle_run_local_paper_validation_smoke(args)

    assert exit_code == 1
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    payload = json.loads((review_output_dir / "smoke_run_result.json").read_text(encoding="utf-8"))
    assert payload["paper_only_intact"] is False
    assert any(
        "paper" in issue.lower() or "guardrail" in issue.lower()
        for issue in payload["blocking_issues"]
    )


def test_run_local_paper_validation_smoke_marks_missing_api_and_dashboard_as_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    service_status_path = _write_smoke_service_status(
        profile,
        service_state="connected",
        cycle_count=1,
    )

    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        lambda **_kwargs: _SmokeTestPaperValidationRuntime(profile),
    )
    monkeypatch.setattr(
        main_module,
        "_run_local_paper_validation_smoke_probes",
        lambda runtime: {
            "internal_api": {
                "url": runtime.internal_api_url.rstrip("/") + "/health",
                "available": False,
                "status_code": None,
                "error": "connection refused",
            },
            "control_api": {
                "url": runtime.control_api_url.rstrip("/") + "/control/safety",
                "available": True,
                "status_code": 200,
                "error": None,
            },
            "dashboard": {
                "url": runtime.dashboard_url,
                "available": False,
                "status_code": None,
                "error": "connection refused",
            },
        },
    )
    monkeypatch.setattr(
        main_module,
        "build_local_paper_validation_summary",
        lambda *_args, **kwargs: _FakeSmokeSummary(
            as_of_date=kwargs["as_of_date"],
            service_status_path=service_status_path,
            cycle_count=1,
        ),
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    exit_code = main_module._handle_run_local_paper_validation_smoke(args)

    assert exit_code == 1
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    payload = json.loads((review_output_dir / "smoke_run_result.json").read_text(encoding="utf-8"))
    assert payload["internal_api_available"] is False
    assert payload["dashboard_available"] is False
    assert any("internal api" in issue.lower() for issue in payload["blocking_issues"])
    assert any("dashboard" in issue.lower() for issue in payload["blocking_issues"])


def test_run_local_paper_validation_smoke_writes_result_on_unexpected_summary_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    _write_smoke_service_status(profile, service_state="connected", cycle_count=1)
    monkeypatch.setattr(
        main_module,
        "create_local_paper_validation_runtime",
        lambda **_kwargs: _SmokeTestPaperValidationRuntime(profile),
    )
    monkeypatch.setattr(
        main_module,
        "_run_local_paper_validation_smoke_probes",
        lambda runtime: _successful_smoke_probes(runtime),
    )
    monkeypatch.setattr(
        main_module,
        "build_local_paper_validation_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AttributeError("unexpected")),
    )

    args = build_parser().parse_args(
        [
            "run-local-paper-validation-smoke",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--validation-root",
            str(profile.validation_root),
        ]
    )

    exit_code = main_module._handle_run_local_paper_validation_smoke(args)

    assert exit_code == 1
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    smoke_result_path = review_output_dir / "smoke_run_result.json"
    assert smoke_result_path.is_file()
    payload = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert any("unexpected" in issue for issue in payload["blocking_issues"])


def _create_runtime(
    tmp_path: Path,
) -> tuple[object, list[FakeRuntimeSupervisor]]:
    supervisors: list[FakeRuntimeSupervisor] = []
    source_env_file = tmp_path / "validation.env"
    source_env_file.write_text("NOTIFICATION_WEBHOOK_URL=https://example.test\n", encoding="utf-8")
    profile = build_local_paper_validation_profile(
        project_root=_repo_root(),
        validation_root=tmp_path / "validation-root",
    )
    runtime_config = build_local_paper_validation_runtime_config(
        profile=profile,
        source_config_dir=_repo_config_dir(),
        source_env_file=source_env_file,
        internal_api_port=0,
        control_api_port=0,
        dashboard_port=0,
    )

    def supervisor_factory(*_args) -> FakeRuntimeSupervisor:
        supervisor = FakeRuntimeSupervisor()
        supervisors.append(supervisor)
        return supervisor

    runtime = create_local_paper_validation_runtime(
        runtime_config=runtime_config,
        validation_root=profile.validation_root,
        supervisor_factory=supervisor_factory,
        internal_api_server_factory=lambda **kwargs: FakeServer(
            host=kwargs["host"],
            port=kwargs["port"],
        ),
        control_api_server_factory=lambda **kwargs: FakeServer(
            host=kwargs["host"],
            port=kwargs["port"],
        ),
        dashboard_server_factory=lambda **kwargs: FakeServer(
            host=kwargs["host"],
            port=kwargs["port"],
        ),
    )
    return runtime, supervisors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_config_dir() -> Path:
    return _repo_root() / "config"


def _append_control_audit_record(
    project_root: Path,
    *,
    command_name: str,
    requested_at_utc: str,
    status: str,
    message: str,
    safety_state: OperatorControlState,
    warnings: tuple[str, ...] = (),
    result_data: dict[str, object] | None = None,
) -> None:
    record = OperatorControlAuditRecord(
        command_name=command_name,
        requested_at_utc=requested_at_utc,
        status=status,
        message=message,
        warnings=warnings,
        command_payload={},
        result_payload={
            "safety_state": safety_state.to_dict(),
            "data": dict(result_data or {}),
        },
    )
    append_operator_control_audit_record(
        record,
        default_operator_control_audit_log_path(project_root),
    )


class _FailingPaperValidationRuntime:
    def __init__(self, profile: object) -> None:
        self.profile = profile
        self.internal_api_url = "http://127.0.0.1:8765"
        self.control_api_url = "http://127.0.0.1:8766/control"
        self.dashboard_url = "http://127.0.0.1:8780/dashboard"
        self.last_supervisor_result = None
        self.stop_calls = 0
        self.write_review_summary_calls: list[tuple[date | None, int, Path | None]] = []

    def configure_logging(self, *, level: str = "INFO", json_logs: bool = False) -> None:
        _ = (level, json_logs)

    def start(self, *, max_iterations: int | None = None) -> None:
        _ = max_iterations

    def wait_for_stack(self, timeout: float | None = None) -> bool:
        _ = timeout
        raise RuntimeError("simulated validation server failure")

    def stop(self) -> None:
        self.stop_calls += 1

    def write_review_summary(
        self,
        *,
        as_of_date: date | None = None,
        window_days: int = 1,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        self.write_review_summary_calls.append((as_of_date, window_days, output_dir))
        resolved_output_dir = output_dir or self.profile.default_review_output_dir(
            as_of_date=as_of_date or date.today()
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = resolved_output_dir / "paper_validation_summary.json"
        checkpoint_path = resolved_output_dir / "paper_validation_checkpoint.json"
        brief_path = resolved_output_dir / "paper_validation_brief.txt"
        summary_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_text("{}", encoding="utf-8")
        brief_path.write_text("failure review written", encoding="utf-8")
        return {
            "paper_validation_summary_json": summary_path,
            "paper_validation_checkpoint_json": checkpoint_path,
            "paper_validation_brief": brief_path,
        }


class _SmokeTestPaperValidationRuntime:
    def __init__(
        self,
        profile: object,
        *,
        wait_error: Exception | None = None,
    ) -> None:
        self.profile = profile
        self.internal_api_url = "http://127.0.0.1:8765"
        self.control_api_url = "http://127.0.0.1:8766"
        self.dashboard_url = "http://127.0.0.1:8780/dashboard"
        self.last_supervisor_result = FakeRuntimeStatus(service_state="connected")
        self.stop_calls = 0
        self.start_calls: list[int | None] = []
        self.wait_error = wait_error

    def configure_logging(self, *, level: str = "INFO", json_logs: bool = False) -> None:
        _ = (level, json_logs)

    def start(self, *, max_iterations: int | None = None) -> None:
        self.start_calls.append(max_iterations)

    def wait_for_stack(self, timeout: float | None = None) -> bool:
        _ = timeout
        if self.wait_error is not None:
            raise self.wait_error
        return True

    def stop(self) -> None:
        self.stop_calls += 1

    def write_review_summary(
        self,
        *,
        as_of_date: date | None = None,
        window_days: int = 1,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        _ = window_days
        resolved_output_dir = output_dir or self.profile.default_review_output_dir(
            as_of_date=as_of_date or date.today()
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = resolved_output_dir / "paper_validation_summary.json"
        checkpoint_path = resolved_output_dir / "paper_validation_checkpoint.json"
        brief_path = resolved_output_dir / "paper_validation_brief.txt"
        summary_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_text("{}", encoding="utf-8")
        brief_path.write_text("smoke review written", encoding="utf-8")
        return {
            "paper_validation_summary_json": summary_path,
            "paper_validation_checkpoint_json": checkpoint_path,
            "paper_validation_brief": brief_path,
        }


@dataclass(frozen=True)
class _FakeSmokeSummary:
    as_of_date: date
    service_status_path: Path
    cycle_count: int
    last_successful_cycle_at_utc: str | None = "2024-01-05T15:01:00+00:00"
    paper_only_intact: bool = True
    execution_mode_current: str = "paper"
    execution_submission_enabled: bool = True
    broker_trading_enabled: bool = False
    paper_guardrail_active: bool = True
    control_state_readable: bool = True
    generated_at_utc: str = "2024-01-05T15:05:00+00:00"
    warnings: tuple[str, ...] = ()
    anomaly_flags: tuple[dict[str, object], ...] = ()
    log_warning_count: int = 0

    @property
    def artifact_paths(self) -> dict[str, str]:
        return {
            "live_market_service_status": str(self.service_status_path.resolve()),
        }

    @property
    def health_checkpoint(self) -> dict[str, object]:
        return {
            "available": True,
            "service_state": "connected",
            "connected": True,
            "stale": False,
            "cycle_count": self.cycle_count,
            "last_successful_flush_at_utc": self.last_successful_cycle_at_utc,
        }

    @property
    def safety(self) -> dict[str, object]:
        return {
            "paper_guardrail_active": self.paper_guardrail_active,
            "current_execution_mode": self.execution_mode_current,
            "execution_submission_enabled": self.execution_submission_enabled,
            "broker_trading_enabled": self.broker_trading_enabled,
            "control_state_readable": self.control_state_readable,
        }

    @property
    def paper_integrity(self) -> dict[str, object]:
        return {
            "paper_only_intact": self.paper_only_intact,
            "integrity_warnings": (),
            "evidence_notes": (),
        }

    @property
    def log_summary(self) -> dict[str, object]:
        return {"warning_count": self.log_warning_count}


def _successful_smoke_probes(runtime: object) -> dict[str, dict[str, object]]:
    internal_api_url = getattr(runtime, "internal_api_url")
    control_api_url = getattr(runtime, "control_api_url")
    dashboard_url = getattr(runtime, "dashboard_url")
    return {
        "internal_api": {
            "url": internal_api_url.rstrip("/") + "/health",
            "available": True,
            "status_code": 200,
            "error": None,
        },
        "control_api": {
            "url": control_api_url.rstrip("/") + "/control/safety",
            "available": True,
            "status_code": 200,
            "error": None,
        },
        "dashboard": {
            "url": dashboard_url,
            "available": True,
            "status_code": 200,
            "error": None,
        },
    }


def _write_smoke_service_status(
    profile: object,
    *,
    service_state: str,
    cycle_count: int,
    last_successful_flush_at_utc: str | None = "2024-01-05T15:01:00+00:00",
    last_cycle_status: str | None = "completed",
) -> Path:
    store = LiveMarketServiceStatusStore(profile.runtime_root)
    store.save(
        LiveMarketServiceStatus(
            service_state=service_state,
            connected=service_state == "connected",
            cycle_count=cycle_count,
            reconnect_attempt_count=0,
            last_successful_flush_at_utc=last_successful_flush_at_utc,
            last_cycle_status=last_cycle_status,
        ),
        updated_at_utc="2024-01-05T15:02:00+00:00",
    )
    return store.path
