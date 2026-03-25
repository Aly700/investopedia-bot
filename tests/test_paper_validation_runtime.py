from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import threading

import pytest

import bot.main as main_module
from bot.api.control_api import PauseExecutionCommand, SetExecutionModeCommand
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
        assert summary.trade_feedback["broker_submitted_count"] == 1
        assert summary.log_summary["warning_count"] == 1
        assert summary.log_summary["error_count"] == 1
        assert outputs["paper_validation_summary_json"].is_file()
        assert outputs["paper_validation_checkpoint_json"].is_file()
        assert outputs["paper_validation_brief"].is_file()

        payload = json.loads(
            outputs["paper_validation_summary_json"].read_text(encoding="utf-8")
        )
        assert payload["health_checkpoint"]["service_state"] == "connected"
        assert "Paper guardrail active" in outputs["paper_validation_brief"].read_text(
            encoding="utf-8"
        )
    finally:
        runtime.stop()


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
