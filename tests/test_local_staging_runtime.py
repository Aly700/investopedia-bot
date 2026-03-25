from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time

import pytest

from bot.api.control_api import (
    ForceBrokerSyncCommand,
    OperatorControlService,
    PauseExecutionCommand,
    SetExecutionModeCommand,
)
from bot.broker.execution import BrokerRequestError
from bot.logging_utils import setup_logging
from bot.main import build_parser
from bot.service.local_staging_runtime import (
    LocalStagingRuntimeConfig,
    create_local_staging_runtime,
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


class BlockingRuntimeSupervisor(FakeRuntimeSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def run(self, *, max_iterations: int | None = None) -> FakeRuntimeStatus:
        self.run_calls.append(max_iterations)
        self._stop_event.wait(timeout=5.0)
        return FakeRuntimeStatus(service_state="stopped-by-stop")

    def stop(self) -> None:
        super().stop()
        self._stop_event.set()


class FakeServer:
    def __init__(self, *, host: str, port: int) -> None:
        self.server_address = (host, port or 10000)
        self.serve_forever_calls = 0
        self.shutdown_calls = 0
        self.server_close_calls = 0
        self._shutdown_event = threading.Event()
        self._started_event = threading.Event()

    def serve_forever(self) -> None:
        self.serve_forever_calls += 1
        self._started_event.set()
        self._shutdown_event.wait(timeout=2.0)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._shutdown_event.set()

    def server_close(self) -> None:
        self.server_close_calls += 1
        self._shutdown_event.set()

    def wait_started(self, timeout: float = 2.0) -> bool:
        return self._started_event.wait(timeout=timeout)


class FailingServer(FakeServer):
    def __init__(self, *, host: str, port: int, failure: Exception | None = None) -> None:
        super().__init__(host=host, port=port)
        self.failure = failure or RuntimeError("server exploded")

    def serve_forever(self) -> None:
        self.serve_forever_calls += 1
        self._started_event.set()
        raise self.failure


class DelayedFailingServer(FakeServer):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        delay_seconds: float = 0.2,
    ) -> None:
        super().__init__(host=host, port=port)
        self.delay_seconds = delay_seconds

    def serve_forever(self) -> None:
        self.serve_forever_calls += 1
        self._started_event.set()
        time.sleep(self.delay_seconds)
        raise RuntimeError("delayed server failure")


def test_build_parser_accepts_run_local_staging_runtime_command() -> None:
    args = build_parser().parse_args(
        [
            "run-local-staging-runtime",
            "data/raw/candidate_symbols.txt",
            "--websocket-url",
            "wss://example.test/live",
            "--runtime-root",
            "tmp/local-runtime",
            "--max-iterations",
            "1",
        ]
    )

    assert args.command == "run-local-staging-runtime"
    assert args.runtime_root == Path("tmp/local-runtime")
    assert args.websocket_url == "wss://example.test/live"


def test_create_local_staging_runtime_starts_with_valid_runtime_config(
    tmp_path: Path,
) -> None:
    supervisors: list[FakeRuntimeSupervisor] = []
    servers: list[FakeServer] = []

    def supervisor_factory(*_args) -> FakeRuntimeSupervisor:
        supervisor = FakeRuntimeSupervisor()
        supervisors.append(supervisor)
        return supervisor

    def server_factory(**kwargs) -> FakeServer:
        server = FakeServer(host=kwargs["host"], port=kwargs["port"])
        servers.append(server)
        return server

    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=supervisor_factory,
        internal_api_server_factory=server_factory,
        control_api_server_factory=server_factory,
        dashboard_server_factory=server_factory,
    )

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        assert all(server.wait_started() for server in servers)
        assert runtime.paths.config_dir.is_dir()
        assert runtime.paths.hot_state_mount_path.is_symlink()
        assert runtime.paths.hot_state_dir.is_dir()
        assert runtime.paths.logs_dir.is_dir()
        assert runtime.paths.archive_dir.is_dir()
        assert supervisors[0].run_calls == [1]

        health_payload = runtime.query_service.health()
        safety_payload = runtime.control_service.safety()

        assert health_payload["endpoint"] == "health"
        assert safety_payload["data"]["safety_state"]["execution_mode"] == "paper"
        assert len(servers) == 3
        assert runtime.internal_api_url == "http://127.0.0.1:10000"
        assert runtime.dashboard_url == "http://127.0.0.1:10000/dashboard"

        runtime.write_runtime_env("ALPACA_API_KEY_ID=test-key\n")
        runtime.disconnect_transport()
        runtime.reconnect_transport()

        assert runtime.paths.env_file.read_text(encoding="utf-8") == "ALPACA_API_KEY_ID=test-key\n"
        assert supervisors[0].adapter.disconnect_calls == 1
        assert supervisors[0].adapter.connect_calls == 1
    finally:
        runtime.stop()
        assert all(server.shutdown_calls == 1 for server in servers)
        assert all(server.server_close_calls == 1 for server in servers)


def test_create_local_staging_runtime_rejects_invalid_runtime_config(
    tmp_path: Path,
) -> None:
    config_dir = _repo_config_dir()
    runtime_config = LocalStagingRuntimeConfig(
        source_config_dir=config_dir,
        runtime_root=config_dir.parent,
        hot_state_dir=tmp_path / "hot-state",
        logs_dir=tmp_path / "logs",
        archive_dir=tmp_path / "archive",
    )

    with pytest.raises(ValueError, match="runtime_root must be different"):
        create_local_staging_runtime(
            runtime_config,
            supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
        )


def test_local_staging_runtime_state_survives_restart(tmp_path: Path) -> None:
    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
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

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        result = runtime.control_service.set_execution_mode(
            SetExecutionModeCommand(execution_mode="live")
        )
        assert result.ok is True

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        assert runtime.control_service.safety_state().execution_mode == "live"
    finally:
        runtime.stop()


def test_local_staging_runtime_malformed_safety_state_fails_closed_after_restart(
    tmp_path: Path,
) -> None:
    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
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

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        runtime.corrupt_safety_state()

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        result = runtime.control_service.pause_execution(
            PauseExecutionCommand(reason="verify fail closed")
        )

        assert result.ok is False
        assert result.status == "failed"
        assert result.error_code == "control_state_unavailable"
    finally:
        runtime.stop()


def test_local_staging_runtime_deleted_safety_state_fails_closed_after_restart(
    tmp_path: Path,
) -> None:
    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
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

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        runtime.delete_safety_state()

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        result = runtime.control_service.pause_execution(
            PauseExecutionCommand(reason="verify missing state fail closed")
        )

        assert result.ok is False
        assert result.status == "failed"
        assert result.error_code == "control_state_unavailable"
    finally:
        runtime.stop()


def test_local_staging_runtime_preserves_runtime_env_override_across_restart(
    tmp_path: Path,
) -> None:
    source_env_file = tmp_path / "source.env"
    source_env_file.write_text("BROKER_MODE=healthy\n", encoding="utf-8")
    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path, source_env_file=source_env_file),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
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

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        override = "BROKER_MODE=broken\nRUNTIME_ONLY=1\n"
        runtime.write_runtime_env(override)

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        assert runtime.paths.env_file.read_text(encoding="utf-8") == override
    finally:
        runtime.stop()


def test_local_staging_runtime_broker_unavailable_override_survives_restart(
    tmp_path: Path,
) -> None:
    source_env_file = tmp_path / "source.env"
    source_env_file.write_text("BROKER_MODE=healthy\n", encoding="utf-8")

    def control_service_factory(_config, paths) -> OperatorControlService:
        def broker_adapter_factory(
            broker_name: str,
            *,
            env_file: Path | None = None,
            target: str | None = None,
        ):
            assert broker_name == "alpaca"
            assert target == "paper"
            if env_file is None:
                raise AssertionError("Expected env_file to be provided.")
            env_text = env_file.read_text(encoding="utf-8")
            if "BROKER_MODE=broken" in env_text:
                raise BrokerRequestError("broker unavailable via runtime env override")
            return _EmptyBrokerAdapter()

        return OperatorControlService(
            project_root=paths.runtime_root,
            env_file=paths.env_file,
            broker_adapter_factory=broker_adapter_factory,
            fail_closed_on_missing_safety_state=True,
        )

    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path, source_env_file=source_env_file),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
        control_service_factory=control_service_factory,
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

    try:
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        runtime.write_runtime_env("BROKER_MODE=broken\n")

        runtime = runtime.restart(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        result = runtime.control_service.force_broker_sync(
            ForceBrokerSyncCommand(broker="alpaca")
        )

        assert result.ok is False
        assert result.status == "failed"
        assert result.error_code == "broker_unavailable"
        assert "runtime env override" in result.message
    finally:
        runtime.stop()


def test_local_staging_runtime_rotate_logs_does_not_break_active_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
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
    errors: list[str] = []

    def record_handle_error(self: logging.Handler, record: logging.LogRecord) -> None:
        errors.append(record.getMessage())

    monkeypatch.setattr(logging.Handler, "handleError", record_handle_error)

    stop_event = threading.Event()
    logger = logging.getLogger("tests.local_staging_runtime.rotation")

    def background_logger() -> None:
        counter = 0
        while not stop_event.is_set():
            logger.info("runtime-log-%s-%s", counter, "x" * 80)
            counter += 1

    try:
        runtime.configure_logging(level="INFO")
        runtime.start(max_iterations=1)
        assert runtime.wait_for_supervisor(timeout=2.0) is True
        worker = threading.Thread(target=background_logger, daemon=True)
        worker.start()
        time.sleep(0.05)

        for _ in range(10):
            runtime.rotate_logs()

        logger.info("post-rotation-sentinel")
        stop_event.set()
        worker.join(timeout=2.0)
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert errors == []
        assert "post-rotation-sentinel" in runtime.paths.active_log_path.read_text(
            encoding="utf-8"
        )
    finally:
        stop_event.set()
        runtime.stop()
        setup_logging(level="INFO")


def test_local_staging_runtime_surfaces_server_thread_failure(
    tmp_path: Path,
) -> None:
    supervisors: list[BlockingRuntimeSupervisor] = []
    servers: list[FakeServer] = []

    def supervisor_factory(*_args) -> BlockingRuntimeSupervisor:
        supervisor = BlockingRuntimeSupervisor()
        supervisors.append(supervisor)
        return supervisor

    def server_factory(**kwargs) -> FakeServer:
        if kwargs["host"] == "127.0.0.1" and not servers:
            server = FailingServer(host=kwargs["host"], port=kwargs["port"])
        else:
            server = FakeServer(host=kwargs["host"], port=kwargs["port"])
        servers.append(server)
        return server

    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=supervisor_factory,
        internal_api_server_factory=server_factory,
        control_api_server_factory=server_factory,
        dashboard_server_factory=server_factory,
    )

    try:
        runtime.start()
        with pytest.raises(
            RuntimeError,
            match="Local staging internal-api server exited with an exception.",
        ):
            runtime.wait_for_supervisor(timeout=2.0)
    finally:
        runtime.stop()


def test_local_staging_runtime_stops_remaining_stack_after_server_failure(
    tmp_path: Path,
) -> None:
    supervisors: list[BlockingRuntimeSupervisor] = []
    servers: list[FakeServer] = []

    def supervisor_factory(*_args) -> BlockingRuntimeSupervisor:
        supervisor = BlockingRuntimeSupervisor()
        supervisors.append(supervisor)
        return supervisor

    def server_factory(**kwargs) -> FakeServer:
        if kwargs["port"] == 0 and not servers:
            server = FailingServer(host=kwargs["host"], port=kwargs["port"])
        else:
            server = FakeServer(host=kwargs["host"], port=kwargs["port"])
        servers.append(server)
        return server

    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=supervisor_factory,
        internal_api_server_factory=server_factory,
        control_api_server_factory=server_factory,
        dashboard_server_factory=server_factory,
    )

    try:
        runtime.start()
        with pytest.raises(RuntimeError):
            runtime.wait_for_supervisor(timeout=2.0)
        assert supervisors[0].stop_calls >= 1
        assert all(server.shutdown_calls >= 1 for server in servers[1:])
    finally:
        runtime.stop()


def test_local_staging_runtime_wait_for_stack_surfaces_delayed_server_failure(
    tmp_path: Path,
) -> None:
    servers: list[FakeServer] = []

    def server_factory(**kwargs) -> FakeServer:
        if not servers:
            server = DelayedFailingServer(host=kwargs["host"], port=kwargs["port"])
        else:
            server = FakeServer(host=kwargs["host"], port=kwargs["port"])
        servers.append(server)
        return server

    runtime = create_local_staging_runtime(
        _runtime_config(tmp_path),
        supervisor_factory=lambda *_args: FakeRuntimeSupervisor(),
        internal_api_server_factory=server_factory,
        control_api_server_factory=server_factory,
        dashboard_server_factory=server_factory,
    )

    try:
        runtime.start(max_iterations=1)
        with pytest.raises(
            RuntimeError,
            match="Local staging internal-api server exited with an exception.",
        ):
            runtime.wait_for_stack(timeout=1.0)
    finally:
        runtime.stop()


class _EmptyBrokerAdapter:
    def list_open_orders(self) -> tuple[object, ...]:
        return ()

    def get_order(
        self,
        broker_order_id: str | None = None,
        *,
        client_order_id: str | None = None,
    ) -> object:
        raise AssertionError("get_order() should not be called for an empty sync.")

    def list_positions(self) -> tuple[object, ...]:
        return ()


def _runtime_config(
    tmp_path: Path,
    *,
    source_env_file: Path | None = None,
) -> LocalStagingRuntimeConfig:
    return LocalStagingRuntimeConfig(
        source_config_dir=_repo_config_dir(),
        source_env_file=source_env_file if source_env_file is not None else _repo_root() / ".env",
        runtime_root=tmp_path / "runtime-root",
        hot_state_dir=tmp_path / "hot-state",
        logs_dir=tmp_path / "logs",
        archive_dir=tmp_path / "archive",
        internal_api_port=0,
        control_api_port=0,
        dashboard_port=0,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_config_dir() -> Path:
    return _repo_root() / "config"
