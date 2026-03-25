from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import pytest

from bot.api.control_api import PauseExecutionCommand, SetExecutionModeCommand
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


def _runtime_config(tmp_path: Path) -> LocalStagingRuntimeConfig:
    return LocalStagingRuntimeConfig(
        source_config_dir=_repo_config_dir(),
        source_env_file=_repo_root() / ".env",
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
