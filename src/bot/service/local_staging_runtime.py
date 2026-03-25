"""Local staging runtime harness for the full always-on stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import threading
from typing import Any, Callable, Protocol

from bot.api.control_api import (
    OperatorControlService,
    default_operator_control_state_path,
)
from bot.api.internal_api import InternalApiQueryService, create_internal_api_server
from bot.config import AppConfig, load_app_config
from bot.dashboard.operator_dashboard import create_operator_dashboard_server
from bot.logging_utils import rotate_archive_log_handlers, setup_logging
from bot.api.control_api import create_operator_control_server


LOGGER = logging.getLogger(__name__)


class RuntimeSupervisor(Protocol):
    """Minimal lifecycle interface required by the local staging harness."""

    def run(self, *, max_iterations: int | None = None) -> object:
        """Run the long-lived supervisor until completion or stop."""

    def stop(self) -> None:
        """Request shutdown."""


@dataclass(frozen=True)
class LocalStagingRuntimeConfig:
    """Filesystem and bind configuration for one local staging runtime."""

    source_config_dir: Path
    runtime_root: Path
    source_env_file: Path | None = None
    hot_state_dir: Path | None = None
    logs_dir: Path | None = None
    archive_dir: Path | None = None
    internal_api_host: str = "127.0.0.1"
    internal_api_port: int = 8765
    control_api_host: str = "127.0.0.1"
    control_api_port: int = 8766
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8780
    dashboard_refresh_seconds: int = 5
    active_log_filename: str = "local_staging_runtime.log"
    log_rotate_max_bytes: int = 250_000
    log_rotate_backup_count: int = 5

    def __post_init__(self) -> None:
        if not self.internal_api_host.strip():
            raise ValueError("internal_api_host cannot be empty.")
        if not self.control_api_host.strip():
            raise ValueError("control_api_host cannot be empty.")
        if not self.dashboard_host.strip():
            raise ValueError("dashboard_host cannot be empty.")
        if self.internal_api_port < 0:
            raise ValueError("internal_api_port must be greater than or equal to zero.")
        if self.control_api_port < 0:
            raise ValueError("control_api_port must be greater than or equal to zero.")
        if self.dashboard_port < 0:
            raise ValueError("dashboard_port must be greater than or equal to zero.")
        if self.dashboard_refresh_seconds <= 0:
            raise ValueError("dashboard_refresh_seconds must be greater than zero.")
        if not self.active_log_filename.strip():
            raise ValueError("active_log_filename cannot be empty.")
        if self.log_rotate_max_bytes <= 0:
            raise ValueError("log_rotate_max_bytes must be greater than zero.")
        if self.log_rotate_backup_count <= 0:
            raise ValueError("log_rotate_backup_count must be greater than zero.")


@dataclass(frozen=True)
class LocalStagingRuntimePaths:
    """Resolved filesystem layout for one prepared local staging runtime."""

    runtime_root: Path
    config_dir: Path
    env_file: Path
    hot_state_dir: Path
    hot_state_mount_path: Path
    logs_dir: Path
    archive_dir: Path
    archived_logs_dir: Path
    active_log_path: Path

    @property
    def operator_control_state_path(self) -> Path:
        return default_operator_control_state_path(self.runtime_root)


@dataclass
class LocalStagingRuntime:
    """Manage the lifecycle of a prepared local staging runtime stack."""

    runtime_config: LocalStagingRuntimeConfig
    paths: LocalStagingRuntimePaths
    app_config: AppConfig
    query_service: InternalApiQueryService
    control_service: OperatorControlService
    supervisor: RuntimeSupervisor
    internal_api_server: ThreadingHTTPServer
    control_api_server: ThreadingHTTPServer
    dashboard_server: ThreadingHTTPServer
    rebuild_factory: Callable[[], "LocalStagingRuntime"] | None = None
    _server_threads: list[threading.Thread] = field(default_factory=list, init=False, repr=False)
    _supervisor_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _supervisor_exception: BaseException | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    last_supervisor_result: object | None = field(default=None, init=False)

    @property
    def internal_api_url(self) -> str:
        return _server_url(self.internal_api_server)

    @property
    def control_api_url(self) -> str:
        return _server_url(self.control_api_server)

    @property
    def dashboard_url(self) -> str:
        return _server_url(self.dashboard_server) + "/dashboard"

    def configure_logging(
        self,
        *,
        level: str = "INFO",
        json_logs: bool = False,
    ) -> None:
        setup_logging(
            level=level,
            json_logs=json_logs,
            log_file=self.paths.active_log_path,
            archive_dir=self.paths.archived_logs_dir,
            rotate_max_bytes=self.runtime_config.log_rotate_max_bytes,
            rotate_backup_count=self.runtime_config.log_rotate_backup_count,
        )

    def start(self, *, max_iterations: int | None = None) -> None:
        if self._started:
            raise RuntimeError("Local staging runtime is already running.")
        self._started = True
        self._server_threads = []
        self._supervisor_exception = None
        for name, server in (
            ("internal-api", self.internal_api_server),
            ("control-api", self.control_api_server),
            ("dashboard", self.dashboard_server),
        ):
            thread = threading.Thread(
                target=self._serve_server,
                args=(name, server),
                daemon=True,
            )
            thread.start()
            self._server_threads.append(thread)
        self._supervisor_thread = threading.Thread(
            target=self._run_supervisor,
            kwargs={"max_iterations": max_iterations},
            daemon=True,
        )
        self._supervisor_thread.start()

    def wait_for_supervisor(self, timeout: float | None = None) -> bool:
        if self._supervisor_thread is None:
            return True
        self._supervisor_thread.join(timeout=timeout)
        if self._supervisor_thread.is_alive():
            return False
        if self._supervisor_exception is not None:
            raise RuntimeError(
                "Local staging supervisor exited with an exception."
            ) from self._supervisor_exception
        return True

    def stop(self) -> None:
        if not self._started:
            return
        if hasattr(self.supervisor, "stop"):
            self.supervisor.stop()
        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=5.0)
        for server in (
            self.internal_api_server,
            self.control_api_server,
            self.dashboard_server,
        ):
            try:
                server.shutdown()
            finally:
                server.server_close()
        for thread in self._server_threads:
            thread.join(timeout=5.0)
        self._server_threads = []
        self._supervisor_thread = None
        self._started = False

    def restart(self, *, max_iterations: int | None = None) -> "LocalStagingRuntime":
        if self.rebuild_factory is None:
            raise RuntimeError("Local staging runtime does not support restart.")
        self.stop()
        restarted = self.rebuild_factory()
        restarted.start(max_iterations=max_iterations)
        return restarted

    def rotate_logs(self) -> tuple[Path, ...]:
        return rotate_archive_log_handlers()

    def corrupt_safety_state(self, payload: str = "{malformed-json") -> Path:
        path = self.paths.operator_control_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path

    def delete_safety_state(self) -> None:
        path = self.paths.operator_control_state_path
        if path.exists() or path.is_symlink():
            path.unlink()

    def write_runtime_env(self, content: str) -> Path:
        self.paths.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.paths.env_file.write_text(content, encoding="utf-8")
        return self.paths.env_file

    def disconnect_transport(self) -> None:
        adapter = getattr(self.supervisor, "adapter", None)
        if adapter is None or not hasattr(adapter, "disconnect"):
            raise RuntimeError("Supervisor adapter does not support disconnect().")
        adapter.disconnect()

    def reconnect_transport(self) -> None:
        adapter = getattr(self.supervisor, "adapter", None)
        if adapter is None or not hasattr(adapter, "connect"):
            raise RuntimeError("Supervisor adapter does not support connect().")
        adapter.connect()

    def _serve_server(self, name: str, server: ThreadingHTTPServer) -> None:
        try:
            server.serve_forever()
        except Exception:
            LOGGER.exception("Local staging %s server exited unexpectedly.", name)
            raise

    def _run_supervisor(self, *, max_iterations: int | None) -> None:
        try:
            self.last_supervisor_result = self.supervisor.run(max_iterations=max_iterations)
        except BaseException as exc:
            self._supervisor_exception = exc


def prepare_local_staging_runtime(
    runtime_config: LocalStagingRuntimeConfig,
) -> LocalStagingRuntimePaths:
    """Prepare and validate the isolated runtime root for local staging."""

    source_config_dir = runtime_config.source_config_dir.resolve()
    source_project_root = source_config_dir.parent.resolve()
    runtime_root = runtime_config.runtime_root.resolve()
    if not source_config_dir.is_dir():
        raise ValueError(f"source_config_dir does not exist: {source_config_dir}")
    if runtime_root == source_project_root:
        raise ValueError("runtime_root must be different from the source project root.")
    if source_config_dir == runtime_root or source_config_dir in runtime_root.parents:
        raise ValueError("runtime_root cannot be inside the source config directory.")
    source_env_file = (
        runtime_config.source_env_file.resolve()
        if runtime_config.source_env_file is not None
        else None
    )
    if source_env_file is not None and source_env_file.exists() and not source_env_file.is_file():
        raise ValueError(f"source_env_file must be a file when provided: {source_env_file}")

    hot_state_dir = (
        runtime_config.hot_state_dir.resolve()
        if runtime_config.hot_state_dir is not None
        else runtime_root / "data" / "processed" / "state"
    )
    logs_dir = (
        runtime_config.logs_dir.resolve()
        if runtime_config.logs_dir is not None
        else runtime_root / "logs"
    )
    archive_dir = (
        runtime_config.archive_dir.resolve()
        if runtime_config.archive_dir is not None
        else runtime_root / "archive"
    )
    _validate_distinct_runtime_paths(
        hot_state_dir=hot_state_dir,
        logs_dir=logs_dir,
        archive_dir=archive_dir,
    )

    runtime_root.mkdir(parents=True, exist_ok=True)
    config_dir = runtime_root / "config"
    shutil.copytree(source_config_dir, config_dir, dirs_exist_ok=True)

    env_file = runtime_root / ".env"
    if source_env_file is not None and source_env_file.exists():
        env_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_env_file, env_file)

    hot_state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_logs_dir = archive_dir / "logs"
    archived_logs_dir.mkdir(parents=True, exist_ok=True)

    hot_state_mount_path = runtime_root / "data" / "processed" / "state"
    _mount_hot_state_dir(
        hot_state_mount_path=hot_state_mount_path,
        hot_state_dir=hot_state_dir,
    )
    active_log_path = logs_dir / runtime_config.active_log_filename
    return LocalStagingRuntimePaths(
        runtime_root=runtime_root,
        config_dir=config_dir.resolve(),
        env_file=env_file.resolve(),
        hot_state_dir=hot_state_dir,
        hot_state_mount_path=hot_state_mount_path,
        logs_dir=logs_dir,
        archive_dir=archive_dir,
        archived_logs_dir=archived_logs_dir,
        active_log_path=active_log_path.resolve(),
    )


def create_local_staging_runtime(
    runtime_config: LocalStagingRuntimeConfig,
    *,
    supervisor_factory: Callable[[AppConfig, LocalStagingRuntimePaths], RuntimeSupervisor],
    query_service_factory: Callable[
        [AppConfig, LocalStagingRuntimePaths], InternalApiQueryService
    ]
    | None = None,
    control_service_factory: Callable[
        [AppConfig, LocalStagingRuntimePaths], OperatorControlService
    ]
    | None = None,
    internal_api_server_factory: Callable[..., ThreadingHTTPServer] | None = None,
    control_api_server_factory: Callable[..., ThreadingHTTPServer] | None = None,
    dashboard_server_factory: Callable[..., ThreadingHTTPServer] | None = None,
) -> LocalStagingRuntime:
    """Build a fully wired local staging runtime around one runtime root."""

    paths = prepare_local_staging_runtime(runtime_config)
    app_config = load_app_config(config_dir=paths.config_dir)
    query_service = (
        query_service_factory(app_config, paths)
        if query_service_factory is not None
        else InternalApiQueryService(project_root=app_config.project_root, config=app_config)
    )
    control_service = (
        control_service_factory(app_config, paths)
        if control_service_factory is not None
        else OperatorControlService(project_root=app_config.project_root, env_file=paths.env_file)
    )
    resolved_internal_api_server_factory = (
        internal_api_server_factory or create_internal_api_server
    )
    resolved_control_api_server_factory = (
        control_api_server_factory or create_operator_control_server
    )
    resolved_dashboard_server_factory = (
        dashboard_server_factory or create_operator_dashboard_server
    )
    internal_api_server = resolved_internal_api_server_factory(
        query_service=query_service,
        host=runtime_config.internal_api_host,
        port=runtime_config.internal_api_port,
    )
    control_api_server = resolved_control_api_server_factory(
        control_service=control_service,
        host=runtime_config.control_api_host,
        port=runtime_config.control_api_port,
    )
    dashboard_server = resolved_dashboard_server_factory(
        query_service=query_service,
        control_service=control_service,
        host=runtime_config.dashboard_host,
        port=runtime_config.dashboard_port,
        refresh_interval_seconds=runtime_config.dashboard_refresh_seconds,
    )

    def rebuild_runtime() -> LocalStagingRuntime:
        return create_local_staging_runtime(
            runtime_config,
            supervisor_factory=supervisor_factory,
            query_service_factory=query_service_factory,
            control_service_factory=control_service_factory,
            internal_api_server_factory=internal_api_server_factory,
            control_api_server_factory=control_api_server_factory,
            dashboard_server_factory=dashboard_server_factory,
        )

    return LocalStagingRuntime(
        runtime_config=runtime_config,
        paths=paths,
        app_config=app_config,
        query_service=query_service,
        control_service=control_service,
        supervisor=supervisor_factory(app_config, paths),
        internal_api_server=internal_api_server,
        control_api_server=control_api_server,
        dashboard_server=dashboard_server,
        rebuild_factory=rebuild_runtime,
    )


def _mount_hot_state_dir(
    *,
    hot_state_mount_path: Path,
    hot_state_dir: Path,
) -> None:
    hot_state_mount_path.parent.mkdir(parents=True, exist_ok=True)
    if hot_state_mount_path.resolve() == hot_state_dir.resolve():
        hot_state_mount_path.mkdir(parents=True, exist_ok=True)
        return
    if hot_state_mount_path.is_symlink():
        if hot_state_mount_path.resolve() == hot_state_dir.resolve():
            return
        hot_state_mount_path.unlink()
    elif hot_state_mount_path.exists():
        if hot_state_mount_path.is_dir() and not any(hot_state_mount_path.iterdir()):
            hot_state_mount_path.rmdir()
        else:
            raise ValueError(
                f"hot_state_mount_path already exists and is not empty: {hot_state_mount_path}"
            )
    hot_state_mount_path.symlink_to(hot_state_dir, target_is_directory=True)


def _validate_distinct_runtime_paths(
    *,
    hot_state_dir: Path,
    logs_dir: Path,
    archive_dir: Path,
) -> None:
    if hot_state_dir == logs_dir:
        raise ValueError("hot_state_dir and logs_dir must be different.")
    if hot_state_dir == archive_dir:
        raise ValueError("hot_state_dir and archive_dir must be different.")
    if logs_dir == archive_dir:
        raise ValueError("logs_dir and archive_dir must be different.")


def _server_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"
