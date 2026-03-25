"""Logging helpers for CLI and batch workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class JsonLogFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


class ArchiveRotatingFileHandler(RotatingFileHandler):
    """Rotate the active log file into a separate archive directory."""

    def __init__(
        self,
        log_file: Path,
        *,
        archive_dir: Path,
        max_bytes: int,
        backup_count: int = 5,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero.")
        if backup_count <= 0:
            raise ValueError("backup_count must be greater than zero.")
        resolved_log_file = log_file.resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir = archive_dir.resolve()
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            resolved_log_file,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    def doRollover(self) -> Path | None:
        if self.stream:
            self.stream.flush()
            self.stream.close()
            self.stream = None
        source_path = Path(self.baseFilename).resolve()
        archived_path: Path | None = None
        if source_path.exists():
            archived_path = self._next_archive_path(source_path)
            source_path.replace(archived_path)
        self._prune_archives(source_path)
        if not self.delay:
            self.stream = self._open()
        return archived_path

    def force_rollover(self) -> Path | None:
        """Serialize a manual rollover with normal emit() calls."""

        self.acquire()
        try:
            if self.stream is not None:
                self.flush()
            return self.doRollover()
        finally:
            self.release()

    def _next_archive_path(self, source_path: Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = source_path.suffix
        stem = source_path.stem
        candidate = self.archive_dir / f"{stem}-{timestamp}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = self.archive_dir / f"{stem}-{timestamp}-{counter}{suffix}"
            counter += 1
        return candidate

    def _prune_archives(self, source_path: Path) -> None:
        suffix = source_path.suffix
        stem = source_path.stem
        archived_paths = sorted(
            self.archive_dir.glob(f"{stem}-*{suffix}"),
            key=lambda path: path.name,
        )
        while len(archived_paths) > self.backupCount:
            oldest_path = archived_paths.pop(0)
            try:
                oldest_path.unlink()
            except OSError:
                logging.getLogger(__name__).warning(
                    "Failed to prune archived log %s.",
                    oldest_path,
                    exc_info=True,
                )


def rotate_archive_log_handlers(
    *,
    logger: logging.Logger | None = None,
) -> tuple[Path, ...]:
    """Force-roll all archive-aware handlers attached to one logger."""

    active_logger = logger or logging.getLogger()
    archived_paths: list[Path] = []
    for handler in active_logger.handlers:
        if not isinstance(handler, ArchiveRotatingFileHandler):
            continue
        archived_path = handler.force_rollover()
        if archived_path is not None:
            archived_paths.append(archived_path)
    return tuple(archived_paths)


def setup_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
    json_logs: bool = False,
    archive_dir: Path | None = None,
    rotate_max_bytes: int | None = None,
    rotate_backup_count: int = 5,
) -> None:
    """Configure root logging for the current process."""

    resolved_level = getattr(logging, level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    formatter: logging.Formatter
    if json_logs:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if archive_dir is not None and rotate_max_bytes is not None:
            file_handler = ArchiveRotatingFileHandler(
                log_file,
                archive_dir=archive_dir,
                max_bytes=rotate_max_bytes,
                backup_count=rotate_backup_count,
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger using the configured root handlers."""

    return logging.getLogger(name)
