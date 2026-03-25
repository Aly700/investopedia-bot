from __future__ import annotations

import logging
from pathlib import Path
import threading
import time

import pytest

from bot.logging_utils import ArchiveRotatingFileHandler, rotate_archive_log_handlers


def test_archive_rotating_file_handler_moves_logs_to_archive_and_keeps_active_file_writable(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "logs" / "runtime.log"
    archive_dir = tmp_path / "archive"
    logger = logging.getLogger("tests.archive_rotation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()

    handler = ArchiveRotatingFileHandler(
        log_file,
        archive_dir=archive_dir,
        max_bytes=80,
        backup_count=2,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    try:
        logger.info("A" * 70)
        logger.info("B" * 70)
        logger.info("after-rotation")
        handler.flush()
    finally:
        logger.removeHandler(handler)
        handler.close()

    archived_logs = sorted(archive_dir.glob("runtime-*.log"))
    assert archived_logs
    assert len(archived_logs) <= 2
    archived_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in archived_logs
    )
    assert "A" * 70 in archived_text
    active_log_text = log_file.read_text(encoding="utf-8")
    assert "after-rotation" in active_log_text


def test_rotate_archive_log_handlers_is_safe_under_concurrent_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_file = tmp_path / "logs" / "runtime.log"
    archive_dir = tmp_path / "archive"
    logger = logging.getLogger("tests.archive_rotation.concurrent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()

    errors: list[str] = []

    def record_handle_error(self: logging.Handler, record: logging.LogRecord) -> None:
        errors.append(record.getMessage())

    monkeypatch.setattr(logging.Handler, "handleError", record_handle_error)

    handler = ArchiveRotatingFileHandler(
        log_file,
        archive_dir=archive_dir,
        max_bytes=512,
        backup_count=3,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    stop_event = threading.Event()

    def background_logger() -> None:
        counter = 0
        while not stop_event.is_set():
            logger.info("background-log-%s-%s", counter, "x" * 80)
            counter += 1

    worker = threading.Thread(target=background_logger, daemon=True)
    worker.start()

    try:
        time.sleep(0.05)
        for _ in range(15):
            rotate_archive_log_handlers(logger=logger)
        logger.info("post-rotation-sentinel")
        stop_event.set()
        worker.join(timeout=2.0)
        handler.flush()
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        logger.removeHandler(handler)
        handler.close()

    assert errors == []
    assert "post-rotation-sentinel" in log_file.read_text(encoding="utf-8")
