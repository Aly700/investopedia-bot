from __future__ import annotations

import logging
from pathlib import Path

from bot.logging_utils import ArchiveRotatingFileHandler


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
