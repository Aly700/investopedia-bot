"""Trade log export helpers for backtests and offline research runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_trade_log_report(
    trade_log: pd.DataFrame,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write a trade log report to CSV or JSON."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(resolved_output_path, output_format)
    prepared = trade_log.copy()

    if resolved_format == "csv":
        prepared.to_csv(resolved_output_path, index=False, date_format="%Y-%m-%d")
    else:
        resolved_output_path.write_text(
            json.dumps(_frame_records(prepared), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return resolved_output_path


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        serialized: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                serialized[key] = value.date().isoformat()
            else:
                serialized[key] = value
        records.append(serialized)
    return records


def _resolve_output_format(output_path: Path, output_format: str | None) -> str:
    normalized = (
        output_format.strip().lower()
        if output_format is not None
        else output_path.suffix.lower().lstrip(".") or "csv"
    )
    if normalized not in {"csv", "json"}:
        raise ValueError("output_format must be either 'csv' or 'json'.")
    return normalized
