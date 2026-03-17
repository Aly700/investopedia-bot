"""Daily signal and order report helpers for manual workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any, Sequence

from bot.execution.interface import ExecutionBatch, ExecutionOrder
from bot.risk.portfolio_rules import RiskAssessedCandidate


DAILY_REPORT_COLUMNS = (
    "date",
    "signal_date",
    "symbol",
    "status",
    "action",
    "quantity",
    "intended_order_type",
    "entry_price_hint",
    "stop_level",
    "strategy_name",
    "rationale",
    "rejection_reasons",
    "metadata_json",
)


@dataclass(frozen=True)
class DailySignalReportRow:
    """A flattened daily signal decision used for CSV and JSON reporting."""

    date: date
    signal_date: date
    symbol: str
    status: str
    action: str
    quantity: int
    intended_order_type: str | None
    entry_price_hint: float | None
    stop_level: float | None
    strategy_name: str | None
    rationale: str
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the report row."""

        return {
            "date": self.date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "symbol": self.symbol,
            "status": self.status,
            "action": self.action,
            "quantity": self.quantity,
            "intended_order_type": self.intended_order_type,
            "entry_price_hint": self.entry_price_hint,
            "stop_level": self.stop_level,
            "strategy_name": self.strategy_name,
            "rationale": self.rationale,
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly representation of the report row."""

        return {
            "date": self.date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "symbol": self.symbol,
            "status": self.status,
            "action": self.action,
            "quantity": str(self.quantity),
            "intended_order_type": self.intended_order_type or "",
            "entry_price_hint": _format_optional_float(self.entry_price_hint),
            "stop_level": _format_optional_float(self.stop_level),
            "strategy_name": self.strategy_name or "",
            "rationale": self.rationale,
            "rejection_reasons": " | ".join(self.rejection_reasons),
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(frozen=True)
class DailySignalReport:
    """Summary and row-level signal report for a manual execution cycle."""

    as_of_date: date
    generated_at_utc: str
    executor_name: str
    universe_symbols: tuple[str, ...]
    no_signal_symbols: tuple[str, ...]
    benchmark_symbol: str | None
    rows: tuple[DailySignalReportRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report payload."""

        approved_count = sum(row.status == "approved" for row in self.rows)
        rejected_count = sum(row.status == "rejected" for row in self.rows)
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "executor_name": self.executor_name,
            "benchmark_symbol": self.benchmark_symbol,
            "universe_count": len(self.universe_symbols),
            "signal_count": len(self.rows),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "no_signal_count": len(self.no_signal_symbols),
            "universe_symbols": list(self.universe_symbols),
            "no_signal_symbols": list(self.no_signal_symbols),
            "rows": [row.to_dict() for row in self.rows],
        }


def build_daily_signal_report(
    *,
    as_of_date: date,
    execution_batch: ExecutionBatch,
    assessed_candidates: Sequence[RiskAssessedCandidate],
    universe_symbols: Sequence[str],
    no_signal_symbols: Sequence[str] = (),
    benchmark_symbol: str | None = None,
) -> DailySignalReport:
    """Build a report covering approved and rejected signals for one daily run."""

    orders_by_symbol = {
        order.symbol: order
        for order in execution_batch.orders
    }
    rows = tuple(
        _row_from_candidate(
            candidate,
            as_of_date=as_of_date,
            order=orders_by_symbol.get(candidate.signal.symbol),
        )
        for candidate in assessed_candidates
    )
    return DailySignalReport(
        as_of_date=as_of_date,
        generated_at_utc=execution_batch.generated_at_utc,
        executor_name=execution_batch.executor_name,
        universe_symbols=tuple(universe_symbols),
        no_signal_symbols=tuple(no_signal_symbols),
        benchmark_symbol=benchmark_symbol,
        rows=rows,
    )


def write_daily_signal_report(
    report: DailySignalReport,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write a daily signal report to CSV or JSON."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(resolved_output_path, output_format)

    if resolved_format == "json":
        resolved_output_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return resolved_output_path

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_REPORT_COLUMNS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.to_record())
    return resolved_output_path


def _row_from_candidate(
    candidate: RiskAssessedCandidate,
    *,
    as_of_date: date,
    order: ExecutionOrder | None,
) -> DailySignalReportRow:
    status = "approved" if candidate.approved else "rejected"
    metadata = {
        "adjusted_risk_per_trade": candidate.adjusted_risk_per_trade,
        "risk_budget": candidate.sizing.risk_budget,
        "per_share_risk": candidate.sizing.per_share_risk,
        "notional_value": candidate.sizing.notional_value,
        "signal_metadata": dict(candidate.signal.metadata),
    }
    if order is not None:
        metadata["execution_metadata"] = dict(order.metadata)

    rationale = order.rationale if order is not None else candidate.signal.entry_reason.replace("_", " ")
    return DailySignalReportRow(
        date=as_of_date,
        signal_date=candidate.signal.date,
        symbol=candidate.signal.symbol,
        status=status,
        action=candidate.signal.side,
        quantity=order.quantity if order is not None else 0,
        intended_order_type=order.intended_order_type if order is not None else None,
        entry_price_hint=candidate.entry_price,
        stop_level=candidate.stop_price,
        strategy_name=candidate.signal.strategy_name,
        rationale=rationale,
        rejection_reasons=candidate.rejection_reasons,
        metadata=metadata,
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _resolve_output_format(output_path: Path, output_format: str | None) -> str:
    normalized = (
        output_format.strip().lower()
        if output_format is not None
        else output_path.suffix.lower().lstrip(".") or "json"
    )
    if normalized not in {"csv", "json"}:
        raise ValueError("output_format must be either 'csv' or 'json'.")
    return normalized
