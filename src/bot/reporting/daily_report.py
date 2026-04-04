"""Daily signal, order, and research summary report helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.execution.interface import ExecutionBatch, ExecutionOrder
from bot.execution.prioritization import (
    execution_priority_note,
    risk_candidate_sort_key,
    score_risk_candidate_opportunity as _score_risk_candidate_opportunity,
)
from bot.features import MarketContext
from bot.risk.portfolio_rules import (
    PORTFOLIO_REVIEW_ACTIONS,
    ExistingPosition,
    RiskAssessedCandidate,
)
from bot.strategy.breakout_momentum import BreakoutStrategyPreset, build_breakout_rationale


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
DAILY_RESEARCH_COLUMNS = (
    "rank",
    "date",
    "signal_date",
    "preset_name",
    "parameter_id",
    "symbol",
    "status",
    "action",
    "quantity",
    "intended_order_type",
    "entry_price_hint",
    "stop_level",
    "strategy_name",
    "rationale",
    "score",
    "breakout_pct",
    "relative_volume_ratio",
    "position_pct_equity",
    "adjusted_risk_per_trade",
    "risk_budget",
    "per_share_risk",
    "notional_value",
    "rejection_reasons",
    "metadata_json",
)
DAILY_PRESET_SUMMARY_COLUMNS = (
    "preset_rank",
    "preset_name",
    "parameter_id",
    "candidate_count",
    "approved_count",
    "rejected_count",
    "no_signal_count",
    "order_count",
    "average_score",
    "top_score",
    "top_symbol",
    "approved_symbols",
    "rejected_symbols",
)
PORTFOLIO_REVIEW_COLUMNS = (
    "date",
    "symbol",
    "quantity",
    "average_entry_price",
    "current_stop",
    "suggested_stop",
    "latest_close",
    "unrealized_pl_pct",
    "distance_to_stop_pct",
    "regime_passed",
    "above_entry",
    "suggested_action",
    "preset_name",
    "rationale",
    "metadata_json",
)
MARKET_MONITOR_COLUMNS = (
    "date",
    "symbol",
    "category",
    "preset_name",
    "latest_close",
    "average_entry_price",
    "current_stop",
    "suggested_stop",
    "entry_price_hint",
    "stop_level",
    "rationale",
    "metadata_json",
)
MARKET_MONITOR_CATEGORIES = (
    "EXIT CANDIDATE",
    "RAISE STOP",
    "BUY CANDIDATE",
    "WATCH CLOSELY",
    "HOLD",
    "NO ACTION",
)
MARKET_MONITOR_CATEGORY_PRIORITY = {
    category: priority for priority, category in enumerate(MARKET_MONITOR_CATEGORIES)
}
MARKET_MONITOR_BRIEF_TOP_BUY_LIMIT = 5
MARKET_MONITOR_BRIEF_LOWER_PRIORITY_LIMIT = 5


def market_monitor_flat_count_key(category: str) -> str:
    """Return the flat ``*_count`` field name for a market-monitor category."""

    return f"{category.lower().replace(' ', '_')}_count"


def ensure_market_monitor_categories_cover_portfolio_actions(
    market_monitor_categories: Sequence[str],
    portfolio_review_actions: Sequence[str],
) -> None:
    """Raise when portfolio-review actions are not representable as monitor categories."""

    missing_categories = sorted(set(portfolio_review_actions) - set(market_monitor_categories))
    if missing_categories:
        raise RuntimeError(
            "Portfolio review actions must also be valid market monitor categories. "
            f"Missing categories: {missing_categories}"
        )


ensure_market_monitor_categories_cover_portfolio_actions(
    MARKET_MONITOR_CATEGORIES,
    PORTFOLIO_REVIEW_ACTIONS,
)


@dataclass(frozen=True)
class PresetCandidateEvaluation:
    """A risk-assessed candidate paired with its preset identity."""

    preset_name: str
    parameter_id: str
    candidate: RiskAssessedCandidate

    def __post_init__(self) -> None:
        if not self.preset_name.strip():
            raise ValueError("preset_name cannot be empty.")
        if not self.parameter_id.strip():
            raise ValueError("parameter_id cannot be empty.")


@dataclass(frozen=True)
class DailyResearchOpportunityRow:
    """A ranked daily opportunity row for preset review and manual execution."""

    rank: int
    date: date
    signal_date: date
    preset_name: str
    parameter_id: str
    symbol: str
    status: str
    action: str
    quantity: int
    intended_order_type: str | None
    entry_price_hint: float | None
    stop_level: float | None
    strategy_name: str | None
    rationale: str
    score: float
    breakout_pct: float
    relative_volume_ratio: float
    position_pct_equity: float
    adjusted_risk_per_trade: float
    risk_budget: float
    per_share_risk: float
    notional_value: float
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the opportunity row."""

        return {
            "rank": self.rank,
            "date": self.date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "preset_name": self.preset_name,
            "parameter_id": self.parameter_id,
            "symbol": self.symbol,
            "status": self.status,
            "action": self.action,
            "quantity": self.quantity,
            "intended_order_type": self.intended_order_type,
            "entry_price_hint": self.entry_price_hint,
            "stop_level": self.stop_level,
            "strategy_name": self.strategy_name,
            "rationale": self.rationale,
            "score": self.score,
            "breakout_pct": self.breakout_pct,
            "relative_volume_ratio": self.relative_volume_ratio,
            "position_pct_equity": self.position_pct_equity,
            "adjusted_risk_per_trade": self.adjusted_risk_per_trade,
            "risk_budget": self.risk_budget,
            "per_share_risk": self.per_share_risk,
            "notional_value": self.notional_value,
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly representation of the opportunity row."""

        return {
            "rank": str(self.rank),
            "date": self.date.isoformat(),
            "signal_date": self.signal_date.isoformat(),
            "preset_name": self.preset_name,
            "parameter_id": self.parameter_id,
            "symbol": self.symbol,
            "status": self.status,
            "action": self.action,
            "quantity": str(self.quantity),
            "intended_order_type": self.intended_order_type or "",
            "entry_price_hint": _format_optional_float(self.entry_price_hint),
            "stop_level": _format_optional_float(self.stop_level),
            "strategy_name": self.strategy_name or "",
            "rationale": self.rationale,
            "score": _format_optional_float(self.score),
            "breakout_pct": _format_optional_float(self.breakout_pct),
            "relative_volume_ratio": _format_optional_float(self.relative_volume_ratio),
            "position_pct_equity": _format_optional_float(self.position_pct_equity),
            "adjusted_risk_per_trade": _format_optional_float(self.adjusted_risk_per_trade),
            "risk_budget": _format_optional_float(self.risk_budget),
            "per_share_risk": _format_optional_float(self.per_share_risk),
            "notional_value": _format_optional_float(self.notional_value),
            "rejection_reasons": " | ".join(self.rejection_reasons),
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(frozen=True)
class DailyPresetSummaryRow:
    """Aggregate summary for one evaluated preset on the current day."""

    preset_rank: int
    preset_name: str
    parameter_id: str
    candidate_count: int
    approved_count: int
    rejected_count: int
    no_signal_count: int
    order_count: int
    average_score: float
    top_score: float
    top_symbol: str | None
    approved_symbols: tuple[str, ...] = ()
    rejected_symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the preset summary."""

        return {
            "preset_rank": self.preset_rank,
            "preset_name": self.preset_name,
            "parameter_id": self.parameter_id,
            "candidate_count": self.candidate_count,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "no_signal_count": self.no_signal_count,
            "order_count": self.order_count,
            "average_score": self.average_score,
            "top_score": self.top_score,
            "top_symbol": self.top_symbol,
            "approved_symbols": list(self.approved_symbols),
            "rejected_symbols": list(self.rejected_symbols),
        }

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly representation of the preset summary."""

        return {
            "preset_rank": str(self.preset_rank),
            "preset_name": self.preset_name,
            "parameter_id": self.parameter_id,
            "candidate_count": str(self.candidate_count),
            "approved_count": str(self.approved_count),
            "rejected_count": str(self.rejected_count),
            "no_signal_count": str(self.no_signal_count),
            "order_count": str(self.order_count),
            "average_score": _format_optional_float(self.average_score),
            "top_score": _format_optional_float(self.top_score),
            "top_symbol": self.top_symbol or "",
            "approved_symbols": " | ".join(self.approved_symbols),
            "rejected_symbols": " | ".join(self.rejected_symbols),
        }


@dataclass(frozen=True)
class DailyResearchSummary:
    """Consolidated daily summary for preset selection and trade review."""

    as_of_date: date
    generated_at_utc: str
    executor_name: str
    selected_presets: tuple[BreakoutStrategyPreset, ...]
    preset_selection_source: str | None
    benchmark_symbol: str | None
    universe_symbols: tuple[str, ...]
    current_positions: tuple[ExistingPosition, ...]
    no_signal_symbols_by_preset: dict[str, tuple[str, ...]]
    preset_summaries: tuple[DailyPresetSummaryRow, ...]
    rows: tuple[DailyResearchOpportunityRow, ...]
    execution_batch: ExecutionBatch
    market_context: MarketContext | None = None
    force_require_relative_volume_confirmation: bool = False
    pending_order_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recommended_preset(self) -> str | None:
        """Return today's top-ranked preset, if any."""

        if not self.preset_summaries:
            return None
        top_summary = self.preset_summaries[0]
        if top_summary.candidate_count == 0:
            return None
        return top_summary.preset_name

    @property
    def relative_volume_policy(self) -> str:
        """Return the effective RV policy across the selected presets."""

        if self.force_require_relative_volume_confirmation:
            return "required"

        selected_policies = {
            bool(preset.require_relative_volume_confirmation)
            for preset in self.selected_presets
        }
        if not selected_policies or selected_policies == {False}:
            return "optional"
        if selected_policies == {True}:
            return "required"
        return "mixed"

    @property
    def relative_volume_confirmation_required(self) -> bool | None:
        """Return the effective RV hard-gate status when it is not mixed."""

        policy = self.relative_volume_policy
        if policy == "mixed":
            return None
        return policy == "required"

    @property
    def relative_volume_policy_by_preset(self) -> dict[str, str]:
        """Return RV policy labels for each selected preset."""

        policy = "required" if self.force_require_relative_volume_confirmation else None
        return {
            preset.name: (
                policy
                if policy is not None
                else (
                    "required"
                    if preset.require_relative_volume_confirmation
                    else "optional"
                )
            )
            for preset in self.selected_presets
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary payload."""

        approved_rows = [row for row in self.rows if row.status == "approved"]
        rejected_rows = [row for row in self.rows if row.status == "rejected"]
        unique_no_signal_symbols = sorted(
            {
                symbol
                for symbols in self.no_signal_symbols_by_preset.values()
                for symbol in symbols
            }
        )
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "executor_name": self.executor_name,
            "benchmark_symbol": self.benchmark_symbol,
            "preset_selection_source": self.preset_selection_source,
            "selected_presets": [preset.to_dict() for preset in self.selected_presets],
            "market_context": (
                self.market_context.to_dict()
                if self.market_context is not None
                else None
            ),
            "relative_volume_confirmation_required": self.relative_volume_confirmation_required,
            "relative_volume_policy": self.relative_volume_policy,
            "relative_volume_policy_by_preset": self.relative_volume_policy_by_preset,
            "recommended_preset": self.recommended_preset,
            "universe_count": len(self.universe_symbols),
            "current_position_count": len(self.current_positions),
            "current_position_symbols": [position.symbol for position in self.current_positions],
            "current_positions": [position.to_dict() for position in self.current_positions],
            "pending_order_summary": (
                dict(self.pending_order_summary)
                if self.pending_order_summary is not None
                else None
            ),
            "metadata": dict(self.metadata),
            "candidate_count": len(self.rows),
            "approved_count": len(approved_rows),
            "rejected_count": len(rejected_rows),
            "order_count": len(self.execution_batch.orders),
            "unique_no_signal_symbol_count": len(unique_no_signal_symbols),
            "total_preset_no_signal_count": sum(
                len(symbols) for symbols in self.no_signal_symbols_by_preset.values()
            ),
            "universe_symbols": list(self.universe_symbols),
            "unique_no_signal_symbols": unique_no_signal_symbols,
            "no_signal_symbols_by_preset": {
                preset_name: list(symbols)
                for preset_name, symbols in self.no_signal_symbols_by_preset.items()
            },
            "preset_summaries": [summary.to_dict() for summary in self.preset_summaries],
            "approved_candidates": [row.to_dict() for row in approved_rows],
            "rejected_candidates": [row.to_dict() for row in rejected_rows],
            "ranked_opportunities": [row.to_dict() for row in self.rows],
            "suggested_order_sheet": self.execution_batch.to_dict(),
        }

    def to_brief(self, *, output_paths: Mapping[str, str] | None = None) -> str:
        """Return a human-readable trader brief for the daily summary."""

        approved_rows = [row for row in self.rows if row.status == "approved"]
        actionable_approved_rows = [row for row in approved_rows if _row_actionable_now(row)]
        deferred_approved_rows = [row for row in approved_rows if not _row_actionable_now(row)]
        rejected_rows = [row for row in self.rows if row.status == "rejected"]
        current_symbols = [position.symbol for position in self.current_positions]

        lines = [f"Daily summary brief for {self.as_of_date.isoformat()}", "", "Headline"]
        lines.append(
            "Recommended preset: "
            f"{self.recommended_preset or 'none'} | "
            f"Universe: {len(self.universe_symbols)} | "
            f"Approved: {len(approved_rows)} | "
            f"Rejected: {len(rejected_rows)} | "
            f"Orders: {len(self.execution_batch.orders)}"
        )
        lines.append(
            f"Relative-volume policy: {self.relative_volume_policy} | "
            f"Preset selection: {self.preset_selection_source or 'n/a'}"
        )
        if self.pending_order_summary is not None:
            pending_fill_notional = float(
                self.pending_order_summary.get("pending_fill_notional", 0.0) or 0.0
            )
            pending_line = (
                "Pending orders: "
                f"{self.pending_order_summary.get('active_order_count', 0)} | "
                f"Reserved capital: {self.pending_order_summary.get('reserved_notional', 0.0):.2f}"
            )
            if pending_fill_notional > 0:
                pending_line += (
                    f" | Pending fills awaiting holdings: {pending_fill_notional:.2f}"
                )
            pending_line += (
                f" | Available capital: {self.pending_order_summary.get('available_capital', 0.0):.2f}"
            )
            lines.append(pending_line)
        breadth_line = _market_breadth_brief_line(self.market_context)
        if breadth_line is not None:
            lines.append(breadth_line)
        volatility_line = _volatility_regime_brief_line(self.market_context)
        if volatility_line is not None:
            lines.append(volatility_line)

        lines.extend(("", "Top opportunities"))
        if actionable_approved_rows:
            for row in actionable_approved_rows[:5]:
                lines.extend(_daily_research_brief_card(row))
        elif approved_rows:
            lines.append(
                "No approved buy candidates are currently actionable under portfolio capacity."
            )
        else:
            lines.append("No approved buy candidates today.")

        lines.extend(("", "Lower-priority / rejected candidates"))
        if deferred_approved_rows or rejected_rows:
            for row in deferred_approved_rows[:3]:
                lines.append(
                    f"- {row.symbol} | preset={row.preset_name} | "
                    f"note={_single_line(row.rationale)}"
                )
            for row in rejected_rows[:5]:
                reason = row.rejection_reasons[0] if row.rejection_reasons else row.rationale
                lines.append(
                    f"- {row.symbol} | preset={row.preset_name} | "
                    f"reason={_single_line(reason)}"
                )
        else:
            lines.append("No lower-priority rejected candidates today.")

        lines.extend(("", "Portfolio context"))
        lines.append(
            f"Current holdings: {len(current_symbols)}"
            + (f" | symbols={', '.join(current_symbols)}" if current_symbols else "")
        )
        if self.preset_summaries:
            top_preset = self.preset_summaries[0]
            lines.append(
                f"Top preset today: {top_preset.preset_name} | "
                f"approved={top_preset.approved_count} | "
                f"top_symbol={top_preset.top_symbol or 'n/a'}"
            )
        else:
            lines.append("Top preset today: none")

        if output_paths:
            lines.extend(("", "Output files"))
            for name, path in sorted(output_paths.items()):
                lines.append(f"- {name}: {path}")

        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class PortfolioReviewRow:
    """One portfolio-management review row for an open holding."""

    date: date
    symbol: str
    quantity: int
    average_entry_price: float
    current_stop: float | None
    suggested_stop: float | None
    latest_close: float
    unrealized_pl_pct: float
    distance_to_stop_pct: float | None
    regime_passed: bool | None
    above_entry: bool
    suggested_action: str
    preset_name: str | None
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.suggested_action not in PORTFOLIO_REVIEW_ACTIONS:
            raise ValueError(
                f"suggested_action must be one of {PORTFOLIO_REVIEW_ACTIONS}, "
                f"got '{self.suggested_action}'."
            )
        if (
            self.suggested_action == "RAISE STOP"
            and self.suggested_stop is not None
            and self.suggested_stop >= self.latest_close
        ):
            raise ValueError(
                "RAISE STOP rows require suggested_stop to remain below latest_close."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the review row."""

        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_stop": self.current_stop,
            "suggested_stop": self.suggested_stop,
            "latest_close": self.latest_close,
            "unrealized_pl_pct": self.unrealized_pl_pct,
            "distance_to_stop_pct": self.distance_to_stop_pct,
            "regime_passed": self.regime_passed,
            "above_entry": self.above_entry,
            "suggested_action": self.suggested_action,
            "preset_name": self.preset_name,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly representation of the review row."""

        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "average_entry_price": _format_optional_float(self.average_entry_price),
            "current_stop": _format_optional_float(self.current_stop),
            "suggested_stop": _format_optional_float(self.suggested_stop),
            "latest_close": _format_optional_float(self.latest_close),
            "unrealized_pl_pct": _format_optional_float(self.unrealized_pl_pct),
            "distance_to_stop_pct": _format_optional_float(self.distance_to_stop_pct),
            "regime_passed": _format_optional_bool(self.regime_passed),
            "above_entry": _format_optional_bool(self.above_entry),
            "suggested_action": self.suggested_action,
            "preset_name": self.preset_name or "",
            "rationale": self.rationale,
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(frozen=True)
class PortfolioReviewReport:
    """Daily portfolio-management review for currently open holdings."""

    as_of_date: date
    generated_at_utc: str
    benchmark_symbol: str | None
    current_positions: tuple[ExistingPosition, ...]
    rows: tuple[PortfolioReviewRow, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly review report payload."""

        action_counts = {
            f"{action.lower().replace(' ', '_')}_count": sum(
                row.suggested_action == action for row in self.rows
            )
            for action in PORTFOLIO_REVIEW_ACTIONS
        }
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "benchmark_symbol": self.benchmark_symbol,
            "position_count": len(self.current_positions),
            "reviewed_symbol_count": len(self.rows),
            "reviewed_symbols": [row.symbol for row in self.rows],
            "current_position_symbols": [position.symbol for position in self.current_positions],
            "current_positions": [position.to_dict() for position in self.current_positions],
            **action_counts,
            "rows": [row.to_dict() for row in self.rows],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IntradayPortfolioReviewReport:
    """Intraday held-position review driven by scheduled bar polling."""

    as_of_date: date
    generated_at_utc: str
    interval_minutes: int
    portfolio_path: str | None
    benchmark_symbol: str | None
    current_positions: tuple[ExistingPosition, ...]
    rows: tuple[PortfolioReviewRow, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly intraday review payload."""

        action_counts = {
            f"{action.lower().replace(' ', '_')}_count": sum(
                row.suggested_action == action for row in self.rows
            )
            for action in PORTFOLIO_REVIEW_ACTIONS
        }
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "interval_minutes": self.interval_minutes,
            "portfolio_path": self.portfolio_path,
            "benchmark_symbol": self.benchmark_symbol,
            "position_count": len(self.current_positions),
            "reviewed_symbol_count": len(self.rows),
            "reviewed_symbols": [row.symbol for row in self.rows],
            "current_position_symbols": [position.symbol for position in self.current_positions],
            "current_positions": [position.to_dict() for position in self.current_positions],
            **action_counts,
            "rows": [row.to_dict() for row in self.rows],
            "metadata": dict(self.metadata),
        }

    def to_brief(self) -> str:
        """Return a human-readable intraday trader brief."""

        sorted_rows = sorted(
            self.rows,
            key=lambda row: (
                MARKET_MONITOR_CATEGORY_PRIORITY.get(row.suggested_action, 99),
                row.symbol,
                row.preset_name or "",
            ),
        )
        urgent_rows = [
            row
            for row in sorted_rows
            if row.suggested_action == "EXIT CANDIDATE"
            or (
                row.suggested_action == "WATCH CLOSELY"
                and (
                    bool(row.metadata.get("failed_intraday_strength"))
                    or (
                        _optional_float(row.metadata.get("session_high_giveback_pct")) is not None
                        and _optional_float(row.metadata.get("session_high_giveback_exit_threshold")) is not None
                        and _optional_float(row.metadata.get("session_high_giveback_pct"))
                        >= _optional_float(row.metadata.get("session_high_giveback_exit_threshold"))
                    )
                )
            )
        ]
        urgent_row_ids = {id(row) for row in urgent_rows}
        pressure_rows = [
            row
            for row in sorted_rows
            if row.suggested_action == "WATCH CLOSELY" and id(row) not in urgent_row_ids
        ]
        healthy_rows = [
            row
            for row in sorted_rows
            if row.suggested_action == "HOLD"
        ]

        exit_count = sum(row.suggested_action == "EXIT CANDIDATE" for row in self.rows)
        watch_count = sum(row.suggested_action == "WATCH CLOSELY" for row in self.rows)
        hold_count = sum(row.suggested_action == "HOLD" for row in self.rows)
        raise_stop_count = sum(row.suggested_action == "RAISE STOP" for row in self.rows)

        lines = [f"Intraday portfolio brief for {self.as_of_date.isoformat()}", "", "Headline"]
        lines.append(
            f"Interval: {self.interval_minutes}m | "
            f"Reviewed: {len(self.rows)} | "
            f"Exit: {exit_count} | "
            f"Watch: {watch_count} | "
            f"Hold: {hold_count} | "
            f"Raise stop: {raise_stop_count}"
        )
        lines.append(
            "Benchmark: "
            + (self.benchmark_symbol or "none")
            + " | Portfolio: "
            + (self.portfolio_path or "n/a")
        )

        lines.extend(("", "Urgent intraday actions"))
        if urgent_rows:
            for row in urgent_rows:
                lines.append(_intraday_portfolio_review_brief_line(row))
        else:
            lines.append("No urgent intraday actions.")

        lines.extend(("", "Current holdings under pressure"))
        if pressure_rows:
            for row in pressure_rows:
                lines.append(_intraday_portfolio_review_brief_line(row))
        else:
            lines.append("No holdings are currently under intraday pressure.")

        lines.extend(("", "Holdings still healthy"))
        if healthy_rows:
            for row in healthy_rows:
                lines.append(_intraday_portfolio_review_brief_line(row))
        else:
            lines.append("No holdings are currently classified as healthy.")

        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class MarketMonitorAlertRow:
    """One compact alert row for scheduled market monitoring."""

    date: date
    symbol: str
    category: str
    preset_name: str | None
    latest_close: float | None
    average_entry_price: float | None
    current_stop: float | None
    suggested_stop: float | None
    entry_price_hint: float | None
    stop_level: float | None
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in MARKET_MONITOR_CATEGORIES:
            raise ValueError(
                f"category must be one of {MARKET_MONITOR_CATEGORIES}, got '{self.category}'."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the alert row."""

        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "category": self.category,
            "preset_name": self.preset_name,
            "latest_close": self.latest_close,
            "average_entry_price": self.average_entry_price,
            "current_stop": self.current_stop,
            "suggested_stop": self.suggested_stop,
            "entry_price_hint": self.entry_price_hint,
            "stop_level": self.stop_level,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly representation of the alert row."""

        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "category": self.category,
            "preset_name": self.preset_name or "",
            "latest_close": _format_optional_float(self.latest_close),
            "average_entry_price": _format_optional_float(self.average_entry_price),
            "current_stop": _format_optional_float(self.current_stop),
            "suggested_stop": _format_optional_float(self.suggested_stop),
            "entry_price_hint": _format_optional_float(self.entry_price_hint),
            "stop_level": _format_optional_float(self.stop_level),
            "rationale": self.rationale,
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(frozen=True)
class MarketMonitorReport:
    """Combined entry and portfolio-management alert summary."""

    as_of_date: date
    generated_at_utc: str
    portfolio_path: str | None
    preset_names: tuple[str, ...]
    benchmark_symbol: str | None
    alerts: tuple[MarketMonitorAlertRow, ...]
    ranked_opportunity_rows: tuple[DailyResearchOpportunityRow, ...] = ()
    portfolio_review_rows: tuple[PortfolioReviewRow, ...] = ()
    market_context: MarketContext | None = None

    @property
    def category_counts(self) -> dict[str, int]:
        """Return alert counts for each supported category."""

        counts = {category: 0 for category in MARKET_MONITOR_CATEGORIES}
        for alert in self.alerts:
            counts[alert.category] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly market-monitor payload."""

        category_counts = self.category_counts
        flat_count_fields = {
            market_monitor_flat_count_key(category): category_counts[category]
            for category in MARKET_MONITOR_CATEGORIES
        }
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at_utc": self.generated_at_utc,
            "portfolio_path": self.portfolio_path,
            "preset_names": list(self.preset_names),
            "benchmark_symbol": self.benchmark_symbol,
            "market_context": (
                self.market_context.to_dict()
                if self.market_context is not None
                else None
            ),
            "alert_count": len(self.alerts),
            "category_counts": dict(category_counts),
            **flat_count_fields,
            "alerts": [alert.to_dict() for alert in self.alerts],
        }

    def to_text(self) -> str:
        """Return a compact plain-text summary suitable for later notifications."""

        counts = self.category_counts
        lines = [f"Market monitor for {self.as_of_date.isoformat()}"]
        lines.extend(f"{category}: {counts[category]}" for category in MARKET_MONITOR_CATEGORIES)
        lines.append("")
        for alert in self.alerts:
            details: list[str] = [alert.category]
            if alert.symbol:
                details.append(alert.symbol)
            if alert.preset_name:
                details.append(f"preset={alert.preset_name}")
            if alert.latest_close is not None:
                details.append(f"close={_format_optional_float(alert.latest_close)}")
            if alert.entry_price_hint is not None:
                details.append(f"entry_hint={_format_optional_float(alert.entry_price_hint)}")
            if alert.average_entry_price is not None:
                details.append(f"avg_entry={_format_optional_float(alert.average_entry_price)}")
            if alert.current_stop is not None:
                details.append(f"current_stop={_format_optional_float(alert.current_stop)}")
            if alert.suggested_stop is not None:
                details.append(f"suggested_stop={_format_optional_float(alert.suggested_stop)}")
            if alert.stop_level is not None:
                details.append(f"stop={_format_optional_float(alert.stop_level)}")
            lines.append(" | ".join(details))
            lines.append(f"  {alert.rationale}")
        return "\n".join(lines).rstrip() + "\n"

    def to_brief(self) -> str:
        """Return a human-readable trader brief for the monitor run."""

        counts = self.category_counts
        approved_buy_rows = [
            row
            for row in self.ranked_opportunity_rows
            if row.status == "approved" and _row_actionable_now(row)
        ]
        deferred_buy_rows = [
            row
            for row in self.ranked_opportunity_rows
            if row.status == "approved" and not _row_actionable_now(row)
        ]
        lower_priority_buy_rows = deferred_buy_rows + list(
            approved_buy_rows[MARKET_MONITOR_BRIEF_TOP_BUY_LIMIT:]
        ) + [
            row for row in self.ranked_opportunity_rows if row.status != "approved"
        ]
        top_buy_rows = approved_buy_rows[:MARKET_MONITOR_BRIEF_TOP_BUY_LIMIT]
        holding_rows = sorted(
            self.portfolio_review_rows,
            key=lambda row: (
                MARKET_MONITOR_CATEGORY_PRIORITY.get(row.suggested_action, 99),
                row.symbol,
                row.preset_name or "",
            ),
        )
        urgent_holding_rows = [
            row
            for row in holding_rows
            if row.suggested_action in {"EXIT CANDIDATE", "RAISE STOP"}
        ]

        lines = [f"Market monitor brief for {self.as_of_date.isoformat()}", "", "Headline"]
        lines.append(
            f"Actionable now: {len(top_buy_rows) + len(urgent_holding_rows)} | "
            f"Top buys shown: {len(top_buy_rows)} | "
            f"Raise stop: {counts['RAISE STOP']} | "
            f"Exit: {counts['EXIT CANDIDATE']} | "
            f"Watch: {counts['WATCH CLOSELY']}"
        )
        lines.append(
            "Presets: "
            + (", ".join(self.preset_names) if self.preset_names else "none")
        )
        breadth_line = _market_breadth_brief_line(self.market_context, label="Breadth")
        if breadth_line is not None:
            lines.append(breadth_line)
        volatility_line = _volatility_regime_brief_line(
            self.market_context,
            label="Volatility",
        )
        if volatility_line is not None:
            lines.append(volatility_line)

        lines.extend(("", "Best actions now"))
        if not urgent_holding_rows and not top_buy_rows:
            lines.append("No immediate action is required.")
        else:
            if urgent_holding_rows:
                lines.append(
                    "- Urgent holdings: "
                    + ", ".join(
                        f"{row.symbol} ({row.suggested_action})"
                        for row in urgent_holding_rows
                    )
                )
            if top_buy_rows:
                lines.append(
                    "- Best buys: "
                    + ", ".join(row.symbol for row in top_buy_rows)
                )

        lines.extend(("", "Current holdings"))
        if holding_rows:
            for row in holding_rows:
                lines.append(_portfolio_review_brief_line(row))
        else:
            lines.append("No held positions were reviewed.")

        lines.extend(("", "Top buy candidates"))
        if top_buy_rows:
            for row in top_buy_rows:
                lines.append(_daily_research_monitor_brief_line(row))
        elif deferred_buy_rows:
            lines.append("No approved buy candidates are currently actionable under portfolio capacity.")
        else:
            lines.append("No approved buy candidates.")

        lines.extend(("", "Lower-priority names"))
        if lower_priority_buy_rows:
            for row in lower_priority_buy_rows[:MARKET_MONITOR_BRIEF_LOWER_PRIORITY_LIMIT]:
                lines.append(_daily_research_monitor_brief_line(row))
            remaining_count = len(lower_priority_buy_rows) - MARKET_MONITOR_BRIEF_LOWER_PRIORITY_LIMIT
            if remaining_count > 0:
                lines.append(f"- {remaining_count} more lower-priority buy candidates not shown.")
        else:
            lines.append("No lower-priority buy candidates.")

        return "\n".join(lines).rstrip() + "\n"


def build_portfolio_review_report(
    *,
    as_of_date: date,
    rows: Sequence[PortfolioReviewRow],
    current_positions: Sequence[ExistingPosition],
    benchmark_symbol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PortfolioReviewReport:
    """Build a portfolio review report from row-level management suggestions."""

    return PortfolioReviewReport(
        as_of_date=as_of_date,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        benchmark_symbol=benchmark_symbol,
        current_positions=tuple(current_positions),
        rows=tuple(rows),
        metadata=dict(metadata or {}),
    )


def build_intraday_portfolio_review_report(
    *,
    as_of_date: date,
    interval_minutes: int,
    portfolio_path: str | None,
    rows: Sequence[PortfolioReviewRow],
    current_positions: Sequence[ExistingPosition],
    benchmark_symbol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IntradayPortfolioReviewReport:
    """Build an intraday portfolio review report from row-level suggestions."""

    return IntradayPortfolioReviewReport(
        as_of_date=as_of_date,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        interval_minutes=interval_minutes,
        portfolio_path=portfolio_path,
        benchmark_symbol=benchmark_symbol,
        current_positions=tuple(current_positions),
        rows=tuple(rows),
        metadata=dict(metadata or {}),
    )


def build_market_monitor_report(
    *,
    as_of_date: date,
    daily_summary: DailyResearchSummary | None = None,
    portfolio_review: PortfolioReviewReport | None = None,
    portfolio_path: str | None = None,
) -> MarketMonitorReport:
    """Build a combined monitoring report from entry and position-review outputs."""

    alerts: list[MarketMonitorAlertRow] = []
    preset_names: list[str] = []
    benchmark_symbol: str | None = None

    if daily_summary is not None:
        preset_names.extend(preset.name for preset in daily_summary.selected_presets)
        benchmark_symbol = daily_summary.benchmark_symbol or benchmark_symbol
        for row in daily_summary.rows:
            if row.status != "approved" or not _row_actionable_now(row):
                continue
            alerts.append(
                MarketMonitorAlertRow(
                    date=as_of_date,
                    symbol=row.symbol,
                    category="BUY CANDIDATE",
                    preset_name=row.preset_name,
                    latest_close=row.entry_price_hint,
                    average_entry_price=None,
                    current_stop=None,
                    suggested_stop=None,
                    entry_price_hint=row.entry_price_hint,
                    stop_level=row.stop_level,
                    rationale=row.rationale,
                    metadata={
                        "parameter_id": row.parameter_id,
                        "score": row.score,
                        "score_components": dict(row.metadata.get("score_components", {})),
                        "execution_priority": dict(row.metadata.get("execution_priority", {})),
                        "strategy_name": row.strategy_name,
                        "signal_metadata": dict(row.metadata.get("signal_metadata", {})),
                    },
                )
            )

    if portfolio_review is not None:
        benchmark_symbol = portfolio_review.benchmark_symbol or benchmark_symbol
        preset_names.extend(
            row.preset_name for row in portfolio_review.rows if row.preset_name is not None
        )
        for row in portfolio_review.rows:
            alerts.append(
                MarketMonitorAlertRow(
                    date=as_of_date,
                    symbol=row.symbol,
                    category=row.suggested_action,
                    preset_name=row.preset_name,
                    latest_close=row.latest_close,
                    average_entry_price=row.average_entry_price,
                    current_stop=row.current_stop,
                    suggested_stop=row.suggested_stop,
                    entry_price_hint=None,
                    stop_level=None,
                    rationale=row.rationale,
                    metadata=dict(row.metadata),
                )
            )

    if not alerts:
        alerts.append(
            MarketMonitorAlertRow(
                date=as_of_date,
                symbol="",
                category="NO ACTION",
                preset_name=None,
                latest_close=None,
                average_entry_price=None,
                current_stop=None,
                suggested_stop=None,
                entry_price_hint=None,
                stop_level=None,
                rationale="No approved buy candidates or portfolio-management alerts for the requested date.",
                metadata={},
            )
        )

    alerts.sort(
        key=lambda alert: (
            MARKET_MONITOR_CATEGORY_PRIORITY.get(alert.category, 99),
            alert.symbol,
            alert.preset_name or "",
        )
    )
    return MarketMonitorReport(
        as_of_date=as_of_date,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        portfolio_path=portfolio_path,
        preset_names=tuple(dict.fromkeys(preset_names)),
        benchmark_symbol=benchmark_symbol,
        alerts=tuple(alerts),
        ranked_opportunity_rows=(
            tuple(daily_summary.rows)
            if daily_summary is not None
            else ()
        ),
        portfolio_review_rows=(
            tuple(portfolio_review.rows)
            if portfolio_review is not None
            else ()
        ),
        market_context=(
            daily_summary.market_context
            if daily_summary is not None
            else None
        ),
    )


def write_portfolio_review_report(
    report: PortfolioReviewReport,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write a portfolio review report to CSV or JSON."""

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
        writer = csv.DictWriter(handle, fieldnames=PORTFOLIO_REVIEW_COLUMNS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.to_record())
    return resolved_output_path


def write_intraday_portfolio_review_report(
    report: IntradayPortfolioReviewReport,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write an intraday portfolio review report to CSV or JSON."""

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
        writer = csv.DictWriter(handle, fieldnames=PORTFOLIO_REVIEW_COLUMNS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.to_record())
    return resolved_output_path


def write_intraday_portfolio_review_brief(
    report: IntradayPortfolioReviewReport,
    output_path: Path,
) -> Path:
    """Write a human-readable intraday portfolio trader brief."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(report.to_brief(), encoding="utf-8")
    return resolved_output_path


def write_market_monitor_report(
    report: MarketMonitorReport,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write a market-monitor report to CSV or JSON."""

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
        writer = csv.DictWriter(handle, fieldnames=MARKET_MONITOR_COLUMNS)
        writer.writeheader()
        for row in report.alerts:
            writer.writerow(row.to_record())
    return resolved_output_path


def write_market_monitor_text_summary(
    report: MarketMonitorReport,
    output_path: Path,
) -> Path:
    """Write a plain-text market-monitor summary suitable for scheduled delivery."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(report.to_text(), encoding="utf-8")
    return resolved_output_path


def write_market_monitor_brief(
    report: MarketMonitorReport,
    output_path: Path,
) -> Path:
    """Write a human-readable market-monitor trader brief."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(report.to_brief(), encoding="utf-8")
    return resolved_output_path


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
    current_positions: tuple[ExistingPosition, ...]
    no_signal_symbols: tuple[str, ...]
    benchmark_symbol: str | None
    rows: tuple[DailySignalReportRow, ...]
    pending_order_summary: dict[str, Any] | None = None

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
            "current_position_count": len(self.current_positions),
            "current_position_symbols": [position.symbol for position in self.current_positions],
            "current_positions": [position.to_dict() for position in self.current_positions],
            "pending_order_summary": (
                dict(self.pending_order_summary)
                if self.pending_order_summary is not None
                else None
            ),
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
    current_positions: Sequence[ExistingPosition] = (),
    no_signal_symbols: Sequence[str] = (),
    benchmark_symbol: str | None = None,
    pending_order_summary: Mapping[str, Any] | None = None,
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
        current_positions=tuple(current_positions),
        no_signal_symbols=tuple(no_signal_symbols),
        benchmark_symbol=benchmark_symbol,
        rows=rows,
        pending_order_summary=(
            dict(pending_order_summary)
            if pending_order_summary is not None
            else None
        ),
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


def score_risk_candidate_opportunity(
    candidate: RiskAssessedCandidate,
    *,
    current_equity: float,
) -> tuple[float, dict[str, float]]:
    """Return a deterministic score and its components for a risk candidate."""
    return _score_risk_candidate_opportunity(
        candidate,
        current_equity=current_equity,
    )


def rank_preset_candidate_evaluations(
    evaluations: Sequence[PresetCandidateEvaluation],
    *,
    current_equity: float,
) -> list[PresetCandidateEvaluation]:
    """Return evaluations sorted by approval status and deterministic opportunity score."""

    def sort_key(evaluation: PresetCandidateEvaluation) -> tuple[int, int, float, int, str, str]:
        return risk_candidate_sort_key(
            evaluation.candidate,
            current_equity=current_equity,
        )

    return sorted(evaluations, key=sort_key)


def build_daily_research_summary(
    *,
    as_of_date: date,
    execution_batch: ExecutionBatch,
    evaluations: Sequence[PresetCandidateEvaluation],
    selected_presets: Sequence[BreakoutStrategyPreset],
    universe_symbols: Sequence[str],
    current_positions: Sequence[ExistingPosition] = (),
    current_equity: float,
    no_signal_symbols_by_preset: Mapping[str, Sequence[str]] | None = None,
    benchmark_symbol: str | None = None,
    market_context: MarketContext | None = None,
    preset_selection_source: str | None = None,
    force_require_relative_volume_confirmation: bool = False,
    pending_order_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DailyResearchSummary:
    """Build a consolidated daily preset summary from evaluated candidates."""

    normalized_no_signal_symbols = {
        preset_name: tuple(symbols)
        for preset_name, symbols in (no_signal_symbols_by_preset or {}).items()
    }
    ranked_evaluations = rank_preset_candidate_evaluations(
        evaluations,
        current_equity=current_equity,
    )
    orders_by_key = {
        (order.symbol, order.strategy_name or ""): order
        for order in execution_batch.orders
    }

    rows = tuple(
        _daily_research_row_from_evaluation(
            evaluation,
            rank=index,
            as_of_date=as_of_date,
            current_equity=current_equity,
            order=orders_by_key.get(
                (
                    evaluation.candidate.signal.symbol,
                    evaluation.candidate.signal.strategy_name or "",
                )
            ),
        )
        for index, evaluation in enumerate(ranked_evaluations, start=1)
    )
    preset_summaries = _build_daily_preset_summaries(
        rows=rows,
        selected_presets=selected_presets,
        no_signal_symbols_by_preset=normalized_no_signal_symbols,
        execution_batch=execution_batch,
    )

    return DailyResearchSummary(
        as_of_date=as_of_date,
        generated_at_utc=execution_batch.generated_at_utc,
        executor_name=execution_batch.executor_name,
        selected_presets=tuple(selected_presets),
        preset_selection_source=preset_selection_source,
        benchmark_symbol=benchmark_symbol,
        universe_symbols=tuple(universe_symbols),
        current_positions=tuple(current_positions),
        no_signal_symbols_by_preset=normalized_no_signal_symbols,
        preset_summaries=preset_summaries,
        rows=rows,
        execution_batch=execution_batch,
        market_context=market_context,
        force_require_relative_volume_confirmation=force_require_relative_volume_confirmation,
        pending_order_summary=(
            dict(pending_order_summary)
            if pending_order_summary is not None
            else None
        ),
        metadata=dict(metadata or {}),
    )


def write_daily_research_summary(
    summary: DailyResearchSummary,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write the daily research summary as JSON or ranked-opportunity CSV."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(resolved_output_path, output_format)

    if resolved_format == "json":
        resolved_output_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return resolved_output_path

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_RESEARCH_COLUMNS)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(row.to_record())
    return resolved_output_path


def write_daily_research_brief(
    summary: DailyResearchSummary,
    output_path: Path,
    *,
    output_paths: Mapping[str, str] | None = None,
) -> Path:
    """Write a human-readable daily-summary trader brief."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        summary.to_brief(output_paths=output_paths),
        encoding="utf-8",
    )
    return resolved_output_path


def write_daily_preset_summary(
    summary: DailyResearchSummary,
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write preset-level daily summary rows as CSV or JSON."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(resolved_output_path, output_format)

    if resolved_format == "json":
        resolved_output_path.write_text(
            json.dumps(
                [row.to_dict() for row in summary.preset_summaries],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return resolved_output_path

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_PRESET_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary.preset_summaries:
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
    if candidate.features is not None and candidate.features.market_context is not None:
        metadata["market_context"] = candidate.features.market_context.to_dict()
    if candidate.features is not None and candidate.features.portfolio_heat_context is not None:
        metadata["portfolio_heat_context"] = (
            candidate.features.portfolio_heat_context.to_dict()
        )
    if candidate.portfolio_heat_projection is not None:
        metadata["portfolio_heat_projection"] = candidate.portfolio_heat_projection.to_dict()
    if candidate.execution_priority is not None:
        metadata["execution_priority"] = candidate.execution_priority.to_dict()
    if candidate.existing_position is not None:
        metadata["existing_position"] = candidate.existing_position.to_dict()
    if order is not None:
        metadata["execution_metadata"] = dict(order.metadata)

    rationale = order.rationale if order is not None else _candidate_rationale(candidate)
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


def _daily_research_row_from_evaluation(
    evaluation: PresetCandidateEvaluation,
    *,
    rank: int,
    as_of_date: date,
    current_equity: float,
    order: ExecutionOrder | None,
) -> DailyResearchOpportunityRow:
    candidate = evaluation.candidate
    score, score_components = score_risk_candidate_opportunity(
        candidate,
        current_equity=current_equity,
    )
    metadata = {
        "score_components": dict(score_components),
        "adjusted_risk_per_trade": candidate.adjusted_risk_per_trade,
        "risk_budget": candidate.sizing.risk_budget,
        "per_share_risk": candidate.sizing.per_share_risk,
        "notional_value": candidate.sizing.notional_value,
        "signal_metadata": dict(candidate.signal.metadata),
    }
    if candidate.features is not None and candidate.features.market_context is not None:
        metadata["market_context"] = candidate.features.market_context.to_dict()
    if candidate.features is not None and candidate.features.portfolio_heat_context is not None:
        metadata["portfolio_heat_context"] = (
            candidate.features.portfolio_heat_context.to_dict()
        )
    if candidate.portfolio_heat_projection is not None:
        metadata["portfolio_heat_projection"] = candidate.portfolio_heat_projection.to_dict()
    if candidate.execution_priority is not None:
        metadata["execution_priority"] = candidate.execution_priority.to_dict()
    if candidate.existing_position is not None:
        metadata["existing_position"] = candidate.existing_position.to_dict()
    if order is not None:
        metadata["execution_metadata"] = dict(order.metadata)

    rationale = order.rationale if order is not None else _candidate_rationale(candidate)
    return DailyResearchOpportunityRow(
        rank=rank,
        date=as_of_date,
        signal_date=candidate.signal.date,
        preset_name=evaluation.preset_name,
        parameter_id=evaluation.parameter_id,
        symbol=candidate.signal.symbol,
        status="approved" if candidate.approved else "rejected",
        action=candidate.signal.side,
        quantity=order.quantity if order is not None else 0,
        intended_order_type=order.intended_order_type if order is not None else None,
        entry_price_hint=candidate.entry_price,
        stop_level=candidate.stop_price,
        strategy_name=candidate.signal.strategy_name,
        rationale=rationale,
        score=score,
        breakout_pct=score_components["breakout_pct"],
        relative_volume_ratio=score_components["relative_volume_ratio"],
        position_pct_equity=score_components["position_pct_equity"],
        adjusted_risk_per_trade=candidate.adjusted_risk_per_trade,
        risk_budget=candidate.sizing.risk_budget,
        per_share_risk=candidate.sizing.per_share_risk,
        notional_value=candidate.sizing.notional_value,
        rejection_reasons=candidate.rejection_reasons,
        metadata=metadata,
    )


def _candidate_rationale(candidate: RiskAssessedCandidate) -> str:
    rationale = build_breakout_rationale(
        candidate.signal.entry_reason,
        candidate.signal.metadata,
    )
    priority_note = execution_priority_note(candidate.execution_priority)
    if priority_note is None:
        return rationale
    return f"{rationale} | {priority_note}"


def _market_breadth_brief_line(
    market_context: MarketContext | None,
    *,
    label: str = "Market breadth",
) -> str | None:
    if market_context is None or market_context.market_breadth_pct_above_200ma is None:
        return None

    line = f"{label}: "
    if market_context.market_breadth_state is not None:
        line += f"{market_context.market_breadth_state} | "
    line += f"{market_context.market_breadth_pct_above_200ma:.0%} above 200d"
    if market_context.market_breadth_pct_above_50ma is not None:
        line += f" | {market_context.market_breadth_pct_above_50ma:.0%} above 50d"
    return line


def _volatility_regime_brief_line(
    market_context: MarketContext | None,
    *,
    label: str = "Volatility regime",
) -> str | None:
    if market_context is None or market_context.vix_close is None:
        return None

    line = f"{label}: "
    if market_context.volatility_regime_state is not None:
        line += f"{market_context.volatility_regime_state} | "
    line += f"VIX {market_context.vix_close:.1f}"
    if market_context.vix_sma_short is not None:
        line += f" | 5d {market_context.vix_sma_short:.1f}"
    if market_context.vix_sma_long is not None:
        line += f" | 20d {market_context.vix_sma_long:.1f}"
    return line


def _build_daily_preset_summaries(
    *,
    rows: Sequence[DailyResearchOpportunityRow],
    selected_presets: Sequence[BreakoutStrategyPreset],
    no_signal_symbols_by_preset: Mapping[str, Sequence[str]],
    execution_batch: ExecutionBatch,
) -> tuple[DailyPresetSummaryRow, ...]:
    orders_by_preset: dict[str, int] = {}
    for order in execution_batch.orders:
        preset_name = _extract_preset_name(order.metadata)
        if preset_name is None:
            continue
        orders_by_preset[preset_name] = orders_by_preset.get(preset_name, 0) + 1

    summary_rows: list[DailyPresetSummaryRow] = []
    rows_by_preset: dict[str, list[DailyResearchOpportunityRow]] = {}
    for row in rows:
        rows_by_preset.setdefault(row.preset_name, []).append(row)

    for preset in selected_presets:
        preset_rows = rows_by_preset.get(preset.name, [])
        approved_symbols = tuple(row.symbol for row in preset_rows if row.status == "approved")
        rejected_symbols = tuple(row.symbol for row in preset_rows if row.status == "rejected")
        scoring_rows = [row for row in preset_rows if row.status == "approved"] or list(preset_rows)
        top_score = max((row.score for row in scoring_rows), default=0.0)
        top_symbol = next((row.symbol for row in scoring_rows if row.score == top_score), None)
        average_score = (
            sum(row.score for row in scoring_rows) / len(scoring_rows)
            if scoring_rows
            else 0.0
        )
        summary_rows.append(
            DailyPresetSummaryRow(
                preset_rank=0,
                preset_name=preset.name,
                parameter_id=preset.parameter_id,
                candidate_count=len(preset_rows),
                approved_count=len(approved_symbols),
                rejected_count=len(rejected_symbols),
                no_signal_count=len(tuple(no_signal_symbols_by_preset.get(preset.name, ()))),
                order_count=orders_by_preset.get(preset.name, 0),
                average_score=average_score,
                top_score=top_score,
                top_symbol=top_symbol,
                approved_symbols=approved_symbols,
                rejected_symbols=rejected_symbols,
            )
        )

    sorted_summaries = sorted(
        summary_rows,
        key=lambda row: (
            -row.approved_count,
            -row.top_score,
            -row.average_score,
            -row.candidate_count,
            row.preset_name,
        ),
    )
    return tuple(
        DailyPresetSummaryRow(
            preset_rank=index,
            preset_name=row.preset_name,
            parameter_id=row.parameter_id,
            candidate_count=row.candidate_count,
            approved_count=row.approved_count,
            rejected_count=row.rejected_count,
            no_signal_count=row.no_signal_count,
            order_count=row.order_count,
            average_score=row.average_score,
            top_score=row.top_score,
            top_symbol=row.top_symbol,
            approved_symbols=row.approved_symbols,
            rejected_symbols=row.rejected_symbols,
        )
        for index, row in enumerate(sorted_summaries, start=1)
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _daily_research_brief_card(row: DailyResearchOpportunityRow) -> list[str]:
    return [
        f"{row.rank}. {row.symbol} | preset={row.preset_name} | score={_format_optional_float(row.score)}",
        "   "
        f"qty={row.quantity} | "
        f"entry={_brief_value(row.entry_price_hint)} | "
        f"stop={_brief_value(row.stop_level)} | "
        f"notional={_brief_value(row.notional_value)}",
        f"   {_single_line(row.rationale)}",
    ]


def _daily_research_monitor_brief_line(row: DailyResearchOpportunityRow) -> str:
    if row.status == "approved":
        return (
            f"- {row.rank}. {row.symbol} | preset={row.preset_name} | "
            f"qty={row.quantity} | entry={_brief_value(row.entry_price_hint)} | "
            f"stop={_brief_value(row.stop_level)} | note={_single_line(row.rationale)}"
        )

    reason = row.rejection_reasons[0] if row.rejection_reasons else row.rationale
    return (
        f"- {row.rank}. {row.symbol} | preset={row.preset_name} | "
        f"status={row.status} | reason={_single_line(reason)}"
    )


def _row_actionable_now(row: DailyResearchOpportunityRow) -> bool:
    execution_priority = row.metadata.get("execution_priority")
    if not isinstance(execution_priority, Mapping):
        return row.status == "approved"
    actionable_now = execution_priority.get("actionable_now")
    if isinstance(actionable_now, bool):
        return actionable_now
    return row.status == "approved"


def _market_monitor_brief_line(alert: MarketMonitorAlertRow) -> str:
    parts = [alert.category]
    if alert.symbol:
        parts.append(alert.symbol)
    if alert.preset_name:
        parts.append(f"preset={alert.preset_name}")
    if alert.entry_price_hint is not None:
        parts.append(f"entry={_brief_value(alert.entry_price_hint)}")
    if alert.stop_level is not None:
        parts.append(f"stop={_brief_value(alert.stop_level)}")
    if alert.current_stop is not None:
        parts.append(f"current_stop={_brief_value(alert.current_stop)}")
    if alert.suggested_stop is not None:
        parts.append(f"suggested_stop={_brief_value(alert.suggested_stop)}")
    if alert.latest_close is not None:
        parts.append(f"close={_brief_value(alert.latest_close)}")
    parts.append(f"note={_single_line(alert.rationale)}")
    return "- " + " | ".join(parts)


def _portfolio_review_brief_line(row: PortfolioReviewRow) -> str:
    parts = [row.suggested_action, row.symbol]
    if row.preset_name:
        parts.append(f"preset={row.preset_name}")
    if row.current_stop is not None:
        parts.append(f"current_stop={_brief_value(row.current_stop)}")
    if row.suggested_stop is not None:
        parts.append(f"suggested_stop={_brief_value(row.suggested_stop)}")
    parts.append(f"close={_brief_value(row.latest_close)}")
    parts.append(f"note={_single_line(row.rationale)}")
    return "- " + " | ".join(parts)


def _intraday_portfolio_review_brief_line(row: PortfolioReviewRow) -> str:
    parts = [row.suggested_action, row.symbol]
    if row.preset_name:
        parts.append(f"preset={row.preset_name}")
    parts.append(f"close={_brief_value(row.latest_close)}")
    session_vwap = _optional_float(row.metadata.get("session_vwap"))
    if session_vwap is not None:
        parts.append(f"vwap={_brief_value(session_vwap)}")
    session_high = _optional_float(row.metadata.get("session_high"))
    if session_high is not None:
        parts.append(f"session_high={_brief_value(session_high)}")
    giveback_pct = _optional_float(row.metadata.get("session_high_giveback_pct"))
    if giveback_pct is not None:
        parts.append(f"giveback={giveback_pct:.1%}")
    parts.append(f"note={_single_line(row.rationale)}")
    return "- " + " | ".join(parts)


def _brief_value(value: float | None) -> str:
    formatted = _format_optional_float(value)
    return formatted if formatted else "n/a"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_preset_name(metadata: Mapping[str, Any]) -> str | None:
    signal_metadata = metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        preset_name = signal_metadata.get("preset_name")
        if isinstance(preset_name, str) and preset_name.strip():
            return preset_name.strip()
    preset_name = metadata.get("preset_name")
    if isinstance(preset_name, str) and preset_name.strip():
        return preset_name.strip()
    return None


def _resolve_output_format(output_path: Path, output_format: str | None) -> str:
    normalized = (
        output_format.strip().lower()
        if output_format is not None
        else output_path.suffix.lower().lstrip(".") or "json"
    )
    if normalized not in {"csv", "json"}:
        raise ValueError("output_format must be either 'csv' or 'json'.")
    return normalized
