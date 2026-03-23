"""Trade-feedback analytics summaries and report writers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bot.data.trade_feedback import TradeFeedbackEvent
from bot.data.state_persistence import write_json_file


_FILTER_CATEGORY_ORDER = (
    "earnings",
    "breadth",
    "volatility",
    "sector",
    "portfolio_heat",
    "other",
)
_PRIORITY_BUCKET_ORDER = (
    "top_priority",
    "secondary_priority",
    "lower_priority",
    "capacity_constrained",
    "unranked",
)
_ACTION_QUALITY_ACTIONS = (
    "WATCH CLOSELY",
    "EXIT CANDIDATE",
    "RAISE STOP",
)
_RESULT_EVENT_TYPES = frozenset({"outcome", "closed"})


@dataclass(frozen=True)
class TradeReviewAnalyticsSummary:
    """Compact analytics summary over persisted trade feedback events."""

    as_of_date: date
    window_days: int | None
    window_start_date: date | None
    feedback_log_path: str
    event_count: int
    event_counts_by_type: dict[str, int]
    event_counts_by_workflow: dict[str, int]
    decision_summary: dict[str, Any]
    outcome_summary: dict[str, Any]
    filter_effectiveness: dict[str, Any]
    execution_priority_summary: dict[str, Any]
    action_quality_summary: dict[str, Any]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly analytics payload."""

        return {
            "as_of_date": self.as_of_date.isoformat(),
            "window_days": self.window_days,
            "window_start_date": (
                self.window_start_date.isoformat()
                if self.window_start_date is not None
                else None
            ),
            "window_end_date": self.as_of_date.isoformat(),
            "feedback_log_path": self.feedback_log_path,
            "event_count": self.event_count,
            "event_counts_by_type": dict(self.event_counts_by_type),
            "event_counts_by_workflow": dict(self.event_counts_by_workflow),
            "decision_summary": dict(self.decision_summary),
            "outcome_summary": dict(self.outcome_summary),
            "filter_effectiveness": dict(self.filter_effectiveness),
            "execution_priority_summary": dict(self.execution_priority_summary),
            "action_quality_summary": dict(self.action_quality_summary),
            "notes": list(self.notes),
        }

    def to_brief(self, *, output_paths: Mapping[str, str] | None = None) -> str:
        """Return a compact human-readable analytics brief."""

        window_label = (
            f"last {self.window_days} days"
            if self.window_days is not None
            else "all available data"
        )
        lines = [
            f"Trade review analytics for {self.as_of_date.isoformat()} ({window_label})",
            "",
        ]
        if self.event_count == 0:
            lines.append("No trade feedback events were available in the selected window.")
        else:
            decision_summary = self.decision_summary
            outcome_summary = self.outcome_summary
            lines.extend(
                [
                    "Headline",
                    (
                        f"- {decision_summary['decision_count']} decisions | "
                        f"{decision_summary['approved_count']} approved | "
                        f"{decision_summary['rejected_count']} rejected | "
                        f"{self.execution_priority_summary['executed_trade_count']} executed trades"
                    ),
                    "",
                    "Outcomes",
                ]
            )
            if outcome_summary["trade_count"] > 0:
                lines.append(
                    (
                        f"- {outcome_summary['trade_count']} trades with outcome snapshots | "
                        f"avg current return {_format_pct(outcome_summary['average_current_return_pct'])} | "
                        f"avg 5d {_format_pct(outcome_summary['average_forward_return_5d'])} | "
                        f"5d positive rate {_format_pct(outcome_summary['positive_forward_return_5d_rate'])}"
                    )
                )
            else:
                lines.append("- No executed trades with outcome snapshots were available.")

            lines.extend(["", "Filters"])
            blocked_by_category = self.filter_effectiveness["blocked_by_reason_category"]
            non_zero_filters = [
                f"{category.replace('_', ' ')}={count}"
                for category, count in blocked_by_category.items()
                if count > 0
            ]
            if non_zero_filters:
                lines.append("- " + " | ".join(non_zero_filters))
            else:
                lines.append("- No rejection filters fired in the selected window.")

            lines.extend(["", "Execution Priority"])
            lines.append(
                (
                    f"- Approved vs executed: "
                    f"{decision_summary['approved_count']} approved / "
                    f"{self.execution_priority_summary['executed_trade_count']} executed"
                )
            )
            for bucket_name in ("top_priority", "lower_priority", "capacity_constrained"):
                bucket_summary = self.execution_priority_summary["by_priority_bucket"].get(
                    bucket_name
                )
                if not isinstance(bucket_summary, Mapping):
                    continue
                if bucket_summary["approved_count"] <= 0:
                    continue
                label = bucket_name.replace("_", " ")
                lines.append(
                    (
                        f"- {label}: {bucket_summary['approved_count']} approved | "
                        f"{bucket_summary['executed_count']} executed | "
                        f"avg latest return {_format_pct(bucket_summary['average_latest_return_pct'])}"
                    )
                )

            lines.extend(["", "Action Quality"])
            rendered_action_summary = False
            for action in _ACTION_QUALITY_ACTIONS:
                summary = self.action_quality_summary.get(action)
                if not isinstance(summary, Mapping):
                    continue
                if summary["decision_count"] <= 0:
                    continue
                rendered_action_summary = True
                if action == "RAISE STOP":
                    lines.append(
                        (
                            f"- {action}: {summary['decision_count']} calls | "
                            f"{summary['linked_trade_count']} linked follow-ups | "
                            f"stop-hit rate {_format_pct(summary['stop_hit_rate'])} | "
                            f"protected gains {_format_pct(summary['protected_gain_rate'])}"
                        )
                    )
                else:
                    lines.append(
                        (
                            f"- {action}: {summary['decision_count']} calls | "
                            f"{summary['linked_trade_count']} linked follow-ups | "
                            f"further weakness rate {_format_pct(summary['further_weakness_rate'])}"
                        )
                    )
            if not rendered_action_summary:
                lines.append("- No linked portfolio-review action follow-through was available.")

        if self.notes:
            lines.extend(["", "Notes"])
            lines.extend(f"- {note}" for note in self.notes)

        if output_paths:
            lines.extend(["", "Outputs"])
            for label, path in output_paths.items():
                lines.append(f"- {label}: {path}")

        return "\n".join(lines)


def default_trade_review_output_dir(
    project_root: Path,
    *,
    as_of_date: date,
    window_days: int | None,
) -> Path:
    """Return the default output directory for one analytics run."""

    window_label = (
        f"{as_of_date.isoformat()}_last_{window_days}_days"
        if window_days is not None
        else f"{as_of_date.isoformat()}_all_available"
    )
    return (
        project_root
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / window_label
    )


def write_trade_review_summary(
    summary: TradeReviewAnalyticsSummary,
    output_path: Path,
) -> Path:
    """Write the analytics summary as JSON."""

    return write_json_file(output_path, summary.to_dict())


def write_trade_review_brief(
    summary: TradeReviewAnalyticsSummary,
    output_path: Path,
    *,
    output_paths: Mapping[str, str] | None = None,
) -> Path:
    """Write the human-readable analytics brief."""

    resolved_path = output_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        summary.to_brief(output_paths=output_paths),
        encoding="utf-8",
    )
    return resolved_path


def build_trade_review_analytics_summary(
    events: Sequence[TradeFeedbackEvent],
    *,
    as_of_date: date,
    feedback_log_path: Path,
    window_days: int | None = 30,
) -> TradeReviewAnalyticsSummary:
    """Build a deterministic analytics summary over the persisted feedback log."""

    if window_days is not None and window_days <= 0:
        raise ValueError("window_days must be greater than zero when provided.")

    window_start_date = (
        as_of_date - timedelta(days=window_days - 1)
        if window_days is not None
        else None
    )
    filtered_events = tuple(
        event
        for event in events
        if _event_is_in_window(
            event,
            as_of_date=as_of_date,
            window_start_date=window_start_date,
        )
    )

    event_counts_by_type = _ordered_counter(
        event.event_type for event in filtered_events
    )
    event_counts_by_workflow = _ordered_counter(
        event.workflow for event in filtered_events
    )

    decision_events = tuple(
        event for event in filtered_events if event.event_type == "decision"
    )
    approved_decisions = tuple(
        event for event in decision_events if event.approved is True
    )
    rejected_decisions = tuple(
        event for event in decision_events if event.approved is False
    )
    executed_events = tuple(
        event for event in filtered_events if event.event_type == "executed"
    )
    outcome_events = tuple(
        event
        for event in filtered_events
        if event.event_type == "outcome" and event.outcome_snapshot is not None
    )
    stop_events = tuple(
        event
        for event in filtered_events
        if event.event_type in {"stop_raised", "stop_updated"}
    )
    closed_events = tuple(
        event for event in filtered_events if event.event_type == "closed"
    )

    approved_decision_ids = {
        event.decision_id
        for event in approved_decisions
        if event.decision_id is not None
    }
    executed_decision_ids = {
        event.linked_decision_id or event.decision_id
        for event in executed_events
        if (event.linked_decision_id or event.decision_id) is not None
    }
    executed_trade_keys = {
        trade_key
        for event in executed_events
        for trade_key in [_trade_lookup_key(event)]
        if trade_key is not None
    }
    executed_trade_count = len(executed_trade_keys) or len(executed_events)
    approved_executed_decision_count = len(approved_decision_ids & executed_decision_ids)

    latest_outcomes_by_trade = _latest_events_by_trade(outcome_events)
    latest_results_by_trade = _latest_events_by_trade(
        tuple(event for event in filtered_events if event.event_type in _RESULT_EVENT_TYPES)
    )
    outcome_summary = _build_outcome_summary(latest_outcomes_by_trade.values())
    execution_priority_summary = _build_execution_priority_summary(
        approved_decisions=approved_decisions,
        executed_events=executed_events,
        latest_outcomes_by_trade=latest_outcomes_by_trade,
        latest_results_by_trade=latest_results_by_trade,
    )
    filter_effectiveness = _build_filter_effectiveness_summary(rejected_decisions)
    action_quality_summary, action_quality_notes = _build_action_quality_summary(
        decision_events=decision_events,
        filtered_events=filtered_events,
    )

    decision_summary = {
        "decision_count": len(decision_events),
        "approved_count": len(approved_decisions),
        "rejected_count": len(rejected_decisions),
        "approved_execution_rate": _ratio(
            approved_executed_decision_count,
            len(approved_decisions),
        ),
        "approved_but_not_executed_count": max(
            len(approved_decisions) - approved_executed_decision_count,
            0,
        ),
        "by_workflow": _ordered_counter(event.workflow for event in decision_events),
        "by_preset": _ordered_counter(
            event.preset_name
            for event in decision_events
            if event.preset_name is not None
        ),
        "by_action": _ordered_counter(
            event.suggested_action
            for event in decision_events
            if event.suggested_action is not None
        ),
    }

    notes: list[str] = []
    if not filtered_events:
        notes.append("No trade feedback events were available in the selected window.")
    if (
        decision_summary["approved_but_not_executed_count"] >= 2
        and decision_summary["approved_but_not_executed_count"] * 4
        >= len(approved_decisions)
    ):
        notes.append(
            "Approved but unexecuted candidates do not currently receive forward outcome tracking."
        )
    if rejected_decisions:
        notes.append(
            "Rejected candidates are counted by filter category, but later outcome comparisons are not available from the current feedback schema."
        )
    if filter_effectiveness["multi_reason_block_count"] > 0:
        notes.append(
            "Some rejected decisions matched multiple filter categories, so category counts can exceed blocked_decision_count."
        )
    notes.extend(action_quality_notes)

    return TradeReviewAnalyticsSummary(
        as_of_date=as_of_date,
        window_days=window_days,
        window_start_date=window_start_date,
        feedback_log_path=str(feedback_log_path.resolve()),
        event_count=len(filtered_events),
        event_counts_by_type=event_counts_by_type,
        event_counts_by_workflow=event_counts_by_workflow,
        decision_summary=decision_summary,
        outcome_summary=outcome_summary,
        filter_effectiveness=filter_effectiveness,
        execution_priority_summary={
            **execution_priority_summary,
            "executed_trade_count": executed_trade_count,
            "closed_trade_count": len(
                {
                    trade_key
                    for event in closed_events
                    for trade_key in [_trade_lookup_key(event)]
                    if trade_key is not None
                }
            )
            or len(closed_events),
            "stop_update_count": len(stop_events),
        },
        action_quality_summary=action_quality_summary,
        notes=tuple(notes),
    )


def _event_is_in_window(
    event: TradeFeedbackEvent,
    *,
    as_of_date: date,
    window_start_date: date | None,
) -> bool:
    if window_start_date is None:
        return True
    if event.as_of_date is None:
        return False
    return window_start_date <= event.as_of_date <= as_of_date


def _build_outcome_summary(
    latest_outcomes: Iterable[TradeFeedbackEvent],
) -> dict[str, Any]:
    outcome_events = tuple(latest_outcomes)
    snapshots = tuple(
        event.outcome_snapshot
        for event in outcome_events
        if event.outcome_snapshot is not None
    )
    return {
        "trade_count": len(snapshots),
        "average_current_return_pct": _mean(
            snapshot.current_return_pct for snapshot in snapshots
        ),
        "average_forward_return_1d": _mean(
            snapshot.forward_return_1d for snapshot in snapshots
        ),
        "average_forward_return_5d": _mean(
            snapshot.forward_return_5d for snapshot in snapshots
        ),
        "average_forward_return_10d": _mean(
            snapshot.forward_return_10d for snapshot in snapshots
        ),
        "positive_forward_return_1d_rate": _positive_rate(
            snapshot.forward_return_1d for snapshot in snapshots
        ),
        "positive_forward_return_5d_rate": _positive_rate(
            snapshot.forward_return_5d for snapshot in snapshots
        ),
        "positive_forward_return_10d_rate": _positive_rate(
            snapshot.forward_return_10d for snapshot in snapshots
        ),
        "average_max_favorable_excursion_pct": _mean(
            snapshot.max_favorable_excursion_pct for snapshot in snapshots
        ),
        "average_max_adverse_excursion_pct": _mean(
            snapshot.max_adverse_excursion_pct for snapshot in snapshots
        ),
        "stop_hit_rate": _boolean_rate(snapshot.stop_hit for snapshot in snapshots),
    }


def _build_filter_effectiveness_summary(
    rejected_decisions: Sequence[TradeFeedbackEvent],
) -> dict[str, Any]:
    blocked_by_category = {category: 0 for category in _FILTER_CATEGORY_ORDER}
    multi_reason_block_count = 0
    for event in rejected_decisions:
        categories = {
            _categorize_rejection_reason(reason)
            for reason in event.rejection_reasons
        }
        if len(categories) > 1:
            multi_reason_block_count += 1
        for category in categories:
            blocked_by_category[category] += 1
    return {
        "blocked_decision_count": len(rejected_decisions),
        "blocked_by_reason_category": blocked_by_category,
        "multi_reason_block_count": multi_reason_block_count,
        "blocked_vs_accepted_outcome_comparison_available": False,
    }


def _build_execution_priority_summary(
    *,
    approved_decisions: Sequence[TradeFeedbackEvent],
    executed_events: Sequence[TradeFeedbackEvent],
    latest_outcomes_by_trade: Mapping[str, TradeFeedbackEvent],
    latest_results_by_trade: Mapping[str, TradeFeedbackEvent],
) -> dict[str, Any]:
    executed_trade_by_decision_id: dict[str, str] = {}
    for event in executed_events:
        trade_key = _trade_lookup_key(event)
        decision_id = event.linked_decision_id or event.decision_id
        if trade_key is None or decision_id is None:
            continue
        executed_trade_by_decision_id.setdefault(decision_id, trade_key)

    bucket_summaries: dict[str, dict[str, Any]] = {}
    crowded_out_approved_count = 0
    crowded_out_executed_count = 0
    for bucket_name in _iter_priority_buckets(approved_decisions):
        bucket_decisions = tuple(
            event
            for event in approved_decisions
            if (event.priority_bucket or "unranked") == bucket_name
        )
        if not bucket_decisions:
            continue
        latest_result_returns: list[float] = []
        forward_returns_5d: list[float] = []
        executed_count = 0
        outcome_trade_count = 0
        for event in bucket_decisions:
            if event.decision_id is None:
                continue
            trade_key = executed_trade_by_decision_id.get(event.decision_id)
            if trade_key is None:
                continue
            executed_count += 1
            latest_result = latest_results_by_trade.get(trade_key)
            latest_return = _event_return_pct(latest_result)
            if latest_return is not None:
                latest_result_returns.append(latest_return)
            latest_outcome = latest_outcomes_by_trade.get(trade_key)
            if latest_outcome is not None and latest_outcome.outcome_snapshot is not None:
                outcome_trade_count += 1
                if latest_outcome.outcome_snapshot.forward_return_5d is not None:
                    forward_returns_5d.append(
                        latest_outcome.outcome_snapshot.forward_return_5d
                    )
        actionable_now_count = sum(
            event.actionable_now is True for event in bucket_decisions
        )
        bucket_summaries[bucket_name] = {
            "approved_count": len(bucket_decisions),
            "actionable_now_count": actionable_now_count,
            "executed_count": executed_count,
            "execution_rate": _ratio(executed_count, len(bucket_decisions)),
            "latest_outcome_trade_count": outcome_trade_count,
            "average_latest_return_pct": _mean(latest_result_returns),
            "average_forward_return_5d": _mean(forward_returns_5d),
        }
        if bucket_name == "capacity_constrained":
            crowded_out_approved_count += len(bucket_decisions)
            crowded_out_executed_count += executed_count

    return {
        "approved_execution_rate": _ratio(
            sum(bucket["executed_count"] for bucket in bucket_summaries.values()),
            len(approved_decisions),
        ),
        "crowded_out_approved_count": crowded_out_approved_count,
        "crowded_out_executed_count": crowded_out_executed_count,
        "by_priority_bucket": bucket_summaries,
    }


def _build_action_quality_summary(
    *,
    decision_events: Sequence[TradeFeedbackEvent],
    filtered_events: Sequence[TradeFeedbackEvent],
) -> tuple[dict[str, Any], list[str]]:
    raw_summary: dict[str, dict[str, Any]] = {
        action: {
            "decision_count": 0,
            "linked_trade_count": 0,
            "further_weakness_count": 0,
            "stop_hit_count": 0,
            "protected_gain_count": 0,
            "_comparable_trade_count": 0,
            "_stop_hit_evaluable_count": 0,
            "_protected_gain_evaluable_count": 0,
            "_return_changes": [],
        }
        for action in _ACTION_QUALITY_ACTIONS
    }

    result_events_by_trade: dict[str, list[TradeFeedbackEvent]] = defaultdict(list)
    for event in filtered_events:
        if event.event_type not in _RESULT_EVENT_TYPES:
            continue
        trade_key = _trade_lookup_key(event)
        if trade_key is None:
            continue
        result_events_by_trade[trade_key].append(event)
    for events in result_events_by_trade.values():
        events.sort(key=_event_sort_key)

    unlinked_review_action_count = 0
    non_comparable_follow_through_count = 0
    for event in decision_events:
        action = event.suggested_action
        if action not in raw_summary:
            continue
        if event.workflow not in {"review-portfolio", "review-portfolio-intraday"}:
            continue
        action_summary = raw_summary[action]
        action_summary["decision_count"] += 1
        trade_key = _trade_lookup_key(event)
        if trade_key is None or event.as_of_date is None:
            unlinked_review_action_count += 1
            continue
        future_events = [
            candidate
            for candidate in result_events_by_trade.get(trade_key, [])
            if candidate.as_of_date is not None and candidate.as_of_date > event.as_of_date
        ]
        if not future_events:
            continue
        action_summary["linked_trade_count"] += 1
        if action == "RAISE STOP":
            stop_hit_outcomes = [
                candidate
                for candidate in future_events
                if candidate.event_type == "outcome"
                and candidate.outcome_snapshot is not None
                and candidate.outcome_snapshot.stop_hit is not None
            ]
            if stop_hit_outcomes:
                action_summary["_stop_hit_evaluable_count"] += 1
            if any(
                candidate.outcome_snapshot is not None
                and candidate.outcome_snapshot.stop_hit is True
                for candidate in stop_hit_outcomes
            ):
                action_summary["stop_hit_count"] += 1

            latest_closed_event = next(
                (
                    candidate
                    for candidate in reversed(future_events)
                    if candidate.event_type == "closed"
                ),
                None,
            )
            realized_return = _event_return_pct(latest_closed_event)
            if realized_return is not None:
                action_summary["_protected_gain_evaluable_count"] += 1
                if realized_return > 0:
                    action_summary["protected_gain_count"] += 1
            continue

        latest_future_event = future_events[-1]
        latest_future_return = _event_return_pct(latest_future_event)
        baseline_return = _mapping_numeric_value(
            event.feature_snapshot.get("unrealized_pl_pct")
        )
        if baseline_return is None or latest_future_return is None:
            non_comparable_follow_through_count += 1
            continue
        action_summary["_comparable_trade_count"] += 1
        change = latest_future_return - baseline_return
        if change < 0:
            action_summary["further_weakness_count"] += 1
        existing_changes = action_summary["_return_changes"]
        if isinstance(existing_changes, list):
            existing_changes.append(change)

    summary: dict[str, Any] = {}
    raise_stop_seen = False
    for action, action_summary in raw_summary.items():
        if action == "RAISE STOP":
            raise_stop_seen = raise_stop_seen or action_summary["decision_count"] > 0
            summary[action] = {
                "decision_count": action_summary["decision_count"],
                "linked_trade_count": action_summary["linked_trade_count"],
                "stop_hit_evaluable_count": action_summary["_stop_hit_evaluable_count"],
                "stop_hit_count": action_summary["stop_hit_count"],
                "stop_hit_rate": _ratio(
                    action_summary["stop_hit_count"],
                    action_summary["_stop_hit_evaluable_count"],
                ),
                "protected_gain_evaluable_count": action_summary[
                    "_protected_gain_evaluable_count"
                ],
                "protected_gain_count": action_summary["protected_gain_count"],
                "protected_gain_rate": _ratio(
                    action_summary["protected_gain_count"],
                    action_summary["_protected_gain_evaluable_count"],
                ),
            }
            continue

        return_changes = action_summary["_return_changes"]
        summary[action] = {
            "decision_count": action_summary["decision_count"],
            "linked_trade_count": action_summary["linked_trade_count"],
            "comparable_trade_count": action_summary["_comparable_trade_count"],
            "further_weakness_count": action_summary["further_weakness_count"],
            "further_weakness_rate": _ratio(
                action_summary["further_weakness_count"],
                action_summary["_comparable_trade_count"],
            ),
            "average_return_change_pct": _mean(return_changes),
        }

    notes: list[str] = []
    if unlinked_review_action_count > 0:
        notes.append(
            f"{unlinked_review_action_count} portfolio-review action decisions lacked trade linkage and were excluded from follow-through summaries."
        )
    if non_comparable_follow_through_count > 0:
        notes.append(
            "WATCH CLOSELY and EXIT CANDIDATE weakness rates use only trades with comparable before/after return snapshots."
        )
    if raise_stop_seen:
        notes.append(
            "Protected-gain rates only count closed trades with positive realized returns after a RAISE STOP call."
        )
    return summary, notes


def _latest_events_by_trade(
    events: Sequence[TradeFeedbackEvent],
) -> dict[str, TradeFeedbackEvent]:
    latest: dict[str, TradeFeedbackEvent] = {}
    for event in events:
        trade_key = _trade_lookup_key(event)
        if trade_key is None:
            continue
        existing = latest.get(trade_key)
        if existing is None or _event_sort_key(event) > _event_sort_key(existing):
            latest[trade_key] = event
    return latest


def _trade_lookup_key(event: TradeFeedbackEvent) -> str | None:
    if event.trade_id is not None:
        return f"trade:{event.trade_id}"
    if event.linked_decision_id is not None:
        return f"decision:{event.linked_decision_id}"
    if event.decision_id is not None:
        return f"decision:{event.decision_id}"
    return None


def _event_sort_key(event: TradeFeedbackEvent) -> tuple[str, str, str, str]:
    return (
        event.as_of_date.isoformat() if event.as_of_date is not None else "",
        event.timestamp_utc or "",
        event.workflow,
        event.event_id,
    )


def _ordered_counter(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(value for value in values if value)
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda key: (-counts[key], key))
    }


def _iter_priority_buckets(
    approved_decisions: Sequence[TradeFeedbackEvent],
) -> tuple[str, ...]:
    present = {event.priority_bucket or "unranked" for event in approved_decisions}
    ordered = [bucket for bucket in _PRIORITY_BUCKET_ORDER if bucket in present]
    extras = sorted(present - set(_PRIORITY_BUCKET_ORDER))
    return tuple((*ordered, *extras))


def _categorize_rejection_reason(reason: str) -> str:
    lowered = reason.casefold()
    if "earnings" in lowered:
        return "earnings"
    if "market breadth" in lowered:
        return "breadth"
    if "volatility regime" in lowered or " vix " in f" {lowered} ":
        return "volatility"
    if (
        "per-sector limit" in lowered
        or "same-industry limit" in lowered
        or "approximate portfolio notional" in lowered
        or "sector notional limit" in lowered
        or lowered.startswith("portfolio already has")
    ):
        return "portfolio_heat"
    if (
        "sector context" in lowered
        or "sector etf" in lowered
        or "sector support" in lowered
        or "sector-relative strength" in lowered
        or ("lagging" in lowered and "sector" in lowered)
    ):
        return "sector"
    return "other"


def _event_return_pct(event: TradeFeedbackEvent | None) -> float | None:
    if event is None:
        return None
    if event.event_type == "outcome" and event.outcome_snapshot is not None:
        return event.outcome_snapshot.current_return_pct
    if event.event_type == "closed":
        return _mapping_numeric_value(event.metadata.get("realized_return_pct"))
    return None


def _mapping_numeric_value(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _positive_rate(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(value > 0 for value in filtered) / len(filtered)


def _boolean_rate(values: Iterable[bool | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(value is True for value in filtered) / len(filtered)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"
