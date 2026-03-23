"""Post-assessment execution prioritization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping, Sequence

from bot.features import MarketContext
from bot.risk.portfolio_rules import ExistingPosition, PortfolioConstraints, RiskAssessedCandidate

EXECUTION_PRIORITY_BUCKETS = (
    "top_priority",
    "secondary_priority",
    "lower_priority",
    "capacity_constrained",
)


@dataclass(frozen=True)
class ExecutionPriority:
    """Execution-priority payload attached to approved candidates after assessment."""

    priority_score: float
    opportunity_score: float
    setup_quality_score: float
    sector_support_score: float
    market_support_score: float
    portfolio_heat_penalty: float
    capital_efficiency_score: float
    queue_rank: int | None = None
    remaining_slots: int = 0
    actionable_now: bool = False
    priority_bucket: str = "lower_priority"
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.priority_score):
            raise ValueError("priority_score must be finite.")
        if self.queue_rank is not None and self.queue_rank <= 0:
            raise ValueError("queue_rank must be greater than zero when provided.")
        if self.remaining_slots < 0:
            raise ValueError("remaining_slots cannot be negative.")
        if self.priority_bucket not in EXECUTION_PRIORITY_BUCKETS:
            raise ValueError(
                f"priority_bucket must be one of {EXECUTION_PRIORITY_BUCKETS}, got '{self.priority_bucket}'."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the execution priority payload."""

        return {
            "priority_score": self.priority_score,
            "opportunity_score": self.opportunity_score,
            "setup_quality_score": self.setup_quality_score,
            "sector_support_score": self.sector_support_score,
            "market_support_score": self.market_support_score,
            "portfolio_heat_penalty": self.portfolio_heat_penalty,
            "capital_efficiency_score": self.capital_efficiency_score,
            "queue_rank": self.queue_rank,
            "remaining_slots": self.remaining_slots,
            "actionable_now": self.actionable_now,
            "priority_bucket": self.priority_bucket,
            "rationale": list(self.rationale),
        }


def score_risk_candidate_opportunity(
    candidate: RiskAssessedCandidate,
    *,
    current_equity: float,
) -> tuple[float, dict[str, float]]:
    """Return the compact opportunity score used by reporting and execution priority."""

    prior_high = (
        candidate.features.prior_high
        if candidate.features is not None
        else _optional_float(candidate.signal.metadata.get("prior_high"))
    )
    relative_volume = (
        candidate.features.relative_volume
        if candidate.features is not None
        else _optional_float(candidate.signal.metadata.get("relative_volume"))
    )
    relative_volume_threshold = _optional_float(candidate.signal.metadata.get("relative_volume_threshold"))

    breakout_pct = (
        candidate.features.breakout_strength
        if candidate.features is not None and candidate.features.breakout_strength is not None
        else 0.0
    )
    if breakout_pct == 0.0 and prior_high is not None and prior_high > 0 and candidate.entry_price > 0:
        breakout_pct = max((candidate.entry_price - prior_high) / prior_high, 0.0)

    relative_volume_ratio = 0.0
    if relative_volume is not None:
        if relative_volume_threshold is not None and relative_volume_threshold > 0:
            relative_volume_ratio = max(relative_volume / relative_volume_threshold, 0.0)
        else:
            relative_volume_ratio = max(relative_volume, 0.0)

    position_pct_equity = (
        max(candidate.sizing.notional_value / current_equity, 0.0)
        if current_equity > 0
        else 0.0
    )
    candidate_memory_score = max(
        (
            candidate.features.setup_quality_score
            if candidate.features is not None
            else _optional_float(candidate.signal.metadata.get("setup_quality_score"))
        )
        or 0.0,
        0.0,
    )
    score = (
        breakout_pct * 1000.0
        + relative_volume_ratio * 100.0
        + position_pct_equity * 25.0
        + candidate_memory_score * 15.0
    )
    return score, {
        "breakout_pct": breakout_pct,
        "relative_volume_ratio": relative_volume_ratio,
        "position_pct_equity": position_pct_equity,
        "candidate_memory_score": candidate_memory_score,
    }


def annotate_execution_priority(
    candidates: Sequence[RiskAssessedCandidate],
    *,
    current_equity: float,
    current_positions: Sequence[ExistingPosition],
    constraints: PortfolioConstraints,
) -> list[RiskAssessedCandidate]:
    """Attach execution-priority metadata to approved candidates."""

    remaining_slots = max(constraints.max_concurrent_positions - len(current_positions), 0)
    prioritized_candidates: list[RiskAssessedCandidate] = list(candidates)

    approved_entries: list[tuple[int, RiskAssessedCandidate, ExecutionPriority]] = []
    for index, candidate in enumerate(candidates):
        if not candidate.approved:
            continue

        opportunity_score, score_components = score_risk_candidate_opportunity(
            candidate,
            current_equity=current_equity,
        )
        setup_quality_score = score_components["candidate_memory_score"]
        sector_support_score = _sector_support_score(candidate)
        market_support_score = _market_support_score(
            candidate.features.market_context if candidate.features is not None else None
        )
        portfolio_heat_penalty = _portfolio_heat_penalty(candidate)
        capital_efficiency_score = _capital_efficiency_score(
            score_components["position_pct_equity"]
        )
        priority_score = (
            opportunity_score
            + setup_quality_score * 10.0
            + sector_support_score * 12.0
            + market_support_score * 8.0
            + capital_efficiency_score * 5.0
            - portfolio_heat_penalty * 25.0
        )
        approved_entries.append(
            (
                index,
                candidate,
                ExecutionPriority(
                    priority_score=priority_score,
                    opportunity_score=opportunity_score,
                    setup_quality_score=setup_quality_score,
                    sector_support_score=sector_support_score,
                    market_support_score=market_support_score,
                    portfolio_heat_penalty=portfolio_heat_penalty,
                    capital_efficiency_score=capital_efficiency_score,
                ),
            )
        )

    approved_entries.sort(
        key=lambda item: (
            -item[2].priority_score,
            -item[2].opportunity_score,
            item[1].signal.strategy_name or "",
            item[1].signal.symbol,
        )
    )

    for queue_rank, (index, candidate, priority) in enumerate(approved_entries, start=1):
        actionable_now = queue_rank <= remaining_slots
        priority_bucket = _priority_bucket_for_rank(
            queue_rank=queue_rank,
            remaining_slots=remaining_slots,
        )
        finalized_priority = replace(
            priority,
            queue_rank=queue_rank,
            remaining_slots=remaining_slots,
            actionable_now=actionable_now,
            priority_bucket=priority_bucket,
            rationale=_priority_rationale(
                priority_bucket=priority_bucket,
                sector_support_score=priority.sector_support_score,
                portfolio_heat_penalty=priority.portfolio_heat_penalty,
            ),
        )
        prioritized_candidates[index] = replace(
            candidate,
            execution_priority=finalized_priority,
        )

    return prioritized_candidates


def rank_risk_assessed_candidates(
    candidates: Sequence[RiskAssessedCandidate],
    *,
    current_equity: float,
) -> list[RiskAssessedCandidate]:
    """Return candidates sorted in queue order, then deterministic fallback order."""

    return sorted(
        candidates,
        key=lambda candidate: risk_candidate_sort_key(
            candidate,
            current_equity=current_equity,
        ),
    )


def risk_candidate_sort_key(
    candidate: RiskAssessedCandidate,
    *,
    current_equity: float,
) -> tuple[int, int, float, int, str, str]:
    """Return the shared deterministic sort key used by execution workflows."""

    approval_rank = 1 if candidate.approved else 0
    priority = candidate.execution_priority
    actionable_rank = 1 if priority is not None and priority.actionable_now else 0
    if priority is not None:
        opportunity_score = priority.opportunity_score
        priority_score = priority.priority_score
    else:
        opportunity_score, _components = score_risk_candidate_opportunity(
            candidate,
            current_equity=current_equity,
        )
        priority_score = opportunity_score
    queue_rank = (
        priority.queue_rank
        if priority is not None and priority.queue_rank is not None
        else 10_000
    )
    return (
        -approval_rank,
        -actionable_rank,
        -priority_score,
        queue_rank,
        candidate.signal.strategy_name or "",
        candidate.signal.symbol,
    )


def execution_priority_note(priority: ExecutionPriority | None) -> str | None:
    """Return a compact one-line user-facing execution-priority note."""

    if priority is None or not priority.rationale:
        return None
    return " ".join(priority.rationale)


def _priority_bucket_for_rank(*, queue_rank: int, remaining_slots: int) -> str:
    if queue_rank > remaining_slots:
        return "capacity_constrained"
    if queue_rank == 1:
        return "top_priority"
    if queue_rank <= min(remaining_slots, 3):
        return "secondary_priority"
    return "lower_priority"


def _priority_rationale(
    *,
    priority_bucket: str,
    sector_support_score: float,
    portfolio_heat_penalty: float,
) -> tuple[str, ...]:
    if priority_bucket == "top_priority":
        rationale = ["Top priority due to the strongest overall execution score."]
        if sector_support_score > 0:
            rationale.append("Sector context is supportive.")
        elif sector_support_score < 0:
            rationale.append("Sector context is not supportive, but this setup still ranks highest.")
        if portfolio_heat_penalty >= 1.0:
            rationale.append("Portfolio heat is elevated but still manageable.")
        elif portfolio_heat_penalty > 0:
            rationale.append("Portfolio crowding remains manageable.")
        return tuple(rationale)
    if priority_bucket == "secondary_priority":
        if portfolio_heat_penalty >= 1.0:
            return (
                "Secondary priority: approved and within current portfolio capacity.",
                "Lower priority because portfolio heat is already elevated in this sector.",
            )
        return (
            "Secondary priority: approved and within current portfolio capacity.",
        )
    if priority_bucket == "lower_priority":
        if portfolio_heat_penalty >= 1.0:
            return (
                "Approved but lower priority than stronger actionable setups.",
                "Lower priority because portfolio heat is already elevated in this sector.",
            )
        if sector_support_score < 0:
            return (
                "Approved but lower priority than stronger actionable setups.",
                "Priority is capped because sector support is not especially strong.",
            )
        return ("Approved but lower priority than stronger actionable setups.",)
    return (
        "Crowded out by higher-ranked approved opportunities under current position limits.",
    )


def _sector_support_score(candidate: RiskAssessedCandidate) -> float:
    score = 0.0
    if candidate.features is None:
        return score

    if candidate.features.sector_regime_passed is True:
        score += 0.5
    elif candidate.features.sector_regime_passed is False:
        score -= 0.5

    if candidate.features.relative_strength_vs_sector is not None:
        score += max(min(candidate.features.relative_strength_vs_sector * 5.0, 0.75), -0.75)

    return score


def _market_support_score(market_context: MarketContext | None) -> float:
    if market_context is None:
        return 0.0

    score = 0.0
    if market_context.market_breadth_state == "healthy":
        score += 0.5
    elif market_context.market_breadth_state == "caution":
        score -= 0.2
    elif market_context.market_breadth_state == "weak":
        score -= 0.5

    if market_context.volatility_regime_state == "calm":
        score += 0.5
    elif market_context.volatility_regime_state == "elevated":
        score -= 0.2
    elif market_context.volatility_regime_state == "stressed":
        score -= 0.5

    return score


def _portfolio_heat_penalty(candidate: RiskAssessedCandidate) -> float:
    if candidate.features is None or candidate.features.portfolio_heat_context is None:
        return 0.0

    context = candidate.features.portfolio_heat_context
    penalty = 0.0
    if context.projected_same_sector_position_count > 0:
        if context.max_positions_per_sector is not None:
            penalty += (
                max(context.projected_same_sector_position_count - 1, 0)
                / context.max_positions_per_sector
            )
        else:
            penalty += max(context.projected_same_sector_position_count - 1, 0) * 0.25
    if context.projected_same_industry_position_count > 0:
        if context.max_same_industry_positions is not None:
            penalty += 0.5 * (
                max(context.projected_same_industry_position_count - 1, 0)
                / context.max_same_industry_positions
            )
        else:
            penalty += max(context.projected_same_industry_position_count - 1, 0) * 0.1

    if (
        candidate.portfolio_heat_projection is not None
        and candidate.portfolio_heat_projection.projected_sector_notional_pct is not None
    ):
        penalty += candidate.portfolio_heat_projection.projected_sector_notional_pct * 0.5
    return penalty


def _capital_efficiency_score(position_pct_equity: float) -> float:
    if position_pct_equity <= 0:
        return 0.0
    return max(1.0 - min(position_pct_equity / 0.25, 1.0), 0.0)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
