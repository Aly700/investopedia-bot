from __future__ import annotations

from datetime import date
import pytest

from bot.risk.candidate_outcome import (
    BlockerCategory,
    BlockerSeverity,
    CandidateBlocker,
    CandidateDisposition,
    CandidateOutcome,
    ContextStatus,
    build_context_availability,
    is_actionable_candidate_disposition,
    is_operator_approved_disposition,
    legacy_candidate_outcome,
    operator_candidate_disposition,
)
from bot.execution.prioritization import ExecutionPriority
from bot.risk.portfolio_rules import RiskAssessedCandidate
from bot.risk.position_sizing import PositionSizingResult
from bot.strategy.signal_models import StrategySignal


def test_candidate_outcome_classifies_capacity_only_as_approved_capacity_blocked() -> None:
    outcome = CandidateOutcome.from_blockers(
        (
            CandidateBlocker(
                category=BlockerCategory.CAPACITY,
                severity=BlockerSeverity.SOFT,
                reason="Max concurrent positions reached.",
                context_key="max_concurrent_positions",
            ),
        )
    )

    assert outcome.disposition == CandidateDisposition.APPROVED_CAPACITY_BLOCKED
    assert outcome.operator_approved is True
    assert outcome.legacy_approved is False
    assert outcome.rejection_reasons == ("Max concurrent positions reached.",)


def test_candidate_outcome_classifies_non_capacity_soft_blockers_as_soft_gated() -> None:
    outcome = CandidateOutcome.from_blockers(
        (
            CandidateBlocker(
                category=BlockerCategory.SECTOR_REGIME,
                severity=BlockerSeverity.SOFT,
                reason="Sector ETF XLK is below its trend filter; new entries require sector support.",
                context_key="sector_context",
            ),
        )
    )

    assert outcome.disposition == CandidateDisposition.SOFT_GATED
    assert outcome.operator_approved is False
    assert outcome.legacy_approved is False


def test_candidate_outcome_classifies_missing_optional_context_as_degraded_approved() -> None:
    outcome = CandidateOutcome.from_blockers(
        (),
        context_availability=build_context_availability(
            sector_context=False,
            volatility_context=True,
            earnings_context=None,
            market_breadth_context=True,
        ),
    )

    assert outcome.disposition == CandidateDisposition.DEGRADED_APPROVED
    assert outcome.operator_approved is True
    assert outcome.legacy_approved is True
    assert outcome.context_availability.sector_context == ContextStatus.UNAVAILABLE
    assert outcome.context_availability.volatility_context == ContextStatus.AVAILABLE


def test_legacy_candidate_outcome_preserves_backward_compatible_fields() -> None:
    approved = legacy_candidate_outcome(approved=True, rejection_reasons=())
    rejected = legacy_candidate_outcome(
        approved=False,
        rejection_reasons=("Calculated share size is zero after risk and notional constraints.",),
    )

    assert approved.disposition == CandidateDisposition.ACTIONABLE
    assert approved.legacy_approved is True
    assert approved.rejection_reasons == ()
    assert rejected.disposition == CandidateDisposition.HARD_REJECTED
    assert rejected.legacy_approved is False
    assert rejected.rejection_reasons == (
        "Calculated share size is zero after risk and notional constraints.",
    )


def test_risk_assessed_candidate_derives_legacy_fields_from_structured_outcome() -> None:
    signal = StrategySignal(
        strategy_name="breakout_momentum",
        symbol="AAPL",
        date=date(2024, 1, 5),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=100.0,
        stop_hint=95.0,
    )
    sizing = PositionSizingResult(
        shares=10,
        risk_budget=500.0,
        per_share_risk=5.0,
        notional_value=1_000.0,
        max_shares_by_risk=10,
        max_shares_by_notional=10,
        capped_by_notional=False,
        is_valid=True,
        rejection_reason=None,
    )
    outcome = CandidateOutcome.from_blockers(
        (
            CandidateBlocker(
                category=BlockerCategory.CAPACITY,
                severity=BlockerSeverity.SOFT,
                reason="Max concurrent positions reached.",
            ),
        )
    )

    candidate = RiskAssessedCandidate(
        signal=signal,
        entry_price=100.0,
        stop_price=95.0,
        adjusted_risk_per_trade=0.01,
        sizing=sizing,
        outcome=outcome,
    )

    assert candidate.approved is False
    assert candidate.operator_approved is True
    assert candidate.rejection_reasons == ("Max concurrent positions reached.",)
    assert candidate.outcome is outcome


def test_operator_candidate_disposition_represents_queue_time_capacity_blocking() -> None:
    outcome = CandidateOutcome.from_blockers(())

    assert operator_candidate_disposition(
        outcome=outcome,
        actionable_now=True,
    ) == CandidateDisposition.ACTIONABLE
    assert operator_candidate_disposition(
        outcome=outcome,
        actionable_now=False,
    ) == CandidateDisposition.APPROVED_CAPACITY_BLOCKED


def test_risk_assessed_candidate_operator_disposition_prefers_final_operator_meaning() -> None:
    signal = StrategySignal(
        strategy_name="breakout_momentum",
        symbol="AAPL",
        date=date(2024, 1, 5),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=100.0,
        stop_hint=95.0,
    )
    sizing = PositionSizingResult(
        shares=10,
        risk_budget=500.0,
        per_share_risk=5.0,
        notional_value=1_000.0,
        max_shares_by_risk=10,
        max_shares_by_notional=10,
        capped_by_notional=False,
        is_valid=True,
        rejection_reason=None,
    )
    candidate = RiskAssessedCandidate(
        signal=signal,
        entry_price=100.0,
        stop_price=95.0,
        adjusted_risk_per_trade=0.01,
        sizing=sizing,
        outcome=CandidateOutcome.from_blockers(()),
        execution_priority=ExecutionPriority(
            priority_score=10.0,
            opportunity_score=10.0,
            setup_quality_score=1.0,
            sector_support_score=0.0,
            market_support_score=0.0,
            portfolio_heat_penalty=0.0,
            capital_efficiency_score=0.0,
            queue_rank=2,
            remaining_slots=1,
            available_capital=50_000.0,
            actionable_now=False,
            priority_bucket="capacity_constrained",
        ),
    )

    assert candidate.assessment_disposition == CandidateDisposition.ACTIONABLE
    assert candidate.operator_disposition == CandidateDisposition.APPROVED_CAPACITY_BLOCKED
    assert is_operator_approved_disposition(candidate.operator_disposition) is True
    assert is_actionable_candidate_disposition(candidate.operator_disposition) is False


def test_build_context_availability_rejects_unknown_status_text() -> None:
    with pytest.raises(ValueError, match="Unsupported context status"):
        build_context_availability(sector_context="broken")
