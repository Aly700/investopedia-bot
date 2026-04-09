"""Structured candidate decision outcomes for recommendation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class BlockerCategory(str, Enum):
    """Canonical blocker categories used by candidate assessment."""

    CAPACITY = "capacity"
    DUPLICATE_POSITION = "duplicate_position"
    SIZING = "sizing"
    EARNINGS = "earnings"
    VOLATILITY = "volatility"
    MARKET_BREADTH = "market_breadth"
    SECTOR_REGIME = "sector_regime"
    SECTOR_RELATIVE_STRENGTH = "sector_relative_strength"
    PORTFOLIO_HEAT = "portfolio_heat"
    SECTOR_EXPOSURE = "sector_exposure"
    INDUSTRY_EXPOSURE = "industry_exposure"
    SECTOR_NOTIONAL_EXPOSURE = "sector_notional_exposure"
    OTHER = "other"


class BlockerSeverity(str, Enum):
    """Coarse blocker severity used to separate hard rejects from soft gates."""

    SOFT = "soft"
    HARD = "hard"


class ContextStatus(str, Enum):
    """Availability state for optional recommendation inputs."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CandidateDisposition(str, Enum):
    """Final structured recommendation disposition for one candidate."""

    ACTIONABLE = "actionable"
    APPROVED_CAPACITY_BLOCKED = "approved_capacity_blocked"
    SOFT_GATED = "soft_gated"
    HARD_REJECTED = "hard_rejected"
    DEGRADED_APPROVED = "degraded_approved"


@dataclass(frozen=True)
class CandidateBlocker:
    """One structured blocker attached to a candidate outcome."""

    category: BlockerCategory
    severity: BlockerSeverity
    reason: str
    context_key: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "reason": self.reason,
            "context_key": self.context_key,
        }


@dataclass(frozen=True)
class ContextAvailability:
    """Availability state for optional contexts used during candidate assessment."""

    sector_context: ContextStatus = ContextStatus.UNKNOWN
    volatility_context: ContextStatus = ContextStatus.UNKNOWN
    earnings_context: ContextStatus = ContextStatus.UNKNOWN
    market_breadth_context: ContextStatus = ContextStatus.UNKNOWN

    def to_dict(self) -> dict[str, str]:
        return {
            "sector_context": self.sector_context.value,
            "volatility_context": self.volatility_context.value,
            "earnings_context": self.earnings_context.value,
            "market_breadth_context": self.market_breadth_context.value,
        }

    @property
    def degraded_or_missing_contexts(self) -> tuple[str, ...]:
        degraded: list[str] = []
        for context_key in (
            "sector_context",
            "volatility_context",
            "earnings_context",
            "market_breadth_context",
        ):
            status = getattr(self, context_key)
            if status in {ContextStatus.DEGRADED, ContextStatus.UNAVAILABLE}:
                degraded.append(context_key)
        return tuple(degraded)

    @property
    def has_degraded_or_missing_optional_context(self) -> bool:
        return bool(self.degraded_or_missing_contexts)


@dataclass(frozen=True)
class CandidateOutcome:
    """Structured decision outcome for one assessed candidate."""

    disposition: CandidateDisposition
    blockers: tuple[CandidateBlocker, ...] = ()
    context_availability: ContextAvailability = field(
        default_factory=ContextAvailability
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "context_availability": self.context_availability.to_dict(),
            "legacy_approved": self.legacy_approved,
            "operator_approved": self.operator_approved,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(blocker.reason for blocker in self.blockers)

    @property
    def legacy_approved(self) -> bool:
        return self.disposition in {
            CandidateDisposition.ACTIONABLE,
            CandidateDisposition.DEGRADED_APPROVED,
        }

    @property
    def operator_approved(self) -> bool:
        return self.disposition in {
            CandidateDisposition.ACTIONABLE,
            CandidateDisposition.APPROVED_CAPACITY_BLOCKED,
            CandidateDisposition.DEGRADED_APPROVED,
        }

    @property
    def soft_blocked(self) -> bool:
        return self.disposition in {
            CandidateDisposition.APPROVED_CAPACITY_BLOCKED,
            CandidateDisposition.SOFT_GATED,
        }

    @classmethod
    def from_blockers(
        cls,
        blockers: Sequence[CandidateBlocker] = (),
        *,
        context_availability: ContextAvailability | None = None,
    ) -> "CandidateOutcome":
        resolved_context_availability = context_availability or ContextAvailability()
        resolved_blockers = tuple(blockers)
        hard_blockers = tuple(
            blocker
            for blocker in resolved_blockers
            if blocker.severity == BlockerSeverity.HARD
        )
        soft_blockers = tuple(
            blocker
            for blocker in resolved_blockers
            if blocker.severity == BlockerSeverity.SOFT
        )

        if hard_blockers:
            disposition = CandidateDisposition.HARD_REJECTED
        elif soft_blockers:
            if all(blocker.category == BlockerCategory.CAPACITY for blocker in soft_blockers):
                disposition = CandidateDisposition.APPROVED_CAPACITY_BLOCKED
            else:
                disposition = CandidateDisposition.SOFT_GATED
        elif resolved_context_availability.has_degraded_or_missing_optional_context:
            disposition = CandidateDisposition.DEGRADED_APPROVED
        else:
            disposition = CandidateDisposition.ACTIONABLE

        return cls(
            disposition=disposition,
            blockers=resolved_blockers,
            context_availability=resolved_context_availability,
        )


def build_context_availability(
    *,
    sector_context: ContextStatus | str | bool | None = None,
    volatility_context: ContextStatus | str | bool | None = None,
    earnings_context: ContextStatus | str | bool | None = None,
    market_breadth_context: ContextStatus | str | bool | None = None,
) -> ContextAvailability:
    """Build normalized context availability from coarse booleans or statuses."""

    return ContextAvailability(
        sector_context=_coerce_context_status(sector_context),
        volatility_context=_coerce_context_status(volatility_context),
        earnings_context=_coerce_context_status(earnings_context),
        market_breadth_context=_coerce_context_status(market_breadth_context),
    )


def legacy_candidate_outcome(
    *,
    approved: bool,
    rejection_reasons: Sequence[str] = (),
    context_availability: ContextAvailability | None = None,
) -> CandidateOutcome:
    """Construct a conservative structured outcome from legacy boolean fields."""

    blockers = tuple(
        CandidateBlocker(
            category=BlockerCategory.OTHER,
            severity=BlockerSeverity.HARD,
            reason=reason,
        )
        for reason in rejection_reasons
        if str(reason).strip()
    )
    if approved and not blockers:
        return CandidateOutcome.from_blockers(
            (),
            context_availability=context_availability,
        )
    if not approved and not blockers:
        blockers = (
            CandidateBlocker(
                category=BlockerCategory.OTHER,
                severity=BlockerSeverity.HARD,
                reason="Candidate was not approved.",
            ),
        )
    return CandidateOutcome.from_blockers(
        blockers,
        context_availability=context_availability,
    )


def _coerce_context_status(value: ContextStatus | str | bool | None) -> ContextStatus:
    if isinstance(value, ContextStatus):
        return value
    if isinstance(value, bool):
        return ContextStatus.AVAILABLE if value else ContextStatus.UNAVAILABLE
    if value is None:
        return ContextStatus.UNKNOWN
    normalized = str(value).strip().lower()
    if normalized == ContextStatus.AVAILABLE.value:
        return ContextStatus.AVAILABLE
    if normalized == ContextStatus.DEGRADED.value:
        return ContextStatus.DEGRADED
    if normalized == ContextStatus.UNAVAILABLE.value:
        return ContextStatus.UNAVAILABLE
    if normalized == ContextStatus.UNKNOWN.value:
        return ContextStatus.UNKNOWN
    raise ValueError(f"Unsupported context status: {value!r}")
