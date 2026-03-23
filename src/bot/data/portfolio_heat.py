"""Derived portfolio heat and exposure helpers for new entry assessment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PortfolioHoldingExposure:
    """Minimal holding snapshot used to derive portfolio exposure context."""

    symbol: str
    approximate_notional: float
    sector_name: str | None = None
    industry_name: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if not isfinite(self.approximate_notional) or self.approximate_notional < 0:
            raise ValueError("approximate_notional must be a finite non-negative number.")


@dataclass(frozen=True)
class PortfolioHeatContext:
    """Typed portfolio-level exposure context for one candidate entry."""

    current_position_count: int
    sector_position_count_by_sector: dict[str, int]
    sector_notional_pct_by_sector: dict[str, float]
    current_total_approximate_notional: float
    current_sector_approximate_notional: float
    candidate_sector: str | None = None
    candidate_industry: str | None = None
    same_sector_position_count: int = 0
    same_industry_position_count: int = 0
    projected_same_sector_position_count: int = 0
    projected_same_industry_position_count: int = 0
    sector_concentration_risk: bool = False
    correlated_exposure_risk: bool = False
    max_positions_per_sector: int | None = None
    max_same_industry_positions: int | None = None
    max_sector_notional_pct: float | None = None

    def __post_init__(self) -> None:
        if self.current_position_count < 0:
            raise ValueError("current_position_count cannot be negative.")
        if (
            not isfinite(self.current_total_approximate_notional)
            or self.current_total_approximate_notional < 0
        ):
            raise ValueError(
                "current_total_approximate_notional must be a finite non-negative number."
            )
        if (
            not isfinite(self.current_sector_approximate_notional)
            or self.current_sector_approximate_notional < 0
        ):
            raise ValueError(
                "current_sector_approximate_notional must be a finite non-negative number."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the portfolio heat context."""

        return {
            "current_position_count": self.current_position_count,
            "sector_position_count_by_sector": dict(self.sector_position_count_by_sector),
            "sector_notional_pct_by_sector": dict(self.sector_notional_pct_by_sector),
            "current_total_approximate_notional": self.current_total_approximate_notional,
            "current_sector_approximate_notional": self.current_sector_approximate_notional,
            "candidate_sector": self.candidate_sector,
            "candidate_industry": self.candidate_industry,
            "same_sector_position_count": self.same_sector_position_count,
            "same_industry_position_count": self.same_industry_position_count,
            "projected_same_sector_position_count": self.projected_same_sector_position_count,
            "projected_same_industry_position_count": self.projected_same_industry_position_count,
            "sector_concentration_risk": self.sector_concentration_risk,
            "correlated_exposure_risk": self.correlated_exposure_risk,
            "max_positions_per_sector": self.max_positions_per_sector,
            "max_same_industry_positions": self.max_same_industry_positions,
            "max_sector_notional_pct": self.max_sector_notional_pct,
        }


@dataclass(frozen=True)
class PortfolioHeatProjection:
    """Projected portfolio heat after adding the candidate position.

    The combined ``sector_concentration_risk`` and ``crowded_exposure_bucket`` fields
    are reporting summaries. Entry gates should rely on the more specific count-based
    and sizing-dependent fields alongside the candidate context.
    """

    projected_same_sector_position_count: int = 0
    projected_same_industry_position_count: int = 0
    projected_sector_notional_pct: float | None = None
    sector_count_concentration_risk: bool = False
    sector_notional_concentration_risk: bool = False
    sector_concentration_risk: bool = False
    correlated_exposure_risk: bool = False
    crowded_exposure_bucket: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly projected risk payload."""

        return {
            "projected_same_sector_position_count": self.projected_same_sector_position_count,
            "projected_same_industry_position_count": self.projected_same_industry_position_count,
            "projected_sector_notional_pct": self.projected_sector_notional_pct,
            "sector_count_concentration_risk": self.sector_count_concentration_risk,
            "sector_notional_concentration_risk": self.sector_notional_concentration_risk,
            "sector_concentration_risk": self.sector_concentration_risk,
            "correlated_exposure_risk": self.correlated_exposure_risk,
            "crowded_exposure_bucket": self.crowded_exposure_bucket,
        }


def build_portfolio_heat_context(
    current_holdings: Sequence[PortfolioHoldingExposure],
    *,
    candidate_sector: str | None,
    candidate_industry: str | None,
    max_positions_per_sector: int | None = None,
    max_same_industry_positions: int | None = None,
    max_sector_notional_pct: float | None = None,
) -> PortfolioHeatContext:
    """Build compact portfolio exposure context for one candidate."""

    _validate_optional_positive_int(
        max_positions_per_sector,
        field_name="max_positions_per_sector",
    )
    _validate_optional_positive_int(
        max_same_industry_positions,
        field_name="max_same_industry_positions",
    )
    _validate_optional_pct(
        max_sector_notional_pct,
        field_name="max_sector_notional_pct",
    )

    candidate_sector_label = _display_label(candidate_sector)
    candidate_industry_label = _display_label(candidate_industry)
    normalized_candidate_sector = _normalize_label(candidate_sector)
    normalized_candidate_industry = _normalize_label(candidate_industry)

    sector_position_count_by_key: dict[str, int] = {}
    sector_notional_by_key: dict[str, float] = {}
    sector_display_name_by_key: dict[str, str] = {}
    same_industry_position_count = 0
    current_total_approximate_notional = 0.0

    for holding in current_holdings:
        current_total_approximate_notional += holding.approximate_notional
        normalized_sector_name = _normalize_label(holding.sector_name)
        sector_display_name = _display_label(holding.sector_name)
        if normalized_sector_name is not None:
            sector_position_count_by_key[normalized_sector_name] = (
                sector_position_count_by_key.get(normalized_sector_name, 0) + 1
            )
            sector_notional_by_key[normalized_sector_name] = (
                sector_notional_by_key.get(normalized_sector_name, 0.0)
                + holding.approximate_notional
            )
            if sector_display_name is not None:
                sector_display_name_by_key.setdefault(
                    normalized_sector_name,
                    sector_display_name,
                )

        if (
            normalized_candidate_industry is not None
            and _normalize_label(holding.industry_name) == normalized_candidate_industry
        ):
            same_industry_position_count += 1

    if normalized_candidate_sector is not None and candidate_sector_label is not None:
        sector_display_name_by_key.setdefault(
            normalized_candidate_sector,
            candidate_sector_label,
        )

    sector_sort_key = lambda item: sector_display_name_by_key.get(item[0], item[0])
    sector_position_count_by_sector = {
        sector_display_name_by_key.get(sector_name, sector_name): count
        for sector_name, count in sorted(
            sector_position_count_by_key.items(),
            key=sector_sort_key,
        )
    }

    sector_notional_pct_by_sector = {
        sector_display_name_by_key.get(sector_name, sector_name): (
            sector_notional / current_total_approximate_notional
            if current_total_approximate_notional > 0
            else 0.0
        )
        for sector_name, sector_notional in sorted(
            sector_notional_by_key.items(),
            key=sector_sort_key,
        )
    }

    same_sector_position_count = (
        sector_position_count_by_key.get(normalized_candidate_sector, 0)
        if normalized_candidate_sector is not None
        else 0
    )
    current_sector_approximate_notional = (
        sector_notional_by_key.get(normalized_candidate_sector, 0.0)
        if normalized_candidate_sector is not None
        else 0.0
    )
    projected_same_sector_position_count = (
        same_sector_position_count + 1
        if normalized_candidate_sector is not None
        else 0
    )
    projected_same_industry_position_count = (
        same_industry_position_count + 1
        if normalized_candidate_industry is not None
        else 0
    )
    sector_count_concentration_risk = (
        normalized_candidate_sector is not None
        and max_positions_per_sector is not None
        and projected_same_sector_position_count > max_positions_per_sector
    )
    correlated_exposure_risk = (
        normalized_candidate_industry is not None
        and max_same_industry_positions is not None
        and projected_same_industry_position_count > max_same_industry_positions
    )

    return PortfolioHeatContext(
        current_position_count=len(current_holdings),
        sector_position_count_by_sector=sector_position_count_by_sector,
        sector_notional_pct_by_sector=sector_notional_pct_by_sector,
        current_total_approximate_notional=current_total_approximate_notional,
        current_sector_approximate_notional=current_sector_approximate_notional,
        candidate_sector=candidate_sector_label,
        candidate_industry=candidate_industry_label,
        same_sector_position_count=same_sector_position_count,
        same_industry_position_count=same_industry_position_count,
        projected_same_sector_position_count=projected_same_sector_position_count,
        projected_same_industry_position_count=projected_same_industry_position_count,
        sector_concentration_risk=sector_count_concentration_risk,
        correlated_exposure_risk=correlated_exposure_risk,
        max_positions_per_sector=max_positions_per_sector,
        max_same_industry_positions=max_same_industry_positions,
        max_sector_notional_pct=max_sector_notional_pct,
    )


def project_portfolio_heat_context(
    context: PortfolioHeatContext | None,
    *,
    candidate_notional_value: float | None,
) -> PortfolioHeatProjection:
    """Project concentration risk after sizing the candidate position."""

    if context is None:
        return PortfolioHeatProjection()
    if candidate_notional_value is not None and (
        not isfinite(candidate_notional_value) or candidate_notional_value < 0
    ):
        raise ValueError("candidate_notional_value must be a finite non-negative number.")

    projected_sector_notional_pct = None
    sector_notional_concentration_risk = False
    if (
        context.candidate_sector is not None
        and candidate_notional_value is not None
        and candidate_notional_value > 0
    ):
        projected_total = context.current_total_approximate_notional + candidate_notional_value
        if projected_total > 0:
            projected_sector_notional_pct = (
                context.current_sector_approximate_notional + candidate_notional_value
            ) / projected_total
            sector_notional_concentration_risk = (
                context.max_sector_notional_pct is not None
                and projected_sector_notional_pct > context.max_sector_notional_pct
            )

    sector_count_concentration_risk = context.sector_concentration_risk
    correlated_exposure_risk = context.correlated_exposure_risk
    sector_concentration_risk = (
        sector_count_concentration_risk or sector_notional_concentration_risk
    )
    return PortfolioHeatProjection(
        projected_same_sector_position_count=context.projected_same_sector_position_count,
        projected_same_industry_position_count=context.projected_same_industry_position_count,
        projected_sector_notional_pct=projected_sector_notional_pct,
        sector_count_concentration_risk=sector_count_concentration_risk,
        sector_notional_concentration_risk=sector_notional_concentration_risk,
        sector_concentration_risk=sector_concentration_risk,
        correlated_exposure_risk=correlated_exposure_risk,
        crowded_exposure_bucket=(sector_concentration_risk or correlated_exposure_risk),
    )


def _normalize_label(value: str | None) -> str | None:
    cleaned = _clean_label(value)
    if cleaned is None:
        return None
    return cleaned.casefold()


def _display_label(value: str | None) -> str | None:
    cleaned = _clean_label(value)
    if cleaned is None:
        return None
    if any(character.isalpha() for character in cleaned) and (
        cleaned.islower() or cleaned.isupper()
    ):
        return cleaned.title()
    return cleaned


def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _validate_optional_positive_int(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero when provided.")


def _validate_optional_pct(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value <= 0 or value >= 1:
        raise ValueError(f"{field_name} must be between zero and one when provided.")
