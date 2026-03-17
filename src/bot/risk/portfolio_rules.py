"""Portfolio-level rules and signal-to-risk assessment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from bot.config import RiskConfig, RulesConfig
from bot.risk.position_sizing import PositionSizingResult, size_position
from bot.strategy.signal_models import StrategySignal


@dataclass(frozen=True)
class ExistingPosition:
    """Minimal existing-position snapshot used by portfolio rules."""

    symbol: str
    shares: int
    average_entry_price: float


@dataclass(frozen=True)
class PortfolioConstraints:
    """Reusable account-level risk and portfolio constraints."""

    max_concurrent_positions: int
    max_position_pct_equity: float | None
    no_averaging_down: bool
    drawdown_risk_reduction_threshold: float | None = None
    drawdown_risk_reduction_factor: float = 0.5

    @classmethod
    def from_configs(
        cls,
        risk_config: RiskConfig,
        rules_config: RulesConfig,
        *,
        drawdown_risk_reduction_factor: float = 0.5,
    ) -> "PortfolioConstraints":
        """Build portfolio constraints from the shared repo config models."""

        return cls(
            max_concurrent_positions=rules_config.max_positions,
            max_position_pct_equity=rules_config.max_position_pct_equity,
            no_averaging_down=rules_config.no_averaging_down,
            drawdown_risk_reduction_threshold=risk_config.drawdown_risk_reduction_threshold,
            drawdown_risk_reduction_factor=drawdown_risk_reduction_factor,
        )

    def __post_init__(self) -> None:
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be greater than zero.")
        if self.max_position_pct_equity is not None and self.max_position_pct_equity <= 0:
            raise ValueError("max_position_pct_equity must be greater than zero when provided.")
        if self.drawdown_risk_reduction_factor <= 0 or self.drawdown_risk_reduction_factor > 1:
            raise ValueError("drawdown_risk_reduction_factor must be in the interval (0, 1].")


@dataclass(frozen=True)
class PortfolioRuleResult:
    """Outcome of deterministic portfolio rule checks."""

    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskAssessedCandidate:
    """A signal combined with sizing and portfolio rule outcomes."""

    signal: StrategySignal
    entry_price: float
    stop_price: float
    adjusted_risk_per_trade: float
    sizing: PositionSizingResult
    approved: bool
    rejection_reasons: tuple[str, ...] = ()


def apply_drawdown_risk_adjustment(
    base_risk_per_trade: float,
    *,
    current_drawdown: float,
    threshold: float | None,
    reduction_factor: float = 0.5,
) -> float:
    """Reduce per-trade risk when drawdown exceeds a configured threshold."""

    _validate_finite_non_negative(base_risk_per_trade, name="base_risk_per_trade")
    _validate_finite_non_negative(current_drawdown, name="current_drawdown")
    if threshold is None:
        return base_risk_per_trade
    _validate_finite_non_negative(threshold, name="threshold")
    if reduction_factor <= 0 or reduction_factor > 1:
        raise ValueError("reduction_factor must be in the interval (0, 1].")
    if current_drawdown >= threshold:
        return base_risk_per_trade * reduction_factor
    return base_risk_per_trade


def evaluate_portfolio_rules(
    signal: StrategySignal,
    *,
    current_positions: Sequence[ExistingPosition],
    current_equity: float,
    proposed_notional: float,
    constraints: PortfolioConstraints,
) -> PortfolioRuleResult:
    """Evaluate deterministic portfolio rules for a candidate position."""

    reasons: list[str] = []
    existing_position = _find_existing_position(current_positions, signal.symbol)

    is_new_symbol = existing_position is None
    if is_new_symbol and len(current_positions) >= constraints.max_concurrent_positions:
        reasons.append("Max concurrent positions reached.")

    if constraints.max_position_pct_equity is not None:
        notional_cap = current_equity * constraints.max_position_pct_equity
        if proposed_notional > notional_cap:
            reasons.append("Proposed position exceeds max position percent of equity.")

    if (
        constraints.no_averaging_down
        and existing_position is not None
        and signal.side == "BUY"
        and signal.entry_price_hint < existing_position.average_entry_price
    ):
        reasons.append("No averaging down is allowed for existing long positions.")

    return PortfolioRuleResult(approved=not reasons, reasons=tuple(reasons))


def assess_signal_candidate(
    signal: StrategySignal,
    *,
    current_equity: float,
    base_risk_per_trade: float,
    constraints: PortfolioConstraints,
    current_positions: Sequence[ExistingPosition] = (),
    current_drawdown: float = 0.0,
    stop_price: float | None = None,
) -> RiskAssessedCandidate:
    """Combine sizing and portfolio rules into one risk-assessed entry candidate."""

    resolved_stop_price = signal.stop_hint if stop_price is None else stop_price
    adjusted_risk_per_trade = apply_drawdown_risk_adjustment(
        base_risk_per_trade,
        current_drawdown=current_drawdown,
        threshold=constraints.drawdown_risk_reduction_threshold,
        reduction_factor=constraints.drawdown_risk_reduction_factor,
    )

    if resolved_stop_price is None:
        sizing = PositionSizingResult(
            shares=0,
            risk_budget=max(current_equity * adjusted_risk_per_trade, 0.0),
            per_share_risk=0.0,
            notional_value=0.0,
            max_shares_by_risk=0,
            max_shares_by_notional=None,
            capped_by_notional=False,
            is_valid=False,
            rejection_reason="A stop price is required for risk assessment.",
        )
        return RiskAssessedCandidate(
            signal=signal,
            entry_price=signal.entry_price_hint,
            stop_price=signal.entry_price_hint,
            adjusted_risk_per_trade=adjusted_risk_per_trade,
            sizing=sizing,
            approved=False,
            rejection_reasons=(sizing.rejection_reason,),
        )

    sizing = size_position(
        current_equity=current_equity,
        risk_per_trade=adjusted_risk_per_trade,
        entry_price=signal.entry_price_hint,
        stop_price=resolved_stop_price,
        max_position_pct_equity=constraints.max_position_pct_equity,
    )

    rejection_reasons: list[str] = []
    if not sizing.is_valid and sizing.rejection_reason is not None:
        rejection_reasons.append(sizing.rejection_reason)

    if sizing.is_valid:
        rule_result = evaluate_portfolio_rules(
            signal,
            current_positions=current_positions,
            current_equity=current_equity,
            proposed_notional=sizing.notional_value,
            constraints=constraints,
        )
        rejection_reasons.extend(rule_result.reasons)

    approved = sizing.is_valid and not rejection_reasons
    return RiskAssessedCandidate(
        signal=signal,
        entry_price=signal.entry_price_hint,
        stop_price=resolved_stop_price,
        adjusted_risk_per_trade=adjusted_risk_per_trade,
        sizing=sizing,
        approved=approved,
        rejection_reasons=tuple(rejection_reasons),
    )


def _find_existing_position(
    positions: Sequence[ExistingPosition],
    symbol: str,
) -> ExistingPosition | None:
    normalized_symbol = symbol.strip().upper()
    for position in positions:
        if position.symbol.strip().upper() == normalized_symbol:
            return position
    return None


def _validate_finite_non_negative(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
