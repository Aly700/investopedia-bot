"""Position sizing helpers for risk-budget-based entry planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

from bot.risk.stops import stop_distance


@dataclass(frozen=True)
class PositionSizingResult:
    """Structured result for a risk-budget sizing calculation."""

    shares: int
    risk_budget: float
    per_share_risk: float
    notional_value: float
    max_shares_by_risk: int
    max_shares_by_notional: int | None
    capped_by_notional: bool
    is_valid: bool
    rejection_reason: str | None = None


def size_position(
    *,
    current_equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    max_position_pct_equity: float | None = None,
) -> PositionSizingResult:
    """Size a position using risk budget and optional notional caps."""

    validation_error = _validate_sizing_inputs(
        current_equity=current_equity,
        risk_per_trade=risk_per_trade,
        entry_price=entry_price,
        stop_price=stop_price,
        max_position_pct_equity=max_position_pct_equity,
    )
    if validation_error is not None:
        return PositionSizingResult(
            shares=0,
            risk_budget=max(current_equity * max(risk_per_trade, 0.0), 0.0),
            per_share_risk=0.0,
            notional_value=0.0,
            max_shares_by_risk=0,
            max_shares_by_notional=None,
            capped_by_notional=False,
            is_valid=False,
            rejection_reason=validation_error,
        )

    risk_budget = current_equity * risk_per_trade

    if stop_price >= entry_price:
        return PositionSizingResult(
            shares=0,
            risk_budget=risk_budget,
            per_share_risk=0.0,
            notional_value=0.0,
            max_shares_by_risk=0,
            max_shares_by_notional=None,
            capped_by_notional=False,
            is_valid=False,
            rejection_reason=(
                f"Stop price ({stop_price}) must be below entry price ({entry_price}) "
                "for a long position."
            ),
        )

    per_share_risk = stop_distance(entry_price, stop_price)
    if per_share_risk <= 0:
        return PositionSizingResult(
            shares=0,
            risk_budget=risk_budget,
            per_share_risk=per_share_risk,
            notional_value=0.0,
            max_shares_by_risk=0,
            max_shares_by_notional=None,
            capped_by_notional=False,
            is_valid=False,
            rejection_reason="Per-share risk must be greater than zero.",
        )

    max_shares_by_risk = floor(risk_budget / per_share_risk)
    max_shares_by_notional: int | None = None
    capped_by_notional = False
    shares = max_shares_by_risk

    if max_position_pct_equity is not None:
        notional_cap = current_equity * max_position_pct_equity
        max_shares_by_notional = floor(notional_cap / entry_price)
        if max_shares_by_notional < shares:
            shares = max_shares_by_notional
            capped_by_notional = True

    if shares <= 0:
        return PositionSizingResult(
            shares=0,
            risk_budget=risk_budget,
            per_share_risk=per_share_risk,
            notional_value=0.0,
            max_shares_by_risk=max_shares_by_risk,
            max_shares_by_notional=max_shares_by_notional,
            capped_by_notional=capped_by_notional,
            is_valid=False,
            rejection_reason="Calculated share size is zero after risk and notional constraints.",
        )

    notional_value = shares * entry_price
    return PositionSizingResult(
        shares=shares,
        risk_budget=risk_budget,
        per_share_risk=per_share_risk,
        notional_value=notional_value,
        max_shares_by_risk=max_shares_by_risk,
        max_shares_by_notional=max_shares_by_notional,
        capped_by_notional=capped_by_notional,
        is_valid=True,
    )


def _validate_sizing_inputs(
    *,
    current_equity: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
    max_position_pct_equity: float | None,
) -> str | None:
    for name, value in (
        ("current_equity", current_equity),
        ("risk_per_trade", risk_per_trade),
        ("entry_price", entry_price),
        ("stop_price", stop_price),
    ):
        if not isfinite(value):
            return f"{name} must be finite."

    if current_equity <= 0:
        return "current_equity must be greater than zero."
    if risk_per_trade <= 0:
        return "risk_per_trade must be greater than zero."
    if entry_price <= 0:
        return "entry_price must be greater than zero."
    if max_position_pct_equity is not None:
        if not isfinite(max_position_pct_equity):
            return "max_position_pct_equity must be finite when provided."
        if max_position_pct_equity <= 0:
            return "max_position_pct_equity must be greater than zero when provided."
    return None
