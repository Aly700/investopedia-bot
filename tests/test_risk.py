from __future__ import annotations

from datetime import date

import pytest

from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioConstraints,
    apply_drawdown_risk_adjustment,
    assess_signal_candidate,
    evaluate_portfolio_rules,
)
from bot.risk.position_sizing import size_position
from bot.risk.stops import atr_stop_distance, initial_stop_price, stop_distance, trailing_stop_reference
from bot.strategy.signal_models import StrategySignal


def test_stop_helpers_use_atr_conventions() -> None:
    assert atr_stop_distance(2.0, 2.5) == pytest.approx(5.0)
    assert initial_stop_price(100.0, 2.0, 2.5) == pytest.approx(95.0)
    assert trailing_stop_reference(110.0, 2.0, 2.5) == pytest.approx(105.0)
    assert stop_distance(100.0, 95.0) == pytest.approx(5.0)


def test_position_sizing_math_returns_expected_share_count() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.01,
        entry_price=50.0,
        stop_price=45.0,
    )

    assert result.is_valid is True
    assert result.risk_budget == pytest.approx(1_000.0)
    assert result.per_share_risk == pytest.approx(5.0)
    assert result.shares == 200
    assert result.notional_value == pytest.approx(10_000.0)


def test_position_sizing_rejects_zero_stop_distance() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.01,
        entry_price=50.0,
        stop_price=50.0,
    )

    assert result.is_valid is False
    assert result.shares == 0
    assert result.rejection_reason == "Per-share risk must be greater than zero."


def test_position_sizing_enforces_max_notional_cap() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.05,
        entry_price=100.0,
        stop_price=90.0,
        max_position_pct_equity=0.25,
    )

    assert result.is_valid is True
    assert result.max_shares_by_risk == 500
    assert result.max_shares_by_notional == 250
    assert result.shares == 250
    assert result.capped_by_notional is True
    assert result.notional_value == pytest.approx(25_000.0)


def test_portfolio_rules_enforce_max_positions() -> None:
    signal = _signal(symbol="CCC", entry_price=100.0, stop_price=95.0)
    positions = [
        ExistingPosition(symbol="AAA", shares=100, average_entry_price=50.0),
        ExistingPosition(symbol="BBB", shares=100, average_entry_price=60.0),
    ]
    constraints = PortfolioConstraints(
        max_concurrent_positions=2,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )

    result = evaluate_portfolio_rules(
        signal,
        current_positions=positions,
        current_equity=100_000.0,
        proposed_notional=10_000.0,
        constraints=constraints,
    )

    assert result.approved is False
    assert result.reasons == ("Max concurrent positions reached.",)


def test_portfolio_rules_block_averaging_down() -> None:
    signal = _signal(symbol="AAA", entry_price=95.0, stop_price=90.0)
    positions = [ExistingPosition(symbol="AAA", shares=100, average_entry_price=100.0)]
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )

    result = evaluate_portfolio_rules(
        signal,
        current_positions=positions,
        current_equity=100_000.0,
        proposed_notional=9_500.0,
        constraints=constraints,
    )

    assert result.approved is False
    assert result.reasons == ("No averaging down is allowed for existing long positions.",)


def test_drawdown_risk_adjustment_reduces_risk_budget() -> None:
    adjusted_risk = apply_drawdown_risk_adjustment(
        0.01,
        current_drawdown=0.20,
        threshold=0.15,
        reduction_factor=0.5,
    )

    assert adjusted_risk == pytest.approx(0.005)


def test_assess_signal_candidate_combines_sizing_and_rules() -> None:
    signal = _signal(symbol="AAA", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
        drawdown_risk_reduction_threshold=0.15,
        drawdown_risk_reduction_factor=0.5,
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        current_positions=(),
        current_drawdown=0.20,
    )

    assert candidate.approved is True
    assert candidate.adjusted_risk_per_trade == pytest.approx(0.005)
    assert candidate.sizing.shares == 100
    assert candidate.sizing.risk_budget == pytest.approx(500.0)


def _signal(*, symbol: str, entry_price: float, stop_price: float | None) -> StrategySignal:
    return StrategySignal(
        strategy_name="breakout_momentum",
        symbol=symbol,
        date=date(2024, 1, 4),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=entry_price,
        stop_hint=stop_price,
        metadata={},
    )
