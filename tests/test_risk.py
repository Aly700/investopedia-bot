from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioConstraints,
    PortfolioInputError,
    PORTFOLIO_REVIEW_ACTIONS,
    apply_drawdown_risk_adjustment,
    assess_signal_candidate,
    evaluate_portfolio_rules,
    initialize_portfolio_snapshot,
    load_existing_positions,
    review_existing_long_position,
    remove_existing_position_snapshot,
    update_existing_position_stop_snapshot,
    upsert_existing_position_snapshot,
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


def test_position_sizing_rejects_stop_equal_to_entry() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.01,
        entry_price=50.0,
        stop_price=50.0,
    )

    assert result.is_valid is False
    assert result.shares == 0
    assert result.per_share_risk == pytest.approx(0.0)
    assert "50" in (result.rejection_reason or "")


def test_position_sizing_rejects_stop_above_entry() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.01,
        entry_price=50.0,
        stop_price=55.0,
    )

    assert result.is_valid is False
    assert result.shares == 0
    assert result.per_share_risk == pytest.approx(0.0)
    assert "55" in (result.rejection_reason or "")
    assert "50" in (result.rejection_reason or "")


def test_position_sizing_accepts_stop_below_entry() -> None:
    result = size_position(
        current_equity=100_000.0,
        risk_per_trade=0.01,
        entry_price=50.0,
        stop_price=45.0,
    )

    assert result.is_valid is True
    assert result.per_share_risk == pytest.approx(5.0)
    assert result.shares == 200


def test_stop_distance_returns_positive_for_valid_long_stop() -> None:
    assert stop_distance(100.0, 95.0) == pytest.approx(5.0)


def test_stop_distance_returns_zero_when_stop_equals_entry() -> None:
    assert stop_distance(100.0, 100.0) == pytest.approx(0.0)


def test_stop_distance_returns_negative_when_stop_above_entry() -> None:
    # Negative result is the signal to callers that the stop placement is invalid.
    assert stop_distance(100.0, 105.0) == pytest.approx(-5.0)


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
        constraints=constraints,
    )

    assert result.approved is False
    assert result.reasons == ("No averaging down is allowed for existing long positions.",)


def test_portfolio_rules_block_duplicate_existing_holding() -> None:
    signal = _signal(symbol="AAA", entry_price=105.0, stop_price=100.0)
    positions = [ExistingPosition(symbol="AAA", shares=100, average_entry_price=100.0)]
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )

    result = evaluate_portfolio_rules(
        signal,
        current_positions=positions,
        constraints=constraints,
    )

    assert result.approved is False
    assert result.reasons == (
        "Existing long position already open for symbol; duplicate entries are not allowed.",
    )


def test_drawdown_risk_adjustment_reduces_risk_budget() -> None:
    adjusted_risk = apply_drawdown_risk_adjustment(
        0.01,
        current_drawdown=0.20,
        threshold=0.15,
        reduction_factor=0.5,
    )

    assert adjusted_risk == pytest.approx(0.005)


def test_drawdown_risk_adjustment_triggers_at_exact_threshold() -> None:
    # The condition is current_drawdown >= threshold, so equality must trigger reduction.
    adjusted_risk = apply_drawdown_risk_adjustment(
        0.01,
        current_drawdown=0.15,
        threshold=0.15,
        reduction_factor=0.5,
    )

    assert adjusted_risk == pytest.approx(0.005)


def test_drawdown_risk_adjustment_does_not_trigger_just_below_threshold() -> None:
    # One basis-point below threshold must leave risk unchanged.
    adjusted_risk = apply_drawdown_risk_adjustment(
        0.01,
        current_drawdown=0.1499,
        threshold=0.15,
        reduction_factor=0.5,
    )

    assert adjusted_risk == pytest.approx(0.01)


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


def test_assess_signal_candidate_carries_existing_position_context() -> None:
    signal = _signal(symbol="AAA", entry_price=105.0, stop_price=100.0)
    existing_position = ExistingPosition(
        symbol="AAA",
        shares=25,
        average_entry_price=100.0,
        current_stop=95.0,
        preset_name="standard_breakout",
        source="portfolio.csv",
    )
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        current_positions=(existing_position,),
    )

    assert candidate.approved is False
    assert candidate.existing_position == existing_position
    assert candidate.rejection_reasons == (
        "Existing long position already open for symbol; duplicate entries are not allowed.",
    )


def test_load_existing_positions_supports_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text(
        (
            "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
            "AAPL,10,150,145,standard_breakout,manual,{\"note\":\"swing\"}\n"
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "portfolio.json"
    json_path.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "MSFT",
                        "quantity": 5,
                        "average_entry_price": 300.0,
                        "current_stop": 285.0,
                        "preset_name": "confirmed_breakout",
                        "source": "manual",
                        "metadata": {"note": "core"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    csv_positions = load_existing_positions(csv_path)
    json_positions = load_existing_positions(json_path)

    assert csv_positions[0].symbol == "AAPL"
    assert csv_positions[0].shares == 10
    assert csv_positions[0].current_stop == pytest.approx(145.0)
    assert csv_positions[0].metadata == {"note": "swing"}
    assert json_positions[0].symbol == "MSFT"
    assert json_positions[0].preset_name == "confirmed_breakout"
    assert json_positions[0].metadata == {"note": "core"}


def test_load_existing_positions_rejects_duplicate_symbols(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text(
        "symbol,quantity,average_entry_price\nAAPL,10,150\nAAPL,5,155\n",
        encoding="utf-8",
    )

    with pytest.raises(PortfolioInputError, match="duplicate symbols"):
        load_existing_positions(csv_path)


def test_initialize_portfolio_snapshot_creates_empty_csv_and_json(tmp_path: Path) -> None:
    csv_path = initialize_portfolio_snapshot(tmp_path / "portfolio.csv")
    json_path = initialize_portfolio_snapshot(tmp_path / "portfolio.json")

    assert load_existing_positions(csv_path) == []
    assert load_existing_positions(json_path) == []
    assert "symbol,quantity,average_entry_price" in csv_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"positions": []}


def test_upsert_existing_position_snapshot_appends_and_updates_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    initialize_portfolio_snapshot(csv_path)

    upsert_existing_position_snapshot(
        csv_path,
        ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=150.0,
            current_stop=145.0,
            preset_name="standard_breakout",
            source="manual",
            metadata={"note": "swing"},
        ),
    )
    upsert_existing_position_snapshot(
        csv_path,
        ExistingPosition(
            symbol="AAPL",
            shares=12,
            average_entry_price=151.5,
            current_stop=146.0,
            preset_name="confirmed_breakout",
            source="manual",
            metadata={"note": "updated"},
        ),
    )
    positions = load_existing_positions(csv_path)

    assert len(positions) == 1
    assert positions[0].shares == 12
    assert positions[0].average_entry_price == pytest.approx(151.5)
    assert positions[0].preset_name == "confirmed_breakout"
    assert positions[0].metadata == {"note": "updated"}


def test_upsert_existing_position_snapshot_appends_json(tmp_path: Path) -> None:
    json_path = tmp_path / "portfolio.json"
    initialize_portfolio_snapshot(json_path)

    upsert_existing_position_snapshot(
        json_path,
        ExistingPosition(symbol="MSFT", shares=5, average_entry_price=300.0),
    )
    upsert_existing_position_snapshot(
        json_path,
        ExistingPosition(symbol="NVDA", shares=3, average_entry_price=900.0, current_stop=850.0),
    )
    positions = load_existing_positions(json_path)

    assert [position.symbol for position in positions] == ["MSFT", "NVDA"]
    assert positions[1].current_stop == pytest.approx(850.0)


def test_update_existing_position_stop_snapshot_updates_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    initialize_portfolio_snapshot(csv_path)
    upsert_existing_position_snapshot(
        csv_path,
        ExistingPosition(symbol="AAPL", shares=10, average_entry_price=150.0, current_stop=145.0),
    )

    written_path = update_existing_position_stop_snapshot(csv_path, "AAPL", 147.5)
    positions = load_existing_positions(written_path)

    assert positions[0].symbol == "AAPL"
    assert positions[0].current_stop == pytest.approx(147.5)


def test_update_existing_position_stop_snapshot_updates_json(tmp_path: Path) -> None:
    json_path = tmp_path / "portfolio.json"
    initialize_portfolio_snapshot(json_path)
    upsert_existing_position_snapshot(
        json_path,
        ExistingPosition(symbol="MSFT", shares=5, average_entry_price=300.0, current_stop=290.0),
    )

    written_path = update_existing_position_stop_snapshot(json_path, "MSFT", 295.0)
    positions = load_existing_positions(written_path)

    assert positions[0].symbol == "MSFT"
    assert positions[0].current_stop == pytest.approx(295.0)


def test_remove_existing_position_snapshot_removes_csv_position(tmp_path: Path) -> None:
    csv_path = tmp_path / "portfolio.csv"
    initialize_portfolio_snapshot(csv_path)
    upsert_existing_position_snapshot(
        csv_path,
        ExistingPosition(symbol="AAPL", shares=10, average_entry_price=150.0),
    )
    upsert_existing_position_snapshot(
        csv_path,
        ExistingPosition(symbol="MSFT", shares=5, average_entry_price=300.0),
    )

    written_path = remove_existing_position_snapshot(csv_path, "AAPL")
    positions = load_existing_positions(written_path)

    assert [position.symbol for position in positions] == ["MSFT"]


def test_remove_existing_position_snapshot_removes_json_position(tmp_path: Path) -> None:
    json_path = tmp_path / "portfolio.json"
    initialize_portfolio_snapshot(json_path)
    upsert_existing_position_snapshot(
        json_path,
        ExistingPosition(symbol="AAPL", shares=10, average_entry_price=150.0),
    )
    upsert_existing_position_snapshot(
        json_path,
        ExistingPosition(symbol="NVDA", shares=3, average_entry_price=900.0),
    )

    written_path = remove_existing_position_snapshot(json_path, "NVDA")
    positions = load_existing_positions(written_path)

    assert [position.symbol for position in positions] == ["AAPL"]


def test_update_existing_position_stop_snapshot_errors_for_missing_symbol(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "portfolio.csv"
    initialize_portfolio_snapshot(csv_path)

    with pytest.raises(PortfolioInputError, match="does not exist"):
        update_existing_position_stop_snapshot(csv_path, "AAPL", 145.0)


def test_remove_existing_position_snapshot_errors_for_missing_symbol(tmp_path: Path) -> None:
    json_path = tmp_path / "portfolio.json"
    initialize_portfolio_snapshot(json_path)

    with pytest.raises(PortfolioInputError, match="does not exist"):
        remove_existing_position_snapshot(json_path, "AAPL")


def test_review_existing_long_position_with_one_holding_returns_hold() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=110.0,
        regime_passed=True,
        trailing_stop_candidate=88.0,
    )

    assert decision.suggested_action == "HOLD"
    assert decision.suggested_action in PORTFOLIO_REVIEW_ACTIONS
    assert decision.unrealized_pl_pct == pytest.approx(0.10)
    assert decision.distance_to_stop_pct == pytest.approx((110.0 - 90.0) / 110.0)
    assert decision.above_entry is True
    assert decision.suggested_stop is None


def test_review_existing_long_position_suggests_raise_stop() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=112.0,
        regime_passed=True,
        trailing_stop_candidate=101.0,
    )

    assert decision.suggested_action == "RAISE STOP"
    assert decision.suggested_stop == pytest.approx(101.0)


def test_review_existing_long_position_does_not_lower_stop() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=98.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=110.0,
        regime_passed=True,
        trailing_stop_candidate=96.0,
    )

    assert decision.suggested_stop is None
    assert decision.suggested_action == "HOLD"


def test_review_existing_long_position_suggests_exit_candidate() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=96.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=95.0,
        regime_passed=False,
        trailing_stop_candidate=97.0,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.suggested_stop is None
    assert decision.trailing_stop_candidate is None
    assert "current stop" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_exits_on_large_profit_giveback() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=102.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=115.0,
        regime_passed=True,
        trailing_stop_candidate=106.0,
        high_water_close=130.0,
        profit_giveback_threshold=0.08,
        profit_giveback_min_unrealized_pct=0.10,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.suggested_stop is None
    assert decision.trailing_stop_candidate is None
    assert decision.metadata["high_water_close"] == pytest.approx(130.0)
    assert decision.metadata["giveback_pct"] == pytest.approx((130.0 - 115.0) / 130.0)
    assert "given back" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_does_not_exit_on_small_profit_giveback() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=116.0,
        regime_passed=True,
        trailing_stop_candidate=None,
        high_water_close=120.0,
        profit_giveback_threshold=0.08,
        profit_giveback_min_unrealized_pct=0.10,
    )

    assert decision.suggested_action == "HOLD"
    assert decision.metadata["giveback_pct"] == pytest.approx((120.0 - 116.0) / 120.0)


def test_review_existing_long_position_exits_on_failed_breakout_reference() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=94.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=101.0,
        regime_passed=True,
        trailing_stop_candidate=None,
        breakout_failure_reference=103.0,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["failed_breakout_detected"] is True
    assert "breakout reference" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_trailing_stop_above_latest_close_does_not_raise_stop() -> None:
    # Regression: trailing_stop_candidate above latest_close must never produce RAISE STOP.
    # Mirrors the live NET case: close=221.27, bad candidate=228.574.
    position = ExistingPosition(
        symbol="NET",
        shares=10,
        average_entry_price=200.0,
        current_stop=210.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=221.27,
        regime_passed=True,
        trailing_stop_candidate=228.574,
    )

    assert decision.suggested_action != "RAISE STOP"
    assert decision.suggested_stop is None


def test_review_existing_long_position_valid_stop_below_latest_close_still_raises_stop() -> None:
    # Regression: a trailing candidate that is above current_stop AND below latest_close
    # must still produce RAISE STOP — the fix must not suppress valid stops.
    position = ExistingPosition(
        symbol="MSFT",
        shares=5,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=120.0,
        regime_passed=True,
        trailing_stop_candidate=105.0,
    )

    assert decision.suggested_action == "RAISE STOP"
    assert decision.suggested_stop == pytest.approx(105.0)


def test_review_existing_long_position_raise_stop_suggested_stop_always_below_latest_close() -> None:
    # Invariant: whenever action is RAISE STOP, suggested_stop must be < latest_close.
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=115.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
    )

    assert decision.suggested_action == "RAISE STOP"
    assert decision.suggested_stop is not None
    assert decision.suggested_stop < decision.latest_close


def test_review_existing_long_position_raise_stop_still_wins_over_stale_and_weak_relative_strength() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=120.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=123.0,
        days_since_new_high=18,
        stale_high_watch_days=15,
        relative_strength_return_diff=-0.07,
        relative_strength_window=20,
    )

    assert decision.suggested_action == "RAISE STOP"
    assert decision.suggested_stop == pytest.approx(108.0)
    assert decision.metadata["stale_position"] is True
    assert decision.metadata["weak_relative_strength"] is True


def test_review_existing_long_position_marks_stale_weak_relative_strength_as_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=112.0,
        regime_passed=True,
        trailing_stop_candidate=None,
        high_water_close=118.0,
        days_since_new_high=16,
        stale_high_watch_days=15,
        relative_strength_return_diff=-0.06,
        relative_strength_window=20,
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    joined_rationale = " ".join(decision.rationale).lower()
    assert "trading days without a new closing high" in joined_rationale
    assert "relative strength" in joined_rationale


def test_review_existing_long_position_clears_stop_for_weak_regime_exit_candidate() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=85.0,
    )

    decision = review_existing_long_position(
        position,
        latest_close=95.0,
        regime_passed=False,
        trailing_stop_candidate=90.0,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.suggested_stop is None
    assert decision.trailing_stop_candidate is None
    assert "weak" in " ".join(decision.rationale).lower()


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
