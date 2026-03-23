from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from bot.data.intraday_state_journal import IntradayTrajectoryFeatures
from bot.data.position_trajectory import PositionTrajectoryFeatures
from bot.features import (
    apply_intraday_trajectory_features,
    apply_position_trajectory_features,
    build_candidate_features,
    build_intraday_position_features,
    build_market_context,
    build_position_features,
)
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
    review_existing_long_position_intraday,
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


def test_assess_signal_candidate_blocks_entry_near_earnings() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
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
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_entry_block_days=3,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Upcoming earnings on 2024-01-10 are 3 trading days away; new entries are blocked within 3 trading days.",
    )


def test_assess_signal_candidate_rejects_weak_sector_regime() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
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
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
        require_sector_regime_for_entries=True,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Sector ETF XLK is below its trend filter; new entries require sector support.",
    )


def test_assess_signal_candidate_rejects_symbol_lagging_sector() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
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
        sector_name="Technology",
        sector_etf_symbol="XLK",
        relative_strength_vs_sector=-0.062,
        sector_relative_strength_window=20,
        sector_relative_strength_entry_reject_threshold=-0.05,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Stock is lagging XLK by 6.2% over 20 trading days; new entries require sector-relative strength.",
    )


def test_assess_signal_candidate_does_not_block_unknown_sector_regime() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
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
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=None,
        require_sector_regime_for_entries=True,
    )

    assert candidate.approved is True
    assert candidate.rejection_reasons == ()


def test_assess_signal_candidate_blocks_weak_market_breadth() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(
            as_of_date=signal.date,
            market_breadth_pct_above_200ma=0.37,
            market_breadth_pct_above_50ma=0.48,
            market_breadth_state="weak",
        ),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        market_breadth_entry_floor_200ma=0.40,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Market breadth is weak: 37% of the universe is above its 200-day moving average; new entries require at least 40%.",
    )


def test_assess_signal_candidate_fallback_preserves_market_breadth_context() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    market_context = build_market_context(
        as_of_date=signal.date,
        market_breadth_pct_above_200ma=0.37,
        market_breadth_pct_above_50ma=0.48,
        market_breadth_state="weak",
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        market_context=market_context,
        market_breadth_entry_floor_200ma=0.40,
    )

    assert candidate.approved is False
    assert candidate.features is not None
    assert candidate.features.market_context == market_context
    assert candidate.rejection_reasons == (
        "Market breadth is weak: 37% of the universe is above its 200-day moving average; new entries require at least 40%.",
    )


def test_assess_signal_candidate_rejects_separate_market_context_when_features_are_provided() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(
            as_of_date=signal.date,
            market_breadth_pct_above_200ma=0.58,
            market_breadth_pct_above_50ma=0.67,
            market_breadth_state="neutral",
        ),
    )
    market_context = build_market_context(
        as_of_date=signal.date,
        market_breadth_pct_above_200ma=0.37,
        market_breadth_pct_above_50ma=0.48,
        market_breadth_state="weak",
    )

    with pytest.raises(
        ValueError,
        match="Pass market_context via candidate_features, not as a separate argument\\.",
    ):
        assess_signal_candidate(
            signal,
            current_equity=100_000.0,
            base_risk_per_trade=0.01,
            constraints=constraints,
            candidate_features=candidate_features,
            market_context=market_context,
            market_breadth_entry_floor_200ma=0.40,
        )


def test_assess_signal_candidate_allows_acceptable_market_breadth() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(
            as_of_date=signal.date,
            market_breadth_pct_above_200ma=0.58,
            market_breadth_pct_above_50ma=0.67,
            market_breadth_state="neutral",
        ),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        market_breadth_entry_floor_200ma=0.40,
    )

    assert candidate.approved is True
    assert candidate.rejection_reasons == ()


def test_assess_signal_candidate_blocks_stressed_volatility_regime() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(
            as_of_date=signal.date,
            vix_close=33.4,
            vix_sma_short=31.0,
            vix_sma_long=26.2,
            volatility_regime_state="stressed",
            volatility_regime_risk_off=True,
        ),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        vix_entry_block_threshold=30.0,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Volatility regime is stressed: VIX is 33.4, above the 30.0 entry block threshold.",
    )


def test_assess_signal_candidate_reports_missing_vix_for_stressed_regime_block() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(
            as_of_date=signal.date,
            vix_sma_short=31.0,
            vix_sma_long=26.2,
            volatility_regime_state="stressed",
            volatility_regime_risk_off=True,
        ),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        vix_entry_block_threshold=30.0,
    )

    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Volatility regime is stressed; VIX data is unavailable but derived regime state indicates stressed conditions.",
    )


def test_assess_signal_candidate_does_not_block_missing_volatility_context() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(as_of_date=signal.date),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        vix_entry_block_threshold=30.0,
    )

    assert candidate.approved is True
    assert candidate.rejection_reasons == ()


def test_assess_signal_candidate_does_not_block_missing_market_breadth() -> None:
    signal = _signal(symbol="AAPL", entry_price=100.0, stop_price=95.0)
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    candidate_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        market_context=build_market_context(as_of_date=signal.date),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=candidate_features,
        market_breadth_entry_floor_200ma=0.40,
    )

    assert candidate.approved is True
    assert candidate.rejection_reasons == ()


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


def test_review_existing_long_position_marks_near_earnings_as_watch() -> None:
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
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_watch_days=7,
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    assert decision.metadata["earnings_days_away"] == 3
    assert decision.metadata["is_earnings_risk"] is True
    joined_rationale = " ".join(decision.rationale).lower()
    assert "upcoming earnings are scheduled on 2024-01-10" in joined_rationale
    assert "event risk" in joined_rationale


def test_review_existing_long_position_marks_sector_lag_as_watch() -> None:
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
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=True,
        relative_strength_vs_sector=-0.062,
        sector_relative_strength_window=20,
        sector_relative_strength_watch_threshold=-0.05,
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    assert decision.metadata["lagging_sector"] is True
    joined_rationale = " ".join(decision.rationale).lower()
    assert "lagging xlk by 6.2% over 20 trading days" in joined_rationale


def test_review_existing_long_position_raise_stop_notes_weak_sector_context() -> None:
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
        trailing_stop_candidate=101.0,
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
    )

    assert decision.suggested_action == "RAISE STOP"
    joined_rationale = " ".join(decision.rationale).lower()
    assert "sector etf xlk is below its trend filter" in joined_rationale
    assert "raising the stop improves risk control" in joined_rationale


def test_review_existing_long_position_reports_earnings_today_cleanly() -> None:
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
        earnings_date=date(2024, 1, 10),
        earnings_days_away=0,
        earnings_watch_days=7,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert "scheduled on 2024-01-10 (today)" in joined_rationale


def test_review_existing_long_position_escalates_multi_day_weakness_after_repeated_watchs() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )
    features = apply_position_trajectory_features(
        build_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            latest_close=97.0,
            current_stop=90.0,
            regime_passed=True,
            high_water_close=110.0,
            days_since_new_high=18,
            stale_high_watch_days=15,
            relative_strength_return_diff=-0.07,
            relative_strength_window=20,
        ),
        PositionTrajectoryFeatures(
            observation_count=4,
            days_in_position_state=4,
            consecutive_days_above_entry=0,
            consecutive_days_below_entry=3,
            consecutive_watch_closely_days=2,
            consecutive_hold_days=0,
            consecutive_stale_position_days=3,
            consecutive_weak_relative_strength_days=3,
            consecutive_weak_position_days=3,
            repeated_weak_position=True,
            persistent_underperformance=True,
            recovery_after_multi_day_weakness=False,
        ),
    )

    decision = review_existing_long_position(
        position,
        position_features=features,
        latest_close=97.0,
        regime_passed=True,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["consecutive_watch_closely_days"] == 2
    assert "watch closely for 2 consecutive review sessions" in joined_rationale
    assert "below entry for 3 consecutive review sessions" in joined_rationale
    assert "multi-day weakness increases exit pressure" in joined_rationale


def test_review_existing_long_position_recovery_after_multi_day_weakness_is_not_over_punished() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_position_trajectory_features(
        build_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            latest_close=103.0,
            current_stop=95.0,
            regime_passed=True,
            high_water_close=104.0,
            days_since_new_high=1,
            stale_high_watch_days=15,
            relative_strength_return_diff=0.01,
            relative_strength_window=20,
        ),
        PositionTrajectoryFeatures(
            observation_count=4,
            days_in_position_state=4,
            consecutive_days_above_entry=1,
            consecutive_days_below_entry=0,
            consecutive_watch_closely_days=0,
            consecutive_hold_days=0,
            consecutive_stale_position_days=0,
            consecutive_weak_relative_strength_days=0,
            consecutive_weak_position_days=0,
            repeated_weak_position=False,
            persistent_underperformance=False,
            recovery_after_multi_day_weakness=True,
        ),
    )

    decision = review_existing_long_position(
        position,
        position_features=features,
        latest_close=103.0,
        regime_passed=True,
    )

    assert decision.suggested_action == "HOLD"
    assert decision.metadata["recovery_after_multi_day_weakness"] is True
    assert "recovered after multiple weak review sessions" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_suppresses_recovery_note_when_earnings_promote_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_position_trajectory_features(
        build_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            latest_close=103.0,
            current_stop=95.0,
            regime_passed=True,
            high_water_close=104.0,
            days_since_new_high=1,
            stale_high_watch_days=15,
            relative_strength_return_diff=0.01,
            relative_strength_window=20,
            earnings_date=date(2024, 1, 10),
            earnings_days_away=3,
            earnings_watch_days=7,
        ),
        PositionTrajectoryFeatures(
            observation_count=4,
            days_in_position_state=4,
            consecutive_days_above_entry=1,
            consecutive_days_below_entry=0,
            consecutive_watch_closely_days=0,
            consecutive_hold_days=0,
            consecutive_stale_position_days=0,
            consecutive_weak_relative_strength_days=0,
            consecutive_weak_position_days=0,
            repeated_weak_position=False,
            persistent_underperformance=False,
            recovery_after_multi_day_weakness=True,
        ),
    )

    decision = review_existing_long_position(
        position,
        position_features=features,
        latest_close=103.0,
        regime_passed=True,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_watch_days=7,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert decision.suggested_action == "WATCH CLOSELY"
    assert "recovered after multiple weak review sessions" not in joined_rationale
    assert "upcoming earnings" in joined_rationale


def test_review_existing_long_position_suppresses_recovery_note_when_sector_promotes_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_position_trajectory_features(
        build_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            latest_close=103.0,
            current_stop=95.0,
            regime_passed=True,
            high_water_close=104.0,
            days_since_new_high=1,
            stale_high_watch_days=15,
            relative_strength_return_diff=0.01,
            relative_strength_window=20,
            sector_name="Technology",
            sector_etf_symbol="XLK",
            sector_regime_passed=False,
        ),
        PositionTrajectoryFeatures(
            observation_count=4,
            days_in_position_state=4,
            consecutive_days_above_entry=1,
            consecutive_days_below_entry=0,
            consecutive_watch_closely_days=0,
            consecutive_hold_days=0,
            consecutive_stale_position_days=0,
            consecutive_weak_relative_strength_days=0,
            consecutive_weak_position_days=0,
            repeated_weak_position=False,
            persistent_underperformance=False,
            recovery_after_multi_day_weakness=True,
        ),
    )

    decision = review_existing_long_position(
        position,
        position_features=features,
        latest_close=103.0,
        regime_passed=True,
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert decision.suggested_action == "WATCH CLOSELY"
    assert "recovered after multiple weak review sessions" not in joined_rationale
    assert "sector etf xlk is below its trend filter" in joined_rationale


def test_review_existing_long_position_intraday_exits_on_stop_breach() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=110.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=96.0,
        latest_low=95.0,
        session_vwap=108.0,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["stop_breached_intraday"] is True
    assert "traded through the current stop" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_marks_sector_lag_as_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=101.2,
        session_low=99.7,
        latest_close=100.9,
        latest_low=100.6,
        session_vwap=100.5,
        relative_strength_vs_sector=-0.061,
        sector_etf_symbol="XLK",
        sector_name="Technology",
        sector_relative_strength_window=20,
        sector_relative_strength_watch_threshold=-0.05,
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    assert decision.metadata["lagging_sector"] is True
    joined_rationale = " ".join(decision.rationale).lower()
    assert "lagging xlk by 6.1% over 20 trading days" in joined_rationale


def test_review_existing_long_position_intraday_exits_on_severe_session_high_giveback() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=110.0,
        session_high=130.0,
        session_low=108.0,
        latest_close=115.0,
        latest_low=114.0,
        session_vwap=120.0,
        session_high_giveback_exit_threshold=0.08,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["session_high_giveback_pct"] == pytest.approx((130.0 - 115.0) / 130.0)
    assert "session high" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_marks_fade_as_watch() -> None:
    # Position entered in a prior session at 90.0; today's intraday move is small (<3% off open)
    # so failed_intraday_strength stays False, but peak_unrealized is in profit and the session
    # is giving back enough to trigger intraday_momentum_fade -> WATCH CLOSELY.
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=90.0,
        current_stop=85.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=102.0,   # peak_intraday = 2% < early_strength_threshold=3% → failed=False
        session_low=93.0,
        latest_close=94.0,    # giveback = (102-94)/102 ≈ 7.8% > watch_threshold=7.5%
        latest_low=93.5,
        session_vwap=97.0,    # latest_close < vwap → below_vwap
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    assert decision.metadata["intraday_momentum_fade"] is True
    assert decision.metadata["failed_intraday_strength"] is False
    joined_rationale = " ".join(decision.rationale).lower()
    assert "fading intraday" in joined_rationale
    assert "session vwap" in joined_rationale


def test_review_existing_long_position_intraday_exits_on_failed_intraday_strength() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=120.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=108.0,
        session_low=96.0,
        latest_close=97.0,
        latest_low=96.5,
        session_vwap=101.0,
        session_high_giveback_exit_threshold=0.08,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["failed_intraday_strength"] is True
    joined_rationale = " ".join(decision.rationale).lower()
    assert "failed to hold" in joined_rationale
    assert "session vwap" in joined_rationale


def test_review_existing_long_position_intraday_keeps_healthy_position_on_hold() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=110.0,
        session_high=114.0,
        session_low=109.0,
        latest_close=113.0,
        latest_low=112.0,
        session_vwap=111.0,
    )

    assert decision.suggested_action == "HOLD"
    assert decision.metadata["intraday_momentum_fade"] is False
    assert decision.metadata["stop_breached_intraday"] is False


def test_review_existing_long_position_intraday_exits_on_stacked_weakness() -> None:
    # Three corroborating weak signals: below VWAP + giveback >= watch_threshold + underperforms benchmark.
    position = ExistingPosition(
        symbol="WDC",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=105.0,
        session_high=108.0,   # peak_unrealized = (108/100)-1 = 8% >= meaningful_profit=5%
        session_low=100.0,
        latest_close=101.7,   # giveback = (108-101.7)/108 ≈ 5.8% >= meaningful_profit=5%; < exit=10%
        latest_low=101.0,
        session_vwap=104.0,   # latest_close < vwap → below_vwap
        session_high_giveback_exit_threshold=0.10,
        # 5.8% is below the 10% exit and below the 7.5% watch threshold, but three
        # corroborating signals (below VWAP + giveback >= meaningful + weak benchmark) → EXIT
        intraday_relative_strength_diff=-0.03,  # underperforms benchmark
        intraday_relative_strength_watch_threshold=-0.02,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["stacked_intraday_weakness"] is True
    assert decision.metadata["weak_intraday_relative_strength"] is True
    assert "three corroborating" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_exits_on_high_profit_giveback() -> None:
    # Position in >15% unrealized profit gives back >7% from session high → exits at tighter threshold.
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=80.0,
        current_stop=75.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=112.0,   # peak_unrealized = (112/80)-1 = 40% >= high_profit_unrealized=15%
        session_low=97.0,
        latest_close=103.0,   # session_high_giveback = (112-103)/112 ≈ 8.0% >= high_profit_giveback=7%
        latest_low=102.0,
        session_vwap=107.0,
        session_high_giveback_exit_threshold=0.10,  # 8.0% < 10% → would be WATCH at normal threshold
        intraday_high_profit_unrealized_pct=0.15,
        intraday_high_profit_giveback_threshold=0.07,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["peak_unrealized_pct"] >= 0.15
    assert decision.metadata["session_high_giveback_pct"] >= 0.07
    assert "protecting profit" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_exits_failed_strength_at_watch_threshold() -> None:
    # failed_intraday_strength + giveback >= watch_threshold (not the higher exit threshold) → EXIT CANDIDATE.
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=120.0,
        current_stop=90.0,
    )

    decision = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=105.0,   # peak_intraday = 5% >= early_strength_threshold=3% → failed candidate
        session_low=96.0,
        latest_close=97.0,    # giveback = (105-97)/105 ≈ 7.6% > watch_threshold≈7.5%; < exit_threshold=10%
        latest_low=96.5,
        session_vwap=101.0,   # latest_close < vwap → confirms failed_intraday_strength
        session_high_giveback_exit_threshold=0.10,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["failed_intraday_strength"] is True
    # giveback is below the main exit threshold (10%) but above the watch threshold
    assert decision.metadata["session_high_giveback_pct"] < 0.10
    assert decision.metadata["session_high_giveback_pct"] >= decision.metadata["session_high_giveback_watch_threshold"]
    assert "failed to hold" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_escalates_persistent_weakness_after_repeated_watchs() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )
    features = apply_intraday_trajectory_features(
        build_intraday_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            current_stop=90.0,
            session_open=100.0,
            session_high=104.0,
            session_low=98.8,
            latest_close=99.0,
            latest_low=98.9,
            session_vwap=101.0,
            intraday_return_vs_benchmark=-0.031,
            early_strength_threshold=0.03,
            meaningful_profit_pct=0.05,
            session_high_giveback_watch_threshold=0.04,
            intraday_relative_strength_watch_threshold=-0.02,
            sector_relative_strength_watch_threshold=-0.05,
        ),
        IntradayTrajectoryFeatures(
            observation_count=4,
            consecutive_polls_below_vwap=3,
            consecutive_polls_weak_relative_strength=3,
            intraday_pressure_persistence_count=3,
            max_session_high_giveback_seen=(104.0 - 99.0) / 104.0,
            worsening_session_high_giveback=True,
            giveback_worsening_polls=3,
            giveback_worsening_from_pct=0.021,
            repeated_watch_closely_count=2,
            weakening_all_session=True,
            recovered_after_weakness=False,
        ),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=100.0,
        session_high=104.0,
        session_low=98.8,
        latest_close=99.0,
        latest_low=98.9,
        session_vwap=101.0,
        intraday_persistent_weakness_poll_threshold=3,
        intraday_repeated_watch_exit_threshold=2,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert decision.metadata["repeated_watch_closely_count"] == 2
    joined_rationale = " ".join(decision.rationale).lower()
    assert "watch closely on 2 consecutive prior polls" in joined_rationale
    assert "below session vwap for 3 consecutive polls" in joined_rationale
    assert "giveback from session high has worsened" in joined_rationale


def test_review_existing_long_position_intraday_recovery_after_weakness_does_not_over_punish() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_intraday_trajectory_features(
        build_intraday_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            current_stop=95.0,
            session_open=100.0,
            session_high=104.0,
            session_low=98.0,
            latest_close=103.2,
            latest_low=102.8,
            session_vwap=102.1,
            intraday_return_vs_benchmark=0.004,
            early_strength_threshold=0.03,
            meaningful_profit_pct=0.05,
            session_high_giveback_watch_threshold=0.04,
            intraday_relative_strength_watch_threshold=-0.02,
            sector_relative_strength_watch_threshold=-0.05,
        ),
        IntradayTrajectoryFeatures(
            observation_count=4,
            consecutive_polls_below_vwap=0,
            consecutive_polls_weak_relative_strength=0,
            intraday_pressure_persistence_count=0,
            max_session_high_giveback_seen=0.041,
            worsening_session_high_giveback=False,
            giveback_worsening_polls=0,
            giveback_worsening_from_pct=None,
            repeated_watch_closely_count=1,
            weakening_all_session=False,
            recovered_after_weakness=True,
        ),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=100.0,
        session_high=104.0,
        session_low=98.0,
        latest_close=103.2,
        latest_low=102.8,
        session_vwap=102.1,
    )

    assert decision.suggested_action == "HOLD"
    assert decision.metadata["recovered_after_weakness"] is True
    assert "recovered after earlier intraday weakness" in " ".join(decision.rationale).lower()


def test_review_existing_long_position_intraday_suppresses_recovery_note_when_earnings_promote_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_intraday_trajectory_features(
        build_intraday_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            current_stop=95.0,
            session_open=100.0,
            session_high=104.0,
            session_low=98.0,
            latest_close=103.2,
            latest_low=102.8,
            session_vwap=102.1,
            intraday_return_vs_benchmark=0.004,
            earnings_date=date(2024, 1, 10),
            earnings_days_away=3,
            earnings_watch_days=7,
            early_strength_threshold=0.03,
            meaningful_profit_pct=0.05,
            session_high_giveback_watch_threshold=0.04,
            intraday_relative_strength_watch_threshold=-0.02,
            sector_relative_strength_watch_threshold=-0.05,
        ),
        IntradayTrajectoryFeatures(
            observation_count=4,
            consecutive_polls_below_vwap=0,
            consecutive_polls_weak_relative_strength=0,
            intraday_pressure_persistence_count=0,
            max_session_high_giveback_seen=0.041,
            worsening_session_high_giveback=False,
            giveback_worsening_polls=0,
            giveback_worsening_from_pct=None,
            repeated_watch_closely_count=1,
            weakening_all_session=False,
            recovered_after_weakness=True,
        ),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=100.0,
        session_high=104.0,
        session_low=98.0,
        latest_close=103.2,
        latest_low=102.8,
        session_vwap=102.1,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_watch_days=7,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert decision.suggested_action == "WATCH CLOSELY"
    assert "recovered after earlier intraday weakness" not in joined_rationale
    assert "upcoming earnings" in joined_rationale


def test_review_existing_long_position_intraday_suppresses_recovery_note_when_sector_promotes_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_intraday_trajectory_features(
        build_intraday_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            current_stop=95.0,
            session_open=100.0,
            session_high=104.0,
            session_low=98.0,
            latest_close=103.2,
            latest_low=102.8,
            session_vwap=102.1,
            intraday_return_vs_benchmark=0.004,
            sector_name="Technology",
            sector_etf_symbol="XLK",
            sector_regime_passed=False,
            early_strength_threshold=0.03,
            meaningful_profit_pct=0.05,
            session_high_giveback_watch_threshold=0.04,
            intraday_relative_strength_watch_threshold=-0.02,
            sector_relative_strength_watch_threshold=-0.05,
        ),
        IntradayTrajectoryFeatures(
            observation_count=4,
            consecutive_polls_below_vwap=0,
            consecutive_polls_weak_relative_strength=0,
            intraday_pressure_persistence_count=0,
            max_session_high_giveback_seen=0.041,
            worsening_session_high_giveback=False,
            giveback_worsening_polls=0,
            giveback_worsening_from_pct=None,
            repeated_watch_closely_count=1,
            weakening_all_session=False,
            recovered_after_weakness=True,
        ),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=100.0,
        session_high=104.0,
        session_low=98.0,
        latest_close=103.2,
        latest_low=102.8,
        session_vwap=102.1,
        sector_name="Technology",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
    )

    joined_rationale = " ".join(decision.rationale).lower()
    assert decision.suggested_action == "WATCH CLOSELY"
    assert "recovered after earlier intraday weakness" not in joined_rationale
    assert "sector etf xlk is below its trend filter" in joined_rationale


def test_review_existing_long_position_intraday_weakening_all_session_alone_triggers_watch() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = apply_intraday_trajectory_features(
        build_intraday_position_features(
            symbol="AAPL",
            as_of_date=date(2024, 1, 5),
            average_entry_price=100.0,
            current_stop=95.0,
            session_open=100.0,
            session_high=101.2,
            session_low=99.6,
            latest_close=100.8,
            latest_low=100.5,
            session_vwap=100.7,
            intraday_return_vs_benchmark=0.004,
            early_strength_threshold=0.03,
            meaningful_profit_pct=0.05,
            session_high_giveback_watch_threshold=0.04,
            intraday_relative_strength_watch_threshold=-0.02,
            sector_relative_strength_watch_threshold=-0.05,
        ),
        IntradayTrajectoryFeatures(
            observation_count=3,
            consecutive_polls_below_vwap=0,
            consecutive_polls_weak_relative_strength=0,
            intraday_pressure_persistence_count=0,
            max_session_high_giveback_seen=0.018,
            worsening_session_high_giveback=False,
            giveback_worsening_polls=0,
            giveback_worsening_from_pct=None,
            repeated_watch_closely_count=0,
            weakening_all_session=True,
            recovered_after_weakness=False,
        ),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=100.0,
        session_high=101.2,
        session_low=99.6,
        latest_close=100.8,
        latest_low=100.5,
        session_vwap=100.7,
    )

    assert decision.suggested_action == "WATCH CLOSELY"
    assert decision.metadata["weakening_all_session"] is True
    assert (
        "intraday pressure has persisted throughout all observed polls this session."
        in " ".join(decision.rationale).lower()
    )


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
