from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bot.data.earnings import EarningsRiskContext
from bot.data.intraday_state_journal import IntradayTrajectoryFeatures
from bot.data.sector_context import SectorFeatureContext
from bot.features import (
    apply_intraday_trajectory_features,
    build_candidate_features,
    build_intraday_position_features,
    build_market_context,
    build_position_features,
    compute_market_breadth_from_frames,
)
from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioConstraints,
    assess_signal_candidate,
    review_existing_long_position,
    review_existing_long_position_intraday,
)
from bot.strategy.signal_models import StrategySignal


def _market_breadth_frame(
    symbol: str,
    *,
    end: str,
    periods: int,
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(end=end, periods=periods, freq="D"),
            "close": closes,
            "symbol": [symbol] * periods,
        }
    )


def test_build_candidate_features_captures_current_contexts() -> None:
    signal = StrategySignal(
        strategy_name="breakout_momentum:standard_breakout",
        symbol="AAPL",
        date=date(2024, 1, 5),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=101.0,
        stop_hint=96.0,
        metadata={
            "preset_name": "standard_breakout",
            "parameter_id": "standard",
            "prior_high": 100.0,
            "relative_volume": 1.9,
            "regime_passed": True,
            "setup_quality_score": 0.8,
            "setup_persistence_days": 3,
            "days_approved": 2,
            "repeated_high_quality_signal": True,
        },
    )

    features = build_candidate_features(
        signal,
        as_of_date=date(2024, 1, 5),
        earnings_context=EarningsRiskContext(
            symbol="AAPL",
            earnings_date=date(2024, 1, 10),
            earnings_days_away=3,
            is_earnings_risk=True,
        ),
        sector_context=SectorFeatureContext(
            symbol="AAPL",
            sector_name="Technology",
            industry_name="Consumer electronics",
            sector_etf_symbol="XLK",
            sector_regime_passed=True,
            sector_return=0.04,
            symbol_return=0.06,
            relative_strength_vs_sector=0.02,
            relative_strength_window=20,
            sector_trend_state="supportive",
            mapping_source="test",
        ),
        market_context=build_market_context(
            as_of_date=date(2024, 1, 5),
            benchmark_symbol="SPY",
            regime_filter_mode="either",
            regime_passed=True,
            market_breadth_pct_above_200ma=0.58,
            market_breadth_pct_above_50ma=0.67,
            market_breadth_state="neutral",
            vix_close=18.4,
            vix_sma_short=17.9,
            vix_sma_long=19.2,
            volatility_regime_state="calm",
            volatility_regime_risk_off=False,
        ),
    )

    assert features.symbol == "AAPL"
    assert features.breakout_strength == pytest.approx(0.01)
    assert features.relative_volume == pytest.approx(1.9)
    assert features.sector_etf_symbol == "XLK"
    assert features.earnings_days_away == 3
    assert features.setup_quality_score == pytest.approx(0.8)
    assert features.repeated_high_quality_signal is True
    assert features.market_context is not None
    assert features.market_context.benchmark_symbol == "SPY"
    assert features.market_context.market_breadth_pct_above_200ma == pytest.approx(0.58)
    assert features.market_context.vix_close == pytest.approx(18.4)
    assert features.market_context.volatility_regime_state == "calm"
    assert features.market_context.volatility_regime_risk_off is False


def test_compute_market_breadth_from_frames_is_deterministic() -> None:
    symbol_frames = {
        "AAA": _market_breadth_frame(
            "AAA",
            end="2023-08-08",
            periods=220,
            closes=[100.0] * 219 + [120.0],
        ),
        "BBB": _market_breadth_frame(
            "BBB",
            end="2023-08-08",
            periods=220,
            closes=[100.0] * 220,
        ),
        "CCC": _market_breadth_frame(
            "CCC",
            end="2023-08-08",
            periods=220,
            closes=[120.0] * 219 + [80.0],
        ),
    }

    breadth = compute_market_breadth_from_frames(
        symbol_frames,
        as_of_date=date(2023, 8, 8),
        entry_floor_200ma=0.40,
    )

    assert breadth["market_breadth_pct_above_200ma"] == pytest.approx(1 / 3)
    assert breadth["market_breadth_pct_above_50ma"] == pytest.approx(1 / 3)
    assert breadth["market_breadth_state"] == "weak"


def test_compute_market_breadth_from_frames_returns_none_for_empty_frames() -> None:
    breadth = compute_market_breadth_from_frames(
        {
            "AAA": pd.DataFrame(columns=["date", "close", "symbol"]),
            "BBB": pd.DataFrame(columns=["date", "close", "symbol"]),
        },
        as_of_date=date(2024, 1, 5),
        entry_floor_200ma=0.40,
    )

    assert breadth == {
        "market_breadth_pct_above_200ma": None,
        "market_breadth_pct_above_50ma": None,
        "market_breadth_state": None,
    }


def test_compute_market_breadth_from_frames_returns_none_when_history_is_insufficient() -> None:
    breadth = compute_market_breadth_from_frames(
        {
            "AAA": _market_breadth_frame(
                "AAA",
                end="2024-01-05",
                periods=49,
                closes=[100.0] * 49,
            ),
            "BBB": _market_breadth_frame(
                "BBB",
                end="2024-01-05",
                periods=49,
                closes=[101.0] * 49,
            ),
        },
        as_of_date=date(2024, 1, 5),
        entry_floor_200ma=0.40,
    )

    assert breadth == {
        "market_breadth_pct_above_200ma": None,
        "market_breadth_pct_above_50ma": None,
        "market_breadth_state": None,
    }


def test_compute_market_breadth_from_frames_excludes_stale_symbols_from_denominator() -> None:
    breadth = compute_market_breadth_from_frames(
        {
            "AAA": _market_breadth_frame(
                "AAA",
                end="2023-08-08",
                periods=220,
                closes=[100.0] * 219 + [120.0],
            ),
            "BBB": _market_breadth_frame(
                "BBB",
                end="2023-07-31",
                periods=220,
                closes=[120.0] * 219 + [80.0],
            ),
        },
        as_of_date=date(2023, 8, 8),
        entry_floor_200ma=0.40,
    )

    assert breadth["market_breadth_pct_above_200ma"] == pytest.approx(1.0)
    assert breadth["market_breadth_pct_above_50ma"] == pytest.approx(1.0)
    assert breadth["market_breadth_state"] == "strong"


def test_compute_market_breadth_from_frames_includes_symbols_at_stale_boundary() -> None:
    breadth = compute_market_breadth_from_frames(
        {
            "AAA": _market_breadth_frame(
                "AAA",
                end="2023-08-08",
                periods=220,
                closes=[100.0] * 219 + [120.0],
            ),
            "BBB": _market_breadth_frame(
                "BBB",
                end="2023-08-01",
                periods=220,
                closes=[120.0] * 219 + [80.0],
            ),
        },
        as_of_date=date(2023, 8, 8),
        entry_floor_200ma=0.40,
    )

    assert breadth["market_breadth_pct_above_200ma"] == pytest.approx(0.5)
    assert breadth["market_breadth_pct_above_50ma"] == pytest.approx(0.5)
    assert breadth["market_breadth_state"] == "neutral"


def test_compute_market_breadth_from_frames_omits_state_when_entry_floor_is_disabled() -> None:
    breadth = compute_market_breadth_from_frames(
        {
            "AAA": _market_breadth_frame(
                "AAA",
                end="2023-08-08",
                periods=220,
                closes=[100.0] * 219 + [120.0],
            ),
            "BBB": _market_breadth_frame(
                "BBB",
                end="2023-08-08",
                periods=220,
                closes=[100.0] * 220,
            ),
            "CCC": _market_breadth_frame(
                "CCC",
                end="2023-08-08",
                periods=220,
                closes=[120.0] * 219 + [80.0],
            ),
        },
        as_of_date=date(2023, 8, 8),
        entry_floor_200ma=None,
    )

    assert breadth["market_breadth_pct_above_200ma"] == pytest.approx(1 / 3)
    assert breadth["market_breadth_pct_above_50ma"] == pytest.approx(1 / 3)
    assert breadth["market_breadth_state"] is None


def test_assess_signal_candidate_consumes_candidate_features() -> None:
    signal = StrategySignal(
        strategy_name="breakout_momentum",
        symbol="AAPL",
        date=date(2024, 1, 5),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=101.0,
        stop_hint=96.0,
        metadata={"prior_high": 100.0, "relative_volume": 2.0},
    )
    features = build_candidate_features(
        signal,
        as_of_date=date(2024, 1, 5),
        earnings_context=EarningsRiskContext(
            symbol="AAPL",
            earnings_date=date(2024, 1, 9),
            earnings_days_away=2,
            is_earnings_risk=True,
        ),
        sector_context=None,
        market_context=build_market_context(as_of_date=date(2024, 1, 5)),
    )

    candidate = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=PortfolioConstraints(
            max_concurrent_positions=10,
            max_position_pct_equity=0.25,
            no_averaging_down=True,
        ),
        candidate_features=features,
        earnings_entry_block_days=3,
    )

    assert candidate.features == features
    assert candidate.approved is False
    assert candidate.rejection_reasons == (
        "Upcoming earnings on 2024-01-09 are 2 trading days away; new entries are blocked within 3 trading days.",
    )


def test_build_position_features_captures_end_of_day_review_state() -> None:
    features = build_position_features(
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        average_entry_price=100.0,
        latest_close=114.0,
        current_stop=90.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=130.0,
        breakout_failure_reference=120.0,
        days_since_new_high=16,
        stale_high_watch_days=15,
        relative_strength_return_diff=-0.06,
        relative_strength_window=20,
        earnings_context=EarningsRiskContext(
            symbol="AAPL",
            earnings_date=date(2024, 1, 10),
            earnings_days_away=3,
            is_earnings_risk=True,
        ),
        earnings_watch_days=7,
        sector_context=SectorFeatureContext(
            symbol="AAPL",
            sector_name="Technology",
            industry_name="Consumer electronics",
            sector_etf_symbol="XLK",
            sector_regime_passed=False,
            sector_return=-0.02,
            symbol_return=-0.08,
            relative_strength_vs_sector=-0.06,
            relative_strength_window=20,
            sector_trend_state="weak",
            mapping_source="test",
        ),
        sector_relative_strength_watch_threshold=-0.05,
        market_context=build_market_context(
            as_of_date=date(2024, 1, 5),
            benchmark_symbol="SPY",
            regime_passed=True,
        ),
    )

    assert features.unrealized_pl_pct == pytest.approx(0.14)
    assert features.giveback_pct == pytest.approx((130.0 - 114.0) / 130.0)
    assert features.failed_breakout is True
    assert features.stale_position is True
    assert features.lagging_sector is True
    assert features.is_earnings_risk is True


def test_review_existing_long_position_consumes_position_features() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )
    features = build_position_features(
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        average_entry_price=100.0,
        latest_close=114.0,
        current_stop=90.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=130.0,
        breakout_failure_reference=None,
        days_since_new_high=2,
        stale_high_watch_days=15,
        relative_strength_return_diff=None,
        relative_strength_window=20,
        earnings_context=None,
        earnings_watch_days=7,
        sector_context=None,
        market_context=build_market_context(as_of_date=date(2024, 1, 5)),
    )

    decision = review_existing_long_position(
        position,
        position_features=features,
        latest_close=features.latest_close,
        regime_passed=features.regime_passed,
        trailing_stop_candidate=features.trailing_stop_candidate,
        high_water_close=features.high_water_close,
        profit_giveback_threshold=0.10,
        profit_giveback_min_unrealized_pct=0.10,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert "given back" in " ".join(decision.rationale).lower()


def test_build_intraday_position_features_captures_intraday_review_state() -> None:
    features = build_intraday_position_features(
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        average_entry_price=100.0,
        current_stop=95.0,
        session_open=100.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=98.0,
        latest_low=97.0,
        session_vwap=104.0,
        intraday_return_vs_benchmark=-0.03,
        early_strength_threshold=0.03,
        meaningful_profit_pct=0.05,
        session_high_giveback_watch_threshold=0.04,
        intraday_relative_strength_watch_threshold=-0.02,
        sector_relative_strength_watch_threshold=-0.05,
        market_context=build_market_context(as_of_date=date(2024, 1, 5)),
    )

    assert features.stop_breached_intraday is True
    assert features.failed_intraday_strength is True
    assert features.weak_intraday_relative_strength is True
    assert features.session_high_giveback_pct == pytest.approx((112.0 - 98.0) / 112.0)


def test_apply_intraday_trajectory_features_enriches_intraday_position_features() -> None:
    features = build_intraday_position_features(
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        average_entry_price=100.0,
        current_stop=95.0,
        session_open=100.0,
        session_high=112.0,
        session_low=97.0,
        latest_close=101.0,
        latest_low=100.5,
        session_vwap=102.0,
        intraday_return_vs_benchmark=-0.01,
        market_context=build_market_context(as_of_date=date(2024, 1, 5)),
    )

    enriched = apply_intraday_trajectory_features(
        features,
        IntradayTrajectoryFeatures(
            observation_count=4,
            consecutive_polls_below_vwap=3,
            consecutive_polls_weak_relative_strength=2,
            intraday_pressure_persistence_count=3,
            max_session_high_giveback_seen=0.057,
            worsening_session_high_giveback=True,
            giveback_worsening_polls=4,
            giveback_worsening_from_pct=0.021,
            repeated_watch_closely_count=2,
            weakening_all_session=False,
            recovered_after_weakness=False,
        ),
    )

    assert enriched.intraday_observation_count == 4
    assert enriched.consecutive_polls_below_vwap == 3
    assert enriched.giveback_worsening_from_pct == pytest.approx(0.021)
    assert enriched.repeated_watch_closely_count == 2


def test_review_existing_long_position_intraday_consumes_position_features() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    features = build_intraday_position_features(
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        average_entry_price=100.0,
        current_stop=95.0,
        session_open=100.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=98.0,
        latest_low=97.0,
        session_vwap=104.0,
        intraday_return_vs_benchmark=-0.03,
        early_strength_threshold=0.03,
        meaningful_profit_pct=0.05,
        session_high_giveback_watch_threshold=0.04,
        intraday_relative_strength_watch_threshold=-0.02,
        sector_relative_strength_watch_threshold=-0.05,
        market_context=build_market_context(as_of_date=date(2024, 1, 5)),
    )

    decision = review_existing_long_position_intraday(
        position,
        position_features=features,
        session_open=features.session_open or 100.0,
        session_high=features.session_high or 112.0,
        session_low=features.session_low or 94.0,
        latest_close=features.latest_close,
        latest_low=features.latest_low or 97.0,
    )

    assert decision.suggested_action == "EXIT CANDIDATE"
    assert "traded through the current stop" in " ".join(decision.rationale).lower()


def test_assess_signal_candidate_fallback_matches_shared_builder() -> None:
    signal = StrategySignal(
        strategy_name="breakout_momentum",
        symbol="AAPL",
        date=date(2024, 1, 5),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=101.0,
        stop_hint=96.0,
        metadata={
            "preset_name": "standard_breakout",
            "parameter_id": "standard",
            "prior_high": 100.0,
            "relative_volume": 2.1,
            "setup_quality_score": 0.7,
            "setup_persistence_days": 2,
            "days_approved": 2,
            "repeated_high_quality_signal": True,
        },
    )
    constraints = PortfolioConstraints(
        max_concurrent_positions=10,
        max_position_pct_equity=0.25,
        no_averaging_down=True,
    )
    expected_features = build_candidate_features(
        signal,
        as_of_date=signal.date,
        stop_hint=96.0,
        earnings_date=date(2024, 1, 16),
        earnings_days_away=7,
        is_earnings_risk=False,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=True,
        relative_strength_vs_sector=0.02,
        sector_relative_strength_window=20,
    )

    expected = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        candidate_features=expected_features,
        earnings_entry_block_days=3,
    )
    actual = assess_signal_candidate(
        signal,
        current_equity=100_000.0,
        base_risk_per_trade=0.01,
        constraints=constraints,
        earnings_date=date(2024, 1, 16),
        earnings_days_away=7,
        earnings_entry_block_days=3,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=True,
        relative_strength_vs_sector=0.02,
        sector_relative_strength_window=20,
    )

    assert actual == expected
    assert actual.features == expected_features
    assert actual.features is not None
    assert actual.features.industry_name == "Application software"


def test_review_existing_long_position_fallback_matches_shared_builder_and_preserves_industry_name() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
    )
    expected_features = build_position_features(
        symbol="AAPL",
        as_of_date=date.today(),
        average_entry_price=100.0,
        latest_close=114.0,
        current_stop=90.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=130.0,
        breakout_failure_reference=120.0,
        days_since_new_high=16,
        stale_high_watch_days=15,
        relative_strength_return_diff=-0.06,
        relative_strength_window=20,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_watch_days=7,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
        relative_strength_vs_sector=-0.06,
        sector_relative_strength_window=20,
        sector_relative_strength_watch_threshold=-0.05,
    )

    expected = review_existing_long_position(
        position,
        position_features=expected_features,
        latest_close=114.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=130.0,
        breakout_failure_reference=120.0,
        days_since_new_high=16,
        relative_strength_return_diff=-0.06,
        relative_strength_window=20,
    )
    actual = review_existing_long_position(
        position,
        latest_close=114.0,
        regime_passed=True,
        trailing_stop_candidate=108.0,
        high_water_close=130.0,
        breakout_failure_reference=120.0,
        days_since_new_high=16,
        relative_strength_return_diff=-0.06,
        relative_strength_window=20,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
        relative_strength_vs_sector=-0.06,
        sector_relative_strength_window=20,
    )

    assert actual == expected
    assert actual.metadata["industry_name"] == "Application software"


def test_review_existing_long_position_intraday_fallback_matches_shared_builder_and_preserves_industry_name() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
    )
    expected_features = build_intraday_position_features(
        symbol="AAPL",
        as_of_date=date.today(),
        average_entry_price=100.0,
        current_stop=95.0,
        session_open=100.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=98.0,
        latest_low=97.0,
        session_vwap=104.0,
        intraday_return_vs_benchmark=-0.03,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        earnings_watch_days=7,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
        relative_strength_vs_sector=-0.06,
        sector_relative_strength_window=20,
        early_strength_threshold=0.03,
        meaningful_profit_pct=0.05,
        session_high_giveback_watch_threshold=0.04,
        intraday_relative_strength_watch_threshold=-0.02,
        sector_relative_strength_watch_threshold=-0.05,
    )

    expected = review_existing_long_position_intraday(
        position,
        position_features=expected_features,
        session_open=100.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=98.0,
        latest_low=97.0,
    )
    actual = review_existing_long_position_intraday(
        position,
        session_open=100.0,
        session_high=112.0,
        session_low=94.0,
        latest_close=98.0,
        latest_low=97.0,
        session_vwap=104.0,
        intraday_relative_strength_diff=-0.03,
        earnings_date=date(2024, 1, 10),
        earnings_days_away=3,
        sector_name="Technology",
        industry_name="Application software",
        sector_etf_symbol="XLK",
        sector_regime_passed=False,
        relative_strength_vs_sector=-0.06,
        sector_relative_strength_window=20,
    )

    assert actual == expected
    assert actual.metadata["industry_name"] == "Application software"
