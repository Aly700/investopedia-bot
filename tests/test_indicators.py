from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from bot.indicators.trend import rolling_high, rolling_low, simple_moving_average, sma
from bot.indicators.volatility import atr, average_true_range, true_range
from bot.indicators.volume import average_dollar_volume, average_volume, relative_volume


def test_simple_moving_average_matches_hand_checked_values() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0], name="close")

    result = simple_moving_average(values, 2)
    expected = pd.Series([np.nan, 1.5, 2.5, 3.5], name="sma_close_2")

    assert_series_equal(result, expected)
    assert_series_equal(sma(values, 2), expected)


def test_rolling_high_and_low_sort_dataframe_by_date() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
            "high": [12.0, 10.0, 11.0],
            "low": [9.0, 7.0, 8.0],
        },
        index=[30, 10, 20],
    )

    high_result = rolling_high(frame, 2)
    low_result = rolling_low(frame, 2)

    expected_high = pd.Series([12.0, np.nan, 11.0], index=[30, 10, 20], name="rolling_high_high_2")
    expected_low = pd.Series([8.0, np.nan, 7.0], index=[30, 10, 20], name="rolling_low_low_2")

    assert_series_equal(high_result.sort_index(), expected_high.sort_index())
    assert_series_equal(low_result.sort_index(), expected_low.sort_index())


def test_simple_moving_average_respects_min_periods_with_nans() -> None:
    values = pd.Series([100.0, np.nan, 300.0], name="volume")

    result = simple_moving_average(values, 2, min_periods=1)
    expected = pd.Series([100.0, 100.0, 300.0], name="sma_volume_2")

    assert_series_equal(result, expected)


def test_true_range_and_atr_match_hand_checked_values() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 12.0, 11.0],
            "low": [8.0, 9.0, 7.0],
            "close": [9.0, 11.0, 8.0],
        }
    )

    tr_result = true_range(frame)
    atr_result = average_true_range(frame, window=2)
    atr_alias = atr(frame, window=2)

    expected_tr = pd.Series([2.0, 3.0, 4.0], name="true_range")
    expected_atr = pd.Series([np.nan, 2.5, 3.5], name="atr_2")

    assert_series_equal(tr_result, expected_tr)
    assert_series_equal(atr_result, expected_atr)
    assert_series_equal(atr_alias, expected_atr)


def test_average_volume_and_relative_volume_match_expected_values() -> None:
    frame = pd.DataFrame({"volume": [100.0, 200.0, 300.0, 400.0]})

    avg_volume = average_volume(frame, 2)
    rel_volume = relative_volume(frame, 2)

    expected_avg_volume = pd.Series([np.nan, 150.0, 250.0, 350.0], name="average_volume_2")
    expected_rel_volume = pd.Series(
        [np.nan, 200.0 / 150.0, 300.0 / 250.0, 400.0 / 350.0],
        name="relative_volume_2",
    )

    assert_series_equal(avg_volume, expected_avg_volume)
    assert_series_equal(rel_volume, expected_rel_volume)


def test_average_dollar_volume_uses_close_times_volume() -> None:
    frame = pd.DataFrame(
        {
            "close": [10.0, 20.0, 30.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )

    result = average_dollar_volume(frame, 2)
    expected = pd.Series([np.nan, 2500.0, 6500.0], name="average_dollar_volume_2")

    assert_series_equal(result, expected)


def test_relative_volume_can_lag_the_average_and_avoid_zero_division() -> None:
    frame = pd.DataFrame({"volume": [100.0, 0.0, 300.0]})

    result = relative_volume(frame, 1, lag_average=1)
    expected = pd.Series([np.nan, 0.0, np.nan], name="relative_volume_1")

    assert_series_equal(result, expected)


def test_indicator_window_validation_rejects_invalid_values() -> None:
    values = pd.Series([1.0, 2.0, 3.0])

    try:
        simple_moving_average(values, 0)
    except ValueError as exc:
        assert "window" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero window.")

    try:
        average_true_range(pd.DataFrame({"high": [1], "low": [0], "close": [1]}), 2, min_periods=3)
    except ValueError as exc:
        assert "min_periods" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid min_periods.")
