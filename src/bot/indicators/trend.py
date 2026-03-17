"""Trend indicator helpers built on pandas rolling windows."""

from __future__ import annotations

from typing import Union

import pandas as pd


SeriesLike = Union[pd.Series, pd.DataFrame]


def simple_moving_average(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Return the simple moving average for a Series or DataFrame column."""

    series = _resolve_series(data, column=column, default_column="close")
    moving_average = series.rolling(
        window=_validate_window(window),
        min_periods=_resolve_min_periods(window, min_periods),
    ).mean()
    moving_average.name = _build_indicator_name("sma", series.name, window)
    return moving_average


def sma(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Alias for :func:`simple_moving_average`."""

    return simple_moving_average(data, window, column=column, min_periods=min_periods)


def rolling_high(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Return the rolling high for a Series or DataFrame column."""

    series = _resolve_series(data, column=column, default_column="high")
    high_series = series.rolling(
        window=_validate_window(window),
        min_periods=_resolve_min_periods(window, min_periods),
    ).max()
    high_series.name = _build_indicator_name("rolling_high", series.name, window)
    return high_series


def rolling_low(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Return the rolling low for a Series or DataFrame column."""

    series = _resolve_series(data, column=column, default_column="low")
    low_series = series.rolling(
        window=_validate_window(window),
        min_periods=_resolve_min_periods(window, min_periods),
    ).min()
    low_series.name = _build_indicator_name("rolling_low", series.name, window)
    return low_series


def _resolve_series(
    data: SeriesLike,
    *,
    column: str | None,
    default_column: str | None,
) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        frame = _prepare_frame(data)
        resolved_column = _resolve_column(frame, column=column, default_column=default_column)
        series = frame[resolved_column]

    return pd.to_numeric(series, errors="coerce")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" in frame.columns:
        return frame.sort_values("date", kind="stable")
    return frame


def _resolve_column(
    frame: pd.DataFrame,
    *,
    column: str | None,
    default_column: str | None,
) -> str:
    if column is not None:
        if column not in frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")
        return column
    if default_column is not None and default_column in frame.columns:
        return default_column
    if len(frame.columns) == 1:
        return str(frame.columns[0])
    raise ValueError("A column name is required when passing a DataFrame with multiple columns.")


def _validate_window(window: int) -> int:
    if window <= 0:
        raise ValueError("window must be greater than zero.")
    return window


def _resolve_min_periods(window: int, min_periods: int | None) -> int:
    resolved_min_periods = window if min_periods is None else min_periods
    if resolved_min_periods <= 0:
        raise ValueError("min_periods must be greater than zero.")
    if resolved_min_periods > window:
        raise ValueError("min_periods cannot be greater than window.")
    return resolved_min_periods


def _build_indicator_name(prefix: str, source_name: object, window: int) -> str:
    if source_name in (None, ""):
        return f"{prefix}_{window}"
    return f"{prefix}_{source_name}_{window}"
