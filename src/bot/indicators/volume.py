"""Volume-based indicators for normalized OHLCV daily bars."""

from __future__ import annotations

from typing import Union

import pandas as pd

from bot.indicators.trend import simple_moving_average


SeriesLike = Union[pd.Series, pd.DataFrame]


def average_volume(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Return the rolling average volume series."""

    average_series = simple_moving_average(
        data,
        window,
        column=column or ("volume" if isinstance(data, pd.DataFrame) else None),
        min_periods=min_periods,
    )
    average_series.name = f"average_volume_{window}"
    return average_series


def average_dollar_volume(
    frame: pd.DataFrame,
    window: int,
    *,
    close_column: str = "close",
    volume_column: str = "volume",
    min_periods: int | None = None,
) -> pd.Series:
    """Return the rolling average dollar volume using close * volume."""

    prepared_frame = _prepare_volume_frame(frame, close_column=close_column, volume_column=volume_column)
    dollar_volume = prepared_frame[close_column] * prepared_frame[volume_column]
    average_dollar_volume_series = dollar_volume.rolling(
        window=_validate_window(window),
        min_periods=_resolve_min_periods(window, min_periods),
    ).mean()
    average_dollar_volume_series.name = f"average_dollar_volume_{window}"
    return average_dollar_volume_series


def relative_volume(
    data: SeriesLike,
    window: int,
    *,
    column: str | None = None,
    min_periods: int | None = None,
    lag_average: int = 0,
) -> pd.Series:
    """Return current volume divided by rolling average volume."""

    if lag_average < 0:
        raise ValueError("lag_average cannot be negative.")

    volume_series = _resolve_volume_series(data, column=column)
    baseline = average_volume(
        data,
        window,
        column=column or ("volume" if isinstance(data, pd.DataFrame) else None),
        min_periods=min_periods,
    )
    if lag_average:
        baseline = baseline.shift(lag_average)

    denominator = baseline.where(baseline != 0)
    relative_volume_series = volume_series.divide(denominator)
    relative_volume_series.name = f"relative_volume_{window}"
    return relative_volume_series


def _resolve_volume_series(data: SeriesLike, *, column: str | None) -> pd.Series:
    if isinstance(data, pd.Series):
        return pd.to_numeric(data.copy(), errors="coerce")

    prepared_frame = _prepare_volume_frame(
        data,
        close_column="close" if "close" in data.columns else data.columns[0],
        volume_column=column or "volume",
    )
    resolved_column = column or "volume"
    if resolved_column not in prepared_frame.columns:
        raise KeyError(f"Column '{resolved_column}' does not exist in the DataFrame.")
    return pd.to_numeric(prepared_frame[resolved_column], errors="coerce")


def _prepare_volume_frame(
    frame: pd.DataFrame,
    *,
    close_column: str,
    volume_column: str,
) -> pd.DataFrame:
    required_columns = (close_column, volume_column)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Missing required volume columns: {missing}.")

    prepared_frame = frame.sort_values("date", kind="stable") if "date" in frame.columns else frame.copy()
    prepared_frame[close_column] = pd.to_numeric(prepared_frame[close_column], errors="coerce")
    prepared_frame[volume_column] = pd.to_numeric(prepared_frame[volume_column], errors="coerce")
    return prepared_frame


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
