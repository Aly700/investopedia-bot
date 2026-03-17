"""Volatility indicators for daily-bar research workflows."""

from __future__ import annotations

import pandas as pd


def true_range(
    frame: pd.DataFrame,
    *,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pd.Series:
    """Return the standard true range series for daily bars."""

    prepared_frame = _prepare_ohlc_frame(
        frame,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
    )
    high = prepared_frame[high_column]
    low = prepared_frame[low_column]
    close = prepared_frame[close_column]
    previous_close = close.shift(1)

    true_range_components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    range_series = true_range_components.max(axis=1)
    range_series.name = "true_range"
    return range_series


def average_true_range(
    frame: pd.DataFrame,
    window: int = 14,
    *,
    min_periods: int | None = None,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pd.Series:
    """Return ATR as a rolling mean of true range."""

    range_series = true_range(
        frame,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
    )
    atr_series = range_series.rolling(
        window=_validate_window(window),
        min_periods=_resolve_min_periods(window, min_periods),
    ).mean()
    atr_series.name = f"atr_{window}"
    return atr_series


def atr(
    frame: pd.DataFrame,
    window: int = 14,
    *,
    min_periods: int | None = None,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pd.Series:
    """Alias for :func:`average_true_range`."""

    return average_true_range(
        frame,
        window=window,
        min_periods=min_periods,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
    )


def _prepare_ohlc_frame(
    frame: pd.DataFrame,
    *,
    high_column: str,
    low_column: str,
    close_column: str,
) -> pd.DataFrame:
    required_columns = (high_column, low_column, close_column)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Missing required OHLC columns: {missing}.")

    prepared_frame = frame.sort_values("date", kind="stable") if "date" in frame.columns else frame.copy()
    for column in required_columns:
        prepared_frame[column] = pd.to_numeric(prepared_frame[column], errors="coerce")
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
