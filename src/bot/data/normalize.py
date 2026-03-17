"""Helpers for normalizing daily OHLCV data frames."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd


DAILY_BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume", "symbol")
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataNormalizationError(ValueError):
    """Raised when provider output cannot be normalized into daily bars."""


def empty_daily_bar_frame() -> pd.DataFrame:
    """Return an empty daily-bar frame with the canonical schema and dtypes."""

    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "symbol": pd.Series(dtype="object"),
        }
    )


def normalize_daily_bars(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize provider output into the repo's canonical daily-bar schema."""

    if frame.empty:
        return empty_daily_bar_frame()

    normalized_symbol = symbol.upper() if symbol is not None else None
    working_frame = frame.copy()
    if column_mapping is not None:
        working_frame = working_frame.rename(columns=dict(column_mapping))

    missing_columns = [
        column
        for column in ("date", "open", "high", "low", "close", "volume")
        if column not in working_frame.columns
    ]
    if missing_columns:
        expected_columns = ", ".join(missing_columns)
        raise DataNormalizationError(f"Missing required columns after normalization: {expected_columns}.")

    if "symbol" not in working_frame.columns:
        if normalized_symbol is None:
            raise DataNormalizationError("A symbol column or symbol argument is required.")
        working_frame["symbol"] = normalized_symbol

    working_frame = working_frame.loc[:, list(DAILY_BAR_COLUMNS)].copy()
    working_frame["date"] = (
        pd.to_datetime(working_frame["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    for column in PRICE_COLUMNS:
        working_frame[column] = pd.to_numeric(working_frame[column], errors="coerce")
    working_frame["volume"] = pd.to_numeric(working_frame["volume"], errors="coerce")
    working_frame["symbol"] = working_frame["symbol"].astype(str).str.strip().str.upper()

    working_frame = working_frame.dropna(subset=list(DAILY_BAR_COLUMNS[:-1]))
    working_frame = working_frame[working_frame["symbol"] != ""]
    for column in PRICE_COLUMNS:
        working_frame = working_frame[working_frame[column] > 0]
    working_frame = working_frame[working_frame["volume"] >= 0]

    if working_frame.empty:
        return empty_daily_bar_frame()

    working_frame["open"] = working_frame["open"].astype("float64")
    working_frame["high"] = working_frame["high"].astype("float64")
    working_frame["low"] = working_frame["low"].astype("float64")
    working_frame["close"] = working_frame["close"].astype("float64")
    working_frame["volume"] = working_frame["volume"].round(0).astype("int64")
    working_frame["symbol"] = working_frame["symbol"].astype("object")

    working_frame = (
        working_frame.sort_values(["date", "symbol"])
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    return working_frame.loc[:, list(DAILY_BAR_COLUMNS)]


def filter_daily_bars_by_date(
    frame: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Filter a normalized daily-bar frame to an inclusive date range."""

    if frame.empty:
        return empty_daily_bar_frame()

    mask = frame["date"].dt.date.between(start_date, end_date)
    filtered = frame.loc[mask].reset_index(drop=True)
    if filtered.empty:
        return empty_daily_bar_frame()
    return filtered.loc[:, list(DAILY_BAR_COLUMNS)]
