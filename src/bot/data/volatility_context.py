"""Derived market-wide volatility regime context from daily VIX-style bars."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import pandas as pd

from bot.data.providers import DailyBarProvider, DataProviderError


DEFAULT_VIX_SYMBOL_CANDIDATES: tuple[str, ...] = ("^VIX", "VIX")
DEFAULT_VIX_SMA_SHORT_WINDOW = 5
DEFAULT_VIX_SMA_LONG_WINDOW = 20
MAX_VOLATILITY_CONTEXT_BAR_AGE_DAYS = 7


def volatility_context_history_start(
    as_of_date: date,
    *,
    long_window: int = DEFAULT_VIX_SMA_LONG_WINDOW,
) -> date:
    """Return a conservative fetch start for VIX-style regime context."""

    if long_window <= 0:
        raise ValueError("long_window must be greater than zero.")
    return as_of_date - timedelta(days=max(long_window * 3, 30))


def build_volatility_regime_context(
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    history_start: date,
    caution_threshold: float | None,
    entry_block_threshold: float | None,
    refresh_cache: bool = False,
    symbol_candidates: Sequence[str] = DEFAULT_VIX_SYMBOL_CANDIDATES,
    short_window: int = DEFAULT_VIX_SMA_SHORT_WINDOW,
    long_window: int = DEFAULT_VIX_SMA_LONG_WINDOW,
) -> dict[str, float | str | bool | None]:
    """Fetch daily volatility data and derive a small explainable regime context."""

    if history_start > as_of_date:
        raise ValueError("history_start must be on or before as_of_date.")
    if short_window <= 0:
        raise ValueError("short_window must be greater than zero.")
    if long_window <= 0:
        raise ValueError("long_window must be greater than zero.")
    if short_window > long_window:
        raise ValueError("short_window must be less than or equal to long_window.")
    if caution_threshold is not None and caution_threshold <= 0:
        raise ValueError("caution_threshold must be greater than zero when provided.")
    if entry_block_threshold is not None and entry_block_threshold <= 0:
        raise ValueError("entry_block_threshold must be greater than zero when provided.")
    if (
        caution_threshold is not None
        and entry_block_threshold is not None
        and caution_threshold >= entry_block_threshold
    ):
        raise ValueError("caution_threshold must be less than entry_block_threshold.")

    for symbol in symbol_candidates:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            continue
        try:
            bars = provider.fetch_daily_bars(
                normalized_symbol,
                history_start,
                as_of_date,
                refresh_cache=refresh_cache,
            )
        except DataProviderError:
            continue
        if bars.empty:
            continue

        prepared = _prepare_daily_frame(bars, as_of_date=as_of_date)
        if len(prepared) < long_window:
            continue

        latest_bar_date = prepared["date"].iloc[-1].date()
        if (as_of_date - latest_bar_date).days > MAX_VOLATILITY_CONTEXT_BAR_AGE_DAYS:
            continue

        vix_close = float(prepared["close"].iloc[-1])
        # These rolling averages are currently reporting-only. Entry gates use
        # spot `vix_close` against the configured thresholds, not SMA crossovers
        # or SMA-relative checks.
        vix_sma_short = float(prepared["close"].rolling(short_window).mean().iloc[-1])
        vix_sma_long = float(prepared["close"].rolling(long_window).mean().iloc[-1])
        state, risk_off = _volatility_regime_state(
            vix_close,
            caution_threshold=caution_threshold,
            entry_block_threshold=entry_block_threshold,
        )
        return {
            "vix_close": vix_close,
            "vix_sma_short": vix_sma_short,
            "vix_sma_long": vix_sma_long,
            "volatility_regime_state": state,
            "volatility_regime_risk_off": risk_off,
        }

    return {
        "vix_close": None,
        "vix_sma_short": None,
        "vix_sma_long": None,
        "volatility_regime_state": None,
        "volatility_regime_risk_off": None,
    }


def _prepare_daily_frame(
    frame: pd.DataFrame,
    *,
    as_of_date: date,
) -> pd.DataFrame:
    prepared = frame.sort_values("date", kind="stable").copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared = prepared.loc[
        prepared["date"].notna()
        & prepared["close"].notna()
        & (prepared["date"].dt.date <= as_of_date)
    ]
    return prepared.reset_index(drop=True)


def _volatility_regime_state(
    vix_close: float,
    *,
    caution_threshold: float | None,
    entry_block_threshold: float | None,
) -> tuple[str | None, bool | None]:
    if entry_block_threshold is not None and vix_close >= entry_block_threshold:
        return "stressed", True
    if caution_threshold is not None and vix_close >= caution_threshold:
        return "elevated", False
    if caution_threshold is None and entry_block_threshold is None:
        return None, None
    return "calm", False
