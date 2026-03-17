"""Long-only breakout momentum signal generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from bot.config import RiskConfig, SignalConfig
from bot.indicators.trend import rolling_high
from bot.indicators.volatility import atr
from bot.indicators.volume import relative_volume
from bot.strategy.regime_filter import BenchmarkRegimeSettings, RegimeFilterMode, regime_is_bullish
from bot.strategy.signal_models import StrategySignal


@dataclass(frozen=True)
class BreakoutMomentumSettings:
    """Configuration for the long-only breakout momentum strategy."""

    breakout_lookback: int
    benchmark_symbol: str
    benchmark_sma_fast: int
    benchmark_sma_slow: int
    relative_volume_threshold: float
    atr_window: int
    stop_atr_multiple: float
    require_relative_volume_confirmation: bool = False
    enable_regime_filter: bool = True
    regime_filter_mode: RegimeFilterMode = "either"
    relative_volume_window: int | None = None

    @classmethod
    def from_configs(
        cls,
        signal_config: SignalConfig,
        risk_config: RiskConfig,
        *,
        require_relative_volume_confirmation: bool = False,
        enable_regime_filter: bool = True,
        regime_filter_mode: RegimeFilterMode = "either",
        relative_volume_window: int | None = None,
    ) -> "BreakoutMomentumSettings":
        """Build settings from the shared repo config models."""

        return cls(
            breakout_lookback=signal_config.breakout_lookback,
            benchmark_symbol=signal_config.benchmark_symbol,
            benchmark_sma_fast=signal_config.benchmark_sma_fast,
            benchmark_sma_slow=signal_config.benchmark_sma_slow,
            relative_volume_threshold=signal_config.relative_volume_threshold,
            atr_window=risk_config.atr_length,
            stop_atr_multiple=risk_config.initial_stop_atr,
            require_relative_volume_confirmation=require_relative_volume_confirmation,
            enable_regime_filter=enable_regime_filter,
            regime_filter_mode=regime_filter_mode,
            relative_volume_window=relative_volume_window,
        )

    def __post_init__(self) -> None:
        if self.breakout_lookback <= 0:
            raise ValueError("breakout_lookback must be greater than zero.")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be greater than zero.")
        if self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be greater than zero.")
        if self.relative_volume_window is not None and self.relative_volume_window <= 0:
            raise ValueError("relative_volume_window must be greater than zero when provided.")

    @property
    def resolved_relative_volume_window(self) -> int:
        """Return the relative-volume lookback used by the strategy."""

        return self.relative_volume_window or self.breakout_lookback

    @property
    def regime_settings(self) -> BenchmarkRegimeSettings:
        """Return the reusable benchmark regime settings."""

        return BenchmarkRegimeSettings(
            benchmark_symbol=self.benchmark_symbol,
            fast_window=self.benchmark_sma_fast,
            slow_window=self.benchmark_sma_slow,
            mode=self.regime_filter_mode,
            enabled=self.enable_regime_filter,
        )


def compute_breakout_features(
    price_frame: pd.DataFrame,
    settings: BreakoutMomentumSettings,
) -> pd.DataFrame:
    """Compute breakout-related features for a normalized symbol price frame."""

    prepared_frame = _prepare_price_frame(price_frame)
    feature_frame = prepared_frame.copy()
    feature_frame["prior_high"] = rolling_high(
        feature_frame,
        settings.breakout_lookback,
        column="high",
    ).shift(1)
    feature_frame["breakout"] = feature_frame["close"] > feature_frame["prior_high"]
    feature_frame["atr"] = atr(feature_frame, window=settings.atr_window)
    feature_frame["relative_volume"] = relative_volume(
        feature_frame,
        settings.resolved_relative_volume_window,
        column="volume",
        lag_average=1,
    )
    feature_frame["passes_relative_volume"] = (
        feature_frame["relative_volume"] >= settings.relative_volume_threshold
    )
    return feature_frame


def generate_breakout_signal(
    price_frame: pd.DataFrame,
    *,
    settings: BreakoutMomentumSettings,
    benchmark_frame: pd.DataFrame | None = None,
    has_open_position: bool = False,
    symbol: str | None = None,
) -> StrategySignal | None:
    """Evaluate the latest bar and return a long breakout signal if conditions pass."""

    if has_open_position:
        return None

    feature_frame = compute_breakout_features(price_frame, settings)
    if feature_frame.empty:
        return None

    latest = feature_frame.iloc[-1]
    if not bool(latest["breakout"]):
        return None

    if settings.require_relative_volume_confirmation and not bool(latest["passes_relative_volume"]):
        return None

    regime_passed = True
    if settings.enable_regime_filter:
        if benchmark_frame is None:
            raise ValueError("benchmark_frame is required when enable_regime_filter is True.")
        regime_passed = regime_is_bullish(benchmark_frame, settings.regime_settings)
        if not regime_passed:
            return None

    resolved_symbol = _resolve_symbol(feature_frame, symbol=symbol)
    close_price = float(latest["close"])
    atr_value = _optional_float(latest["atr"])
    stop_hint = (
        close_price - (settings.stop_atr_multiple * atr_value)
        if atr_value is not None
        else None
    )

    return StrategySignal(
        strategy_name="breakout_momentum",
        symbol=resolved_symbol,
        date=_coerce_signal_date(latest["date"]),
        side="BUY",
        entry_reason=f"close_above_prior_{settings.breakout_lookback}_day_high",
        entry_price_hint=close_price,
        stop_hint=stop_hint,
        metadata=_build_signal_metadata(latest, settings, regime_passed=regime_passed),
    )


def _prepare_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ("date", "high", "low", "close", "volume")
    missing_columns = [column for column in required_columns if column not in price_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Price frame is missing required columns: {missing}.")

    prepared_frame = price_frame.sort_values("date", kind="stable").reset_index(drop=True).copy()
    for column in ("high", "low", "close", "volume"):
        prepared_frame[column] = pd.to_numeric(prepared_frame[column], errors="coerce")
    prepared_frame["date"] = pd.to_datetime(prepared_frame["date"], errors="coerce")
    return prepared_frame


def _resolve_symbol(price_frame: pd.DataFrame, *, symbol: str | None) -> str:
    if symbol is not None and symbol.strip():
        return symbol.strip().upper()
    if "symbol" not in price_frame.columns:
        raise ValueError("A symbol column or explicit symbol argument is required.")

    resolved_symbol = str(price_frame["symbol"].iloc[-1]).strip().upper()
    if not resolved_symbol:
        raise ValueError("Symbol cannot be empty.")
    return resolved_symbol


def _coerce_signal_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _build_signal_metadata(
    latest_row: pd.Series,
    settings: BreakoutMomentumSettings,
    *,
    regime_passed: bool,
) -> dict[str, Any]:
    return {
        "breakout_lookback": settings.breakout_lookback,
        "prior_high": _optional_float(latest_row["prior_high"]),
        "close": _optional_float(latest_row["close"]),
        "relative_volume": _optional_float(latest_row["relative_volume"]),
        "relative_volume_threshold": settings.relative_volume_threshold,
        "relative_volume_confirmed": bool(latest_row["passes_relative_volume"]),
        "relative_volume_window": settings.resolved_relative_volume_window,
        "regime_filter_enabled": settings.enable_regime_filter,
        "regime_filter_mode": settings.regime_filter_mode,
        "benchmark_symbol": settings.benchmark_symbol,
        "regime_passed": regime_passed,
        "atr": _optional_float(latest_row["atr"]),
    }
