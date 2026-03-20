"""Reusable market regime filters for benchmark-aware strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import pandas as pd

from bot.config import SignalConfig
from bot.indicators.trend import simple_moving_average


RegimeFilterMode = Literal["close_above_slow", "fast_above_slow", "either"]
VALID_REGIME_FILTER_MODES: frozenset[str] = frozenset(get_args(RegimeFilterMode))


@dataclass(frozen=True)
class BenchmarkRegimeSettings:
    """Configuration for a benchmark-based market regime filter."""

    benchmark_symbol: str
    fast_window: int
    slow_window: int
    mode: RegimeFilterMode = "either"
    enabled: bool = True

    @classmethod
    def from_signal_config(
        cls,
        signal_config: SignalConfig,
        *,
        mode: RegimeFilterMode = "either",
        enabled: bool = True,
    ) -> "BenchmarkRegimeSettings":
        """Build settings from the shared signal config."""

        return cls(
            benchmark_symbol=signal_config.benchmark_symbol,
            fast_window=signal_config.benchmark_sma_fast,
            slow_window=signal_config.benchmark_sma_slow,
            mode=mode,
            enabled=enabled,
        )

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("Benchmark moving-average windows must be greater than zero.")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be strictly less than "
                f"slow_window ({self.slow_window})."
            )
        if self.mode not in VALID_REGIME_FILTER_MODES:
            raise ValueError(
                "Unsupported regime filter mode: "
                f"{self.mode}. Expected one of {sorted(VALID_REGIME_FILTER_MODES)}."
            )


def benchmark_regime_series(
    benchmark_frame: pd.DataFrame,
    settings: BenchmarkRegimeSettings,
) -> pd.Series:
    """Return a boolean series indicating whether the benchmark regime is bullish."""

    if "close" not in benchmark_frame.columns:
        raise KeyError("Benchmark frame must contain a 'close' column.")

    prepared_frame = benchmark_frame.sort_values("date", kind="stable").reset_index(drop=True)
    close = pd.to_numeric(prepared_frame["close"], errors="coerce")
    slow_sma = simple_moving_average(close, settings.slow_window)
    fast_sma = simple_moving_average(close, settings.fast_window)

    close_above_slow = close > slow_sma
    fast_above_slow = fast_sma > slow_sma

    if settings.mode == "close_above_slow":
        regime_series = close_above_slow
    elif settings.mode == "fast_above_slow":
        regime_series = fast_above_slow
    else:
        regime_series = close_above_slow | fast_above_slow

    regime_series.name = f"benchmark_regime_{settings.mode}"
    return regime_series.fillna(False)


def regime_is_bullish(
    benchmark_frame: pd.DataFrame,
    settings: BenchmarkRegimeSettings,
) -> bool:
    """Return whether the latest benchmark bar passes the configured regime filter."""

    if not settings.enabled:
        return True

    regime_series = benchmark_regime_series(benchmark_frame, settings)
    if regime_series.empty:
        return False
    return bool(regime_series.iloc[-1])
