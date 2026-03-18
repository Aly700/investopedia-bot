from __future__ import annotations

from datetime import date

import pytest
import pandas as pd

from bot.config import RiskConfig, SignalConfig
from bot.strategy.breakout_momentum import (
    BreakoutMomentumSettings,
    build_default_breakout_presets,
    compute_breakout_features,
    generate_breakout_signal,
)
from bot.strategy.regime_filter import BenchmarkRegimeSettings, benchmark_regime_series, regime_is_bullish


def test_compute_breakout_features_detects_breakout() -> None:
    price_frame = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 300],
        symbol="AAA",
    )
    settings = _settings(require_relative_volume_confirmation=True)

    features = compute_breakout_features(price_frame, settings)

    assert bool(features["breakout"].iloc[-1]) is True
    assert features["prior_high"].iloc[-1] == pytest.approx(11.0)
    assert features["relative_volume"].iloc[-1] == pytest.approx(3.0)


def test_generate_breakout_signal_returns_none_when_breakout_fails() -> None:
    price_frame = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 10.9],
        highs=[10.0, 11.0, 11.0, 11.2],
        lows=[9.0, 10.0, 10.5, 10.6],
        volumes=[100, 100, 100, 300],
        symbol="AAA",
    )

    signal = generate_breakout_signal(
        price_frame,
        settings=_settings(enable_regime_filter=False),
    )

    assert signal is None


def test_regime_filter_blocks_signal_when_benchmark_is_weak() -> None:
    price_frame = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 300],
        symbol="AAA",
    )
    weak_benchmark = _benchmark_frame([103.0, 102.0, 101.0, 100.0], symbol="SPY")

    signal = generate_breakout_signal(
        price_frame,
        settings=_settings(enable_regime_filter=True),
        benchmark_frame=weak_benchmark,
    )

    assert signal is None


def test_relative_volume_confirmation_can_block_or_allow_signal() -> None:
    low_volume_breakout = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 120],
        symbol="AAA",
    )

    blocked_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=_settings(require_relative_volume_confirmation=True, enable_regime_filter=False),
    )
    allowed_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=_settings(require_relative_volume_confirmation=False, enable_regime_filter=False),
    )

    assert blocked_signal is None
    assert allowed_signal is not None
    assert allowed_signal.metadata["relative_volume"] == pytest.approx(1.2)
    assert allowed_signal.metadata["relative_volume_confirmed"] is False
    assert allowed_signal.metadata["relative_volume_required"] is False
    assert allowed_signal.metadata["relative_volume_policy"] == "optional"
    assert allowed_signal.metadata["relative_volume_gate_passed"] is True


def test_confirmed_breakout_blocks_low_relative_volume_that_standard_allows() -> None:
    low_volume_breakout = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 120],
        symbol="AAA",
    )
    presets = build_default_breakout_presets(_signal_config(), _risk_config())
    base_settings = BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )

    standard_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=presets["standard_breakout"].apply_to_settings(base_settings),
    )
    confirmed_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=presets["confirmed_breakout"].apply_to_settings(base_settings),
    )

    assert standard_signal is not None
    assert standard_signal.metadata["relative_volume_policy"] == "optional"
    assert confirmed_signal is None


def test_confirmed_conservative_breakout_blocks_low_relative_volume_that_conservative_allows() -> None:
    low_volume_breakout = _symbol_frame(
        closes=[9.5] * 13 + [11.6],
        highs=[10.0] * 13 + [12.0],
        lows=[9.0] * 13 + [11.0],
        volumes=[100] * 13 + [150],
        symbol="AAA",
    )
    presets = build_default_breakout_presets(_signal_config(), _risk_config())
    base_settings = BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )

    conservative_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=presets["conservative_breakout"].apply_to_settings(base_settings),
    )
    confirmed_conservative_signal = generate_breakout_signal(
        low_volume_breakout,
        settings=presets["confirmed_conservative_breakout"].apply_to_settings(base_settings),
    )

    assert conservative_signal is not None
    assert conservative_signal.metadata["relative_volume_policy"] == "optional"
    assert confirmed_conservative_signal is None


def test_signal_contents_include_stop_and_metadata() -> None:
    price_frame = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 300],
        symbol="AAA",
    )
    strong_benchmark = _benchmark_frame([100.0, 101.0, 102.0, 103.0], symbol="SPY")

    signal = generate_breakout_signal(
        price_frame,
        settings=_settings(require_relative_volume_confirmation=True, enable_regime_filter=True),
        benchmark_frame=strong_benchmark,
    )

    assert signal is not None
    assert signal.strategy_name == "breakout_momentum"
    assert signal.symbol == "AAA"
    assert signal.date == date(2024, 1, 4)
    assert signal.side == "BUY"
    assert signal.entry_reason == "close_above_prior_3_day_high"
    assert signal.entry_price_hint == pytest.approx(11.6)
    assert signal.stop_hint == pytest.approx(9.9)
    assert signal.metadata["prior_high"] == pytest.approx(11.0)
    assert signal.metadata["relative_volume"] == pytest.approx(3.0)
    assert signal.metadata["regime_passed"] is True


def test_open_position_blocks_new_signal() -> None:
    price_frame = _symbol_frame(
        closes=[9.5, 10.5, 10.8, 11.6],
        highs=[10.0, 11.0, 11.0, 12.0],
        lows=[9.0, 10.0, 10.5, 11.0],
        volumes=[100, 100, 100, 300],
        symbol="AAA",
    )

    signal = generate_breakout_signal(
        price_frame,
        settings=_settings(enable_regime_filter=False),
        has_open_position=True,
    )

    assert signal is None


def test_regime_filter_supports_configured_modes() -> None:
    benchmark_frame = _benchmark_frame([100.0, 101.0, 102.0, 103.0], symbol="SPY")

    close_only = BenchmarkRegimeSettings("SPY", fast_window=2, slow_window=3, mode="close_above_slow")
    fast_only = BenchmarkRegimeSettings("SPY", fast_window=2, slow_window=3, mode="fast_above_slow")

    close_series = benchmark_regime_series(benchmark_frame, close_only)
    fast_series = benchmark_regime_series(benchmark_frame, fast_only)

    assert bool(close_series.iloc[-1]) is True
    assert bool(fast_series.iloc[-1]) is True
    assert regime_is_bullish(benchmark_frame, close_only) is True


def _settings(
    *,
    require_relative_volume_confirmation: bool = False,
    enable_regime_filter: bool = True,
) -> BreakoutMomentumSettings:
    return BreakoutMomentumSettings(
        breakout_lookback=3,
        benchmark_symbol="SPY",
        benchmark_sma_fast=2,
        benchmark_sma_slow=3,
        relative_volume_threshold=1.5,
        atr_window=2,
        stop_atr_multiple=2.0,
        trailing_stop_atr=3.0,
        require_relative_volume_confirmation=require_relative_volume_confirmation,
        enable_regime_filter=enable_regime_filter,
        regime_filter_mode="either",
    )


def _symbol_frame(
    *,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    symbol: str,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "symbol": [symbol] * len(closes),
        }
    )


def _benchmark_frame(closes: list[float], *, symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
            "symbol": [symbol] * len(closes),
        }
    )


def _signal_config() -> SignalConfig:
    return SignalConfig(
        breakout_lookback=3,
        benchmark_symbol="SPY",
        benchmark_sma_fast=2,
        benchmark_sma_slow=3,
        relative_volume_threshold=1.5,
    )


def _risk_config() -> RiskConfig:
    return RiskConfig(
        risk_per_trade=0.01,
        atr_length=2,
        initial_stop_atr=2.0,
        trailing_stop_atr=3.0,
        drawdown_risk_reduction_threshold=0.15,
    )
