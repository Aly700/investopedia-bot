"""Long-only breakout momentum signal generation and preset helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Mapping, Sequence

import pandas as pd

from bot.config import RiskConfig, SignalConfig
from bot.indicators.trend import rolling_high
from bot.indicators.volatility import atr
from bot.indicators.volume import relative_volume
from bot.strategy.regime_filter import (
    VALID_REGIME_FILTER_MODES,
    BenchmarkRegimeSettings,
    RegimeFilterMode,
    regime_is_bullish,
)
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
    trailing_stop_atr: float
    require_relative_volume_confirmation: bool = False
    enable_regime_filter: bool = True
    regime_filter_mode: RegimeFilterMode = "either"
    relative_volume_window: int | None = None
    profit_giveback_threshold: float = 0.10
    profit_giveback_min_unrealized_pct: float = 0.10
    stale_high_watch_days: int = 15
    relative_strength_window: int = 20
    relative_strength_watch_threshold: float = -0.05
    intraday_high_profit_unrealized_pct: float = 0.15
    intraday_high_profit_giveback_threshold: float = 0.07
    earnings_entry_block_days: int = 3
    earnings_watch_days: int = 7
    market_breadth_entry_floor_200ma: float | None = 0.40
    vix_caution_threshold: float | None = 25.0
    vix_entry_block_threshold: float | None = 30.0
    require_sector_regime_for_entries: bool = True
    sector_relative_strength_window: int = 20
    sector_relative_strength_entry_reject_threshold: float = -0.05
    sector_relative_strength_watch_threshold: float = -0.05
    max_positions_per_sector: int | None = 3
    max_same_industry_positions: int | None = 2
    max_sector_notional_pct: float | None = None

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
            trailing_stop_atr=risk_config.trailing_stop_atr,
            require_relative_volume_confirmation=require_relative_volume_confirmation,
            enable_regime_filter=enable_regime_filter,
            regime_filter_mode=regime_filter_mode,
            relative_volume_window=relative_volume_window,
            earnings_entry_block_days=signal_config.earnings_entry_block_days,
            earnings_watch_days=signal_config.earnings_watch_days,
            market_breadth_entry_floor_200ma=signal_config.market_breadth_entry_floor_200ma,
            vix_caution_threshold=signal_config.vix_caution_threshold,
            vix_entry_block_threshold=signal_config.vix_entry_block_threshold,
            require_sector_regime_for_entries=signal_config.require_sector_regime_for_entries,
            sector_relative_strength_window=signal_config.sector_relative_strength_window,
            sector_relative_strength_entry_reject_threshold=(
                signal_config.sector_relative_strength_entry_reject_threshold
            ),
            sector_relative_strength_watch_threshold=(
                signal_config.sector_relative_strength_watch_threshold
            ),
            max_positions_per_sector=signal_config.max_positions_per_sector,
            max_same_industry_positions=signal_config.max_same_industry_positions,
            max_sector_notional_pct=signal_config.max_sector_notional_pct,
        )

    def __post_init__(self) -> None:
        if self.breakout_lookback <= 0:
            raise ValueError("breakout_lookback must be greater than zero.")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be greater than zero.")
        if self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be greater than zero.")
        if self.trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be greater than zero.")
        if self.relative_volume_window is not None and self.relative_volume_window <= 0:
            raise ValueError("relative_volume_window must be greater than zero when provided.")
        if self.profit_giveback_threshold <= 0 or self.profit_giveback_threshold >= 1:
            raise ValueError("profit_giveback_threshold must be between zero and one.")
        if self.profit_giveback_min_unrealized_pct <= 0:
            raise ValueError("profit_giveback_min_unrealized_pct must be greater than zero.")
        if self.stale_high_watch_days <= 0:
            raise ValueError("stale_high_watch_days must be greater than zero.")
        if self.relative_strength_window <= 0:
            raise ValueError("relative_strength_window must be greater than zero.")
        if self.intraday_high_profit_unrealized_pct <= 0:
            raise ValueError("intraday_high_profit_unrealized_pct must be greater than zero.")
        if self.intraday_high_profit_giveback_threshold <= 0 or self.intraday_high_profit_giveback_threshold >= 1:
            raise ValueError("intraday_high_profit_giveback_threshold must be between zero and one.")
        if self.earnings_entry_block_days <= 0:
            raise ValueError("earnings_entry_block_days must be greater than zero.")
        if self.earnings_watch_days <= 0:
            raise ValueError("earnings_watch_days must be greater than zero.")
        if self.market_breadth_entry_floor_200ma is not None and (
            self.market_breadth_entry_floor_200ma <= 0
            or self.market_breadth_entry_floor_200ma >= 1
        ):
            raise ValueError(
                "market_breadth_entry_floor_200ma must be between zero and one when provided."
            )
        if self.vix_caution_threshold is not None and self.vix_caution_threshold <= 0:
            raise ValueError("vix_caution_threshold must be greater than zero when provided.")
        if self.vix_entry_block_threshold is not None and self.vix_entry_block_threshold <= 0:
            raise ValueError(
                "vix_entry_block_threshold must be greater than zero when provided."
            )
        if (
            self.vix_caution_threshold is not None
            and self.vix_entry_block_threshold is not None
            and self.vix_caution_threshold >= self.vix_entry_block_threshold
        ):
            raise ValueError(
                "vix_caution_threshold must be less than vix_entry_block_threshold when both are provided."
            )
        if self.sector_relative_strength_window <= 0:
            raise ValueError("sector_relative_strength_window must be greater than zero.")
        if self.sector_relative_strength_entry_reject_threshold > 0:
            raise ValueError(
                "sector_relative_strength_entry_reject_threshold must be less than or equal to zero."
            )
        if self.sector_relative_strength_watch_threshold > 0:
            raise ValueError(
                "sector_relative_strength_watch_threshold must be less than or equal to zero."
            )
        if self.max_positions_per_sector is not None and self.max_positions_per_sector <= 0:
            raise ValueError(
                "max_positions_per_sector must be greater than zero when provided."
            )
        if (
            self.max_same_industry_positions is not None
            and self.max_same_industry_positions <= 0
        ):
            raise ValueError(
                "max_same_industry_positions must be greater than zero when provided."
            )
        if self.max_sector_notional_pct is not None and (
            self.max_sector_notional_pct <= 0 or self.max_sector_notional_pct >= 1
        ):
            raise ValueError(
                "max_sector_notional_pct must be between zero and one when provided."
            )
        if self.regime_filter_mode not in VALID_REGIME_FILTER_MODES:
            raise ValueError(
                "regime_filter_mode must be one of "
                f"{sorted(VALID_REGIME_FILTER_MODES)}, got '{self.regime_filter_mode}'."
            )

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


@dataclass(frozen=True)
class BreakoutStrategyPreset:
    """A named breakout/risk preset for backtest and comparison workflows."""

    name: str
    breakout_lookback: int
    relative_volume_threshold: float
    initial_stop_atr: float
    trailing_stop_atr: float
    risk_per_trade: float
    require_relative_volume_confirmation: bool = False
    regime_filter_mode: RegimeFilterMode = "either"
    profit_giveback_threshold: float = 0.10

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Preset name cannot be empty.")
        if self.breakout_lookback <= 0:
            raise ValueError("breakout_lookback must be greater than zero.")
        if self.relative_volume_threshold <= 0:
            raise ValueError("relative_volume_threshold must be greater than zero.")
        if self.initial_stop_atr <= 0:
            raise ValueError("initial_stop_atr must be greater than zero.")
        if self.trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be greater than zero.")
        if self.risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be greater than zero.")
        if self.profit_giveback_threshold <= 0 or self.profit_giveback_threshold >= 1:
            raise ValueError("profit_giveback_threshold must be between zero and one.")
        if self.regime_filter_mode not in VALID_REGIME_FILTER_MODES:
            raise ValueError(
                "regime_filter_mode must be one of "
                f"{sorted(VALID_REGIME_FILTER_MODES)}, got '{self.regime_filter_mode}'."
            )

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "BreakoutStrategyPreset":
        """Build a preset from a config-style mapping."""

        return cls(
            name=name.strip(),
            breakout_lookback=_mapping_int(data, "breakout_lookback"),
            relative_volume_threshold=_mapping_float(data, "relative_volume_threshold"),
            initial_stop_atr=_mapping_float(data, "initial_stop_atr"),
            trailing_stop_atr=_mapping_float(data, "trailing_stop_atr"),
            risk_per_trade=_mapping_float(data, "risk_per_trade"),
            require_relative_volume_confirmation=_mapping_bool(
                data,
                "require_relative_volume_confirmation",
                default=False,
            ),
            regime_filter_mode=_mapping_regime_filter_mode(data),
            profit_giveback_threshold=(
                _mapping_float(data, "profit_giveback_threshold")
                if "profit_giveback_threshold" in data
                else 0.10
            ),
        )

    @property
    def parameter_id(self) -> str:
        """Return a stable identifier for the preset parameters."""

        return (
            f"lookback={self.breakout_lookback}|"
            f"rv={self.relative_volume_threshold:g}|"
            f"rv_required={int(self.require_relative_volume_confirmation)}|"
            f"initial_stop={self.initial_stop_atr:g}|"
            f"trailing_stop={self.trailing_stop_atr:g}|"
            f"risk={self.risk_per_trade:g}|"
            f"regime={self.regime_filter_mode}|"
            f"giveback={self.profit_giveback_threshold:g}"
        )

    def apply_to_settings(
        self,
        settings: BreakoutMomentumSettings,
        *,
        force_require_relative_volume_confirmation: bool | None = None,
    ) -> BreakoutMomentumSettings:
        """Return strategy settings with the preset's fields applied.

        When ``force_require_relative_volume_confirmation`` is provided, it overrides
        the preset's own relative-volume confirmation policy. This preserves the
        existing CLI behavior where a global flag can require RV for every preset.
        """

        return replace(
            settings,
            breakout_lookback=self.breakout_lookback,
            relative_volume_threshold=self.relative_volume_threshold,
            stop_atr_multiple=self.initial_stop_atr,
            trailing_stop_atr=self.trailing_stop_atr,
            require_relative_volume_confirmation=(
                self.require_relative_volume_confirmation
                if force_require_relative_volume_confirmation is None
                else force_require_relative_volume_confirmation
            ),
            regime_filter_mode=self.regime_filter_mode,
            profit_giveback_threshold=self.profit_giveback_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the preset."""

        return {
            "preset_name": self.name,
            "parameter_id": self.parameter_id,
            "breakout_lookback": self.breakout_lookback,
            "relative_volume_threshold": self.relative_volume_threshold,
            "require_relative_volume_confirmation": self.require_relative_volume_confirmation,
            "initial_stop_atr": self.initial_stop_atr,
            "trailing_stop_atr": self.trailing_stop_atr,
            "risk_per_trade": self.risk_per_trade,
            "regime_filter_mode": self.regime_filter_mode,
            "profit_giveback_threshold": self.profit_giveback_threshold,
        }


def build_default_breakout_presets(
    signal_config: SignalConfig,
    risk_config: RiskConfig,
) -> dict[str, BreakoutStrategyPreset]:
    """Return the built-in preset catalog derived from the repo defaults."""

    standard = BreakoutStrategyPreset(
        name="standard_breakout",
        breakout_lookback=signal_config.breakout_lookback,
        relative_volume_threshold=signal_config.relative_volume_threshold,
        initial_stop_atr=risk_config.initial_stop_atr,
        trailing_stop_atr=risk_config.trailing_stop_atr,
        risk_per_trade=risk_config.risk_per_trade,
        require_relative_volume_confirmation=False,
        profit_giveback_threshold=0.10,
    )
    confirmed = BreakoutStrategyPreset(
        name="confirmed_breakout",
        breakout_lookback=signal_config.breakout_lookback,
        relative_volume_threshold=signal_config.relative_volume_threshold,
        initial_stop_atr=risk_config.initial_stop_atr,
        trailing_stop_atr=risk_config.trailing_stop_atr,
        risk_per_trade=risk_config.risk_per_trade,
        require_relative_volume_confirmation=True,
        profit_giveback_threshold=0.10,
    )
    conservative = BreakoutStrategyPreset(
        name="conservative_breakout",
        breakout_lookback=max(signal_config.breakout_lookback + 10, int(round(signal_config.breakout_lookback * 1.5))),
        relative_volume_threshold=signal_config.relative_volume_threshold + 0.5,
        initial_stop_atr=risk_config.initial_stop_atr + 0.5,
        trailing_stop_atr=risk_config.trailing_stop_atr + 0.5,
        risk_per_trade=max(risk_config.risk_per_trade * 0.5, 0.0025),
        require_relative_volume_confirmation=False,
        profit_giveback_threshold=0.12,
    )
    confirmed_conservative = BreakoutStrategyPreset(
        name="confirmed_conservative_breakout",
        breakout_lookback=conservative.breakout_lookback,
        relative_volume_threshold=conservative.relative_volume_threshold,
        initial_stop_atr=conservative.initial_stop_atr,
        trailing_stop_atr=conservative.trailing_stop_atr,
        risk_per_trade=conservative.risk_per_trade,
        require_relative_volume_confirmation=True,
        profit_giveback_threshold=0.12,
    )
    aggressive = BreakoutStrategyPreset(
        name="aggressive_breakout",
        breakout_lookback=max(5, signal_config.breakout_lookback - 10),
        relative_volume_threshold=max(1.0, signal_config.relative_volume_threshold - 0.5),
        initial_stop_atr=max(1.0, risk_config.initial_stop_atr - 0.5),
        trailing_stop_atr=max(1.0, risk_config.trailing_stop_atr - 0.5),
        risk_per_trade=min(0.05, risk_config.risk_per_trade * 1.5),
        require_relative_volume_confirmation=False,
        regime_filter_mode="fast_above_slow",
        profit_giveback_threshold=0.08,
    )
    return {
        conservative.name: conservative,
        confirmed_conservative.name: confirmed_conservative,
        standard.name: standard,
        confirmed.name: confirmed,
        aggressive.name: aggressive,
    }


def breakout_presets_from_mapping(
    preset_mapping: Mapping[str, Any] | None,
) -> dict[str, BreakoutStrategyPreset]:
    """Parse breakout presets from a config-style mapping."""

    if preset_mapping is None:
        return {}
    presets: dict[str, BreakoutStrategyPreset] = {}
    for name, raw_value in preset_mapping.items():
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"Preset '{name}' must be a mapping of breakout parameters.")
        preset = BreakoutStrategyPreset.from_mapping(str(name), raw_value)
        presets[preset.name] = preset
    return presets


def breakout_preset_from_cli_definition(raw_value: str) -> BreakoutStrategyPreset:
    """Parse an inline CLI preset definition.

    Expected format:
    ``name=my_preset,breakout_lookback=20,relative_volume_threshold=1.5,initial_stop_atr=2.5,trailing_stop_atr=3.0,risk_per_trade=0.01,regime_filter_mode=fast_above_slow``

    Optional fields:
    ``require_relative_volume_confirmation=true|false``,
    ``profit_giveback_threshold=0.08``, and
    ``regime_filter_mode=close_above_slow|fast_above_slow|either``.
    Valid ``regime_filter_mode`` values are ``close_above_slow``,
    ``fast_above_slow``, and ``either``. These values are case-insensitive.
    """

    cleaned = raw_value.strip()
    if not cleaned:
        raise ValueError("Preset definitions cannot be empty.")

    values: dict[str, str] = {}
    for chunk in cleaned.split(","):
        part = chunk.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(
                "Invalid preset definition segment "
                f"'{part}'. Expected key=value pairs separated by commas."
            )
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key or not normalized_value:
            raise ValueError(
                f"Invalid preset definition segment '{part}'. Keys and values cannot be empty."
            )
        values[normalized_key] = normalized_value

    name = values.pop("name", "").strip()
    if not name:
        raise ValueError("Preset definitions must include a non-empty name=... field.")
    return BreakoutStrategyPreset.from_mapping(name, values)


def resolve_breakout_strategy_presets(
    signal_config: SignalConfig,
    risk_config: RiskConfig,
    *,
    configured_presets: Mapping[str, Any] | None = None,
    cli_preset_definitions: Sequence[str] = (),
    preset_names: Sequence[str] | None = None,
) -> list[BreakoutStrategyPreset]:
    """Resolve the preset catalog from defaults, config overrides, and CLI definitions."""

    resolved = build_default_breakout_presets(signal_config, risk_config)
    resolved.update(breakout_presets_from_mapping(configured_presets))
    for raw_definition in cli_preset_definitions:
        preset = breakout_preset_from_cli_definition(raw_definition)
        resolved[preset.name] = preset

    if preset_names:
        selected: list[BreakoutStrategyPreset] = []
        missing: list[str] = []
        for raw_name in preset_names:
            name = raw_name.strip()
            if not name:
                continue
            preset = resolved.get(name)
            if preset is None:
                missing.append(name)
                continue
            selected.append(preset)
        if missing:
            available = ", ".join(sorted(resolved))
            raise ValueError(
                f"Unknown preset name(s): {', '.join(missing)}. Available presets: {available}."
            )
        if not selected:
            raise ValueError("Preset selection cannot be empty.")
        return selected

    return list(resolved.values())


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
    # lag_average=1 shifts the rolling-average baseline back one bar before
    # dividing, so today's volume is compared against the average of the N bars
    # ending *yesterday*.  This keeps today's own volume out of its own
    # baseline and prevents any look-ahead bias in the RV reading.
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


def build_breakout_rationale(
    entry_reason: str,
    metadata: Mapping[str, Any],
) -> str:
    """Return a human-readable rationale that matches the actual RV policy."""

    parts = [entry_reason.replace("_", " ")]

    prior_high = _metadata_float(metadata.get("prior_high"))
    if prior_high is not None:
        parts.append(f"prior_high={prior_high:.2f}")

    relative_volume = _metadata_float(metadata.get("relative_volume"))
    if relative_volume is not None:
        relative_volume_threshold = _metadata_float(metadata.get("relative_volume_threshold"))
        relative_volume_confirmed = bool(metadata.get("relative_volume_confirmed"))
        relative_volume_required = bool(metadata.get("relative_volume_required"))
        rv_status = "confirmed" if relative_volume_confirmed else "not confirmed"
        rv_policy = "required" if relative_volume_required else "optional"
        if relative_volume_threshold is not None:
            parts.append(
                "relative_volume="
                f"{relative_volume:.2f} ({rv_policy}; threshold={relative_volume_threshold:.2f}; {rv_status})"
            )
        else:
            parts.append(f"relative_volume={relative_volume:.2f} ({rv_policy}; {rv_status})")

    earnings_days_away = _metadata_int(metadata.get("earnings_days_away"))
    earnings_date = _metadata_text(metadata.get("earnings_date"))
    earnings_watch_days = _metadata_int(metadata.get("earnings_watch_days"))
    if (
        earnings_days_away is not None
        and earnings_date is not None
        and (
            bool(metadata.get("is_earnings_risk"))
            or earnings_watch_days is None
            or earnings_days_away <= earnings_watch_days
        )
    ):
        parts.append(
            f"earnings={earnings_date} ({earnings_days_away} trading days)"
        )

    setup_persistence_days = _metadata_int(metadata.get("setup_persistence_days"))
    days_approved = _metadata_int(metadata.get("days_approved"))
    setup_notes: list[str] = []
    if bool(metadata.get("repeated_high_quality_signal")):
        setup_notes.append("high-confidence repeat signal")
    if setup_persistence_days is not None and setup_persistence_days > 1:
        setup_notes.append(f"persisted for {setup_persistence_days} sessions")
    if days_approved is not None and days_approved > 1:
        setup_notes.append(f"approved on {days_approved} sessions")
    if setup_notes:
        parts.append(f"setup={', '.join(setup_notes)}")

    sector_etf_symbol = _metadata_text(metadata.get("sector_etf_symbol"))
    sector_regime_passed = metadata.get("sector_regime_passed")
    relative_strength_vs_sector = _metadata_float(metadata.get("relative_strength_vs_sector"))
    sector_relative_strength_window = _metadata_int(
        metadata.get("sector_relative_strength_window")
    )
    if sector_etf_symbol is not None:
        sector_notes: list[str] = []
        if sector_regime_passed is True:
            sector_notes.append("trend supportive")
        elif sector_regime_passed is False:
            sector_notes.append("below trend filter")
        if relative_strength_vs_sector is not None:
            window_label = (
                f" over {sector_relative_strength_window}d"
                if sector_relative_strength_window is not None
                else ""
            )
            if relative_strength_vs_sector > 0:
                sector_notes.append(f"leads by {relative_strength_vs_sector:.1%}{window_label}")
            elif relative_strength_vs_sector < 0:
                sector_notes.append(
                    f"lags by {abs(relative_strength_vs_sector):.1%}{window_label}"
                )
        if sector_notes:
            parts.append(f"sector={sector_etf_symbol} ({', '.join(sector_notes)})")

    market_breadth_pct_above_200ma = _metadata_float(
        metadata.get("market_breadth_pct_above_200ma")
    )
    market_breadth_state = _metadata_text(metadata.get("market_breadth_state"))
    if market_breadth_pct_above_200ma is not None and market_breadth_state in {
        "weak",
        "strong",
    }:
        parts.append(
            f"breadth={market_breadth_state} ({market_breadth_pct_above_200ma:.0%} above 200d)"
        )

    volatility_regime_state = _metadata_text(metadata.get("volatility_regime_state"))
    vix_close = _metadata_float(metadata.get("vix_close"))
    vix_caution_threshold = _metadata_float(metadata.get("vix_caution_threshold"))
    vix_entry_block_threshold = _metadata_float(metadata.get("vix_entry_block_threshold"))
    if volatility_regime_state == "elevated":
        detail = f"VIX {vix_close:.1f}" if vix_close is not None else "VIX elevated"
        if vix_close is not None and vix_caution_threshold is not None:
            detail += f" above {vix_caution_threshold:.1f} caution"
        parts.append(f"volatility={volatility_regime_state} ({detail})")
    elif volatility_regime_state == "stressed":
        detail = f"VIX {vix_close:.1f}" if vix_close is not None else "VIX elevated"
        if vix_close is not None and vix_entry_block_threshold is not None:
            detail += f" above {vix_entry_block_threshold:.1f} block"
        parts.append(f"volatility={volatility_regime_state} ({detail})")

    return "; ".join(parts)


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
        "relative_volume_required": settings.require_relative_volume_confirmation,
        "relative_volume_policy": (
            "required"
            if settings.require_relative_volume_confirmation
            else "optional"
        ),
        "relative_volume_gate_passed": (
            bool(latest_row["passes_relative_volume"])
            or not settings.require_relative_volume_confirmation
        ),
        "relative_volume_window": settings.resolved_relative_volume_window,
        "regime_filter_enabled": settings.enable_regime_filter,
        "regime_filter_mode": settings.regime_filter_mode,
        "benchmark_symbol": settings.benchmark_symbol,
        "regime_passed": regime_passed,
        "atr": _optional_float(latest_row["atr"]),
        "earnings_entry_block_days": settings.earnings_entry_block_days,
        "earnings_watch_days": settings.earnings_watch_days,
        "require_sector_regime_for_entries": settings.require_sector_regime_for_entries,
        "sector_relative_strength_window": settings.sector_relative_strength_window,
        "sector_relative_strength_entry_reject_threshold": (
            settings.sector_relative_strength_entry_reject_threshold
        ),
        "sector_relative_strength_watch_threshold": (
            settings.sector_relative_strength_watch_threshold
        ),
    }


def _mapping_int(data: Mapping[str, Any], key: str) -> int:
    try:
        return int(data[key])
    except KeyError as exc:
        raise ValueError(f"Preset mapping is missing required field '{key}'.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Preset field '{key}' must be an integer.") from exc


def _mapping_float(data: Mapping[str, Any], key: str) -> float:
    try:
        return float(data[key])
    except KeyError as exc:
        raise ValueError(f"Preset mapping is missing required field '{key}'.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Preset field '{key}' must be a float.") from exc


def _mapping_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in data:
        return default

    value = data[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Preset field '{key}' must be a boolean.")


def _mapping_regime_filter_mode(data: Mapping[str, Any]) -> RegimeFilterMode:
    key = "regime_filter_mode"
    if key not in data:
        return "either"
    value = str(data[key]).strip().lower()
    if value not in VALID_REGIME_FILTER_MODES:
        raise ValueError(
            f"Preset field '{key}' must be one of {sorted(VALID_REGIME_FILTER_MODES)}, got '{value}'."
        )
    return value  # type: ignore[return-value]


def _metadata_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metadata_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _metadata_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
