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
    trailing_stop_atr: float
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
            trailing_stop_atr=risk_config.trailing_stop_atr,
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
        if self.trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be greater than zero.")
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
            f"risk={self.risk_per_trade:g}"
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
    )
    confirmed = BreakoutStrategyPreset(
        name="confirmed_breakout",
        breakout_lookback=signal_config.breakout_lookback,
        relative_volume_threshold=signal_config.relative_volume_threshold,
        initial_stop_atr=risk_config.initial_stop_atr,
        trailing_stop_atr=risk_config.trailing_stop_atr,
        risk_per_trade=risk_config.risk_per_trade,
        require_relative_volume_confirmation=True,
    )
    conservative = BreakoutStrategyPreset(
        name="conservative_breakout",
        breakout_lookback=max(signal_config.breakout_lookback + 10, int(round(signal_config.breakout_lookback * 1.5))),
        relative_volume_threshold=signal_config.relative_volume_threshold + 0.5,
        initial_stop_atr=risk_config.initial_stop_atr + 0.5,
        trailing_stop_atr=risk_config.trailing_stop_atr + 0.5,
        risk_per_trade=max(risk_config.risk_per_trade * 0.5, 0.0025),
        require_relative_volume_confirmation=False,
    )
    confirmed_conservative = BreakoutStrategyPreset(
        name="confirmed_conservative_breakout",
        breakout_lookback=conservative.breakout_lookback,
        relative_volume_threshold=conservative.relative_volume_threshold,
        initial_stop_atr=conservative.initial_stop_atr,
        trailing_stop_atr=conservative.trailing_stop_atr,
        risk_per_trade=conservative.risk_per_trade,
        require_relative_volume_confirmation=True,
    )
    aggressive = BreakoutStrategyPreset(
        name="aggressive_breakout",
        breakout_lookback=max(5, signal_config.breakout_lookback - 10),
        relative_volume_threshold=max(1.0, signal_config.relative_volume_threshold - 0.5),
        initial_stop_atr=max(1.0, risk_config.initial_stop_atr - 0.5),
        trailing_stop_atr=max(1.0, risk_config.trailing_stop_atr - 0.5),
        risk_per_trade=min(0.05, risk_config.risk_per_trade * 1.5),
        require_relative_volume_confirmation=False,
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
    ``name=my_preset,breakout_lookback=20,relative_volume_threshold=1.5,initial_stop_atr=2.5,trailing_stop_atr=3.0,risk_per_trade=0.01``
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


def _metadata_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
