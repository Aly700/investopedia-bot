"""Typed configuration loading for the Investopedia bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when repo configuration files are missing or malformed."""


@dataclass(frozen=True)
class UniverseConfig:
    """Universe selection parameters for research and signal generation."""

    min_price: float
    min_avg_dollar_volume: int
    max_symbols: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UniverseConfig":
        return cls(
            min_price=_as_float(data, "min_price"),
            min_avg_dollar_volume=_as_int(data, "min_avg_dollar_volume"),
            max_symbols=_as_int(data, "max_symbols"),
        )


@dataclass(frozen=True)
class SignalConfig:
    """Signal model parameters for daily-bar strategies."""

    breakout_lookback: int
    benchmark_symbol: str
    benchmark_sma_fast: int
    benchmark_sma_slow: int
    relative_volume_threshold: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SignalConfig":
        return cls(
            breakout_lookback=_as_int(data, "breakout_lookback"),
            benchmark_symbol=_as_str(data, "benchmark_symbol"),
            benchmark_sma_fast=_as_int(data, "benchmark_sma_fast"),
            benchmark_sma_slow=_as_int(data, "benchmark_sma_slow"),
            relative_volume_threshold=_as_float(data, "relative_volume_threshold"),
        )


@dataclass(frozen=True)
class RiskConfig:
    """Risk model configuration shared by research and backtests."""

    risk_per_trade: float
    atr_length: int
    initial_stop_atr: float
    trailing_stop_atr: float
    drawdown_risk_reduction_threshold: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RiskConfig":
        return cls(
            risk_per_trade=_as_float(data, "risk_per_trade"),
            atr_length=_as_int(data, "atr_length"),
            initial_stop_atr=_as_float(data, "initial_stop_atr"),
            trailing_stop_atr=_as_float(data, "trailing_stop_atr"),
            drawdown_risk_reduction_threshold=_as_float(
                data,
                "drawdown_risk_reduction_threshold",
            ),
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Top-level strategy configuration."""

    strategy_name: str
    universe: UniverseConfig
    signals: SignalConfig
    risk: RiskConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StrategyConfig":
        return cls(
            strategy_name=_as_str(data, "strategy_name"),
            universe=UniverseConfig.from_mapping(_as_mapping(data, "universe")),
            signals=SignalConfig.from_mapping(_as_mapping(data, "signals")),
            risk=RiskConfig.from_mapping(_as_mapping(data, "risk")),
        )


@dataclass(frozen=True)
class ProviderConfig:
    """Credential metadata for a market data provider."""

    api_key_env: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderConfig":
        return cls(api_key_env=_as_str(data, "api_key_env"))


@dataclass(frozen=True)
class DataSourcesConfig:
    """Data provider configuration for research workflows."""

    provider: str
    alphavantage: ProviderConfig
    tiingo: ProviderConfig
    polygon: ProviderConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DataSourcesConfig":
        return cls(
            provider=_as_str(data, "provider"),
            alphavantage=ProviderConfig.from_mapping(_as_mapping(data, "alphavantage")),
            tiingo=ProviderConfig.from_mapping(_as_mapping(data, "tiingo")),
            polygon=ProviderConfig.from_mapping(_as_mapping(data, "polygon")),
        )

    def active_provider(self) -> ProviderConfig:
        """Return the credential config for the selected provider."""

        provider_name = self.provider.lower()
        providers = {
            "alphavantage": self.alphavantage,
            "tiingo": self.tiingo,
            "polygon": self.polygon,
        }
        try:
            return providers[provider_name]
        except KeyError as exc:
            valid_providers = ", ".join(sorted(providers))
            raise ConfigError(
                f"Unsupported provider '{self.provider}'. Expected one of: {valid_providers}."
            ) from exc


@dataclass(frozen=True)
class CommissionsConfig:
    """Commission assumptions for simulator order types."""

    market_order: float
    stop_limit_order: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CommissionsConfig":
        return cls(
            market_order=_as_float(data, "market_order"),
            stop_limit_order=_as_float(data, "stop_limit_order"),
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """Fill and slippage assumptions for the simulator."""

    market_data_delay_minutes: int
    fill_model: str
    slippage_bps: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExecutionConfig":
        return cls(
            market_data_delay_minutes=_as_int(data, "market_data_delay_minutes"),
            fill_model=_as_str(data, "fill_model"),
            slippage_bps=_as_int(data, "slippage_bps"),
        )


@dataclass(frozen=True)
class RulesConfig:
    """Account-level simulator rules and constraints."""

    allow_shorting: bool
    allow_options: bool
    max_positions: int
    max_position_pct_equity: float
    no_averaging_down: bool
    use_quick_sell_guard: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RulesConfig":
        return cls(
            allow_shorting=_as_bool(data, "allow_shorting"),
            allow_options=_as_bool(data, "allow_options"),
            max_positions=_as_int(data, "max_positions"),
            max_position_pct_equity=_as_float(data, "max_position_pct_equity"),
            no_averaging_down=_as_bool(data, "no_averaging_down"),
            use_quick_sell_guard=_as_bool(data, "use_quick_sell_guard"),
        )


@dataclass(frozen=True)
class GameRulesConfig:
    """Simulator and portfolio rules."""

    starting_cash: float
    commissions: CommissionsConfig
    execution: ExecutionConfig
    rules: RulesConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GameRulesConfig":
        return cls(
            starting_cash=_as_float(data, "starting_cash"),
            commissions=CommissionsConfig.from_mapping(_as_mapping(data, "commissions")),
            execution=ExecutionConfig.from_mapping(_as_mapping(data, "execution")),
            rules=RulesConfig.from_mapping(_as_mapping(data, "rules")),
        )


@dataclass(frozen=True)
class AppConfig:
    """Merged repository configuration."""

    project_root: Path
    config_dir: Path
    strategy: StrategyConfig
    data_sources: DataSourcesConfig
    game_rules: GameRulesConfig

    def required_environment_variables(self) -> tuple[str, ...]:
        """Return the environment variables required by the active provider."""

        api_key_env = self.data_sources.active_provider().api_key_env
        return (api_key_env,) if api_key_env else ()

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly copy of the config."""

        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["config_dir"] = str(self.config_dir)
        return payload


@dataclass(frozen=True)
class EnvironmentValidationResult:
    """Result of checking environment variables required by the active provider."""

    provider: str
    required: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether all required variables are available."""

        return not self.missing


def default_project_root() -> Path:
    """Return the repository root inferred from this module location."""

    return Path(__file__).resolve().parents[2]


def default_config_dir(project_root: Path | None = None) -> Path:
    """Return the default config directory under the repository root."""

    root = project_root or default_project_root()
    return root / "config"


def default_env_file(project_root: Path | None = None) -> Path:
    """Return the default .env location under the repository root."""

    root = project_root or default_project_root()
    return root / ".env"


def load_app_config(config_dir: Path | None = None) -> AppConfig:
    """Load the full application config from the repo's YAML files."""

    project_root = default_project_root()
    resolved_config_dir = (config_dir or default_config_dir(project_root)).resolve()
    if not resolved_config_dir.exists():
        raise ConfigError(f"Config directory does not exist: {resolved_config_dir}")

    strategy = StrategyConfig.from_mapping(_load_yaml_file(resolved_config_dir / "strategy.yaml"))
    data_sources = DataSourcesConfig.from_mapping(
        _load_yaml_file(resolved_config_dir / "data_sources.yaml")
    )
    game_rules = GameRulesConfig.from_mapping(_load_yaml_file(resolved_config_dir / "game_rules.yaml"))

    return AppConfig(
        project_root=project_root,
        config_dir=resolved_config_dir,
        strategy=strategy,
        data_sources=data_sources,
        game_rules=game_rules,
    )


def validate_environment(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> EnvironmentValidationResult:
    """Validate environment variables required by the active data provider."""

    merged_environment = _merge_environment(env_file=env_file, environ=environ)
    required = config.required_environment_variables()
    present = tuple(name for name in required if merged_environment.get(name))
    missing = tuple(name for name in required if name not in present)
    return EnvironmentValidationResult(
        provider=config.data_sources.provider,
        required=required,
        present=present,
        missing=missing,
    )


def _load_yaml_file(path: Path) -> Mapping[str, Any]:
    """Read a YAML file and ensure it contains a top-level mapping."""

    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(f"Config file must contain a top-level mapping: {path}")
    return raw_data


def _read_env_file(env_file: Path | None) -> dict[str, str]:
    """Parse a simple .env file without mutating process environment."""

    if env_file is None or not env_file.exists():
        return {}

    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()

        key, separator, value = line.partition("=")
        if not separator:
            raise ConfigError(
                f"Invalid environment line in {env_file} at line {line_number}: {raw_line}"
            )

        cleaned_key = key.strip()
        if not cleaned_key:
            raise ConfigError(
                f"Invalid environment key in {env_file} at line {line_number}: {raw_line}"
            )

        parsed[cleaned_key] = value.strip().strip("'\"")
    return parsed


def _merge_environment(
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge .env values with the live process environment."""

    merged = _read_env_file(env_file)
    merged.update(dict(environ or os.environ))
    return merged


def _as_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected '{key}' to be a mapping.")
    return value


def _as_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected '{key}' to be a non-empty string.")
    return value.strip()


def _as_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Expected '{key}' to be an integer.")
    return int(value)


def _as_float(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Expected '{key}' to be numeric.")
    return float(value)


def _as_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Expected '{key}' to be a boolean.")
    return value
