"""Typed configuration loading for the Investopedia bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when repo configuration files are missing or malformed."""


REQUIRED_CONFIG_FILENAMES = (
    "strategy.yaml",
    "universe.yaml",
    "data_sources.yaml",
    "game_rules.yaml",
)


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
class UniverseProfileConfig:
    """Configurable filters for one generated candidate universe."""

    description: str
    active_only: bool
    allowed_exchanges: tuple[str, ...]
    allowed_ticker_types: tuple[str, ...]
    min_price: float | None
    min_average_daily_volume: int | None
    min_dollar_volume: int | None
    min_market_cap: int | None
    allowed_sectors: tuple[str, ...]
    allowed_industries: tuple[str, ...]
    include_etfs: bool
    include_adrs: bool
    common_stock_only: bool
    exclude_weird_instruments: bool
    max_symbols: int | None
    include_symbols: tuple[str, ...]
    exclude_symbols: tuple[str, ...]
    preferred_symbol_groups: dict[str, tuple[str, ...]]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UniverseProfileConfig":
        include_symbols = _as_text_tuple(data, "include_symbols", default=())
        exclude_symbols = _as_text_tuple(data, "exclude_symbols", default=())
        preferred_symbol_groups = _as_symbol_group_mapping(
            data,
            "preferred_symbol_groups",
        )
        overlap = sorted(set(include_symbols) & set(exclude_symbols))
        if overlap:
            joined = ", ".join(overlap)
            raise ConfigError(
                f"Universe profile include_symbols and exclude_symbols overlap: {joined}"
            )

        return cls(
            description=_as_str(data, "description", default=""),
            active_only=_as_bool(data, "active_only", default=True),
            allowed_exchanges=_as_text_tuple(data, "allowed_exchanges", default=()),
            allowed_ticker_types=_as_text_tuple(data, "allowed_ticker_types", default=()),
            min_price=_as_optional_float(data, "min_price"),
            min_average_daily_volume=_as_optional_int(data, "min_average_daily_volume"),
            min_dollar_volume=_as_optional_int(data, "min_dollar_volume"),
            min_market_cap=_as_optional_int(data, "min_market_cap"),
            allowed_sectors=_as_text_tuple(data, "allowed_sectors", default=()),
            allowed_industries=_as_text_tuple(data, "allowed_industries", default=()),
            include_etfs=_as_bool(data, "include_etfs", default=False),
            include_adrs=_as_bool(data, "include_adrs", default=False),
            common_stock_only=_as_bool(data, "common_stock_only", default=True),
            exclude_weird_instruments=_as_bool(
                data,
                "exclude_weird_instruments",
                default=True,
            ),
            max_symbols=_as_optional_int(data, "max_symbols"),
            include_symbols=include_symbols,
            exclude_symbols=exclude_symbols,
            preferred_symbol_groups=preferred_symbol_groups,
        )


@dataclass(frozen=True)
class UniverseMasterConfig:
    """Stage-1 and stage-2 settings for the universe-builder pipeline."""

    market: str
    locale: str
    active_only: bool
    allowed_ticker_types: tuple[str, ...]
    liquidity_lookback_days: int
    reference_cache_dir: str
    processed_output_dir: str
    candidate_output_dir: str
    raw_reference_filename: str
    master_universe_filename: str
    profile_summary_dir: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UniverseMasterConfig":
        return cls(
            market=_as_str(data, "market"),
            locale=_as_str(data, "locale"),
            active_only=_as_bool(data, "active_only", default=True),
            allowed_ticker_types=_as_text_tuple(
                data,
                "allowed_ticker_types",
                default=(),
            ),
            liquidity_lookback_days=_as_int(data, "liquidity_lookback_days"),
            reference_cache_dir=_as_str(data, "reference_cache_dir"),
            processed_output_dir=_as_str(data, "processed_output_dir"),
            candidate_output_dir=_as_str(data, "candidate_output_dir"),
            raw_reference_filename=_as_str(data, "raw_reference_filename"),
            master_universe_filename=_as_str(data, "master_universe_filename"),
            profile_summary_dir=_as_str(data, "profile_summary_dir"),
        )


@dataclass(frozen=True)
class UniverseBuilderConfig:
    """Universe-builder configuration, including profiles and output paths."""

    master: UniverseMasterConfig
    profiles: dict[str, UniverseProfileConfig]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UniverseBuilderConfig":
        master = UniverseMasterConfig.from_mapping(_as_mapping(data, "master_universe"))
        raw_profiles = _as_mapping(data, "profiles")
        if not raw_profiles:
            raise ConfigError("Expected at least one configured universe profile.")

        profiles: dict[str, UniverseProfileConfig] = {}
        for profile_name, profile_payload in raw_profiles.items():
            if not isinstance(profile_name, str) or not profile_name.strip():
                raise ConfigError("Universe profile names must be non-empty strings.")
            if not isinstance(profile_payload, dict):
                raise ConfigError(
                    f"Universe profile '{profile_name}' must be a mapping."
                )
            profiles[profile_name.strip()] = UniverseProfileConfig.from_mapping(profile_payload)

        return cls(master=master, profiles=profiles)


@dataclass(frozen=True)
class SignalConfig:
    """Signal model parameters for daily-bar strategies."""

    breakout_lookback: int
    benchmark_symbol: str
    benchmark_sma_fast: int
    benchmark_sma_slow: int
    relative_volume_threshold: float
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
    def from_mapping(cls, data: Mapping[str, Any]) -> "SignalConfig":
        earnings_entry_block_days = _as_optional_int(data, "earnings_entry_block_days")
        earnings_watch_days = _as_optional_int(data, "earnings_watch_days")
        market_breadth_entry_floor_200ma = _as_optional_float(
            data,
            "market_breadth_entry_floor_200ma",
        )
        vix_caution_threshold = _as_optional_float(data, "vix_caution_threshold")
        vix_entry_block_threshold = _as_optional_float(
            data,
            "vix_entry_block_threshold",
        )
        sector_relative_strength_window = _as_optional_int(data, "sector_relative_strength_window")
        sector_relative_strength_entry_reject_threshold = _as_optional_float(
            data,
            "sector_relative_strength_entry_reject_threshold",
        )
        sector_relative_strength_watch_threshold = _as_optional_float(
            data,
            "sector_relative_strength_watch_threshold",
        )
        max_positions_per_sector = _as_optional_int(data, "max_positions_per_sector")
        max_same_industry_positions = _as_optional_int(
            data,
            "max_same_industry_positions",
        )
        max_sector_notional_pct = _as_optional_float(data, "max_sector_notional_pct")
        return cls(
            breakout_lookback=_as_int(data, "breakout_lookback"),
            benchmark_symbol=_as_str(data, "benchmark_symbol"),
            benchmark_sma_fast=_as_int(data, "benchmark_sma_fast"),
            benchmark_sma_slow=_as_int(data, "benchmark_sma_slow"),
            relative_volume_threshold=_as_float(data, "relative_volume_threshold"),
            earnings_entry_block_days=(
                earnings_entry_block_days
                if earnings_entry_block_days is not None
                else 3
            ),
            earnings_watch_days=(
                earnings_watch_days
                if earnings_watch_days is not None
                else 7
            ),
            market_breadth_entry_floor_200ma=market_breadth_entry_floor_200ma,
            vix_caution_threshold=(
                vix_caution_threshold
                if vix_caution_threshold is not None
                else 25.0
            ),
            vix_entry_block_threshold=(
                vix_entry_block_threshold
                if vix_entry_block_threshold is not None
                else 30.0
            ),
            require_sector_regime_for_entries=_as_bool(
                data,
                "require_sector_regime_for_entries",
                default=True,
            ),
            sector_relative_strength_window=(
                sector_relative_strength_window
                if sector_relative_strength_window is not None
                else 20
            ),
            sector_relative_strength_entry_reject_threshold=(
                sector_relative_strength_entry_reject_threshold
                if sector_relative_strength_entry_reject_threshold is not None
                else -0.05
            ),
            sector_relative_strength_watch_threshold=(
                sector_relative_strength_watch_threshold
                if sector_relative_strength_watch_threshold is not None
                else -0.05
            ),
            max_positions_per_sector=(
                max_positions_per_sector
                if max_positions_per_sector is not None
                else 3
            ),
            max_same_industry_positions=(
                max_same_industry_positions
                if max_same_industry_positions is not None
                else 2
            ),
            max_sector_notional_pct=(
                max_sector_notional_pct
                if max_sector_notional_pct is not None
                else None
            ),
        )

    def __post_init__(self) -> None:
        if self.breakout_lookback <= 0:
            raise ConfigError("Expected 'breakout_lookback' to be greater than zero.")
        if self.benchmark_sma_fast <= 0:
            raise ConfigError("Expected 'benchmark_sma_fast' to be greater than zero.")
        if self.benchmark_sma_slow <= 0:
            raise ConfigError("Expected 'benchmark_sma_slow' to be greater than zero.")
        if self.relative_volume_threshold <= 0:
            raise ConfigError("Expected 'relative_volume_threshold' to be greater than zero.")
        if self.earnings_entry_block_days <= 0:
            raise ConfigError("Expected 'earnings_entry_block_days' to be greater than zero.")
        if self.earnings_watch_days <= 0:
            raise ConfigError("Expected 'earnings_watch_days' to be greater than zero.")
        if self.market_breadth_entry_floor_200ma is not None and (
            self.market_breadth_entry_floor_200ma <= 0
            or self.market_breadth_entry_floor_200ma >= 1
        ):
            raise ConfigError(
                "Expected 'market_breadth_entry_floor_200ma' to be between zero and one when provided."
            )
        if self.vix_caution_threshold is not None and self.vix_caution_threshold <= 0:
            raise ConfigError(
                "Expected 'vix_caution_threshold' to be greater than zero when provided."
            )
        if self.vix_entry_block_threshold is not None and self.vix_entry_block_threshold <= 0:
            raise ConfigError(
                "Expected 'vix_entry_block_threshold' to be greater than zero when provided."
            )
        if (
            self.vix_caution_threshold is not None
            and self.vix_entry_block_threshold is not None
            and self.vix_caution_threshold >= self.vix_entry_block_threshold
        ):
            raise ConfigError(
                "Expected 'vix_caution_threshold' to be less than 'vix_entry_block_threshold'."
            )
        if self.sector_relative_strength_window <= 0:
            raise ConfigError(
                "Expected 'sector_relative_strength_window' to be greater than zero."
            )
        if self.sector_relative_strength_entry_reject_threshold > 0:
            raise ConfigError(
                "Expected 'sector_relative_strength_entry_reject_threshold' to be less than or equal to zero."
            )
        if self.sector_relative_strength_watch_threshold > 0:
            raise ConfigError(
                "Expected 'sector_relative_strength_watch_threshold' to be less than or equal to zero."
            )
        if self.max_positions_per_sector is not None and self.max_positions_per_sector <= 0:
            raise ConfigError(
                "Expected 'max_positions_per_sector' to be greater than zero when provided."
            )
        if (
            self.max_same_industry_positions is not None
            and self.max_same_industry_positions <= 0
        ):
            raise ConfigError(
                "Expected 'max_same_industry_positions' to be greater than zero when provided."
            )
        if self.max_sector_notional_pct is not None and (
            self.max_sector_notional_pct <= 0
            or self.max_sector_notional_pct >= 1
        ):
            raise ConfigError(
                "Expected 'max_sector_notional_pct' to be between zero and one when provided."
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
    universe_builder: UniverseBuilderConfig
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
    """Return the runtime root that owns the active config bundle."""

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        discovered_root = _find_project_root(start)
        if discovered_root is not None:
            return discovered_root
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

    if config_dir is None:
        project_root = default_project_root()
        resolved_config_dir = default_config_dir(project_root).resolve()
    else:
        resolved_config_dir = config_dir.resolve()
        project_root = resolved_config_dir.parent
    if not resolved_config_dir.exists():
        raise ConfigError(f"Config directory does not exist: {resolved_config_dir}")

    strategy = StrategyConfig.from_mapping(_load_yaml_file(resolved_config_dir / "strategy.yaml"))
    universe_builder = UniverseBuilderConfig.from_mapping(
        _load_yaml_file(resolved_config_dir / "universe.yaml")
    )
    data_sources = DataSourcesConfig.from_mapping(
        _load_yaml_file(resolved_config_dir / "data_sources.yaml")
    )
    game_rules = GameRulesConfig.from_mapping(_load_yaml_file(resolved_config_dir / "game_rules.yaml"))

    return AppConfig(
        project_root=project_root,
        config_dir=resolved_config_dir,
        strategy=strategy,
        universe_builder=universe_builder,
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

        parsed[cleaned_key] = _strip_paired_quotes(value.strip())
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


def _strip_paired_quotes(value: str) -> str:
    """Strip surrounding quotes only when they form a matching pair."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _find_project_root(start: Path) -> Path | None:
    """Search upward for a directory that contains the full config bundle."""

    resolved_start = start.resolve()
    candidates = (
        (resolved_start,)
        if resolved_start.is_dir()
        else ()
    )
    search_roots = candidates or (resolved_start.parent,)
    for root in search_roots:
        for candidate in (root, *root.parents):
            if _has_required_config_bundle(candidate):
                return candidate
    return None


def _has_required_config_bundle(project_root: Path) -> bool:
    """Return whether the path has the full expected config directory."""

    config_dir = project_root / "config"
    return config_dir.is_dir() and all(
        (config_dir / filename).is_file()
        for filename in REQUIRED_CONFIG_FILENAMES
    )


def _as_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected '{key}' to be a mapping.")
    return value


def _as_str(data: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"Expected '{key}' to be a string.")
    if default == "" and value == "":
        return ""
    if not value.strip():
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


def _as_optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Expected '{key}' to be an integer when provided.")
    return int(value)


def _as_optional_float(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Expected '{key}' to be numeric when provided.")
    return float(value)


def _as_bool(data: Mapping[str, Any], key: str, *, default: bool | None = None) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Expected '{key}' to be a boolean.")
    return value


def _as_text_tuple(
    data: Mapping[str, Any],
    key: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    value = data.get(key, list(default))
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"Expected '{key}' to be a list of strings.")

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"Expected all '{key}' values to be non-empty strings.")
        cleaned.append(item.strip())
    return tuple(cleaned)


def _as_symbol_group_mapping(
    data: Mapping[str, Any],
    key: str,
) -> dict[str, tuple[str, ...]]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Expected '{key}' to be a mapping of preferred symbols to symbol lists.")

    cleaned_mapping: dict[str, tuple[str, ...]] = {}
    seen_symbols: dict[str, str] = {}
    for preferred_symbol, raw_group in value.items():
        if not isinstance(preferred_symbol, str) or not preferred_symbol.strip():
            raise ConfigError(f"Expected '{key}' mapping keys to be non-empty strings.")
        if not isinstance(raw_group, list):
            raise ConfigError(f"Expected '{key}.{preferred_symbol}' to be a list of symbols.")

        normalized_preferred = preferred_symbol.strip().upper()
        group_symbols: list[str] = []
        for raw_symbol in raw_group:
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                raise ConfigError(
                    f"Expected all '{key}.{preferred_symbol}' values to be non-empty strings."
                )
            normalized_symbol = raw_symbol.strip().upper()
            if normalized_symbol not in group_symbols:
                group_symbols.append(normalized_symbol)

        if normalized_preferred not in group_symbols:
            group_symbols.insert(0, normalized_preferred)

        for symbol in group_symbols:
            existing_group = seen_symbols.get(symbol)
            if existing_group is not None and existing_group != normalized_preferred:
                raise ConfigError(
                    f"Symbol '{symbol}' appears in multiple '{key}' groups: "
                    f"{existing_group}, {normalized_preferred}."
                )
            seen_symbols[symbol] = normalized_preferred

        cleaned_mapping[normalized_preferred] = tuple(group_symbols)

    return cleaned_mapping
