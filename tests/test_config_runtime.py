from pathlib import Path

from bot.config import _read_env_file, default_project_root, load_app_config, validate_environment


def test_load_app_config_reads_repo_files() -> None:
    config = load_app_config()

    assert config.strategy.strategy_name == "breakout_momentum"
    assert "broad_momentum" in config.universe_builder.profiles
    assert "growth_momentum" in config.universe_builder.profiles
    assert "quality_liquid" in config.universe_builder.profiles
    growth_profile = config.universe_builder.profiles["growth_momentum"]
    assert growth_profile.common_stock_only is True
    assert growth_profile.include_etfs is False
    assert growth_profile.include_adrs is False
    assert growth_profile.min_market_cap == 10_000_000_000
    assert "semiconductor" in growth_profile.allowed_industries
    assert growth_profile.include_symbols[:3] == ("AMZN", "GOOGL", "META")
    quality_profile = config.universe_builder.profiles["quality_liquid"]
    assert quality_profile.common_stock_only is True
    assert quality_profile.include_etfs is False
    assert quality_profile.include_adrs is False
    assert quality_profile.min_market_cap == 15_000_000_000
    assert quality_profile.min_dollar_volume == 60_000_000
    assert "HOOD" in quality_profile.exclude_symbols
    assert config.data_sources.provider == "alpaca"
    assert config.data_sources.resolved_roles().stream_market_data == "alpaca"
    assert config.data_sources.resolved_roles().historical_bars == "alpaca"
    assert config.data_sources.resolved_roles().reference_data == "polygon"
    assert config.data_sources.resolved_roles().earnings_calendar == "polygon"
    assert config.data_sources.resolved_roles().execution_broker == "alpaca"
    assert config.data_sources.resolved_roles().broker_update_stream is None
    assert config.game_rules.rules.max_positions == 4


def test_validate_environment_reads_env_file(tmp_path: Path) -> None:
    config = load_app_config()
    required_names = config.required_environment_variables()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "".join(f"{name}=test-{index}\n" for index, name in enumerate(required_names, start=1)),
        encoding="utf-8",
    )

    result = validate_environment(config, env_file=env_file, environ={})

    assert result.is_valid is True
    assert result.provider == config.data_sources.provider
    assert result.role_assignments == {
        "stream_market_data": "alpaca",
        "historical_bars": "alpaca",
        "reference_data": "polygon",
        "earnings_calendar": "polygon",
    }
    assert result.present == required_names
    assert result.missing == ()


def test_read_env_file_strips_only_paired_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'DOUBLE_QUOTED="double value"\n'
        "SINGLE_QUOTED='single value'\n"
        "UNQUOTED=plain-value\n",
        encoding="utf-8",
    )

    parsed = _read_env_file(env_file)

    assert parsed["DOUBLE_QUOTED"] == "double value"
    assert parsed["SINGLE_QUOTED"] == "single value"
    assert parsed["UNQUOTED"] == "plain-value"


def test_load_app_config_uses_explicit_config_dir_parent_as_project_root(tmp_path: Path) -> None:
    config_dir = _copy_repo_config_bundle(tmp_path / "config")

    config = load_app_config(config_dir=config_dir)

    assert config.config_dir == config_dir.resolve()
    assert config.project_root == tmp_path.resolve()


def test_load_app_config_preserves_legacy_single_provider_defaults_when_roles_missing(
    tmp_path: Path,
) -> None:
    config_dir = _copy_repo_config_bundle(tmp_path / "config")
    (config_dir / "data_sources.yaml").write_text(
        (
            "provider: tiingo\n\n"
            "alphavantage:\n"
            "  api_key_env: ALPHAVANTAGE_API_KEY\n\n"
            "tiingo:\n"
            "  api_key_env: TIINGO_API_KEY\n\n"
            "polygon:\n"
            "  api_key_env: POLYGON_API_KEY\n\n"
            "alpaca:\n"
            "  api_key_env: ALPACA_API_KEY_ID\n"
            "  api_secret_env: ALPACA_SECRET_KEY\n"
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_dir=config_dir)
    roles = config.data_sources.resolved_roles()

    assert roles.historical_bars == "tiingo"
    assert roles.reference_data == "tiingo"
    assert roles.earnings_calendar == "tiingo"
    assert roles.stream_market_data is None
    assert roles.execution_broker is None
    assert roles.broker_update_stream is None


def test_load_app_config_reads_role_based_provider_overrides_with_legacy_fallbacks(
    tmp_path: Path,
) -> None:
    config_dir = _copy_repo_config_bundle(tmp_path / "config")
    (config_dir / "data_sources.yaml").write_text(
        (
            "provider: polygon\n\n"
            "roles:\n"
            "  stream_market_data: alpaca\n"
            "  historical_bars: tiingo\n"
            "  earnings_calendar: polygon\n"
            "  execution_broker: alpaca\n\n"
            "alphavantage:\n"
            "  api_key_env: ALPHAVANTAGE_API_KEY\n\n"
            "tiingo:\n"
            "  api_key_env: TIINGO_API_KEY\n\n"
            "polygon:\n"
            "  api_key_env: POLYGON_API_KEY\n\n"
            "alpaca:\n"
            "  api_key_env: ALPACA_API_KEY_ID\n"
            "  api_secret_env: ALPACA_SECRET_KEY\n"
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_dir=config_dir)
    roles = config.data_sources.resolved_roles()

    assert roles.stream_market_data == "alpaca"
    assert roles.historical_bars == "tiingo"
    assert roles.reference_data == "polygon"
    assert roles.earnings_calendar == "polygon"
    assert roles.execution_broker == "alpaca"
    assert roles.broker_update_stream is None
    assert config.data_sources.provider_name_for_role("historical_bars") == "tiingo"
    assert config.data_sources.active_provider().api_key_env == "TIINGO_API_KEY"


def test_validate_environment_requires_both_alpaca_historical_credentials(tmp_path: Path) -> None:
    config_dir = _copy_repo_config_bundle(tmp_path / "config")
    (config_dir / "data_sources.yaml").write_text(
        (
            "provider: polygon\n\n"
            "roles:\n"
            "  historical_bars: alpaca\n"
            "  reference_data: polygon\n"
            "  earnings_calendar: polygon\n\n"
            "alphavantage:\n"
            "  api_key_env: ALPHAVANTAGE_API_KEY\n\n"
            "tiingo:\n"
            "  api_key_env: TIINGO_API_KEY\n\n"
            "polygon:\n"
            "  api_key_env: POLYGON_API_KEY\n\n"
            "alpaca:\n"
            "  api_key_env: ALPACA_API_KEY_ID\n"
            "  api_secret_env: ALPACA_SECRET_KEY\n"
        ),
        encoding="utf-8",
    )
    config = load_app_config(config_dir=config_dir)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY_ID=test-key\nPOLYGON_API_KEY=polygon-key\n",
        encoding="utf-8",
    )

    result = validate_environment(config, env_file=env_file, environ={})

    assert result.provider == "alpaca"
    assert result.present == ("ALPACA_API_KEY_ID", "POLYGON_API_KEY")
    assert result.missing == ("ALPACA_SECRET_KEY",)


def test_default_project_root_prefers_runtime_config_bundle_in_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = _copy_repo_config_bundle(tmp_path / "config")
    monkeypatch.chdir(tmp_path)

    project_root = default_project_root()
    config = load_app_config()

    assert project_root == tmp_path.resolve()
    assert config.project_root == tmp_path.resolve()
    assert config.config_dir == config_dir.resolve()


def _copy_repo_config_bundle(destination: Path) -> Path:
    source_dir = Path(__file__).resolve().parents[1] / "config"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("strategy.yaml", "universe.yaml", "data_sources.yaml", "game_rules.yaml"):
        (destination / filename).write_text(
            (source_dir / filename).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return destination
