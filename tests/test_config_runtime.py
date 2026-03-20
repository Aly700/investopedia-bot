from pathlib import Path

from bot.config import load_app_config, validate_environment


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
    assert config.data_sources.provider == "polygon"
    assert config.game_rules.rules.max_positions == 4


def test_validate_environment_reads_env_file(tmp_path: Path) -> None:
    config = load_app_config()
    api_key_env = config.data_sources.active_provider().api_key_env
    env_file = tmp_path / ".env"
    env_file.write_text(f"{api_key_env}=test-key\n", encoding="utf-8")

    result = validate_environment(config, env_file=env_file, environ={})

    assert result.is_valid is True
    assert result.provider == config.data_sources.provider
    assert result.present == (api_key_env,)
    assert result.missing == ()
