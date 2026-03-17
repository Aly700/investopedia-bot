from pathlib import Path

from bot.config import load_app_config, validate_environment


def test_load_app_config_reads_repo_files() -> None:
    config = load_app_config()

    assert config.strategy.strategy_name == "breakout_momentum"
    assert config.data_sources.provider == "alphavantage"
    assert config.game_rules.rules.max_positions == 4


def test_validate_environment_reads_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALPHAVANTAGE_API_KEY=test-key\n", encoding="utf-8")

    config = load_app_config()
    result = validate_environment(config, env_file=env_file, environ={})

    assert result.is_valid is True
    assert result.present == ("ALPHAVANTAGE_API_KEY",)
    assert result.missing == ()
