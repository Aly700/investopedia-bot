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
