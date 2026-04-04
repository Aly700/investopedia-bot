from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from bot.config import load_app_config
from bot.providers.bundle import build_provider_bundle, provider_role_metadata_from_config


def test_build_provider_bundle_resolves_roles_and_degraded_metadata(
    tmp_path: Path,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        (
            "ALPACA_API_KEY_ID=alpaca-key\n"
            "ALPACA_SECRET_KEY=alpaca-secret\n"
            "POLYGON_API_KEY=polygon-key\n"
        ),
        encoding="utf-8",
    )

    bundle = build_provider_bundle(config, env_file=env_file)
    metadata = bundle.provider_metadata(
        include_stream=True,
        include_execution=True,
        stream_provider=config.data_sources.resolved_roles().stream_market_data,
        execution_broker=config.data_sources.resolved_roles().execution_broker,
        broker_update_stream_provider=config.data_sources.resolved_roles().broker_update_stream,
    )

    assert bundle.role_assignments["stream_market_data"] == "alpaca"
    assert bundle.role_assignments["historical_bars"] == "alpaca"
    assert bundle.role_assignments["reference_data"] == "polygon"
    assert bundle.role_assignments["earnings_calendar"] == "polygon"
    assert bundle.historical_bars_provider is not None
    assert bundle.historical_bars_provider.provider_name == "alpaca"
    assert bundle.reference_data_provider is not None
    assert bundle.reference_data_provider.provider_name == "polygon"
    assert bundle.earnings_calendar_provider is not None
    assert bundle.earnings_calendar_provider.provider_name == "polygon"
    assert metadata["stream_provider"] == "alpaca"
    assert metadata["historical_provider"] == "alpaca"
    assert metadata["reference_provider"] == "polygon"
    assert metadata["earnings_provider"] == "polygon"
    assert metadata["execution_broker"] == "alpaca"
    assert metadata["broker_update_stream_provider"] is None
    assert "historical_bars" in metadata["degraded_provider_roles"]
    assert metadata["provider_roles"]["historical_bars"]["degraded"] is True
    assert (
        "VIX/index-style symbols"
        in metadata["provider_roles"]["historical_bars"]["reason"]
    )
    assert metadata["provider_roles"]["reference_data"]["available"] is True
    assert metadata["provider_roles"]["earnings_calendar"]["available"] is True


def test_provider_role_metadata_from_legacy_config_preserves_legacy_research_provider() -> None:
    legacy_config = SimpleNamespace(data_sources=SimpleNamespace(provider="polygon"))

    metadata = provider_role_metadata_from_config(
        legacy_config,
        include_stream=True,
        include_execution=True,
        stream_provider="alpaca",
    )

    assert metadata["stream_provider"] == "alpaca"
    assert metadata["historical_provider"] == "polygon"
    assert metadata["reference_provider"] == "polygon"
    assert metadata["earnings_provider"] == "polygon"
    assert metadata["execution_broker"] is None
    assert metadata["broker_update_stream_provider"] is None
    assert "historical_bars" in metadata["degraded_provider_roles"]
    assert "reference_data" in metadata["unavailable_provider_roles"]
    assert "earnings_calendar" in metadata["unavailable_provider_roles"]


def test_provider_bundle_marks_configured_broker_update_stream_unavailable_until_wired(
    tmp_path: Path,
) -> None:
    base_config = replace(load_app_config(), project_root=tmp_path)
    config = replace(
        base_config,
        data_sources=replace(
            base_config.data_sources,
            roles=base_config.data_sources.roles.__class__(
                stream_market_data="alpaca",
                historical_bars="alpaca",
                reference_data="polygon",
                earnings_calendar="polygon",
                execution_broker="alpaca",
                broker_update_stream="alpaca",
            ),
        ),
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        (
            "ALPACA_API_KEY_ID=alpaca-key\n"
            "ALPACA_SECRET_KEY=alpaca-secret\n"
            "POLYGON_API_KEY=polygon-key\n"
        ),
        encoding="utf-8",
    )

    bundle = build_provider_bundle(config, env_file=env_file)
    metadata = bundle.provider_metadata(
        include_stream=True,
        include_execution=True,
        stream_provider="alpaca",
        execution_broker="alpaca",
        broker_update_stream_provider="alpaca",
    )

    broker_stream_status = metadata["provider_roles"]["broker_update_stream"]
    assert broker_stream_status["provider"] == "alpaca"
    assert broker_stream_status["configured"] is True
    assert broker_stream_status["available"] is False
    assert broker_stream_status["degraded"] is True
    assert broker_stream_status["implemented"] is False
    assert "broker-update stream adapter is wired" in broker_stream_status["reason"]
    assert "broker_update_stream" in metadata["degraded_provider_roles"]
    assert "broker_update_stream" in metadata["unavailable_provider_roles"]
