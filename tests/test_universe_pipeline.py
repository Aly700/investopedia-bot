from dataclasses import replace
from datetime import date
import json
import logging
from pathlib import Path

import pandas as pd

import bot.main as main_module
from bot.config import UniverseProfileConfig, load_app_config
from bot.data.normalize import empty_daily_bar_frame
from bot.data.providers import DailyBarProvider
from bot.data.universe_pipeline import apply_universe_filters, write_candidate_symbols


class FakeDailyBarProvider(DailyBarProvider):
    provider_name = "fake"

    def __init__(self, *, frames_by_symbol: dict[str, pd.DataFrame], cache_dir: Path) -> None:
        super().__init__(api_key="test-key", cache_dir=cache_dir)
        self.frames_by_symbol = frames_by_symbol

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        return self.frames_by_symbol.get(symbol, empty_daily_bar_frame()).copy()


def test_apply_universe_filters_respects_profile_rules() -> None:
    master_frame = pd.DataFrame(
        [
            _master_row(
                symbol="SOFT",
                exchange="XNAS",
                industry="Application software",
                sector="Technology",
                price=45.0,
                average_daily_volume=1_500_000,
                dollar_volume=67_500_000,
                market_cap=12_000_000_000,
            ),
            _master_row(
                symbol="CHIP",
                exchange="XNYS",
                industry="Semiconductors",
                sector="Technology",
                price=80.0,
                average_daily_volume=900_000,
                dollar_volume=72_000_000,
                market_cap=25_000_000_000,
            ),
            _master_row(
                symbol="SMALL",
                exchange="XNAS",
                industry="Application software",
                sector="Technology",
                price=6.0,
                average_daily_volume=300_000,
                dollar_volume=1_800_000,
                market_cap=900_000_000,
            ),
            _master_row(
                symbol="ETF1",
                exchange="ARCX",
                type_code="ETF",
                industry="Exchange traded fund",
                sector="Fund",
                price=50.0,
                average_daily_volume=3_000_000,
                dollar_volume=150_000_000,
                market_cap=5_000_000_000,
                is_etf=True,
                is_common_stock=False,
            ),
        ]
    )
    profile = UniverseProfileConfig(
        description="Technology common-stock screen",
        active_only=True,
        allowed_exchanges=("XNAS", "XNYS"),
        allowed_ticker_types=("CS",),
        min_price=10.0,
        min_average_daily_volume=500_000,
        min_dollar_volume=20_000_000,
        min_market_cap=2_000_000_000,
        allowed_sectors=("technology",),
        allowed_industries=("software", "semiconductor"),
        include_etfs=False,
        include_adrs=False,
        common_stock_only=True,
        exclude_weird_instruments=True,
        max_symbols=None,
        include_symbols=(),
        exclude_symbols=(),
        preferred_symbol_groups={},
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="tech_profile",
        profile_config=profile,
    )

    assert result.symbols == ("CHIP", "SOFT")


def test_apply_universe_filters_supports_manual_include_and_exclude_overrides() -> None:
    master_frame = pd.DataFrame(
        [
            _master_row(symbol="KEEP", price=40.0, average_daily_volume=800_000, dollar_volume=32_000_000),
            _master_row(symbol="DROP", price=35.0, average_daily_volume=900_000, dollar_volume=31_500_000),
        ]
    )
    profile = UniverseProfileConfig(
        description="Manual override profile",
        active_only=True,
        allowed_exchanges=("XNAS", "XNYS", "ARCX"),
        allowed_ticker_types=("CS",),
        min_price=20.0,
        min_average_daily_volume=500_000,
        min_dollar_volume=10_000_000,
        min_market_cap=1_000_000_000,
        allowed_sectors=(),
        allowed_industries=(),
        include_etfs=False,
        include_adrs=False,
        common_stock_only=True,
        exclude_weird_instruments=True,
        max_symbols=5,
        include_symbols=("MISSING", "KEEP"),
        exclude_symbols=("DROP",),
        preferred_symbol_groups={},
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="manual_merge",
        profile_config=profile,
    )

    assert result.symbols == ("MISSING", "KEEP")
    assert result.summary["missing_force_include_symbols"] == ["MISSING"]


def test_common_stock_only_filter_excludes_etfs_adrs_and_weird_instruments() -> None:
    master_frame = pd.DataFrame(
        [
            _master_row(symbol="CS1"),
            _master_row(
                symbol="ETF1",
                type_code="ETF",
                industry="Exchange traded fund",
                is_etf=True,
                is_common_stock=False,
            ),
            _master_row(
                symbol="ADR1",
                type_code="ADRC",
                industry="Depositary shares",
                is_adr=True,
                is_common_stock=False,
            ),
            _master_row(
                symbol="UNIT.U",
                type_code="UNIT",
                industry="Blank check company",
                is_common_stock=False,
            ),
        ]
    )
    profile = UniverseProfileConfig(
        description="Common stocks only",
        active_only=True,
        allowed_exchanges=("XNAS", "XNYS", "ARCX"),
        allowed_ticker_types=(),
        min_price=None,
        min_average_daily_volume=None,
        min_dollar_volume=None,
        min_market_cap=None,
        allowed_sectors=(),
        allowed_industries=(),
        include_etfs=False,
        include_adrs=False,
        common_stock_only=True,
        exclude_weird_instruments=True,
        max_symbols=None,
        include_symbols=(),
        exclude_symbols=(),
        preferred_symbol_groups={},
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="common_stock_only",
        profile_config=profile,
    )

    assert result.symbols == ("CS1",)


def test_write_candidate_symbols_uses_plain_text_one_symbol_per_line(tmp_path: Path) -> None:
    output_path = tmp_path / "candidate_symbols_test.txt"

    written_path = write_candidate_symbols(["aapl", "MSFT", "nvda"], output_path)

    assert written_path == output_path.resolve()
    assert output_path.read_text(encoding="utf-8") == "AAPL\nMSFT\nNVDA\n"


def test_preferred_symbol_groups_deduplicate_dual_class_symbols_deterministically() -> None:
    master_frame = pd.DataFrame(
        [
            _master_row(symbol="GOOG", dollar_volume=120_000_000, market_cap=220_000_000_000),
            _master_row(symbol="MSFT", dollar_volume=110_000_000, market_cap=210_000_000_000),
            _master_row(symbol="AAPL", dollar_volume=100_000_000, market_cap=200_000_000_000),
            _master_row(symbol="GOOGL", dollar_volume=90_000_000, market_cap=220_000_000_000),
        ]
    )
    profile = UniverseProfileConfig(
        description="Preferred symbol group test",
        active_only=True,
        allowed_exchanges=("XNAS", "XNYS", "ARCX"),
        allowed_ticker_types=("CS",),
        min_price=10.0,
        min_average_daily_volume=100_000,
        min_dollar_volume=1_000_000,
        min_market_cap=None,
        allowed_sectors=(),
        allowed_industries=(),
        include_etfs=False,
        include_adrs=False,
        common_stock_only=True,
        exclude_weird_instruments=True,
        max_symbols=None,
        include_symbols=(),
        exclude_symbols=(),
        preferred_symbol_groups={"GOOGL": ("GOOGL", "GOOG")},
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="preferred_share_class",
        profile_config=profile,
    )

    assert result.symbols == ("GOOGL", "MSFT", "AAPL")
    assert result.summary["preference_removed_symbols"] == ["GOOG"]


def test_blacklist_application_removes_symbols_before_final_output() -> None:
    master_frame = pd.DataFrame(
        [
            _master_row(symbol="KEEP1", dollar_volume=80_000_000),
            _master_row(symbol="DROP1", dollar_volume=75_000_000),
            _master_row(symbol="KEEP2", dollar_volume=70_000_000),
        ]
    )
    profile = UniverseProfileConfig(
        description="Blacklist test",
        active_only=True,
        allowed_exchanges=("XNAS", "XNYS", "ARCX"),
        allowed_ticker_types=("CS",),
        min_price=10.0,
        min_average_daily_volume=100_000,
        min_dollar_volume=1_000_000,
        min_market_cap=None,
        allowed_sectors=(),
        allowed_industries=(),
        include_etfs=False,
        include_adrs=False,
        common_stock_only=True,
        exclude_weird_instruments=True,
        max_symbols=None,
        include_symbols=(),
        exclude_symbols=("DROP1",),
        preferred_symbol_groups={},
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="blacklist_profile",
        profile_config=profile,
    )

    assert result.symbols == ("KEEP1", "KEEP2")


def test_growth_momentum_profile_filters_growth_themes_and_manual_platform_includes() -> None:
    repo_profile = load_app_config().universe_builder.profiles["growth_momentum"]
    profile = replace(
        repo_profile,
        include_symbols=("AMZN", "UBER"),
        exclude_symbols=("IONQ",),
        max_symbols=None,
    )
    master_frame = pd.DataFrame(
        [
            _master_row(
                symbol="NVDA",
                industry="SEMICONDUCTORS & RELATED DEVICES",
                price=120.0,
                average_daily_volume=45_000_000,
                dollar_volume=5_400_000_000,
                market_cap=4_300_000_000_000,
            ),
            _master_row(
                symbol="NOW",
                industry="SERVICES-PREPACKAGED SOFTWARE",
                price=780.0,
                average_daily_volume=1_200_000,
                dollar_volume=936_000_000,
                market_cap=120_000_000_000,
            ),
            _master_row(
                symbol="ANET",
                industry="COMPUTER COMMUNICATIONS EQUIPMENT",
                price=95.0,
                average_daily_volume=7_500_000,
                dollar_volume=712_500_000,
                market_cap=170_000_000_000,
            ),
            _master_row(
                symbol="AMZN",
                industry="RETAIL-CATALOG & MAIL-ORDER HOUSES",
                price=210.0,
                average_daily_volume=38_000_000,
                dollar_volume=7_980_000_000,
                market_cap=2_200_000_000_000,
            ),
            _master_row(
                symbol="UBER",
                industry="SERVICES-BUSINESS SERVICES, NEC",
                price=88.0,
                average_daily_volume=21_000_000,
                dollar_volume=1_848_000_000,
                market_cap=160_000_000_000,
            ),
            _master_row(
                symbol="IONQ",
                industry="SERVICES-COMPUTER INTEGRATED SYSTEMS DESIGN",
                price=27.0,
                average_daily_volume=16_000_000,
                dollar_volume=432_000_000,
                market_cap=12_000_000_000,
            ),
            _master_row(
                symbol="CHEAP",
                industry="SERVICES-PREPACKAGED SOFTWARE",
                price=9.0,
                average_daily_volume=2_000_000,
                dollar_volume=18_000_000,
                market_cap=15_000_000_000,
            ),
            _master_row(
                symbol="SMALL",
                industry="SERVICES-PREPACKAGED SOFTWARE",
                price=42.0,
                average_daily_volume=1_000_000,
                dollar_volume=42_000_000,
                market_cap=3_000_000_000,
            ),
        ]
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="growth_momentum",
        profile_config=profile,
    )

    assert result.symbols == ("AMZN", "UBER", "NVDA", "NOW", "ANET")
    assert result.summary["missing_force_include_symbols"] == []


def test_quality_liquid_profile_applies_stricter_liquidity_and_blacklist_rules() -> None:
    repo_profile = load_app_config().universe_builder.profiles["quality_liquid"]
    profile = replace(repo_profile, max_symbols=None)
    master_frame = pd.DataFrame(
        [
            _master_row(
                symbol="SAFE1",
                price=85.0,
                average_daily_volume=2_500_000,
                dollar_volume=212_500_000,
                market_cap=55_000_000_000,
            ),
            _master_row(
                symbol="SAFE2",
                price=48.0,
                average_daily_volume=1_600_000,
                dollar_volume=76_800_000,
                market_cap=22_000_000_000,
            ),
            _master_row(
                symbol="HOOD",
                price=40.0,
                average_daily_volume=9_000_000,
                dollar_volume=360_000_000,
                market_cap=65_000_000_000,
            ),
            _master_row(
                symbol="THIN",
                price=120.0,
                average_daily_volume=300_000,
                dollar_volume=36_000_000,
                market_cap=45_000_000_000,
            ),
            _master_row(
                symbol="SMCAP",
                price=70.0,
                average_daily_volume=2_000_000,
                dollar_volume=140_000_000,
                market_cap=8_000_000_000,
            ),
            _master_row(
                symbol="ETF1",
                type_code="ETF",
                industry="Exchange traded fund",
                is_etf=True,
                is_common_stock=False,
                price=100.0,
                average_daily_volume=5_000_000,
                dollar_volume=500_000_000,
                market_cap=50_000_000_000,
            ),
        ]
    )

    result = apply_universe_filters(
        master_frame,
        profile_name="quality_liquid",
        profile_config=profile,
    )

    assert result.symbols == ("SAFE1", "SAFE2")


def test_build_universe_cli_writes_master_and_profile_outputs_from_master_input(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    master_input = tmp_path / "master_input.csv"
    pd.DataFrame(
        [
            _master_row(symbol="AAPL", market_cap=50_000_000_000),
            _master_row(symbol="MSFT", market_cap=60_000_000_000),
        ]
    ).to_csv(master_input, index=False)

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    frames_by_symbol = {
        "AAPL": pd.DataFrame(
            {
                "date": dates,
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [2_000_000, 2_100_000, 2_200_000],
                "symbol": ["AAPL", "AAPL", "AAPL"],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "date": dates,
                "open": [90.0, 91.0, 92.0],
                "high": [91.0, 92.0, 93.0],
                "low": [89.0, 90.0, 91.0],
                "close": [90.0, 91.0, 92.0],
                "volume": [1_500_000, 1_600_000, 1_700_000],
                "symbol": ["MSFT", "MSFT", "MSFT"],
            }
        ),
    }
    fake_provider = FakeDailyBarProvider(frames_by_symbol=frames_by_symbol, cache_dir=tmp_path)
    runtime_config = replace(
        load_app_config(),
        project_root=tmp_path,
        config_dir=tmp_path / "config",
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: runtime_config)
    monkeypatch.setattr(
        main_module,
        "create_daily_bar_provider",
        lambda config, env_file: fake_provider,
    )

    exit_code = main_module.main(
        [
            "build-universe",
            "--master-input",
            str(master_input),
            "--as-of",
            "2024-01-04",
            "--profile",
            "large_cap_liquid",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    candidate_output = tmp_path / "data" / "raw" / "candidate_symbols_large_cap_liquid.txt"
    master_output = tmp_path / "data" / "processed" / "universe" / "master_universe.csv"

    assert exit_code == 0
    assert payload["profile_names"] == ["large_cap_liquid"]
    assert candidate_output.exists()
    assert candidate_output.read_text(encoding="utf-8") == "AAPL\nMSFT\n"
    assert master_output.exists()


def test_build_universe_warns_when_master_input_disables_reference_detail_enrichment(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    master_input = tmp_path / "master_input.csv"
    pd.DataFrame(
        [
            _master_row(
                symbol="NVDA",
                industry="Semiconductors",
                sector="Technology",
                market_cap=2_000_000_000_000,
                price=120.0,
                average_daily_volume=5_000_000,
                dollar_volume=600_000_000,
            ),
            _master_row(
                symbol="MSFT",
                industry="Application software",
                sector="Technology",
                market_cap=3_000_000_000_000,
                price=400.0,
                average_daily_volume=3_000_000,
                dollar_volume=1_200_000_000,
            ),
        ]
    ).to_csv(master_input, index=False)
    frames_by_symbol = {
        "NVDA": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "open": [118.0, 119.0, 120.0],
                "high": [121.0, 122.0, 123.0],
                "low": [117.0, 118.0, 119.0],
                "close": [120.0, 121.0, 122.0],
                "volume": [5_000_000, 5_100_000, 5_200_000],
                "symbol": ["NVDA", "NVDA", "NVDA"],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "open": [398.0, 399.0, 400.0],
                "high": [401.0, 402.0, 403.0],
                "low": [397.0, 398.0, 399.0],
                "close": [400.0, 401.0, 402.0],
                "volume": [3_000_000, 3_100_000, 3_200_000],
                "symbol": ["MSFT", "MSFT", "MSFT"],
            }
        ),
    }
    fake_provider = FakeDailyBarProvider(frames_by_symbol=frames_by_symbol, cache_dir=tmp_path)
    runtime_config = replace(
        load_app_config(),
        project_root=tmp_path,
        config_dir=tmp_path / "config",
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: runtime_config)
    monkeypatch.setattr(
        main_module,
        "create_daily_bar_provider",
        lambda config, env_file: fake_provider,
    )

    args = main_module.build_parser().parse_args(
        [
            "build-universe",
            "--master-input",
            str(master_input),
            "--as-of",
            "2024-01-04",
            "--profile",
            "growth_momentum",
            "--format",
            "json",
        ]
    )

    with caplog.at_level(logging.WARNING):
        exit_code = main_module._handle_build_universe(args)

    assert exit_code == 0
    assert "--master-input disables provider-backed reference detail enrichment." in caplog.text


def _master_row(
    *,
    symbol: str,
    exchange: str = "XNAS",
    market: str = "stocks",
    type_code: str = "CS",
    industry: str = "Application software",
    sector: str = "Technology",
    price: float = 50.0,
    average_daily_volume: float = 1_000_000,
    dollar_volume: float = 50_000_000,
    market_cap: float = 5_000_000_000,
    active: bool = True,
    is_etf: bool = False,
    is_adr: bool = False,
    is_common_stock: bool = True,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": symbol,
        "active": active,
        "exchange": exchange,
        "market": market,
        "type": type_code,
        "market_cap": market_cap,
        "sector": sector,
        "industry": industry,
        "price": price,
        "average_daily_volume": average_daily_volume,
        "dollar_volume": dollar_volume,
        "is_etf": is_etf,
        "is_adr": is_adr,
        "is_common_stock": is_common_stock,
        "source": "test",
        "updated_at": "2026-03-19T00:00:00Z",
    }
