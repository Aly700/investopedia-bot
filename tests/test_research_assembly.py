from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.config import load_app_config
from bot.data.earnings import EarningsCalendarProvider
from bot.data.providers import DataProviderRequestError
from bot.providers.bundle import ProviderBundle
from bot.providers.capabilities import ProviderCapabilities
import bot.research_assembly as research_assembly_module


class _FakeHistoricalProvider:
    provider_name = "alpaca"
    capabilities = ProviderCapabilities(
        supports_daily_bars=True,
        supports_intraday_bars=True,
        supports_benchmark_symbols=True,
    )

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        _ = (start_date, end_date, refresh_cache)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1_000_000.0, 1_100_000.0, 1_200_000.0],
                "symbol": [symbol, symbol, symbol],
            }
        )

    def fetch_intraday_bars(
        self,
        symbol: str,
        session_date: date,
        *,
        interval_minutes: int = 15,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        _ = (session_date, interval_minutes, refresh_cache)
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                ),
                "open": [100.0, 100.5, 101.0],
                "high": [100.6, 101.0, 101.4],
                "low": [99.8, 100.4, 100.9],
                "close": [100.4, 100.9, 101.2],
                "volume": [1_000_000, 1_100_000, 1_200_000],
                "vwap": [100.2, 100.6, 100.9],
                "symbol": [symbol, symbol, symbol],
            }
        )


class _FakeEarningsProvider(EarningsCalendarProvider):
    provider_name = "polygon"

    def _fetch_upcoming_earnings_from_api(self, *, start_date: date, end_date: date):
        _ = (start_date, end_date)
        return []


def _result(*, role_name: str, operation: str, provider_name: str | None, value, status: str = "ok"):
    return research_assembly_module.ResearchDataResult(
        role_name=role_name,
        operation=operation,
        provider_name=provider_name,
        status=status,
        value=value,
    )


def test_load_volatility_context_result_marks_unsupported_vix_provider_degraded() -> None:
    class _NoVixProvider:
        provider_name = "alpaca"
        capabilities = SimpleNamespace(supports_vix_symbols=False)

    result = research_assembly_module.load_volatility_context_result(
        provider=_NoVixProvider(),
        as_of_date=date(2024, 1, 5),
        vix_caution_threshold=25.0,
        vix_entry_block_threshold=30.0,
        refresh_cache=False,
        log_label="daily-summary",
    )

    assert result.status == "degraded"
    assert result.issue_code == "unsupported_capability"
    assert result.provider_name == "alpaca"
    assert result.value["vix_close"] is None
    assert "VIX/index-style symbols" in (result.message or "")


def test_load_daily_summary_benchmark_frame_result_classifies_timeout_degradation() -> None:
    class _TimeoutProvider:
        provider_name = "alpaca"

        def fetch_daily_bars(self, *args, **kwargs):
            raise DataProviderRequestError("alpaca request failed: timed out")

    config = load_app_config()
    args = SimpleNamespace(
        disable_regime_filter=False,
        benchmark_symbol=None,
        as_of=date(2024, 1, 5),
        refresh_cache=False,
    )

    result = research_assembly_module.load_daily_summary_benchmark_frame_result(
        args=args,
        config=config,
        provider=_TimeoutProvider(),
        symbol_frames={},
        fetch_start=date(2023, 12, 1),
        universe_members=(SimpleNamespace(symbol="AAPL"),),
        log_label="daily-summary",
    )

    assert result.status == "degraded"
    assert result.issue_code == "timeout"
    assert result.value.symbol == "SPY"
    assert result.value.frame is None
    assert "timed out" in (result.message or "")


def test_load_earnings_contexts_result_classifies_entitlement_failure() -> None:
    provider = _FakeEarningsProvider(
        api_key="polygon-key",
        cache_dir=Path("/tmp/investopedia-earnings-cache"),
    )
    original = research_assembly_module.build_earnings_risk_contexts

    def _raise_entitlement(*args, **kwargs):
        raise DataProviderRequestError("polygon request failed with HTTP 403: NOT_AUTHORIZED")

    research_assembly_module.build_earnings_risk_contexts = _raise_entitlement
    try:
        result = research_assembly_module.load_earnings_contexts_result(
            config=load_app_config(),
            env_file=None,
            provider=provider,
            symbols=["AAPL"],
            as_of_date=date(2024, 1, 5),
            earnings_watch_days=5,
            refresh_cache=False,
            log_label="daily-summary",
        )
    finally:
        research_assembly_module.build_earnings_risk_contexts = original

    assert result.status == "degraded"
    assert result.issue_code == "entitlement_limited"
    assert result.provider_name == "polygon"
    assert result.value == {}
    assert "not entitled" in (result.message or "")


def test_daily_summary_research_assembler_uses_provider_bundle_directly(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("AAPL\nMSFT\n", encoding="utf-8")
    provider = _FakeHistoricalProvider()
    config = replace(load_app_config(), project_root=tmp_path)
    provider_bundle = ProviderBundle(
        role_assignments={
            "stream_market_data": "alpaca",
            "historical_bars": "alpaca",
            "reference_data": "polygon",
            "earnings_calendar": "polygon",
            "execution_broker": "alpaca",
            "broker_update_stream": None,
        },
        historical_bars_provider=provider,
        historical_bars_capabilities=provider.capabilities,
        reference_data_provider=None,
        reference_data_capabilities=ProviderCapabilities(),
        earnings_calendar_provider=None,
        earnings_calendar_capabilities=ProviderCapabilities(),
        role_statuses={},
    )
    assembler = research_assembly_module.DailySummaryResearchAssembler(
        config=config,
        provider_bundle=provider_bundle,
        benchmark_loader=lambda **kwargs: _result(
            role_name="historical_bars",
            operation="load_daily_summary_benchmark_frame",
            provider_name="alpaca",
            value=research_assembly_module.BenchmarkFrameData(symbol="SPY", frame=None),
            status="unavailable",
        ),
        earnings_loader=lambda **kwargs: _result(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name="polygon",
            value={},
        ),
        position_classification_loader=lambda *args, **kwargs: _result(
            role_name="reference_data",
            operation="load_position_sector_classifications",
            provider_name="polygon",
            value={},
        ),
        sector_context_loader=lambda **kwargs: _result(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name="polygon",
            value={},
        ),
        volatility_loader=lambda **kwargs: _result(
            role_name="historical_bars",
            operation="load_volatility_context",
            provider_name="alpaca",
            value={
                "vix_close": None,
                "vix_sma_short": None,
                "vix_sma_long": None,
                "volatility_regime_state": None,
                "volatility_regime_risk_off": None,
            },
        ),
    )

    args = SimpleNamespace(
        candidate_path=candidate_path,
        as_of=date(2024, 1, 5),
        lookback_days=20,
        refresh_cache=False,
        disable_regime_filter=False,
        benchmark_symbol=None,
        env_file=None,
    )
    assembled = assembler.assemble(
        args=args,
        current_positions=(),
        fetch_start=date(2023, 12, 1),
        log_label="daily-summary",
    )

    assert assembled.candidate_data.provider_name == "alpaca"
    assert assembled.candidate_data.status == "ok"
    assert [member.symbol for member in assembled.candidate_data.value.universe_members] == [
        "AAPL",
        "MSFT",
    ]
    assert set(assembled.candidate_data.value.symbol_frames) == {"AAPL", "MSFT"}


def test_load_position_daily_symbol_frames_result_tracks_partial_skips() -> None:
    class _PartialProvider(_FakeHistoricalProvider):
        def fetch_daily_bars(self, symbol: str, *args, **kwargs) -> pd.DataFrame:
            if symbol == "MSFT":
                raise DataProviderRequestError("alpaca request failed: timed out")
            return super().fetch_daily_bars(symbol, *args, **kwargs)

    result = research_assembly_module.load_position_daily_symbol_frames_result(
        [SimpleNamespace(symbol="AAPL"), SimpleNamespace(symbol="MSFT")],
        provider=_PartialProvider(),
        history_start=date(2023, 12, 1),
        as_of_date=date(2024, 1, 5),
        refresh_cache=False,
        log_label="review-portfolio",
    )

    assert result.status == "degraded"
    assert result.issue_code == "partial_symbol_frames"
    assert set(result.value.symbol_frames) == {"AAPL"}
    assert result.value.skipped_symbols == ("MSFT",)


def test_portfolio_review_research_assembler_uses_provider_bundle_directly() -> None:
    provider = _FakeHistoricalProvider()
    config = load_app_config()
    provider_bundle = ProviderBundle(
        role_assignments={
            "stream_market_data": "alpaca",
            "historical_bars": "alpaca",
            "reference_data": "polygon",
            "earnings_calendar": "polygon",
            "execution_broker": "alpaca",
            "broker_update_stream": None,
        },
        historical_bars_provider=provider,
        historical_bars_capabilities=provider.capabilities,
        reference_data_provider=None,
        reference_data_capabilities=ProviderCapabilities(),
        earnings_calendar_provider=None,
        earnings_calendar_capabilities=ProviderCapabilities(),
        role_statuses={},
    )
    assembler = research_assembly_module.PortfolioReviewResearchAssembler(
        config=config,
        provider_bundle=provider_bundle,
        benchmark_loader=lambda **kwargs: _result(
            role_name="historical_bars",
            operation="load_review_benchmark_frame",
            provider_name="alpaca",
            value=research_assembly_module.BenchmarkFrameData(symbol="SPY", frame=pd.DataFrame()),
            status="failed",
        ),
        earnings_loader=lambda **kwargs: _result(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name="polygon",
            value={},
        ),
        sector_context_loader=lambda **kwargs: _result(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name="polygon",
            value={},
        ),
    )

    assembled = assembler.assemble(
        args=SimpleNamespace(as_of=date(2024, 1, 5), env_file=None, refresh_cache=False),
        current_positions=(SimpleNamespace(symbol="AAPL"),),
        position_plans=(
            {
                "settings": SimpleNamespace(enable_regime_filter=True, benchmark_symbol="SPY"),
                "fetch_start": date(2023, 12, 1),
            },
        ),
        log_label="review-portfolio",
    )

    assert assembled.benchmark.provider_name == "alpaca"
    assert assembled.benchmark.status == "failed"
    assert assembled.earnings_contexts.provider_name == "polygon"
    assert assembled.position_daily_symbol_frames.provider_name == "alpaca"
    assert set(assembled.position_daily_symbol_frames.value.symbol_frames) == {"AAPL"}


def test_intraday_review_research_assembler_uses_provider_bundle_directly() -> None:
    provider = _FakeHistoricalProvider()
    config = load_app_config()
    provider_bundle = ProviderBundle(
        role_assignments={
            "stream_market_data": "alpaca",
            "historical_bars": "alpaca",
            "reference_data": "polygon",
            "earnings_calendar": "polygon",
            "execution_broker": "alpaca",
            "broker_update_stream": None,
        },
        historical_bars_provider=provider,
        historical_bars_capabilities=provider.capabilities,
        reference_data_provider=None,
        reference_data_capabilities=ProviderCapabilities(),
        earnings_calendar_provider=None,
        earnings_calendar_capabilities=ProviderCapabilities(),
        role_statuses={},
    )
    assembler = research_assembly_module.IntradayReviewResearchAssembler(
        config=config,
        provider_bundle=provider_bundle,
        earnings_loader=lambda **kwargs: _result(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name="polygon",
            value={},
        ),
        sector_context_loader=lambda **kwargs: _result(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name="polygon",
            value={},
        ),
        intraday_metrics_builder=lambda frame: {"intraday_return_vs_open": 0.01},
    )

    assembled = assembler.assemble(
        args=SimpleNamespace(as_of=date(2024, 1, 5), env_file=None, interval_minutes=15),
        current_positions=(SimpleNamespace(symbol="AAPL"),),
        benchmark_symbol="SPY",
        log_label="review-portfolio-intraday",
    )

    assert assembled.benchmark_intraday_metrics.provider_name == "alpaca"
    assert assembled.benchmark_intraday_metrics.status == "ok"
    assert assembled.benchmark_intraday_metrics.value.symbol == "SPY"
    assert assembled.benchmark_intraday_metrics.value.metrics["intraday_return_vs_open"] == pytest.approx(0.01)
    assert set(assembled.position_daily_symbol_frames.value.symbol_frames) == {"AAPL"}
