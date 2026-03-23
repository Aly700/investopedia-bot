from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bot.execution.manual_executor import ManualExecutor
from bot.ingestion.market_data import (
    MarketCycleIngestionEnvelope,
    MarketDataIngestionUpdate,
    PollingIngestionAdapter,
)
from bot.orchestration.live_runner import (
    LiveMarketCycleRequest,
    LiveMarketRunner,
    PersistedMarketStateArtifacts,
)
from bot.reporting.daily_report import (
    PortfolioReviewRow,
    build_daily_research_summary,
    build_intraday_portfolio_review_report,
    build_market_monitor_report,
    build_portfolio_review_report,
)
from bot.risk.portfolio_rules import ExistingPosition
from bot.state.market_state import (
    MarketStateChangeSet,
    MarketStateSnapshot,
    MarketStateUpdateResult,
)


def test_polling_ingestion_adapter_builds_stable_cycle_envelope() -> None:
    positions = (
        ExistingPosition(
            symbol="MSFT",
            shares=5,
            average_entry_price=100.0,
            current_stop=95.0,
        ),
    )
    request = LiveMarketCycleRequest(
        workflow="monitor-market",
        as_of_date=date(2024, 1, 5),
        portfolio_path="/tmp/portfolio.csv",
        include_daily_summary=True,
        include_portfolio_review=True,
        daily_summary_required=True,
        portfolio_review_required=True,
    )
    ingestion = PollingIngestionAdapter(
        provider_name="test-provider",
        portfolio_path="/tmp/portfolio.csv",
        load_current_positions=lambda: positions,
        create_provider=lambda: object(),
        run_daily_summary=lambda provider, current_positions: {"summary": _daily_summary()},
        run_portfolio_review=lambda provider, current_positions: _portfolio_review_report(),
        candidate_path="/tmp/candidates.csv",
        benchmark_symbol="SPY",
        refresh_cache=True,
    ).build_cycle_ingestion(request)

    assert ingestion.to_dict() == {
        "adapter_name": "polling",
        "ingestion_mode": "polling",
        "as_of_date": "2024-01-05",
        "portfolio_path": "/tmp/portfolio.csv",
        "provider_name": "test-provider",
        "current_position_symbols": ["MSFT"],
        "update_count": 4,
        "updates": [
            {
                "update_type": "benchmark_context_refresh",
                "as_of_date": "2024-01-05",
                "symbol_count": 1,
                "symbols": ["SPY"],
                "metadata": {"provider_name": "test-provider"},
            },
            {
                "update_type": "candidate_universe_refresh_batch",
                "as_of_date": "2024-01-05",
                "symbol_count": 0,
                "symbols": [],
                "metadata": {
                    "candidate_path": "/tmp/candidates.csv",
                    "provider_name": "test-provider",
                    "refresh_cache": True,
                },
            },
            {
                "update_type": "market_context_refresh_batch",
                "as_of_date": "2024-01-05",
                "symbol_count": 0,
                "symbols": [],
                "metadata": {
                    "benchmark_symbol": "SPY",
                    "provider_name": "test-provider",
                    "refresh_cache": True,
                },
            },
            {
                "update_type": "portfolio_symbol_refresh_batch",
                "as_of_date": "2024-01-05",
                "symbol_count": 1,
                "symbols": ["MSFT"],
                "metadata": {
                    "portfolio_path": "/tmp/portfolio.csv",
                    "provider_name": "test-provider",
                },
            },
        ],
    }


def test_live_market_runner_consumes_polling_ingestion_envelope(
    tmp_path: Path,
) -> None:
    call_order: list[str] = []
    request = LiveMarketCycleRequest(
        workflow="monitor-market",
        as_of_date=date(2024, 1, 5),
        portfolio_path=str(tmp_path / "portfolio.csv"),
        include_daily_summary=True,
        include_portfolio_review=True,
        daily_summary_required=True,
        portfolio_review_required=True,
    )
    positions = (
        ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=100.0,
            current_stop=95.0,
        ),
    )
    ingestion = PollingIngestionAdapter(
        provider_name="test-provider",
        portfolio_path=str(tmp_path / "portfolio.csv"),
        load_current_positions=lambda: positions,
        create_provider=lambda: _record_call(call_order, "create_provider", object()),
        run_daily_summary=lambda provider, current_positions: _record_call(
            call_order,
            "daily_summary",
            {"summary": _daily_summary()},
        ),
        run_portfolio_review=lambda provider, current_positions: _record_call(
            call_order,
            "portfolio_review",
            _portfolio_review_report(),
        ),
        candidate_path=str(tmp_path / "candidates.csv"),
        benchmark_symbol="SPY",
        refresh_cache=False,
    ).build_cycle_ingestion(request)
    state_artifacts = _state_artifacts(tmp_path)
    runner = LiveMarketRunner(
        build_monitor_report=lambda as_of_date, summary, review, intraday, portfolio_path: _record_call(
            call_order,
            "monitor_report",
            build_market_monitor_report(
                as_of_date=as_of_date,
                daily_summary=summary,
                portfolio_review=review,
                portfolio_path=portfolio_path,
            ),
        ),
        persist_market_state=lambda as_of_date, workflow, summary, review, intraday, portfolio_path: _record_call(
            call_order,
            "persist_market_state",
            state_artifacts,
        ),
    )

    result = runner.run_cycle(request, ingestion=ingestion)

    assert call_order == [
        "create_provider",
        "daily_summary",
        "portfolio_review",
        "monitor_report",
        "persist_market_state",
    ]
    assert result.status == "ok"
    assert result.ingestion_adapter_name == "polling"
    assert result.ingestion_mode == "polling"
    assert len(result.ingestion_updates) == 4
    assert result.monitor_report is not None
    assert result.market_state_snapshot == state_artifacts.update_result.current_snapshot
    assert result.to_dict()["ingestion_update_count"] == 4


def test_market_data_ingestion_update_rejects_unknown_update_type() -> None:
    with pytest.raises(ValueError, match="update_type must be one of"):
        MarketDataIngestionUpdate(
            update_type="unsupported",
            as_of_date=date(2024, 1, 5),
        )


def test_market_data_ingestion_update_rejects_non_string_symbol() -> None:
    with pytest.raises(TypeError, match="symbols\\[1\\] must be a string"):
        MarketDataIngestionUpdate(
            update_type="quote_bar_batch_refresh",
            as_of_date=date(2024, 1, 5),
            symbols=("AAPL", 123),
        )


def test_market_data_ingestion_update_rejects_bare_string_symbols_argument() -> None:
    with pytest.raises(
        TypeError,
        match="symbols must be an iterable of symbol strings, not a single string",
    ):
        MarketDataIngestionUpdate(
            update_type="quote_bar_batch_refresh",
            as_of_date=date(2024, 1, 5),
            symbols="AAPL",
        )


def test_market_data_ingestion_update_rejects_blank_symbol() -> None:
    with pytest.raises(ValueError, match="symbols\\[1\\] cannot be blank"):
        MarketDataIngestionUpdate(
            update_type="quote_bar_batch_refresh",
            as_of_date=date(2024, 1, 5),
            symbols=("AAPL", "   "),
        )


def test_market_cycle_ingestion_envelope_rejects_malformed_updates() -> None:
    with pytest.raises(TypeError, match="updates\\[0\\] must be a MarketDataIngestionUpdate"):
        MarketCycleIngestionEnvelope(
            adapter_name="polling",
            ingestion_mode="polling",
            as_of_date=date(2024, 1, 5),
            updates=("bad-update",),
        )


def test_market_cycle_ingestion_envelope_rejects_non_callable_daily_summary() -> None:
    with pytest.raises(TypeError, match="run_daily_summary must be callable"):
        MarketCycleIngestionEnvelope(
            adapter_name="polling",
            ingestion_mode="polling",
            as_of_date=date(2024, 1, 5),
            run_daily_summary="bad-callback",
        )


def test_market_cycle_ingestion_envelope_rejects_non_callable_portfolio_review() -> None:
    with pytest.raises(TypeError, match="run_portfolio_review must be callable"):
        MarketCycleIngestionEnvelope(
            adapter_name="polling",
            ingestion_mode="polling",
            as_of_date=date(2024, 1, 5),
            run_portfolio_review={"not": "callable"},
        )


def test_market_cycle_ingestion_envelope_rejects_non_callable_intraday_review() -> None:
    with pytest.raises(TypeError, match="run_intraday_review must be callable"):
        MarketCycleIngestionEnvelope(
            adapter_name="polling",
            ingestion_mode="polling",
            as_of_date=date(2024, 1, 5),
            run_intraday_review="bad-callback",
        )


def test_live_market_runner_rejects_non_envelope_ingestion() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        TypeError,
        match="ingestion must be a MarketCycleIngestionEnvelope",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="monitor-market",
                as_of_date=date(2024, 1, 5),
            ),
            ingestion=object(),
        )


def test_live_market_runner_rejects_mismatched_ingestion_as_of_date() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="ingestion.as_of_date must match request.as_of_date",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="monitor-market",
                as_of_date=date(2024, 1, 5),
            ),
            ingestion=MarketCycleIngestionEnvelope(
                adapter_name="streaming",
                ingestion_mode="streaming",
                as_of_date=date(2024, 1, 4),
            ),
        )


def test_live_market_runner_rejects_mismatched_ingestion_portfolio_path() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="ingestion.portfolio_path must match request.portfolio_path",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio",
                as_of_date=date(2024, 1, 5),
                portfolio_path="/tmp/portfolio.csv",
            ),
            ingestion=MarketCycleIngestionEnvelope(
                adapter_name="streaming",
                ingestion_mode="streaming",
                as_of_date=date(2024, 1, 5),
                portfolio_path="/tmp/other.csv",
            ),
        )


def test_live_market_runner_rejects_missing_daily_summary_update_type() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="candidate_universe_refresh_batch",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="monitor-market",
                as_of_date=date(2024, 1, 5),
                include_daily_summary=True,
            ),
            ingestion=MarketCycleIngestionEnvelope(
                adapter_name="streaming",
                ingestion_mode="streaming",
                as_of_date=date(2024, 1, 5),
                updates=(),
            ),
        )


def test_live_market_runner_rejects_missing_portfolio_review_update_type() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="portfolio_symbol_refresh_batch",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio",
                as_of_date=date(2024, 1, 5),
                include_portfolio_review=True,
            ),
            ingestion=MarketCycleIngestionEnvelope(
                adapter_name="streaming",
                ingestion_mode="streaming",
                as_of_date=date(2024, 1, 5),
                updates=(),
            ),
        )


def test_live_market_runner_rejects_missing_portfolio_refresh_for_intraday_review() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="portfolio_symbol_refresh_batch",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio-intraday",
                as_of_date=date(2024, 1, 5),
                include_intraday_review=True,
            ),
            ingestion=MarketCycleIngestionEnvelope(
                adapter_name="streaming",
                ingestion_mode="streaming",
                as_of_date=date(2024, 1, 5),
                updates=(
                    MarketDataIngestionUpdate(
                        update_type="intraday_portfolio_refresh_batch",
                        as_of_date=date(2024, 1, 5),
                    ),
                ),
            ),
        )


def test_polling_ingestion_adapter_adds_portfolio_refresh_for_intraday_cycle() -> None:
    request = LiveMarketCycleRequest(
        workflow="review-portfolio-intraday",
        as_of_date=date(2024, 1, 5),
        portfolio_path="/tmp/portfolio.csv",
        include_intraday_review=True,
        intraday_review_required=True,
    )
    ingestion = PollingIngestionAdapter(
        provider_name="test-provider",
        portfolio_path="/tmp/portfolio.csv",
        load_current_positions=lambda: (
            ExistingPosition(
                symbol="AAPL",
                shares=10,
                average_entry_price=100.0,
                current_stop=95.0,
            ),
        ),
        create_provider=lambda: object(),
        run_intraday_review=lambda provider, current_positions: _intraday_review_report(),
        interval_minutes=15,
    ).build_cycle_ingestion(request)

    assert [update.update_type for update in ingestion.updates] == [
        "portfolio_symbol_refresh_batch",
        "intraday_portfolio_refresh_batch",
    ]


def _record_call(call_order: list[str], name: str, value):
    call_order.append(name)
    return value


def _daily_summary():
    execution_batch = ManualExecutor().build_execution_batch(
        [],
        as_of_date=date(2024, 1, 5),
    )
    return build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[],
        selected_presets=[],
        universe_symbols=[],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={},
        benchmark_symbol="SPY",
        preset_selection_source="test",
    )


def _portfolio_review_report():
    return build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        current_positions=[],
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=None,
                latest_close=98.0,
                unrealized_pl_pct=-0.02,
                distance_to_stop_pct=0.02,
                regime_passed=True,
                above_entry=False,
                suggested_action="HOLD",
                preset_name="standard_breakout",
                rationale="Hold rationale.",
            )
        ],
        benchmark_symbol="SPY",
    )


def _intraday_review_report():
    return build_intraday_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        interval_minutes=15,
        portfolio_path="/tmp/portfolio.csv",
        current_positions=[],
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=None,
                latest_close=98.0,
                unrealized_pl_pct=-0.02,
                distance_to_stop_pct=0.02,
                regime_passed=None,
                above_entry=False,
                suggested_action="HOLD",
                preset_name="standard_breakout",
                rationale="Hold rationale.",
            )
        ],
        benchmark_symbol="SPY",
    )


def _state_artifacts(tmp_path: Path) -> PersistedMarketStateArtifacts:
    snapshot = MarketStateSnapshot(
        schema_version=1,
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
    )
    change_set = MarketStateChangeSet(
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
        baseline_established=False,
    )
    return PersistedMarketStateArtifacts(
        update_result=MarketStateUpdateResult(
            current_snapshot=snapshot,
            previous_snapshot=None,
            change_set=change_set,
            current_path=tmp_path / "current_market_state.json",
            previous_path=tmp_path / "previous_market_state.json",
        ),
        output_paths={
            "current_market_state_snapshot": str(tmp_path / "current_market_state.json"),
            "market_state_changes_json": str(tmp_path / "market_state_changes.json"),
            "market_state_changes_text": str(tmp_path / "market_state_changes.txt"),
        },
    )
