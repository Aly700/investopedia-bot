from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bot.execution.manual_executor import ManualExecutor
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
from bot.state.market_state import (
    MarketStateAlertableState,
    MarketStateChangeSet,
    MarketStateSnapshot,
    MarketStateTransition,
    MarketStateUpdateResult,
)


def test_live_market_runner_runs_cycle_end_to_end_with_mocked_components(
    tmp_path: Path,
) -> None:
    call_order: list[str] = []
    daily_summary = _daily_summary()
    portfolio_review = _portfolio_review_report("WATCH CLOSELY")
    intraday_review = _intraday_review_report("EXIT CANDIDATE")
    state_artifacts = _state_artifacts(
        tmp_path,
        transitions=(
            MarketStateTransition(
                transition_type="WATCH_CLOSELY_TO_EXIT_CANDIDATE",
                symbol="AAPL",
                previous_value="WATCH CLOSELY",
                current_value="EXIT CANDIDATE",
                message="AAPL escalated from WATCH CLOSELY to EXIT CANDIDATE.",
            ),
        ),
    )
    runner = LiveMarketRunner(
        run_daily_summary=lambda: _record_call(
            call_order,
            "daily_summary",
            {"summary": daily_summary, "candidate_score_journal_path": tmp_path / "candidate.json"},
        ),
        run_portfolio_review=lambda: _record_call(
            call_order,
            "portfolio_review",
            portfolio_review,
        ),
        run_intraday_review=lambda: _record_call(
            call_order,
            "intraday_review",
            intraday_review,
        ),
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

    result = runner.run_cycle(
        LiveMarketCycleRequest(
            workflow="monitor-market",
            as_of_date=date(2024, 1, 5),
            portfolio_path=str(tmp_path / "portfolio.csv"),
            include_daily_summary=True,
            include_portfolio_review=True,
            include_intraday_review=True,
            daily_summary_required=True,
            portfolio_review_required=True,
            intraday_review_required=True,
        )
    )

    assert call_order == [
        "daily_summary",
        "portfolio_review",
        "intraday_review",
        "monitor_report",
        "persist_market_state",
    ]
    assert result.status == "ok"
    assert result.daily_summary is daily_summary
    assert result.portfolio_review is portfolio_review
    assert result.intraday_review is intraday_review
    assert result.monitor_report is not None
    assert result.market_state_snapshot == state_artifacts.update_result.current_snapshot
    assert result.state_change_count == 1
    assert result.alertable_states == ()
    assert result.paths_written["current_market_state_snapshot"].endswith(
        "current_market_state.json"
    )


def test_live_market_runner_degrades_gracefully_on_optional_stage_failure(
    tmp_path: Path,
) -> None:
    call_order: list[str] = []
    daily_summary = _daily_summary()
    state_artifacts = _state_artifacts(tmp_path)

    def failing_portfolio_review():
        call_order.append("portfolio_review")
        raise RuntimeError("portfolio review failed")

    runner = LiveMarketRunner(
        run_daily_summary=lambda: _record_call(
            call_order,
            "daily_summary",
            {"summary": daily_summary},
        ),
        run_portfolio_review=failing_portfolio_review,
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

    result = runner.run_cycle(
        LiveMarketCycleRequest(
            workflow="monitor-market",
            as_of_date=date(2024, 1, 5),
            include_daily_summary=True,
            include_portfolio_review=True,
            daily_summary_required=True,
            portfolio_review_required=False,
        )
    )

    assert call_order == [
        "daily_summary",
        "portfolio_review",
        "monitor_report",
        "persist_market_state",
    ]
    assert result.status == "partial"
    assert result.warning_count == 1
    assert result.warnings[0].stage == "portfolio_review"
    assert result.monitor_report is not None
    assert result.market_state_snapshot is not None


def test_live_market_cycle_result_distinguishes_state_transitions_from_current_alertable_states(
    tmp_path: Path,
) -> None:
    state_artifacts = _state_artifacts(
        tmp_path,
        transitions=(
            MarketStateTransition(
                transition_type="HOLD_TO_WATCH_CLOSELY",
                symbol="AAPL",
                previous_value="HOLD",
                current_value="WATCH CLOSELY",
                message="AAPL moved from HOLD to WATCH CLOSELY.",
            ),
        ),
        alertable_states=(
            MarketStateAlertableState(
                state_key="symbol:AAPL:WATCH CLOSELY",
                category="WATCH CLOSELY",
                source_workflow="review-portfolio",
                symbol="AAPL",
                action="WATCH CLOSELY",
            ),
            MarketStateAlertableState(
                state_key="candidate:NVDA:BUY CANDIDATE",
                category="BUY CANDIDATE",
                source_workflow="monitor-market",
                symbol="NVDA",
                action="BUY CANDIDATE",
            ),
        ),
    )
    runner = LiveMarketRunner(
        run_daily_summary=lambda: {"summary": _daily_summary()},
        persist_market_state=lambda as_of_date, workflow, summary, review, intraday, portfolio_path: state_artifacts,
    )

    result = runner.run_cycle(
        LiveMarketCycleRequest(
            workflow="monitor-market",
            as_of_date=date(2024, 1, 5),
            include_daily_summary=True,
            daily_summary_required=True,
        )
    )

    assert result.state_change_count == 1
    assert len(result.alertable_states) == 2
    assert result.to_dict()["state_change_count"] == 1
    assert result.to_dict()["alertable_state_count"] == 2


def test_live_market_cycle_result_shape_is_stable(tmp_path: Path) -> None:
    state_artifacts = _state_artifacts(tmp_path)
    runner = LiveMarketRunner(
        run_daily_summary=lambda: {"summary": _daily_summary()},
        persist_market_state=lambda as_of_date, workflow, summary, review, intraday, portfolio_path: state_artifacts,
    )

    result = runner.run_cycle(
        LiveMarketCycleRequest(
            workflow="monitor-market",
            as_of_date=date(2024, 1, 5),
            include_daily_summary=True,
            daily_summary_required=True,
        )
    )

    assert result.to_dict() == {
        "workflow": "monitor-market",
        "as_of_date": "2024-01-05",
        "status": "ok",
        "candidate_summary": {
            "approved_count": 0,
            "rejected_count": 0,
            "order_count": 0,
            "recommended_preset": None,
        },
        "portfolio_review_summary": None,
        "intraday_review_summary": None,
        "has_monitor_report": False,
        "has_market_state_snapshot": True,
        "state_change_count": 0,
        "alertable_state_count": 0,
        "warning_count": 0,
        "warnings": [],
        "paths_written": {
            "current_market_state_snapshot": str(tmp_path / "current_market_state.json"),
            "market_state_changes_json": str(tmp_path / "market_state_changes.json"),
            "market_state_changes_text": str(tmp_path / "market_state_changes.txt"),
        },
    }


def test_live_market_runner_rejects_required_daily_summary_when_not_included() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="daily_summary_required=True requires include_daily_summary=True",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="monitor-market",
                as_of_date=date(2024, 1, 5),
                include_daily_summary=False,
                daily_summary_required=True,
            )
        )


def test_live_market_runner_rejects_required_portfolio_review_when_not_included() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="portfolio_review_required=True requires include_portfolio_review=True",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio",
                as_of_date=date(2024, 1, 5),
                include_portfolio_review=False,
                portfolio_review_required=True,
            )
        )


def test_live_market_runner_rejects_required_intraday_review_when_not_included() -> None:
    runner = LiveMarketRunner()

    with pytest.raises(
        ValueError,
        match="intraday_review_required=True requires include_intraday_review=True",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio-intraday",
                as_of_date=date(2024, 1, 5),
                include_intraday_review=False,
                intraday_review_required=True,
            )
        )


def test_live_market_runner_rejects_wrong_portfolio_review_type() -> None:
    runner = LiveMarketRunner(
        run_portfolio_review=lambda: {"bad": "type"},
    )

    with pytest.raises(
        TypeError,
        match="portfolio_review workflow must return a PortfolioReviewReport",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio",
                as_of_date=date(2024, 1, 5),
                include_portfolio_review=True,
                portfolio_review_required=True,
            )
        )


def test_live_market_runner_rejects_wrong_intraday_review_type() -> None:
    runner = LiveMarketRunner(
        run_intraday_review=lambda: {"bad": "type"},
    )

    with pytest.raises(
        TypeError,
        match="intraday_review workflow must return an IntradayPortfolioReviewReport",
    ):
        runner.run_cycle(
            LiveMarketCycleRequest(
                workflow="review-portfolio-intraday",
                as_of_date=date(2024, 1, 5),
                include_intraday_review=True,
                intraday_review_required=True,
            )
        )


def test_live_market_runner_builds_monitor_report_for_intraday_only_cycle() -> None:
    call_order: list[str] = []
    runner = LiveMarketRunner(
        run_intraday_review=lambda: _record_call(
            call_order,
            "intraday_review",
            _intraday_review_report("WATCH CLOSELY"),
        ),
        build_monitor_report=lambda as_of_date, summary, review, intraday, portfolio_path: _record_call(
            call_order,
            "monitor_report",
            build_market_monitor_report(
                as_of_date=as_of_date,
                portfolio_path=portfolio_path,
            ),
        ),
    )

    result = runner.run_cycle(
        LiveMarketCycleRequest(
            workflow="review-portfolio-intraday",
            as_of_date=date(2024, 1, 5),
            include_intraday_review=True,
            intraday_review_required=True,
        )
    )

    assert call_order == ["intraday_review", "monitor_report"]
    assert result.monitor_report is not None


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


def _portfolio_review_report(action: str):
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
                suggested_stop=(96.0 if action == "EXIT CANDIDATE" else None),
                latest_close=94.0 if action == "EXIT CANDIDATE" else 98.0,
                unrealized_pl_pct=-0.06 if action == "EXIT CANDIDATE" else -0.02,
                distance_to_stop_pct=0.02,
                regime_passed=True,
                above_entry=False,
                suggested_action=action,
                preset_name="standard_breakout",
                rationale=f"{action} rationale.",
            )
        ],
        benchmark_symbol="SPY",
    )


def _intraday_review_report(action: str):
    return build_intraday_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        interval_minutes=15,
        portfolio_path=None,
        current_positions=[],
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=96.0,
                latest_close=94.0,
                unrealized_pl_pct=-0.06,
                distance_to_stop_pct=0.01,
                regime_passed=None,
                above_entry=False,
                suggested_action=action,
                preset_name="standard_breakout",
                rationale=f"{action} rationale.",
            )
        ],
        benchmark_symbol="SPY",
    )


def _state_artifacts(
    tmp_path: Path,
    *,
    transitions: tuple[MarketStateTransition, ...] = (),
    alertable_states: tuple[MarketStateAlertableState, ...] = (),
) -> PersistedMarketStateArtifacts:
    snapshot = MarketStateSnapshot(
        schema_version=1,
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
        current_alertable_states=alertable_states,
    )
    change_set = MarketStateChangeSet(
        as_of_timestamp="2024-01-05T15:30:00+00:00",
        as_of_date=date(2024, 1, 5),
        baseline_established=False,
        transitions=transitions,
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
