from __future__ import annotations

import csv
from dataclasses import replace
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import bot.main as main_module
from bot.config import load_app_config
from bot.data.earnings import EarningsRiskContext
from bot.data.intraday_state_journal import default_intraday_state_journal_path
from bot.data.position_trajectory import (
    PositionTrajectoryJournal,
    PositionTrajectoryObservation,
    default_position_trajectory_journal_path,
)
from bot.data.sector_context import SectorFeatureContext, SymbolSectorClassification
from bot.data.providers import DataProviderError
from bot.execution.manual_executor import ManualExecutor
from bot.features import build_market_context
from bot.main import _load_top_preset_name_from_results, _parse_text_list, build_parser
from bot.reporting.daily_report import (
    MARKET_MONITOR_CATEGORIES,
    IntradayPortfolioReviewReport,
    PresetCandidateEvaluation,
    PortfolioReviewRow,
    build_daily_research_summary,
    build_intraday_portfolio_review_report,
    build_market_monitor_report,
    build_portfolio_review_report,
    ensure_market_monitor_categories_cover_portfolio_actions,
    market_monitor_flat_count_key,
    write_intraday_portfolio_review_brief,
    write_market_monitor_report,
    write_market_monitor_text_summary,
    write_daily_preset_summary,
    write_daily_research_brief,
    write_portfolio_review_report,
    write_daily_research_summary,
    write_market_monitor_brief,
)
from bot.risk.portfolio_rules import (
    PORTFOLIO_REVIEW_ACTIONS,
    ExistingPosition,
    RiskAssessedCandidate,
)
from bot.risk.position_sizing import PositionSizingResult
from bot.strategy.breakout_momentum import BreakoutStrategyPreset, build_default_breakout_presets
from bot.strategy.signal_models import StrategySignal


def test_build_daily_research_summary_ranks_approved_candidates_before_rejected() -> None:
    presets = _selected_presets("standard_breakout", "aggressive_breakout")
    approved = _evaluation(
        presets[0],
        symbol="AAA",
        approved=True,
        shares=50,
        prior_high=99.0,
        entry_price=101.0,
        relative_volume=2.2,
    )
    rejected = _evaluation(
        presets[1],
        symbol="BBB",
        approved=False,
        shares=0,
        prior_high=95.0,
        entry_price=101.0,
        relative_volume=3.0,
        rejection_reasons=("Calculated share size is zero after risk and notional constraints.",),
    )
    lower_score_approved = _evaluation(
        presets[1],
        symbol="CCC",
        approved=True,
        shares=25,
        prior_high=99.5,
        entry_price=100.0,
        relative_volume=1.6,
    )

    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate, rejected.candidate, lower_score_approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved, rejected, lower_score_approved],
        selected_presets=presets,
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": (), "aggressive_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    assert [row.symbol for row in summary.rows] == ["AAA", "CCC", "BBB"]
    assert [row.status for row in summary.rows] == ["approved", "approved", "rejected"]
    assert summary.rows[0].preset_name == "standard_breakout"
    assert summary.rows[0].quantity == 50
    assert summary.rows[2].rejection_reasons == (
        "Calculated share size is zero after risk and notional constraints.",
    )
    assert summary.recommended_preset == "standard_breakout"


def test_daily_research_summary_reports_counts_and_rejections() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=40)
    rejected = _evaluation(
        preset,
        symbol="BBB",
        approved=False,
        shares=0,
        rejection_reasons=("Max concurrent positions reached.",),
    )

    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate, rejected.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved, rejected],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ("CCC",)},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    payload = summary.to_dict()

    assert payload["approved_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["order_count"] == 1
    assert payload["unique_no_signal_symbol_count"] == 1
    assert payload["rejected_candidates"][0]["rejection_reasons"] == ["Max concurrent positions reached."]
    assert payload["suggested_order_sheet"]["order_count"] == 1


def test_build_daily_research_summary_uses_candidate_memory_to_break_score_ties() -> None:
    preset = _selected_presets("standard_breakout")[0]
    persistent = _evaluation(
        preset,
        symbol="AAA",
        approved=True,
        shares=40,
        prior_high=99.0,
        entry_price=101.0,
        relative_volume=2.0,
    )
    fresh = _evaluation(
        preset,
        symbol="BBB",
        approved=True,
        shares=40,
        prior_high=99.0,
        entry_price=101.0,
        relative_volume=2.0,
    )
    persistent = replace(
        persistent,
        candidate=replace(
            persistent.candidate,
            signal=replace(
                persistent.candidate.signal,
                metadata={
                    **dict(persistent.candidate.signal.metadata),
                    "setup_quality_score": 0.9,
                    "setup_persistence_days": 4,
                    "days_approved": 2,
                    "repeated_high_quality_signal": True,
                },
            ),
        ),
    )
    fresh = replace(
        fresh,
        candidate=replace(
            fresh.candidate,
            signal=replace(
                fresh.candidate.signal,
                metadata={
                    **dict(fresh.candidate.signal.metadata),
                    "setup_quality_score": 0.0,
                },
            ),
        ),
    )

    execution_batch = ManualExecutor().build_execution_batch(
        [persistent.candidate, fresh.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[fresh, persistent],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={preset.name: ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    assert [row.symbol for row in summary.rows] == ["AAA", "BBB"]
    assert summary.rows[0].metadata["score_components"]["candidate_memory_score"] == pytest.approx(0.9)
    assert "setup=high-confidence repeat signal, persisted for 4 sessions, approved on 2 sessions" in summary.rows[0].rationale


def test_daily_research_summary_handles_empty_no_signal_days(tmp_path: Path) -> None:
    preset = _selected_presets("standard_breakout")[0]
    execution_batch = ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5))
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ("AAA", "BBB")},
        benchmark_symbol="SPY",
        preset_selection_source="default_standard_breakout",
    )

    summary_json = write_daily_research_summary(summary, tmp_path / "daily_summary.json")
    opportunities_csv = write_daily_research_summary(summary, tmp_path / "ranked_opportunities.csv")
    preset_csv = write_daily_preset_summary(summary, tmp_path / "preset_rankings.csv")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    with opportunities_csv.open("r", encoding="utf-8", newline="") as handle:
        opportunity_rows = list(csv.DictReader(handle))
    with preset_csv.open("r", encoding="utf-8", newline="") as handle:
        preset_rows = list(csv.DictReader(handle))

    assert payload["candidate_count"] == 0
    assert payload["approved_candidates"] == []
    assert payload["rejected_candidates"] == []
    assert payload["current_position_count"] == 0
    assert payload["recommended_preset"] is None
    assert opportunity_rows == []
    assert preset_rows[0]["no_signal_count"] == "2"


def test_portfolio_review_report_exports_one_holding(tmp_path: Path) -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        preset_name="standard_breakout",
    )
    row = PortfolioReviewRow(
        date=date(2024, 1, 5),
        symbol="AAPL",
        quantity=10,
        average_entry_price=100.0,
        current_stop=95.0,
        suggested_stop=101.0,
        latest_close=110.0,
        unrealized_pl_pct=0.10,
        distance_to_stop_pct=(110.0 - 95.0) / 110.0,
        regime_passed=True,
        above_entry=True,
        suggested_action="RAISE STOP",
        preset_name="standard_breakout",
        rationale="Trailing-stop logic supports a higher stop.",
        metadata={"trailing_stop_candidate": 101.0},
    )
    report = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[row],
        current_positions=[position],
        benchmark_symbol="SPY",
    )

    json_path = write_portfolio_review_report(report, tmp_path / "portfolio_review.json")
    csv_path = write_portfolio_review_report(report, tmp_path / "portfolio_review.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert payload["position_count"] == 1
    assert payload["raise_stop_count"] == 1
    assert payload["reviewed_symbols"] == ["AAPL"]
    assert payload["rows"][0]["suggested_action"] == "RAISE STOP"
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["suggested_stop"] == "101"


def test_portfolio_review_report_to_dict_counts_actions_from_declared_action_list() -> None:
    positions = [
        ExistingPosition(
            symbol=f"SYM{index}",
            shares=10,
            average_entry_price=100.0 + index,
            current_stop=95.0 + index,
            preset_name="standard_breakout",
        )
        for index, _ in enumerate(PORTFOLIO_REVIEW_ACTIONS, start=1)
    ]
    rows = [
        PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=(
                position.current_stop + 1.0
                if action == "RAISE STOP" and position.current_stop is not None
                else position.current_stop
            ),
            latest_close=position.average_entry_price + 5.0,
            unrealized_pl_pct=0.05,
            distance_to_stop_pct=0.10,
            regime_passed=True,
            above_entry=True,
            suggested_action=action,
            preset_name="standard_breakout",
            rationale=f"{action} rationale.",
            metadata={},
        )
        for position, action in zip(positions, PORTFOLIO_REVIEW_ACTIONS)
    ]

    payload = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=rows,
        current_positions=positions,
        benchmark_symbol="SPY",
    ).to_dict()

    expected_count_fields = {
        f"{action.lower().replace(' ', '_')}_count": 1
        for action in PORTFOLIO_REVIEW_ACTIONS
    }
    for key, expected_value in expected_count_fields.items():
        assert payload[key] == expected_value


def test_portfolio_review_row_rejects_invalid_suggested_action() -> None:
    with pytest.raises(ValueError, match="got 'TRIM'"):
        PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol="AAPL",
            quantity=10,
            average_entry_price=100.0,
            current_stop=95.0,
            suggested_stop=101.0,
            latest_close=110.0,
            unrealized_pl_pct=0.10,
            distance_to_stop_pct=(110.0 - 95.0) / 110.0,
            regime_passed=True,
            above_entry=True,
            suggested_action="TRIM",
            preset_name="standard_breakout",
            rationale="Invalid action for regression coverage.",
            metadata={},
        )


def test_portfolio_review_row_rejects_raise_stop_at_or_above_latest_close() -> None:
    with pytest.raises(ValueError, match="remain below latest_close"):
        PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol="AAPL",
            quantity=10,
            average_entry_price=100.0,
            current_stop=95.0,
            suggested_stop=110.0,
            latest_close=110.0,
            unrealized_pl_pct=0.10,
            distance_to_stop_pct=(110.0 - 95.0) / 110.0,
            regime_passed=True,
            above_entry=True,
            suggested_action="RAISE STOP",
            preset_name="standard_breakout",
            rationale="Invalid stop recommendation for regression coverage.",
            metadata={},
        )


def test_market_monitor_report_handles_no_alert_day(tmp_path: Path) -> None:
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5)),
        evaluations=[],
        selected_presets=[_selected_presets("standard_breakout")[0]],
        universe_symbols=["AAA", "BBB"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ("AAA", "BBB")},
        benchmark_symbol="SPY",
        preset_selection_source="default_standard_breakout",
    )

    report = build_market_monitor_report(as_of_date=date(2024, 1, 5), daily_summary=summary)
    json_path = write_market_monitor_report(report, tmp_path / "market_monitor.json")
    text_path = write_market_monitor_text_summary(report, tmp_path / "market_monitor.txt")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["no_action_count"] == 1
    assert payload["alerts"][0]["category"] == "NO ACTION"
    assert payload["alerts"][0]["symbol"] == ""
    assert "NO ACTION: 1" in text


def test_market_monitor_report_flat_count_keys_match_category_counts() -> None:
    report = build_market_monitor_report(as_of_date=date(2024, 1, 5))
    payload = report.to_dict()

    for category in MARKET_MONITOR_CATEGORIES:
        assert payload[market_monitor_flat_count_key(category)] == payload["category_counts"][category]


def test_market_monitor_report_handles_no_action_when_inputs_are_absent() -> None:
    report = build_market_monitor_report(as_of_date=date(2024, 1, 5))

    assert report.category_counts["NO ACTION"] == 1
    assert len(report.alerts) == 1
    assert report.alerts[0].category == "NO ACTION"
    assert report.alerts[0].symbol == ""
    assert report.preset_names == ()


def test_market_monitor_report_includes_buy_candidate_alert() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    report = build_market_monitor_report(as_of_date=date(2024, 1, 5), daily_summary=summary)

    assert report.category_counts["BUY CANDIDATE"] == 1
    assert report.alerts[0].category == "BUY CANDIDATE"
    assert report.alerts[0].symbol == "AAA"
    assert report.alerts[0].preset_name == "standard_breakout"


def test_market_monitor_report_includes_portfolio_review_alert() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        preset_name="standard_breakout",
    )
    review = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=101.0,
                latest_close=110.0,
                unrealized_pl_pct=0.10,
                distance_to_stop_pct=(110.0 - 95.0) / 110.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="RAISE STOP",
                preset_name="standard_breakout",
                rationale="Trailing-stop logic supports a higher stop.",
                metadata={"trailing_stop_candidate": 101.0},
            )
        ],
        current_positions=[position],
        benchmark_symbol="SPY",
    )

    report = build_market_monitor_report(as_of_date=date(2024, 1, 5), portfolio_review=review)

    assert report.category_counts["RAISE STOP"] == 1
    assert report.alerts[0].category == "RAISE STOP"
    assert report.alerts[0].symbol == "AAPL"


def test_market_monitor_category_guard_accepts_current_action_coverage() -> None:
    ensure_market_monitor_categories_cover_portfolio_actions(
        MARKET_MONITOR_CATEGORIES,
        PORTFOLIO_REVIEW_ACTIONS,
    )


def test_market_monitor_category_guard_rejects_missing_portfolio_action_categories() -> None:
    with pytest.raises(RuntimeError, match="Missing categories: \\['HOLD'\\]"):
        ensure_market_monitor_categories_cover_portfolio_actions(
            tuple(category for category in MARKET_MONITOR_CATEGORIES if category != "HOLD"),
            PORTFOLIO_REVIEW_ACTIONS,
        )


def test_market_monitor_report_includes_review_preset_names_without_daily_summary() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        preset_name="confirmed_conservative_breakout",
    )
    review = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=101.0,
                latest_close=110.0,
                unrealized_pl_pct=0.10,
                distance_to_stop_pct=(110.0 - 95.0) / 110.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="RAISE STOP",
                preset_name="confirmed_conservative_breakout",
                rationale="Trailing-stop logic supports a higher stop.",
                metadata={"trailing_stop_candidate": 101.0},
            )
        ],
        current_positions=[position],
        benchmark_symbol="SPY",
    )

    report = build_market_monitor_report(as_of_date=date(2024, 1, 5), portfolio_review=review)

    assert report.preset_names == ("confirmed_conservative_breakout",)


def test_market_monitor_report_combines_entry_and_position_management_alerts(
    tmp_path: Path,
) -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch(
            [approved.candidate],
            as_of_date=date(2024, 1, 5),
        ),
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        preset_name="standard_breakout",
    )
    review = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=101.0,
                latest_close=110.0,
                unrealized_pl_pct=0.10,
                distance_to_stop_pct=(110.0 - 95.0) / 110.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="RAISE STOP",
                preset_name="standard_breakout",
                rationale="Trailing-stop logic supports a higher stop.",
                metadata={"trailing_stop_candidate": 101.0},
            )
        ],
        current_positions=[position],
        benchmark_symbol="SPY",
    )

    report = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
        portfolio_review=review,
        portfolio_path="/tmp/portfolio.json",
    )
    csv_path = write_market_monitor_report(report, tmp_path / "market_monitor.csv")
    text_path = write_market_monitor_text_summary(report, tmp_path / "market_monitor.txt")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    text = text_path.read_text(encoding="utf-8")

    assert report.category_counts["BUY CANDIDATE"] == 1
    assert report.category_counts["RAISE STOP"] == 1
    assert [row["category"] for row in rows] == ["RAISE STOP", "BUY CANDIDATE"]
    assert "BUY CANDIDATE: 1" in text
    assert "RAISE STOP: 1" in text


def test_market_monitor_text_summary_header_matches_alert_priority_order() -> None:
    report = build_market_monitor_report(as_of_date=date(2024, 1, 5))

    assert report.to_text().splitlines()[:7] == [
        "Market monitor for 2024-01-05",
        "EXIT CANDIDATE: 0",
        "RAISE STOP: 0",
        "BUY CANDIDATE: 0",
        "WATCH CLOSELY: 0",
        "HOLD: 0",
        "NO ACTION: 1",
    ]


def test_market_monitor_brief_separates_holdings_and_buy_candidates(tmp_path: Path) -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved_candidates = [
        _evaluation(preset, symbol="BBB", approved=True, shares=25, relative_volume=2.2),
        _evaluation(preset, symbol="AAA", approved=True, shares=20, relative_volume=2.8),
        _evaluation(preset, symbol="CCC", approved=True, shares=15, relative_volume=1.9),
    ]
    lower_priority_candidate = [
        _evaluation(
            preset,
            symbol="DDD",
            approved=False,
            shares=0,
            relative_volume=1.2,
            rejection_reasons=("Relative volume confirmation was not strong enough.",),
        )
    ]
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch(
            [evaluation.candidate for evaluation in approved_candidates],
            as_of_date=date(2024, 1, 5),
        ),
        evaluations=[*approved_candidates, *lower_priority_candidate],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC", "DDD", "AAPL", "AMD"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    review = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=101.0,
                latest_close=110.0,
                unrealized_pl_pct=0.10,
                distance_to_stop_pct=(110.0 - 95.0) / 110.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="RAISE STOP",
                preset_name="standard_breakout",
                rationale="Trailing-stop logic supports a higher stop.",
                metadata={},
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AMD",
                quantity=8,
                average_entry_price=150.0,
                current_stop=145.0,
                suggested_stop=145.0,
                latest_close=147.0,
                unrealized_pl_pct=-0.02,
                distance_to_stop_pct=(147.0 - 145.0) / 147.0,
                regime_passed=True,
                above_entry=False,
                suggested_action="WATCH CLOSELY",
                preset_name="standard_breakout",
                rationale="Position is below entry and close to the stop.",
                metadata={},
            ),
        ],
        current_positions=[
            ExistingPosition(
                symbol="AAPL",
                shares=10,
                average_entry_price=100.0,
                current_stop=95.0,
                preset_name="standard_breakout",
            ),
            ExistingPosition(
                symbol="AMD",
                shares=8,
                average_entry_price=150.0,
                current_stop=145.0,
                preset_name="standard_breakout",
            ),
        ],
        benchmark_symbol="SPY",
    )
    report = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
        portfolio_review=review,
    )

    brief_path = write_market_monitor_brief(report, tmp_path / "market_monitor_brief.txt")
    text = brief_path.read_text(encoding="utf-8")

    assert "Headline" in text
    assert "Best actions now" in text
    assert "Current holdings" in text
    assert "Top buy candidates" in text
    assert "Lower-priority names" in text
    best_actions_section = text.split("Best actions now\n", 1)[1].split("\n\nCurrent holdings", 1)[0]
    holdings_section = text.split("Current holdings\n", 1)[1].split("\n\nTop buy candidates", 1)[0]
    buy_section = text.split("Top buy candidates\n", 1)[1].split("\n\nLower-priority names", 1)[0]
    lower_priority_section = text.split("Lower-priority names\n", 1)[1]
    assert "- Urgent holdings: AAPL (RAISE STOP)" in best_actions_section
    assert "- Best buys: AAA, BBB, CCC" in best_actions_section
    assert "AAPL" in holdings_section
    assert "AMD" in holdings_section
    assert "AAA" not in holdings_section
    assert "DDD" not in holdings_section
    assert "1. AAA" in buy_section
    assert "2. BBB" in buy_section
    assert "3. CCC" in buy_section
    assert "AAPL" not in buy_section
    assert "AMD" not in buy_section
    assert buy_section.index("1. AAA") < buy_section.index("2. BBB") < buy_section.index("3. CCC")
    assert "DDD" in lower_priority_section
    assert "AMD" not in lower_priority_section
    assert "AAPL" not in lower_priority_section


def test_market_monitor_brief_limits_top_buy_section_and_moves_remainder_lower(
    tmp_path: Path,
) -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved_candidates = [
        _evaluation(preset, symbol="AAA", approved=True, shares=30, relative_volume=3.0),
        _evaluation(preset, symbol="BBB", approved=True, shares=30, relative_volume=2.8),
        _evaluation(preset, symbol="CCC", approved=True, shares=30, relative_volume=2.6),
        _evaluation(preset, symbol="DDD", approved=True, shares=30, relative_volume=2.4),
        _evaluation(preset, symbol="EEE", approved=True, shares=30, relative_volume=2.2),
        _evaluation(preset, symbol="FFF", approved=True, shares=30, relative_volume=2.0),
    ]
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch(
            [evaluation.candidate for evaluation in approved_candidates],
            as_of_date=date(2024, 1, 5),
        ),
        evaluations=approved_candidates,
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    report = build_market_monitor_report(as_of_date=date(2024, 1, 5), daily_summary=summary)

    brief_path = write_market_monitor_brief(report, tmp_path / "market_monitor_brief.txt")
    text = brief_path.read_text(encoding="utf-8")

    best_actions_section = text.split("Best actions now\n", 1)[1].split("\n\nCurrent holdings", 1)[0]
    buy_section = text.split("Top buy candidates\n", 1)[1].split("\n\nLower-priority names", 1)[0]
    lower_priority_section = text.split("Lower-priority names\n", 1)[1]
    assert "- Best buys: AAA, BBB, CCC, DDD, EEE" in best_actions_section
    assert "FFF" not in best_actions_section
    assert "1. AAA" in buy_section
    assert "5. EEE" in buy_section
    assert "6. FFF" not in buy_section
    assert "6. FFF" in lower_priority_section


def test_market_monitor_report_orders_multiple_alert_categories_by_priority() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="MSFT", approved=True, shares=25)
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch(
            [approved.candidate],
            as_of_date=date(2024, 1, 5),
        ),
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["MSFT"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    current_positions = [
        ExistingPosition(
            symbol="TSLA",
            shares=5,
            average_entry_price=200.0,
            current_stop=195.0,
            preset_name="standard_breakout",
        ),
        ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=100.0,
            current_stop=95.0,
            preset_name="standard_breakout",
        ),
        ExistingPosition(
            symbol="AMD",
            shares=10,
            average_entry_price=150.0,
            current_stop=145.0,
            preset_name="standard_breakout",
        ),
        ExistingPosition(
            symbol="GOOG",
            shares=8,
            average_entry_price=120.0,
            current_stop=110.0,
            preset_name="standard_breakout",
        ),
    ]
    review = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="TSLA",
                quantity=5,
                average_entry_price=200.0,
                current_stop=195.0,
                suggested_stop=195.0,
                latest_close=190.0,
                unrealized_pl_pct=-0.05,
                distance_to_stop_pct=(190.0 - 195.0) / 190.0,
                regime_passed=False,
                above_entry=False,
                suggested_action="EXIT CANDIDATE",
                preset_name="standard_breakout",
                rationale="Close is below the active stop.",
                metadata={},
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=101.0,
                latest_close=110.0,
                unrealized_pl_pct=0.10,
                distance_to_stop_pct=(110.0 - 95.0) / 110.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="RAISE STOP",
                preset_name="standard_breakout",
                rationale="Trailing-stop logic supports a higher stop.",
                metadata={"trailing_stop_candidate": 101.0},
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AMD",
                quantity=10,
                average_entry_price=150.0,
                current_stop=145.0,
                suggested_stop=145.0,
                latest_close=147.0,
                unrealized_pl_pct=-0.02,
                distance_to_stop_pct=(147.0 - 145.0) / 147.0,
                regime_passed=True,
                above_entry=False,
                suggested_action="WATCH CLOSELY",
                preset_name="standard_breakout",
                rationale="Position is below entry and close to the stop.",
                metadata={},
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="GOOG",
                quantity=8,
                average_entry_price=120.0,
                current_stop=110.0,
                suggested_stop=110.0,
                latest_close=130.0,
                unrealized_pl_pct=0.0833,
                distance_to_stop_pct=(130.0 - 110.0) / 130.0,
                regime_passed=True,
                above_entry=True,
                suggested_action="HOLD",
                preset_name="standard_breakout",
                rationale="Trend remains healthy above the stop.",
                metadata={},
            ),
        ],
        current_positions=current_positions,
        benchmark_symbol="SPY",
    )

    report = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
        portfolio_review=review,
    )

    assert [alert.category for alert in report.alerts] == [
        "EXIT CANDIDATE",
        "RAISE STOP",
        "BUY CANDIDATE",
        "WATCH CLOSELY",
        "HOLD",
    ]


def test_handle_monitor_market_payload_includes_daily_summary_status_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    rejected = _evaluation(
        preset,
        symbol="BBB",
        approved=False,
        shares=0,
        rejection_reasons=("Max concurrent positions reached.",),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch(
            [approved.candidate, rejected.candidate],
            as_of_date=date(2024, 1, 5),
        ),
        evaluations=[approved, rejected],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ("CCC",)},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: SimpleNamespace(project_root=tmp_path))
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: object())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: [])
    monkeypatch.setattr(
        main_module,
        "_run_daily_summary_workflow",
        lambda args, *, config, provider, current_positions=None: {"summary": summary},
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--config-dir",
            str(tmp_path / "config"),
            "--env-file",
            str(tmp_path / ".env"),
            "monitor-market",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--output-dir",
            str(tmp_path / "monitor"),
            "--format",
            "json",
        ]
    )

    result = args.handler(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["approved_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["approved_count"] == payload["buy_candidate_count"]
    assert payload["universe_count"] == 3
    assert payload["buy_candidate_count"] == 1
    assert Path(payload["outputs"]["market_monitor_json"]).exists()
    assert Path(payload["outputs"]["market_monitor_brief"]).exists()


def test_run_portfolio_review_workflow_skips_symbols_with_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_app_config()
    current_positions = [
        ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=100.0,
            current_stop=95.0,
            preset_name="standard_breakout",
        ),
        ExistingPosition(
            symbol="MSFT",
            shares=5,
            average_entry_price=200.0,
            current_stop=190.0,
            preset_name="standard_breakout",
        ),
    ]
    args = SimpleNamespace(
        as_of=date(2024, 1, 5),
        portfolio_file=None,
        benchmark_symbol=None,
        refresh_cache=False,
        config_dir=tmp_path,
    )

    monkeypatch.setattr(main_module, "_portfolio_review_preset_catalog", lambda args, config: {})
    monkeypatch.setattr(
        main_module,
        "_build_portfolio_review_plan",
        lambda position, *, preset_catalog, base_settings, as_of_date: {
            "position": position,
            "settings": SimpleNamespace(enable_regime_filter=False),
            "fetch_start": as_of_date,
        },
    )

    def fake_build_review_row(plan: dict[str, object], **_: object) -> PortfolioReviewRow:
        position = plan["position"]
        if not isinstance(position, ExistingPosition):
            raise TypeError("position must be an ExistingPosition")
        if position.symbol == "MSFT":
            raise DataProviderError("provider failed")
        return PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=101.0,
            latest_close=110.0,
            unrealized_pl_pct=0.10,
            distance_to_stop_pct=(110.0 - 95.0) / 110.0,
            regime_passed=True,
            above_entry=True,
            suggested_action="RAISE STOP",
            preset_name="standard_breakout",
            rationale="Trailing-stop logic supports a higher stop.",
            metadata={},
        )

    monkeypatch.setattr(main_module, "_build_portfolio_review_row", fake_build_review_row)

    report = main_module._run_portfolio_review_workflow(
        args,
        config=config,
        provider=object(),
        current_positions=current_positions,
    )

    assert [row.symbol for row in report.rows] == ["AAPL"]
    assert report.to_dict()["position_count"] == 2
    assert report.to_dict()["reviewed_symbol_count"] == 1


def test_run_portfolio_review_workflow_skips_symbols_with_value_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_app_config()
    current_positions = [
        ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=100.0,
            current_stop=95.0,
            preset_name="standard_breakout",
        ),
        ExistingPosition(
            symbol="MSFT",
            shares=5,
            average_entry_price=200.0,
            current_stop=190.0,
            preset_name="standard_breakout",
        ),
    ]
    args = SimpleNamespace(
        as_of=date(2024, 1, 5),
        portfolio_file=None,
        benchmark_symbol=None,
        refresh_cache=False,
        config_dir=tmp_path,
    )

    monkeypatch.setattr(main_module, "_portfolio_review_preset_catalog", lambda args, config: {})
    monkeypatch.setattr(
        main_module,
        "_build_portfolio_review_plan",
        lambda position, *, preset_catalog, base_settings, as_of_date: {
            "position": position,
            "settings": SimpleNamespace(enable_regime_filter=False),
            "fetch_start": as_of_date,
        },
    )

    def fake_build_review_row(plan: dict[str, object], **_: object) -> PortfolioReviewRow:
        position = plan["position"]
        if not isinstance(position, ExistingPosition):
            raise TypeError("position must be an ExistingPosition")
        if position.symbol == "MSFT":
            raise ValueError("unusable close prices")
        return PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=101.0,
            latest_close=110.0,
            unrealized_pl_pct=0.10,
            distance_to_stop_pct=(110.0 - 95.0) / 110.0,
            regime_passed=True,
            above_entry=True,
            suggested_action="RAISE STOP",
            preset_name="standard_breakout",
            rationale="Trailing-stop logic supports a higher stop.",
            metadata={},
        )

    monkeypatch.setattr(main_module, "_build_portfolio_review_row", fake_build_review_row)

    report = main_module._run_portfolio_review_workflow(
        args,
        config=config,
        provider=object(),
        current_positions=current_positions,
    )

    payload = report.to_dict()
    assert [row.symbol for row in report.rows] == ["AAPL"]
    assert payload["position_count"] == 2
    assert payload["reviewed_symbol_count"] == 1


def test_handle_review_portfolio_writes_position_trajectory_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: object())
    monkeypatch.setattr(main_module, "_portfolio_review_preset_catalog", lambda args, config: {})
    monkeypatch.setattr(
        main_module,
        "_build_portfolio_review_plan",
        lambda position, *, preset_catalog, base_settings, as_of_date: {
            "position": position,
            "settings": SimpleNamespace(enable_regime_filter=False),
            "fetch_start": as_of_date,
        },
    )

    def fake_build_review_row(plan: dict[str, object], **_: object) -> PortfolioReviewRow:
        position = plan["position"]
        if not isinstance(position, ExistingPosition):
            raise TypeError("position must be an ExistingPosition")
        observation = PositionTrajectoryObservation(
            symbol=position.symbol,
            as_of_date=date(2024, 1, 5),
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            latest_close=98.0,
            unrealized_pl_pct=-0.02,
            above_entry=False,
            high_water_close=110.0,
            high_water_close_date=date(2024, 1, 2),
            days_since_new_high=3,
            stale_position=False,
            relative_strength_return_diff=-0.01,
            weak_relative_strength=False,
            suggested_action="WATCH CLOSELY",
        )
        return PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=None,
            latest_close=98.0,
            unrealized_pl_pct=-0.02,
            distance_to_stop_pct=(98.0 - 90.0) / 98.0,
            regime_passed=True,
            above_entry=False,
            suggested_action="WATCH CLOSELY",
            preset_name="standard_breakout",
            rationale="Position is below the average entry price.",
            metadata={
                main_module._POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY: (
                    observation.to_dict()
                )
            },
        )

    monkeypatch.setattr(main_module, "_build_portfolio_review_row", fake_build_review_row)

    args = build_parser().parse_args(
        [
            "review-portfolio",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--output-dir",
            str(tmp_path / "daily"),
            "--format",
            "json",
        ]
    )

    result = main_module._handle_review_portfolio(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["watch_closely_count"] == 1
    assert Path(payload["outputs"]["portfolio_review_json"]).exists()
    assert Path(payload["outputs"]["portfolio_review_csv"]).exists()
    assert Path(payload["outputs"]["position_trajectory_journal"]).exists()

    journal_payload = json.loads(
        Path(payload["outputs"]["position_trajectory_journal"]).read_text(encoding="utf-8")
    )
    assert journal_payload["symbols"]["AAPL"]["observations"][0]["suggested_action"] == (
        "WATCH CLOSELY"
    )


def test_handle_review_portfolio_same_day_position_trajectory_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: object())
    monkeypatch.setattr(main_module, "_portfolio_review_preset_catalog", lambda args, config: {})
    monkeypatch.setattr(
        main_module,
        "_build_portfolio_review_plan",
        lambda position, *, preset_catalog, base_settings, as_of_date: {
            "position": position,
            "settings": SimpleNamespace(enable_regime_filter=False),
            "fetch_start": as_of_date,
        },
    )

    state = {"calls": 0}

    def fake_build_review_row(plan: dict[str, object], **_: object) -> PortfolioReviewRow:
        state["calls"] += 1
        position = plan["position"]
        if not isinstance(position, ExistingPosition):
            raise TypeError("position must be an ExistingPosition")
        latest_close = 98.0 if state["calls"] == 1 else 101.0
        suggested_action = "WATCH CLOSELY" if state["calls"] == 1 else "HOLD"
        observation = PositionTrajectoryObservation(
            symbol=position.symbol,
            as_of_date=date(2024, 1, 5),
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            latest_close=latest_close,
            unrealized_pl_pct=(latest_close / position.average_entry_price) - 1.0,
            above_entry=latest_close >= position.average_entry_price,
            high_water_close=110.0,
            high_water_close_date=date(2024, 1, 2),
            days_since_new_high=1,
            stale_position=False,
            relative_strength_return_diff=0.01,
            weak_relative_strength=False,
            suggested_action=suggested_action,
        )
        return PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=None,
            latest_close=latest_close,
            unrealized_pl_pct=(latest_close / position.average_entry_price) - 1.0,
            distance_to_stop_pct=(latest_close - 90.0) / latest_close,
            regime_passed=True,
            above_entry=latest_close >= position.average_entry_price,
            suggested_action=suggested_action,
            preset_name="standard_breakout",
            rationale=f"{suggested_action} rationale.",
            metadata={
                main_module._POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY: (
                    observation.to_dict()
                )
            },
        )

    monkeypatch.setattr(main_module, "_build_portfolio_review_row", fake_build_review_row)

    args = build_parser().parse_args(
        [
            "review-portfolio",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--output-dir",
            str(tmp_path / "daily"),
            "--format",
            "json",
        ]
    )

    assert main_module._handle_review_portfolio(args) == 0
    capsys.readouterr()
    assert main_module._handle_review_portfolio(args) == 0
    capsys.readouterr()

    journal_path = default_position_trajectory_journal_path(tmp_path)
    payload = json.loads(journal_path.read_text(encoding="utf-8"))

    assert len(payload["symbols"]["AAPL"]["observations"]) == 1
    assert payload["symbols"]["AAPL"]["observations"][0]["latest_close"] == pytest.approx(101.0)
    assert payload["symbols"]["AAPL"]["observations"][0]["suggested_action"] == "HOLD"


def test_handle_review_portfolio_ignores_corrupt_position_trajectory_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )
    journal_path = default_position_trajectory_journal_path(tmp_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{bad-json", encoding="utf-8")

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: object())
    monkeypatch.setattr(main_module, "_portfolio_review_preset_catalog", lambda args, config: {})
    monkeypatch.setattr(
        main_module,
        "_build_portfolio_review_plan",
        lambda position, *, preset_catalog, base_settings, as_of_date: {
            "position": position,
            "settings": SimpleNamespace(enable_regime_filter=False),
            "fetch_start": as_of_date,
        },
    )

    def fake_build_review_row(plan: dict[str, object], **_: object) -> PortfolioReviewRow:
        position = plan["position"]
        if not isinstance(position, ExistingPosition):
            raise TypeError("position must be an ExistingPosition")
        observation = PositionTrajectoryObservation(
            symbol=position.symbol,
            as_of_date=date(2024, 1, 5),
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            latest_close=101.0,
            unrealized_pl_pct=0.01,
            above_entry=True,
            high_water_close=110.0,
            high_water_close_date=date(2024, 1, 2),
            days_since_new_high=1,
            stale_position=False,
            relative_strength_return_diff=0.01,
            weak_relative_strength=False,
            suggested_action="HOLD",
        )
        return PortfolioReviewRow(
            date=date(2024, 1, 5),
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=None,
            latest_close=101.0,
            unrealized_pl_pct=0.01,
            distance_to_stop_pct=(101.0 - 90.0) / 101.0,
            regime_passed=True,
            above_entry=True,
            suggested_action="HOLD",
            preset_name="standard_breakout",
            rationale="Position remains healthy.",
            metadata={
                main_module._POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY: (
                    observation.to_dict()
                )
            },
        )

    monkeypatch.setattr(main_module, "_build_portfolio_review_row", fake_build_review_row)

    args = build_parser().parse_args(
        [
            "review-portfolio",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--output-dir",
            str(tmp_path / "daily"),
            "--format",
            "json",
        ]
    )

    with caplog.at_level("WARNING"):
        assert main_module._handle_review_portfolio(args) == 0
    capsys.readouterr()

    rewritten_payload = json.loads(journal_path.read_text(encoding="utf-8"))
    assert rewritten_payload["symbols"]["AAPL"]["observations"]
    assert "Ignoring position trajectory journal" in caplog.text


def test_intraday_portfolio_review_brief_avoids_duplicate_exit_rows_across_sections(
    tmp_path: Path,
) -> None:
    current_positions = [
        ExistingPosition(symbol="AAPL", shares=10, average_entry_price=100.0, current_stop=95.0, preset_name="standard_breakout"),
        ExistingPosition(symbol="ABNB", shares=7, average_entry_price=100.0, current_stop=91.0, preset_name="standard_breakout"),
        ExistingPosition(symbol="AMD", shares=8, average_entry_price=100.0, current_stop=90.0, preset_name="standard_breakout"),
        ExistingPosition(symbol="MSFT", shares=6, average_entry_price=100.0, current_stop=95.0, preset_name="standard_breakout"),
    ]
    report = build_intraday_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        interval_minutes=15,
        portfolio_path="/tmp/portfolio.csv",
        current_positions=current_positions,
        benchmark_symbol="SPY",
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=None,
                latest_close=96.0,
                unrealized_pl_pct=-0.04,
                distance_to_stop_pct=(96.0 - 95.0) / 96.0,
                regime_passed=None,
                above_entry=False,
                suggested_action="EXIT CANDIDATE",
                preset_name="standard_breakout",
                rationale="An intraday bar traded through the current stop.",
                metadata={
                    "stop_breached_intraday": True,
                    "session_vwap": 108.0,
                    "session_high": 112.0,
                    "session_high_giveback_pct": 0.143,
                },
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="ABNB",
                quantity=7,
                average_entry_price=100.0,
                current_stop=91.0,
                suggested_stop=None,
                latest_close=104.0,
                unrealized_pl_pct=0.04,
                distance_to_stop_pct=(104.0 - 91.0) / 104.0,
                regime_passed=None,
                above_entry=True,
                suggested_action="WATCH CLOSELY",
                preset_name="standard_breakout",
                rationale="Strong intraday move is no longer holding the session range.",
                metadata={
                    "failed_intraday_strength": False,
                    "session_vwap": 106.0,
                    "session_high": 109.0,
                    "session_high_giveback_pct": (109.0 - 104.0) / 109.0,
                    "session_high_giveback_exit_threshold": 0.10,
                },
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AMD",
                quantity=8,
                average_entry_price=100.0,
                current_stop=90.0,
                suggested_stop=None,
                latest_close=107.0,
                unrealized_pl_pct=0.07,
                distance_to_stop_pct=(107.0 - 90.0) / 107.0,
                regime_passed=None,
                above_entry=True,
                suggested_action="WATCH CLOSELY",
                preset_name="standard_breakout",
                rationale="Profitable position is fading intraday after a strong session high.",
                metadata={
                    "session_vwap": 111.0,
                    "session_high": 116.0,
                    "session_high_giveback_pct": (116.0 - 107.0) / 116.0,
                },
            ),
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="MSFT",
                quantity=6,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=None,
                latest_close=113.0,
                unrealized_pl_pct=0.13,
                distance_to_stop_pct=(113.0 - 95.0) / 113.0,
                regime_passed=None,
                above_entry=True,
                suggested_action="HOLD",
                preset_name="standard_breakout",
                rationale="Intraday structure remains healthy and no stop or fade signal is active.",
                metadata={
                    "session_vwap": 111.0,
                    "session_high": 114.0,
                    "session_high_giveback_pct": (114.0 - 113.0) / 114.0,
                },
            ),
        ],
    )

    brief_path = write_intraday_portfolio_review_brief(
        report,
        tmp_path / "portfolio_review_intraday_brief.txt",
    )
    text = brief_path.read_text(encoding="utf-8")

    assert "Headline" in text
    assert "Urgent intraday actions" in text
    assert "Current holdings under pressure" in text
    assert "Holdings still healthy" in text
    assert text.count("AAPL") == 1
    urgent_section = text.split("Urgent intraday actions\n", 1)[1].split("\n\nCurrent holdings under pressure", 1)[0]
    pressure_section = text.split("Current holdings under pressure\n", 1)[1].split("\n\nHoldings still healthy", 1)[0]
    healthy_section = text.split("Holdings still healthy\n", 1)[1]
    assert "AAPL" in urgent_section
    assert "AMD" not in urgent_section
    assert pressure_section.count("AAPL") == 0
    assert "ABNB" in pressure_section
    assert "AMD" in pressure_section
    assert "MSFT" not in pressure_section
    assert "MSFT" in healthy_section
    assert "AAPL" not in healthy_section
    assert healthy_section.count("MSFT") == 1
    pressure_lines = [line for line in pressure_section.splitlines() if line.startswith("- ")]
    assert "| ABNB |" in pressure_lines[0]
    assert "| AMD |" in pressure_lines[1]


def test_intraday_session_metrics_uses_full_multi_bar_session() -> None:
    metrics = main_module._intraday_session_metrics(
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2024-01-05 14:30:00", "2024-01-05 14:45:00", "2024-01-05 15:00:00"]
                ),
                "open": [110.0, 112.0, 111.0],
                "high": [113.0, 116.0, 114.0],
                "low": [109.0, 110.0, 108.0],
                "close": [112.0, 111.0, 109.0],
                "volume": [500_000, 600_000, 700_000],
                "vwap": [111.0, 113.0, 112.0],
                "symbol": ["AAPL", "AAPL", "AAPL"],
            }
        )
    )

    assert metrics["session_open"] == 110.0
    assert metrics["session_high"] == 116.0
    assert metrics["session_low"] == 108.0
    assert metrics["latest_close"] == 109.0
    assert metrics["latest_low"] == 108.0
    assert metrics["latest_bar_time"] == "2024-01-05T15:00:00"


def test_intraday_session_vwap_aggregates_explicit_bar_vwap() -> None:
    vwap = main_module._intraday_session_vwap(
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-05 14:30:00", "2024-01-05 14:45:00"]),
                "open": [10.0, 20.0],
                "high": [10.5, 20.5],
                "low": [9.5, 19.5],
                "close": [10.2, 20.2],
                "volume": [100, 300],
                "vwap": [10.0, 20.0],
                "symbol": ["AAPL", "AAPL"],
            }
        )
    )

    assert vwap == pytest.approx((10.0 * 100 + 20.0 * 300) / 400)


def test_build_portfolio_review_intraday_row_uses_provider_and_metrics() -> None:
    config = replace(load_app_config(), project_root=Path("/tmp/intraday-row-project-root"))
    preset = _selected_presets("standard_breakout")[0]
    settings = main_module.BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
    )
    plan = {
        "position": ExistingPosition(
            symbol="AAPL",
            shares=10,
            average_entry_price=100.0,
            current_stop=90.0,
            preset_name="standard_breakout",
            source="manual",
            metadata={"note": "starter"},
        ),
        "preset": preset,
        "preset_resolution": "position_preset",
        "settings": settings,
    }

    class FakeProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def fetch_intraday_bars(
            self,
            symbol: str,
            session_date: date,
            *,
            interval_minutes: int = 15,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            self.calls.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    "interval_minutes": interval_minutes,
                    "refresh_cache": refresh_cache,
                }
            )
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        ["2024-01-05 14:30:00", "2024-01-05 14:45:00", "2024-01-05 15:00:00"]
                    ),
                    "open": [110.0, 113.0, 109.0],
                    "high": [113.0, 116.0, 110.0],
                    "low": [109.0, 108.0, 107.0],
                    "close": [113.0, 109.0, 107.0],
                    "volume": [500_000, 600_000, 550_000],
                    "vwap": [111.0, 111.5, 111.0],
                    "symbol": [symbol] * 3,
                }
            )

    provider = FakeProvider()
    row, observation = main_module._build_portfolio_review_intraday_row(
        plan,
        provider=provider,
        as_of_date=date(2024, 1, 5),
        interval_minutes=15,
        benchmark_symbol="SPY",
        benchmark_intraday_metrics={"intraday_return_vs_open": 0.01},
        refresh_cache=True,
    )

    assert provider.calls == [
        {
            "symbol": "AAPL",
            "session_date": date(2024, 1, 5),
            "interval_minutes": 15,
            "refresh_cache": True,
        }
    ]
    assert row.symbol == "AAPL"
    # Session: AAPL close -2.7% from open, below VWAP, underperforms benchmark +1% by ~3.7%,
    # and gives back 7.8% from the session high — stacked weakness escalates to EXIT CANDIDATE.
    assert row.suggested_action == "EXIT CANDIDATE"
    assert row.metadata["position_source"] == "manual"
    assert row.metadata["position_metadata"] == {"note": "starter"}
    assert row.metadata["session_high"] == 116.0
    assert row.metadata["session_vwap"] == pytest.approx((111.0 * 500_000 + 111.5 * 600_000 + 111.0 * 550_000) / 1_650_000)
    assert row.metadata["intraday_relative_strength_diff"] is not None
    assert observation.timestamp == "2024-01-05T15:00:00"
    assert observation.suggested_action == "EXIT CANDIDATE"


def test_handle_review_portfolio_intraday_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    preset = _selected_presets("standard_breakout")[0]
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )

    class FakeProvider:
        def __init__(self) -> None:
            self.refresh_values: list[bool] = []

        def fetch_intraday_bars(
            self,
            symbol: str,
            session_date: date,
            *,
            interval_minutes: int = 15,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            self.refresh_values.append(refresh_cache)
            if symbol == "SPY":
                return pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(
                            ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                        ),
                        "open": [100.0, 100.5, 100.8],
                        "high": [100.8, 101.0, 101.2],
                        "low": [99.9, 100.4, 100.7],
                        "close": [100.5, 100.8, 101.0],
                        "volume": [1_000_000, 1_100_000, 1_200_000],
                        "vwap": [100.4, 100.6, 100.8],
                        "symbol": [symbol] * 3,
                    }
                )
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                    ),
                    "open": [110.0, 113.0, 109.0],
                    "high": [113.0, 116.0, 110.0],
                    "low": [109.0, 108.0, 107.0],
                    "close": [113.0, 109.0, 107.0],
                    "volume": [500_000, 600_000, 550_000],
                    "vwap": [111.0, 111.5, 111.0],
                    "symbol": [symbol] * 3,
                }
            )

    provider = FakeProvider()
    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: provider)
    monkeypatch.setattr(
        main_module,
        "_portfolio_review_preset_catalog",
        lambda args, config: {preset.name: preset},
    )

    args = build_parser().parse_args(
        [
            "review-portfolio-intraday",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--interval-minutes",
            "15",
            "--output-dir",
            str(tmp_path / "intraday"),
            "--format",
            "json",
        ]
    )

    result = main_module._handle_review_portfolio_intraday(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["interval_minutes"] == 15
    # AAPL: -2.7% from open, below VWAP, underperforms benchmark, 7.8% giveback → EXIT CANDIDATE
    assert payload["watch_closely_count"] == 0
    assert payload["exit_candidate_count"] == 1
    assert Path(payload["outputs"]["portfolio_review_intraday_json"]).exists()
    assert Path(payload["outputs"]["portfolio_review_intraday_csv"]).exists()
    assert Path(payload["outputs"]["portfolio_review_intraday_brief"]).exists()
    assert Path(payload["outputs"]["intraday_state_journal"]).exists()
    assert provider.refresh_values == [True, True]


def test_handle_review_portfolio_intraday_same_timestamp_state_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    preset = _selected_presets("standard_breakout")[0]
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )

    class FakeProvider:
        def fetch_intraday_bars(
            self,
            symbol: str,
            session_date: date,
            *,
            interval_minutes: int = 15,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            if symbol == "SPY":
                return pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(
                            ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                        ),
                        "open": [100.0, 100.5, 100.8],
                        "high": [100.8, 101.0, 101.2],
                        "low": [99.9, 100.4, 100.7],
                        "close": [100.5, 100.8, 101.0],
                        "volume": [1_000_000, 1_100_000, 1_200_000],
                        "vwap": [100.4, 100.6, 100.8],
                        "symbol": [symbol] * 3,
                    }
                )
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                    ),
                    "open": [110.0, 113.0, 109.0],
                    "high": [113.0, 116.0, 110.0],
                    "low": [109.0, 108.0, 107.0],
                    "close": [113.0, 109.0, 107.0],
                    "volume": [500_000, 600_000, 550_000],
                    "vwap": [111.0, 111.5, 111.0],
                    "symbol": [symbol] * 3,
                }
            )

    provider = FakeProvider()
    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: provider)
    monkeypatch.setattr(
        main_module,
        "_portfolio_review_preset_catalog",
        lambda args, config: {preset.name: preset},
    )

    args = build_parser().parse_args(
        [
            "review-portfolio-intraday",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--interval-minutes",
            "15",
            "--output-dir",
            str(tmp_path / "intraday"),
            "--format",
            "json",
        ]
    )

    assert main_module._handle_review_portfolio_intraday(args) == 0
    capsys.readouterr()
    assert main_module._handle_review_portfolio_intraday(args) == 0
    capsys.readouterr()

    journal_path = default_intraday_state_journal_path(tmp_path, date(2024, 1, 5))
    payload = json.loads(journal_path.read_text(encoding="utf-8"))

    assert len(payload["symbols"]["AAPL"]["observations"]) == 1
    assert payload["symbols"]["AAPL"]["observations"][0]["timestamp"] == "2024-01-05T10:00:00"


def test_handle_review_portfolio_intraday_ignores_corrupt_state_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    preset = _selected_presets("standard_breakout")[0]
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )
    journal_path = default_intraday_state_journal_path(tmp_path, date(2024, 1, 5))
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{bad-json", encoding="utf-8")

    class FakeProvider:
        def fetch_intraday_bars(
            self,
            symbol: str,
            session_date: date,
            *,
            interval_minutes: int = 15,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            if symbol == "SPY":
                return pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(
                            ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                        ),
                        "open": [100.0, 100.5, 100.8],
                        "high": [100.8, 101.0, 101.2],
                        "low": [99.9, 100.4, 100.7],
                        "close": [100.5, 100.8, 101.0],
                        "volume": [1_000_000, 1_100_000, 1_200_000],
                        "vwap": [100.4, 100.6, 100.8],
                        "symbol": [symbol] * 3,
                    }
                )
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        ["2024-01-05 09:30:00", "2024-01-05 09:45:00", "2024-01-05 10:00:00"]
                    ),
                    "open": [110.0, 113.0, 109.0],
                    "high": [113.0, 116.0, 110.0],
                    "low": [109.0, 108.0, 107.0],
                    "close": [113.0, 109.0, 107.0],
                    "volume": [500_000, 600_000, 550_000],
                    "vwap": [111.0, 111.5, 111.0],
                    "symbol": [symbol] * 3,
                }
            )

    provider = FakeProvider()
    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: provider)
    monkeypatch.setattr(
        main_module,
        "_portfolio_review_preset_catalog",
        lambda args, config: {preset.name: preset},
    )

    args = build_parser().parse_args(
        [
            "review-portfolio-intraday",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-05",
            "--interval-minutes",
            "15",
            "--output-dir",
            str(tmp_path / "intraday"),
            "--format",
            "json",
        ]
    )

    with caplog.at_level("WARNING"):
        assert main_module._handle_review_portfolio_intraday(args) == 0
    capsys.readouterr()

    rewritten_payload = json.loads(journal_path.read_text(encoding="utf-8"))
    assert rewritten_payload["symbols"]["AAPL"]["observations"]
    assert "Ignoring intraday state journal" in caplog.text


def test_intraday_portfolio_review_brief_includes_trajectory_note(tmp_path: Path) -> None:
    report = build_intraday_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        interval_minutes=15,
        portfolio_path="/tmp/portfolio.csv",
        current_positions=[
            ExistingPosition(
                symbol="AAPL",
                shares=10,
                average_entry_price=100.0,
                current_stop=95.0,
                preset_name="standard_breakout",
            )
        ],
        benchmark_symbol="SPY",
        rows=[
            PortfolioReviewRow(
                date=date(2024, 1, 5),
                symbol="AAPL",
                quantity=10,
                average_entry_price=100.0,
                current_stop=95.0,
                suggested_stop=None,
                latest_close=101.0,
                unrealized_pl_pct=0.01,
                distance_to_stop_pct=(101.0 - 95.0) / 101.0,
                regime_passed=None,
                above_entry=True,
                suggested_action="WATCH CLOSELY",
                preset_name="standard_breakout",
                rationale="Position has remained below session VWAP for 3 consecutive polls.",
                metadata={
                    "session_vwap": 102.5,
                    "session_high": 104.0,
                    "session_high_giveback_pct": 0.03,
                },
            )
        ],
    )

    brief_path = write_intraday_portfolio_review_brief(
        report,
        tmp_path / "portfolio_review_intraday_brief.txt",
    )
    text = brief_path.read_text(encoding="utf-8")

    assert "below session VWAP for 3 consecutive polls" in text


def test_handle_review_portfolio_intraday_returns_skipped_when_no_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_app_config()
    preset = _selected_presets("standard_breakout")[0]
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n"
        "AAPL,10,100,90,standard_breakout,manual,{}\n",
        encoding="utf-8",
    )

    class EmptyProvider:
        def fetch_intraday_bars(
            self,
            symbol: str,
            session_date: date,
            *,
            interval_minutes: int = 15,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "datetime": pd.Series(dtype="datetime64[ns]"),
                    "open": pd.Series(dtype="float64"),
                    "high": pd.Series(dtype="float64"),
                    "low": pd.Series(dtype="float64"),
                    "close": pd.Series(dtype="float64"),
                    "volume": pd.Series(dtype="int64"),
                    "vwap": pd.Series(dtype="float64"),
                    "symbol": pd.Series(dtype="object"),
                }
            )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: EmptyProvider())
    monkeypatch.setattr(
        main_module,
        "_portfolio_review_preset_catalog",
        lambda args, config: {preset.name: preset},
    )

    args = build_parser().parse_args(
        [
            "review-portfolio-intraday",
            "--portfolio-file",
            str(portfolio_path),
            "--as-of",
            "2024-01-06",
            "--interval-minutes",
            "15",
            "--format",
            "json",
        ]
    )

    result = main_module._handle_review_portfolio_intraday(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["status"] == "skipped"
    assert payload["reason"] == "no_intraday_data"
    assert payload["position_count"] == 1
    assert payload["symbols_reviewed"] == []
    assert payload["outputs"] == {}


def test_run_daily_summary_workflow_treats_held_symbol_as_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    current_positions = [
        ExistingPosition(
            symbol=" aaa ",
            shares=10,
            average_entry_price=100.0,
            current_stop=95.0,
            preset_name="standard_breakout",
        )
    ]
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAA")
        ],
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    signal_calls: list[tuple[str, bool]] = []

    def fake_generate_breakout_signal(
        bars: pd.DataFrame,
        *,
        settings: object,
        benchmark_frame: pd.DataFrame | None,
        has_open_position: bool,
        symbol: str,
    ) -> None:
        signal_calls.append((symbol, has_open_position))
        return None

    monkeypatch.setattr(main_module, "generate_breakout_signal", fake_generate_breakout_signal)

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=current_positions,
    )
    summary = workflow["summary"]
    payload = summary.to_dict()

    assert signal_calls == [("AAA", True)]
    assert payload["candidate_count"] == 0
    assert payload["rejected_count"] == 0
    assert payload["unique_no_signal_symbols"] == ["AAA"]


def test_run_daily_summary_workflow_uses_full_candidate_file_without_strategy_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    candidate_path = tmp_path / "candidates.txt"
    symbols = [f"SYM{index:03d}" for index in range(105)]
    candidate_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "daily-summary",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [50.0],
                    "high": [51.0],
                    "low": [49.0],
                    "close": [50.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda *args, **kwargs: None,
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    payload = workflow["summary"].to_dict()

    assert payload["universe_count"] == 105
    assert payload["candidate_count"] == 0
    assert payload["unique_no_signal_symbol_count"] == 105


def test_run_daily_summary_workflow_rejects_candidate_near_earnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(
        main_module,
        "_load_earnings_contexts",
        lambda **kwargs: {
            "AAPL": EarningsRiskContext(
                symbol="AAPL",
                earnings_date=date(2024, 1, 9),
                earnings_days_away=2,
                is_earnings_risk=True,
                status="confirmed",
                name="Q1 earnings",
                source="polygon",
            )
        },
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.8},
        ),
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    payload = workflow["summary"].to_dict()

    assert payload["candidate_count"] == 1
    assert payload["approved_count"] == 0
    assert payload["rejected_count"] == 1
    assert payload["rejected_candidates"][0]["rejection_reasons"] == [
        "Upcoming earnings on 2024-01-09 are 2 trading days away; new entries are blocked within 3 trading days."
    ]
    assert payload["rejected_candidates"][0]["metadata"]["signal_metadata"]["earnings_days_away"] == 2
    assert payload["rejected_candidates"][0]["metadata"]["signal_metadata"]["is_earnings_risk"] is True


def test_run_daily_summary_workflow_rejects_candidate_on_weak_sector_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_sector_contexts",
        lambda **kwargs: {
            "AAPL": SectorFeatureContext(
                symbol="AAPL",
                sector_name="Technology",
                industry_name="Application software",
                sector_etf_symbol="XLK",
                sector_regime_passed=False,
                sector_return=-0.04,
                symbol_return=-0.08,
                relative_strength_vs_sector=-0.04,
                relative_strength_window=20,
                sector_trend_state="weak",
                mapping_source="test",
            )
        },
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.8},
        ),
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    payload = workflow["summary"].to_dict()

    assert payload["approved_count"] == 0
    assert payload["rejected_count"] == 1
    assert payload["rejected_candidates"][0]["rejection_reasons"] == [
        "Sector ETF XLK is below its trend filter; new entries require sector support."
    ]
    assert payload["rejected_candidates"][0]["metadata"]["signal_metadata"]["sector_etf_symbol"] == "XLK"
    assert "sector=XLK (below trend filter, lags by 4.0% over 20d)" in payload["rejected_candidates"][0]["rationale"]


def test_run_daily_summary_workflow_rejects_candidate_on_portfolio_heat_and_reports_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )
    current_positions = [
        ExistingPosition(symbol="MSFT", shares=10, average_entry_price=100.0),
        ExistingPosition(symbol="NVDA", shares=10, average_entry_price=100.0),
        ExistingPosition(symbol="AMD", shares=10, average_entry_price=100.0),
    ]

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_sector_contexts",
        lambda **kwargs: {
            "AAPL": SectorFeatureContext(
                symbol="AAPL",
                sector_name="Technology",
                industry_name="Software",
                sector_etf_symbol="XLK",
                sector_regime_passed=True,
                sector_return=0.04,
                symbol_return=0.06,
                relative_strength_vs_sector=0.02,
                relative_strength_window=20,
                sector_trend_state="supportive",
                mapping_source="test",
            )
        },
    )
    monkeypatch.setattr(
        main_module,
        "_load_position_sector_classifications",
        lambda *args, **kwargs: {
            "MSFT": SymbolSectorClassification(
                symbol="MSFT",
                sector="Technology",
                industry="Software",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
            "NVDA": SymbolSectorClassification(
                symbol="NVDA",
                sector="Technology",
                industry="Semiconductors",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
            "AMD": SymbolSectorClassification(
                symbol="AMD",
                sector="Technology",
                industry="Semiconductors",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
        },
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.8},
        ),
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=current_positions,
    )
    summary = workflow["summary"]
    payload = summary.to_dict()
    brief = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
    ).to_brief()

    assert payload["approved_count"] == 0
    assert payload["rejected_count"] == 1
    assert payload["rejected_candidates"][0]["rejection_reasons"] == [
        "Portfolio already has 3 Technology positions; adding another would exceed the per-sector limit of 3."
    ]
    assert payload["rejected_candidates"][0]["metadata"]["portfolio_heat_context"][
        "same_sector_position_count"
    ] == 3
    assert "portfolio already has 3 technology positions" in brief.lower()


def test_run_daily_summary_workflow_serializes_projected_portfolio_heat_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    config = replace(
        config,
        strategy=replace(
            config.strategy,
            signals=replace(
                config.strategy.signals,
                max_positions_per_sector=4,
                max_same_industry_positions=4,
                max_sector_notional_pct=0.60,
            ),
        ),
    )
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )
    current_positions = [
        ExistingPosition(symbol="MSFT", shares=100, average_entry_price=100.0),
        ExistingPosition(symbol="NVDA", shares=150, average_entry_price=100.0),
        ExistingPosition(symbol="JPM", shares=250, average_entry_price=100.0),
    ]

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_sector_contexts",
        lambda **kwargs: {
            "AAPL": SectorFeatureContext(
                symbol="AAPL",
                sector_name="Technology",
                industry_name="Software",
                sector_etf_symbol="XLK",
                sector_regime_passed=True,
                sector_return=0.03,
                symbol_return=0.05,
                relative_strength_vs_sector=0.02,
                relative_strength_window=20,
                sector_trend_state="supportive",
                mapping_source="test",
            )
        },
    )
    monkeypatch.setattr(
        main_module,
        "_load_position_sector_classifications",
        lambda *args, **kwargs: {
            "MSFT": SymbolSectorClassification(
                symbol="MSFT",
                sector="Technology",
                industry="Software",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
            "NVDA": SymbolSectorClassification(
                symbol="NVDA",
                sector="Technology",
                industry="Semiconductors",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
            "JPM": SymbolSectorClassification(
                symbol="JPM",
                sector="Financials",
                industry="Banks",
                sector_etf_symbol="XLF",
                mapping_source="test",
            ),
        },
    )
    monkeypatch.setattr(
        main_module,
        "compute_market_breadth_from_frames",
        lambda *args, **kwargs: {
            "market_breadth_pct_above_200ma": 0.72,
            "market_breadth_pct_above_50ma": 0.76,
            "market_breadth_state": "healthy",
        },
    )
    monkeypatch.setattr(
        main_module,
        "_load_volatility_context",
        lambda **kwargs: {
            "vix_close": 20.0,
            "vix_sma_short": 19.5,
            "vix_sma_long": 18.8,
            "volatility_regime_state": "normal",
            "volatility_regime_risk_off": False,
        },
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=100.0,
            stop_hint=95.0,
            metadata={"prior_high": 99.0, "relative_volume": 1.8},
        ),
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=current_positions,
    )
    payload = workflow["summary"].to_dict()
    rejected_candidate = payload["rejected_candidates"][0]
    projection = rejected_candidate["metadata"]["portfolio_heat_projection"]

    assert rejected_candidate["rejection_reasons"] == [
        "Technology already accounts for 50.0% of approximate portfolio notional; adding this position would raise it to 64.3%, above the 60% limit."
    ]
    assert projection["projected_sector_notional_pct"] == pytest.approx(45_000.0 / 70_000.0)
    assert projection["sector_notional_concentration_risk"] is True
    assert projection["sector_concentration_risk"] is True
    assert projection["crowded_exposure_bucket"] is True


def test_run_daily_summary_workflow_warns_and_proceeds_without_sector_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = load_app_config()
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.8},
        ),
    )

    with caplog.at_level("WARNING"):
        workflow = main_module._run_daily_summary_workflow(
            args,
            config=config,
            provider=FakeProvider(),
            current_positions=[],
        )

    payload = workflow["summary"].to_dict()

    assert payload["approved_count"] == 1
    assert payload["rejected_count"] == 0
    assert "Sector regime enforcement is enabled but sector context is unavailable for AAPL during daily-summary:standard_breakout; proceeding without sector gate." in caplog.text


def test_run_daily_summary_workflow_updates_candidate_score_journal_across_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1_500_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.9},
        ),
    )

    first_workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    first_journal_path = Path(first_workflow["candidate_score_journal_path"])
    first_payload = json.loads(first_journal_path.read_text(encoding="utf-8"))

    second_args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-08",
            "--disable-regime-filter",
        ]
    )
    second_workflow = main_module._run_daily_summary_workflow(
        second_args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    second_payload = json.loads(first_journal_path.read_text(encoding="utf-8"))
    second_summary = second_workflow["summary"].to_dict()

    assert first_journal_path.exists()
    assert first_payload["symbols"]["AAPL"]["days_near_breakout"] == 1
    assert first_payload["symbols"]["AAPL"]["days_approved"] == 1
    assert second_payload["symbols"]["AAPL"]["days_near_breakout"] == 2
    assert second_payload["symbols"]["AAPL"]["days_approved"] == 2
    assert second_payload["symbols"]["AAPL"]["last_rank"] == 1
    assert second_summary["approved_candidates"][0]["metadata"]["signal_metadata"]["setup_persistence_days"] == 2
    assert "setup=high-confidence repeat signal, persisted for 2 sessions, approved on 2 sessions" in second_summary["approved_candidates"][0]["rationale"]


def test_run_daily_summary_workflow_ignores_corrupt_candidate_score_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    journal_path = tmp_path / "data" / "processed" / "state" / "candidate_scores.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{bad-json", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1_500_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.9},
        ),
    )

    with caplog.at_level("WARNING"):
        workflow = main_module._run_daily_summary_workflow(
            args,
            config=config,
            provider=FakeProvider(),
            current_positions=[],
        )

    payload = workflow["summary"].to_dict()
    rewritten_payload = json.loads(journal_path.read_text(encoding="utf-8"))

    assert payload["approved_count"] == 1
    assert rewritten_payload["symbols"]["AAPL"]["days_near_breakout"] == 1
    assert "Ignoring candidate score journal" in caplog.text


def test_comparison_warmup_start_accounts_for_relative_volume_window() -> None:
    presets = (
        BreakoutStrategyPreset(
            name="rv_heavy",
            breakout_lookback=20,
            relative_volume_threshold=1.5,
            initial_stop_atr=2.0,
            trailing_stop_atr=3.0,
            risk_per_trade=0.01,
        ),
    )

    fetch_start = main_module._comparison_warmup_start(
        start_date=date(2024, 1, 10),
        atr_window=14,
        benchmark_sma_slow=50,
        presets=presets,
        max_relative_volume_window=80,
        max_sector_relative_strength_window=20,
        market_breadth_long_window=1,
        enable_regime_filter=False,
    )

    assert fetch_start == date(2024, 1, 10) - timedelta(days=243)


def test_handle_generate_orders_uses_full_candidate_file_without_strategy_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    symbols = [f"SYM{index:03d}" for index in range(105)]
    candidate_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "generate-orders",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
            "--output-dir",
            str(tmp_path / "orders"),
            "--format",
            "json",
        ]
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [50.0],
                    "high": [51.0],
                    "low": [49.0],
                    "close": [50.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: FakeProvider())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: [])
    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda *args, **kwargs: None,
    )

    result = main_module._handle_generate_orders(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["universe_count"] == 105
    assert payload["signal_count"] == 0


def test_handle_generate_orders_blocks_candidate_on_weak_market_breadth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("AAPL\n", encoding="utf-8")
    output_dir = tmp_path / "orders"
    args = build_parser().parse_args(
        [
            "generate-orders",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [99.0],
                    "high": [101.0],
                    "low": [98.0],
                    "close": [100.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: FakeProvider())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: [])
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "compute_market_breadth_from_frames",
        lambda *args, **kwargs: {
            "market_breadth_pct_above_200ma": 0.37,
            "market_breadth_pct_above_50ma": 0.48,
            "market_breadth_state": "weak",
        },
    )
    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda *args, **kwargs: StrategySignal(
            strategy_name="breakout_momentum",
            symbol="AAPL",
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=100.0,
            stop_hint=95.0,
            metadata={"prior_high": 99.0, "relative_volume": 1.8},
        ),
    )

    result = main_module._handle_generate_orders(args)
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(
        (output_dir / "daily_signal_report.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert payload["market_breadth_state"] == "weak"
    assert payload["approved_order_count"] == 0
    assert payload["rejected_signal_count"] == 1
    assert report_payload["rows"][0]["rejection_reasons"] == [
        "Market breadth is weak: 37% of the universe is above its 200-day moving average; new entries require at least 40%."
    ]


def test_handle_generate_orders_blocks_candidate_on_stressed_volatility_regime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("AAPL\n", encoding="utf-8")
    output_dir = tmp_path / "orders"
    args = build_parser().parse_args(
        [
            "generate-orders",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [99.0],
                    "high": [101.0],
                    "low": [98.0],
                    "close": [100.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: FakeProvider())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: [])
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_volatility_context",
        lambda **kwargs: {
            "vix_close": 33.4,
            "vix_sma_short": 31.0,
            "vix_sma_long": 26.2,
            "volatility_regime_state": "stressed",
            "volatility_regime_risk_off": True,
        },
    )
    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda *args, **kwargs: StrategySignal(
            strategy_name="breakout_momentum",
            symbol="AAPL",
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=100.0,
            stop_hint=95.0,
            metadata={"prior_high": 99.0, "relative_volume": 1.8},
        ),
    )

    result = main_module._handle_generate_orders(args)
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(
        (output_dir / "daily_signal_report.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert payload["vix_close"] == pytest.approx(33.4)
    assert payload["volatility_regime_state"] == "stressed"
    assert payload["approved_order_count"] == 0
    assert payload["rejected_signal_count"] == 1
    assert report_payload["rows"][0]["rejection_reasons"] == [
        "Volatility regime is stressed: VIX is 33.4, above the 30.0 entry block threshold."
    ]


def test_handle_generate_orders_blocks_candidate_near_earnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("AAPL\n", encoding="utf-8")
    output_dir = tmp_path / "orders"
    args = build_parser().parse_args(
        [
            "generate-orders",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [99.0],
                    "high": [101.0],
                    "low": [98.0],
                    "close": [100.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: FakeProvider())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: [])
    monkeypatch.setattr(
        main_module,
        "_load_earnings_contexts",
        lambda **kwargs: {
            "AAPL": EarningsRiskContext(
                symbol="AAPL",
                earnings_date=date(2024, 1, 9),
                earnings_days_away=2,
                is_earnings_risk=True,
            )
        },
    )
    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda *args, **kwargs: StrategySignal(
            strategy_name="breakout_momentum",
            symbol="AAPL",
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_high",
            entry_price_hint=100.0,
            stop_hint=95.0,
            metadata={"prior_high": 99.0, "relative_volume": 2.0},
        ),
    )

    result = main_module._handle_generate_orders(args)
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(
        Path(payload["outputs"]["daily_signal_report_json"]).read_text(encoding="utf-8")
    )

    assert result == 0
    assert payload["signal_count"] == 1
    assert payload["approved_order_count"] == 0
    assert payload["rejected_signal_count"] == 1
    assert report_payload["rows"][0]["status"] == "rejected"
    assert report_payload["rows"][0]["rejection_reasons"] == [
        "Upcoming earnings on 2024-01-09 are 2 trading days away; new entries are blocked within 3 trading days."
    ]
    assert report_payload["rows"][0]["metadata"]["signal_metadata"]["earnings_days_away"] == 2


def test_handle_generate_orders_rolls_forward_approved_sector_exposure_between_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path = tmp_path / "candidates.txt"
    candidate_path.write_text("AAPL\nSNOW\n", encoding="utf-8")
    output_dir = tmp_path / "orders"
    args = build_parser().parse_args(
        [
            "generate-orders",
            str(candidate_path),
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    )
    current_positions = [
        ExistingPosition(symbol="MSFT", shares=100, average_entry_price=100.0),
        ExistingPosition(symbol="NVDA", shares=100, average_entry_price=100.0),
    ]

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [99.0],
                    "high": [101.0],
                    "low": [98.0],
                    "close": [100.0],
                    "volume": [2_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: FakeProvider())
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: current_positions)
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_sector_contexts",
        lambda **kwargs: {
            "AAPL": SectorFeatureContext(
                symbol="AAPL",
                sector_name="Technology",
                industry_name="Software",
                sector_etf_symbol="XLK",
                sector_regime_passed=True,
                sector_return=0.03,
                symbol_return=0.05,
                relative_strength_vs_sector=0.02,
                relative_strength_window=20,
                sector_trend_state="supportive",
                mapping_source="test",
            ),
            "SNOW": SectorFeatureContext(
                symbol="SNOW",
                sector_name="Technology",
                industry_name="Semiconductors",
                sector_etf_symbol="XLK",
                sector_regime_passed=True,
                sector_return=0.03,
                symbol_return=0.05,
                relative_strength_vs_sector=0.02,
                relative_strength_window=20,
                sector_trend_state="supportive",
                mapping_source="test",
            ),
        },
    )
    monkeypatch.setattr(
        main_module,
        "_load_position_sector_classifications",
        lambda *args, **kwargs: {
            "MSFT": SymbolSectorClassification(
                symbol="MSFT",
                sector="Technology",
                industry="Software",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
            "NVDA": SymbolSectorClassification(
                symbol="NVDA",
                sector="Technology",
                industry="Semiconductors",
                sector_etf_symbol="XLK",
                mapping_source="test",
            ),
        },
    )
    monkeypatch.setattr(
        main_module,
        "compute_market_breadth_from_frames",
        lambda *args, **kwargs: {
            "market_breadth_pct_above_200ma": 0.72,
            "market_breadth_pct_above_50ma": 0.76,
            "market_breadth_state": "healthy",
        },
    )
    monkeypatch.setattr(
        main_module,
        "_load_volatility_context",
        lambda **kwargs: {
            "vix_close": 20.0,
            "vix_sma_short": 19.5,
            "vix_sma_long": 18.8,
            "volatility_regime_state": "normal",
            "volatility_regime_risk_off": False,
        },
    )
    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=100.0,
            stop_hint=95.0,
            metadata={"prior_high": 99.0, "relative_volume": 1.8},
        ),
    )

    result = main_module._handle_generate_orders(args)
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(
        (output_dir / "daily_signal_report.json").read_text(encoding="utf-8")
    )

    assert result == 0
    assert payload["signal_count"] == 2
    assert payload["approved_order_count"] == 1
    assert payload["rejected_signal_count"] == 1
    assert report_payload["rows"][0]["status"] == "approved"
    assert report_payload["rows"][1]["status"] == "rejected"
    assert report_payload["rows"][1]["rejection_reasons"] == [
        "Portfolio already has 3 Technology positions; adding another would exceed the per-sector limit of 3."
    ]


def test_handle_daily_summary_uses_workflow_universe_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preset = _selected_presets("aggressive_breakout")[0]
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5)),
        evaluations=[],
        selected_presets=[preset],
        universe_symbols=[f"SYM{index:03d}" for index in range(105)],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={preset.name: tuple(f"SYM{index:03d}" for index in range(105))},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
        current_positions=[],
    )

    monkeypatch.setattr(main_module, "load_app_config", lambda config_dir: SimpleNamespace(project_root=tmp_path, data_sources=SimpleNamespace(provider="polygon")))
    monkeypatch.setattr(main_module, "create_daily_bar_provider", lambda config, env_file: object())
    loaded_positions = [
        ExistingPosition(
            symbol="AAPL",
            shares=5,
            average_entry_price=100.0,
            current_stop=95.0,
            preset_name="aggressive_breakout",
        )
    ]
    monkeypatch.setattr(main_module, "_load_current_positions", lambda portfolio_file: loaded_positions)
    captured_current_positions: dict[str, object] = {}

    def fake_run_daily_summary_workflow(args, *, config, provider, current_positions=None):
        captured_current_positions["value"] = current_positions
        return {
            "summary": summary,
            "presets": [preset],
            "preset_selection_source": "named_presets",
            "current_positions": loaded_positions,
            "current_equity": 100_000.0,
            "execution_batch": ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5)),
        }

    monkeypatch.setattr(
        main_module,
        "_run_daily_summary_workflow",
        fake_run_daily_summary_workflow,
    )

    args = build_parser().parse_args(
        [
            "--config-dir",
            str(tmp_path / "config"),
            "--env-file",
            str(tmp_path / ".env"),
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--portfolio-file",
            str(tmp_path / "portfolio.csv"),
            "--output-dir",
            str(tmp_path / "daily"),
            "--format",
            "json",
        ]
    )

    result = main_module._handle_daily_summary(args)
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert captured_current_positions["value"] == loaded_positions
    assert payload["universe_count"] == 105
    assert payload["current_position_count"] == 1
    assert payload["current_position_symbols"] == ["AAPL"]
    assert Path(payload["outputs"]["daily_summary_json"]).exists()
    assert Path(payload["outputs"]["daily_summary_brief"]).exists()


def test_position_entry_date_returns_date_object() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        metadata={"entry_date": "2024-01-03"},
    )

    entry_date = main_module._position_entry_date(position)

    assert entry_date == date(2024, 1, 3)
    assert not isinstance(entry_date, str)


def test_portfolio_review_reference_close_filters_to_post_entry_dates_only() -> None:
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=95.0,
        metadata={"entry_date": "2024-01-03"},
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [200.0, 150.0, 160.0],
        }
    )

    reference_close = main_module._portfolio_review_reference_close(bars, position=position)

    assert reference_close == pytest.approx(160.0)


def test_build_portfolio_review_row_includes_profit_protection_metadata() -> None:
    preset = _selected_presets("aggressive_breakout")[0]
    settings = main_module.BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
        preset_name=preset.name,
        metadata={"entry_date": "2024-01-02"},
    )
    plan = {
        "position": position,
        "preset": preset,
        "preset_resolution": "portfolio_snapshot",
        "settings": preset.apply_to_settings(settings),
        "fetch_start": date(2024, 1, 1),
    }

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                    ),
                    "open": [100.0, 101.0, 129.0, 120.0, 114.0],
                    "high": [101.0, 130.0, 131.0, 121.0, 115.0],
                    "low": [99.0, 100.0, 118.0, 112.0, 110.0],
                    "close": [100.0, 101.0, 130.0, 120.0, 114.0],
                    "volume": [1_000_000.0] * 5,
                    "symbol": [symbol] * 5,
                }
            )

    row = main_module._build_portfolio_review_row(
        plan,
        provider=FakeProvider(),
        as_of_date=date(2024, 1, 5),
        benchmark_frame=None,
        refresh_cache=False,
    )

    assert row.suggested_action == "EXIT CANDIDATE"
    assert row.metadata["high_water_close"] == pytest.approx(130.0)
    assert row.metadata["giveback_pct"] == pytest.approx((130.0 - 114.0) / 130.0)
    assert row.metadata["days_since_new_high"] == 2
    assert row.metadata["failed_breakout_detected"] is False


def test_build_portfolio_review_row_includes_earnings_metadata() -> None:
    preset = _selected_presets("standard_breakout")[0]
    settings = main_module.BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
        preset_name=preset.name,
    )
    plan = {
        "position": position,
        "preset": preset,
        "preset_resolution": "portfolio_snapshot",
        "settings": preset.apply_to_settings(settings),
        "fetch_start": date(2024, 1, 1),
    }

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1_000_000.0] * 3,
                    "symbol": [symbol] * 3,
                }
            )

    row = main_module._build_portfolio_review_row(
        plan,
        provider=FakeProvider(),
        as_of_date=date(2024, 1, 5),
        benchmark_frame=None,
        refresh_cache=False,
        earnings_context=EarningsRiskContext(
            symbol="AAPL",
            earnings_date=date(2024, 1, 10),
            earnings_days_away=3,
            is_earnings_risk=True,
            status="confirmed",
            name="Q1 earnings",
            source="polygon",
        ),
    )

    assert row.suggested_action == "WATCH CLOSELY"
    assert row.metadata["earnings_days_away"] == 3
    assert row.metadata["is_earnings_risk"] is True
    assert "upcoming earnings are scheduled on 2024-01-10" in row.rationale.lower()


def test_build_portfolio_review_row_includes_sector_context() -> None:
    preset = _selected_presets("standard_breakout")[0]
    settings = main_module.BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
        preset_name=preset.name,
    )
    plan = {
        "position": position,
        "preset": preset,
        "preset_resolution": "portfolio_snapshot",
        "settings": preset.apply_to_settings(settings),
        "fetch_start": date(2024, 1, 1),
    }

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1_000_000.0] * 3,
                    "symbol": [symbol] * 3,
                }
            )

    row = main_module._build_portfolio_review_row(
        plan,
        provider=FakeProvider(),
        as_of_date=date(2024, 1, 5),
        benchmark_frame=None,
        refresh_cache=False,
        sector_context=SectorFeatureContext(
            symbol="AAPL",
            sector_name="Technology",
            industry_name="Application software",
            sector_etf_symbol="XLK",
            sector_regime_passed=False,
            sector_return=-0.02,
            symbol_return=-0.08,
            relative_strength_vs_sector=-0.06,
            relative_strength_window=20,
            sector_trend_state="weak",
            mapping_source="test",
        ),
    )

    assert row.suggested_action == "WATCH CLOSELY"
    assert row.metadata["sector_etf_symbol"] == "XLK"
    assert row.metadata["lagging_sector"] is True
    assert "sector etf xlk is below its trend filter" in row.rationale.lower()
    assert "lagging xlk by 6.0% over 20 trading days" in row.rationale.lower()


def test_build_portfolio_review_row_and_monitor_brief_include_position_trajectory_note() -> None:
    preset = _selected_presets("standard_breakout")[0]
    settings = main_module.BreakoutMomentumSettings.from_configs(
        _signal_config(),
        _risk_config(),
        enable_regime_filter=False,
    )
    position = ExistingPosition(
        symbol="AAPL",
        shares=10,
        average_entry_price=100.0,
        current_stop=90.0,
        preset_name=preset.name,
    )
    plan = {
        "position": position,
        "preset": preset,
        "preset_resolution": "portfolio_snapshot",
        "settings": preset.apply_to_settings(settings),
        "fetch_start": date(2024, 1, 1),
    }

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                    "open": [100.0, 99.0, 98.0],
                    "high": [101.0, 100.0, 99.0],
                    "low": [98.0, 97.0, 96.0],
                    "close": [99.0, 98.0, 97.0],
                    "volume": [1_000_000.0] * 3,
                    "symbol": [symbol] * 3,
                }
            )

    position_trajectory_journal = PositionTrajectoryJournal(
        entries={
            "AAPL": (
                PositionTrajectoryObservation(
                    symbol="AAPL",
                    as_of_date=date(2024, 1, 3),
                    average_entry_price=100.0,
                    current_stop=90.0,
                    latest_close=99.0,
                    unrealized_pl_pct=-0.01,
                    above_entry=False,
                    high_water_close=110.0,
                    high_water_close_date=date(2024, 1, 2),
                    days_since_new_high=1,
                    stale_position=False,
                    relative_strength_return_diff=0.01,
                    weak_relative_strength=False,
                    suggested_action="WATCH CLOSELY",
                ),
                PositionTrajectoryObservation(
                    symbol="AAPL",
                    as_of_date=date(2024, 1, 4),
                    average_entry_price=100.0,
                    current_stop=90.0,
                    latest_close=98.0,
                    unrealized_pl_pct=-0.02,
                    above_entry=False,
                    high_water_close=110.0,
                    high_water_close_date=date(2024, 1, 2),
                    days_since_new_high=2,
                    stale_position=False,
                    relative_strength_return_diff=0.01,
                    weak_relative_strength=False,
                    suggested_action="WATCH CLOSELY",
                ),
            )
        }
    )

    row = main_module._build_portfolio_review_row(
        plan,
        provider=FakeProvider(),
        as_of_date=date(2024, 1, 5),
        benchmark_frame=None,
        refresh_cache=False,
        position_trajectory_journal=position_trajectory_journal,
    )

    report = build_portfolio_review_report(
        as_of_date=date(2024, 1, 5),
        rows=[row],
        current_positions=[position],
    )
    brief = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        portfolio_review=report,
    ).to_brief()

    assert row.suggested_action == "EXIT CANDIDATE"
    assert "below entry for 3 consecutive review sessions" in row.rationale.lower()
    assert "below entry for 3 consecutive review sessions" in brief.lower()


def test_daily_research_summary_reports_confirmed_breakout_rv_policy_in_header() -> None:
    preset = _selected_presets("confirmed_breakout")[0]
    execution_batch = ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5))
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[],
        selected_presets=[preset],
        universe_symbols=["MU"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"confirmed_breakout": ("MU",)},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    payload = summary.to_dict()

    assert payload["relative_volume_confirmation_required"] is True
    assert payload["relative_volume_policy"] == "required"
    assert payload["relative_volume_policy_by_preset"] == {"confirmed_breakout": "required"}
    assert payload["candidate_count"] == 0


def test_daily_research_summary_reports_confirmed_conservative_breakout_rv_policy_in_header() -> None:
    preset = _selected_presets("confirmed_conservative_breakout")[0]
    execution_batch = ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5))
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[],
        selected_presets=[preset],
        universe_symbols=["MU"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"confirmed_conservative_breakout": ("MU",)},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    payload = summary.to_dict()

    assert payload["relative_volume_confirmation_required"] is True
    assert payload["relative_volume_policy"] == "required"
    assert payload["relative_volume_policy_by_preset"] == {
        "confirmed_conservative_breakout": "required"
    }
    assert payload["selected_presets"][0]["preset_name"] == "confirmed_conservative_breakout"


def test_daily_research_summary_exports_ranked_rows_and_preset_summaries(tmp_path: Path) -> None:
    presets = _selected_presets("standard_breakout", "aggressive_breakout")
    evaluations = [
        _evaluation(presets[0], symbol="AAA", approved=True, shares=30),
        _evaluation(
            presets[1],
            symbol="BBB",
            approved=False,
            shares=0,
            rejection_reasons=("No averaging down is allowed for existing long positions.",),
        ),
    ]
    execution_batch = ManualExecutor().build_execution_batch(
        [evaluation.candidate for evaluation in evaluations],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=evaluations,
        selected_presets=presets,
        universe_symbols=["AAA", "BBB"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": (), "aggressive_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    preset_json = write_daily_preset_summary(summary, tmp_path / "preset_rankings.json")
    ranked_csv = write_daily_research_summary(summary, tmp_path / "ranked_opportunities.csv")

    preset_payload = json.loads(preset_json.read_text(encoding="utf-8"))
    with ranked_csv.open("r", encoding="utf-8", newline="") as handle:
        ranked_rows = list(csv.DictReader(handle))

    assert preset_payload[0]["preset_rank"] == 1
    assert ranked_rows[0]["preset_name"] == "standard_breakout"
    assert ranked_rows[0]["status"] == "approved"
    assert ranked_rows[1]["rejection_reasons"] == "No averaging down is allowed for existing long positions."


def test_daily_research_brief_includes_trade_cards_and_lower_priority_section(
    tmp_path: Path,
) -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=40)
    rejected = _evaluation(
        preset,
        symbol="BBB",
        approved=False,
        shares=0,
        rejection_reasons=("Max concurrent positions reached.",),
    )
    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate, rejected.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved, rejected],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ("CCC",)},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
        current_positions=[
            ExistingPosition(
                symbol="MSFT",
                shares=5,
                average_entry_price=200.0,
                current_stop=190.0,
                preset_name="standard_breakout",
            )
        ],
    )

    brief_path = write_daily_research_brief(
        summary,
        tmp_path / "daily_summary_brief.txt",
        output_paths={
            "daily_summary_json": "/tmp/daily_summary.json",
            "ranked_opportunities_csv": "/tmp/ranked_opportunities.csv",
        },
    )
    text = brief_path.read_text(encoding="utf-8")

    assert "Headline" in text
    assert "Top opportunities" in text
    assert "Lower-priority / rejected candidates" in text
    assert "Portfolio context" in text
    assert "Output files" in text
    assert "1. AAA" in text
    assert "qty=40" in text
    assert "entry=" in text
    assert "stop=" in text
    assert "notional=" in text
    assert "- BBB | preset=standard_breakout | reason=Max concurrent positions reached." in text
    assert "Current holdings: 1 | symbols=MSFT" in text
    assert "- daily_summary_json: /tmp/daily_summary.json" in text
    assert text.index("1. AAA") < text.index("- BBB | preset=standard_breakout")


def test_daily_research_summary_includes_current_holdings_context_for_rejections() -> None:
    preset = _selected_presets("standard_breakout")[0]
    existing_position = ExistingPosition(
        symbol="AAA",
        shares=15,
        average_entry_price=110.0,
        current_stop=100.0,
        preset_name="standard_breakout",
        source="portfolio.csv",
    )
    rejected = _evaluation(
        preset,
        symbol="AAA",
        approved=False,
        shares=0,
        rejection_reasons=("No averaging down is allowed for existing long positions.",),
        existing_position=existing_position,
    )

    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=ManualExecutor().build_execution_batch([], as_of_date=date(2024, 1, 5)),
        evaluations=[rejected],
        selected_presets=[preset],
        universe_symbols=["AAA"],
        current_positions=[existing_position],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )
    payload = summary.to_dict()

    assert payload["current_position_count"] == 1
    assert payload["current_positions"][0]["symbol"] == "AAA"
    assert payload["rejected_candidates"][0]["metadata"]["existing_position"]["current_stop"] == 100.0


def test_daily_research_and_monitor_briefs_include_market_breadth_context() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        market_context=build_market_context(
            as_of_date=date(2024, 1, 5),
            benchmark_symbol="SPY",
            market_breadth_pct_above_200ma=0.37,
            market_breadth_pct_above_50ma=0.51,
            market_breadth_state="weak",
        ),
        preset_selection_source="named_presets",
    )

    summary_brief = summary.to_brief()
    monitor_brief = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
    ).to_brief()

    assert "Market breadth: weak | 37% above 200d | 51% above 50d" in summary_brief
    assert "Breadth: weak | 37% above 200d | 51% above 50d" in monitor_brief


def test_daily_research_and_monitor_briefs_include_volatility_context() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        market_context=build_market_context(
            as_of_date=date(2024, 1, 5),
            benchmark_symbol="SPY",
            vix_close=26.4,
            vix_sma_short=25.1,
            vix_sma_long=21.7,
            volatility_regime_state="elevated",
            volatility_regime_risk_off=False,
        ),
        preset_selection_source="named_presets",
    )

    summary_brief = summary.to_brief()
    monitor_brief = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
    ).to_brief()

    assert "Volatility regime: elevated | VIX 26.4 | 5d 25.1 | 20d 21.7" in summary_brief
    assert "Volatility: elevated | VIX 26.4 | 5d 25.1 | 20d 21.7" in monitor_brief


def test_daily_research_and_monitor_briefs_handle_unknown_market_breadth_state() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(preset, symbol="AAA", approved=True, shares=25)
    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        market_context=build_market_context(
            as_of_date=date(2024, 1, 5),
            benchmark_symbol="SPY",
            market_breadth_pct_above_200ma=0.37,
            market_breadth_pct_above_50ma=0.51,
            market_breadth_state=None,
        ),
        preset_selection_source="named_presets",
    )

    summary_brief = summary.to_brief()
    monitor_brief = build_market_monitor_report(
        as_of_date=date(2024, 1, 5),
        daily_summary=summary,
    ).to_brief()

    assert "Market breadth: 37% above 200d | 51% above 50d" in summary_brief
    assert "Market breadth: n/a" not in summary_brief
    assert "Breadth: 37% above 200d | 51% above 50d" in monitor_brief
    assert "Breadth: n/a" not in monitor_brief


def test_run_daily_summary_workflow_includes_elevated_volatility_in_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config()
    args = build_parser().parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--disable-regime-filter",
        ]
    )

    monkeypatch.setattr(
        main_module.UniverseBuilder,
        "screen_candidates",
        lambda self, candidate_path, *, as_of_date, lookback_days, refresh_cache, enforce_max_symbols=True: [
            SimpleNamespace(symbol="AAPL")
        ],
    )
    monkeypatch.setattr(main_module, "_load_earnings_contexts", lambda **kwargs: {})
    monkeypatch.setattr(main_module, "_load_sector_contexts", lambda **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "_load_volatility_context",
        lambda **kwargs: {
            "vix_close": 26.4,
            "vix_sma_short": 25.1,
            "vix_sma_long": 21.7,
            "volatility_regime_state": "elevated",
            "volatility_regime_risk_off": False,
        },
    )

    class FakeProvider:
        def fetch_daily_bars(
            self,
            symbol: str,
            start_date: date,
            end_date: date,
            *,
            refresh_cache: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-05"]),
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000_000.0],
                    "symbol": [symbol],
                }
            )

    monkeypatch.setattr(
        main_module,
        "generate_breakout_signal",
        lambda bars, *, settings, benchmark_frame, has_open_position, symbol: StrategySignal(
            strategy_name="breakout_momentum",
            symbol=symbol,
            date=date(2024, 1, 5),
            side="BUY",
            entry_reason="close_above_prior_20_day_high",
            entry_price_hint=101.0,
            stop_hint=96.0,
            metadata={"prior_high": 100.0, "relative_volume": 1.8},
        ),
    )

    workflow = main_module._run_daily_summary_workflow(
        args,
        config=config,
        provider=FakeProvider(),
        current_positions=[],
    )
    payload = workflow["summary"].to_dict()

    assert payload["approved_count"] == 1
    assert payload["market_context"]["vix_close"] == pytest.approx(26.4)
    assert payload["market_context"]["volatility_regime_state"] == "elevated"
    assert (
        "volatility=elevated (VIX 26.4 above 25.0 caution)"
        in payload["approved_candidates"][0]["rationale"]
    )


def test_daily_research_summary_labels_relative_volume_policy_in_rationale() -> None:
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(
        preset,
        symbol="AAA",
        approved=True,
        shares=20,
        relative_volume=1.2,
        relative_volume_confirmed=False,
        relative_volume_required=False,
    )
    rejected = _evaluation(
        preset,
        symbol="BBB",
        approved=False,
        shares=0,
        relative_volume=1.2,
        relative_volume_confirmed=False,
        relative_volume_required=False,
        rejection_reasons=("Calculated share size is zero after risk and notional constraints.",),
    )

    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate, rejected.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved, rejected],
        selected_presets=[preset],
        universe_symbols=["AAA", "BBB"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    approved_row = next(row for row in summary.rows if row.symbol == "AAA")
    rejected_row = next(row for row in summary.rows if row.symbol == "BBB")

    assert "relative_volume=1.20 (optional; threshold=1.50; not confirmed)" in approved_row.rationale
    assert "relative_volume=1.20 (optional; threshold=1.50; not confirmed)" in rejected_row.rationale
    assert approved_row.metadata["signal_metadata"]["relative_volume_policy"] == "optional"
    assert rejected_row.metadata["signal_metadata"]["relative_volume_gate_passed"] is True


def test_daily_research_summary_labels_required_relative_volume_when_enabled() -> None:
    preset = _selected_presets("confirmed_breakout")[0]
    approved = _evaluation(
        preset,
        symbol="AAA",
        approved=True,
        shares=20,
        relative_volume=2.0,
        relative_volume_confirmed=True,
    )

    execution_batch = ManualExecutor().build_execution_batch(
        [approved.candidate],
        as_of_date=date(2024, 1, 5),
    )
    summary = build_daily_research_summary(
        as_of_date=date(2024, 1, 5),
        execution_batch=execution_batch,
        evaluations=[approved],
        selected_presets=[preset],
        universe_symbols=["AAA"],
        current_equity=100_000.0,
        no_signal_symbols_by_preset={"standard_breakout": ()},
        benchmark_symbol="SPY",
        preset_selection_source="named_presets",
    )

    assert preset.require_relative_volume_confirmation is True
    assert "relative_volume=2.00 (required; threshold=1.50; confirmed)" in summary.rows[0].rationale


def test_load_top_preset_name_from_results_supports_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "ranked_presets.csv"
    csv_path.write_text(
        "rank,preset_name\n1,standard_breakout\n2,aggressive_breakout\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "summary.json"
    json_path.write_text(
        json.dumps({"top_presets": [{"preset_name": "conservative_breakout"}]}),
        encoding="utf-8",
    )

    assert _load_top_preset_name_from_results(csv_path) == "standard_breakout"
    assert _load_top_preset_name_from_results(json_path) == "conservative_breakout"


def test_cli_parser_exposes_daily_summary_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--preset-names",
            "standard_breakout,aggressive_breakout",
            "--portfolio-file",
            "data/processed/portfolio.csv",
        ]
    )

    assert args.command == "daily-summary"
    assert args.as_of == date(2024, 1, 5)
    assert args.preset_names == "standard_breakout,aggressive_breakout"
    assert args.portfolio_file == Path("data/processed/portfolio.csv")


def test_daily_summary_preset_names_with_spaces_are_parsed_correctly() -> None:
    # Verifies the daily-summary CLI path handles "name, name" (comma-space) input.
    # The raw argparse value retains the user's string; _parse_text_list strips it.
    parser = build_parser()
    args = parser.parse_args(
        [
            "daily-summary",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--preset-names",
            "standard_breakout, aggressive_breakout",
        ]
    )
    # argparse stores the raw string unchanged — that is expected.
    assert args.preset_names == "standard_breakout, aggressive_breakout"
    # _parse_text_list (the CLI split/parse boundary) must strip whitespace.
    parsed = _parse_text_list(args.preset_names)
    assert parsed == ("standard_breakout", "aggressive_breakout")


def test_cli_parser_exposes_portfolio_snapshot_commands() -> None:
    parser = build_parser()

    init_args = parser.parse_args(
        [
            "init-portfolio",
            "data/processed/portfolio/current_positions.json",
            "--snapshot-format",
            "json",
        ]
    )
    upsert_args = parser.parse_args(
        [
            "upsert-position",
            "data/processed/portfolio/current_positions.csv",
            "AAPL",
            "--quantity",
            "10",
            "--average-entry-price",
            "150",
            "--current-stop",
            "145",
            "--preset-name",
            "standard_breakout",
        ]
    )
    update_stop_args = parser.parse_args(
        [
            "update-stop",
            "data/processed/portfolio/current_positions.csv",
            "AAPL",
            "--current-stop",
            "145",
        ]
    )
    remove_position_args = parser.parse_args(
        [
            "remove-position",
            "data/processed/portfolio/current_positions.json",
            "AAPL",
        ]
    )

    assert init_args.command == "init-portfolio"
    assert init_args.output_path == Path("data/processed/portfolio/current_positions.json")
    assert init_args.snapshot_format == "json"
    assert upsert_args.command == "upsert-position"
    assert upsert_args.portfolio_path == Path("data/processed/portfolio/current_positions.csv")
    assert upsert_args.symbol == "AAPL"
    assert upsert_args.quantity == 10
    assert upsert_args.average_entry_price == 150.0
    assert update_stop_args.command == "update-stop"
    assert update_stop_args.portfolio_path == Path(
        "data/processed/portfolio/current_positions.csv"
    )
    assert update_stop_args.symbol == "AAPL"
    assert update_stop_args.current_stop == 145.0
    assert remove_position_args.command == "remove-position"
    assert remove_position_args.portfolio_path == Path(
        "data/processed/portfolio/current_positions.json"
    )
    assert remove_position_args.symbol == "AAPL"


def _evaluation(
    preset: BreakoutStrategyPreset,
    *,
    symbol: str,
    approved: bool,
    shares: int,
    prior_high: float = 98.0,
    entry_price: float = 100.0,
    relative_volume: float = 1.8,
    relative_volume_confirmed: bool | None = None,
    relative_volume_required: bool | None = None,
    rejection_reasons: tuple[str, ...] = (),
    existing_position: ExistingPosition | None = None,
) -> PresetCandidateEvaluation:
    relative_volume_threshold = preset.relative_volume_threshold
    required = (
        preset.require_relative_volume_confirmation
        if relative_volume_required is None
        else relative_volume_required
    )
    confirmed = (
        relative_volume >= relative_volume_threshold
        if relative_volume_confirmed is None
        else relative_volume_confirmed
    )
    strategy_name = f"breakout_momentum:{preset.name}"
    signal = StrategySignal(
        strategy_name=strategy_name,
        symbol=symbol,
        date=date(2024, 1, 4),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=entry_price,
        stop_hint=95.0,
        metadata={
            "preset_name": preset.name,
            "parameter_id": preset.parameter_id,
            "prior_high": prior_high,
            "relative_volume": relative_volume,
            "relative_volume_threshold": relative_volume_threshold,
            "relative_volume_confirmed": confirmed,
            "relative_volume_required": required,
            "relative_volume_policy": (
                "required" if required else "optional"
            ),
            "relative_volume_gate_passed": confirmed or not required,
        },
    )
    sizing = PositionSizingResult(
        shares=shares,
        risk_budget=500.0,
        per_share_risk=5.0,
        notional_value=shares * entry_price,
        max_shares_by_risk=max(shares, 0),
        max_shares_by_notional=max(shares, 0),
        capped_by_notional=False,
        is_valid=approved,
        rejection_reason=rejection_reasons[0] if rejection_reasons else None,
    )
    return PresetCandidateEvaluation(
        preset_name=preset.name,
        parameter_id=preset.parameter_id,
        candidate=RiskAssessedCandidate(
            signal=signal,
            entry_price=entry_price,
            stop_price=95.0,
            adjusted_risk_per_trade=preset.risk_per_trade,
            sizing=sizing,
            approved=approved,
            rejection_reasons=rejection_reasons,
            existing_position=existing_position,
        ),
    )


def _selected_presets(*names: str) -> list[BreakoutStrategyPreset]:
    presets = build_default_breakout_presets(_signal_config(), _risk_config())
    return [presets[name] for name in names]


def _signal_config():
    from bot.config import SignalConfig

    return SignalConfig(
        breakout_lookback=20,
        benchmark_symbol="SPY",
        benchmark_sma_fast=50,
        benchmark_sma_slow=200,
        relative_volume_threshold=1.5,
    )


def _risk_config():
    from bot.config import RiskConfig

    return RiskConfig(
        risk_per_trade=0.01,
        atr_length=14,
        initial_stop_atr=2.5,
        trailing_stop_atr=3.0,
        drawdown_risk_reduction_threshold=0.15,
    )
