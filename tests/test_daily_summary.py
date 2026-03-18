from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import bot.main as main_module
from bot.execution.manual_executor import ManualExecutor
from bot.main import _load_top_preset_name_from_results, _parse_text_list, build_parser
from bot.reporting.daily_report import (
    MARKET_MONITOR_CATEGORIES,
    PresetCandidateEvaluation,
    PortfolioReviewRow,
    build_daily_research_summary,
    build_market_monitor_report,
    build_portfolio_review_report,
    ensure_market_monitor_categories_cover_portfolio_actions,
    market_monitor_flat_count_key,
    write_market_monitor_report,
    write_market_monitor_text_summary,
    write_daily_preset_summary,
    write_portfolio_review_report,
    write_daily_research_summary,
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
            suggested_stop=position.current_stop,
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
