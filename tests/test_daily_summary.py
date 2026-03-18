from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from bot.execution.manual_executor import ManualExecutor
from bot.main import _load_top_preset_name_from_results, build_parser
from bot.reporting.daily_report import (
    PresetCandidateEvaluation,
    build_daily_research_summary,
    write_daily_preset_summary,
    write_daily_research_summary,
)
from bot.risk.portfolio_rules import RiskAssessedCandidate
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
    assert payload["recommended_preset"] is None
    assert opportunity_rows == []
    assert preset_rows[0]["no_signal_count"] == "2"


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
    preset = _selected_presets("standard_breakout")[0]
    approved = _evaluation(
        preset,
        symbol="AAA",
        approved=True,
        shares=20,
        relative_volume=2.0,
        relative_volume_confirmed=True,
        relative_volume_required=True,
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
        ]
    )

    assert args.command == "daily-summary"
    assert args.as_of == date(2024, 1, 5)
    assert args.preset_names == "standard_breakout,aggressive_breakout"


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
    relative_volume_required: bool = False,
    rejection_reasons: tuple[str, ...] = (),
) -> PresetCandidateEvaluation:
    relative_volume_threshold = preset.relative_volume_threshold
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
            "relative_volume_required": relative_volume_required,
            "relative_volume_policy": (
                "required" if relative_volume_required else "optional"
            ),
            "relative_volume_gate_passed": confirmed or not relative_volume_required,
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
