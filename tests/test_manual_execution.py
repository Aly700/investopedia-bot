from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd

from bot.execution.interface import ExecutionBatch
from bot.execution.manual_executor import ManualExecutor, write_execution_batch
from bot.main import build_parser
from bot.reporting.daily_report import build_daily_signal_report, write_daily_signal_report
from bot.reporting.equity_curve import write_equity_curve_report
from bot.reporting.trade_log import write_trade_log_report
from bot.risk.portfolio_rules import ExistingPosition, RiskAssessedCandidate
from bot.risk.position_sizing import PositionSizingResult
from bot.strategy.signal_models import StrategySignal


def test_manual_executor_renders_order_sheet_from_approved_candidate(tmp_path: Path) -> None:
    candidate = _candidate(symbol="AAA", approved=True, shares=50, rejection_reasons=())
    executor = ManualExecutor()

    batch = executor.build_execution_batch([candidate], as_of_date=date(2024, 1, 5))
    output_path = write_execution_batch(batch, tmp_path / "manual_order_sheet.csv")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(batch.orders) == 1
    assert rows[0]["date"] == "2024-01-05"
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["action"] == "BUY"
    assert rows[0]["quantity"] == "50"
    assert rows[0]["intended_order_type"] == "MARKET"
    assert rows[0]["entry_price_hint"] == "100"
    assert rows[0]["stop_level"] == "95"
    assert "close above prior 20 day high" in rows[0]["rationale"]


def test_manual_executor_labels_optional_relative_volume_in_rationale(tmp_path: Path) -> None:
    candidate = _candidate(
        symbol="AAA",
        approved=True,
        shares=50,
        rejection_reasons=(),
        relative_volume=1.2,
        relative_volume_confirmed=False,
        relative_volume_required=False,
    )
    batch = ManualExecutor().build_execution_batch([candidate], as_of_date=date(2024, 1, 5))
    output_path = write_execution_batch(batch, tmp_path / "manual_order_sheet.csv")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert "relative_volume=1.20 (optional; threshold=1.50; not confirmed)" in rows[0]["rationale"]


def test_manual_executor_labels_required_relative_volume_in_rationale(tmp_path: Path) -> None:
    candidate = _candidate(
        symbol="AAA",
        approved=True,
        shares=50,
        rejection_reasons=(),
        relative_volume=2.0,
        relative_volume_confirmed=True,
        relative_volume_required=True,
    )
    batch = ManualExecutor().build_execution_batch([candidate], as_of_date=date(2024, 1, 5))
    output_path = write_execution_batch(batch, tmp_path / "manual_order_sheet.csv")

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert "relative_volume=2.00 (required; threshold=1.50; confirmed)" in rows[0]["rationale"]


def test_write_execution_batch_handles_empty_order_sets(tmp_path: Path) -> None:
    batch = ExecutionBatch(
        executor_name="manual",
        as_of_date=date(2024, 1, 5),
        generated_at_utc="2024-01-05T00:00:00+00:00",
        orders=(),
    )

    csv_path = write_execution_batch(batch, tmp_path / "orders.csv")
    json_path = write_execution_batch(batch, tmp_path / "orders.json")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert rows == []
    assert payload["order_count"] == 0
    assert payload["orders"] == []


def test_reporting_exports_trade_log_and_equity_curve(tmp_path: Path) -> None:
    trade_log = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "entry_date": pd.to_datetime(["2024-01-03"]),
            "exit_date": pd.to_datetime(["2024-01-05"]),
            "net_pnl": [125.0],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "equity": [10_000.0, 10_125.0],
        }
    )

    trade_csv = write_trade_log_report(trade_log, tmp_path / "trade_log.csv")
    trade_json = write_trade_log_report(trade_log, tmp_path / "trade_log.json")
    curve_csv = write_equity_curve_report(equity_curve, tmp_path / "equity_curve.csv")
    curve_json = write_equity_curve_report(equity_curve, tmp_path / "equity_curve.json")

    trade_payload = json.loads(trade_json.read_text(encoding="utf-8"))
    curve_payload = json.loads(curve_json.read_text(encoding="utf-8"))

    assert trade_csv.exists()
    assert curve_csv.exists()
    assert trade_payload[0]["entry_date"] == "2024-01-03"
    assert curve_payload[1]["equity"] == 10_125.0


def test_daily_signal_report_exports_rejections_and_no_signal_summary(tmp_path: Path) -> None:
    approved = _candidate(symbol="AAA", approved=True, shares=25, rejection_reasons=())
    existing_position = ExistingPosition(
        symbol="BBB",
        shares=20,
        average_entry_price=105.0,
        current_stop=95.0,
        preset_name="standard_breakout",
        source="portfolio.csv",
    )
    rejected = _candidate(
        symbol="BBB",
        approved=False,
        shares=0,
        rejection_reasons=("Existing long position already open for symbol; duplicate entries are not allowed.",),
        relative_volume=1.2,
        relative_volume_confirmed=False,
        relative_volume_required=False,
        existing_position=existing_position,
    )
    batch = ManualExecutor().build_execution_batch([approved, rejected], as_of_date=date(2024, 1, 5))
    report = build_daily_signal_report(
        as_of_date=date(2024, 1, 5),
        execution_batch=batch,
        assessed_candidates=[approved, rejected],
        universe_symbols=["AAA", "BBB", "CCC"],
        current_positions=[existing_position],
        no_signal_symbols=["CCC"],
        benchmark_symbol="SPY",
    )

    json_path = write_daily_signal_report(report, tmp_path / "daily_signal_report.json")
    csv_path = write_daily_signal_report(report, tmp_path / "daily_signal_report.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert payload["approved_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["no_signal_count"] == 1
    assert payload["current_position_count"] == 1
    assert payload["benchmark_symbol"] == "SPY"
    assert [row["status"] for row in rows] == ["approved", "rejected"]
    assert "relative_volume=1.20 (optional; threshold=1.50; not confirmed)" in rows[1]["rationale"]
    assert rows[1]["rejection_reasons"] == (
        "Existing long position already open for symbol; duplicate entries are not allowed."
    )
    assert payload["rows"][1]["metadata"]["existing_position"]["average_entry_price"] == 105.0


def test_cli_parser_exposes_generate_orders_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate-orders",
            "data/raw/candidate_symbols.txt",
            "--as-of",
            "2024-01-05",
            "--portfolio-file",
            "data/processed/portfolio.json",
        ]
    )

    assert args.command == "generate-orders"
    assert args.as_of == date(2024, 1, 5)
    assert args.portfolio_file == Path("data/processed/portfolio.json")


def _candidate(
    *,
    symbol: str,
    approved: bool,
    shares: int,
    rejection_reasons: tuple[str, ...],
    relative_volume: float = 1.8,
    relative_volume_confirmed: bool | None = None,
    relative_volume_required: bool = False,
    existing_position: ExistingPosition | None = None,
) -> RiskAssessedCandidate:
    relative_volume_threshold = 1.5
    confirmed = (
        relative_volume >= relative_volume_threshold
        if relative_volume_confirmed is None
        else relative_volume_confirmed
    )
    signal = StrategySignal(
        strategy_name="breakout_momentum",
        symbol=symbol,
        date=date(2024, 1, 4),
        side="BUY",
        entry_reason="close_above_prior_20_day_high",
        entry_price_hint=100.0,
        stop_hint=95.0,
        metadata={
            "prior_high": 99.5,
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
        notional_value=shares * 100.0,
        max_shares_by_risk=max(shares, 0),
        max_shares_by_notional=max(shares, 0),
        capped_by_notional=False,
        is_valid=approved,
        rejection_reason=rejection_reasons[0] if rejection_reasons else None,
    )
    return RiskAssessedCandidate(
        signal=signal,
        entry_price=100.0,
        stop_price=95.0,
        adjusted_risk_per_trade=0.01,
        sizing=sizing,
        approved=approved,
        rejection_reasons=rejection_reasons,
        existing_position=existing_position,
    )
