from __future__ import annotations

from datetime import date

import pytest

from bot.backtest.metrics import (
    COMPARISON_COLUMNS,
    build_strategy_comparison_frame,
    empty_summary_metrics,
    rank_strategy_comparisons,
)
from bot.config import RiskConfig, SignalConfig
from bot.main import build_parser
from bot.strategy.breakout_momentum import (
    build_default_breakout_presets,
    resolve_breakout_strategy_presets,
)


def test_resolve_breakout_strategy_presets_supports_defaults_config_and_cli() -> None:
    presets = resolve_breakout_strategy_presets(
        _signal_config(),
        _risk_config(),
        configured_presets={
            "research_breakout": {
                "breakout_lookback": 30,
                "relative_volume_threshold": 1.7,
                "initial_stop_atr": 2.7,
                "trailing_stop_atr": 3.3,
                "risk_per_trade": 0.008,
            }
        },
        cli_preset_definitions=(
            "name=cli_fast,breakout_lookback=12,relative_volume_threshold=1.2,"
            "initial_stop_atr=2.0,trailing_stop_atr=2.5,risk_per_trade=0.015",
        ),
        preset_names=("conservative_breakout", "research_breakout", "cli_fast"),
    )

    assert [preset.name for preset in presets] == [
        "conservative_breakout",
        "research_breakout",
        "cli_fast",
    ]
    assert presets[1].breakout_lookback == 30
    assert presets[2].risk_per_trade == pytest.approx(0.015)


def test_build_default_breakout_presets_exposes_named_variants() -> None:
    presets = build_default_breakout_presets(_signal_config(), _risk_config())

    assert list(presets) == [
        "conservative_breakout",
        "standard_breakout",
        "aggressive_breakout",
    ]
    assert presets["standard_breakout"].breakout_lookback == 20
    assert presets["conservative_breakout"].risk_per_trade < presets["standard_breakout"].risk_per_trade
    assert presets["aggressive_breakout"].risk_per_trade > presets["standard_breakout"].risk_per_trade


def test_rank_strategy_comparisons_orders_rows_by_objective() -> None:
    frame = build_strategy_comparison_frame(
        [
            _comparison_row(
                "conservative_breakout",
                total_return=0.08,
                sharpe_ratio=1.1,
                max_drawdown=0.04,
                expectancy=40.0,
            ),
            _comparison_row(
                "standard_breakout",
                total_return=0.10,
                sharpe_ratio=1.3,
                max_drawdown=0.06,
                expectancy=45.0,
            ),
            _comparison_row(
                "aggressive_breakout",
                total_return=0.09,
                sharpe_ratio=1.3,
                max_drawdown=0.03,
                expectancy=42.0,
            ),
        ]
    )

    ranked = rank_strategy_comparisons(frame, objective="sharpe_ratio", top_n=2)

    assert ranked["preset_name"].tolist() == ["aggressive_breakout", "standard_breakout"]
    assert ranked["rank"].tolist() == [1, 2]


def test_build_strategy_comparison_frame_returns_expected_schema() -> None:
    frame = build_strategy_comparison_frame([_comparison_row("standard_breakout")])

    assert tuple(frame.columns) == COMPARISON_COLUMNS
    assert frame.iloc[0]["preset_name"] == "standard_breakout"
    assert frame.iloc[0]["trade_count"] == 3


def test_rank_strategy_comparisons_handles_no_trade_rows_cleanly() -> None:
    frame = build_strategy_comparison_frame(
        [
            _comparison_row("conservative_breakout", metrics=empty_summary_metrics()),
            _comparison_row("standard_breakout", metrics=empty_summary_metrics()),
        ]
    )

    ranked = rank_strategy_comparisons(frame, objective="total_return")

    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["trade_count"].tolist() == [0, 0]
    assert ranked["total_return"].tolist() == [0.0, 0.0]


def test_cli_parser_exposes_compare_strategies_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-strategies",
            "data/raw/candidate_symbols.txt",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--preset-names",
            "standard_breakout,aggressive_breakout",
        ]
    )

    assert args.command == "compare-strategies"
    assert args.start == date(2024, 1, 1)
    assert args.end == date(2024, 3, 31)
    assert args.preset_names == "standard_breakout,aggressive_breakout"


def _comparison_row(
    preset_name: str,
    *,
    total_return: float = 0.05,
    sharpe_ratio: float = 1.0,
    max_drawdown: float = 0.05,
    expectancy: float = 25.0,
    metrics: dict[str, float | int] | None = None,
) -> dict[str, float | int | str]:
    preset = build_default_breakout_presets(_signal_config(), _risk_config())[preset_name]
    row = preset.to_dict()
    defaults = {
        "starting_equity": 100_000.0,
        "ending_equity": 105_000.0,
        "total_return": total_return,
        "cagr": 0.12,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "trade_count": 3,
        "win_rate": 2.0 / 3.0,
        "average_win": 120.0,
        "average_loss": -60.0,
        "expectancy": expectancy,
    }
    if metrics is None:
        resolved_metrics = dict(defaults)
    else:
        resolved_metrics = dict(metrics)
        for key, value in defaults.items():
            resolved_metrics.setdefault(key, value)
    row.update(resolved_metrics)
    return row


def _signal_config() -> SignalConfig:
    return SignalConfig(
        breakout_lookback=20,
        benchmark_symbol="SPY",
        benchmark_sma_fast=50,
        benchmark_sma_slow=200,
        relative_volume_threshold=1.5,
    )


def _risk_config() -> RiskConfig:
    return RiskConfig(
        risk_per_trade=0.01,
        atr_length=14,
        initial_stop_atr=2.5,
        trailing_stop_atr=3.0,
        drawdown_risk_reduction_threshold=0.15,
    )
