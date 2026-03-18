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
from bot.main import _parse_text_list, build_parser
from bot.strategy.breakout_momentum import (
    BreakoutMomentumSettings,
    BreakoutStrategyPreset,
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
                "require_relative_volume_confirmation": True,
            }
        },
        cli_preset_definitions=(
            "name=cli_fast,breakout_lookback=12,relative_volume_threshold=1.2,"
            "initial_stop_atr=2.0,trailing_stop_atr=2.5,risk_per_trade=0.015,"
            "require_relative_volume_confirmation=false",
        ),
        preset_names=("conservative_breakout", "research_breakout", "cli_fast"),
    )

    assert [preset.name for preset in presets] == [
        "conservative_breakout",
        "research_breakout",
        "cli_fast",
    ]
    assert presets[1].breakout_lookback == 30
    assert presets[1].require_relative_volume_confirmation is True
    assert presets[2].risk_per_trade == pytest.approx(0.015)
    assert presets[2].require_relative_volume_confirmation is False


def test_build_default_breakout_presets_exposes_named_variants() -> None:
    presets = build_default_breakout_presets(_signal_config(), _risk_config())

    assert list(presets) == [
        "conservative_breakout",
        "confirmed_conservative_breakout",
        "standard_breakout",
        "confirmed_breakout",
        "aggressive_breakout",
    ]
    assert presets["standard_breakout"].breakout_lookback == 20
    assert (
        presets["confirmed_conservative_breakout"].breakout_lookback
        == presets["conservative_breakout"].breakout_lookback
    )
    assert (
        presets["confirmed_conservative_breakout"].relative_volume_threshold
        == presets["conservative_breakout"].relative_volume_threshold
    )
    assert (
        presets["confirmed_conservative_breakout"].initial_stop_atr
        == presets["conservative_breakout"].initial_stop_atr
    )
    assert (
        presets["confirmed_conservative_breakout"].trailing_stop_atr
        == presets["conservative_breakout"].trailing_stop_atr
    )
    assert (
        presets["confirmed_conservative_breakout"].risk_per_trade
        == presets["conservative_breakout"].risk_per_trade
    )
    assert presets["conservative_breakout"].require_relative_volume_confirmation is False
    assert presets["confirmed_conservative_breakout"].require_relative_volume_confirmation is True
    assert (
        presets["confirmed_breakout"].breakout_lookback
        == presets["standard_breakout"].breakout_lookback
    )
    assert (
        presets["confirmed_breakout"].relative_volume_threshold
        == presets["standard_breakout"].relative_volume_threshold
    )
    assert (
        presets["confirmed_breakout"].initial_stop_atr
        == presets["standard_breakout"].initial_stop_atr
    )
    assert (
        presets["confirmed_breakout"].trailing_stop_atr
        == presets["standard_breakout"].trailing_stop_atr
    )
    assert (
        presets["confirmed_breakout"].risk_per_trade
        == presets["standard_breakout"].risk_per_trade
    )
    assert presets["standard_breakout"].require_relative_volume_confirmation is False
    assert presets["confirmed_breakout"].require_relative_volume_confirmation is True
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


def test_parse_text_list_strips_whitespace_at_split_boundary() -> None:
    # Simulates "--preset-names standard_breakout, aggressive_breakout" (comma-space).
    # Stripping must happen in _parse_text_list itself, not only in downstream lookup.
    result = _parse_text_list("standard_breakout, aggressive_breakout")
    assert result == ("standard_breakout", "aggressive_breakout")


def test_parse_text_list_handles_mixed_whitespace_and_empty_segments() -> None:
    result = _parse_text_list("  standard_breakout ,, aggressive_breakout  ")
    assert result == ("standard_breakout", "aggressive_breakout")


def test_preset_names_with_spaces_resolve_to_correct_presets() -> None:
    # End-to-end: spaced names from CLI → resolve_breakout_strategy_presets → correct presets.
    presets = resolve_breakout_strategy_presets(
        _signal_config(),
        _risk_config(),
        preset_names=("standard_breakout ", " aggressive_breakout"),
    )
    assert [p.name for p in presets] == ["standard_breakout", "aggressive_breakout"]


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


def test_apply_to_settings_propagates_trailing_stop_atr() -> None:
    """apply_to_settings must wire trailing_stop_atr from the preset into BreakoutMomentumSettings.

    This test would fail before the fix because BreakoutMomentumSettings had no
    trailing_stop_atr field, so apply_to_settings silently dropped the preset value.
    """
    base_settings = BreakoutMomentumSettings.from_configs(_signal_config(), _risk_config())
    # _risk_config() has trailing_stop_atr=3.0; use a clearly different preset value.
    preset = BreakoutStrategyPreset(
        name="custom_trail",
        breakout_lookback=15,
        relative_volume_threshold=1.5,
        initial_stop_atr=2.5,
        trailing_stop_atr=1.0,
        risk_per_trade=0.01,
    )

    applied = preset.apply_to_settings(base_settings)

    assert applied.trailing_stop_atr == pytest.approx(1.0)
    assert applied.trailing_stop_atr != base_settings.trailing_stop_atr


def test_apply_to_settings_preserves_trailing_stop_atr_across_all_default_presets() -> None:
    """Every default preset's trailing_stop_atr must survive a round-trip through apply_to_settings."""
    base_settings = BreakoutMomentumSettings.from_configs(_signal_config(), _risk_config())
    presets = build_default_breakout_presets(_signal_config(), _risk_config())

    for preset in presets.values():
        applied = preset.apply_to_settings(base_settings)
        assert applied.trailing_stop_atr == pytest.approx(preset.trailing_stop_atr), (
            f"Preset '{preset.name}': trailing_stop_atr not propagated. "
            f"Expected {preset.trailing_stop_atr}, got {applied.trailing_stop_atr}."
        )


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
