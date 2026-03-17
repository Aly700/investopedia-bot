from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bot.backtest.metrics import aggregate_metric_frame, empty_summary_metrics, select_top_parameter_sets
from bot.backtest.walkforward import (
    SweepParameterSet,
    build_parameter_grid,
    evaluate_walkforward,
    generate_walkforward_folds,
)


def test_generate_walkforward_folds_supports_rolling_and_expanding() -> None:
    rolling = generate_walkforward_folds(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        train_window_days=10,
        test_window_days=5,
        expanding_train=False,
    )
    expanding = generate_walkforward_folds(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        train_window_days=10,
        test_window_days=5,
        expanding_train=True,
    )

    assert rolling[0].train_start == date(2024, 1, 1)
    assert rolling[0].train_end == date(2024, 1, 10)
    assert rolling[0].test_start == date(2024, 1, 11)
    assert rolling[1].train_start == date(2024, 1, 6)
    assert rolling[1].train_end == date(2024, 1, 15)
    assert expanding[1].train_start == date(2024, 1, 1)
    assert expanding[1].train_end == date(2024, 1, 15)


def test_build_parameter_grid_returns_cartesian_product() -> None:
    grid = build_parameter_grid(
        breakout_lookbacks=[20, 40],
        relative_volume_thresholds=[1.0],
        initial_stop_atrs=[2.0, 2.5],
        trailing_stop_atrs=[3.0],
        risk_per_trade_values=[0.005, 0.01],
    )

    assert len(grid) == 8
    assert grid[0].parameter_id.startswith("lookback=")
    assert {item.breakout_lookback for item in grid} == {20, 40}
    assert {item.risk_per_trade for item in grid} == {0.005, 0.01}


def test_evaluate_walkforward_records_train_and_test_metrics() -> None:
    folds = generate_walkforward_folds(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 20),
        train_window_days=5,
        test_window_days=5,
    )
    parameter_sets = [
        SweepParameterSet(10, 1.0, 2.0, 3.0, 0.01),
        SweepParameterSet(20, 1.0, 2.0, 3.0, 0.01),
    ]

    def runner(parameter_set: SweepParameterSet, window_start: date, window_end: date) -> dict[str, float | int]:
        metrics = empty_summary_metrics()
        if window_end <= folds[0].train_end:
            metrics["sharpe_ratio"] = 2.0 if parameter_set.breakout_lookback == 20 else 1.0
            metrics["total_return"] = 0.10 if parameter_set.breakout_lookback == 20 else 0.05
        else:
            metrics["sharpe_ratio"] = 1.5 if parameter_set.breakout_lookback == 20 else 0.8
            metrics["total_return"] = 0.08 if parameter_set.breakout_lookback == 20 else 0.03
        metrics["trade_count"] = 3
        return metrics

    result = evaluate_walkforward(
        folds=folds[:1],
        parameter_sets=parameter_sets,
        runner=runner,
        objective="sharpe_ratio",
        top_n=1,
    )

    assert len(result.fold_metrics) == 4
    assert set(result.fold_metrics["split"]) == {"train", "test"}
    assert len(result.selected_fold_metrics) == 1
    assert result.selected_fold_metrics.iloc[0]["parameter_id"] == parameter_sets[1].parameter_id
    assert result.best_parameter_sets.iloc[0]["parameter_id"] == parameter_sets[1].parameter_id


def test_metric_aggregation_and_ranking_use_test_split() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "parameter_id": "A",
                "split": "test",
                "total_return": 0.10,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
                "trade_count": 5,
            },
            {
                "parameter_id": "A",
                "split": "test",
                "total_return": 0.06,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.04,
                "trade_count": 4,
            },
            {
                "parameter_id": "B",
                "split": "test",
                "total_return": 0.04,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.10,
                "trade_count": 6,
            },
        ]
    )

    aggregate = aggregate_metric_frame(fold_metrics)
    best = select_top_parameter_sets(
        aggregate,
        objective="sharpe_ratio",
        split="test",
        top_n=1,
    )

    assert aggregate.loc[aggregate["parameter_id"] == "A", "mean_total_return"].iloc[0] == pytest.approx(0.08)
    assert best.iloc[0]["parameter_id"] == "B"


def test_evaluate_walkforward_handles_no_trade_metrics_cleanly() -> None:
    folds = generate_walkforward_folds(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 12),
        train_window_days=5,
        test_window_days=3,
    )
    parameter_sets = [SweepParameterSet(20, 1.5, 2.5, 3.0, 0.01)]

    result = evaluate_walkforward(
        folds=folds[:1],
        parameter_sets=parameter_sets,
        runner=lambda _params, _start, _end: empty_summary_metrics(),
        objective="total_return",
        top_n=1,
    )

    assert len(result.fold_metrics) == 2
    assert result.fold_metrics["trade_count"].tolist() == [0, 0]
    assert result.best_parameter_sets.iloc[0]["mean_total_return"] == pytest.approx(0.0)
