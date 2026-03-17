"""Walk-forward validation and parameter sweep helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from itertools import product
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import pandas as pd

from bot.backtest.engine import DailyBarBacktestEngine
from bot.backtest.metrics import (
    OBJECTIVE_DIRECTIONS,
    SUMMARY_METRIC_FIELDS,
    aggregate_metric_frame,
    empty_summary_metrics,
    metrics_to_serializable_dict,
    select_top_metric_rows,
    select_top_parameter_sets,
)
from bot.backtest.slippage_costs import TransactionCostModel
from bot.config import AppConfig
from bot.risk.portfolio_rules import PortfolioConstraints
from bot.strategy.breakout_momentum import BreakoutMomentumSettings
from bot.strategy.regime_filter import RegimeFilterMode


MetricValue = Union[float, int]
MetricRunner = Callable[["SweepParameterSet", date, date], Mapping[str, MetricValue]]

FOLD_METRIC_COLUMNS = (
    "fold_index",
    "split",
    "period_start",
    "period_end",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "parameter_id",
    "breakout_lookback",
    "relative_volume_threshold",
    "initial_stop_atr",
    "trailing_stop_atr",
    "risk_per_trade",
    *SUMMARY_METRIC_FIELDS,
)


@dataclass(frozen=True)
class WalkForwardFold:
    """A sequential train/test fold for walk-forward validation."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the fold."""

        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


@dataclass(frozen=True)
class SweepParameterSet:
    """A parameter combination evaluated during a walk-forward sweep."""

    breakout_lookback: int
    relative_volume_threshold: float
    initial_stop_atr: float
    trailing_stop_atr: float
    risk_per_trade: float

    def __post_init__(self) -> None:
        if self.breakout_lookback <= 0:
            raise ValueError("breakout_lookback must be greater than zero.")
        if self.relative_volume_threshold <= 0:
            raise ValueError("relative_volume_threshold must be greater than zero.")
        if self.initial_stop_atr <= 0:
            raise ValueError("initial_stop_atr must be greater than zero.")
        if self.trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be greater than zero.")
        if self.risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be greater than zero.")

    @property
    def parameter_id(self) -> str:
        """Return a stable string identifier for the parameter set."""

        return (
            f"lookback={self.breakout_lookback}|"
            f"rv={self.relative_volume_threshold:g}|"
            f"initial_stop={self.initial_stop_atr:g}|"
            f"trailing_stop={self.trailing_stop_atr:g}|"
            f"risk={self.risk_per_trade:g}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the parameter set."""

        return {
            "parameter_id": self.parameter_id,
            "breakout_lookback": self.breakout_lookback,
            "relative_volume_threshold": self.relative_volume_threshold,
            "initial_stop_atr": self.initial_stop_atr,
            "trailing_stop_atr": self.trailing_stop_atr,
            "risk_per_trade": self.risk_per_trade,
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """Pandas-friendly output bundle for walk-forward analysis."""

    folds: tuple[WalkForwardFold, ...]
    parameter_sets: tuple[SweepParameterSet, ...]
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    selected_fold_metrics: pd.DataFrame
    best_parameter_sets: pd.DataFrame


def generate_walkforward_folds(
    *,
    start_date: date,
    end_date: date,
    train_window_days: int,
    test_window_days: int,
    expanding_train: bool = False,
) -> list[WalkForwardFold]:
    """Build sequential walk-forward folds using calendar-day windows."""

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")
    if train_window_days <= 0:
        raise ValueError("train_window_days must be greater than zero.")
    if test_window_days <= 0:
        raise ValueError("test_window_days must be greater than zero.")

    folds: list[WalkForwardFold] = []
    train_start = start_date
    train_end = train_start + timedelta(days=train_window_days - 1)
    fold_index = 1

    while train_end < end_date:
        test_start = train_end + timedelta(days=1)
        if test_start > end_date:
            break

        test_end = min(test_start + timedelta(days=test_window_days - 1), end_date)
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_index += 1

        if expanding_train:
            train_end = test_end
        else:
            train_start = train_start + timedelta(days=test_window_days)
            train_end = train_start + timedelta(days=train_window_days - 1)

    return folds


def build_parameter_grid(
    *,
    breakout_lookbacks: Sequence[int],
    relative_volume_thresholds: Sequence[float],
    initial_stop_atrs: Sequence[float],
    trailing_stop_atrs: Sequence[float],
    risk_per_trade_values: Sequence[float],
) -> list[SweepParameterSet]:
    """Return the cartesian product of the supplied parameter sequences."""

    sequences = {
        "breakout_lookbacks": breakout_lookbacks,
        "relative_volume_thresholds": relative_volume_thresholds,
        "initial_stop_atrs": initial_stop_atrs,
        "trailing_stop_atrs": trailing_stop_atrs,
        "risk_per_trade_values": risk_per_trade_values,
    }
    for name, values in sequences.items():
        if not values:
            raise ValueError(f"{name} must contain at least one value.")

    return [
        SweepParameterSet(
            breakout_lookback=int(breakout_lookback),
            relative_volume_threshold=float(relative_volume_threshold),
            initial_stop_atr=float(initial_stop_atr),
            trailing_stop_atr=float(trailing_stop_atr),
            risk_per_trade=float(risk_per_trade),
        )
        for breakout_lookback, relative_volume_threshold, initial_stop_atr, trailing_stop_atr, risk_per_trade in product(
            breakout_lookbacks,
            relative_volume_thresholds,
            initial_stop_atrs,
            trailing_stop_atrs,
            risk_per_trade_values,
        )
    ]


def parameter_grid_from_config(
    config: AppConfig,
    *,
    breakout_lookbacks: Sequence[int] | None = None,
    relative_volume_thresholds: Sequence[float] | None = None,
    initial_stop_atrs: Sequence[float] | None = None,
    trailing_stop_atrs: Sequence[float] | None = None,
    risk_per_trade_values: Sequence[float] | None = None,
) -> list[SweepParameterSet]:
    """Build a parameter grid using explicit sequences or config defaults."""

    return build_parameter_grid(
        breakout_lookbacks=breakout_lookbacks or (config.strategy.signals.breakout_lookback,),
        relative_volume_thresholds=relative_volume_thresholds
        or (config.strategy.signals.relative_volume_threshold,),
        initial_stop_atrs=initial_stop_atrs or (config.strategy.risk.initial_stop_atr,),
        trailing_stop_atrs=trailing_stop_atrs or (config.strategy.risk.trailing_stop_atr,),
        risk_per_trade_values=risk_per_trade_values or (config.strategy.risk.risk_per_trade,),
    )


def evaluate_walkforward(
    *,
    folds: Sequence[WalkForwardFold],
    parameter_sets: Sequence[SweepParameterSet],
    runner: MetricRunner,
    objective: str = "sharpe_ratio",
    top_n: int = 5,
) -> WalkForwardResult:
    """Evaluate all parameter sets across sequential train/test folds."""

    if objective not in OBJECTIVE_DIRECTIONS:
        expected = ", ".join(sorted(OBJECTIVE_DIRECTIONS))
        raise ValueError(f"Unsupported objective '{objective}'. Expected one of: {expected}.")
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    fold_metric_rows: list[dict[str, Any]] = []
    selected_fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_rows_for_fold: list[dict[str, Any]] = []
        test_rows_by_parameter_id: dict[str, dict[str, Any]] = {}

        for parameter_set in parameter_sets:
            train_metrics = _normalize_metrics(runner(parameter_set, fold.train_start, fold.train_end))
            train_row = _build_fold_metric_row(
                fold=fold,
                parameter_set=parameter_set,
                split="train",
                period_start=fold.train_start,
                period_end=fold.train_end,
                metrics=train_metrics,
            )
            fold_metric_rows.append(train_row)
            train_rows_for_fold.append(train_row)

            test_metrics = _normalize_metrics(runner(parameter_set, fold.test_start, fold.test_end))
            test_row = _build_fold_metric_row(
                fold=fold,
                parameter_set=parameter_set,
                split="test",
                period_start=fold.test_start,
                period_end=fold.test_end,
                metrics=test_metrics,
            )
            fold_metric_rows.append(test_row)
            test_rows_by_parameter_id[parameter_set.parameter_id] = test_row

        if train_rows_for_fold:
            best_train_row = select_top_metric_rows(
                pd.DataFrame(train_rows_for_fold),
                objective=objective,
                top_n=1,
            )
            if not best_train_row.empty:
                best_train = best_train_row.iloc[0].to_dict()
                selected_fold_rows.append(
                    _build_selected_fold_row(
                        best_train_row=best_train,
                        matching_test_row=test_rows_by_parameter_id[best_train["parameter_id"]],
                        objective=objective,
                    )
                )

    fold_metrics = pd.DataFrame(fold_metric_rows, columns=FOLD_METRIC_COLUMNS)
    aggregate_metrics = aggregate_metric_frame(fold_metrics)
    selected_fold_metrics = pd.DataFrame(selected_fold_rows)
    best_parameter_sets = select_top_parameter_sets(
        aggregate_metrics,
        objective=objective,
        split="test",
        top_n=top_n,
    )

    return WalkForwardResult(
        folds=tuple(folds),
        parameter_sets=tuple(parameter_sets),
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
        selected_fold_metrics=selected_fold_metrics,
        best_parameter_sets=best_parameter_sets,
    )


def run_breakout_walkforward(
    *,
    config: AppConfig,
    symbol_frames: Mapping[str, pd.DataFrame],
    benchmark_frame: pd.DataFrame | None,
    folds: Sequence[WalkForwardFold],
    parameter_sets: Sequence[SweepParameterSet],
    objective: str = "sharpe_ratio",
    top_n: int = 5,
    benchmark_symbol_override: str | None = None,
    require_relative_volume_confirmation: bool = False,
    enable_regime_filter: bool = True,
    regime_filter_mode: RegimeFilterMode = "either",
    relative_volume_window: int | None = None,
    close_positions_at_end: bool = True,
    drawdown_risk_reduction_factor: float = 0.5,
) -> WalkForwardResult:
    """Run walk-forward evaluation with the repo's breakout strategy and backtest engine."""

    cost_model = TransactionCostModel.from_game_rules(config.game_rules)
    portfolio_constraints = PortfolioConstraints.from_configs(
        config.strategy.risk,
        config.game_rules.rules,
        drawdown_risk_reduction_factor=drawdown_risk_reduction_factor,
    )

    def runner(parameter_set: SweepParameterSet, window_start: date, window_end: date) -> Mapping[str, float | int]:
        strategy_settings = BreakoutMomentumSettings.from_configs(
            config.strategy.signals,
            config.strategy.risk,
            require_relative_volume_confirmation=require_relative_volume_confirmation,
            enable_regime_filter=enable_regime_filter,
            regime_filter_mode=regime_filter_mode,
            relative_volume_window=relative_volume_window,
        )
        strategy_settings = replace(
            strategy_settings,
            breakout_lookback=parameter_set.breakout_lookback,
            relative_volume_threshold=parameter_set.relative_volume_threshold,
            stop_atr_multiple=parameter_set.initial_stop_atr,
        )
        if benchmark_symbol_override is not None and benchmark_symbol_override.strip():
            strategy_settings = replace(
                strategy_settings,
                benchmark_symbol=benchmark_symbol_override.strip().upper(),
            )

        engine = DailyBarBacktestEngine(
            strategy_settings=strategy_settings,
            portfolio_constraints=portfolio_constraints,
            starting_cash=config.game_rules.starting_cash,
            base_risk_per_trade=parameter_set.risk_per_trade,
            cost_model=cost_model,
            trailing_stop_atr_multiple=parameter_set.trailing_stop_atr,
            close_positions_at_end=close_positions_at_end,
        )
        result = engine.run(
            symbol_frames=symbol_frames,
            benchmark_frame=benchmark_frame,
            start_date=window_start,
            end_date=window_end,
        )
        return result.summary_metrics

    return evaluate_walkforward(
        folds=folds,
        parameter_sets=parameter_sets,
        runner=runner,
        objective=objective,
        top_n=top_n,
    )


def walkforward_fetch_start(
    *,
    initial_start_date: date,
    parameter_sets: Sequence[SweepParameterSet],
    atr_window: int,
    benchmark_sma_slow: int,
    enable_regime_filter: bool,
) -> date:
    """Return the earliest warmup start required for a walk-forward sweep."""

    if atr_window <= 0:
        raise ValueError("atr_window must be greater than zero.")
    max_breakout_lookback = max(
        (parameter_set.breakout_lookback for parameter_set in parameter_sets),
        default=1,
    )
    largest_window = max(
        max_breakout_lookback + 1,
        atr_window,
        max_breakout_lookback + 1,
        benchmark_sma_slow if enable_regime_filter else 1,
    )
    return initial_start_date - timedelta(days=max(largest_window * 3, 30))


def write_walkforward_reports(
    result: WalkForwardResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write walk-forward analysis outputs to CSV and JSON files."""

    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = {
        "fold_metrics_csv": str(_write_frame(result.fold_metrics, resolved_output_dir / "fold_metrics.csv")),
        "fold_metrics_json": str(_write_frame(result.fold_metrics, resolved_output_dir / "fold_metrics.json")),
        "aggregate_metrics_csv": str(
            _write_frame(result.aggregate_metrics, resolved_output_dir / "aggregate_metrics.csv")
        ),
        "aggregate_metrics_json": str(
            _write_frame(result.aggregate_metrics, resolved_output_dir / "aggregate_metrics.json")
        ),
        "selected_fold_metrics_csv": str(
            _write_frame(result.selected_fold_metrics, resolved_output_dir / "selected_fold_metrics.csv")
        ),
        "selected_fold_metrics_json": str(
            _write_frame(result.selected_fold_metrics, resolved_output_dir / "selected_fold_metrics.json")
        ),
        "best_parameter_sets_csv": str(
            _write_frame(result.best_parameter_sets, resolved_output_dir / "best_parameter_sets.csv")
        ),
        "best_parameter_sets_json": str(
            _write_frame(result.best_parameter_sets, resolved_output_dir / "best_parameter_sets.json")
        ),
    }

    summary_payload = {
        "fold_count": len(result.folds),
        "parameter_set_count": len(result.parameter_sets),
        "best_parameter_sets": _frame_records(result.best_parameter_sets),
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written_paths["summary_json"] = str(summary_path)
    return written_paths


def _build_fold_metric_row(
    *,
    fold: WalkForwardFold,
    parameter_set: SweepParameterSet,
    split: str,
    period_start: date,
    period_end: date,
    metrics: Mapping[str, float | int],
) -> dict[str, Any]:
    row = {
        "fold_index": fold.fold_index,
        "split": split,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        **parameter_set.to_dict(),
    }
    row.update(metrics_to_serializable_dict(dict(metrics)))
    return row


def _build_selected_fold_row(
    *,
    best_train_row: Mapping[str, Any],
    matching_test_row: Mapping[str, Any],
    objective: str,
) -> dict[str, Any]:
    selected_row: dict[str, Any] = {
        "fold_index": best_train_row["fold_index"],
        "parameter_id": best_train_row["parameter_id"],
        "breakout_lookback": best_train_row["breakout_lookback"],
        "relative_volume_threshold": best_train_row["relative_volume_threshold"],
        "initial_stop_atr": best_train_row["initial_stop_atr"],
        "trailing_stop_atr": best_train_row["trailing_stop_atr"],
        "risk_per_trade": best_train_row["risk_per_trade"],
        "objective": objective,
        "train_period_start": best_train_row["period_start"],
        "train_period_end": best_train_row["period_end"],
        "test_period_start": matching_test_row["period_start"],
        "test_period_end": matching_test_row["period_end"],
        "train_objective_value": best_train_row[objective],
        "test_objective_value": matching_test_row[objective],
    }
    for metric_name in SUMMARY_METRIC_FIELDS:
        selected_row[f"train_{metric_name}"] = best_train_row.get(metric_name, 0.0)
        selected_row[f"test_{metric_name}"] = matching_test_row.get(metric_name, 0.0)
    return selected_row


def _normalize_metrics(metrics: Mapping[str, float | int]) -> dict[str, float | int]:
    normalized = empty_summary_metrics()
    for metric_name in SUMMARY_METRIC_FIELDS:
        if metric_name in metrics:
            normalized[metric_name] = metrics[metric_name]
    return normalized


def _write_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_output_path.suffix.lower() == ".json":
        resolved_output_path.write_text(
            json.dumps(_frame_records(frame), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        frame.to_csv(resolved_output_path, index=False)
    return resolved_output_path


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        serialized: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                serialized[key] = value.date().isoformat()
            else:
                serialized[key] = value
        records.append(serialized)
    return records
