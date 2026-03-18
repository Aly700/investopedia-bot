"""Summary statistics for deterministic daily-bar backtests."""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

import pandas as pd


SUMMARY_METRIC_FIELDS = (
    "starting_equity",
    "ending_equity",
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe_ratio",
    "trade_count",
    "win_rate",
    "average_win",
    "average_loss",
    "expectancy",
)

OBJECTIVE_DIRECTIONS = {
    "starting_equity": "max",
    "ending_equity": "max",
    "total_return": "max",
    "cagr": "max",
    "max_drawdown": "min",
    "sharpe_ratio": "max",
    "trade_count": "max",
    "win_rate": "max",
    "average_win": "max",
    "average_loss": "max",
    "expectancy": "max",
}
OBJECTIVE_CHOICES = tuple(OBJECTIVE_DIRECTIONS.keys())

COMPARISON_PARAMETER_FIELDS = (
    "preset_name",
    "parameter_id",
    "breakout_lookback",
    "relative_volume_threshold",
    "require_relative_volume_confirmation",
    "initial_stop_atr",
    "trailing_stop_atr",
    "risk_per_trade",
)
COMPARISON_COLUMNS = (*COMPARISON_PARAMETER_FIELDS, *SUMMARY_METRIC_FIELDS)


def empty_summary_metrics() -> dict[str, float | int]:
    """Return the default metric payload for empty or no-trade runs."""

    return {
        "starting_equity": 0.0,
        "ending_equity": 0.0,
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "expectancy": 0.0,
    }


def calculate_summary_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> dict[str, float | int]:
    """Return a compact set of summary metrics for a backtest result."""

    if equity_curve.empty:
        return empty_summary_metrics()

    prepared_curve = equity_curve.copy()
    prepared_curve["date"] = pd.to_datetime(prepared_curve["date"], errors="coerce")
    prepared_curve["equity"] = pd.to_numeric(prepared_curve["equity"], errors="coerce")
    prepared_curve = prepared_curve.dropna(subset=["date", "equity"]).sort_values("date", kind="stable")

    if prepared_curve.empty:
        return calculate_summary_metrics(pd.DataFrame(), trade_log)

    starting_equity = float(prepared_curve["equity"].iloc[0])
    ending_equity = float(prepared_curve["equity"].iloc[-1])
    total_return = (
        (ending_equity / starting_equity) - 1.0 if starting_equity > 0 else 0.0
    )

    first_date = prepared_curve["date"].iloc[0]
    last_date = prepared_curve["date"].iloc[-1]
    elapsed_days = max(int((last_date - first_date).days), 0)
    cagr = 0.0
    if starting_equity > 0 and ending_equity > 0 and elapsed_days > 0:
        cagr = (ending_equity / starting_equity) ** (365.25 / elapsed_days) - 1.0

    drawdown_series = 1.0 - (
        prepared_curve["equity"] / prepared_curve["equity"].cummax().replace(0, pd.NA)
    )
    max_drawdown = float(drawdown_series.fillna(0.0).max())

    daily_returns = prepared_curve["equity"].pct_change().dropna()
    sharpe_ratio = 0.0
    if len(daily_returns) >= 2:
        volatility = float(daily_returns.std(ddof=1))
        if volatility > 0:
            sharpe_ratio = float(daily_returns.mean()) / volatility * sqrt(252.0)

    prepared_trades = trade_log.copy()
    if "net_pnl" in prepared_trades.columns:
        prepared_trades["net_pnl"] = pd.to_numeric(prepared_trades["net_pnl"], errors="coerce")
        prepared_trades = prepared_trades.dropna(subset=["net_pnl"])
    else:
        prepared_trades = prepared_trades.iloc[0:0]

    net_pnl = prepared_trades["net_pnl"] if not prepared_trades.empty else pd.Series(dtype=float)
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl < 0]

    trade_count = int(len(net_pnl))
    win_rate = float(len(wins) / trade_count) if trade_count else 0.0
    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0
    expectancy = float(net_pnl.mean()) if trade_count else 0.0

    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "expectancy": expectancy,
    }


def build_strategy_comparison_frame(
    rows: Sequence[dict[str, Any]] | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return a normalized frame for preset comparison runs."""

    if not rows:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    frame = pd.DataFrame(rows).copy()
    if "preset_name" not in frame.columns:
        raise KeyError("Comparison rows must include a 'preset_name' column.")

    if "parameter_id" not in frame.columns:
        frame["parameter_id"] = frame["preset_name"].astype(str)

    numeric_columns = [
        "breakout_lookback",
        "relative_volume_threshold",
        "initial_stop_atr",
        "trailing_stop_atr",
        "risk_per_trade",
        *SUMMARY_METRIC_FIELDS,
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.reindex(columns=COMPARISON_COLUMNS)


def aggregate_metric_frame(
    metric_frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("parameter_id", "split"),
) -> pd.DataFrame:
    """Aggregate fold-level metric rows into grouped summary statistics."""

    if metric_frame.empty:
        columns = list(group_columns) + ["fold_count"]
        for metric_name in SUMMARY_METRIC_FIELDS:
            columns.extend(
                [
                    f"mean_{metric_name}",
                    f"median_{metric_name}",
                    f"min_{metric_name}",
                    f"max_{metric_name}",
                ]
            )
        return pd.DataFrame(columns=columns)

    missing_columns = [column for column in group_columns if column not in metric_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Metric frame is missing required group columns: {missing}.")

    available_metrics = [
        metric_name for metric_name in SUMMARY_METRIC_FIELDS if metric_name in metric_frame.columns
    ]
    passthrough_columns = [
        column
        for column in metric_frame.columns
        if column not in group_columns and column not in available_metrics
    ]
    prepared = metric_frame.copy()
    for metric_name in available_metrics:
        prepared[metric_name] = pd.to_numeric(prepared[metric_name], errors="coerce")

    aggregate_rows: list[dict[str, Any]] = []
    grouped = prepared.groupby(list(group_columns), dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            column: value for column, value in zip(group_columns, group_key)
        }
        row["fold_count"] = int(len(group))
        for column in passthrough_columns:
            unique_values = group[column].dropna().unique().tolist()
            if len(unique_values) == 1:
                row[column] = unique_values[0]
        for metric_name in available_metrics:
            series = group[metric_name].dropna()
            row[f"mean_{metric_name}"] = float(series.mean()) if not series.empty else 0.0
            row[f"median_{metric_name}"] = float(series.median()) if not series.empty else 0.0
            row[f"min_{metric_name}"] = float(series.min()) if not series.empty else 0.0
            row[f"max_{metric_name}"] = float(series.max()) if not series.empty else 0.0
        aggregate_rows.append(row)

    return pd.DataFrame(aggregate_rows)


def select_top_metric_rows(
    metric_frame: pd.DataFrame,
    *,
    objective: str,
    top_n: int = 1,
) -> pd.DataFrame:
    """Return the highest-ranked metric rows for the requested objective."""

    _validate_objective(objective)
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if metric_frame.empty:
        return metric_frame.copy()
    if objective not in metric_frame.columns:
        raise KeyError(f"Metric frame does not contain objective column '{objective}'.")

    sort_columns: list[str] = [objective]
    ascending: list[bool] = [OBJECTIVE_DIRECTIONS[objective] == "min"]

    if objective != "max_drawdown" and "max_drawdown" in metric_frame.columns:
        sort_columns.append("max_drawdown")
        ascending.append(True)
    if objective != "total_return" and "total_return" in metric_frame.columns:
        sort_columns.append("total_return")
        ascending.append(False)
    if objective != "trade_count" and "trade_count" in metric_frame.columns:
        sort_columns.append("trade_count")
        ascending.append(False)
    if "parameter_id" in metric_frame.columns:
        sort_columns.append("parameter_id")
        ascending.append(True)

    return metric_frame.sort_values(sort_columns, ascending=ascending, kind="stable").head(top_n)


def select_top_parameter_sets(
    aggregate_metrics: pd.DataFrame,
    *,
    objective: str,
    split: str = "test",
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top aggregated parameter rows for the requested split and objective."""

    _validate_objective(objective)
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if aggregate_metrics.empty:
        return aggregate_metrics.copy()

    filtered = aggregate_metrics.copy()
    if "split" in filtered.columns:
        filtered = filtered.loc[filtered["split"] == split].copy()
    if filtered.empty:
        return filtered

    objective_column = f"mean_{objective}"
    if objective_column not in filtered.columns:
        raise KeyError(f"Aggregate metric frame does not contain '{objective_column}'.")

    sort_columns: list[str] = [objective_column]
    ascending: list[bool] = [OBJECTIVE_DIRECTIONS[objective] == "min"]

    if objective != "max_drawdown" and "mean_max_drawdown" in filtered.columns:
        sort_columns.append("mean_max_drawdown")
        ascending.append(True)
    if objective != "total_return" and "mean_total_return" in filtered.columns:
        sort_columns.append("mean_total_return")
        ascending.append(False)
    if objective != "trade_count" and "mean_trade_count" in filtered.columns:
        sort_columns.append("mean_trade_count")
        ascending.append(False)
    if "parameter_id" in filtered.columns:
        sort_columns.append("parameter_id")
        ascending.append(True)

    return filtered.sort_values(sort_columns, ascending=ascending, kind="stable").head(top_n)


def rank_strategy_comparisons(
    comparison_frame: pd.DataFrame,
    *,
    objective: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank preset comparison rows by the requested objective."""

    _validate_objective(objective)
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be greater than zero when provided.")
    if comparison_frame.empty:
        ranked = comparison_frame.copy()
        if "rank" not in ranked.columns:
            ranked.insert(0, "rank", pd.Series(dtype=int))
        return ranked

    limit = len(comparison_frame) if top_n is None else min(int(top_n), len(comparison_frame))
    ranked = select_top_metric_rows(
        comparison_frame,
        objective=objective,
        top_n=len(comparison_frame),
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.head(limit)


def metrics_to_serializable_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of a metric dictionary."""

    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, str, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable


def _validate_objective(objective: str) -> None:
    if objective not in OBJECTIVE_DIRECTIONS:
        expected = ", ".join(sorted(OBJECTIVE_DIRECTIONS))
        raise ValueError(f"Unsupported objective '{objective}'. Expected one of: {expected}.")
