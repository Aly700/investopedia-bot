"""Summary statistics for deterministic daily-bar backtests."""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd


def calculate_summary_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> dict[str, float | int]:
    """Return a compact set of summary metrics for a backtest result."""

    if equity_curve.empty:
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


def metrics_to_serializable_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of a metric dictionary."""

    serializable: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, str, bool)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable
