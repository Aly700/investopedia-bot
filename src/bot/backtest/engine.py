"""Deterministic daily-bar backtest engine for the breakout strategy stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from math import floor
from typing import Any, Mapping, Sequence

import pandas as pd

from bot.backtest.metrics import calculate_summary_metrics
from bot.backtest.slippage_costs import TransactionCostModel
from bot.config import AppConfig
from bot.indicators.volatility import atr
from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioConstraints,
    RiskAssessedCandidate,
    assess_signal_candidate,
)
from bot.risk.stops import trailing_stop_reference
from bot.strategy.breakout_momentum import BreakoutMomentumSettings, generate_breakout_signal
from bot.strategy.regime_filter import RegimeFilterMode


TRADE_LOG_COLUMNS = (
    "symbol",
    "strategy_name",
    "entry_signal_date",
    "entry_date",
    "exit_date",
    "entry_reason",
    "exit_reason",
    "shares",
    "entry_price",
    "exit_price",
    "initial_stop",
    "final_stop",
    "gross_pnl",
    "net_pnl",
    "total_commission",
    "holding_period_days",
    "return_pct",
)

EQUITY_CURVE_COLUMNS = (
    "date",
    "cash",
    "market_value",
    "equity",
    "realized_pnl",
    "unrealized_pnl",
    "open_positions",
    "drawdown",
)


@dataclass(frozen=True)
class PendingEntry:
    """An approved signal that will be filled on a future bar open."""

    candidate: RiskAssessedCandidate
    fill_date: date


@dataclass
class BacktestPosition:
    """Mutable in-memory position state for the engine loop."""

    symbol: str
    strategy_name: str
    entry_reason: str
    entry_signal_date: date
    entry_date: date
    shares: int
    entry_price: float
    initial_stop: float
    stop_price: float
    highest_close: float
    entry_commission: float
    last_close: float


@dataclass(frozen=True)
class BacktestResult:
    """Pandas-friendly output bundle from a backtest run."""

    trade_log: pd.DataFrame
    equity_curve: pd.DataFrame
    summary_metrics: dict[str, float | int]


class DailyBarBacktestEngine:
    """Run a deterministic daily-bar breakout backtest with next-open fills."""

    def __init__(
        self,
        *,
        strategy_settings: BreakoutMomentumSettings,
        portfolio_constraints: PortfolioConstraints,
        starting_cash: float,
        base_risk_per_trade: float,
        cost_model: TransactionCostModel,
        close_positions_at_end: bool = True,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be greater than zero.")
        if base_risk_per_trade <= 0:
            raise ValueError("base_risk_per_trade must be greater than zero.")

        self.strategy_settings = strategy_settings
        self.portfolio_constraints = portfolio_constraints
        self.starting_cash = float(starting_cash)
        self.base_risk_per_trade = float(base_risk_per_trade)
        self.cost_model = cost_model
        self.close_positions_at_end = close_positions_at_end

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        benchmark_symbol_override: str | None = None,
        require_relative_volume_confirmation: bool = False,
        enable_regime_filter: bool = True,
        regime_filter_mode: RegimeFilterMode = "either",
        relative_volume_window: int | None = None,
        close_positions_at_end: bool = True,
        drawdown_risk_reduction_factor: float = 0.5,
    ) -> "DailyBarBacktestEngine":
        """Build the engine from the shared repo config."""

        fill_model = config.game_rules.execution.fill_model.strip().lower()
        if fill_model != "next_bar_open":
            raise ValueError(
                "Only the 'next_bar_open' fill model is supported by the daily-bar backtest engine."
            )

        strategy_settings = BreakoutMomentumSettings.from_configs(
            config.strategy.signals,
            config.strategy.risk,
            require_relative_volume_confirmation=require_relative_volume_confirmation,
            enable_regime_filter=enable_regime_filter,
            regime_filter_mode=regime_filter_mode,
            relative_volume_window=relative_volume_window,
        )
        if benchmark_symbol_override is not None and benchmark_symbol_override.strip():
            strategy_settings = replace(
                strategy_settings,
                benchmark_symbol=benchmark_symbol_override.strip().upper(),
            )

        portfolio_constraints = PortfolioConstraints.from_configs(
            config.strategy.risk,
            config.game_rules.rules,
            drawdown_risk_reduction_factor=drawdown_risk_reduction_factor,
        )
        return cls(
            strategy_settings=strategy_settings,
            portfolio_constraints=portfolio_constraints,
            starting_cash=config.game_rules.starting_cash,
            base_risk_per_trade=config.strategy.risk.risk_per_trade,
            cost_model=TransactionCostModel.from_game_rules(config.game_rules),
            close_positions_at_end=close_positions_at_end,
        )

    def warmup_start(self, start_date: date) -> date:
        """Return a conservative warmup start for indicators and regime filters."""

        largest_window = max(
            self.strategy_settings.breakout_lookback + 1,
            self.strategy_settings.atr_window,
            self.strategy_settings.resolved_relative_volume_window + 1,
            self.strategy_settings.benchmark_sma_slow if self.strategy_settings.enable_regime_filter else 1,
        )
        return start_date - timedelta(days=max(largest_window * 3, 30))

    def run(
        self,
        symbol_frames: Mapping[str, pd.DataFrame],
        *,
        benchmark_frame: pd.DataFrame | None,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        """Run the backtest and return trade log, equity curve, and summary metrics."""

        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")
        if self.strategy_settings.enable_regime_filter and benchmark_frame is None:
            raise ValueError("benchmark_frame is required when the regime filter is enabled.")

        prepared_symbol_frames = {
            symbol.strip().upper(): _prepare_symbol_frame(
                frame,
                symbol=symbol,
                atr_window=self.strategy_settings.atr_window,
            )
            for symbol, frame in symbol_frames.items()
            if not frame.empty
        }
        prepared_symbol_frames = {
            symbol: frame for symbol, frame in prepared_symbol_frames.items() if not frame.empty
        }
        prepared_benchmark = (
            _prepare_symbol_frame(
                benchmark_frame,
                symbol=self.strategy_settings.benchmark_symbol,
                atr_window=self.strategy_settings.atr_window,
            )
            if benchmark_frame is not None
            else None
        )

        trading_calendar = _build_trading_calendar(
            prepared_symbol_frames,
            benchmark_frame=prepared_benchmark,
            start_date=start_date,
            end_date=end_date,
        )
        if not trading_calendar:
            trade_log = _empty_trade_log()
            equity_curve = _empty_equity_curve()
            return BacktestResult(
                trade_log=trade_log,
                equity_curve=equity_curve,
                summary_metrics=calculate_summary_metrics(equity_curve, trade_log),
            )

        indexed_symbol_frames = {
            symbol: frame.set_index("date", drop=False)
            for symbol, frame in prepared_symbol_frames.items()
        }

        cash = self.starting_cash
        realized_pnl = 0.0
        peak_equity = self.starting_cash
        open_positions: dict[str, BacktestPosition] = {}
        pending_entries: list[PendingEntry] = []
        trade_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        for current_timestamp in trading_calendar:
            current_date = current_timestamp.date()

            gap_exit_symbols = self._symbols_with_gap_stop_exit(
                current_timestamp,
                indexed_symbol_frames,
                open_positions,
            )
            for symbol in gap_exit_symbols:
                position = open_positions[symbol]
                bar = indexed_symbol_frames[symbol].loc[current_timestamp]
                exit_price = self.cost_model.market_fill_price(float(bar["open"]), side="SELL")
                exit_commission = self.cost_model.commission(order_type="stop")
                cash, realized_pnl = _close_position(
                    position=position,
                    exit_date=current_date,
                    exit_price=exit_price,
                    exit_reason=_stop_exit_reason(position),
                    exit_commission=exit_commission,
                    cash=cash,
                    realized_pnl=realized_pnl,
                    trade_rows=trade_rows,
                )
                del open_positions[symbol]

            due_entries = [entry for entry in pending_entries if entry.fill_date == current_date]
            pending_entries = [entry for entry in pending_entries if entry.fill_date != current_date]
            for pending_entry in sorted(due_entries, key=lambda item: item.candidate.signal.symbol):
                symbol = pending_entry.candidate.signal.symbol
                if symbol in open_positions:
                    continue

                bar = indexed_symbol_frames.get(symbol)
                if bar is None or current_timestamp not in bar.index:
                    continue

                current_bar = bar.loc[current_timestamp]
                open_price = float(current_bar["open"])
                fill_price = self.cost_model.market_fill_price(open_price, side="BUY")
                if fill_price <= pending_entry.candidate.stop_price:
                    continue

                entry_commission = self.cost_model.commission(order_type="market")
                affordable_shares = _affordable_share_count(
                    cash=cash,
                    fill_price=fill_price,
                    commission=entry_commission,
                )
                shares = min(pending_entry.candidate.sizing.shares, affordable_shares)
                if shares <= 0:
                    continue

                cash -= shares * fill_price + entry_commission
                open_positions[symbol] = BacktestPosition(
                    symbol=symbol,
                    strategy_name=pending_entry.candidate.signal.strategy_name,
                    entry_reason=pending_entry.candidate.signal.entry_reason,
                    entry_signal_date=pending_entry.candidate.signal.date,
                    entry_date=current_date,
                    shares=shares,
                    entry_price=fill_price,
                    initial_stop=pending_entry.candidate.stop_price,
                    stop_price=pending_entry.candidate.stop_price,
                    highest_close=pending_entry.candidate.entry_price,
                    entry_commission=entry_commission,
                    last_close=float(current_bar["close"]),
                )

            intraday_exit_symbols = self._symbols_with_intraday_stop_exit(
                current_timestamp,
                indexed_symbol_frames,
                open_positions,
            )
            for symbol in intraday_exit_symbols:
                position = open_positions[symbol]
                bar = indexed_symbol_frames[symbol].loc[current_timestamp]
                exit_price = self.cost_model.long_stop_fill_price(
                    stop_price=position.stop_price,
                    bar_open=float(bar["open"]),
                    bar_low=float(bar["low"]),
                )
                if exit_price is None:
                    continue

                exit_commission = self.cost_model.commission(order_type="stop")
                cash, realized_pnl = _close_position(
                    position=position,
                    exit_date=current_date,
                    exit_price=exit_price,
                    exit_reason=_stop_exit_reason(position),
                    exit_commission=exit_commission,
                    cash=cash,
                    realized_pnl=realized_pnl,
                    trade_rows=trade_rows,
                )
                del open_positions[symbol]

            market_value = self._mark_positions(
                current_timestamp,
                indexed_symbol_frames,
                open_positions,
            )
            equity = cash + market_value
            peak_equity = max(peak_equity, equity)
            current_drawdown = 0.0 if peak_equity <= 0 else max(0.0, 1.0 - (equity / peak_equity))
            unrealized_pnl = equity - self.starting_cash - realized_pnl
            equity_rows.append(
                {
                    "date": pd.Timestamp(current_date),
                    "cash": cash,
                    "market_value": market_value,
                    "equity": equity,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "open_positions": len(open_positions),
                    "drawdown": current_drawdown,
                }
            )

            for symbol in sorted(prepared_symbol_frames):
                if symbol in open_positions or symbol in {entry.candidate.signal.symbol for entry in pending_entries}:
                    continue
                if current_timestamp not in indexed_symbol_frames[symbol].index:
                    continue

                next_fill_date = _next_available_trade_date(
                    prepared_symbol_frames[symbol],
                    current_timestamp,
                    end_date=end_date,
                )
                if next_fill_date is None:
                    continue

                price_slice = prepared_symbol_frames[symbol].loc[
                    prepared_symbol_frames[symbol]["date"] <= current_timestamp
                ]
                benchmark_slice = (
                    prepared_benchmark.loc[prepared_benchmark["date"] <= current_timestamp]
                    if prepared_benchmark is not None
                    else None
                )
                signal = generate_breakout_signal(
                    price_slice,
                    settings=self.strategy_settings,
                    benchmark_frame=benchmark_slice,
                    has_open_position=False,
                    symbol=symbol,
                )
                if signal is None:
                    continue

                candidate = assess_signal_candidate(
                    signal,
                    current_equity=equity,
                    base_risk_per_trade=self.base_risk_per_trade,
                    constraints=self.portfolio_constraints,
                    current_positions=_positions_snapshot(open_positions, pending_entries),
                    current_drawdown=current_drawdown,
                )
                if candidate.approved:
                    pending_entries.append(
                        PendingEntry(candidate=candidate, fill_date=next_fill_date)
                    )

            self._update_trailing_stops(
                current_timestamp,
                indexed_symbol_frames,
                open_positions,
            )

        if self.close_positions_at_end and open_positions:
            final_timestamp = trading_calendar[-1]
            final_date = final_timestamp.date()
            for symbol in sorted(tuple(open_positions)):
                position = open_positions[symbol]
                if final_timestamp in indexed_symbol_frames[symbol].index:
                    bar = indexed_symbol_frames[symbol].loc[final_timestamp]
                    reference_price = float(bar["close"])
                else:
                    reference_price = position.last_close
                exit_price = self.cost_model.market_fill_price(reference_price, side="SELL")
                exit_commission = self.cost_model.commission(order_type="market")
                cash, realized_pnl = _close_position(
                    position=position,
                    exit_date=final_date,
                    exit_price=exit_price,
                    exit_reason="end_of_data",
                    exit_commission=exit_commission,
                    cash=cash,
                    realized_pnl=realized_pnl,
                    trade_rows=trade_rows,
                )
                del open_positions[symbol]

            peak_equity = max(peak_equity, cash)
            equity_rows[-1] = {
                "date": pd.Timestamp(final_date),
                "cash": cash,
                "market_value": 0.0,
                "equity": cash,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": 0.0,
                "open_positions": 0,
                "drawdown": 0.0 if peak_equity <= 0 else max(0.0, 1.0 - (cash / peak_equity)),
            }

        trade_log = pd.DataFrame(trade_rows, columns=TRADE_LOG_COLUMNS)
        equity_curve = pd.DataFrame(equity_rows, columns=EQUITY_CURVE_COLUMNS)
        summary_metrics = calculate_summary_metrics(equity_curve, trade_log)
        return BacktestResult(
            trade_log=trade_log,
            equity_curve=equity_curve,
            summary_metrics=summary_metrics,
        )

    def _symbols_with_gap_stop_exit(
        self,
        current_timestamp: pd.Timestamp,
        indexed_symbol_frames: Mapping[str, pd.DataFrame],
        open_positions: Mapping[str, BacktestPosition],
    ) -> list[str]:
        symbols: list[str] = []
        for symbol in sorted(open_positions):
            if current_timestamp not in indexed_symbol_frames[symbol].index:
                continue
            bar = indexed_symbol_frames[symbol].loc[current_timestamp]
            if float(bar["open"]) <= open_positions[symbol].stop_price:
                symbols.append(symbol)
        return symbols

    def _symbols_with_intraday_stop_exit(
        self,
        current_timestamp: pd.Timestamp,
        indexed_symbol_frames: Mapping[str, pd.DataFrame],
        open_positions: Mapping[str, BacktestPosition],
    ) -> list[str]:
        symbols: list[str] = []
        for symbol in sorted(open_positions):
            if current_timestamp not in indexed_symbol_frames[symbol].index:
                continue
            bar = indexed_symbol_frames[symbol].loc[current_timestamp]
            if float(bar["open"]) <= open_positions[symbol].stop_price:
                continue
            if float(bar["low"]) <= open_positions[symbol].stop_price:
                symbols.append(symbol)
        return symbols

    def _mark_positions(
        self,
        current_timestamp: pd.Timestamp,
        indexed_symbol_frames: Mapping[str, pd.DataFrame],
        open_positions: Mapping[str, BacktestPosition],
    ) -> float:
        market_value = 0.0
        for symbol, position in open_positions.items():
            if current_timestamp in indexed_symbol_frames[symbol].index:
                bar = indexed_symbol_frames[symbol].loc[current_timestamp]
                position.last_close = float(bar["close"])
            market_value += position.shares * position.last_close
        return market_value

    def _update_trailing_stops(
        self,
        current_timestamp: pd.Timestamp,
        indexed_symbol_frames: Mapping[str, pd.DataFrame],
        open_positions: Mapping[str, BacktestPosition],
    ) -> None:
        for symbol, position in open_positions.items():
            if current_timestamp not in indexed_symbol_frames[symbol].index:
                continue

            bar = indexed_symbol_frames[symbol].loc[current_timestamp]
            current_close = float(bar["close"])
            atr_value = bar.get("_atr")
            if pd.isna(atr_value):
                position.highest_close = max(position.highest_close, current_close)
                position.last_close = current_close
                continue

            position.highest_close = max(position.highest_close, current_close)
            trailing_stop = trailing_stop_reference(
                position.highest_close,
                float(atr_value),
                self.strategy_settings.trailing_stop_atr,
            )
            position.stop_price = max(position.stop_price, trailing_stop)
            position.last_close = current_close


def _prepare_symbol_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    atr_window: int,
) -> pd.DataFrame:
    required_columns = ("date", "open", "high", "low", "close", "volume")
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Frame for {symbol} is missing required columns: {missing}.")

    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["date"]).sort_values("date", kind="stable")
    prepared = prepared.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["open", "high", "low", "close"])
    prepared["symbol"] = symbol.strip().upper()
    prepared["_atr"] = atr(prepared, window=atr_window)
    return prepared


def _build_trading_calendar(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    benchmark_frame: pd.DataFrame | None,
    start_date: date,
    end_date: date,
) -> list[pd.Timestamp]:
    calendar_dates: set[pd.Timestamp] = set()
    for frame in symbol_frames.values():
        in_range = frame.loc[
            (frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date)),
            "date",
        ]
        calendar_dates.update(pd.Timestamp(value) for value in in_range.tolist())

    if benchmark_frame is not None:
        in_range = benchmark_frame.loc[
            (benchmark_frame["date"] >= pd.Timestamp(start_date))
            & (benchmark_frame["date"] <= pd.Timestamp(end_date)),
            "date",
        ]
        calendar_dates.update(pd.Timestamp(value) for value in in_range.tolist())

    return sorted(calendar_dates)


def _next_available_trade_date(
    frame: pd.DataFrame,
    current_timestamp: pd.Timestamp,
    *,
    end_date: date,
) -> date | None:
    future_rows = frame.loc[frame["date"] > current_timestamp, "date"]
    if future_rows.empty:
        return None

    next_date = pd.Timestamp(future_rows.iloc[0]).date()
    if next_date > end_date:
        return None
    return next_date


def _positions_snapshot(
    open_positions: Mapping[str, BacktestPosition],
    pending_entries: Sequence[PendingEntry],
) -> list[ExistingPosition]:
    positions = [
        ExistingPosition(
            symbol=position.symbol,
            shares=position.shares,
            average_entry_price=position.entry_price,
        )
        for position in open_positions.values()
    ]
    positions.extend(
        ExistingPosition(
            symbol=entry.candidate.signal.symbol,
            shares=max(entry.candidate.sizing.shares, 1),
            average_entry_price=entry.candidate.entry_price,
        )
        for entry in pending_entries
    )
    return positions


def _affordable_share_count(*, cash: float, fill_price: float, commission: float) -> int:
    if fill_price <= 0:
        return 0
    available_cash = cash - commission
    if available_cash <= 0:
        return 0
    return max(int(floor(available_cash / fill_price)), 0)


def _close_position(
    *,
    position: BacktestPosition,
    exit_date: date,
    exit_price: float,
    exit_reason: str,
    exit_commission: float,
    cash: float,
    realized_pnl: float,
    trade_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    gross_pnl = position.shares * (exit_price - position.entry_price)
    total_commission = position.entry_commission + exit_commission
    net_pnl = gross_pnl - total_commission
    cash += position.shares * exit_price - exit_commission
    realized_pnl += net_pnl

    entry_notional = position.entry_price * position.shares
    return_pct = net_pnl / entry_notional if entry_notional > 0 else 0.0
    trade_rows.append(
        {
            "symbol": position.symbol,
            "strategy_name": position.strategy_name,
            "entry_signal_date": pd.Timestamp(position.entry_signal_date),
            "entry_date": pd.Timestamp(position.entry_date),
            "exit_date": pd.Timestamp(exit_date),
            "entry_reason": position.entry_reason,
            "exit_reason": exit_reason,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "initial_stop": position.initial_stop,
            "final_stop": position.stop_price,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_commission": total_commission,
            "holding_period_days": max((exit_date - position.entry_date).days, 0),
            "return_pct": return_pct,
        }
    )
    return cash, realized_pnl


def _stop_exit_reason(position: BacktestPosition) -> str:
    if position.stop_price > position.initial_stop:
        return "trailing_stop"
    return "initial_stop"


def _empty_trade_log() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def _empty_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)
