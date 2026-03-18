from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from bot.backtest.engine import DailyBarBacktestEngine
from bot.backtest.metrics import calculate_summary_metrics
from bot.backtest.slippage_costs import TransactionCostModel, apply_slippage
from bot.risk.portfolio_rules import PortfolioConstraints
from bot.strategy.breakout_momentum import BreakoutMomentumSettings


def test_backtest_fills_on_next_bar_open_and_closes_at_end_of_data() -> None:
    engine = _engine(slippage_bps=10, market_commission=1.0, stop_commission=1.0)
    symbol_frame = _symbol_frame(
        opens=[9.0, 10.0, 11.0, 12.5, 12.8],
        highs=[10.0, 11.0, 12.0, 13.0, 13.1],
        lows=[8.0, 9.0, 10.0, 11.5, 12.0],
        closes=[9.0, 10.0, 12.0, 12.8, 12.9],
        symbol="AAA",
    )

    result = engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
    )

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["entry_date"].date() == date(2024, 1, 4)
    assert trade["exit_date"].date() == date(2024, 1, 5)
    assert trade["shares"] == 100
    assert trade["entry_price"] == pytest.approx(apply_slippage(12.5, side="BUY", slippage_bps=10))
    assert trade["exit_price"] == pytest.approx(apply_slippage(12.9, side="SELL", slippage_bps=10))
    assert trade["total_commission"] == pytest.approx(2.0)

    final_equity = float(result.equity_curve["equity"].iloc[-1])
    expected_net_pnl = 100 * (
        apply_slippage(12.9, side="SELL", slippage_bps=10)
        - apply_slippage(12.5, side="BUY", slippage_bps=10)
    ) - 2.0
    assert final_equity == pytest.approx(10_000.0 + expected_net_pnl)


def test_backtest_exits_on_intraday_stop() -> None:
    engine = _engine(slippage_bps=0, market_commission=0.0, stop_commission=0.0)
    symbol_frame = _symbol_frame(
        opens=[9.0, 10.0, 11.0, 12.0],
        highs=[10.0, 11.0, 12.0, 12.2],
        lows=[8.0, 9.0, 10.0, 9.0],
        closes=[9.0, 10.0, 12.0, 9.5],
        symbol="AAA",
    )

    result = engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
    )

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["entry_date"].date() == date(2024, 1, 4)
    assert trade["exit_date"].date() == date(2024, 1, 4)
    assert trade["exit_reason"] == "initial_stop"
    assert trade["exit_price"] == pytest.approx(10.0)
    assert trade["net_pnl"] == pytest.approx(-200.0)
    assert float(result.equity_curve["cash"].iloc[-1]) == pytest.approx(9_800.0)


def test_backtest_updates_trailing_stop_before_later_exit() -> None:
    engine = _engine(
        slippage_bps=0,
        market_commission=0.0,
        stop_commission=0.0,
        trailing_stop_atr_multiple=0.5,
    )
    symbol_frame = _symbol_frame(
        opens=[9.0, 10.0, 11.0, 12.5, 13.0, 11.8],
        highs=[10.0, 11.0, 12.0, 13.5, 14.0, 12.0],
        lows=[8.0, 9.0, 10.0, 11.5, 12.0, 11.7],
        closes=[9.0, 10.0, 12.0, 13.0, 13.0, 11.8],
        symbol="AAA",
    )

    result = engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 6),
    )

    assert len(result.trade_log) == 1
    trade = result.trade_log.iloc[0]
    assert trade["exit_date"].date() == date(2024, 1, 5)
    assert trade["exit_reason"] == "trailing_stop"
    assert trade["final_stop"] == pytest.approx(12.0)
    assert trade["exit_price"] == pytest.approx(12.0)


def test_trailing_stop_atr_from_settings_affects_final_stop() -> None:
    # Two engines identical except trailing_stop_atr: tight=0.5, wide=5.0.
    # Price rises then pulls back; tight stop should ratchet higher than wide stop.
    tight_engine = _engine(slippage_bps=0, market_commission=0.0, stop_commission=0.0, trailing_stop_atr_multiple=0.5)
    wide_engine = _engine(slippage_bps=0, market_commission=0.0, stop_commission=0.0, trailing_stop_atr_multiple=5.0)

    symbol_frame = _symbol_frame(
        opens=[9.0, 10.0, 11.0, 12.5, 13.0, 11.8],
        highs=[10.0, 11.0, 12.0, 13.5, 14.0, 12.0],
        lows=[8.0, 9.0, 10.0, 11.5, 12.0, 11.7],
        closes=[9.0, 10.0, 12.0, 13.0, 13.0, 11.8],
        symbol="AAA",
    )

    tight_result = tight_engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 6),
    )
    wide_result = wide_engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 6),
    )

    assert len(tight_result.trade_log) == 1
    assert len(wide_result.trade_log) == 1
    tight_final_stop = float(tight_result.trade_log.iloc[0]["final_stop"])
    wide_final_stop = float(wide_result.trade_log.iloc[0]["final_stop"])
    # Tight trailing stop stays closer to price, so its floor is higher.
    assert tight_final_stop > wide_final_stop


def test_backtest_handles_no_trade_runs_cleanly() -> None:
    engine = _engine(slippage_bps=0, market_commission=0.0, stop_commission=0.0)
    symbol_frame = _symbol_frame(
        opens=[9.0, 9.5, 9.6, 9.7],
        highs=[10.0, 10.0, 10.0, 10.0],
        lows=[8.5, 9.0, 9.1, 9.2],
        closes=[9.0, 9.5, 9.6, 9.7],
        symbol="AAA",
    )

    result = engine.run(
        {"AAA": symbol_frame},
        benchmark_frame=None,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
    )

    assert result.trade_log.empty
    assert result.summary_metrics["trade_count"] == 0
    assert result.summary_metrics["total_return"] == pytest.approx(0.0)
    assert result.equity_curve["equity"].tolist() == [10_000.0, 10_000.0, 10_000.0, 10_000.0]


def test_transaction_cost_model_applies_commission_and_slippage() -> None:
    costs = TransactionCostModel(
        market_order_commission=19.99,
        stop_order_commission=29.95,
        slippage_bps=5,
    )

    assert costs.market_fill_price(100.0, side="BUY") == pytest.approx(100.05)
    assert costs.market_fill_price(100.0, side="SELL") == pytest.approx(99.95)
    assert costs.long_stop_fill_price(stop_price=95.0, bar_open=97.0, bar_low=94.0) == pytest.approx(94.9525)
    assert costs.commission(order_type="market") == pytest.approx(19.99)
    assert costs.commission(order_type="stop") == pytest.approx(29.95)


def test_summary_metrics_match_hand_checked_values() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "equity": [100.0, 110.0, 105.0],
        }
    )
    trade_log = pd.DataFrame({"net_pnl": [10.0, -5.0]})

    metrics = calculate_summary_metrics(equity_curve, trade_log)

    assert metrics["starting_equity"] == pytest.approx(100.0)
    assert metrics["ending_equity"] == pytest.approx(105.0)
    assert metrics["total_return"] == pytest.approx(0.05)
    assert metrics["max_drawdown"] == pytest.approx(1.0 - (105.0 / 110.0))
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["average_win"] == pytest.approx(10.0)
    assert metrics["average_loss"] == pytest.approx(-5.0)
    assert metrics["expectancy"] == pytest.approx(2.5)


def _engine(
    *,
    slippage_bps: int,
    market_commission: float,
    stop_commission: float,
    trailing_stop_atr_multiple: float = 1.0,
) -> DailyBarBacktestEngine:
    settings = BreakoutMomentumSettings(
        breakout_lookback=2,
        benchmark_symbol="SPY",
        benchmark_sma_fast=2,
        benchmark_sma_slow=3,
        relative_volume_threshold=1.0,
        atr_window=2,
        stop_atr_multiple=1.0,
        trailing_stop_atr=trailing_stop_atr_multiple,
        require_relative_volume_confirmation=False,
        enable_regime_filter=False,
        regime_filter_mode="either",
    )
    constraints = PortfolioConstraints(
        max_concurrent_positions=5,
        max_position_pct_equity=1.0,
        no_averaging_down=True,
    )
    return DailyBarBacktestEngine(
        strategy_settings=settings,
        portfolio_constraints=constraints,
        starting_cash=10_000.0,
        base_risk_per_trade=0.02,
        cost_model=TransactionCostModel(
            market_order_commission=market_commission,
            stop_order_commission=stop_commission,
            slippage_bps=slippage_bps,
        ),
        close_positions_at_end=True,
    )


def _symbol_frame(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    symbol: str,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * len(closes),
            "symbol": [symbol] * len(closes),
        }
    )
