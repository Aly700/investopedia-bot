"""CLI entrypoint for offline research and manual order workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from bot.backtest.engine import DailyBarBacktestEngine
from bot.backtest.metrics import metrics_to_serializable_dict
from bot.backtest.walkforward import (
    generate_walkforward_folds,
    parameter_grid_from_config,
    run_breakout_walkforward,
    walkforward_fetch_start,
    write_walkforward_reports,
)
from bot.config import (
    ConfigError,
    default_config_dir,
    default_env_file,
    load_app_config,
    validate_environment,
)
from bot.data.providers import DataProviderConfigurationError, DataProviderError, create_daily_bar_provider
from bot.data.universe import UniverseBuilder, load_candidate_symbols
from bot.execution.manual_executor import (
    ManualExecutor,
    ManualOrderError,
    load_orders_from_csv,
    write_execution_batch,
    write_manual_order_sheet,
)
from bot.logging_utils import get_logger, setup_logging
from bot.reporting.daily_report import build_daily_signal_report, write_daily_signal_report
from bot.reporting.equity_curve import write_equity_curve_report
from bot.reporting.trade_log import write_trade_log_report
from bot.risk.portfolio_rules import PortfolioConstraints, assess_signal_candidate
from bot.strategy.breakout_momentum import BreakoutMomentumSettings, generate_breakout_signal


LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="investopedia-bot",
        description="Daily-bar research, configuration validation, and manual order tooling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir(),
        help="Directory containing strategy, data source, and game rule YAML files.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=default_env_file(),
        help="Optional .env file used for provider credential checks.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Root log level for the current command.",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Emit structured JSON logs instead of plain text.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path for command output.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config_parser = subparsers.add_parser(
        "show-config",
        help="Load the repo config and print the merged result.",
    )
    show_config_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format for the rendered config.",
    )
    show_config_parser.set_defaults(handler=_handle_show_config)

    check_env_parser = subparsers.add_parser(
        "check-env",
        help="Validate environment variables for the active research data provider.",
    )
    check_env_parser.set_defaults(handler=_handle_check_env)

    fetch_data_parser = subparsers.add_parser(
        "fetch-data",
        help="Fetch normalized daily bars for one symbol using the configured provider.",
    )
    fetch_data_parser.add_argument("symbol", help="Ticker symbol to fetch.")
    fetch_data_parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    fetch_data_parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    fetch_data_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the normalized daily bars as CSV.",
    )
    fetch_data_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Summary output format.",
    )
    fetch_data_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force a provider fetch.",
    )
    fetch_data_parser.set_defaults(handler=_handle_fetch_data)

    build_universe_parser = subparsers.add_parser(
        "build-universe",
        help="Build a filtered trading universe from a candidate symbol list.",
    )
    build_universe_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    build_universe_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Date used as the right edge of the screening window.",
    )
    build_universe_parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Number of recent daily bars to use when computing average dollar volume.",
    )
    build_universe_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the selected universe.",
    )
    build_universe_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    build_universe_parser.set_defaults(handler=_handle_build_universe)

    generate_orders_parser = subparsers.add_parser(
        "generate-orders",
        help="Generate a manual daily order sheet from current signals and risk checks.",
    )
    generate_orders_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    generate_orders_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge of the signal evaluation window.",
    )
    generate_orders_parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Current account equity used for position sizing. Defaults to config starting_cash.",
    )
    generate_orders_parser.add_argument(
        "--current-drawdown",
        type=float,
        default=0.0,
        help="Current portfolio drawdown as a decimal fraction for drawdown-aware risk reduction.",
    )
    generate_orders_parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Lookback window used for universe liquidity screening.",
    )
    generate_orders_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    generate_orders_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    generate_orders_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    generate_orders_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the manual order sheet and daily reports.",
    )
    generate_orders_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    generate_orders_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    generate_orders_parser.set_defaults(handler=_handle_generate_orders)

    render_orders_parser = subparsers.add_parser(
        "render-orders",
        help="Render offline signal output into a manual order blotter CSV.",
    )
    render_orders_parser.add_argument(
        "input_path",
        type=Path,
        help="CSV file containing candidate orders.",
    )
    render_orders_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the normalized manual order sheet.",
    )
    render_orders_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=None,
        help="Trading date to stamp on the output file in YYYY-MM-DD format.",
    )
    render_orders_parser.set_defaults(handler=_handle_render_orders)

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run a deterministic daily-bar backtest over a symbol list.",
    )
    backtest_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    backtest_parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Inclusive backtest start date in YYYY-MM-DD format.",
    )
    backtest_parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="Inclusive backtest end date in YYYY-MM-DD format.",
    )
    backtest_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    backtest_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    backtest_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    backtest_parser.add_argument(
        "--no-end-of-data-closeout",
        action="store_true",
        help="Leave positions open instead of force-closing them on the final bar.",
    )
    backtest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for trade log, equity curve, and summary output files.",
    )
    backtest_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Summary output format.",
    )
    backtest_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    backtest_parser.set_defaults(handler=_handle_backtest)

    walkforward_parser = subparsers.add_parser(
        "walkforward",
        help="Run rolling or expanding walk-forward validation with parameter sweeps.",
    )
    walkforward_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    walkforward_parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Inclusive walk-forward start date in YYYY-MM-DD format.",
    )
    walkforward_parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="Inclusive walk-forward end date in YYYY-MM-DD format.",
    )
    walkforward_parser.add_argument(
        "--train-days",
        type=int,
        required=True,
        help="Number of calendar days in each training window.",
    )
    walkforward_parser.add_argument(
        "--test-days",
        type=int,
        required=True,
        help="Number of calendar days in each validation/test window.",
    )
    walkforward_parser.add_argument(
        "--expanding-train",
        action="store_true",
        help="Use expanding training windows instead of rolling windows.",
    )
    walkforward_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    walkforward_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    walkforward_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    walkforward_parser.add_argument(
        "--breakout-lookbacks",
        default=None,
        help="Comma-separated breakout lookback values to sweep.",
    )
    walkforward_parser.add_argument(
        "--relative-volume-thresholds",
        default=None,
        help="Comma-separated relative-volume thresholds to sweep.",
    )
    walkforward_parser.add_argument(
        "--initial-stop-atrs",
        default=None,
        help="Comma-separated initial ATR stop multiples to sweep.",
    )
    walkforward_parser.add_argument(
        "--trailing-stop-atrs",
        default=None,
        help="Comma-separated trailing ATR stop multiples to sweep.",
    )
    walkforward_parser.add_argument(
        "--risk-per-trade-values",
        default=None,
        help="Comma-separated per-trade risk values to sweep.",
    )
    walkforward_parser.add_argument(
        "--objective",
        choices=(
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
        ),
        default="sharpe_ratio",
        help="Objective used to rank parameter sets.",
    )
    walkforward_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top aggregate parameter sets to include in the summary.",
    )
    walkforward_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for walk-forward CSV and JSON outputs.",
    )
    walkforward_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    walkforward_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    walkforward_parser.set_defaults(handler=_handle_walkforward)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file, json_logs=args.json_logs)

    try:
        return int(args.handler(args))
    except (ConfigError, DataProviderConfigurationError, DataProviderError, ManualOrderError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1


def _handle_show_config(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    payload = config.to_dict()

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(payload, sort_keys=False))
    return 0


def _handle_check_env(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    validation = validate_environment(config, env_file=args.env_file)

    if validation.is_valid:
        print(
            f"Provider '{validation.provider}' is ready. "
            f"Found: {', '.join(validation.present) or 'no credentials required'}."
        )
        return 0

    print(
        f"Provider '{validation.provider}' is missing environment variables: "
        f"{', '.join(validation.missing)}.",
        file=sys.stderr,
    )
    return 1


def _handle_fetch_data(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    bars = provider.fetch_daily_bars(
        args.symbol,
        args.start,
        args.end,
        refresh_cache=args.refresh_cache,
    )

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        LOGGER.info("Saved %s rows to %s", len(bars), output_path)

    summary = {
        "provider": config.data_sources.provider,
        "symbol": args.symbol.upper(),
        "rows": int(len(bars)),
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "first_bar": bars["date"].iloc[0].date().isoformat() if not bars.empty else None,
        "last_bar": bars["date"].iloc[-1].date().isoformat() if not bars.empty else None,
    }
    _print_structured(summary, output_format=args.format)
    return 0


def _handle_build_universe(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    builder = UniverseBuilder(provider, config.strategy.universe)
    members = builder.screen_candidates(
        args.candidate_path,
        as_of_date=args.as_of,
        lookback_days=args.lookback_days,
        refresh_cache=args.refresh_cache,
    )

    if args.format == "json":
        payload = {
            "provider": config.data_sources.provider,
            "as_of_date": args.as_of.isoformat(),
            "lookback_days": args.lookback_days,
            "count": len(members),
            "symbols": [member.symbol for member in members],
            "members": [member.to_dict() for member in members],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for member in members:
        print(member.symbol)
    return 0


def _handle_generate_orders(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    builder = UniverseBuilder(provider, config.strategy.universe)
    universe_members = builder.screen_candidates(
        args.candidate_path,
        as_of_date=args.as_of,
        lookback_days=args.lookback_days,
        refresh_cache=args.refresh_cache,
    )

    strategy_settings = BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
        require_relative_volume_confirmation=args.require_relative_volume,
        enable_regime_filter=not args.disable_regime_filter,
    )
    if args.benchmark_symbol:
        strategy_settings = replace(
            strategy_settings,
            benchmark_symbol=args.benchmark_symbol.strip().upper(),
        )

    current_equity = (
        config.game_rules.starting_cash if args.equity is None else float(args.equity)
    )
    if current_equity <= 0:
        raise ValueError("equity must be greater than zero.")
    if args.current_drawdown < 0:
        raise ValueError("current_drawdown must be non-negative.")

    benchmark_frame = None
    fetch_start = _strategy_warmup_start(args.as_of, strategy_settings)
    if strategy_settings.enable_regime_filter and universe_members:
        benchmark_frame = provider.fetch_daily_bars(
            strategy_settings.benchmark_symbol,
            fetch_start,
            args.as_of,
            refresh_cache=args.refresh_cache,
        )
        if benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{strategy_settings.benchmark_symbol}'."
            )

    constraints = PortfolioConstraints.from_configs(
        config.strategy.risk,
        config.game_rules.rules,
    )
    assessed_candidates = []
    no_signal_symbols: list[str] = []

    for member in universe_members:
        try:
            bars = provider.fetch_daily_bars(
                member.symbol,
                fetch_start,
                args.as_of,
                refresh_cache=args.refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning("Skipping %s due to provider error: %s", member.symbol, exc)
            no_signal_symbols.append(member.symbol)
            continue

        if bars.empty:
            no_signal_symbols.append(member.symbol)
            continue

        signal = generate_breakout_signal(
            bars,
            settings=strategy_settings,
            benchmark_frame=benchmark_frame,
            has_open_position=False,
            symbol=member.symbol,
        )
        if signal is None:
            no_signal_symbols.append(member.symbol)
            continue

        assessed_candidates.append(
            assess_signal_candidate(
                signal,
                current_equity=current_equity,
                base_risk_per_trade=config.strategy.risk.risk_per_trade,
                constraints=constraints,
                current_positions=(),
                current_drawdown=float(args.current_drawdown),
            )
        )

    executor = ManualExecutor()
    batch = executor.build_execution_batch(assessed_candidates, as_of_date=args.as_of)
    report = build_daily_signal_report(
        as_of_date=args.as_of,
        execution_batch=batch,
        assessed_candidates=assessed_candidates,
        universe_symbols=[member.symbol for member in universe_members],
        no_signal_symbols=no_signal_symbols,
        benchmark_symbol=strategy_settings.benchmark_symbol if strategy_settings.enable_regime_filter else None,
    )

    output_dir = (args.output_dir or _default_daily_output_dir(config.project_root, args.as_of)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    order_sheet_path = write_execution_batch(batch, output_dir / "manual_order_sheet.csv")
    signal_report_json_path = write_daily_signal_report(report, output_dir / "daily_signal_report.json")
    signal_report_csv_path = write_daily_signal_report(report, output_dir / "daily_signal_report.csv")

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "equity": current_equity,
        "current_drawdown": float(args.current_drawdown),
        "universe_count": len(universe_members),
        "signal_count": len(assessed_candidates),
        "approved_order_count": len(batch.orders),
        "rejected_signal_count": sum(not candidate.approved for candidate in assessed_candidates),
        "no_signal_count": len(no_signal_symbols),
        "order_symbols": [order.symbol for order in batch.orders],
        "outputs": {
            "manual_order_sheet": str(order_sheet_path),
            "daily_signal_report_json": str(signal_report_json_path),
            "daily_signal_report_csv": str(signal_report_csv_path),
        },
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_render_orders(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    orders = load_orders_from_csv(args.input_path)
    as_of_date = args.as_of or date.today()
    output_path = args.output or _default_order_output_path(config.project_root, as_of_date)
    written_path = write_manual_order_sheet(orders, output_path, as_of_date=as_of_date)

    buy_count = sum(order.side == "BUY" for order in orders)
    sell_count = sum(order.side == "SELL" for order in orders)
    LOGGER.info(
        "Wrote %s manual orders (%s buy, %s sell) to %s",
        len(orders),
        buy_count,
        sell_count,
        written_path,
    )
    print(written_path)
    return 0


def _handle_backtest(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    engine = DailyBarBacktestEngine.from_config(
        config,
        benchmark_symbol_override=args.benchmark_symbol,
        require_relative_volume_confirmation=args.require_relative_volume,
        enable_regime_filter=not args.disable_regime_filter,
        close_positions_at_end=not args.no_end_of_data_closeout,
    )

    symbols = load_candidate_symbols(args.candidate_path)
    if not symbols:
        raise ValueError(f"No symbols were found in {args.candidate_path.resolve()}.")

    fetch_start = engine.warmup_start(args.start)
    symbol_frames: dict[str, object] = {}
    for symbol in symbols:
        try:
            bars = provider.fetch_daily_bars(
                symbol,
                fetch_start,
                args.end,
                refresh_cache=args.refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning("Skipping %s due to provider error: %s", symbol, exc)
            continue

        if bars.empty:
            LOGGER.warning("Skipping %s because no bars were returned in the requested range.", symbol)
            continue
        symbol_frames[symbol] = bars

    if not symbol_frames:
        raise ValueError("No symbol data was available for the requested backtest.")

    benchmark_frame = None
    if engine.strategy_settings.enable_regime_filter:
        benchmark_symbol = engine.strategy_settings.benchmark_symbol
        if benchmark_symbol in symbol_frames:
            benchmark_frame = symbol_frames[benchmark_symbol]
        else:
            benchmark_frame = provider.fetch_daily_bars(
                benchmark_symbol,
                fetch_start,
                args.end,
                refresh_cache=args.refresh_cache,
            )
        if benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{benchmark_symbol}'."
            )

    result = engine.run(
        symbol_frames=symbol_frames,
        benchmark_frame=benchmark_frame,
        start_date=args.start,
        end_date=args.end,
    )

    written_files: dict[str, str] = {}
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        trade_log_path = output_dir / "trade_log.csv"
        equity_curve_path = output_dir / "equity_curve.csv"
        summary_path = output_dir / f"summary.{args.format}"

        write_trade_log_report(result.trade_log, trade_log_path)
        write_equity_curve_report(result.equity_curve, equity_curve_path)
        if args.format == "json":
            summary_path.write_text(
                json.dumps(metrics_to_serializable_dict(result.summary_metrics), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            summary_path.write_text(
                yaml.safe_dump(metrics_to_serializable_dict(result.summary_metrics), sort_keys=False),
                encoding="utf-8",
            )
        written_files = {
            "trade_log": str(trade_log_path),
            "equity_curve": str(equity_curve_path),
            "summary": str(summary_path),
        }

    payload = {
        "symbols_requested": len(symbols),
        "symbols_tested": len(symbol_frames),
        "benchmark_symbol": engine.strategy_settings.benchmark_symbol
        if engine.strategy_settings.enable_regime_filter
        else None,
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "trade_log_rows": int(len(result.trade_log)),
        "equity_curve_rows": int(len(result.equity_curve)),
        "metrics": metrics_to_serializable_dict(result.summary_metrics),
        "outputs": written_files,
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_walkforward(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    parameter_sets = parameter_grid_from_config(
        config,
        breakout_lookbacks=_parse_int_list(args.breakout_lookbacks),
        relative_volume_thresholds=_parse_float_list(args.relative_volume_thresholds),
        initial_stop_atrs=_parse_float_list(args.initial_stop_atrs),
        trailing_stop_atrs=_parse_float_list(args.trailing_stop_atrs),
        risk_per_trade_values=_parse_float_list(args.risk_per_trade_values),
    )
    folds = generate_walkforward_folds(
        start_date=args.start,
        end_date=args.end,
        train_window_days=args.train_days,
        test_window_days=args.test_days,
        expanding_train=args.expanding_train,
    )
    if not folds:
        raise ValueError("No walk-forward folds could be generated for the requested date range.")

    symbols = load_candidate_symbols(args.candidate_path)
    if not symbols:
        raise ValueError(f"No symbols were found in {args.candidate_path.resolve()}.")

    fetch_start = walkforward_fetch_start(
        initial_start_date=folds[0].train_start,
        parameter_sets=parameter_sets,
        atr_window=config.strategy.risk.atr_length,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        enable_regime_filter=not args.disable_regime_filter,
    )

    symbol_frames: dict[str, object] = {}
    for symbol in symbols:
        try:
            bars = provider.fetch_daily_bars(
                symbol,
                fetch_start,
                args.end,
                refresh_cache=args.refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning("Skipping %s due to provider error: %s", symbol, exc)
            continue

        if bars.empty:
            LOGGER.warning("Skipping %s because no bars were returned in the requested range.", symbol)
            continue
        symbol_frames[symbol] = bars

    if not symbol_frames:
        raise ValueError("No symbol data was available for the requested walk-forward analysis.")

    benchmark_frame = None
    if not args.disable_regime_filter:
        benchmark_symbol = (
            args.benchmark_symbol.strip().upper()
            if args.benchmark_symbol
            else config.strategy.signals.benchmark_symbol
        )
        if benchmark_symbol in symbol_frames:
            benchmark_frame = symbol_frames[benchmark_symbol]
        else:
            benchmark_frame = provider.fetch_daily_bars(
                benchmark_symbol,
                fetch_start,
                args.end,
                refresh_cache=args.refresh_cache,
            )
        if benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{benchmark_symbol}'."
            )

    result = run_breakout_walkforward(
        config=config,
        symbol_frames=symbol_frames,
        benchmark_frame=benchmark_frame,
        folds=folds,
        parameter_sets=parameter_sets,
        objective=args.objective,
        top_n=args.top_n,
        benchmark_symbol_override=args.benchmark_symbol,
        require_relative_volume_confirmation=args.require_relative_volume,
        enable_regime_filter=not args.disable_regime_filter,
    )

    output_dir = (
        args.output_dir
        or (config.project_root / "data" / "processed" / "walkforward" / f"{args.start.isoformat()}_{args.end.isoformat()}")
    ).resolve()
    outputs = write_walkforward_reports(result, output_dir)

    payload = {
        "provider": config.data_sources.provider,
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "train_days": args.train_days,
        "test_days": args.test_days,
        "expanding_train": bool(args.expanding_train),
        "fold_count": len(result.folds),
        "parameter_set_count": len(result.parameter_sets),
        "objective": args.objective,
        "symbols_requested": len(symbols),
        "symbols_tested": len(symbol_frames),
        "top_parameter_sets": _dataframe_records(result.best_parameter_sets),
        "outputs": outputs,
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _default_order_output_path(project_root: Path, as_of_date: date) -> Path:
    output_dir = project_root / "data" / "processed" / "orders"
    return output_dir / f"{as_of_date.isoformat()}_manual_orders.csv"


def _default_daily_output_dir(project_root: Path, as_of_date: date) -> Path:
    return project_root / "data" / "processed" / "daily" / as_of_date.isoformat()


def _strategy_warmup_start(
    as_of_date: date,
    settings: BreakoutMomentumSettings,
) -> date:
    largest_window = max(
        settings.breakout_lookback + 1,
        settings.atr_window,
        settings.resolved_relative_volume_window + 1,
        settings.benchmark_sma_slow if settings.enable_regime_filter else 1,
    )
    return as_of_date - timedelta(days=max(largest_window * 3, 30))


def _print_structured(payload: dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(yaml.safe_dump(payload, sort_keys=False))


def _parse_iso_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{raw_value}'. Expected YYYY-MM-DD."
        ) from exc


def _parse_int_list(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    cleaned_values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not cleaned_values:
        raise ValueError("Parameter lists cannot be empty.")
    return tuple(int(value) for value in cleaned_values)


def _parse_float_list(raw_value: str | None) -> tuple[float, ...] | None:
    if raw_value is None:
        return None
    cleaned_values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not cleaned_values:
        raise ValueError("Parameter lists cannot be empty.")
    return tuple(float(value) for value in cleaned_values)


def _dataframe_records(frame: object) -> list[dict[str, object]]:
    if not hasattr(frame, "empty") or not hasattr(frame, "to_dict"):
        return []
    if frame.empty:
        return []
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        records.append(normalized)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
