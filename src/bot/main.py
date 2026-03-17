"""CLI entrypoint for offline research and manual order workflows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import date
from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import yaml

from bot.backtest.engine import DailyBarBacktestEngine
from bot.backtest.metrics import (
    OBJECTIVE_CHOICES,
    build_strategy_comparison_frame,
    metrics_to_serializable_dict,
    rank_strategy_comparisons,
)
from bot.backtest.slippage_costs import TransactionCostModel
from bot.backtest.walkforward import (
    generate_walkforward_folds,
    parameter_grid_from_config,
    run_breakout_walkforward,
    walkforward_fetch_start,
    write_walkforward_reports,
)
from bot.config import (
    AppConfig,
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
from bot.reporting.daily_report import (
    PresetCandidateEvaluation,
    build_daily_research_summary,
    build_daily_signal_report,
    rank_preset_candidate_evaluations,
    write_daily_preset_summary,
    write_daily_research_summary,
    write_daily_signal_report,
)
from bot.reporting.equity_curve import write_equity_curve_report
from bot.reporting.trade_log import write_trade_log_report
from bot.risk.portfolio_rules import PortfolioConstraints, assess_signal_candidate
from bot.strategy.breakout_momentum import (
    BreakoutMomentumSettings,
    BreakoutStrategyPreset,
    generate_breakout_signal,
    resolve_breakout_strategy_presets,
)


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
        choices=OBJECTIVE_CHOICES,
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

    compare_parser = subparsers.add_parser(
        "compare-strategies",
        help="Compare named breakout strategy presets over the same symbols and date range.",
    )
    compare_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    compare_parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Inclusive comparison start date in YYYY-MM-DD format.",
    )
    compare_parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="Inclusive comparison end date in YYYY-MM-DD format.",
    )
    compare_parser.add_argument(
        "--preset-names",
        default=None,
        help="Comma-separated preset names to compare. Defaults to all built-in and config-defined presets.",
    )
    compare_parser.add_argument(
        "--preset",
        action="append",
        default=None,
        help=(
            "Inline custom preset definition. "
            "Format: name=my_preset,breakout_lookback=20,relative_volume_threshold=1.5,"
            "initial_stop_atr=2.5,trailing_stop_atr=3.0,risk_per_trade=0.01"
        ),
    )
    compare_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    compare_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for all compared presets.",
    )
    compare_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter for all compared presets.",
    )
    compare_parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default="sharpe_ratio",
        help="Objective used to rank presets.",
    )
    compare_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top-ranked presets to include in the summary output.",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for comparison CSV and JSON outputs.",
    )
    compare_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    compare_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    compare_parser.set_defaults(handler=_handle_compare_strategies)

    daily_summary_parser = subparsers.add_parser(
        "daily-summary",
        help="Generate a consolidated daily preset summary and suggested order sheet.",
    )
    daily_summary_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    daily_summary_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge of the signal evaluation window.",
    )
    daily_summary_parser.add_argument(
        "--preset-names",
        default=None,
        help="Comma-separated preset names to evaluate. Defaults to standard_breakout unless comparison results are provided.",
    )
    daily_summary_parser.add_argument(
        "--comparison-results",
        type=Path,
        default=None,
        help="Optional ranked preset output from compare-strategies. When provided, the top preset is included automatically.",
    )
    daily_summary_parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Current account equity used for position sizing. Defaults to config starting_cash.",
    )
    daily_summary_parser.add_argument(
        "--current-drawdown",
        type=float,
        default=0.0,
        help="Current portfolio drawdown as a decimal fraction for drawdown-aware risk reduction.",
    )
    daily_summary_parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Lookback window used for universe liquidity screening.",
    )
    daily_summary_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    daily_summary_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    daily_summary_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    daily_summary_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the daily summary reports and suggested order sheet.",
    )
    daily_summary_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    daily_summary_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    daily_summary_parser.set_defaults(handler=_handle_daily_summary)

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


def _handle_compare_strategies(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    preset_names = _parse_text_list(args.preset_names)
    configured_presets = _load_strategy_comparison_presets(args.config_dir)
    presets = resolve_breakout_strategy_presets(
        config.strategy.signals,
        config.strategy.risk,
        configured_presets=configured_presets,
        cli_preset_definitions=tuple(args.preset or ()),
        preset_names=preset_names,
    )

    symbols = load_candidate_symbols(args.candidate_path)
    if not symbols:
        raise ValueError(f"No symbols were found in {args.candidate_path.resolve()}.")

    fetch_start = _comparison_warmup_start(
        start_date=args.start,
        atr_window=config.strategy.risk.atr_length,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        presets=presets,
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
        raise ValueError("No symbol data was available for the requested strategy comparison.")

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

    comparison_frame = _run_breakout_strategy_comparison(
        config=config,
        symbol_frames=symbol_frames,
        benchmark_frame=benchmark_frame,
        start_date=args.start,
        end_date=args.end,
        presets=presets,
        benchmark_symbol_override=args.benchmark_symbol,
        require_relative_volume_confirmation=args.require_relative_volume,
        enable_regime_filter=not args.disable_regime_filter,
    )
    ranked_presets = rank_strategy_comparisons(
        comparison_frame,
        objective=args.objective,
        top_n=args.top_n,
    )

    output_dir = (
        args.output_dir
        or (
            config.project_root
            / "data"
            / "processed"
            / "strategy_comparison"
            / f"{args.start.isoformat()}_{args.end.isoformat()}"
        )
    ).resolve()
    outputs = _write_strategy_comparison_reports(
        comparison_frame,
        ranked_presets,
        output_dir=output_dir,
        objective=args.objective,
    )

    payload = {
        "provider": config.data_sources.provider,
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "objective": args.objective,
        "preset_count": len(presets),
        "preset_names": [preset.name for preset in presets],
        "symbols_requested": len(symbols),
        "symbols_tested": len(symbol_frames),
        "ranked_presets": _dataframe_records(ranked_presets),
        "outputs": outputs,
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_daily_summary(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    presets, preset_selection_source = _resolve_daily_summary_presets(args, config)

    current_equity = (
        config.game_rules.starting_cash if args.equity is None else float(args.equity)
    )
    if current_equity <= 0:
        raise ValueError("equity must be greater than zero.")
    if args.current_drawdown < 0:
        raise ValueError("current_drawdown must be non-negative.")

    builder = UniverseBuilder(provider, config.strategy.universe)
    universe_members = builder.screen_candidates(
        args.candidate_path,
        as_of_date=args.as_of,
        lookback_days=args.lookback_days,
        refresh_cache=args.refresh_cache,
    )

    fetch_start = _comparison_warmup_start(
        start_date=args.as_of,
        atr_window=config.strategy.risk.atr_length,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        presets=presets,
        enable_regime_filter=not args.disable_regime_filter,
    )

    symbol_frames: dict[str, pd.DataFrame] = {}
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
            continue
        if bars.empty:
            LOGGER.warning("Skipping %s because no bars were returned in the requested range.", member.symbol)
            continue
        symbol_frames[member.symbol] = bars

    benchmark_frame = None
    benchmark_symbol = None
    if not args.disable_regime_filter and universe_members:
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
                args.as_of,
                refresh_cache=args.refresh_cache,
            )
        if benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{benchmark_symbol}'."
            )

    constraints = PortfolioConstraints.from_configs(
        config.strategy.risk,
        config.game_rules.rules,
    )
    evaluations: list[PresetCandidateEvaluation] = []
    no_signal_symbols_by_preset: dict[str, list[str]] = {
        preset.name: []
        for preset in presets
    }

    for preset in presets:
        strategy_settings = BreakoutMomentumSettings.from_configs(
            config.strategy.signals,
            config.strategy.risk,
            require_relative_volume_confirmation=args.require_relative_volume,
            enable_regime_filter=not args.disable_regime_filter,
        )
        strategy_settings = preset.apply_to_settings(strategy_settings)
        if args.benchmark_symbol:
            strategy_settings = replace(
                strategy_settings,
                benchmark_symbol=args.benchmark_symbol.strip().upper(),
            )

        for member in universe_members:
            bars = symbol_frames.get(member.symbol)
            if bars is None or bars.empty:
                no_signal_symbols_by_preset[preset.name].append(member.symbol)
                continue

            signal = generate_breakout_signal(
                bars,
                settings=strategy_settings,
                benchmark_frame=benchmark_frame,
                has_open_position=False,
                symbol=member.symbol,
            )
            if signal is None:
                no_signal_symbols_by_preset[preset.name].append(member.symbol)
                continue

            signal_metadata = dict(signal.metadata)
            signal_metadata["preset_name"] = preset.name
            signal_metadata["parameter_id"] = preset.parameter_id
            signal = replace(
                signal,
                strategy_name=f"{signal.strategy_name}:{preset.name}",
                metadata=signal_metadata,
            )
            evaluations.append(
                PresetCandidateEvaluation(
                    preset_name=preset.name,
                    parameter_id=preset.parameter_id,
                    candidate=assess_signal_candidate(
                        signal,
                        current_equity=current_equity,
                        base_risk_per_trade=preset.risk_per_trade,
                        constraints=constraints,
                        current_positions=(),
                        current_drawdown=float(args.current_drawdown),
                    ),
                )
            )

    ranked_evaluations = rank_preset_candidate_evaluations(
        evaluations,
        current_equity=current_equity,
    )
    executor = ManualExecutor()
    execution_batch = executor.build_execution_batch(
        [evaluation.candidate for evaluation in ranked_evaluations],
        as_of_date=args.as_of,
    )
    summary = build_daily_research_summary(
        as_of_date=args.as_of,
        execution_batch=execution_batch,
        evaluations=evaluations,
        selected_presets=presets,
        universe_symbols=[member.symbol for member in universe_members],
        current_equity=current_equity,
        no_signal_symbols_by_preset=no_signal_symbols_by_preset,
        benchmark_symbol=benchmark_symbol,
        preset_selection_source=preset_selection_source,
    )

    output_dir = (
        args.output_dir
        or (config.project_root / "data" / "processed" / "daily_summary" / args.as_of.isoformat())
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = write_daily_research_summary(summary, output_dir / "daily_summary.json")
    opportunities_csv_path = write_daily_research_summary(summary, output_dir / "ranked_opportunities.csv")
    preset_csv_path = write_daily_preset_summary(summary, output_dir / "preset_rankings.csv")
    preset_json_path = write_daily_preset_summary(summary, output_dir / "preset_rankings.json")
    orders_csv_path = write_execution_batch(execution_batch, output_dir / "suggested_order_sheet.csv")
    orders_json_path = write_execution_batch(execution_batch, output_dir / "suggested_order_sheet.json")

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "preset_names": [preset.name for preset in presets],
        "preset_selection_source": preset_selection_source,
        "recommended_preset": summary.recommended_preset,
        "equity": current_equity,
        "current_drawdown": float(args.current_drawdown),
        "universe_count": len(universe_members),
        "candidate_count": len(summary.rows),
        "approved_count": sum(row.status == "approved" for row in summary.rows),
        "rejected_count": sum(row.status == "rejected" for row in summary.rows),
        "order_count": len(execution_batch.orders),
        "top_opportunities": [row.to_dict() for row in summary.rows[:5]],
        "outputs": {
            "daily_summary_json": str(summary_json_path),
            "ranked_opportunities_csv": str(opportunities_csv_path),
            "preset_rankings_csv": str(preset_csv_path),
            "preset_rankings_json": str(preset_json_path),
            "suggested_order_sheet_csv": str(orders_csv_path),
            "suggested_order_sheet_json": str(orders_json_path),
        },
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


def _comparison_warmup_start(
    *,
    start_date: date,
    atr_window: int,
    benchmark_sma_slow: int,
    presets: Sequence[BreakoutStrategyPreset],
    enable_regime_filter: bool,
) -> date:
    max_breakout_lookback = max((preset.breakout_lookback for preset in presets), default=1)
    largest_window = max(
        max_breakout_lookback + 1,
        atr_window,
        max_breakout_lookback + 1,
        benchmark_sma_slow if enable_regime_filter else 1,
    )
    return start_date - timedelta(days=max(largest_window * 3, 30))


def _run_breakout_strategy_comparison(
    *,
    config: AppConfig,
    symbol_frames: dict[str, pd.DataFrame],
    benchmark_frame: pd.DataFrame | None,
    start_date: date,
    end_date: date,
    presets: Sequence[BreakoutStrategyPreset],
    benchmark_symbol_override: str | None,
    require_relative_volume_confirmation: bool,
    enable_regime_filter: bool,
) -> pd.DataFrame:
    base_settings = BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
        require_relative_volume_confirmation=require_relative_volume_confirmation,
        enable_regime_filter=enable_regime_filter,
    )
    if benchmark_symbol_override is not None and benchmark_symbol_override.strip():
        base_settings = replace(
            base_settings,
            benchmark_symbol=benchmark_symbol_override.strip().upper(),
        )

    cost_model = TransactionCostModel.from_game_rules(config.game_rules)
    portfolio_constraints = PortfolioConstraints.from_configs(
        config.strategy.risk,
        config.game_rules.rules,
    )

    rows: list[dict[str, object]] = []
    for preset in presets:
        engine = DailyBarBacktestEngine(
            strategy_settings=preset.apply_to_settings(base_settings),
            portfolio_constraints=portfolio_constraints,
            starting_cash=config.game_rules.starting_cash,
            base_risk_per_trade=preset.risk_per_trade,
            cost_model=cost_model,
            trailing_stop_atr_multiple=preset.trailing_stop_atr,
            close_positions_at_end=True,
        )
        result = engine.run(
            symbol_frames=symbol_frames,
            benchmark_frame=benchmark_frame,
            start_date=start_date,
            end_date=end_date,
        )
        row = preset.to_dict()
        row.update(metrics_to_serializable_dict(result.summary_metrics))
        rows.append(row)

    return build_strategy_comparison_frame(rows)


def _write_strategy_comparison_reports(
    comparison_frame: pd.DataFrame,
    ranked_presets: pd.DataFrame,
    *,
    output_dir: Path,
    objective: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = output_dir / "comparison_results.csv"
    comparison_json = output_dir / "comparison_results.json"
    ranked_csv = output_dir / "ranked_presets.csv"
    ranked_json = output_dir / "ranked_presets.json"
    summary_json = output_dir / "summary.json"

    comparison_frame.to_csv(comparison_csv, index=False)
    ranked_presets.to_csv(ranked_csv, index=False)
    comparison_json.write_text(
        json.dumps(_dataframe_records(comparison_frame), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ranked_json.write_text(
        json.dumps(_dataframe_records(ranked_presets), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(
            {
                "objective": objective,
                "preset_count": int(len(comparison_frame)),
                "top_presets": _dataframe_records(ranked_presets),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "comparison_results_csv": str(comparison_csv),
        "comparison_results_json": str(comparison_json),
        "ranked_presets_csv": str(ranked_csv),
        "ranked_presets_json": str(ranked_json),
        "summary_json": str(summary_json),
    }


def _load_strategy_comparison_presets(config_dir: Path) -> dict[str, object]:
    strategy_path = config_dir / "strategy.yaml"
    if not strategy_path.exists():
        return {}

    raw_config = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) or {}
    configured_presets = raw_config.get("comparison_presets", {})
    if configured_presets is None:
        return {}
    if not isinstance(configured_presets, dict):
        raise ValueError("comparison_presets in strategy.yaml must be a mapping when provided.")
    return dict(configured_presets)


def _resolve_daily_summary_presets(
    args: argparse.Namespace,
    config: AppConfig,
) -> tuple[list[BreakoutStrategyPreset], str]:
    configured_presets = _load_strategy_comparison_presets(args.config_dir)
    requested_names = list(_parse_text_list(args.preset_names) or ())
    selection_source = "named_presets" if requested_names else "default_standard_breakout"

    if args.comparison_results is not None:
        top_preset_name = _load_top_preset_name_from_results(args.comparison_results)
        if top_preset_name not in requested_names:
            requested_names.append(top_preset_name)
        selection_source = f"comparison_results:{args.comparison_results.resolve()}"

    if not requested_names:
        requested_names = ["standard_breakout"]

    presets = resolve_breakout_strategy_presets(
        config.strategy.signals,
        config.strategy.risk,
        configured_presets=configured_presets,
        preset_names=tuple(requested_names),
    )
    return presets, selection_source


def _load_top_preset_name_from_results(results_path: Path) -> str:
    resolved_path = results_path.resolve()
    if not resolved_path.exists():
        raise ValueError(f"Comparison results file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Comparison results CSV is empty: {resolved_path}")
        preset_name = first_row.get("preset_name")
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise ValueError(
                f"Comparison results CSV does not contain a usable preset_name column: {resolved_path}"
            )
        return preset_name.strip()

    if suffix == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        records: list[object]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            top_presets = payload.get("top_presets")
            if isinstance(top_presets, list):
                records = top_presets
            else:
                records = [payload]
        else:
            raise ValueError(f"Unsupported JSON payload for comparison results: {resolved_path}")

        for record in records:
            if not isinstance(record, dict):
                continue
            preset_name = record.get("preset_name")
            if isinstance(preset_name, str) and preset_name.strip():
                return preset_name.strip()
        raise ValueError(
            f"Comparison results JSON does not contain a usable preset_name: {resolved_path}"
        )

    raise ValueError(
        f"Unsupported comparison results format '{resolved_path.suffix}'. Expected CSV or JSON."
    )


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


def _parse_text_list(raw_value: str | None) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    cleaned_values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not cleaned_values:
        raise ValueError("Text lists cannot be empty.")
    return cleaned_values


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
