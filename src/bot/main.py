"""CLI entrypoint for offline research and manual order workflows."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from bot.backtest.engine import DailyBarBacktestEngine
from bot.backtest.metrics import metrics_to_serializable_dict
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
    ManualOrderError,
    load_orders_from_csv,
    write_manual_order_sheet,
)
from bot.logging_utils import get_logger, setup_logging


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

        result.trade_log.to_csv(trade_log_path, index=False, date_format="%Y-%m-%d")
        result.equity_curve.to_csv(equity_curve_path, index=False, date_format="%Y-%m-%d")
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


def _default_order_output_path(project_root: Path, as_of_date: date) -> Path:
    output_dir = project_root / "data" / "processed" / "orders"
    return output_dir / f"{as_of_date.isoformat()}_manual_orders.csv"


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


if __name__ == "__main__":
    raise SystemExit(main())
