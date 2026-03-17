"""CLI entrypoint for offline research and manual order workflows."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from bot.config import (
    ConfigError,
    default_config_dir,
    default_env_file,
    load_app_config,
    validate_environment,
)
from bot.data.providers import DataProviderConfigurationError, DataProviderError, create_daily_bar_provider
from bot.data.universe import UniverseBuilder
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
