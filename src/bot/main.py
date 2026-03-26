"""CLI entrypoint for offline research and manual order workflows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import date
from datetime import datetime, timezone
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml

from bot.api.control_api import (
    CancelPendingOrderCommand,
    ForceBrokerSyncCommand,
    OperatorControlService,
    ReplacePendingOrderCommand,
    SubmitPendingOrderCommand,
    serve_operator_control_api,
)
from bot.api.internal_api import InternalApiQueryService, serve_internal_api
from bot.dashboard.operator_dashboard import serve_operator_dashboard
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
from bot.broker import (
    BrokerExecutionAdapter,
    BrokerExecutionError,
    BrokerOrderUpdate,
    BrokerReplaceOrderRequest,
    create_broker_execution_adapter,
    reconcile_broker_order_updates,
    submit_pending_orders_via_broker,
)
from bot.config import (
    AppConfig,
    ConfigError,
    UniverseProfileConfig,
    _read_env_file,
    default_config_dir,
    default_env_file,
    load_app_config,
    validate_environment,
)
from bot.data.candidate_journal import (
    CandidateScoreJournal,
    CandidateJournalObservation,
    CandidateScoreJournalStore,
    build_candidate_memory_features,
    update_candidate_score_journal,
)
from bot.data.earnings import (
    EarningsRiskContext,
    build_earnings_risk_contexts,
    create_earnings_calendar_provider,
    earnings_context_metadata,
)
from bot.data.intraday_state_journal import (
    IntradayStateJournal,
    IntradayStateObservation,
    IntradayStateJournalStore,
    default_intraday_state_journal_path,
    preview_intraday_trajectory_features,
    update_intraday_state_journal,
)
from bot.data.pending_orders import (
    OrderReservationState,
    PendingOrderRecord,
    PendingOrderState,
    PendingOrderStateStore,
    build_pending_order_record_from_execution_order,
    build_order_reservation_state,
    build_pending_order_holding_exposures,
    create_pending_order,
    replace_pending_order,
    upsert_pending_order_state,
)
from bot.data.portfolio_heat import (
    PortfolioHoldingExposure,
    build_portfolio_heat_context,
)
from bot.data.position_trajectory import (
    PositionTrajectoryJournal,
    PositionTrajectoryObservation,
    PositionTrajectoryJournalStore,
    default_position_trajectory_journal_path,
    preview_position_trajectory_features,
    update_position_trajectory_journal,
)
from bot.data.sector_context import (
    SectorFeatureContext,
    SymbolSectorClassification,
    build_sector_feature_contexts,
    load_symbol_sector_classifications,
    sector_context_history_start,
    sector_context_metadata,
)
from bot.data.trade_feedback import (
    TradeFeedbackEvent,
    TradeFeedbackLogStore,
    TradeOutcomeSnapshot,
    build_trade_decision_id,
    build_trade_id,
    compute_trade_outcome_snapshot,
    default_trade_feedback_log_path,
    load_trade_feedback_events,
)
from bot.data.providers import (
    DailyBarProvider,
    DataProviderConfigurationError,
    DataProviderError,
    create_daily_bar_provider,
)
from bot.data.volatility_context import (
    build_volatility_regime_context,
    volatility_context_history_start,
)
from bot.data.reference import create_reference_universe_provider
from bot.data.universe import UniverseBuilder, load_candidate_symbols
from bot.data.universe_pipeline import (
    UniverseProfileBuildResult,
    apply_universe_filters,
    enrich_universe_metadata,
    fetch_master_universe,
    write_universe_outputs,
)
from bot.execution.manual_executor import (
    ManualExecutor,
    ManualOrderError,
    load_orders_from_csv,
    write_execution_batch,
    write_manual_order_sheet,
)
from bot.execution.interface import ExecutionBatch, ExecutionOrder
from bot.execution.prioritization import (
    annotate_execution_priority,
    rank_risk_assessed_candidates,
)
from bot.features import (
    PositionFeatures,
    apply_intraday_trajectory_features,
    apply_position_trajectory_features,
    build_candidate_features,
    build_intraday_position_features,
    build_market_context,
    build_position_features,
    compute_market_breadth_from_frames,
    market_context_metadata,
)
from bot.ingestion.market_data import PollingIngestionAdapter
from bot.ingestion.streaming import LiveUpdateBufferSnapshot, WebsocketIngestionAdapter
from bot.ingestion.websocket_transport import (
    JsonWebsocketTransport,
    normalize_polygon_websocket_message,
)
from bot.indicators.volatility import atr
from bot.logging_utils import get_logger, setup_logging
from bot.notifications import (
    NotificationEvent,
    NotificationRoute,
    NotificationRouter,
    NotificationDeliveryStateStore,
    WebhookNotificationChannel,
    build_market_state_transition_notification,
    build_trade_feedback_notification,
    build_warning_notification,
)
from bot.orchestration.live_runner import (
    LiveMarketCycleRequest,
    LiveMarketCycleResult,
    LiveMarketRunner,
    PersistedMarketStateArtifacts,
)
from bot.reporting.daily_report import (
    DailyResearchSummary,
    IntradayPortfolioReviewReport,
    MarketMonitorReport,
    PresetCandidateEvaluation,
    PortfolioReviewReport,
    PortfolioReviewRow,
    build_daily_research_summary,
    build_intraday_portfolio_review_report,
    build_market_monitor_report,
    build_portfolio_review_report,
    build_daily_signal_report,
    market_monitor_flat_count_key,
    rank_preset_candidate_evaluations,
    write_daily_research_brief,
    write_market_monitor_report,
    write_market_monitor_brief,
    write_market_monitor_text_summary,
    write_daily_preset_summary,
    write_intraday_portfolio_review_brief,
    write_intraday_portfolio_review_report,
    write_portfolio_review_report,
    write_daily_research_summary,
    write_daily_signal_report,
)
from bot.reporting.equity_curve import write_equity_curve_report
from bot.reporting.trade_review import (
    build_trade_review_analytics_summary,
    default_trade_review_output_dir,
    write_trade_review_brief,
    write_trade_review_summary,
)
from bot.reporting.trade_log import write_trade_log_report
from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioConstraints,
    PortfolioInputError,
    PORTFOLIO_REVIEW_ACTIONS,
    RiskAssessedCandidate,
    assess_signal_candidate,
    initialize_portfolio_snapshot,
    load_existing_positions,
    remove_existing_position_snapshot,
    review_existing_long_position,
    review_existing_long_position_intraday,
    update_existing_position_stop_snapshot,
    upsert_existing_position_snapshot,
)
from bot.risk.stops import trailing_stop_reference
from bot.strategy.breakout_momentum import (
    BreakoutMomentumSettings,
    BreakoutStrategyPreset,
    generate_breakout_signal,
    resolve_breakout_strategy_presets,
)
from bot.strategy.regime_filter import regime_is_bullish
from bot.state.market_state import (
    MarketStateStore,
    write_market_state_change_set,
    write_market_state_change_summary,
)
from bot.service.live_market_service import (
    LiveMarketServiceStatusStore,
    LiveMarketSupervisor,
    ReconnectBackoffPolicy,
)
from bot.service.local_staging_runtime import (
    LocalStagingRuntimeConfig,
    create_local_staging_runtime,
)
from bot.service.paper_validation import (
    build_local_paper_validation_smoke_result,
    build_local_paper_validation_profile,
    build_local_paper_validation_runtime_config,
    build_local_paper_validation_summary,
    create_local_paper_validation_runtime,
    write_local_paper_validation_smoke_result,
    write_local_paper_validation_summary,
)


LOGGER = get_logger(__name__)

_POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY = "_position_trajectory_observation"
_TRADE_OUTCOME_SNAPSHOT_METADATA_KEY = "_trade_outcome_snapshot"


@dataclass
class _LiveMarketServiceBundle:
    supervisor: LiveMarketSupervisor
    latest_outputs: dict[str, str]
    notification_router: NotificationRouter | None


def _configured_provider_name(config: object) -> str | None:
    data_sources = getattr(config, "data_sources", None)
    provider_name = getattr(data_sources, "provider", None)
    return provider_name if isinstance(provider_name, str) and provider_name else None


def _add_notification_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--notification-webhook-url",
        default=None,
        help="Optional outbound webhook URL for notification delivery.",
    )
    parser.add_argument(
        "--notification-webhook-format",
        choices=("generic", "discord"),
        default=None,
        help="Optional webhook payload format override.",
    )
    parser.add_argument(
        "--notification-min-severity",
        choices=("info", "warning", "critical"),
        default=None,
        help="Optional minimum notification severity to deliver.",
    )


def _notification_environment(env_file: Path | None) -> dict[str, str]:
    merged = _read_env_file(env_file)
    merged.update(dict(os.environ))
    return merged


def _build_notification_router(
    *,
    project_root: Path,
    env_file: Path | None,
    webhook_url: str | None,
    webhook_format: str | None,
    minimum_severity: str | None,
) -> NotificationRouter | None:
    merged_env = _notification_environment(env_file)
    resolved_webhook_url = _clean_optional_text(webhook_url)
    resolved_format = _clean_optional_text(webhook_format)
    if resolved_webhook_url is None:
        resolved_webhook_url = _clean_optional_text(merged_env.get("NOTIFICATION_WEBHOOK_URL"))
    if resolved_webhook_url is None:
        discord_webhook_url = _clean_optional_text(merged_env.get("DISCORD_WEBHOOK_URL"))
        if discord_webhook_url is not None:
            resolved_webhook_url = discord_webhook_url
            resolved_format = resolved_format or "discord"
    if resolved_webhook_url is None:
        return None
    resolved_minimum_severity = (
        _clean_optional_text(minimum_severity)
        or _clean_optional_text(merged_env.get("NOTIFICATION_MIN_SEVERITY"))
    )
    channel = WebhookNotificationChannel(
        webhook_url=resolved_webhook_url,
        payload_format=resolved_format or "generic",
    )
    return NotificationRouter(
        routes=(
            NotificationRoute(
                channel=channel,
                minimum_severity=resolved_minimum_severity,
            ),
        ),
        state_store=NotificationDeliveryStateStore(project_root),
    )


def _route_notification_events(
    router: NotificationRouter | None,
    events: Sequence[NotificationEvent],
) -> dict[str, object] | None:
    if router is None or not events:
        return None
    batch_result = router.route_events(events)
    return batch_result.to_dict()


def _notification_events_from_cycle_result(
    cycle_result: LiveMarketCycleResult,
) -> tuple[NotificationEvent, ...]:
    if cycle_result.market_state_snapshot is None:
        return ()
    created_at = cycle_result.market_state_snapshot.as_of_timestamp
    return tuple(
        build_market_state_transition_notification(
            transition,
            created_at=created_at,
        )
        for transition in cycle_result.state_transitions
    )


def _notification_events_from_trade_feedback_events(
    events: Sequence[TradeFeedbackEvent],
) -> tuple[NotificationEvent, ...]:
    notifications: list[NotificationEvent] = []
    for event in events:
        notification = build_trade_feedback_notification(event)
        if notification is not None:
            notifications.append(notification)
    return tuple(notifications)


def _notification_events_from_warning_messages(
    *,
    workflow: str,
    created_at: str,
    warnings: Sequence[str],
    related_ids: Mapping[str, str] | None = None,
) -> tuple[NotificationEvent, ...]:
    notifications: list[NotificationEvent] = []
    for warning in warnings:
        message_hash = hashlib.sha1(
            f"{workflow}:{warning}".encode("utf-8")
        ).hexdigest()[:16]
        notifications.append(
            build_warning_notification(
                event_type="execution_warning",
                severity="warning",
                title=f"{workflow} warning",
                body=warning,
                created_at=created_at,
                related_ids=dict(related_ids or {}),
                dedupe_key=f"execution-warning:{message_hash}",
                suppression_scope="once",
            )
        )
    return tuple(notifications)


class NoIntradayDataError(ValueError):
    """Raised when an intraday review session has no usable regular-session bars."""


@dataclass(frozen=True)
class PortfolioReviewRowState:
    """A portfolio review row plus persistence-side artifacts derived from it."""

    row: PortfolioReviewRow
    position_trajectory_observation: PositionTrajectoryObservation | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.row, name)

    def to_dict(self) -> dict[str, Any]:
        return self.row.to_dict()

    def to_record(self) -> dict[str, str]:
        return self.row.to_record()


def _persist_market_state(
    *,
    project_root: Path,
    output_dir: Path,
    as_of_date: date,
    workflow: str,
    portfolio_path: str | None = None,
    daily_summary: DailyResearchSummary | None = None,
    portfolio_review: PortfolioReviewReport | None = None,
    intraday_review: IntradayPortfolioReviewReport | None = None,
) -> PersistedMarketStateArtifacts:
    """Update the live market state snapshot and write compact change outputs."""

    state_store = MarketStateStore(project_root)
    update_result = state_store.update(
        as_of_date=as_of_date,
        workflow=workflow,
        daily_summary=daily_summary,
        portfolio_review=portfolio_review,
        intraday_review=intraday_review,
        portfolio_path=portfolio_path,
    )
    change_json_path = write_market_state_change_set(
        update_result.change_set,
        output_dir / "market_state_changes.json",
    )
    change_text_path = write_market_state_change_summary(
        update_result.change_set,
        output_dir / "market_state_changes.txt",
    )
    outputs = {
        "current_market_state_snapshot": str(update_result.current_path),
        "market_state_changes_json": str(change_json_path),
        "market_state_changes_text": str(change_text_path),
    }
    if update_result.previous_path.exists():
        outputs["previous_market_state_snapshot"] = str(update_result.previous_path)
    return PersistedMarketStateArtifacts(
        update_result=update_result,
        output_paths=outputs,
    )


def _add_live_market_service_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_dir_help: str,
) -> None:
    parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    parser.add_argument(
        "--websocket-url",
        required=True,
        help="Websocket endpoint used for live market updates.",
    )
    parser.add_argument(
        "--subscription-message",
        action="append",
        default=None,
        type=_parse_json_message,
        help="Optional JSON websocket subscription message. Repeat to send multiple subscriptions.",
    )
    parser.add_argument(
        "--transport-name",
        default="json-websocket",
        help="Label for the live websocket transport.",
    )
    parser.add_argument(
        "--portfolio-file",
        type=Path,
        default=None,
        help="Optional CSV or JSON portfolio snapshot to include in the live cycle.",
    )
    parser.add_argument(
        "--include-intraday-review",
        action="store_true",
        help="Include intraday portfolio review cycles when a portfolio file is provided.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Trading session date used for the live cycle.",
    )
    parser.add_argument(
        "--preset-names",
        default=None,
        help="Comma-separated preset names to evaluate for buy candidates.",
    )
    parser.add_argument(
        "--comparison-results",
        type=Path,
        default=None,
        help="Optional ranked preset output from compare-strategies.",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Current account equity used for position sizing. Defaults to config starting_cash.",
    )
    parser.add_argument(
        "--current-drawdown",
        type=float,
        default=0.0,
        help="Current portfolio drawdown as a decimal fraction for drawdown-aware risk reduction.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Lookback window used for universe liquidity screening.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=15,
        help="Intraday aggregate interval in minutes when intraday review is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=output_dir_help,
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Base supervisor sleep interval between live loop iterations.",
    )
    parser.add_argument(
        "--flush-interval-seconds",
        type=int,
        default=5,
        help="Streaming buffer flush interval in seconds.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=30.0,
        help="Mark the service stale when no messages arrive within this many seconds.",
    )
    parser.add_argument(
        "--receive-timeout-seconds",
        type=float,
        default=0.25,
        help="Per-read websocket timeout for the concrete transport.",
    )
    parser.add_argument(
        "--max-messages-per-poll",
        type=int,
        default=100,
        help="Maximum websocket messages to process in one supervisor iteration.",
    )
    parser.add_argument(
        "--initial-reconnect-delay-seconds",
        type=float,
        default=1.0,
        help="Initial reconnect backoff in seconds.",
    )
    parser.add_argument(
        "--max-reconnect-delay-seconds",
        type=float,
        default=30.0,
        help="Maximum reconnect backoff in seconds.",
    )
    parser.add_argument(
        "--reconnect-backoff-multiplier",
        type=float,
        default=2.0,
        help="Reconnect backoff multiplier.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional maximum supervisor iterations before exiting.",
    )
    _add_notification_arguments(parser)


def _add_local_paper_validation_runtime_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Optional isolated root for the local paper-validation profile.",
    )
    parser.add_argument(
        "--preserve-existing-safety-state",
        action="store_true",
        help="Do not reset the validation safety state to the forced paper profile on startup.",
    )
    parser.add_argument(
        "--internal-api-host",
        default="127.0.0.1",
        help="Bind host for the local paper-validation internal API.",
    )
    parser.add_argument(
        "--internal-api-port",
        type=int,
        default=8765,
        help="Bind port for the local paper-validation internal API.",
    )
    parser.add_argument(
        "--control-api-host",
        default="127.0.0.1",
        help="Bind host for the local paper-validation operator control API.",
    )
    parser.add_argument(
        "--control-api-port",
        type=int,
        default=8766,
        help="Bind port for the local paper-validation operator control API.",
    )
    parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Bind host for the local paper-validation dashboard.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8780,
        help="Bind port for the local paper-validation dashboard.",
    )
    parser.add_argument(
        "--dashboard-refresh-seconds",
        type=int,
        default=5,
        help="Dashboard polling cadence in seconds.",
    )
    parser.add_argument(
        "--active-log-filename",
        default="local_paper_validation.log",
        help="Active runtime log filename under the profile logs directory.",
    )
    parser.add_argument(
        "--log-rotate-max-bytes",
        type=int,
        default=250_000,
        help="Rotate the active paper-validation log after this many bytes.",
    )
    parser.add_argument(
        "--log-rotate-backup-count",
        type=int,
        default=5,
        help="Maximum archived paper-validation log files to keep.",
    )


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
        help="Build profile-based candidate universes from a broad master universe.",
    )
    build_universe_parser.add_argument(
        "candidate_path",
        type=Path,
        nargs="?",
        help="Optional legacy screen-only mode: path to a text or CSV file containing candidate symbols.",
    )
    build_universe_parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        default=None,
        help="Configured universe profile to build. Repeat to build more than one profile. Defaults to all configured profiles.",
    )
    build_universe_parser.add_argument(
        "--master-input",
        type=Path,
        default=None,
        help="Optional existing master universe CSV to load instead of fetching provider reference data.",
    )
    build_universe_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Date used as the right edge of the liquidity enrichment window.",
    )
    build_universe_parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional override for the number of recent daily bars used when computing liquidity metrics.",
    )
    build_universe_parser.add_argument(
        "--format",
        choices=("text", "yaml", "json"),
        default="yaml",
        help="Summary output format. Use text only with the legacy screen-only mode.",
    )
    build_universe_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local reference and daily-bar caches.",
    )
    build_universe_parser.set_defaults(handler=_handle_build_universe)

    init_portfolio_parser = subparsers.add_parser(
        "init-portfolio",
        help="Initialize an empty CSV or JSON portfolio snapshot.",
    )
    init_portfolio_parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the portfolio snapshot file to create.",
    )
    init_portfolio_parser.add_argument(
        "--snapshot-format",
        choices=("csv", "json"),
        default=None,
        help="Optional portfolio snapshot format override. Defaults to the file extension.",
    )
    init_portfolio_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    init_portfolio_parser.set_defaults(handler=_handle_init_portfolio)

    upsert_position_parser = subparsers.add_parser(
        "upsert-position",
        help="Append or update one current holding in a CSV or JSON portfolio snapshot.",
    )
    upsert_position_parser.add_argument(
        "portfolio_path",
        type=Path,
        help="Path to the portfolio snapshot file to update.",
    )
    upsert_position_parser.add_argument(
        "symbol",
        help="Ticker symbol for the holding to append or update.",
    )
    upsert_position_parser.add_argument(
        "--quantity",
        type=int,
        required=True,
        help="Current share quantity for the holding.",
    )
    upsert_position_parser.add_argument(
        "--average-entry-price",
        type=float,
        required=True,
        help="Average entry price for the holding.",
    )
    upsert_position_parser.add_argument(
        "--current-stop",
        type=float,
        default=None,
        help="Optional current stop level for the holding.",
    )
    upsert_position_parser.add_argument(
        "--preset-name",
        default=None,
        help="Optional preset name associated with the position.",
    )
    upsert_position_parser.add_argument(
        "--source",
        default=None,
        help="Optional source label for the position snapshot.",
    )
    upsert_position_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON object string to store as position metadata.",
    )
    upsert_position_parser.add_argument(
        "--decision-id",
        default=None,
        help="Optional linked decision id for execution logging.",
    )
    upsert_position_parser.add_argument(
        "--trade-id",
        default=None,
        help="Optional stable trade id override for execution logging.",
    )
    upsert_position_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Execution date used for feedback logging and entry-date defaults.",
    )
    upsert_position_parser.add_argument(
        "--snapshot-format",
        choices=("csv", "json"),
        default=None,
        help="Optional portfolio snapshot format override. Defaults to the file extension.",
    )
    upsert_position_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    upsert_position_parser.set_defaults(handler=_handle_upsert_position)

    update_stop_parser = subparsers.add_parser(
        "update-stop",
        help="Update the current stop for an existing holding in a portfolio snapshot.",
    )
    update_stop_parser.add_argument(
        "portfolio_path",
        type=Path,
        help="Path to the portfolio snapshot file to update.",
    )
    update_stop_parser.add_argument(
        "symbol",
        help="Ticker symbol for the holding to update.",
    )
    update_stop_parser.add_argument(
        "--current-stop",
        type=float,
        required=True,
        help="Updated stop level for the holding.",
    )
    update_stop_parser.add_argument(
        "--decision-id",
        default=None,
        help="Optional linked decision id for stop-update logging.",
    )
    update_stop_parser.add_argument(
        "--trade-id",
        default=None,
        help="Optional stable trade id override for stop-update logging.",
    )
    update_stop_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Execution date used for stop-update logging.",
    )
    update_stop_parser.add_argument(
        "--snapshot-format",
        choices=("csv", "json"),
        default=None,
        help="Optional portfolio snapshot format override. Defaults to the file extension.",
    )
    update_stop_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    update_stop_parser.set_defaults(handler=_handle_update_stop)

    remove_position_parser = subparsers.add_parser(
        "remove-position",
        help="Remove an existing holding from a CSV or JSON portfolio snapshot.",
    )
    remove_position_parser.add_argument(
        "portfolio_path",
        type=Path,
        help="Path to the portfolio snapshot file to update.",
    )
    remove_position_parser.add_argument(
        "symbol",
        help="Ticker symbol for the holding to remove.",
    )
    remove_position_parser.add_argument(
        "--decision-id",
        default=None,
        help="Optional linked decision id for close logging.",
    )
    remove_position_parser.add_argument(
        "--trade-id",
        default=None,
        help="Optional stable trade id override for close logging.",
    )
    remove_position_parser.add_argument(
        "--close-price",
        type=float,
        default=None,
        help="Optional close price used for realized-return logging.",
    )
    remove_position_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Execution date used for close logging.",
    )
    remove_position_parser.add_argument(
        "--snapshot-format",
        choices=("csv", "json"),
        default=None,
        help="Optional portfolio snapshot format override. Defaults to the file extension.",
    )
    remove_position_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    remove_position_parser.set_defaults(handler=_handle_remove_position)

    review_portfolio_parser = subparsers.add_parser(
        "review-portfolio",
        help="Review current holdings and suggest simple stop/exit management actions.",
    )
    review_portfolio_parser.add_argument(
        "--portfolio-file",
        type=Path,
        required=True,
        help="CSV or JSON portfolio snapshot to review.",
    )
    review_portfolio_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge of the portfolio review window.",
    )
    review_portfolio_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter review.",
    )
    review_portfolio_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for portfolio review CSV and JSON outputs.",
    )
    review_portfolio_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    review_portfolio_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    review_portfolio_parser.set_defaults(handler=_handle_review_portfolio)

    review_portfolio_intraday_parser = subparsers.add_parser(
        "review-portfolio-intraday",
        help="Review current holdings with intraday bars for scheduled sell monitoring.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--portfolio-file",
        type=Path,
        required=True,
        help="CSV or JSON portfolio snapshot to review.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Trading session date to review.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--interval-minutes",
        type=int,
        default=15,
        help="Intraday aggregate interval in minutes.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for intraday relative-strength context.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for intraday review CSV, JSON, and brief outputs.",
    )
    review_portfolio_intraday_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    review_portfolio_intraday_parser.set_defaults(handler=_handle_review_portfolio_intraday)

    monitor_market_parser = subparsers.add_parser(
        "monitor-market",
        help="Combine new-entry scans and portfolio-management alerts into one scheduled summary.",
    )
    monitor_market_parser.add_argument(
        "candidate_path",
        type=Path,
        help="Path to a text or CSV file containing candidate symbols.",
    )
    monitor_market_parser.add_argument(
        "--portfolio-file",
        type=Path,
        default=None,
        help="Optional CSV or JSON portfolio snapshot to include in the monitor run.",
    )
    monitor_market_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge of the monitoring window.",
    )
    monitor_market_parser.add_argument(
        "--preset-names",
        default=None,
        help="Comma-separated preset names to evaluate for buy candidates.",
    )
    monitor_market_parser.add_argument(
        "--comparison-results",
        type=Path,
        default=None,
        help="Optional ranked preset output from compare-strategies. When provided, the top preset is included automatically.",
    )
    monitor_market_parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Current account equity used for position sizing. Defaults to config starting_cash.",
    )
    monitor_market_parser.add_argument(
        "--current-drawdown",
        type=float,
        default=0.0,
        help="Current portfolio drawdown as a decimal fraction for drawdown-aware risk reduction.",
    )
    monitor_market_parser.add_argument(
        "--lookback-days",
        type=int,
        default=20,
        help="Lookback window used for universe liquidity screening.",
    )
    monitor_market_parser.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark override for the regime filter.",
    )
    monitor_market_parser.add_argument(
        "--require-relative-volume",
        action="store_true",
        help="Require relative-volume confirmation for breakout entries.",
    )
    monitor_market_parser.add_argument(
        "--disable-regime-filter",
        action="store_true",
        help="Disable the benchmark-based regime filter.",
    )
    monitor_market_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for monitoring JSON, CSV, and text outputs.",
    )
    monitor_market_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    monitor_market_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass local cache and force provider fetches.",
    )
    monitor_market_parser.set_defaults(handler=_handle_monitor_market)

    run_live_market_service_parser = subparsers.add_parser(
        "run-live-market-service",
        help="Run the long-lived websocket-driven live market supervisor.",
    )
    _add_live_market_service_arguments(
        run_live_market_service_parser,
        output_dir_help="Optional directory for live report and state outputs.",
    )
    run_live_market_service_parser.set_defaults(handler=_handle_run_live_market_service)

    run_local_staging_runtime_parser = subparsers.add_parser(
        "run-local-staging-runtime",
        help="Launch the full local staging runtime stack under an isolated runtime root.",
    )
    _add_live_market_service_arguments(
        run_local_staging_runtime_parser,
        output_dir_help=(
            "Optional archive directory for live report outputs. Defaults to "
            "<archive-dir>/live_market/<as-of-date>."
        ),
    )
    run_local_staging_runtime_parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Filesystem root for the isolated local staging runtime.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--hot-state-dir",
        type=Path,
        default=None,
        help="Optional hot-state directory mounted into the isolated runtime root.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Optional active-log directory for the isolated runtime.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Optional archive root for rotated logs and staged live outputs.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--active-log-filename",
        default="local_staging_runtime.log",
        help="Active runtime log filename under the configured logs directory.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--log-rotate-max-bytes",
        type=int,
        default=250_000,
        help="Rotate the active runtime log after this many bytes.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--log-rotate-backup-count",
        type=int,
        default=5,
        help="Maximum archived runtime log files to keep.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--internal-api-host",
        default="127.0.0.1",
        help="Bind host for the local staging internal API.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--internal-api-port",
        type=int,
        default=8765,
        help="Bind port for the local staging internal API.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--control-api-host",
        default="127.0.0.1",
        help="Bind host for the local staging operator control API.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--control-api-port",
        type=int,
        default=8766,
        help="Bind port for the local staging operator control API.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Bind host for the local staging dashboard.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8780,
        help="Bind port for the local staging dashboard.",
    )
    run_local_staging_runtime_parser.add_argument(
        "--dashboard-refresh-seconds",
        type=int,
        default=5,
        help="Dashboard polling cadence in seconds.",
    )
    run_local_staging_runtime_parser.set_defaults(handler=_handle_run_local_staging_runtime)

    run_local_paper_validation_parser = subparsers.add_parser(
        "run-local-paper-validation",
        help="Launch the isolated always-on paper-validation runtime profile.",
    )
    _add_live_market_service_arguments(
        run_local_paper_validation_parser,
        output_dir_help=(
            "Optional archive directory for live report outputs. Defaults to "
            "<validation-root>/archive/live_market/<as-of-date>."
        ),
    )
    _add_local_paper_validation_runtime_arguments(run_local_paper_validation_parser)
    run_local_paper_validation_parser.set_defaults(
        handler=_handle_run_local_paper_validation
    )

    run_local_paper_validation_smoke_parser = subparsers.add_parser(
        "run-local-paper-validation-smoke",
        help="Launch a bounded Day-1 smoke run over the isolated paper-validation profile.",
    )
    _add_live_market_service_arguments(
        run_local_paper_validation_smoke_parser,
        output_dir_help=(
            "Optional archive directory for live report outputs. Defaults to "
            "<validation-root>/archive/live_market/<as-of-date>."
        ),
    )
    _add_local_paper_validation_runtime_arguments(
        run_local_paper_validation_smoke_parser
    )
    run_local_paper_validation_smoke_parser.set_defaults(
        max_iterations=10,
        handler=_handle_run_local_paper_validation_smoke,
    )

    summarize_local_paper_validation_parser = subparsers.add_parser(
        "summarize-local-paper-validation",
        help="Write daily checkpoint, operator review, and checklist artifacts for the paper-validation runtime.",
    )
    summarize_local_paper_validation_parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Optional isolated root for the local paper-validation profile.",
    )
    summarize_local_paper_validation_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge date for the validation review window.",
    )
    summarize_local_paper_validation_parser.add_argument(
        "--window-days",
        type=int,
        default=1,
        help="Number of recent days to include in the validation review window.",
    )
    summarize_local_paper_validation_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the validation summary outputs.",
    )
    summarize_local_paper_validation_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    summarize_local_paper_validation_parser.set_defaults(
        handler=_handle_summarize_local_paper_validation
    )

    serve_internal_api_parser = subparsers.add_parser(
        "serve-internal-api",
        help="Serve the local read-only internal backend API for operational state.",
    )
    serve_internal_api_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local internal API server.",
    )
    serve_internal_api_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for the local internal API server.",
    )
    serve_internal_api_parser.set_defaults(handler=_handle_serve_internal_api)

    serve_operator_control_api_parser = subparsers.add_parser(
        "serve-operator-control-api",
        help="Serve the local write-capable operator control API.",
    )
    serve_operator_control_api_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local operator control API server.",
    )
    serve_operator_control_api_parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Bind port for the local operator control API server.",
    )
    serve_operator_control_api_parser.set_defaults(handler=_handle_serve_operator_control_api)

    serve_operator_dashboard_parser = subparsers.add_parser(
        "serve-operator-dashboard",
        help="Serve the local read-only operator dashboard and internal API.",
    )
    serve_operator_dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local operator dashboard server.",
    )
    serve_operator_dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8780,
        help="Bind port for the local operator dashboard server.",
    )
    serve_operator_dashboard_parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=5,
        help="Polling cadence for the dashboard client.",
    )
    serve_operator_dashboard_parser.set_defaults(handler=_handle_serve_operator_dashboard)

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
        "--portfolio-file",
        type=Path,
        default=None,
        help="Optional CSV or JSON file describing currently open holdings.",
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
    generate_orders_parser.add_argument(
        "--stage-pending-orders",
        action="store_true",
        help="Persist generated buy orders into pending-order state for later capacity-aware runs.",
    )
    generate_orders_parser.set_defaults(handler=_handle_generate_orders)

    submit_broker_orders_parser = subparsers.add_parser(
        "submit-broker-orders",
        help="Submit active staged pending orders through the configured broker adapter.",
    )
    submit_broker_orders_parser.add_argument(
        "--broker",
        default="alpaca",
        help="Broker adapter name.",
    )
    submit_broker_orders_parser.add_argument(
        "--order-id",
        dest="order_ids",
        action="append",
        default=None,
        help="Optional local pending order id to submit. Repeat for multiple orders.",
    )
    submit_broker_orders_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    submit_broker_orders_parser.add_argument(
        "--confirm-live-action",
        action="store_true",
        help="Assert explicit operator intent for live broker submissions when required by policy.",
    )
    _add_notification_arguments(submit_broker_orders_parser)
    submit_broker_orders_parser.set_defaults(handler=_handle_submit_broker_orders)

    sync_broker_orders_parser = subparsers.add_parser(
        "sync-broker-orders",
        help="Reconcile broker order truth back into local pending-order state.",
    )
    sync_broker_orders_parser.add_argument(
        "--broker",
        default="alpaca",
        help="Broker adapter name.",
    )
    sync_broker_orders_parser.add_argument(
        "--order-id",
        dest="order_ids",
        action="append",
        default=None,
        help="Optional local pending order id to sync. Repeat for multiple orders.",
    )
    sync_broker_orders_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    _add_notification_arguments(sync_broker_orders_parser)
    sync_broker_orders_parser.set_defaults(handler=_handle_sync_broker_orders)

    cancel_broker_order_parser = subparsers.add_parser(
        "cancel-broker-order",
        help="Cancel one active broker-linked pending order.",
    )
    cancel_broker_order_parser.add_argument(
        "order_id",
        help="Local pending order id to cancel.",
    )
    cancel_broker_order_parser.add_argument(
        "--broker",
        default="alpaca",
        help="Broker adapter name.",
    )
    cancel_broker_order_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    cancel_broker_order_parser.add_argument(
        "--confirm-live-action",
        action="store_true",
        help="Assert explicit operator intent for live broker cancels when required by policy.",
    )
    _add_notification_arguments(cancel_broker_order_parser)
    cancel_broker_order_parser.set_defaults(handler=_handle_cancel_broker_order)

    replace_broker_order_parser = subparsers.add_parser(
        "replace-broker-order",
        help="Replace one active broker-linked pending order.",
    )
    replace_broker_order_parser.add_argument(
        "order_id",
        help="Local pending order id to replace.",
    )
    replace_broker_order_parser.add_argument(
        "--broker",
        default="alpaca",
        help="Broker adapter name.",
    )
    replace_broker_order_parser.add_argument(
        "--quantity",
        type=int,
        default=None,
        help="Optional replacement quantity.",
    )
    replace_broker_order_parser.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help="Optional replacement limit price.",
    )
    replace_broker_order_parser.add_argument(
        "--stop-price",
        type=float,
        default=None,
        help="Optional replacement stop price.",
    )
    replace_broker_order_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    replace_broker_order_parser.add_argument(
        "--confirm-live-action",
        action="store_true",
        help="Assert explicit operator intent for live broker replacements when required by policy.",
    )
    _add_notification_arguments(replace_broker_order_parser)
    replace_broker_order_parser.set_defaults(handler=_handle_replace_broker_order)

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
            "initial_stop_atr=2.5,trailing_stop_atr=3.0,risk_per_trade=0.01,"
            "require_relative_volume_confirmation=true"
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
        "--portfolio-file",
        type=Path,
        default=None,
        help="Optional CSV or JSON file describing currently open holdings.",
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

    trade_review_parser = subparsers.add_parser(
        "review-trade-analytics",
        help="Summarize persisted trade feedback decisions, executions, and outcomes.",
    )
    trade_review_parser.add_argument(
        "--as-of",
        type=_parse_iso_date,
        default=date.today(),
        help="Right edge of the analytics window.",
    )
    trade_review_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Trailing window in calendar days. Ignored when --all-history is set.",
    )
    trade_review_parser.add_argument(
        "--all-history",
        action="store_true",
        help="Summarize all available feedback records instead of a trailing window.",
    )
    trade_review_parser.add_argument(
        "--feedback-log",
        type=Path,
        default=None,
        help="Optional trade feedback JSONL path override.",
    )
    trade_review_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for analytics JSON and brief outputs.",
    )
    trade_review_parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Console summary output format.",
    )
    trade_review_parser.set_defaults(handler=_handle_review_trade_analytics)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file, json_logs=args.json_logs)

    try:
        return int(args.handler(args))
    except (
        BrokerExecutionError,
        ConfigError,
        DataProviderConfigurationError,
        DataProviderError,
        ManualOrderError,
        ValueError,
    ) as exc:
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
    if args.candidate_path is not None and args.profiles is None and args.master_input is None:
        return _handle_build_universe_legacy(args)

    if args.candidate_path is not None:
        raise ValueError(
            "candidate_path is only supported in legacy screen-only mode. "
            "Use --profile for master-universe builds."
        )
    if args.format == "text":
        raise ValueError("format=text is only supported in legacy screen-only mode.")

    config = load_app_config(config_dir=args.config_dir)
    daily_bar_provider = create_daily_bar_provider(config, env_file=args.env_file)
    reference_provider = (
        None
        if args.master_input is not None
        else create_reference_universe_provider(config, env_file=args.env_file)
    )

    lookback_days = (
        config.universe_builder.master.liquidity_lookback_days
        if args.lookback_days is None
        else int(args.lookback_days)
    )
    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero.")

    profile_names = _resolve_universe_profiles(args.profiles, config)
    if args.master_input is not None:
        degraded_profiles = [
            profile_name
            for profile_name in profile_names
            if _profile_uses_reference_detail_filters(config.universe_builder.profiles[profile_name])
        ]
        if degraded_profiles:
            LOGGER.warning(
                "--master-input disables provider-backed reference detail enrichment. "
                "Profiles %s rely on market-cap/sector/industry filters and will only use "
                "metadata already present in %s.",
                ", ".join(degraded_profiles),
                args.master_input.resolve(),
            )
    profile_label = ", ".join(profile_names)
    LOGGER.info(
        "Build-universe starting for profile(s): %s.",
        profile_label,
    )
    LOGGER.info(
        "Starting master universe fetch%s.",
        f" from {args.master_input.resolve()}" if args.master_input is not None else "",
    )
    try:
        raw_reference_frame = fetch_master_universe(
            master_config=config.universe_builder.master,
            as_of_date=args.as_of,
            reference_provider=reference_provider,
            master_input=args.master_input,
            refresh_cache=args.refresh_cache,
        )
    except DataProviderError as exc:
        raise DataProviderError(f"Master universe fetch failed: {exc}") from exc

    LOGGER.info("Fetched %s tickers into the master reference universe.", len(raw_reference_frame))
    LOGGER.info("Starting metadata enrichment.")
    try:
        master_frame = enrich_universe_metadata(
            raw_reference_frame,
            as_of_date=args.as_of,
            lookback_days=lookback_days,
            daily_bar_provider=daily_bar_provider,
            reference_provider=reference_provider,
            refresh_cache=args.refresh_cache,
        )
    except DataProviderError as exc:
        raise DataProviderError(f"Metadata enrichment failed: {exc}") from exc

    LOGGER.info(
        "Metadata enrichment completed for %s symbols. Applying profile filters.",
        len(master_frame),
    )
    profile_results: dict[str, UniverseProfileBuildResult] = {}
    for profile_name in profile_names:
        LOGGER.info("Applying filters for profile %s.", profile_name)
        try:
            profile_results[profile_name] = apply_universe_filters(
                master_frame,
                profile_name=profile_name,
                profile_config=config.universe_builder.profiles[profile_name],
                reference_provider=reference_provider,
                as_of_date=args.as_of,
                refresh_cache=args.refresh_cache,
            )
        except DataProviderError as exc:
            raise DataProviderError(f"Applying filters failed for profile '{profile_name}': {exc}") from exc

    LOGGER.info("Writing universe outputs.")
    outputs = write_universe_outputs(
        project_root=config.project_root,
        master_config=config.universe_builder.master,
        raw_reference_frame=raw_reference_frame,
        master_frame=master_frame,
        profile_results=profile_results,
    )
    for profile_name in profile_names:
        LOGGER.info(
            "Profile %s output written to %s.",
            profile_name,
            outputs["profiles"][profile_name]["candidate_output_path"],
        )

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "lookback_days": lookback_days,
        "master_input": str(args.master_input.resolve()) if args.master_input is not None else None,
        "profile_names": list(profile_names),
        "master_universe_count": int(len(master_frame)),
        "profiles": {
            profile_name: {
                "count": len(result.symbols),
                "filtered_count": result.summary["filtered_count"],
                "force_include_count": result.summary["force_include_count"],
                "missing_force_include_symbols": result.summary["missing_force_include_symbols"],
                "candidate_output_path": outputs["profiles"][profile_name]["candidate_output_path"],
                "summary_output_path": outputs["profiles"][profile_name]["summary_output_path"],
            }
            for profile_name, result in profile_results.items()
        },
        "outputs": outputs,
    }
    LOGGER.info(
        "Build-universe completed. Master universe: %s. Raw reference: %s. Profiles: %s.",
        outputs["master_universe_output_path"],
        outputs["raw_reference_output_path"],
        ", ".join(
            f"{name}={profile_results[name].summary['output_count']}"
            for name in profile_names
        ),
    )
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_build_universe_legacy(args: argparse.Namespace) -> int:
    if args.candidate_path is None:
        raise ValueError("candidate_path is required for legacy screen-only mode.")

    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    builder = UniverseBuilder(provider, config.strategy.universe)
    resolved_lookback_days = (
        config.universe_builder.master.liquidity_lookback_days
        if args.lookback_days is None
        else int(args.lookback_days)
    )
    members = builder.screen_candidates(
        args.candidate_path,
        as_of_date=args.as_of,
        lookback_days=resolved_lookback_days,
        refresh_cache=args.refresh_cache,
    )

    if args.format == "text":
        for member in members:
            print(member.symbol)
        return 0

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "lookback_days": resolved_lookback_days,
        "count": len(members),
        "symbols": [member.symbol for member in members],
        "members": [member.to_dict() for member in members],
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_init_portfolio(args: argparse.Namespace) -> int:
    written_path = initialize_portfolio_snapshot(
        args.output_path,
        output_format=args.snapshot_format,
    )
    payload = {
        "portfolio_path": str(written_path),
        "snapshot_format": written_path.suffix.lower().lstrip("."),
        "position_count": 0,
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_upsert_position(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    existing_positions = (
        load_existing_positions(args.portfolio_path.resolve())
        if args.portfolio_path.resolve().exists()
        else []
    )
    existing_position = next(
        (
            position
            for position in existing_positions
            if position.symbol == args.symbol.strip().upper()
        ),
        None,
    )
    explicit_metadata = _parse_metadata_json(args.metadata_json)
    merged_metadata = (
        dict(existing_position.metadata)
        if existing_position is not None
        else {}
    )
    merged_metadata.update(explicit_metadata)
    execution_decision_id = _resolve_execution_decision_id(
        explicit_decision_id=args.decision_id,
        metadata=merged_metadata,
        existing_position=existing_position,
    )
    entry_date = _position_entry_date(existing_position) if existing_position is not None else None
    if entry_date is None:
        raw_entry_date = merged_metadata.get("entry_date")
        if isinstance(raw_entry_date, str) and raw_entry_date.strip():
            try:
                entry_date = date.fromisoformat(raw_entry_date.strip())
            except ValueError:
                entry_date = None
    if entry_date is None:
        entry_date = args.as_of
    trade_id = _resolve_execution_trade_id(
        explicit_trade_id=args.trade_id,
        metadata=merged_metadata,
        existing_position=existing_position,
        symbol=args.symbol.strip().upper(),
        entry_date=entry_date,
        average_entry_price=float(args.average_entry_price),
        decision_id=execution_decision_id,
    )
    merged_metadata["trade_id"] = trade_id
    if execution_decision_id is not None:
        merged_metadata["linked_decision_id"] = execution_decision_id
    merged_metadata.setdefault("entry_date", entry_date.isoformat())
    position = ExistingPosition(
        symbol=args.symbol.strip().upper(),
        shares=int(args.quantity),
        average_entry_price=float(args.average_entry_price),
        current_stop=(None if args.current_stop is None else float(args.current_stop)),
        preset_name=(args.preset_name.strip() if isinstance(args.preset_name, str) and args.preset_name.strip() else None),
        source=(args.source.strip() if isinstance(args.source, str) and args.source.strip() else None),
        metadata=merged_metadata,
    )
    written_path = upsert_existing_position_snapshot(
        args.portfolio_path,
        position,
        output_format=args.snapshot_format,
    )
    current_positions = load_existing_positions(written_path)
    event = TradeFeedbackEvent(
        event_type="executed",
        workflow="upsert-position",
        symbol=position.symbol,
        as_of_date=args.as_of,
        timestamp_utc=_trade_feedback_timestamp_utc(),
        decision_id=execution_decision_id,
        trade_id=trade_id,
        linked_decision_id=execution_decision_id,
        preset_name=position.preset_name,
        strategy_name=_extract_strategy_name_from_metadata(position.metadata),
        suggested_action="BUY",
        quantity=position.shares,
        entry_price_hint=position.average_entry_price,
        stop_price=position.current_stop,
        notional_value=position.average_entry_price * position.shares,
        metadata={"position_source": position.source},
    )
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[event],
    )
    payload = {
        "portfolio_path": str(written_path),
        "snapshot_format": written_path.suffix.lower().lstrip("."),
        "updated_symbol": position.symbol,
        "position_count": len(current_positions),
        "current_position_symbols": [current_position.symbol for current_position in current_positions],
    }
    if trade_feedback_log_path is not None:
        payload["trade_decision_log"] = str(trade_feedback_log_path)
        payload["trade_feedback_log"] = str(trade_feedback_log_path)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_update_stop(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    existing_positions = load_existing_positions(args.portfolio_path.resolve())
    normalized_symbol = args.symbol.strip().upper()
    existing_position = next(
        (position for position in existing_positions if position.symbol == normalized_symbol),
        None,
    )
    if existing_position is None:
        raise PortfolioInputError(
            f"Symbol '{normalized_symbol}' does not exist in portfolio snapshot."
        )
    written_path = update_existing_position_stop_snapshot(
        args.portfolio_path,
        args.symbol,
        float(args.current_stop),
        output_format=args.snapshot_format,
    )
    current_positions = load_existing_positions(written_path)
    payload = {
        "portfolio_path": str(written_path),
        "snapshot_format": written_path.suffix.lower().lstrip("."),
        "updated_symbol": args.symbol.strip().upper(),
        "position_count": len(current_positions),
        "current_position_symbols": [current_position.symbol for current_position in current_positions],
    }
    execution_decision_id = _resolve_execution_decision_id(
        explicit_decision_id=args.decision_id,
        metadata=existing_position.metadata,
        existing_position=existing_position,
    )
    trade_id = _resolve_execution_trade_id(
        explicit_trade_id=args.trade_id,
        metadata=existing_position.metadata,
        existing_position=existing_position,
        symbol=existing_position.symbol,
        entry_date=_position_entry_date(existing_position),
        average_entry_price=existing_position.average_entry_price,
        decision_id=execution_decision_id,
    )
    event_type = (
        "stop_raised"
        if existing_position.current_stop is not None
        and float(args.current_stop) > existing_position.current_stop
        else "stop_updated"
    )
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[
            TradeFeedbackEvent(
                event_type=event_type,
                workflow="update-stop",
                symbol=existing_position.symbol,
                as_of_date=args.as_of,
                timestamp_utc=_trade_feedback_timestamp_utc(),
                decision_id=execution_decision_id,
                trade_id=trade_id,
                linked_decision_id=execution_decision_id,
                preset_name=existing_position.preset_name,
                strategy_name=_extract_strategy_name_from_metadata(existing_position.metadata),
                suggested_action="RAISE STOP",
                quantity=existing_position.shares,
                entry_price_hint=existing_position.average_entry_price,
                stop_price=float(args.current_stop),
                notional_value=existing_position.average_entry_price * existing_position.shares,
                metadata={
                    "previous_stop": existing_position.current_stop,
                    "position_source": existing_position.source,
                },
            )
        ],
    )
    if trade_feedback_log_path is not None:
        payload["trade_decision_log"] = str(trade_feedback_log_path)
        payload["trade_feedback_log"] = str(trade_feedback_log_path)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_remove_position(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    existing_positions = load_existing_positions(args.portfolio_path.resolve())
    normalized_symbol = args.symbol.strip().upper()
    existing_position = next(
        (position for position in existing_positions if position.symbol == normalized_symbol),
        None,
    )
    if existing_position is None:
        raise PortfolioInputError(
            f"Symbol '{normalized_symbol}' does not exist in portfolio snapshot."
        )
    written_path = remove_existing_position_snapshot(
        args.portfolio_path,
        args.symbol,
        output_format=args.snapshot_format,
    )
    current_positions = load_existing_positions(written_path)
    payload = {
        "portfolio_path": str(written_path),
        "snapshot_format": written_path.suffix.lower().lstrip("."),
        "removed_symbol": args.symbol.strip().upper(),
        "position_count": len(current_positions),
        "current_position_symbols": [current_position.symbol for current_position in current_positions],
    }
    execution_decision_id = _resolve_execution_decision_id(
        explicit_decision_id=args.decision_id,
        metadata=existing_position.metadata,
        existing_position=existing_position,
    )
    trade_id = _resolve_execution_trade_id(
        explicit_trade_id=args.trade_id,
        metadata=existing_position.metadata,
        existing_position=existing_position,
        symbol=existing_position.symbol,
        entry_date=_position_entry_date(existing_position),
        average_entry_price=existing_position.average_entry_price,
        decision_id=execution_decision_id,
    )
    close_price = None if args.close_price is None else float(args.close_price)
    close_metadata: dict[str, Any] = {
        "position_source": existing_position.source,
    }
    if close_price is not None:
        close_metadata["close_price"] = close_price
        close_metadata["realized_return_pct"] = (
            close_price / existing_position.average_entry_price
        ) - 1.0
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[
            TradeFeedbackEvent(
                event_type="closed",
                workflow="remove-position",
                symbol=existing_position.symbol,
                as_of_date=args.as_of,
                timestamp_utc=_trade_feedback_timestamp_utc(),
                decision_id=execution_decision_id,
                trade_id=trade_id,
                linked_decision_id=execution_decision_id,
                preset_name=existing_position.preset_name,
                strategy_name=_extract_strategy_name_from_metadata(existing_position.metadata),
                suggested_action="SELL",
                quantity=existing_position.shares,
                entry_price_hint=existing_position.average_entry_price,
                stop_price=existing_position.current_stop,
                notional_value=(
                    close_price * existing_position.shares
                    if close_price is not None
                    else None
                ),
                metadata=close_metadata,
            )
        ],
    )
    if trade_feedback_log_path is not None:
        payload["trade_decision_log"] = str(trade_feedback_log_path)
        payload["trade_feedback_log"] = str(trade_feedback_log_path)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_review_portfolio(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    request = LiveMarketCycleRequest(
        workflow="review-portfolio",
        as_of_date=args.as_of,
        portfolio_path=str(args.portfolio_file.resolve()),
        include_portfolio_review=True,
        portfolio_review_required=True,
    )
    ingestion = PollingIngestionAdapter(
        provider_name=_configured_provider_name(config),
        portfolio_path=str(args.portfolio_file.resolve()),
        load_current_positions=lambda: _load_current_positions(args.portfolio_file),
        create_provider=lambda: create_daily_bar_provider(config, env_file=args.env_file),
        run_portfolio_review=lambda provider, current_positions: _run_portfolio_review_workflow(
            args,
            config=config,
            provider=cast(Optional[DailyBarProvider], provider),
            current_positions=current_positions,
        ),
        benchmark_symbol=args.benchmark_symbol,
        refresh_cache=getattr(args, "refresh_cache", False),
    ).build_cycle_ingestion(request)
    current_positions = list(ingestion.current_positions)
    output_dir = (
        args.output_dir or _default_daily_output_dir(config.project_root, args.as_of)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle = LiveMarketRunner(
        persist_market_state=lambda as_of_date, workflow, daily_summary, portfolio_review, intraday_review, portfolio_path: _persist_market_state(
            project_root=config.project_root,
            output_dir=output_dir,
            as_of_date=as_of_date,
            workflow=workflow,
            portfolio_path=portfolio_path,
            portfolio_review=(
                _strip_internal_portfolio_review_report(portfolio_review)
                if portfolio_review is not None
                else None
            ),
        ),
    ).run_cycle(request, ingestion=ingestion)
    report = cast(PortfolioReviewReport, cycle.portfolio_review)
    report = replace(
        report,
        rows=tuple(
            replace(
                row,
                metadata={
                    **dict(row.metadata),
                    "decision_id": _portfolio_review_decision_id(
                        row,
                        workflow="review-portfolio",
                    ),
                },
            )
            for row in report.rows
        ),
    )
    trade_feedback_events: list[TradeFeedbackEvent] = [
        _portfolio_review_decision_event(
            row,
            workflow="review-portfolio",
            timestamp_utc=report.generated_at_utc,
        )
        for row in report.rows
    ]
    trade_feedback_events.extend(
        event
        for row in report.rows
        for event in [
            _portfolio_review_outcome_event(
                row,
                workflow="review-portfolio",
                timestamp_utc=report.generated_at_utc,
            )
        ]
        if event is not None
    )
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=trade_feedback_events,
    )
    sanitized_report = _strip_internal_portfolio_review_report(report)
    review_json_path = write_portfolio_review_report(
        sanitized_report,
        output_dir / "portfolio_review.json",
    )
    review_csv_path = write_portfolio_review_report(
        sanitized_report,
        output_dir / "portfolio_review.csv",
    )
    state_artifacts = cast(PersistedMarketStateArtifacts, cycle.state_artifacts)
    position_trajectory_journal_path = default_position_trajectory_journal_path(
        config.project_root,
    ).resolve()
    report_payload = sanitized_report.to_dict()
    action_counts = {
        f"{action.lower().replace(' ', '_')}_count": report_payload[
            f"{action.lower().replace(' ', '_')}_count"
        ]
        for action in PORTFOLIO_REVIEW_ACTIONS
    }
    payload = {
        "portfolio_path": str(args.portfolio_file.resolve()),
        "position_count": len(current_positions),
        "symbols_reviewed": report_payload["reviewed_symbols"],
        "state_change_count": cycle.state_change_count,
        "market_state_baseline_established": state_artifacts.baseline_established,
        **action_counts,
        "outputs": {
            "portfolio_review_json": str(review_json_path),
            "portfolio_review_csv": str(review_csv_path),
            **cycle.paths_written,
        },
    }
    if position_trajectory_journal_path.exists():
        payload["outputs"]["position_trajectory_journal"] = str(
            position_trajectory_journal_path
        )
    if trade_feedback_log_path is not None:
        payload["outputs"]["trade_decision_log"] = str(trade_feedback_log_path)
        payload["outputs"]["trade_feedback_log"] = str(trade_feedback_log_path)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_review_portfolio_intraday(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    request = LiveMarketCycleRequest(
        workflow="review-portfolio-intraday",
        as_of_date=args.as_of,
        portfolio_path=str(args.portfolio_file.resolve()),
        include_intraday_review=True,
        intraday_review_required=True,
    )
    ingestion = PollingIngestionAdapter(
        provider_name=_configured_provider_name(config),
        portfolio_path=str(args.portfolio_file.resolve()),
        load_current_positions=lambda: _load_current_positions(args.portfolio_file),
        create_provider=lambda: create_daily_bar_provider(config, env_file=args.env_file),
        run_intraday_review=lambda provider, current_positions: _run_portfolio_review_intraday_workflow(
            args,
            config=config,
            provider=cast(Optional[DailyBarProvider], provider),
            current_positions=current_positions,
        ),
        benchmark_symbol=args.benchmark_symbol,
        refresh_cache=getattr(args, "refresh_cache", False),
        interval_minutes=args.interval_minutes,
    ).build_cycle_ingestion(request)
    current_positions = list(ingestion.current_positions)
    output_dir = (
        args.output_dir
        or _default_intraday_portfolio_output_dir(config.project_root, args.as_of)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cycle = LiveMarketRunner(
            persist_market_state=lambda as_of_date, workflow, daily_summary, portfolio_review, intraday_review, portfolio_path: _persist_market_state(
                project_root=config.project_root,
                output_dir=output_dir,
                as_of_date=as_of_date,
                workflow=workflow,
                portfolio_path=portfolio_path,
                intraday_review=intraday_review,
            ),
        ).run_cycle(request, ingestion=ingestion)
    except NoIntradayDataError as exc:
        payload = {
            "status": "skipped",
            "reason": "no_intraday_data",
            "message": str(exc),
            "portfolio_path": str(args.portfolio_file.resolve()),
            "interval_minutes": args.interval_minutes,
            "position_count": len(current_positions),
            "symbols_reviewed": [],
            "outputs": {},
        }
        _print_structured(payload, output_format=args.format)
        return 0
    report = cast(IntradayPortfolioReviewReport, cycle.intraday_review)
    report = replace(
        report,
        rows=tuple(
            replace(
                row,
                metadata={
                    **dict(row.metadata),
                    "decision_id": _portfolio_review_decision_id(
                        row,
                        workflow="review-portfolio-intraday",
                    ),
                },
            )
            for row in report.rows
        ),
    )
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[
            _portfolio_review_decision_event(
                row,
                workflow="review-portfolio-intraday",
                timestamp_utc=report.generated_at_utc,
            )
            for row in report.rows
        ],
    )
    review_json_path = write_intraday_portfolio_review_report(
        report,
        output_dir / "portfolio_review_intraday.json",
    )
    review_csv_path = write_intraday_portfolio_review_report(
        report,
        output_dir / "portfolio_review_intraday.csv",
    )
    review_brief_path = write_intraday_portfolio_review_brief(
        report,
        output_dir / "portfolio_review_intraday_brief.txt",
    )
    state_artifacts = cast(PersistedMarketStateArtifacts, cycle.state_artifacts)
    intraday_state_journal_path = default_intraday_state_journal_path(
        config.project_root,
        args.as_of,
    ).resolve()
    report_payload = report.to_dict()
    action_counts = {
        f"{action.lower().replace(' ', '_')}_count": report_payload[
            f"{action.lower().replace(' ', '_')}_count"
        ]
        for action in PORTFOLIO_REVIEW_ACTIONS
    }
    payload = {
        "portfolio_path": str(args.portfolio_file.resolve()),
        "interval_minutes": args.interval_minutes,
        "position_count": len(current_positions),
        "symbols_reviewed": report_payload["reviewed_symbols"],
        "state_change_count": cycle.state_change_count,
        "market_state_baseline_established": state_artifacts.baseline_established,
        **action_counts,
        "outputs": {
            "portfolio_review_intraday_json": str(review_json_path),
            "portfolio_review_intraday_csv": str(review_csv_path),
            "portfolio_review_intraday_brief": str(review_brief_path),
            **cycle.paths_written,
        },
    }
    if intraday_state_journal_path.exists():
        payload["outputs"]["intraday_state_journal"] = str(intraday_state_journal_path)
    if trade_feedback_log_path is not None:
        payload["outputs"]["trade_decision_log"] = str(trade_feedback_log_path)
        payload["outputs"]["trade_feedback_log"] = str(trade_feedback_log_path)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_monitor_market(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    request = LiveMarketCycleRequest(
        workflow="monitor-market",
        as_of_date=args.as_of,
        portfolio_path=(
            str(args.portfolio_file.resolve()) if args.portfolio_file else None
        ),
        include_daily_summary=True,
        include_portfolio_review=args.portfolio_file is not None,
        daily_summary_required=True,
        portfolio_review_required=False,
    )
    ingestion = PollingIngestionAdapter(
        provider_name=_configured_provider_name(config),
        portfolio_path=(
            str(args.portfolio_file.resolve()) if args.portfolio_file else None
        ),
        load_current_positions=lambda: _load_current_positions(args.portfolio_file),
        create_provider=lambda: create_daily_bar_provider(config, env_file=args.env_file),
        run_daily_summary=lambda provider, current_positions: _run_daily_summary_workflow(
            args,
            config=config,
            provider=cast(DailyBarProvider, provider),
            current_positions=current_positions,
        ),
        run_portfolio_review=(
            lambda provider, current_positions: _run_portfolio_review_workflow(
                args,
                config=config,
                provider=cast(Optional[DailyBarProvider], provider),
                current_positions=current_positions,
            )
            if args.portfolio_file is not None
            else None
        ),
        candidate_path=str(args.candidate_path),
        benchmark_symbol=args.benchmark_symbol,
        refresh_cache=args.refresh_cache,
    ).build_cycle_ingestion(request)
    current_positions = list(ingestion.current_positions)
    output_dir = (
        args.output_dir
        or (config.project_root / "data" / "processed" / "monitor_market" / args.as_of.isoformat())
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle = LiveMarketRunner(
        build_monitor_report=lambda as_of_date, daily_summary, portfolio_review, intraday_review, portfolio_path: build_market_monitor_report(
            as_of_date=as_of_date,
            daily_summary=daily_summary,
            portfolio_review=(
                _strip_internal_portfolio_review_report(portfolio_review)
                if portfolio_review is not None
                else None
            ),
            portfolio_path=portfolio_path,
        ),
        persist_market_state=lambda as_of_date, workflow, daily_summary, portfolio_review, intraday_review, portfolio_path: _persist_market_state(
            project_root=config.project_root,
            output_dir=output_dir,
            as_of_date=as_of_date,
            workflow=workflow,
            portfolio_path=portfolio_path,
            daily_summary=daily_summary,
            portfolio_review=(
                _strip_internal_portfolio_review_report(portfolio_review)
                if portfolio_review is not None
                else None
            ),
        ),
    ).run_cycle(request, ingestion=ingestion)
    summary_result = cast(Mapping[str, object], cycle.daily_summary_result)
    summary = cast(DailyResearchSummary, cycle.daily_summary)
    monitor_report = cast(MarketMonitorReport, cycle.monitor_report)
    state_artifacts = cast(PersistedMarketStateArtifacts, cycle.state_artifacts)

    alert_json_path = write_market_monitor_report(monitor_report, output_dir / "market_monitor.json")
    alert_csv_path = write_market_monitor_report(monitor_report, output_dir / "market_monitor.csv")
    alert_text_path = write_market_monitor_text_summary(
        monitor_report,
        output_dir / "market_monitor.txt",
    )
    alert_brief_path = write_market_monitor_brief(
        monitor_report,
        output_dir / "market_monitor_brief.txt",
    )
    report_payload = monitor_report.to_dict()
    summary_payload = summary.to_dict()
    candidate_score_journal_path = summary_result.get("candidate_score_journal_path")
    category_counts = report_payload["category_counts"]
    flat_category_counts = {
        market_monitor_flat_count_key(category): count
        for category, count in category_counts.items()
    }
    outputs = {
        "market_monitor_json": str(alert_json_path),
        "market_monitor_csv": str(alert_csv_path),
        "market_monitor_text": str(alert_text_path),
        "market_monitor_brief": str(alert_brief_path),
    }
    if candidate_score_journal_path is not None:
        outputs["candidate_score_journal"] = str(candidate_score_journal_path)

    payload = {
        "as_of_date": args.as_of.isoformat(),
        "portfolio_file": str(args.portfolio_file.resolve()) if args.portfolio_file else None,
        "preset_names": list(report_payload["preset_names"]),
        "market_breadth_pct_above_200ma": (
            summary.market_context.market_breadth_pct_above_200ma
            if summary.market_context is not None
            else None
        ),
        "market_breadth_pct_above_50ma": (
            summary.market_context.market_breadth_pct_above_50ma
            if summary.market_context is not None
            else None
        ),
        "market_breadth_state": (
            summary.market_context.market_breadth_state
            if summary.market_context is not None
            else None
        ),
        "vix_close": (
            summary.market_context.vix_close
            if summary.market_context is not None
            else None
        ),
        "volatility_regime_state": (
            summary.market_context.volatility_regime_state
            if summary.market_context is not None
            else None
        ),
        "volatility_regime_risk_off": (
            summary.market_context.volatility_regime_risk_off
            if summary.market_context is not None
            else None
        ),
        "universe_count": summary_payload["universe_count"],
        "approved_count": summary_payload["approved_count"],
        "rejected_count": summary_payload["rejected_count"],
        "alert_count": report_payload["alert_count"],
        "state_change_count": cycle.state_change_count,
        "market_state_baseline_established": state_artifacts.baseline_established,
        **flat_category_counts,
        "outputs": outputs,
    }
    payload["outputs"].update(cycle.paths_written)
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_run_live_market_service(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    output_dir = (
        args.output_dir
        or _default_live_market_output_dir(config.project_root, args.as_of)
    ).resolve()
    bundle = _build_live_market_service_bundle(
        args,
        config=config,
        env_file=args.env_file,
        output_dir=output_dir,
    )
    status = bundle.supervisor.run(max_iterations=args.max_iterations)
    payload = {
        "command": "run-live-market-service",
        "as_of_date": args.as_of.isoformat(),
        "websocket_url": args.websocket_url,
        "transport_name": args.transport_name,
        "portfolio_file": (
            str(args.portfolio_file.resolve()) if args.portfolio_file is not None else None
        ),
        "include_intraday_review": bool(args.include_intraday_review),
        "notifications_enabled": bundle.notification_router is not None,
        "status": status.to_dict(),
        "outputs": dict(sorted(bundle.latest_outputs.items())),
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _build_live_market_service_bundle(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    env_file: Path | None,
    output_dir: Path,
) -> _LiveMarketServiceBundle:
    if args.include_intraday_review and args.portfolio_file is None:
        raise ValueError("--include-intraday-review requires --portfolio-file.")

    provider = create_daily_bar_provider(config, env_file=env_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = LiveMarketCycleRequest(
        workflow="monitor-market",
        as_of_date=args.as_of,
        portfolio_path=(
            str(args.portfolio_file.resolve()) if args.portfolio_file is not None else None
        ),
        include_daily_summary=True,
        include_portfolio_review=args.portfolio_file is not None,
        include_intraday_review=(
            bool(args.include_intraday_review) and args.portfolio_file is not None
        ),
        daily_summary_required=True,
        portfolio_review_required=False,
        intraday_review_required=False,
    )
    transport = JsonWebsocketTransport(
        url=args.websocket_url,
        subscription_messages=tuple(args.subscription_message or ()),
        receive_timeout_seconds=args.receive_timeout_seconds,
        max_messages_per_read=args.max_messages_per_poll,
        transport_name=args.transport_name,
    )
    adapter = WebsocketIngestionAdapter(
        transport_name=transport.transport_name,
        transport=transport,
        portfolio_path=request.portfolio_path,
        load_current_positions=lambda: _load_current_positions(args.portfolio_file),
        run_daily_summary=lambda snapshot, current_positions: _run_streaming_daily_summary_workflow(
            args,
            config=config,
            current_positions=current_positions,
            snapshot=snapshot,
            provider=provider,
        ),
        run_portfolio_review=(
            lambda snapshot, current_positions: _run_streaming_portfolio_review_workflow(
                args,
                config=config,
                current_positions=current_positions,
                snapshot=snapshot,
                provider=provider,
            )
            if args.portfolio_file is not None
            else None
        ),
        run_intraday_review=(
            lambda snapshot, current_positions: _run_streaming_intraday_review_workflow(
                args,
                config=config,
                current_positions=current_positions,
                snapshot=snapshot,
                provider=provider,
            )
            if args.portfolio_file is not None and args.include_intraday_review
            else None
        ),
        provider_name=_configured_provider_name(config),
        flush_interval_seconds=args.flush_interval_seconds,
        message_normalizer=(
            normalize_polygon_websocket_message
            if "polygon" in args.websocket_url.lower()
            else None
        ),
    )
    latest_outputs: dict[str, str] = {}
    notification_router = _build_notification_router(
        project_root=config.project_root,
        env_file=env_file,
        webhook_url=getattr(args, "notification_webhook_url", None),
        webhook_format=getattr(args, "notification_webhook_format", None),
        minimum_severity=getattr(args, "notification_min_severity", None),
    )

    def handle_cycle_result(cycle_result: LiveMarketCycleResult) -> None:
        latest_outputs.update(
            _write_live_market_cycle_outputs(
                cycle_result,
                output_dir=output_dir,
            )
        )
        _route_notification_events(
            notification_router,
            _notification_events_from_cycle_result(cycle_result),
        )

    return _LiveMarketServiceBundle(
        supervisor=LiveMarketSupervisor(
            runner=LiveMarketRunner(
                build_monitor_report=lambda as_of_date, daily_summary, portfolio_review, intraday_review, portfolio_path: build_market_monitor_report(
                    as_of_date=as_of_date,
                    daily_summary=daily_summary,
                    portfolio_review=(
                        _strip_internal_portfolio_review_report(portfolio_review)
                        if portfolio_review is not None
                        else None
                    ),
                    portfolio_path=portfolio_path,
                ),
                persist_market_state=lambda as_of_date, workflow, daily_summary, portfolio_review, intraday_review, portfolio_path: _persist_market_state(
                    project_root=config.project_root,
                    output_dir=output_dir,
                    as_of_date=as_of_date,
                    workflow=workflow,
                    portfolio_path=portfolio_path,
                    daily_summary=daily_summary,
                    portfolio_review=(
                        _strip_internal_portfolio_review_report(portfolio_review)
                        if portfolio_review is not None
                        else None
                    ),
                    intraday_review=intraday_review,
                ),
            ),
            adapter=adapter,
            request=request,
            poll_interval_seconds=args.poll_interval_seconds,
            stale_after_seconds=args.stale_after_seconds,
            max_messages_per_poll=args.max_messages_per_poll,
            reconnect_backoff=ReconnectBackoffPolicy(
                initial_delay_seconds=args.initial_reconnect_delay_seconds,
                multiplier=args.reconnect_backoff_multiplier,
                max_delay_seconds=args.max_reconnect_delay_seconds,
            ),
            handle_cycle_result=handle_cycle_result,
            status_store=LiveMarketServiceStatusStore(config.project_root),
            notification_router=notification_router,
        ),
        latest_outputs=latest_outputs,
        notification_router=notification_router,
    )


def _handle_run_local_staging_runtime(args: argparse.Namespace) -> int:
    runtime_config = LocalStagingRuntimeConfig(
        source_config_dir=args.config_dir,
        runtime_root=args.runtime_root,
        source_env_file=args.env_file,
        hot_state_dir=args.hot_state_dir,
        logs_dir=args.logs_dir,
        archive_dir=args.archive_dir,
        internal_api_host=args.internal_api_host,
        internal_api_port=args.internal_api_port,
        control_api_host=args.control_api_host,
        control_api_port=args.control_api_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        dashboard_refresh_seconds=args.dashboard_refresh_seconds,
        active_log_filename=args.active_log_filename,
        log_rotate_max_bytes=args.log_rotate_max_bytes,
        log_rotate_backup_count=args.log_rotate_backup_count,
    )
    bundle_holder: dict[str, _LiveMarketServiceBundle] = {}
    try:
        runtime = create_local_staging_runtime(
            runtime_config,
            supervisor_factory=lambda config, paths: _capture_staging_supervisor(
                bundle_holder,
                _build_live_market_service_bundle(
                    args,
                    config=config,
                    env_file=paths.env_file,
                    output_dir=(
                        args.output_dir
                        or paths.archive_dir / "live_market" / args.as_of.isoformat()
                    ).resolve(),
                ),
            ),
        )
    except OSError as exc:
        raise ValueError(f"Failed to initialize local staging runtime: {exc}") from exc
    runtime.configure_logging(level=args.log_level, json_logs=args.json_logs)
    LOGGER.info(
        "Starting local staging runtime: internal_api=%s control_api=%s dashboard=%s",
        runtime.internal_api_url,
        runtime.control_api_url,
        runtime.dashboard_url,
    )
    try:
        runtime.start(max_iterations=args.max_iterations)
        runtime.wait_for_stack()
    except KeyboardInterrupt:
        LOGGER.info("Stopping local staging runtime.")
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    finally:
        runtime.stop()

    bundle = bundle_holder.get("live_market_bundle")
    supervisor_status = runtime.last_supervisor_result
    payload = {
        "command": "run-local-staging-runtime",
        "runtime_root": str(runtime.paths.runtime_root),
        "config_dir": str(runtime.paths.config_dir),
        "env_file": str(runtime.paths.env_file),
        "hot_state_dir": str(runtime.paths.hot_state_dir),
        "logs_dir": str(runtime.paths.logs_dir),
        "active_log_path": str(runtime.paths.active_log_path),
        "archive_dir": str(runtime.paths.archive_dir),
        "archived_logs_dir": str(runtime.paths.archived_logs_dir),
        "internal_api_url": runtime.internal_api_url,
        "control_api_url": runtime.control_api_url,
        "dashboard_url": runtime.dashboard_url,
        "as_of_date": args.as_of.isoformat(),
        "websocket_url": args.websocket_url,
        "transport_name": args.transport_name,
        "portfolio_file": (
            str(args.portfolio_file.resolve()) if args.portfolio_file is not None else None
        ),
        "include_intraday_review": bool(args.include_intraday_review),
        "notifications_enabled": (
            bundle.notification_router is not None if bundle is not None else False
        ),
        "status": (
            supervisor_status.to_dict()
            if hasattr(supervisor_status, "to_dict")
            else supervisor_status
        ),
        "outputs": (
            dict(sorted(bundle.latest_outputs.items()))
            if bundle is not None
            else {}
        ),
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _local_paper_validation_profile_and_config(
    args: argparse.Namespace,
) -> tuple[Any, LocalStagingRuntimeConfig]:
    profile = build_local_paper_validation_profile(
        project_root=args.config_dir.resolve().parent,
        validation_root=args.validation_root,
    )
    runtime_config = build_local_paper_validation_runtime_config(
        profile=profile,
        source_config_dir=args.config_dir,
        source_env_file=args.env_file,
        internal_api_host=args.internal_api_host,
        internal_api_port=args.internal_api_port,
        control_api_host=args.control_api_host,
        control_api_port=args.control_api_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        dashboard_refresh_seconds=args.dashboard_refresh_seconds,
        active_log_filename=args.active_log_filename,
        log_rotate_max_bytes=args.log_rotate_max_bytes,
        log_rotate_backup_count=args.log_rotate_backup_count,
    )
    return profile, runtime_config


def _probe_local_paper_validation_surface(
    url: str,
    *,
    expect_json: bool,
    timeout_seconds: float = 0.5,
    attempts: int = 5,
    retry_delay_seconds: float = 0.1,
) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "investopedia-bot-smoke"})
    last_error: str | None = None
    last_status_code: int | None = None
    for attempt_index in range(max(attempts, 1)):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.getcode()
                last_status_code = status_code
                body = response.read()
                if expect_json:
                    try:
                        payload = json.loads(body.decode("utf-8") or "{}")
                    except json.JSONDecodeError as exc:
                        last_error = f"Invalid JSON response: {exc}"
                    else:
                        return {
                            "url": url,
                            "available": 200 <= status_code < 400,
                            "status_code": status_code,
                            "error": None,
                            "json_payload": payload,
                        }
                else:
                    return {
                        "url": url,
                        "available": 200 <= status_code < 400,
                        "status_code": status_code,
                        "error": None,
                    }
        except HTTPError as exc:
            last_status_code = exc.code
            last_error = f"HTTP {exc.code}: {exc.reason}"
        except (OSError, URLError) as exc:
            last_error = str(exc)
        if attempt_index + 1 < max(attempts, 1):
            time.sleep(retry_delay_seconds)
    return {
        "url": url,
        "available": False,
        "status_code": last_status_code,
        "error": last_error,
    }


def _run_local_paper_validation_smoke_probes(runtime: Any) -> dict[str, dict[str, Any]]:
    return {
        "internal_api": _probe_local_paper_validation_surface(
            runtime.internal_api_url.rstrip("/") + "/health",
            expect_json=True,
        ),
        "control_api": _probe_local_paper_validation_surface(
            runtime.control_api_url.rstrip("/") + "/control/safety",
            expect_json=True,
        ),
        "dashboard": _probe_local_paper_validation_surface(
            runtime.dashboard_url,
            expect_json=False,
        ),
    }


def _handle_run_local_paper_validation(args: argparse.Namespace) -> int:
    profile, runtime_config = _local_paper_validation_profile_and_config(args)
    bundle_holder: dict[str, _LiveMarketServiceBundle] = {}
    try:
        runtime = create_local_paper_validation_runtime(
            runtime_config=runtime_config,
            validation_root=profile.validation_root,
            reset_safety_state=not args.preserve_existing_safety_state,
            supervisor_factory=lambda config, paths: _capture_staging_supervisor(
                bundle_holder,
                _build_live_market_service_bundle(
                    args,
                    config=config,
                    env_file=paths.env_file,
                    output_dir=(
                        args.output_dir
                        or profile.default_live_output_dir(as_of_date=args.as_of)
                    ).resolve(),
                ),
            ),
        )
    except OSError as exc:
        raise ValueError(f"Failed to initialize local paper validation runtime: {exc}") from exc

    runtime.configure_logging(level=args.log_level, json_logs=args.json_logs)
    LOGGER.info(
        "Starting local paper validation runtime: internal_api=%s control_api=%s dashboard=%s",
        runtime.internal_api_url,
        runtime.control_api_url,
        runtime.dashboard_url,
    )
    runtime_failure: RuntimeError | None = None
    try:
        runtime.start(max_iterations=args.max_iterations)
        runtime.wait_for_stack()
    except KeyboardInterrupt:
        LOGGER.info("Stopping local paper validation runtime.")
    except RuntimeError as exc:
        runtime_failure = exc
    finally:
        runtime.stop()

    review_paths = runtime.write_review_summary(
        as_of_date=args.as_of,
        window_days=1,
    )
    if runtime_failure is not None:
        raise ValueError(str(runtime_failure)) from runtime_failure
    bundle = bundle_holder.get("live_market_bundle")
    supervisor_status = runtime.last_supervisor_result
    payload = {
        "command": "run-local-paper-validation",
        "validation_profile": profile.to_dict(),
        "as_of_date": args.as_of.isoformat(),
        "websocket_url": args.websocket_url,
        "transport_name": args.transport_name,
        "portfolio_file": (
            str(args.portfolio_file.resolve()) if args.portfolio_file is not None else None
        ),
        "include_intraday_review": bool(args.include_intraday_review),
        "notifications_enabled": (
            bundle.notification_router is not None if bundle is not None else False
        ),
        "status": (
            supervisor_status.to_dict()
            if hasattr(supervisor_status, "to_dict")
            else supervisor_status
        ),
        "outputs": (
            dict(sorted(bundle.latest_outputs.items()))
            if bundle is not None
            else {}
        ),
        "review_outputs": {
            label: str(path.resolve()) for label, path in sorted(review_paths.items())
        },
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_run_local_paper_validation_smoke(args: argparse.Namespace) -> int:
    profile, runtime_config = _local_paper_validation_profile_and_config(args)
    bundle_holder: dict[str, _LiveMarketServiceBundle] = {}
    review_output_dir = profile.default_review_output_dir(as_of_date=args.as_of)
    runtime: Any | None = None
    runtime_failure_message: str | None = None
    review_failure_message: str | None = None
    summary_failure_message: str | None = None
    startup_succeeded = False
    runtime_stopped_cleanly = False
    review_paths: dict[str, Path] = {}
    summary = None
    probes: dict[str, dict[str, Any]] = {
        "internal_api": {"url": None, "available": False, "status_code": None, "error": None},
        "control_api": {"url": None, "available": False, "status_code": None, "error": None},
        "dashboard": {"url": None, "available": False, "status_code": None, "error": None},
    }
    try:
        runtime = create_local_paper_validation_runtime(
            runtime_config=runtime_config,
            validation_root=profile.validation_root,
            reset_safety_state=not args.preserve_existing_safety_state,
            supervisor_factory=lambda config, paths: _capture_staging_supervisor(
                bundle_holder,
                _build_live_market_service_bundle(
                    args,
                    config=config,
                    env_file=paths.env_file,
                    output_dir=(
                        args.output_dir
                        or profile.default_live_output_dir(as_of_date=args.as_of)
                    ).resolve(),
                ),
            ),
        )
    except OSError as exc:
        runtime_failure_message = (
            f"Failed to initialize local paper validation smoke runtime: {exc}"
        )
    if runtime is not None:
        runtime.configure_logging(level=args.log_level, json_logs=args.json_logs)
        LOGGER.info(
            "Starting local paper validation smoke run: internal_api=%s control_api=%s dashboard=%s",
            runtime.internal_api_url,
            runtime.control_api_url,
            runtime.dashboard_url,
        )
        try:
            runtime.start(max_iterations=args.max_iterations)
            startup_succeeded = True
            probes = _run_local_paper_validation_smoke_probes(runtime)
            runtime.wait_for_stack()
        except KeyboardInterrupt:
            runtime_failure_message = (
                "Local paper validation smoke run was interrupted before bounded completion."
            )
        except RuntimeError as exc:
            runtime_failure_message = str(exc)
        finally:
            try:
                runtime.stop()
                runtime_stopped_cleanly = True
            except OSError as exc:
                stop_error = f"Failed to stop local paper validation smoke runtime cleanly: {exc}"
                runtime_failure_message = (
                    f"{runtime_failure_message} {stop_error}"
                    if runtime_failure_message is not None
                    else stop_error
                )
    if runtime is not None:
        try:
            review_paths = runtime.write_review_summary(
                as_of_date=args.as_of,
                window_days=1,
            )
        except OSError as exc:
            review_failure_message = (
                f"Failed to write paper-validation review artifacts: {exc}"
            )
    try:
        summary = build_local_paper_validation_summary(
            profile,
            as_of_date=args.as_of,
            window_days=1,
        )
    except Exception as exc:
        summary_failure_message = (
            f"Failed to build paper-validation smoke summary: {exc}"
        )
    smoke_result = build_local_paper_validation_smoke_result(
        profile,
        as_of_date=args.as_of,
        max_iterations=args.max_iterations,
        summary=summary,
        review_paths=review_paths,
        internal_api_probe=probes["internal_api"],
        control_api_probe=probes["control_api"],
        dashboard_probe=probes["dashboard"],
        startup_succeeded=startup_succeeded,
        runtime_stopped_cleanly=runtime_stopped_cleanly,
        runtime_failure_message=runtime_failure_message,
        review_failure_message=review_failure_message,
        summary_failure_message=summary_failure_message,
    )
    smoke_paths = write_local_paper_validation_smoke_result(
        smoke_result,
        output_dir=review_output_dir,
    )
    bundle = bundle_holder.get("live_market_bundle")
    supervisor_status = runtime.last_supervisor_result if runtime is not None else None
    payload = {
        "command": "run-local-paper-validation-smoke",
        "validation_profile": profile.to_dict(),
        "as_of_date": args.as_of.isoformat(),
        "websocket_url": args.websocket_url,
        "transport_name": args.transport_name,
        "portfolio_file": (
            str(args.portfolio_file.resolve()) if args.portfolio_file is not None else None
        ),
        "include_intraday_review": bool(args.include_intraday_review),
        "notifications_enabled": (
            bundle.notification_router is not None if bundle is not None else False
        ),
        "status": (
            supervisor_status.to_dict()
            if hasattr(supervisor_status, "to_dict")
            else supervisor_status
        ),
        "outputs": (
            dict(sorted(bundle.latest_outputs.items()))
            if bundle is not None
            else {}
        ),
        "review_outputs": {
            label: str(path.resolve()) for label, path in sorted(review_paths.items())
        },
        "smoke_outputs": {
            label: str(path.resolve()) for label, path in sorted(smoke_paths.items())
        },
        "smoke_result": smoke_result.to_dict(),
    }
    _print_structured(payload, output_format=args.format)
    return 0 if smoke_result.passed else 1


def _handle_summarize_local_paper_validation(args: argparse.Namespace) -> int:
    profile = build_local_paper_validation_profile(
        project_root=args.config_dir.resolve().parent,
        validation_root=args.validation_root,
    )
    summary = build_local_paper_validation_summary(
        profile,
        as_of_date=args.as_of,
        window_days=args.window_days,
    )
    review_output_dir = (
        args.output_dir
        or profile.default_review_output_dir(as_of_date=args.as_of)
    ).resolve()
    written_paths = write_local_paper_validation_summary(
        summary,
        output_dir=review_output_dir,
    )
    payload = summary.to_dict()
    payload["command"] = "summarize-local-paper-validation"
    payload["review_outputs"] = {
        label: str(path.resolve()) for label, path in sorted(written_paths.items())
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _capture_staging_supervisor(
    holder: dict[str, _LiveMarketServiceBundle],
    bundle: _LiveMarketServiceBundle,
) -> LiveMarketSupervisor:
    holder["live_market_bundle"] = bundle
    return bundle.supervisor


def _handle_serve_internal_api(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    if args.port <= 0:
        raise ValueError("port must be greater than zero.")
    LOGGER.info(
        "Starting internal API server on http://%s:%s",
        args.host,
        args.port,
    )
    try:
        try:
            serve_internal_api(
                query_service=InternalApiQueryService(
                    project_root=config.project_root,
                    config=config,
                ),
                host=args.host,
                port=args.port,
            ),
        except OSError as exc:
            raise ValueError(f"Failed to start internal API server: {exc}") from exc
    except KeyboardInterrupt:
        LOGGER.info("Stopping internal API server.")
    return 0


def _handle_serve_operator_control_api(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    if args.port <= 0:
        raise ValueError("port must be greater than zero.")
    LOGGER.info(
        "Starting operator control API server on http://%s:%s/control",
        args.host,
        args.port,
    )
    try:
        try:
            serve_operator_control_api(
                control_service=OperatorControlService(
                    project_root=config.project_root,
                    env_file=getattr(args, "env_file", None),
                ),
                host=args.host,
                port=args.port,
            )
        except OSError as exc:
            raise ValueError(f"Failed to start operator control API server: {exc}") from exc
    except KeyboardInterrupt:
        LOGGER.info("Stopping operator control API server.")
    return 0


def _handle_serve_operator_dashboard(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    if args.port <= 0:
        raise ValueError("port must be greater than zero.")
    if args.refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be greater than zero.")
    LOGGER.info(
        "Starting operator dashboard on http://%s:%s/dashboard",
        args.host,
        args.port,
    )
    try:
        try:
            serve_operator_dashboard(
                query_service=InternalApiQueryService(
                    project_root=config.project_root,
                    config=config,
                ),
                control_service=OperatorControlService(
                    project_root=config.project_root,
                    env_file=getattr(args, "env_file", None),
                ),
                host=args.host,
                port=args.port,
                refresh_interval_seconds=args.refresh_seconds,
            )
        except OSError as exc:
            raise ValueError(f"Failed to start operator dashboard server: {exc}") from exc
    except KeyboardInterrupt:
        LOGGER.info("Stopping operator dashboard server.")
    return 0


def _write_live_market_cycle_outputs(
    cycle_result: LiveMarketCycleResult,
    *,
    output_dir: Path,
) -> dict[str, str]:
    outputs = dict(cycle_result.paths_written)
    if cycle_result.monitor_report is not None:
        outputs["market_monitor_json"] = str(
            write_market_monitor_report(
                cycle_result.monitor_report,
                output_dir / "market_monitor.json",
            )
        )
        outputs["market_monitor_csv"] = str(
            write_market_monitor_report(
                cycle_result.monitor_report,
                output_dir / "market_monitor.csv",
            )
        )
        outputs["market_monitor_text"] = str(
            write_market_monitor_text_summary(
                cycle_result.monitor_report,
                output_dir / "market_monitor.txt",
            )
        )
        outputs["market_monitor_brief"] = str(
            write_market_monitor_brief(
                cycle_result.monitor_report,
                output_dir / "market_monitor_brief.txt",
            )
        )
    if cycle_result.portfolio_review is not None:
        sanitized_review = _strip_internal_portfolio_review_report(
            cycle_result.portfolio_review,
        )
        outputs["portfolio_review_json"] = str(
            write_portfolio_review_report(
                sanitized_review,
                output_dir / "portfolio_review.json",
            )
        )
        outputs["portfolio_review_csv"] = str(
            write_portfolio_review_report(
                sanitized_review,
                output_dir / "portfolio_review.csv",
            )
        )
    if cycle_result.intraday_review is not None:
        outputs["portfolio_review_intraday_json"] = str(
            write_intraday_portfolio_review_report(
                cycle_result.intraday_review,
                output_dir / "portfolio_review_intraday.json",
            )
        )
        outputs["portfolio_review_intraday_csv"] = str(
            write_intraday_portfolio_review_report(
                cycle_result.intraday_review,
                output_dir / "portfolio_review_intraday.csv",
            )
        )
        outputs["portfolio_review_intraday_brief"] = str(
            write_intraday_portfolio_review_brief(
                cycle_result.intraday_review,
                output_dir / "portfolio_review_intraday_brief.txt",
            )
        )
    if cycle_result.daily_summary_result is not None:
        candidate_score_journal_path = cycle_result.daily_summary_result.get(
            "candidate_score_journal_path"
        )
        if candidate_score_journal_path is not None:
            outputs["candidate_score_journal"] = str(candidate_score_journal_path)
    return outputs


def _handle_generate_orders(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    provider = create_daily_bar_provider(config, env_file=args.env_file)
    current_positions = _load_current_positions(args.portfolio_file)
    builder = UniverseBuilder(provider, config.strategy.universe)
    universe_members = builder.screen_candidates(
        args.candidate_path,
        as_of_date=args.as_of,
        lookback_days=args.lookback_days,
        refresh_cache=args.refresh_cache,
        enforce_max_symbols=False,
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
    pending_order_state, pending_order_reservation_state = (
        _build_pending_order_reservation_state(
            project_root=config.project_root,
            current_equity=current_equity,
            current_positions=current_positions,
            constraints=constraints,
        )
    )
    assessed_candidates = []
    no_signal_symbols: list[str] = []
    symbol_frames: dict[str, pd.DataFrame] = {}
    earnings_contexts = _load_earnings_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        symbols=[member.symbol for member in universe_members],
        as_of_date=args.as_of,
        earnings_watch_days=config.strategy.signals.earnings_watch_days,
        refresh_cache=args.refresh_cache,
        log_label="generate-orders",
    )
    sector_history_start = sector_context_history_start(
        args.as_of,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        relative_strength_window=config.strategy.signals.sector_relative_strength_window,
    )

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

        symbol_frames[member.symbol] = bars

    breadth_metrics = _compute_entry_market_breadth(
        symbol_frames,
        as_of_date=args.as_of,
        market_breadth_entry_floor_200ma=(
            strategy_settings.market_breadth_entry_floor_200ma
        ),
        log_label="generate-orders",
    )
    market_context = build_market_context(
        as_of_date=args.as_of,
        benchmark_symbol=(
            strategy_settings.benchmark_symbol
            if strategy_settings.enable_regime_filter
            else None
        ),
        regime_filter_mode=strategy_settings.regime_filter_mode,
        market_breadth_pct_above_200ma=(
            breadth_metrics["market_breadth_pct_above_200ma"]
        ),
        market_breadth_pct_above_50ma=(
            breadth_metrics["market_breadth_pct_above_50ma"]
        ),
        market_breadth_state=breadth_metrics["market_breadth_state"],
        **_load_volatility_context(
            provider=provider,
            as_of_date=args.as_of,
            vix_caution_threshold=strategy_settings.vix_caution_threshold,
            vix_entry_block_threshold=strategy_settings.vix_entry_block_threshold,
            refresh_cache=args.refresh_cache,
            log_label="generate-orders",
        ),
    )

    sector_contexts = _load_sector_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        provider=provider,
        symbol_frames=symbol_frames,
        as_of_date=args.as_of,
        history_start=sector_history_start,
        benchmark_sma_fast=config.strategy.signals.benchmark_sma_fast,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        relative_strength_window=config.strategy.signals.sector_relative_strength_window,
        refresh_cache=args.refresh_cache,
        log_label="generate-orders",
    )
    current_position_classifications = _load_position_sector_classifications(
        current_positions,
        config=config,
        env_file=getattr(args, "env_file", None),
        as_of_date=args.as_of,
        refresh_cache=args.refresh_cache,
        log_label="generate-orders",
    )
    portfolio_holding_exposures = _build_portfolio_holding_exposures(
        current_positions,
        current_position_classifications,
    )
    portfolio_holding_exposures.extend(
        build_pending_order_holding_exposures(
            pending_order_state,
            current_positions=current_positions,
        )
    )

    for member in universe_members:
        bars = symbol_frames.get(member.symbol)
        if bars is None or bars.empty:
            continue
        # Candidate journal memory is intentionally not applied here. generate-orders is a
        # pure execution path, while daily-summary/monitor-market own the memory-enhanced
        # research ranking layer.
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

        earnings_context = earnings_contexts.get(member.symbol.strip().upper())
        sector_context = sector_contexts.get(member.symbol.strip().upper())
        portfolio_heat_context = build_portfolio_heat_context(
            portfolio_holding_exposures,
            candidate_sector=(
                sector_context.sector_name if sector_context is not None else None
            ),
            candidate_industry=(
                sector_context.industry_name if sector_context is not None else None
            ),
            max_positions_per_sector=strategy_settings.max_positions_per_sector,
            max_same_industry_positions=strategy_settings.max_same_industry_positions,
            max_sector_notional_pct=strategy_settings.max_sector_notional_pct,
        )
        if strategy_settings.require_sector_regime_for_entries and sector_context is None:
            _warn_missing_sector_context_for_entry(
                member.symbol,
                log_label="generate-orders",
            )
        signal = replace(
            signal,
            metadata={
                **dict(signal.metadata),
                **earnings_context_metadata(earnings_context),
                **sector_context_metadata(sector_context),
                **market_context_metadata(
                    market_context,
                    market_breadth_entry_floor_200ma=(
                        strategy_settings.market_breadth_entry_floor_200ma
                    ),
                    vix_caution_threshold=strategy_settings.vix_caution_threshold,
                    vix_entry_block_threshold=(
                        strategy_settings.vix_entry_block_threshold
                    ),
                ),
            },
        )
        candidate_features = build_candidate_features(
            signal,
            as_of_date=args.as_of,
            earnings_context=earnings_context,
            sector_context=sector_context,
            market_context=market_context,
            portfolio_heat_context=portfolio_heat_context,
        )
        assessed_candidate = assess_signal_candidate(
            signal,
            current_equity=current_equity,
            base_risk_per_trade=config.strategy.risk.risk_per_trade,
            constraints=constraints,
            candidate_features=candidate_features,
            current_positions=current_positions,
            current_drawdown=float(args.current_drawdown),
            earnings_date=(
                earnings_context.earnings_date
                if earnings_context is not None
                else None
            ),
            earnings_days_away=(
                earnings_context.earnings_days_away
                if earnings_context is not None
                else None
            ),
            earnings_entry_block_days=strategy_settings.earnings_entry_block_days,
            sector_name=(
                sector_context.sector_name
                if sector_context is not None
                else None
            ),
            sector_etf_symbol=(
                sector_context.sector_etf_symbol
                if sector_context is not None
                else None
            ),
            sector_regime_passed=(
                sector_context.sector_regime_passed
                if sector_context is not None
                else None
            ),
            relative_strength_vs_sector=(
                sector_context.relative_strength_vs_sector
                if sector_context is not None
                else None
            ),
            sector_relative_strength_window=(
                sector_context.relative_strength_window
                if sector_context is not None
                else None
            ),
            market_breadth_entry_floor_200ma=(
                strategy_settings.market_breadth_entry_floor_200ma
            ),
            vix_entry_block_threshold=(
                strategy_settings.vix_entry_block_threshold
            ),
            require_sector_regime_for_entries=(
                strategy_settings.require_sector_regime_for_entries
            ),
            sector_relative_strength_entry_reject_threshold=(
                strategy_settings.sector_relative_strength_entry_reject_threshold
            ),
        )
        assessed_candidates.append(assessed_candidate)
        _roll_forward_portfolio_holding_exposures(
            portfolio_holding_exposures,
            assessed_candidate,
            sector_context=sector_context,
        )

    assessed_candidates = rank_risk_assessed_candidates(
        annotate_execution_priority(
            assessed_candidates,
            current_equity=current_equity,
            current_positions=current_positions,
            constraints=constraints,
            reservation_state=pending_order_reservation_state,
        ),
        current_equity=current_equity,
    )
    assessed_candidates = _attach_candidate_decision_ids(
        assessed_candidates,
        workflow="generate-orders",
        as_of_date=args.as_of,
    )

    executor = ManualExecutor()
    batch = executor.build_execution_batch(assessed_candidates, as_of_date=args.as_of)
    staged_pending_order_count = 0
    staged_pending_order_path: Path | None = None
    post_stage_pending_order_summary: dict[str, Any] | None = None
    if args.stage_pending_orders and batch.orders:
        staged_pending_order_count, staged_pending_order_path, staged_state = (
            _stage_execution_batch_pending_orders(
                project_root=config.project_root,
                batch=batch,
            )
        )
        if staged_pending_order_path is not None:
            post_stage_pending_order_summary = build_order_reservation_state(
                staged_state,
                current_equity=current_equity,
                current_positions=current_positions,
                max_concurrent_positions=constraints.max_concurrent_positions,
            ).to_dict()
    effective_pending_order_summary = (
        post_stage_pending_order_summary
        if post_stage_pending_order_summary is not None
        else pending_order_reservation_state.to_dict()
    )
    report = build_daily_signal_report(
        as_of_date=args.as_of,
        execution_batch=batch,
        assessed_candidates=assessed_candidates,
        universe_symbols=[member.symbol for member in universe_members],
        current_positions=current_positions,
        no_signal_symbols=no_signal_symbols,
        benchmark_symbol=strategy_settings.benchmark_symbol if strategy_settings.enable_regime_filter else None,
        pending_order_summary=effective_pending_order_summary,
    )

    output_dir = (args.output_dir or _default_daily_output_dir(config.project_root, args.as_of)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    order_sheet_path = write_execution_batch(batch, output_dir / "manual_order_sheet.csv")
    signal_report_json_path = write_daily_signal_report(report, output_dir / "daily_signal_report.json")
    signal_report_csv_path = write_daily_signal_report(report, output_dir / "daily_signal_report.csv")
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[
            _candidate_trade_feedback_event(
                candidate,
                workflow="generate-orders",
                as_of_date=args.as_of,
                timestamp_utc=batch.generated_at_utc,
            )
            for candidate in assessed_candidates
        ],
    )

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "equity": current_equity,
        "current_drawdown": float(args.current_drawdown),
        "portfolio_file": str(args.portfolio_file.resolve()) if args.portfolio_file else None,
        "current_position_count": len(current_positions),
        "current_position_symbols": [position.symbol for position in current_positions],
        "pending_order_summary": effective_pending_order_summary,
        "blocked_by_pending_reservations_count": sum(
            candidate.approved
            and candidate.execution_priority is not None
            and candidate.execution_priority.capacity_blocked_by_pending
            and not candidate.execution_priority.actionable_now
            for candidate in assessed_candidates
        ),
        "relative_volume_confirmation_required": (
            strategy_settings.require_relative_volume_confirmation
        ),
        "relative_volume_policy": (
            "required"
            if strategy_settings.require_relative_volume_confirmation
            else "optional"
        ),
        "market_breadth_pct_above_200ma": market_context.market_breadth_pct_above_200ma,
        "market_breadth_pct_above_50ma": market_context.market_breadth_pct_above_50ma,
        "market_breadth_state": market_context.market_breadth_state,
        "vix_close": market_context.vix_close,
        "volatility_regime_state": market_context.volatility_regime_state,
        "volatility_regime_risk_off": market_context.volatility_regime_risk_off,
        "universe_count": len(universe_members),
        "signal_count": len(assessed_candidates),
        "approved_signal_count": sum(candidate.approved for candidate in assessed_candidates),
        "approved_order_count": len(batch.orders),
        "deferred_approved_count": sum(
            candidate.approved
            and candidate.execution_priority is not None
            and not candidate.execution_priority.actionable_now
            for candidate in assessed_candidates
        ),
        "rejected_signal_count": sum(not candidate.approved for candidate in assessed_candidates),
        "no_signal_count": len(no_signal_symbols),
        "order_symbols": [order.symbol for order in batch.orders],
        "outputs": {
            "manual_order_sheet": str(order_sheet_path),
            "daily_signal_report_json": str(signal_report_json_path),
            "daily_signal_report_csv": str(signal_report_csv_path),
        },
        "staged_pending_order_count": staged_pending_order_count,
    }
    if trade_feedback_log_path is not None:
        payload["outputs"]["trade_decision_log"] = str(trade_feedback_log_path)
        payload["outputs"]["trade_feedback_log"] = str(trade_feedback_log_path)
    if staged_pending_order_path is not None:
        payload["outputs"]["pending_order_state"] = str(staged_pending_order_path)
    if post_stage_pending_order_summary is not None:
        payload["post_stage_pending_order_summary"] = post_stage_pending_order_summary
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_submit_broker_orders(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    store = PendingOrderStateStore(config.project_root)
    starting_state = store.load()
    control_service = _broker_cli_control_service(
        project_root=config.project_root,
        env_file=args.env_file,
    )
    requested_ids = {
        order_id.strip() for order_id in (args.order_ids or ()) if order_id and order_id.strip()
    }
    candidate_orders = [
        record
        for record in starting_state.active_orders()
        if not requested_ids or record.order_id in requested_ids
    ]
    trade_feedback_before = _trade_feedback_event_count(config.project_root)
    notification_router = _build_notification_router(
        project_root=config.project_root,
        env_file=args.env_file,
        webhook_url=getattr(args, "notification_webhook_url", None),
        webhook_format=getattr(args, "notification_webhook_format", None),
        minimum_severity=getattr(args, "notification_min_severity", None),
    )
    submitted_updates: list[dict[str, object]] = []
    skipped_order_ids: list[str] = []
    warnings: list[str] = []
    order_results: list[dict[str, object]] = []
    artifact_paths: dict[str, str] = {}
    for record in candidate_orders:
        if record.broker_order_id is not None and record.broker_status not in {
            "cancelled",
            "expired",
            "replaced",
            "rejected",
        }:
            skipped_order_ids.append(record.order_id)
            continue
        result = control_service.submit_pending_order(
            SubmitPendingOrderCommand(
                broker=args.broker,
                order_id=record.order_id,
                confirm_live_action=getattr(args, "confirm_live_action", False),
            )
        )
        warnings.extend(result.warnings)
        if not result.ok and not result.warnings:
            warnings.append(result.message)
        if result.ok and result.status == "completed":
            submitted_updates.extend(result.data.get("submitted_updates", ()))
        artifact_paths.update(result.artifact_paths)
        order_results.append(
            {
                "order_id": record.order_id,
                "ok": result.ok,
                "status": result.status,
                "error_code": result.error_code,
                "message": result.message,
            }
        )
    warnings.extend(
        f"Pending order '{order_id}' was not found in active local state."
        for order_id in sorted(requested_ids - {record.order_id for record in candidate_orders})
    )
    feedback_events = _new_trade_feedback_events_since(
        config.project_root,
        trade_feedback_before,
    )
    notification_delivery = _route_notification_events(
        notification_router,
        _notification_events_from_trade_feedback_events(feedback_events)
        + _notification_events_from_warning_messages(
            workflow="submit-broker-orders",
            created_at=_trade_feedback_timestamp_utc(),
            warnings=warnings,
        ),
    )
    payload = {
        "command": "submit-broker-orders",
        "broker": args.broker,
        "submitted_count": len(submitted_updates),
        "skipped_order_ids": skipped_order_ids,
        "warnings": warnings,
        "submitted_updates": submitted_updates,
        "order_results": order_results,
        "outputs": {},
    }
    if artifact_paths:
        payload["outputs"].update(dict(sorted(artifact_paths.items())))
    if notification_delivery is not None:
        payload["notification_delivery"] = notification_delivery
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_sync_broker_orders(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    control_service = _broker_cli_control_service(
        project_root=config.project_root,
        env_file=args.env_file,
    )
    trade_feedback_before = _trade_feedback_event_count(config.project_root)
    requested_order_ids = tuple(
        order_id.strip()
        for order_id in (args.order_ids or ())
        if order_id and order_id.strip()
    )
    result = control_service.force_broker_sync(
        ForceBrokerSyncCommand(
            broker=args.broker,
            order_ids=requested_order_ids,
        )
    )
    notification_router = _build_notification_router(
        project_root=config.project_root,
        env_file=args.env_file,
        webhook_url=getattr(args, "notification_webhook_url", None),
        webhook_format=getattr(args, "notification_webhook_format", None),
        minimum_severity=getattr(args, "notification_min_severity", None),
    )
    warnings = list(result.warnings)
    if not result.ok and not warnings:
        warnings.append(result.message)
    feedback_events = _new_trade_feedback_events_since(
        config.project_root,
        trade_feedback_before,
    )
    notification_delivery = _route_notification_events(
        notification_router,
        _notification_events_from_trade_feedback_events(feedback_events)
        + _notification_events_from_warning_messages(
            workflow="sync-broker-orders",
            created_at=_trade_feedback_timestamp_utc(),
            warnings=warnings,
        ),
    )
    payload = {
        "command": "sync-broker-orders",
        "broker": args.broker,
        "ok": result.ok,
        "status": result.status,
        "message": result.message,
        "error_code": result.error_code,
        "selected_order_count": result.data.get("selected_order_count", 0),
        "matched_update_count": result.data.get("matched_update_count", 0),
        "unmatched_update_count": result.data.get("unmatched_update_count", 0),
        "warnings": warnings,
        "matched_updates": list(result.data.get("matched_updates", ())),
        "unmatched_updates": list(result.data.get("unmatched_updates", ())),
        "outputs": dict(sorted(result.artifact_paths.items())),
    }
    if notification_delivery is not None:
        payload["notification_delivery"] = notification_delivery
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_cancel_broker_order(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    control_service = _broker_cli_control_service(
        project_root=config.project_root,
        env_file=args.env_file,
    )
    trade_feedback_before = _trade_feedback_event_count(config.project_root)
    result = control_service.cancel_pending_order(
        CancelPendingOrderCommand(
            broker=args.broker,
            order_id=args.order_id,
            confirm_live_action=getattr(args, "confirm_live_action", False),
        )
    )
    notification_router = _build_notification_router(
        project_root=config.project_root,
        env_file=args.env_file,
        webhook_url=getattr(args, "notification_webhook_url", None),
        webhook_format=getattr(args, "notification_webhook_format", None),
        minimum_severity=getattr(args, "notification_min_severity", None),
    )
    warnings = list(result.warnings)
    if not result.ok and not warnings:
        warnings.append(result.message)
    feedback_events = _new_trade_feedback_events_since(
        config.project_root,
        trade_feedback_before,
    )
    notification_delivery = _route_notification_events(
        notification_router,
        _notification_events_from_trade_feedback_events(feedback_events)
        + _notification_events_from_warning_messages(
            workflow="cancel-broker-order",
            created_at=_trade_feedback_timestamp_utc(),
            warnings=warnings,
            related_ids={"order_id": args.order_id},
        ),
    )
    payload = {
        "command": "cancel-broker-order",
        "broker": args.broker,
        "order_id": args.order_id,
        "ok": result.ok,
        "status": result.status,
        "message": result.message,
        "error_code": result.error_code,
        "warnings": warnings,
        "outputs": dict(sorted(result.artifact_paths.items())),
    }
    if result.data.get("broker_update") is not None:
        payload["broker_update"] = result.data["broker_update"]
    if notification_delivery is not None:
        payload["notification_delivery"] = notification_delivery
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_replace_broker_order(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    control_service = _broker_cli_control_service(
        project_root=config.project_root,
        env_file=args.env_file,
    )
    trade_feedback_before = _trade_feedback_event_count(config.project_root)
    result = control_service.replace_pending_order(
        ReplacePendingOrderCommand(
            broker=args.broker,
            order_id=args.order_id,
            quantity=args.quantity,
            limit_price=args.limit_price,
            stop_price=args.stop_price,
            confirm_live_action=getattr(args, "confirm_live_action", False),
        )
    )
    notification_router = _build_notification_router(
        project_root=config.project_root,
        env_file=args.env_file,
        webhook_url=getattr(args, "notification_webhook_url", None),
        webhook_format=getattr(args, "notification_webhook_format", None),
        minimum_severity=getattr(args, "notification_min_severity", None),
    )
    warnings = list(result.warnings)
    if not result.ok and not warnings:
        warnings.append(result.message)
    feedback_events = _new_trade_feedback_events_since(
        config.project_root,
        trade_feedback_before,
    )
    notification_delivery = _route_notification_events(
        notification_router,
        _notification_events_from_trade_feedback_events(feedback_events)
        + _notification_events_from_warning_messages(
            workflow="replace-broker-order",
            created_at=_trade_feedback_timestamp_utc(),
            warnings=warnings,
            related_ids={
                "order_id": args.order_id,
                "replacement_order_id": str(result.data.get("replacement_order_id") or ""),
            },
        ),
    )
    payload = {
        "command": "replace-broker-order",
        "broker": args.broker,
        "replaced_order_id": args.order_id,
        "replacement_order_id": result.data.get("replacement_order_id"),
        "ok": result.ok,
        "status": result.status,
        "message": result.message,
        "error_code": result.error_code,
        "warnings": warnings,
        "outputs": dict(sorted(result.artifact_paths.items())),
    }
    if result.data.get("broker_update") is not None:
        payload["broker_update"] = result.data["broker_update"]
    if notification_delivery is not None:
        payload["notification_delivery"] = notification_delivery
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
        max_relative_volume_window=max(
            (parameter_set.breakout_lookback for parameter_set in parameter_sets),
            default=config.strategy.signals.breakout_lookback,
        ),
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
        max_relative_volume_window=_max_resolved_relative_volume_window_for_presets(
            config=config,
            presets=presets,
            require_relative_volume_confirmation=args.require_relative_volume,
            enable_regime_filter=not args.disable_regime_filter,
        ),
        max_sector_relative_strength_window=(
            config.strategy.signals.sector_relative_strength_window
        ),
        market_breadth_long_window=(
            200
            if config.strategy.signals.market_breadth_entry_floor_200ma is not None
            else 1
        ),
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
    current_positions = _load_current_positions(args.portfolio_file)
    workflow = _run_daily_summary_workflow(
        args,
        config=config,
        provider=provider,
        current_positions=current_positions,
    )
    summary = workflow["summary"]
    presets = workflow["presets"]
    preset_selection_source = workflow["preset_selection_source"]
    current_positions = workflow["current_positions"]
    current_equity = workflow["current_equity"]
    execution_batch = workflow["execution_batch"]
    ranked_evaluations = workflow["ranked_evaluations"]
    summary_payload = summary.to_dict()
    candidate_score_journal_path = workflow.get("candidate_score_journal_path")
    trade_feedback_log_path = _append_trade_feedback_events_safe(
        project_root=config.project_root,
        events=[
            _candidate_trade_feedback_event(
                evaluation.candidate,
                workflow="daily-summary",
                as_of_date=args.as_of,
                timestamp_utc=execution_batch.generated_at_utc,
            )
            for evaluation in ranked_evaluations
            if isinstance(evaluation, PresetCandidateEvaluation)
        ],
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
    brief_output_paths = {
        "daily_summary_json": str(summary_json_path),
        "ranked_opportunities_csv": str(opportunities_csv_path),
        "preset_rankings_csv": str(preset_csv_path),
        "preset_rankings_json": str(preset_json_path),
        "suggested_order_sheet_csv": str(orders_csv_path),
        "suggested_order_sheet_json": str(orders_json_path),
        "daily_summary_brief": str((output_dir / "daily_summary_brief.txt").resolve()),
    }
    if candidate_score_journal_path is not None:
        brief_output_paths["candidate_score_journal"] = str(candidate_score_journal_path)
    if trade_feedback_log_path is not None:
        brief_output_paths["trade_feedback_log"] = str(trade_feedback_log_path)
    brief_text_path = write_daily_research_brief(
        summary,
        output_dir / "daily_summary_brief.txt",
        output_paths=brief_output_paths,
    )

    outputs = {
        "daily_summary_json": str(summary_json_path),
        "ranked_opportunities_csv": str(opportunities_csv_path),
        "preset_rankings_csv": str(preset_csv_path),
        "preset_rankings_json": str(preset_json_path),
        "suggested_order_sheet_csv": str(orders_csv_path),
        "suggested_order_sheet_json": str(orders_json_path),
        "daily_summary_brief": str(brief_text_path),
    }
    if candidate_score_journal_path is not None:
        outputs["candidate_score_journal"] = str(candidate_score_journal_path)
    if trade_feedback_log_path is not None:
        outputs["trade_decision_log"] = str(trade_feedback_log_path)
        outputs["trade_feedback_log"] = str(trade_feedback_log_path)

    payload = {
        "provider": config.data_sources.provider,
        "as_of_date": args.as_of.isoformat(),
        "preset_names": [preset.name for preset in presets],
        "preset_selection_source": preset_selection_source,
        "portfolio_file": str(args.portfolio_file.resolve()) if args.portfolio_file else None,
        "current_position_count": len(current_positions),
        "current_position_symbols": [position.symbol for position in current_positions],
        "relative_volume_confirmation_required": summary.relative_volume_confirmation_required,
        "relative_volume_policy": summary.relative_volume_policy,
        "relative_volume_policy_by_preset": summary.relative_volume_policy_by_preset,
        "recommended_preset": summary.recommended_preset,
        "market_breadth_pct_above_200ma": (
            summary.market_context.market_breadth_pct_above_200ma
            if summary.market_context is not None
            else None
        ),
        "market_breadth_pct_above_50ma": (
            summary.market_context.market_breadth_pct_above_50ma
            if summary.market_context is not None
            else None
        ),
        "market_breadth_state": (
            summary.market_context.market_breadth_state
            if summary.market_context is not None
            else None
        ),
        "vix_close": (
            summary.market_context.vix_close
            if summary.market_context is not None
            else None
        ),
        "volatility_regime_state": (
            summary.market_context.volatility_regime_state
            if summary.market_context is not None
            else None
        ),
        "volatility_regime_risk_off": (
            summary.market_context.volatility_regime_risk_off
            if summary.market_context is not None
            else None
        ),
        "equity": current_equity,
        "current_drawdown": float(args.current_drawdown),
        "universe_count": summary_payload["universe_count"],
        "candidate_count": len(summary.rows),
        "approved_count": sum(row.status == "approved" for row in summary.rows),
        "rejected_count": sum(row.status == "rejected" for row in summary.rows),
        "order_count": len(execution_batch.orders),
        "top_opportunities": [row.to_dict() for row in summary.rows[:5]],
        "outputs": outputs,
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _handle_review_trade_analytics(args: argparse.Namespace) -> int:
    config = load_app_config(config_dir=args.config_dir)
    window_days = None if args.all_history else args.days
    if window_days is not None and window_days <= 0:
        raise ValueError("--days must be greater than zero unless --all-history is set.")

    trade_feedback_store = TradeFeedbackLogStore(config.project_root)
    feedback_log_path = (args.feedback_log or trade_feedback_store.path).resolve()
    summary = build_trade_review_analytics_summary(
        load_trade_feedback_events(feedback_log_path),
        as_of_date=args.as_of,
        feedback_log_path=feedback_log_path,
        window_days=window_days,
    )

    output_dir = (
        args.output_dir
        or default_trade_review_output_dir(
            config.project_root,
            as_of_date=args.as_of,
            window_days=window_days,
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = write_trade_review_summary(
        summary,
        output_dir / "trade_review_analytics.json",
    )
    brief_output_paths = {
        "trade_review_analytics_json": str(summary_json_path),
        "trade_feedback_log": str(feedback_log_path),
    }
    brief_text_path = write_trade_review_brief(
        summary,
        output_dir / "trade_review_analytics_brief.txt",
        output_paths=brief_output_paths,
    )

    payload = {
        **summary.to_dict(),
        "outputs": {
            "trade_review_analytics_json": str(summary_json_path),
            "trade_review_analytics_brief": str(brief_text_path),
        },
    }
    _print_structured(payload, output_format=args.format)
    return 0


def _default_order_output_path(project_root: Path, as_of_date: date) -> Path:
    output_dir = project_root / "data" / "processed" / "orders"
    return output_dir / f"{as_of_date.isoformat()}_manual_orders.csv"


def _default_daily_output_dir(project_root: Path, as_of_date: date) -> Path:
    return project_root / "data" / "processed" / "daily" / as_of_date.isoformat()


def _default_intraday_portfolio_output_dir(project_root: Path, as_of_date: date) -> Path:
    return project_root / "data" / "processed" / "portfolio_review_intraday" / as_of_date.isoformat()


def _default_live_market_output_dir(project_root: Path, as_of_date: date) -> Path:
    return project_root / "data" / "processed" / "live_market" / as_of_date.isoformat()


def _trade_feedback_timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_trade_feedback_events_safe(
    *,
    project_root: Path,
    events: Sequence[TradeFeedbackEvent],
) -> Path | None:
    if not events:
        return None
    store = TradeFeedbackLogStore(project_root)
    log_path = store.path.resolve()
    try:
        return store.append(events)
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Skipping trade feedback log update at %s: %s", log_path, exc)
        return None


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_numeric_value(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _candidate_decision_id(candidate: RiskAssessedCandidate, *, workflow: str, as_of_date: date) -> str:
    signal_metadata = candidate.signal.metadata
    existing = _clean_optional_text(signal_metadata.get("decision_id"))
    if existing is not None:
        return existing
    return build_trade_decision_id(
        workflow=workflow,
        as_of_date=as_of_date,
        symbol=candidate.signal.symbol,
        signal_date=candidate.signal.date,
        preset_name=_clean_optional_text(signal_metadata.get("preset_name")),
        parameter_id=_clean_optional_text(signal_metadata.get("parameter_id")),
        strategy_name=candidate.signal.strategy_name,
    )


def _attach_candidate_decision_ids(
    candidates: Sequence[RiskAssessedCandidate],
    *,
    workflow: str,
    as_of_date: date,
) -> list[RiskAssessedCandidate]:
    annotated: list[RiskAssessedCandidate] = []
    for candidate in candidates:
        decision_id = _candidate_decision_id(
            candidate,
            workflow=workflow,
            as_of_date=as_of_date,
        )
        signal_metadata = dict(candidate.signal.metadata)
        signal_metadata["decision_id"] = decision_id
        annotated.append(
            replace(
                candidate,
                signal=replace(candidate.signal, metadata=signal_metadata),
            )
        )
    return annotated


def _attach_evaluation_decision_ids(
    evaluations: Sequence[PresetCandidateEvaluation],
    *,
    workflow: str,
    as_of_date: date,
) -> list[PresetCandidateEvaluation]:
    return [
        replace(
            evaluation,
            candidate=_attach_candidate_decision_ids(
                [evaluation.candidate],
                workflow=workflow,
                as_of_date=as_of_date,
            )[0],
        )
        for evaluation in evaluations
    ]


def _candidate_feature_snapshot(candidate: RiskAssessedCandidate) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if candidate.execution_priority is not None:
        snapshot["opportunity_score"] = candidate.execution_priority.opportunity_score
    if candidate.features is not None:
        if candidate.features.breakout_strength is not None:
            snapshot["breakout_strength"] = candidate.features.breakout_strength
        if candidate.features.relative_volume is not None:
            snapshot["relative_volume"] = candidate.features.relative_volume
        if candidate.features.setup_quality_score is not None:
            snapshot["setup_quality_score"] = candidate.features.setup_quality_score
        if candidate.features.setup_persistence_days is not None:
            snapshot["setup_persistence_days"] = candidate.features.setup_persistence_days
        if candidate.features.days_approved is not None:
            snapshot["days_approved"] = candidate.features.days_approved
        if candidate.features.sector_name is not None:
            snapshot["sector_name"] = candidate.features.sector_name
        if candidate.features.industry_name is not None:
            snapshot["industry_name"] = candidate.features.industry_name
        if candidate.features.sector_regime_passed is not None:
            snapshot["sector_regime_passed"] = candidate.features.sector_regime_passed
        if candidate.features.relative_strength_vs_sector is not None:
            snapshot["relative_strength_vs_sector"] = (
                candidate.features.relative_strength_vs_sector
            )
        if candidate.features.is_earnings_risk:
            snapshot["is_earnings_risk"] = True
        if candidate.features.earnings_days_away is not None:
            snapshot["earnings_days_away"] = candidate.features.earnings_days_away
        if candidate.features.market_context is not None:
            market_context = candidate.features.market_context
            if market_context.market_breadth_state is not None:
                snapshot["market_breadth_state"] = market_context.market_breadth_state
            if market_context.market_breadth_pct_above_200ma is not None:
                snapshot["market_breadth_pct_above_200ma"] = (
                    market_context.market_breadth_pct_above_200ma
                )
            if market_context.volatility_regime_state is not None:
                snapshot["volatility_regime_state"] = (
                    market_context.volatility_regime_state
                )
            if market_context.vix_close is not None:
                snapshot["vix_close"] = market_context.vix_close
        if candidate.features.portfolio_heat_context is not None:
            heat_context = candidate.features.portfolio_heat_context
            snapshot["same_sector_position_count"] = heat_context.same_sector_position_count
            snapshot["same_industry_position_count"] = (
                heat_context.same_industry_position_count
            )
    if candidate.portfolio_heat_projection is not None:
        projection = candidate.portfolio_heat_projection
        snapshot["projected_same_sector_position_count"] = (
            projection.projected_same_sector_position_count
        )
        snapshot["projected_same_industry_position_count"] = (
            projection.projected_same_industry_position_count
        )
        if projection.projected_sector_notional_pct is not None:
            snapshot["projected_sector_notional_pct"] = (
                projection.projected_sector_notional_pct
            )
    return snapshot


def _order_decision_id(order: object | None) -> str | None:
    if order is None:
        return None
    metadata = getattr(order, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return _extract_linked_decision_id(metadata)


def _pending_order_trade_id(order: ExecutionOrder) -> str:
    return build_trade_id(
        symbol=order.symbol,
        entry_date=order.date,
        average_entry_price=order.entry_price_hint,
        decision_id=_order_decision_id(order),
    )


def _pending_order_id(order: ExecutionOrder) -> str:
    decision_id = _order_decision_id(order)
    if decision_id is not None:
        identity_payload = {
            "symbol": order.symbol.strip().upper(),
            "date": order.date.isoformat(),
            "action": order.action,
            "decision_id": decision_id,
            "strategy_name": order.strategy_name,
        }
    else:
        identity_payload = {
            "symbol": order.symbol.strip().upper(),
            "date": order.date.isoformat(),
            "action": order.action,
            "quantity": order.quantity,
            "intended_order_type": order.intended_order_type,
            "entry_price_hint": order.entry_price_hint,
            "stop_level": order.stop_level,
            "strategy_name": order.strategy_name,
            "decision_id": None,
        }
    serialized = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"order_{digest}"


def _stage_execution_batch_pending_orders(
    *,
    project_root: Path,
    batch: ExecutionBatch,
) -> tuple[int, Path | None, PendingOrderState]:
    store = PendingOrderStateStore(project_root)
    state = store.load()
    staged_count = 0
    changed = False
    for order in batch.orders:
        decision_id = _order_decision_id(order)
        record = build_pending_order_record_from_execution_order(
            order,
            order_id=_pending_order_id(order),
            created_at=batch.generated_at_utc,
            trade_id=_pending_order_trade_id(order),
            linked_decision_id=decision_id,
        )
        existing = state.orders.get(record.order_id)
        if existing is not None:
            updated_record = replace(
                record,
                created_at=existing.created_at,
                trade_id=existing.trade_id or record.trade_id,
            )
            if updated_record == existing:
                continue
            state = upsert_pending_order_state(
                state,
                updated_record,
                updated_at=batch.generated_at_utc,
            )
            staged_count += 1
            changed = True
            continue
        existing_active_decision_order = _find_active_pending_order_for_decision(
            state,
            symbol=record.symbol,
            decision_id=decision_id,
        )
        if existing_active_decision_order is not None:
            replacement = replace(
                record,
                trade_id=existing_active_decision_order.trade_id or record.trade_id,
            )
            result = replace_pending_order(
                state,
                existing_active_decision_order.order_id,
                replacement=replacement,
                occurred_at=batch.generated_at_utc,
            )
        else:
            result = create_pending_order(state, record)
        if result.state != state:
            staged_count += 1
            changed = True
        state = result.state
    if not changed:
        return staged_count, None, state
    return staged_count, store.save(state), state


def _find_active_pending_order_for_decision(
    state: PendingOrderState,
    *,
    symbol: str,
    decision_id: str | None,
) -> PendingOrderRecord | None:
    if decision_id is None:
        return None
    normalized_symbol = symbol.strip().upper()
    for record in state.active_orders():
        if record.symbol != normalized_symbol:
            continue
        if record.linked_decision_id == decision_id:
            return record
    return None


def _create_broker_adapter(
    broker_name: str,
    *,
    env_file: Path | None,
    target: str | None = None,
) -> BrokerExecutionAdapter:
    return create_broker_execution_adapter(
        broker_name,
        env_file=env_file,
        target=target,
    )


def _broker_cli_control_service(
    *,
    project_root: Path,
    env_file: Path | None,
) -> OperatorControlService:
    return OperatorControlService(
        project_root=project_root,
        env_file=env_file,
        broker_adapter_factory=_broker_cli_adapter_factory,
    )


def _broker_cli_adapter_factory(
    broker_name: str,
    *,
    env_file: Path | None = None,
    target: str | None = None,
) -> BrokerExecutionAdapter:
    try:
        return _create_broker_adapter(
            broker_name,
            env_file=env_file,
            target=target,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'target'" not in str(exc):
            raise
        return _create_broker_adapter(
            broker_name,
            env_file=env_file,
        )


def _trade_feedback_event_count(project_root: Path) -> int:
    return len(load_trade_feedback_events(default_trade_feedback_log_path(project_root)))


def _new_trade_feedback_events_since(
    project_root: Path,
    previous_count: int,
) -> tuple[TradeFeedbackEvent, ...]:
    events = load_trade_feedback_events(default_trade_feedback_log_path(project_root))
    return tuple(events[previous_count:])


def _save_pending_order_state_if_changed(
    store: PendingOrderStateStore,
    *,
    previous_state: PendingOrderState,
    updated_state: PendingOrderState,
) -> Path | None:
    if updated_state == previous_state:
        return None
    return store.save(updated_state)


def _mark_pending_order_broker_cancel_pending(
    state: PendingOrderState,
    *,
    order_id: str,
    occurred_at: str,
) -> PendingOrderState:
    record = state.orders.get(order_id)
    if record is None:
        raise ValueError(f"Unknown pending order_id '{order_id}'.")
    updated_record = replace(
        record,
        broker_status="pending_cancel",
        last_broker_update_at=occurred_at,
    )
    if updated_record == record:
        return state
    return upsert_pending_order_state(
        state,
        updated_record,
        updated_at=occurred_at,
    )


def _broker_update_identity_key(update: BrokerOrderUpdate) -> tuple[str, str]:
    return (
        update.broker_order_id,
        update.client_order_id or "",
    )


def _pending_order_matches_broker_update(
    record: PendingOrderRecord,
    update: BrokerOrderUpdate,
) -> bool:
    if record.broker_order_id is not None and record.broker_order_id == update.broker_order_id:
        return True
    candidate_client_order_id = record.client_order_id or record.order_id
    if update.client_order_id is not None and candidate_client_order_id == update.client_order_id:
        return True
    return False


def _select_pending_orders_for_broker_sync(
    state: PendingOrderState,
    *,
    order_ids: Sequence[str] | None,
) -> tuple[PendingOrderRecord, ...]:
    requested_ids = {order_id.strip() for order_id in (order_ids or ()) if order_id.strip()}
    selected = [
        record
        for record in state.active_orders()
        if not requested_ids or record.order_id in requested_ids
    ]
    return tuple(selected)


def _load_broker_updates_for_pending_orders(
    adapter: BrokerExecutionAdapter,
    orders: Sequence[PendingOrderRecord],
    *,
    use_open_order_scan: bool = False,
) -> tuple[tuple[BrokerOrderUpdate, ...], tuple[str, ...]]:
    updates_by_identity: dict[tuple[str, str], BrokerOrderUpdate] = {}
    warnings: list[str] = []
    broker_open_updates: tuple[BrokerOrderUpdate, ...] = ()
    if use_open_order_scan:
        try:
            broker_open_updates = adapter.list_open_orders()
        except BrokerExecutionError as exc:
            warnings.append(f"Broker open-order scan unavailable: {exc}")
        else:
            for update in broker_open_updates:
                updates_by_identity[_broker_update_identity_key(update)] = update
    for record in orders:
        if use_open_order_scan and any(
            _pending_order_matches_broker_update(record, update)
            for update in broker_open_updates
        ):
            continue
        try:
            if record.broker_order_id is not None:
                update = adapter.get_order(record.broker_order_id)
                updates_by_identity[_broker_update_identity_key(update)] = update
                continue
            client_order_id = record.client_order_id or record.order_id
            update = adapter.get_order(client_order_id=client_order_id)
            updates_by_identity[_broker_update_identity_key(update)] = update
        except BrokerExecutionError as exc:
            warnings.append(
                f"Broker update unavailable for local pending order '{record.order_id}': {exc}"
            )
    return tuple(updates_by_identity.values()), tuple(warnings)


def _replacement_pending_order_id(
    record: PendingOrderRecord,
    *,
    quantity: int | None,
    limit_price: float | None,
    stop_price: float | None,
) -> str:
    identity_payload = {
        "existing_order_id": record.order_id,
        "quantity": quantity,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "broker_order_id": record.broker_order_id,
    }
    serialized = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"order_{digest}"


def _build_replacement_pending_order_record(
    existing: PendingOrderRecord,
    *,
    local_order_id: str,
    occurred_at: str,
    quantity: int | None,
    limit_price: float | None,
    stop_price: float | None,
) -> PendingOrderRecord:
    requested_quantity = quantity if quantity is not None else existing.requested_quantity
    if requested_quantity <= existing.filled_quantity:
        raise ValueError(
            "Replacement quantity must exceed the already filled quantity for the order."
        )
    requested_price = (
        limit_price
        if existing.order_type in {"LIMIT", "STOP_LIMIT"} and limit_price is not None
        else existing.requested_price
    )
    resolved_entry_hint = requested_price or existing.entry_hint
    remaining_quantity = requested_quantity - existing.filled_quantity
    reservation_price = requested_price or existing.entry_hint or 0.0
    return PendingOrderRecord(
        order_id=local_order_id,
        symbol=existing.symbol,
        side=existing.side,
        order_type=existing.order_type,
        created_at=occurred_at,
        status="pending",
        requested_quantity=requested_quantity,
        filled_quantity=existing.filled_quantity,
        requested_price=requested_price,
        entry_hint=resolved_entry_hint,
        stop_price=stop_price if stop_price is not None else existing.stop_price,
        reserved_notional=(
            remaining_quantity * reservation_price
            if existing.side == "BUY"
            else 0.0
        ),
        reserved_slot=existing.side == "BUY" and remaining_quantity > 0,
        preset_name=existing.preset_name,
        trade_id=existing.trade_id,
        linked_decision_id=existing.linked_decision_id,
        sector_name=existing.sector_name,
        industry_name=existing.industry_name,
        client_order_id=local_order_id,
        metadata=dict(existing.metadata),
    )


def _broker_submission_feedback_events(
    *,
    workflow: str,
    state: PendingOrderState,
    submitted_updates: Sequence[BrokerOrderUpdate],
    lifecycle_events: Sequence[Any],
) -> tuple[TradeFeedbackEvent, ...]:
    events: list[TradeFeedbackEvent] = []
    for update in submitted_updates:
        record = _find_pending_order_record_for_broker_update(state, update)
        if record is None:
            continue
        timestamp_utc = (
            update.submitted_at
            or update.updated_at
            or record.last_broker_update_at
            or record.created_at
        )
        event = _broker_trade_feedback_event(
            event_type="broker_submitted",
            workflow=workflow,
            record=record,
            timestamp_utc=timestamp_utc,
        )
        if event is not None:
            events.append(event)
    events.extend(
        _broker_reconciliation_feedback_events(
            workflow=workflow,
            state=state,
            lifecycle_events=lifecycle_events,
            matched_updates=submitted_updates,
        )
    )
    return tuple(events)


def _broker_reconciliation_feedback_events(
    *,
    workflow: str,
    state: PendingOrderState,
    lifecycle_events: Sequence[Any],
    matched_updates: Sequence[BrokerOrderUpdate],
) -> tuple[TradeFeedbackEvent, ...]:
    updates_by_order_id: dict[str, BrokerOrderUpdate] = {}
    for update in matched_updates:
        record = _find_pending_order_record_for_broker_update(state, update)
        if record is not None:
            updates_by_order_id[record.order_id] = update
    events: list[TradeFeedbackEvent] = []
    for lifecycle_event in lifecycle_events:
        order_id = getattr(lifecycle_event, "order_id", None)
        if not isinstance(order_id, str):
            continue
        record = state.orders.get(order_id)
        if record is None:
            continue
        update = updates_by_order_id.get(order_id)
        event_type = _broker_feedback_event_type_for_lifecycle(
            getattr(lifecycle_event, "event_type", None)
        )
        if event_type is None:
            continue
        timestamp_utc = (
            record.last_broker_update_at
            or (update.updated_at if update is not None else None)
            or (update.submitted_at if update is not None else None)
            or getattr(lifecycle_event, "occurred_at", None)
            or record.created_at
        )
        event = _broker_trade_feedback_event(
            event_type=event_type,
            workflow=workflow,
            record=record,
            timestamp_utc=timestamp_utc,
        )
        if event is not None:
            events.append(event)
    return tuple(events)


def _broker_replacement_feedback_events(
    *,
    state: PendingOrderState,
    replacement_update: BrokerOrderUpdate,
    replacement_order_id: str,
) -> tuple[TradeFeedbackEvent, ...]:
    record = state.orders.get(replacement_order_id)
    if record is None:
        return ()
    timestamp_utc = (
        replacement_update.updated_at
        or replacement_update.submitted_at
        or record.last_broker_update_at
        or record.created_at
    )
    event = _broker_trade_feedback_event(
        event_type="broker_replaced",
        workflow="replace-broker-order",
        record=record,
        timestamp_utc=timestamp_utc,
    )
    return (event,) if event is not None else ()


def _find_pending_order_record_for_broker_update(
    state: PendingOrderState,
    update: BrokerOrderUpdate,
) -> PendingOrderRecord | None:
    for record in state.orders.values():
        if record.broker_order_id == update.broker_order_id:
            return record
    if update.client_order_id is not None:
        record = state.orders.get(update.client_order_id)
        if record is not None:
            return record
        for existing in state.orders.values():
            if existing.client_order_id == update.client_order_id:
                return existing
    return None


def _broker_feedback_event_type_for_lifecycle(
    lifecycle_event_type: object,
) -> str | None:
    if lifecycle_event_type == "filled":
        return "executed"
    if lifecycle_event_type == "cancelled":
        return "broker_cancelled"
    if lifecycle_event_type == "expired":
        return "broker_expired"
    if lifecycle_event_type == "replaced":
        return "broker_replaced"
    return None


def _broker_trade_feedback_event(
    *,
    event_type: str,
    workflow: str,
    record: PendingOrderRecord,
    timestamp_utc: str,
) -> TradeFeedbackEvent | None:
    as_of_date = _broker_event_date(timestamp_utc)
    if as_of_date is None:
        return None
    quantity = record.filled_quantity if event_type == "executed" else record.requested_quantity
    return TradeFeedbackEvent(
        event_type=event_type,
        workflow=workflow,
        symbol=record.symbol,
        as_of_date=as_of_date,
        timestamp_utc=timestamp_utc,
        decision_id=record.linked_decision_id,
        trade_id=record.trade_id,
        linked_decision_id=record.linked_decision_id,
        preset_name=record.preset_name,
        strategy_name=_extract_strategy_name_from_metadata(record.metadata),
        suggested_action=record.side,
        approved=True,
        quantity=quantity,
        entry_price_hint=(
            record.average_fill_price
            if event_type == "executed"
            else record.requested_price or record.entry_hint
        ),
        stop_price=record.stop_price,
        notional_value=(
            quantity
            * (
                record.average_fill_price
                if event_type == "executed"
                else record.requested_price or record.entry_hint or 0.0
            )
            if quantity is not None
            else None
        ),
        metadata={
            "order_id": record.order_id,
            "broker_name": record.broker_name,
            "broker_order_id": record.broker_order_id,
            "client_order_id": record.client_order_id,
            "broker_status": record.broker_status,
        },
    )


def _broker_event_date(timestamp_utc: str | None) -> date | None:
    cleaned = _clean_optional_text(timestamp_utc)
    if cleaned is None:
        return None
    normalized = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _candidate_trade_feedback_event(
    candidate: RiskAssessedCandidate,
    *,
    workflow: str,
    as_of_date: date,
    timestamp_utc: str,
) -> TradeFeedbackEvent:
    signal_metadata = candidate.signal.metadata
    decision_id = _candidate_decision_id(candidate, workflow=workflow, as_of_date=as_of_date)
    priority = candidate.execution_priority
    return TradeFeedbackEvent(
        event_type="decision",
        workflow=workflow,
        symbol=candidate.signal.symbol,
        as_of_date=as_of_date,
        timestamp_utc=timestamp_utc,
        decision_id=decision_id,
        preset_name=_clean_optional_text(signal_metadata.get("preset_name")),
        parameter_id=_clean_optional_text(signal_metadata.get("parameter_id")),
        strategy_name=candidate.signal.strategy_name,
        suggested_action=candidate.signal.side,
        approved=candidate.approved,
        rejection_reasons=tuple(candidate.rejection_reasons),
        queue_rank=priority.queue_rank if priority is not None else None,
        priority_bucket=priority.priority_bucket if priority is not None else None,
        actionable_now=priority.actionable_now if priority is not None else None,
        quantity=(candidate.sizing.shares if candidate.sizing.shares > 0 else None),
        entry_price_hint=candidate.entry_price,
        stop_price=candidate.stop_price,
        notional_value=(
            candidate.sizing.notional_value
            if candidate.sizing.notional_value > 0
            else None
        ),
        feature_snapshot=_candidate_feature_snapshot(candidate),
    )


def _portfolio_review_decision_id(
    row: PortfolioReviewRow,
    *,
    workflow: str,
) -> str:
    existing = _clean_optional_text(row.metadata.get("decision_id"))
    if existing is not None:
        return existing
    return build_trade_decision_id(
        workflow=workflow,
        as_of_date=row.date,
        symbol=row.symbol,
        preset_name=row.preset_name,
        review_timestamp=_clean_optional_text(row.metadata.get("latest_bar_time")),
    )


def _portfolio_review_feature_snapshot(row: PortfolioReviewRow) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "above_entry": row.above_entry,
        "unrealized_pl_pct": row.unrealized_pl_pct,
    }
    if row.distance_to_stop_pct is not None:
        snapshot["distance_to_stop_pct"] = row.distance_to_stop_pct
    for key in (
        "days_since_new_high",
        "relative_strength_return_diff",
        "sector_name",
        "earnings_days_away",
        "session_vwap",
        "session_high_giveback_pct",
        "intraday_relative_strength_diff",
        "consecutive_days_below_entry",
        "consecutive_watch_closely_days",
        "consecutive_polls_below_vwap",
        "repeated_watch_closely_count",
    ):
        value = row.metadata.get(key)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _trade_outcome_snapshot_from_row(row: PortfolioReviewRow) -> TradeOutcomeSnapshot | None:
    raw_snapshot = row.metadata.get(_TRADE_OUTCOME_SNAPSHOT_METADATA_KEY)
    if not isinstance(raw_snapshot, Mapping):
        return None
    return TradeOutcomeSnapshot.from_mapping(raw_snapshot)


def _portfolio_review_position_metadata(row: PortfolioReviewRow) -> Mapping[str, Any]:
    raw_metadata = row.metadata.get("position_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        return {}
    return raw_metadata


def _strip_position_trajectory_observation_from_row(
    row: PortfolioReviewRow,
) -> PortfolioReviewRow:
    return replace(
        row,
        metadata={
            key: value
            for key, value in row.metadata.items()
            if key != _POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY
        },
    )


def _coerce_portfolio_review_row_state(
    value: PortfolioReviewRow | PortfolioReviewRowState,
) -> PortfolioReviewRowState:
    if isinstance(value, PortfolioReviewRowState):
        return PortfolioReviewRowState(
            row=_strip_position_trajectory_observation_from_row(value.row),
            position_trajectory_observation=value.position_trajectory_observation,
        )
    if isinstance(value, PortfolioReviewRow):
        return PortfolioReviewRowState(
            row=_strip_position_trajectory_observation_from_row(value),
            position_trajectory_observation=_position_trajectory_observation_from_row(value),
        )
    raise TypeError(
        "portfolio review row builder must return PortfolioReviewRow or "
        "PortfolioReviewRowState."
    )


def _strip_internal_portfolio_review_row(row: PortfolioReviewRow) -> PortfolioReviewRow:
    return replace(
        row,
        metadata={
            key: value
            for key, value in row.metadata.items()
            if key not in (
                _POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY,
                _TRADE_OUTCOME_SNAPSHOT_METADATA_KEY,
            )
        },
    )


def _strip_internal_portfolio_review_report(
    report: PortfolioReviewReport,
) -> PortfolioReviewReport:
    return replace(
        report,
        rows=tuple(_strip_internal_portfolio_review_row(row) for row in report.rows),
    )


def _portfolio_review_decision_event(
    row: PortfolioReviewRow,
    *,
    workflow: str,
    timestamp_utc: str,
) -> TradeFeedbackEvent:
    position_metadata = _portfolio_review_position_metadata(row)
    trade_id = _clean_optional_text(position_metadata.get("trade_id"))
    linked_decision_id = (
        _clean_optional_text(position_metadata.get("linked_decision_id"))
        or _extract_linked_decision_id(position_metadata)
    )
    return TradeFeedbackEvent(
        event_type="decision",
        workflow=workflow,
        symbol=row.symbol,
        as_of_date=row.date,
        timestamp_utc=timestamp_utc,
        decision_id=_portfolio_review_decision_id(row, workflow=workflow),
        trade_id=trade_id,
        linked_decision_id=linked_decision_id,
        preset_name=row.preset_name,
        suggested_action=row.suggested_action,
        quantity=row.quantity,
        entry_price_hint=row.average_entry_price,
        stop_price=row.current_stop,
        feature_snapshot=_portfolio_review_feature_snapshot(row),
        metadata={"rationale": row.rationale},
    )


def _portfolio_review_outcome_event(
    row: PortfolioReviewRow,
    *,
    workflow: str,
    timestamp_utc: str,
) -> TradeFeedbackEvent | None:
    outcome_snapshot = _trade_outcome_snapshot_from_row(row)
    if outcome_snapshot is None:
        return None
    position_metadata = _portfolio_review_position_metadata(row)
    trade_id = _clean_optional_text(position_metadata.get("trade_id"))
    if trade_id is None:
        LOGGER.debug(
            "Skipping portfolio review outcome event for %s on %s because trade_id is absent.",
            row.symbol,
            row.date.isoformat(),
        )
        return None
    linked_decision_id = (
        _clean_optional_text(position_metadata.get("linked_decision_id"))
        or _extract_linked_decision_id(position_metadata)
    )
    return TradeFeedbackEvent(
        event_type="outcome",
        workflow=workflow,
        symbol=row.symbol,
        as_of_date=row.date,
        timestamp_utc=timestamp_utc,
        decision_id=linked_decision_id,
        trade_id=trade_id,
        linked_decision_id=linked_decision_id,
        preset_name=row.preset_name,
        suggested_action=row.suggested_action,
        quantity=row.quantity,
        entry_price_hint=row.average_entry_price,
        stop_price=row.current_stop,
        feature_snapshot=_portfolio_review_feature_snapshot(row),
        outcome_snapshot=outcome_snapshot,
    )


def _extract_linked_decision_id(metadata: Mapping[str, Any]) -> str | None:
    for key in ("linked_decision_id", "decision_id"):
        value = _clean_optional_text(metadata.get(key))
        if value is not None:
            return value
    signal_metadata = metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        value = _clean_optional_text(signal_metadata.get("decision_id"))
        if value is not None:
            return value
    execution_metadata = metadata.get("execution_metadata")
    if isinstance(execution_metadata, Mapping):
        for key in ("decision_id", "linked_decision_id"):
            value = _clean_optional_text(execution_metadata.get(key))
            if value is not None:
                return value
        nested_signal_metadata = execution_metadata.get("signal_metadata")
        if isinstance(nested_signal_metadata, Mapping):
            value = _clean_optional_text(nested_signal_metadata.get("decision_id"))
            if value is not None:
                return value
    return None


def _extract_strategy_name_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    direct = _clean_optional_text(metadata.get("strategy_name"))
    if direct is not None:
        return direct
    signal_metadata = metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        direct = _clean_optional_text(signal_metadata.get("strategy_name"))
        if direct is not None:
            return direct
    execution_metadata = metadata.get("execution_metadata")
    if isinstance(execution_metadata, Mapping):
        direct = _clean_optional_text(execution_metadata.get("strategy_name"))
        if direct is not None:
            return direct
        nested_signal_metadata = execution_metadata.get("signal_metadata")
        if isinstance(nested_signal_metadata, Mapping):
            direct = _clean_optional_text(nested_signal_metadata.get("strategy_name"))
            if direct is not None:
                return direct
    return None


def _resolve_execution_decision_id(
    *,
    explicit_decision_id: object,
    metadata: Mapping[str, Any],
    existing_position: ExistingPosition | None,
) -> str | None:
    decision_id = _clean_optional_text(explicit_decision_id)
    if decision_id is not None:
        return decision_id
    decision_id = _extract_linked_decision_id(metadata)
    if decision_id is not None:
        return decision_id
    if existing_position is not None:
        decision_id = _extract_linked_decision_id(existing_position.metadata)
        if decision_id is not None:
            return decision_id
    return None


def _resolve_execution_trade_id(
    *,
    explicit_trade_id: object,
    metadata: Mapping[str, Any],
    existing_position: ExistingPosition | None,
    symbol: str,
    entry_date: date | None,
    average_entry_price: float | None,
    decision_id: str | None,
) -> str:
    trade_id = _clean_optional_text(explicit_trade_id)
    if trade_id is not None:
        return trade_id
    trade_id = _clean_optional_text(metadata.get("trade_id"))
    if trade_id is not None:
        return trade_id
    if existing_position is not None:
        trade_id = _clean_optional_text(existing_position.metadata.get("trade_id"))
        if trade_id is not None:
            return trade_id
    return build_trade_id(
        symbol=symbol,
        entry_date=entry_date,
        average_entry_price=average_entry_price,
        decision_id=decision_id,
    )

def _benchmark_symbol_override(raw_value: object) -> str | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    return raw_value.strip().upper()


def _run_daily_summary_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    provider: DailyBarProvider,
    current_positions: list[ExistingPosition] | None = None,
) -> dict[str, object]:
    presets, preset_selection_source = _resolve_daily_summary_presets(args, config)
    resolved_current_positions = (
        _load_current_positions(args.portfolio_file)
        if current_positions is None
        else current_positions
    )
    open_position_symbols = {
        position.symbol.strip().upper()
        for position in resolved_current_positions
        if position.symbol.strip()
    }

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
        enforce_max_symbols=False,
    )

    fetch_start = _comparison_warmup_start(
        start_date=args.as_of,
        atr_window=config.strategy.risk.atr_length,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        presets=presets,
        max_relative_volume_window=_max_resolved_relative_volume_window_for_presets(
            config=config,
            presets=presets,
            require_relative_volume_confirmation=args.require_relative_volume,
            enable_regime_filter=not args.disable_regime_filter,
        ),
        max_sector_relative_strength_window=(
            config.strategy.signals.sector_relative_strength_window
        ),
        market_breadth_long_window=(
            200
            if config.strategy.signals.market_breadth_entry_floor_200ma is not None
            else 1
        ),
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
            LOGGER.warning(
                "Skipping %s because no bars were returned in the requested range.",
                member.symbol,
            )
            continue
        symbol_frames[member.symbol] = bars

    benchmark_frame = None
    benchmark_symbol = None
    benchmark_override = _benchmark_symbol_override(args.benchmark_symbol)
    if not args.disable_regime_filter and universe_members:
        benchmark_symbol = benchmark_override or config.strategy.signals.benchmark_symbol
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
    pending_order_state, pending_order_reservation_state = (
        _build_pending_order_reservation_state(
            project_root=config.project_root,
            current_equity=current_equity,
            current_positions=resolved_current_positions,
            constraints=constraints,
        )
    )
    evaluations: list[PresetCandidateEvaluation] = []
    no_signal_symbols_by_preset: dict[str, list[str]] = {
        preset.name: []
        for preset in presets
    }
    earnings_contexts = _load_earnings_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        symbols=[member.symbol for member in universe_members],
        as_of_date=args.as_of,
        earnings_watch_days=config.strategy.signals.earnings_watch_days,
        refresh_cache=args.refresh_cache,
        log_label="daily-summary",
    )
    current_position_classifications = _load_position_sector_classifications(
        resolved_current_positions,
        config=config,
        env_file=getattr(args, "env_file", None),
        as_of_date=args.as_of,
        refresh_cache=args.refresh_cache,
        log_label="daily-summary",
    )
    portfolio_holding_exposures = _build_portfolio_holding_exposures(
        resolved_current_positions,
        current_position_classifications,
    )
    portfolio_holding_exposures.extend(
        build_pending_order_holding_exposures(
            pending_order_state,
            current_positions=resolved_current_positions,
        )
    )
    sector_contexts = _load_sector_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        provider=provider,
        symbol_frames=symbol_frames,
        as_of_date=args.as_of,
        history_start=sector_context_history_start(
            args.as_of,
            benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=config.strategy.signals.sector_relative_strength_window,
        ),
        benchmark_sma_fast=config.strategy.signals.benchmark_sma_fast,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        relative_strength_window=config.strategy.signals.sector_relative_strength_window,
        refresh_cache=args.refresh_cache,
        log_label="daily-summary",
    )
    breadth_metrics = _compute_entry_market_breadth(
        symbol_frames,
        as_of_date=args.as_of,
        market_breadth_entry_floor_200ma=(
            config.strategy.signals.market_breadth_entry_floor_200ma
        ),
        log_label="daily-summary",
    )
    summary_market_context = build_market_context(
        as_of_date=args.as_of,
        benchmark_symbol=benchmark_symbol,
        market_breadth_pct_above_200ma=(
            breadth_metrics["market_breadth_pct_above_200ma"]
        ),
        market_breadth_pct_above_50ma=(
            breadth_metrics["market_breadth_pct_above_50ma"]
        ),
        market_breadth_state=breadth_metrics["market_breadth_state"],
        **_load_volatility_context(
            provider=provider,
            as_of_date=args.as_of,
            vix_caution_threshold=config.strategy.signals.vix_caution_threshold,
            vix_entry_block_threshold=(
                config.strategy.signals.vix_entry_block_threshold
            ),
            refresh_cache=args.refresh_cache,
            log_label="daily-summary",
        ),
    )
    candidate_score_journal_store = CandidateScoreJournalStore(config.project_root)
    candidate_score_journal = candidate_score_journal_store.load()

    for preset in presets:
        strategy_settings = BreakoutMomentumSettings.from_configs(
            config.strategy.signals,
            config.strategy.risk,
            require_relative_volume_confirmation=args.require_relative_volume,
            enable_regime_filter=not args.disable_regime_filter,
        )
        strategy_settings = preset.apply_to_settings(
            strategy_settings,
            force_require_relative_volume_confirmation=(
                True if args.require_relative_volume else None
            ),
        )
        if benchmark_override is not None:
            strategy_settings = replace(
                strategy_settings,
                benchmark_symbol=benchmark_override,
            )
        market_context = build_market_context(
            as_of_date=args.as_of,
            benchmark_symbol=(
                strategy_settings.benchmark_symbol
                if strategy_settings.enable_regime_filter
                else None
            ),
            regime_filter_mode=strategy_settings.regime_filter_mode,
            market_breadth_pct_above_200ma=(
                breadth_metrics["market_breadth_pct_above_200ma"]
            ),
            market_breadth_pct_above_50ma=(
                breadth_metrics["market_breadth_pct_above_50ma"]
            ),
            market_breadth_state=breadth_metrics["market_breadth_state"],
            vix_close=summary_market_context.vix_close,
            vix_sma_short=summary_market_context.vix_sma_short,
            vix_sma_long=summary_market_context.vix_sma_long,
            volatility_regime_state=summary_market_context.volatility_regime_state,
            volatility_regime_risk_off=summary_market_context.volatility_regime_risk_off,
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
                has_open_position=member.symbol.strip().upper() in open_position_symbols,
                symbol=member.symbol,
            )
            if signal is None:
                no_signal_symbols_by_preset[preset.name].append(member.symbol)
                continue

            earnings_context = earnings_contexts.get(member.symbol.strip().upper())
            sector_context = sector_contexts.get(member.symbol.strip().upper())
            portfolio_heat_context = build_portfolio_heat_context(
                portfolio_holding_exposures,
                candidate_sector=(
                    sector_context.sector_name if sector_context is not None else None
                ),
                candidate_industry=(
                    sector_context.industry_name if sector_context is not None else None
                ),
                max_positions_per_sector=strategy_settings.max_positions_per_sector,
                max_same_industry_positions=strategy_settings.max_same_industry_positions,
                max_sector_notional_pct=strategy_settings.max_sector_notional_pct,
            )
            if strategy_settings.require_sector_regime_for_entries and sector_context is None:
                _warn_missing_sector_context_for_entry(
                    member.symbol,
                    log_label="daily-summary",
                    preset_name=preset.name,
                )
            signal_metadata = dict(signal.metadata)
            signal_metadata.update(earnings_context_metadata(earnings_context))
            signal_metadata.update(sector_context_metadata(sector_context))
            signal_metadata.update(
                market_context_metadata(
                    market_context,
                    market_breadth_entry_floor_200ma=(
                        strategy_settings.market_breadth_entry_floor_200ma
                    ),
                    vix_caution_threshold=strategy_settings.vix_caution_threshold,
                    vix_entry_block_threshold=(
                        strategy_settings.vix_entry_block_threshold
                    ),
                )
            )
            signal_metadata["preset_name"] = preset.name
            signal_metadata["parameter_id"] = preset.parameter_id
            signal = replace(
                signal,
                strategy_name=f"{signal.strategy_name}:{preset.name}",
                metadata=signal_metadata,
            )
            candidate_features = build_candidate_features(
                signal,
                as_of_date=args.as_of,
                earnings_context=earnings_context,
                sector_context=sector_context,
                market_context=market_context,
                portfolio_heat_context=portfolio_heat_context,
            )
            assessed_candidate = assess_signal_candidate(
                signal,
                current_equity=current_equity,
                base_risk_per_trade=preset.risk_per_trade,
                constraints=constraints,
                candidate_features=candidate_features,
                current_positions=resolved_current_positions,
                current_drawdown=float(args.current_drawdown),
                earnings_date=(
                    earnings_context.earnings_date
                    if earnings_context is not None
                    else None
                ),
                earnings_days_away=(
                    earnings_context.earnings_days_away
                    if earnings_context is not None
                    else None
                ),
                earnings_entry_block_days=strategy_settings.earnings_entry_block_days,
                sector_name=(
                    sector_context.sector_name
                    if sector_context is not None
                    else None
                ),
                sector_etf_symbol=(
                    sector_context.sector_etf_symbol
                    if sector_context is not None
                    else None
                ),
                sector_regime_passed=(
                    sector_context.sector_regime_passed
                    if sector_context is not None
                    else None
                ),
                relative_strength_vs_sector=(
                    sector_context.relative_strength_vs_sector
                    if sector_context is not None
                    else None
                ),
                sector_relative_strength_window=(
                    sector_context.relative_strength_window
                    if sector_context is not None
                    else None
                ),
                market_breadth_entry_floor_200ma=(
                    strategy_settings.market_breadth_entry_floor_200ma
                ),
                vix_entry_block_threshold=(
                    strategy_settings.vix_entry_block_threshold
                ),
                require_sector_regime_for_entries=(
                    strategy_settings.require_sector_regime_for_entries
                ),
                sector_relative_strength_entry_reject_threshold=(
                    strategy_settings.sector_relative_strength_entry_reject_threshold
                ),
            )
            evaluations.append(
                PresetCandidateEvaluation(
                    preset_name=preset.name,
                    parameter_id=preset.parameter_id,
                    candidate=assessed_candidate,
                )
            )
            _roll_forward_portfolio_holding_exposures(
                portfolio_holding_exposures,
                assessed_candidate,
                sector_context=sector_context,
            )

    symbol_observations = _candidate_journal_observations(evaluations)
    evaluations = _apply_candidate_memory_to_evaluations(
        evaluations,
        observations=symbol_observations,
        candidate_score_journal=candidate_score_journal,
        as_of_date=args.as_of,
    )
    prioritized_candidates = annotate_execution_priority(
        [evaluation.candidate for evaluation in evaluations],
        current_equity=current_equity,
        current_positions=resolved_current_positions,
        constraints=constraints,
        reservation_state=pending_order_reservation_state,
    )
    evaluations = [
        replace(evaluation, candidate=candidate)
        for evaluation, candidate in zip(evaluations, prioritized_candidates)
    ]
    evaluations = _attach_evaluation_decision_ids(
        evaluations,
        workflow="daily-summary",
        as_of_date=args.as_of,
    )
    ranked_evaluations = rank_preset_candidate_evaluations(
        evaluations,
        current_equity=current_equity,
    )
    ranked_symbol_observations = _candidate_journal_observations(
        ranked_evaluations,
        include_ranks=True,
    )
    candidate_score_journal = update_candidate_score_journal(
        candidate_score_journal,
        observations=ranked_symbol_observations,
        as_of_date=args.as_of,
    )
    journal_path = candidate_score_journal_store.save(candidate_score_journal)
    execution_batch = ManualExecutor().build_execution_batch(
        [evaluation.candidate for evaluation in ranked_evaluations],
        as_of_date=args.as_of,
    )
    summary = build_daily_research_summary(
        as_of_date=args.as_of,
        execution_batch=execution_batch,
        evaluations=evaluations,
        selected_presets=presets,
        universe_symbols=[member.symbol for member in universe_members],
        current_positions=resolved_current_positions,
        current_equity=current_equity,
        no_signal_symbols_by_preset=no_signal_symbols_by_preset,
        benchmark_symbol=benchmark_symbol,
        market_context=summary_market_context,
        preset_selection_source=preset_selection_source,
        force_require_relative_volume_confirmation=bool(args.require_relative_volume),
        pending_order_summary=pending_order_reservation_state.to_dict(),
    )
    return {
        "summary": summary,
        "execution_batch": execution_batch,
        "ranked_evaluations": ranked_evaluations,
        "presets": presets,
        "preset_selection_source": preset_selection_source,
        "current_positions": resolved_current_positions,
        "current_equity": current_equity,
        "pending_order_summary": pending_order_reservation_state.to_dict(),
        "candidate_score_journal_path": journal_path,
    }


def _candidate_journal_observations(
    evaluations: Sequence[PresetCandidateEvaluation],
    *,
    include_ranks: bool = False,
) -> dict[str, CandidateJournalObservation]:
    observations: dict[str, CandidateJournalObservation] = {}
    ranked_positions: dict[str, int] = {}

    if include_ranks:
        for rank, evaluation in enumerate(evaluations, start=1):
            symbol = evaluation.candidate.signal.symbol.strip().upper()
            ranked_positions[symbol] = min(rank, ranked_positions.get(symbol, rank))

    for evaluation in evaluations:
        candidate = evaluation.candidate
        signal = candidate.signal
        symbol = signal.symbol.strip().upper()
        existing_observation = observations.get(symbol)
        breakout_strength = _signal_breakout_strength(candidate)
        relative_volume = _optional_float(signal.metadata.get("relative_volume"))
        sector_etf_symbol = _optional_text(signal.metadata.get("sector_etf_symbol"))
        sector_regime_passed = _optional_bool(signal.metadata.get("sector_regime_passed"))
        if existing_observation is None:
            observations[symbol] = CandidateJournalObservation(
                symbol=symbol,
                approved_today=candidate.approved,
                breakout_strength=breakout_strength,
                relative_volume=relative_volume,
                sector_etf_symbol=sector_etf_symbol,
                sector_regime_passed=sector_regime_passed,
                best_rank_today=ranked_positions.get(symbol),
            )
            continue

        observations[symbol] = CandidateJournalObservation(
            symbol=symbol,
            approved_today=(existing_observation.approved_today or candidate.approved),
            breakout_strength=_peak_optional_float(
                existing_observation.breakout_strength,
                breakout_strength,
            ),
            relative_volume=_peak_optional_float(
                existing_observation.relative_volume,
                relative_volume,
            ),
            sector_etf_symbol=sector_etf_symbol or existing_observation.sector_etf_symbol,
            sector_regime_passed=(
                sector_regime_passed
                if sector_regime_passed is not None
                else existing_observation.sector_regime_passed
            ),
            best_rank_today=ranked_positions.get(symbol, existing_observation.best_rank_today),
        )

    return observations


def _apply_candidate_memory_to_evaluations(
    evaluations: Sequence[PresetCandidateEvaluation],
    *,
    observations: Mapping[str, CandidateJournalObservation],
    candidate_score_journal: CandidateScoreJournal,
    as_of_date: date,
) -> list[PresetCandidateEvaluation]:
    enriched_evaluations: list[PresetCandidateEvaluation] = []
    for evaluation in evaluations:
        candidate = evaluation.candidate
        signal = candidate.signal
        symbol = signal.symbol.strip().upper()
        observation = observations.get(symbol)
        if observation is None:
            enriched_evaluations.append(evaluation)
            continue
        memory_features = build_candidate_memory_features(
            candidate_score_journal.entries.get(symbol),
            observation=observation,
            as_of_date=as_of_date,
        )
        enriched_signal = replace(
            signal,
            metadata={**dict(signal.metadata), **memory_features},
        )
        enriched_evaluations.append(
            replace(
                evaluation,
                candidate=replace(candidate, signal=enriched_signal),
            )
        )
    return enriched_evaluations


def _signal_breakout_strength(candidate: RiskAssessedCandidate) -> float | None:
    entry_price = _optional_float(candidate.entry_price)
    if entry_price is None or entry_price <= 0:
        return None
    prior_high = _optional_float(candidate.signal.metadata.get("prior_high"))
    if prior_high is None or prior_high <= 0:
        return None
    return max((entry_price - prior_high) / prior_high, 0.0)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _peak_optional_float(existing_value: float | None, new_value: float | None) -> float | None:
    if existing_value is None:
        return new_value
    if new_value is None:
        return existing_value
    return max(existing_value, new_value)


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _run_portfolio_review_intraday_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    provider: DailyBarProvider | None,
    current_positions: list[ExistingPosition] | None = None,
) -> IntradayPortfolioReviewReport:
    resolved_current_positions = (
        _load_current_positions(args.portfolio_file)
        if current_positions is None
        else current_positions
    )
    benchmark_symbol_override = _benchmark_symbol_override(args.benchmark_symbol)

    if args.interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero.")
    if not resolved_current_positions:
        return build_intraday_portfolio_review_report(
            as_of_date=args.as_of,
            interval_minutes=args.interval_minutes,
            portfolio_path=str(args.portfolio_file.resolve()),
            rows=[],
            current_positions=[],
            benchmark_symbol=benchmark_symbol_override,
        )
    if provider is None:
        raise ValueError("A data provider is required when reviewing non-empty portfolios.")
    intraday_state_store = IntradayStateJournalStore(config.project_root, args.as_of)
    intraday_state_journal_path = intraday_state_store.path
    intraday_state_journal = intraday_state_store.load()

    preset_catalog = _portfolio_review_preset_catalog(args, config)
    base_settings = BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
    )
    if benchmark_symbol_override is not None:
        base_settings = replace(base_settings, benchmark_symbol=benchmark_symbol_override)

    position_plans = [
        _build_portfolio_review_plan(
            position,
            preset_catalog=preset_catalog,
            base_settings=base_settings,
            as_of_date=args.as_of,
        )
        for position in resolved_current_positions
    ]

    benchmark_symbol = base_settings.benchmark_symbol
    benchmark_intraday_metrics: Mapping[str, float | str | None] | None = None
    if benchmark_symbol:
        benchmark_frame = provider.fetch_intraday_bars(
            benchmark_symbol,
            args.as_of,
            interval_minutes=args.interval_minutes,
            refresh_cache=True,
        )
        if not benchmark_frame.empty:
            benchmark_intraday_metrics = _intraday_session_metrics(benchmark_frame)

    earnings_contexts = _load_earnings_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        symbols=[position.symbol for position in resolved_current_positions],
        as_of_date=args.as_of,
        earnings_watch_days=config.strategy.signals.earnings_watch_days,
        refresh_cache=False,
        log_label="review-portfolio-intraday",
    )
    intraday_sector_history_start = sector_context_history_start(
        args.as_of,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        relative_strength_window=config.strategy.signals.sector_relative_strength_window,
    )
    sector_contexts: dict[str, SectorFeatureContext] = {}
    if hasattr(provider, "fetch_daily_bars"):
        intraday_sector_symbol_frames: dict[str, pd.DataFrame] = {}
        for position in resolved_current_positions:
            try:
                bars = provider.fetch_daily_bars(
                    position.symbol,
                    intraday_sector_history_start,
                    args.as_of,
                    refresh_cache=False,
                )
            except DataProviderError as exc:
                LOGGER.warning(
                    "Skipping sector context for held symbol %s due to provider error: %s",
                    position.symbol,
                    exc,
                )
                continue
            if not bars.empty:
                intraday_sector_symbol_frames[position.symbol] = bars
        sector_contexts = _load_sector_contexts(
            config=config,
            env_file=getattr(args, "env_file", None),
            provider=provider,
            symbol_frames=intraday_sector_symbol_frames,
            as_of_date=args.as_of,
            history_start=intraday_sector_history_start,
            benchmark_sma_fast=config.strategy.signals.benchmark_sma_fast,
            benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=config.strategy.signals.sector_relative_strength_window,
            refresh_cache=False,
            log_label="review-portfolio-intraday",
        )

    rows: list[PortfolioReviewRow] = []
    intraday_observations: dict[str, IntradayStateObservation] = {}
    for plan in position_plans:
        position = plan.get("position")
        symbol = position.symbol if isinstance(position, ExistingPosition) else "<unknown>"
        try:
            row, observation = _build_portfolio_review_intraday_row(
                plan,
                provider=provider,
                as_of_date=args.as_of,
                interval_minutes=args.interval_minutes,
                benchmark_symbol=benchmark_symbol,
                benchmark_intraday_metrics=benchmark_intraday_metrics,
                refresh_cache=True,
                earnings_context=earnings_contexts.get(symbol),
                sector_context=sector_contexts.get(symbol),
                intraday_state_journal=intraday_state_journal,
            )
            rows.append(row)
            intraday_observations[observation.symbol] = observation
        except DataProviderConfigurationError:
            raise
        except (DataProviderError, ValueError) as exc:
            LOGGER.warning("Skipping held symbol %s due to intraday review error: %s", symbol, exc)

    if not rows:
        raise NoIntradayDataError(
            "No held symbols had usable regular-session intraday data for the requested session."
        )
    updated_intraday_state_journal = update_intraday_state_journal(
        intraday_state_journal,
        observations=intraday_observations,
    )
    intraday_state_store.save(updated_intraday_state_journal)

    return build_intraday_portfolio_review_report(
        as_of_date=args.as_of,
        interval_minutes=args.interval_minutes,
        portfolio_path=str(args.portfolio_file.resolve()),
        rows=rows,
        current_positions=resolved_current_positions,
        benchmark_symbol=benchmark_symbol,
    )


def _build_portfolio_review_intraday_row(
    plan: dict[str, object],
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    interval_minutes: int,
    benchmark_symbol: str | None,
    benchmark_intraday_metrics: Mapping[str, float | str | None] | None,
    refresh_cache: bool,
    earnings_context: EarningsRiskContext | None = None,
    sector_context: SectorFeatureContext | None = None,
    intraday_state_journal: IntradayStateJournal | None = None,
) -> tuple[PortfolioReviewRow, IntradayStateObservation]:
    position = plan["position"]
    preset = plan["preset"]
    preset_resolution = plan["preset_resolution"]
    settings = plan["settings"]
    if not isinstance(position, ExistingPosition):
        raise TypeError("plan['position'] must be an ExistingPosition.")
    if not isinstance(preset, BreakoutStrategyPreset):
        raise TypeError("plan['preset'] must be a BreakoutStrategyPreset.")
    if not isinstance(preset_resolution, str):
        raise TypeError("plan['preset_resolution'] must be a string.")
    if not isinstance(settings, BreakoutMomentumSettings):
        raise TypeError("plan['settings'] must be BreakoutMomentumSettings.")

    bars = provider.fetch_intraday_bars(
        position.symbol,
        as_of_date,
        interval_minutes=interval_minutes,
        refresh_cache=refresh_cache,
    )
    if bars.empty:
        raise ValueError(f"No intraday bars were available for held symbol '{position.symbol}'.")

    intraday_metrics = _intraday_session_metrics(bars)
    intraday_relative_strength_diff = _intraday_relative_strength_diff(
        intraday_metrics,
        benchmark_intraday_metrics,
    )
    market_context = build_market_context(
        as_of_date=as_of_date,
        benchmark_symbol=benchmark_symbol,
        benchmark_intraday_return_vs_open=(
            _mapping_float_or_none(benchmark_intraday_metrics, "intraday_return_vs_open")
            if benchmark_intraday_metrics is not None
            else None
        ),
    )
    position_features = build_intraday_position_features(
        symbol=position.symbol,
        as_of_date=as_of_date,
        average_entry_price=position.average_entry_price,
        current_stop=position.current_stop,
        session_open=float(intraday_metrics["session_open"]),
        session_high=float(intraday_metrics["session_high"]),
        session_low=float(intraday_metrics["session_low"]),
        latest_close=float(intraday_metrics["latest_close"]),
        latest_low=float(intraday_metrics["latest_low"]),
        session_vwap=_mapping_float_or_none(intraday_metrics, "session_vwap"),
        intraday_return_vs_benchmark=intraday_relative_strength_diff,
        earnings_context=earnings_context,
        earnings_watch_days=settings.earnings_watch_days,
        sector_context=sector_context,
        early_strength_threshold=0.03,
        meaningful_profit_pct=0.05,
        session_high_giveback_watch_threshold=max(
            0.04,
            settings.profit_giveback_threshold * 0.75,
        ),
        intraday_relative_strength_watch_threshold=-0.02,
        sector_relative_strength_watch_threshold=(
            settings.sector_relative_strength_watch_threshold
        ),
        market_context=market_context,
    )
    effective_intraday_state_journal = (
        intraday_state_journal
        if intraday_state_journal is not None
        else IntradayStateJournal(session_date=as_of_date, entries={})
    )
    latest_bar_time = _mapping_text_or_none(intraday_metrics, "latest_bar_time")
    state_observation = _build_intraday_state_observation(
        position_features,
        timestamp=latest_bar_time or f"{as_of_date.isoformat()}T00:00:00",
    )
    trajectory_features = preview_intraday_trajectory_features(
        effective_intraday_state_journal,
        observation=state_observation,
    )
    position_features = apply_intraday_trajectory_features(
        position_features,
        trajectory_features,
    )
    decision = review_existing_long_position_intraday(
        position,
        position_features=position_features,
        session_open=float(intraday_metrics["session_open"]),
        session_high=float(intraday_metrics["session_high"]),
        session_low=float(intraday_metrics["session_low"]),
        latest_close=float(intraday_metrics["latest_close"]),
        latest_low=float(intraday_metrics["latest_low"]),
        session_vwap=_mapping_float_or_none(intraday_metrics, "session_vwap"),
        session_high_giveback_exit_threshold=settings.profit_giveback_threshold,
        intraday_relative_strength_diff=intraday_relative_strength_diff,
        intraday_high_profit_unrealized_pct=settings.intraday_high_profit_unrealized_pct,
        intraday_high_profit_giveback_threshold=settings.intraday_high_profit_giveback_threshold,
        earnings_date=earnings_context.earnings_date if earnings_context is not None else None,
        earnings_days_away=(
            earnings_context.earnings_days_away
            if earnings_context is not None
            else None
        ),
        earnings_watch_days=settings.earnings_watch_days,
        sector_name=(
            sector_context.sector_name
            if sector_context is not None
            else None
        ),
        sector_etf_symbol=(
            sector_context.sector_etf_symbol
            if sector_context is not None
            else None
        ),
        sector_regime_passed=(
            sector_context.sector_regime_passed
            if sector_context is not None
            else None
        ),
        relative_strength_vs_sector=(
            sector_context.relative_strength_vs_sector
            if sector_context is not None
            else None
        ),
        sector_relative_strength_window=(
            sector_context.relative_strength_window
            if sector_context is not None
            else None
        ),
        sector_relative_strength_watch_threshold=(
            settings.sector_relative_strength_watch_threshold
        ),
    )
    finalized_state_observation = replace(
        state_observation,
        suggested_action=decision.suggested_action,
    )
    entry_date_used = _position_entry_date(position)
    return (
        PortfolioReviewRow(
            date=as_of_date,
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=decision.suggested_stop,
            latest_close=decision.latest_close,
            unrealized_pl_pct=decision.unrealized_pl_pct,
            distance_to_stop_pct=decision.distance_to_stop_pct,
            regime_passed=None,
            above_entry=decision.above_entry,
            suggested_action=decision.suggested_action,
            preset_name=preset.name,
            rationale=" | ".join(decision.rationale),
            metadata={
                "preset_resolution": preset_resolution,
                "position_source": position.source,
                "position_metadata": dict(position.metadata),
                "entry_date_used": (
                    entry_date_used.isoformat() if entry_date_used is not None else None
                ),
                "interval_minutes": interval_minutes,
                "latest_bar_time": intraday_metrics.get("latest_bar_time"),
                "benchmark_symbol": benchmark_symbol,
                "intraday_relative_strength_diff": intraday_relative_strength_diff,
                "benchmark_intraday_return_vs_open": (
                    benchmark_intraday_metrics.get("intraday_return_vs_open")
                    if benchmark_intraday_metrics is not None
                    else None
                ),
                **dict(intraday_metrics),
                **decision.metadata,
            },
        ),
        finalized_state_observation,
    )


def _build_intraday_state_observation(
    features: PositionFeatures,
    *,
    timestamp: str,
) -> IntradayStateObservation:
    return IntradayStateObservation(
        symbol=features.symbol,
        timestamp=timestamp,
        latest_close=features.latest_close,
        session_vwap=features.session_vwap,
        close_vs_vwap_pct=features.close_vs_vwap_pct,
        session_high=features.session_high,
        session_low=features.session_low,
        session_high_giveback_pct=features.session_high_giveback_pct,
        intraday_return_vs_open=features.intraday_return_vs_open,
        intraday_return_vs_benchmark=features.intraday_return_vs_benchmark,
        weak_intraday_relative_strength=features.weak_intraday_relative_strength,
        failed_intraday_strength=features.failed_intraday_strength,
        intraday_momentum_fade=features.intraday_momentum_fade,
        stacked_intraday_weakness=features.stacked_intraday_weakness,
        stop_breached_intraday=features.stop_breached_intraday,
    )


def _build_position_trajectory_observation(
    features: PositionFeatures,
    *,
    high_water_close_date: date | None,
    weak_relative_strength: bool,
) -> PositionTrajectoryObservation:
    return PositionTrajectoryObservation(
        symbol=features.symbol,
        as_of_date=features.as_of_date,
        average_entry_price=features.average_entry_price,
        current_stop=features.current_stop,
        latest_close=features.latest_close,
        unrealized_pl_pct=features.unrealized_pl_pct,
        above_entry=features.above_entry,
        high_water_close=features.high_water_close,
        high_water_close_date=high_water_close_date,
        days_since_new_high=features.days_since_new_high,
        stale_position=features.stale_position,
        relative_strength_return_diff=features.relative_strength_return_diff,
        weak_relative_strength=weak_relative_strength,
    )


def _build_trade_outcome_snapshot(
    position: ExistingPosition,
    *,
    price_frame: pd.DataFrame,
    as_of_date: date,
) -> TradeOutcomeSnapshot | None:
    trade_id = _clean_optional_text(position.metadata.get("trade_id"))
    if trade_id is None:
        LOGGER.debug(
            "Skipping trade outcome tracking for %s on %s because trade_id is absent.",
            position.symbol,
            as_of_date.isoformat(),
        )
        return None
    entry_date = _position_entry_date(position)
    if entry_date is None:
        LOGGER.warning(
            "Skipping trade outcome tracking for %s on %s because trade_id %s has no entry_date.",
            position.symbol,
            as_of_date.isoformat(),
            trade_id,
        )
        return None
    return compute_trade_outcome_snapshot(
        price_frame,
        entry_date=entry_date,
        entry_price=position.average_entry_price,
        stop_price=position.current_stop,
        as_of_date=as_of_date,
    )


def _position_trajectory_observation_from_row(
    row: PortfolioReviewRow,
) -> PositionTrajectoryObservation | None:
    raw_observation = row.metadata.get(_POSITION_TRAJECTORY_OBSERVATION_METADATA_KEY)
    if not isinstance(raw_observation, Mapping):
        return None
    return PositionTrajectoryObservation.from_mapping(row.symbol, raw_observation)


def _intraday_session_metrics(
    intraday_bars: pd.DataFrame,
) -> dict[str, float | str | None]:
    prepared = intraday_bars.sort_values("datetime", kind="stable").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("intraday_bars cannot be empty.")

    session_open = float(prepared.iloc[0]["open"])
    session_high = float(pd.to_numeric(prepared["high"], errors="coerce").max())
    session_low = float(pd.to_numeric(prepared["low"], errors="coerce").min())
    latest_bar = prepared.iloc[-1]
    latest_close = float(latest_bar["close"])
    latest_low = float(latest_bar["low"])
    latest_bar_time = pd.to_datetime(latest_bar["datetime"], errors="coerce")
    session_vwap = _intraday_session_vwap(prepared)
    intraday_return_vs_open = (latest_close / session_open) - 1.0
    peak_intraday_return_vs_open = (session_high / session_open) - 1.0

    return {
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "latest_close": latest_close,
        "latest_low": latest_low,
        "latest_bar_time": (
            latest_bar_time.isoformat() if not pd.isna(latest_bar_time) else None
        ),
        "session_vwap": session_vwap,
        "intraday_return_vs_open": intraday_return_vs_open,
        "peak_intraday_return_vs_open": peak_intraday_return_vs_open,
    }


def _intraday_session_vwap(intraday_bars: pd.DataFrame) -> float | None:
    if "vwap" in intraday_bars.columns:
        explicit_vwap = pd.to_numeric(intraday_bars["vwap"], errors="coerce")
        volume = pd.to_numeric(intraday_bars["volume"], errors="coerce")
        valid_explicit = explicit_vwap.notna() & volume.notna() & (volume > 0)
        if bool(valid_explicit.any()):
            weighted_vwap = float(
                (explicit_vwap.loc[valid_explicit] * volume.loc[valid_explicit]).sum()
            )
            total_volume = float(volume.loc[valid_explicit].sum())
            if total_volume > 0:
                return weighted_vwap / total_volume

    volume = pd.to_numeric(intraday_bars["volume"], errors="coerce")
    close = pd.to_numeric(intraday_bars["close"], errors="coerce")
    valid = volume.notna() & close.notna() & (volume > 0)
    if not bool(valid.any()):
        return None
    weighted_close = float((close.loc[valid] * volume.loc[valid]).sum())
    total_volume = float(volume.loc[valid].sum())
    if total_volume <= 0:
        return None
    return weighted_close / total_volume


def _intraday_relative_strength_diff(
    intraday_metrics: Mapping[str, float | str | None],
    benchmark_intraday_metrics: Mapping[str, float | str | None] | None,
) -> float | None:
    if benchmark_intraday_metrics is None:
        return None
    symbol_return = _mapping_float_or_none(intraday_metrics, "intraday_return_vs_open")
    benchmark_return = _mapping_float_or_none(
        benchmark_intraday_metrics,
        "intraday_return_vs_open",
    )
    if symbol_return is None or benchmark_return is None:
        return None
    return symbol_return - benchmark_return


def _mapping_float_or_none(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mapping_text_or_none(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_int_or_none(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _iso_date_or_none(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _portfolio_review_preset_catalog(
    args: argparse.Namespace,
    config: AppConfig,
) -> dict[str, BreakoutStrategyPreset]:
    configured_presets = _load_strategy_comparison_presets(args.config_dir)
    presets = resolve_breakout_strategy_presets(
        config.strategy.signals,
        config.strategy.risk,
        configured_presets=configured_presets,
    )
    return {preset.name: preset for preset in presets}


def _resolve_portfolio_review_preset(
    position: ExistingPosition,
    preset_catalog: dict[str, BreakoutStrategyPreset],
) -> tuple[BreakoutStrategyPreset, str]:
    default_preset = preset_catalog.get("standard_breakout")
    if default_preset is None:
        default_preset = next(iter(preset_catalog.values()))

    requested_name = (
        position.preset_name.strip()
        if isinstance(position.preset_name, str) and position.preset_name.strip()
        else None
    )
    if requested_name is None:
        return default_preset, "default_standard_breakout"

    preset = preset_catalog.get(requested_name)
    if preset is None:
        return default_preset, f"fallback_default_standard_breakout:{requested_name}"
    return preset, "portfolio_snapshot"


def _run_portfolio_review_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    provider: DailyBarProvider | None,
    current_positions: list[ExistingPosition] | None = None,
) -> PortfolioReviewReport:
    resolved_current_positions = (
        _load_current_positions(args.portfolio_file)
        if current_positions is None
        else current_positions
    )
    benchmark_symbol_override = _benchmark_symbol_override(args.benchmark_symbol)

    if not resolved_current_positions:
        return build_portfolio_review_report(
            as_of_date=args.as_of,
            rows=[],
            current_positions=[],
            benchmark_symbol=benchmark_symbol_override,
        )
    if provider is None:
        raise ValueError("A data provider is required when reviewing non-empty portfolios.")
    position_trajectory_journal: PositionTrajectoryJournal | None = None
    position_trajectory_store: PositionTrajectoryJournalStore | None = None
    if getattr(args, "portfolio_file", None) is not None:
        position_trajectory_store = PositionTrajectoryJournalStore(
            config.project_root,
        )
        position_trajectory_journal = position_trajectory_store.load()

    preset_catalog = _portfolio_review_preset_catalog(args, config)
    base_settings = BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
    )
    if benchmark_symbol_override is not None:
        base_settings = replace(base_settings, benchmark_symbol=benchmark_symbol_override)

    position_plans = [
        _build_portfolio_review_plan(
            position,
            preset_catalog=preset_catalog,
            base_settings=base_settings,
            as_of_date=args.as_of,
        )
        for position in resolved_current_positions
    ]

    benchmark_symbol = (
        base_settings.benchmark_symbol
        if any(plan["settings"].enable_regime_filter for plan in position_plans)
        else None
    )
    benchmark_frame = None
    if benchmark_symbol is not None:
        earliest_fetch_start = min(plan["fetch_start"] for plan in position_plans)
        benchmark_frame = provider.fetch_daily_bars(
            benchmark_symbol,
            earliest_fetch_start,
            args.as_of,
            refresh_cache=args.refresh_cache,
        )
        if benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{benchmark_symbol}'."
            )

    earnings_contexts = _load_earnings_contexts(
        config=config,
        env_file=getattr(args, "env_file", None),
        symbols=[position.symbol for position in resolved_current_positions],
        as_of_date=args.as_of,
        earnings_watch_days=config.strategy.signals.earnings_watch_days,
        refresh_cache=args.refresh_cache,
        log_label="review-portfolio",
    )
    daily_sector_history_start = sector_context_history_start(
        args.as_of,
        benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
        relative_strength_window=config.strategy.signals.sector_relative_strength_window,
    )
    sector_contexts: dict[str, SectorFeatureContext] = {}
    if hasattr(provider, "fetch_daily_bars"):
        daily_sector_symbol_frames: dict[str, pd.DataFrame] = {}
        for position in resolved_current_positions:
            try:
                bars = provider.fetch_daily_bars(
                    position.symbol,
                    daily_sector_history_start,
                    args.as_of,
                    refresh_cache=args.refresh_cache,
                )
            except DataProviderError as exc:
                LOGGER.warning(
                    "Skipping sector context for held symbol %s due to provider error: %s",
                    position.symbol,
                    exc,
                )
                continue
            if not bars.empty:
                daily_sector_symbol_frames[position.symbol] = bars
        sector_contexts = _load_sector_contexts(
            config=config,
            env_file=getattr(args, "env_file", None),
            provider=provider,
            symbol_frames=daily_sector_symbol_frames,
            as_of_date=args.as_of,
            history_start=daily_sector_history_start,
            benchmark_sma_fast=config.strategy.signals.benchmark_sma_fast,
            benchmark_sma_slow=config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=config.strategy.signals.sector_relative_strength_window,
            refresh_cache=args.refresh_cache,
            log_label="review-portfolio",
        )

    rows: list[PortfolioReviewRow] = []
    position_trajectory_observations: dict[str, PositionTrajectoryObservation] = {}
    for plan in position_plans:
        position = plan.get("position")
        symbol = position.symbol if isinstance(position, ExistingPosition) else "<unknown>"
        try:
            review_row_state = _coerce_portfolio_review_row_state(
                _build_portfolio_review_row(
                    plan,
                    provider=provider,
                    as_of_date=args.as_of,
                    benchmark_frame=benchmark_frame,
                    refresh_cache=args.refresh_cache,
                    earnings_context=earnings_contexts.get(symbol),
                    sector_context=sector_contexts.get(symbol),
                    position_trajectory_journal=position_trajectory_journal,
                )
            )
            row = review_row_state.row
            position_trajectory_observation = review_row_state.position_trajectory_observation
            if position_trajectory_observation is not None:
                position_trajectory_observations[position_trajectory_observation.symbol] = (
                    position_trajectory_observation
                )
            rows.append(row)
        except (DataProviderError, ValueError) as exc:
            LOGGER.warning("Skipping held symbol %s due to review error: %s", symbol, exc)
    if position_trajectory_store is not None and position_trajectory_journal is not None:
        updated_position_trajectory_journal = update_position_trajectory_journal(
            position_trajectory_journal,
            observations=position_trajectory_observations,
            active_symbols=[position.symbol for position in resolved_current_positions],
        )
        position_trajectory_store.save(updated_position_trajectory_journal)
    return build_portfolio_review_report(
        as_of_date=args.as_of,
        rows=rows,
        current_positions=resolved_current_positions,
        benchmark_symbol=benchmark_symbol,
    )


def _build_portfolio_review_plan(
    position: ExistingPosition,
    *,
    preset_catalog: dict[str, BreakoutStrategyPreset],
    base_settings: BreakoutMomentumSettings,
    as_of_date: date,
) -> dict[str, object]:
    preset, preset_resolution = _resolve_portfolio_review_preset(position, preset_catalog)
    settings = preset.apply_to_settings(base_settings)
    fetch_start = _strategy_warmup_start(as_of_date, settings)
    entry_date = _position_entry_date(position)
    if entry_date is not None and entry_date < fetch_start:
        fetch_start = entry_date
    return {
        "position": position,
        "preset": preset,
        "preset_resolution": preset_resolution,
        "settings": settings,
        "fetch_start": fetch_start,
    }


def _build_portfolio_review_row(
    plan: dict[str, object],
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    benchmark_frame: pd.DataFrame | None,
    refresh_cache: bool,
    earnings_context: EarningsRiskContext | None = None,
    sector_context: SectorFeatureContext | None = None,
    position_trajectory_journal: PositionTrajectoryJournal | None = None,
) -> PortfolioReviewRowState:
    position = plan["position"]
    preset = plan["preset"]
    preset_resolution = plan["preset_resolution"]
    settings = plan["settings"]
    fetch_start = plan["fetch_start"]
    if not isinstance(position, ExistingPosition):
        raise TypeError("plan['position'] must be an ExistingPosition.")
    if not isinstance(preset, BreakoutStrategyPreset):
        raise TypeError("plan['preset'] must be a BreakoutStrategyPreset.")
    if not isinstance(preset_resolution, str):
        raise TypeError("plan['preset_resolution'] must be a string.")
    if not isinstance(settings, BreakoutMomentumSettings):
        raise TypeError("plan['settings'] must be BreakoutMomentumSettings.")
    if not isinstance(fetch_start, date):
        raise TypeError("plan['fetch_start'] must be a date.")

    bars = provider.fetch_daily_bars(
        position.symbol,
        fetch_start,
        as_of_date,
        refresh_cache=refresh_cache,
    )
    if bars.empty:
        raise ValueError(f"No daily bars were available for held symbol '{position.symbol}'.")

    prepared_bars = bars.sort_values("date", kind="stable").reset_index(drop=True)
    latest_close = _latest_close_from_bars(prepared_bars, symbol=position.symbol)
    trailing_stop_candidate = _portfolio_review_trailing_stop_candidate(
        prepared_bars,
        position=position,
        settings=settings,
    )
    high_water_metrics = _portfolio_review_high_water_metrics(
        prepared_bars,
        position=position,
    )
    relative_strength_metrics = _portfolio_review_relative_strength_metrics(
        prepared_bars,
        benchmark_frame=benchmark_frame,
        window=settings.relative_strength_window,
    )
    regime_passed: bool | None = None
    if settings.enable_regime_filter:
        if benchmark_frame is None or benchmark_frame.empty:
            raise ValueError(
                f"No benchmark data was available for regime symbol '{settings.benchmark_symbol}'."
            )
        regime_passed = regime_is_bullish(benchmark_frame, settings.regime_settings)

    breakout_failure_reference = _portfolio_review_breakout_failure_reference(position)
    market_context = build_market_context(
        as_of_date=as_of_date,
        benchmark_symbol=(
            settings.benchmark_symbol if settings.enable_regime_filter else None
        ),
        regime_filter_mode=settings.regime_filter_mode,
        regime_passed=regime_passed,
        benchmark_return=_mapping_float_or_none(relative_strength_metrics, "benchmark_return"),
    )
    position_features = build_position_features(
        symbol=position.symbol,
        as_of_date=as_of_date,
        average_entry_price=position.average_entry_price,
        latest_close=latest_close,
        current_stop=position.current_stop,
        regime_passed=regime_passed,
        trailing_stop_candidate=trailing_stop_candidate,
        high_water_close=_mapping_float_or_none(high_water_metrics, "high_water_close"),
        breakout_failure_reference=breakout_failure_reference,
        days_since_new_high=_mapping_int_or_none(high_water_metrics, "days_since_new_high"),
        stale_high_watch_days=settings.stale_high_watch_days,
        relative_strength_return_diff=_mapping_float_or_none(
            relative_strength_metrics,
            "relative_strength_return_diff",
        ),
        relative_strength_window=_mapping_int_or_none(
            relative_strength_metrics,
            "relative_strength_window",
        ),
        earnings_context=earnings_context,
        earnings_watch_days=settings.earnings_watch_days,
        sector_context=sector_context,
        sector_relative_strength_watch_threshold=(
            settings.sector_relative_strength_watch_threshold
        ),
        market_context=market_context,
    )
    weak_relative_strength = (
        position_features.relative_strength_return_diff is not None
        and position_features.relative_strength_return_diff
        <= settings.relative_strength_watch_threshold
    )
    finalized_position_trajectory_observation: PositionTrajectoryObservation | None = None
    if position_trajectory_journal is not None:
        position_trajectory_observation = _build_position_trajectory_observation(
            position_features,
            high_water_close_date=_iso_date_or_none(
                high_water_metrics.get("high_water_close_date")
            ),
            weak_relative_strength=weak_relative_strength,
        )
        position_trajectory_features = preview_position_trajectory_features(
            position_trajectory_journal,
            observation=position_trajectory_observation,
        )
        position_features = apply_position_trajectory_features(
            position_features,
            position_trajectory_features,
        )
    decision = review_existing_long_position(
        position,
        position_features=position_features,
        latest_close=latest_close,
        regime_passed=regime_passed,
        trailing_stop_candidate=trailing_stop_candidate,
        high_water_close=position_features.high_water_close,
        profit_giveback_threshold=settings.profit_giveback_threshold,
        profit_giveback_min_unrealized_pct=settings.profit_giveback_min_unrealized_pct,
        breakout_failure_reference=breakout_failure_reference,
        days_since_new_high=position_features.days_since_new_high,
        stale_high_watch_days=settings.stale_high_watch_days,
        relative_strength_return_diff=position_features.relative_strength_return_diff,
        relative_strength_window=position_features.relative_strength_window,
        relative_strength_watch_threshold=settings.relative_strength_watch_threshold,
        earnings_date=earnings_context.earnings_date if earnings_context is not None else None,
        earnings_days_away=(
            earnings_context.earnings_days_away
            if earnings_context is not None
            else None
        ),
        earnings_watch_days=settings.earnings_watch_days,
        sector_name=(
            sector_context.sector_name
            if sector_context is not None
            else None
        ),
        sector_etf_symbol=(
            sector_context.sector_etf_symbol
            if sector_context is not None
            else None
        ),
        sector_regime_passed=(
            sector_context.sector_regime_passed
            if sector_context is not None
            else None
        ),
        relative_strength_vs_sector=(
            sector_context.relative_strength_vs_sector
            if sector_context is not None
            else None
        ),
        sector_relative_strength_window=(
            sector_context.relative_strength_window
            if sector_context is not None
            else None
        ),
        sector_relative_strength_watch_threshold=(
            settings.sector_relative_strength_watch_threshold
        ),
    )
    if position_trajectory_journal is not None:
        finalized_position_trajectory_observation = replace(
            position_trajectory_observation,
            suggested_action=decision.suggested_action,
        )
    trade_outcome_snapshot = _build_trade_outcome_snapshot(
        position,
        price_frame=prepared_bars,
        as_of_date=as_of_date,
    )
    entry_date_used = _position_entry_date(position)
    metadata = {
        "preset_resolution": preset_resolution,
        "position_source": position.source,
        "position_metadata": dict(position.metadata),
        "trailing_stop_candidate": trailing_stop_candidate,
        "regime_filter_enabled": settings.enable_regime_filter,
        "regime_filter_mode": settings.regime_filter_mode,
        "benchmark_symbol": settings.benchmark_symbol if settings.enable_regime_filter else None,
        "entry_date_used": entry_date_used.isoformat() if entry_date_used is not None else None,
        "high_water_close": high_water_metrics.get("high_water_close"),
        "high_water_close_date": high_water_metrics.get("high_water_close_date"),
        "days_since_new_high": high_water_metrics.get("days_since_new_high"),
        "relative_strength_window": relative_strength_metrics.get("relative_strength_window"),
        "relative_strength_symbol_return": relative_strength_metrics.get("symbol_return"),
        "relative_strength_benchmark_return": relative_strength_metrics.get("benchmark_return"),
        "relative_strength_return_diff": relative_strength_metrics.get("relative_strength_return_diff"),
        **decision.metadata,
    }
    if trade_outcome_snapshot is not None:
        metadata[_TRADE_OUTCOME_SNAPSHOT_METADATA_KEY] = trade_outcome_snapshot.to_dict()
    return PortfolioReviewRowState(
        row=PortfolioReviewRow(
            date=as_of_date,
            symbol=position.symbol,
            quantity=position.shares,
            average_entry_price=position.average_entry_price,
            current_stop=position.current_stop,
            suggested_stop=decision.suggested_stop,
            latest_close=decision.latest_close,
            unrealized_pl_pct=decision.unrealized_pl_pct,
            distance_to_stop_pct=decision.distance_to_stop_pct,
            regime_passed=decision.regime_passed,
            above_entry=decision.above_entry,
            suggested_action=decision.suggested_action,
            preset_name=preset.name,
            rationale=" | ".join(decision.rationale),
            metadata=metadata,
        ),
        position_trajectory_observation=finalized_position_trajectory_observation,
    )


def _latest_close_from_bars(price_frame: pd.DataFrame, *, symbol: str) -> float:
    close_series = pd.to_numeric(price_frame["close"], errors="coerce").dropna()
    if close_series.empty:
        raise ValueError(f"No usable close prices were available for held symbol '{symbol}'.")
    return float(close_series.iloc[-1])


def _portfolio_review_trailing_stop_candidate(
    price_frame: pd.DataFrame,
    *,
    position: ExistingPosition,
    settings: BreakoutMomentumSettings,
) -> float | None:
    atr_series = pd.to_numeric(atr(price_frame, window=settings.atr_window), errors="coerce").dropna()
    if atr_series.empty:
        return None

    reference_close = _portfolio_review_reference_close(price_frame, position=position)
    if reference_close is None:
        return None
    return trailing_stop_reference(
        reference_close,
        float(atr_series.iloc[-1]),
        settings.trailing_stop_atr,
    )


def _portfolio_review_reference_close(
    price_frame: pd.DataFrame,
    *,
    position: ExistingPosition,
) -> float | None:
    prepared = _portfolio_review_history_since_entry(price_frame, position=position)
    if prepared.empty:
        return None
    return float(prepared["close"].max())


def _portfolio_review_history_since_entry(
    price_frame: pd.DataFrame,
    *,
    position: ExistingPosition,
) -> pd.DataFrame:
    prepared = price_frame.copy()
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["close", "date"]).sort_values("date", kind="stable")
    if prepared.empty:
        return prepared.reset_index(drop=True)

    entry_date = _position_entry_date(position)
    if entry_date is not None:
        entry_mask = prepared["date"].dt.date >= entry_date
        if bool(entry_mask.any()):
            prepared = prepared.loc[entry_mask]
    return prepared.reset_index(drop=True)


def _portfolio_review_high_water_metrics(
    price_frame: pd.DataFrame,
    *,
    position: ExistingPosition,
) -> dict[str, float | int | str | None]:
    prepared = _portfolio_review_history_since_entry(price_frame, position=position)
    if prepared.empty:
        return {
            "high_water_close": None,
            "high_water_close_date": None,
            "days_since_new_high": None,
        }

    high_water_close = float(prepared["close"].max())
    high_indices = prepared.index[prepared["close"] == high_water_close]
    if len(high_indices) == 0:
        return {
            "high_water_close": None,
            "high_water_close_date": None,
            "days_since_new_high": None,
        }

    last_high_index = int(high_indices[-1])
    last_high_date = prepared.iloc[last_high_index]["date"]
    high_water_close_date = (
        last_high_date.date().isoformat()
        if isinstance(last_high_date, pd.Timestamp)
        else None
    )
    return {
        "high_water_close": high_water_close,
        "high_water_close_date": high_water_close_date,
        "days_since_new_high": len(prepared) - last_high_index - 1,
    }


def _portfolio_review_relative_strength_metrics(
    price_frame: pd.DataFrame,
    *,
    benchmark_frame: pd.DataFrame | None,
    window: int,
) -> dict[str, float | int | None]:
    if benchmark_frame is None or benchmark_frame.empty or window <= 0:
        return {
            "relative_strength_window": window,
            "symbol_return": None,
            "benchmark_return": None,
            "relative_strength_return_diff": None,
        }

    symbol_history = price_frame.loc[:, ["date", "close"]].copy()
    symbol_history["date"] = pd.to_datetime(symbol_history["date"], errors="coerce")
    symbol_history["close"] = pd.to_numeric(symbol_history["close"], errors="coerce")
    symbol_history = symbol_history.dropna(subset=["date", "close"]).sort_values("date", kind="stable")

    benchmark_history = benchmark_frame.loc[:, ["date", "close"]].copy()
    benchmark_history["date"] = pd.to_datetime(benchmark_history["date"], errors="coerce")
    benchmark_history["close"] = pd.to_numeric(benchmark_history["close"], errors="coerce")
    benchmark_history = benchmark_history.dropna(subset=["date", "close"]).sort_values("date", kind="stable")

    merged = symbol_history.merge(
        benchmark_history,
        on="date",
        how="inner",
        suffixes=("_symbol", "_benchmark"),
    )
    if len(merged) <= window:
        return {
            "relative_strength_window": window,
            "symbol_return": None,
            "benchmark_return": None,
            "relative_strength_return_diff": None,
        }

    window_frame = merged.iloc[-(window + 1):]
    symbol_start = float(window_frame.iloc[0]["close_symbol"])
    symbol_end = float(window_frame.iloc[-1]["close_symbol"])
    benchmark_start = float(window_frame.iloc[0]["close_benchmark"])
    benchmark_end = float(window_frame.iloc[-1]["close_benchmark"])
    if symbol_start <= 0 or benchmark_start <= 0:
        return {
            "relative_strength_window": window,
            "symbol_return": None,
            "benchmark_return": None,
            "relative_strength_return_diff": None,
        }

    symbol_return = (symbol_end / symbol_start) - 1.0
    benchmark_return = (benchmark_end / benchmark_start) - 1.0
    return {
        "relative_strength_window": window,
        "symbol_return": symbol_return,
        "benchmark_return": benchmark_return,
        "relative_strength_return_diff": symbol_return - benchmark_return,
    }


def _portfolio_review_breakout_failure_reference(position: ExistingPosition) -> float | None:
    metadata_containers: list[Mapping[str, Any]] = [position.metadata]
    signal_metadata = position.metadata.get("signal_metadata")
    if isinstance(signal_metadata, Mapping):
        metadata_containers.append(signal_metadata)
    execution_metadata = position.metadata.get("execution_metadata")
    if isinstance(execution_metadata, Mapping):
        metadata_containers.append(execution_metadata)
        nested_signal_metadata = execution_metadata.get("signal_metadata")
        if isinstance(nested_signal_metadata, Mapping):
            metadata_containers.append(nested_signal_metadata)

    for container in metadata_containers:
        for key in ("breakout_reference", "breakout_level", "prior_high", "entry_price_hint"):
            value = _portfolio_review_metadata_float(container.get(key))
            if value is not None and value > 0:
                return value
    return None


def _portfolio_review_metadata_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _position_entry_date(position: ExistingPosition) -> date | None:
    for key in ("entry_date", "entry_datetime", "opened_at"):
        raw_value = position.metadata.get(key)
        if raw_value is None:
            continue
        parsed_value = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(parsed_value):
            continue
        return parsed_value.date()
    return None


def _load_current_positions(portfolio_file: Path | None) -> list[ExistingPosition]:
    if portfolio_file is None:
        return []
    try:
        return load_existing_positions(portfolio_file)
    except PortfolioInputError as exc:
        raise ValueError(str(exc)) from exc


def _load_pending_order_state(project_root: Path) -> PendingOrderState:
    return PendingOrderStateStore(project_root).load()


def _build_pending_order_reservation_state(
    *,
    project_root: Path,
    current_equity: float,
    current_positions: Sequence[ExistingPosition],
    constraints: PortfolioConstraints,
) -> tuple[PendingOrderState, OrderReservationState]:
    pending_order_state = _load_pending_order_state(project_root)
    reservation_state = build_order_reservation_state(
        pending_order_state,
        current_equity=current_equity,
        current_positions=current_positions,
        max_concurrent_positions=constraints.max_concurrent_positions,
    )
    return pending_order_state, reservation_state


def _load_earnings_contexts(
    *,
    config: AppConfig,
    env_file: Path | None,
    symbols: Sequence[str],
    as_of_date: date,
    earnings_watch_days: int,
    refresh_cache: bool,
    log_label: str,
) -> dict[str, EarningsRiskContext]:
    normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not normalized_symbols:
        return {}

    try:
        provider = create_earnings_calendar_provider(config, env_file=env_file)
        return build_earnings_risk_contexts(
            normalized_symbols,
            as_of_date=as_of_date,
            provider=provider,
            risk_window_days=earnings_watch_days,
            lookahead_calendar_days=max(30, earnings_watch_days + 7),
            refresh_cache=refresh_cache,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        LOGGER.warning("Earnings context unavailable for %s: %s", log_label, exc)
        return {}


def _load_sector_contexts(
    *,
    config: AppConfig,
    env_file: Path | None,
    provider: DailyBarProvider,
    symbol_frames: Mapping[str, pd.DataFrame],
    as_of_date: date,
    history_start: date,
    benchmark_sma_fast: int,
    benchmark_sma_slow: int,
    relative_strength_window: int,
    refresh_cache: bool,
    log_label: str,
) -> dict[str, SectorFeatureContext]:
    normalized_symbol_frames = {
        symbol.strip().upper(): frame
        for symbol, frame in symbol_frames.items()
        if symbol.strip() and not frame.empty
    }
    if not normalized_symbol_frames:
        return {}

    try:
        classifications = load_symbol_sector_classifications(
            tuple(normalized_symbol_frames),
            config=config,
            env_file=env_file,
            as_of_date=as_of_date,
            refresh_cache=refresh_cache,
        )
        return build_sector_feature_contexts(
            normalized_symbol_frames,
            provider=provider,
            as_of_date=as_of_date,
            history_start=history_start,
            sector_classifications=classifications,
            benchmark_sma_fast=benchmark_sma_fast,
            benchmark_sma_slow=benchmark_sma_slow,
            relative_strength_window=relative_strength_window,
            refresh_cache=refresh_cache,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        LOGGER.warning("Sector context unavailable for %s: %s", log_label, exc)
        return {}


def _load_volatility_context(
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    vix_caution_threshold: float | None,
    vix_entry_block_threshold: float | None,
    refresh_cache: bool,
    log_label: str,
) -> dict[str, float | str | bool | None]:
    try:
        return build_volatility_regime_context(
            provider=provider,
            as_of_date=as_of_date,
            history_start=volatility_context_history_start(as_of_date),
            caution_threshold=vix_caution_threshold,
            entry_block_threshold=vix_entry_block_threshold,
            refresh_cache=refresh_cache,
        )
    except ValueError as exc:
        LOGGER.warning("Volatility context unavailable for %s: %s", log_label, exc)
        return {
            "vix_close": None,
            "vix_sma_short": None,
            "vix_sma_long": None,
            "volatility_regime_state": None,
            "volatility_regime_risk_off": None,
        }


def _warn_missing_sector_context_for_entry(
    symbol: str,
    *,
    log_label: str,
    preset_name: str | None = None,
) -> None:
    context_label = f"{log_label}:{preset_name}" if preset_name is not None else log_label
    LOGGER.warning(
        "Sector regime enforcement is enabled but sector context is unavailable for %s during %s; proceeding without sector gate.",
        symbol,
        context_label,
    )


def _load_position_sector_classifications(
    current_positions: Sequence[ExistingPosition],
    *,
    config: AppConfig,
    env_file: Path | None,
    as_of_date: date,
    refresh_cache: bool,
    log_label: str,
) -> dict[str, SymbolSectorClassification]:
    if not current_positions:
        return {}

    try:
        return load_symbol_sector_classifications(
            [position.symbol for position in current_positions],
            config=config,
            env_file=env_file,
            as_of_date=as_of_date,
            refresh_cache=refresh_cache,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        LOGGER.warning("Portfolio heat context unavailable for %s: %s", log_label, exc)
        return {}


def _build_portfolio_holding_exposures(
    current_positions: Sequence[ExistingPosition],
    classifications: Mapping[str, SymbolSectorClassification],
) -> list[PortfolioHoldingExposure]:
    exposures: list[PortfolioHoldingExposure] = []
    for position in sorted(current_positions, key=lambda current_position: current_position.symbol):
        classification = classifications.get(position.symbol.strip().upper())
        exposures.append(
            PortfolioHoldingExposure(
                symbol=position.symbol,
                approximate_notional=position.average_entry_price * position.shares,
                sector_name=classification.sector if classification is not None else None,
                industry_name=classification.industry if classification is not None else None,
            )
        )
    return exposures


def _roll_forward_portfolio_holding_exposures(
    portfolio_holding_exposures: list[PortfolioHoldingExposure],
    candidate: RiskAssessedCandidate,
    *,
    sector_context: SectorFeatureContext | None,
) -> None:
    """Update the same-run exposure baseline after an approved candidate."""

    if (
        not candidate.approved
        or not candidate.sizing.is_valid
        or candidate.sizing.notional_value <= 0
    ):
        return

    normalized_symbol = candidate.signal.symbol.strip().upper()
    if any(
        exposure.symbol.strip().upper() == normalized_symbol
        for exposure in portfolio_holding_exposures
    ):
        return

    portfolio_holding_exposures.append(
        PortfolioHoldingExposure(
            symbol=normalized_symbol,
            approximate_notional=candidate.sizing.notional_value,
            sector_name=sector_context.sector_name if sector_context is not None else None,
            industry_name=sector_context.industry_name if sector_context is not None else None,
        )
    )


def _compute_entry_market_breadth(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    as_of_date: date,
    market_breadth_entry_floor_200ma: float | None,
    log_label: str,
) -> dict[str, float | str | None]:
    breadth_metrics = compute_market_breadth_from_frames(
        symbol_frames,
        as_of_date=as_of_date,
        entry_floor_200ma=market_breadth_entry_floor_200ma,
    )
    if (
        market_breadth_entry_floor_200ma is not None
        and breadth_metrics["market_breadth_pct_above_200ma"] is None
        and symbol_frames
    ):
        LOGGER.warning(
            "Market breadth gate is configured for %s but breadth could not be computed from the available universe bars; proceeding without breadth gate.",
            log_label,
        )
    return breadth_metrics


def _strategy_warmup_start(
    as_of_date: date,
    settings: BreakoutMomentumSettings,
) -> date:
    largest_window = max(
        settings.breakout_lookback + 1,
        settings.atr_window,
        settings.resolved_relative_volume_window + 1,
        settings.stale_high_watch_days + 1,
        settings.relative_strength_window + 1,
        settings.sector_relative_strength_window + 1,
        200 if settings.market_breadth_entry_floor_200ma is not None else 1,
        settings.benchmark_sma_slow if settings.enable_regime_filter else 1,
    )
    return as_of_date - timedelta(days=max(largest_window * 3, 30))


def _comparison_warmup_start(
    *,
    start_date: date,
    atr_window: int,
    benchmark_sma_slow: int,
    presets: Sequence[BreakoutStrategyPreset],
    max_relative_volume_window: int,
    max_sector_relative_strength_window: int,
    market_breadth_long_window: int,
    enable_regime_filter: bool,
) -> date:
    max_breakout_lookback = max((preset.breakout_lookback for preset in presets), default=1)
    largest_window = max(
        max_breakout_lookback + 1,
        atr_window,
        max_relative_volume_window + 1,
        max_sector_relative_strength_window + 1,
        market_breadth_long_window,
        benchmark_sma_slow if enable_regime_filter else 1,
    )
    return start_date - timedelta(days=max(largest_window * 3, 30))


def _profile_uses_reference_detail_filters(profile_config: UniverseProfileConfig) -> bool:
    return (
        profile_config.min_market_cap is not None
        or bool(profile_config.allowed_sectors)
        or bool(profile_config.allowed_industries)
    )


def _max_resolved_relative_volume_window_for_presets(
    *,
    config: AppConfig,
    presets: Sequence[BreakoutStrategyPreset],
    require_relative_volume_confirmation: bool,
    enable_regime_filter: bool,
) -> int:
    base_settings = BreakoutMomentumSettings.from_configs(
        config.strategy.signals,
        config.strategy.risk,
        require_relative_volume_confirmation=require_relative_volume_confirmation,
        enable_regime_filter=enable_regime_filter,
    )
    return max(
        (
            preset.apply_to_settings(
                base_settings,
                force_require_relative_volume_confirmation=(
                    True if require_relative_volume_confirmation else None
                ),
            ).resolved_relative_volume_window
            for preset in presets
        ),
        default=base_settings.resolved_relative_volume_window,
    )


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
            strategy_settings=preset.apply_to_settings(
                base_settings,
                force_require_relative_volume_confirmation=(
                    True if require_relative_volume_confirmation else None
                ),
            ),
            portfolio_constraints=portfolio_constraints,
            starting_cash=config.game_rules.starting_cash,
            base_risk_per_trade=preset.risk_per_trade,
            cost_model=cost_model,
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


def _run_streaming_daily_summary_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    current_positions: list[ExistingPosition],
    snapshot: LiveUpdateBufferSnapshot,
    provider: DailyBarProvider,
) -> dict[str, object]:
    _ = snapshot
    return _run_daily_summary_workflow(
        args,
        config=config,
        provider=provider,
        current_positions=current_positions,
    )


def _run_streaming_portfolio_review_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    current_positions: list[ExistingPosition],
    snapshot: LiveUpdateBufferSnapshot,
    provider: DailyBarProvider,
) -> PortfolioReviewReport:
    _ = snapshot
    return _run_portfolio_review_workflow(
        args,
        config=config,
        provider=provider,
        current_positions=current_positions,
    )


def _run_streaming_intraday_review_workflow(
    args: argparse.Namespace,
    *,
    config: AppConfig,
    current_positions: list[ExistingPosition],
    snapshot: LiveUpdateBufferSnapshot,
    provider: DailyBarProvider,
) -> IntradayPortfolioReviewReport:
    _ = snapshot
    return _run_portfolio_review_intraday_workflow(
        args,
        config=config,
        provider=provider,
        current_positions=current_positions,
    )


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


def _resolve_universe_profiles(
    raw_profiles: list[str] | None,
    config: AppConfig,
) -> tuple[str, ...]:
    configured_profiles = config.universe_builder.profiles
    if raw_profiles is None:
        return tuple(configured_profiles.keys())

    ordered_profiles = tuple(
        dict.fromkeys(profile.strip() for profile in raw_profiles if profile and profile.strip())
    )
    if not ordered_profiles:
        raise ValueError("At least one non-empty --profile value is required.")

    invalid_profiles = [profile for profile in ordered_profiles if profile not in configured_profiles]
    if invalid_profiles:
        valid_profiles = ", ".join(sorted(configured_profiles))
        invalid_rendered = ", ".join(invalid_profiles)
        raise ValueError(
            f"Unknown universe profile(s): {invalid_rendered}. Valid profiles: {valid_profiles}."
        )
    return ordered_profiles


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


def _parse_json_message(raw_value: str) -> dict[str, object] | str:
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON message '{raw_value}': {exc.msg}."
        ) from exc
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        return payload
    raise argparse.ArgumentTypeError(
        "Subscription messages must decode to a JSON object or JSON string."
    )


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


def _parse_metadata_json(raw_value: str | None) -> dict[str, object]:
    if raw_value is None or not raw_value.strip():
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata-json must contain valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("--metadata-json must decode to a JSON object.")
    return dict(payload)


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
