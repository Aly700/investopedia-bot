"""Read-only internal backend API queries and lightweight HTTP server."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from bot.config import AppConfig
from bot.data.pending_orders import (
    PendingOrderStateStore,
    build_pending_order_holding_exposures,
    pending_order_reserves_unmatched_slot,
)
from bot.logging_utils import get_logger
from bot.risk.portfolio_rules import (
    ExistingPosition,
    PortfolioInputError,
    load_existing_positions,
)
from bot.service.live_market_service import LiveMarketServiceStatusStore
from bot.state.market_state import (
    MarketStateStore,
    MarketStateSnapshot,
    diff_market_state_snapshots,
)
from bot.workflow_status import (
    build_workflow_input_overview,
    summarize_workflow_input_statuses,
)


LOGGER = get_logger(__name__)

INTERNAL_API_VERSION = 1
INTERNAL_API_ENDPOINTS = (
    "/",
    "/health",
    "/market-state",
    "/market-state/transitions",
    "/pending-orders",
    "/portfolio",
    "/analytics/trade-review",
)


@dataclass(frozen=True)
class InternalApiQueryService:
    """Read-only backend query surface over persisted operational state."""

    project_root: Path
    config: AppConfig | None = None

    def index(self) -> dict[str, Any]:
        return {
            "service": "investopedia-bot-internal-api",
            "version": INTERNAL_API_VERSION,
            "project_root": str(self.project_root.resolve()),
            "endpoints": list(INTERNAL_API_ENDPOINTS),
        }

    def health(self) -> dict[str, Any]:
        status_store = LiveMarketServiceStatusStore(self.project_root)
        artifact = status_store.load()
        status_available = artifact.updated_at_utc is not None
        return _endpoint_response(
            endpoint="health",
            available=status_available,
            not_available_reason=(
                None if status_available else "live_service_status_not_available_yet"
            ),
            artifact_paths={
                "live_market_service_status": str(status_store.path.resolve()),
            },
            data={
                "updated_at_utc": artifact.updated_at_utc,
                "status": artifact.status.to_dict(),
            },
        )

    def market_state(self) -> dict[str, Any]:
        market_state_store = MarketStateStore(self.project_root)
        current_snapshot = market_state_store.load_current()
        previous_snapshot = market_state_store.load_previous()
        if current_snapshot is None:
            return _endpoint_response(
                endpoint="market_state",
                available=False,
                not_available_reason="market_state_not_available_yet",
                artifact_paths={
                    "current_market_state": str(market_state_store.current_path.resolve()),
                    "previous_market_state": str(market_state_store.previous_path.resolve()),
                },
                data=None,
            )

        change_set = diff_market_state_snapshots(previous_snapshot, current_snapshot)
        top_priority_candidates = [
            candidate.to_dict()
            for candidate in current_snapshot.approved_candidate_queue
            if candidate.priority_bucket == "top_priority"
        ]
        recommendation_state = _market_state_recommendation_state(
            project_root=self.project_root,
            current_snapshot=current_snapshot,
            top_priority_candidates=top_priority_candidates,
        )
        return _endpoint_response(
            endpoint="market_state",
            available=True,
            artifact_paths={
                "current_market_state": str(market_state_store.current_path.resolve()),
                "previous_market_state": str(market_state_store.previous_path.resolve()),
            },
            data={
                "snapshot": current_snapshot.to_dict(),
                "current_alertable_states": [
                    state.to_dict() for state in current_snapshot.current_alertable_states
                ],
                "current_action_states_by_symbol": {
                    symbol: state.to_dict()
                    for symbol, state in sorted(
                        current_snapshot.current_action_states_by_symbol.items(),
                        key=lambda item: item[0],
                    )
                },
                "top_priority_candidates": top_priority_candidates,
                "recommendation_state": recommendation_state,
                "recent_transitions": [transition.to_dict() for transition in change_set.transitions],
                "transition_count": change_set.transition_count,
                "baseline_established": change_set.baseline_established,
            },
        )

    def market_state_transitions(self) -> dict[str, Any]:
        market_state_store = MarketStateStore(self.project_root)
        current_snapshot = market_state_store.load_current()
        previous_snapshot = market_state_store.load_previous()
        if current_snapshot is None:
            return _endpoint_response(
                endpoint="market_state_transitions",
                available=False,
                not_available_reason="market_state_not_available_yet",
                artifact_paths={
                    "current_market_state": str(market_state_store.current_path.resolve()),
                    "previous_market_state": str(market_state_store.previous_path.resolve()),
                },
                data=None,
            )
        change_set = diff_market_state_snapshots(previous_snapshot, current_snapshot)
        return _endpoint_response(
            endpoint="market_state_transitions",
            available=True,
            artifact_paths={
                "current_market_state": str(market_state_store.current_path.resolve()),
                "previous_market_state": str(market_state_store.previous_path.resolve()),
            },
            data=change_set.to_dict(),
        )

    def pending_orders(self) -> dict[str, Any]:
        pending_order_store = PendingOrderStateStore(self.project_root)
        pending_order_state = pending_order_store.load()
        market_state_store = MarketStateStore(self.project_root)
        current_snapshot = market_state_store.load_current()
        if current_snapshot is None:
            current_positions = []
            position_context_available = False
            position_warnings = (
                "Live portfolio context is not available because no current market-state snapshot exists yet.",
            )
        else:
            (
                current_positions,
                position_warnings,
                position_context_available,
            ) = _load_current_positions(current_snapshot.portfolio_path)
        max_concurrent_positions = _max_concurrent_positions(self.config)
        active_orders = list(pending_order_state.active_orders())
        capacity_reserving_orders = [
            record for record in active_orders if record.reserves_capacity
        ]
        reserved_notional = sum(
            record.reserved_notional for record in capacity_reserving_orders
        )
        exposures = build_pending_order_holding_exposures(
            pending_order_state,
            current_positions=current_positions,
        )
        pending_fill_notional: float | None = None
        if position_context_available:
            pending_fill_notional = max(
                sum(exposure.approximate_notional for exposure in exposures) - reserved_notional,
                0.0,
            )
        reserved_slot_count: int | None = None
        if position_context_available:
            reserved_slot_count = sum(
                1
                for record in capacity_reserving_orders
                if pending_order_reserves_unmatched_slot(
                    record,
                    current_positions=current_positions,
                )
            )
        pending_order_count_by_sector: dict[str, int] = {}
        reserved_notional_by_sector: dict[str, float] = {}
        for record in capacity_reserving_orders:
            sector_name = record.sector_name.strip() if record.sector_name else "Unknown"
            pending_order_count_by_sector[sector_name] = (
                pending_order_count_by_sector.get(sector_name, 0) + 1
            )
            reserved_notional_by_sector[sector_name] = (
                reserved_notional_by_sector.get(sector_name, 0.0) + record.reserved_notional
            )
        available_slots = None
        current_position_notional = None
        if position_context_available:
            available_slots = (
                max(max_concurrent_positions - len(current_positions) - reserved_slot_count, 0)
                if max_concurrent_positions is not None
                else None
            )
            current_position_notional = sum(
                float(position.shares) * float(position.average_entry_price)
                for position in current_positions
            )
        return _endpoint_response(
            endpoint="pending_orders",
            available=True,
            artifact_paths={
                "pending_orders": str(pending_order_store.path.resolve()),
            },
            warnings=position_warnings,
            data={
                "schema_version": pending_order_state.schema_version,
                "updated_at": pending_order_state.updated_at,
                "active_order_count": len(active_orders),
                "summary": {
                    "active_order_count": len(active_orders),
                    "capacity_reserving_buy_order_count": len(capacity_reserving_orders),
                    "reserved_notional": reserved_notional,
                    "pending_fill_notional": pending_fill_notional,
                    "reserved_slot_count": reserved_slot_count,
                    "current_position_notional": current_position_notional,
                    "position_context_available": position_context_available,
                    "available_capital": None,
                    "current_equity": None,
                    "equity_context_available": False,
                    "available_slots": available_slots,
                    "pending_symbols": sorted(
                        {record.symbol for record in capacity_reserving_orders}
                    ),
                    "pending_order_count_by_sector": dict(
                        sorted(pending_order_count_by_sector.items())
                    ),
                    "reserved_notional_by_sector": dict(
                        sorted(reserved_notional_by_sector.items())
                    ),
                },
                "active_orders": [record.to_dict() for record in active_orders],
            },
        )

    def portfolio(self) -> dict[str, Any]:
        current_snapshot = MarketStateStore(self.project_root).load_current()
        if current_snapshot is None:
            return _endpoint_response(
                endpoint="portfolio",
                available=False,
                not_available_reason="market_state_not_available_yet",
                data=None,
            )
        action_states = [
            state.to_dict()
            for _, state in sorted(
                current_snapshot.current_action_states_by_symbol.items(),
                key=lambda item: item[0],
            )
        ]
        workflow_input_summaries = _load_workflow_input_summaries_for_date(
            self.project_root,
            as_of_date=current_snapshot.as_of_date,
        )
        return _endpoint_response(
            endpoint="portfolio",
            available=True,
            data={
                "portfolio_review_summary": (
                    current_snapshot.portfolio_review_summary.to_dict()
                    if current_snapshot.portfolio_review_summary is not None
                    else None
                ),
                "intraday_review_summary": (
                    current_snapshot.intraday_review_summary.to_dict()
                    if current_snapshot.intraday_review_summary is not None
                    else None
                ),
                "current_action_states": action_states,
                "exit_candidates": [
                    state["symbol"]
                    for state in action_states
                    if state["action"] == "EXIT CANDIDATE"
                ],
                "watch_closely": [
                    state["symbol"]
                    for state in action_states
                    if state["action"] == "WATCH CLOSELY"
                ],
                "raise_stop": [
                    state["symbol"]
                    for state in action_states
                    if state["action"] == "RAISE STOP"
                ],
                "workflow_input_overview": build_workflow_input_overview(
                    {
                        workflow_name: summary
                        for workflow_name, summary in workflow_input_summaries.items()
                        if workflow_name in {"portfolio_review", "intraday_review"}
                    }
                ).to_dict(),
                "workflow_input_summaries": {
                    workflow_name: summary
                    for workflow_name, summary in workflow_input_summaries.items()
                    if workflow_name in {"portfolio_review", "intraday_review"}
                },
            },
        )

    def trade_review_analytics(self) -> dict[str, Any]:
        latest_summary_path, payload, warnings = _latest_trade_review_summary(
            self.project_root
        )
        if latest_summary_path is None:
            return _endpoint_response(
                endpoint="trade_review_analytics",
                available=False,
                not_available_reason="trade_review_analytics_not_available_yet",
                data=None,
            )
        if payload is None:
            return _endpoint_response(
                endpoint="trade_review_analytics",
                available=False,
                not_available_reason="trade_review_analytics_unreadable",
                artifact_paths={
                    "trade_review_analytics": str(latest_summary_path.resolve()),
                },
                warnings=warnings,
                data=None,
            )
        return _endpoint_response(
            endpoint="trade_review_analytics",
            available=True,
            artifact_paths={
                "trade_review_analytics": str(latest_summary_path.resolve()),
            },
            data={
                "summary_path": str(latest_summary_path.resolve()),
                "summary": dict(payload),
            },
        )


class InternalApiHttpServer(ThreadingHTTPServer):
    """Small HTTP server that exposes the internal query surface."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        query_service: InternalApiQueryService,
    ) -> None:
        self.query_service = query_service
        super().__init__(server_address, InternalApiRequestHandler)


class InternalApiRequestHandler(BaseHTTPRequestHandler):
    """Serve the read-only internal API as compact JSON."""

    server: InternalApiHttpServer

    def do_GET(self) -> None:
        status, payload = internal_api_response_for_path(
            query_service=self.server.query_service,
            path=self.path,
        )
        self._write_json(status, payload)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("internal-api %s", format % args)

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_internal_api_server(
    *,
    query_service: InternalApiQueryService,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> InternalApiHttpServer:
    """Return a configured local HTTP server for the internal API."""

    return InternalApiHttpServer((host, port), query_service=query_service)


def serve_internal_api(
    *,
    query_service: InternalApiQueryService,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run the local internal API server until interrupted."""

    server = create_internal_api_server(
        query_service=query_service,
        host=host,
        port=port,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def internal_api_response_for_path(
    *,
    query_service: InternalApiQueryService,
    path: str,
) -> tuple[HTTPStatus, dict[str, Any]]:
    """Return the JSON response payload for one internal API path."""

    normalized_path = urlparse(path).path.rstrip("/") or "/"
    routes: dict[str, Callable[[], dict[str, Any]]] = {
        "/": query_service.index,
        "/health": query_service.health,
        "/market-state": query_service.market_state,
        "/market-state/transitions": query_service.market_state_transitions,
        "/pending-orders": query_service.pending_orders,
        "/portfolio": query_service.portfolio,
        "/analytics/trade-review": query_service.trade_review_analytics,
    }
    handler = routes.get(normalized_path)
    if handler is None:
        return HTTPStatus.NOT_FOUND, {
            "error": "not_found",
            "path": normalized_path,
            "endpoints": list(INTERNAL_API_ENDPOINTS),
        }
    return HTTPStatus.OK, handler()


def _endpoint_response(
    *,
    endpoint: str,
    available: bool,
    data: Mapping[str, Any] | None,
    not_available_reason: str | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "available": available,
        "not_available_reason": not_available_reason,
        "artifact_paths": dict(sorted((artifact_paths or {}).items())),
        "warnings": list(warnings),
        "data": dict(data) if data is not None else None,
    }


def _load_current_positions(
    portfolio_path: str | None,
) -> tuple[list[ExistingPosition], tuple[str, ...], bool]:
    if portfolio_path is None:
        return (
            [],
            (
                "Live portfolio context is not available because the current market-state snapshot does not include a portfolio path.",
            ),
            False,
        )
    resolved_path = Path(portfolio_path).resolve()
    if not resolved_path.exists():
        return (
            [],
            (f"Portfolio snapshot does not exist at {resolved_path}.",),
            False,
        )
    try:
        return load_existing_positions(resolved_path), (), True
    except (PortfolioInputError, OSError, UnicodeDecodeError) as exc:
        return [], (f"Failed to load portfolio snapshot at {resolved_path}: {exc}",), False


def _market_state_recommendation_state(
    *,
    project_root: Path,
    current_snapshot: MarketStateSnapshot,
    top_priority_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    latest_monitor_market_output_path, latest_monitor_market_output_date = (
        _latest_monitor_market_output(project_root)
    )
    workflow_input_summaries = _load_workflow_input_summaries_for_date(
        project_root,
        as_of_date=current_snapshot.as_of_date,
    )
    empty_reasons: list[str] = []
    if not current_snapshot.approved_candidate_queue:
        if current_snapshot.top_rejected_reasons_summary:
            rejection_summary = "; ".join(
                f"{reason.reason} ({reason.count})"
                for reason in current_snapshot.top_rejected_reasons_summary
            )
            empty_reasons.append(
                f"No approved candidates remain after screening. Top rejection reasons: {rejection_summary}."
            )
        else:
            empty_reasons.append(
                "No approved candidates are present in the current market-state snapshot."
            )
    if not top_priority_candidates and current_snapshot.approved_candidate_queue:
        empty_reasons.append(
            "Approved candidates are present, but none are currently marked top priority."
        )
    return {
        "approved_candidate_count": len(current_snapshot.approved_candidate_queue),
        "top_priority_candidate_count": len(top_priority_candidates),
        "queue_empty": not current_snapshot.approved_candidate_queue,
        "top_priority_empty": not top_priority_candidates,
        "empty_reasons": empty_reasons,
        "workflow_input_overview": build_workflow_input_overview(
            workflow_input_summaries
        ).to_dict(),
        "workflow_input_summaries": workflow_input_summaries,
        "context_notes": _market_state_recommendation_context_notes(
            current_snapshot=current_snapshot,
            latest_monitor_market_output_date=latest_monitor_market_output_date,
        ),
        "latest_successful_monitor_market_as_of_date": (
            latest_monitor_market_output_date.isoformat()
            if latest_monitor_market_output_date is not None
            else None
        ),
        "latest_successful_monitor_market_output_path": (
            str(latest_monitor_market_output_path.resolve())
            if latest_monitor_market_output_path is not None
            else None
        ),
        "monitor_market_fresh_for_snapshot_date": (
            latest_monitor_market_output_date == current_snapshot.as_of_date
            if latest_monitor_market_output_date is not None
            else False
        ),
    }


def _market_state_recommendation_context_notes(
    *,
    current_snapshot: MarketStateSnapshot,
    latest_monitor_market_output_date: date | None,
) -> list[str]:
    notes: list[str] = []
    if (
        current_snapshot.benchmark_context is None
        or current_snapshot.benchmark_context.benchmark_return is None
    ):
        notes.append("Benchmark context is unavailable in the current market-state snapshot.")
    if (
        current_snapshot.volatility_context is None
        or current_snapshot.volatility_context.vix_close is None
    ):
        notes.append("Volatility context is unavailable in the current market-state snapshot.")
    if not current_snapshot.sector_context_summary:
        notes.append("Sector context summary is empty in the current market-state snapshot.")
    if latest_monitor_market_output_date is None:
        notes.append("No successful monitor-market output artifact has been written yet.")
    elif latest_monitor_market_output_date < current_snapshot.as_of_date:
        notes.append(
            "No successful monitor-market output artifact was written for "
            f"{current_snapshot.as_of_date.isoformat()}; latest successful monitor-market "
            f"output is {latest_monitor_market_output_date.isoformat()}."
        )
    return notes


def _latest_monitor_market_output(project_root: Path) -> tuple[Path | None, date | None]:
    candidate_paths: list[Path] = []
    for root_name in ("monitor_market", "live_market"):
        monitor_market_root = project_root / "data" / "processed" / root_name
        if not monitor_market_root.exists():
            continue
        try:
            candidate_paths.extend(monitor_market_root.rglob("market_monitor.json"))
        except OSError as exc:
            LOGGER.warning(
                "Ignoring monitor-market artifact lookup under %s because the directory could not be scanned: %s",
                monitor_market_root,
                exc,
            )
    latest_candidate: tuple[date, int, Path] | None = None
    for path in candidate_paths:
        try:
            artifact_date = date.fromisoformat(path.parent.name)
            mtime_ns = path.stat().st_mtime_ns
        except (OSError, ValueError):
            continue
        candidate = (artifact_date, mtime_ns, path)
        if latest_candidate is None or candidate > latest_candidate:
            latest_candidate = candidate
    if latest_candidate is None:
        return None, None
    return latest_candidate[2], latest_candidate[0]


def _load_workflow_input_summaries_for_date(
    project_root: Path,
    *,
    as_of_date: date,
) -> dict[str, dict[str, Any]]:
    artifact_specs = {
        "daily_summary": (
            "daily_summary.json",
            "research_input_statuses",
            ("live_market", "monitor_market", "daily_summary"),
        ),
        "portfolio_review": (
            "portfolio_review.json",
            "review_input_statuses",
            ("live_market", "monitor_market", "portfolio_review"),
        ),
        "intraday_review": (
            "portfolio_review_intraday.json",
            "review_input_statuses",
            ("live_market", "portfolio_review_intraday"),
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    for workflow_name, (filename, raw_status_key, roots) in artifact_specs.items():
        payload = _load_latest_workflow_artifact_payload_for_date(
            project_root,
            as_of_date=as_of_date,
            filename=filename,
            processed_roots=roots,
        )
        if payload is None:
            continue
        summary = _workflow_input_summary_from_artifact_payload(
            payload,
            raw_status_key=raw_status_key,
        )
        if summary is not None:
            results[workflow_name] = summary
    return results


def _load_latest_workflow_artifact_payload_for_date(
    project_root: Path,
    *,
    as_of_date: date,
    filename: str,
    processed_roots: Sequence[str],
) -> Mapping[str, Any] | None:
    best_candidate: tuple[int, Mapping[str, Any]] | None = None
    for priority, root_name in enumerate(processed_roots):
        path = (
            project_root
            / "data"
            / "processed"
            / root_name
            / as_of_date.isoformat()
            / filename
        )
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Ignoring workflow artifact %s because it could not be loaded: %s",
                path,
                exc,
            )
            continue
        if not isinstance(payload, Mapping):
            continue
        candidate = (priority, dict(payload))
        if best_candidate is None or candidate[0] < best_candidate[0]:
            best_candidate = candidate
    if best_candidate is None:
        return None
    return best_candidate[1]


def _workflow_input_summary_from_artifact_payload(
    payload: Mapping[str, Any],
    *,
    raw_status_key: str,
) -> dict[str, Any] | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    summary = metadata.get("workflow_input_summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    raw_statuses = metadata.get(raw_status_key)
    if not isinstance(raw_statuses, Mapping):
        return None
    return summarize_workflow_input_statuses(raw_statuses).to_dict()


@dataclass(frozen=True)
class _TradeReviewSummaryCandidate:
    path: Path
    payload: Mapping[str, Any] | None
    payload_date: date | None
    mtime_ns: int
    warning: str | None = None


def _latest_trade_review_summary(
    project_root: Path,
) -> tuple[Path | None, Mapping[str, Any] | None, tuple[str, ...]]:
    analytics_root = (
        project_root
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
    )
    if not analytics_root.exists():
        return None, None, ()
    try:
        candidate_paths = list(analytics_root.rglob("trade_review_analytics.json"))
    except OSError as exc:
        LOGGER.warning(
            "Ignoring trade review analytics lookup under %s because the directory could not be scanned: %s",
            analytics_root,
            exc,
        )
        return None, None, ()
    candidates = tuple(
        candidate
        for path in candidate_paths
        if (candidate := _load_trade_review_summary_candidate(path)) is not None
    )
    if not candidates:
        return None, None, ()
    dated_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.payload is not None and candidate.payload_date is not None
    )
    if dated_candidates:
        selected = max(
            dated_candidates,
            key=lambda candidate: (
                candidate.payload_date,
                candidate.mtime_ns,
                str(candidate.path),
            ),
        )
        return selected.path, selected.payload, ()
    readable_candidates = tuple(
        candidate for candidate in candidates if candidate.payload is not None
    )
    if readable_candidates:
        selected = max(
            readable_candidates,
            key=lambda candidate: (candidate.mtime_ns, str(candidate.path)),
        )
        return selected.path, selected.payload, ()
    selected = max(
        candidates,
        key=lambda candidate: (candidate.mtime_ns, str(candidate.path)),
    )
    warnings = (selected.warning,) if selected.warning is not None else ()
    return selected.path, None, warnings


def _load_trade_review_summary_candidate(
    path: Path,
) -> _TradeReviewSummaryCandidate | None:
    mtime_ns = _safe_path_mtime_ns(path)
    if mtime_ns is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("trade review analytics payload must be a mapping.")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "Ignoring trade review analytics at %s because it could not be loaded: %s",
            path,
            exc,
        )
        return _TradeReviewSummaryCandidate(
            path=path,
            payload=None,
            payload_date=None,
            mtime_ns=mtime_ns,
            warning=str(exc),
        )
    return _TradeReviewSummaryCandidate(
        path=path,
        payload=dict(payload),
        payload_date=_trade_review_payload_date(payload),
        mtime_ns=mtime_ns,
    )


def _trade_review_payload_date(payload: Mapping[str, Any]) -> date | None:
    for key in ("window_end_date", "as_of_date"):
        raw_value = payload.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        try:
            return date.fromisoformat(raw_value.strip())
        except ValueError:
            continue
    return None


def _safe_path_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError as exc:
        LOGGER.warning(
            "Ignoring trade review analytics at %s because metadata could not be read: %s",
            path,
            exc,
        )
        return None


def _max_concurrent_positions(config: AppConfig | None) -> int | None:
    if config is None:
        return None
    return int(config.game_rules.rules.max_positions)
