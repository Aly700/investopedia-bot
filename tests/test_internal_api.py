from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
import os
from pathlib import Path

import pytest

import bot.api.internal_api as internal_api_module
from bot.api.internal_api import (
    InternalApiQueryService,
    internal_api_response_for_path,
)
from bot.config import load_app_config
from bot.data.pending_orders import (
    PendingOrderRecord,
    create_pending_order,
    default_pending_order_state_path,
    load_pending_order_state,
    write_pending_order_state,
)
from bot.data.state_persistence import write_json_file
from bot.service.live_market_service import (
    LiveMarketServiceStatus,
    LiveMarketServiceStatusStore,
)
from bot.state.market_state import (
    MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
    MarketStateActionState,
    MarketStateAlertableState,
    MarketStateCandidateState,
    MarketStatePortfolioSummary,
    MarketStateSnapshot,
    default_current_market_state_path,
    default_previous_market_state_path,
)


def test_internal_api_market_state_returns_snapshot_and_recent_transitions(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    _write_market_state_snapshots(tmp_path)

    payload = query_service.market_state()

    assert payload["available"] is True
    assert payload["data"]["snapshot"]["as_of_date"] == "2024-01-05"
    assert payload["data"]["top_priority_candidates"][0]["symbol"] == "NVDA"
    assert payload["data"]["current_alertable_states"][0]["symbol"] == "AAPL"
    assert payload["data"]["recent_transitions"][0]["transition_type"] == "HOLD_TO_WATCH_CLOSELY"


def test_internal_api_market_state_degrades_cleanly_when_missing_or_malformed(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)

    missing_payload = query_service.market_state()
    assert missing_payload["available"] is False
    assert missing_payload["not_available_reason"] == "market_state_not_available_yet"

    current_path = default_current_market_state_path(tmp_path)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text("{bad-json", encoding="utf-8")

    malformed_payload = query_service.market_state()
    assert malformed_payload["available"] is False
    assert malformed_payload["not_available_reason"] == "market_state_not_available_yet"


def test_internal_api_health_reports_defaults_and_persisted_status(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)

    default_payload = query_service.health()
    assert default_payload["available"] is False
    assert default_payload["data"]["status"]["service_state"] == "starting"

    status_store = LiveMarketServiceStatusStore(tmp_path)
    status_store.save(
        LiveMarketServiceStatus(
            service_state="connected",
            connected=True,
            cycle_count=3,
            last_successful_flush_at_utc="2024-01-05T15:05:00+00:00",
        ),
        updated_at_utc="2024-01-05T15:06:00+00:00",
    )

    persisted_payload = query_service.health()
    assert persisted_payload["available"] is True
    assert persisted_payload["data"]["updated_at_utc"] == "2024-01-05T15:06:00+00:00"
    assert persisted_payload["data"]["status"]["connected"] is True
    assert persisted_payload["data"]["status"]["cycle_count"] == 3


def test_internal_api_pending_orders_returns_current_summary(
    tmp_path: Path,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    portfolio_path = tmp_path / "portfolio.csv"
    _write_portfolio_csv(portfolio_path)
    _write_market_state_snapshots(tmp_path, portfolio_path=str(portfolio_path))
    _write_pending_orders(
        tmp_path,
        PendingOrderRecord(
            order_id="ord_aapl_1",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T14:30:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            sector_name="Technology",
            industry_name="Software",
        ),
    )
    query_service = InternalApiQueryService(project_root=tmp_path, config=config)

    payload = query_service.pending_orders()

    assert payload["available"] is True
    assert payload["data"]["active_order_count"] == 1
    assert payload["data"]["summary"]["position_context_available"] is True
    assert payload["data"]["summary"]["reserved_notional"] == pytest.approx(1_000.0)
    assert payload["data"]["summary"]["reserved_slot_count"] == 1
    assert payload["data"]["summary"]["current_position_notional"] == pytest.approx(0.0)
    assert payload["data"]["summary"]["available_slots"] == (
        config.game_rules.rules.max_positions - 1
    )
    assert payload["data"]["summary"]["pending_order_count_by_sector"] == {"Technology": 1}
    assert payload["data"]["active_orders"][0]["symbol"] == "AAPL"


def test_internal_api_pending_orders_marks_position_context_unknown_without_market_state(
    tmp_path: Path,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    _write_pending_orders(
        tmp_path,
        PendingOrderRecord(
            order_id="ord_msft_1",
            symbol="MSFT",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T14:30:00+00:00",
            status="pending",
            requested_quantity=5,
            requested_price=200.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            sector_name="Technology",
        ),
    )
    query_service = InternalApiQueryService(project_root=tmp_path, config=config)

    payload = query_service.pending_orders()

    assert payload["available"] is True
    assert payload["data"]["summary"]["position_context_available"] is False
    assert payload["data"]["summary"]["available_slots"] is None
    assert payload["data"]["summary"]["current_position_notional"] is None
    assert payload["data"]["summary"]["pending_fill_notional"] is None
    assert payload["warnings"]


def test_internal_api_pending_orders_marks_unreadable_portfolio_context_unavailable(
    tmp_path: Path,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    portfolio_path = tmp_path / "portfolio.csv"
    portfolio_path.write_bytes(b"\xff\xfe\x00\x00")
    _write_market_state_snapshots(tmp_path, portfolio_path=str(portfolio_path))
    _write_pending_orders(
        tmp_path,
        PendingOrderRecord(
            order_id="ord_nvda_1",
            symbol="NVDA",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T14:30:00+00:00",
            status="pending",
            requested_quantity=2,
            requested_price=500.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            sector_name="Technology",
        ),
    )
    query_service = InternalApiQueryService(project_root=tmp_path, config=config)

    payload = query_service.pending_orders()

    assert payload["available"] is True
    assert payload["data"]["summary"]["position_context_available"] is False
    assert payload["data"]["summary"]["current_position_notional"] is None
    assert payload["data"]["summary"]["available_slots"] is None
    assert payload["warnings"]


def test_internal_api_pending_orders_includes_all_active_orders_but_reserves_from_buys(
    tmp_path: Path,
) -> None:
    config = replace(load_app_config(), project_root=tmp_path)
    _write_pending_orders(
        tmp_path,
        PendingOrderRecord(
            order_id="ord_buy_aapl",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            created_at="2024-01-05T14:30:00+00:00",
            status="pending",
            requested_quantity=10,
            requested_price=100.0,
            reserved_notional=1_000.0,
            reserved_slot=True,
            sector_name="Technology",
        ),
        PendingOrderRecord(
            order_id="ord_sell_msft",
            symbol="MSFT",
            side="SELL",
            order_type="LIMIT",
            created_at="2024-01-05T14:31:00+00:00",
            status="pending",
            requested_quantity=5,
            requested_price=300.0,
            reserved_notional=0.0,
            reserved_slot=False,
            sector_name="Technology",
        ),
    )
    query_service = InternalApiQueryService(project_root=tmp_path, config=config)

    payload = query_service.pending_orders()

    assert payload["data"]["active_order_count"] == 2
    assert payload["data"]["summary"]["capacity_reserving_buy_order_count"] == 1
    assert payload["data"]["summary"]["reserved_notional"] == pytest.approx(1_000.0)
    assert sorted(order["symbol"] for order in payload["data"]["active_orders"]) == ["AAPL", "MSFT"]


def test_internal_api_trade_review_analytics_handles_malformed_summary_safely(
    tmp_path: Path,
) -> None:
    analytics_path = (
        tmp_path
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / "2024-01-10_last_30_days"
        / "trade_review_analytics.json"
    )
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    analytics_path.write_text("{bad-json", encoding="utf-8")
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.trade_review_analytics()

    assert payload["available"] is False
    assert payload["not_available_reason"] == "trade_review_analytics_unreadable"
    assert payload["artifact_paths"]["trade_review_analytics"] == str(analytics_path.resolve())
    assert payload["warnings"]


def test_internal_api_trade_review_analytics_prefers_payload_date_over_newer_mtime(
    tmp_path: Path,
) -> None:
    older_path = (
        tmp_path
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / "2024-01-10_last_30_days"
        / "trade_review_analytics.json"
    )
    newer_path = (
        tmp_path
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / "2024-01-12_last_30_days"
        / "trade_review_analytics.json"
    )
    older_path.parent.mkdir(parents=True, exist_ok=True)
    newer_path.parent.mkdir(parents=True, exist_ok=True)
    older_path.write_text(
        json.dumps({"as_of_date": "2024-01-10", "window_end_date": "2024-01-10"}),
        encoding="utf-8",
    )
    newer_path.write_text(
        json.dumps({"as_of_date": "2024-01-12", "window_end_date": "2024-01-12"}),
        encoding="utf-8",
    )
    os.utime(older_path, (2_000_000_000, 2_000_000_000))
    os.utime(newer_path, (1_000_000_000, 1_000_000_000))
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.trade_review_analytics()

    assert payload["available"] is True
    assert payload["data"]["summary"]["window_end_date"] == "2024-01-12"
    assert payload["artifact_paths"]["trade_review_analytics"] == str(newer_path.resolve())


def test_internal_api_trade_review_analytics_ignores_stat_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flaky_path = (
        tmp_path
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / "2024-01-11_last_30_days"
        / "trade_review_analytics.json"
    )
    good_path = (
        tmp_path
        / "data"
        / "processed"
        / "analytics"
        / "trade_review"
        / "2024-01-12_last_30_days"
        / "trade_review_analytics.json"
    )
    flaky_path.parent.mkdir(parents=True, exist_ok=True)
    good_path.parent.mkdir(parents=True, exist_ok=True)
    flaky_path.write_text(
        json.dumps({"as_of_date": "2024-01-11", "window_end_date": "2024-01-11"}),
        encoding="utf-8",
    )
    good_path.write_text(
        json.dumps({"as_of_date": "2024-01-12", "window_end_date": "2024-01-12"}),
        encoding="utf-8",
    )
    original_safe_path_mtime_ns = internal_api_module._safe_path_mtime_ns

    def flaky_safe_path_mtime_ns(path: Path) -> int | None:
        if path == flaky_path:
            return None
        return original_safe_path_mtime_ns(path)

    monkeypatch.setattr(
        internal_api_module,
        "_safe_path_mtime_ns",
        flaky_safe_path_mtime_ns,
    )
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.trade_review_analytics()

    assert payload["available"] is True
    assert payload["artifact_paths"]["trade_review_analytics"] == str(good_path.resolve())


def test_internal_api_route_dispatch_serves_market_state_endpoint(
    tmp_path: Path,
) -> None:
    _write_market_state_snapshots(tmp_path)
    status, payload = internal_api_response_for_path(
        query_service=InternalApiQueryService(project_root=tmp_path),
        path="/market-state",
    )

    assert int(status) == 200
    assert payload["endpoint"] == "market_state"
    assert payload["available"] is True
    assert payload["data"]["snapshot"]["as_of_date"] == "2024-01-05"


def test_internal_api_market_state_transitions_returns_change_set(
    tmp_path: Path,
) -> None:
    _write_market_state_snapshots(tmp_path)
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.market_state_transitions()

    assert payload["available"] is True
    assert payload["data"]["transition_count"] == 1
    assert payload["data"]["transitions"][0]["transition_type"] == "HOLD_TO_WATCH_CLOSELY"


def test_internal_api_portfolio_returns_current_review_state(
    tmp_path: Path,
) -> None:
    _write_market_state_snapshots(tmp_path)
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.portfolio()

    assert payload["available"] is True
    assert payload["data"]["watch_closely"] == ["AAPL"]
    assert payload["data"]["exit_candidates"] == []


def test_internal_api_portfolio_degrades_cleanly_when_market_state_missing(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.portfolio()

    assert payload["available"] is False
    assert payload["not_available_reason"] == "market_state_not_available_yet"


def _write_market_state_snapshots(
    project_root: Path,
    *,
    portfolio_path: str | None = None,
) -> None:
    previous_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-05T14:30:00+00:00",
        as_of_date=date(2024, 1, 5),
        portfolio_path=portfolio_path,
        source_workflows=("review-portfolio",),
        approved_candidate_queue=(
            MarketStateCandidateState(
                symbol="NVDA",
                rank=1,
                preset_name="standard_breakout",
                actionable_now=True,
                priority_bucket="top_priority",
            ),
        ),
        portfolio_review_summary=MarketStatePortfolioSummary(
            position_count=1,
            hold_count=1,
            watch_closely_count=0,
            exit_candidate_count=0,
            raise_stop_count=0,
        ),
        current_action_states_by_symbol={
            "AAPL": MarketStateActionState(
                symbol="AAPL",
                action="HOLD",
                source_workflow="review-portfolio",
            )
        },
    )
    current_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-05T15:00:00+00:00",
        as_of_date=date(2024, 1, 5),
        portfolio_path=portfolio_path,
        source_workflows=("monitor-market", "review-portfolio"),
        approved_candidate_queue=(
            MarketStateCandidateState(
                symbol="NVDA",
                rank=1,
                preset_name="standard_breakout",
                actionable_now=True,
                priority_bucket="top_priority",
            ),
        ),
        portfolio_review_summary=MarketStatePortfolioSummary(
            position_count=1,
            hold_count=0,
            watch_closely_count=1,
            exit_candidate_count=0,
            raise_stop_count=0,
            urgent_symbols=("AAPL",),
        ),
        current_alertable_states=(
            MarketStateAlertableState(
                state_key="AAPL:WATCH_CLOSELY",
                category="WATCH CLOSELY",
                source_workflow="review-portfolio",
                symbol="AAPL",
                action="WATCH CLOSELY",
                rationale="AAPL weakened relative to its stop buffer.",
            ),
        ),
        current_action_states_by_symbol={
            "AAPL": MarketStateActionState(
                symbol="AAPL",
                action="WATCH CLOSELY",
                source_workflow="review-portfolio",
                rationale="AAPL weakened relative to its stop buffer.",
            )
        },
    )
    write_json_file(
        default_previous_market_state_path(project_root),
        previous_snapshot.to_dict(),
    )
    write_json_file(
        default_current_market_state_path(project_root),
        current_snapshot.to_dict(),
    )


def _write_pending_orders(project_root: Path, *records: PendingOrderRecord) -> None:
    state = load_pending_order_state(default_pending_order_state_path(project_root))
    for record in records:
        state = create_pending_order(state, record).state
    write_pending_order_state(state, default_pending_order_state_path(project_root))


def _write_portfolio_csv(path: Path) -> None:
    path.write_text(
        "symbol,quantity,average_entry_price,current_stop,preset_name,source,metadata_json\n",
        encoding="utf-8",
    )
