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
    MarketStateRejectedReasonSummary,
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
    assert payload["data"]["top_priority_candidates"][0]["candidate_disposition"] == "actionable"
    assert payload["data"]["current_alertable_states"][0]["symbol"] == "AAPL"
    assert payload["data"]["recent_transitions"][0]["transition_type"] == "HOLD_TO_WATCH_CLOSELY"


def test_internal_api_market_state_reports_fresh_recommendation_state_when_queue_is_populated(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    _write_market_state_snapshots(tmp_path)
    monitor_output_path = write_json_file(
        tmp_path / "data" / "processed" / "live_market" / "2024-01-05" / "market_monitor.json",
        {
            "as_of_date": "2024-01-05",
            "recommendation_sections": {
                "actionable_buy_candidates": [
                    {
                        "symbol": "NVDA",
                        "candidate_disposition": "actionable",
                        "blocker_categories": [],
                        "blocker_severities": [],
                        "degraded_or_missing_contexts": [],
                    }
                ],
                "capacity_blocked_candidates": [],
                "sector_gated_candidates": [],
                "degraded_context_candidates": [],
                "fundamental_rejected_candidates": [],
            },
        },
    )
    write_json_file(
        tmp_path / "data" / "processed" / "live_market" / "2024-01-05" / "daily_summary.json",
        {
            "metadata": {
                "workflow_input_summary": {
                    "total_count": 2,
                    "ok_count": 1,
                    "degraded_count": 0,
                    "unavailable_count": 1,
                    "failed_count": 0,
                    "issue_count": 1,
                    "healthy": False,
                    "problematic_inputs": ["volatility_context"],
                    "issue_codes": ["unsupported_capability"],
                    "issues": [
                        {
                            "input_name": "volatility_context",
                            "status": "unavailable",
                            "role_name": "historical_bars",
                            "provider": "alpaca",
                            "issue_code": "unsupported_capability",
                            "message": "VIX symbols are not supported.",
                        }
                    ],
                }
            }
        },
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert recommendation_state["queue_empty"] is False
    assert recommendation_state["actionable_queue_empty"] is False
    assert recommendation_state["top_priority_empty"] is False
    assert recommendation_state["empty_reasons"] == []
    assert recommendation_state["actionable_candidate_count"] == 1
    assert recommendation_state["capacity_blocked_candidate_count"] == 0
    assert recommendation_state["candidate_disposition_counts"] == {"actionable": 1}
    assert recommendation_state["sector_gated_candidate_count"] == 0
    assert recommendation_state["degraded_context_candidate_count"] == 0
    assert recommendation_state["latest_successful_monitor_market_as_of_date"] == "2024-01-05"
    assert (
        recommendation_state["latest_successful_monitor_market_output_path"]
        == str(monitor_output_path.resolve())
    )
    assert recommendation_state["monitor_market_fresh_for_snapshot_date"] is True
    assert recommendation_state["workflow_input_overview"]["workflow_count"] == 1
    assert recommendation_state["workflow_input_overview"]["highest_severity"] == "unavailable"
    assert (
        recommendation_state["workflow_input_summaries"]["daily_summary"][
            "unavailable_count"
        ]
        == 1
    )


def test_internal_api_market_state_reports_capacity_blocked_candidates_when_no_top_priority_is_actionable(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    current_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-05T15:00:00+00:00",
        as_of_date=date(2024, 1, 5),
        source_workflows=("monitor-market",),
        approved_candidate_queue=(
            MarketStateCandidateState(
                symbol="NVDA",
                rank=1,
                preset_name="standard_breakout",
                actionable_now=False,
                priority_bucket="capacity_constrained",
                candidate_disposition="approved_capacity_blocked",
                blocker_categories=("capacity",),
                blocker_severities=("soft",),
                degraded_or_missing_contexts=("earnings_context",),
            ),
        ),
    )
    write_json_file(
        default_current_market_state_path(tmp_path),
        current_snapshot.to_dict(),
    )
    write_json_file(
        tmp_path / "data" / "processed" / "live_market" / "2024-01-05" / "market_monitor.json",
        {"as_of_date": "2024-01-05"},
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert recommendation_state["approved_candidate_count"] == 1
    assert recommendation_state["actionable_candidate_count"] == 0
    assert recommendation_state["capacity_blocked_candidate_count"] == 1
    assert recommendation_state["queue_empty"] is False
    assert recommendation_state["actionable_queue_empty"] is True
    assert recommendation_state["top_priority_empty"] is True
    assert recommendation_state["top_capacity_blocked_candidates"][0]["symbol"] == "NVDA"
    assert (
        recommendation_state["top_capacity_blocked_candidates"][0]["candidate_disposition"]
        == "approved_capacity_blocked"
    )
    assert recommendation_state["top_capacity_blocked_candidates"][0]["blocker_categories"] == [
        "capacity"
    ]
    assert recommendation_state["top_capacity_blocked_candidates"][0][
        "degraded_or_missing_contexts"
    ] == ["earnings_context"]
    assert recommendation_state["candidate_disposition_counts"] == {
        "approved_capacity_blocked": 1
    }
    assert recommendation_state["empty_reasons"] == [
        "No approved candidates are actionable right now; 1 approved candidate is blocked by current portfolio capacity."
    ]


def test_internal_api_market_state_surfaces_structured_recommendation_sections_from_monitor_market_artifact(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    _write_market_state_snapshots(tmp_path)
    write_json_file(
        tmp_path / "data" / "processed" / "live_market" / "2024-01-05" / "market_monitor.json",
        {
            "as_of_date": "2024-01-05",
            "recommendation_sections": {
                "actionable_buy_candidates": [
                    {
                        "symbol": "NVDA",
                        "candidate_disposition": "actionable",
                        "blocker_categories": [],
                        "blocker_severities": [],
                        "degraded_or_missing_contexts": [],
                    }
                ],
                "capacity_blocked_candidates": [],
                "sector_gated_candidates": [
                    {
                        "symbol": "AMD",
                        "candidate_disposition": "soft_gated",
                        "blocker_categories": ["sector_regime"],
                        "blocker_severities": ["soft"],
                        "degraded_or_missing_contexts": [],
                    }
                ],
                "degraded_context_candidates": [
                    {
                        "symbol": "SHOP",
                        "candidate_disposition": "hard_rejected",
                        "blocker_categories": ["sizing"],
                        "blocker_severities": ["hard"],
                        "degraded_or_missing_contexts": ["sector_context"],
                    }
                ],
                "fundamental_rejected_candidates": [
                    {
                        "symbol": "AAPL",
                        "candidate_disposition": "hard_rejected",
                        "blocker_categories": ["duplicate_position"],
                        "blocker_severities": ["hard"],
                        "degraded_or_missing_contexts": [],
                    }
                ],
            },
        },
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert recommendation_state["sector_gated_candidate_count"] == 1
    assert recommendation_state["degraded_context_candidate_count"] == 1
    assert recommendation_state["fundamental_rejected_candidate_count"] == 1
    assert recommendation_state["top_sector_gated_candidates"][0]["symbol"] == "AMD"
    assert recommendation_state["top_sector_gated_candidates"][0]["blocker_categories"] == [
        "sector_regime"
    ]
    assert recommendation_state["top_degraded_context_candidates"][0]["symbol"] == "SHOP"
    assert recommendation_state["top_degraded_context_candidates"][0][
        "degraded_or_missing_contexts"
    ] == ["sector_context"]
    assert recommendation_state["top_fundamental_rejected_candidates"][0]["symbol"] == "AAPL"
    assert recommendation_state["candidate_disposition_counts"] == {
        "actionable": 1,
        "soft_gated": 1,
        "hard_rejected": 2,
    }


def test_internal_api_market_state_treats_archived_live_market_output_as_successful_with_zero_approved_candidates(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "runtime-root"
    query_service = InternalApiQueryService(project_root=project_root)
    current_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-05T15:00:00+00:00",
        as_of_date=date(2024, 1, 5),
        source_workflows=("monitor-market",),
        top_rejected_reasons_summary=(
            MarketStateRejectedReasonSummary(
                reason="Max concurrent positions reached.",
                count=5,
            ),
        ),
    )
    write_json_file(
        default_current_market_state_path(project_root),
        current_snapshot.to_dict(),
    )
    monitor_output_path = write_json_file(
        tmp_path / "archive" / "live_market" / "2024-01-05" / "market_monitor.json",
        {"as_of_date": "2024-01-05"},
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert recommendation_state["approved_candidate_count"] == 0
    assert recommendation_state["queue_empty"] is True
    assert recommendation_state["latest_successful_monitor_market_as_of_date"] == "2024-01-05"
    assert (
        recommendation_state["latest_successful_monitor_market_output_path"]
        == str(monitor_output_path.resolve())
    )
    assert recommendation_state["monitor_market_fresh_for_snapshot_date"] is True
    assert (
        "No successful monitor-market output artifact has been written yet."
        not in recommendation_state["context_notes"]
    )
    assert (
        "No successful monitor-market output artifact was written for 2024-01-05; latest successful monitor-market output is 2024-01-04."
        not in recommendation_state["context_notes"]
    )


def test_internal_api_market_state_surfaces_empty_queue_reasons_and_stale_monitor_market_output(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    previous_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-04T15:00:00+00:00",
        as_of_date=date(2024, 1, 4),
        source_workflows=("monitor-market",),
    )
    current_snapshot = MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp="2024-01-05T15:00:00+00:00",
        as_of_date=date(2024, 1, 5),
        source_workflows=("monitor-market",),
        top_rejected_reasons_summary=(
            MarketStateRejectedReasonSummary(
                reason="Max concurrent positions reached.",
                count=5,
            ),
        ),
    )
    write_json_file(
        default_previous_market_state_path(tmp_path),
        previous_snapshot.to_dict(),
    )
    write_json_file(
        default_current_market_state_path(tmp_path),
        current_snapshot.to_dict(),
    )
    write_json_file(
        tmp_path / "data" / "processed" / "monitor_market" / "2024-01-04" / "market_monitor.json",
        {"as_of_date": "2024-01-04"},
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert payload["data"]["baseline_established"] is False
    assert recommendation_state["queue_empty"] is True
    assert recommendation_state["top_priority_empty"] is True
    assert recommendation_state["latest_successful_monitor_market_as_of_date"] == "2024-01-04"
    assert recommendation_state["monitor_market_fresh_for_snapshot_date"] is False
    assert recommendation_state["empty_reasons"] == [
        "No approved candidates remain after screening. Top rejection reasons: Max concurrent positions reached. (5)."
    ]
    assert (
        "Benchmark context is unavailable in the current market-state snapshot."
        in recommendation_state["context_notes"]
    )
    assert (
        "Volatility context is unavailable in the current market-state snapshot."
        in recommendation_state["context_notes"]
    )
    assert (
        "Sector context summary is empty in the current market-state snapshot."
        in recommendation_state["context_notes"]
    )
    assert (
        "No successful monitor-market output artifact was written for 2024-01-05; latest successful monitor-market output is 2024-01-04."
        in recommendation_state["context_notes"]
    )


def test_internal_api_market_state_falls_back_to_raw_statuses_for_workflow_input_summary(
    tmp_path: Path,
) -> None:
    query_service = InternalApiQueryService(project_root=tmp_path)
    _write_market_state_snapshots(tmp_path)
    write_json_file(
        tmp_path / "data" / "processed" / "monitor_market" / "2024-01-05" / "market_monitor.json",
        {"as_of_date": "2024-01-05"},
    )
    write_json_file(
        tmp_path / "data" / "processed" / "monitor_market" / "2024-01-05" / "daily_summary.json",
        {
            "metadata": {
                "research_input_statuses": {
                    "benchmark": {"status": "ok"},
                    "earnings_contexts": {
                        "status": "degraded",
                        "issue_code": "entitlement_limited",
                    },
                }
            }
        },
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert (
        recommendation_state["workflow_input_summaries"]["daily_summary"][
            "degraded_count"
        ]
        == 1
    )
    assert (
        recommendation_state["workflow_input_summaries"]["daily_summary"][
            "issue_codes"
        ]
        == ["entitlement_limited"]
    )
    assert recommendation_state["workflow_input_overview"]["problematic_workflow_count"] == 1


def test_internal_api_market_state_loads_workflow_summaries_from_archived_live_market_runtime(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "runtime-root"
    query_service = InternalApiQueryService(project_root=project_root)
    _write_market_state_snapshots(project_root)
    write_json_file(
        tmp_path / "archive" / "live_market" / "2024-01-05" / "market_monitor.json",
        {"as_of_date": "2024-01-05"},
    )
    write_json_file(
        tmp_path / "archive" / "live_market" / "2024-01-05" / "portfolio_review.json",
        {
            "metadata": {
                "workflow_input_summary": {
                    "total_count": 4,
                    "ok_count": 3,
                    "degraded_count": 1,
                    "unavailable_count": 0,
                    "failed_count": 0,
                    "issue_count": 1,
                    "healthy": False,
                    "problematic_inputs": ["earnings_contexts"],
                    "issue_codes": ["entitlement_limited"],
                    "issues": [
                        {
                            "input_name": "earnings_contexts",
                            "status": "degraded",
                            "role_name": "earnings_calendar",
                            "provider": "polygon",
                            "issue_code": "entitlement_limited",
                            "message": "Earnings context degraded.",
                        }
                    ],
                }
            }
        },
    )
    write_json_file(
        tmp_path / "archive" / "live_market" / "2024-01-05" / "portfolio_review_intraday.json",
        {
            "metadata": {
                "review_input_statuses": {
                    "benchmark_intraday_metrics": {"status": "ok"},
                    "earnings_contexts": {
                        "status": "degraded",
                        "issue_code": "entitlement_limited",
                    },
                }
            }
        },
    )

    payload = query_service.market_state()
    recommendation_state = payload["data"]["recommendation_state"]

    assert recommendation_state["workflow_input_overview"]["workflow_count"] == 2
    assert recommendation_state["workflow_input_overview"]["problematic_workflow_count"] == 2
    assert recommendation_state["workflow_input_overview"]["highest_severity"] == "degraded"
    assert (
        recommendation_state["workflow_input_summaries"]["portfolio_review"][
            "problematic_inputs"
        ]
        == ["earnings_contexts"]
    )
    assert (
        recommendation_state["workflow_input_summaries"]["intraday_review"][
            "issue_codes"
        ]
        == ["entitlement_limited"]
    )


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
            raw_message_count=12,
            accepted_message_count=9,
            last_raw_message_at_utc="2024-01-05T15:05:30+00:00",
            last_accepted_message_at_utc="2024-01-05T15:05:00+00:00",
            last_raw_message_type="b",
            last_accepted_message_type="quote_bar_batch_refresh",
            stream_provider="alpaca",
            historical_provider="polygon",
            reference_provider="polygon",
            earnings_provider="polygon",
            execution_broker="alpaca",
            broker_update_stream_provider=None,
            provider_roles={
                "stream_market_data": {
                    "provider": "alpaca",
                    "configured": True,
                    "available": True,
                },
                "historical_bars": {
                    "provider": "polygon",
                    "configured": True,
                    "available": True,
                    "degraded": True,
                },
            },
            degraded_provider_roles=("historical_bars",),
            unavailable_provider_roles=(),
            subscription_status="acknowledged",
            last_subscription_message="alpaca websocket subscription acknowledged: bars=3",
            last_successful_flush_at_utc="2024-01-05T15:05:00+00:00",
        ),
        updated_at_utc="2024-01-05T15:06:00+00:00",
    )

    persisted_payload = query_service.health()
    assert persisted_payload["available"] is True
    assert persisted_payload["data"]["updated_at_utc"] == "2024-01-05T15:06:00+00:00"
    assert persisted_payload["data"]["status"]["connected"] is True
    assert persisted_payload["data"]["status"]["cycle_count"] == 3
    assert persisted_payload["data"]["status"]["raw_message_count"] == 12
    assert persisted_payload["data"]["status"]["accepted_message_count"] == 9
    assert persisted_payload["data"]["status"]["last_raw_message_type"] == "b"
    assert (
        persisted_payload["data"]["status"]["last_accepted_message_type"]
        == "quote_bar_batch_refresh"
    )
    assert persisted_payload["data"]["status"]["stream_provider"] == "alpaca"
    assert persisted_payload["data"]["status"]["historical_provider"] == "polygon"
    assert persisted_payload["data"]["status"]["reference_provider"] == "polygon"
    assert persisted_payload["data"]["status"]["earnings_provider"] == "polygon"
    assert persisted_payload["data"]["status"]["execution_broker"] == "alpaca"
    assert persisted_payload["data"]["status"]["broker_update_stream_provider"] is None
    assert (
        persisted_payload["data"]["status"]["provider_roles"]["stream_market_data"][
            "provider"
        ]
        == "alpaca"
    )
    assert persisted_payload["data"]["status"]["degraded_provider_roles"] == [
        "historical_bars"
    ]
    assert persisted_payload["data"]["status"]["unavailable_provider_roles"] == []
    assert persisted_payload["data"]["status"]["subscription_status"] == "acknowledged"


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
    assert payload["data"]["summary"]["reserved_slot_count"] is None
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
    write_json_file(
        tmp_path / "data" / "processed" / "portfolio_review" / "2024-01-05" / "portfolio_review.json",
        {
            "metadata": {
                "review_input_statuses": {
                    "benchmark": {"status": "ok"},
                    "position_daily_symbol_frames": {
                        "status": "degraded",
                        "issue_code": "partial_symbol_frames",
                    },
                }
            }
        },
    )
    write_json_file(
        tmp_path
        / "data"
        / "processed"
        / "portfolio_review_intraday"
        / "2024-01-05"
        / "portfolio_review_intraday.json",
        {
            "metadata": {
                "workflow_input_summary": {
                    "total_count": 2,
                    "ok_count": 1,
                    "degraded_count": 1,
                    "unavailable_count": 0,
                    "failed_count": 0,
                    "issue_count": 1,
                    "healthy": False,
                    "problematic_inputs": ["position_daily_symbol_frames"],
                    "issue_codes": ["partial_symbol_frames"],
                    "issues": [
                        {
                            "input_name": "position_daily_symbol_frames",
                            "status": "degraded",
                            "role_name": "historical_bars",
                            "provider": "alpaca",
                            "issue_code": "partial_symbol_frames",
                            "message": "One or more held symbols could not be loaded.",
                        }
                    ],
                }
            }
        },
    )
    query_service = InternalApiQueryService(project_root=tmp_path)

    payload = query_service.portfolio()

    assert payload["available"] is True
    assert payload["data"]["watch_closely"] == ["AAPL"]
    assert payload["data"]["exit_candidates"] == []
    assert payload["data"]["workflow_input_overview"]["workflow_count"] == 2
    assert payload["data"]["workflow_input_overview"]["problematic_workflow_count"] == 2
    assert payload["data"]["workflow_input_summaries"]["portfolio_review"][
        "degraded_count"
    ] == 1
    assert payload["data"]["workflow_input_summaries"]["intraday_review"][
        "issue_codes"
    ] == ["partial_symbol_frames"]


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
                candidate_disposition="actionable",
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
                candidate_disposition="actionable",
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
