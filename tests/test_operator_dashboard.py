from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path

import pytest

import bot.dashboard.operator_dashboard as operator_dashboard_module
from bot.api.internal_api import InternalApiQueryService
from bot.dashboard.operator_dashboard import (
    OperatorDashboardHttpServer,
    dashboard_asset_for_path,
    operator_dashboard_response_for_path,
)
from bot.main import build_parser


def test_dashboard_asset_for_path_serves_html_shell() -> None:
    asset = dashboard_asset_for_path(path="/dashboard", refresh_interval_seconds=7)

    assert asset is not None
    assert asset.status == HTTPStatus.OK
    assert asset.content_type == "text/html; charset=utf-8"
    html = asset.body.decode("utf-8")
    assert "Operator Dashboard" in html
    assert '"refreshIntervalSeconds": 7' in html
    assert "/dashboard/styles.css" in html
    assert "/dashboard/app.js" in html
    assert "service-health-body" in html
    assert "market-state-body" in html
    assert "portfolio-state-body" in html
    assert "pending-orders-body" in html
    assert "candidate-queue-body" in html
    assert "analytics-body" in html


def test_dashboard_app_asset_references_internal_api_endpoints() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    assert asset is not None
    assert asset.status == HTTPStatus.OK
    assert asset.content_type == "application/javascript; charset=utf-8"
    javascript = asset.body.decode("utf-8")
    assert 'health: "/health"' in javascript
    assert 'marketState: "/market-state"' in javascript
    assert 'marketTransitions: "/market-state/transitions"' in javascript
    assert 'pendingOrders: "/pending-orders"' in javascript
    assert 'portfolio: "/portfolio"' in javascript
    assert 'analytics: "/analytics/trade-review"' in javascript
    assert "setPanelUnavailable" in javascript
    assert "snapshot.top_rejected_reasons_summary" in javascript
    assert "data.top_rejected_reasons_summary" not in javascript


def test_dashboard_app_asset_formats_ratio_percent_correctly() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert "function formatRatioPercent(value)" in javascript
    assert '(value * 100).toFixed(1)' in javascript
    assert "12.3%" not in javascript


def test_dashboard_app_asset_marks_reserved_slots_unknown_when_context_missing() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert 'summary.position_context_available ? formatNumber(summary.reserved_slot_count) : "Unknown"' in javascript
    assert "Capacity figures are partial." in javascript


def test_dashboard_app_asset_handles_missing_actionable_now_and_portfolio_counts() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert 'typeof item.actionable_now !== "boolean"' in javascript
    assert "Actionability unavailable." in javascript
    assert "function formatOptionalCount(value)" in javascript
    assert 'formatOptionalCount(portfolioSummary.hold_count)' in javascript
    assert 'formatOptionalCount(portfolioSummary.position_count)' in javascript


def test_dashboard_asset_for_path_serves_stylesheet() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/styles.css", refresh_interval_seconds=5)

    assert asset is not None
    assert asset.status == HTTPStatus.OK
    assert asset.content_type == "text/css; charset=utf-8"
    stylesheet = asset.body.decode("utf-8")
    assert "--accent:" in stylesheet
    assert ".panel-grid" in stylesheet
    assert ".metric-grid" in stylesheet
    assert ".note.warn" in stylesheet


def test_dashboard_asset_for_path_returns_none_for_unknown_path() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/unknown", refresh_interval_seconds=5)

    assert asset is None


def test_operator_dashboard_response_for_non_asset_path_uses_internal_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, str] = {}

    def fake_internal_api_response_for_path(
        *,
        query_service: InternalApiQueryService,
        path: str,
    ) -> tuple[HTTPStatus, dict[str, object]]:
        called["path"] = path
        assert query_service.project_root == tmp_path
        return HTTPStatus.OK, {"endpoint": "health", "available": False}

    monkeypatch.setattr(
        operator_dashboard_module,
        "internal_api_response_for_path",
        fake_internal_api_response_for_path,
    )

    response = operator_dashboard_response_for_path(
        query_service=InternalApiQueryService(project_root=tmp_path),
        path="/health",
        refresh_interval_seconds=5,
    )

    assert called["path"] == "/health"
    assert response.status == HTTPStatus.OK
    assert response.content_type == "application/json; charset=utf-8"
    assert json.loads(response.body.decode("utf-8")) == {
        "available": False,
        "endpoint": "health",
    }


def test_operator_dashboard_http_server_rejects_non_positive_refresh_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="refresh_interval_seconds must be greater than zero."):
        OperatorDashboardHttpServer(
            ("127.0.0.1", 8780),
            query_service=InternalApiQueryService(project_root=tmp_path),
            refresh_interval_seconds=0,
        )


def test_build_parser_accepts_serve_operator_dashboard_command() -> None:
    args = build_parser().parse_args(
        [
            "serve-operator-dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            "8788",
            "--refresh-seconds",
            "9",
        ]
    )

    assert args.command == "serve-operator-dashboard"
    assert args.host == "127.0.0.1"
    assert args.port == 8788
    assert args.refresh_seconds == 9
