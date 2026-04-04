from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

import bot.dashboard.operator_dashboard as operator_dashboard_module
from bot.api.control_api import OperatorControlService
from bot.api.internal_api import InternalApiQueryService
from bot.dashboard.operator_dashboard import (
    OperatorDashboardHttpServer,
    dashboard_asset_for_path,
    operator_dashboard_response_for_request,
    operator_dashboard_response_for_path,
)
from bot.main import build_parser


def _dashboard_app_source(refresh_interval_seconds: int = 5) -> str:
    asset = dashboard_asset_for_path(
        path="/dashboard/app.js",
        refresh_interval_seconds=refresh_interval_seconds,
    )
    assert asset is not None
    return asset.body.decode("utf-8")


def _run_dashboard_app_runtime_test(
    tmp_path: Path,
    *,
    test_body: str,
    refresh_interval_seconds: int = 5,
) -> None:
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node is required for dashboard runtime tests")
    source = _dashboard_app_source(refresh_interval_seconds=refresh_interval_seconds)
    bootstrap = textwrap.dedent(
        """\
        loadDashboard();
        window.setInterval(() => {
          loadDashboard();
        }, REFRESH_INTERVAL_SECONDS * 1000);
        """
    )
    assert bootstrap in source
    source = source.replace(bootstrap, "", 1)
    app_path = tmp_path / "dashboard_app.js"
    app_path.write_text(source, encoding="utf-8")
    script = textwrap.dedent(
        """\
        const fs = require("fs");
        const vm = require("vm");

        let source = fs.readFileSync(__APP_PATH__, "utf8");
        source += `
        ;globalThis.__dashboardTestExports = {
          state,
          els,
          loadDashboard,
          executeControlAction,
          handleControlAction,
          renderPendingOrders,
          renderCurrentDashboard,
        };
        `;

        function makeElement(id) {
          return {
            id,
            textContent: "",
            innerHTML: "",
            hidden: false,
            className: "",
            disabled: false,
            dataset: {},
            value: "",
            addEventListener() {},
            closest() { return null; },
            classList: {
              add() {},
              remove() {},
            },
          };
        }

        const elements = new Map();
        const document = {
          getElementById(id) {
            if (!elements.has(id)) {
              elements.set(id, makeElement(id));
            }
            return elements.get(id);
          },
          addEventListener() {},
        };

        function jsonResponse(payload, status = 200) {
          return {
            ok: status >= 200 && status < 300,
            status,
            text: async () => JSON.stringify(payload),
          };
        }

        const fetchCalls = [];
        let fetchImpl = async (path, options = {}) => jsonResponse({
          available: false,
          not_available_reason: "test-default",
          warnings: [],
          data: {},
        });

        const context = {
          window: {
            __BOT_DASHBOARD_CONFIG__: { refreshIntervalSeconds: __REFRESH_INTERVAL__ },
            setInterval() {},
            confirm() { return true; },
          },
          document,
          fetch: async (path, options = {}) => {
            fetchCalls.push({
              path,
              method: options.method || "GET",
              body: options.body || null,
            });
            return fetchImpl(path, options);
          },
          console,
          setTimeout,
          clearTimeout,
        };
        context.globalThis = context;

        vm.createContext(context);
        vm.runInContext(source, context, { filename: "dashboard_app.js" });

        const {
          state,
          els,
          loadDashboard,
          executeControlAction,
          handleControlAction,
          renderPendingOrders,
          renderCurrentDashboard,
        } = context.__dashboardTestExports;

        (async () => {
        __TEST_BODY__
        })().catch((error) => {
          console.error(error);
          process.exit(1);
        });
        """
    )
    script = script.replace("__APP_PATH__", json.dumps(str(app_path)))
    script = script.replace("__REFRESH_INTERVAL__", str(refresh_interval_seconds))
    script = script.replace("__TEST_BODY__", textwrap.dedent(test_body))
    result = subprocess.run(
        [node_path, "--input-type=commonjs", "-e", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


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
    assert "control-feedback" in html
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
    assert 'controlSafety: "/control/safety"' in javascript
    assert 'pauseExecution: "/control/execution/pause"' in javascript
    assert 'resumeExecution: "/control/execution/resume"' in javascript
    assert 'setExecutionMode: "/control/execution/mode"' in javascript
    assert 'setLiveConfirmation: "/control/execution/live-confirmation"' in javascript
    assert 'setBrokerTrading: "/control/execution/broker-trading"' in javascript
    assert 'submitOrder: "/control/orders/submit"' in javascript
    assert 'cancelOrder: "/control/orders/cancel"' in javascript
    assert 'replaceOrder: "/control/orders/replace"' in javascript
    assert 'forceBrokerSync: "/control/broker/sync"' in javascript
    assert "setPanelUnavailable" in javascript
    assert "snapshot.top_rejected_reasons_summary" in javascript
    assert "data.top_rejected_reasons_summary" not in javascript
    assert "function workflowOverviewStatusLabel(status)" in javascript
    assert "function renderWorkflowInputOverviewBlock(" in javascript
    assert "recommendationState.workflow_input_overview" in javascript
    assert "data.workflow_input_overview" in javascript
    assert "Recommendation input health" in javascript


def test_dashboard_app_asset_wires_control_client_and_refresh_after_actions() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert "async function postControlCommand(path, payload)" in javascript
    assert 'method: "POST"' in javascript
    assert '"Content-Type": "application/json"' in javascript
    assert "await loadDashboard({ manual: true });" in javascript
    assert "renderControlFeedback();" in javascript
    assert "state.pendingRefresh = true;" in javascript


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


def test_dashboard_app_asset_handles_transitional_and_rejected_control_states() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert 'payload.status === "needs_confirmation"' in javascript
    assert "payload.data.transitional_state" in javascript
    assert "payload.data.reservation_active === true" in javascript
    assert 'payload.error_code === "control_state_unavailable"' in javascript
    assert 'order.broker_status === "pending_cancel"' in javascript


def test_dashboard_app_asset_handles_missing_actionable_now_and_portfolio_counts() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert 'typeof item.actionable_now !== "boolean"' in javascript
    assert "Actionability unavailable." in javascript
    assert "function formatOptionalCount(value)" in javascript
    assert 'formatOptionalCount(portfolioSummary.hold_count)' in javascript
    assert 'formatOptionalCount(portfolioSummary.position_count)' in javascript


def test_dashboard_app_asset_disables_submit_when_execution_is_paused_or_unreadable() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert "control.executionEnabled !== true" in javascript
    assert "!control.readable" in javascript
    assert "Execution submissions are paused. Submit actions are disabled." in javascript


def test_dashboard_app_asset_renders_replace_editor_form_and_force_sync_control() -> None:
    asset = dashboard_asset_for_path(path="/dashboard/app.js", refresh_interval_seconds=5)

    javascript = asset.body.decode("utf-8")

    assert "function renderReplaceEditor(order)" in javascript
    assert "replace-limit-price" in javascript
    assert "replace-stop-price" in javascript
    assert "Force broker sync" in javascript
    assert "Switch to live" in javascript
    assert "Switch to paper" in javascript
    assert "Enable live trading" in javascript
    assert "Require live confirm" in javascript
    assert 'action: "submit-replace-order"' in javascript
    assert "Only changed fields will be sent to the backend replacement request." in javascript


def test_dashboard_runtime_renders_workflow_input_overview_blocks(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        state.currentResults = {
          health: {
            ok: true,
            payload: { available: false, warnings: [], data: {} },
          },
          marketState: {
            ok: true,
            payload: {
              available: true,
              warnings: [],
              data: {
                snapshot: {
                  as_of_timestamp: "2024-01-05T15:00:00+00:00",
                  source_workflows: ["monitor-market", "review-portfolio"],
                },
                current_alertable_states: [],
                recent_transitions: [],
                recommendation_state: {
                  workflow_input_overview: {
                    workflow_count: 2,
                    healthy_workflow_count: 1,
                    degraded_workflow_count: 0,
                    unavailable_workflow_count: 1,
                    failed_workflow_count: 0,
                    problematic_workflow_count: 1,
                    highest_severity: "unavailable",
                    problematic_workflows: [
                      {
                        workflow_name: "daily_summary",
                        status: "unavailable",
                        issue_count: 1,
                        issue_codes: ["unsupported_capability"],
                        problematic_inputs: ["volatility_context"],
                      },
                    ],
                  },
                },
              },
            },
          },
          marketTransitions: {
            ok: true,
            payload: { available: false, warnings: [], data: {} },
          },
          portfolio: {
            ok: true,
            payload: {
              available: true,
              warnings: [],
              data: {
                portfolio_review_summary: {
                  position_count: 3,
                  hold_count: 1,
                  watch_closely_count: 1,
                  exit_candidate_count: 1,
                  raise_stop_count: 0,
                },
                intraday_review_summary: {
                  watch_closely_count: 1,
                },
                current_action_states: [
                  { symbol: "AAPL", action: "HOLD" },
                  { symbol: "MSFT", action: "WATCH CLOSELY" },
                ],
                watch_closely: ["MSFT"],
                exit_candidates: ["NVDA"],
                raise_stop: [],
                workflow_input_overview: {
                  workflow_count: 2,
                  healthy_workflow_count: 0,
                  degraded_workflow_count: 2,
                  unavailable_workflow_count: 0,
                  failed_workflow_count: 0,
                  problematic_workflow_count: 2,
                  highest_severity: "degraded",
                  problematic_workflows: [
                    {
                      workflow_name: "portfolio_review",
                      status: "degraded",
                      issue_count: 1,
                      issue_codes: ["partial_symbol_frames"],
                      problematic_inputs: ["position_daily_symbol_frames"],
                    },
                    {
                      workflow_name: "intraday_review",
                      status: "degraded",
                      issue_count: 1,
                      issue_codes: ["partial_symbol_frames"],
                      problematic_inputs: ["position_daily_symbol_frames"],
                    },
                  ],
                },
              },
            },
          },
          analytics: {
            ok: true,
            payload: { available: false, warnings: [], data: {} },
          },
          controlSafety: {
            ok: true,
            payload: { available: false, warnings: [], data: {} },
          },
          pendingOrders: {
            ok: true,
            payload: { available: false, warnings: [], data: {} },
          },
        };

        renderCurrentDashboard();

        if (!els.marketStateBody.innerHTML.includes("Workflow input health")) {
          throw new Error(`Expected workflow input health block in market-state panel: ${els.marketStateBody.innerHTML}`);
        }
        if (!els.marketStateBody.innerHTML.includes("daily_summary | Unavailable")) {
          throw new Error(`Expected daily_summary details in market-state panel: ${els.marketStateBody.innerHTML}`);
        }
        if (!els.marketStateBody.innerHTML.includes("Unavailable highest severity")) {
          throw new Error(`Expected unavailable severity wording in market-state panel: ${els.marketStateBody.innerHTML}`);
        }
        if (!els.portfolioStateBody.innerHTML.includes("Review input health")) {
          throw new Error(`Expected review input health block in portfolio panel: ${els.portfolioStateBody.innerHTML}`);
        }
        if (!els.portfolioStateBody.innerHTML.includes("portfolio_review | Degraded")) {
          throw new Error(`Expected portfolio_review details in portfolio panel: ${els.portfolioStateBody.innerHTML}`);
        }
        if (!els.portfolioStateBody.innerHTML.includes("intraday_review | Degraded")) {
          throw new Error(`Expected intraday_review details in portfolio panel: ${els.portfolioStateBody.innerHTML}`);
        }
        if (!els.candidateQueueBody.innerHTML.includes("Recommendation input health")) {
          throw new Error(`Expected recommendation input health block in candidate queue panel: ${els.candidateQueueBody.innerHTML}`);
        }
        if (!els.candidateQueueBody.innerHTML.includes("daily_summary | Unavailable")) {
          throw new Error(`Expected daily_summary details in candidate queue panel: ${els.candidateQueueBody.innerHTML}`);
        }
        """,
    )


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
        control_service=OperatorControlService(project_root=tmp_path),
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


def test_operator_dashboard_response_for_control_get_uses_control_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, str] = {}

    def fake_operator_control_response_for_request(
        *,
        control_service: OperatorControlService,
        method: str,
        path: str,
        body: bytes | None,
    ) -> tuple[HTTPStatus, dict[str, object]]:
        called["method"] = method
        called["path"] = path
        assert control_service.project_root == tmp_path
        assert body is None
        return HTTPStatus.OK, {"endpoint": "control_safety", "available": True}

    monkeypatch.setattr(
        operator_dashboard_module,
        "operator_control_response_for_request",
        fake_operator_control_response_for_request,
    )

    response = operator_dashboard_response_for_request(
        query_service=InternalApiQueryService(project_root=tmp_path),
        control_service=OperatorControlService(project_root=tmp_path),
        method="GET",
        path="/control/safety",
        body=None,
        refresh_interval_seconds=5,
    )

    assert called == {"method": "GET", "path": "/control/safety"}
    assert response.status == HTTPStatus.OK
    assert json.loads(response.body.decode("utf-8")) == {
        "available": True,
        "endpoint": "control_safety",
    }


def test_operator_dashboard_response_for_control_post_uses_control_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_operator_control_response_for_request(
        *,
        control_service: OperatorControlService,
        method: str,
        path: str,
        body: bytes | None,
    ) -> tuple[HTTPStatus, dict[str, object]]:
        called["method"] = method
        called["path"] = path
        called["body"] = body.decode("utf-8") if body is not None else None
        assert control_service.project_root == tmp_path
        return HTTPStatus.OK, {"command_name": "submit_pending_order", "ok": True}

    monkeypatch.setattr(
        operator_dashboard_module,
        "operator_control_response_for_request",
        fake_operator_control_response_for_request,
    )

    response = operator_dashboard_response_for_request(
        query_service=InternalApiQueryService(project_root=tmp_path),
        control_service=OperatorControlService(project_root=tmp_path),
        method="POST",
        path="/control/orders/submit",
        body=b'{"broker":"alpaca","order_id":"order_aapl_1"}',
        refresh_interval_seconds=5,
    )

    assert called == {
        "method": "POST",
        "path": "/control/orders/submit",
        "body": '{"broker":"alpaca","order_id":"order_aapl_1"}',
    }
    assert response.status == HTTPStatus.OK
    assert json.loads(response.body.decode("utf-8")) == {
        "command_name": "submit_pending_order",
        "ok": True,
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


def test_dashboard_runtime_queues_exactly_one_post_action_refresh_during_inflight_load(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        function endpointPayload(path) {
          if (path === "/control/safety") {
            return {
              available: true,
              warnings: [],
              data: {
                control_state_readable: true,
                safety_state: {
                  execution_submission_enabled: true,
                  execution_mode: "paper",
                },
              },
            };
          }
          if (path === "/pending-orders") {
            return {
              available: true,
              warnings: [],
              data: {
                active_order_count: 0,
                active_orders: [],
                summary: {
                  capacity_reserving_buy_order_count: 0,
                  reserved_notional: 0,
                  pending_fill_notional: 0,
                  reserved_slot_count: 0,
                  position_context_available: true,
                  available_slots: 4,
                },
              },
            };
          }
          return {
            available: false,
            not_available_reason: "not-ready",
            warnings: [],
            data: {},
          };
        }

        let releaseFirstBatch = null;
        const firstBatchGate = new Promise((resolve) => {
          releaseFirstBatch = resolve;
        });
        let remainingBlockedGets = 7;

        fetchImpl = async (path, options = {}) => {
          const method = options.method || "GET";
          if (method === "POST") {
            return jsonResponse({
              command_name: "submit_pending_order",
              ok: true,
              status: "completed",
              message: "Submitted.",
              warnings: [],
              data: {},
            });
          }
          if (remainingBlockedGets > 0) {
            remainingBlockedGets -= 1;
            await firstBatchGate;
          }
          return jsonResponse(endpointPayload(path));
        };

        const firstLoadPromise = loadDashboard();
        await Promise.resolve();
        const actionPromise = executeControlAction("/control/orders/submit", {
          broker: "alpaca",
          order_id: "order_aapl_1",
        });
        await Promise.resolve();

        if (state.loading !== true) {
          throw new Error("Expected the initial dashboard load to still be in flight.");
        }

        releaseFirstBatch();
        await Promise.all([firstLoadPromise, actionPromise]);

        const getCalls = fetchCalls.filter((call) => call.method === "GET");
        const postCalls = fetchCalls.filter((call) => call.method === "POST");
        const pendingOrderLoads = getCalls.filter((call) => call.path === "/pending-orders");

        if (postCalls.length !== 1) {
          throw new Error(`Expected one control POST, saw ${postCalls.length}.`);
        }
        if (getCalls.length !== 14) {
          throw new Error(`Expected exactly two dashboard fetch rounds, saw ${getCalls.length} GETs.`);
        }
        if (pendingOrderLoads.length !== 2) {
          throw new Error(`Expected two pending-order refreshes, saw ${pendingOrderLoads.length}.`);
        }
        if (state.loading !== false) {
          throw new Error("Dashboard loading flag did not clear.");
        }
        if (state.actionInFlight !== false) {
          throw new Error("Action-in-flight flag did not clear.");
        }
        """,
    )


def test_dashboard_runtime_unreadable_control_state_disables_cancel_and_replace(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        const pendingOrdersResult = {
          ok: true,
          payload: {
            warnings: [],
            data: {
              active_order_count: 1,
              active_orders: [
                {
                  order_id: "order_aapl_1",
                  symbol: "AAPL",
                  side: "BUY",
                  status: "pending",
                  broker_status: "accepted",
                  broker_order_id: "alpaca-ord-1",
                  broker_name: "alpaca",
                  reserved_notional: 1000,
                },
              ],
              summary: {
                capacity_reserving_buy_order_count: 1,
                reserved_notional: 1000,
                pending_fill_notional: 0,
                reserved_slot_count: 1,
                position_context_available: true,
                available_slots: 3,
              },
            },
          },
        };
        const unreadableControl = {
          ok: true,
          payload: {
            available: true,
            warnings: ["state unreadable"],
            data: {
              control_state_readable: false,
              safety_state: {
                execution_submission_enabled: true,
                execution_mode: "paper",
              },
            },
          },
        };

        renderPendingOrders(pendingOrdersResult, unreadableControl);
        const html = els.pendingOrdersBody.innerHTML;

        if (!/data-control-action="cancel-order"[^>]*disabled/.test(html)) {
          throw new Error("Cancel action should be disabled when control state is unreadable.");
        }
        if (!/data-control-action="open-replace-editor"[^>]*disabled/.test(html)) {
          throw new Error("Replace action should be disabled when control state is unreadable.");
        }
        """,
    )


def test_dashboard_runtime_closes_replace_editor_when_order_enters_pending_cancel(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        const controlResult = {
          ok: true,
          payload: {
            available: true,
            warnings: [],
            data: {
              control_state_readable: true,
              safety_state: {
                execution_submission_enabled: true,
                execution_mode: "paper",
              },
            },
          },
        };

        function pendingOrdersResult(brokerStatus) {
          return {
            ok: true,
            payload: {
              warnings: [],
              data: {
                active_order_count: 1,
                active_orders: [
                  {
                    order_id: "order_aapl_1",
                    symbol: "AAPL",
                    side: "BUY",
                    status: "pending",
                    broker_status: brokerStatus,
                    broker_order_id: "alpaca-ord-1",
                    broker_name: "alpaca",
                    requested_quantity: 10,
                    requested_price: 100,
                    stop_price: 95,
                    reserved_notional: 1000,
                  },
                ],
                summary: {
                  capacity_reserving_buy_order_count: 1,
                  reserved_notional: 1000,
                  pending_fill_notional: 0,
                  reserved_slot_count: 1,
                  position_context_available: true,
                  available_slots: 3,
                },
              },
            },
          };
        }

        state.replaceEditorOrderId = "order_aapl_1";
        renderPendingOrders(pendingOrdersResult("accepted"), controlResult);
        if (!els.pendingOrdersBody.innerHTML.includes("replace-quantity")) {
          throw new Error("Expected replace editor to be visible before pending_cancel.");
        }

        renderPendingOrders(pendingOrdersResult("pending_cancel"), controlResult);
        if (state.replaceEditorOrderId !== null) {
          throw new Error("Replace editor should close when broker_status becomes pending_cancel.");
        }
        if (els.pendingOrdersBody.innerHTML.includes("replace-quantity")) {
          throw new Error("Replace editor should not remain visible after pending_cancel.");
        }
        """,
    )


def test_dashboard_runtime_replace_without_changes_does_not_send_unchanged_fields(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        function endpointPayload(path) {
          if (path === "/control/safety") {
            return {
              available: true,
              warnings: [],
              data: {
                control_state_readable: true,
                safety_state: {
                  execution_submission_enabled: true,
                  execution_mode: "paper",
                },
              },
            };
          }
          if (path === "/pending-orders") {
            return {
              available: true,
              warnings: [],
              data: {
                active_order_count: 1,
                active_orders: [
                  {
                    order_id: "order_aapl_1",
                    symbol: "AAPL",
                    side: "BUY",
                    status: "pending",
                    broker_status: "accepted",
                    broker_order_id: "alpaca-ord-1",
                    broker_name: "alpaca",
                    requested_quantity: 10,
                    requested_price: 100,
                    stop_price: 95,
                    reserved_notional: 1000,
                  },
                ],
                summary: {
                  capacity_reserving_buy_order_count: 1,
                  reserved_notional: 1000,
                  pending_fill_notional: 0,
                  reserved_slot_count: 1,
                  position_context_available: true,
                  available_slots: 3,
                },
              },
            };
          }
          return {
            available: false,
            not_available_reason: "not-ready",
            warnings: [],
            data: {},
          };
        }

        state.currentResults = {
          pendingOrders: {
            ok: true,
            payload: endpointPayload("/pending-orders"),
          },
        };

        document.getElementById("replace-quantity").value = "10";
        document.getElementById("replace-limit-price").value = "100";
        document.getElementById("replace-stop-price").value = "95";

        fetchImpl = async (path, options = {}) => {
          const method = options.method || "GET";
          if (method === "POST") {
            const payload = JSON.parse(options.body || "{}");
            if ("quantity" in payload || "limit_price" in payload || "stop_price" in payload) {
              throw new Error(`Expected unchanged replacement fields to be omitted, saw ${options.body}.`);
            }
            return jsonResponse({
              command_name: "replace_pending_order",
              ok: false,
              status: "rejected",
              message: "No changes were provided.",
              warnings: [],
              data: {},
            }, 409);
          }
          return jsonResponse(endpointPayload(path));
        };

        await handleControlAction("submit-replace-order", "order_aapl_1");

        const postCalls = fetchCalls.filter((call) => call.method === "POST");
        if (postCalls.length !== 0) {
          throw new Error(`Expected no replace request for unchanged values, saw ${postCalls.length}.`);
        }
        if (!state.lastControlResult || !state.lastControlResult.payload) {
          throw new Error("Expected a local control result after unchanged replace submission.");
        }
        if (state.lastControlResult.payload.status !== "rejected") {
          throw new Error(`Expected a local rejected result, saw ${state.lastControlResult.payload.status}.`);
        }
        if (
          state.lastControlResult.payload.message !==
          "Change at least one replacement field before submitting the order amendment."
        ) {
          throw new Error(`Unexpected local rejection message: ${state.lastControlResult.payload.message}`);
        }
        """,
    )


def test_dashboard_runtime_live_submit_sends_confirmation_flag_when_required(
    tmp_path: Path,
) -> None:
    _run_dashboard_app_runtime_test(
        tmp_path,
        test_body="""
        function endpointPayload(path) {
          if (path === "/control/safety") {
            return {
              available: true,
              warnings: [],
              data: {
                control_state_readable: true,
                safety_state: {
                  execution_submission_enabled: true,
                  execution_mode: "live",
                  live_actions_require_confirmation: true,
                  broker_trading_enabled: true,
                },
                policy_summary: {
                  execution_mode: "live",
                  execution_submission_enabled: true,
                  live_actions_require_confirmation: true,
                  broker_trading_enabled: true,
                  is_live_mode: true,
                  live_broker_mutations_allowed: true,
                },
              },
            };
          }
          if (path === "/pending-orders") {
            return {
              available: true,
              warnings: [],
              data: {
                active_order_count: 1,
                active_orders: [
                  {
                    order_id: "order_aapl_1",
                    symbol: "AAPL",
                    side: "BUY",
                    status: "pending",
                    broker_status: null,
                    broker_order_id: null,
                    broker_name: "alpaca",
                    requested_quantity: 10,
                    requested_price: 100,
                    reserved_notional: 1000,
                  },
                ],
                summary: {
                  capacity_reserving_buy_order_count: 1,
                  reserved_notional: 1000,
                  pending_fill_notional: 0,
                  reserved_slot_count: 1,
                  position_context_available: true,
                  available_slots: 3,
                },
              },
            };
          }
          return {
            available: false,
            not_available_reason: "not-ready",
            warnings: [],
            data: {},
          };
        }

        state.currentResults = {
          health: {
            ok: true,
            payload: endpointPayload("/health"),
          },
          marketState: {
            ok: true,
            payload: endpointPayload("/market-state"),
          },
          marketTransitions: {
            ok: true,
            payload: endpointPayload("/market-state/transitions"),
          },
          portfolio: {
            ok: true,
            payload: endpointPayload("/portfolio"),
          },
          analytics: {
            ok: true,
            payload: endpointPayload("/analytics/trade-review"),
          },
          controlSafety: {
            ok: true,
            payload: endpointPayload("/control/safety"),
          },
          pendingOrders: {
            ok: true,
            payload: endpointPayload("/pending-orders"),
          },
        };

        let confirmCalls = 0;
        context.window.confirm = () => {
          confirmCalls += 1;
          return true;
        };

        fetchImpl = async (path, options = {}) => {
          const method = options.method || "GET";
          if (method === "POST") {
            return jsonResponse({
              command_name: "submit_pending_order",
              ok: true,
              status: "completed",
              message: "Submitted.",
              warnings: [],
              data: {},
            });
          }
          return jsonResponse(endpointPayload(path));
        };

        await handleControlAction("submit-order", "order_aapl_1");

        const postCalls = fetchCalls.filter((call) => call.method === "POST");
        if (postCalls.length !== 1) {
          throw new Error(`Expected one live submit request, saw ${postCalls.length}.`);
        }
        const payload = JSON.parse(postCalls[0].body);
        if (payload.confirm_live_action !== true) {
          throw new Error(`Expected confirm_live_action=true, saw ${postCalls[0].body}`);
        }
        if (confirmCalls !== 1) {
          throw new Error(`Expected one confirmation prompt, saw ${confirmCalls}.`);
        }
        """,
    )
