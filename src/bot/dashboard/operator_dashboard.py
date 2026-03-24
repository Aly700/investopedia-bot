"""Static operator dashboard served alongside the internal and control APIs."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlparse

from bot.api.control_api import (
    OperatorControlService,
    operator_control_response_for_request,
)
from bot.api.internal_api import InternalApiQueryService, internal_api_response_for_path
from bot.logging_utils import get_logger


LOGGER = get_logger(__name__)

OPERATOR_DASHBOARD_ASSET_PATHS = (
    "/dashboard",
    "/dashboard/",
    "/dashboard/app.js",
    "/dashboard/styles.css",
)


@dataclass(frozen=True)
class OperatorDashboardAssetResponse:
    """One static dashboard asset response."""

    status: HTTPStatus
    content_type: str
    body: bytes


class OperatorDashboardHttpServer(ThreadingHTTPServer):
    """Serve the operator dashboard and internal API from one local host."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        query_service: InternalApiQueryService,
        control_service: OperatorControlService | None = None,
        refresh_interval_seconds: int,
    ) -> None:
        if refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be greater than zero.")
        self.query_service = query_service
        self.control_service = control_service or OperatorControlService(
            project_root=query_service.project_root
        )
        self.refresh_interval_seconds = refresh_interval_seconds
        super().__init__(server_address, OperatorDashboardRequestHandler)


class OperatorDashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the dashboard UI and pass through the internal API endpoints."""

    server: OperatorDashboardHttpServer

    def do_GET(self) -> None:
        response = operator_dashboard_response_for_request(
            query_service=self.server.query_service,
            control_service=self.server.control_service,
            method="GET",
            path=self.path,
            body=None,
            refresh_interval_seconds=self.server.refresh_interval_seconds,
        )
        self._write_content(
            status=response.status,
            content_type=response.content_type,
            body=response.body,
        )

    def do_POST(self) -> None:
        content_length_header = self.headers.get("Content-Length", "0") or "0"
        try:
            content_length = int(content_length_header)
        except ValueError:
            self._write_content(
                status=HTTPStatus.BAD_REQUEST,
                content_type="application/json; charset=utf-8",
                body=json.dumps(
                    {
                        "error": "invalid_content_length",
                        "message": "Content-Length must be an integer.",
                    },
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )
            return
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""
        response = operator_dashboard_response_for_request(
            query_service=self.server.query_service,
            control_service=self.server.control_service,
            method="POST",
            path=self.path,
            body=raw_body,
            refresh_interval_seconds=self.server.refresh_interval_seconds,
        )
        self._write_content(
            status=response.status,
            content_type=response.content_type,
            body=response.body,
        )

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("operator-dashboard %s", format % args)

    def _write_content(
        self,
        *,
        status: HTTPStatus,
        content_type: str,
        body: bytes,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_operator_dashboard_server(
    *,
    query_service: InternalApiQueryService,
    control_service: OperatorControlService | None = None,
    host: str = "127.0.0.1",
    port: int = 8780,
    refresh_interval_seconds: int = 5,
) -> OperatorDashboardHttpServer:
    """Return a configured local server for the operator dashboard."""

    return OperatorDashboardHttpServer(
        (host, port),
        query_service=query_service,
        control_service=control_service,
        refresh_interval_seconds=refresh_interval_seconds,
    )


def serve_operator_dashboard(
    *,
    query_service: InternalApiQueryService,
    control_service: OperatorControlService | None = None,
    host: str = "127.0.0.1",
    port: int = 8780,
    refresh_interval_seconds: int = 5,
) -> None:
    """Run the operator dashboard server until interrupted."""

    server = create_operator_dashboard_server(
        query_service=query_service,
        control_service=control_service,
        host=host,
        port=port,
        refresh_interval_seconds=refresh_interval_seconds,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def dashboard_asset_for_path(
    *,
    path: str,
    refresh_interval_seconds: int,
) -> OperatorDashboardAssetResponse | None:
    """Return one static dashboard asset for the requested path."""

    normalized_path = urlparse(path).path
    if normalized_path in {"/dashboard", "/dashboard/"}:
        return OperatorDashboardAssetResponse(
            status=HTTPStatus.OK,
            content_type="text/html; charset=utf-8",
            body=_dashboard_html(refresh_interval_seconds).encode("utf-8"),
        )
    if normalized_path == "/dashboard/styles.css":
        return OperatorDashboardAssetResponse(
            status=HTTPStatus.OK,
            content_type="text/css; charset=utf-8",
            body=_dashboard_styles().encode("utf-8"),
        )
    if normalized_path == "/dashboard/app.js":
        return OperatorDashboardAssetResponse(
            status=HTTPStatus.OK,
            content_type="application/javascript; charset=utf-8",
            body=_dashboard_app_js().encode("utf-8"),
        )
    return None


def operator_dashboard_response_for_path(
    *,
    query_service: InternalApiQueryService,
    control_service: OperatorControlService | None = None,
    path: str,
    refresh_interval_seconds: int,
) -> OperatorDashboardAssetResponse:
    """Return either a dashboard asset or a proxied internal API JSON payload."""

    return operator_dashboard_response_for_request(
        query_service=query_service,
        control_service=control_service,
        method="GET",
        path=path,
        body=None,
        refresh_interval_seconds=refresh_interval_seconds,
    )


def operator_dashboard_response_for_request(
    *,
    query_service: InternalApiQueryService,
    control_service: OperatorControlService | None = None,
    method: str,
    path: str,
    body: bytes | None,
    refresh_interval_seconds: int,
) -> OperatorDashboardAssetResponse:
    """Return a dashboard asset or proxied API/control JSON payload."""

    asset = dashboard_asset_for_path(
        path=path,
        refresh_interval_seconds=refresh_interval_seconds,
    )
    if asset is not None:
        return asset
    normalized_path = urlparse(path).path.rstrip("/") or "/"
    if normalized_path == "/control" or normalized_path.startswith("/control/"):
        status, payload = operator_control_response_for_request(
            control_service=control_service
            or OperatorControlService(project_root=query_service.project_root),
            method=method,
            path=path,
            body=body,
        )
    else:
        if method != "GET":
            status, payload = HTTPStatus.METHOD_NOT_ALLOWED, {
                "error": "method_not_allowed",
                "method": method,
                "allowed_methods": ["GET"],
            }
        else:
            status, payload = internal_api_response_for_path(
                query_service=query_service,
                path=path,
            )
    return OperatorDashboardAssetResponse(
        status=status,
        content_type="application/json; charset=utf-8",
        body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )


def _dashboard_html(refresh_interval_seconds: int) -> str:
    config_json = json.dumps({"refreshIntervalSeconds": refresh_interval_seconds})
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Investopedia Bot Operator Dashboard</title>
    <link rel="stylesheet" href="/dashboard/styles.css" />
    <script>
      window.__BOT_DASHBOARD_CONFIG__ = {config_json};
    </script>
    <script type="module" src="/dashboard/app.js"></script>
  </head>
  <body>
    <div class="page-shell">
      <header class="hero">
        <div class="hero-copy">
          <p class="eyebrow">Investopedia Bot</p>
          <h1>Operator Dashboard</h1>
          <p class="hero-lede">
            Live operator console over the internal backend and control APIs.
            Polling every <span id="refresh-interval-label">{refresh_interval_seconds}s</span>.
          </p>
        </div>
        <div class="hero-side">
          <button class="refresh-button" id="refresh-button" type="button">Refresh now</button>
          <div class="pill neutral" id="connection-pill">Loading</div>
        </div>
      </header>

      <section class="overview-strip">
        <div class="overview-item">
          <span class="overview-label">API host</span>
          <strong>same origin</strong>
        </div>
        <div class="overview-item">
          <span class="overview-label">Last refresh</span>
          <strong id="last-refresh-value">Waiting for first poll</strong>
        </div>
        <div class="overview-item">
          <span class="overview-label">Fetch status</span>
          <strong id="fetch-status-value">Idle</strong>
        </div>
      </section>

      <section class="feedback-strip" id="control-feedback" hidden></section>

      <main class="panel-grid">
        <section class="panel panel-wide">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Health</p>
              <h2>Service Health</h2>
            </div>
            <p class="panel-note">Supervisor liveness, connectivity, and recent warnings.</p>
          </div>
          <div class="panel-body is-loading" id="service-health-body">Loading service health...</div>
        </section>

        <section class="panel panel-wide">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">State</p>
              <h2>Market State</h2>
            </div>
            <p class="panel-note">Current alertable states and recent transitions.</p>
          </div>
          <div class="panel-body is-loading" id="market-state-body">Loading market state...</div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Portfolio</p>
              <h2>Portfolio State</h2>
            </div>
            <p class="panel-note">Latest review summaries and current action buckets.</p>
          </div>
          <div class="panel-body is-loading" id="portfolio-state-body">Loading portfolio state...</div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Execution</p>
              <h2>Pending Orders</h2>
            </div>
            <p class="panel-note">Active orders, capacity reservation, and execution realism.</p>
          </div>
          <div class="panel-body is-loading" id="pending-orders-body">Loading pending orders...</div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Queue</p>
              <h2>Candidate Queue</h2>
            </div>
            <p class="panel-note">Top-priority names and blocked or rejected pressure points.</p>
          </div>
          <div class="panel-body is-loading" id="candidate-queue-body">Loading candidate queue...</div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <p class="panel-kicker">Analytics</p>
              <h2>Trade Review Snapshot</h2>
            </div>
            <p class="panel-note">Latest compact trade feedback metrics.</p>
          </div>
          <div class="panel-body is-loading" id="analytics-body">Loading analytics snapshot...</div>
        </section>
      </main>
    </div>
  </body>
</html>
"""


def _dashboard_styles() -> str:
    return """
:root {
  --bg: #f4efe6;
  --bg-alt: #efe6d4;
  --panel: rgba(255, 252, 247, 0.92);
  --panel-border: rgba(32, 46, 43, 0.14);
  --panel-shadow: 0 24px 60px rgba(42, 54, 49, 0.14);
  --ink: #1d2624;
  --muted: #5f6c67;
  --accent: #b6502f;
  --accent-soft: rgba(182, 80, 47, 0.12);
  --good: #24664c;
  --good-soft: rgba(36, 102, 76, 0.14);
  --warn: #8d5b12;
  --warn-soft: rgba(141, 91, 18, 0.14);
  --bad: #8d2f2f;
  --bad-soft: rgba(141, 47, 47, 0.14);
  --neutral: #51615b;
  --neutral-soft: rgba(81, 97, 91, 0.14);
  --radius: 24px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(182, 80, 47, 0.18), transparent 34%),
    radial-gradient(circle at top right, rgba(36, 102, 76, 0.14), transparent 28%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 100%);
  color: var(--ink);
  font-family: "Trebuchet MS", "Avenir Next", "Segoe UI", sans-serif;
}

.page-shell {
  width: min(1380px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.hero {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-end;
  padding: 28px;
  background: linear-gradient(135deg, rgba(255, 249, 240, 0.94), rgba(243, 235, 220, 0.78));
  border: 1px solid rgba(29, 38, 36, 0.1);
  border-radius: calc(var(--radius) + 6px);
  box-shadow: var(--panel-shadow);
}

.eyebrow,
.panel-kicker {
  margin: 0 0 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: var(--accent);
  font-weight: 700;
}

.hero h1,
.panel h2 {
  margin: 0;
  font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-weight: 700;
  line-height: 1.05;
}

.hero h1 {
  font-size: clamp(2.1rem, 4vw, 3.7rem);
  max-width: 10ch;
}

.hero-lede,
.panel-note,
.empty-row,
.note,
.detail-line {
  color: var(--muted);
}

.hero-lede {
  margin: 14px 0 0;
  max-width: 58ch;
  line-height: 1.55;
}

.hero-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 12px;
}

.refresh-button {
  appearance: none;
  border: none;
  border-radius: 999px;
  padding: 12px 18px;
  background: linear-gradient(135deg, #1e302b, #355249);
  color: #fffaf2;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 14px 28px rgba(29, 38, 36, 0.18);
}

.refresh-button:hover {
  transform: translateY(-1px);
}

.overview-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 0;
}

.overview-item,
.panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  box-shadow: var(--panel-shadow);
  backdrop-filter: blur(12px);
}

.feedback-strip {
  margin-top: 14px;
}

.overview-item {
  border-radius: 20px;
  padding: 16px 18px;
}

.overview-label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
}

.panel-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 18px;
}

.panel {
  border-radius: var(--radius);
  padding: 22px;
}

.panel-wide {
  grid-column: span 2;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.panel-note {
  margin: 6px 0 0;
  max-width: 32ch;
  text-align: right;
  font-size: 0.94rem;
}

.panel-body {
  margin-top: 18px;
}

.control-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}

.action-button {
  appearance: none;
  border: none;
  border-radius: 999px;
  padding: 10px 14px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 120ms ease, opacity 120ms ease;
}

.action-button:hover:not(:disabled) {
  transform: translateY(-1px);
}

.action-button:disabled {
  cursor: not-allowed;
  opacity: 0.56;
}

.action-button.good {
  background: var(--good-soft);
  color: var(--good);
}

.action-button.warn {
  background: var(--warn-soft);
  color: var(--warn);
}

.action-button.bad {
  background: var(--bad-soft);
  color: var(--bad);
}

.action-button.neutral {
  background: var(--neutral-soft);
  color: var(--neutral);
}

.action-cluster {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.inline-form {
  margin-top: 14px;
  padding: 14px;
  border-radius: 18px;
  background: rgba(255, 249, 241, 0.92);
  border: 1px solid rgba(29, 38, 36, 0.1);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.field-block label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.field-block input {
  width: 100%;
  border: 1px solid rgba(29, 38, 36, 0.14);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.88);
  color: var(--ink);
  font: inherit;
}

.form-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.is-loading {
  color: var(--muted);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.metric-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255, 249, 241, 0.86);
  border: 1px solid rgba(29, 38, 36, 0.08);
}

.metric-label {
  display: block;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: var(--muted);
}

.metric-value {
  font-size: 1.04rem;
  font-weight: 700;
}

.pill,
.tag {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 7px 11px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.pill.good,
.tag.good {
  background: var(--good-soft);
  color: var(--good);
}

.pill.warn,
.tag.warn {
  background: var(--warn-soft);
  color: var(--warn);
}

.pill.bad,
.tag.bad {
  background: var(--bad-soft);
  color: var(--bad);
}

.pill.neutral,
.tag.neutral {
  background: var(--neutral-soft);
  color: var(--neutral);
}

.detail-stack,
.chip-row,
.list-block,
.table-wrap {
  margin-top: 14px;
}

.detail-line,
.note,
.empty-row {
  font-size: 0.94rem;
  line-height: 1.5;
}

.note {
  margin-bottom: 10px;
  padding: 11px 13px;
  border-radius: 16px;
  background: rgba(255, 247, 233, 0.9);
  border: 1px solid rgba(141, 91, 18, 0.15);
}

.note.bad {
  background: rgba(255, 239, 239, 0.92);
  border-color: rgba(141, 47, 47, 0.16);
  color: var(--bad);
}

.note.warn {
  background: rgba(255, 247, 233, 0.95);
  border-color: rgba(141, 91, 18, 0.2);
  color: var(--warn);
}

.list-block {
  display: grid;
  gap: 10px;
}

.list-row {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(250, 244, 236, 0.88);
  border: 1px solid rgba(29, 38, 36, 0.08);
}

.list-row strong {
  display: block;
  margin-bottom: 4px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.94rem;
}

th,
td {
  padding: 10px 10px 10px 0;
  border-bottom: 1px solid rgba(29, 38, 36, 0.08);
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

code {
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 0.88rem;
  background: rgba(29, 38, 36, 0.07);
  padding: 2px 6px;
  border-radius: 8px;
}

@media (max-width: 980px) {
  .hero,
  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .hero-side {
    width: 100%;
    align-items: flex-start;
  }

  .panel-grid,
  .overview-strip {
    grid-template-columns: 1fr;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }

  .panel-wide {
    grid-column: span 1;
  }

  .panel-note {
    text-align: left;
  }
}
"""


def _dashboard_app_js() -> str:
    return """
const CONFIG = window.__BOT_DASHBOARD_CONFIG__ || {};
const REFRESH_INTERVAL_SECONDS = Number(CONFIG.refreshIntervalSeconds || 5);
const ENDPOINTS = {
  health: "/health",
  marketState: "/market-state",
  marketTransitions: "/market-state/transitions",
  pendingOrders: "/pending-orders",
  portfolio: "/portfolio",
  analytics: "/analytics/trade-review",
  controlSafety: "/control/safety",
};
const CONTROL_ENDPOINTS = {
  pauseExecution: "/control/execution/pause",
  resumeExecution: "/control/execution/resume",
  submitOrder: "/control/orders/submit",
  cancelOrder: "/control/orders/cancel",
  replaceOrder: "/control/orders/replace",
  forceBrokerSync: "/control/broker/sync",
};
const DEFAULT_CONTROL_BROKER = "alpaca";

const els = {
  refreshButton: document.getElementById("refresh-button"),
  refreshIntervalLabel: document.getElementById("refresh-interval-label"),
  lastRefreshValue: document.getElementById("last-refresh-value"),
  fetchStatusValue: document.getElementById("fetch-status-value"),
  connectionPill: document.getElementById("connection-pill"),
  controlFeedback: document.getElementById("control-feedback"),
  serviceHealthBody: document.getElementById("service-health-body"),
  marketStateBody: document.getElementById("market-state-body"),
  portfolioStateBody: document.getElementById("portfolio-state-body"),
  pendingOrdersBody: document.getElementById("pending-orders-body"),
  candidateQueueBody: document.getElementById("candidate-queue-body"),
  analyticsBody: document.getElementById("analytics-body"),
};

const state = {
  loading: false,
  loadPromise: null,
  pendingRefresh: false,
  lastRefreshAt: null,
  currentResults: null,
  actionInFlight: false,
  lastControlResult: null,
  replaceEditorOrderId: null,
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function isRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function formatDateTime(value) {
  if (!value) return "n/a";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
}

function formatNumber(value) {
  return typeof value === "number" ? value.toLocaleString() : "n/a";
}

function formatCurrency(value) {
  return typeof value === "number"
    ? value.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 })
    : "n/a";
}

function formatRatioPercent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "n/a";
}

function formatOptionalCount(value) {
  return typeof value === "number" ? formatNumber(value) : "Unavailable";
}

function formatControlStatus(value) {
  return value ? String(value).replace(/_/g, " ") : "n/a";
}

function describeCandidateActionability(item) {
  if (typeof item.actionable_now !== "boolean") {
    return "Actionability unavailable.";
  }
  return item.actionable_now ? "Actionable now." : "Tracked but not actionable right now.";
}

function toneForStatus(text) {
  const value = String(text || "").toLowerCase();
  if (value.includes("connected") || value.includes("ok")) return "good";
  if (value.includes("retry") || value.includes("stale") || value.includes("watch")) return "warn";
  if (value.includes("exit") || value.includes("error") || value.includes("down")) return "bad";
  return "neutral";
}

function renderMetricGrid(items) {
  return `
    <div class="metric-grid">
      ${items
        .map(
          ([label, value]) => `
            <div class="metric-card">
              <span class="metric-label">${escapeHtml(label)}</span>
              <strong class="metric-value">${escapeHtml(value)}</strong>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderActionButton({ label, tone = "neutral", action, orderId = null, disabled = false }) {
  const parts = [
    `class="action-button ${tone}"`,
    'type="button"',
    `data-control-action="${escapeHtml(action)}"`,
  ];
  if (orderId) {
    parts.push(`data-order-id="${escapeHtml(orderId)}"`);
  }
  if (disabled) {
    parts.push("disabled");
    parts.push('aria-disabled="true"');
  }
  return `<button ${parts.join(" ")}>${escapeHtml(label)}</button>`;
}

function renderChipRow(items, tone = "neutral") {
  if (!items.length) {
    return '<div class="empty-row">Nothing notable right now.</div>';
  }
  return `
    <div class="chip-row">
      ${items.map((item) => `<span class="tag ${toneForStatus(item.tone || tone)}">${escapeHtml(item.label)}</span>`).join("")}
    </div>
  `;
}

function renderList(rows, emptyText) {
  if (!rows.length) {
    return `<div class="empty-row">${escapeHtml(emptyText)}</div>`;
  }
  return `
    <div class="list-block">
      ${rows
        .map(
          (row) => `
            <div class="list-row">
              ${row.title ? `<strong>${escapeHtml(row.title)}</strong>` : ""}
              ${row.body ? `<div class="detail-line">${escapeHtml(row.body)}</div>` : ""}
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderNotes(messages, tone = "neutral") {
  return messages
    .filter(Boolean)
    .map((message) => `<div class="note ${tone}">${escapeHtml(message)}</div>`)
    .join("");
}

function controlSafetySnapshot(result) {
  const fallback = {
    available: false,
    readable: false,
    executionEnabled: null,
    executionMode: "unknown",
    pausedReason: null,
    warnings: [],
    fetchError: result && !result.ok ? result.error : null,
    payload: {},
  };
  if (!result || !result.ok) {
    return fallback;
  }
  const payload = asObject(result.payload);
  const data = asObject(payload.data);
  const safetyState = asObject(data.safety_state);
  return {
    available: payload.available === true,
    readable: data.control_state_readable !== false,
    executionEnabled:
      typeof safetyState.execution_submission_enabled === "boolean"
        ? safetyState.execution_submission_enabled
        : null,
    executionMode: safetyState.execution_mode || "unknown",
    pausedReason: safetyState.paused_reason || null,
    warnings: asArray(payload.warnings),
    fetchError: payload.available ? null : payload.not_available_reason || "control_safety_unavailable",
    payload,
  };
}

function resolveControlBroker(order = null) {
  if (isRecord(order) && typeof order.broker_name === "string" && order.broker_name.trim()) {
    return order.broker_name.trim().toLowerCase();
  }
  const pendingPayload =
    state.currentResults && state.currentResults.pendingOrders
      ? asObject(state.currentResults.pendingOrders.payload)
      : {};
  const pendingData = asObject(pendingPayload.data);
  const activeOrders = asArray(pendingData.active_orders);
  const brokerOrder = activeOrders.find(
    (item) => typeof item.broker_name === "string" && item.broker_name.trim()
  );
  if (brokerOrder) {
    return brokerOrder.broker_name.trim().toLowerCase();
  }
  return DEFAULT_CONTROL_BROKER;
}

function renderControlFeedback() {
  const element = els.controlFeedback;
  const result = state.lastControlResult;
  if (!result) {
    element.hidden = true;
    element.innerHTML = "";
    return;
  }

  const payload = asObject(result.payload);
  const warnings = asArray(payload.warnings);
  let tone = "neutral";
  let primaryMessage = result.error || payload.message || payload.error || "Operator action completed.";
  const extraNotes = [];

  if (payload.status === "completed") {
    tone = "good";
  } else if (payload.status === "needs_confirmation") {
    tone = "warn";
    if (payload.data && payload.data.transitional_state) {
      extraNotes.push(
        `Broker state is ${payload.data.transitional_state}. Reservation remains active until broker confirmation arrives.`
      );
    }
    if (payload.data && payload.data.reservation_active === true) {
      extraNotes.push("The order still reserves execution capacity while broker confirmation is pending.");
    }
  } else if (payload.status === "no_op") {
    tone = "neutral";
  } else if (payload.status === "rejected" || payload.status === "failed" || payload.error) {
    tone = "bad";
  } else if (!result.ok) {
    tone = "bad";
  }

  if (payload.error_code === "control_state_unavailable") {
    extraNotes.push(
      "Operator control state is unreadable, so mutating actions are blocked until the backend state is repaired."
    );
  }

  element.hidden = false;
  element.innerHTML = `
    <div class="note ${tone}">
      <strong>${escapeHtml(payload.command_name || "Operator action")}</strong>
      <div class="detail-line">${escapeHtml(primaryMessage)}</div>
      ${payload.status ? `<div class="detail-line">Result: ${escapeHtml(formatControlStatus(payload.status))}</div>` : ""}
    </div>
    ${renderNotes(extraNotes, tone === "bad" ? "bad" : "warn")}
    ${renderNotes(warnings, warnings.length ? "warn" : "neutral")}
  `;
}

function setPanelUnavailable(element, label, reason) {
  element.classList.remove("is-loading");
  element.innerHTML = `
    <div class="note bad">${escapeHtml(label)}</div>
    <div class="empty-row">${escapeHtml(reason || "Data is not available yet.")}</div>
  `;
}

async function fetchEndpoint(path) {
  try {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (error) {
        return {
          ok: false,
          error: `Invalid JSON from ${path}: ${error.message}`,
        };
      }
    }
    if (!response.ok) {
      return {
        ok: false,
        error: `HTTP ${response.status} from ${path}`,
        payload,
      };
    }
    return { ok: true, payload: asObject(payload) };
  } catch (error) {
    return {
      ok: false,
      error: `Fetch failed for ${path}: ${error.message}`,
    };
  }
}

async function postControlCommand(path, payload) {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      cache: "no-store",
      body: JSON.stringify(payload || {}),
    });
    const text = await response.text();
    let parsedPayload = {};
    if (text) {
      try {
        parsedPayload = JSON.parse(text);
      } catch (error) {
        return {
          ok: false,
          statusCode: response.status,
          error: `Invalid JSON from ${path}: ${error.message}`,
        };
      }
    }
    return {
      ok: response.ok,
      statusCode: response.status,
      payload: asObject(parsedPayload),
    };
  } catch (error) {
    return {
      ok: false,
      error: `Control request failed for ${path}: ${error.message}`,
    };
  }
}

function renderHealth(result, controlResult) {
  const element = els.serviceHealthBody;
  element.classList.remove("is-loading");
  const control = controlSafetySnapshot(controlResult);
  let payload = {};
  let data = {};
  let status = {};
  let serviceState = "unknown";
  const warningNotes = [];
  const errorNotes = [];

  if (!result.ok) {
    errorNotes.push(result.error || "Health endpoint unavailable.");
    els.connectionPill.className = "pill bad";
    els.connectionPill.textContent = "API error";
  } else {
    payload = asObject(result.payload);
    if (!payload.available) {
      warningNotes.push(
        payload.not_available_reason || "Live service status has not been written yet."
      );
      els.connectionPill.className = "pill neutral";
      els.connectionPill.textContent = "No status";
    } else {
      data = asObject(payload.data);
      status = asObject(data.status);
      serviceState = status.service_state || "unknown";
      els.connectionPill.className = `pill ${toneForStatus(serviceState)}`;
      els.connectionPill.textContent = serviceState.replace(/_/g, " ");
    }
  }

  if (status.last_warning) warningNotes.push(status.last_warning);
  if (status.last_error) errorNotes.push(status.last_error);
  if (!control.available) {
    warningNotes.push(control.fetchError || "Control safety endpoint unavailable.");
  } else {
    warningNotes.push(...control.warnings);
    if (!control.readable) {
      errorNotes.push(
        "Operator control state is unreadable. Mutating actions are blocked until backend state is repaired."
      );
    } else if (control.executionEnabled === false) {
      warningNotes.push(
        control.pausedReason
          ? `Execution submissions paused: ${control.pausedReason}`
          : "Execution submissions are paused."
      );
    }
  }

  const pauseDisabled =
    state.actionInFlight ||
    !control.available ||
    !control.readable ||
    control.executionEnabled !== true;
  const resumeDisabled =
    state.actionInFlight ||
    !control.available ||
    !control.readable ||
    control.executionEnabled !== false;
  const syncDisabled = state.actionInFlight || !control.available || !control.readable;

  element.innerHTML = `
    ${renderMetricGrid([
      ["Service state", serviceState.replace(/_/g, " ")],
      ["Connected", status.connected ? "Yes" : "No"],
      ["Stale", status.stale ? "Yes" : "No"],
      ["Reconnect attempts", formatNumber(status.reconnect_attempt_count)],
      ["Cycle count", formatNumber(status.cycle_count)],
      ["Last cycle warnings", formatNumber(status.last_cycle_warning_count)],
      ["Last message", formatDateTime(status.last_message_at_utc)],
      ["Last successful cycle", formatDateTime(status.last_successful_flush_at_utc)],
      ["Execution submissions", control.executionEnabled === null ? "Unavailable" : control.executionEnabled ? "Enabled" : "Paused"],
      ["Execution mode", formatControlStatus(control.executionMode)],
    ])}
    ${renderNotes(errorNotes, "bad")}
    ${renderNotes(warningNotes, warningNotes.length ? "warn" : "neutral")}
    <div class="detail-stack">
      <div class="detail-line"><strong>Execution controls</strong></div>
      <div class="control-toolbar">
        ${renderActionButton({
          label: "Pause submissions",
          tone: "warn",
          action: "pause-execution",
          disabled: pauseDisabled,
        })}
        ${renderActionButton({
          label: "Resume submissions",
          tone: "good",
          action: "resume-execution",
          disabled: resumeDisabled,
        })}
        ${renderActionButton({
          label: "Force broker sync",
          tone: "neutral",
          action: "force-broker-sync",
          disabled: syncDisabled,
        })}
      </div>
    </div>
  `;
}

function renderMarketState(marketResult, transitionsResult) {
  const element = els.marketStateBody;
  element.classList.remove("is-loading");
  if (!marketResult.ok) {
    setPanelUnavailable(element, "Market state endpoint unavailable", marketResult.error);
    return;
  }

  const payload = asObject(marketResult.payload);
  if (!payload.available) {
    setPanelUnavailable(
      element,
      "Market state unavailable",
      payload.not_available_reason || "No current market-state snapshot is available yet."
    );
    return;
  }

  const data = asObject(payload.data);
  const snapshot = asObject(data.snapshot);
  const transitionsPayload = transitionsResult.ok ? asObject(transitionsResult.payload) : {};
  const transitionsData = transitionsPayload.available ? asObject(transitionsPayload.data) : {};
  const recentTransitions = asArray(
    transitionsPayload.available ? transitionsData.transitions : data.recent_transitions
  );
  const alertableStates = asArray(data.current_alertable_states);

  element.innerHTML = `
    ${renderMetricGrid([
      ["As of", formatDateTime(snapshot.as_of_timestamp)],
      ["Workflows", asArray(snapshot.source_workflows).join(", ") || "n/a"],
      ["Alertable states", formatNumber(alertableStates.length)],
      ["Transitions", formatNumber(
        transitionsPayload.available ? transitionsData.transition_count : data.transition_count
      )],
    ])}
    <div class="detail-stack">
      <div class="detail-line"><strong>Alertable now</strong></div>
      ${renderList(
        alertableStates.map((item) => ({
          title: `${item.symbol || item.state_key || "State"} | ${item.action || item.category || "Alert"}`,
          body: item.rationale || item.message || "No rationale recorded.",
        })),
        "No current alertable states."
      )}
    </div>
    <div class="detail-stack">
      <div class="detail-line"><strong>Recent transitions</strong></div>
      ${renderList(
        recentTransitions.map((item) => ({
          title: `${item.transition_type || "Transition"}${item.symbol ? ` | ${item.symbol}` : ""}`,
          body: item.message || `${item.previous_value || "n/a"} -> ${item.current_value || "n/a"}`,
        })),
        "No meaningful transitions detected."
      )}
    </div>
  `;
}

function renderPortfolio(result) {
  const element = els.portfolioStateBody;
  element.classList.remove("is-loading");
  if (!result.ok) {
    setPanelUnavailable(element, "Portfolio endpoint unavailable", result.error);
    return;
  }

  const payload = asObject(result.payload);
  if (!payload.available) {
    setPanelUnavailable(
      element,
      "Portfolio state unavailable",
      payload.not_available_reason || "Portfolio review data is not available yet."
    );
    return;
  }

  const data = asObject(payload.data);
  const portfolioSummary = asObject(data.portfolio_review_summary);
  const intradaySummary = asObject(data.intraday_review_summary);
  const actionStates = asArray(data.current_action_states);
  const holdNames = actionStates.filter((item) => item.action === "HOLD").map((item) => item.symbol);
  const watchNames = asArray(data.watch_closely);
  const exitNames = asArray(data.exit_candidates);
  const raiseStopNames = asArray(data.raise_stop);

  element.innerHTML = `
    ${renderMetricGrid([
      ["Positions", formatOptionalCount(portfolioSummary.position_count)],
      ["Hold", formatOptionalCount(portfolioSummary.hold_count)],
      ["Watch closely", formatOptionalCount(portfolioSummary.watch_closely_count)],
      ["Exit candidate", formatOptionalCount(portfolioSummary.exit_candidate_count)],
      ["Raise stop", formatOptionalCount(portfolioSummary.raise_stop_count)],
      ["Intraday watch", formatOptionalCount(intradaySummary.watch_closely_count)],
    ])}
    <div class="detail-stack">
      <div class="detail-line"><strong>Current action buckets</strong></div>
      ${renderChipRow(holdNames.map((label) => ({ label })))}
      ${renderChipRow(watchNames.map((label) => ({ label, tone: "warn" })), "warn")}
      ${renderChipRow(exitNames.map((label) => ({ label, tone: "bad" })), "bad")}
      ${renderChipRow(raiseStopNames.map((label) => ({ label, tone: "good" })), "good")}
    </div>
  `;
}

function renderPendingOrderActions(order, control) {
  const isActiveOrder = order.status === "pending" || order.status === "partially_filled";
  const controlBlocked = !control.available || !control.readable;
  const submitDisabled =
    state.actionInFlight ||
    controlBlocked ||
    control.executionEnabled !== true ||
    !isActiveOrder ||
    Boolean(order.broker_order_id);
  const cancelDisabled =
    state.actionInFlight ||
    controlBlocked ||
    !isActiveOrder ||
    !order.broker_order_id ||
    order.broker_status === "pending_cancel";
  const replaceDisabled =
    state.actionInFlight ||
    controlBlocked ||
    !isActiveOrder ||
    !order.broker_order_id ||
    order.broker_status === "pending_cancel";
  return `
    <div class="action-cluster">
      ${renderActionButton({
        label: "Submit",
        tone: "good",
        action: "submit-order",
        orderId: order.order_id,
        disabled: submitDisabled,
      })}
      ${renderActionButton({
        label: "Cancel",
        tone: "bad",
        action: "cancel-order",
        orderId: order.order_id,
        disabled: cancelDisabled,
      })}
      ${renderActionButton({
        label: state.replaceEditorOrderId === order.order_id ? "Editing…" : "Replace",
        tone: "warn",
        action: "open-replace-editor",
        orderId: order.order_id,
        disabled: replaceDisabled,
      })}
    </div>
  `;
}

function renderReplaceEditor(order) {
  return `
    <div class="inline-form">
      <div class="detail-line">
        <strong>Replace ${escapeHtml(order.symbol || order.order_id || "order")}</strong>
      </div>
      <div class="detail-line">
        Only changed fields will be sent to the backend replacement request.
      </div>
      <div class="form-grid">
        <div class="field-block">
          <label for="replace-quantity">Quantity</label>
          <input id="replace-quantity" type="number" min="1" step="1" value="${escapeHtml(order.requested_quantity || "")}" />
        </div>
        <div class="field-block">
          <label for="replace-limit-price">Limit price</label>
          <input id="replace-limit-price" type="number" min="0.01" step="0.01" value="${escapeHtml(order.requested_price || "")}" />
        </div>
        <div class="field-block">
          <label for="replace-stop-price">Stop price</label>
          <input id="replace-stop-price" type="number" min="0.01" step="0.01" value="${escapeHtml(order.stop_price || "")}" />
        </div>
      </div>
      <div class="form-actions">
        ${renderActionButton({
          label: "Submit replacement",
          tone: "warn",
          action: "submit-replace-order",
          orderId: order.order_id,
          disabled: state.actionInFlight,
        })}
        ${renderActionButton({
          label: "Dismiss",
          tone: "neutral",
          action: "dismiss-replace-editor",
          orderId: order.order_id,
          disabled: state.actionInFlight,
        })}
      </div>
    </div>
  `;
}

function renderPendingOrders(result, controlResult) {
  const element = els.pendingOrdersBody;
  element.classList.remove("is-loading");
  if (!result.ok) {
    setPanelUnavailable(element, "Pending-orders endpoint unavailable", result.error);
    return;
  }

  const payload = asObject(result.payload);
  const warnings = asArray(payload.warnings);
  const data = asObject(payload.data);
  const summary = asObject(data.summary);
  const activeOrders = asArray(data.active_orders);
  const control = controlSafetySnapshot(controlResult);
  const controlNotes = [];
  if (!control.available) {
    controlNotes.push("Control API state is unavailable. Order actions are disabled.");
  } else if (!control.readable) {
    controlNotes.push("Operator control state is unreadable. Mutating actions are blocked.");
  } else if (control.executionEnabled === false) {
    controlNotes.push("Execution submissions are paused. Submit actions are disabled.");
  }
  let replaceOrder = activeOrders.find((order) => order.order_id === state.replaceEditorOrderId) || null;
  const replaceEditorBlocked =
    !control.available ||
    !control.readable ||
    (replaceOrder && replaceOrder.broker_status === "pending_cancel");
  if ((!replaceOrder || replaceEditorBlocked) && state.replaceEditorOrderId) {
    state.replaceEditorOrderId = null;
    replaceOrder = null;
  }

  element.innerHTML = `
    ${renderMetricGrid([
      ["Active orders", formatNumber(data.active_order_count)],
      ["Capacity-reserving buys", formatNumber(summary.capacity_reserving_buy_order_count)],
      ["Reserved notional", formatCurrency(summary.reserved_notional)],
      ["Pending fill notional", summary.pending_fill_notional === null ? "Unknown" : formatCurrency(summary.pending_fill_notional)],
      ["Reserved slots", summary.position_context_available ? formatNumber(summary.reserved_slot_count) : "Unknown"],
      ["Available slots", summary.available_slots === null ? "Unknown" : formatNumber(summary.available_slots)],
    ])}
    ${summary.position_context_available ? "" : renderNotes(
      [warnings[0] || "Live position context is unavailable. Capacity figures are partial."],
      "bad"
    )}
    ${renderNotes(controlNotes, controlNotes.length ? "warn" : "neutral")}
    <div class="table-wrap">
      ${
        activeOrders.length
          ? `
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Status</th>
                  <th>Broker</th>
                  <th>Preset</th>
                  <th>Reserved</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                ${activeOrders
                  .map(
                    (order) => `
                      <tr>
                        <td><strong>${escapeHtml(order.symbol || "n/a")}</strong></td>
                        <td>${escapeHtml(order.side || "n/a")}</td>
                        <td>${escapeHtml(order.broker_status || order.status || "n/a")}</td>
                        <td>${escapeHtml(order.broker_name || "local")}</td>
                        <td>${escapeHtml(order.preset_name || "n/a")}</td>
                        <td>${formatCurrency(order.reserved_notional)}</td>
                        <td>${renderPendingOrderActions(order, control)}</td>
                      </tr>
                    `
                  )
                  .join("")}
              </tbody>
            </table>
          `
          : '<div class="empty-row">No active pending orders.</div>'
      }
    </div>
    ${replaceOrder ? renderReplaceEditor(replaceOrder) : ""}
  `;
}

function renderCandidateQueue(result) {
  const element = els.candidateQueueBody;
  element.classList.remove("is-loading");
  if (!result.ok) {
    setPanelUnavailable(element, "Candidate queue unavailable", result.error);
    return;
  }

  const payload = asObject(result.payload);
  if (!payload.available) {
    setPanelUnavailable(
      element,
      "Candidate queue unavailable",
      payload.not_available_reason || "Market state is not available yet."
    );
    return;
  }

  const data = asObject(payload.data);
  const snapshot = asObject(data.snapshot);
  const topPriority = asArray(data.top_priority_candidates);
  const approvedQueue = asArray(snapshot.approved_candidate_queue);
  const constrained = approvedQueue.filter((item) => item.priority_bucket === "capacity_constrained");
  const rejectedReasons = asArray(snapshot.top_rejected_reasons_summary);

  element.innerHTML = `
    ${renderMetricGrid([
      ["Top priority", formatNumber(topPriority.length)],
      ["Approved queue", formatNumber(approvedQueue.length)],
      ["Capacity constrained", formatNumber(constrained.length)],
      ["Rejected reasons tracked", formatNumber(rejectedReasons.length)],
    ])}
    <div class="detail-stack">
      <div class="detail-line"><strong>Top-priority names</strong></div>
      ${renderList(
        topPriority.map((item) => ({
          title: `${item.symbol || "Symbol"} | rank ${item.rank || "n/a"} | ${item.preset_name || "preset"}`,
          body: describeCandidateActionability(item),
        })),
        "No top-priority candidates."
      )}
    </div>
    <div class="detail-stack">
      <div class="detail-line"><strong>Common rejected reasons</strong></div>
      ${renderList(
        rejectedReasons.map((item) => ({
          title: `${item.reason || "Reason"} (${item.count || 0})`,
          body: "Recent pressure point in the candidate pipeline.",
        })),
        "No rejected-reason summary is available."
      )}
    </div>
  `;
}

function renderAnalytics(result) {
  const element = els.analyticsBody;
  element.classList.remove("is-loading");
  if (!result.ok) {
    setPanelUnavailable(element, "Analytics endpoint unavailable", result.error);
    return;
  }

  const payload = asObject(result.payload);
  if (!payload.available) {
    setPanelUnavailable(
      element,
      "Trade-review analytics unavailable",
      payload.not_available_reason || "No analytics artifact is available yet."
    );
    return;
  }

  const summary = asObject(asObject(payload.data).summary);
  const decisionSummary = asObject(summary.decision_summary);
  const outcomeSummary = asObject(summary.outcome_summary);
  const executionSummary = asObject(summary.execution_priority_summary);

  element.innerHTML = `
    ${renderMetricGrid([
      ["As of date", summary.as_of_date || "n/a"],
      ["Events", formatNumber(summary.event_count)],
      ["Approved", formatNumber(decisionSummary.approved_count)],
      ["Rejected", formatNumber(decisionSummary.rejected_count)],
      ["Executed trades", formatNumber(executionSummary.executed_trade_count)],
      ["Avg current return", formatRatioPercent(outcomeSummary.average_current_return_pct)],
    ])}
    ${renderList(
      asArray(summary.notes).map((note) => ({ title: "Note", body: note })),
      "No analytics notes were recorded."
    )}
  `;
}

function renderDashboard(results) {
  renderControlFeedback();
  renderHealth(results.health, results.controlSafety);
  renderMarketState(results.marketState, results.marketTransitions);
  renderPortfolio(results.portfolio);
  renderPendingOrders(results.pendingOrders, results.controlSafety);
  renderCandidateQueue(results.marketState);
  renderAnalytics(results.analytics);
}

function renderCurrentDashboard() {
  if (state.currentResults) {
    renderDashboard(state.currentResults);
  } else {
    renderControlFeedback();
  }
}

async function loadDashboard({ manual = false } = {}) {
  if (state.loading) {
    state.pendingRefresh = true;
    return state.loadPromise || Promise.resolve();
  }
  state.loading = true;
  const currentLoadPromise = (async () => {
    let loadError = null;
    try {
      els.fetchStatusValue.textContent = manual ? "Manual refresh..." : "Refreshing...";
      const [health, marketState, marketTransitions, pendingOrders, portfolio, analytics, controlSafety] = await Promise.all([
        fetchEndpoint(ENDPOINTS.health),
        fetchEndpoint(ENDPOINTS.marketState),
        fetchEndpoint(ENDPOINTS.marketTransitions),
        fetchEndpoint(ENDPOINTS.pendingOrders),
        fetchEndpoint(ENDPOINTS.portfolio),
        fetchEndpoint(ENDPOINTS.analytics),
        fetchEndpoint(ENDPOINTS.controlSafety),
      ]);
      state.currentResults = {
        health,
        marketState,
        marketTransitions,
        pendingOrders,
        portfolio,
        analytics,
        controlSafety,
      };
      renderDashboard(state.currentResults);
      state.lastRefreshAt = new Date();
      els.lastRefreshValue.textContent = state.lastRefreshAt.toLocaleTimeString();
      const failedCount = [health, marketState, marketTransitions, pendingOrders, portfolio, analytics, controlSafety].filter(
        (result) => !result.ok
      ).length;
      els.fetchStatusValue.textContent = failedCount ? `${failedCount} fetch issue(s)` : "Fresh";
    } catch (error) {
      loadError = error;
    } finally {
      state.loading = false;
    }
    const shouldRefreshAgain = state.pendingRefresh;
    state.pendingRefresh = false;
    if (shouldRefreshAgain) {
      await loadDashboard({ manual: true });
    }
    if (loadError) {
      throw loadError;
    }
  })();
  state.loadPromise = currentLoadPromise;
  try {
    await currentLoadPromise;
  } finally {
    if (state.loadPromise === currentLoadPromise) {
      state.loadPromise = null;
    }
  }
}

function parseOptionalNumberInput(id, label) {
  const input = document.getElementById(id);
  if (!input) {
    return { present: false };
  }
  const rawValue = input.value.trim();
  if (!rawValue) {
    return { present: false };
  }
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return { error: `${label} must be a positive number.` };
  }
  return { present: true, value: parsed };
}

function setLocalControlError(message, commandName) {
  state.lastControlResult = {
    ok: false,
    payload: {
      command_name: commandName,
      status: "rejected",
      message,
      warnings: [],
    },
  };
  renderControlFeedback();
}

async function executeControlAction(path, payload) {
  if (state.actionInFlight) {
    return;
  }
  state.actionInFlight = true;
  renderCurrentDashboard();
  els.fetchStatusValue.textContent = "Applying operator action...";
  try {
    const result = await postControlCommand(path, payload);
    state.lastControlResult = result;
    await loadDashboard({ manual: true });
  } finally {
    state.actionInFlight = false;
    renderCurrentDashboard();
  }
}

async function handleControlAction(action, orderId) {
  if (action === "pause-execution") {
    await executeControlAction(CONTROL_ENDPOINTS.pauseExecution, {});
    return;
  }
  if (action === "resume-execution") {
    await executeControlAction(CONTROL_ENDPOINTS.resumeExecution, {});
    return;
  }
  if (action === "force-broker-sync") {
    await executeControlAction(CONTROL_ENDPOINTS.forceBrokerSync, {
      broker: resolveControlBroker(),
    });
    return;
  }

  const pendingPayload =
    state.currentResults && state.currentResults.pendingOrders
      ? asObject(state.currentResults.pendingOrders.payload)
      : {};
  const pendingData = asObject(pendingPayload.data);
  const activeOrders = asArray(pendingData.active_orders);
  const order = activeOrders.find((item) => item.order_id === orderId);
  if (!order) {
    setLocalControlError(`Pending order '${orderId}' is no longer present in the dashboard state.`, action);
    return;
  }

  if (action === "submit-order") {
    if (!window.confirm(`Submit pending order ${order.order_id} for ${order.symbol}?`)) {
      return;
    }
    await executeControlAction(CONTROL_ENDPOINTS.submitOrder, {
      broker: resolveControlBroker(order),
      order_id: order.order_id,
    });
    return;
  }

  if (action === "cancel-order") {
    if (!window.confirm(`Cancel broker order for ${order.order_id} (${order.symbol})?`)) {
      return;
    }
    await executeControlAction(CONTROL_ENDPOINTS.cancelOrder, {
      broker: resolveControlBroker(order),
      order_id: order.order_id,
    });
    return;
  }

  if (action === "open-replace-editor") {
    state.replaceEditorOrderId = order.order_id;
    renderCurrentDashboard();
    return;
  }

  if (action === "dismiss-replace-editor") {
    state.replaceEditorOrderId = null;
    renderCurrentDashboard();
    return;
  }

  if (action === "submit-replace-order") {
    const quantityInput = parseOptionalNumberInput("replace-quantity", "Quantity");
    const limitInput = parseOptionalNumberInput("replace-limit-price", "Limit price");
    const stopInput = parseOptionalNumberInput("replace-stop-price", "Stop price");
    const inputError = quantityInput.error || limitInput.error || stopInput.error;
    if (inputError) {
      setLocalControlError(inputError, "replace_pending_order");
      return;
    }
    const payload = {
      broker: resolveControlBroker(order),
      order_id: order.order_id,
    };
    const existingQuantity =
      typeof order.requested_quantity === "number" ? Math.round(order.requested_quantity) : null;
    const existingLimitPrice =
      typeof order.requested_price === "number" ? order.requested_price : null;
    const existingStopPrice =
      typeof order.stop_price === "number" ? order.stop_price : null;
    if (
      quantityInput.present &&
      (existingQuantity === null || Math.round(quantityInput.value) !== existingQuantity)
    ) {
      payload.quantity = Math.round(quantityInput.value);
    }
    if (
      limitInput.present &&
      (existingLimitPrice === null || limitInput.value !== existingLimitPrice)
    ) {
      payload.limit_price = limitInput.value;
    }
    if (
      stopInput.present &&
      (existingStopPrice === null || stopInput.value !== existingStopPrice)
    ) {
      payload.stop_price = stopInput.value;
    }
    if (!("quantity" in payload) && !("limit_price" in payload) && !("stop_price" in payload)) {
      setLocalControlError(
        "Change at least one replacement field before submitting the order amendment.",
        "replace_pending_order"
      );
      return;
    }
    state.replaceEditorOrderId = null;
    await executeControlAction(CONTROL_ENDPOINTS.replaceOrder, payload);
  }
}

els.refreshIntervalLabel.textContent = `${REFRESH_INTERVAL_SECONDS}s`;
els.refreshButton.addEventListener("click", () => {
  loadDashboard({ manual: true });
});
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-control-action]");
  if (!button || button.disabled) {
    return;
  }
  event.preventDefault();
  void handleControlAction(button.dataset.controlAction, button.dataset.orderId || null);
});

loadDashboard();
window.setInterval(() => {
  loadDashboard();
}, REFRESH_INTERVAL_SECONDS * 1000);
"""
