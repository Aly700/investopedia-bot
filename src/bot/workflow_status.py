"""Shared workflow input-status summarization and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_PROBLEM_STATUSES = {"degraded", "unavailable", "failed"}
_KNOWN_STATUSES = {"ok", "degraded", "unavailable", "failed"}


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class WorkflowInputIssue:
    """Compact description of one degraded or unavailable workflow input."""

    input_name: str
    status: str
    role_name: str | None = None
    provider: str | None = None
    issue_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "status": self.status,
            "role_name": self.role_name,
            "provider": self.provider,
            "issue_code": self.issue_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class WorkflowInputStatusSummary:
    """Compact summary for a workflow's structured input statuses."""

    total_count: int
    ok_count: int
    degraded_count: int
    unavailable_count: int
    failed_count: int
    issue_count: int
    healthy: bool
    problematic_inputs: tuple[str, ...] = ()
    issue_codes: tuple[str, ...] = ()
    issues: tuple[WorkflowInputIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "ok_count": self.ok_count,
            "degraded_count": self.degraded_count,
            "unavailable_count": self.unavailable_count,
            "failed_count": self.failed_count,
            "issue_count": self.issue_count,
            "healthy": self.healthy,
            "problematic_inputs": list(self.problematic_inputs),
            "issue_codes": list(self.issue_codes),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class WorkflowInputOverviewItem:
    """Compact operator-facing status for one workflow summary."""

    workflow_name: str
    status: str
    issue_count: int
    issue_codes: tuple[str, ...] = ()
    problematic_inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status,
            "issue_count": self.issue_count,
            "issue_codes": list(self.issue_codes),
            "problematic_inputs": list(self.problematic_inputs),
        }


@dataclass(frozen=True)
class WorkflowInputOverview:
    """Compact operator-facing overview across workflow summaries."""

    workflow_count: int
    healthy_workflow_count: int
    degraded_workflow_count: int
    unavailable_workflow_count: int
    failed_workflow_count: int
    problematic_workflow_count: int
    highest_severity: str | None = None
    problematic_workflows: tuple[WorkflowInputOverviewItem, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_count": self.workflow_count,
            "healthy_workflow_count": self.healthy_workflow_count,
            "degraded_workflow_count": self.degraded_workflow_count,
            "unavailable_workflow_count": self.unavailable_workflow_count,
            "failed_workflow_count": self.failed_workflow_count,
            "problematic_workflow_count": self.problematic_workflow_count,
            "highest_severity": self.highest_severity,
            "problematic_workflows": [
                workflow.to_dict() for workflow in self.problematic_workflows
            ],
        }


def summarize_workflow_input_statuses(
    raw_statuses: Mapping[str, Any] | None,
) -> WorkflowInputStatusSummary:
    """Summarize structured workflow-input status metadata."""

    if not isinstance(raw_statuses, Mapping):
        return WorkflowInputStatusSummary(
            total_count=0,
            ok_count=0,
            degraded_count=0,
            unavailable_count=0,
            failed_count=0,
            issue_count=0,
            healthy=True,
        )

    total_count = 0
    ok_count = 0
    degraded_count = 0
    unavailable_count = 0
    failed_count = 0
    issues: list[WorkflowInputIssue] = []
    issue_codes: list[str] = []
    problematic_inputs: list[str] = []

    for input_name, raw_value in raw_statuses.items():
        if not isinstance(input_name, str) or not isinstance(raw_value, Mapping):
            continue
        total_count += 1
        status = _clean_optional_text(raw_value.get("status")) or "failed"
        if status not in _KNOWN_STATUSES:
            status = "failed"

        if status == "ok":
            ok_count += 1
            continue
        if status == "degraded":
            degraded_count += 1
        elif status == "unavailable":
            unavailable_count += 1
        elif status == "failed":
            failed_count += 1

        problematic_inputs.append(input_name)
        issue_code = _clean_optional_text(raw_value.get("issue_code"))
        if issue_code is not None and issue_code not in issue_codes:
            issue_codes.append(issue_code)
        issues.append(
            WorkflowInputIssue(
                input_name=input_name,
                status=status,
                role_name=_clean_optional_text(raw_value.get("role_name")),
                provider=_clean_optional_text(raw_value.get("provider")),
                issue_code=issue_code,
                message=_clean_optional_text(raw_value.get("message")),
            )
        )

    return WorkflowInputStatusSummary(
        total_count=total_count,
        ok_count=ok_count,
        degraded_count=degraded_count,
        unavailable_count=unavailable_count,
        failed_count=failed_count,
        issue_count=len(issues),
        healthy=not issues,
        problematic_inputs=tuple(problematic_inputs),
        issue_codes=tuple(issue_codes),
        issues=tuple(issues),
    )


def summarize_named_workflow_input_statuses(
    workflow_statuses: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Summarize multiple workflow input-status mappings by workflow name."""

    if not isinstance(workflow_statuses, Mapping):
        return {}
    return {
        workflow_name: summarize_workflow_input_statuses(raw_statuses).to_dict()
        for workflow_name, raw_statuses in workflow_statuses.items()
        if isinstance(workflow_name, str) and isinstance(raw_statuses, Mapping)
    }


def workflow_input_summary_status(summary: Mapping[str, Any] | None) -> str:
    """Return the highest-severity status for one workflow input summary."""

    if not isinstance(summary, Mapping):
        return "ok"
    if int(summary.get("failed_count", 0) or 0) > 0:
        return "failed"
    if int(summary.get("unavailable_count", 0) or 0) > 0:
        return "unavailable"
    if int(summary.get("degraded_count", 0) or 0) > 0:
        return "degraded"
    return "ok"


def build_workflow_input_overview(
    workflow_input_summaries: Mapping[str, Any] | None,
) -> WorkflowInputOverview:
    """Build a compact overview across named workflow summaries."""

    if not isinstance(workflow_input_summaries, Mapping):
        return WorkflowInputOverview(
            workflow_count=0,
            healthy_workflow_count=0,
            degraded_workflow_count=0,
            unavailable_workflow_count=0,
            failed_workflow_count=0,
            problematic_workflow_count=0,
        )

    workflow_count = 0
    healthy_workflow_count = 0
    degraded_workflow_count = 0
    unavailable_workflow_count = 0
    failed_workflow_count = 0
    problematic_workflows: list[WorkflowInputOverviewItem] = []

    for workflow_name, raw_summary in workflow_input_summaries.items():
        if not isinstance(workflow_name, str) or not isinstance(raw_summary, Mapping):
            continue
        workflow_count += 1
        status = workflow_input_summary_status(raw_summary)
        if status == "ok":
            healthy_workflow_count += 1
            continue
        if status == "degraded":
            degraded_workflow_count += 1
        elif status == "unavailable":
            unavailable_workflow_count += 1
        elif status == "failed":
            failed_workflow_count += 1
        problematic_workflows.append(
            WorkflowInputOverviewItem(
                workflow_name=workflow_name,
                status=status,
                issue_count=int(raw_summary.get("issue_count", 0) or 0),
                issue_codes=tuple(
                    str(item)
                    for item in raw_summary.get("issue_codes", ())
                    if str(item).strip()
                ),
                problematic_inputs=tuple(
                    str(item)
                    for item in raw_summary.get("problematic_inputs", ())
                    if str(item).strip()
                ),
            )
        )

    highest_severity: str | None = None
    if failed_workflow_count > 0:
        highest_severity = "failed"
    elif unavailable_workflow_count > 0:
        highest_severity = "unavailable"
    elif degraded_workflow_count > 0:
        highest_severity = "degraded"

    return WorkflowInputOverview(
        workflow_count=workflow_count,
        healthy_workflow_count=healthy_workflow_count,
        degraded_workflow_count=degraded_workflow_count,
        unavailable_workflow_count=unavailable_workflow_count,
        failed_workflow_count=failed_workflow_count,
        problematic_workflow_count=len(problematic_workflows),
        highest_severity=highest_severity,
        problematic_workflows=tuple(problematic_workflows),
    )


def render_workflow_input_summary_lines(
    workflow_input_summaries: Mapping[str, Any] | None,
) -> list[str]:
    """Render a compact operator-facing text section for workflow summaries."""

    overview = build_workflow_input_overview(workflow_input_summaries)
    if overview.workflow_count == 0:
        return []

    lines = [
        "Workflow inputs",
        (
            f"- Workflows={overview.workflow_count} | "
            f"healthy={overview.healthy_workflow_count} | "
            f"degraded={overview.degraded_workflow_count} | "
            f"unavailable={overview.unavailable_workflow_count} | "
            f"failed={overview.failed_workflow_count}"
        ),
    ]
    for workflow in overview.problematic_workflows:
        detail = (
            f"- {workflow.workflow_name}: "
            f"status={workflow.status} | "
            f"issues={workflow.issue_count}"
        )
        if workflow.problematic_inputs:
            detail += f" | inputs={', '.join(workflow.problematic_inputs)}"
        if workflow.issue_codes:
            detail += f" | issue_codes={', '.join(workflow.issue_codes)}"
        lines.append(detail)
    return lines


def build_workflow_input_metadata(
    status_field_name: str,
    raw_statuses: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return report metadata containing raw statuses and a compact summary."""

    return {
        status_field_name: {
            str(key): dict(value)
            for key, value in raw_statuses.items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
        if isinstance(raw_statuses, Mapping)
        else {},
        "workflow_input_summary": summarize_workflow_input_statuses(raw_statuses).to_dict(),
    }
