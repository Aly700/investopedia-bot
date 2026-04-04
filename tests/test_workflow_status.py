from bot.workflow_status import (
    build_workflow_input_overview,
    build_workflow_input_metadata,
    render_workflow_input_summary_lines,
    summarize_named_workflow_input_statuses,
    summarize_workflow_input_statuses,
)


def test_summarize_workflow_input_statuses_counts_problem_inputs() -> None:
    summary = summarize_workflow_input_statuses(
        {
            "benchmark": {
                "status": "ok",
                "role_name": "historical_bars",
                "provider": "alpaca",
            },
            "volatility_context": {
                "status": "unavailable",
                "role_name": "historical_bars",
                "provider": "alpaca",
                "issue_code": "unsupported_capability",
                "message": "VIX symbols are not supported.",
            },
            "earnings_contexts": {
                "status": "degraded",
                "role_name": "earnings_calendar",
                "provider": "polygon",
                "issue_code": "entitlement_limited",
                "message": "Not entitled.",
            },
        }
    ).to_dict()

    assert summary["total_count"] == 3
    assert summary["ok_count"] == 1
    assert summary["degraded_count"] == 1
    assert summary["unavailable_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["healthy"] is False
    assert summary["problematic_inputs"] == ["volatility_context", "earnings_contexts"]
    assert summary["issue_codes"] == ["unsupported_capability", "entitlement_limited"]
    assert summary["issues"][0]["input_name"] == "volatility_context"


def test_build_workflow_input_metadata_includes_raw_statuses_and_summary() -> None:
    metadata = build_workflow_input_metadata(
        "review_input_statuses",
        {
            "benchmark": {"status": "ok"},
            "position_daily_symbol_frames": {
                "status": "degraded",
                "issue_code": "partial_symbol_frames",
            },
        },
    )

    assert metadata["review_input_statuses"]["benchmark"]["status"] == "ok"
    assert metadata["workflow_input_summary"]["degraded_count"] == 1
    assert metadata["workflow_input_summary"]["problematic_inputs"] == [
        "position_daily_symbol_frames"
    ]


def test_summarize_named_workflow_input_statuses_reuses_shared_pattern() -> None:
    summaries = summarize_named_workflow_input_statuses(
        {
            "daily_summary": {
                "benchmark": {"status": "ok"},
                "volatility_context": {"status": "unavailable", "issue_code": "empty_data"},
            },
            "portfolio_review": {
                "benchmark": {"status": "ok"},
            },
        }
    )

    assert summaries["daily_summary"]["unavailable_count"] == 1
    assert summaries["portfolio_review"]["healthy"] is True


def test_build_workflow_input_overview_classifies_problematic_workflows() -> None:
    overview = build_workflow_input_overview(
        {
            "daily_summary": {
                "ok_count": 1,
                "degraded_count": 0,
                "unavailable_count": 1,
                "failed_count": 0,
                "issue_count": 1,
                "issue_codes": ["unsupported_capability"],
                "problematic_inputs": ["volatility_context"],
            },
            "portfolio_review": {
                "ok_count": 2,
                "degraded_count": 1,
                "unavailable_count": 0,
                "failed_count": 0,
                "issue_count": 1,
                "issue_codes": ["partial_symbol_frames"],
                "problematic_inputs": ["position_daily_symbol_frames"],
            },
            "intraday_review": {
                "ok_count": 2,
                "degraded_count": 0,
                "unavailable_count": 0,
                "failed_count": 0,
                "issue_count": 0,
                "issue_codes": [],
                "problematic_inputs": [],
            },
        }
    ).to_dict()

    assert overview["workflow_count"] == 3
    assert overview["healthy_workflow_count"] == 1
    assert overview["problematic_workflow_count"] == 2
    assert overview["highest_severity"] == "unavailable"
    assert overview["problematic_workflows"][0]["workflow_name"] == "daily_summary"


def test_render_workflow_input_summary_lines_reuses_overview_shape() -> None:
    lines = render_workflow_input_summary_lines(
        {
            "daily_summary": {
                "ok_count": 1,
                "degraded_count": 0,
                "unavailable_count": 1,
                "failed_count": 0,
                "issue_count": 1,
                "issue_codes": ["unsupported_capability"],
                "problematic_inputs": ["volatility_context"],
            }
        }
    )

    assert lines[0] == "Workflow inputs"
    assert "unavailable=1" in lines[1]
    assert "inputs=volatility_context" in lines[2]
