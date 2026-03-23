"""Unified live market-cycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping

from bot.ingestion.market_data import MarketCycleIngestionEnvelope, MarketDataIngestionUpdate
from bot.reporting.daily_report import (
    DailyResearchSummary,
    IntradayPortfolioReviewReport,
    MarketMonitorReport,
    PortfolioReviewReport,
)
from bot.state.market_state import (
    MarketStateAlertableState,
    MarketStateSnapshot,
    MarketStateTransition,
    MarketStateUpdateResult,
)


@dataclass(frozen=True)
class LiveMarketCycleRequest:
    """Requested stages for one coordinated market cycle."""

    workflow: str
    as_of_date: date
    portfolio_path: str | None = None
    include_daily_summary: bool = False
    include_portfolio_review: bool = False
    include_intraday_review: bool = False
    daily_summary_required: bool = False
    portfolio_review_required: bool = False
    intraday_review_required: bool = False


@dataclass(frozen=True)
class LiveMarketCycleWarning:
    """One non-fatal warning collected during cycle orchestration."""

    stage: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "message": self.message,
        }


@dataclass(frozen=True)
class PersistedMarketStateArtifacts:
    """State persistence outputs for one cycle."""

    update_result: MarketStateUpdateResult
    output_paths: dict[str, str] = field(default_factory=dict)

    @property
    def transition_count(self) -> int:
        return self.update_result.change_set.transition_count

    @property
    def baseline_established(self) -> bool:
        return self.update_result.change_set.baseline_established


@dataclass(frozen=True)
class LiveMarketCycleResult:
    """Structured result for one unified market cycle."""

    workflow: str
    as_of_date: date
    status: str
    daily_summary_result: Mapping[str, object] | None = None
    daily_summary: DailyResearchSummary | None = None
    portfolio_review: PortfolioReviewReport | None = None
    intraday_review: IntradayPortfolioReviewReport | None = None
    monitor_report: MarketMonitorReport | None = None
    market_state_snapshot: MarketStateSnapshot | None = None
    state_transitions: tuple[MarketStateTransition, ...] = ()
    alertable_states: tuple[MarketStateAlertableState, ...] = ()
    ingestion_adapter_name: str | None = None
    ingestion_mode: str | None = None
    ingestion_updates: tuple[MarketDataIngestionUpdate, ...] = ()
    warnings: tuple[LiveMarketCycleWarning, ...] = ()
    paths_written: dict[str, str] = field(default_factory=dict)
    state_artifacts: PersistedMarketStateArtifacts | None = None

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def state_change_count(self) -> int:
        return len(self.state_transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "as_of_date": self.as_of_date.isoformat(),
            "status": self.status,
            "candidate_summary": _candidate_summary(self.daily_summary),
            "portfolio_review_summary": _portfolio_review_summary(self.portfolio_review),
            "intraday_review_summary": _intraday_review_summary(self.intraday_review),
            "has_monitor_report": self.monitor_report is not None,
            "has_market_state_snapshot": self.market_state_snapshot is not None,
            "ingestion_adapter_name": self.ingestion_adapter_name,
            "ingestion_mode": self.ingestion_mode,
            "ingestion_update_count": len(self.ingestion_updates),
            "state_change_count": self.state_change_count,
            "alertable_state_count": len(self.alertable_states),
            "warning_count": self.warning_count,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "paths_written": dict(sorted(self.paths_written.items())),
        }


class LiveMarketRunner:
    """Coordinate one coherent market cycle over the existing workflow components."""

    def __init__(
        self,
        *,
        run_daily_summary: Callable[[], Mapping[str, object]] | None = None,
        run_portfolio_review: Callable[[], PortfolioReviewReport] | None = None,
        run_intraday_review: Callable[[], IntradayPortfolioReviewReport] | None = None,
        build_monitor_report: Callable[
            [
                date,
                DailyResearchSummary | None,
                PortfolioReviewReport | None,
                IntradayPortfolioReviewReport | None,
                str | None,
            ],
            MarketMonitorReport,
        ]
        | None = None,
        persist_market_state: Callable[
            [date, str, DailyResearchSummary | None, PortfolioReviewReport | None, IntradayPortfolioReviewReport | None, str | None],
            PersistedMarketStateArtifacts,
        ]
        | None = None,
    ) -> None:
        self._run_daily_summary = run_daily_summary
        self._run_portfolio_review = run_portfolio_review
        self._run_intraday_review = run_intraday_review
        self._build_monitor_report = build_monitor_report
        self._persist_market_state = persist_market_state

    def run_cycle(
        self,
        request: LiveMarketCycleRequest,
        *,
        ingestion: MarketCycleIngestionEnvelope | None = None,
    ) -> LiveMarketCycleResult:
        """Run the requested market-cycle stages in orchestration order."""

        _validate_request(request)
        _validate_ingestion(request, ingestion)
        warnings: list[LiveMarketCycleWarning] = []
        daily_summary_result: Mapping[str, object] | None = None
        daily_summary: DailyResearchSummary | None = None
        portfolio_review: PortfolioReviewReport | None = None
        intraday_review: IntradayPortfolioReviewReport | None = None
        monitor_report: MarketMonitorReport | None = None
        state_artifacts: PersistedMarketStateArtifacts | None = None
        resolved_portfolio_path = (
            request.portfolio_path
            if request.portfolio_path is not None
            else (ingestion.portfolio_path if ingestion is not None else None)
        )
        run_daily_summary = (
            ingestion.run_daily_summary
            if ingestion is not None and ingestion.run_daily_summary is not None
            else self._run_daily_summary
        )
        run_portfolio_review = (
            ingestion.run_portfolio_review
            if ingestion is not None and ingestion.run_portfolio_review is not None
            else self._run_portfolio_review
        )
        run_intraday_review = (
            ingestion.run_intraday_review
            if ingestion is not None and ingestion.run_intraday_review is not None
            else self._run_intraday_review
        )

        if request.include_daily_summary:
            daily_summary_result = self._run_stage(
                stage_name="daily_summary",
                required=request.daily_summary_required,
                callback=run_daily_summary,
                warnings=warnings,
            )
            if daily_summary_result is not None:
                summary_object = daily_summary_result.get("summary")
                if not isinstance(summary_object, DailyResearchSummary):
                    raise TypeError("daily_summary workflow must return a mapping with a DailyResearchSummary at 'summary'.")
                daily_summary = summary_object

        if request.include_portfolio_review:
            portfolio_review = self._run_stage(
                stage_name="portfolio_review",
                required=request.portfolio_review_required,
                callback=run_portfolio_review,
                warnings=warnings,
            )
            if portfolio_review is not None and not isinstance(
                portfolio_review,
                PortfolioReviewReport,
            ):
                raise TypeError(
                    "portfolio_review workflow must return a PortfolioReviewReport."
                )

        if request.include_intraday_review:
            intraday_review = self._run_stage(
                stage_name="intraday_review",
                required=request.intraday_review_required,
                callback=run_intraday_review,
                warnings=warnings,
            )
            if intraday_review is not None and not isinstance(
                intraday_review,
                IntradayPortfolioReviewReport,
            ):
                raise TypeError(
                    "intraday_review workflow must return an IntradayPortfolioReviewReport."
                )

        if (
            self._build_monitor_report is not None
            and (
                daily_summary is not None
                or portfolio_review is not None
                or intraday_review is not None
            )
        ):
            monitor_report = self._build_monitor_report(
                request.as_of_date,
                daily_summary,
                portfolio_review,
                intraday_review,
                resolved_portfolio_path,
            )

        if (
            self._persist_market_state is not None
            and (
                daily_summary is not None
                or portfolio_review is not None
                or intraday_review is not None
            )
        ):
            state_artifacts = self._persist_market_state(
                request.as_of_date,
                request.workflow,
                daily_summary,
                portfolio_review,
                intraday_review,
                resolved_portfolio_path,
            )

        state_transitions = (
            tuple(state_artifacts.update_result.change_set.transitions)
            if state_artifacts is not None
            else ()
        )
        return LiveMarketCycleResult(
            workflow=request.workflow,
            as_of_date=request.as_of_date,
            status="partial" if warnings else "ok",
            daily_summary_result=daily_summary_result,
            daily_summary=daily_summary,
            portfolio_review=portfolio_review,
            intraday_review=intraday_review,
            monitor_report=monitor_report,
            market_state_snapshot=(
                state_artifacts.update_result.current_snapshot
                if state_artifacts is not None
                else None
            ),
            state_transitions=state_transitions,
            alertable_states=(
                tuple(state_artifacts.update_result.current_snapshot.current_alertable_states)
                if state_artifacts is not None
                else ()
            ),
            ingestion_adapter_name=(
                ingestion.adapter_name if ingestion is not None else None
            ),
            ingestion_mode=(
                ingestion.ingestion_mode if ingestion is not None else None
            ),
            ingestion_updates=(
                tuple(ingestion.updates) if ingestion is not None else ()
            ),
            warnings=tuple(warnings),
            paths_written=(
                dict(state_artifacts.output_paths)
                if state_artifacts is not None
                else {}
            ),
            state_artifacts=state_artifacts,
        )

    def _run_stage(
        self,
        *,
        stage_name: str,
        required: bool,
        callback: Callable[[], Any] | None,
        warnings: list[LiveMarketCycleWarning],
    ) -> Any:
        if callback is None:
            if required:
                raise ValueError(f"{stage_name} stage is required but no callback was provided.")
            return None
        try:
            return callback()
        except Exception as exc:
            if required:
                raise
            warnings.append(LiveMarketCycleWarning(stage=stage_name, message=str(exc)))
            return None


def run_market_cycle(
    runner: LiveMarketRunner,
    request: LiveMarketCycleRequest,
    *,
    ingestion: MarketCycleIngestionEnvelope | None = None,
) -> LiveMarketCycleResult:
    """Convenience wrapper around ``LiveMarketRunner.run_cycle``."""

    return runner.run_cycle(request, ingestion=ingestion)


def _validate_request(request: LiveMarketCycleRequest) -> None:
    if request.daily_summary_required and not request.include_daily_summary:
        raise ValueError(
            "daily_summary_required=True requires include_daily_summary=True."
        )
    if request.portfolio_review_required and not request.include_portfolio_review:
        raise ValueError(
            "portfolio_review_required=True requires include_portfolio_review=True."
        )
    if request.intraday_review_required and not request.include_intraday_review:
        raise ValueError(
            "intraday_review_required=True requires include_intraday_review=True."
        )


def _validate_ingestion(
    request: LiveMarketCycleRequest,
    ingestion: MarketCycleIngestionEnvelope | None,
) -> None:
    if ingestion is None:
        return
    if not isinstance(ingestion, MarketCycleIngestionEnvelope):
        raise TypeError("ingestion must be a MarketCycleIngestionEnvelope.")
    if ingestion.as_of_date != request.as_of_date:
        raise ValueError("ingestion.as_of_date must match request.as_of_date.")
    if (
        request.portfolio_path is not None
        and ingestion.portfolio_path is not None
        and request.portfolio_path != ingestion.portfolio_path
    ):
        raise ValueError("ingestion.portfolio_path must match request.portfolio_path.")
    update_types = {update.update_type for update in ingestion.updates}
    if (
        request.include_daily_summary
        and "candidate_universe_refresh_batch" not in update_types
    ):
        raise ValueError(
            "ingestion.updates must include 'candidate_universe_refresh_batch' when include_daily_summary=True."
        )
    if (
        request.include_portfolio_review
        and "portfolio_symbol_refresh_batch" not in update_types
    ):
        raise ValueError(
            "ingestion.updates must include 'portfolio_symbol_refresh_batch' when include_portfolio_review=True."
        )
    if (
        request.include_intraday_review
        and "portfolio_symbol_refresh_batch" not in update_types
    ):
        raise ValueError(
            "ingestion.updates must include 'portfolio_symbol_refresh_batch' when include_intraday_review=True."
        )


def _candidate_summary(summary: DailyResearchSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    payload = summary.to_dict()
    return {
        "approved_count": payload["approved_count"],
        "rejected_count": payload["rejected_count"],
        "order_count": payload["order_count"],
        "recommended_preset": payload["recommended_preset"],
    }


def _portfolio_review_summary(report: PortfolioReviewReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    payload = report.to_dict()
    return {
        "position_count": payload["position_count"],
        "reviewed_symbol_count": payload["reviewed_symbol_count"],
        "hold_count": payload["hold_count"],
        "watch_closely_count": payload["watch_closely_count"],
        "exit_candidate_count": payload["exit_candidate_count"],
        "raise_stop_count": payload["raise_stop_count"],
    }


def _intraday_review_summary(
    report: IntradayPortfolioReviewReport | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    payload = report.to_dict()
    return {
        "position_count": payload["position_count"],
        "reviewed_symbol_count": payload["reviewed_symbol_count"],
        "interval_minutes": payload["interval_minutes"],
        "hold_count": payload["hold_count"],
        "watch_closely_count": payload["watch_closely_count"],
        "exit_candidate_count": payload["exit_candidate_count"],
        "raise_stop_count": payload["raise_stop_count"],
    }
