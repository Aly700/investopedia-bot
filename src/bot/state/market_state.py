"""Typed live market state snapshots, persistence, and transition diffing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from datetime import datetime, timezone
from logging import Logger
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    coerce_positive_int,
    load_json_mapping_file,
    write_json_file,
)
from bot.features import MarketContext
from bot.reporting.daily_report import (
    DailyResearchSummary,
    IntradayPortfolioReviewReport,
    PortfolioReviewReport,
    PortfolioReviewRow,
)


LOGGER = logging.getLogger(__name__)

MARKET_STATE_SNAPSHOT_SCHEMA_VERSION = 1
MARKET_STATE_TRANSITION_TYPES = (
    "HOLD_TO_WATCH_CLOSELY",
    "WATCH_CLOSELY_TO_EXIT_CANDIDATE",
    "POSITION_ACTION_ESCALATED",
    "POSITION_RISK_EASED",
    "TOP_PRIORITY_CANDIDATE_ENTERED",
    "TOP_PRIORITY_CANDIDATE_EXITED",
    "BREADTH_REGIME_CHANGED",
    "VOLATILITY_REGIME_CHANGED",
    "SECTOR_BLOCK_ENTERED",
    "SECTOR_BLOCK_CLEARED",
)
MARKET_STATE_ALERTABLE_CATEGORIES = (
    "EXIT CANDIDATE",
    "WATCH CLOSELY",
    "RAISE STOP",
    "BUY CANDIDATE",
    "SECTOR BLOCK",
)
_PORTFOLIO_ACTION_SEVERITY = {
    "HOLD": 0,
    "RAISE STOP": 1,
    "WATCH CLOSELY": 2,
    "EXIT CANDIDATE": 3,
}
_CURRENT_MARKET_STATE_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "current_market_state.json",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "current_market_state.json"),
    ),
)
_PREVIOUS_MARKET_STATE_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "snapshots",
        "previous_market_state.json",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "previous_market_state.json"),
    ),
)


def default_current_market_state_path(project_root: Path) -> Path:
    """Return the preferred resolved path for the current market snapshot."""

    return _CURRENT_MARKET_STATE_PATH.preferred_path(project_root).resolve()


def default_previous_market_state_path(project_root: Path) -> Path:
    """Return the preferred resolved path for the previous market snapshot."""

    return _PREVIOUS_MARKET_STATE_PATH.preferred_path(project_root).resolve()


@dataclass(frozen=True)
class MarketStateBenchmarkContext:
    """Compact benchmark/regime state for the live snapshot."""

    benchmark_symbol: str | None = None
    regime_filter_mode: str | None = None
    regime_passed: bool | None = None
    benchmark_return: float | None = None
    benchmark_intraday_return_vs_open: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "regime_filter_mode": self.regime_filter_mode,
            "regime_passed": self.regime_passed,
            "benchmark_return": self.benchmark_return,
            "benchmark_intraday_return_vs_open": self.benchmark_intraday_return_vs_open,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateBenchmarkContext:
        return cls(
            benchmark_symbol=_optional_text(payload.get("benchmark_symbol")),
            regime_filter_mode=_optional_text(payload.get("regime_filter_mode")),
            regime_passed=_optional_bool(payload.get("regime_passed")),
            benchmark_return=_optional_float(payload.get("benchmark_return")),
            benchmark_intraday_return_vs_open=_optional_float(
                payload.get("benchmark_intraday_return_vs_open")
            ),
        )


@dataclass(frozen=True)
class MarketStateBreadthContext:
    """Compact breadth state for the live snapshot."""

    market_breadth_pct_above_200ma: float | None = None
    market_breadth_pct_above_50ma: float | None = None
    market_breadth_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_breadth_pct_above_200ma": self.market_breadth_pct_above_200ma,
            "market_breadth_pct_above_50ma": self.market_breadth_pct_above_50ma,
            "market_breadth_state": self.market_breadth_state,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateBreadthContext:
        return cls(
            market_breadth_pct_above_200ma=_optional_float(
                payload.get("market_breadth_pct_above_200ma")
            ),
            market_breadth_pct_above_50ma=_optional_float(
                payload.get("market_breadth_pct_above_50ma")
            ),
            market_breadth_state=_optional_text(payload.get("market_breadth_state")),
        )


@dataclass(frozen=True)
class MarketStateVolatilityContext:
    """Compact volatility state for the live snapshot."""

    vix_close: float | None = None
    vix_sma_short: float | None = None
    vix_sma_long: float | None = None
    volatility_regime_state: str | None = None
    volatility_regime_risk_off: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "vix_close": self.vix_close,
            "vix_sma_short": self.vix_sma_short,
            "vix_sma_long": self.vix_sma_long,
            "volatility_regime_state": self.volatility_regime_state,
            "volatility_regime_risk_off": self.volatility_regime_risk_off,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateVolatilityContext:
        return cls(
            vix_close=_optional_float(payload.get("vix_close")),
            vix_sma_short=_optional_float(payload.get("vix_sma_short")),
            vix_sma_long=_optional_float(payload.get("vix_sma_long")),
            volatility_regime_state=_optional_text(payload.get("volatility_regime_state")),
            volatility_regime_risk_off=_optional_bool(payload.get("volatility_regime_risk_off")),
        )


@dataclass(frozen=True)
class MarketStateSectorSummary:
    """Compact sector-level entry state derived from current candidates."""

    sector_name: str
    sector_etf_symbol: str | None = None
    approved_candidate_count: int = 0
    actionable_candidate_count: int = 0
    blocked_candidate_count: int = 0
    blocked_by_portfolio_heat: bool = False
    top_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sector_name.strip():
            raise ValueError("sector_name cannot be empty.")
        if self.approved_candidate_count < 0:
            raise ValueError("approved_candidate_count cannot be negative.")
        if self.actionable_candidate_count < 0:
            raise ValueError("actionable_candidate_count cannot be negative.")
        if self.blocked_candidate_count < 0:
            raise ValueError("blocked_candidate_count cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_name": self.sector_name,
            "sector_etf_symbol": self.sector_etf_symbol,
            "approved_candidate_count": self.approved_candidate_count,
            "actionable_candidate_count": self.actionable_candidate_count,
            "blocked_candidate_count": self.blocked_candidate_count,
            "blocked_by_portfolio_heat": self.blocked_by_portfolio_heat,
            "top_symbols": list(self.top_symbols),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateSectorSummary:
        sector_name = _required_text(payload.get("sector_name"), "sector_name")
        return cls(
            sector_name=sector_name,
            sector_etf_symbol=_optional_text(payload.get("sector_etf_symbol")),
            approved_candidate_count=_non_negative_int(
                payload.get("approved_candidate_count"),
                "approved_candidate_count",
            ),
            actionable_candidate_count=_non_negative_int(
                payload.get("actionable_candidate_count"),
                "actionable_candidate_count",
            ),
            blocked_candidate_count=_non_negative_int(
                payload.get("blocked_candidate_count"),
                "blocked_candidate_count",
            ),
            blocked_by_portfolio_heat=_required_bool(
                payload.get("blocked_by_portfolio_heat"),
                "blocked_by_portfolio_heat",
            ),
            top_symbols=_string_tuple(payload.get("top_symbols"), "top_symbols"),
        )


@dataclass(frozen=True)
class MarketStateCandidateState:
    """Compact ranked candidate queue state for live monitoring."""

    symbol: str
    rank: int
    preset_name: str | None = None
    status: str = "approved"
    score: float | None = None
    actionable_now: bool = False
    priority_bucket: str | None = None
    entry_price_hint: float | None = None
    stop_level: float | None = None
    sector_name: str | None = None
    blocked_by_portfolio_heat: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "preset_name": self.preset_name,
            "status": self.status,
            "score": self.score,
            "actionable_now": self.actionable_now,
            "priority_bucket": self.priority_bucket,
            "entry_price_hint": self.entry_price_hint,
            "stop_level": self.stop_level,
            "sector_name": self.sector_name,
            "blocked_by_portfolio_heat": self.blocked_by_portfolio_heat,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateCandidateState:
        return cls(
            symbol=_required_text(payload.get("symbol"), "symbol"),
            rank=coerce_positive_int(payload.get("rank"), "rank"),
            preset_name=_optional_text(payload.get("preset_name")),
            status=_required_text(payload.get("status"), "status"),
            score=_optional_float(payload.get("score")),
            actionable_now=_required_bool(payload.get("actionable_now"), "actionable_now"),
            priority_bucket=_optional_text(payload.get("priority_bucket")),
            entry_price_hint=_optional_float(payload.get("entry_price_hint")),
            stop_level=_optional_float(payload.get("stop_level")),
            sector_name=_optional_text(payload.get("sector_name")),
            blocked_by_portfolio_heat=_required_bool(
                payload.get("blocked_by_portfolio_heat"),
                "blocked_by_portfolio_heat",
            ),
        )


@dataclass(frozen=True)
class MarketStatePortfolioSummary:
    """Compact daily held-position review summary."""

    position_count: int
    hold_count: int
    watch_closely_count: int
    exit_candidate_count: int
    raise_stop_count: int
    urgent_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "position_count",
            "hold_count",
            "watch_closely_count",
            "exit_candidate_count",
            "raise_stop_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_count": self.position_count,
            "hold_count": self.hold_count,
            "watch_closely_count": self.watch_closely_count,
            "exit_candidate_count": self.exit_candidate_count,
            "raise_stop_count": self.raise_stop_count,
            "urgent_symbols": list(self.urgent_symbols),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStatePortfolioSummary:
        return cls(
            position_count=_non_negative_int(payload.get("position_count"), "position_count"),
            hold_count=_non_negative_int(payload.get("hold_count"), "hold_count"),
            watch_closely_count=_non_negative_int(
                payload.get("watch_closely_count"),
                "watch_closely_count",
            ),
            exit_candidate_count=_non_negative_int(
                payload.get("exit_candidate_count"),
                "exit_candidate_count",
            ),
            raise_stop_count=_non_negative_int(
                payload.get("raise_stop_count"),
                "raise_stop_count",
            ),
            urgent_symbols=_string_tuple(payload.get("urgent_symbols"), "urgent_symbols"),
        )


@dataclass(frozen=True)
class MarketStateIntradaySummary:
    """Compact intraday held-position review summary."""

    position_count: int
    interval_minutes: int
    hold_count: int
    watch_closely_count: int
    exit_candidate_count: int
    raise_stop_count: int
    urgent_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.position_count < 0:
            raise ValueError("position_count cannot be negative.")
        if self.hold_count < 0:
            raise ValueError("hold_count cannot be negative.")
        if self.watch_closely_count < 0:
            raise ValueError("watch_closely_count cannot be negative.")
        if self.exit_candidate_count < 0:
            raise ValueError("exit_candidate_count cannot be negative.")
        if self.raise_stop_count < 0:
            raise ValueError("raise_stop_count cannot be negative.")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_count": self.position_count,
            "interval_minutes": self.interval_minutes,
            "hold_count": self.hold_count,
            "watch_closely_count": self.watch_closely_count,
            "exit_candidate_count": self.exit_candidate_count,
            "raise_stop_count": self.raise_stop_count,
            "urgent_symbols": list(self.urgent_symbols),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateIntradaySummary:
        return cls(
            position_count=_non_negative_int(payload.get("position_count"), "position_count"),
            interval_minutes=coerce_positive_int(
                payload.get("interval_minutes"),
                "interval_minutes",
            ),
            hold_count=_non_negative_int(payload.get("hold_count"), "hold_count"),
            watch_closely_count=_non_negative_int(
                payload.get("watch_closely_count"),
                "watch_closely_count",
            ),
            exit_candidate_count=_non_negative_int(
                payload.get("exit_candidate_count"),
                "exit_candidate_count",
            ),
            raise_stop_count=_non_negative_int(
                payload.get("raise_stop_count"),
                "raise_stop_count",
            ),
            urgent_symbols=_string_tuple(payload.get("urgent_symbols"), "urgent_symbols"),
        )


@dataclass(frozen=True)
class MarketStateActionState:
    """Current actionable state for one symbol."""

    symbol: str
    action: str
    source_workflow: str
    preset_name: str | None = None
    latest_close: float | None = None
    current_stop: float | None = None
    suggested_stop: float | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if not self.action.strip():
            raise ValueError("action cannot be empty.")
        if not self.source_workflow.strip():
            raise ValueError("source_workflow cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "source_workflow": self.source_workflow,
            "preset_name": self.preset_name,
            "latest_close": self.latest_close,
            "current_stop": self.current_stop,
            "suggested_stop": self.suggested_stop,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateActionState:
        return cls(
            symbol=_required_text(payload.get("symbol"), "symbol"),
            action=_required_text(payload.get("action"), "action"),
            source_workflow=_required_text(payload.get("source_workflow"), "source_workflow"),
            preset_name=_optional_text(payload.get("preset_name")),
            latest_close=_optional_float(payload.get("latest_close")),
            current_stop=_optional_float(payload.get("current_stop")),
            suggested_stop=_optional_float(payload.get("suggested_stop")),
            rationale=_optional_text(payload.get("rationale")),
        )


@dataclass(frozen=True)
class MarketStateAlertableState:
    """Compact current alertable state for notifications and dashboards."""

    state_key: str
    category: str
    source_workflow: str
    symbol: str | None = None
    sector_name: str | None = None
    priority_bucket: str | None = None
    action: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not self.state_key.strip():
            raise ValueError("state_key cannot be empty.")
        if self.category not in MARKET_STATE_ALERTABLE_CATEGORIES:
            raise ValueError(
                "category must be one of "
                f"{MARKET_STATE_ALERTABLE_CATEGORIES}, got '{self.category}'."
            )
        if not self.source_workflow.strip():
            raise ValueError("source_workflow cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "category": self.category,
            "source_workflow": self.source_workflow,
            "symbol": self.symbol,
            "sector_name": self.sector_name,
            "priority_bucket": self.priority_bucket,
            "action": self.action,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateAlertableState:
        return cls(
            state_key=_required_text(payload.get("state_key"), "state_key"),
            category=_required_text(payload.get("category"), "category"),
            source_workflow=_required_text(payload.get("source_workflow"), "source_workflow"),
            symbol=_optional_text(payload.get("symbol")),
            sector_name=_optional_text(payload.get("sector_name")),
            priority_bucket=_optional_text(payload.get("priority_bucket")),
            action=_optional_text(payload.get("action")),
            rationale=_optional_text(payload.get("rationale")),
        )


@dataclass(frozen=True)
class MarketStateRejectedReasonSummary:
    """Compact aggregate of current rejection reasons."""

    reason: str
    count: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason cannot be empty.")
        if self.count <= 0:
            raise ValueError("count must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "count": self.count,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateRejectedReasonSummary:
        return cls(
            reason=_required_text(payload.get("reason"), "reason"),
            count=coerce_positive_int(payload.get("count"), "count"),
        )


@dataclass(frozen=True)
class MarketStateSnapshot:
    """Persisted compact view of what is true now."""

    schema_version: int
    as_of_timestamp: str
    as_of_date: date
    portfolio_path: str | None = None
    source_workflows: tuple[str, ...] = ()
    benchmark_context: MarketStateBenchmarkContext | None = None
    breadth_context: MarketStateBreadthContext | None = None
    volatility_context: MarketStateVolatilityContext | None = None
    sector_context_summary: tuple[MarketStateSectorSummary, ...] = ()
    approved_candidate_queue: tuple[MarketStateCandidateState, ...] = ()
    portfolio_review_summary: MarketStatePortfolioSummary | None = None
    intraday_review_summary: MarketStateIntradaySummary | None = None
    current_alertable_states: tuple[MarketStateAlertableState, ...] = ()
    current_action_states_by_symbol: dict[str, MarketStateActionState] = field(
        default_factory=dict
    )
    top_rejected_reasons_summary: tuple[MarketStateRejectedReasonSummary, ...] = ()

    def __post_init__(self) -> None:
        coerce_positive_int(self.schema_version, "schema_version")
        coerce_iso8601_datetime_text(self.as_of_timestamp, "as_of_timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "as_of_timestamp": self.as_of_timestamp,
            "as_of_date": self.as_of_date.isoformat(),
            "portfolio_path": self.portfolio_path,
            "source_workflows": list(self.source_workflows),
            "benchmark_context": (
                self.benchmark_context.to_dict()
                if self.benchmark_context is not None
                else None
            ),
            "breadth_context": (
                self.breadth_context.to_dict()
                if self.breadth_context is not None
                else None
            ),
            "volatility_context": (
                self.volatility_context.to_dict()
                if self.volatility_context is not None
                else None
            ),
            "sector_context_summary": [
                sector.to_dict() for sector in self.sector_context_summary
            ],
            "approved_candidate_queue": [
                candidate.to_dict() for candidate in self.approved_candidate_queue
            ],
            "portfolio_review_summary": (
                self.portfolio_review_summary.to_dict()
                if self.portfolio_review_summary is not None
                else None
            ),
            "intraday_review_summary": (
                self.intraday_review_summary.to_dict()
                if self.intraday_review_summary is not None
                else None
            ),
            "current_alertable_states": [
                state.to_dict() for state in self.current_alertable_states
            ],
            "current_action_states_by_symbol": {
                symbol: state.to_dict()
                for symbol, state in sorted(
                    self.current_action_states_by_symbol.items(),
                    key=lambda item: item[0],
                )
            },
            "top_rejected_reasons_summary": [
                reason.to_dict() for reason in self.top_rejected_reasons_summary
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> MarketStateSnapshot:
        source_name = str(source_path) if source_path is not None else "market_state"
        as_of_date = _require_iso_date(payload.get("as_of_date"), "as_of_date")
        source_workflows = _string_tuple(payload.get("source_workflows"), "source_workflows")

        benchmark_context = _optional_mapping(payload.get("benchmark_context"))
        breadth_context = _optional_mapping(payload.get("breadth_context"))
        volatility_context = _optional_mapping(payload.get("volatility_context"))
        portfolio_review_summary = _optional_mapping(payload.get("portfolio_review_summary"))
        intraday_review_summary = _optional_mapping(payload.get("intraday_review_summary"))
        current_action_states = _mapping_or_empty(
            payload.get("current_action_states_by_symbol"),
            "current_action_states_by_symbol",
        )

        return cls(
            schema_version=coerce_positive_int(
                payload.get("schema_version"),
                "schema_version",
            ),
            as_of_timestamp=coerce_iso8601_datetime_text(
                payload.get("as_of_timestamp"),
                "as_of_timestamp",
            ),
            as_of_date=as_of_date,
            portfolio_path=_optional_text(payload.get("portfolio_path")),
            source_workflows=source_workflows,
            benchmark_context=(
                MarketStateBenchmarkContext.from_mapping(benchmark_context)
                if benchmark_context is not None
                else None
            ),
            breadth_context=(
                MarketStateBreadthContext.from_mapping(breadth_context)
                if breadth_context is not None
                else None
            ),
            volatility_context=(
                MarketStateVolatilityContext.from_mapping(volatility_context)
                if volatility_context is not None
                else None
            ),
            sector_context_summary=tuple(
                MarketStateSectorSummary.from_mapping(item)
                for item in _sequence_of_mappings(
                    payload.get("sector_context_summary"),
                    f"{source_name}:sector_context_summary",
                )
            ),
            approved_candidate_queue=tuple(
                MarketStateCandidateState.from_mapping(item)
                for item in _sequence_of_mappings(
                    payload.get("approved_candidate_queue"),
                    f"{source_name}:approved_candidate_queue",
                )
            ),
            portfolio_review_summary=(
                MarketStatePortfolioSummary.from_mapping(portfolio_review_summary)
                if portfolio_review_summary is not None
                else None
            ),
            intraday_review_summary=(
                MarketStateIntradaySummary.from_mapping(intraday_review_summary)
                if intraday_review_summary is not None
                else None
            ),
            current_alertable_states=tuple(
                MarketStateAlertableState.from_mapping(item)
                for item in _sequence_of_mappings(
                    payload.get("current_alertable_states"),
                    f"{source_name}:current_alertable_states",
                )
            ),
            current_action_states_by_symbol={
                symbol: MarketStateActionState.from_mapping(_required_mapping(item, symbol))
                for symbol, item in sorted(current_action_states.items(), key=lambda entry: entry[0])
            },
            top_rejected_reasons_summary=tuple(
                MarketStateRejectedReasonSummary.from_mapping(item)
                for item in _sequence_of_mappings(
                    payload.get("top_rejected_reasons_summary"),
                    f"{source_name}:top_rejected_reasons_summary",
                )
            ),
        )


@dataclass(frozen=True)
class MarketStateTransition:
    """One meaningful change between the previous and current snapshot."""

    transition_type: str
    message: str
    symbol: str | None = None
    sector_name: str | None = None
    previous_value: str | None = None
    current_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.transition_type not in MARKET_STATE_TRANSITION_TYPES:
            raise ValueError(
                "transition_type must be one of "
                f"{MARKET_STATE_TRANSITION_TYPES}, got '{self.transition_type}'."
            )
        if not self.message.strip():
            raise ValueError("message cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_type": self.transition_type,
            "message": self.message,
            "symbol": self.symbol,
            "sector_name": self.sector_name,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateTransition:
        metadata = _mapping_or_empty(payload.get("metadata"), "metadata")
        return cls(
            transition_type=_required_text(payload.get("transition_type"), "transition_type"),
            message=_required_text(payload.get("message"), "message"),
            symbol=_optional_text(payload.get("symbol")),
            sector_name=_optional_text(payload.get("sector_name")),
            previous_value=_optional_text(payload.get("previous_value")),
            current_value=_optional_text(payload.get("current_value")),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class MarketStateChangeSet:
    """Compact transition set emitted for the current poll."""

    as_of_timestamp: str
    as_of_date: date
    baseline_established: bool
    transitions: tuple[MarketStateTransition, ...] = ()

    def __post_init__(self) -> None:
        coerce_iso8601_datetime_text(self.as_of_timestamp, "as_of_timestamp")

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_timestamp": self.as_of_timestamp,
            "as_of_date": self.as_of_date.isoformat(),
            "baseline_established": self.baseline_established,
            "transition_count": self.transition_count,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def to_text(self) -> str:
        lines = [f"State changes for {self.as_of_date.isoformat()}"]
        if self.baseline_established:
            lines.append("Baseline established. No prior market state snapshot was available.")
            return "\n".join(lines).rstrip() + "\n"
        if not self.transitions:
            lines.append("No meaningful state changes detected.")
            return "\n".join(lines).rstrip() + "\n"
        for transition in self.transitions:
            details: list[str] = [transition.transition_type]
            if transition.symbol is not None:
                details.append(transition.symbol)
            if transition.sector_name is not None:
                details.append(transition.sector_name)
            if transition.previous_value is not None or transition.current_value is not None:
                details.append(
                    f"{transition.previous_value or 'n/a'} -> {transition.current_value or 'n/a'}"
                )
            lines.append("- " + " | ".join(details))
            lines.append(f"  {transition.message}")
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MarketStateChangeSet:
        return cls(
            as_of_timestamp=coerce_iso8601_datetime_text(
                payload.get("as_of_timestamp"),
                "as_of_timestamp",
            ),
            as_of_date=_require_iso_date(payload.get("as_of_date"), "as_of_date"),
            baseline_established=_required_bool(
                payload.get("baseline_established"),
                "baseline_established",
            ),
            transitions=tuple(
                MarketStateTransition.from_mapping(item)
                for item in _sequence_of_mappings(payload.get("transitions"), "transitions")
            ),
        )


@dataclass(frozen=True)
class MarketStateUpdateResult:
    """Result of updating the persisted market state snapshot."""

    current_snapshot: MarketStateSnapshot
    previous_snapshot: MarketStateSnapshot | None
    change_set: MarketStateChangeSet
    current_path: Path
    previous_path: Path


def load_market_state_snapshot(
    path: Path,
    *,
    logger: Logger | None = None,
) -> MarketStateSnapshot | None:
    """Load a market state snapshot, returning ``None`` on failure."""

    effective_logger = logger or LOGGER
    return load_json_mapping_file(
        path,
        artifact_label="market state snapshot",
        logger=effective_logger,
        empty_value=lambda: None,
        build=lambda payload, source_path: MarketStateSnapshot.from_mapping(
            payload,
            source_path=source_path,
        ),
    )


class MarketStateStore:
    """Load and update the current and previous live market state snapshots."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.current_path = _CURRENT_MARKET_STATE_PATH.resolve(project_root)
        self.previous_path = _PREVIOUS_MARKET_STATE_PATH.resolve(project_root)

    def load_current(self) -> MarketStateSnapshot | None:
        return load_market_state_snapshot(self.current_path)

    def load_previous(self) -> MarketStateSnapshot | None:
        return load_market_state_snapshot(self.previous_path)

    def update(
        self,
        *,
        as_of_date: date,
        workflow: str,
        daily_summary: DailyResearchSummary | None = None,
        portfolio_review: PortfolioReviewReport | None = None,
        intraday_review: IntradayPortfolioReviewReport | None = None,
        portfolio_path: str | None = None,
    ) -> MarketStateUpdateResult:
        previous_snapshot = self.load_current()
        current_snapshot = build_market_state_snapshot(
            as_of_date=as_of_date,
            workflow=workflow,
            previous_snapshot=previous_snapshot,
            daily_summary=daily_summary,
            portfolio_review=portfolio_review,
            intraday_review=intraday_review,
            portfolio_path=portfolio_path,
        )
        change_set = diff_market_state_snapshots(previous_snapshot, current_snapshot)
        if previous_snapshot is not None:
            write_json_file(self.previous_path, previous_snapshot.to_dict())
        write_json_file(self.current_path, current_snapshot.to_dict())
        return MarketStateUpdateResult(
            current_snapshot=current_snapshot,
            previous_snapshot=previous_snapshot,
            change_set=change_set,
            current_path=self.current_path.resolve(),
            previous_path=self.previous_path.resolve(),
        )


def build_market_state_snapshot(
    *,
    as_of_date: date,
    workflow: str,
    previous_snapshot: MarketStateSnapshot | None = None,
    daily_summary: DailyResearchSummary | None = None,
    portfolio_review: PortfolioReviewReport | None = None,
    intraday_review: IntradayPortfolioReviewReport | None = None,
    portfolio_path: str | None = None,
) -> MarketStateSnapshot:
    """Build a compact live market snapshot from the latest workflow outputs."""

    if not workflow.strip():
        raise ValueError("workflow cannot be empty.")

    benchmark_context = (
        previous_snapshot.benchmark_context if previous_snapshot is not None else None
    )
    breadth_context = (
        previous_snapshot.breadth_context if previous_snapshot is not None else None
    )
    volatility_context = (
        previous_snapshot.volatility_context if previous_snapshot is not None else None
    )
    sector_context_summary = (
        previous_snapshot.sector_context_summary if previous_snapshot is not None else ()
    )
    approved_candidate_queue = (
        previous_snapshot.approved_candidate_queue if previous_snapshot is not None else ()
    )
    top_rejected_reasons_summary = (
        previous_snapshot.top_rejected_reasons_summary if previous_snapshot is not None else ()
    )
    portfolio_review_summary = (
        previous_snapshot.portfolio_review_summary if previous_snapshot is not None else None
    )
    intraday_review_summary = (
        previous_snapshot.intraday_review_summary if previous_snapshot is not None else None
    )
    current_action_states = (
        dict(previous_snapshot.current_action_states_by_symbol)
        if previous_snapshot is not None
        else {}
    )
    resolved_portfolio_path = (
        portfolio_path
        if portfolio_path is not None
        else (previous_snapshot.portfolio_path if previous_snapshot is not None else None)
    )
    source_workflows = (
        set(previous_snapshot.source_workflows) if previous_snapshot is not None else set()
    )
    source_workflows.add(workflow.strip())

    if daily_summary is not None:
        benchmark_context, breadth_context, volatility_context = _market_context_sections(
            daily_summary.market_context
        )
        approved_candidate_queue = _approved_candidate_queue(daily_summary)
        sector_context_summary = _sector_context_summary(daily_summary)
        top_rejected_reasons_summary = _top_rejected_reasons_summary(daily_summary)

    if portfolio_review is not None:
        portfolio_review_summary = _portfolio_review_summary(portfolio_review)
        current_action_states = _action_states_from_rows(
            portfolio_review.rows,
            source_workflow=workflow,
        )

    if intraday_review is not None:
        intraday_review_summary = _intraday_review_summary(intraday_review)
        current_action_states = _action_states_from_rows(
            intraday_review.rows,
            source_workflow=workflow,
        )
        if intraday_review.portfolio_path is not None:
            resolved_portfolio_path = intraday_review.portfolio_path

    as_of_timestamp = _latest_timestamp_for_snapshot(
        daily_summary=daily_summary,
        portfolio_review=portfolio_review,
        intraday_review=intraday_review,
    )
    current_alertable_states = _alertable_states(
        candidate_queue=approved_candidate_queue,
        current_action_states=current_action_states,
        sector_context_summary=sector_context_summary,
    )

    return MarketStateSnapshot(
        schema_version=MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        as_of_timestamp=as_of_timestamp,
        as_of_date=as_of_date,
        portfolio_path=resolved_portfolio_path,
        source_workflows=tuple(sorted(source_workflows)),
        benchmark_context=benchmark_context,
        breadth_context=breadth_context,
        volatility_context=volatility_context,
        sector_context_summary=sector_context_summary,
        approved_candidate_queue=approved_candidate_queue,
        portfolio_review_summary=portfolio_review_summary,
        intraday_review_summary=intraday_review_summary,
        current_alertable_states=current_alertable_states,
        current_action_states_by_symbol=current_action_states,
        top_rejected_reasons_summary=top_rejected_reasons_summary,
    )


def diff_market_state_snapshots(
    previous_snapshot: MarketStateSnapshot | None,
    current_snapshot: MarketStateSnapshot,
) -> MarketStateChangeSet:
    """Return the meaningful transition set between two snapshots."""

    if previous_snapshot is None:
        return MarketStateChangeSet(
            as_of_timestamp=current_snapshot.as_of_timestamp,
            as_of_date=current_snapshot.as_of_date,
            baseline_established=True,
            transitions=(),
        )

    transitions: list[MarketStateTransition] = []
    transitions.extend(_portfolio_action_transitions(previous_snapshot, current_snapshot))
    transitions.extend(_top_priority_candidate_transitions(previous_snapshot, current_snapshot))
    transitions.extend(_breadth_transitions(previous_snapshot, current_snapshot))
    transitions.extend(_volatility_transitions(previous_snapshot, current_snapshot))
    transitions.extend(_sector_block_transitions(previous_snapshot, current_snapshot))
    transitions.sort(
        key=lambda transition: (
            transition.transition_type,
            transition.symbol or "",
            transition.sector_name or "",
            transition.current_value or "",
            transition.previous_value or "",
        )
    )
    return MarketStateChangeSet(
        as_of_timestamp=current_snapshot.as_of_timestamp,
        as_of_date=current_snapshot.as_of_date,
        baseline_established=False,
        transitions=tuple(transitions),
    )


def write_market_state_change_set(
    change_set: MarketStateChangeSet,
    output_path: Path,
) -> Path:
    """Write a structured state-change payload as JSON."""

    return write_json_file(output_path, change_set.to_dict())


def write_market_state_change_summary(
    change_set: MarketStateChangeSet,
    output_path: Path,
) -> Path:
    """Write a compact human-readable state-change summary."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(change_set.to_text(), encoding="utf-8")
    return resolved_output_path


def _market_context_sections(
    market_context: MarketContext | None,
) -> tuple[
    MarketStateBenchmarkContext | None,
    MarketStateBreadthContext | None,
    MarketStateVolatilityContext | None,
]:
    if market_context is None:
        return None, None, None
    return (
        MarketStateBenchmarkContext(
            benchmark_symbol=market_context.benchmark_symbol,
            regime_filter_mode=market_context.regime_filter_mode,
            regime_passed=market_context.regime_passed,
            benchmark_return=market_context.benchmark_return,
            benchmark_intraday_return_vs_open=market_context.benchmark_intraday_return_vs_open,
        ),
        MarketStateBreadthContext(
            market_breadth_pct_above_200ma=market_context.market_breadth_pct_above_200ma,
            market_breadth_pct_above_50ma=market_context.market_breadth_pct_above_50ma,
            market_breadth_state=market_context.market_breadth_state,
        ),
        MarketStateVolatilityContext(
            vix_close=market_context.vix_close,
            vix_sma_short=market_context.vix_sma_short,
            vix_sma_long=market_context.vix_sma_long,
            volatility_regime_state=market_context.volatility_regime_state,
            volatility_regime_risk_off=market_context.volatility_regime_risk_off,
        ),
    )


def _approved_candidate_queue(summary: DailyResearchSummary) -> tuple[MarketStateCandidateState, ...]:
    queue: list[MarketStateCandidateState] = []
    for row in summary.rows:
        if row.status != "approved":
            continue
        execution_priority = _optional_mapping(row.metadata.get("execution_priority"))
        signal_metadata = _optional_mapping(row.metadata.get("signal_metadata")) or {}
        queue.append(
            MarketStateCandidateState(
                symbol=row.symbol,
                rank=row.rank,
                preset_name=row.preset_name,
                status=row.status,
                score=row.score,
                actionable_now=_required_bool_with_default(
                    execution_priority,
                    key="actionable_now",
                    default=row.status == "approved",
                ),
                priority_bucket=_optional_text_from_mapping(
                    execution_priority,
                    key="priority_bucket",
                ),
                entry_price_hint=row.entry_price_hint,
                stop_level=row.stop_level,
                sector_name=_optional_text(signal_metadata.get("sector_name")),
                blocked_by_portfolio_heat=_candidate_blocked_by_portfolio_heat(row.metadata),
            )
        )
    queue.sort(key=lambda candidate: (candidate.rank, candidate.symbol, candidate.preset_name or ""))
    return tuple(queue)


def _sector_context_summary(
    summary: DailyResearchSummary,
) -> tuple[MarketStateSectorSummary, ...]:
    sector_state: dict[str, dict[str, Any]] = {}
    for row in summary.rows:
        signal_metadata = _optional_mapping(row.metadata.get("signal_metadata")) or {}
        sector_name = _optional_text(signal_metadata.get("sector_name"))
        if sector_name is None:
            continue
        entry = sector_state.setdefault(
            sector_name,
            {
                "sector_etf_symbol": _optional_text(signal_metadata.get("sector_etf_symbol")),
                "approved_candidate_count": 0,
                "actionable_candidate_count": 0,
                "blocked_candidate_count": 0,
                "blocked_by_portfolio_heat": False,
                "top_symbols": [],
            },
        )
        if entry["sector_etf_symbol"] is None:
            entry["sector_etf_symbol"] = _optional_text(signal_metadata.get("sector_etf_symbol"))
        if row.status == "approved":
            entry["approved_candidate_count"] += 1
            if _candidate_actionable_now(row.metadata):
                entry["actionable_candidate_count"] += 1
        if _candidate_blocked_by_portfolio_heat(row.metadata):
            entry["blocked_candidate_count"] += 1
            entry["blocked_by_portfolio_heat"] = True
        if len(entry["top_symbols"]) < 5 and row.symbol not in entry["top_symbols"]:
            entry["top_symbols"].append(row.symbol)

    sectors = [
        MarketStateSectorSummary(
            sector_name=sector_name,
            sector_etf_symbol=_optional_text(payload.get("sector_etf_symbol")),
            approved_candidate_count=int(payload["approved_candidate_count"]),
            actionable_candidate_count=int(payload["actionable_candidate_count"]),
            blocked_candidate_count=int(payload["blocked_candidate_count"]),
            blocked_by_portfolio_heat=bool(payload["blocked_by_portfolio_heat"]),
            top_symbols=tuple(sorted(payload["top_symbols"])),
        )
        for sector_name, payload in sector_state.items()
    ]
    sectors.sort(key=lambda sector: (sector.sector_name, sector.sector_etf_symbol or ""))
    return tuple(sectors)


def _top_rejected_reasons_summary(
    summary: DailyResearchSummary,
) -> tuple[MarketStateRejectedReasonSummary, ...]:
    counts: dict[str, int] = {}
    for row in summary.rows:
        if row.status != "rejected":
            continue
        reasons = row.rejection_reasons or (row.rationale,)
        for reason in reasons:
            cleaned = reason.strip()
            if not cleaned:
                continue
            counts[cleaned] = counts.get(cleaned, 0) + 1
    summaries = [
        MarketStateRejectedReasonSummary(reason=reason, count=count)
        for reason, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]
    return tuple(summaries)


def _portfolio_review_summary(report: PortfolioReviewReport) -> MarketStatePortfolioSummary:
    urgent_symbols = tuple(
        sorted(
            row.symbol
            for row in report.rows
            if row.suggested_action in {"EXIT CANDIDATE", "WATCH CLOSELY", "RAISE STOP"}
        )
    )
    return MarketStatePortfolioSummary(
        position_count=len(report.current_positions),
        hold_count=sum(row.suggested_action == "HOLD" for row in report.rows),
        watch_closely_count=sum(
            row.suggested_action == "WATCH CLOSELY" for row in report.rows
        ),
        exit_candidate_count=sum(
            row.suggested_action == "EXIT CANDIDATE" for row in report.rows
        ),
        raise_stop_count=sum(row.suggested_action == "RAISE STOP" for row in report.rows),
        urgent_symbols=urgent_symbols,
    )


def _intraday_review_summary(
    report: IntradayPortfolioReviewReport,
) -> MarketStateIntradaySummary:
    urgent_symbols = tuple(
        sorted(
            row.symbol
            for row in report.rows
            if row.suggested_action in {"EXIT CANDIDATE", "WATCH CLOSELY", "RAISE STOP"}
        )
    )
    return MarketStateIntradaySummary(
        position_count=len(report.current_positions),
        interval_minutes=report.interval_minutes,
        hold_count=sum(row.suggested_action == "HOLD" for row in report.rows),
        watch_closely_count=sum(
            row.suggested_action == "WATCH CLOSELY" for row in report.rows
        ),
        exit_candidate_count=sum(
            row.suggested_action == "EXIT CANDIDATE" for row in report.rows
        ),
        raise_stop_count=sum(row.suggested_action == "RAISE STOP" for row in report.rows),
        urgent_symbols=urgent_symbols,
    )


def _action_states_from_rows(
    rows: Sequence[PortfolioReviewRow],
    *,
    source_workflow: str,
) -> dict[str, MarketStateActionState]:
    action_states = {
        row.symbol: MarketStateActionState(
            symbol=row.symbol,
            action=row.suggested_action,
            source_workflow=source_workflow,
            preset_name=row.preset_name,
            latest_close=row.latest_close,
            current_stop=row.current_stop,
            suggested_stop=row.suggested_stop,
            rationale=row.rationale,
        )
        for row in sorted(rows, key=lambda row: (row.symbol, row.preset_name or ""))
    }
    return action_states


def _alertable_states(
    *,
    candidate_queue: Sequence[MarketStateCandidateState],
    current_action_states: Mapping[str, MarketStateActionState],
    sector_context_summary: Sequence[MarketStateSectorSummary],
) -> tuple[MarketStateAlertableState, ...]:
    states: list[MarketStateAlertableState] = []
    for action_state in sorted(current_action_states.values(), key=lambda state: state.symbol):
        if action_state.action == "HOLD":
            continue
        states.append(
            MarketStateAlertableState(
                state_key=f"symbol:{action_state.symbol}:{action_state.action}",
                category=action_state.action,
                source_workflow=action_state.source_workflow,
                symbol=action_state.symbol,
                action=action_state.action,
                rationale=action_state.rationale,
            )
        )

    for candidate in candidate_queue:
        if not candidate.actionable_now or candidate.priority_bucket != "top_priority":
            continue
        states.append(
            MarketStateAlertableState(
                state_key=f"candidate:{candidate.symbol}:BUY CANDIDATE",
                category="BUY CANDIDATE",
                source_workflow="monitor-market",
                symbol=candidate.symbol,
                priority_bucket=candidate.priority_bucket,
                action="BUY CANDIDATE",
                rationale=(
                    f"Rank {candidate.rank}"
                    + (
                        f" | preset={candidate.preset_name}"
                        if candidate.preset_name is not None
                        else ""
                    )
                ),
            )
        )

    for sector in sector_context_summary:
        if not sector.blocked_by_portfolio_heat:
            continue
        states.append(
            MarketStateAlertableState(
                state_key=f"sector:{sector.sector_name}:SECTOR BLOCK",
                category="SECTOR BLOCK",
                source_workflow="monitor-market",
                sector_name=sector.sector_name,
                action="SECTOR BLOCK",
                rationale=(
                    f"{sector.blocked_candidate_count} candidate(s) blocked by portfolio heat."
                ),
            )
        )

    states.sort(
        key=lambda state: (
            state.category,
            state.symbol or "",
            state.sector_name or "",
            state.state_key,
        )
    )
    return tuple(states)


def _portfolio_action_transitions(
    previous_snapshot: MarketStateSnapshot,
    current_snapshot: MarketStateSnapshot,
) -> list[MarketStateTransition]:
    transitions: list[MarketStateTransition] = []
    previous_states = previous_snapshot.current_action_states_by_symbol
    current_states = current_snapshot.current_action_states_by_symbol

    for symbol in sorted(set(previous_states) | set(current_states)):
        previous_state = previous_states.get(symbol)
        current_state = current_states.get(symbol)
        previous_action = previous_state.action if previous_state is not None else None
        current_action = current_state.action if current_state is not None else None
        if previous_action == current_action:
            continue
        if previous_action is None or current_action is None:
            continue
        transition_type = _portfolio_transition_type(previous_action, current_action)
        if transition_type is None:
            continue
        transitions.append(
            MarketStateTransition(
                transition_type=transition_type,
                symbol=symbol,
                previous_value=previous_action,
                current_value=current_action,
                message=_portfolio_transition_message(
                    symbol,
                    previous_action=previous_action,
                    current_action=current_action,
                ),
                metadata={
                    "source_workflow": current_state.source_workflow,
                    "rationale": current_state.rationale,
                },
            )
        )
    return transitions


def _top_priority_candidate_transitions(
    previous_snapshot: MarketStateSnapshot,
    current_snapshot: MarketStateSnapshot,
) -> list[MarketStateTransition]:
    previous_top = {
        candidate.symbol: candidate
        for candidate in previous_snapshot.approved_candidate_queue
        if candidate.actionable_now and candidate.priority_bucket == "top_priority"
    }
    current_top = {
        candidate.symbol: candidate
        for candidate in current_snapshot.approved_candidate_queue
        if candidate.actionable_now and candidate.priority_bucket == "top_priority"
    }
    transitions: list[MarketStateTransition] = []
    for symbol in sorted(set(current_top) - set(previous_top)):
        candidate = current_top[symbol]
        transitions.append(
            MarketStateTransition(
                transition_type="TOP_PRIORITY_CANDIDATE_ENTERED",
                symbol=symbol,
                current_value="top_priority",
                message=(
                    f"{symbol} entered the top-priority buy queue"
                    + (
                        f" at rank {candidate.rank}"
                        if candidate.rank > 0
                        else ""
                    )
                    + (
                        f" under {candidate.preset_name}."
                        if candidate.preset_name is not None
                        else "."
                    )
                ),
                metadata={
                    "rank": candidate.rank,
                    "preset_name": candidate.preset_name,
                },
            )
        )
    for symbol in sorted(set(previous_top) - set(current_top)):
        previous_candidate = previous_top[symbol]
        transitions.append(
            MarketStateTransition(
                transition_type="TOP_PRIORITY_CANDIDATE_EXITED",
                symbol=symbol,
                previous_value="top_priority",
                message=f"{symbol} left the top-priority buy queue.",
                metadata={
                    "previous_rank": previous_candidate.rank,
                    "preset_name": previous_candidate.preset_name,
                },
            )
        )
    return transitions


def _breadth_transitions(
    previous_snapshot: MarketStateSnapshot,
    current_snapshot: MarketStateSnapshot,
) -> list[MarketStateTransition]:
    previous_state = (
        previous_snapshot.breadth_context.market_breadth_state
        if previous_snapshot.breadth_context is not None
        else None
    )
    current_state = (
        current_snapshot.breadth_context.market_breadth_state
        if current_snapshot.breadth_context is not None
        else None
    )
    if previous_state == current_state or current_state is None:
        return []
    return [
        MarketStateTransition(
            transition_type="BREADTH_REGIME_CHANGED",
            previous_value=previous_state,
            current_value=current_state,
            message=(
                "Market breadth changed"
                f" from {previous_state or 'unknown'} to {current_state}."
            ),
            metadata={
                "previous_pct_above_200ma": (
                    previous_snapshot.breadth_context.market_breadth_pct_above_200ma
                    if previous_snapshot.breadth_context is not None
                    else None
                ),
                "current_pct_above_200ma": (
                    current_snapshot.breadth_context.market_breadth_pct_above_200ma
                    if current_snapshot.breadth_context is not None
                    else None
                ),
            },
        )
    ]


def _volatility_transitions(
    previous_snapshot: MarketStateSnapshot,
    current_snapshot: MarketStateSnapshot,
) -> list[MarketStateTransition]:
    previous_state = (
        previous_snapshot.volatility_context.volatility_regime_state
        if previous_snapshot.volatility_context is not None
        else None
    )
    current_state = (
        current_snapshot.volatility_context.volatility_regime_state
        if current_snapshot.volatility_context is not None
        else None
    )
    previous_risk_off = (
        previous_snapshot.volatility_context.volatility_regime_risk_off
        if previous_snapshot.volatility_context is not None
        else None
    )
    current_risk_off = (
        current_snapshot.volatility_context.volatility_regime_risk_off
        if current_snapshot.volatility_context is not None
        else None
    )
    if previous_state == current_state and previous_risk_off == current_risk_off:
        return []
    if current_state is None and current_risk_off is None:
        return []
    return [
        MarketStateTransition(
            transition_type="VOLATILITY_REGIME_CHANGED",
            previous_value=_volatility_transition_label(previous_state, previous_risk_off),
            current_value=_volatility_transition_label(current_state, current_risk_off),
            message=(
                "Volatility regime changed"
                f" from {_volatility_transition_label(previous_state, previous_risk_off) or 'unknown'}"
                f" to {_volatility_transition_label(current_state, current_risk_off) or 'unknown'}."
            ),
            metadata={
                "previous_vix_close": (
                    previous_snapshot.volatility_context.vix_close
                    if previous_snapshot.volatility_context is not None
                    else None
                ),
                "current_vix_close": (
                    current_snapshot.volatility_context.vix_close
                    if current_snapshot.volatility_context is not None
                    else None
                ),
            },
        )
    ]


def _sector_block_transitions(
    previous_snapshot: MarketStateSnapshot,
    current_snapshot: MarketStateSnapshot,
) -> list[MarketStateTransition]:
    previous_blocked = {
        sector.sector_name
        for sector in previous_snapshot.sector_context_summary
        if sector.blocked_by_portfolio_heat
    }
    current_blocked = {
        sector.sector_name
        for sector in current_snapshot.sector_context_summary
        if sector.blocked_by_portfolio_heat
    }
    transitions: list[MarketStateTransition] = []
    for sector_name in sorted(current_blocked - previous_blocked):
        transitions.append(
            MarketStateTransition(
                transition_type="SECTOR_BLOCK_ENTERED",
                sector_name=sector_name,
                current_value="blocked",
                message=f"Portfolio heat is now blocking new entries in {sector_name}.",
            )
        )
    for sector_name in sorted(previous_blocked - current_blocked):
        transitions.append(
            MarketStateTransition(
                transition_type="SECTOR_BLOCK_CLEARED",
                sector_name=sector_name,
                previous_value="blocked",
                current_value="clear",
                message=f"Portfolio heat is no longer blocking new entries in {sector_name}.",
            )
        )
    return transitions


def _portfolio_transition_type(
    previous_action: str,
    current_action: str,
) -> str | None:
    if previous_action == "HOLD" and current_action == "WATCH CLOSELY":
        return "HOLD_TO_WATCH_CLOSELY"
    if previous_action == "WATCH CLOSELY" and current_action == "EXIT CANDIDATE":
        return "WATCH_CLOSELY_TO_EXIT_CANDIDATE"
    previous_severity = _PORTFOLIO_ACTION_SEVERITY.get(previous_action)
    current_severity = _PORTFOLIO_ACTION_SEVERITY.get(current_action)
    if previous_severity is None or current_severity is None:
        return None
    if current_severity > previous_severity:
        return "POSITION_ACTION_ESCALATED"
    if current_severity < previous_severity:
        return "POSITION_RISK_EASED"
    return None


def _portfolio_transition_message(
    symbol: str,
    *,
    previous_action: str,
    current_action: str,
) -> str:
    if previous_action == "HOLD" and current_action == "WATCH CLOSELY":
        return f"{symbol} moved from HOLD to WATCH CLOSELY."
    if previous_action == "WATCH CLOSELY" and current_action == "EXIT CANDIDATE":
        return f"{symbol} escalated from WATCH CLOSELY to EXIT CANDIDATE."
    if _PORTFOLIO_ACTION_SEVERITY.get(current_action, 0) > _PORTFOLIO_ACTION_SEVERITY.get(
        previous_action,
        0,
    ):
        return f"{symbol} risk escalated from {previous_action} to {current_action}."
    return f"{symbol} risk eased from {previous_action} to {current_action}."


def _latest_timestamp_for_snapshot(
    *,
    daily_summary: DailyResearchSummary | None,
    portfolio_review: PortfolioReviewReport | None,
    intraday_review: IntradayPortfolioReviewReport | None,
) -> str:
    timestamps = [
        report.generated_at_utc
        for report in (daily_summary, portfolio_review, intraday_review)
        if report is not None and isinstance(report.generated_at_utc, str)
    ]
    if not timestamps:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamps.sort()
    return timestamps[-1]


def _candidate_actionable_now(metadata: Mapping[str, Any]) -> bool:
    execution_priority = _optional_mapping(metadata.get("execution_priority"))
    return _required_bool_with_default(execution_priority, key="actionable_now", default=False)


def _candidate_blocked_by_portfolio_heat(metadata: Mapping[str, Any]) -> bool:
    heat_projection = _optional_mapping(metadata.get("portfolio_heat_projection")) or {}
    heat_context = _optional_mapping(metadata.get("portfolio_heat_context")) or {}
    return bool(
        _optional_bool(heat_projection.get("sector_concentration_risk"))
        or _optional_bool(heat_projection.get("sector_count_concentration_risk"))
        or _optional_bool(heat_projection.get("sector_notional_concentration_risk"))
        or _optional_bool(heat_projection.get("correlated_exposure_risk"))
        or _optional_bool(heat_projection.get("crowded_exposure_bucket"))
        or _optional_bool(heat_context.get("sector_concentration_risk"))
        or _optional_bool(heat_context.get("correlated_exposure_risk"))
    )


def _volatility_transition_label(
    regime_state: str | None,
    risk_off: bool | None,
) -> str | None:
    if regime_state is None and risk_off is None:
        return None
    if risk_off is None:
        return regime_state
    risk_label = "risk_off" if risk_off else "risk_on"
    return f"{regime_state or 'unknown'}:{risk_label}"


def _required_text(value: Any, field_name: str) -> str:
    cleaned = _optional_text(value)
    if cleaned is None:
        raise ValueError(f"Expected '{field_name}' to be a non-empty string.")
    return cleaned


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _required_bool(value: Any, field_name: str) -> bool:
    resolved = _optional_bool(value)
    if resolved is None:
        raise ValueError(f"Expected '{field_name}' to be a boolean.")
    return resolved


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected '{field_name}' to be an integer, not a boolean.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{field_name}' to be an integer.") from exc
    if resolved < 0:
        raise ValueError(f"Expected '{field_name}' to be non-negative.")
    return resolved


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{field_name}' to be a mapping.")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _mapping_or_empty(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, field_name)


def _sequence_of_mappings(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Expected '{field_name}' to be a sequence of mappings.")
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Expected '{field_name}[{index}]' to be a mapping."
            )
        items.append(item)
    return tuple(items)


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Expected '{field_name}' to be a sequence of strings.")
    resolved: list[str] = []
    for index, item in enumerate(value):
        cleaned = _optional_text(item)
        if cleaned is None:
            raise ValueError(f"Expected '{field_name}[{index}]' to be a non-empty string.")
        resolved.append(cleaned)
    return tuple(resolved)


def _require_iso_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{field_name}' to be an ISO date string.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"Expected '{field_name}' to be an ISO date string.") from exc


def _required_bool_with_default(
    mapping_value: Mapping[str, Any] | None,
    *,
    key: str,
    default: bool,
) -> bool:
    if mapping_value is None:
        return default
    resolved = _optional_bool(mapping_value.get(key))
    if resolved is None:
        return default
    return resolved


def _optional_text_from_mapping(
    mapping_value: Mapping[str, Any] | None,
    *,
    key: str,
) -> str | None:
    if mapping_value is None:
        return None
    return _optional_text(mapping_value.get(key))
