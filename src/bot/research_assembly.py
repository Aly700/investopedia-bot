"""Structured research-data assembly for daily-summary style workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import logging
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import pandas as pd

from bot.config import AppConfig
from bot.data.earnings import (
    EarningsCalendarProvider,
    EarningsRiskContext,
    build_earnings_risk_contexts,
    create_earnings_calendar_provider,
)
from bot.data.providers import (
    DailyBarProvider,
    DataProviderConfigurationError,
    DataProviderError,
    provider_error_is_entitlement_limited,
    provider_error_is_timeout,
)
from bot.data.reference import ReferenceUniverseProvider
from bot.data.sector_context import (
    SectorFeatureContext,
    SymbolSectorClassification,
    build_sector_feature_contexts,
    load_symbol_sector_classifications,
    sector_context_history_start,
)
from bot.data.universe import UniverseBuilder, UniverseMember
from bot.data.volatility_context import (
    build_volatility_regime_context,
    volatility_context_history_start,
)
from bot.providers.bundle import ProviderBundle


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

_EMPTY_VOLATILITY_CONTEXT: dict[str, float | str | bool | None] = {
    "vix_close": None,
    "vix_sma_short": None,
    "vix_sma_long": None,
    "volatility_regime_state": None,
    "volatility_regime_risk_off": None,
}


@dataclass(frozen=True)
class ResearchDataResult(Generic[T]):
    """Structured availability/degradation result for one assembly operation."""

    role_name: str | None
    operation: str
    provider_name: str | None
    status: str
    value: T | None = None
    issue_code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "degraded", "unavailable", "failed"}:
            raise ValueError(f"Unsupported research-data status '{self.status}'.")

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_usable(self) -> bool:
        return self.status == "ok" and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_name": self.role_name,
            "operation": self.operation,
            "provider": self.provider_name,
            "status": self.status,
            "issue_code": self.issue_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateUniverseData:
    """Screened candidate universe plus any loaded symbol frames."""

    fetch_start: date
    universe_members: tuple[UniverseMember, ...]
    symbol_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkFrameData:
    """Benchmark symbol context for regime-driven workflows."""

    symbol: str | None
    frame: pd.DataFrame | None


@dataclass(frozen=True)
class DailySummaryResearchInputs:
    """Structured assembly payload for daily-summary style workflows."""

    candidate_data: ResearchDataResult[CandidateUniverseData]
    benchmark: ResearchDataResult[BenchmarkFrameData]
    earnings_contexts: ResearchDataResult[dict[str, EarningsRiskContext]]
    position_classifications: ResearchDataResult[dict[str, SymbolSectorClassification]]
    sector_contexts: ResearchDataResult[dict[str, SectorFeatureContext]]
    volatility_context: ResearchDataResult[dict[str, float | str | bool | None]]

    def results(self) -> tuple[ResearchDataResult[Any], ...]:
        return (
            self.candidate_data,
            self.benchmark,
            self.earnings_contexts,
            self.position_classifications,
            self.sector_contexts,
            self.volatility_context,
        )

    def result_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            "candidate_data": self.candidate_data.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "earnings_contexts": self.earnings_contexts.to_dict(),
            "position_classifications": self.position_classifications.to_dict(),
            "sector_contexts": self.sector_contexts.to_dict(),
            "volatility_context": self.volatility_context.to_dict(),
        }


@dataclass(frozen=True)
class ReviewPositionDailyFramesData:
    """Daily-bar history loaded for held symbols during review workflows."""

    history_start: date
    symbol_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntradayBenchmarkMetricsData:
    """Intraday benchmark metrics used for relative-strength review context."""

    symbol: str | None
    metrics: Mapping[str, float | str | None] | None


@dataclass(frozen=True)
class PortfolioReviewResearchInputs:
    """Structured assembly payload for daily portfolio review workflows."""

    benchmark: ResearchDataResult[BenchmarkFrameData]
    earnings_contexts: ResearchDataResult[dict[str, EarningsRiskContext]]
    position_daily_symbol_frames: ResearchDataResult[ReviewPositionDailyFramesData]
    sector_contexts: ResearchDataResult[dict[str, SectorFeatureContext]]

    def results(self) -> tuple[ResearchDataResult[Any], ...]:
        return (
            self.benchmark,
            self.earnings_contexts,
            self.position_daily_symbol_frames,
            self.sector_contexts,
        )

    def result_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            "benchmark": self.benchmark.to_dict(),
            "earnings_contexts": self.earnings_contexts.to_dict(),
            "position_daily_symbol_frames": self.position_daily_symbol_frames.to_dict(),
            "sector_contexts": self.sector_contexts.to_dict(),
        }


@dataclass(frozen=True)
class IntradayReviewResearchInputs:
    """Structured assembly payload for intraday portfolio review workflows."""

    benchmark_intraday_metrics: ResearchDataResult[IntradayBenchmarkMetricsData]
    earnings_contexts: ResearchDataResult[dict[str, EarningsRiskContext]]
    position_daily_symbol_frames: ResearchDataResult[ReviewPositionDailyFramesData]
    sector_contexts: ResearchDataResult[dict[str, SectorFeatureContext]]

    def results(self) -> tuple[ResearchDataResult[Any], ...]:
        return (
            self.benchmark_intraday_metrics,
            self.earnings_contexts,
            self.position_daily_symbol_frames,
            self.sector_contexts,
        )

    def result_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            "benchmark_intraday_metrics": self.benchmark_intraday_metrics.to_dict(),
            "earnings_contexts": self.earnings_contexts.to_dict(),
            "position_daily_symbol_frames": self.position_daily_symbol_frames.to_dict(),
            "sector_contexts": self.sector_contexts.to_dict(),
        }


def load_earnings_contexts_result(
    *,
    config: AppConfig,
    env_file: Path | None,
    provider: EarningsCalendarProvider | None,
    symbols: Sequence[str],
    as_of_date: date,
    earnings_watch_days: int,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[dict[str, EarningsRiskContext]]:
    normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    provider_name = _provider_name(provider)
    if not normalized_symbols:
        return ResearchDataResult(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name=provider_name,
            status="ok",
            value={},
        )

    try:
        resolved_provider = (
            provider
            if provider is not None
            else create_earnings_calendar_provider(config, env_file=env_file)
        )
        contexts = build_earnings_risk_contexts(
            normalized_symbols,
            as_of_date=as_of_date,
            provider=resolved_provider,
            risk_window_days=earnings_watch_days,
            lookahead_calendar_days=max(30, earnings_watch_days + 7),
            refresh_cache=refresh_cache,
        )
        return ResearchDataResult(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name=_provider_name(resolved_provider),
            status="ok",
            value=contexts,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        return _degraded_result(
            role_name="earnings_calendar",
            operation="load_earnings_contexts",
            provider_name=provider_name,
            issue_code=_issue_code_for_error(exc),
            message=_log_earnings_context_warning(log_label=log_label, error=exc),
            value={},
        )


def load_position_sector_classifications_result(
    current_positions: Sequence[object],
    *,
    config: AppConfig,
    env_file: Path | None,
    reference_provider: ReferenceUniverseProvider | None,
    as_of_date: date,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[dict[str, SymbolSectorClassification]]:
    if not current_positions:
        return ResearchDataResult(
            role_name="reference_data",
            operation="load_position_sector_classifications",
            provider_name=_provider_name(reference_provider),
            status="ok",
            value={},
        )

    try:
        classifications = load_symbol_sector_classifications(
            [str(getattr(position, "symbol", "")) for position in current_positions],
            config=config,
            env_file=env_file,
            reference_provider=reference_provider,
            as_of_date=as_of_date,
            refresh_cache=refresh_cache,
        )
        return ResearchDataResult(
            role_name="reference_data",
            operation="load_position_sector_classifications",
            provider_name=_provider_name(reference_provider),
            status="ok",
            value=classifications,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        message = f"Portfolio heat context unavailable for {log_label}: {exc}"
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="reference_data",
            operation="load_position_sector_classifications",
            provider_name=_provider_name(reference_provider),
            issue_code=_issue_code_for_error(exc),
            message=message,
            value={},
        )


def load_sector_contexts_result(
    *,
    config: AppConfig,
    env_file: Path | None,
    provider: DailyBarProvider,
    reference_provider: ReferenceUniverseProvider | None,
    symbol_frames: Mapping[str, pd.DataFrame],
    as_of_date: date,
    history_start: date,
    benchmark_sma_fast: int,
    benchmark_sma_slow: int,
    relative_strength_window: int,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[dict[str, SectorFeatureContext]]:
    normalized_symbol_frames = {
        symbol.strip().upper(): frame
        for symbol, frame in symbol_frames.items()
        if symbol.strip() and not frame.empty
    }
    if not normalized_symbol_frames:
        return ResearchDataResult(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name=_provider_name(reference_provider),
            status="ok",
            value={},
        )

    try:
        classifications = load_symbol_sector_classifications(
            tuple(normalized_symbol_frames),
            config=config,
            env_file=env_file,
            reference_provider=reference_provider,
            as_of_date=as_of_date,
            refresh_cache=refresh_cache,
        )
        contexts = build_sector_feature_contexts(
            normalized_symbol_frames,
            provider=provider,
            as_of_date=as_of_date,
            history_start=history_start,
            sector_classifications=classifications,
            benchmark_sma_fast=benchmark_sma_fast,
            benchmark_sma_slow=benchmark_sma_slow,
            relative_strength_window=relative_strength_window,
            refresh_cache=refresh_cache,
        )
        return ResearchDataResult(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name=_provider_name(reference_provider),
            status="ok",
            value=contexts,
        )
    except (DataProviderConfigurationError, DataProviderError, ValueError) as exc:
        message = f"Sector context unavailable for {log_label}: {exc}"
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="reference_data",
            operation="load_sector_contexts",
            provider_name=_provider_name(reference_provider),
            issue_code=_issue_code_for_error(exc),
            message=message,
            value={},
        )


def load_volatility_context_result(
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    vix_caution_threshold: float | None,
    vix_entry_block_threshold: float | None,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[dict[str, float | str | bool | None]]:
    provider_name = _provider_name(provider)
    provider_capabilities = getattr(provider, "capabilities", None)
    if (
        provider_capabilities is not None
        and not getattr(provider_capabilities, "supports_vix_symbols", False)
    ):
        message = (
            "Volatility context unavailable for "
            f"{log_label} because historical-bars provider '{provider_name or 'unknown'}' "
            "does not support VIX/index-style symbols; continuing without volatility regime context."
        )
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="historical_bars",
            operation="load_volatility_context",
            provider_name=provider_name,
            issue_code="unsupported_capability",
            message=message,
            value=dict(_EMPTY_VOLATILITY_CONTEXT),
        )
    try:
        context = build_volatility_regime_context(
            provider=provider,
            as_of_date=as_of_date,
            history_start=volatility_context_history_start(as_of_date),
            caution_threshold=vix_caution_threshold,
            entry_block_threshold=vix_entry_block_threshold,
            refresh_cache=refresh_cache,
        )
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_volatility_context",
            provider_name=provider_name,
            status="ok",
            value=context,
        )
    except (DataProviderError, ValueError) as exc:
        message = f"Volatility context unavailable for {log_label}: {exc}"
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="historical_bars",
            operation="load_volatility_context",
            provider_name=provider_name,
            issue_code=_issue_code_for_error(exc),
            message=message,
            value=dict(_EMPTY_VOLATILITY_CONTEXT),
        )


def load_daily_summary_benchmark_frame_result(
    *,
    args: object,
    config: AppConfig,
    provider: DailyBarProvider,
    symbol_frames: Mapping[str, pd.DataFrame],
    fetch_start: date,
    universe_members: Sequence[UniverseMember],
    log_label: str,
) -> ResearchDataResult[BenchmarkFrameData]:
    if bool(getattr(args, "disable_regime_filter", False)) or not universe_members:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_daily_summary_benchmark_frame",
            provider_name=_provider_name(provider),
            status="unavailable",
            issue_code="not_requested",
            value=BenchmarkFrameData(symbol=None, frame=None),
        )

    benchmark_override = _benchmark_symbol_override(getattr(args, "benchmark_symbol", None))
    benchmark_symbol = benchmark_override or config.strategy.signals.benchmark_symbol
    if benchmark_symbol in symbol_frames:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_daily_summary_benchmark_frame",
            provider_name=_provider_name(provider),
            status="ok",
            value=BenchmarkFrameData(
                symbol=benchmark_symbol,
                frame=symbol_frames[benchmark_symbol],
            ),
        )

    try:
        benchmark_frame = provider.fetch_daily_bars(
            benchmark_symbol,
            fetch_start,
            getattr(args, "as_of"),
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
        )
    except DataProviderError as exc:
        return _degraded_result(
            role_name="historical_bars",
            operation="load_daily_summary_benchmark_frame",
            provider_name=_provider_name(provider),
            issue_code=_issue_code_for_error(exc),
            message=_log_missing_benchmark_context_warning(
                benchmark_symbol=benchmark_symbol,
                log_label=log_label,
                error=exc,
            ),
            value=BenchmarkFrameData(symbol=benchmark_symbol, frame=None),
        )
    if benchmark_frame.empty:
        return _degraded_result(
            role_name="historical_bars",
            operation="load_daily_summary_benchmark_frame",
            provider_name=_provider_name(provider),
            issue_code="empty_data",
            message=_log_missing_benchmark_context_warning(
                benchmark_symbol=benchmark_symbol,
                log_label=log_label,
                error=None,
            ),
            value=BenchmarkFrameData(symbol=benchmark_symbol, frame=None),
        )
    return ResearchDataResult(
        role_name="historical_bars",
        operation="load_daily_summary_benchmark_frame",
        provider_name=_provider_name(provider),
        status="ok",
        value=BenchmarkFrameData(symbol=benchmark_symbol, frame=benchmark_frame),
    )


def load_position_daily_symbol_frames_result(
    positions: Sequence[object],
    *,
    provider: DailyBarProvider,
    history_start: date,
    as_of_date: date,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[ReviewPositionDailyFramesData]:
    provider_name = _provider_name(provider)
    if not positions:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_position_daily_symbol_frames",
            provider_name=provider_name,
            status="ok",
            value=ReviewPositionDailyFramesData(history_start=history_start),
        )

    if not hasattr(provider, "fetch_daily_bars"):
        message = (
            "Daily-bar review context unavailable for "
            f"{log_label} because historical-bars provider '{provider_name or 'unknown'}' "
            "does not expose daily bar fetching; continuing without sector review context."
        )
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="historical_bars",
            operation="load_position_daily_symbol_frames",
            provider_name=provider_name,
            issue_code="unsupported_capability",
            message=message,
            value=ReviewPositionDailyFramesData(history_start=history_start),
        )

    provider_capabilities = getattr(provider, "capabilities", None)
    if (
        provider_capabilities is not None
        and not getattr(provider_capabilities, "supports_daily_bars", False)
    ):
        message = (
            "Daily-bar review context unavailable for "
            f"{log_label} because historical-bars provider '{provider_name or 'unknown'}' "
            "does not support daily bars; continuing without sector review context."
        )
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="historical_bars",
            operation="load_position_daily_symbol_frames",
            provider_name=provider_name,
            issue_code="unsupported_capability",
            message=message,
            value=ReviewPositionDailyFramesData(history_start=history_start),
        )

    symbol_frames: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[str] = []
    normalized_symbols = sorted(
        {
            str(getattr(position, "symbol", "")).strip().upper()
            for position in positions
            if str(getattr(position, "symbol", "")).strip()
        }
    )
    for symbol in normalized_symbols:
        try:
            bars = provider.fetch_daily_bars(
                symbol,
                history_start,
                as_of_date,
                refresh_cache=refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning(
                "Skipping sector context for held symbol %s due to provider error: %s",
                symbol,
                exc,
            )
            skipped_symbols.append(symbol)
            continue
        if bars.empty:
            skipped_symbols.append(symbol)
            continue
        symbol_frames[symbol] = bars

    result_value = ReviewPositionDailyFramesData(
        history_start=history_start,
        symbol_frames=symbol_frames,
        skipped_symbols=tuple(skipped_symbols),
    )
    if not symbol_frames and skipped_symbols:
        return _degraded_result(
            role_name="historical_bars",
            operation="load_position_daily_symbol_frames",
            provider_name=provider_name,
            issue_code="no_symbol_frames",
            message=(
                f"No usable held-symbol daily-bar history was available for {log_label}; "
                "continuing without sector review context."
            ),
            value=result_value,
        )
    if skipped_symbols:
        return _degraded_result(
            role_name="historical_bars",
            operation="load_position_daily_symbol_frames",
            provider_name=provider_name,
            issue_code="partial_symbol_frames",
            message=(
                f"Historical bars were unavailable for {len(skipped_symbols)} held symbols during "
                f"{log_label}; continuing with remaining sector review context."
            ),
            value=result_value,
        )
    return ResearchDataResult(
        role_name="historical_bars",
        operation="load_position_daily_symbol_frames",
        provider_name=provider_name,
        status="ok",
        value=result_value,
    )


def load_review_benchmark_frame_result(
    *,
    provider: DailyBarProvider,
    benchmark_symbol: str | None,
    fetch_start: date,
    as_of_date: date,
    refresh_cache: bool,
    log_label: str,
) -> ResearchDataResult[BenchmarkFrameData]:
    provider_name = _provider_name(provider)
    if benchmark_symbol is None:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_review_benchmark_frame",
            provider_name=provider_name,
            status="unavailable",
            issue_code="not_requested",
            value=BenchmarkFrameData(symbol=None, frame=None),
        )
    try:
        benchmark_frame = provider.fetch_daily_bars(
            benchmark_symbol,
            fetch_start,
            as_of_date,
            refresh_cache=refresh_cache,
        )
    except DataProviderError as exc:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_review_benchmark_frame",
            provider_name=provider_name,
            status="failed",
            issue_code=_issue_code_for_error(exc),
            message=_log_missing_benchmark_context_warning(
                benchmark_symbol=benchmark_symbol,
                log_label=log_label,
                error=exc,
            ),
            value=BenchmarkFrameData(symbol=benchmark_symbol, frame=None),
        )
    if benchmark_frame.empty:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_review_benchmark_frame",
            provider_name=provider_name,
            status="failed",
            issue_code="empty_data",
            message=_log_missing_benchmark_context_warning(
                benchmark_symbol=benchmark_symbol,
                log_label=log_label,
                error=None,
            ),
            value=BenchmarkFrameData(symbol=benchmark_symbol, frame=None),
        )
    return ResearchDataResult(
        role_name="historical_bars",
        operation="load_review_benchmark_frame",
        provider_name=provider_name,
        status="ok",
        value=BenchmarkFrameData(symbol=benchmark_symbol, frame=benchmark_frame),
    )


def load_intraday_benchmark_metrics_result(
    *,
    provider: DailyBarProvider,
    benchmark_symbol: str | None,
    as_of_date: date,
    interval_minutes: int,
    refresh_cache: bool,
    log_label: str,
    metrics_builder: Callable[[pd.DataFrame], Mapping[str, float | str | None]],
) -> ResearchDataResult[IntradayBenchmarkMetricsData]:
    provider_name = _provider_name(provider)
    if benchmark_symbol is None:
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_intraday_benchmark_metrics",
            provider_name=provider_name,
            status="unavailable",
            issue_code="not_requested",
            value=IntradayBenchmarkMetricsData(symbol=None, metrics=None),
        )
    if not hasattr(provider, "fetch_intraday_bars"):
        message = (
            "Intraday benchmark context unavailable for "
            f"{log_label} because historical-bars provider '{provider_name or 'unknown'}' "
            "does not expose intraday bar fetching."
        )
        LOGGER.warning("%s", message)
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_intraday_benchmark_metrics",
            provider_name=provider_name,
            status="failed",
            issue_code="unsupported_capability",
            message=message,
            value=IntradayBenchmarkMetricsData(symbol=benchmark_symbol, metrics=None),
        )
    try:
        benchmark_frame = provider.fetch_intraday_bars(
            benchmark_symbol,
            as_of_date,
            interval_minutes=interval_minutes,
            refresh_cache=refresh_cache,
        )
    except DataProviderError as exc:
        message = (
            f"Intraday benchmark context unavailable for {log_label}: {exc}"
        )
        LOGGER.warning("%s", message)
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_intraday_benchmark_metrics",
            provider_name=provider_name,
            status="failed",
            issue_code=_issue_code_for_error(exc),
            message=message,
            value=IntradayBenchmarkMetricsData(symbol=benchmark_symbol, metrics=None),
        )
    if benchmark_frame.empty:
        message = (
            "Intraday benchmark context unavailable for "
            f"{log_label} because no intraday bars were returned for regime symbol "
            f"'{benchmark_symbol}'; continuing without benchmark-relative intraday context."
        )
        LOGGER.warning("%s", message)
        return _degraded_result(
            role_name="historical_bars",
            operation="load_intraday_benchmark_metrics",
            provider_name=provider_name,
            issue_code="empty_data",
            message=message,
            value=IntradayBenchmarkMetricsData(symbol=benchmark_symbol, metrics=None),
        )
    try:
        metrics = dict(metrics_builder(benchmark_frame))
    except ValueError as exc:
        message = (
            f"Intraday benchmark context unavailable for {log_label}: {exc}"
        )
        LOGGER.warning("%s", message)
        return ResearchDataResult(
            role_name="historical_bars",
            operation="load_intraday_benchmark_metrics",
            provider_name=provider_name,
            status="failed",
            issue_code="provider_error",
            message=message,
            value=IntradayBenchmarkMetricsData(symbol=benchmark_symbol, metrics=None),
        )
    return ResearchDataResult(
        role_name="historical_bars",
        operation="load_intraday_benchmark_metrics",
        provider_name=provider_name,
        status="ok",
        value=IntradayBenchmarkMetricsData(symbol=benchmark_symbol, metrics=metrics),
    )


@dataclass
class DailySummaryResearchAssembler:
    """Assemble daily-summary research inputs from the configured provider bundle."""

    config: AppConfig
    provider_bundle: ProviderBundle
    benchmark_loader: Callable[..., ResearchDataResult[BenchmarkFrameData]] = (
        load_daily_summary_benchmark_frame_result
    )
    earnings_loader: Callable[..., ResearchDataResult[dict[str, EarningsRiskContext]]] = (
        load_earnings_contexts_result
    )
    position_classification_loader: Callable[
        ...,
        ResearchDataResult[dict[str, SymbolSectorClassification]],
    ] = load_position_sector_classifications_result
    sector_context_loader: Callable[..., ResearchDataResult[dict[str, SectorFeatureContext]]] = (
        load_sector_contexts_result
    )
    volatility_loader: Callable[
        ...,
        ResearchDataResult[dict[str, float | str | bool | None]],
    ] = load_volatility_context_result

    def assemble(
        self,
        *,
        args: object,
        current_positions: Sequence[object],
        fetch_start: date,
        log_label: str,
    ) -> DailySummaryResearchInputs:
        provider = self.provider_bundle.historical_bars_provider
        if provider is None:
            raise ValueError("Historical-bars provider bundle is unavailable.")
        candidate_data = self._assemble_candidate_data(
            args=args,
            provider=provider,
            fetch_start=fetch_start,
            log_label=log_label,
        )
        candidate_value = candidate_data.value or CandidateUniverseData(
            fetch_start=fetch_start,
            universe_members=(),
            symbol_frames={},
            skipped_symbols=(),
        )
        benchmark = self.benchmark_loader(
            args=args,
            config=self.config,
            provider=provider,
            symbol_frames=candidate_value.symbol_frames,
            fetch_start=fetch_start,
            universe_members=candidate_value.universe_members,
            log_label=log_label,
        )
        earnings_contexts = self.earnings_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=self.provider_bundle.earnings_calendar_provider,
            symbols=[member.symbol for member in candidate_value.universe_members],
            as_of_date=getattr(args, "as_of"),
            earnings_watch_days=self.config.strategy.signals.earnings_watch_days,
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        position_classifications = self.position_classification_loader(
            current_positions,
            config=self.config,
            env_file=getattr(args, "env_file", None),
            reference_provider=self.provider_bundle.reference_data_provider,
            as_of_date=getattr(args, "as_of"),
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        sector_contexts = self.sector_context_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=provider,
            reference_provider=self.provider_bundle.reference_data_provider,
            symbol_frames=candidate_value.symbol_frames,
            as_of_date=getattr(args, "as_of"),
            history_start=sector_context_history_start(
                getattr(args, "as_of"),
                benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
                relative_strength_window=(
                    self.config.strategy.signals.sector_relative_strength_window
                ),
            ),
            benchmark_sma_fast=self.config.strategy.signals.benchmark_sma_fast,
            benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=self.config.strategy.signals.sector_relative_strength_window,
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        volatility_context = self.volatility_loader(
            provider=provider,
            as_of_date=getattr(args, "as_of"),
            vix_caution_threshold=self.config.strategy.signals.vix_caution_threshold,
            vix_entry_block_threshold=self.config.strategy.signals.vix_entry_block_threshold,
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        return DailySummaryResearchInputs(
            candidate_data=candidate_data,
            benchmark=benchmark,
            earnings_contexts=earnings_contexts,
            position_classifications=position_classifications,
            sector_contexts=sector_contexts,
            volatility_context=volatility_context,
        )

    def _assemble_candidate_data(
        self,
        *,
        args: object,
        provider: DailyBarProvider,
        fetch_start: date,
        log_label: str,
    ) -> ResearchDataResult[CandidateUniverseData]:
        builder = UniverseBuilder(provider, self.config.strategy.universe)
        universe_members = tuple(
            builder.screen_candidates(
                getattr(args, "candidate_path"),
                as_of_date=getattr(args, "as_of"),
                lookback_days=int(getattr(args, "lookback_days")),
                refresh_cache=bool(getattr(args, "refresh_cache", False)),
                enforce_max_symbols=False,
            )
        )
        symbol_frames: dict[str, pd.DataFrame] = {}
        skipped_symbols: list[str] = []
        for member in universe_members:
            try:
                bars = provider.fetch_daily_bars(
                    member.symbol,
                    fetch_start,
                    getattr(args, "as_of"),
                    refresh_cache=bool(getattr(args, "refresh_cache", False)),
                )
            except DataProviderError as exc:
                LOGGER.warning("Skipping %s due to provider error: %s", member.symbol, exc)
                skipped_symbols.append(member.symbol)
                continue
            if bars.empty:
                LOGGER.warning(
                    "Skipping %s because no bars were returned in the requested range.",
                    member.symbol,
                )
                skipped_symbols.append(member.symbol)
                continue
            symbol_frames[member.symbol] = bars

        candidate_value = CandidateUniverseData(
            fetch_start=fetch_start,
            universe_members=universe_members,
            symbol_frames=symbol_frames,
            skipped_symbols=tuple(skipped_symbols),
        )
        if not universe_members:
            return ResearchDataResult(
                role_name="historical_bars",
                operation="screen_candidate_universe",
                provider_name=_provider_name(provider),
                status="unavailable",
                issue_code="empty_universe",
                value=candidate_value,
            )
        if not symbol_frames:
            return _degraded_result(
                role_name="historical_bars",
                operation="screen_candidate_universe",
                provider_name=_provider_name(provider),
                issue_code="no_symbol_frames",
                message=(
                    f"No usable daily-bar history was available for {log_label} after "
                    "screening the candidate universe."
                ),
                value=candidate_value,
            )
        if skipped_symbols:
            return _degraded_result(
                role_name="historical_bars",
                operation="screen_candidate_universe",
                provider_name=_provider_name(provider),
                issue_code="partial_symbol_frames",
                message=(
                    f"Historical bars were unavailable for {len(skipped_symbols)} screened "
                    f"symbols during {log_label}; continuing with remaining symbol frames."
                ),
                value=candidate_value,
            )
        return ResearchDataResult(
            role_name="historical_bars",
            operation="screen_candidate_universe",
            provider_name=_provider_name(provider),
            status="ok",
            value=candidate_value,
        )


@dataclass
class PortfolioReviewResearchAssembler:
    """Assemble shared portfolio-review research inputs from the provider bundle."""

    config: AppConfig
    provider_bundle: ProviderBundle
    benchmark_loader: Callable[..., ResearchDataResult[BenchmarkFrameData]] = (
        load_review_benchmark_frame_result
    )
    earnings_loader: Callable[..., ResearchDataResult[dict[str, EarningsRiskContext]]] = (
        load_earnings_contexts_result
    )
    position_daily_frames_loader: Callable[
        ...,
        ResearchDataResult[ReviewPositionDailyFramesData],
    ] = load_position_daily_symbol_frames_result
    sector_context_loader: Callable[..., ResearchDataResult[dict[str, SectorFeatureContext]]] = (
        load_sector_contexts_result
    )

    def assemble(
        self,
        *,
        args: object,
        current_positions: Sequence[object],
        position_plans: Sequence[Mapping[str, object]],
        log_label: str,
    ) -> PortfolioReviewResearchInputs:
        provider = self.provider_bundle.historical_bars_provider
        if provider is None:
            raise ValueError("Historical-bars provider bundle is unavailable.")
        benchmark_symbol, benchmark_fetch_start = _portfolio_review_benchmark_request(
            position_plans
        )
        benchmark = self.benchmark_loader(
            provider=provider,
            benchmark_symbol=benchmark_symbol,
            fetch_start=benchmark_fetch_start or getattr(args, "as_of"),
            as_of_date=getattr(args, "as_of"),
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        earnings_contexts = self.earnings_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=self.provider_bundle.earnings_calendar_provider,
            symbols=[str(getattr(position, "symbol", "")) for position in current_positions],
            as_of_date=getattr(args, "as_of"),
            earnings_watch_days=self.config.strategy.signals.earnings_watch_days,
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        history_start = sector_context_history_start(
            getattr(args, "as_of"),
            benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=self.config.strategy.signals.sector_relative_strength_window,
        )
        position_daily_symbol_frames = self.position_daily_frames_loader(
            current_positions,
            provider=provider,
            history_start=history_start,
            as_of_date=getattr(args, "as_of"),
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        daily_symbol_frames = (
            position_daily_symbol_frames.value.symbol_frames
            if position_daily_symbol_frames.value is not None
            else {}
        )
        sector_contexts = self.sector_context_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=provider,
            reference_provider=self.provider_bundle.reference_data_provider,
            symbol_frames=daily_symbol_frames,
            as_of_date=getattr(args, "as_of"),
            history_start=history_start,
            benchmark_sma_fast=self.config.strategy.signals.benchmark_sma_fast,
            benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=self.config.strategy.signals.sector_relative_strength_window,
            refresh_cache=bool(getattr(args, "refresh_cache", False)),
            log_label=log_label,
        )
        return PortfolioReviewResearchInputs(
            benchmark=benchmark,
            earnings_contexts=earnings_contexts,
            position_daily_symbol_frames=position_daily_symbol_frames,
            sector_contexts=sector_contexts,
        )


@dataclass
class IntradayReviewResearchAssembler:
    """Assemble shared intraday-review research inputs from the provider bundle."""

    config: AppConfig
    provider_bundle: ProviderBundle
    benchmark_loader: Callable[..., ResearchDataResult[IntradayBenchmarkMetricsData]] = (
        load_intraday_benchmark_metrics_result
    )
    earnings_loader: Callable[..., ResearchDataResult[dict[str, EarningsRiskContext]]] = (
        load_earnings_contexts_result
    )
    position_daily_frames_loader: Callable[
        ...,
        ResearchDataResult[ReviewPositionDailyFramesData],
    ] = load_position_daily_symbol_frames_result
    sector_context_loader: Callable[..., ResearchDataResult[dict[str, SectorFeatureContext]]] = (
        load_sector_contexts_result
    )
    intraday_metrics_builder: Callable[[pd.DataFrame], Mapping[str, float | str | None]] | None = None

    def assemble(
        self,
        *,
        args: object,
        current_positions: Sequence[object],
        benchmark_symbol: str | None,
        log_label: str,
    ) -> IntradayReviewResearchInputs:
        provider = self.provider_bundle.historical_bars_provider
        if provider is None:
            raise ValueError("Historical-bars provider bundle is unavailable.")
        if self.intraday_metrics_builder is None:
            raise ValueError("intraday_metrics_builder must be configured.")
        benchmark_intraday_metrics = self.benchmark_loader(
            provider=provider,
            benchmark_symbol=benchmark_symbol,
            as_of_date=getattr(args, "as_of"),
            interval_minutes=int(getattr(args, "interval_minutes")),
            refresh_cache=True,
            log_label=log_label,
            metrics_builder=self.intraday_metrics_builder,
        )
        earnings_contexts = self.earnings_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=self.provider_bundle.earnings_calendar_provider,
            symbols=[str(getattr(position, "symbol", "")) for position in current_positions],
            as_of_date=getattr(args, "as_of"),
            earnings_watch_days=self.config.strategy.signals.earnings_watch_days,
            refresh_cache=False,
            log_label=log_label,
        )
        history_start = sector_context_history_start(
            getattr(args, "as_of"),
            benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=self.config.strategy.signals.sector_relative_strength_window,
        )
        position_daily_symbol_frames = self.position_daily_frames_loader(
            current_positions,
            provider=provider,
            history_start=history_start,
            as_of_date=getattr(args, "as_of"),
            refresh_cache=False,
            log_label=log_label,
        )
        daily_symbol_frames = (
            position_daily_symbol_frames.value.symbol_frames
            if position_daily_symbol_frames.value is not None
            else {}
        )
        sector_contexts = self.sector_context_loader(
            config=self.config,
            env_file=getattr(args, "env_file", None),
            provider=provider,
            reference_provider=self.provider_bundle.reference_data_provider,
            symbol_frames=daily_symbol_frames,
            as_of_date=getattr(args, "as_of"),
            history_start=history_start,
            benchmark_sma_fast=self.config.strategy.signals.benchmark_sma_fast,
            benchmark_sma_slow=self.config.strategy.signals.benchmark_sma_slow,
            relative_strength_window=self.config.strategy.signals.sector_relative_strength_window,
            refresh_cache=False,
            log_label=log_label,
        )
        return IntradayReviewResearchInputs(
            benchmark_intraday_metrics=benchmark_intraday_metrics,
            earnings_contexts=earnings_contexts,
            position_daily_symbol_frames=position_daily_symbol_frames,
            sector_contexts=sector_contexts,
        )


def _provider_name(provider: object | None) -> str | None:
    provider_name = getattr(provider, "provider_name", None)
    return provider_name if isinstance(provider_name, str) and provider_name else None


def _issue_code_for_error(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    if provider_error_is_entitlement_limited(exc):
        return "entitlement_limited"
    if provider_error_is_timeout(exc):
        return "timeout"
    return "provider_error"


def _degraded_result(
    *,
    role_name: str,
    operation: str,
    provider_name: str | None,
    issue_code: str | None,
    message: str | None,
    value: T,
) -> ResearchDataResult[T]:
    return ResearchDataResult(
        role_name=role_name,
        operation=operation,
        provider_name=provider_name,
        status="degraded",
        issue_code=issue_code,
        message=message,
        value=value,
    )


def _log_earnings_context_warning(*, log_label: str, error: BaseException) -> str:
    if provider_error_is_entitlement_limited(error):
        message = (
            "Earnings context unavailable for "
            f"{log_label} because the configured provider is not entitled to "
            f"earnings-calendar data; continuing without earnings gate. Original error: {error}"
        )
    elif provider_error_is_timeout(error):
        message = (
            "Earnings context unavailable for "
            f"{log_label} due to provider timeout; continuing without earnings gate. "
            f"Original error: {error}"
        )
    else:
        message = (
            f"Earnings context unavailable for {log_label}; continuing without "
            f"earnings gate: {error}"
        )
    LOGGER.warning("%s", message)
    return message


def _log_missing_benchmark_context_warning(
    *,
    benchmark_symbol: str,
    log_label: str,
    error: BaseException | None,
) -> str:
    if error is None:
        message = (
            "Benchmark context unavailable for "
            f"{log_label} because no data was returned for regime symbol '{benchmark_symbol}'; "
            "proceeding without regime filter."
        )
    elif provider_error_is_entitlement_limited(error):
        message = (
            "Benchmark context unavailable for "
            f"{log_label} because the configured provider is not entitled to regime symbol "
            f"'{benchmark_symbol}'; proceeding without regime filter. Original error: {error}"
        )
    elif provider_error_is_timeout(error):
        message = (
            "Benchmark context unavailable for "
            f"{log_label} because regime symbol '{benchmark_symbol}' timed out; "
            f"proceeding without regime filter. Original error: {error}"
        )
    else:
        message = (
            "Benchmark context unavailable for "
            f"{log_label} because regime symbol '{benchmark_symbol}' could not be fetched; "
            f"proceeding without regime filter. Original error: {error}"
        )
    LOGGER.warning("%s", message)
    return message


def _benchmark_symbol_override(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().upper()
    return cleaned or None


def _portfolio_review_benchmark_request(
    position_plans: Sequence[Mapping[str, object]],
) -> tuple[str | None, date | None]:
    benchmark_symbol: str | None = None
    earliest_fetch_start: date | None = None
    for plan in position_plans:
        settings = plan.get("settings")
        if not getattr(settings, "enable_regime_filter", False):
            continue
        candidate_symbol = _optional_benchmark_symbol(getattr(settings, "benchmark_symbol", None))
        if candidate_symbol is not None and benchmark_symbol is None:
            benchmark_symbol = candidate_symbol
        fetch_start = plan.get("fetch_start")
        if isinstance(fetch_start, date):
            earliest_fetch_start = (
                fetch_start
                if earliest_fetch_start is None or fetch_start < earliest_fetch_start
                else earliest_fetch_start
            )
    return benchmark_symbol, earliest_fetch_start


def _optional_benchmark_symbol(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None
    cleaned = raw_value.strip().upper()
    return cleaned or None
