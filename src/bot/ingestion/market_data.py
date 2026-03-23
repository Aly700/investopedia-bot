"""Typed market-data ingestion adapters for live market cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

from bot.reporting.daily_report import IntradayPortfolioReviewReport, PortfolioReviewReport
from bot.risk.portfolio_rules import ExistingPosition


if TYPE_CHECKING:
    from bot.orchestration.live_runner import LiveMarketCycleRequest


MARKET_DATA_INGESTION_UPDATE_TYPES = (
    "benchmark_context_refresh",
    "candidate_universe_refresh_batch",
    "market_context_refresh_batch",
    "portfolio_symbol_refresh_batch",
    "intraday_portfolio_refresh_batch",
    "quote_bar_batch_refresh",
)


@dataclass(frozen=True)
class MarketDataIngestionUpdate:
    """One structured ingress update describing what was refreshed for a cycle."""

    update_type: str
    as_of_date: date
    symbols: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.update_type not in MARKET_DATA_INGESTION_UPDATE_TYPES:
            raise ValueError(
                "update_type must be one of "
                f"{MARKET_DATA_INGESTION_UPDATE_TYPES}, got '{self.update_type}'."
            )
        if isinstance(self.symbols, str):
            raise TypeError("symbols must be an iterable of symbol strings, not a single string.")
        validated_symbols: list[str] = []
        for index, symbol in enumerate(self.symbols):
            if not isinstance(symbol, str):
                raise TypeError(f"symbols[{index}] must be a string.")
            if not symbol.strip():
                raise ValueError(f"symbols[{index}] cannot be blank.")
            validated_symbols.append(symbol)
        normalized_symbols = tuple(
            sorted(
                {
                    symbol.strip().upper()
                    for symbol in validated_symbols
                }
            )
        )
        object.__setattr__(self, "symbols", normalized_symbols)
        object.__setattr__(self, "metadata", dict(sorted(self.metadata.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_type": self.update_type,
            "as_of_date": self.as_of_date.isoformat(),
            "symbol_count": len(self.symbols),
            "symbols": list(self.symbols),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MarketCycleIngestionEnvelope:
    """Typed ingress bundle consumed by the live market runner for one cycle."""

    adapter_name: str
    ingestion_mode: str
    as_of_date: date
    portfolio_path: str | None = None
    provider_name: str | None = None
    current_positions: tuple[ExistingPosition, ...] = ()
    updates: tuple[MarketDataIngestionUpdate, ...] = ()
    run_daily_summary: Callable[[], Mapping[str, object]] | None = None
    run_portfolio_review: Callable[[], PortfolioReviewReport] | None = None
    run_intraday_review: Callable[[], IntradayPortfolioReviewReport] | None = None

    def __post_init__(self) -> None:
        if not self.adapter_name.strip():
            raise ValueError("adapter_name cannot be empty.")
        if self.ingestion_mode not in {"polling", "streaming", "event_queue"}:
            raise ValueError(
                "ingestion_mode must be one of ('polling', 'streaming', 'event_queue')."
            )
        if self.portfolio_path is not None and not self.portfolio_path.strip():
            raise ValueError("portfolio_path cannot be an empty string.")
        for index, position in enumerate(self.current_positions):
            if not isinstance(position, ExistingPosition):
                raise TypeError(
                    f"current_positions[{index}] must be an ExistingPosition."
                )
        for index, update in enumerate(self.updates):
            if not isinstance(update, MarketDataIngestionUpdate):
                raise TypeError(
                    f"updates[{index}] must be a MarketDataIngestionUpdate."
                )
        _validate_optional_callable(
            self.run_daily_summary,
            field_name="run_daily_summary",
        )
        _validate_optional_callable(
            self.run_portfolio_review,
            field_name="run_portfolio_review",
        )
        _validate_optional_callable(
            self.run_intraday_review,
            field_name="run_intraday_review",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "ingestion_mode": self.ingestion_mode,
            "as_of_date": self.as_of_date.isoformat(),
            "portfolio_path": self.portfolio_path,
            "provider_name": self.provider_name,
            "current_position_symbols": [position.symbol for position in self.current_positions],
            "update_count": len(self.updates),
            "updates": [update.to_dict() for update in self.updates],
        }


class MarketDataIngestionAdapter(Protocol):
    """Interface for cycle-level market data ingress."""

    def build_cycle_ingestion(
        self,
        request: LiveMarketCycleRequest,
    ) -> MarketCycleIngestionEnvelope:
        """Return the typed ingestion envelope for one live market cycle."""


def required_ingestion_update_types(
    *,
    include_daily_summary: bool,
    include_portfolio_review: bool,
    include_intraday_review: bool,
) -> tuple[str, ...]:
    """Return the minimum update types required for a request shape."""

    required: list[str] = []
    if include_daily_summary:
        required.append("candidate_universe_refresh_batch")
    if include_portfolio_review or include_intraday_review:
        required.append("portfolio_symbol_refresh_batch")
    return tuple(required)


def sort_ingestion_updates(
    updates: Sequence[MarketDataIngestionUpdate],
) -> tuple[MarketDataIngestionUpdate, ...]:
    """Return updates in the shared deterministic adapter order."""

    order = {
        update_type: index
        for index, update_type in enumerate(MARKET_DATA_INGESTION_UPDATE_TYPES)
    }
    return tuple(
        sorted(
            updates,
            key=lambda update: (
                order.get(update.update_type, len(order)),
                update.as_of_date.isoformat(),
                update.symbols,
            ),
        )
    )


@dataclass(frozen=True)
class PollingIngestionAdapter:
    """Polling-backed adapter that packages one cycle's ingress state."""

    provider_name: str | None
    portfolio_path: str | None
    load_current_positions: Callable[[], Sequence[ExistingPosition]]
    create_provider: Callable[[], object | None]
    run_daily_summary: Callable[[object, list[ExistingPosition]], Mapping[str, object]] | None = None
    run_portfolio_review: Callable[[object | None, list[ExistingPosition]], PortfolioReviewReport] | None = None
    run_intraday_review: Callable[[object | None, list[ExistingPosition]], IntradayPortfolioReviewReport] | None = None
    candidate_path: str | None = None
    benchmark_symbol: str | None = None
    refresh_cache: bool | None = None
    interval_minutes: int | None = None

    def build_cycle_ingestion(
        self,
        request: LiveMarketCycleRequest,
    ) -> MarketCycleIngestionEnvelope:
        """Build the polling-backed ingress envelope for one cycle."""

        current_positions = list(self.load_current_positions())
        provider_required = bool(
            request.include_daily_summary
            or (
                (request.include_portfolio_review or request.include_intraday_review)
                and current_positions
            )
        )
        provider = self.create_provider() if provider_required else None
        if request.include_daily_summary and provider is None:
            raise ValueError(
                "Polling ingestion requires a provider when daily_summary is included."
            )
        if (
            provider is None
            and current_positions
            and (request.include_portfolio_review or request.include_intraday_review)
        ):
            raise ValueError(
                "Polling ingestion requires a provider for non-empty portfolio review cycles."
            )

        resolved_portfolio_path = request.portfolio_path or self.portfolio_path
        updates = _build_polling_updates(
            request=request,
            portfolio_path=resolved_portfolio_path,
            current_positions=current_positions,
            candidate_path=self.candidate_path,
            benchmark_symbol=self.benchmark_symbol,
            refresh_cache=self.refresh_cache,
            interval_minutes=self.interval_minutes,
            provider_name=self.provider_name,
        )
        current_positions_snapshot = tuple(current_positions)
        return MarketCycleIngestionEnvelope(
            adapter_name="polling",
            ingestion_mode="polling",
            as_of_date=request.as_of_date,
            portfolio_path=resolved_portfolio_path,
            provider_name=self.provider_name,
            current_positions=current_positions_snapshot,
            updates=updates,
            run_daily_summary=(
                _daily_summary_callback(
                    self.run_daily_summary,
                    provider=provider,
                    current_positions=current_positions_snapshot,
                )
                if request.include_daily_summary
                else None
            ),
            run_portfolio_review=(
                _portfolio_review_callback(
                    self.run_portfolio_review,
                    provider=provider,
                    current_positions=current_positions_snapshot,
                )
                if request.include_portfolio_review
                else None
            ),
            run_intraday_review=(
                _intraday_review_callback(
                    self.run_intraday_review,
                    provider=provider,
                    current_positions=current_positions_snapshot,
                )
                if request.include_intraday_review
                else None
            ),
        )


def _build_polling_updates(
    *,
    request: LiveMarketCycleRequest,
    portfolio_path: str | None,
    current_positions: Sequence[ExistingPosition],
    candidate_path: str | None,
    benchmark_symbol: str | None,
    refresh_cache: bool | None,
    interval_minutes: int | None,
    provider_name: str | None,
) -> tuple[MarketDataIngestionUpdate, ...]:
    updates: list[MarketDataIngestionUpdate] = []
    if benchmark_symbol is not None:
        updates.append(
            MarketDataIngestionUpdate(
                update_type="benchmark_context_refresh",
                as_of_date=request.as_of_date,
                symbols=(benchmark_symbol,),
                metadata={"provider_name": provider_name},
            )
        )
    if request.include_daily_summary:
        updates.append(
            MarketDataIngestionUpdate(
                update_type="candidate_universe_refresh_batch",
                as_of_date=request.as_of_date,
                metadata={
                    "candidate_path": candidate_path,
                    "refresh_cache": refresh_cache,
                    "provider_name": provider_name,
                },
            )
        )
        updates.append(
            MarketDataIngestionUpdate(
                update_type="market_context_refresh_batch",
                as_of_date=request.as_of_date,
                metadata={
                    "benchmark_symbol": benchmark_symbol,
                    "refresh_cache": refresh_cache,
                    "provider_name": provider_name,
                },
            )
        )
    if request.include_portfolio_review or request.include_intraday_review:
        updates.append(
            MarketDataIngestionUpdate(
                update_type="portfolio_symbol_refresh_batch",
                as_of_date=request.as_of_date,
                symbols=tuple(position.symbol for position in current_positions),
                metadata={
                    "portfolio_path": portfolio_path,
                    "provider_name": provider_name,
                },
            )
        )
    if request.include_intraday_review:
        updates.append(
            MarketDataIngestionUpdate(
                update_type="intraday_portfolio_refresh_batch",
                as_of_date=request.as_of_date,
                symbols=tuple(position.symbol for position in current_positions),
                metadata={
                    "portfolio_path": portfolio_path,
                    "interval_minutes": interval_minutes,
                    "provider_name": provider_name,
                },
            )
        )
    return tuple(updates)


def _daily_summary_callback(
    callback: Callable[[object, list[ExistingPosition]], Mapping[str, object]] | None,
    *,
    provider: object | None,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], Mapping[str, object]] | None:
    if callback is None:
        return None

    def run() -> Mapping[str, object]:
        return callback(provider, list(current_positions))

    return run


def _portfolio_review_callback(
    callback: Callable[[object | None, list[ExistingPosition]], PortfolioReviewReport] | None,
    *,
    provider: object | None,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], PortfolioReviewReport] | None:
    if callback is None:
        return None

    def run() -> PortfolioReviewReport:
        return callback(provider, list(current_positions))

    return run


def _intraday_review_callback(
    callback: Callable[[object | None, list[ExistingPosition]], IntradayPortfolioReviewReport] | None,
    *,
    provider: object | None,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], IntradayPortfolioReviewReport] | None:
    if callback is None:
        return None

    def run() -> IntradayPortfolioReviewReport:
        return callback(provider, list(current_positions))

    return run


def _validate_optional_callable(
    value: object | None,
    *,
    field_name: str,
) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable when provided.")
