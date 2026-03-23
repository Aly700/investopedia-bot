"""Streaming ingestion adapters and websocket-backed buffering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

from bot.ingestion.market_data import (
    MARKET_DATA_INGESTION_UPDATE_TYPES,
    MarketCycleIngestionEnvelope,
    MarketDataIngestionUpdate,
    required_ingestion_update_types,
    sort_ingestion_updates,
)
from bot.logging_utils import get_logger
from bot.reporting.daily_report import (
    IntradayPortfolioReviewReport,
    PortfolioReviewReport,
)
from bot.risk.portfolio_rules import ExistingPosition


if TYPE_CHECKING:
    from bot.orchestration.live_runner import LiveMarketCycleRequest


LOGGER = get_logger(__name__)

STREAM_CONTROL_EVENT_TYPES = (
    "disconnect",
    "reconnect",
)
STREAMING_MARKET_EVENT_TYPES = MARKET_DATA_INGESTION_UPDATE_TYPES + STREAM_CONTROL_EVENT_TYPES


@dataclass(frozen=True)
class StreamingMarketDataEvent:
    """One normalized streaming event before buffer coalescing."""

    event_type: str
    as_of_date: date | None = None
    symbols: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in STREAMING_MARKET_EVENT_TYPES:
            raise ValueError(
                "event_type must be one of "
                f"{STREAMING_MARKET_EVENT_TYPES}, got '{self.event_type}'."
            )
        if self.event_type in MARKET_DATA_INGESTION_UPDATE_TYPES and self.as_of_date is None:
            raise ValueError(
                f"as_of_date is required for streaming update event '{self.event_type}'."
            )
        if self.event_type in STREAM_CONTROL_EVENT_TYPES:
            object.__setattr__(self, "symbols", ())
        object.__setattr__(self, "metadata", dict(sorted(self.metadata.items())))

    @property
    def is_control_event(self) -> bool:
        return self.event_type in STREAM_CONTROL_EVENT_TYPES

    def to_ingestion_update(self) -> MarketDataIngestionUpdate | None:
        if self.is_control_event:
            return None
        if self.as_of_date is None:
            raise ValueError(
                f"Streaming update event '{self.event_type}' is missing as_of_date."
            )
        return MarketDataIngestionUpdate(
            update_type=self.event_type,
            as_of_date=self.as_of_date,
            symbols=self.symbols,
            metadata=self.metadata,
        )

    @classmethod
    def from_message(
        cls,
        message: Mapping[str, Any],
    ) -> "StreamingMarketDataEvent":
        """Normalize one transport message into a streaming event."""

        if not isinstance(message, Mapping):
            raise TypeError("stream message must be a mapping.")
        raw_type = message.get("type")
        if not isinstance(raw_type, str) or not raw_type.strip():
            raise ValueError("stream message must include a non-empty 'type'.")
        event_type = raw_type.strip()
        raw_metadata = message.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("stream message 'metadata' must be a mapping when provided.")
        metadata = dict(raw_metadata)
        if event_type in MARKET_DATA_INGESTION_UPDATE_TYPES:
            update = MarketDataIngestionUpdate(
                update_type=event_type,
                as_of_date=_coerce_message_date(message.get("as_of_date")),
                symbols=tuple(message.get("symbols", ())),
                metadata=metadata,
            )
            return cls(
                event_type=event_type,
                as_of_date=update.as_of_date,
                symbols=update.symbols,
                metadata=update.metadata,
            )
        if event_type in STREAM_CONTROL_EVENT_TYPES:
            return cls(
                event_type=event_type,
                as_of_date=_coerce_optional_message_date(message.get("as_of_date")),
                metadata=metadata,
            )
        raise ValueError(
            "stream message type must be one of "
            f"{STREAMING_MARKET_EVENT_TYPES}, got '{event_type}'."
        )


@dataclass(frozen=True)
class StreamingIngestionPollResult:
    """Summary of one websocket poll operation."""

    received_count: int
    accepted_count: int
    skipped_count: int
    warnings: tuple[str, ...] = ()
    connected: bool = False

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


@dataclass(frozen=True)
class LiveUpdateBufferSnapshot:
    """Immutable buffer view captured for one emitted runner cycle."""

    as_of_date: date
    updates: tuple[MarketDataIngestionUpdate, ...]
    connected: bool
    last_event_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "connected": self.connected,
            "last_event_at_utc": self.last_event_at_utc,
            "update_count": len(self.updates),
            "update_types": [update.update_type for update in self.updates],
        }


@dataclass
class LiveUpdateBuffer:
    """Coalesce high-frequency stream updates into cycle-level batches."""

    latest_updates: dict[str, MarketDataIngestionUpdate] = field(default_factory=dict)
    dirty_since_utc: datetime | None = None
    last_event_at_utc: datetime | None = None
    last_flush_at_utc: datetime | None = None

    def add_update(
        self,
        update: MarketDataIngestionUpdate,
        *,
        observed_at_utc: datetime | None = None,
    ) -> None:
        observed_at = observed_at_utc or _utcnow()
        existing = self.latest_updates.get(update.update_type)
        merged_update = _merge_updates(existing, update)
        self.latest_updates[update.update_type] = merged_update
        self.last_event_at_utc = observed_at
        if self.dirty_since_utc is None:
            self.dirty_since_utc = observed_at

    def current_updates(
        self,
        *,
        as_of_date: date,
    ) -> tuple[MarketDataIngestionUpdate, ...]:
        return sort_ingestion_updates(
            [
                update
                for update in self.latest_updates.values()
                if update.as_of_date == as_of_date
            ]
        )

    def flush_due(
        self,
        *,
        as_of_date: date,
        now_utc: datetime,
        flush_interval: timedelta,
    ) -> bool:
        if self.dirty_since_utc is None:
            return False
        if not any(update.as_of_date == as_of_date for update in self.latest_updates.values()):
            return False
        return now_utc >= self.dirty_since_utc + flush_interval

    def mark_flushed(
        self,
        *,
        flushed_at_utc: datetime,
    ) -> None:
        self.last_flush_at_utc = flushed_at_utc
        self.dirty_since_utc = None


class WebsocketMessageTransport(Protocol):
    """Transport abstraction for websocket-style message polling."""

    def connect(self) -> None:
        """Connect the underlying stream transport."""

    def disconnect(self) -> None:
        """Disconnect the underlying stream transport."""

    def read_messages(
        self,
        *,
        max_messages: int | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Return the currently available transport messages."""


@dataclass
class WebsocketIngestionAdapter:
    """Streaming adapter that buffers websocket updates into runner envelopes."""

    transport_name: str
    transport: WebsocketMessageTransport
    portfolio_path: str | None
    load_current_positions: Callable[[], Sequence[ExistingPosition]]
    run_daily_summary: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], Mapping[str, object]] | None = None
    run_portfolio_review: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], PortfolioReviewReport] | None = None
    run_intraday_review: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], IntradayPortfolioReviewReport] | None = None
    provider_name: str | None = None
    flush_interval_seconds: int = 5
    _buffer: LiveUpdateBuffer = field(default_factory=LiveUpdateBuffer, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.transport_name.strip():
            raise ValueError("transport_name cannot be empty.")
        if self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be greater than zero.")
        if self.portfolio_path is not None and not self.portfolio_path.strip():
            raise ValueError("portfolio_path cannot be an empty string.")
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

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self.transport.connect()
        self._connected = True

    def disconnect(self) -> None:
        self.transport.disconnect()
        self._connected = False

    def poll_messages(
        self,
        *,
        max_messages: int | None = None,
        observed_at_utc: datetime | None = None,
    ) -> StreamingIngestionPollResult:
        """Read, normalize, and buffer currently available websocket messages."""

        observed_at = observed_at_utc or _utcnow()
        try:
            messages = self.transport.read_messages(max_messages=max_messages)
        except OSError as exc:
            self._connected = False
            warning = f"websocket transport disconnected: {exc}"
            LOGGER.warning("%s", warning)
            return StreamingIngestionPollResult(
                received_count=0,
                accepted_count=0,
                skipped_count=0,
                warnings=(warning,),
                connected=self._connected,
            )

        received_count = 0
        accepted_count = 0
        skipped_count = 0
        warnings: list[str] = []
        for message in messages:
            received_count += 1
            try:
                event = StreamingMarketDataEvent.from_message(message)
            except (TypeError, ValueError) as exc:
                skipped_count += 1
                warning = f"skipped malformed stream message: {exc}"
                warnings.append(warning)
                LOGGER.warning("%s", warning)
                continue
            accepted_count += 1
            if event.event_type == "disconnect":
                self._connected = False
                continue
            if event.event_type == "reconnect":
                self._connected = True
                continue
            update = event.to_ingestion_update()
            if update is None:
                continue
            self._buffer.add_update(update, observed_at_utc=observed_at)
        return StreamingIngestionPollResult(
            received_count=received_count,
            accepted_count=accepted_count,
            skipped_count=skipped_count,
            warnings=tuple(warnings),
            connected=self._connected,
        )

    def flush_ready_cycle_ingestion(
        self,
        request: LiveMarketCycleRequest,
        *,
        now_utc: datetime | None = None,
    ) -> MarketCycleIngestionEnvelope | None:
        """Return the next periodic flush envelope when the buffer is ready."""

        now = now_utc or _utcnow()
        return self._build_cycle_ingestion(request, now_utc=now, force=False)

    def build_cycle_ingestion(
        self,
        request: LiveMarketCycleRequest,
    ) -> MarketCycleIngestionEnvelope:
        """Force-build a cycle envelope from the current buffered stream state."""

        envelope = self._build_cycle_ingestion(
            request,
            now_utc=_utcnow(),
            force=True,
        )
        if envelope is None:
            raise ValueError("No buffered streaming updates are ready for cycle ingestion.")
        return envelope

    def has_pending_updates(
        self,
        *,
        as_of_date: date,
    ) -> bool:
        """Return whether the buffer is currently dirty for the requested session date."""

        if self._buffer.dirty_since_utc is None:
            return False
        return bool(self._buffer.current_updates(as_of_date=as_of_date))

    def _build_cycle_ingestion(
        self,
        request: LiveMarketCycleRequest,
        *,
        now_utc: datetime,
        force: bool,
    ) -> MarketCycleIngestionEnvelope | None:
        updates = self._buffer.current_updates(as_of_date=request.as_of_date)
        if not updates:
            if force:
                raise ValueError(
                    f"No buffered streaming updates are available for {request.as_of_date.isoformat()}."
                )
            return None
        required_types = required_ingestion_update_types(
            include_daily_summary=request.include_daily_summary,
            include_portfolio_review=request.include_portfolio_review,
            include_intraday_review=request.include_intraday_review,
        )
        available_types = {update.update_type for update in updates}
        missing_types = tuple(
            update_type
            for update_type in required_types
            if update_type not in available_types
        )
        if missing_types:
            if force:
                raise ValueError(
                    "Streaming ingestion buffer is missing required update types: "
                    + ", ".join(missing_types)
                    + "."
                )
            return None
        if (
            not force
            and not self._buffer.flush_due(
                as_of_date=request.as_of_date,
                now_utc=now_utc,
                flush_interval=timedelta(seconds=self.flush_interval_seconds),
            )
        ):
            return None

        current_positions = tuple(self.load_current_positions())
        snapshot = LiveUpdateBufferSnapshot(
            as_of_date=request.as_of_date,
            updates=updates,
            connected=self._connected,
            last_event_at_utc=(
                self._buffer.last_event_at_utc.astimezone(timezone.utc).isoformat()
                if self._buffer.last_event_at_utc is not None
                else None
            ),
        )
        resolved_portfolio_path = request.portfolio_path or self.portfolio_path
        envelope = MarketCycleIngestionEnvelope(
            adapter_name="websocket",
            ingestion_mode="streaming",
            as_of_date=request.as_of_date,
            portfolio_path=resolved_portfolio_path,
            provider_name=self.provider_name or self.transport_name,
            current_positions=current_positions,
            updates=updates,
            run_daily_summary=(
                _streaming_daily_summary_callback(
                    self.run_daily_summary,
                    snapshot=snapshot,
                    current_positions=current_positions,
                )
                if request.include_daily_summary
                else None
            ),
            run_portfolio_review=(
                _streaming_portfolio_review_callback(
                    self.run_portfolio_review,
                    snapshot=snapshot,
                    current_positions=current_positions,
                )
                if request.include_portfolio_review
                else None
            ),
            run_intraday_review=(
                _streaming_intraday_review_callback(
                    self.run_intraday_review,
                    snapshot=snapshot,
                    current_positions=current_positions,
                )
                if request.include_intraday_review
                else None
            ),
        )
        self._buffer.mark_flushed(flushed_at_utc=now_utc)
        return envelope


def _merge_updates(
    existing: MarketDataIngestionUpdate | None,
    incoming: MarketDataIngestionUpdate,
) -> MarketDataIngestionUpdate:
    if existing is None or existing.as_of_date != incoming.as_of_date:
        return incoming
    merged_symbols = tuple(sorted(set(existing.symbols) | set(incoming.symbols)))
    merged_metadata = dict(existing.metadata)
    merged_metadata.update(incoming.metadata)
    return MarketDataIngestionUpdate(
        update_type=incoming.update_type,
        as_of_date=incoming.as_of_date,
        symbols=merged_symbols,
        metadata=merged_metadata,
    )


def _streaming_daily_summary_callback(
    callback: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], Mapping[str, object]] | None,
    *,
    snapshot: LiveUpdateBufferSnapshot,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], Mapping[str, object]] | None:
    if callback is None:
        return None

    def run() -> Mapping[str, object]:
        return callback(snapshot, list(current_positions))

    return run


def _streaming_portfolio_review_callback(
    callback: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], PortfolioReviewReport] | None,
    *,
    snapshot: LiveUpdateBufferSnapshot,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], PortfolioReviewReport] | None:
    if callback is None:
        return None

    def run() -> PortfolioReviewReport:
        return callback(snapshot, list(current_positions))

    return run


def _streaming_intraday_review_callback(
    callback: Callable[[LiveUpdateBufferSnapshot, list[ExistingPosition]], IntradayPortfolioReviewReport] | None,
    *,
    snapshot: LiveUpdateBufferSnapshot,
    current_positions: Sequence[ExistingPosition],
) -> Callable[[], IntradayPortfolioReviewReport] | None:
    if callback is None:
        return None

    def run() -> IntradayPortfolioReviewReport:
        return callback(snapshot, list(current_positions))

    return run


def _coerce_message_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("stream message 'as_of_date' must be a date or ISO 8601 date string.")


def _coerce_optional_message_date(value: object) -> date | None:
    if value is None:
        return None
    return _coerce_message_date(value)


def _validate_optional_callable(
    value: object | None,
    *,
    field_name: str,
) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{field_name} must be callable when provided.")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
