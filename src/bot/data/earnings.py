"""Cached earnings-calendar providers and derived risk context helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

from bot.config import AppConfig
from bot.data.providers import (
    DEFAULT_TIMEOUT_SECONDS,
    DataProviderConfigurationError,
    DataProviderError,
    DataProviderRequestError,
    provider_capabilities_for_role,
    resolve_provider_credentials,
)


EARNINGS_CACHE_TTL = timedelta(hours=24)


class NyseHolidayCalendar(AbstractHolidayCalendar):
    """Standard NYSE holiday schedule used for earnings-risk day counting."""

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=USMartinLutherKingJr.offset,
            start_date="1998-01-01",
        ),
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date="2022-06-19",
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_HOLIDAY_CALENDAR = NyseHolidayCalendar()


@dataclass(frozen=True)
class EarningsEvent:
    """One normalized earnings-calendar event for a symbol."""

    symbol: str
    earnings_date: date
    status: str | None = None
    name: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the event."""

        return {
            "symbol": self.symbol,
            "earnings_date": self.earnings_date.isoformat(),
            "status": self.status,
            "name": self.name,
            "source": self.source,
        }


@dataclass(frozen=True)
class EarningsRiskContext:
    """Derived earnings-event context for one symbol."""

    symbol: str
    earnings_date: date | None
    earnings_days_away: int | None
    is_earnings_risk: bool
    status: str | None = None
    name: str | None = None
    source: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return the context in JSON-friendly metadata form."""

        return {
            "earnings_date": self.earnings_date.isoformat() if self.earnings_date else None,
            "earnings_days_away": self.earnings_days_away,
            "is_earnings_risk": self.is_earnings_risk,
            "earnings_event_status": self.status,
            "earnings_event_name": self.name,
            "earnings_source": self.source,
        }


class EarningsCalendarProvider(ABC):
    """Abstract interface for upcoming earnings-calendar lookups."""

    provider_name: str
    capabilities = provider_capabilities_for_role("earnings_calendar", "polygon")

    def __init__(
        self,
        *,
        api_key: str,
        cache_dir: Path,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        cache_ttl: timedelta = EARNINGS_CACHE_TTL,
    ) -> None:
        if not api_key:
            raise DataProviderConfigurationError(
                f"{self.provider_name} requires a non-empty API key."
            )

        self.api_key = api_key
        self.cache = EarningsCalendarCache(cache_dir=cache_dir, provider_name=self.provider_name)
        self.timeout_seconds = timeout_seconds
        self.cache_ttl = cache_ttl

    def fetch_upcoming_earnings(
        self,
        *,
        start_date: date,
        end_date: date,
        symbols: Sequence[str] = (),
        refresh_cache: bool = False,
    ) -> dict[str, EarningsEvent]:
        """Fetch normalized upcoming earnings events keyed by symbol."""

        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")

        normalized_symbols = tuple(
            sorted({_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()})
        )
        raw_records: list[dict[str, Any]] | None = None
        if not refresh_cache:
            raw_records = self.cache.load_window(
                start_date=start_date,
                end_date=end_date,
                ttl=self.cache_ttl,
            )
        if raw_records is None:
            raw_records = self._fetch_upcoming_earnings_from_api(
                start_date=start_date,
                end_date=end_date,
            )
            self.cache.store_window(
                start_date=start_date,
                end_date=end_date,
                payload=raw_records,
            )

        return _normalize_earnings_records(
            raw_records,
            provider_name=self.provider_name,
            symbols=normalized_symbols,
        )

    @abstractmethod
    def _fetch_upcoming_earnings_from_api(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch raw earnings records for a date window."""

    def _get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore").strip()
            raise DataProviderRequestError(
                f"{self.provider_name} request failed with HTTP {exc.code}: {message or exc.reason}"
            ) from exc
        except URLError as exc:
            raise DataProviderRequestError(
                f"{self.provider_name} request failed: {exc.reason}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise DataProviderRequestError(
                f"{self.provider_name} request timed out: {exc}"
            ) from exc

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DataProviderRequestError(
                f"{self.provider_name} returned invalid JSON."
            ) from exc


class PolygonEarningsCalendarProvider(EarningsCalendarProvider):
    """Polygon TMX corporate-events adapter for upcoming earnings dates."""

    provider_name = "polygon"
    capabilities = provider_capabilities_for_role("earnings_calendar", provider_name)
    base_url = "https://api.polygon.io/tmx/v1/corporate-events"

    def _fetch_upcoming_earnings_from_api(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        params = {
            "type": "earnings_announcement_date",
            "date.gte": start_date.isoformat(),
            "date.lte": end_date.isoformat(),
            "order": "asc",
            "sort": "date",
            "limit": 1000,
            "apiKey": self.api_key,
        }
        url = f"{self.base_url}?{urlencode(params)}"
        return self._fetch_paginated_results(url)

    def _fetch_paginated_results(self, url: str) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        next_url: str | None = url

        while next_url:
            payload = self._get_json(next_url)
            if not isinstance(payload, dict):
                raise DataProviderRequestError("Polygon returned an unexpected earnings payload.")
            if payload.get("status") == "ERROR" or payload.get("error"):
                raise DataProviderRequestError(
                    f"Polygon returned an error: {payload.get('error') or payload.get('message') or payload}"
                )

            page_results = payload.get("results")
            if isinstance(page_results, list):
                collected.extend(record for record in page_results if isinstance(record, dict))

            raw_next_url = payload.get("next_url")
            if not isinstance(raw_next_url, str) or not raw_next_url:
                next_url = None
                continue

            separator = "&" if "?" in raw_next_url else "?"
            next_url = f"{raw_next_url}{separator}apiKey={quote(self.api_key, safe='')}"

        return collected


class EarningsCalendarCache:
    """File-backed cache for upcoming earnings-calendar windows."""

    def __init__(self, *, cache_dir: Path, provider_name: str) -> None:
        self.cache_dir = cache_dir.resolve()
        self.provider_name = provider_name

    def load_window(
        self,
        *,
        start_date: date,
        end_date: date,
        ttl: timedelta,
    ) -> list[dict[str, Any]] | None:
        path = self._window_path(start_date=start_date, end_date=end_date)
        payload = self._load_json(path)
        if not isinstance(payload, dict):
            return None

        fetched_at = _parse_cached_timestamp(payload.get("fetched_at_utc"))
        if fetched_at is None or datetime.now(timezone.utc) - fetched_at > ttl:
            return None

        results = payload.get("results")
        if not isinstance(results, list):
            return None
        return [dict(record) for record in results if isinstance(record, dict)]

    def store_window(
        self,
        *,
        start_date: date,
        end_date: date,
        payload: Sequence[Mapping[str, Any]],
    ) -> Path:
        path = self._window_path(start_date=start_date, end_date=end_date)
        return self._store_json(
            path,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "results": [dict(record) for record in payload],
            },
        )

    def _window_path(self, *, start_date: date, end_date: date) -> Path:
        provider_dir = self.cache_dir / self.provider_name
        filename = f"earnings_{start_date.isoformat()}_{end_date.isoformat()}.json"
        return provider_dir / filename

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataProviderError(f"Cached earnings data at {path} is invalid JSON.") from exc

    def _store_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


def create_earnings_calendar_provider(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
) -> EarningsCalendarProvider:
    """Create the configured earnings-calendar provider."""

    credentials = resolve_provider_credentials(
        config,
        role_name="earnings_calendar",
        env_file=env_file,
        environment=environment,
    )
    provider_name = credentials.provider_name
    provider_capabilities_for_role("earnings_calendar", provider_name)

    resolved_cache_dir = cache_dir or config.project_root / "data" / "cache" / "earnings"
    providers: dict[str, type[EarningsCalendarProvider]] = {
        "polygon": PolygonEarningsCalendarProvider,
    }
    provider_class = providers[provider_name]
    return provider_class(api_key=credentials.api_key, cache_dir=resolved_cache_dir)


def build_earnings_risk_contexts(
    symbols: Sequence[str],
    *,
    as_of_date: date,
    provider: EarningsCalendarProvider,
    risk_window_days: int,
    lookahead_calendar_days: int = 30,
    refresh_cache: bool = False,
) -> dict[str, EarningsRiskContext]:
    """Build per-symbol earnings context for the requested date."""

    if risk_window_days <= 0:
        raise ValueError("risk_window_days must be greater than zero.")
    if lookahead_calendar_days <= 0:
        raise ValueError("lookahead_calendar_days must be greater than zero.")

    normalized_symbols = sorted(
        {_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()}
    )
    if not normalized_symbols:
        return {}

    earnings_by_symbol = provider.fetch_upcoming_earnings(
        start_date=as_of_date,
        end_date=as_of_date + timedelta(days=lookahead_calendar_days),
        symbols=normalized_symbols,
        refresh_cache=refresh_cache,
    )

    contexts: dict[str, EarningsRiskContext] = {}
    for symbol in normalized_symbols:
        event = earnings_by_symbol.get(symbol)
        if event is not None and event.earnings_date < as_of_date:
            event = None
        earnings_days_away = (
            trading_days_until(as_of_date, event.earnings_date)
            if event is not None
            else None
        )
        contexts[symbol] = EarningsRiskContext(
            symbol=symbol,
            earnings_date=event.earnings_date if event is not None else None,
            earnings_days_away=earnings_days_away,
            is_earnings_risk=(
                earnings_days_away is not None
                and earnings_days_away <= risk_window_days
            ),
            status=event.status if event is not None else None,
            name=event.name if event is not None else None,
            source=event.source if event is not None else None,
        )
    return contexts


def earnings_context_metadata(
    context: EarningsRiskContext | None,
    *,
    context_status: str | None = None,
) -> dict[str, Any]:
    """Return JSON-friendly earnings metadata for a symbol."""

    if context is None:
        return {
            "earnings_date": None,
            "earnings_days_away": None,
            "is_earnings_risk": False,
            "earnings_event_status": None,
            "earnings_event_name": None,
            "earnings_source": None,
            "earnings_context_status": context_status,
        }
    return {
        **context.to_metadata(),
        "earnings_context_status": context_status or "available",
    }


def trading_days_until(as_of_date: date, target_date: date) -> int:
    """Return NYSE trading days from ``as_of_date`` until ``target_date``."""

    if as_of_date == target_date:
        return 0
    holiday_dates = _nyse_holiday_dates_between(as_of_date, target_date)
    return int(
        np.busday_count(
            as_of_date.isoformat(),
            target_date.isoformat(),
            holidays=holiday_dates,
        )
    )


@lru_cache(maxsize=256)
def _nyse_holiday_dates_between(start_date: date, end_date: date) -> tuple[str, ...]:
    if start_date <= end_date:
        range_start = start_date
        range_end = end_date
    else:
        range_start = end_date
        range_end = start_date

    holiday_index = _NYSE_HOLIDAY_CALENDAR.holidays(
        start=range_start.isoformat(),
        end=range_end.isoformat(),
    )
    return tuple(timestamp.date().isoformat() for timestamp in holiday_index)


def _normalize_earnings_records(
    records: Sequence[Mapping[str, Any]],
    *,
    provider_name: str,
    symbols: Sequence[str] = (),
) -> dict[str, EarningsEvent]:
    symbol_filter = set(symbols)
    earliest_events: dict[str, EarningsEvent] = {}

    for record in records:
        raw_symbol = record.get("ticker", record.get("symbol"))
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            continue
        symbol = _normalize_symbol(raw_symbol)
        if symbol_filter and symbol not in symbol_filter:
            continue

        earnings_date = _parse_record_date(record.get("date"))
        if earnings_date is None:
            continue

        event = EarningsEvent(
            symbol=symbol,
            earnings_date=earnings_date,
            status=_optional_text(record.get("status")),
            name=_optional_text(record.get("name")),
            source=provider_name,
        )
        existing_event = earliest_events.get(symbol)
        if existing_event is None or event.earnings_date < existing_event.earnings_date:
            earliest_events[symbol] = event

    return earliest_events


def _normalize_symbol(symbol: str) -> str:
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string.")
    return normalized_symbol


def _parse_record_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_cached_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
