"""Historical daily-bar providers and transparent local caching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

from bot.config import AppConfig, ConfigError, _read_env_file as _read_config_env_file
from bot.data.normalize import (
    DAILY_BAR_COLUMNS,
    INTRADAY_BAR_COLUMNS,
    DataNormalizationError,
    empty_daily_bar_frame,
    empty_intraday_bar_frame,
    filter_daily_bars_by_date,
    filter_intraday_bars_by_session_date,
    normalize_intraday_bars,
    normalize_daily_bars,
)
from bot.providers.capabilities import (
    ProviderCapabilities,
    provider_capabilities_for_role as _provider_capabilities_for_role,
    supported_provider_names_for_role as _supported_provider_names_for_role,
)


DEFAULT_TIMEOUT_SECONDS = 30

class DataProviderError(RuntimeError):
    """Base exception for provider and cache failures."""


class DataProviderConfigurationError(DataProviderError):
    """Raised when the configured provider cannot be initialized."""


class DataProviderRequestError(DataProviderError):
    """Raised when a remote provider rejects a request or returns malformed data."""


def supported_provider_names_for_role(role_name: str) -> tuple[str, ...]:
    """Return supported provider names for one explicit provider role."""

    try:
        return _supported_provider_names_for_role(role_name)
    except ValueError:
        raise


def provider_capabilities_for_role(
    role_name: str,
    provider_name: str,
) -> ProviderCapabilities:
    """Return capabilities for one role/provider pair or raise a role-specific error."""

    try:
        return _provider_capabilities_for_role(role_name, provider_name)
    except ValueError as exc:
        raise DataProviderConfigurationError(str(exc)) from exc


ProviderRoleCapabilities = ProviderCapabilities


def provider_error_is_timeout(exc: BaseException) -> bool:
    """Return whether the provider error was caused by a network timeout."""

    message = str(exc).strip().lower()
    return "timed out" in message or "timeout" in message


def provider_error_is_entitlement_limited(exc: BaseException) -> bool:
    """Return whether the provider error looks like a plan/entitlement restriction."""

    message = str(exc).strip().lower()
    entitlement_markers = (
        "not_authorized",
        "not authorized",
        "not entitled",
        "not available with your current plan",
        "permission denied",
        "http 401",
        "http 403",
    )
    return any(marker in message for marker in entitlement_markers)


class DailyBarProvider(ABC):
    """Abstract interface for historical daily OHLCV providers."""

    provider_name: str
    capabilities = ProviderCapabilities()

    def __init__(
        self,
        *,
        api_key: str,
        cache_dir: Path,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key:
            raise DataProviderConfigurationError(
                f"{self.provider_name} requires a non-empty API key."
            )

        self.api_key = api_key
        self.cache = DailyBarCache(cache_dir=cache_dir, provider_name=self.provider_name)
        self.intraday_cache = IntradayBarCache(cache_dir=cache_dir, provider_name=self.provider_name)
        self.timeout_seconds = timeout_seconds

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        """Fetch normalized daily bars for a symbol over an inclusive date range."""

        normalized_symbol = _normalize_symbol(symbol)
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")

        if not refresh_cache:
            cached_frame = self.cache.load(normalized_symbol, start_date, end_date)
            if cached_frame is not None:
                return cached_frame

        frame = self._fetch_daily_bars_from_api(normalized_symbol, start_date, end_date)
        normalized_frame = normalize_daily_bars(frame, symbol=normalized_symbol)
        filtered_frame = filter_daily_bars_by_date(
            normalized_frame,
            start_date=start_date,
            end_date=end_date,
        )
        self.cache.store(normalized_symbol, start_date, end_date, filtered_frame)
        return filtered_frame

    def fetch_intraday_bars(
        self,
        symbol: str,
        session_date: date,
        *,
        interval_minutes: int = 15,
        refresh_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch normalized intraday OHLCV bars for one symbol and session date."""

        normalized_symbol = _normalize_symbol(symbol)
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero.")

        if not refresh_cache:
            cached_frame = self.intraday_cache.load(
                normalized_symbol,
                session_date,
                interval_minutes,
            )
            if cached_frame is not None:
                return cached_frame

        frame = self._fetch_intraday_bars_from_api(
            normalized_symbol,
            session_date,
            interval_minutes,
        )
        normalized_frame = normalize_intraday_bars(frame, symbol=normalized_symbol)
        filtered_frame = filter_intraday_bars_by_session_date(
            normalized_frame,
            session_date=session_date,
        )
        self.intraday_cache.store(
            normalized_symbol,
            session_date,
            interval_minutes,
            filtered_frame,
        )
        return filtered_frame

    @abstractmethod
    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch remote data for the requested symbol and date range."""

    def _fetch_intraday_bars_from_api(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        """Fetch remote intraday data for the requested symbol and session date."""

        raise DataProviderConfigurationError(
            f"{self.provider_name} does not support intraday aggregate bars."
        )

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


class AlphaVantageDailyBarProvider(DailyBarProvider):
    """Alpha Vantage adapter for daily OHLCV data."""

    provider_name = "alphavantage"
    capabilities = provider_capabilities_for_role("historical_bars", provider_name)
    base_url = "https://www.alphavantage.co/query"

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        query = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            }
        )
        payload = self._get_json(f"{self.base_url}?{query}")
        if not isinstance(payload, dict):
            raise DataProviderRequestError("Alpha Vantage returned an unexpected payload.")
        if "Error Message" in payload:
            raise DataProviderRequestError(str(payload["Error Message"]))
        if "Information" in payload:
            raise DataProviderRequestError(str(payload["Information"]))
        if "Note" in payload:
            raise DataProviderRequestError(str(payload["Note"]))

        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            return empty_daily_bar_frame()

        frame = pd.DataFrame.from_dict(series, orient="index").reset_index()
        frame = normalize_daily_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "index": "date",
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "6. volume": "volume",
            },
        )
        return filter_daily_bars_by_date(frame, start_date=start_date, end_date=end_date)


class TiingoDailyBarProvider(DailyBarProvider):
    """Tiingo adapter for daily OHLCV data."""

    provider_name = "tiingo"
    capabilities = provider_capabilities_for_role("historical_bars", provider_name)
    base_url = "https://api.tiingo.com/tiingo/daily"

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        query = urlencode(
            {
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "token": self.api_key,
            }
        )
        url = f"{self.base_url}/{quote(symbol, safe='')}/prices?{query}"
        payload = self._get_json(url)
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            raise DataProviderRequestError(f"Tiingo returned an error: {detail or payload}")
        if not isinstance(payload, list):
            raise DataProviderRequestError("Tiingo returned an unexpected payload.")
        if not payload:
            return empty_daily_bar_frame()

        frame = pd.DataFrame(payload)
        return normalize_daily_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        )


class PolygonDailyBarProvider(DailyBarProvider):
    """Polygon adapter for daily OHLCV data."""

    provider_name = "polygon"
    capabilities = provider_capabilities_for_role("historical_bars", provider_name)
    base_url = "https://api.polygon.io/v2/aggs/ticker"

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        query = urlencode(
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key,
            }
        )
        symbol_path = quote(symbol, safe="")
        url = (
            f"{self.base_url}/{symbol_path}/range/1/day/"
            f"{start_date.isoformat()}/{end_date.isoformat()}?{query}"
        )
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise DataProviderRequestError("Polygon returned an unexpected payload.")
        if payload.get("status") == "ERROR" or payload.get("error"):
            raise DataProviderRequestError(
                f"Polygon returned an error: {payload.get('error') or payload.get('message') or payload}"
            )

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return empty_daily_bar_frame()

        frame = pd.DataFrame(results)
        frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
        return normalize_daily_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "date": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
        )

    def _fetch_intraday_bars_from_api(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        query = urlencode(
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key,
            }
        )
        symbol_path = quote(symbol, safe="")
        url = (
            f"{self.base_url}/{symbol_path}/range/{interval_minutes}/minute/"
            f"{session_date.isoformat()}/{session_date.isoformat()}?{query}"
        )
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise DataProviderRequestError("Polygon returned an unexpected payload.")
        if payload.get("status") == "ERROR" or payload.get("error"):
            raise DataProviderRequestError(
                f"Polygon returned an error: {payload.get('error') or payload.get('message') or payload}"
            )

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return empty_intraday_bar_frame()

        frame = pd.DataFrame(results)
        frame["datetime"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
        return normalize_intraday_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "datetime": "datetime",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
            },
        )


class AlpacaDailyBarProvider(DailyBarProvider):
    """Alpaca Market Data adapter for historical stock OHLCV bars."""

    provider_name = "alpaca"
    capabilities = provider_capabilities_for_role("historical_bars", provider_name)
    base_url = "https://data.alpaca.markets/v2/stocks"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        cache_dir: Path,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        default_feed: str | None = None,
        default_adjustment: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
        )
        if not api_secret:
            raise DataProviderConfigurationError(
                f"{self.provider_name} requires a non-empty API secret."
            )
        self.api_secret = api_secret
        self.default_feed = default_feed.strip().lower() if default_feed else None
        self.default_adjustment = (
            default_adjustment.strip().lower() if default_adjustment else None
        )

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        records = self._fetch_bars(
            symbol=symbol,
            timeframe="1Day",
            start_timestamp=_iso_start_of_day(start_date),
            end_timestamp=_iso_end_of_day(end_date),
        )
        if not records:
            return empty_daily_bar_frame()

        frame = pd.DataFrame(records)
        frame["date"] = pd.to_datetime(frame["t"], utc=True)
        return normalize_daily_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "date": "date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
        )

    def _fetch_intraday_bars_from_api(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        records = self._fetch_bars(
            symbol=symbol,
            timeframe=f"{interval_minutes}Min",
            start_timestamp=_iso_start_of_day(session_date),
            end_timestamp=_iso_start_of_day(_next_date(session_date)),
        )
        if not records:
            return empty_intraday_bar_frame()

        frame = pd.DataFrame(records)
        frame["datetime"] = pd.to_datetime(frame["t"], utc=True)
        return normalize_intraday_bars(
            frame,
            symbol=symbol,
            column_mapping={
                "datetime": "datetime",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
            },
        )

    def _fetch_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_timestamp: str,
        end_timestamp: str,
    ) -> list[dict[str, Any]]:
        collected_records: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while True:
            params = {
                "timeframe": timeframe,
                "start": start_timestamp,
                "end": end_timestamp,
                "limit": 10_000,
            }
            if self.default_feed is not None:
                params["feed"] = self.default_feed
            if self.default_adjustment is not None:
                params["adjustment"] = self.default_adjustment
            if next_page_token is not None:
                params["page_token"] = next_page_token
            url = (
                f"{self.base_url}/{quote(symbol, safe='')}/bars?{urlencode(params)}"
            )
            payload = self._get_json(url, headers=self._auth_headers())
            if not isinstance(payload, dict):
                raise DataProviderRequestError("alpaca returned an unexpected payload.")
            raw_records = payload.get("bars")
            if raw_records is None:
                message = payload.get("message") or payload.get("error") or payload
                raise DataProviderRequestError(
                    f"alpaca returned an unexpected bars payload: {message}"
                )
            if not isinstance(raw_records, list):
                raise DataProviderRequestError("alpaca returned invalid bars data.")
            collected_records.extend(
                record for record in raw_records if isinstance(record, dict)
            )
            raw_next_page_token = payload.get("next_page_token")
            if not isinstance(raw_next_page_token, str) or not raw_next_page_token.strip():
                break
            next_page_token = raw_next_page_token.strip()
        return collected_records

    def _auth_headers(self) -> Mapping[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }


class DailyBarCache:
    """Simple file-based cache for normalized daily bars."""

    def __init__(self, *, cache_dir: Path, provider_name: str) -> None:
        self.cache_dir = cache_dir.resolve()
        self.provider_name = provider_name

    def cache_path(self, symbol: str, start_date: date, end_date: date) -> Path:
        safe_symbol = re.sub(r"[^A-Z0-9._-]+", "_", symbol.upper())
        provider_dir = self.cache_dir / self.provider_name
        filename = f"{safe_symbol}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        return provider_dir / filename

    def load(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        path = self.cache_path(symbol, start_date, end_date)
        if not path.exists():
            return None

        frame = pd.read_csv(path, parse_dates=["date"])
        try:
            return normalize_daily_bars(frame)
        except DataNormalizationError as exc:
            raise DataProviderError(f"Cached data at {path} is invalid: {exc}") from exc

    def store(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frame: pd.DataFrame,
    ) -> Path:
        path = self.cache_path(symbol, start_date, end_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(DAILY_BAR_COLUMNS)].to_csv(path, index=False, date_format="%Y-%m-%d")
        return path


class IntradayBarCache:
    """Simple file-based cache for normalized intraday bars."""

    def __init__(self, *, cache_dir: Path, provider_name: str) -> None:
        self.cache_dir = cache_dir.resolve()
        self.provider_name = provider_name

    def cache_path(self, symbol: str, session_date: date, interval_minutes: int) -> Path:
        safe_symbol = re.sub(r"[^A-Z0-9._-]+", "_", symbol.upper())
        provider_dir = self.cache_dir / self.provider_name
        filename = (
            f"{safe_symbol}_intraday_{session_date.isoformat()}_{interval_minutes}min.csv"
        )
        return provider_dir / filename

    def load(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame | None:
        path = self.cache_path(symbol, session_date, interval_minutes)
        if not path.exists():
            return None

        frame = pd.read_csv(path, parse_dates=["datetime"])
        try:
            return normalize_intraday_bars(frame)
        except DataNormalizationError as exc:
            raise DataProviderError(f"Cached data at {path} is invalid: {exc}") from exc

    def store(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
        frame: pd.DataFrame,
    ) -> Path:
        path = self.cache_path(symbol, session_date, interval_minutes)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, list(INTRADAY_BAR_COLUMNS)].to_csv(
            path,
            index=False,
            date_format="%Y-%m-%dT%H:%M:%S",
        )
        return path


@dataclass(frozen=True)
class ResolvedProviderCredentials:
    """Resolved credential values for one configured role/provider assignment."""

    provider_name: str
    api_key: str
    api_secret: str | None = None
    default_feed: str | None = None
    default_adjustment: str | None = None


def resolve_provider_credentials(
    config: AppConfig,
    *,
    role_name: str,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedProviderCredentials:
    """Resolve required credentials and defaults for one configured provider role."""

    provider_name = config.data_sources.provider_name_for_role(role_name)
    if provider_name is None:
        raise DataProviderConfigurationError(f"Provider role '{role_name}' is not configured.")
    provider_capabilities_for_role(role_name, provider_name)
    try:
        provider_config = config.data_sources.provider_config(provider_name)
    except ConfigError as exc:
        supported_names = supported_provider_names_for_role(role_name)
        raise DataProviderConfigurationError(
            "Unsupported provider assignment for role "
            f"'{role_name}': requested '{provider_name}', supported providers are "
            f"{list(supported_names)}."
        ) from exc
    merged_environment = load_provider_environment(env_file=env_file, environment=environment)
    missing_environment_variables = tuple(
        env_name
        for env_name in provider_config.required_environment_variables()
        if not merged_environment.get(env_name, "").strip()
    )
    if missing_environment_variables:
        missing_list = ", ".join(missing_environment_variables)
        raise DataProviderConfigurationError(
            f"Role '{role_name}' is configured as '{provider_name}' but required credentials "
            f"are missing: {missing_list}."
        )
    return ResolvedProviderCredentials(
        provider_name=provider_name,
        api_key=merged_environment[provider_config.api_key_env].strip(),
        api_secret=(
            merged_environment[provider_config.api_secret_env].strip()
            if provider_config.api_secret_env is not None
            else None
        ),
        default_feed=provider_config.default_feed,
        default_adjustment=provider_config.default_adjustment,
    )


def create_historical_bars_provider(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
) -> DailyBarProvider:
    """Create the configured historical-bars provider from application config."""

    credentials = resolve_provider_credentials(
        config,
        role_name="historical_bars",
        env_file=env_file,
        environment=environment,
    )
    resolved_cache_dir = cache_dir or config.project_root / "data" / "cache" / "daily_bars"
    provider_name = credentials.provider_name
    provider_capabilities_for_role("historical_bars", provider_name)
    providers: dict[str, type[DailyBarProvider]] = {
        "alphavantage": AlphaVantageDailyBarProvider,
        "tiingo": TiingoDailyBarProvider,
        "polygon": PolygonDailyBarProvider,
        "alpaca": AlpacaDailyBarProvider,
    }
    provider_class = providers[provider_name]
    if provider_name == "alpaca":
        return provider_class(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret or "",
            cache_dir=resolved_cache_dir,
            default_feed=credentials.default_feed,
            default_adjustment=credentials.default_adjustment,
        )
    return provider_class(api_key=credentials.api_key, cache_dir=resolved_cache_dir)


def create_daily_bar_provider(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
) -> DailyBarProvider:
    """Backward-compatible alias for the historical-bars provider factory."""

    return create_historical_bars_provider(
        config,
        env_file=env_file,
        environment=environment,
        cache_dir=cache_dir,
    )


def load_provider_environment(
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge a simple .env file with the live process environment."""

    try:
        merged_environment = _read_config_env_file(env_file)
    except ConfigError as exc:
        raise DataProviderConfigurationError(str(exc)) from exc
    merged_environment.update(dict(environment or os.environ))
    return merged_environment


def _normalize_symbol(symbol: str) -> str:
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string.")
    return normalized_symbol


def _iso_start_of_day(value: date) -> str:
    return datetime.combine(value, time.min, tzinfo=timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def _iso_end_of_day(value: date) -> str:
    return datetime.combine(value, time.max, tzinfo=timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def _next_date(value: date) -> date:
    return value + timedelta(days=1)
