"""Historical daily-bar providers and transparent local caching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
import json
import os
from pathlib import Path
import re
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


DEFAULT_TIMEOUT_SECONDS = 30


class DataProviderError(RuntimeError):
    """Base exception for provider and cache failures."""


class DataProviderConfigurationError(DataProviderError):
    """Raised when the configured provider cannot be initialized."""


class DataProviderRequestError(DataProviderError):
    """Raised when a remote provider rejects a request or returns malformed data."""


class DailyBarProvider(ABC):
    """Abstract interface for historical daily OHLCV providers."""

    provider_name: str

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

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DataProviderRequestError(
                f"{self.provider_name} returned invalid JSON."
            ) from exc


class AlphaVantageDailyBarProvider(DailyBarProvider):
    """Alpha Vantage adapter for daily OHLCV data."""

    provider_name = "alphavantage"
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


def create_daily_bar_provider(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
) -> DailyBarProvider:
    """Create the configured daily-bar provider from application config."""

    provider_name = config.data_sources.provider.lower()
    active_provider_config = config.data_sources.active_provider()
    merged_environment = load_provider_environment(env_file=env_file, environment=environment)
    api_key = merged_environment.get(active_provider_config.api_key_env, "").strip()
    if not api_key:
        raise DataProviderConfigurationError(
            f"Missing API key for provider '{provider_name}'. "
            f"Set {active_provider_config.api_key_env} in the shell environment or .env file."
        )

    resolved_cache_dir = cache_dir or config.project_root / "data" / "cache" / "daily_bars"
    providers: dict[str, type[DailyBarProvider]] = {
        "alphavantage": AlphaVantageDailyBarProvider,
        "tiingo": TiingoDailyBarProvider,
        "polygon": PolygonDailyBarProvider,
    }
    try:
        provider_class = providers[provider_name]
    except KeyError as exc:
        valid_provider_names = ", ".join(sorted(providers))
        raise DataProviderConfigurationError(
            f"Unsupported provider '{provider_name}'. Expected one of: {valid_provider_names}."
        ) from exc

    return provider_class(api_key=api_key, cache_dir=resolved_cache_dir)


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
