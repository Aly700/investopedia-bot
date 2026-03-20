"""Reference-data providers for staged universe construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from bot.config import AppConfig
from bot.data.providers import (
    DEFAULT_TIMEOUT_SECONDS,
    DataProviderConfigurationError,
    DataProviderRequestError,
    load_provider_environment,
)


LOGGER = logging.getLogger(__name__)


class ReferenceUniverseProvider(ABC):
    """Abstract interface for master-universe and symbol-reference lookups."""

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
        self.cache = ReferenceDataCache(cache_dir=cache_dir, provider_name=self.provider_name)
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def fetch_master_universe(
        self,
        *,
        market: str,
        locale: str,
        active_only: bool,
        allowed_ticker_types: tuple[str, ...] = (),
        as_of_date: date | None = None,
        refresh_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch a broad reference universe from the remote provider."""

    @abstractmethod
    def fetch_ticker_details(
        self,
        symbol: str,
        *,
        as_of_date: date | None = None,
        refresh_cache: bool = False,
    ) -> dict[str, Any]:
        """Fetch reference metadata for one symbol."""

    def fetch_grouped_daily_market(
        self,
        market_date: date,
        *,
        refresh_cache: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Fetch one grouped daily market summary for all symbols on a date."""

        raise NotImplementedError(
            f"{self.provider_name} does not support grouped daily market summaries."
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


class PolygonReferenceUniverseProvider(ReferenceUniverseProvider):
    """Polygon reference-data adapter for master universe construction."""

    provider_name = "polygon"
    tickers_url = "https://api.polygon.io/v3/reference/tickers"
    grouped_daily_url = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

    def fetch_master_universe(
        self,
        *,
        market: str,
        locale: str,
        active_only: bool,
        allowed_ticker_types: tuple[str, ...] = (),
        as_of_date: date | None = None,
        refresh_cache: bool = False,
    ) -> list[dict[str, Any]]:
        cached_payload = None
        if not refresh_cache:
            cached_payload = self.cache.load_master(
                market=market,
                locale=locale,
                active_only=active_only,
                allowed_ticker_types=allowed_ticker_types,
                as_of_date=as_of_date,
            )
        if cached_payload is not None:
            return list(cached_payload)

        results: dict[str, dict[str, Any]] = {}
        ticker_types = allowed_ticker_types or ("",)
        for ticker_type in ticker_types:
            normalized_ticker_type = ticker_type.strip().upper()
            type_label = normalized_ticker_type or "all types"
            LOGGER.info("Master universe fetch: requesting tickers for %s.", type_label)
            params = {
                "market": market,
                "locale": locale,
                "active": "true" if active_only else "false",
                "order": "asc",
                "sort": "ticker",
                "limit": 1000,
                "apiKey": self.api_key,
            }
            if normalized_ticker_type:
                params["type"] = normalized_ticker_type
            if as_of_date is not None:
                params["date"] = as_of_date.isoformat()

            url = f"{self.tickers_url}?{urlencode(params)}"
            fetched_records = self._fetch_paginated_results(
                url,
                log_label=f"master universe ({type_label})",
            )
            LOGGER.info(
                "Master universe fetch: received %s records for %s.",
                len(fetched_records),
                type_label,
            )
            for record in fetched_records:
                symbol = str(record.get("ticker", "")).strip().upper()
                if not symbol:
                    continue
                results[symbol] = dict(record)

        ordered_results = [results[symbol] for symbol in sorted(results)]
        self.cache.store_master(
            market=market,
            locale=locale,
            active_only=active_only,
            allowed_ticker_types=allowed_ticker_types,
            as_of_date=as_of_date,
            payload=ordered_results,
        )
        return ordered_results

    def fetch_ticker_details(
        self,
        symbol: str,
        *,
        as_of_date: date | None = None,
        refresh_cache: bool = False,
    ) -> dict[str, Any]:
        normalized_symbol = _normalize_symbol(symbol)
        if not refresh_cache:
            cached_payload = self.cache.load_details(normalized_symbol, as_of_date=as_of_date)
            if cached_payload is not None:
                return dict(cached_payload)

        params = {"apiKey": self.api_key}
        if as_of_date is not None:
            params["date"] = as_of_date.isoformat()
        url = f"{self.tickers_url}/{quote(normalized_symbol, safe='')}?{urlencode(params)}"
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise DataProviderRequestError("Polygon returned an unexpected ticker details payload.")

        results = payload.get("results")
        if isinstance(results, dict):
            record = dict(results)
        elif isinstance(payload.get("ticker"), str):
            record = dict(payload)
        else:
            raise DataProviderRequestError(
                f"Polygon returned an unexpected ticker details payload for {normalized_symbol}."
            )

        self.cache.store_details(normalized_symbol, as_of_date=as_of_date, payload=record)
        return record

    def fetch_grouped_daily_market(
        self,
        market_date: date,
        *,
        refresh_cache: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if not refresh_cache:
            cached_payload = self.cache.load_grouped_daily(market_date)
            if cached_payload is not None:
                return dict(cached_payload)

        params = {
            "adjusted": "true",
            "include_otc": "false",
            "apiKey": self.api_key,
        }
        url = f"{self.grouped_daily_url}/{market_date.isoformat()}?{urlencode(params)}"
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise DataProviderRequestError("Polygon returned an unexpected grouped daily payload.")
        if payload.get("status") == "ERROR" or payload.get("error"):
            raise DataProviderRequestError(
                f"Polygon returned an error: {payload.get('error') or payload.get('message') or payload}"
            )

        results = payload.get("results")
        if not isinstance(results, list):
            return {}

        grouped_records: dict[str, dict[str, Any]] = {}
        for record in results:
            if not isinstance(record, dict):
                continue
            symbol = str(record.get("T") or record.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            grouped_records[symbol] = dict(record)

        self.cache.store_grouped_daily(market_date, payload=grouped_records)
        return grouped_records

    def _fetch_paginated_results(
        self,
        url: str,
        *,
        log_label: str,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        next_url: str | None = url
        page_number = 0

        while next_url:
            page_number += 1
            payload = self._get_json(next_url)
            if not isinstance(payload, dict):
                raise DataProviderRequestError("Polygon returned an unexpected payload.")
            if payload.get("status") == "ERROR" or payload.get("error"):
                raise DataProviderRequestError(
                    f"Polygon returned an error: {payload.get('error') or payload.get('message') or payload}"
                )

            page_results = payload.get("results")
            if isinstance(page_results, list):
                collected.extend(record for record in page_results if isinstance(record, dict))
                LOGGER.info(
                    "%s: fetched page %s (%s records, %s total).",
                    log_label,
                    page_number,
                    len(page_results),
                    len(collected),
                )

            raw_next_url = payload.get("next_url")
            if not isinstance(raw_next_url, str) or not raw_next_url:
                next_url = None
                continue

            separator = "&" if "?" in raw_next_url else "?"
            next_url = f"{raw_next_url}{separator}apiKey={quote(self.api_key, safe='')}"

        return collected


class ReferenceDataCache:
    """Simple file cache for master-universe and ticker-details JSON payloads."""

    def __init__(self, *, cache_dir: Path, provider_name: str) -> None:
        self.cache_dir = cache_dir.resolve()
        self.provider_name = provider_name

    def load_master(
        self,
        *,
        market: str,
        locale: str,
        active_only: bool,
        allowed_ticker_types: tuple[str, ...],
        as_of_date: date | None,
    ) -> list[dict[str, Any]] | None:
        path = self._master_path(
            market=market,
            locale=locale,
            active_only=active_only,
            allowed_ticker_types=allowed_ticker_types,
            as_of_date=as_of_date,
        )
        payload = self._load_json(path)
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        if not isinstance(results, list):
            return None
        return [dict(record) for record in results if isinstance(record, dict)]

    def store_master(
        self,
        *,
        market: str,
        locale: str,
        active_only: bool,
        allowed_ticker_types: tuple[str, ...],
        as_of_date: date | None,
        payload: list[dict[str, Any]],
    ) -> Path:
        path = self._master_path(
            market=market,
            locale=locale,
            active_only=active_only,
            allowed_ticker_types=allowed_ticker_types,
            as_of_date=as_of_date,
        )
        return self._store_json(
            path,
            {
                "market": market,
                "locale": locale,
                "active_only": active_only,
                "allowed_ticker_types": list(allowed_ticker_types),
                "as_of_date": as_of_date.isoformat() if as_of_date else None,
                "results": payload,
            },
        )

    def load_details(
        self,
        symbol: str,
        *,
        as_of_date: date | None,
    ) -> dict[str, Any] | None:
        path = self._details_path(symbol, as_of_date=as_of_date)
        payload = self._load_json(path)
        return dict(payload) if isinstance(payload, dict) else None

    def store_details(
        self,
        symbol: str,
        *,
        as_of_date: date | None,
        payload: dict[str, Any],
    ) -> Path:
        return self._store_json(
            self._details_path(symbol, as_of_date=as_of_date),
            payload,
        )

    def load_grouped_daily(self, market_date: date) -> dict[str, dict[str, Any]] | None:
        path = self._grouped_daily_path(market_date)
        payload = self._load_json(path)
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        if not isinstance(results, dict):
            return None

        grouped_records: dict[str, dict[str, Any]] = {}
        for symbol, record in results.items():
            if not isinstance(symbol, str) or not isinstance(record, dict):
                continue
            grouped_records[symbol] = dict(record)
        return grouped_records

    def store_grouped_daily(
        self,
        market_date: date,
        *,
        payload: dict[str, dict[str, Any]],
    ) -> Path:
        return self._store_json(
            self._grouped_daily_path(market_date),
            {
                "date": market_date.isoformat(),
                "results": payload,
            },
        )

    def _master_path(
        self,
        *,
        market: str,
        locale: str,
        active_only: bool,
        allowed_ticker_types: tuple[str, ...],
        as_of_date: date | None,
    ) -> Path:
        type_suffix = "-".join(
            _safe_path_component(value) for value in allowed_ticker_types if value.strip()
        )
        type_fragment = type_suffix or "all-types"
        date_fragment = as_of_date.isoformat() if as_of_date else "latest"
        filename = (
            f"master_{_safe_path_component(market)}_{_safe_path_component(locale)}_"
            f"{'active' if active_only else 'all'}_{type_fragment}_{date_fragment}.json"
        )
        return self.cache_dir / self.provider_name / "master" / filename

    def _details_path(self, symbol: str, *, as_of_date: date | None) -> Path:
        date_fragment = as_of_date.isoformat() if as_of_date else "latest"
        filename = f"{_safe_path_component(symbol)}_{date_fragment}.json"
        return self.cache_dir / self.provider_name / "ticker_details" / filename

    def _grouped_daily_path(self, market_date: date) -> Path:
        filename = f"{market_date.isoformat()}.json"
        return self.cache_dir / self.provider_name / "grouped_daily" / filename

    def _load_json(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _store_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def create_reference_universe_provider(
    config: AppConfig,
    *,
    env_file: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
) -> ReferenceUniverseProvider:
    """Create the configured reference-data provider for universe building."""

    provider_name = config.data_sources.provider.lower()
    if provider_name != "polygon":
        raise DataProviderConfigurationError(
            "Master-universe fetching is currently implemented for the polygon provider only. "
            "Switch config/data_sources.yaml to polygon or pass --master-input."
        )

    active_provider_config = config.data_sources.active_provider()
    merged_environment = load_provider_environment(env_file=env_file, environment=environment)
    api_key = merged_environment.get(active_provider_config.api_key_env, "").strip()
    if not api_key:
        raise DataProviderConfigurationError(
            f"Missing API key for provider '{provider_name}'. "
            f"Set {active_provider_config.api_key_env} in the shell environment or .env file."
        )

    resolved_cache_dir = cache_dir or (
        config.project_root / config.universe_builder.master.reference_cache_dir
    )
    return PolygonReferenceUniverseProvider(api_key=api_key, cache_dir=resolved_cache_dir)


def _normalize_symbol(symbol: str) -> str:
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string.")
    return normalized_symbol


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "unknown"
