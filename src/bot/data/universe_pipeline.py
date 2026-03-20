"""Staged master-universe pipeline and profile-based candidate outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bot.config import UniverseMasterConfig, UniverseProfileConfig
from bot.data.providers import DailyBarProvider, DataProviderError
from bot.data.reference import ReferenceUniverseProvider


LOGGER = logging.getLogger(__name__)

MASTER_UNIVERSE_COLUMNS = (
    "symbol",
    "name",
    "active",
    "exchange",
    "market",
    "type",
    "market_cap",
    "sector",
    "industry",
    "price",
    "average_daily_volume",
    "dollar_volume",
    "is_etf",
    "is_adr",
    "is_common_stock",
    "source",
    "updated_at",
)

WEIRD_INSTRUMENT_PATTERNS = (
    ".W",
    ".WS",
    ".WT",
    ".U",
    ".R",
    ".P",
    "-W",
    "-WS",
    "-WT",
    "-U",
    "-R",
    "-P",
    "/W",
    "/WS",
    "/WT",
    "/U",
    "/R",
    "/P",
)

ETF_TYPE_CODES = {"ETF", "ETN", "ETV", "ETMF", "FUND", "MF"}
COMMON_STOCK_TYPE_CODES = {"CS"}
ADR_TYPE_TOKENS = ("ADR", "ADS", "DEPOSITARY")
WEIRD_TYPE_CODES = {"WARRANT", "WT", "RIGHT", "RT", "UNIT", "PFD", "PREF"}


@dataclass
class UniverseProfileBuildResult:
    """Materialized output for one configured universe profile."""

    profile_name: str
    frame: pd.DataFrame
    symbols: tuple[str, ...]
    summary: dict[str, Any]


def fetch_master_universe(
    *,
    master_config: UniverseMasterConfig,
    as_of_date: date,
    reference_provider: ReferenceUniverseProvider | None = None,
    master_input: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch or load the broad master universe used by profile builders."""

    if master_input is not None:
        resolved_input = master_input.resolve()
        if resolved_input.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported master input format: {resolved_input}")
        frame = pd.read_csv(resolved_input)
        return normalize_master_universe_frame(frame)

    if reference_provider is None:
        raise ValueError("A reference universe provider is required when --master-input is not used.")

    records = reference_provider.fetch_master_universe(
        market=master_config.market,
        locale=master_config.locale,
        active_only=master_config.active_only,
        allowed_ticker_types=master_config.allowed_ticker_types,
        as_of_date=as_of_date,
        refresh_cache=refresh_cache,
    )
    generated_at = _generated_timestamp()
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        symbol = str(record.get("ticker", "")).strip().upper()
        if not symbol:
            continue

        name = _as_clean_string(record.get("name"))
        type_code = _as_clean_string(record.get("type")).upper()
        industry = _as_clean_string(record.get("sic_description"))
        normalized_records.append(
            {
                "symbol": symbol,
                "name": name,
                "active": bool(record.get("active", False)),
                "exchange": _as_clean_string(record.get("primary_exchange")),
                "market": _as_clean_string(record.get("market")) or master_config.market,
                "type": type_code,
                "market_cap": None,
                "sector": "",
                "industry": industry,
                "price": None,
                "average_daily_volume": None,
                "dollar_volume": None,
                "is_etf": _infer_is_etf(type_code, name=name, industry=industry),
                "is_adr": _infer_is_adr(type_code, name=name, industry=industry),
                "is_common_stock": _infer_is_common_stock(
                    type_code,
                    name=name,
                    industry=industry,
                ),
                "source": reference_provider.provider_name,
                "updated_at": generated_at,
            }
        )
    return normalize_master_universe_frame(pd.DataFrame(normalized_records))


def enrich_universe_metadata(
    master_frame: pd.DataFrame,
    *,
    as_of_date: date,
    lookback_days: int,
    daily_bar_provider: DailyBarProvider,
    reference_provider: ReferenceUniverseProvider | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Enrich the master universe with provider metadata and liquidity metrics."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero.")

    normalized_master = normalize_master_universe_frame(master_frame)
    if normalized_master.empty:
        return normalized_master

    LOGGER.info(
        "Starting metadata enrichment for %s symbols with a %s-day liquidity lookback.",
        len(normalized_master),
        lookback_days,
    )

    if reference_provider is not None:
        try:
            grouped_metrics, trading_days = _load_grouped_market_metrics(
                reference_provider,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
                refresh_cache=refresh_cache,
            )
        except NotImplementedError:
            LOGGER.info(
                "Reference provider %s does not support grouped daily summaries; falling back to per-symbol bar fetches.",
                reference_provider.provider_name,
            )
        else:
            enriched_frame = _apply_grouped_market_metrics(normalized_master, grouped_metrics)
            LOGGER.info(
                "Metadata enrichment completed using grouped market data for %s trading days.",
                trading_days,
            )
            return enriched_frame

    LOGGER.info(
        "Metadata enrichment fallback: fetching per-symbol daily bars for %s symbols.",
        len(normalized_master),
    )
    fetch_start = as_of_date - timedelta(days=max(lookback_days * 4, 45))
    generated_at = _generated_timestamp()
    enriched_records: list[dict[str, Any]] = []
    total_symbols = len(normalized_master)
    progress_interval = max(25, min(200, total_symbols // 10 or 1))

    for index, record in enumerate(normalized_master.to_dict(orient="records"), start=1):
        symbol = str(record["symbol"]).upper()
        enriched = dict(record)
        try:
            bars = daily_bar_provider.fetch_daily_bars(
                symbol,
                fetch_start,
                as_of_date,
                refresh_cache=refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning("Skipping liquidity enrichment for %s due to provider error: %s", symbol, exc)
            bars = pd.DataFrame()

        if not bars.empty:
            recent_bars = bars.tail(lookback_days)
            if not recent_bars.empty:
                enriched["price"] = float(recent_bars["close"].iloc[-1])
                enriched["average_daily_volume"] = float(recent_bars["volume"].mean())
                enriched["dollar_volume"] = float(
                    (recent_bars["close"] * recent_bars["volume"]).mean()
                )

        enriched["is_etf"] = _infer_is_etf(
            str(enriched.get("type", "")),
            name=str(enriched.get("name", "")),
            industry=str(enriched.get("industry", "")),
        )
        enriched["is_adr"] = _infer_is_adr(
            str(enriched.get("type", "")),
            name=str(enriched.get("name", "")),
            industry=str(enriched.get("industry", "")),
        )
        enriched["is_common_stock"] = _infer_is_common_stock(
            str(enriched.get("type", "")),
            name=str(enriched.get("name", "")),
            industry=str(enriched.get("industry", "")),
        )
        enriched["updated_at"] = generated_at
        enriched_records.append(enriched)

        if index % progress_interval == 0 or index == total_symbols:
            LOGGER.info(
                "Metadata enrichment progress: %s/%s symbols processed.",
                index,
                total_symbols,
            )

    LOGGER.info("Metadata enrichment completed for %s symbols.", len(enriched_records))
    return normalize_master_universe_frame(pd.DataFrame(enriched_records))


def apply_universe_filters(
    master_frame: pd.DataFrame,
    *,
    profile_name: str,
    profile_config: UniverseProfileConfig,
    reference_provider: ReferenceUniverseProvider | None = None,
    as_of_date: date | None = None,
    refresh_cache: bool = False,
) -> UniverseProfileBuildResult:
    """Apply one configured profile to the enriched master universe."""

    normalized_master = normalize_master_universe_frame(master_frame)
    base_frame = normalized_master.copy()
    start_count = int(len(base_frame))

    if profile_config.active_only:
        base_frame = base_frame[base_frame["active"]]

    if profile_config.allowed_exchanges:
        allowed_exchanges = {value.strip().upper() for value in profile_config.allowed_exchanges}
        base_frame = base_frame[base_frame["exchange"].str.upper().isin(allowed_exchanges)]

    if profile_config.allowed_ticker_types:
        allowed_types = {value.strip().upper() for value in profile_config.allowed_ticker_types}
        base_frame = base_frame[base_frame["type"].str.upper().isin(allowed_types)]

    if profile_config.common_stock_only:
        base_frame = base_frame[base_frame["is_common_stock"]]

    if not profile_config.include_etfs:
        base_frame = base_frame[~base_frame["is_etf"]]

    if not profile_config.include_adrs:
        base_frame = base_frame[~base_frame["is_adr"]]

    if profile_config.exclude_weird_instruments:
        base_frame = base_frame[~base_frame.apply(_row_looks_like_weird_instrument, axis=1)]

    if profile_config.min_price is not None:
        base_frame = base_frame[base_frame["price"] >= float(profile_config.min_price)]

    if profile_config.min_average_daily_volume is not None:
        base_frame = base_frame[
            base_frame["average_daily_volume"] >= float(profile_config.min_average_daily_volume)
        ]

    if profile_config.min_dollar_volume is not None:
        base_frame = base_frame[base_frame["dollar_volume"] >= float(profile_config.min_dollar_volume)]

    exclude_symbols = {symbol.strip().upper() for symbol in profile_config.exclude_symbols}
    if exclude_symbols:
        base_frame = base_frame[~base_frame["symbol"].isin(exclude_symbols)]

    base_frame = _sort_candidates(base_frame)
    post_liquidity_count = int(len(base_frame))

    detail_enriched_count = 0
    if _profile_requires_reference_details(profile_config, base_frame):
        if reference_provider is None:
            LOGGER.warning(
                "Profile %s requires reference detail filters, but no reference provider is available. "
                "Using the existing master-universe metadata only.",
                profile_name,
            )
        else:
            LOGGER.info(
                "Applying filters for profile %s: enriching reference details for %s candidate symbols.",
                profile_name,
                len(base_frame),
            )
            base_frame = _enrich_frame_with_reference_details(
                base_frame,
                reference_provider=reference_provider,
                as_of_date=as_of_date,
                refresh_cache=refresh_cache,
                progress_label=profile_name,
            )
            detail_enriched_count = int(len(base_frame))

    if profile_config.min_market_cap is not None:
        base_frame = base_frame[base_frame["market_cap"] >= float(profile_config.min_market_cap)]

    if profile_config.allowed_sectors:
        base_frame = base_frame[
            base_frame["sector"].apply(
                lambda value: _matches_keywords(value, profile_config.allowed_sectors)
            )
        ]

    if profile_config.allowed_industries:
        base_frame = base_frame[
            base_frame["industry"].apply(
                lambda value: _matches_keywords(value, profile_config.allowed_industries)
            )
        ]

    base_frame = _sort_candidates(base_frame)
    preference_removed_symbols: tuple[str, ...] = ()
    if profile_config.preferred_symbol_groups:
        base_frame, preference_removed_symbols = _apply_preferred_symbol_groups(
            base_frame,
            preferred_symbol_groups=profile_config.preferred_symbol_groups,
        )

    filtered_count = int(len(base_frame))

    include_symbols = tuple(symbol.strip().upper() for symbol in profile_config.include_symbols)
    manual_frame, missing_from_master = _build_manual_include_frame(
        normalized_master,
        include_symbols=include_symbols,
    )
    if not manual_frame.empty:
        base_frame = base_frame[~base_frame["symbol"].isin(set(include_symbols))]
        final_frame = pd.concat([manual_frame, base_frame], ignore_index=True)
    else:
        final_frame = base_frame.copy()

    if profile_config.max_symbols is not None:
        final_frame = final_frame.head(profile_config.max_symbols)

    final_frame = normalize_master_universe_frame(final_frame, sort_by_symbol=False)
    symbols = tuple(final_frame["symbol"].tolist())
    summary = {
        "profile": profile_name,
        "description": profile_config.description,
        "filters": asdict(profile_config),
        "input_master_count": start_count,
        "post_liquidity_count": post_liquidity_count,
        "detail_enriched_count": detail_enriched_count,
        "filtered_count": filtered_count,
        "force_include_count": len(include_symbols),
        "missing_force_include_symbols": list(missing_from_master),
        "preference_removed_symbols": list(preference_removed_symbols),
        "output_count": len(symbols),
        "symbols": list(symbols),
    }
    return UniverseProfileBuildResult(
        profile_name=profile_name,
        frame=final_frame,
        symbols=symbols,
        summary=summary,
    )


def write_universe_outputs(
    *,
    project_root: Path,
    master_config: UniverseMasterConfig,
    raw_reference_frame: pd.DataFrame,
    master_frame: pd.DataFrame,
    profile_results: dict[str, UniverseProfileBuildResult],
) -> dict[str, Any]:
    """Write processed master data plus profile symbol outputs and summaries."""

    processed_dir = (project_root / master_config.processed_output_dir).resolve()
    candidate_dir = (project_root / master_config.candidate_output_dir).resolve()
    summary_dir = processed_dir / master_config.profile_summary_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    raw_reference_path = processed_dir / master_config.raw_reference_filename
    master_universe_path = processed_dir / master_config.master_universe_filename
    normalize_master_universe_frame(raw_reference_frame).to_csv(raw_reference_path, index=False)
    normalize_master_universe_frame(master_frame).to_csv(master_universe_path, index=False)

    profile_payloads: dict[str, Any] = {}
    for profile_name, result in profile_results.items():
        candidate_path = candidate_dir / f"candidate_symbols_{profile_name}.txt"
        summary_path = summary_dir / f"{profile_name}.json"
        write_candidate_symbols(result.symbols, candidate_path)
        summary_payload = dict(result.summary)
        summary_payload["candidate_output_path"] = str(candidate_path)
        summary_payload["summary_output_path"] = str(summary_path)
        summary_payload["master_universe_path"] = str(master_universe_path)
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        profile_payloads[profile_name] = {
            "count": len(result.symbols),
            "candidate_output_path": str(candidate_path),
            "summary_output_path": str(summary_path),
        }

    return {
        "raw_reference_output_path": str(raw_reference_path),
        "master_universe_output_path": str(master_universe_path),
        "profiles": profile_payloads,
    }


def write_candidate_symbols(symbols: tuple[str, ...] | list[str], output_path: Path) -> Path:
    """Write one plain-text symbol file compatible with the rest of the bot."""

    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(symbol.strip().upper() for symbol in symbols if symbol.strip())
    if payload:
        payload = f"{payload}\n"
    resolved_output.write_text(payload, encoding="utf-8")
    return resolved_output


def _load_grouped_market_metrics(
    reference_provider: ReferenceUniverseProvider,
    *,
    as_of_date: date,
    lookback_days: int,
    refresh_cache: bool,
) -> tuple[dict[str, dict[str, float]], int]:
    grouped_by_symbol: dict[str, dict[str, list[float] | float]] = {}
    trading_days = 0
    search_date = as_of_date
    max_calendar_days = max(lookback_days * 4, 45)

    while trading_days < lookback_days and max_calendar_days > 0:
        daily_records = reference_provider.fetch_grouped_daily_market(
            search_date,
            refresh_cache=refresh_cache,
        )
        if daily_records:
            trading_days += 1
            for symbol, record in daily_records.items():
                close_value = _as_optional_float(record.get("c"))
                volume_value = _as_optional_float(record.get("v"))
                if close_value is None or volume_value is None:
                    continue
                symbol_metrics = grouped_by_symbol.setdefault(
                    symbol,
                    {
                        "closes": [],
                        "volumes": [],
                        "last_close": close_value,
                    },
                )
                closes = symbol_metrics["closes"]
                volumes = symbol_metrics["volumes"]
                if isinstance(closes, list):
                    closes.append(close_value)
                if isinstance(volumes, list):
                    volumes.append(volume_value)

            if trading_days == 1 or trading_days % 5 == 0 or trading_days == lookback_days:
                LOGGER.info(
                    "Metadata enrichment progress: collected grouped daily data for %s/%s trading days (latest %s).",
                    trading_days,
                    lookback_days,
                    search_date.isoformat(),
                )

        search_date -= timedelta(days=1)
        max_calendar_days -= 1

    if trading_days == 0:
        raise DataProviderError(
            f"No grouped daily market data was available in the lookback window ending {as_of_date.isoformat()}."
        )
    if trading_days < lookback_days:
        LOGGER.warning(
            "Only %s grouped trading days were available for the requested %s-day lookback ending %s.",
            trading_days,
            lookback_days,
            as_of_date.isoformat(),
        )

    metrics: dict[str, dict[str, float]] = {}
    for symbol, symbol_metrics in grouped_by_symbol.items():
        closes = symbol_metrics.get("closes")
        volumes = symbol_metrics.get("volumes")
        last_close = symbol_metrics.get("last_close")
        if not isinstance(closes, list) or not isinstance(volumes, list) or not closes or not volumes:
            continue
        metrics[symbol] = {
            "price": float(last_close) if isinstance(last_close, (int, float)) else float(closes[-1]),
            "average_daily_volume": float(sum(volumes) / len(volumes)),
            "dollar_volume": float(
                sum(close * volume for close, volume in zip(closes, volumes)) / len(closes)
            ),
        }
    return metrics, trading_days


def _apply_grouped_market_metrics(
    master_frame: pd.DataFrame,
    grouped_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    enriched_records: list[dict[str, Any]] = []
    generated_at = _generated_timestamp()

    for record in normalize_master_universe_frame(master_frame).to_dict(orient="records"):
        enriched = dict(record)
        metrics = grouped_metrics.get(str(record["symbol"]).upper())
        if metrics is not None:
            enriched["price"] = metrics.get("price")
            enriched["average_daily_volume"] = metrics.get("average_daily_volume")
            enriched["dollar_volume"] = metrics.get("dollar_volume")
        enriched["updated_at"] = generated_at
        enriched_records.append(enriched)

    return normalize_master_universe_frame(pd.DataFrame(enriched_records))


def normalize_master_universe_frame(
    frame: pd.DataFrame,
    *,
    sort_by_symbol: bool = True,
) -> pd.DataFrame:
    """Normalize master-universe data into a stable schema and sort order."""

    if frame.empty:
        return pd.DataFrame(columns=list(MASTER_UNIVERSE_COLUMNS))

    normalized = frame.copy()
    defaults: dict[str, Any] = {
        "symbol": "",
        "name": "",
        "active": False,
        "exchange": "",
        "market": "",
        "type": "",
        "market_cap": None,
        "sector": "",
        "industry": "",
        "price": None,
        "average_daily_volume": None,
        "dollar_volume": None,
        "is_etf": False,
        "is_adr": False,
        "is_common_stock": False,
        "source": "",
        "updated_at": "",
    }
    for column, default_value in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default_value

    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.strip().str.upper()
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(subset=["symbol"], keep="first")

    for column in ("name", "exchange", "market", "type", "sector", "industry", "source", "updated_at"):
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    for column in ("active", "is_etf", "is_adr", "is_common_stock"):
        normalized[column] = _coerce_bool_series(normalized[column])

    for column in ("market_cap", "price", "average_daily_volume", "dollar_volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.loc[:, list(MASTER_UNIVERSE_COLUMNS)]
    if sort_by_symbol:
        normalized = normalized.sort_values(by=["symbol"], kind="stable")
    normalized = normalized.reset_index(drop=True)
    return normalized


def _apply_reference_details(record: dict[str, Any], details: dict[str, Any]) -> None:
    if not details:
        return

    record["name"] = _first_non_empty(details.get("name"), record.get("name"))
    record["exchange"] = _first_non_empty(
        details.get("primary_exchange"),
        details.get("exchange"),
        record.get("exchange"),
    )
    record["market"] = _first_non_empty(details.get("market"), record.get("market"))
    record["type"] = _first_non_empty(details.get("type"), record.get("type"))
    record["market_cap"] = _first_non_empty(details.get("market_cap"), record.get("market_cap"))
    record["sector"] = _extract_sector(details, fallback=str(record.get("sector", "")))
    record["industry"] = _extract_industry(details, fallback=str(record.get("industry", "")))
    if "active" in details:
        record["active"] = bool(details.get("active"))


def _extract_sector(details: dict[str, Any], *, fallback: str) -> str:
    classification = details.get("standard_industrial_classification")
    if isinstance(classification, dict):
        sector = _as_clean_string(classification.get("sector"))
        if sector:
            return sector
    for key in ("sector", "sector_name", "gics_sector"):
        value = _as_clean_string(details.get(key))
        if value:
            return value
    return _as_clean_string(fallback)


def _extract_industry(details: dict[str, Any], *, fallback: str) -> str:
    classification = details.get("standard_industrial_classification")
    if isinstance(classification, dict):
        description = _as_clean_string(classification.get("description"))
        if description:
            return description
    for key in ("industry", "industry_name", "sic_description"):
        value = _as_clean_string(details.get(key))
        if value:
            return value
    return _as_clean_string(fallback)


def _build_manual_include_frame(
    master_frame: pd.DataFrame,
    *,
    include_symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not include_symbols:
        return pd.DataFrame(columns=list(MASTER_UNIVERSE_COLUMNS)), ()

    generated_at = _generated_timestamp()
    indexed_master = normalize_master_universe_frame(master_frame).set_index("symbol", drop=False)
    manual_records: list[dict[str, Any]] = []
    missing_symbols: list[str] = []
    for symbol in include_symbols:
        if symbol in indexed_master.index:
            row = indexed_master.loc[symbol]
            if isinstance(row, pd.DataFrame):
                manual_records.append(row.iloc[0].to_dict())
            else:
                manual_records.append(row.to_dict())
            continue

        missing_symbols.append(symbol)
        manual_records.append(
            {
                "symbol": symbol,
                "name": "",
                "active": False,
                "exchange": "",
                "market": "",
                "type": "",
                "market_cap": None,
                "sector": "",
                "industry": "",
                "price": None,
                "average_daily_volume": None,
                "dollar_volume": None,
                "is_etf": False,
                "is_adr": False,
                "is_common_stock": False,
                "source": "manual_override",
                "updated_at": generated_at,
            }
        )

    return normalize_master_universe_frame(
        pd.DataFrame(manual_records),
        sort_by_symbol=False,
    ), tuple(missing_symbols)


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return normalize_master_universe_frame(frame)
    sorted_frame = frame.sort_values(
        by=["dollar_volume", "market_cap", "average_daily_volume", "symbol"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="stable",
    )
    return normalize_master_universe_frame(sorted_frame, sort_by_symbol=False)


def _apply_preferred_symbol_groups(
    frame: pd.DataFrame,
    *,
    preferred_symbol_groups: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    normalized_frame = normalize_master_universe_frame(frame, sort_by_symbol=False)
    if normalized_frame.empty or not preferred_symbol_groups:
        return normalized_frame, ()

    working = normalized_frame.copy()
    working["_profile_rank"] = range(len(working))

    symbol_to_group: dict[str, str] = {}
    for preferred_symbol, group_symbols in preferred_symbol_groups.items():
        for symbol in group_symbols:
            symbol_to_group[symbol] = preferred_symbol

    kept_symbols: set[str] = set()
    removed_symbols: list[str] = []
    effective_ranks: dict[str, int] = {}

    for preferred_symbol, group_symbols in preferred_symbol_groups.items():
        present = working[working["symbol"].isin(group_symbols)].copy()
        if present.empty:
            continue

        present = present.sort_values(by=["_profile_rank"], kind="stable")
        keep_symbol = preferred_symbol if preferred_symbol in set(present["symbol"]) else str(
            present.iloc[0]["symbol"]
        )
        kept_symbols.add(keep_symbol)
        effective_ranks[keep_symbol] = int(present["_profile_rank"].min())
        removed_symbols.extend(
            symbol for symbol in present["symbol"].tolist() if symbol != keep_symbol
        )

    filtered = working[
        working["symbol"].apply(
            lambda symbol: symbol not in symbol_to_group or symbol in kept_symbols
        )
    ].copy()
    filtered["_effective_rank"] = filtered["symbol"].map(effective_ranks)
    filtered["_effective_rank"] = filtered["_effective_rank"].fillna(filtered["_profile_rank"])
    filtered = filtered.sort_values(
        by=["_effective_rank", "_profile_rank"],
        kind="stable",
    ).drop(columns=["_profile_rank", "_effective_rank"])

    return (
        normalize_master_universe_frame(filtered, sort_by_symbol=False),
        tuple(removed_symbols),
    )


def _profile_requires_reference_details(
    profile_config: UniverseProfileConfig,
    frame: pd.DataFrame,
) -> bool:
    if frame.empty:
        return False
    if profile_config.min_market_cap is not None:
        return True
    if profile_config.allowed_sectors:
        return True
    if profile_config.allowed_industries and frame["industry"].fillna("").eq("").any():
        return True
    return False


def _enrich_frame_with_reference_details(
    frame: pd.DataFrame,
    *,
    reference_provider: ReferenceUniverseProvider,
    as_of_date: date | None,
    refresh_cache: bool,
    progress_label: str,
    batch_size: int = 50,
    max_workers: int = 8,
) -> pd.DataFrame:
    normalized_frame = normalize_master_universe_frame(frame, sort_by_symbol=False)
    if normalized_frame.empty:
        return normalized_frame

    source_records = normalized_frame.to_dict(orient="records")
    enriched_records: list[dict[str, Any]] = []
    total_symbols = len(source_records)
    completed_symbols = 0
    successful_details = 0
    failed_details = 0

    for batch_start in range(0, total_symbols, batch_size):
        batch_records = source_records[batch_start : batch_start + batch_size]
        resolved_workers = max(1, min(max_workers, len(batch_records)))
        detail_payloads: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    reference_provider.fetch_ticker_details,
                    str(record["symbol"]),
                    as_of_date=as_of_date,
                    refresh_cache=refresh_cache,
                ): str(record["symbol"])
                for record in batch_records
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    detail_payloads[symbol] = future.result()
                    successful_details += 1
                except DataProviderError as exc:
                    failed_details += 1
                    LOGGER.warning(
                        "Reference detail lookup failed for %s during profile %s: %s",
                        symbol,
                        progress_label,
                        exc,
                    )

        for record in batch_records:
            enriched = dict(record)
            details = detail_payloads.get(str(record["symbol"]))
            if details:
                _apply_reference_details(enriched, details)
            enriched["is_etf"] = _infer_is_etf(
                str(enriched.get("type", "")),
                name=str(enriched.get("name", "")),
                industry=str(enriched.get("industry", "")),
            )
            enriched["is_adr"] = _infer_is_adr(
                str(enriched.get("type", "")),
                name=str(enriched.get("name", "")),
                industry=str(enriched.get("industry", "")),
            )
            enriched["is_common_stock"] = _infer_is_common_stock(
                str(enriched.get("type", "")),
                name=str(enriched.get("name", "")),
                industry=str(enriched.get("industry", "")),
            )
            enriched_records.append(enriched)

        completed_symbols += len(batch_records)
        LOGGER.info(
            "Reference detail enrichment progress for profile %s: %s/%s symbols completed (%s successful, %s failed).",
            progress_label,
            completed_symbols,
            total_symbols,
            successful_details,
            failed_details,
        )

    if successful_details == 0 and total_symbols > 0:
        raise DataProviderError(
            f"Reference detail enrichment failed for all {total_symbols} symbols in profile '{progress_label}'."
        )

    return normalize_master_universe_frame(pd.DataFrame(enriched_records), sort_by_symbol=False)


def _row_looks_like_weird_instrument(row: pd.Series) -> bool:
    return _looks_like_weird_instrument(
        symbol=str(row.get("symbol", "")),
        type_code=str(row.get("type", "")),
        name=str(row.get("name", "")),
        industry=str(row.get("industry", "")),
    )


def _looks_like_weird_instrument(
    *,
    symbol: str,
    type_code: str,
    name: str,
    industry: str,
) -> bool:
    normalized_type = type_code.strip().upper()
    if normalized_type in ETF_TYPE_CODES or normalized_type in COMMON_STOCK_TYPE_CODES:
        return False
    if normalized_type in WEIRD_TYPE_CODES:
        return True

    normalized_symbol = symbol.strip().upper()
    if any(normalized_symbol.endswith(pattern) for pattern in WEIRD_INSTRUMENT_PATTERNS):
        return True

    text = f" {name} {industry} ".upper()
    return any(token in text for token in (" WARRANT ", " RIGHT ", " RIGHTS ", " UNIT "))


def _infer_is_etf(type_code: str, *, name: str, industry: str) -> bool:
    normalized_type = type_code.strip().upper()
    if normalized_type in ETF_TYPE_CODES:
        return True
    text = f" {name} {industry} ".upper()
    return " ETF " in text or "EXCHANGE TRADED FUND" in text


def _infer_is_adr(type_code: str, *, name: str, industry: str) -> bool:
    normalized_type = type_code.strip().upper()
    if any(token in normalized_type for token in ADR_TYPE_TOKENS):
        return True
    text = f" {name} {industry} ".upper()
    return any(token in text for token in (" ADR ", " ADS ", " AMERICAN DEPOSITARY ", " DEPOSITARY "))


def _infer_is_common_stock(type_code: str, *, name: str, industry: str) -> bool:
    normalized_type = type_code.strip().upper()
    if normalized_type in COMMON_STOCK_TYPE_CODES:
        return True
    if _infer_is_etf(type_code, name=name, industry=industry):
        return False
    if _infer_is_adr(type_code, name=name, industry=industry):
        return False
    text = f" {name} {industry} ".upper()
    return " COMMON STOCK " in text


def _matches_keywords(value: object, keywords: tuple[str, ...]) -> bool:
    normalized_value = str(value or "").casefold()
    if not normalized_value:
        return False
    return any(keyword.casefold() in normalized_value for keyword in keywords if keyword.strip())


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def convert(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}

    return series.apply(convert)


def _first_non_empty(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return ""


def _as_clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _generated_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
