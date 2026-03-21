"""Sector ETF mapping and derived sector-context features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from bot.config import AppConfig
from bot.data.providers import DailyBarProvider, DataProviderConfigurationError, DataProviderError
from bot.data.reference import create_reference_universe_provider
from bot.strategy.regime_filter import BenchmarkRegimeSettings, regime_is_bullish


LOGGER = logging.getLogger(__name__)

_SECTOR_ETF_RULES: tuple[tuple[str, str, str], ...] = (
    ("information technology", "XLK", "Technology"),
    ("technology", "XLK", "Technology"),
    ("financial", "XLF", "Financials"),
    ("energy", "XLE", "Energy"),
    ("health care", "XLV", "Health Care"),
    ("healthcare", "XLV", "Health Care"),
    ("consumer discretionary", "XLY", "Consumer Discretionary"),
    ("industrials", "XLI", "Industrials"),
    ("industrial", "XLI", "Industrials"),
    ("materials", "XLB", "Materials"),
    ("real estate", "XLRE", "Real Estate"),
    ("utilities", "XLU", "Utilities"),
    ("communication services", "XLC", "Communication Services"),
    ("communication", "XLC", "Communication Services"),
    ("consumer staples", "XLP", "Consumer Staples"),
)

_INDUSTRY_ETF_HINTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        (
            "semiconductor",
            "software",
            "cloud",
            "computer",
            "internet software",
            "communications equipment",
            "electronic component",
            "data processing",
            "it service",
        ),
        "XLK",
        "Technology",
    ),
    (
        (
            "bank",
            "insurance",
            "asset management",
            "capital market",
            "consumer finance",
            "financial",
        ),
        "XLF",
        "Financials",
    ),
    (
        (
            "oil",
            "gas",
            "petroleum",
            "exploration",
            "pipeline",
            "drilling",
            "refining",
        ),
        "XLE",
        "Energy",
    ),
    (
        (
            "biotech",
            "pharmaceutical",
            "drug",
            "medical",
            "health",
            "diagnostic",
            "hospital",
        ),
        "XLV",
        "Health Care",
    ),
    (
        (
            "retail",
            "apparel",
            "auto",
            "restaurant",
            "hotel",
            "leisure",
            "travel",
            "homebuilding",
        ),
        "XLY",
        "Consumer Discretionary",
    ),
    (
        (
            "machinery",
            "aerospace",
            "defense",
            "transportation",
            "railroad",
            "air freight",
            "industrial",
            "construction",
        ),
        "XLI",
        "Industrials",
    ),
    (
        (
            "chemical",
            "metal",
            "mining",
            "paper",
            "forest",
            "container",
            "packaging",
        ),
        "XLB",
        "Materials",
    ),
    (
        (
            "reit",
            "real estate",
            "property",
            "mortgage reit",
        ),
        "XLRE",
        "Real Estate",
    ),
    (
        (
            "electric utility",
            "gas utility",
            "water utility",
            "independent power",
            "renewable utility",
        ),
        "XLU",
        "Utilities",
    ),
    (
        (
            "telecom",
            "media",
            "entertainment",
            "broadcast",
            "publishing",
            "interactive media",
        ),
        "XLC",
        "Communication Services",
    ),
    (
        (
            "beverage",
            "food",
            "household",
            "tobacco",
            "personal product",
            "consumer staples",
            "grocery",
        ),
        "XLP",
        "Consumer Staples",
    ),
)


@dataclass(frozen=True)
class SymbolSectorClassification:
    """Normalized sector/industry classification for one symbol."""

    symbol: str
    sector: str | None
    industry: str | None
    sector_etf_symbol: str | None
    mapping_source: str | None


@dataclass(frozen=True)
class SectorFeatureContext:
    """Derived sector ETF context for one symbol."""

    symbol: str
    sector_name: str | None
    industry_name: str | None
    sector_etf_symbol: str | None
    sector_regime_passed: bool | None
    sector_return: float | None
    symbol_return: float | None
    relative_strength_vs_sector: float | None
    relative_strength_window: int
    sector_trend_state: str | None
    mapping_source: str | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sector_name": self.sector_name,
            "industry_name": self.industry_name,
            "sector_etf_symbol": self.sector_etf_symbol,
            "sector_regime_passed": self.sector_regime_passed,
            "sector_return": self.sector_return,
            "symbol_return": self.symbol_return,
            "relative_strength_vs_sector": self.relative_strength_vs_sector,
            "sector_relative_strength_window": self.relative_strength_window,
            "sector_trend_state": self.sector_trend_state,
            "sector_mapping_source": self.mapping_source,
            "sector_context_available": self.sector_etf_symbol is not None,
        }


def sector_context_history_start(
    as_of_date: date,
    *,
    benchmark_sma_slow: int,
    relative_strength_window: int,
) -> date:
    """Return a conservative daily-bar start date for sector context."""

    required_sessions = max(benchmark_sma_slow, relative_strength_window + 1, 30)
    return as_of_date - timedelta(days=required_sessions * 3)


def sector_context_metadata(context: SectorFeatureContext | None) -> dict[str, Any]:
    """Return JSON-friendly sector metadata for one symbol."""

    if context is None:
        return {
            "sector_name": None,
            "industry_name": None,
            "sector_etf_symbol": None,
            "sector_regime_passed": None,
            "sector_return": None,
            "symbol_return": None,
            "relative_strength_vs_sector": None,
            "sector_relative_strength_window": None,
            "sector_trend_state": None,
            "sector_mapping_source": None,
            "sector_context_available": False,
        }
    return context.to_metadata()


def load_symbol_sector_classifications(
    symbols: Sequence[str],
    *,
    config: AppConfig,
    env_file: Path | None = None,
    as_of_date: date | None = None,
    refresh_cache: bool = False,
) -> dict[str, SymbolSectorClassification]:
    """Load sector classifications from local universe data and reference details."""

    normalized_symbols = sorted({_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()})
    if not normalized_symbols:
        return {}

    classifications = _load_local_master_universe_classifications(config, normalized_symbols)
    unresolved_symbols = [symbol for symbol in normalized_symbols if symbol not in classifications]
    if unresolved_symbols:
        try:
            reference_provider = create_reference_universe_provider(config, env_file=env_file)
        except DataProviderConfigurationError:
            reference_provider = None

        if reference_provider is not None:
            for symbol in unresolved_symbols:
                try:
                    details = reference_provider.fetch_ticker_details(
                        symbol,
                        as_of_date=as_of_date,
                        refresh_cache=refresh_cache,
                    )
                except DataProviderError as exc:
                    LOGGER.warning(
                        "Skipping sector classification for %s due to reference-data error: %s",
                        symbol,
                        exc,
                    )
                    continue
                classification = _classification_from_reference_details(
                    symbol,
                    details,
                    source="reference_details",
                )
                if (
                    classification.sector is None
                    and classification.industry is None
                    and classification.sector_etf_symbol is None
                ):
                    continue
                classifications[symbol] = classification

    return classifications


def build_sector_feature_contexts(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    provider: DailyBarProvider,
    as_of_date: date,
    history_start: date,
    sector_classifications: Mapping[str, SymbolSectorClassification],
    benchmark_sma_fast: int,
    benchmark_sma_slow: int,
    relative_strength_window: int,
    refresh_cache: bool = False,
) -> dict[str, SectorFeatureContext]:
    """Build per-symbol sector ETF context from symbol bars and mapped sector ETFs."""

    if benchmark_sma_fast <= 0 or benchmark_sma_slow <= 0:
        raise ValueError("benchmark SMA windows must be greater than zero.")
    if benchmark_sma_fast >= benchmark_sma_slow:
        raise ValueError("benchmark_sma_fast must be less than benchmark_sma_slow.")
    if relative_strength_window <= 0:
        raise ValueError("relative_strength_window must be greater than zero.")

    prepared_symbol_frames: dict[str, pd.DataFrame] = {}
    for raw_symbol, frame in symbol_frames.items():
        symbol = _normalize_symbol(raw_symbol)
        if frame.empty:
            continue
        prepared_symbol_frames[symbol] = _prepare_daily_frame(frame)

    needed_sector_etfs = sorted(
        {
            classification.sector_etf_symbol
            for symbol, classification in sector_classifications.items()
            if symbol in prepared_symbol_frames and classification.sector_etf_symbol is not None
        }
    )

    sector_frames: dict[str, pd.DataFrame] = {}
    for sector_etf_symbol in needed_sector_etfs:
        if sector_etf_symbol is None:
            continue
        try:
            bars = provider.fetch_daily_bars(
                sector_etf_symbol,
                history_start,
                as_of_date,
                refresh_cache=refresh_cache,
            )
        except DataProviderError as exc:
            LOGGER.warning(
                "Skipping sector ETF %s due to provider error: %s",
                sector_etf_symbol,
                exc,
            )
            continue
        if bars.empty:
            continue
        sector_frames[sector_etf_symbol] = _prepare_daily_frame(bars)

    contexts: dict[str, SectorFeatureContext] = {}
    for symbol, symbol_frame in prepared_symbol_frames.items():
        classification = sector_classifications.get(symbol)
        if classification is None or classification.sector_etf_symbol is None:
            continue

        sector_frame = sector_frames.get(classification.sector_etf_symbol)
        sector_regime_passed: bool | None = None
        sector_trend_state: str | None = None
        sector_return: float | None = None
        symbol_return = _trailing_return(symbol_frame, relative_strength_window)
        relative_strength_vs_sector: float | None = None

        if sector_frame is not None:
            if len(sector_frame) >= benchmark_sma_slow:
                sector_regime_passed = regime_is_bullish(
                    sector_frame,
                    BenchmarkRegimeSettings(
                        benchmark_symbol=classification.sector_etf_symbol,
                        fast_window=benchmark_sma_fast,
                        slow_window=benchmark_sma_slow,
                        mode="either",
                        enabled=True,
                    ),
                )
                sector_trend_state = "bullish" if sector_regime_passed else "weak"
            sector_return = _trailing_return(sector_frame, relative_strength_window)
            if symbol_return is not None and sector_return is not None:
                relative_strength_vs_sector = symbol_return - sector_return

        contexts[symbol] = SectorFeatureContext(
            symbol=symbol,
            sector_name=classification.sector,
            industry_name=classification.industry,
            sector_etf_symbol=classification.sector_etf_symbol,
            sector_regime_passed=sector_regime_passed,
            sector_return=sector_return,
            symbol_return=symbol_return,
            relative_strength_vs_sector=relative_strength_vs_sector,
            relative_strength_window=relative_strength_window,
            sector_trend_state=sector_trend_state,
            mapping_source=classification.mapping_source,
        )

    return contexts


def resolve_sector_etf(
    sector: str | None,
    *,
    industry: str | None = None,
) -> tuple[str | None, str | None]:
    """Map sector/industry text to a canonical sector ETF and sector name."""

    cleaned_sector = _clean_text(sector)
    cleaned_industry = _clean_text(industry)
    normalized_sector = _normalize_text(cleaned_sector)
    normalized_industry = _normalize_text(cleaned_industry)

    for keyword, sector_etf_symbol, canonical_sector in _SECTOR_ETF_RULES:
        if keyword in normalized_sector:
            return sector_etf_symbol, canonical_sector

    for keywords, sector_etf_symbol, canonical_sector in _INDUSTRY_ETF_HINTS:
        if any(keyword in normalized_industry for keyword in keywords):
            return sector_etf_symbol, canonical_sector

    return None, cleaned_sector


def _load_local_master_universe_classifications(
    config: AppConfig,
    symbols: Sequence[str],
) -> dict[str, SymbolSectorClassification]:
    path = (
        config.project_root
        / config.universe_builder.master.processed_output_dir
        / config.universe_builder.master.master_universe_filename
    )
    if not path.exists():
        return {}

    try:
        frame = pd.read_csv(path, usecols=["symbol", "sector", "industry"])
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        LOGGER.warning("Unable to read local master-universe sector metadata from %s: %s", path, exc)
        return {}

    filtered = frame[frame["symbol"].isin(symbols)].copy()
    classifications: dict[str, SymbolSectorClassification] = {}
    for record in filtered.to_dict(orient="records"):
        symbol = _normalize_symbol(record.get("symbol", ""))
        sector = _clean_text(record.get("sector"))
        industry = _clean_text(record.get("industry"))
        sector_etf_symbol, resolved_sector_name = resolve_sector_etf(sector, industry=industry)
        if sector_etf_symbol is None and resolved_sector_name is None and industry is None:
            continue
        classifications[symbol] = SymbolSectorClassification(
            symbol=symbol,
            sector=resolved_sector_name,
            industry=industry,
            sector_etf_symbol=sector_etf_symbol,
            mapping_source="master_universe",
        )
    return classifications


def _classification_from_reference_details(
    symbol: str,
    details: Mapping[str, Any],
    *,
    source: str,
) -> SymbolSectorClassification:
    sector = _extract_sector(details)
    industry = _extract_industry(details)
    sector_etf_symbol, resolved_sector_name = resolve_sector_etf(sector, industry=industry)
    return SymbolSectorClassification(
        symbol=_normalize_symbol(symbol),
        sector=resolved_sector_name,
        industry=industry,
        sector_etf_symbol=sector_etf_symbol,
        mapping_source=source,
    )


def _extract_sector(details: Mapping[str, Any]) -> str | None:
    classification = details.get("sic_classification")
    if isinstance(classification, Mapping):
        sector = _clean_text(classification.get("sector"))
        if sector is not None:
            return sector
    for key in ("sector", "sector_name", "gics_sector"):
        sector = _clean_text(details.get(key))
        if sector is not None:
            return sector
    return None


def _extract_industry(details: Mapping[str, Any]) -> str | None:
    classification = details.get("sic_classification")
    if isinstance(classification, Mapping):
        industry = _clean_text(classification.get("industry"))
        if industry is not None:
            return industry
    for key in ("industry", "industry_name", "sic_description"):
        industry = _clean_text(details.get(key))
        if industry is not None:
            return industry
    return None


def _prepare_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared_frame = frame.sort_values("date", kind="stable").reset_index(drop=True).copy()
    prepared_frame["close"] = pd.to_numeric(prepared_frame["close"], errors="coerce")
    prepared_frame = prepared_frame.dropna(subset=["close"])
    return prepared_frame


def _trailing_return(frame: pd.DataFrame, window: int) -> float | None:
    if len(frame) <= window:
        return None
    latest_close = float(frame["close"].iloc[-1])
    prior_close = float(frame["close"].iloc[-(window + 1)])
    if prior_close <= 0:
        return None
    return (latest_close / prior_close) - 1.0


def _normalize_symbol(symbol: object) -> str:
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string.")
    return normalized_symbol


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()
