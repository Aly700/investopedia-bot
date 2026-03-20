"""Universe construction utilities for daily-bar research workflows."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
from typing import Sequence

from bot.config import UniverseConfig
from bot.data.providers import DataProviderError, DailyBarProvider


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseMember:
    """A symbol that passed the universe screen."""

    symbol: str
    last_close: float
    average_dollar_volume: float
    bars_used: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly representation of the screened symbol."""

        return asdict(self)


class UniverseBuilder:
    """Build a tradable candidate universe from recent daily bars."""

    def __init__(self, provider: DailyBarProvider, universe_config: UniverseConfig) -> None:
        self.provider = provider
        self.universe_config = universe_config

    def build(
        self,
        candidates: Sequence[str] | Path,
        *,
        as_of_date: date,
        lookback_days: int = 20,
        refresh_cache: bool = False,
        enforce_max_symbols: bool = True,
    ) -> list[str]:
        """Return eligible symbols after filtering and ranking."""

        return [
            member.symbol
            for member in self.screen_candidates(
                candidates,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
                refresh_cache=refresh_cache,
                enforce_max_symbols=enforce_max_symbols,
            )
        ]

    def screen_candidates(
        self,
        candidates: Sequence[str] | Path,
        *,
        as_of_date: date,
        lookback_days: int = 20,
        refresh_cache: bool = False,
        enforce_max_symbols: bool = True,
    ) -> list[UniverseMember]:
        """Return screened universe members with summary metrics."""

        if lookback_days <= 0:
            raise ValueError("lookback_days must be greater than zero.")

        symbols = (
            load_candidate_symbols(candidates) if isinstance(candidates, Path) else _dedupe_symbols(candidates)
        )
        if not symbols:
            return []

        start_date = as_of_date - timedelta(days=max(lookback_days * 3, 30))
        eligible_members: list[UniverseMember] = []

        for symbol in symbols:
            try:
                bars = self.provider.fetch_daily_bars(
                    symbol,
                    start_date,
                    as_of_date,
                    refresh_cache=refresh_cache,
                )
            except DataProviderError as exc:
                LOGGER.warning("Skipping %s due to provider error: %s", symbol, exc)
                continue

            if bars.empty:
                LOGGER.warning("Skipping %s because no daily bars were returned.", symbol)
                continue

            recent_bars = bars.tail(lookback_days)
            last_close = float(recent_bars["close"].iloc[-1])
            average_dollar_volume = float((recent_bars["close"] * recent_bars["volume"]).mean())
            if last_close < self.universe_config.min_price:
                continue
            if average_dollar_volume < float(self.universe_config.min_avg_dollar_volume):
                continue

            eligible_members.append(
                UniverseMember(
                    symbol=symbol,
                    last_close=last_close,
                    average_dollar_volume=average_dollar_volume,
                    bars_used=int(len(recent_bars)),
                )
            )

        eligible_members.sort(key=lambda member: (-member.average_dollar_volume, member.symbol))
        if enforce_max_symbols:
            return eligible_members[: self.universe_config.max_symbols]
        return eligible_members


def load_candidate_symbols(path: Path) -> list[str]:
    """Load candidate symbols from a newline-delimited text file or CSV."""

    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Candidate symbol file does not exist: {resolved_path}")

    if resolved_path.suffix.lower() == ".csv":
        return _load_symbols_from_csv(resolved_path)
    return _load_symbols_from_text(resolved_path)


def _load_symbols_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        symbol_column = "symbol" if "symbol" in reader.fieldnames else reader.fieldnames[0]
        symbols = [row.get(symbol_column, "") for row in reader]
    return _dedupe_symbols(symbols)


def _load_symbols_from_text(path: Path) -> list[str]:
    raw_symbols: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        raw_symbols.extend(part.strip() for part in line.split(","))
    return _dedupe_symbols(raw_symbols)


def _dedupe_symbols(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered_symbols: list[str] = []
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered_symbols.append(symbol)
    return ordered_symbols
