"""Lightweight persistent candidate-memory journal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

from bot.logging_utils import get_logger


LOGGER = get_logger(__name__)

CANDIDATE_SCORE_JOURNAL_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_SCORE_JOURNAL_STALE_AFTER_DAYS = 30
CONSECUTIVE_SETUP_GAP_RESET_DAYS = 5


@dataclass(frozen=True)
class CandidateJournalObservation:
    """A symbol-level setup observation derived from one evaluation run."""

    symbol: str
    approved_today: bool
    breakout_strength: float | None
    relative_volume: float | None
    sector_etf_symbol: str | None = None
    sector_regime_passed: bool | None = None
    best_rank_today: int | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.best_rank_today is not None and self.best_rank_today <= 0:
            raise ValueError("best_rank_today must be greater than zero when provided.")


@dataclass(frozen=True)
class CandidateJournalEntry:
    """Persisted setup-memory state for one symbol."""

    symbol: str
    last_seen_date: date
    days_near_breakout: int
    days_approved: int
    last_rank: int | None = None
    best_rank: int | None = None
    peak_relative_volume: float | None = None
    peak_breakout_strength: float | None = None
    last_sector_etf_symbol: str | None = None
    last_sector_regime_passed: bool | None = None
    last_approved_date: date | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.days_near_breakout <= 0:
            raise ValueError("days_near_breakout must be greater than zero.")
        if self.days_approved < 0:
            raise ValueError("days_approved must be non-negative.")
        if self.last_rank is not None and self.last_rank <= 0:
            raise ValueError("last_rank must be greater than zero when provided.")
        if self.best_rank is not None and self.best_rank <= 0:
            raise ValueError("best_rank must be greater than zero when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly journal entry payload."""

        return {
            "last_seen_date": self.last_seen_date.isoformat(),
            "days_near_breakout": self.days_near_breakout,
            "days_approved": self.days_approved,
            "last_rank": self.last_rank,
            "best_rank": self.best_rank,
            "peak_relative_volume": self.peak_relative_volume,
            "peak_breakout_strength": self.peak_breakout_strength,
            "last_sector_etf_symbol": self.last_sector_etf_symbol,
            "last_sector_regime_passed": self.last_sector_regime_passed,
            "last_approved_date": (
                self.last_approved_date.isoformat()
                if self.last_approved_date is not None
                else None
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        data: Mapping[str, Any],
    ) -> "CandidateJournalEntry":
        """Build a journal entry from a stored JSON mapping."""

        return cls(
            symbol=symbol.strip().upper(),
            last_seen_date=_mapping_date(data, "last_seen_date"),
            days_near_breakout=_mapping_positive_int(data, "days_near_breakout"),
            days_approved=_mapping_non_negative_int(data, "days_approved"),
            last_rank=_mapping_optional_positive_int(data, "last_rank"),
            best_rank=_mapping_optional_positive_int(data, "best_rank"),
            peak_relative_volume=_mapping_optional_float(data, "peak_relative_volume"),
            peak_breakout_strength=_mapping_optional_float(data, "peak_breakout_strength"),
            last_sector_etf_symbol=_mapping_optional_text(data, "last_sector_etf_symbol"),
            last_sector_regime_passed=_mapping_optional_bool(
                data,
                "last_sector_regime_passed",
            ),
            last_approved_date=_mapping_optional_date(data, "last_approved_date"),
        )


@dataclass(frozen=True)
class CandidateScoreJournal:
    """Persisted candidate-memory state across evaluation dates."""

    entries: dict[str, CandidateJournalEntry]
    updated_at: date | None = None
    stale_after_days: int = DEFAULT_CANDIDATE_SCORE_JOURNAL_STALE_AFTER_DAYS
    schema_version: int = CANDIDATE_SCORE_JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stale_after_days <= 0:
            raise ValueError("stale_after_days must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly journal payload."""

        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at.isoformat() if self.updated_at is not None else None,
            "stale_after_days": self.stale_after_days,
            "symbols": {
                symbol: entry.to_dict()
                for symbol, entry in sorted(self.entries.items())
            },
        }


def default_candidate_score_journal_path(project_root: Path) -> Path:
    """Return the repo-relative journal path used by daily candidate workflows."""

    return project_root / "data" / "processed" / "state" / "candidate_scores.json"


def load_candidate_score_journal(
    journal_path: Path,
    *,
    stale_after_days: int = DEFAULT_CANDIDATE_SCORE_JOURNAL_STALE_AFTER_DAYS,
) -> CandidateScoreJournal:
    """Load the journal, falling back to an empty state on missing/corrupt files."""

    resolved_path = journal_path.resolve()
    if not resolved_path.exists():
        return CandidateScoreJournal(entries={}, stale_after_days=stale_after_days)

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Journal payload must be a mapping.")
        raw_symbols = payload.get("symbols", {})
        if not isinstance(raw_symbols, dict):
            raise ValueError("Journal payload 'symbols' must be a mapping.")

        entries: dict[str, CandidateJournalEntry] = {}
        for symbol, raw_entry in raw_symbols.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Journal symbol keys must be non-empty strings.")
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Journal entry for {symbol!r} must be a mapping.")
            normalized_symbol = symbol.strip().upper()
            entries[normalized_symbol] = CandidateJournalEntry.from_mapping(
                normalized_symbol,
                raw_entry,
            )

        updated_at = _optional_date(payload.get("updated_at"))
        configured_stale_after_days = payload.get("stale_after_days", stale_after_days)
        return CandidateScoreJournal(
            entries=entries,
            updated_at=updated_at,
            stale_after_days=_coerce_positive_int(
                configured_stale_after_days,
                "stale_after_days",
            ),
            schema_version=_coerce_positive_int(
                payload.get("schema_version", CANDIDATE_SCORE_JOURNAL_SCHEMA_VERSION),
                "schema_version",
            ),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "Ignoring candidate score journal at %s because it could not be loaded: %s",
            resolved_path,
            exc,
        )
        return CandidateScoreJournal(entries={}, stale_after_days=stale_after_days)


def write_candidate_score_journal(
    journal: CandidateScoreJournal,
    journal_path: Path,
) -> Path:
    """Persist the journal as JSON."""

    resolved_path = journal_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(journal.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_path


def build_candidate_memory_features(
    journal_entry: CandidateJournalEntry | None,
    *,
    observation: CandidateJournalObservation,
    as_of_date: date,
) -> dict[str, Any]:
    """Return preview memory features for ranking/rationale on the current date."""

    preview_entry = _next_candidate_journal_entry(
        journal_entry,
        observation=observation,
        as_of_date=as_of_date,
    )
    setup_quality_score = _candidate_setup_quality_score(preview_entry)
    repeated_high_quality_signal = (
        preview_entry.days_near_breakout >= 3
        or preview_entry.days_approved >= 2
        or setup_quality_score >= 0.8
    )
    return {
        "last_seen_date": preview_entry.last_seen_date.isoformat(),
        "setup_persistence_days": preview_entry.days_near_breakout,
        "days_near_breakout": preview_entry.days_near_breakout,
        "days_approved": preview_entry.days_approved,
        "last_rank": preview_entry.last_rank,
        "best_rank": preview_entry.best_rank,
        "peak_relative_volume": preview_entry.peak_relative_volume,
        "peak_breakout_strength": preview_entry.peak_breakout_strength,
        "last_sector_etf_symbol": preview_entry.last_sector_etf_symbol,
        "last_sector_regime_passed": preview_entry.last_sector_regime_passed,
        "setup_quality_score": setup_quality_score,
        "repeated_high_quality_signal": repeated_high_quality_signal,
    }


def update_candidate_score_journal(
    journal: CandidateScoreJournal,
    *,
    observations: Mapping[str, CandidateJournalObservation],
    as_of_date: date,
) -> CandidateScoreJournal:
    """Return a new journal with today's symbol observations merged in."""

    updated_entries: dict[str, CandidateJournalEntry] = {}

    for symbol, observation in observations.items():
        existing_entry = journal.entries.get(symbol)
        updated_entries[symbol] = _next_candidate_journal_entry(
            existing_entry,
            observation=observation,
            as_of_date=as_of_date,
        )

    for symbol, existing_entry in journal.entries.items():
        if symbol in updated_entries:
            continue
        if (as_of_date - existing_entry.last_seen_date).days > journal.stale_after_days:
            continue
        updated_entries[symbol] = existing_entry

    return CandidateScoreJournal(
        entries=dict(sorted(updated_entries.items())),
        updated_at=as_of_date,
        stale_after_days=journal.stale_after_days,
        schema_version=journal.schema_version,
    )


def _next_candidate_journal_entry(
    existing_entry: CandidateJournalEntry | None,
    *,
    observation: CandidateJournalObservation,
    as_of_date: date,
) -> CandidateJournalEntry:
    normalized_symbol = observation.symbol.strip().upper()
    if existing_entry is None:
        days_near_breakout = 1
        days_approved = 1 if observation.approved_today else 0
        last_approved_date = as_of_date if observation.approved_today else None
        best_rank = observation.best_rank_today
        last_rank = observation.best_rank_today
        peak_relative_volume = observation.relative_volume
        peak_breakout_strength = observation.breakout_strength
    else:
        is_new_day = as_of_date > existing_entry.last_seen_date
        days_near_breakout = (
            _next_days_near_breakout(existing_entry, as_of_date)
            if is_new_day
            else existing_entry.days_near_breakout
        )
        approved_already_counted_today = existing_entry.last_approved_date == as_of_date
        approved_today = observation.approved_today and not approved_already_counted_today
        days_approved = existing_entry.days_approved + (1 if approved_today else 0)
        last_approved_date = (
            as_of_date
            if approved_today
            else existing_entry.last_approved_date
        )
        last_rank = (
            observation.best_rank_today
            if observation.best_rank_today is not None
            else existing_entry.last_rank
        )
        best_rank = _best_rank(existing_entry.best_rank, observation.best_rank_today)
        peak_relative_volume = _peak_value(
            existing_entry.peak_relative_volume,
            observation.relative_volume,
        )
        peak_breakout_strength = _peak_value(
            existing_entry.peak_breakout_strength,
            observation.breakout_strength,
        )

    return CandidateJournalEntry(
        symbol=normalized_symbol,
        last_seen_date=max(
            as_of_date,
            existing_entry.last_seen_date if existing_entry is not None else as_of_date,
        ),
        days_near_breakout=days_near_breakout,
        days_approved=days_approved,
        last_rank=last_rank,
        best_rank=best_rank,
        peak_relative_volume=peak_relative_volume,
        peak_breakout_strength=peak_breakout_strength,
        last_sector_etf_symbol=(
            observation.sector_etf_symbol
            if observation.sector_etf_symbol is not None
            else (
                existing_entry.last_sector_etf_symbol
                if existing_entry is not None
                else None
            )
        ),
        last_sector_regime_passed=(
            observation.sector_regime_passed
            if observation.sector_regime_passed is not None
            else (
                existing_entry.last_sector_regime_passed
                if existing_entry is not None
                else None
            )
        ),
        last_approved_date=last_approved_date,
    )


def _candidate_setup_quality_score(entry: CandidateJournalEntry) -> float:
    persistence_component = min(entry.days_near_breakout, 6) * 0.12
    approval_component = min(entry.days_approved, 3) * 0.12
    relative_volume_component = (
        min(max((entry.peak_relative_volume or 0.0) - 1.0, 0.0), 2.5) * 0.08
    )
    breakout_component = min(max(entry.peak_breakout_strength or 0.0, 0.0), 0.08) * 3.0
    return round(
        persistence_component
        + approval_component
        + relative_volume_component
        + breakout_component,
        6,
    )


def _next_days_near_breakout(
    existing_entry: CandidateJournalEntry,
    as_of_date: date,
) -> int:
    gap_days = (as_of_date - existing_entry.last_seen_date).days
    if gap_days <= 0:
        return existing_entry.days_near_breakout
    if gap_days <= CONSECUTIVE_SETUP_GAP_RESET_DAYS:
        return existing_entry.days_near_breakout + 1
    return 1


def _peak_value(existing_value: float | None, new_value: float | None) -> float | None:
    if existing_value is None:
        return new_value
    if new_value is None:
        return existing_value
    return max(existing_value, new_value)


def _best_rank(existing_rank: int | None, new_rank: int | None) -> int | None:
    if existing_rank is None:
        return new_rank
    if new_rank is None:
        return existing_rank
    return min(existing_rank, new_rank)


def _mapping_date(data: Mapping[str, Any], key: str) -> date:
    try:
        return _coerce_date(data[key], key)
    except KeyError as exc:
        raise ValueError(f"Expected journal field '{key}'.") from exc


def _mapping_optional_date(data: Mapping[str, Any], key: str) -> date | None:
    if key not in data or data[key] is None:
        return None
    return _coerce_date(data[key], key)


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    return _coerce_date(value, "updated_at")


def _coerce_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{field_name}' to be a non-empty ISO date string.")
    return date.fromisoformat(value.strip())


def _mapping_positive_int(data: Mapping[str, Any], key: str) -> int:
    value = _coerce_positive_int(data.get(key), key)
    return value


def _mapping_non_negative_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Expected '{key}' to be a non-negative integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{key}' to be a non-negative integer.") from exc
    if resolved < 0:
        raise ValueError(f"Expected '{key}' to be a non-negative integer.")
    return resolved


def _mapping_optional_positive_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _coerce_positive_int(value, key)


def _coerce_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Expected '{field_name}' to be a positive integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{field_name}' to be a positive integer.") from exc
    if resolved <= 0:
        raise ValueError(f"Expected '{field_name}' to be a positive integer.")
    return resolved


def _mapping_optional_float(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected '{key}' to be a float when provided.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{key}' to be a float when provided.") from exc


def _mapping_optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{key}' to be a non-empty string when provided.")
    return value.strip().upper()


def _mapping_optional_bool(data: Mapping[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Expected '{key}' to be a boolean when provided.")
    return value
