"""Lightweight multi-day held-position trajectory journaling and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.logging_utils import get_logger
from bot.data.state_persistence import (
    StateArtifactPath,
    load_json_mapping_file,
    write_json_file,
)


LOGGER = get_logger(__name__)

POSITION_TRAJECTORY_JOURNAL_SCHEMA_VERSION = 1
DEFAULT_POSITION_PERSISTENT_WEAKNESS_DAY_THRESHOLD = 3
DEFAULT_POSITION_RECOVERY_MIN_WEAK_DAYS = 2
_POSITION_TRAJECTORY_JOURNAL_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "journals",
        "position_trajectory.json",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "position_trajectory.json"),
    ),
)


@dataclass(frozen=True)
class PositionTrajectoryObservation:
    """One compact end-of-day state snapshot for a held position."""

    symbol: str
    as_of_date: date
    average_entry_price: float
    current_stop: float | None
    latest_close: float
    unrealized_pl_pct: float
    above_entry: bool
    high_water_close: float | None = None
    high_water_close_date: date | None = None
    days_since_new_high: int | None = None
    stale_position: bool = False
    relative_strength_return_diff: float | None = None
    weak_relative_strength: bool = False
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.days_since_new_high is not None and self.days_since_new_high < 0:
            raise ValueError("days_since_new_high cannot be negative.")
        if self.suggested_action is not None and not self.suggested_action.strip():
            raise ValueError("suggested_action cannot be blank when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly observation payload."""

        return {
            "as_of_date": self.as_of_date.isoformat(),
            "average_entry_price": self.average_entry_price,
            "current_stop": self.current_stop,
            "latest_close": self.latest_close,
            "unrealized_pl_pct": self.unrealized_pl_pct,
            "above_entry": self.above_entry,
            "high_water_close": self.high_water_close,
            "high_water_close_date": (
                self.high_water_close_date.isoformat()
                if self.high_water_close_date is not None
                else None
            ),
            "days_since_new_high": self.days_since_new_high,
            "stale_position": self.stale_position,
            "relative_strength_return_diff": self.relative_strength_return_diff,
            "weak_relative_strength": self.weak_relative_strength,
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        data: Mapping[str, Any],
    ) -> "PositionTrajectoryObservation":
        """Build an observation from stored JSON."""

        return cls(
            symbol=symbol.strip().upper(),
            as_of_date=_mapping_date(data, "as_of_date"),
            average_entry_price=_mapping_float(data, "average_entry_price"),
            current_stop=_mapping_optional_float(data, "current_stop"),
            latest_close=_mapping_float(data, "latest_close"),
            unrealized_pl_pct=_mapping_float(data, "unrealized_pl_pct"),
            above_entry=_mapping_bool(data, "above_entry"),
            high_water_close=_mapping_optional_float(data, "high_water_close"),
            high_water_close_date=_mapping_optional_date(data, "high_water_close_date"),
            days_since_new_high=_mapping_optional_non_negative_int(
                data,
                "days_since_new_high",
            ),
            stale_position=_mapping_bool(data, "stale_position"),
            relative_strength_return_diff=_mapping_optional_float(
                data,
                "relative_strength_return_diff",
            ),
            weak_relative_strength=_mapping_bool(data, "weak_relative_strength"),
            suggested_action=_mapping_optional_text(data, "suggested_action"),
        )


@dataclass(frozen=True)
class PositionTrajectoryFeatures:
    """Small explainable multi-day trajectory features for held positions."""

    observation_count: int = 0
    days_in_position_state: int = 0
    consecutive_days_above_entry: int = 0
    consecutive_days_below_entry: int = 0
    consecutive_watch_closely_days: int = 0
    consecutive_hold_days: int = 0
    consecutive_stale_position_days: int = 0
    consecutive_weak_relative_strength_days: int = 0
    consecutive_weak_position_days: int = 0
    repeated_weak_position: bool = False
    persistent_underperformance: bool = False
    recovery_after_multi_day_weakness: bool = False


@dataclass(frozen=True)
class PositionTrajectoryJournal:
    """Persisted multi-day position state across review sessions."""

    entries: dict[str, tuple[PositionTrajectoryObservation, ...]]
    updated_at: str | None = None
    schema_version: int = POSITION_TRAJECTORY_JOURNAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly journal payload."""

        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "symbols": {
                symbol: {
                    "observations": [observation.to_dict() for observation in observations]
                }
                for symbol, observations in sorted(self.entries.items())
            },
        }


@dataclass(frozen=True)
class PositionTrajectoryJournalStore:
    """Project-scoped loader/saver for the position trajectory journal."""

    project_root: Path

    @property
    def path(self) -> Path:
        return default_position_trajectory_journal_path(self.project_root)

    def load(self) -> PositionTrajectoryJournal:
        return load_position_trajectory_journal(self.path)

    def save(self, journal: PositionTrajectoryJournal) -> Path:
        return write_position_trajectory_journal(journal, self.path)


def default_position_trajectory_journal_path(project_root: Path) -> Path:
    """Return the repo-relative journal path for daily held-position state."""

    return _POSITION_TRAJECTORY_JOURNAL_PATH.resolve(project_root)


def load_position_trajectory_journal(
    journal_path: Path,
) -> PositionTrajectoryJournal:
    """Load the journal, falling back to empty state on missing/corrupt files."""

    return load_json_mapping_file(
        journal_path,
        artifact_label="position trajectory journal",
        logger=LOGGER,
        empty_value=lambda: PositionTrajectoryJournal(entries={}),
        build=lambda payload, _resolved_path: _build_position_trajectory_journal(
            payload,
        ),
    )


def write_position_trajectory_journal(
    journal: PositionTrajectoryJournal,
    journal_path: Path,
) -> Path:
    """Persist the journal as JSON with an atomic replace in the target directory."""

    return write_json_file(journal_path, journal.to_dict())


def update_position_trajectory_journal(
    journal: PositionTrajectoryJournal,
    *,
    observations: Mapping[str, PositionTrajectoryObservation],
    active_symbols: Sequence[str] | None = None,
) -> PositionTrajectoryJournal:
    """Return a new journal with per-symbol daily observations merged idempotently."""

    updated_entries: dict[str, list[PositionTrajectoryObservation]] = {
        symbol: list(existing_observations)
        for symbol, existing_observations in journal.entries.items()
    }

    for symbol, observation in observations.items():
        normalized_symbol = symbol.strip().upper()
        if observation.symbol.strip().upper() != normalized_symbol:
            raise ValueError("Observation symbol must match the journal update key.")
        existing = [
            existing_observation
            for existing_observation in updated_entries.get(normalized_symbol, [])
            if existing_observation.as_of_date != observation.as_of_date
        ]
        existing.append(observation)
        existing.sort(key=lambda current: current.as_of_date)
        updated_entries[normalized_symbol] = existing

    if active_symbols is not None:
        normalized_active_symbols = {
            symbol.strip().upper()
            for symbol in active_symbols
            if isinstance(symbol, str) and symbol.strip()
        }
        updated_entries = {
            symbol: observations_for_symbol
            for symbol, observations_for_symbol in updated_entries.items()
            if symbol in normalized_active_symbols
        }

    return PositionTrajectoryJournal(
        entries={
            symbol: tuple(observations_for_symbol)
            for symbol, observations_for_symbol in sorted(updated_entries.items())
        },
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        schema_version=journal.schema_version,
    )


def _build_position_trajectory_journal(
    payload: Mapping[str, Any],
) -> PositionTrajectoryJournal:
    raw_symbols = payload.get("symbols", {})
    if not isinstance(raw_symbols, Mapping):
        raise ValueError("Journal payload 'symbols' must be a mapping.")

    entries: dict[str, tuple[PositionTrajectoryObservation, ...]] = {}
    for symbol, raw_entry in raw_symbols.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Journal symbol keys must be non-empty strings.")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Journal entry for {symbol!r} must be a mapping.")
        raw_observations = raw_entry.get("observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError(f"Journal observations for {symbol!r} must be a list.")
        normalized_symbol = symbol.strip().upper()
        observations = tuple(
            sorted(
                (
                    PositionTrajectoryObservation.from_mapping(
                        normalized_symbol,
                        raw_observation,
                    )
                    for raw_observation in raw_observations
                ),
                key=lambda observation: observation.as_of_date,
            )
        )
        entries[normalized_symbol] = observations

    return PositionTrajectoryJournal(
        entries=dict(sorted(entries.items())),
        updated_at=_mapping_optional_text(payload, "updated_at"),
        schema_version=_coerce_positive_int(
            payload.get(
                "schema_version",
                POSITION_TRAJECTORY_JOURNAL_SCHEMA_VERSION,
            ),
            "schema_version",
        ),
    )


def preview_position_trajectory_features(
    journal: PositionTrajectoryJournal,
    *,
    observation: PositionTrajectoryObservation,
    persistent_weakness_day_threshold: int = DEFAULT_POSITION_PERSISTENT_WEAKNESS_DAY_THRESHOLD,
    recovery_min_weak_days: int = DEFAULT_POSITION_RECOVERY_MIN_WEAK_DAYS,
) -> PositionTrajectoryFeatures:
    """Return features as if the latest unresolved daily observation were merged now.

    The current observation must be unresolved, meaning ``suggested_action`` is still
    ``None``. This preview path is used before the daily portfolio review assigns the
    final action for the current session, so repeated prior ``WATCH CLOSELY`` counts
    only come from already-finalized observations in the journal.
    """

    if observation.suggested_action is not None:
        raise ValueError(
            "preview_position_trajectory_features expects an unresolved observation with "
            "suggested_action=None."
        )

    preview_journal = update_position_trajectory_journal(
        journal,
        observations={observation.symbol: observation},
    )
    return build_position_trajectory_features(
        preview_journal.entries.get(observation.symbol, ()),
        persistent_weakness_day_threshold=persistent_weakness_day_threshold,
        recovery_min_weak_days=recovery_min_weak_days,
    )


def build_position_trajectory_features(
    observations: Sequence[PositionTrajectoryObservation],
    *,
    persistent_weakness_day_threshold: int = DEFAULT_POSITION_PERSISTENT_WEAKNESS_DAY_THRESHOLD,
    recovery_min_weak_days: int = DEFAULT_POSITION_RECOVERY_MIN_WEAK_DAYS,
) -> PositionTrajectoryFeatures:
    """Derive compact multi-day position trajectory features from stored observations.

    When called directly on finalized sequences, a resolved last observation is
    included in repeated ``WATCH CLOSELY`` counts. Only an unresolved last
    observation with ``suggested_action=None`` is excluded from that trailing count.
    """

    if persistent_weakness_day_threshold <= 0:
        raise ValueError("persistent_weakness_day_threshold must be greater than zero.")
    if recovery_min_weak_days <= 0:
        raise ValueError("recovery_min_weak_days must be greater than zero.")

    ordered = sorted(observations, key=lambda observation: observation.as_of_date)
    if not ordered:
        return PositionTrajectoryFeatures()

    weak_position_flags = [
        _observation_has_weakness(observation)
        for observation in ordered
    ]
    repeated_action_source = ordered[:-1] if ordered[-1].suggested_action is None else ordered
    consecutive_days_above_entry = _count_trailing(
        ordered,
        lambda observation: observation.above_entry,
    )
    consecutive_days_below_entry = _count_trailing(
        ordered,
        lambda observation: not observation.above_entry,
    )
    consecutive_stale_position_days = _count_trailing(
        ordered,
        lambda observation: observation.stale_position,
    )
    consecutive_weak_relative_strength_days = _count_trailing(
        ordered,
        lambda observation: observation.weak_relative_strength,
    )
    consecutive_weak_position_days = _count_trailing(
        weak_position_flags,
        lambda is_weak: is_weak,
    )
    trailing_weak_days_before_latest = 0
    if len(ordered) >= 2 and not weak_position_flags[-1]:
        trailing_weak_days_before_latest = _count_trailing(
            weak_position_flags[:-1],
            lambda is_weak: is_weak,
        )

    return PositionTrajectoryFeatures(
        observation_count=len(ordered),
        days_in_position_state=len(ordered),
        consecutive_days_above_entry=consecutive_days_above_entry,
        consecutive_days_below_entry=consecutive_days_below_entry,
        consecutive_watch_closely_days=_count_trailing(
            repeated_action_source,
            lambda observation: observation.suggested_action == "WATCH CLOSELY",
        ),
        consecutive_hold_days=_count_trailing(
            repeated_action_source,
            lambda observation: observation.suggested_action == "HOLD",
        ),
        consecutive_stale_position_days=consecutive_stale_position_days,
        consecutive_weak_relative_strength_days=consecutive_weak_relative_strength_days,
        consecutive_weak_position_days=consecutive_weak_position_days,
        repeated_weak_position=consecutive_weak_position_days >= 2,
        persistent_underperformance=(
            consecutive_days_below_entry >= persistent_weakness_day_threshold
            or consecutive_stale_position_days >= persistent_weakness_day_threshold
            or consecutive_weak_relative_strength_days >= persistent_weakness_day_threshold
            or consecutive_weak_position_days >= persistent_weakness_day_threshold
        ),
        recovery_after_multi_day_weakness=(
            trailing_weak_days_before_latest >= recovery_min_weak_days
        ),
    )


def _observation_has_weakness(observation: PositionTrajectoryObservation) -> bool:
    return bool(
        not observation.above_entry
        or observation.stale_position
        or observation.weak_relative_strength
    )


def _count_trailing(
    values: Sequence[Any],
    predicate,
) -> int:
    count = 0
    for value in reversed(values):
        if not predicate(value):
            break
        count += 1
    return count


def _mapping_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected journal field '{key}' to be a non-empty string.")
    return value.strip()


def _mapping_optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected journal field '{key}' to be a string when provided.")
    cleaned = value.strip()
    return cleaned or None


def _mapping_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected journal field '{key}' to be a boolean.")
    return value


def _mapping_float(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Expected journal field '{key}' to be a float, not a boolean.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected journal field '{key}' to be a float.") from exc


def _mapping_optional_float(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"Expected journal field '{key}' to be a float when provided, not a boolean."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected journal field '{key}' to be a float when provided.") from exc


def _mapping_date(data: Mapping[str, Any], key: str) -> date:
    raw_value = _mapping_text(data, key)
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(f"Expected journal field '{key}' to be a valid ISO date.") from exc


def _mapping_optional_date(data: Mapping[str, Any], key: str) -> date | None:
    raw_value = data.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Expected journal field '{key}' to be an ISO date string.")
    try:
        return date.fromisoformat(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"Expected journal field '{key}' to be a valid ISO date.") from exc


def _mapping_optional_non_negative_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"Expected journal field '{key}' to be an integer when provided, not a boolean."
        )
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected journal field '{key}' to be an integer when provided."
        ) from exc
    if coerced < 0:
        raise ValueError(f"Expected journal field '{key}' to be non-negative.")
    return coerced


def _coerce_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected '{field_name}' to be an integer, not a boolean.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{field_name}' to be an integer.") from exc
    if coerced <= 0:
        raise ValueError(f"Expected '{field_name}' to be greater than zero.")
    return coerced
