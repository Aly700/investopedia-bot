"""Lightweight per-session intraday state journaling and trajectory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bot.logging_utils import get_logger
from bot.data.state_persistence import (
    StateArtifactPath,
    coerce_iso8601_datetime_text,
    coerce_positive_int,
    load_json_mapping_file,
    write_json_file,
)


LOGGER = get_logger(__name__)

INTRADAY_STATE_JOURNAL_SCHEMA_VERSION = 1
DEFAULT_INTRADAY_GIVEBACK_WORSENING_THRESHOLD = 0.005
DEFAULT_INTRADAY_ALL_SESSION_MIN_OBSERVATIONS = 3
_INTRADAY_STATE_JOURNAL_PATH = StateArtifactPath(
    preferred_relative_path=(
        "data",
        "processed",
        "state",
        "journals",
        "intraday",
    ),
    legacy_relative_paths=(
        ("data", "processed", "state", "intraday_journal"),
    ),
)


@dataclass(frozen=True)
class IntradayStateObservation:
    """One compact per-poll intraday state snapshot for one held symbol."""

    symbol: str
    timestamp: str
    latest_close: float
    session_vwap: float | None = None
    close_vs_vwap_pct: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_high_giveback_pct: float | None = None
    intraday_return_vs_open: float | None = None
    intraday_return_vs_benchmark: float | None = None
    weak_intraday_relative_strength: bool = False
    failed_intraday_strength: bool = False
    intraday_momentum_fade: bool = False
    stacked_intraday_weakness: bool = False
    stop_breached_intraday: bool = False
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        object.__setattr__(
            self,
            "timestamp",
            coerce_iso8601_datetime_text(self.timestamp, "timestamp"),
        )
        if self.suggested_action is not None and not self.suggested_action.strip():
            raise ValueError("suggested_action cannot be blank when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly observation payload."""

        return {
            "timestamp": self.timestamp,
            "latest_close": self.latest_close,
            "session_vwap": self.session_vwap,
            "close_vs_vwap_pct": self.close_vs_vwap_pct,
            "session_high": self.session_high,
            "session_low": self.session_low,
            "session_high_giveback_pct": self.session_high_giveback_pct,
            "intraday_return_vs_open": self.intraday_return_vs_open,
            "intraday_return_vs_benchmark": self.intraday_return_vs_benchmark,
            "weak_intraday_relative_strength": self.weak_intraday_relative_strength,
            "failed_intraday_strength": self.failed_intraday_strength,
            "intraday_momentum_fade": self.intraday_momentum_fade,
            "stacked_intraday_weakness": self.stacked_intraday_weakness,
            "stop_breached_intraday": self.stop_breached_intraday,
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        data: Mapping[str, Any],
    ) -> "IntradayStateObservation":
        """Build an observation from stored JSON."""

        return cls(
            symbol=symbol.strip().upper(),
            timestamp=_mapping_text(data, "timestamp"),
            latest_close=_mapping_float(data, "latest_close"),
            session_vwap=_mapping_optional_float(data, "session_vwap"),
            close_vs_vwap_pct=_mapping_optional_float(data, "close_vs_vwap_pct"),
            session_high=_mapping_optional_float(data, "session_high"),
            session_low=_mapping_optional_float(data, "session_low"),
            session_high_giveback_pct=_mapping_optional_float(
                data,
                "session_high_giveback_pct",
            ),
            intraday_return_vs_open=_mapping_optional_float(
                data,
                "intraday_return_vs_open",
            ),
            intraday_return_vs_benchmark=_mapping_optional_float(
                data,
                "intraday_return_vs_benchmark",
            ),
            weak_intraday_relative_strength=_mapping_bool(
                data,
                "weak_intraday_relative_strength",
            ),
            failed_intraday_strength=_mapping_bool(data, "failed_intraday_strength"),
            intraday_momentum_fade=_mapping_bool(data, "intraday_momentum_fade"),
            stacked_intraday_weakness=_mapping_bool(data, "stacked_intraday_weakness"),
            stop_breached_intraday=_mapping_bool(data, "stop_breached_intraday"),
            suggested_action=_mapping_optional_text(data, "suggested_action"),
        )


@dataclass(frozen=True)
class IntradayTrajectoryFeatures:
    """Small explainable trajectory features derived from prior session observations."""

    observation_count: int = 0
    consecutive_polls_below_vwap: int = 0
    consecutive_polls_weak_relative_strength: int = 0
    intraday_pressure_persistence_count: int = 0
    max_session_high_giveback_seen: float | None = None
    worsening_session_high_giveback: bool = False
    giveback_worsening_polls: int = 0
    giveback_worsening_from_pct: float | None = None
    repeated_watch_closely_count: int = 0
    weakening_all_session: bool = False
    recovered_after_weakness: bool = False


@dataclass(frozen=True)
class IntradayStateJournal:
    """Persisted intraday state for one trading session."""

    session_date: date
    entries: dict[str, tuple[IntradayStateObservation, ...]]
    updated_at: str | None = None
    schema_version: int = INTRADAY_STATE_JOURNAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly journal payload."""

        return {
            "schema_version": self.schema_version,
            "session_date": self.session_date.isoformat(),
            "updated_at": self.updated_at,
            "symbols": {
                symbol: {
                    "observations": [observation.to_dict() for observation in observations]
                }
                for symbol, observations in sorted(self.entries.items())
            },
        }


@dataclass(frozen=True)
class IntradayStateJournalStore:
    """Project-scoped loader/saver for one intraday session journal."""

    project_root: Path
    session_date: date

    @property
    def path(self) -> Path:
        return default_intraday_state_journal_path(self.project_root, self.session_date)

    def load(self) -> IntradayStateJournal:
        return load_intraday_state_journal(self.path, session_date=self.session_date)

    def save(self, journal: IntradayStateJournal) -> Path:
        return write_intraday_state_journal(journal, self.path)


def default_intraday_state_journal_path(project_root: Path, session_date: date) -> Path:
    """Return the repo-relative per-session intraday journal path."""

    filename = f"{session_date.isoformat()}.json"
    preferred_path = _INTRADAY_STATE_JOURNAL_PATH.preferred_path(project_root) / filename
    if preferred_path.exists():
        return preferred_path
    for legacy_dir in _INTRADAY_STATE_JOURNAL_PATH.legacy_paths(project_root):
        legacy_path = legacy_dir / filename
        if legacy_path.exists():
            return legacy_path
    return preferred_path


def load_intraday_state_journal(
    journal_path: Path,
    *,
    session_date: date,
) -> IntradayStateJournal:
    """Load the intraday journal, falling back to empty state when unavailable."""

    return load_json_mapping_file(
        journal_path,
        artifact_label="intraday state journal",
        logger=LOGGER,
        empty_value=lambda: IntradayStateJournal(session_date=session_date, entries={}),
        build=lambda payload, _resolved_path: _build_intraday_state_journal(
            payload,
            session_date=session_date,
        ),
    )


def write_intraday_state_journal(
    journal: IntradayStateJournal,
    journal_path: Path,
) -> Path:
    """Persist the journal as JSON with an atomic replace in the target directory."""

    return write_json_file(journal_path, journal.to_dict())


def update_intraday_state_journal(
    journal: IntradayStateJournal,
    *,
    observations: Mapping[str, IntradayStateObservation],
) -> IntradayStateJournal:
    """Return a new journal with per-symbol observations merged idempotently."""

    updated_entries: dict[str, list[IntradayStateObservation]] = {
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
            if existing_observation.timestamp != observation.timestamp
        ]
        existing.append(observation)
        existing.sort(key=lambda current: current.timestamp)
        updated_entries[normalized_symbol] = existing

    return IntradayStateJournal(
        session_date=journal.session_date,
        entries={
            symbol: tuple(observations_for_symbol)
            for symbol, observations_for_symbol in sorted(updated_entries.items())
        },
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        schema_version=journal.schema_version,
    )


def _build_intraday_state_journal(
    payload: Mapping[str, Any],
    *,
    session_date: date,
) -> IntradayStateJournal:
    raw_symbols = payload.get("symbols", {})
    if not isinstance(raw_symbols, Mapping):
        raise ValueError("Journal payload 'symbols' must be a mapping.")

    stored_session_date = _mapping_date(payload, "session_date")
    if stored_session_date != session_date:
        raise ValueError(
            f"Journal session_date {stored_session_date.isoformat()} does not match "
            f"requested session_date {session_date.isoformat()}."
        )

    entries: dict[str, tuple[IntradayStateObservation, ...]] = {}
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
                    IntradayStateObservation.from_mapping(normalized_symbol, raw_observation)
                    for raw_observation in raw_observations
                ),
                key=lambda observation: observation.timestamp,
            )
        )
        entries[normalized_symbol] = observations

    return IntradayStateJournal(
        session_date=session_date,
        entries=dict(sorted(entries.items())),
        updated_at=_mapping_optional_text(payload, "updated_at"),
        schema_version=_coerce_positive_int(
            payload.get("schema_version", INTRADAY_STATE_JOURNAL_SCHEMA_VERSION),
            "schema_version",
        ),
    )


def preview_intraday_trajectory_features(
    journal: IntradayStateJournal,
    *,
    observation: IntradayStateObservation,
    giveback_worsening_threshold: float = DEFAULT_INTRADAY_GIVEBACK_WORSENING_THRESHOLD,
    all_session_min_observations: int = DEFAULT_INTRADAY_ALL_SESSION_MIN_OBSERVATIONS,
) -> IntradayTrajectoryFeatures:
    """Return trajectory features as if the latest unresolved observation were merged now.

    The current observation must be unresolved, meaning ``suggested_action`` is still
    ``None``. This preview path is used before the intraday review assigns the final
    action for the current poll, so repeated prior ``WATCH CLOSELY`` counts only come
    from already-finalized observations in the journal.
    """

    if observation.suggested_action is not None:
        raise ValueError(
            "preview_intraday_trajectory_features expects an unresolved observation with "
            "suggested_action=None."
        )

    preview_journal = update_intraday_state_journal(
        journal,
        observations={observation.symbol: observation},
    )
    return build_intraday_trajectory_features(
        preview_journal.entries.get(observation.symbol, ()),
        giveback_worsening_threshold=giveback_worsening_threshold,
        all_session_min_observations=all_session_min_observations,
    )


def build_intraday_trajectory_features(
    observations: Sequence[IntradayStateObservation],
    *,
    giveback_worsening_threshold: float = DEFAULT_INTRADAY_GIVEBACK_WORSENING_THRESHOLD,
    all_session_min_observations: int = DEFAULT_INTRADAY_ALL_SESSION_MIN_OBSERVATIONS,
) -> IntradayTrajectoryFeatures:
    """Derive a small set of explainable trajectory features from session observations.

    When called directly on finalized sequences, a resolved last observation is included
    in repeated ``WATCH CLOSELY`` counts. Only an unresolved last observation with
    ``suggested_action=None`` is excluded from that trailing watch count.
    """

    if giveback_worsening_threshold < 0:
        raise ValueError("giveback_worsening_threshold must be non-negative.")
    if all_session_min_observations <= 0:
        raise ValueError("all_session_min_observations must be greater than zero.")

    ordered = sorted(observations, key=lambda observation: observation.timestamp)
    if not ordered:
        return IntradayTrajectoryFeatures()

    pressure_flags = [_observation_has_pressure(observation) for observation in ordered]
    max_session_high_giveback_seen = _max_optional(
        observation.session_high_giveback_pct for observation in ordered
    )
    giveback_worsening_polls, giveback_worsening_from_pct = _giveback_worsening_streak(
        ordered,
        giveback_worsening_threshold=giveback_worsening_threshold,
    )
    repeated_watch_source = ordered[:-1] if ordered[-1].suggested_action is None else ordered

    return IntradayTrajectoryFeatures(
        observation_count=len(ordered),
        consecutive_polls_below_vwap=_count_trailing(
            ordered,
            lambda observation: (
                observation.close_vs_vwap_pct is not None
                and observation.close_vs_vwap_pct < 0
            ),
        ),
        consecutive_polls_weak_relative_strength=_count_trailing(
            ordered,
            lambda observation: observation.weak_intraday_relative_strength,
        ),
        intraday_pressure_persistence_count=_count_trailing(
            pressure_flags,
            lambda is_active: is_active,
        ),
        max_session_high_giveback_seen=max_session_high_giveback_seen,
        worsening_session_high_giveback=giveback_worsening_polls >= 2,
        giveback_worsening_polls=giveback_worsening_polls,
        giveback_worsening_from_pct=giveback_worsening_from_pct,
        repeated_watch_closely_count=_count_trailing(
            repeated_watch_source,
            lambda observation: observation.suggested_action == "WATCH CLOSELY",
        ),
        weakening_all_session=(
            len(ordered) >= all_session_min_observations and all(pressure_flags)
        ),
        recovered_after_weakness=(
            len(ordered) >= 2
            and not pressure_flags[-1]
            and any(pressure_flags[:-1])
        ),
    )


def _observation_has_pressure(observation: IntradayStateObservation) -> bool:
    return bool(
        observation.stop_breached_intraday
        or observation.failed_intraday_strength
        or observation.intraday_momentum_fade
        or observation.weak_intraday_relative_strength
        or observation.stacked_intraday_weakness
        or (
            observation.close_vs_vwap_pct is not None
            and observation.close_vs_vwap_pct < 0
        )
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


def _giveback_worsening_streak(
    observations: Sequence[IntradayStateObservation],
    *,
    giveback_worsening_threshold: float,
) -> tuple[int, float | None]:
    if not observations:
        return 0, None
    latest_giveback = observations[-1].session_high_giveback_pct
    if latest_giveback is None:
        return 0, None

    streak = 1
    baseline = latest_giveback
    current = latest_giveback
    for observation in reversed(observations[:-1]):
        previous = observation.session_high_giveback_pct
        if previous is None or (current - previous) < giveback_worsening_threshold:
            break
        streak += 1
        baseline = previous
        current = previous
    if streak < 2:
        return streak, None
    return streak, baseline


def _max_optional(values: Iterable[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


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
    raw_value = data.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Expected journal field '{key}' to be an ISO date string.")
    try:
        return date.fromisoformat(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"Expected journal field '{key}' to be a valid ISO date.") from exc


def _coerce_positive_int(value: Any, field_name: str) -> int:
    return coerce_positive_int(value, field_name)
