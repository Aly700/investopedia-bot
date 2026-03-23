"""Shared path and file helpers for persisted state artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class StateArtifactPath:
    """Preferred and legacy repo-relative paths for one persisted artifact."""

    preferred_relative_path: tuple[str, ...]
    legacy_relative_paths: tuple[tuple[str, ...], ...] = ()

    def preferred_path(self, project_root: Path) -> Path:
        return project_root.joinpath(*self.preferred_relative_path)

    def legacy_paths(self, project_root: Path) -> tuple[Path, ...]:
        return tuple(
            project_root.joinpath(*relative_path)
            for relative_path in self.legacy_relative_paths
        )

    def resolve(self, project_root: Path) -> Path:
        preferred_path = self.preferred_path(project_root)
        if preferred_path.exists():
            return preferred_path
        for legacy_path in self.legacy_paths(project_root):
            if legacy_path.exists():
                return legacy_path
        return preferred_path


def load_json_mapping_file(
    path: Path,
    *,
    artifact_label: str,
    logger: Logger,
    empty_value: Callable[[], T],
    build: Callable[[Mapping[str, Any], Path], T],
) -> T:
    """Load a JSON mapping, logging and returning ``empty_value`` on failure."""

    resolved_path = path.resolve()
    if not resolved_path.exists():
        return empty_value()

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{artifact_label} payload must be a mapping.")
        return build(payload, resolved_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Ignoring %s at %s because it could not be loaded: %s",
            artifact_label,
            resolved_path,
            exc,
        )
        return empty_value()


def write_json_file(
    path: Path,
    payload: Mapping[str, Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Write JSON atomically in the destination directory."""

    resolved_path = path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=indent, sort_keys=sort_keys),
        encoding="utf-8",
    )
    temp_path.replace(resolved_path)
    return resolved_path


def write_text_file(path: Path, content: str) -> Path:
    """Write text atomically in the destination directory."""

    resolved_path = path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(resolved_path)
    return resolved_path


def coerce_positive_int(value: Any, field_name: str) -> int:
    """Return a positive integer while rejecting booleans."""

    if isinstance(value, bool):
        raise ValueError(f"Expected '{field_name}' to be an integer, not a boolean.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected '{field_name}' to be an integer.") from exc
    if coerced <= 0:
        raise ValueError(f"Expected '{field_name}' to be greater than zero.")
    return coerced


def coerce_iso8601_datetime_text(value: Any, field_name: str) -> str:
    """Return a validated ISO 8601 datetime string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{field_name}' to be an ISO 8601 datetime string.")
    cleaned = value.strip()
    if "T" not in cleaned and " " not in cleaned:
        raise ValueError(f"Expected '{field_name}' to be an ISO 8601 datetime string.")
    normalized = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Expected '{field_name}' to be an ISO 8601 datetime string.") from exc
    return cleaned
