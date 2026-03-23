"""Portfolio-level rules and signal-to-risk assessment helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import date
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.config import RiskConfig, RulesConfig
from bot.features import (
    CandidateFeatures,
    MarketContext,
    PositionFeatures,
    build_candidate_features,
    build_intraday_position_features,
    build_position_features,
)
from bot.risk.position_sizing import PositionSizingResult, size_position
from bot.strategy.signal_models import StrategySignal


class PortfolioInputError(ValueError):
    """Raised when a portfolio input file is missing or malformed."""


PORTFOLIO_SNAPSHOT_COLUMNS = (
    "symbol",
    "quantity",
    "average_entry_price",
    "current_stop",
    "preset_name",
    "source",
    "metadata_json",
)
PORTFOLIO_REVIEW_ACTIONS = (
    "HOLD",
    "WATCH CLOSELY",
    "RAISE STOP",
    "EXIT CANDIDATE",
)
DEFAULT_POSITION_PERSISTENT_WEAKNESS_DAY_THRESHOLD = 3
DEFAULT_POSITION_REPEATED_WATCH_EXIT_THRESHOLD = 2


@dataclass(frozen=True)
class ExistingPosition:
    """Minimal existing-position snapshot used by portfolio rules."""

    symbol: str
    shares: int
    average_entry_price: float
    current_stop: float | None = None
    preset_name: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty.")
        if self.shares <= 0:
            raise ValueError("shares must be greater than zero.")
        if not isfinite(self.average_entry_price) or self.average_entry_price <= 0:
            raise ValueError("average_entry_price must be a positive finite number.")
        if self.current_stop is not None and (
            not isfinite(self.current_stop) or self.current_stop <= 0
        ):
            raise ValueError("current_stop must be a positive finite number when provided.")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        row_number: int | None = None,
    ) -> "ExistingPosition":
        """Create an existing-position snapshot from CSV/JSON input."""

        row_prefix = f"Row {row_number}: " if row_number is not None else ""
        symbol = _required_text(data, "symbol", row_prefix=row_prefix).upper()
        raw_quantity = data.get("quantity", data.get("shares"))
        shares = _positive_int(raw_quantity, field_name="quantity", row_prefix=row_prefix)
        average_entry_price = _positive_float(
            data.get("average_entry_price"),
            field_name="average_entry_price",
            row_prefix=row_prefix,
        )
        current_stop = _optional_positive_float(data.get("current_stop"), field_name="current_stop")
        if current_stop is None:
            current_stop = _optional_positive_float(data.get("stop_price"), field_name="stop_price")

        preset_name = _optional_text(data.get("preset_name"))
        source = _optional_text(data.get("source"))
        metadata = _extract_position_metadata(data)
        return cls(
            symbol=symbol,
            shares=shares,
            average_entry_price=average_entry_price,
            current_stop=current_stop,
            preset_name=preset_name,
            source=source,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the current holding."""

        return {
            "symbol": self.symbol,
            "quantity": self.shares,
            "average_entry_price": self.average_entry_price,
            "current_stop": self.current_stop,
            "preset_name": self.preset_name,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PortfolioConstraints:
    """Reusable account-level risk and portfolio constraints."""

    max_concurrent_positions: int
    max_position_pct_equity: float | None
    no_averaging_down: bool
    drawdown_risk_reduction_threshold: float | None = None
    drawdown_risk_reduction_factor: float = 0.5

    @classmethod
    def from_configs(
        cls,
        risk_config: RiskConfig,
        rules_config: RulesConfig,
        *,
        drawdown_risk_reduction_factor: float = 0.5,
    ) -> "PortfolioConstraints":
        """Build portfolio constraints from the shared repo config models."""

        return cls(
            max_concurrent_positions=rules_config.max_positions,
            max_position_pct_equity=rules_config.max_position_pct_equity,
            no_averaging_down=rules_config.no_averaging_down,
            drawdown_risk_reduction_threshold=risk_config.drawdown_risk_reduction_threshold,
            drawdown_risk_reduction_factor=drawdown_risk_reduction_factor,
        )

    def __post_init__(self) -> None:
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be greater than zero.")
        if self.max_position_pct_equity is not None and self.max_position_pct_equity <= 0:
            raise ValueError("max_position_pct_equity must be greater than zero when provided.")
        if self.drawdown_risk_reduction_factor <= 0 or self.drawdown_risk_reduction_factor > 1:
            raise ValueError("drawdown_risk_reduction_factor must be in the interval (0, 1].")


@dataclass(frozen=True)
class PortfolioRuleResult:
    """Outcome of deterministic portfolio rule checks."""

    approved: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskAssessedCandidate:
    """A signal combined with sizing and portfolio rule outcomes."""

    signal: StrategySignal
    entry_price: float
    stop_price: float
    adjusted_risk_per_trade: float
    sizing: PositionSizingResult
    approved: bool
    rejection_reasons: tuple[str, ...] = ()
    existing_position: ExistingPosition | None = None
    features: CandidateFeatures | None = None


@dataclass(frozen=True)
class PortfolioReviewDecision:
    """Deterministic position-management suggestion for one open holding."""

    latest_close: float
    unrealized_pl_pct: float
    distance_to_stop_pct: float | None
    regime_passed: bool | None
    above_entry: bool
    suggested_action: str
    suggested_stop: float | None = None
    trailing_stop_candidate: float | None = None
    rationale: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.suggested_action not in PORTFOLIO_REVIEW_ACTIONS:
            raise ValueError(
                f"suggested_action must be one of {PORTFOLIO_REVIEW_ACTIONS}, "
                f"got '{self.suggested_action}'."
            )


def load_existing_positions(portfolio_path: Path) -> list[ExistingPosition]:
    """Load current holdings from a CSV or JSON portfolio snapshot."""

    resolved_path = portfolio_path.resolve()
    if not resolved_path.exists():
        raise PortfolioInputError(f"Portfolio file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        positions = _load_existing_positions_from_csv(resolved_path)
    elif suffix == ".json":
        positions = _load_existing_positions_from_json(resolved_path)
    else:
        raise PortfolioInputError("portfolio file must use a .csv or .json extension.")

    _validate_unique_position_symbols(positions, source=resolved_path)
    return positions


def initialize_portfolio_snapshot(
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Create an empty portfolio snapshot in CSV or JSON format."""

    return write_existing_positions_snapshot(
        (),
        output_path,
        output_format=output_format,
    )


def write_existing_positions_snapshot(
    positions: Sequence[ExistingPosition],
    output_path: Path,
    *,
    output_format: str | None = None,
) -> Path:
    """Write a portfolio snapshot in the same format consumed by ``--portfolio-file``."""

    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_snapshot_format(resolved_output_path, output_format)

    ordered_positions = sorted(positions, key=lambda position: position.symbol)
    _validate_unique_position_symbols(ordered_positions, source=resolved_output_path)

    if resolved_format == "json":
        resolved_output_path.write_text(
            json.dumps(
                {"positions": [position.to_dict() for position in ordered_positions]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return resolved_output_path

    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PORTFOLIO_SNAPSHOT_COLUMNS)
        writer.writeheader()
        for position in ordered_positions:
            writer.writerow(_position_record(position))
    return resolved_output_path


def upsert_existing_position_snapshot(
    output_path: Path,
    position: ExistingPosition,
    *,
    output_format: str | None = None,
) -> Path:
    """Append a new holding or update an existing symbol in a portfolio snapshot."""

    resolved_output_path = output_path.resolve()
    if resolved_output_path.exists():
        positions = load_existing_positions(resolved_output_path)
    else:
        positions = []

    updated_positions: list[ExistingPosition] = []
    replaced = False
    for existing_position in positions:
        if existing_position.symbol == position.symbol:
            updated_positions.append(position)
            replaced = True
        else:
            updated_positions.append(existing_position)
    if not replaced:
        updated_positions.append(position)

    return write_existing_positions_snapshot(
        updated_positions,
        resolved_output_path,
        output_format=output_format,
    )


def update_existing_position_stop_snapshot(
    output_path: Path,
    symbol: str,
    current_stop: float,
    *,
    output_format: str | None = None,
) -> Path:
    """Update ``current_stop`` for an existing symbol in a portfolio snapshot."""

    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise PortfolioInputError("symbol cannot be empty.")

    positions = load_existing_positions(output_path.resolve())
    updated_positions: list[ExistingPosition] = []
    matched = False
    for existing_position in positions:
        if existing_position.symbol == normalized_symbol:
            updated_positions.append(replace(existing_position, current_stop=current_stop))
            matched = True
        else:
            updated_positions.append(existing_position)

    if not matched:
        raise PortfolioInputError(
            f"Symbol '{normalized_symbol}' does not exist in portfolio snapshot."
        )

    return write_existing_positions_snapshot(
        updated_positions,
        output_path.resolve(),
        output_format=output_format,
    )


def remove_existing_position_snapshot(
    output_path: Path,
    symbol: str,
    *,
    output_format: str | None = None,
) -> Path:
    """Remove an existing symbol from a portfolio snapshot."""

    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise PortfolioInputError("symbol cannot be empty.")

    positions = load_existing_positions(output_path.resolve())
    remaining_positions = [
        existing_position
        for existing_position in positions
        if existing_position.symbol != normalized_symbol
    ]
    if len(remaining_positions) == len(positions):
        raise PortfolioInputError(
            f"Symbol '{normalized_symbol}' does not exist in portfolio snapshot."
        )

    return write_existing_positions_snapshot(
        remaining_positions,
        output_path.resolve(),
        output_format=output_format,
    )


def suggest_long_stop_update(
    *,
    current_stop: float | None,
    trailing_stop_candidate: float | None,
) -> float | None:
    """Return a higher long stop when the trailing candidate improves it."""

    if trailing_stop_candidate is None:
        return None
    _validate_finite_positive(trailing_stop_candidate, name="trailing_stop_candidate")
    if current_stop is None:
        return trailing_stop_candidate
    _validate_finite_positive(current_stop, name="current_stop")
    if trailing_stop_candidate > current_stop:
        return trailing_stop_candidate
    return None


def review_existing_long_position(
    position: ExistingPosition,
    *,
    position_features: PositionFeatures | None = None,
    latest_close: float,
    regime_passed: bool | None,
    trailing_stop_candidate: float | None = None,
    watch_distance_pct: float = 0.03,
    high_water_close: float | None = None,
    profit_giveback_threshold: float = 0.10,
    profit_giveback_min_unrealized_pct: float = 0.10,
    breakout_failure_reference: float | None = None,
    days_since_new_high: int | None = None,
    stale_high_watch_days: int = 15,
    relative_strength_return_diff: float | None = None,
    relative_strength_window: int | None = None,
    relative_strength_watch_threshold: float = -0.05,
    position_persistent_weakness_day_threshold: int = (
        DEFAULT_POSITION_PERSISTENT_WEAKNESS_DAY_THRESHOLD
    ),
    position_repeated_watch_exit_threshold: int = (
        DEFAULT_POSITION_REPEATED_WATCH_EXIT_THRESHOLD
    ),
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    earnings_watch_days: int = 7,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
    sector_relative_strength_watch_threshold: float = -0.05,
) -> PortfolioReviewDecision:
    """Review one long position and suggest a simple management action."""

    _validate_finite_positive(latest_close, name="latest_close")
    _validate_finite_non_negative(watch_distance_pct, name="watch_distance_pct")
    if profit_giveback_threshold <= 0 or profit_giveback_threshold >= 1:
        raise ValueError("profit_giveback_threshold must be between zero and one.")
    if profit_giveback_min_unrealized_pct <= 0:
        raise ValueError("profit_giveback_min_unrealized_pct must be greater than zero.")
    if stale_high_watch_days <= 0:
        raise ValueError("stale_high_watch_days must be greater than zero.")
    if position_persistent_weakness_day_threshold <= 0:
        raise ValueError(
            "position_persistent_weakness_day_threshold must be greater than zero."
        )
    if position_repeated_watch_exit_threshold <= 0:
        raise ValueError(
            "position_repeated_watch_exit_threshold must be greater than zero."
        )
    if days_since_new_high is not None and days_since_new_high < 0:
        raise ValueError("days_since_new_high cannot be negative.")
    if relative_strength_window is not None and relative_strength_window <= 0:
        raise ValueError("relative_strength_window must be greater than zero when provided.")
    if earnings_days_away is not None and earnings_days_away < 0:
        raise ValueError("earnings_days_away cannot be negative.")
    if earnings_watch_days <= 0:
        raise ValueError("earnings_watch_days must be greater than zero.")
    if sector_relative_strength_window is not None and sector_relative_strength_window <= 0:
        raise ValueError(
            "sector_relative_strength_window must be greater than zero when provided."
        )
    if sector_relative_strength_watch_threshold > 0:
        raise ValueError(
            "sector_relative_strength_watch_threshold must be less than or equal to zero."
        )
    if high_water_close is not None:
        _validate_finite_positive(high_water_close, name="high_water_close")

    features = position_features or build_position_features(
        symbol=position.symbol,
        as_of_date=date.today(),
        average_entry_price=position.average_entry_price,
        latest_close=latest_close,
        current_stop=position.current_stop,
        regime_passed=regime_passed,
        trailing_stop_candidate=trailing_stop_candidate,
        high_water_close=high_water_close,
        breakout_failure_reference=breakout_failure_reference,
        days_since_new_high=days_since_new_high,
        relative_strength_return_diff=relative_strength_return_diff,
        relative_strength_window=relative_strength_window,
        earnings_date=earnings_date,
        earnings_days_away=earnings_days_away,
        sector_name=sector_name,
        industry_name=industry_name,
        sector_etf_symbol=sector_etf_symbol,
        sector_regime_passed=sector_regime_passed,
        relative_strength_vs_sector=relative_strength_vs_sector,
        sector_relative_strength_window=sector_relative_strength_window,
        earnings_watch_days=earnings_watch_days,
        sector_relative_strength_watch_threshold=sector_relative_strength_watch_threshold,
        stale_high_watch_days=stale_high_watch_days,
    )
    return _review_existing_long_position_from_features(
        position,
        features=features,
        watch_distance_pct=watch_distance_pct,
        profit_giveback_threshold=profit_giveback_threshold,
        profit_giveback_min_unrealized_pct=profit_giveback_min_unrealized_pct,
        stale_high_watch_days=stale_high_watch_days,
        relative_strength_watch_threshold=relative_strength_watch_threshold,
        position_persistent_weakness_day_threshold=(
            position_persistent_weakness_day_threshold
        ),
        position_repeated_watch_exit_threshold=position_repeated_watch_exit_threshold,
        earnings_watch_days=earnings_watch_days,
        sector_relative_strength_watch_threshold=sector_relative_strength_watch_threshold,
    )


def _review_existing_long_position_from_features(
    position: ExistingPosition,
    *,
    features: PositionFeatures,
    watch_distance_pct: float,
    profit_giveback_threshold: float,
    profit_giveback_min_unrealized_pct: float,
    stale_high_watch_days: int,
    relative_strength_watch_threshold: float,
    position_persistent_weakness_day_threshold: int,
    position_repeated_watch_exit_threshold: int,
    earnings_watch_days: int,
    sector_relative_strength_watch_threshold: float,
) -> PortfolioReviewDecision:
    weak_relative_strength = (
        features.relative_strength_return_diff is not None
        and features.relative_strength_return_diff <= relative_strength_watch_threshold
    )
    weak_sector_regime = features.sector_regime_passed is False
    persistent_below_entry = (
        features.consecutive_days_below_entry >= position_persistent_weakness_day_threshold
    )
    persistent_stale_position = (
        features.consecutive_stale_position_days >= position_persistent_weakness_day_threshold
    )
    persistent_relative_weakness = (
        features.consecutive_weak_relative_strength_days
        >= position_persistent_weakness_day_threshold
    )
    repeated_watch_pressure = (
        features.consecutive_watch_closely_days >= position_repeated_watch_exit_threshold
    )

    suggested_stop = suggest_long_stop_update(
        current_stop=position.current_stop,
        trailing_stop_candidate=features.trailing_stop_candidate,
    )
    if suggested_stop is not None and suggested_stop >= features.latest_close:
        suggested_stop = None

    rationale: list[str] = []
    metadata = {
        "high_water_close": features.high_water_close,
        "giveback_pct": features.giveback_pct,
        "profit_giveback_threshold": profit_giveback_threshold,
        "profit_giveback_min_unrealized_pct": profit_giveback_min_unrealized_pct,
        "breakout_failure_reference": features.breakout_failure_reference,
        "failed_breakout_detected": features.failed_breakout,
        "days_since_new_high": features.days_since_new_high,
        "stale_high_watch_days": stale_high_watch_days,
        "stale_position": features.stale_position,
        "relative_strength_return_diff": features.relative_strength_return_diff,
        "relative_strength_window": features.relative_strength_window,
        "relative_strength_watch_threshold": relative_strength_watch_threshold,
        "weak_relative_strength": weak_relative_strength,
        "days_in_position_state": features.days_in_position_state,
        "consecutive_days_above_entry": features.consecutive_days_above_entry,
        "consecutive_days_below_entry": features.consecutive_days_below_entry,
        "consecutive_watch_closely_days": features.consecutive_watch_closely_days,
        "consecutive_hold_days": features.consecutive_hold_days,
        "consecutive_stale_position_days": features.consecutive_stale_position_days,
        "consecutive_weak_relative_strength_days": (
            features.consecutive_weak_relative_strength_days
        ),
        "consecutive_weak_position_days": features.consecutive_weak_position_days,
        "repeated_weak_position": features.repeated_weak_position,
        "persistent_underperformance": features.persistent_underperformance,
        "recovery_after_multi_day_weakness": features.recovery_after_multi_day_weakness,
        "position_persistent_weakness_day_threshold": (
            position_persistent_weakness_day_threshold
        ),
        "position_repeated_watch_exit_threshold": (
            position_repeated_watch_exit_threshold
        ),
        "persistent_below_entry": persistent_below_entry,
        "persistent_stale_position": persistent_stale_position,
        "persistent_relative_weakness": persistent_relative_weakness,
        "repeated_watch_pressure": repeated_watch_pressure,
        "earnings_date": (
            features.earnings_date.isoformat() if features.earnings_date is not None else None
        ),
        "earnings_days_away": features.earnings_days_away,
        "earnings_watch_days": earnings_watch_days,
        "is_earnings_risk": features.is_earnings_risk,
        "sector_name": features.sector_name,
        "industry_name": features.industry_name,
        "sector_etf_symbol": features.sector_etf_symbol,
        "sector_regime_passed": features.sector_regime_passed,
        "relative_strength_vs_sector": features.relative_strength_vs_sector,
        "sector_relative_strength_window": features.sector_relative_strength_window,
        "sector_relative_strength_watch_threshold": sector_relative_strength_watch_threshold,
        "weak_sector_regime": weak_sector_regime,
        "lagging_sector": features.lagging_sector,
    }

    if position.current_stop is not None and features.latest_close <= position.current_stop:
        rationale.append("Latest close is at or below the current stop.")
        action = "EXIT CANDIDATE"
    elif features.regime_passed is False and not features.above_entry:
        rationale.append("Benchmark regime is weak and the position is below the average entry price.")
        action = "EXIT CANDIDATE"
    elif (
        features.giveback_pct is not None
        and features.unrealized_pl_pct > profit_giveback_min_unrealized_pct
        and features.giveback_pct > profit_giveback_threshold
        and features.high_water_close is not None
    ):
        rationale.append(
            "Profitable position has given back "
            f"{features.giveback_pct:.1%} from the high-water close of {features.high_water_close:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif features.failed_breakout and features.breakout_failure_reference is not None:
        rationale.append(
            "Latest close has fallen back below the breakout reference of "
            f"{features.breakout_failure_reference:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif repeated_watch_pressure and features.persistent_underperformance:
        rationale.append(
            "Position has already been WATCH CLOSELY for "
            f"{features.consecutive_watch_closely_days} consecutive review sessions."
        )
        _append_position_trajectory_rationale(
            rationale,
            features=features,
            persistent_weakness_day_threshold=position_persistent_weakness_day_threshold,
        )
        rationale.append("Multi-day weakness increases exit pressure.")
        action = "EXIT CANDIDATE"
    elif suggested_stop is not None:
        rationale.append("Trailing-stop logic supports a higher stop without lowering risk control.")
        action = "RAISE STOP"
    elif (
        (features.distance_to_stop_pct is not None and features.distance_to_stop_pct <= watch_distance_pct)
        or features.regime_passed is False
        or not features.above_entry
        or features.stale_position
        or weak_relative_strength
        or persistent_below_entry
        or persistent_stale_position
        or persistent_relative_weakness
        or features.repeated_weak_position
    ):
        if features.distance_to_stop_pct is not None and features.distance_to_stop_pct <= watch_distance_pct:
            rationale.append("Latest close is close to the current stop.")
        if features.regime_passed is False:
            rationale.append("Benchmark regime filter is not currently passing.")
        if not features.above_entry:
            rationale.append("Position is below the average entry price.")
        if features.stale_position and features.days_since_new_high is not None:
            rationale.append(
                f"Position has gone {features.days_since_new_high} trading days without a new closing high."
            )
        if weak_relative_strength and features.relative_strength_return_diff is not None:
            if features.relative_strength_window is not None:
                rationale.append(
                    f"Relative strength vs benchmark over {features.relative_strength_window} trading days is "
                    f"{features.relative_strength_return_diff:.1%}."
                )
            else:
                rationale.append(
                    f"Relative strength vs benchmark is {features.relative_strength_return_diff:.1%}."
                )
        _append_position_trajectory_rationale(
            rationale,
            features=features,
            persistent_weakness_day_threshold=position_persistent_weakness_day_threshold,
        )
        action = "WATCH CLOSELY"
    else:
        rationale.append("Position remains above entry and sufficiently above the current stop.")
        action = "HOLD"

    if features.is_earnings_risk:
        rationale.append(_earnings_risk_note(features.earnings_date, features.earnings_days_away))
        if action == "HOLD":
            rationale.append("Upcoming earnings add event risk to the position.")
            action = "WATCH CLOSELY"
        elif action == "RAISE STOP":
            rationale.append(
                "Upcoming earnings still increase gap risk even though a higher stop is justified."
            )

    if weak_sector_regime or features.lagging_sector:
        if weak_sector_regime:
            rationale.append(_sector_regime_note(features.sector_etf_symbol, features.sector_name))
        if features.lagging_sector and features.relative_strength_vs_sector is not None:
            rationale.append(
                _sector_lag_note(
                    features.sector_etf_symbol,
                    sector_name=features.sector_name,
                    relative_strength_vs_sector=features.relative_strength_vs_sector,
                    window=features.sector_relative_strength_window,
                )
            )
        if action == "HOLD":
            action = "WATCH CLOSELY"
        elif action == "RAISE STOP":
            rationale.append(
                "Weak sector context still argues for caution, but raising the stop improves risk control."
            )

    if action == "HOLD" and features.recovery_after_multi_day_weakness:
        rationale.append(_daily_recovery_note(features))

    if action == "EXIT CANDIDATE":
        suggested_stop = None

    return PortfolioReviewDecision(
        latest_close=features.latest_close,
        unrealized_pl_pct=features.unrealized_pl_pct,
        distance_to_stop_pct=features.distance_to_stop_pct,
        regime_passed=features.regime_passed,
        above_entry=features.above_entry,
        suggested_action=action,
        suggested_stop=suggested_stop,
        trailing_stop_candidate=(
            None if action == "EXIT CANDIDATE" else features.trailing_stop_candidate
        ),
        rationale=tuple(rationale),
        metadata=metadata,
    )


def _append_position_trajectory_rationale(
    rationale: list[str],
    *,
    features: PositionFeatures,
    persistent_weakness_day_threshold: int,
) -> None:
    if features.consecutive_days_below_entry >= persistent_weakness_day_threshold:
        rationale.append(
            "Position has remained below entry for "
            f"{features.consecutive_days_below_entry} consecutive review sessions."
        )
    if (
        features.consecutive_weak_relative_strength_days >= persistent_weakness_day_threshold
        and features.relative_strength_return_diff is not None
    ):
        rationale.append(
            "Relative strength versus the benchmark has remained weak for "
            f"{features.consecutive_weak_relative_strength_days} consecutive review sessions."
        )
    if features.consecutive_stale_position_days >= persistent_weakness_day_threshold:
        rationale.append(
            "Stale-position pressure has persisted for "
            f"{features.consecutive_stale_position_days} consecutive review sessions."
        )
    if (
        features.repeated_weak_position
        and features.consecutive_weak_position_days >= 2
        and not (
            features.consecutive_days_below_entry >= persistent_weakness_day_threshold
            or features.consecutive_weak_relative_strength_days
            >= persistent_weakness_day_threshold
            or features.consecutive_stale_position_days >= persistent_weakness_day_threshold
        )
    ):
        rationale.append(
            "Weak position behavior has persisted across "
            f"{features.consecutive_weak_position_days} consecutive review sessions."
        )


def _daily_recovery_note(features: PositionFeatures) -> str:
    if features.above_entry:
        return "Position recovered after multiple weak review sessions and is back above entry."
    return "Position recovered after multiple weak review sessions."


def review_existing_long_position_intraday(
    position: ExistingPosition,
    *,
    position_features: PositionFeatures | None = None,
    session_open: float,
    session_high: float,
    session_low: float,
    latest_close: float,
    latest_low: float,
    session_vwap: float | None = None,
    session_high_giveback_exit_threshold: float = 0.10,
    session_high_giveback_watch_threshold: float | None = None,
    meaningful_profit_pct: float = 0.05,
    early_strength_threshold: float = 0.03,
    intraday_relative_strength_diff: float | None = None,
    intraday_relative_strength_watch_threshold: float = -0.02,
    intraday_persistent_weakness_poll_threshold: int = 3,
    intraday_repeated_watch_exit_threshold: int = 2,
    intraday_high_profit_unrealized_pct: float = 0.15,
    intraday_high_profit_giveback_threshold: float = 0.07,
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    earnings_watch_days: int = 7,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
    sector_relative_strength_watch_threshold: float = -0.05,
) -> PortfolioReviewDecision:
    """Review one long position using a single intraday session snapshot."""

    _validate_finite_positive(session_open, name="session_open")
    _validate_finite_positive(session_high, name="session_high")
    _validate_finite_positive(session_low, name="session_low")
    _validate_finite_positive(latest_close, name="latest_close")
    _validate_finite_positive(latest_low, name="latest_low")
    if session_vwap is not None:
        _validate_finite_positive(session_vwap, name="session_vwap")
    if session_high_giveback_exit_threshold <= 0 or session_high_giveback_exit_threshold >= 1:
        raise ValueError(
            "session_high_giveback_exit_threshold must be between zero and one."
        )
    if session_high_giveback_watch_threshold is None:
        session_high_giveback_watch_threshold = max(
            0.04,
            session_high_giveback_exit_threshold * 0.75,
        )
    if session_high_giveback_watch_threshold <= 0 or session_high_giveback_watch_threshold >= 1:
        raise ValueError(
            "session_high_giveback_watch_threshold must be between zero and one."
        )
    if meaningful_profit_pct <= 0:
        raise ValueError("meaningful_profit_pct must be greater than zero.")
    if early_strength_threshold <= 0:
        raise ValueError("early_strength_threshold must be greater than zero.")
    if intraday_persistent_weakness_poll_threshold <= 0:
        raise ValueError(
            "intraday_persistent_weakness_poll_threshold must be greater than zero."
        )
    if intraday_repeated_watch_exit_threshold <= 0:
        raise ValueError("intraday_repeated_watch_exit_threshold must be greater than zero.")
    if intraday_high_profit_unrealized_pct <= 0:
        raise ValueError("intraday_high_profit_unrealized_pct must be greater than zero.")
    if intraday_high_profit_giveback_threshold <= 0 or intraday_high_profit_giveback_threshold >= 1:
        raise ValueError("intraday_high_profit_giveback_threshold must be between zero and one.")
    if earnings_days_away is not None and earnings_days_away < 0:
        raise ValueError("earnings_days_away cannot be negative.")
    if earnings_watch_days <= 0:
        raise ValueError("earnings_watch_days must be greater than zero.")
    if sector_relative_strength_window is not None and sector_relative_strength_window <= 0:
        raise ValueError(
            "sector_relative_strength_window must be greater than zero when provided."
        )
    if sector_relative_strength_watch_threshold > 0:
        raise ValueError(
            "sector_relative_strength_watch_threshold must be less than or equal to zero."
        )
    features = position_features or build_intraday_position_features(
        symbol=position.symbol,
        as_of_date=date.today(),
        average_entry_price=position.average_entry_price,
        current_stop=position.current_stop,
        session_open=session_open,
        session_high=session_high,
        session_low=session_low,
        latest_close=latest_close,
        latest_low=latest_low,
        session_vwap=session_vwap,
        intraday_return_vs_benchmark=intraday_relative_strength_diff,
        earnings_date=earnings_date,
        earnings_days_away=earnings_days_away,
        sector_name=sector_name,
        industry_name=industry_name,
        sector_etf_symbol=sector_etf_symbol,
        sector_regime_passed=sector_regime_passed,
        relative_strength_vs_sector=relative_strength_vs_sector,
        sector_relative_strength_window=sector_relative_strength_window,
        earnings_watch_days=earnings_watch_days,
        early_strength_threshold=early_strength_threshold,
        meaningful_profit_pct=meaningful_profit_pct,
        session_high_giveback_watch_threshold=session_high_giveback_watch_threshold,
        intraday_relative_strength_watch_threshold=intraday_relative_strength_watch_threshold,
        sector_relative_strength_watch_threshold=sector_relative_strength_watch_threshold,
    )
    return _review_existing_long_position_intraday_from_features(
        features=features,
        session_high_giveback_exit_threshold=session_high_giveback_exit_threshold,
        session_high_giveback_watch_threshold=session_high_giveback_watch_threshold,
        meaningful_profit_pct=meaningful_profit_pct,
        intraday_relative_strength_watch_threshold=intraday_relative_strength_watch_threshold,
        intraday_persistent_weakness_poll_threshold=(
            intraday_persistent_weakness_poll_threshold
        ),
        intraday_repeated_watch_exit_threshold=intraday_repeated_watch_exit_threshold,
        intraday_high_profit_unrealized_pct=intraday_high_profit_unrealized_pct,
        intraday_high_profit_giveback_threshold=intraday_high_profit_giveback_threshold,
        earnings_watch_days=earnings_watch_days,
        sector_relative_strength_watch_threshold=sector_relative_strength_watch_threshold,
    )


def _review_existing_long_position_intraday_from_features(
    *,
    features: PositionFeatures,
    session_high_giveback_exit_threshold: float,
    session_high_giveback_watch_threshold: float,
    meaningful_profit_pct: float,
    intraday_relative_strength_watch_threshold: float,
    intraday_persistent_weakness_poll_threshold: int,
    intraday_repeated_watch_exit_threshold: int,
    intraday_high_profit_unrealized_pct: float,
    intraday_high_profit_giveback_threshold: float,
    earnings_watch_days: int,
    sector_relative_strength_watch_threshold: float,
) -> PortfolioReviewDecision:
    weak_sector_regime = features.sector_regime_passed is False
    persistent_below_vwap = (
        features.consecutive_polls_below_vwap >= intraday_persistent_weakness_poll_threshold
    )
    persistent_relative_weakness = (
        features.consecutive_polls_weak_relative_strength
        >= intraday_persistent_weakness_poll_threshold
    )
    persistent_intraday_pressure = (
        features.intraday_pressure_persistence_count
        >= intraday_persistent_weakness_poll_threshold
    )
    repeated_watch_pressure = (
        features.repeated_watch_closely_count >= intraday_repeated_watch_exit_threshold
    )
    worsening_giveback = (
        features.worsening_session_high_giveback
        and features.giveback_worsening_polls >= 2
    )

    rationale: list[str] = []
    metadata = {
        "session_open": features.session_open,
        "session_high": features.session_high,
        "session_low": features.session_low,
        "latest_low": features.latest_low,
        "session_vwap": features.session_vwap,
        "intraday_return_vs_open": features.intraday_return_vs_open,
        "peak_intraday_return_vs_open": features.peak_intraday_return_vs_open,
        "peak_unrealized_pct": features.peak_unrealized_pct,
        "session_high_giveback_pct": features.session_high_giveback_pct,
        "session_high_giveback_watch_threshold": session_high_giveback_watch_threshold,
        "session_high_giveback_exit_threshold": session_high_giveback_exit_threshold,
        "close_vs_vwap_pct": features.close_vs_vwap_pct,
        "stop_breached_intraday": features.stop_breached_intraday,
        "failed_intraday_strength": features.failed_intraday_strength,
        "intraday_momentum_fade": features.intraday_momentum_fade,
        "intraday_relative_strength_diff": features.intraday_return_vs_benchmark,
        "intraday_relative_strength_watch_threshold": intraday_relative_strength_watch_threshold,
        "weak_intraday_relative_strength": features.weak_intraday_relative_strength,
        "stacked_intraday_weakness": features.stacked_intraday_weakness,
        "intraday_observation_count": features.intraday_observation_count,
        "consecutive_polls_below_vwap": features.consecutive_polls_below_vwap,
        "consecutive_polls_weak_relative_strength": (
            features.consecutive_polls_weak_relative_strength
        ),
        "intraday_pressure_persistence_count": (
            features.intraday_pressure_persistence_count
        ),
        "max_session_high_giveback_seen": features.max_session_high_giveback_seen,
        "worsening_session_high_giveback": features.worsening_session_high_giveback,
        "giveback_worsening_polls": features.giveback_worsening_polls,
        "giveback_worsening_from_pct": features.giveback_worsening_from_pct,
        "repeated_watch_closely_count": features.repeated_watch_closely_count,
        "weakening_all_session": features.weakening_all_session,
        "recovered_after_weakness": features.recovered_after_weakness,
        "intraday_persistent_weakness_poll_threshold": (
            intraday_persistent_weakness_poll_threshold
        ),
        "intraday_repeated_watch_exit_threshold": intraday_repeated_watch_exit_threshold,
        "persistent_below_vwap": persistent_below_vwap,
        "persistent_weak_intraday_relative_strength": persistent_relative_weakness,
        "persistent_intraday_pressure": persistent_intraday_pressure,
        "intraday_high_profit_unrealized_pct": intraday_high_profit_unrealized_pct,
        "intraday_high_profit_giveback_threshold": intraday_high_profit_giveback_threshold,
        "earnings_date": (
            features.earnings_date.isoformat() if features.earnings_date is not None else None
        ),
        "earnings_days_away": features.earnings_days_away,
        "earnings_watch_days": earnings_watch_days,
        "is_earnings_risk": features.is_earnings_risk,
        "sector_name": features.sector_name,
        "industry_name": features.industry_name,
        "sector_etf_symbol": features.sector_etf_symbol,
        "sector_regime_passed": features.sector_regime_passed,
        "relative_strength_vs_sector": features.relative_strength_vs_sector,
        "sector_relative_strength_window": features.sector_relative_strength_window,
        "sector_relative_strength_watch_threshold": sector_relative_strength_watch_threshold,
        "weak_sector_regime": weak_sector_regime,
        "lagging_sector": features.lagging_sector,
    }

    if features.stop_breached_intraday:
        rationale.append("An intraday bar traded through the current stop.")
        action = "EXIT CANDIDATE"
    elif (
        features.peak_unrealized_pct is not None
        and features.session_high_giveback_pct is not None
        and features.peak_unrealized_pct >= meaningful_profit_pct
        and features.session_high_giveback_pct >= session_high_giveback_exit_threshold
        and features.session_high is not None
    ):
        rationale.append(
            "Profitable position has given back "
            f"{features.session_high_giveback_pct:.1%} from the session high of {features.session_high:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif features.stacked_intraday_weakness and features.session_high_giveback_pct is not None:
        rationale.append(
            "Three corroborating weakness signals: below VWAP, "
            f"{features.session_high_giveback_pct:.1%} giveback from session high, "
            "and underperforming the benchmark intraday."
        )
        action = "EXIT CANDIDATE"
    elif (
        repeated_watch_pressure
        and persistent_intraday_pressure
        and (
            (features.session_high_giveback_pct is not None and features.session_high_giveback_pct >= session_high_giveback_watch_threshold)
            or features.failed_intraday_strength
            or features.intraday_momentum_fade
        )
    ):
        rationale.append(
            f"Intraday pressure has persisted for {features.intraday_pressure_persistence_count} consecutive polls."
        )
        rationale.append(
            "Position has already triggered WATCH CLOSELY on "
            f"{features.repeated_watch_closely_count} consecutive prior polls."
        )
        _append_intraday_trajectory_rationale(
            rationale,
            features=features,
            persistent_weakness_poll_threshold=intraday_persistent_weakness_poll_threshold,
        )
        action = "EXIT CANDIDATE"
    elif (
        features.peak_unrealized_pct is not None
        and features.session_high_giveback_pct is not None
        and features.session_high is not None
        and features.peak_unrealized_pct >= intraday_high_profit_unrealized_pct
        and features.session_high_giveback_pct >= intraday_high_profit_giveback_threshold
    ):
        rationale.append(
            f"Position has unrealized gains of {features.peak_unrealized_pct:.1%} but has given back "
            f"{features.session_high_giveback_pct:.1%} from the session high of {features.session_high:.2f}; "
            "protecting profit at a tighter threshold."
        )
        action = "EXIT CANDIDATE"
    elif (
        features.failed_intraday_strength
        and features.session_high_giveback_pct is not None
        and features.session_high_giveback_pct >= session_high_giveback_watch_threshold
    ):
        rationale.append("Strong intraday move failed to hold into the latest bar.")
        if features.session_vwap is not None and features.latest_close < features.session_vwap:
            rationale.append(
                f"Latest close is below session VWAP ({features.session_vwap:.2f})."
            )
        action = "EXIT CANDIDATE"
    elif (
        features.intraday_momentum_fade
        or features.weak_intraday_relative_strength
        or features.failed_intraday_strength
        or persistent_below_vwap
        or persistent_relative_weakness
        or worsening_giveback
        or features.weakening_all_session
    ):
        if features.intraday_momentum_fade:
            rationale.append(
                "Profitable position is fading intraday after a strong session high."
            )
        if features.failed_intraday_strength:
            rationale.append("Strong intraday move is no longer holding the session range.")
        if features.session_vwap is not None and features.latest_close < features.session_vwap:
            rationale.append(
                f"Latest close is below session VWAP ({features.session_vwap:.2f})."
            )
        if (
            features.weak_intraday_relative_strength
            and features.intraday_return_vs_benchmark is not None
        ):
            rationale.append(
                "Intraday return vs benchmark is "
                f"{features.intraday_return_vs_benchmark:.1%}."
            )
        _append_intraday_trajectory_rationale(
            rationale,
            features=features,
            persistent_weakness_poll_threshold=intraday_persistent_weakness_poll_threshold,
        )
        action = "WATCH CLOSELY"
    else:
        rationale.append("Intraday structure remains healthy and no stop or fade signal is active.")
        action = "HOLD"

    if features.is_earnings_risk:
        rationale.append(_earnings_risk_note(features.earnings_date, features.earnings_days_away))
        if action == "HOLD":
            rationale.append("Upcoming earnings add event risk to the position.")
            action = "WATCH CLOSELY"
        elif action == "RAISE STOP":
            rationale.append(
                "Upcoming earnings still increase gap risk even though a higher stop is justified."
            )

    if weak_sector_regime or features.lagging_sector:
        if weak_sector_regime:
            rationale.append(_sector_regime_note(features.sector_etf_symbol, features.sector_name))
        if features.lagging_sector and features.relative_strength_vs_sector is not None:
            rationale.append(
                _sector_lag_note(
                    features.sector_etf_symbol,
                    sector_name=features.sector_name,
                    relative_strength_vs_sector=features.relative_strength_vs_sector,
                    window=features.sector_relative_strength_window,
                )
            )
        if action == "HOLD":
            action = "WATCH CLOSELY"

    if action == "HOLD" and features.recovered_after_weakness:
        rationale.append(_intraday_recovery_note(features))

    return PortfolioReviewDecision(
        latest_close=features.latest_close,
        unrealized_pl_pct=features.unrealized_pl_pct,
        distance_to_stop_pct=features.distance_to_stop_pct,
        regime_passed=features.regime_passed,
        above_entry=features.above_entry,
        suggested_action=action,
        suggested_stop=None,
        trailing_stop_candidate=None,
        rationale=tuple(rationale),
        metadata=metadata,
    )


def _append_intraday_trajectory_rationale(
    rationale: list[str],
    *,
    features: PositionFeatures,
    persistent_weakness_poll_threshold: int,
) -> None:
    if (
        features.consecutive_polls_below_vwap >= persistent_weakness_poll_threshold
        and features.session_vwap is not None
    ):
        rationale.append(
            "Position has remained below session VWAP for "
            f"{features.consecutive_polls_below_vwap} consecutive polls."
        )
    if features.worsening_session_high_giveback and (
        features.giveback_worsening_from_pct is not None
        and features.session_high_giveback_pct is not None
        and features.giveback_worsening_polls >= 2
    ):
        rationale.append(
            "Giveback from session high has worsened from "
            f"{features.giveback_worsening_from_pct:.1%} to "
            f"{features.session_high_giveback_pct:.1%} over the last "
            f"{features.giveback_worsening_polls} polls."
        )
    if (
        features.consecutive_polls_weak_relative_strength >= persistent_weakness_poll_threshold
        and features.intraday_return_vs_benchmark is not None
    ):
        rationale.append(
            "Intraday weakness versus the benchmark has persisted for "
            f"{features.consecutive_polls_weak_relative_strength} consecutive polls."
        )
    if features.weakening_all_session:
        rationale.append("Intraday pressure has persisted throughout all observed polls this session.")


def _intraday_recovery_note(features: PositionFeatures) -> str:
    if (
        features.session_vwap is not None
        and features.close_vs_vwap_pct is not None
        and features.close_vs_vwap_pct >= 0
    ):
        return "Position recovered after earlier intraday weakness and is back above session VWAP."
    return "Position recovered after earlier intraday weakness."


def apply_drawdown_risk_adjustment(
    base_risk_per_trade: float,
    *,
    current_drawdown: float,
    threshold: float | None,
    reduction_factor: float = 0.5,
) -> float:
    """Reduce per-trade risk when drawdown exceeds a configured threshold."""

    _validate_finite_non_negative(base_risk_per_trade, name="base_risk_per_trade")
    _validate_finite_non_negative(current_drawdown, name="current_drawdown")
    if threshold is None:
        return base_risk_per_trade
    _validate_finite_non_negative(threshold, name="threshold")
    if reduction_factor <= 0 or reduction_factor > 1:
        raise ValueError("reduction_factor must be in the interval (0, 1].")
    if current_drawdown >= threshold:
        return base_risk_per_trade * reduction_factor
    return base_risk_per_trade


def evaluate_portfolio_rules(
    signal: StrategySignal,
    *,
    current_positions: Sequence[ExistingPosition],
    constraints: PortfolioConstraints,
) -> PortfolioRuleResult:
    """Evaluate deterministic portfolio rules for a candidate position.

    Checks covered here: concurrent-position cap, no-averaging-down, and
    duplicate-entry blocking.  The notional/position-size cap is enforced
    upstream by ``size_position`` before this function is ever called, so it
    is not re-checked here.
    """

    reasons: list[str] = []
    existing_position = _find_existing_position(current_positions, signal.symbol)

    is_new_symbol = existing_position is None
    if is_new_symbol and len(current_positions) >= constraints.max_concurrent_positions:
        reasons.append("Max concurrent positions reached.")

    if existing_position is not None and signal.side == "BUY":
        if (
            constraints.no_averaging_down
            and signal.entry_price_hint < existing_position.average_entry_price
        ):
            reasons.append("No averaging down is allowed for existing long positions.")
        else:
            reasons.append("Existing long position already open for symbol; duplicate entries are not allowed.")

    return PortfolioRuleResult(approved=not reasons, reasons=tuple(reasons))


def assess_signal_candidate(
    signal: StrategySignal,
    *,
    current_equity: float,
    base_risk_per_trade: float,
    constraints: PortfolioConstraints,
    candidate_features: CandidateFeatures | None = None,
    current_positions: Sequence[ExistingPosition] = (),
    current_drawdown: float = 0.0,
    stop_price: float | None = None,
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    earnings_entry_block_days: int | None = None,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
    market_context: MarketContext | None = None,
    market_breadth_entry_floor_200ma: float | None = None,
    vix_entry_block_threshold: float | None = None,
    require_sector_regime_for_entries: bool = False,
    sector_relative_strength_entry_reject_threshold: float | None = None,
) -> RiskAssessedCandidate:
    """Combine sizing and portfolio rules into one risk-assessed entry candidate."""

    if candidate_features is not None and market_context is not None:
        raise ValueError(
            "Pass market_context via candidate_features, not as a separate argument."
        )

    existing_position = _find_existing_position(current_positions, signal.symbol)
    resolved_stop_price = signal.stop_hint if stop_price is None else stop_price
    adjusted_risk_per_trade = apply_drawdown_risk_adjustment(
        base_risk_per_trade,
        current_drawdown=current_drawdown,
        threshold=constraints.drawdown_risk_reduction_threshold,
        reduction_factor=constraints.drawdown_risk_reduction_factor,
    )
    features = candidate_features or build_candidate_features(
        signal,
        as_of_date=signal.date,
        stop_hint=resolved_stop_price,
        earnings_date=earnings_date,
        earnings_days_away=earnings_days_away,
        is_earnings_risk=bool(
            earnings_days_away is not None
            and earnings_entry_block_days is not None
            and earnings_days_away <= earnings_entry_block_days
        ),
        sector_name=sector_name,
        industry_name=industry_name,
        sector_etf_symbol=sector_etf_symbol,
        sector_regime_passed=sector_regime_passed,
        relative_strength_vs_sector=relative_strength_vs_sector,
        sector_relative_strength_window=sector_relative_strength_window,
        market_context=market_context,
    )

    if resolved_stop_price is None:
        sizing = PositionSizingResult(
            shares=0,
            risk_budget=max(current_equity * adjusted_risk_per_trade, 0.0),
            per_share_risk=0.0,
            notional_value=0.0,
            max_shares_by_risk=0,
            max_shares_by_notional=None,
            capped_by_notional=False,
            is_valid=False,
            rejection_reason="A stop price is required for risk assessment.",
        )
        return RiskAssessedCandidate(
            signal=signal,
            entry_price=signal.entry_price_hint,
            stop_price=signal.entry_price_hint,
            adjusted_risk_per_trade=adjusted_risk_per_trade,
            sizing=sizing,
            approved=False,
            rejection_reasons=(sizing.rejection_reason,),
            existing_position=existing_position,
            features=features,
        )
    return _assess_signal_candidate_from_features(
        signal,
        features=features,
        current_equity=current_equity,
        adjusted_risk_per_trade=adjusted_risk_per_trade,
        constraints=constraints,
        current_positions=current_positions,
        existing_position=existing_position,
        stop_price=resolved_stop_price,
        earnings_entry_block_days=earnings_entry_block_days,
        market_breadth_entry_floor_200ma=market_breadth_entry_floor_200ma,
        vix_entry_block_threshold=vix_entry_block_threshold,
        require_sector_regime_for_entries=require_sector_regime_for_entries,
        sector_relative_strength_entry_reject_threshold=(
            sector_relative_strength_entry_reject_threshold
        ),
    )


def _assess_signal_candidate_from_features(
    signal: StrategySignal,
    *,
    features: CandidateFeatures,
    current_equity: float,
    adjusted_risk_per_trade: float,
    constraints: PortfolioConstraints,
    current_positions: Sequence[ExistingPosition],
    existing_position: ExistingPosition | None,
    stop_price: float,
    earnings_entry_block_days: int | None,
    market_breadth_entry_floor_200ma: float | None,
    vix_entry_block_threshold: float | None,
    require_sector_regime_for_entries: bool,
    sector_relative_strength_entry_reject_threshold: float | None,
) -> RiskAssessedCandidate:
    if features.earnings_days_away is not None and features.earnings_days_away < 0:
        raise ValueError("earnings_days_away cannot be negative.")
    if earnings_entry_block_days is not None and earnings_entry_block_days <= 0:
        raise ValueError("earnings_entry_block_days must be greater than zero when provided.")
    if market_breadth_entry_floor_200ma is not None and (
        market_breadth_entry_floor_200ma <= 0
        or market_breadth_entry_floor_200ma >= 1
    ):
        raise ValueError(
            "market_breadth_entry_floor_200ma must be between zero and one when provided."
        )
    if vix_entry_block_threshold is not None and vix_entry_block_threshold <= 0:
        raise ValueError("vix_entry_block_threshold must be greater than zero when provided.")
    if (
        features.sector_relative_strength_window is not None
        and features.sector_relative_strength_window <= 0
    ):
        raise ValueError(
            "sector_relative_strength_window must be greater than zero when provided."
        )
    if (
        sector_relative_strength_entry_reject_threshold is not None
        and sector_relative_strength_entry_reject_threshold > 0
    ):
        raise ValueError(
            "sector_relative_strength_entry_reject_threshold must be less than or equal to zero when provided."
        )

    sizing = size_position(
        current_equity=current_equity,
        risk_per_trade=adjusted_risk_per_trade,
        entry_price=signal.entry_price_hint,
        stop_price=stop_price,
        max_position_pct_equity=constraints.max_position_pct_equity,
    )

    rejection_reasons: list[str] = []
    if not sizing.is_valid and sizing.rejection_reason is not None:
        rejection_reasons.append(sizing.rejection_reason)

    if sizing.is_valid:
        rule_result = evaluate_portfolio_rules(
            signal,
            current_positions=current_positions,
            constraints=constraints,
        )
        rejection_reasons.extend(rule_result.reasons)

    if (
        features.earnings_days_away is not None
        and earnings_entry_block_days is not None
        and features.earnings_days_away <= earnings_entry_block_days
    ):
        rejection_reasons.append(
            _earnings_entry_block_reason(
                earnings_date=features.earnings_date,
                earnings_days_away=features.earnings_days_away,
                earnings_entry_block_days=earnings_entry_block_days,
            )
        )

    market_breadth_pct_above_200ma = (
        features.market_context.market_breadth_pct_above_200ma
        if features.market_context is not None
        else None
    )
    if (
        market_breadth_entry_floor_200ma is not None
        and market_breadth_pct_above_200ma is not None
        and market_breadth_pct_above_200ma < market_breadth_entry_floor_200ma
    ):
        rejection_reasons.append(
            _market_breadth_entry_reject_reason(
                market_breadth_pct_above_200ma=market_breadth_pct_above_200ma,
                market_breadth_entry_floor_200ma=market_breadth_entry_floor_200ma,
            )
        )

    vix_close = (
        features.market_context.vix_close
        if features.market_context is not None
        else None
    )
    volatility_regime_state = (
        features.market_context.volatility_regime_state
        if features.market_context is not None
        else None
    )
    volatility_regime_risk_off = (
        features.market_context.volatility_regime_risk_off
        if features.market_context is not None
        else None
    )
    if vix_entry_block_threshold is not None and (
        volatility_regime_risk_off is True
        or (vix_close is not None and vix_close >= vix_entry_block_threshold)
    ):
        rejection_reasons.append(
            _volatility_entry_reject_reason(
                vix_close=vix_close,
                vix_entry_block_threshold=vix_entry_block_threshold,
                volatility_regime_state=volatility_regime_state,
            )
        )

    if require_sector_regime_for_entries and features.sector_regime_passed is False:
        rejection_reasons.append(
            _sector_entry_regime_reject_reason(
                features.sector_etf_symbol,
                features.sector_name,
            )
        )

    if (
        features.relative_strength_vs_sector is not None
        and sector_relative_strength_entry_reject_threshold is not None
        and features.relative_strength_vs_sector <= sector_relative_strength_entry_reject_threshold
    ):
        rejection_reasons.append(
            _sector_entry_lag_reject_reason(
                features.sector_etf_symbol,
                sector_name=features.sector_name,
                relative_strength_vs_sector=features.relative_strength_vs_sector,
                window=features.sector_relative_strength_window,
            )
        )

    approved = sizing.is_valid and not rejection_reasons
    return RiskAssessedCandidate(
        signal=signal,
        entry_price=signal.entry_price_hint,
        stop_price=stop_price,
        adjusted_risk_per_trade=adjusted_risk_per_trade,
        sizing=sizing,
        approved=approved,
        rejection_reasons=tuple(rejection_reasons),
        existing_position=existing_position,
        features=features,
    )


def _find_existing_position(
    positions: Sequence[ExistingPosition],
    symbol: str,
) -> ExistingPosition | None:
    normalized_symbol = symbol.strip().upper()
    for position in positions:
        if position.symbol.strip().upper() == normalized_symbol:
            return position
    return None


def _earnings_risk_note(earnings_date: date | None, earnings_days_away: int | None) -> str:
    if earnings_date is None or earnings_days_away is None:
        return "Upcoming earnings add event risk."
    if earnings_days_away == 0:
        return f"Upcoming earnings are scheduled on {earnings_date.isoformat()} (today)."
    return (
        f"Upcoming earnings are scheduled on {earnings_date.isoformat()} "
        f"({earnings_days_away} trading days away)."
    )


def _earnings_entry_block_reason(
    *,
    earnings_date: date | None,
    earnings_days_away: int,
    earnings_entry_block_days: int,
) -> str:
    if earnings_date is None:
        return (
            f"Upcoming earnings are within {earnings_entry_block_days} trading days; "
            "new entries are blocked."
        )
    if earnings_days_away == 0:
        return (
            f"Upcoming earnings on {earnings_date.isoformat()} are scheduled today; "
            f"new entries are blocked within {earnings_entry_block_days} trading days."
        )
    return (
        f"Upcoming earnings on {earnings_date.isoformat()} are {earnings_days_away} "
        f"trading days away; new entries are blocked within {earnings_entry_block_days} trading days."
    )


def _market_breadth_entry_reject_reason(
    *,
    market_breadth_pct_above_200ma: float,
    market_breadth_entry_floor_200ma: float,
) -> str:
    return (
        "Market breadth is weak: "
        f"{market_breadth_pct_above_200ma:.0%} of the universe is above its 200-day moving average; "
        f"new entries require at least {market_breadth_entry_floor_200ma:.0%}."
    )


def _volatility_entry_reject_reason(
    *,
    vix_close: float | None,
    vix_entry_block_threshold: float,
    volatility_regime_state: str | None,
) -> str:
    state = volatility_regime_state or "stressed"
    if vix_close is None:
        return (
            f"Volatility regime is {state}; VIX data is unavailable but derived regime state "
            f"indicates {state} conditions."
        )
    return (
        f"Volatility regime is {state}: VIX is {vix_close:.1f}, above the "
        f"{vix_entry_block_threshold:.1f} entry block threshold."
    )


def _sector_regime_note(sector_etf_symbol: str | None, sector_name: str | None) -> str:
    if sector_etf_symbol is not None:
        return f"Sector ETF {sector_etf_symbol} is below its trend filter."
    if sector_name is not None:
        return f"Sector context for {sector_name} is weak."
    return "Sector context is weak."


def _sector_lag_note(
    sector_etf_symbol: str | None,
    *,
    sector_name: str | None,
    relative_strength_vs_sector: float,
    window: int | None,
) -> str:
    target = sector_etf_symbol or sector_name or "its sector"
    window_text = f" over {window} trading days" if window is not None else ""
    return (
        f"Stock is lagging {target} by {abs(relative_strength_vs_sector):.1%}{window_text}."
    )


def _sector_entry_regime_reject_reason(
    sector_etf_symbol: str | None,
    sector_name: str | None,
) -> str:
    if sector_etf_symbol is not None:
        return f"Sector ETF {sector_etf_symbol} is below its trend filter; new entries require sector support."
    if sector_name is not None:
        return f"Sector context for {sector_name} is weak; new entries require sector support."
    return "Sector context is weak; new entries require sector support."


def _sector_entry_lag_reject_reason(
    sector_etf_symbol: str | None,
    *,
    sector_name: str | None,
    relative_strength_vs_sector: float,
    window: int | None,
) -> str:
    target = sector_etf_symbol or sector_name or "its sector"
    window_text = f" over {window} trading days" if window is not None else ""
    return (
        f"Stock is lagging {target} by {abs(relative_strength_vs_sector):.1%}{window_text}; "
        "new entries require sector-relative strength."
    )


def _validate_finite_non_negative(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_finite_positive(value: float, *, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _load_existing_positions_from_csv(portfolio_path: Path) -> list[ExistingPosition]:
    with portfolio_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PortfolioInputError(f"Portfolio CSV is missing a header row: {portfolio_path}")

        positions: list[ExistingPosition] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            positions.append(ExistingPosition.from_mapping(row, row_number=row_number))
    return positions


def _load_existing_positions_from_json(portfolio_path: Path) -> list[ExistingPosition]:
    try:
        payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortfolioInputError(f"Portfolio JSON is invalid: {portfolio_path}") from exc

    raw_positions: Any
    if isinstance(payload, list):
        raw_positions = payload
    elif isinstance(payload, dict):
        raw_positions = payload.get("positions", payload.get("current_positions"))
    else:
        raw_positions = None

    if raw_positions is None:
        return []
    if not isinstance(raw_positions, list):
        raise PortfolioInputError("Portfolio JSON must be a list or an object with a 'positions' list.")

    positions: list[ExistingPosition] = []
    for index, item in enumerate(raw_positions, start=1):
        if not isinstance(item, Mapping):
            raise PortfolioInputError("Each portfolio position must be a JSON object.")
        positions.append(ExistingPosition.from_mapping(item, row_number=index))
    return positions


def _validate_unique_position_symbols(
    positions: Sequence[ExistingPosition],
    *,
    source: Path,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for position in positions:
        symbol = position.symbol.strip().upper()
        if symbol in seen:
            duplicates.add(symbol)
        seen.add(symbol)
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates))
        raise PortfolioInputError(
            f"Portfolio file contains duplicate symbols ({duplicate_text}): {source}"
        )


def _required_text(data: Mapping[str, Any], key: str, *, row_prefix: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PortfolioInputError(f"{row_prefix}'{key}' is required.")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _positive_int(value: Any, *, field_name: str, row_prefix: str) -> int:
    if value is None:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' is required.")
    try:
        resolved = int(str(value).strip())
    except ValueError as exc:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' must be an integer.") from exc
    if resolved <= 0:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' must be greater than zero.")
    return resolved


def _positive_float(value: Any, *, field_name: str, row_prefix: str) -> float:
    if value is None:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' is required.")
    try:
        resolved = float(str(value).strip())
    except ValueError as exc:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' must be numeric.") from exc
    if not isfinite(resolved) or resolved <= 0:
        raise PortfolioInputError(f"{row_prefix}'{field_name}' must be greater than zero.")
    return resolved


def _optional_positive_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        resolved = float(cleaned)
    except ValueError as exc:
        raise PortfolioInputError(f"'{field_name}' must be numeric when provided.") from exc
    if not isfinite(resolved) or resolved <= 0:
        raise PortfolioInputError(f"'{field_name}' must be greater than zero when provided.")
    return resolved


def _extract_position_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    raw_metadata_json = data.get("metadata_json")
    if raw_metadata_json is not None:
        cleaned_metadata_json = str(raw_metadata_json).strip()
        if cleaned_metadata_json:
            try:
                parsed_metadata = json.loads(cleaned_metadata_json)
            except json.JSONDecodeError as exc:
                raise PortfolioInputError("'metadata_json' must contain valid JSON when provided.") from exc
            if not isinstance(parsed_metadata, Mapping):
                raise PortfolioInputError("'metadata_json' must decode to an object when provided.")
            metadata.update(dict(parsed_metadata))

    explicit_metadata = data.get("metadata")
    if isinstance(explicit_metadata, Mapping):
        metadata.update(dict(explicit_metadata))

    known_keys = {
        "symbol",
        "quantity",
        "shares",
        "average_entry_price",
        "current_stop",
        "stop_price",
        "preset_name",
        "source",
        "metadata_json",
        "metadata",
    }
    for key, value in data.items():
        if key in known_keys or value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                continue
            metadata.setdefault(key, cleaned)
        else:
            metadata.setdefault(key, value)

    return metadata


def _position_record(position: ExistingPosition) -> dict[str, str]:
    metadata_json = (
        json.dumps(position.metadata, sort_keys=True)
        if position.metadata
        else ""
    )
    return {
        "symbol": position.symbol,
        "quantity": str(position.shares),
        "average_entry_price": _format_optional_float(position.average_entry_price),
        "current_stop": _format_optional_float(position.current_stop),
        "preset_name": position.preset_name or "",
        "source": position.source or "",
        "metadata_json": metadata_json,
    }


def _resolve_snapshot_format(output_path: Path, output_format: str | None) -> str:
    suffix_format = output_path.suffix.lower().lstrip(".")
    normalized_output_format = output_format.strip().lower() if output_format is not None else None

    if normalized_output_format is not None and normalized_output_format not in {"csv", "json"}:
        raise PortfolioInputError("output_format must be either 'csv' or 'json'.")

    if normalized_output_format is not None:
        if suffix_format and suffix_format != normalized_output_format:
            raise PortfolioInputError(
                "output_format does not match the portfolio snapshot file extension."
            )
        return normalized_output_format

    if suffix_format in {"csv", "json"}:
        return suffix_format

    raise PortfolioInputError("Portfolio snapshot paths must use a .csv or .json extension.")


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")
