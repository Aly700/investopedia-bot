"""Portfolio-level rules and signal-to-risk assessment helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot.config import RiskConfig, RulesConfig
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
    if days_since_new_high is not None and days_since_new_high < 0:
        raise ValueError("days_since_new_high cannot be negative.")
    if relative_strength_window is not None and relative_strength_window <= 0:
        raise ValueError("relative_strength_window must be greater than zero when provided.")

    unrealized_pl_pct = (latest_close / position.average_entry_price) - 1.0
    above_entry = latest_close >= position.average_entry_price
    distance_to_stop_pct: float | None = None
    if position.current_stop is not None:
        distance_to_stop_pct = (latest_close - position.current_stop) / latest_close

    giveback_pct: float | None = None
    if high_water_close is not None:
        _validate_finite_positive(high_water_close, name="high_water_close")
        giveback_pct = max((high_water_close - latest_close) / high_water_close, 0.0)

    failed_breakout = (
        breakout_failure_reference is not None
        and latest_close < breakout_failure_reference
    )
    stale_position = (
        days_since_new_high is not None
        and days_since_new_high >= stale_high_watch_days
    )
    weak_relative_strength = (
        relative_strength_return_diff is not None
        and relative_strength_return_diff <= relative_strength_watch_threshold
    )

    suggested_stop = suggest_long_stop_update(
        current_stop=position.current_stop,
        trailing_stop_candidate=trailing_stop_candidate,
    )
    # A long protective stop must always be strictly below the latest close.
    # If the trailing-stop formula produced a value at or above the market price
    # (e.g. because ATR or a multiplier over-shot), treat it as unusable.
    if suggested_stop is not None and suggested_stop >= latest_close:
        suggested_stop = None
    rationale: list[str] = []
    metadata = {
        "high_water_close": high_water_close,
        "giveback_pct": giveback_pct,
        "profit_giveback_threshold": profit_giveback_threshold,
        "profit_giveback_min_unrealized_pct": profit_giveback_min_unrealized_pct,
        "breakout_failure_reference": breakout_failure_reference,
        "failed_breakout_detected": failed_breakout,
        "days_since_new_high": days_since_new_high,
        "stale_high_watch_days": stale_high_watch_days,
        "stale_position": stale_position,
        "relative_strength_return_diff": relative_strength_return_diff,
        "relative_strength_window": relative_strength_window,
        "relative_strength_watch_threshold": relative_strength_watch_threshold,
        "weak_relative_strength": weak_relative_strength,
    }

    if position.current_stop is not None and latest_close <= position.current_stop:
        rationale.append("Latest close is at or below the current stop.")
        action = "EXIT CANDIDATE"
    elif regime_passed is False and not above_entry:
        rationale.append("Benchmark regime is weak and the position is below the average entry price.")
        action = "EXIT CANDIDATE"
    elif (
        giveback_pct is not None
        and unrealized_pl_pct > profit_giveback_min_unrealized_pct
        and giveback_pct > profit_giveback_threshold
    ):
        rationale.append(
            "Profitable position has given back "
            f"{giveback_pct:.1%} from the high-water close of {high_water_close:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif failed_breakout:
        rationale.append(
            "Latest close has fallen back below the breakout reference of "
            f"{breakout_failure_reference:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif suggested_stop is not None:
        rationale.append("Trailing-stop logic supports a higher stop without lowering risk control.")
        action = "RAISE STOP"
    elif (
        (distance_to_stop_pct is not None and distance_to_stop_pct <= watch_distance_pct)
        or regime_passed is False
        or not above_entry
        or stale_position
        or weak_relative_strength
    ):
        if distance_to_stop_pct is not None and distance_to_stop_pct <= watch_distance_pct:
            rationale.append("Latest close is close to the current stop.")
        if regime_passed is False:
            rationale.append("Benchmark regime filter is not currently passing.")
        if not above_entry:
            rationale.append("Position is below the average entry price.")
        if stale_position and days_since_new_high is not None:
            rationale.append(
                f"Position has gone {days_since_new_high} trading days without a new closing high."
            )
        if weak_relative_strength and relative_strength_return_diff is not None:
            if relative_strength_window is not None:
                rationale.append(
                    f"Relative strength vs benchmark over {relative_strength_window} trading days is "
                    f"{relative_strength_return_diff:.1%}."
                )
            else:
                rationale.append(
                    f"Relative strength vs benchmark is {relative_strength_return_diff:.1%}."
                )
        action = "WATCH CLOSELY"
    else:
        rationale.append("Position remains above entry and sufficiently above the current stop.")
        action = "HOLD"

    if action == "EXIT CANDIDATE":
        suggested_stop = None
        trailing_stop_candidate = None

    return PortfolioReviewDecision(
        latest_close=latest_close,
        unrealized_pl_pct=unrealized_pl_pct,
        distance_to_stop_pct=distance_to_stop_pct,
        regime_passed=regime_passed,
        above_entry=above_entry,
        suggested_action=action,
        suggested_stop=suggested_stop,
        trailing_stop_candidate=trailing_stop_candidate,
        rationale=tuple(rationale),
        metadata=metadata,
    )


def review_existing_long_position_intraday(
    position: ExistingPosition,
    *,
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

    unrealized_pl_pct = (latest_close / position.average_entry_price) - 1.0
    above_entry = latest_close >= position.average_entry_price
    distance_to_stop_pct: float | None = None
    if position.current_stop is not None:
        distance_to_stop_pct = (latest_close - position.current_stop) / latest_close

    intraday_return_vs_open = (latest_close / session_open) - 1.0
    peak_intraday_return_vs_open = (session_high / session_open) - 1.0
    peak_unrealized_pct = (session_high / position.average_entry_price) - 1.0
    session_high_giveback_pct = max((session_high - latest_close) / session_high, 0.0)
    close_vs_vwap_pct = (
        (latest_close - session_vwap) / session_vwap
        if session_vwap is not None
        else None
    )
    stop_breached_intraday = (
        position.current_stop is not None
        and session_low <= position.current_stop
    )
    failed_intraday_strength = (
        peak_intraday_return_vs_open >= early_strength_threshold
        and (
            (session_vwap is not None and latest_close < session_vwap)
            or latest_close <= session_open
        )
    )
    intraday_momentum_fade = (
        peak_unrealized_pct >= meaningful_profit_pct
        and session_high_giveback_pct >= session_high_giveback_watch_threshold
        and (
            (session_vwap is not None and latest_close < session_vwap)
            or intraday_return_vs_open < 0
        )
    )
    weak_intraday_relative_strength = (
        intraday_relative_strength_diff is not None
        and intraday_relative_strength_diff <= intraday_relative_strength_watch_threshold
    )

    rationale: list[str] = []
    metadata = {
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "latest_low": latest_low,
        "session_vwap": session_vwap,
        "intraday_return_vs_open": intraday_return_vs_open,
        "peak_intraday_return_vs_open": peak_intraday_return_vs_open,
        "peak_unrealized_pct": peak_unrealized_pct,
        "session_high_giveback_pct": session_high_giveback_pct,
        "session_high_giveback_watch_threshold": session_high_giveback_watch_threshold,
        "session_high_giveback_exit_threshold": session_high_giveback_exit_threshold,
        "close_vs_vwap_pct": close_vs_vwap_pct,
        "stop_breached_intraday": stop_breached_intraday,
        "failed_intraday_strength": failed_intraday_strength,
        "intraday_momentum_fade": intraday_momentum_fade,
        "intraday_relative_strength_diff": intraday_relative_strength_diff,
        "intraday_relative_strength_watch_threshold": intraday_relative_strength_watch_threshold,
        "weak_intraday_relative_strength": weak_intraday_relative_strength,
    }

    if stop_breached_intraday:
        rationale.append("An intraday bar traded through the current stop.")
        action = "EXIT CANDIDATE"
    elif (
        peak_unrealized_pct >= meaningful_profit_pct
        and session_high_giveback_pct >= session_high_giveback_exit_threshold
    ):
        rationale.append(
            "Profitable position has given back "
            f"{session_high_giveback_pct:.1%} from the session high of {session_high:.2f}."
        )
        action = "EXIT CANDIDATE"
    elif failed_intraday_strength and session_high_giveback_pct >= session_high_giveback_exit_threshold:
        rationale.append("Strong intraday move failed to hold into the latest bar.")
        if session_vwap is not None and latest_close < session_vwap:
            rationale.append(
                f"Latest close is below session VWAP ({session_vwap:.2f})."
            )
        action = "EXIT CANDIDATE"
    elif intraday_momentum_fade or weak_intraday_relative_strength or failed_intraday_strength:
        if intraday_momentum_fade:
            rationale.append(
                "Profitable position is fading intraday after a strong session high."
            )
        if failed_intraday_strength:
            rationale.append("Strong intraday move is no longer holding the session range.")
        if session_vwap is not None and latest_close < session_vwap:
            rationale.append(
                f"Latest close is below session VWAP ({session_vwap:.2f})."
            )
        if weak_intraday_relative_strength and intraday_relative_strength_diff is not None:
            rationale.append(
                "Intraday return vs benchmark is "
                f"{intraday_relative_strength_diff:.1%}."
            )
        action = "WATCH CLOSELY"
    else:
        rationale.append("Intraday structure remains healthy and no stop or fade signal is active.")
        action = "HOLD"

    return PortfolioReviewDecision(
        latest_close=latest_close,
        unrealized_pl_pct=unrealized_pl_pct,
        distance_to_stop_pct=distance_to_stop_pct,
        regime_passed=None,
        above_entry=above_entry,
        suggested_action=action,
        suggested_stop=None,
        trailing_stop_candidate=None,
        rationale=tuple(rationale),
        metadata=metadata,
    )


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
    current_positions: Sequence[ExistingPosition] = (),
    current_drawdown: float = 0.0,
    stop_price: float | None = None,
) -> RiskAssessedCandidate:
    """Combine sizing and portfolio rules into one risk-assessed entry candidate."""

    existing_position = _find_existing_position(current_positions, signal.symbol)
    resolved_stop_price = signal.stop_hint if stop_price is None else stop_price
    adjusted_risk_per_trade = apply_drawdown_risk_adjustment(
        base_risk_per_trade,
        current_drawdown=current_drawdown,
        threshold=constraints.drawdown_risk_reduction_threshold,
        reduction_factor=constraints.drawdown_risk_reduction_factor,
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
        )

    sizing = size_position(
        current_equity=current_equity,
        risk_per_trade=adjusted_risk_per_trade,
        entry_price=signal.entry_price_hint,
        stop_price=resolved_stop_price,
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

    approved = sizing.is_valid and not rejection_reasons
    return RiskAssessedCandidate(
        signal=signal,
        entry_price=signal.entry_price_hint,
        stop_price=resolved_stop_price,
        adjusted_risk_per_trade=adjusted_risk_per_trade,
        sizing=sizing,
        approved=approved,
        rejection_reasons=tuple(rejection_reasons),
        existing_position=existing_position,
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
