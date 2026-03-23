"""Typed feature vectors and builders for decision workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Mapping

import pandas as pd

from bot.data.earnings import EarningsRiskContext
from bot.data.intraday_state_journal import IntradayTrajectoryFeatures
from bot.data.position_trajectory import PositionTrajectoryFeatures
from bot.data.sector_context import SectorFeatureContext
from bot.strategy.signal_models import StrategySignal

MAX_MARKET_BREADTH_BAR_AGE_DAYS = 7


@dataclass(frozen=True)
class MarketContext:
    """Shared market-level context consumed by decision workflows."""

    as_of_date: date
    benchmark_symbol: str | None = None
    regime_filter_mode: str | None = None
    regime_passed: bool | None = None
    benchmark_return: float | None = None
    benchmark_intraday_return_vs_open: float | None = None
    market_breadth_pct_above_200ma: float | None = None
    market_breadth_pct_above_50ma: float | None = None
    market_breadth_state: str | None = None
    vix_close: float | None = None
    vix_sma_short: float | None = None
    vix_sma_long: float | None = None
    volatility_regime_state: str | None = None
    volatility_regime_risk_off: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the market context."""

        return {
            "as_of_date": self.as_of_date.isoformat(),
            "benchmark_symbol": self.benchmark_symbol,
            "regime_filter_mode": self.regime_filter_mode,
            "regime_passed": self.regime_passed,
            "benchmark_return": self.benchmark_return,
            "benchmark_intraday_return_vs_open": self.benchmark_intraday_return_vs_open,
            "market_breadth_pct_above_200ma": self.market_breadth_pct_above_200ma,
            "market_breadth_pct_above_50ma": self.market_breadth_pct_above_50ma,
            "market_breadth_state": self.market_breadth_state,
            "vix_close": self.vix_close,
            "vix_sma_short": self.vix_sma_short,
            "vix_sma_long": self.vix_sma_long,
            "volatility_regime_state": self.volatility_regime_state,
            "volatility_regime_risk_off": self.volatility_regime_risk_off,
        }


@dataclass(frozen=True)
class CandidateFeatures:
    """Derived candidate-level feature vector for entry assessment and ranking."""

    symbol: str
    as_of_date: date
    preset_name: str | None = None
    parameter_id: str | None = None
    strategy_name: str | None = None
    entry_reason: str | None = None
    entry_price_hint: float | None = None
    stop_hint: float | None = None
    breakout_strength: float | None = None
    relative_volume: float | None = None
    prior_high: float | None = None
    regime_passed: bool | None = None
    benchmark_symbol: str | None = None
    sector_name: str | None = None
    industry_name: str | None = None
    sector_etf_symbol: str | None = None
    sector_regime_passed: bool | None = None
    relative_strength_vs_sector: float | None = None
    sector_relative_strength_window: int | None = None
    earnings_date: date | None = None
    earnings_days_away: int | None = None
    is_earnings_risk: bool = False
    setup_quality_score: float | None = None
    setup_persistence_days: int | None = None
    days_approved: int | None = None
    repeated_high_quality_signal: bool = False
    market_context: MarketContext | None = None


@dataclass(frozen=True)
class PositionFeatures:
    """Derived held-position feature vector for daily and intraday review."""

    symbol: str
    as_of_date: date
    average_entry_price: float
    latest_close: float
    current_stop: float | None
    unrealized_pl_pct: float
    above_entry: bool
    distance_to_stop_pct: float | None
    regime_passed: bool | None
    trailing_stop_candidate: float | None = None
    high_water_close: float | None = None
    giveback_pct: float | None = None
    breakout_failure_reference: float | None = None
    failed_breakout: bool = False
    days_since_new_high: int | None = None
    stale_position: bool = False
    relative_strength_return_diff: float | None = None
    relative_strength_window: int | None = None
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
    sector_name: str | None = None
    industry_name: str | None = None
    sector_etf_symbol: str | None = None
    sector_regime_passed: bool | None = None
    relative_strength_vs_sector: float | None = None
    sector_relative_strength_window: int | None = None
    lagging_sector: bool = False
    earnings_date: date | None = None
    earnings_days_away: int | None = None
    is_earnings_risk: bool = False
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    latest_low: float | None = None
    session_vwap: float | None = None
    session_high_giveback_pct: float | None = None
    intraday_return_vs_open: float | None = None
    intraday_return_vs_benchmark: float | None = None
    peak_intraday_return_vs_open: float | None = None
    peak_unrealized_pct: float | None = None
    close_vs_vwap_pct: float | None = None
    failed_intraday_strength: bool = False
    stop_breached_intraday: bool = False
    intraday_momentum_fade: bool = False
    weak_intraday_relative_strength: bool = False
    stacked_intraday_weakness: bool = False
    intraday_observation_count: int = 0
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
    market_context: MarketContext | None = None


def build_market_context(
    *,
    as_of_date: date,
    benchmark_symbol: str | None = None,
    regime_filter_mode: str | None = None,
    regime_passed: bool | None = None,
    benchmark_return: float | None = None,
    benchmark_intraday_return_vs_open: float | None = None,
    market_breadth_pct_above_200ma: float | None = None,
    market_breadth_pct_above_50ma: float | None = None,
    market_breadth_state: str | None = None,
    vix_close: float | None = None,
    vix_sma_short: float | None = None,
    vix_sma_long: float | None = None,
    volatility_regime_state: str | None = None,
    volatility_regime_risk_off: bool | None = None,
) -> MarketContext:
    """Build a small typed market context for downstream decision logic."""

    return MarketContext(
        as_of_date=as_of_date,
        benchmark_symbol=benchmark_symbol,
        regime_filter_mode=regime_filter_mode,
        regime_passed=regime_passed,
        benchmark_return=benchmark_return,
        benchmark_intraday_return_vs_open=benchmark_intraday_return_vs_open,
        market_breadth_pct_above_200ma=market_breadth_pct_above_200ma,
        market_breadth_pct_above_50ma=market_breadth_pct_above_50ma,
        market_breadth_state=market_breadth_state,
        vix_close=vix_close,
        vix_sma_short=vix_sma_short,
        vix_sma_long=vix_sma_long,
        volatility_regime_state=volatility_regime_state,
        volatility_regime_risk_off=volatility_regime_risk_off,
    )


def compute_market_breadth_from_frames(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    as_of_date: date,
    long_window: int = 200,
    short_window: int = 50,
    entry_floor_200ma: float | None = None,
) -> dict[str, float | str | None]:
    """Compute deterministic breadth metrics from the concrete workflow universe."""

    pct_above_200ma = _pct_above_moving_average(
        symbol_frames,
        as_of_date=as_of_date,
        window=long_window,
    )
    pct_above_50ma = _pct_above_moving_average(
        symbol_frames,
        as_of_date=as_of_date,
        window=short_window,
    )
    return {
        "market_breadth_pct_above_200ma": pct_above_200ma,
        "market_breadth_pct_above_50ma": pct_above_50ma,
        "market_breadth_state": _market_breadth_state(
            pct_above_200ma,
            entry_floor_200ma=entry_floor_200ma,
        ),
    }


def market_context_metadata(
    market_context: MarketContext | None,
    *,
    market_breadth_entry_floor_200ma: float | None = None,
    vix_caution_threshold: float | None = None,
    vix_entry_block_threshold: float | None = None,
) -> dict[str, Any]:
    """Serialize market-context fields into signal/report metadata."""

    if market_context is None:
        return {}

    metadata: dict[str, Any] = {}
    if market_context.market_breadth_pct_above_200ma is not None:
        metadata["market_breadth_pct_above_200ma"] = (
            market_context.market_breadth_pct_above_200ma
        )
    if market_context.market_breadth_pct_above_50ma is not None:
        metadata["market_breadth_pct_above_50ma"] = (
            market_context.market_breadth_pct_above_50ma
        )
    if market_context.market_breadth_state is not None:
        metadata["market_breadth_state"] = market_context.market_breadth_state
    if market_context.vix_close is not None:
        metadata["vix_close"] = market_context.vix_close
    if market_context.vix_sma_short is not None:
        metadata["vix_sma_short"] = market_context.vix_sma_short
    if market_context.vix_sma_long is not None:
        metadata["vix_sma_long"] = market_context.vix_sma_long
    if market_context.volatility_regime_state is not None:
        metadata["volatility_regime_state"] = (
            market_context.volatility_regime_state
        )
    if market_context.volatility_regime_risk_off is not None:
        metadata["volatility_regime_risk_off"] = (
            market_context.volatility_regime_risk_off
        )
    if market_breadth_entry_floor_200ma is not None:
        metadata["market_breadth_entry_floor_200ma"] = market_breadth_entry_floor_200ma
    if vix_caution_threshold is not None:
        metadata["vix_caution_threshold"] = vix_caution_threshold
    if vix_entry_block_threshold is not None:
        metadata["vix_entry_block_threshold"] = vix_entry_block_threshold
    return metadata


def build_candidate_features(
    signal: StrategySignal,
    *,
    as_of_date: date,
    earnings_context: EarningsRiskContext | None = None,
    sector_context: SectorFeatureContext | None = None,
    market_context: MarketContext | None = None,
    stop_hint: float | None = None,
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    is_earnings_risk: bool | None = None,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
) -> CandidateFeatures:
    """Build a typed candidate feature vector from one evaluated signal."""

    prior_high = _optional_float(signal.metadata.get("prior_high"))
    entry_price_hint = _optional_float(signal.entry_price_hint)
    resolved_stop_hint = (
        _optional_float(stop_hint)
        if stop_hint is not None
        else _optional_float(signal.stop_hint)
    )
    breakout_strength = None
    if (
        prior_high is not None
        and prior_high > 0
        and entry_price_hint is not None
        and entry_price_hint > 0
    ):
        breakout_strength = max((entry_price_hint - prior_high) / prior_high, 0.0)

    resolved_earnings_date = (
        earnings_context.earnings_date
        if earnings_context is not None
        else earnings_date or _optional_date(signal.metadata.get("earnings_date"))
    )
    resolved_earnings_days_away = (
        earnings_context.earnings_days_away
        if earnings_context is not None
        else (
            earnings_days_away
            if earnings_days_away is not None
            else _optional_int(signal.metadata.get("earnings_days_away"))
        )
    )
    resolved_is_earnings_risk = (
        earnings_context.is_earnings_risk
        if earnings_context is not None
        else (
            is_earnings_risk
            if is_earnings_risk is not None
            else bool(signal.metadata.get("is_earnings_risk"))
        )
    )

    resolved_sector_name = (
        sector_context.sector_name
        if sector_context is not None
        else sector_name or _optional_text(signal.metadata.get("sector_name"))
    )
    resolved_industry_name = (
        sector_context.industry_name
        if sector_context is not None
        else industry_name or _optional_text(signal.metadata.get("industry_name"))
    )
    resolved_sector_etf_symbol = (
        sector_context.sector_etf_symbol
        if sector_context is not None
        else sector_etf_symbol or _optional_text(signal.metadata.get("sector_etf_symbol"))
    )
    resolved_sector_regime_passed = (
        sector_context.sector_regime_passed
        if sector_context is not None
        else (
            sector_regime_passed
            if sector_regime_passed is not None
            else _optional_bool(signal.metadata.get("sector_regime_passed"))
        )
    )
    resolved_relative_strength_vs_sector = (
        sector_context.relative_strength_vs_sector
        if sector_context is not None
        else (
            relative_strength_vs_sector
            if relative_strength_vs_sector is not None
            else _optional_float(signal.metadata.get("relative_strength_vs_sector"))
        )
    )
    resolved_sector_relative_strength_window = (
        sector_context.relative_strength_window
        if sector_context is not None
        else (
            sector_relative_strength_window
            if sector_relative_strength_window is not None
            else _optional_int(signal.metadata.get("sector_relative_strength_window"))
        )
    )

    return CandidateFeatures(
        symbol=signal.symbol,
        as_of_date=as_of_date,
        preset_name=_optional_text(signal.metadata.get("preset_name")),
        parameter_id=_optional_text(signal.metadata.get("parameter_id")),
        strategy_name=signal.strategy_name,
        entry_reason=signal.entry_reason,
        entry_price_hint=entry_price_hint,
        stop_hint=resolved_stop_hint,
        breakout_strength=breakout_strength,
        relative_volume=_optional_float(signal.metadata.get("relative_volume")),
        prior_high=prior_high,
        regime_passed=_optional_bool(signal.metadata.get("regime_passed")),
        benchmark_symbol=(
            _optional_text(signal.metadata.get("benchmark_symbol"))
            or (market_context.benchmark_symbol if market_context is not None else None)
        ),
        sector_name=resolved_sector_name,
        industry_name=resolved_industry_name,
        sector_etf_symbol=resolved_sector_etf_symbol,
        sector_regime_passed=resolved_sector_regime_passed,
        relative_strength_vs_sector=resolved_relative_strength_vs_sector,
        sector_relative_strength_window=resolved_sector_relative_strength_window,
        earnings_date=resolved_earnings_date,
        earnings_days_away=resolved_earnings_days_away,
        is_earnings_risk=resolved_is_earnings_risk,
        setup_quality_score=_optional_float(signal.metadata.get("setup_quality_score")),
        setup_persistence_days=_optional_int(signal.metadata.get("setup_persistence_days")),
        days_approved=_optional_int(signal.metadata.get("days_approved")),
        repeated_high_quality_signal=bool(signal.metadata.get("repeated_high_quality_signal")),
        market_context=market_context,
    )


def build_position_features(
    *,
    symbol: str,
    as_of_date: date,
    average_entry_price: float,
    latest_close: float,
    current_stop: float | None,
    regime_passed: bool | None,
    trailing_stop_candidate: float | None = None,
    high_water_close: float | None = None,
    breakout_failure_reference: float | None = None,
    days_since_new_high: int | None = None,
    stale_high_watch_days: int,
    relative_strength_return_diff: float | None = None,
    relative_strength_window: int | None = None,
    earnings_context: EarningsRiskContext | None = None,
    earnings_watch_days: int = 7,
    sector_context: SectorFeatureContext | None = None,
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    is_earnings_risk: bool | None = None,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
    sector_relative_strength_watch_threshold: float = -0.05,
    market_context: MarketContext | None = None,
) -> PositionFeatures:
    """Build an end-of-day held-position feature vector."""

    unrealized_pl_pct = (latest_close / average_entry_price) - 1.0
    above_entry = latest_close >= average_entry_price
    distance_to_stop_pct = None
    if current_stop is not None:
        distance_to_stop_pct = (latest_close - current_stop) / latest_close
    giveback_pct = None
    if high_water_close is not None and high_water_close > 0:
        giveback_pct = max((high_water_close - latest_close) / high_water_close, 0.0)
    failed_breakout = (
        breakout_failure_reference is not None
        and latest_close < breakout_failure_reference
    )
    stale_position = (
        days_since_new_high is not None
        and days_since_new_high >= stale_high_watch_days
    )
    resolved_sector_name = (
        sector_context.sector_name if sector_context is not None else sector_name
    )
    resolved_industry_name = (
        sector_context.industry_name if sector_context is not None else industry_name
    )
    resolved_sector_etf_symbol = (
        sector_context.sector_etf_symbol if sector_context is not None else sector_etf_symbol
    )
    resolved_sector_regime_passed = (
        sector_context.sector_regime_passed
        if sector_context is not None
        else sector_regime_passed
    )
    resolved_relative_strength_vs_sector = (
        sector_context.relative_strength_vs_sector
        if sector_context is not None
        else relative_strength_vs_sector
    )
    resolved_sector_relative_strength_window = (
        sector_context.relative_strength_window
        if sector_context is not None
        else sector_relative_strength_window
    )
    resolved_earnings_date = (
        earnings_context.earnings_date if earnings_context is not None else earnings_date
    )
    resolved_earnings_days_away = (
        earnings_context.earnings_days_away
        if earnings_context is not None
        else earnings_days_away
    )
    resolved_is_earnings_risk = (
        bool(
            earnings_context is not None
            and earnings_context.earnings_days_away is not None
            and earnings_context.earnings_days_away <= earnings_watch_days
        )
        if earnings_context is not None
        else bool(
            is_earnings_risk
            if is_earnings_risk is not None
            else (
                resolved_earnings_days_away is not None
                and resolved_earnings_days_away <= earnings_watch_days
            )
        )
    )
    lagging_sector = False
    if resolved_relative_strength_vs_sector is not None:
        lagging_sector = (
            resolved_relative_strength_vs_sector <= sector_relative_strength_watch_threshold
        )

    return PositionFeatures(
        symbol=symbol,
        as_of_date=as_of_date,
        average_entry_price=average_entry_price,
        latest_close=latest_close,
        current_stop=current_stop,
        unrealized_pl_pct=unrealized_pl_pct,
        above_entry=above_entry,
        distance_to_stop_pct=distance_to_stop_pct,
        regime_passed=regime_passed,
        trailing_stop_candidate=trailing_stop_candidate,
        high_water_close=high_water_close,
        giveback_pct=giveback_pct,
        breakout_failure_reference=breakout_failure_reference,
        failed_breakout=failed_breakout,
        days_since_new_high=days_since_new_high,
        stale_position=stale_position,
        relative_strength_return_diff=relative_strength_return_diff,
        relative_strength_window=relative_strength_window,
        sector_name=resolved_sector_name,
        industry_name=resolved_industry_name,
        sector_etf_symbol=resolved_sector_etf_symbol,
        sector_regime_passed=resolved_sector_regime_passed,
        relative_strength_vs_sector=resolved_relative_strength_vs_sector,
        sector_relative_strength_window=resolved_sector_relative_strength_window,
        lagging_sector=lagging_sector,
        earnings_date=resolved_earnings_date,
        earnings_days_away=resolved_earnings_days_away,
        is_earnings_risk=resolved_is_earnings_risk,
        market_context=market_context,
    )


def build_intraday_position_features(
    *,
    symbol: str,
    as_of_date: date,
    average_entry_price: float,
    current_stop: float | None,
    session_open: float,
    session_high: float,
    session_low: float,
    latest_close: float,
    latest_low: float,
    session_vwap: float | None,
    intraday_return_vs_benchmark: float | None,
    earnings_context: EarningsRiskContext | None = None,
    earnings_watch_days: int = 7,
    sector_context: SectorFeatureContext | None = None,
    earnings_date: date | None = None,
    earnings_days_away: int | None = None,
    is_earnings_risk: bool | None = None,
    sector_name: str | None = None,
    industry_name: str | None = None,
    sector_etf_symbol: str | None = None,
    sector_regime_passed: bool | None = None,
    relative_strength_vs_sector: float | None = None,
    sector_relative_strength_window: int | None = None,
    early_strength_threshold: float = 0.03,
    meaningful_profit_pct: float = 0.05,
    session_high_giveback_watch_threshold: float = 0.04,
    intraday_relative_strength_watch_threshold: float = -0.02,
    sector_relative_strength_watch_threshold: float = -0.05,
    market_context: MarketContext | None = None,
) -> PositionFeatures:
    """Build an intraday held-position feature vector."""

    unrealized_pl_pct = (latest_close / average_entry_price) - 1.0
    above_entry = latest_close >= average_entry_price
    distance_to_stop_pct = None
    if current_stop is not None:
        distance_to_stop_pct = (latest_close - current_stop) / latest_close

    intraday_return_vs_open = (latest_close / session_open) - 1.0
    peak_intraday_return_vs_open = (session_high / session_open) - 1.0
    peak_unrealized_pct = (session_high / average_entry_price) - 1.0
    session_high_giveback_pct = max((session_high - latest_close) / session_high, 0.0)
    close_vs_vwap_pct = (
        (latest_close - session_vwap) / session_vwap
        if session_vwap is not None
        else None
    )
    stop_breached_intraday = (
        current_stop is not None
        and session_low <= current_stop
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
        intraday_return_vs_benchmark is not None
        and intraday_return_vs_benchmark <= intraday_relative_strength_watch_threshold
    )
    stacked_intraday_weakness = (
        close_vs_vwap_pct is not None
        and close_vs_vwap_pct < 0
        and session_high_giveback_pct >= meaningful_profit_pct
        and weak_intraday_relative_strength
    )
    resolved_sector_name = (
        sector_context.sector_name if sector_context is not None else sector_name
    )
    resolved_industry_name = (
        sector_context.industry_name if sector_context is not None else industry_name
    )
    resolved_sector_etf_symbol = (
        sector_context.sector_etf_symbol if sector_context is not None else sector_etf_symbol
    )
    resolved_sector_regime_passed = (
        sector_context.sector_regime_passed
        if sector_context is not None
        else sector_regime_passed
    )
    resolved_relative_strength_vs_sector = (
        sector_context.relative_strength_vs_sector
        if sector_context is not None
        else relative_strength_vs_sector
    )
    resolved_sector_relative_strength_window = (
        sector_context.relative_strength_window
        if sector_context is not None
        else sector_relative_strength_window
    )
    resolved_earnings_date = (
        earnings_context.earnings_date if earnings_context is not None else earnings_date
    )
    resolved_earnings_days_away = (
        earnings_context.earnings_days_away
        if earnings_context is not None
        else earnings_days_away
    )
    resolved_is_earnings_risk = (
        bool(
            earnings_context is not None
            and earnings_context.earnings_days_away is not None
            and earnings_context.earnings_days_away <= earnings_watch_days
        )
        if earnings_context is not None
        else bool(
            is_earnings_risk
            if is_earnings_risk is not None
            else (
                resolved_earnings_days_away is not None
                and resolved_earnings_days_away <= earnings_watch_days
            )
        )
    )
    lagging_sector = False
    if resolved_relative_strength_vs_sector is not None:
        lagging_sector = (
            resolved_relative_strength_vs_sector <= sector_relative_strength_watch_threshold
        )

    return PositionFeatures(
        symbol=symbol,
        as_of_date=as_of_date,
        average_entry_price=average_entry_price,
        latest_close=latest_close,
        current_stop=current_stop,
        unrealized_pl_pct=unrealized_pl_pct,
        above_entry=above_entry,
        distance_to_stop_pct=distance_to_stop_pct,
        regime_passed=market_context.regime_passed if market_context is not None else None,
        earnings_date=resolved_earnings_date,
        earnings_days_away=resolved_earnings_days_away,
        is_earnings_risk=resolved_is_earnings_risk,
        sector_name=resolved_sector_name,
        industry_name=resolved_industry_name,
        sector_etf_symbol=resolved_sector_etf_symbol,
        sector_regime_passed=resolved_sector_regime_passed,
        relative_strength_vs_sector=resolved_relative_strength_vs_sector,
        sector_relative_strength_window=resolved_sector_relative_strength_window,
        lagging_sector=lagging_sector,
        session_open=session_open,
        session_high=session_high,
        session_low=session_low,
        latest_low=latest_low,
        session_vwap=session_vwap,
        session_high_giveback_pct=session_high_giveback_pct,
        intraday_return_vs_open=intraday_return_vs_open,
        intraday_return_vs_benchmark=intraday_return_vs_benchmark,
        peak_intraday_return_vs_open=peak_intraday_return_vs_open,
        peak_unrealized_pct=peak_unrealized_pct,
        close_vs_vwap_pct=close_vs_vwap_pct,
        failed_intraday_strength=failed_intraday_strength,
        stop_breached_intraday=stop_breached_intraday,
        intraday_momentum_fade=intraday_momentum_fade,
        weak_intraday_relative_strength=weak_intraday_relative_strength,
        stacked_intraday_weakness=stacked_intraday_weakness,
        market_context=market_context,
    )


def apply_intraday_trajectory_features(
    features: PositionFeatures,
    trajectory_features: IntradayTrajectoryFeatures | None,
) -> PositionFeatures:
    """Return a copy of intraday features enriched with journal-derived trajectory state."""

    if trajectory_features is None:
        return features

    return replace(
        features,
        intraday_observation_count=trajectory_features.observation_count,
        consecutive_polls_below_vwap=trajectory_features.consecutive_polls_below_vwap,
        consecutive_polls_weak_relative_strength=(
            trajectory_features.consecutive_polls_weak_relative_strength
        ),
        intraday_pressure_persistence_count=(
            trajectory_features.intraday_pressure_persistence_count
        ),
        max_session_high_giveback_seen=trajectory_features.max_session_high_giveback_seen,
        worsening_session_high_giveback=trajectory_features.worsening_session_high_giveback,
        giveback_worsening_polls=trajectory_features.giveback_worsening_polls,
        giveback_worsening_from_pct=trajectory_features.giveback_worsening_from_pct,
        repeated_watch_closely_count=trajectory_features.repeated_watch_closely_count,
        weakening_all_session=trajectory_features.weakening_all_session,
        recovered_after_weakness=trajectory_features.recovered_after_weakness,
    )


def apply_position_trajectory_features(
    features: PositionFeatures,
    trajectory_features: PositionTrajectoryFeatures | None,
) -> PositionFeatures:
    """Return a copy of daily position features enriched with multi-day state."""

    if trajectory_features is None:
        return features

    return replace(
        features,
        days_in_position_state=trajectory_features.days_in_position_state,
        consecutive_days_above_entry=trajectory_features.consecutive_days_above_entry,
        consecutive_days_below_entry=trajectory_features.consecutive_days_below_entry,
        consecutive_watch_closely_days=trajectory_features.consecutive_watch_closely_days,
        consecutive_hold_days=trajectory_features.consecutive_hold_days,
        consecutive_stale_position_days=trajectory_features.consecutive_stale_position_days,
        consecutive_weak_relative_strength_days=(
            trajectory_features.consecutive_weak_relative_strength_days
        ),
        consecutive_weak_position_days=trajectory_features.consecutive_weak_position_days,
        repeated_weak_position=trajectory_features.repeated_weak_position,
        persistent_underperformance=trajectory_features.persistent_underperformance,
        recovery_after_multi_day_weakness=(
            trajectory_features.recovery_after_multi_day_weakness
        ),
    )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _pct_above_moving_average(
    symbol_frames: Mapping[str, pd.DataFrame],
    *,
    as_of_date: date,
    window: int,
) -> float | None:
    eligible_count = 0
    above_count = 0
    for frame in symbol_frames.values():
        if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
            continue

        prepared = frame.sort_values("date", kind="stable").copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
        prepared = prepared.loc[
            prepared["date"].notna()
            & prepared["close"].notna()
            & (prepared["date"].dt.date <= as_of_date)
        ]
        if len(prepared) < window:
            continue
        latest_bar_date = prepared["date"].iloc[-1]
        if (as_of_date - latest_bar_date.date()).days > MAX_MARKET_BREADTH_BAR_AGE_DAYS:
            continue

        moving_average = prepared["close"].rolling(window).mean().iloc[-1]
        latest_close = float(prepared["close"].iloc[-1])
        if pd.isna(moving_average):
            continue

        eligible_count += 1
        if latest_close > float(moving_average):
            above_count += 1

    if eligible_count == 0:
        return None
    return above_count / eligible_count


def _market_breadth_state(
    pct_above_200ma: float | None,
    *,
    entry_floor_200ma: float | None,
) -> str | None:
    if pct_above_200ma is None or entry_floor_200ma is None:
        return None

    weak_floor = entry_floor_200ma
    strong_floor = max(0.60, weak_floor + 0.20)
    if pct_above_200ma < weak_floor:
        return "weak"
    if pct_above_200ma >= strong_floor:
        return "strong"
    return "neutral"
