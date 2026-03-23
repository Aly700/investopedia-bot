from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bot.config import UniverseConfig, load_app_config
from bot.data.candidate_journal import (
    CandidateJournalEntry,
    CandidateJournalObservation,
    build_candidate_memory_features,
    default_candidate_score_journal_path,
    load_candidate_score_journal,
    update_candidate_score_journal,
    write_candidate_score_journal,
)
from bot.data.earnings import (
    EarningsCalendarProvider,
    build_earnings_risk_contexts,
    trading_days_until,
)
from bot.data.intraday_state_journal import (
    IntradayStateObservation,
    build_intraday_trajectory_features,
    default_intraday_state_journal_path,
    load_intraday_state_journal,
    preview_intraday_trajectory_features,
    update_intraday_state_journal,
    write_intraday_state_journal,
)
from bot.data.portfolio_heat import (
    PortfolioHoldingExposure,
    build_portfolio_heat_context,
    project_portfolio_heat_context,
)
from bot.data.position_trajectory import (
    PositionTrajectoryObservation,
    build_position_trajectory_features,
    default_position_trajectory_journal_path,
    load_position_trajectory_journal,
    update_position_trajectory_journal,
    write_position_trajectory_journal,
)
from bot.data.sector_context import (
    SymbolSectorClassification,
    build_sector_feature_contexts,
)
from bot.data.trade_feedback import (
    TradeFeedbackEvent,
    TradeOutcomeSnapshot,
    append_trade_feedback_events,
    compute_trade_outcome_snapshot,
    default_trade_feedback_log_path,
    load_trade_feedback_events,
    summarize_trade_feedback_events,
)
from bot.data.volatility_context import (
    MAX_VOLATILITY_CONTEXT_BAR_AGE_DAYS,
    build_volatility_regime_context,
)
from bot.data.normalize import (
    DAILY_BAR_COLUMNS,
    INTRADAY_BAR_COLUMNS,
    empty_daily_bar_frame,
    empty_intraday_bar_frame,
    filter_intraday_bars_by_session_date,
    normalize_daily_bars,
    normalize_intraday_bars,
)
from bot.data.providers import (
    DailyBarProvider,
    create_daily_bar_provider,
    load_provider_environment,
)
from bot.data.universe import UniverseBuilder, load_candidate_symbols


class FakeDailyBarProvider(DailyBarProvider):
    provider_name = "fake"

    def __init__(
        self,
        *,
        frames_by_symbol: dict[str, pd.DataFrame],
        intraday_frames_by_symbol: dict[str, pd.DataFrame] | None = None,
        cache_dir: Path,
    ) -> None:
        super().__init__(api_key="test-key", cache_dir=cache_dir)
        self.frames_by_symbol = frames_by_symbol
        self.intraday_frames_by_symbol = intraday_frames_by_symbol or {}
        self.fetch_count = 0
        self.intraday_fetch_count = 0
        self.intraday_fetch_count_by_symbol: dict[str, int] = {}

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        self.fetch_count += 1
        return self.frames_by_symbol.get(symbol, empty_daily_bar_frame()).copy()

    def _fetch_intraday_bars_from_api(
        self,
        symbol: str,
        session_date: date,
        interval_minutes: int,
    ) -> pd.DataFrame:
        self.intraday_fetch_count += 1
        fetch_index = self.intraday_fetch_count_by_symbol.get(symbol, 0)
        self.intraday_fetch_count_by_symbol[symbol] = fetch_index + 1

        raw_frame = self.intraday_frames_by_symbol.get(symbol, empty_intraday_bar_frame())
        if isinstance(raw_frame, list):
            if not raw_frame:
                return empty_intraday_bar_frame()
            return raw_frame[min(fetch_index, len(raw_frame) - 1)].copy()
        return raw_frame.copy()


class FakeEarningsCalendarProvider(EarningsCalendarProvider):
    provider_name = "fake"

    def __init__(
        self,
        *,
        records_by_window: dict[tuple[date, date], list[dict[str, object]]],
        cache_dir: Path,
    ) -> None:
        super().__init__(api_key="test-key", cache_dir=cache_dir)
        self.records_by_window = records_by_window
        self.fetch_count = 0

    def _fetch_upcoming_earnings_from_api(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, object]]:
        self.fetch_count += 1
        return [
            dict(record)
            for record in self.records_by_window.get((start_date, end_date), [])
        ]


def test_normalize_daily_bars_enforces_schema_and_ordering() -> None:
    raw_frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-03", "2024-01-02", "2024-01-02", "bad-date"],
            "o": ["11.0", "10.0", "10.5", "12.0"],
            "h": ["12.0", "11.0", "11.5", "13.0"],
            "l": ["10.5", "9.5", "10.0", "11.0"],
            "c": ["11.5", "10.5", "10.7", "12.5"],
            "v": ["1500", "1000", "1100", "900"],
        }
    )

    normalized = normalize_daily_bars(
        raw_frame,
        symbol="aapl",
        column_mapping={
            "timestamp": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        },
    )

    assert list(normalized.columns) == list(DAILY_BAR_COLUMNS)
    assert normalized["symbol"].tolist() == ["AAPL", "AAPL"]
    assert normalized["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert normalized["close"].tolist() == [10.7, 11.5]
    assert normalized["volume"].dtype == "int64"


def test_normalize_daily_bars_returns_empty_schema_for_empty_input() -> None:
    normalized = normalize_daily_bars(pd.DataFrame(), symbol="MSFT")

    assert list(normalized.columns) == list(DAILY_BAR_COLUMNS)
    assert normalized.empty


def test_provider_cache_avoids_duplicate_remote_fetches(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.5, 10.5],
            "close": [10.5, 11.5],
            "volume": [1000, 1500],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    provider = FakeDailyBarProvider(frames_by_symbol={"AAPL": frame}, cache_dir=tmp_path)

    first = provider.fetch_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    second = provider.fetch_daily_bars("AAPL", date(2024, 1, 2), date(2024, 1, 3))

    assert provider.fetch_count == 1
    assert_frame_equal(first, second)
    assert (tmp_path / "fake" / "AAPL_2024-01-02_2024-01-03.csv").exists()


def test_provider_intraday_cache_avoids_duplicate_remote_fetches(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-03 14:30:00", "2024-01-03 14:45:00"]),
            "open": [10.0, 10.5],
            "high": [10.6, 10.8],
            "low": [9.9, 10.4],
            "close": [10.5, 10.7],
            "volume": [1000, 1200],
            "vwap": [10.3, 10.5],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    provider = FakeDailyBarProvider(
        frames_by_symbol={},
        intraday_frames_by_symbol={"AAPL": frame},
        cache_dir=tmp_path,
    )

    first = provider.fetch_intraday_bars(
        "AAPL",
        date(2024, 1, 3),
        interval_minutes=15,
        refresh_cache=False,
    )
    second = provider.fetch_intraday_bars(
        "AAPL",
        date(2024, 1, 3),
        interval_minutes=15,
        refresh_cache=False,
    )

    assert provider.intraday_fetch_count == 1
    assert list(first.columns) == list(INTRADAY_BAR_COLUMNS)
    assert_frame_equal(first, second)
    assert (
        tmp_path / "fake" / "AAPL_intraday_2024-01-03_15min.csv"
    ).exists()


def test_provider_intraday_fetch_bypasses_cache_by_default(tmp_path: Path) -> None:
    first_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-03 14:30:00", "2024-01-03 14:45:00"]),
            "open": [10.0, 10.5],
            "high": [10.6, 10.8],
            "low": [9.9, 10.4],
            "close": [10.5, 10.7],
            "volume": [1000, 1200],
            "vwap": [10.3, 10.5],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    second_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-03 14:30:00", "2024-01-03 14:45:00"]),
            "open": [11.0, 11.5],
            "high": [11.6, 11.8],
            "low": [10.9, 11.4],
            "close": [11.5, 11.7],
            "volume": [1500, 1800],
            "vwap": [11.3, 11.5],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    provider = FakeDailyBarProvider(
        frames_by_symbol={},
        intraday_frames_by_symbol={"AAPL": [first_frame, second_frame]},
        cache_dir=tmp_path,
    )

    first = provider.fetch_intraday_bars("AAPL", date(2024, 1, 3), interval_minutes=15)
    second = provider.fetch_intraday_bars("AAPL", date(2024, 1, 3), interval_minutes=15)

    assert provider.intraday_fetch_count == 2
    assert first["close"].tolist() == [10.5, 10.7]
    assert second["close"].tolist() == [11.5, 11.7]


def test_build_volatility_regime_context_uses_vix_alias_fallback(tmp_path: Path) -> None:
    dates = pd.date_range("2023-12-12", periods=25, freq="D")
    vix_frame = pd.DataFrame(
        {
            "date": dates,
            "open": [24.0] * 25,
            "high": [25.0] * 25,
            "low": [23.0] * 25,
            "close": [24.0] * 24 + [26.4],
            "volume": [0] * 25,
            "symbol": ["VIX"] * 25,
        }
    )
    provider = FakeDailyBarProvider(frames_by_symbol={"VIX": vix_frame}, cache_dir=tmp_path)

    context = build_volatility_regime_context(
        provider=provider,
        as_of_date=date(2024, 1, 5),
        history_start=date(2023, 12, 1),
        caution_threshold=25.0,
        entry_block_threshold=30.0,
    )

    assert context["vix_close"] == pytest.approx(26.4)
    assert context["volatility_regime_state"] == "elevated"
    assert context["volatility_regime_risk_off"] is False
    assert context["vix_sma_short"] is not None
    assert context["vix_sma_long"] is not None


def test_build_volatility_regime_context_marks_stressed_regime_when_vix_exceeds_block_threshold(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2023-12-12", periods=25, freq="D")
    vix_frame = pd.DataFrame(
        {
            "date": dates,
            "open": [28.0] * 25,
            "high": [32.0] * 25,
            "low": [27.0] * 25,
            "close": [28.0] * 24 + [31.2],
            "volume": [0] * 25,
            "symbol": ["^VIX"] * 25,
        }
    )
    provider = FakeDailyBarProvider(frames_by_symbol={"^VIX": vix_frame}, cache_dir=tmp_path)

    context = build_volatility_regime_context(
        provider=provider,
        as_of_date=date(2024, 1, 5),
        history_start=date(2023, 12, 1),
        caution_threshold=25.0,
        entry_block_threshold=30.0,
    )

    assert context["vix_close"] == pytest.approx(31.2)
    assert context["volatility_regime_state"] == "stressed"
    assert context["volatility_regime_risk_off"] is True


def test_build_volatility_regime_context_degrades_gracefully_when_vix_is_missing(
    tmp_path: Path,
) -> None:
    provider = FakeDailyBarProvider(frames_by_symbol={}, cache_dir=tmp_path)

    context = build_volatility_regime_context(
        provider=provider,
        as_of_date=date(2024, 1, 5),
        history_start=date(2023, 12, 1),
        caution_threshold=25.0,
        entry_block_threshold=30.0,
    )

    assert context == {
        "vix_close": None,
        "vix_sma_short": None,
        "vix_sma_long": None,
        "volatility_regime_state": None,
        "volatility_regime_risk_off": None,
    }


def test_build_volatility_regime_context_degrades_gracefully_when_vix_bars_are_stale(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2023-12-01", periods=25, freq="D")
    vix_frame = pd.DataFrame(
        {
            "date": dates,
            "open": [24.0] * 25,
            "high": [25.0] * 25,
            "low": [23.0] * 25,
            "close": [24.0] * 25,
            "volume": [0] * 25,
            "symbol": ["^VIX"] * 25,
        }
    )
    provider = FakeDailyBarProvider(frames_by_symbol={"^VIX": vix_frame}, cache_dir=tmp_path)

    context = build_volatility_regime_context(
        provider=provider,
        as_of_date=(dates[-1] + timedelta(days=MAX_VOLATILITY_CONTEXT_BAR_AGE_DAYS + 1)).date(),
        history_start=date(2023, 12, 1),
        caution_threshold=25.0,
        entry_block_threshold=30.0,
    )

    assert context == {
        "vix_close": None,
        "vix_sma_short": None,
        "vix_sma_long": None,
        "volatility_regime_state": None,
        "volatility_regime_risk_off": None,
    }


def test_provider_intraday_fetch_excludes_premarket_and_after_hours(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-03 13:00:00",  # 08:00 ET premarket
                    "2024-01-03 14:30:00",  # 09:30 ET regular session
                    "2024-01-03 21:15:00",  # 16:15 ET after hours
                ]
            ),
            "open": [9.0, 10.0, 10.8],
            "high": [9.2, 10.6, 10.9],
            "low": [8.9, 9.9, 10.7],
            "close": [9.1, 10.5, 10.75],
            "volume": [500, 1500, 400],
            "vwap": [9.05, 10.3, 10.8],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        }
    )
    provider = FakeDailyBarProvider(
        frames_by_symbol={},
        intraday_frames_by_symbol={"AAPL": frame},
        cache_dir=tmp_path,
    )

    session_frame = provider.fetch_intraday_bars(
        "AAPL",
        date(2024, 1, 3),
        interval_minutes=15,
    )

    assert session_frame["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2024-01-03 14:30:00"
    ]
    assert session_frame["open"].tolist() == [10.0]


def test_filter_intraday_bars_by_session_date_handles_dst_transition() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-03-11 13:00:00",  # 09:00 ET premarket after spring DST shift
                    "2024-03-11 13:30:00",  # 09:30 ET regular-session open
                    "2024-03-11 20:00:00",  # 16:00 ET session close
                    "2024-03-11 20:15:00",  # 16:15 ET after hours
                ]
            ),
            "open": [9.0, 10.0, 10.8, 10.9],
            "high": [9.2, 10.6, 11.0, 11.1],
            "low": [8.9, 9.9, 10.7, 10.8],
            "close": [9.1, 10.5, 10.9, 11.0],
            "volume": [500, 1500, 1400, 300],
            "vwap": [9.05, 10.3, 10.85, 10.95],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        }
    )

    filtered = filter_intraday_bars_by_session_date(
        normalize_intraday_bars(frame),
        session_date=date(2024, 3, 11),
    )

    assert filtered["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2024-03-11 13:30:00",
        "2024-03-11 20:00:00",
    ]
    assert filtered["open"].tolist() == [10.0, 10.8]


def test_load_provider_environment_uses_paired_quote_parsing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'DOUBLE_QUOTED="double value"\n'
        "SINGLE_QUOTED='single value'\n"
        "UNQUOTED=plain-value\n",
        encoding="utf-8",
    )

    parsed = load_provider_environment(env_file=env_file, environment={})

    assert parsed["DOUBLE_QUOTED"] == "double value"
    assert parsed["SINGLE_QUOTED"] == "single value"
    assert parsed["UNQUOTED"] == "plain-value"


def test_earnings_provider_cache_avoids_duplicate_remote_fetches(tmp_path: Path) -> None:
    provider = FakeEarningsCalendarProvider(
        records_by_window={
            (
                date(2024, 1, 5),
                date(2024, 2, 4),
            ): [
                {"ticker": "AAPL", "date": "2024-01-10", "status": "confirmed"},
                {"ticker": "MSFT", "date": "2024-01-12", "status": "confirmed"},
            ]
        },
        cache_dir=tmp_path,
    )

    first = provider.fetch_upcoming_earnings(
        start_date=date(2024, 1, 5),
        end_date=date(2024, 2, 4),
        symbols=("AAPL", "MSFT"),
    )
    second = provider.fetch_upcoming_earnings(
        start_date=date(2024, 1, 5),
        end_date=date(2024, 2, 4),
        symbols=("AAPL", "MSFT"),
    )

    assert provider.fetch_count == 1
    assert first["AAPL"].earnings_date == date(2024, 1, 10)
    assert second["MSFT"].earnings_date == date(2024, 1, 12)


def test_build_earnings_risk_contexts_computes_trading_days_away(tmp_path: Path) -> None:
    provider = FakeEarningsCalendarProvider(
        records_by_window={
            (
                date(2024, 1, 5),
                date(2024, 2, 4),
            ): [
                {"ticker": "AAPL", "date": "2024-01-10", "status": "confirmed"},
            ]
        },
        cache_dir=tmp_path,
    )

    contexts = build_earnings_risk_contexts(
        ["AAPL", "MSFT"],
        as_of_date=date(2024, 1, 5),
        provider=provider,
        risk_window_days=7,
        lookahead_calendar_days=30,
    )

    assert contexts["AAPL"].earnings_days_away == 3
    assert contexts["AAPL"].is_earnings_risk is True
    assert contexts["MSFT"].earnings_days_away is None
    assert contexts["MSFT"].is_earnings_risk is False


def test_trading_days_until_accounts_for_nyse_holiday_week() -> None:
    assert trading_days_until(date(2024, 12, 23), date(2024, 12, 27)) == 3


def test_build_earnings_risk_contexts_ignores_past_earnings_dates(tmp_path: Path) -> None:
    provider = FakeEarningsCalendarProvider(
        records_by_window={
            (
                date(2024, 1, 5),
                date(2024, 2, 4),
            ): [
                {"ticker": "AAPL", "date": "2024-01-03", "status": "confirmed"},
            ]
        },
        cache_dir=tmp_path,
    )

    contexts = build_earnings_risk_contexts(
        ["AAPL"],
        as_of_date=date(2024, 1, 5),
        provider=provider,
        risk_window_days=7,
        lookahead_calendar_days=30,
    )

    assert contexts["AAPL"].earnings_date is None
    assert contexts["AAPL"].earnings_days_away is None
    assert contexts["AAPL"].is_earnings_risk is False


def test_build_sector_feature_contexts_fetches_shared_sector_etf_once(tmp_path: Path) -> None:
    symbol_dates = pd.date_range("2024-01-02", periods=25, freq="B")
    sector_dates = pd.date_range("2023-03-01", periods=240, freq="B")
    provider = FakeDailyBarProvider(
        frames_by_symbol={
            "XLK": pd.DataFrame(
                {
                    "date": sector_dates,
                    "open": [100.0 + index for index in range(len(sector_dates))],
                    "high": [101.0 + index for index in range(len(sector_dates))],
                    "low": [99.0 + index for index in range(len(sector_dates))],
                    "close": [100.0 + index for index in range(len(sector_dates))],
                    "volume": [1_000_000 for _ in range(len(sector_dates))],
                    "symbol": ["XLK" for _ in range(len(sector_dates))],
                }
            ),
        },
        cache_dir=tmp_path,
    )
    symbol_frames = {
        "AAPL": pd.DataFrame(
            {
                "date": symbol_dates,
                "open": [100.0 + index for index in range(len(symbol_dates))],
                "high": [101.0 + index for index in range(len(symbol_dates))],
                "low": [99.0 + index for index in range(len(symbol_dates))],
                "close": [100.0 + index for index in range(len(symbol_dates))],
                "volume": [1_000_000 for _ in range(len(symbol_dates))],
                "symbol": ["AAPL" for _ in range(len(symbol_dates))],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "date": symbol_dates,
                "open": [120.0 + index for index in range(len(symbol_dates))],
                "high": [121.0 + index for index in range(len(symbol_dates))],
                "low": [119.0 + index for index in range(len(symbol_dates))],
                "close": [120.0 + index for index in range(len(symbol_dates))],
                "volume": [1_200_000 for _ in range(len(symbol_dates))],
                "symbol": ["MSFT" for _ in range(len(symbol_dates))],
            }
        ),
    }
    sector_classifications = {
        "AAPL": SymbolSectorClassification(
            symbol="AAPL",
            sector="Technology",
            industry="Consumer electronics",
            sector_etf_symbol="XLK",
            mapping_source="test",
        ),
        "MSFT": SymbolSectorClassification(
            symbol="MSFT",
            sector="Technology",
            industry="Application software",
            sector_etf_symbol="XLK",
            mapping_source="test",
        ),
    }

    contexts = build_sector_feature_contexts(
        symbol_frames,
        provider=provider,
        as_of_date=date(2024, 2, 5),
        history_start=date(2023, 3, 1),
        sector_classifications=sector_classifications,
        benchmark_sma_fast=50,
        benchmark_sma_slow=200,
        relative_strength_window=20,
    )

    assert provider.fetch_count == 1
    assert contexts["AAPL"].sector_etf_symbol == "XLK"
    assert contexts["MSFT"].sector_etf_symbol == "XLK"
    assert contexts["AAPL"].sector_regime_passed is True


def test_build_sector_feature_contexts_degrades_gracefully_without_mappings(tmp_path: Path) -> None:
    symbol_dates = pd.date_range("2024-01-02", periods=25, freq="B")
    provider = FakeDailyBarProvider(frames_by_symbol={}, cache_dir=tmp_path)
    symbol_frames = {
        "AAPL": pd.DataFrame(
            {
                "date": symbol_dates,
                "open": [100.0 + index for index in range(len(symbol_dates))],
                "high": [101.0 + index for index in range(len(symbol_dates))],
                "low": [99.0 + index for index in range(len(symbol_dates))],
                "close": [100.0 + index for index in range(len(symbol_dates))],
                "volume": [1_000_000 for _ in range(len(symbol_dates))],
                "symbol": ["AAPL" for _ in range(len(symbol_dates))],
            }
        ),
    }

    contexts = build_sector_feature_contexts(
        symbol_frames,
        provider=provider,
        as_of_date=date(2024, 2, 5),
        history_start=date(2024, 1, 2),
        sector_classifications={},
        benchmark_sma_fast=50,
        benchmark_sma_slow=200,
        relative_strength_window=20,
    )

    assert contexts == {}
    assert provider.fetch_count == 0


def test_create_provider_uses_configured_provider() -> None:
    config = load_app_config()
    api_key_env = config.data_sources.active_provider().api_key_env
    provider = create_daily_bar_provider(
        config,
        environment={api_key_env: "demo-key"},
        cache_dir=Path("/tmp/investopedia-provider-cache"),
    )

    assert provider.provider_name == config.data_sources.provider.lower()


def test_candidate_score_journal_updates_and_persists_repeated_candidates(
    tmp_path: Path,
) -> None:
    journal_path = default_candidate_score_journal_path(tmp_path)
    journal = load_candidate_score_journal(journal_path)
    first_observation = CandidateJournalObservation(
        symbol="AAPL",
        approved_today=False,
        breakout_strength=0.02,
        relative_volume=1.8,
        sector_etf_symbol="XLK",
        sector_regime_passed=True,
    )

    first_features = build_candidate_memory_features(
        None,
        observation=first_observation,
        as_of_date=date(2024, 1, 5),
    )
    journal = update_candidate_score_journal(
        journal,
        observations={"AAPL": first_observation},
        as_of_date=date(2024, 1, 5),
    )
    write_candidate_score_journal(journal, journal_path)
    reloaded = load_candidate_score_journal(journal_path)

    second_observation = CandidateJournalObservation(
        symbol="AAPL",
        approved_today=True,
        breakout_strength=0.03,
        relative_volume=2.1,
        sector_etf_symbol="XLK",
        sector_regime_passed=True,
        best_rank_today=2,
    )
    second_features = build_candidate_memory_features(
        reloaded.entries["AAPL"],
        observation=second_observation,
        as_of_date=date(2024, 1, 8),
    )
    updated = update_candidate_score_journal(
        reloaded,
        observations={"AAPL": second_observation},
        as_of_date=date(2024, 1, 8),
    )
    payload = json.loads(write_candidate_score_journal(updated, journal_path).read_text(encoding="utf-8"))

    assert first_features["setup_persistence_days"] == 1
    assert first_features["days_approved"] == 0
    assert second_features["setup_persistence_days"] == 2
    assert second_features["days_approved"] == 1
    assert second_features["setup_quality_score"] > first_features["setup_quality_score"]
    assert updated.entries["AAPL"] == CandidateJournalEntry(
        symbol="AAPL",
        last_seen_date=date(2024, 1, 8),
        days_near_breakout=2,
        days_approved=1,
        last_rank=2,
        best_rank=2,
        peak_relative_volume=2.1,
        peak_breakout_strength=0.03,
        last_sector_etf_symbol="XLK",
        last_sector_regime_passed=True,
        last_approved_date=date(2024, 1, 8),
    )
    assert payload["symbols"]["AAPL"]["days_near_breakout"] == 2


def test_candidate_score_journal_same_day_rerun_is_idempotent() -> None:
    journal = update_candidate_score_journal(
        load_candidate_score_journal(Path("/tmp/does-not-exist.json"), stale_after_days=30),
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=True,
                breakout_strength=0.02,
                relative_volume=1.8,
            )
        },
        as_of_date=date(2024, 1, 5),
    )

    rerun = update_candidate_score_journal(
        journal,
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=True,
                breakout_strength=0.03,
                relative_volume=2.0,
            )
        },
        as_of_date=date(2024, 1, 5),
    )

    assert rerun.entries["AAPL"].days_near_breakout == 1
    assert rerun.entries["AAPL"].days_approved == 1
    assert rerun.entries["AAPL"].peak_relative_volume == 2.0
    assert rerun.entries["AAPL"].peak_breakout_strength == 0.03


def test_candidate_score_journal_five_day_gap_preserves_setup_streak() -> None:
    journal = update_candidate_score_journal(
        load_candidate_score_journal(Path("/tmp/does-not-exist.json"), stale_after_days=30),
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=False,
                breakout_strength=0.02,
                relative_volume=1.8,
            )
        },
        as_of_date=date(2024, 1, 5),
    )

    updated = update_candidate_score_journal(
        journal,
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=False,
                breakout_strength=0.02,
                relative_volume=1.8,
            )
        },
        as_of_date=date(2024, 1, 10),
    )

    assert updated.entries["AAPL"].days_near_breakout == 2


def test_candidate_score_journal_six_day_gap_resets_setup_streak() -> None:
    journal = update_candidate_score_journal(
        load_candidate_score_journal(Path("/tmp/does-not-exist.json"), stale_after_days=30),
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=False,
                breakout_strength=0.02,
                relative_volume=1.8,
            )
        },
        as_of_date=date(2024, 1, 5),
    )

    updated = update_candidate_score_journal(
        journal,
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=False,
                breakout_strength=0.02,
                relative_volume=1.8,
            )
        },
        as_of_date=date(2024, 1, 11),
    )

    assert updated.entries["AAPL"].days_near_breakout == 1


def test_candidate_score_journal_ages_out_stale_symbols() -> None:
    journal = update_candidate_score_journal(
        load_candidate_score_journal(Path("/tmp/does-not-exist.json"), stale_after_days=30),
        observations={
            "AAPL": CandidateJournalObservation(
                symbol="AAPL",
                approved_today=False,
                breakout_strength=0.02,
                relative_volume=1.6,
            )
        },
        as_of_date=date(2024, 1, 5),
    )

    pruned = update_candidate_score_journal(
        journal,
        observations={},
        as_of_date=date(2024, 2, 10),
    )

    assert pruned.entries == {}


def test_candidate_score_journal_ignores_corrupt_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal_path = default_candidate_score_journal_path(tmp_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{not-json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        journal = load_candidate_score_journal(journal_path)

    assert journal.entries == {}
    assert "Ignoring candidate score journal" in caplog.text


def test_intraday_state_journal_updates_and_persists_repeated_polls(
    tmp_path: Path,
) -> None:
    session_date = date(2024, 1, 5)
    journal_path = default_intraday_state_journal_path(tmp_path, session_date)
    journal = load_intraday_state_journal(journal_path, session_date=session_date)

    first_observation = _intraday_state_observation(
        timestamp="2024-01-05T09:45:00",
        latest_close=101.0,
        close_vs_vwap_pct=-0.004,
        session_high_giveback_pct=0.021,
        intraday_return_vs_benchmark=-0.010,
        weak_intraday_relative_strength=False,
        suggested_action="WATCH CLOSELY",
    )
    second_observation = _intraday_state_observation(
        timestamp="2024-01-05T10:00:00",
        latest_close=99.8,
        close_vs_vwap_pct=-0.012,
        session_high_giveback_pct=0.038,
        intraday_return_vs_benchmark=-0.024,
        weak_intraday_relative_strength=True,
        suggested_action="EXIT CANDIDATE",
    )

    journal = update_intraday_state_journal(
        journal,
        observations={"AAPL": first_observation},
    )
    write_intraday_state_journal(journal, journal_path)
    reloaded = load_intraday_state_journal(journal_path, session_date=session_date)
    updated = update_intraday_state_journal(
        reloaded,
        observations={"AAPL": second_observation},
    )
    reloaded = load_intraday_state_journal(
        write_intraday_state_journal(updated, journal_path),
        session_date=session_date,
    )

    assert journal_path.exists()
    assert [observation.timestamp for observation in reloaded.entries["AAPL"]] == [
        "2024-01-05T09:45:00",
        "2024-01-05T10:00:00",
    ]
    assert reloaded.entries["AAPL"][-1].suggested_action == "EXIT CANDIDATE"


def test_intraday_state_journal_same_timestamp_is_idempotent() -> None:
    session_date = date(2024, 1, 5)
    journal = update_intraday_state_journal(
        load_intraday_state_journal(
            Path("/tmp/does-not-exist-intraday-state.json"),
            session_date=session_date,
        ),
        observations={
            "AAPL": _intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=101.0,
                close_vs_vwap_pct=-0.006,
                session_high_giveback_pct=0.024,
                intraday_return_vs_benchmark=-0.011,
                weak_intraday_relative_strength=False,
                suggested_action="WATCH CLOSELY",
            )
        },
    )

    rerun = update_intraday_state_journal(
        journal,
        observations={
            "AAPL": _intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=99.5,
                close_vs_vwap_pct=-0.015,
                session_high_giveback_pct=0.041,
                intraday_return_vs_benchmark=-0.028,
                weak_intraday_relative_strength=True,
                suggested_action="EXIT CANDIDATE",
            )
        },
    )

    assert len(rerun.entries["AAPL"]) == 1
    assert rerun.entries["AAPL"][0].latest_close == pytest.approx(99.5)
    assert rerun.entries["AAPL"][0].suggested_action == "EXIT CANDIDATE"


def test_intraday_state_journal_ignores_corrupt_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_date = date(2024, 1, 5)
    journal_path = default_intraday_state_journal_path(tmp_path, session_date)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{not-json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        journal = load_intraday_state_journal(journal_path, session_date=session_date)

    assert journal.entries == {}
    assert "Ignoring intraday state journal" in caplog.text


def test_preview_intraday_trajectory_features_rejects_resolved_observation() -> None:
    with pytest.raises(
        ValueError,
        match="expects an unresolved observation with suggested_action=None",
    ):
        preview_intraday_trajectory_features(
            load_intraday_state_journal(
                Path("/tmp/does-not-exist-intraday-state.json"),
                session_date=date(2024, 1, 5),
            ),
            observation=_intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=99.5,
                close_vs_vwap_pct=-0.015,
                session_high_giveback_pct=0.041,
                suggested_action="WATCH CLOSELY",
            ),
        )


def test_build_intraday_trajectory_features_counts_resolved_last_watch_observation() -> None:
    trajectory = build_intraday_trajectory_features(
        [
            _intraday_state_observation(
                timestamp="2024-01-05T09:45:00",
                latest_close=100.8,
                close_vs_vwap_pct=0.002,
                session_high_giveback_pct=0.012,
                suggested_action="HOLD",
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=99.7,
                close_vs_vwap_pct=-0.009,
                session_high_giveback_pct=0.031,
                suggested_action="WATCH CLOSELY",
            ),
        ]
    )

    assert trajectory.repeated_watch_closely_count == 1


def test_intraday_trajectory_features_count_consecutive_below_vwap_polls() -> None:
    trajectory = build_intraday_trajectory_features(
        [
            _intraday_state_observation(
                timestamp="2024-01-05T09:30:00",
                latest_close=101.0,
                close_vs_vwap_pct=0.002,
                session_high_giveback_pct=0.010,
                intraday_return_vs_benchmark=0.002,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T09:45:00",
                latest_close=100.5,
                close_vs_vwap_pct=-0.003,
                session_high_giveback_pct=0.019,
                intraday_return_vs_benchmark=-0.005,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=99.8,
                close_vs_vwap_pct=-0.008,
                session_high_giveback_pct=0.028,
                intraday_return_vs_benchmark=-0.024,
                weak_intraday_relative_strength=True,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T10:15:00",
                latest_close=99.2,
                close_vs_vwap_pct=-0.015,
                session_high_giveback_pct=0.041,
                intraday_return_vs_benchmark=-0.031,
                weak_intraday_relative_strength=True,
            ),
        ]
    )

    assert trajectory.observation_count == 4
    assert trajectory.consecutive_polls_below_vwap == 3
    assert trajectory.consecutive_polls_weak_relative_strength == 2
    assert trajectory.intraday_pressure_persistence_count == 3
    assert trajectory.weakening_all_session is False


def test_intraday_trajectory_features_detect_worsening_giveback() -> None:
    trajectory = build_intraday_trajectory_features(
        [
            _intraday_state_observation(
                timestamp="2024-01-05T09:30:00",
                latest_close=101.5,
                close_vs_vwap_pct=0.003,
                session_high_giveback_pct=0.021,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T09:45:00",
                latest_close=101.0,
                close_vs_vwap_pct=-0.001,
                session_high_giveback_pct=0.035,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T10:00:00",
                latest_close=100.4,
                close_vs_vwap_pct=-0.006,
                session_high_giveback_pct=0.048,
            ),
            _intraday_state_observation(
                timestamp="2024-01-05T10:15:00",
                latest_close=99.9,
                close_vs_vwap_pct=-0.011,
                session_high_giveback_pct=0.057,
            ),
        ]
    )

    assert trajectory.worsening_session_high_giveback is True
    assert trajectory.giveback_worsening_polls == 4
    assert trajectory.giveback_worsening_from_pct == pytest.approx(0.021)
    assert trajectory.max_session_high_giveback_seen == pytest.approx(0.057)


def test_intraday_state_journal_ignores_boolean_float_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_date = date(2024, 1, 5)
    journal_path = default_intraday_state_journal_path(tmp_path, session_date)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_date": "2024-01-05",
                "updated_at": None,
                "symbols": {
                    "AAPL": {
                        "observations": [
                            {
                                "timestamp": "2024-01-05T10:00:00",
                                "latest_close": True,
                                "session_vwap": 100.0,
                                "close_vs_vwap_pct": -0.01,
                                "session_high": 103.0,
                                "session_low": 98.0,
                                "session_high_giveback_pct": 0.03,
                                "intraday_return_vs_open": -0.01,
                                "intraday_return_vs_benchmark": -0.02,
                                "weak_intraday_relative_strength": False,
                                "failed_intraday_strength": False,
                                "intraday_momentum_fade": False,
                                "stacked_intraday_weakness": False,
                                "stop_breached_intraday": False,
                                "suggested_action": "WATCH CLOSELY",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        journal = load_intraday_state_journal(journal_path, session_date=session_date)
    assert journal.entries == {}

    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_date": "2024-01-05",
                "updated_at": None,
                "symbols": {
                    "AAPL": {
                        "observations": [
                            {
                                "timestamp": "2024-01-05T10:00:00",
                                "latest_close": 100.0,
                                "session_vwap": True,
                                "close_vs_vwap_pct": -0.01,
                                "session_high": 103.0,
                                "session_low": 98.0,
                                "session_high_giveback_pct": 0.03,
                                "intraday_return_vs_open": -0.01,
                                "intraday_return_vs_benchmark": -0.02,
                                "weak_intraday_relative_strength": False,
                                "failed_intraday_strength": False,
                                "intraday_momentum_fade": False,
                                "stacked_intraday_weakness": False,
                                "stop_breached_intraday": False,
                                "suggested_action": "WATCH CLOSELY",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        journal = load_intraday_state_journal(journal_path, session_date=session_date)
    assert journal.entries == {}
    assert "Ignoring intraday state journal" in caplog.text


def test_position_trajectory_journal_updates_and_persists_repeated_reviews(
    tmp_path: Path,
) -> None:
    journal_path = default_position_trajectory_journal_path(tmp_path)
    journal = load_position_trajectory_journal(journal_path)

    first_observation = _position_trajectory_observation(
        as_of_date=date(2024, 1, 5),
        latest_close=98.0,
        above_entry=False,
        stale_position=False,
        weak_relative_strength=False,
        suggested_action="WATCH CLOSELY",
    )
    second_observation = _position_trajectory_observation(
        as_of_date=date(2024, 1, 8),
        latest_close=96.5,
        above_entry=False,
        stale_position=True,
        weak_relative_strength=True,
        suggested_action="EXIT CANDIDATE",
    )

    journal = update_position_trajectory_journal(
        journal,
        observations={"AAPL": first_observation},
        active_symbols=("AAPL",),
    )
    write_position_trajectory_journal(journal, journal_path)
    reloaded = load_position_trajectory_journal(journal_path)
    updated = update_position_trajectory_journal(
        reloaded,
        observations={"AAPL": second_observation},
        active_symbols=("AAPL",),
    )
    reloaded = load_position_trajectory_journal(
        write_position_trajectory_journal(updated, journal_path),
    )

    assert journal_path.exists()
    assert [observation.as_of_date for observation in reloaded.entries["AAPL"]] == [
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    assert reloaded.entries["AAPL"][-1].suggested_action == "EXIT CANDIDATE"


def test_position_trajectory_journal_same_day_rerun_is_idempotent() -> None:
    journal = update_position_trajectory_journal(
        load_position_trajectory_journal(Path("/tmp/does-not-exist-position-trajectory.json")),
        observations={
            "AAPL": _position_trajectory_observation(
                as_of_date=date(2024, 1, 5),
                latest_close=98.0,
                above_entry=False,
                stale_position=False,
                weak_relative_strength=False,
                suggested_action="WATCH CLOSELY",
            )
        },
        active_symbols=("AAPL",),
    )

    rerun = update_position_trajectory_journal(
        journal,
        observations={
            "AAPL": _position_trajectory_observation(
                as_of_date=date(2024, 1, 5),
                latest_close=101.0,
                above_entry=True,
                stale_position=False,
                weak_relative_strength=False,
                suggested_action="HOLD",
            )
        },
        active_symbols=("AAPL",),
    )

    assert len(rerun.entries["AAPL"]) == 1
    assert rerun.entries["AAPL"][0].latest_close == pytest.approx(101.0)
    assert rerun.entries["AAPL"][0].suggested_action == "HOLD"


def test_position_trajectory_journal_ignores_corrupt_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal_path = default_position_trajectory_journal_path(tmp_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("{not-json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        journal = load_position_trajectory_journal(journal_path)

    assert journal.entries == {}
    assert "Ignoring position trajectory journal" in caplog.text


def test_position_trajectory_journal_ignores_boolean_float_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal_path = default_position_trajectory_journal_path(tmp_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": None,
                "symbols": {
                    "AAPL": {
                        "observations": [
                            {
                                "as_of_date": "2024-01-05",
                                "average_entry_price": True,
                                "current_stop": 95.0,
                                "latest_close": 98.0,
                                "unrealized_pl_pct": -0.02,
                                "above_entry": False,
                                "high_water_close": 110.0,
                                "high_water_close_date": "2024-01-02",
                                "days_since_new_high": 3,
                                "stale_position": False,
                                "relative_strength_return_diff": -0.01,
                                "weak_relative_strength": False,
                                "suggested_action": "WATCH CLOSELY",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        journal = load_position_trajectory_journal(journal_path)

    assert journal.entries == {}
    assert "Ignoring position trajectory journal" in caplog.text


def test_position_trajectory_features_count_consecutive_below_entry_days() -> None:
    trajectory = build_position_trajectory_features(
        [
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 3),
                latest_close=101.0,
                above_entry=True,
                stale_position=False,
                weak_relative_strength=False,
                suggested_action="HOLD",
            ),
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 4),
                latest_close=98.0,
                above_entry=False,
                stale_position=False,
                weak_relative_strength=False,
                suggested_action="WATCH CLOSELY",
            ),
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 5),
                latest_close=97.0,
                above_entry=False,
                stale_position=False,
                weak_relative_strength=True,
                suggested_action="WATCH CLOSELY",
            ),
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 8),
                latest_close=96.0,
                above_entry=False,
                stale_position=True,
                weak_relative_strength=True,
                suggested_action=None,
            ),
        ]
    )

    assert trajectory.observation_count == 4
    assert trajectory.days_in_position_state == 4
    assert trajectory.consecutive_days_below_entry == 3
    assert trajectory.consecutive_weak_position_days == 3
    assert trajectory.persistent_underperformance is True


def test_position_trajectory_features_count_repeated_watch_states() -> None:
    trajectory = build_position_trajectory_features(
        [
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 3),
                latest_close=99.0,
                above_entry=False,
                stale_position=False,
                weak_relative_strength=False,
                suggested_action="WATCH CLOSELY",
            ),
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 4),
                latest_close=98.0,
                above_entry=False,
                stale_position=True,
                weak_relative_strength=False,
                suggested_action="WATCH CLOSELY",
            ),
            _position_trajectory_observation(
                as_of_date=date(2024, 1, 5),
                latest_close=97.5,
                above_entry=False,
                stale_position=True,
                weak_relative_strength=True,
                suggested_action=None,
            ),
        ]
    )

    assert trajectory.consecutive_watch_closely_days == 2
    assert trajectory.consecutive_hold_days == 0
    assert trajectory.consecutive_stale_position_days == 2


def test_trade_feedback_log_appends_deduplicates_and_stays_compact(
    tmp_path: Path,
) -> None:
    log_path = default_trade_feedback_log_path(tmp_path)
    first_event = TradeFeedbackEvent(
        event_type="decision",
        workflow="daily-summary",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_abc123",
        preset_name="standard_breakout",
        strategy_name="breakout_momentum:standard_breakout",
        suggested_action="BUY",
        approved=True,
        queue_rank=1,
        priority_bucket="top_priority",
        actionable_now=True,
        quantity=100,
        entry_price_hint=100.0,
        stop_price=95.0,
        notional_value=10_000.0,
        feature_snapshot={
            "opportunity_score": 1.25,
            "setup_quality_score": 0.8,
            "volatility_regime_state": "calm",
        },
    )
    duplicate_event = TradeFeedbackEvent(
        event_type="decision",
        workflow="daily-summary",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_abc123",
        preset_name="standard_breakout",
        strategy_name="breakout_momentum:standard_breakout",
        suggested_action="BUY",
        approved=True,
        queue_rank=1,
        priority_bucket="top_priority",
        actionable_now=True,
        quantity=100,
        entry_price_hint=100.0,
        stop_price=95.0,
        notional_value=10_000.0,
        feature_snapshot={
            "opportunity_score": 1.25,
            "setup_quality_score": 0.8,
            "volatility_regime_state": "calm",
        },
    )
    executed_event = TradeFeedbackEvent(
        event_type="executed",
        workflow="upsert-position",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_abc123",
        trade_id="trade_abc123",
        linked_decision_id="decision_abc123",
        suggested_action="BUY",
        quantity=100,
        entry_price_hint=100.0,
        stop_price=95.0,
        notional_value=10_000.0,
    )

    append_trade_feedback_events([first_event, duplicate_event, executed_event], log_path)
    loaded = load_trade_feedback_events(log_path)
    summary = summarize_trade_feedback_events(loaded)
    persisted_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    persisted_payload = json.loads(persisted_lines[0])

    assert len(loaded) == 2
    assert [event.event_type for event in loaded] == ["decision", "executed"]
    assert summary["approved_decision_count"] == 1
    assert summary["executed_approved_decision_count"] == 1
    assert summary["approved_execution_rate"] == pytest.approx(1.0)
    assert set(persisted_payload) == {
        "actionable_now",
        "approved",
        "as_of_date",
        "decision_id",
        "entry_price_hint",
        "event_id",
        "event_type",
        "feature_snapshot",
        "notional_value",
        "preset_name",
        "priority_bucket",
        "quantity",
        "queue_rank",
        "schema_version",
        "stop_price",
        "strategy_name",
        "suggested_action",
        "symbol",
        "workflow",
    }
    assert "price_frame" not in persisted_payload
    assert "bars" not in persisted_payload


def test_trade_feedback_log_ignores_malformed_lines(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log_path = default_trade_feedback_log_path(tmp_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "{bad-json\n"
        + json.dumps(
            TradeFeedbackEvent(
                event_type="decision",
                workflow="generate-orders",
                symbol="MSFT",
                as_of_date=date(2024, 1, 5),
                decision_id="decision_msft",
                suggested_action="BUY",
                approved=False,
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        loaded = load_trade_feedback_events(log_path)

    assert len(loaded) == 1
    assert loaded[0].symbol == "MSFT"
    assert "Ignoring malformed trade feedback line" in caplog.text


def test_trade_feedback_event_deserialization_round_trips_zero_sessions_since_entry() -> None:
    snapshot = TradeOutcomeSnapshot(
        as_of_date=date(2024, 1, 5),
        entry_date=date(2024, 1, 5),
        sessions_since_entry=0,
        current_return_pct=0.0,
        max_favorable_excursion_pct=0.01,
        max_adverse_excursion_pct=-0.01,
        stop_hit=False,
        forward_return_1d=None,
        forward_return_5d=None,
        forward_return_10d=None,
        above_entry_after_1d=None,
        above_entry_after_5d=None,
        above_entry_after_10d=None,
    )
    event = TradeFeedbackEvent(
        event_type="outcome",
        workflow="review-portfolio",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_aapl",
        trade_id="trade_aapl",
        linked_decision_id="decision_aapl",
        outcome_snapshot=snapshot,
    )

    reloaded_event = TradeFeedbackEvent.from_mapping(event.to_dict())

    assert reloaded_event.outcome_snapshot is not None
    assert reloaded_event.outcome_snapshot.sessions_since_entry == 0
    assert reloaded_event.outcome_snapshot.entry_date == date(2024, 1, 5)


def test_trade_feedback_log_round_trips_zero_sessions_since_entry(
    tmp_path: Path,
) -> None:
    log_path = default_trade_feedback_log_path(tmp_path)
    event = TradeFeedbackEvent(
        event_type="outcome",
        workflow="review-portfolio",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_aapl",
        trade_id="trade_aapl",
        linked_decision_id="decision_aapl",
        outcome_snapshot=TradeOutcomeSnapshot(
            as_of_date=date(2024, 1, 5),
            entry_date=date(2024, 1, 5),
            sessions_since_entry=0,
            current_return_pct=0.0,
            max_favorable_excursion_pct=0.01,
            max_adverse_excursion_pct=-0.01,
            stop_hit=False,
            forward_return_1d=None,
            forward_return_5d=None,
            forward_return_10d=None,
            above_entry_after_1d=None,
            above_entry_after_5d=None,
            above_entry_after_10d=None,
        ),
    )

    append_trade_feedback_events([event], log_path)
    loaded = load_trade_feedback_events(log_path)

    assert len(loaded) == 1
    assert loaded[0].outcome_snapshot is not None
    assert loaded[0].outcome_snapshot.sessions_since_entry == 0


def test_trade_feedback_log_deduplicates_logically_identical_events_with_different_feature_snapshots(
    tmp_path: Path,
) -> None:
    log_path = default_trade_feedback_log_path(tmp_path)
    first_event = TradeFeedbackEvent(
        event_type="decision",
        workflow="daily-summary",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_abc123",
        suggested_action="BUY",
        approved=True,
        feature_snapshot={"opportunity_score": 1.25},
    )
    rerun_event = TradeFeedbackEvent(
        event_type="decision",
        workflow="daily-summary",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_abc123",
        suggested_action="BUY",
        approved=True,
        feature_snapshot={"opportunity_score": 0.85, "setup_quality_score": 0.6},
    )

    append_trade_feedback_events([first_event], log_path)
    append_trade_feedback_events([rerun_event], log_path)
    loaded = load_trade_feedback_events(log_path)

    assert len(loaded) == 1
    assert loaded[0].decision_id == "decision_abc123"


def test_trade_feedback_log_persists_distinct_same_day_stop_events(
    tmp_path: Path,
) -> None:
    log_path = default_trade_feedback_log_path(tmp_path)
    first_stop = TradeFeedbackEvent(
        event_type="stop_raised",
        workflow="update-stop",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_aapl",
        trade_id="trade_aapl",
        linked_decision_id="decision_aapl",
        suggested_action="RAISE STOP",
        stop_price=97.0,
    )
    second_stop = TradeFeedbackEvent(
        event_type="stop_raised",
        workflow="update-stop",
        symbol="AAPL",
        as_of_date=date(2024, 1, 5),
        decision_id="decision_aapl",
        trade_id="trade_aapl",
        linked_decision_id="decision_aapl",
        suggested_action="RAISE STOP",
        stop_price=98.5,
    )

    append_trade_feedback_events([first_stop, second_stop], log_path)
    loaded = load_trade_feedback_events(log_path)

    assert len(loaded) == 2
    assert [event.event_type for event in loaded] == ["stop_raised", "stop_raised"]
    assert {event.stop_price for event in loaded} == {97.0, 98.5}


def test_compute_trade_outcome_snapshot_returns_expected_forward_metrics() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                ]
            ),
            "open": [100.0, 101.0, 103.0, 102.0, 104.0, 103.0, 106.0],
            "high": [101.0, 104.0, 105.0, 103.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 97.0, 102.0, 101.0, 105.0],
            "close": [100.0, 103.0, 102.0, 98.0, 105.0, 106.0, 107.0],
            "volume": [1_000_000] * 7,
            "symbol": ["AAPL"] * 7,
        }
    )

    outcome = compute_trade_outcome_snapshot(
        frame,
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        stop_price=98.0,
        as_of_date=date(2024, 1, 10),
    )

    assert outcome is not None
    assert outcome.sessions_since_entry == 6
    assert outcome.current_return_pct == pytest.approx(0.07)
    assert outcome.forward_return_1d == pytest.approx(0.03)
    assert outcome.forward_return_5d == pytest.approx(0.06)
    assert outcome.forward_return_10d is None
    assert outcome.max_favorable_excursion_pct == pytest.approx(0.08)
    assert outcome.max_adverse_excursion_pct == pytest.approx(-0.03)
    assert outcome.stop_hit is True
    assert outcome.above_entry_after_1d is True
    assert outcome.above_entry_after_5d is True
    assert outcome.above_entry_after_10d is None


def test_build_portfolio_heat_context_tracks_sector_counts_and_notional_share() -> None:
    context = build_portfolio_heat_context(
        [
            PortfolioHoldingExposure(
                symbol="MSFT",
                approximate_notional=10_000.0,
                sector_name="Technology",
                industry_name="Software",
            ),
            PortfolioHoldingExposure(
                symbol="NVDA",
                approximate_notional=15_000.0,
                sector_name="Technology",
                industry_name="Semiconductors",
            ),
            PortfolioHoldingExposure(
                symbol="JPM",
                approximate_notional=25_000.0,
                sector_name="Financials",
                industry_name="Banks",
            ),
        ],
        candidate_sector="Technology",
        candidate_industry="Software",
        max_positions_per_sector=3,
        max_same_industry_positions=2,
        max_sector_notional_pct=0.60,
    )

    assert context.current_position_count == 3
    assert context.same_sector_position_count == 2
    assert context.same_industry_position_count == 1
    assert context.projected_same_sector_position_count == 3
    assert context.sector_concentration_risk is False
    assert context.sector_notional_pct_by_sector["Technology"] == pytest.approx(0.50)


def test_project_portfolio_heat_context_flags_projected_notional_breach() -> None:
    context = build_portfolio_heat_context(
        [
            PortfolioHoldingExposure(
                symbol="MSFT",
                approximate_notional=10_000.0,
                sector_name="Technology",
                industry_name="Software",
            ),
            PortfolioHoldingExposure(
                symbol="NVDA",
                approximate_notional=15_000.0,
                sector_name="Technology",
                industry_name="Semiconductors",
            ),
            PortfolioHoldingExposure(
                symbol="JPM",
                approximate_notional=25_000.0,
                sector_name="Financials",
                industry_name="Banks",
            ),
        ],
        candidate_sector="Technology",
        candidate_industry="Software",
        max_positions_per_sector=4,
        max_same_industry_positions=3,
        max_sector_notional_pct=0.60,
    )

    projection = project_portfolio_heat_context(
        context,
        candidate_notional_value=20_000.0,
    )

    assert projection.projected_sector_notional_pct == pytest.approx(45_000.0 / 70_000.0)
    assert projection.sector_notional_concentration_risk is True
    assert projection.sector_concentration_risk is True
    assert projection.crowded_exposure_bucket is True


def test_build_portfolio_heat_context_normalizes_sector_labels_case_insensitively() -> None:
    context = build_portfolio_heat_context(
        [
            PortfolioHoldingExposure(
                symbol="MSFT",
                approximate_notional=10_000.0,
                sector_name="TECHNOLOGY",
                industry_name="SOFTWARE",
            ),
            PortfolioHoldingExposure(
                symbol="NVDA",
                approximate_notional=15_000.0,
                sector_name="Technology",
                industry_name="Semiconductors",
            ),
        ],
        candidate_sector="technology",
        candidate_industry="software",
        max_positions_per_sector=2,
        max_same_industry_positions=1,
    )

    assert context.candidate_sector == "Technology"
    assert context.candidate_industry == "Software"
    assert context.same_sector_position_count == 2
    assert context.same_industry_position_count == 1
    assert context.projected_same_sector_position_count == 3
    assert context.sector_concentration_risk is True
    assert context.correlated_exposure_risk is True
    assert context.sector_position_count_by_sector["Technology"] == 2


def test_load_candidate_symbols_supports_text_and_csv(tmp_path: Path) -> None:
    text_path = tmp_path / "candidates.txt"
    text_path.write_text("aapl\nmsft, nvda\nbrk.b\n# comment\nAAPL\n", encoding="utf-8")
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("symbol\nspy\nqqq\n", encoding="utf-8")

    assert load_candidate_symbols(text_path) == ["AAPL", "MSFT", "NVDA", "BRK.B"]
    assert load_candidate_symbols(csv_path) == ["SPY", "QQQ"]


def _intraday_state_observation(
    *,
    timestamp: str,
    latest_close: float,
    close_vs_vwap_pct: float,
    session_high_giveback_pct: float,
    intraday_return_vs_benchmark: float | None = None,
    weak_intraday_relative_strength: bool = False,
    suggested_action: str | None = None,
) -> IntradayStateObservation:
    return IntradayStateObservation(
        symbol="AAPL",
        timestamp=timestamp,
        latest_close=latest_close,
        session_vwap=100.0,
        close_vs_vwap_pct=close_vs_vwap_pct,
        session_high=103.0,
        session_low=98.0,
        session_high_giveback_pct=session_high_giveback_pct,
        intraday_return_vs_open=-0.01,
        intraday_return_vs_benchmark=intraday_return_vs_benchmark,
        weak_intraday_relative_strength=weak_intraday_relative_strength,
        failed_intraday_strength=False,
        intraday_momentum_fade=False,
        stacked_intraday_weakness=False,
        stop_breached_intraday=False,
        suggested_action=suggested_action,
    )


def _position_trajectory_observation(
    *,
    as_of_date: date,
    latest_close: float,
    above_entry: bool,
    stale_position: bool,
    weak_relative_strength: bool,
    suggested_action: str | None,
) -> PositionTrajectoryObservation:
    return PositionTrajectoryObservation(
        symbol="AAPL",
        as_of_date=as_of_date,
        average_entry_price=100.0,
        current_stop=95.0,
        latest_close=latest_close,
        unrealized_pl_pct=(latest_close / 100.0) - 1.0,
        above_entry=above_entry,
        high_water_close=110.0,
        high_water_close_date=date(2024, 1, 2),
        days_since_new_high=3 if stale_position else 1,
        stale_position=stale_position,
        relative_strength_return_diff=(-0.06 if weak_relative_strength else 0.01),
        weak_relative_strength=weak_relative_strength,
        suggested_action=suggested_action,
    )


def test_universe_builder_filters_and_ranks_symbols(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    frames_by_symbol = {
        "AAA": pd.DataFrame(
            {
                "date": dates,
                "open": [20.0, 20.0, 20.0],
                "high": [21.0, 21.0, 21.0],
                "low": [19.0, 19.0, 19.0],
                "close": [20.0, 20.0, 20.0],
                "volume": [2_000_000, 2_000_000, 2_000_000],
                "symbol": ["AAA", "AAA", "AAA"],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "date": dates,
                "open": [5.0, 5.0, 5.0],
                "high": [5.5, 5.5, 5.5],
                "low": [4.5, 4.5, 4.5],
                "close": [5.0, 5.0, 5.0],
                "volume": [5_000_000, 5_000_000, 5_000_000],
                "symbol": ["BBB", "BBB", "BBB"],
            }
        ),
        "CCC": pd.DataFrame(
            {
                "date": dates,
                "open": [30.0, 30.0, 30.0],
                "high": [31.0, 31.0, 31.0],
                "low": [29.0, 29.0, 29.0],
                "close": [30.0, 30.0, 30.0],
                "volume": [100_000, 100_000, 100_000],
                "symbol": ["CCC", "CCC", "CCC"],
            }
        ),
        "DDD": pd.DataFrame(
            {
                "date": dates,
                "open": [40.0, 40.0, 40.0],
                "high": [41.0, 41.0, 41.0],
                "low": [39.0, 39.0, 39.0],
                "close": [40.0, 40.0, 40.0],
                "volume": [1_000_000, 1_000_000, 1_000_000],
                "symbol": ["DDD", "DDD", "DDD"],
            }
        ),
    }
    provider = FakeDailyBarProvider(frames_by_symbol=frames_by_symbol, cache_dir=tmp_path)
    builder = UniverseBuilder(
        provider,
        UniverseConfig(min_price=10.0, min_avg_dollar_volume=20_000_000, max_symbols=2),
    )

    selected = builder.build(
        ["AAA", "BBB", "CCC", "DDD"],
        as_of_date=date(2024, 1, 4),
        lookback_days=3,
    )

    assert selected == ["AAA", "DDD"]


def test_universe_builder_can_skip_configured_max_symbol_cap(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    symbols = [f"SYM{index:03d}" for index in range(105)]
    frames_by_symbol = {
        symbol: pd.DataFrame(
            {
                "date": dates,
                "open": [50.0, 50.0, 50.0],
                "high": [51.0, 51.0, 51.0],
                "low": [49.0, 49.0, 49.0],
                "close": [50.0, 50.0, 50.0],
                "volume": [2_000_000, 2_000_000, 2_000_000],
                "symbol": [symbol, symbol, symbol],
            }
        )
        for symbol in symbols
    }
    provider = FakeDailyBarProvider(frames_by_symbol=frames_by_symbol, cache_dir=tmp_path)
    builder = UniverseBuilder(
        provider,
        UniverseConfig(min_price=10.0, min_avg_dollar_volume=20_000_000, max_symbols=100),
    )

    selected = builder.build(
        symbols,
        as_of_date=date(2024, 1, 4),
        lookback_days=3,
        enforce_max_symbols=False,
    )

    assert len(selected) == 105
    assert selected[:3] == ["SYM000", "SYM001", "SYM002"]
    assert selected[-1] == "SYM104"
