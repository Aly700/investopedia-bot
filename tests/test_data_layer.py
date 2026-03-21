from __future__ import annotations

from datetime import date
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
from bot.data.sector_context import (
    SymbolSectorClassification,
    build_sector_feature_contexts,
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


def test_load_candidate_symbols_supports_text_and_csv(tmp_path: Path) -> None:
    text_path = tmp_path / "candidates.txt"
    text_path.write_text("aapl\nmsft, nvda\nbrk.b\n# comment\nAAPL\n", encoding="utf-8")
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("symbol\nspy\nqqq\n", encoding="utf-8")

    assert load_candidate_symbols(text_path) == ["AAPL", "MSFT", "NVDA", "BRK.B"]
    assert load_candidate_symbols(csv_path) == ["SPY", "QQQ"]


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
