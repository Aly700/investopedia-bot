from datetime import date
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from bot.config import UniverseConfig, load_app_config
from bot.data.normalize import DAILY_BAR_COLUMNS, empty_daily_bar_frame, normalize_daily_bars
from bot.data.providers import DailyBarProvider, create_daily_bar_provider
from bot.data.universe import UniverseBuilder, load_candidate_symbols


class FakeDailyBarProvider(DailyBarProvider):
    provider_name = "fake"

    def __init__(self, *, frames_by_symbol: dict[str, pd.DataFrame], cache_dir: Path) -> None:
        super().__init__(api_key="test-key", cache_dir=cache_dir)
        self.frames_by_symbol = frames_by_symbol
        self.fetch_count = 0

    def _fetch_daily_bars_from_api(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        self.fetch_count += 1
        return self.frames_by_symbol.get(symbol, empty_daily_bar_frame()).copy()


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


def test_create_provider_uses_configured_provider() -> None:
    config = load_app_config()
    provider = create_daily_bar_provider(
        config,
        environment={"ALPHAVANTAGE_API_KEY": "demo-key"},
        cache_dir=Path("/tmp/investopedia-provider-cache"),
    )

    assert provider.provider_name == "alphavantage"


def test_load_candidate_symbols_supports_text_and_csv(tmp_path: Path) -> None:
    text_path = tmp_path / "candidates.txt"
    text_path.write_text("aapl\nmsft, nvda\n# comment\nAAPL\n", encoding="utf-8")
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("symbol\nspy\nqqq\n", encoding="utf-8")

    assert load_candidate_symbols(text_path) == ["AAPL", "MSFT", "NVDA"]
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
