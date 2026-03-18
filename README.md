# Investopedia Bot

`investopedia-bot` is a Python repo for daily-bar research, backtesting, and manual order output for an Investopedia simulator workflow.

Current scope:

- Research and backtesting only
- Historical daily-bar ingestion, normalization, caching, and universe construction
- Configuration-driven workflows
- Manual order sheet generation from offline signals
- No browser automation
- No live web execution

Python requirement: 3.9+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables live in `.env`. A starter file is provided in `.env.example`.

## Repo Layout

```text
config/                  YAML configuration for strategy, data sources, and simulator rules
data/                    Local research inputs, caches, processed outputs, and logs
notebooks/               Ad hoc research notebooks
src/bot/backtest/        Backtest engine, metrics, slippage, and walk-forward modules
src/bot/data/            Universe, provider, and normalization modules
src/bot/execution/       Manual execution helpers and future executor interfaces
src/bot/indicators/      Indicator library for daily-bar signals
src/bot/reporting/       Equity curves, trade logs, and daily summaries
src/bot/risk/            Position sizing, stops, and portfolio guardrails
src/bot/strategy/        Signal and regime models
src/bot/config.py        Typed config loader for the repo
src/bot/logging_utils.py Logging bootstrap for CLI and batch jobs
src/bot/main.py          CLI entrypoint
tests/                   Test suite
```

## Configuration

The repo uses three YAML files under `config/`:

- `strategy.yaml`: universe filters, signal parameters, and per-trade risk settings
- `data_sources.yaml`: active research data provider and the environment variable name for each API key
- `game_rules.yaml`: simulator cash, commissions, fill assumptions, and account-level constraints

The CLI loads and validates these files through `src/bot/config.py`. Config is kept separate from execution so strategy research, backtests, and order generation can share the same baseline assumptions.

Historical daily-bar data is selected by `config/data_sources.yaml`. Local cache files are written under `data/cache/daily_bars/<provider>/`.

## CLI Usage

After installation, use either the console script or `python -m bot.main`.

### Show merged config

```bash
investopedia-bot show-config
investopedia-bot show-config --format json
```

### Validate environment variables for the active data provider

```bash
investopedia-bot check-env
```

This checks the provider selected in `config/data_sources.yaml` and verifies that the required API key environment variable exists in `.env` or the shell environment.

### Fetch normalized daily bars

```bash
investopedia-bot fetch-data AAPL --start 2025-01-01 --end 2025-03-31
investopedia-bot fetch-data AAPL --start 2025-01-01 --end 2025-03-31 --output data/processed/aapl_daily.csv
investopedia-bot fetch-data AAPL --start 2025-01-01 --end 2025-03-31 --refresh-cache --format json
```

This uses the configured provider, normalizes the response to the canonical daily-bar schema, and caches the requested range locally.

### Build a filtered universe

```bash
investopedia-bot build-universe data/raw/candidate_symbols.txt --as-of 2026-03-17
investopedia-bot build-universe data/raw/candidate_symbols.csv --as-of 2026-03-17 --lookback-days 20 --format json
```

Candidate lists can be either:

- text files with one symbol per line or comma-separated symbols
- CSV files with a `symbol` column

Universe filtering uses the thresholds already defined in `config/strategy.yaml`:

- `universe.min_price`
- `universe.min_avg_dollar_volume`
- `universe.max_symbols`

### Generate a manual daily order sheet

```bash
investopedia-bot generate-orders data/raw/candidate_symbols.txt --as-of 2026-03-17
investopedia-bot generate-orders data/raw/candidate_symbols.txt --as-of 2026-03-17 --equity 125000 --current-drawdown 0.08
investopedia-bot generate-orders data/raw/candidate_symbols.txt --as-of 2026-03-17 --portfolio-file data/processed/portfolio/current_positions.csv
investopedia-bot generate-orders data/raw/candidate_symbols.csv --as-of 2026-03-17 --require-relative-volume --output-dir data/processed/daily/2026-03-17
```

This command:

- screens the configured universe using the candidate list and the universe thresholds in `config/strategy.yaml`
- fetches enough warmup history to evaluate the breakout, ATR, relative-volume, and regime conditions without look-ahead
- generates current breakout signals on the latest available daily bar
- runs each signal through the existing risk sizing and portfolio checks
- writes a human-readable `manual_order_sheet.csv` plus `daily_signal_report.json` and `daily_signal_report.csv`

If `--portfolio-file` is supplied, the workflow treats those holdings as currently open positions when enforcing max positions, duplicate-entry blocking, and no-averaging-down checks. If no portfolio file is supplied, the behavior stays the same as before and the CLI assumes there are no current holdings.
Relative-volume confirmation is optional by default. Use `--require-relative-volume` to make it a hard entry gate; the generated reports and order sheet label whether RV was `optional` or `required` for each candidate.

The portfolio snapshot can be CSV or JSON and should include at least:

- `symbol`
- `quantity`
- `average_entry_price`

Optional fields:

- `current_stop`
- `preset_name`
- `source`
- `metadata_json` for CSV or `metadata` for JSON

Example CSV:

```csv
symbol,quantity,average_entry_price,current_stop,preset_name,source
MU,50,96.25,90.00,confirmed_breakout,investopedia
```

Example JSON:

```json
{
  "positions": [
    {
      "symbol": "MU",
      "quantity": 50,
      "average_entry_price": 96.25,
      "current_stop": 90.0,
      "preset_name": "confirmed_breakout",
      "source": "investopedia"
    }
  ]
}
```

To create and maintain that snapshot locally:

```bash
investopedia-bot init-portfolio data/processed/portfolio/current_positions.csv
investopedia-bot upsert-position data/processed/portfolio/current_positions.csv MU --quantity 50 --average-entry-price 96.25 --current-stop 90 --preset-name confirmed_breakout --source investopedia
investopedia-bot upsert-position data/processed/portfolio/current_positions.json NVDA --quantity 10 --average-entry-price 870 --metadata-json '{"note":"starter"}'
investopedia-bot update-stop data/processed/portfolio/current_positions.csv MU --current-stop 92
investopedia-bot remove-position data/processed/portfolio/current_positions.csv MU
```

`init-portfolio` creates an empty CSV or JSON snapshot compatible with `--portfolio-file`. `upsert-position` appends a new symbol or replaces the existing row/object for that symbol. `update-stop` changes `current_stop` for an existing symbol, and `remove-position` deletes a symbol from the snapshot after an exit. Both commands fail clearly if the symbol is missing.

### Render manual orders from offline signals

```bash
investopedia-bot render-orders signals/orders.csv
investopedia-bot render-orders signals/orders.csv --as-of 2026-03-17
investopedia-bot render-orders signals/orders.csv --output data/processed/orders/today.csv
```

The input CSV is intentionally execution-agnostic. Required columns:

- `symbol`
- `side`
- `quantity`

Optional columns:

- `order_type` (`MARKET`, `LIMIT`, or `STOP_LIMIT`)
- `limit_price`
- `stop_price`
- `time_in_force`
- `strategy_name`
- `thesis`

Example:

```csv
symbol,side,quantity,order_type,limit_price,stop_price,time_in_force,strategy_name,thesis
NVDA,BUY,10,MARKET,,,DAY,breakout_momentum,20-day breakout with strong volume
MSFT,SELL,5,LIMIT,420.00,,DAY,rebalance,Trim exposure after earnings gap
```

The CLI writes a normalized manual order blotter to `data/processed/orders/` by default.

### Run a daily-bar backtest

```bash
investopedia-bot backtest data/raw/candidate_symbols.txt --start 2025-01-01 --end 2025-03-31
investopedia-bot backtest data/raw/candidate_symbols.txt --start 2025-01-01 --end 2025-03-31 --require-relative-volume --output-dir data/processed/backtests/q1_2025
investopedia-bot backtest data/raw/candidate_symbols.csv --start 2025-01-01 --end 2025-03-31 --disable-regime-filter --format json
```

The backtest command:

- fetches a warmup window before the requested start date so breakout, ATR, and benchmark moving averages can initialize cleanly
- generates signals on the close and fills approved entries on the next bar open
- applies configured commissions and slippage from `config/game_rules.yaml`
- manages ATR-based initial and trailing stops
- optionally writes `trade_log.csv`, `equity_curve.csv`, and a summary file to the requested output directory

### Run walk-forward validation

```bash
investopedia-bot walkforward data/raw/candidate_symbols.txt --start 2024-01-01 --end 2025-12-31 --train-days 252 --test-days 63
investopedia-bot walkforward data/raw/candidate_symbols.txt --start 2024-01-01 --end 2025-12-31 --train-days 252 --test-days 63 --expanding-train --breakout-lookbacks 20,40 --initial-stop-atrs 2.0,2.5 --trailing-stop-atrs 2.5,3.0 --risk-per-trade-values 0.005,0.01
investopedia-bot walkforward data/raw/candidate_symbols.csv --start 2024-01-01 --end 2025-12-31 --train-days 252 --test-days 63 --objective sharpe_ratio --output-dir data/processed/walkforward/two_year_review
```

This command:

- generates rolling or expanding train/test folds over the requested date range
- sweeps the requested breakout and risk parameter combinations
- records train and out-of-sample test metrics for every fold and parameter set
- writes `fold_metrics`, `aggregate_metrics`, `selected_fold_metrics`, and `best_parameter_sets` in both CSV and JSON formats

If sweep arguments are omitted, the command uses the current single values from `config/strategy.yaml`.

### Compare named strategy presets

```bash
investopedia-bot compare-strategies data/raw/candidate_symbols.txt --start 2025-01-01 --end 2025-03-31
investopedia-bot compare-strategies data/raw/candidate_symbols.txt --start 2025-01-01 --end 2025-03-31 --preset-names conservative_breakout,confirmed_conservative_breakout,confirmed_breakout --objective sharpe_ratio
investopedia-bot compare-strategies data/raw/candidate_symbols.txt --start 2025-01-01 --end 2025-03-31 --preset "name=research_fast,breakout_lookback=15,relative_volume_threshold=1.2,initial_stop_atr=2.0,trailing_stop_atr=2.5,risk_per_trade=0.012,require_relative_volume_confirmation=true"
```

This command:

- compares named breakout/risk presets over the same symbols and date range
- reuses the deterministic backtest engine and the existing summary metric logic
- writes `comparison_results`, `ranked_presets`, and `summary.json` in machine-readable formats

Built-in presets are `conservative_breakout`, `confirmed_conservative_breakout`, `standard_breakout`, `confirmed_breakout`, and `aggressive_breakout`.
`confirmed_conservative_breakout` matches `conservative_breakout` except that relative-volume confirmation is a hard entry gate.
`confirmed_breakout` matches `standard_breakout` except that relative-volume confirmation is a hard entry gate.
You can also add repo-local presets in `config/strategy.yaml` with an optional `comparison_presets` section:

```yaml
comparison_presets:
  research_breakout:
    breakout_lookback: 30
    relative_volume_threshold: 1.7
    initial_stop_atr: 2.7
    trailing_stop_atr: 3.3
    risk_per_trade: 0.008
    require_relative_volume_confirmation: true
```

### Generate a daily research summary

```bash
investopedia-bot daily-summary data/raw/candidate_symbols.txt --as-of 2026-03-17
investopedia-bot daily-summary data/raw/candidate_symbols.txt --as-of 2026-03-17 --preset-names conservative_breakout,confirmed_conservative_breakout
investopedia-bot daily-summary data/raw/candidate_symbols.txt --as-of 2026-03-17 --preset-names conservative_breakout,confirmed_conservative_breakout --portfolio-file data/processed/portfolio/current_positions.json
investopedia-bot daily-summary data/raw/candidate_symbols.txt --as-of 2026-03-17 --comparison-results data/processed/strategy_comparison/2025-01-01_2025-03-31/ranked_presets.csv
```

This command:

- screens the universe, evaluates one or more presets, and applies the existing risk checks
- ranks approved and rejected opportunities with a deterministic score built from breakout strength, relative-volume confirmation, and position size as a percent of equity
- writes a consolidated `daily_summary.json`, row-level `ranked_opportunities.csv`, preset-level `preset_rankings.csv/json`, and a `suggested_order_sheet.csv/json`

If no preset selection is supplied, the command defaults to `standard_breakout`. If `--comparison-results` is supplied, the top preset from that file is included automatically.
Like `generate-orders`, relative-volume confirmation is optional unless `--require-relative-volume` is enabled, and the ranked outputs preserve that policy in their rationale text and metadata. When `--portfolio-file` is provided, the summary also includes current holdings context and rejection reasons for duplicate entries or averaging-down attempts.

## Architecture Notes

- Signal generation should stay separate from execution. Research code can emit candidate trades, while execution modules only format or route orders.
- The data layer normalizes every provider into one daily-bar schema before strategy or backtest code sees it.
- Cache files are plain CSVs so they are easy to inspect, delete, or regenerate.
- Daily-bar assumptions are first-class. The repo is meant for end-of-day research and next-session decision support, not intraday automation.
- Simulator-specific rules live in config, not in strategy logic. That keeps backtests and manual order prep aligned.
- The backtest engine uses decision-on-close and fill-on-next-open semantics to avoid look-ahead bias. Trailing stops only update after the close and become active on the following session.
- The manual execution layer is intentionally report-first. It produces human-readable orders and research artifacts, but it does not submit anything to Investopedia yet.
- Walk-forward tooling is the main robustness check. Prefer parameter sets that remain acceptable across many test folds over ones that win one in-sample run.
- Strategy comparison is a faster preset-level screen. Use it to narrow candidates, then confirm the survivors with walk-forward outputs before trusting a setting.
- The daily summary is the short-horizon decision layer. Use robust presets from strategy comparison and walk-forward first, then use the daily summary to decide whether today’s setups are worth placing manually.
- Browser automation and any live web executor are intentionally out of scope for this baseline.

## Development

```bash
pytest
ruff check .
mypy src
```
