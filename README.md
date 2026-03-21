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
config/                  YAML configuration for strategy, universe-builder, data sources, and simulator rules
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

The repo uses four YAML files under `config/`:

- `strategy.yaml`: universe filters, signal parameters, and per-trade risk settings
- `universe.yaml`: master-universe settings, profile filters, and universe-builder output paths
- `data_sources.yaml`: active research data provider and the environment variable name for each API key
- `game_rules.yaml`: simulator cash, commissions, fill assumptions, and account-level constraints

The CLI loads and validates these files through `src/bot/config.py`. Config is kept separate from execution so strategy research, backtests, and order generation can share the same baseline assumptions.

Deployment note: container and cloud runtimes need the full `config/` directory copied alongside the bot runtime, not just the Python package files. If the runtime root is not the current working directory, pass `--config-dir /path/to/config`; the loader will treat that directory's parent as the app root for relative `data/` outputs.

Historical daily-bar data is selected by `config/data_sources.yaml`. Local cache files are written under `data/cache/daily_bars/<provider>/`.

## CLI Usage

After installation, use either the console script or `python -m bot.main`.

### Developer shortcuts

For normal local use, the repo includes small bash wrappers under `scripts/` plus shared defaults in `scripts/common.sh`.

Start by copying the local template:

```bash
cp .env.local.example .env.local
```

Then use the short commands:

```bash
./scripts/build_quality_universe.sh
./scripts/monitor_quality.sh
./scripts/daily_quality_summary.sh
./scripts/review_portfolio.sh
./scripts/upsert_position.sh LNG 67 281.87 259.55
```

You can temporarily override the default date or any other helper variable without editing files:

```bash
BOT_DATE=2026-03-20 ./scripts/monitor_quality.sh
```

If you prefer `make`, equivalent shortcuts are available:

```bash
make build-quality
make monitor
make summary
make review
make test-fast
```

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

### Build and refresh candidate universes

```bash
investopedia-bot build-universe --profile broad_momentum
investopedia-bot build-universe --profile growth_momentum
investopedia-bot build-universe --profile quality_liquid
investopedia-bot build-universe --profile semis_ai --profile growth_software --as-of 2026-03-17
investopedia-bot build-universe --master-input data/processed/universe/master_universe.csv --profile large_cap_liquid --format json
```

This command is the staged universe-builder pipeline. It:

- fetches or loads a broad master universe
- enriches symbols with reference metadata plus recent price and liquidity metrics
- applies one or more named profile filters from `config/universe.yaml`
- writes fresh plain-text candidate files that stay compatible with the rest of the bot

By default it writes:

- `data/processed/universe/master_reference.csv`
- `data/processed/universe/master_universe.csv`
- `data/processed/universe/profiles/<profile>.json`
- `data/raw/candidate_symbols_<profile>.txt`

The generated text files are plain one-symbol-per-line watchlists, so they plug directly into the existing commands:

```bash
investopedia-bot daily-summary data/raw/candidate_symbols_broad_momentum.txt --as-of 2026-03-17
investopedia-bot monitor-market data/raw/candidate_symbols_semis_ai.txt --as-of 2026-03-17 --portfolio-file data/processed/portfolio/current_positions.json
investopedia-bot compare-strategies data/raw/candidate_symbols_large_cap_liquid.txt --start 2025-01-01 --end 2025-03-31
```

The pipeline is cache-friendly and deterministic for a given master input or cached refresh:

1. stage 1 fetches the broad reference list into `data/cache/reference_universe/`
2. stage 2 enriches the master file with metadata and daily-bar liquidity metrics
3. stage 3 applies profile filters and writes candidate outputs

Universe profile rules live in `config/universe.yaml`. Each profile can define:

- `allowed_exchanges`
- `allowed_ticker_types`
- `active_only`
- `common_stock_only`
- `include_etfs`
- `include_adrs`
- `min_price`
- `min_average_daily_volume`
- `min_dollar_volume`
- `min_market_cap`
- `allowed_sectors`
- `allowed_industries`
- `include_symbols`
- `exclude_symbols`
- `preferred_symbol_groups`
- `max_symbols`

`allowed_sectors` and `allowed_industries` are case-insensitive substring matches, which makes thematic profiles like semis or software practical even when provider metadata is verbose.
`preferred_symbol_groups` lets a profile collapse obvious dual-class duplicates, such as keeping `GOOGL` over `GOOG`, without hardcoding ticker-specific logic in the pipeline.
`growth_momentum` is intended to sit between `broad_momentum` and the narrower thematic profiles: it keeps the universe liquid and growth-heavy, leans into semis, software, cloud, cybersecurity, internet platforms, and networking/AI infrastructure, and still emits a plain-text candidate file for the existing bot commands.
`quality_liquid` is the more conservative large-universe option: it stays cross-sector, raises the market-cap and liquidity bars above `broad_momentum`, and uses a small cleanup blacklist to keep the output cleaner for lower-noise scans.

To add a new profile, copy one of the existing entries in `config/universe.yaml`, rename it, and adjust the filters:

```yaml
profiles:
  software_cloud:
    description: Cloud software names with liquidity and market-cap gates.
    active_only: true
    allowed_exchanges: [XNAS, XNYS, ARCX]
    allowed_ticker_types: [CS]
    common_stock_only: true
    include_etfs: false
    include_adrs: false
    min_price: 10
    min_average_daily_volume: 250000
    min_dollar_volume: 7500000
    min_market_cap: 1000000000
    allowed_industries: [software, cloud, data processing]
    include_symbols: [CRM, NOW, SNOW]
    exclude_symbols: []
    max_symbols: 175
```

The manual override layer is built into every profile:

- `include_symbols` force-adds names even if they fail the normal filters
- `exclude_symbols` force-removes names without editing generated files by hand

Generated candidate files keep provider-native symbols exactly as written. Symbols with punctuation such as `BRK.B` are preserved in the text output, and the downstream candidate loader and provider calls do not rewrite them.

If you already have an enriched master CSV and only want to re-run profile filters, pass `--master-input` to skip the reference-fetch stage.
For compatibility, the older screen-only behavior still works if you pass a text or CSV candidate file positionally:

```bash
investopedia-bot build-universe data/raw/candidate_symbols.txt --as-of 2026-03-17 --format text
```

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

### Review existing holdings

```bash
investopedia-bot review-portfolio --portfolio-file data/processed/portfolio/current_positions.csv --as-of 2026-03-17
investopedia-bot review-portfolio --portfolio-file data/processed/portfolio/current_positions.json --as-of 2026-03-17 --benchmark-symbol SPY
```

This command:

- loads the current portfolio snapshot and fetches fresh daily bars for each open position
- computes latest close, unrealized P/L percent, distance to the current stop, and regime-filter status
- checks whether the current ATR trailing-stop logic would justify a higher stop without ever suggesting a lower one
- writes `portfolio_review.json` and `portfolio_review.csv` with management suggestions such as `HOLD`, `WATCH CLOSELY`, `RAISE STOP`, or `EXIT CANDIDATE`

The intended daily workflow is:

1. update the portfolio snapshot after fills, stop changes, or exits
2. run `review-portfolio` to manage existing holdings
3. run `daily-summary` or `generate-orders` to evaluate fresh entries against the same snapshot

### Review existing holdings intraday

```bash
investopedia-bot review-portfolio-intraday --portfolio-file data/processed/portfolio/current_positions.csv --as-of 2026-03-20 --interval-minutes 15
investopedia-bot review-portfolio-intraday --portfolio-file data/processed/portfolio/current_positions.csv --as-of 2026-03-20 --interval-minutes 15 --benchmark-symbol SPY
```

This command is the scheduled intraday sell-monitoring layer for held positions only. It does not scan new buys or change sizing. Instead, it fetches one session of intraday aggregate bars for each open position and checks:

- intraday stop breaches
- session-high giveback after a profitable move
- failed intraday strength, including fades back below VWAP or the session open
- intraday momentum fade and simple benchmark-relative weakness

It writes:

- `portfolio_review_intraday.json`
- `portfolio_review_intraday.csv`
- `portfolio_review_intraday_brief.txt`

The brief is the decision-first human-readable layer. It groups names into:

- `Urgent intraday actions`
- `Current holdings under pressure`
- `Holdings still healthy`

Use it as a market-hours polling job, for example every 15 minutes during the session. Keep the existing daily `review-portfolio` path for end-of-day confirmation and stop-raising logic.

The intraday review is intentionally separate from `monitor-market`:

- `review-portfolio-intraday` is for held-position monitoring only
- it does not scan for new buy candidates
- it is suitable for a separate market-hours LaunchAgent that polls existing positions during the session

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

### Monitor the market on a schedule

```bash
investopedia-bot monitor-market data/raw/candidate_symbols.txt --as-of 2026-03-17
investopedia-bot monitor-market data/raw/candidate_symbols.txt --as-of 2026-03-17 --portfolio-file data/processed/portfolio/current_positions.json --preset-names conservative_breakout,confirmed_conservative_breakout
investopedia-bot monitor-market data/raw/candidate_symbols.txt --as-of 2026-03-17 --portfolio-file data/processed/portfolio/current_positions.json --output-dir data/processed/monitor_market/2026-03-17
```

This command:

- reuses the existing `daily-summary` scan to find approved `BUY CANDIDATE` entries
- reuses `review-portfolio` logic to classify current holdings as `HOLD`, `WATCH CLOSELY`, `RAISE STOP`, or `EXIT CANDIDATE`
- combines both into a compact machine-readable alert payload
- writes `market_monitor.json`, `market_monitor.csv`, the existing raw `market_monitor.txt`, and a human-readable `market_monitor_brief.txt` for later notification delivery

This is the background-friendly layer for automation. On macOS, prefer `launchd` over `cron` or a constantly running shell loop, then hand the JSON or text file to a future email, Discord, Telegram, or other notifier without changing the trading logic itself.

Typical flow:

1. update the portfolio snapshot after fills, stop changes, or exits
2. run `monitor-market` on a schedule
3. inspect the alert summary or route the generated files into a later notification step

### macOS wrapper and launchd examples

The repo includes a small wrapper at `scripts/monitor_market.sh`. It runs the existing `monitor-market` command, defaults `--as-of` to today, writes outputs to `data/processed/monitor_market/YYYY-MM-DD/`, and is safe to call from `launchd`.

Default wrapper inputs:

- `INVESTOPEDIA_BOT_CANDIDATE_FILE`: `data/raw/candidate_symbols.txt`
- `INVESTOPEDIA_BOT_PORTFOLIO_FILE`: `data/processed/portfolio/current_positions.json`
- `INVESTOPEDIA_BOT_PRESET_NAMES`: `standard_breakout`
- `INVESTOPEDIA_BOT_OUTPUT_BASE`: `data/processed/monitor_market`
- `INVESTOPEDIA_BOT_NOTIFY`: `true`
- `INVESTOPEDIA_BOT_NOTIFY_DISCORD`: `true`
- `INVESTOPEDIA_BOT_NOTIFY_LOCAL`: `true`
- `INVESTOPEDIA_BOT_NOTIFY_ON_WATCH`: `true`
- `INVESTOPEDIA_BOT_NOTIFY_SOUND`: `default`
- `INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN`: optional explicit notifier path
- `INVESTOPEDIA_BOT_DISCORD_WEBHOOK`: optional Discord webhook URL

For Discord alerts, create a webhook in your target server/channel:

1. open the channel settings in Discord
2. go to `Integrations`
3. create a new webhook
4. copy the webhook URL
5. paste it into `INVESTOPEDIA_BOT_DISCORD_WEBHOOK` in the LaunchAgent plist or export it before running the wrapper manually

For local macOS notifications, install `terminal-notifier` once:

```bash
brew install terminal-notifier
```

On Apple Silicon Macs, Homebrew commonly installs `terminal-notifier` at `/opt/homebrew/bin/terminal-notifier`. `launchd` jobs usually do not inherit the same `PATH` as an interactive Terminal shell, so a command that works in Terminal may still be missing when the LaunchAgent runs.

After `monitor-market` finishes, the wrapper still checks the category counts in `market_monitor.txt` to preserve the current trigger rules:

- sends an actionable notification when `BUY CANDIDATE`, `RAISE STOP`, or `EXIT CANDIDATE` is greater than zero
- sends a lower-priority notification for `WATCH CLOSELY` when there are no actionable alerts and `INVESTOPEDIA_BOT_NOTIFY_ON_WATCH=true`
- sends no notification for pure `NO ACTION` runs by default

Discord webhook alerts are the primary remote path in the wrapper. If `INVESTOPEDIA_BOT_DISCORD_WEBHOOK` is set and `INVESTOPEDIA_BOT_NOTIFY_DISCORD=true`, the wrapper prefers `market_monitor_brief.txt` for the human-facing message and falls back to `market_monitor.txt` if the brief is unavailable.
When the brief exists, the Discord payload uses a compact version of:

- the headline summary
- the best-actions section
- the top buy candidates
- the current-holdings section

The wrapper keeps the message compact and trims it before sending so it stays readable in Discord.
Local macOS notifications follow the same preference order: brief first when present, raw count-based fallback when not.

The wrapper resolves `terminal-notifier` in this order:

1. `INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN`, if set and executable
2. `/opt/homebrew/bin/terminal-notifier`
3. `/usr/local/bin/terminal-notifier`
4. `command -v terminal-notifier`

If no local notifier is found, the wrapper does not fail the monitor run; it logs a short message to stderr and continues.
If the Discord webhook is missing or the delivery fails, the monitor run also continues without failing.

Run it manually:

```bash
chmod +x scripts/monitor_market.sh
./scripts/monitor_market.sh
./scripts/monitor_market.sh 2026-03-17
INVESTOPEDIA_BOT_CANDIDATE_FILE="$PWD/data/raw/candidate_symbols.txt" \
INVESTOPEDIA_BOT_PORTFOLIO_FILE="$PWD/data/processed/portfolio/current_positions.json" \
INVESTOPEDIA_BOT_PRESET_NAMES="conservative_breakout,confirmed_conservative_breakout" \
INVESTOPEDIA_BOT_DISCORD_WEBHOOK="https://discord.com/api/webhooks/REPLACE_ME/REPLACE_ME" \
INVESTOPEDIA_BOT_NOTIFY_DISCORD="true" \
INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN="/opt/homebrew/bin/terminal-notifier" \
INVESTOPEDIA_BOT_NOTIFY_SOUND="default" \
./scripts/monitor_market.sh
```

You can also pass extra CLI flags after the optional date:

```bash
./scripts/monitor_market.sh 2026-03-17 --current-drawdown 0.05
```

Three example LaunchAgents are provided:

- `ops/launchd/com.investopedia.bot.monitor-market.market-hours.plist`
- `ops/launchd/com.investopedia.bot.monitor-market.after-close.plist`
- `ops/launchd/com.investopedia.bot.monitor-market.market-open.plist`
- `ops/launchd/com.investopedia.bot.review-portfolio-intraday.market-hours.plist`

Recommended setup: `com.investopedia.bot.monitor-market.market-hours.plist`.
It keeps `monitor-market` as a short batch job, but runs it automatically several times per weekday with `StartCalendarInterval` entries at:

- `09:35`
- `11:00`
- `13:00`
- `15:30`
- `16:15`

That gives you automatic background monitoring during market hours without redesigning the bot into a permanently running daemon or shell loop.

The single-run examples are still available:

- `market-open`: weekdays at `09:35`
- `after-close`: weekdays at `16:15`

All repo plist examples are templates. They intentionally keep `__REPO_ROOT__` and `REPLACE_ME` placeholders so machine-local paths and secrets do not live in the repo. Customize the copied plist in `~/Library/LaunchAgents/` before loading it.

Install the recommended market-hours LaunchAgent:

```bash
REPO_ROOT="$PWD"
AGENT_NAME="com.investopedia.bot.monitor-market.market-hours"
PLIST_SRC="$REPO_ROOT/ops/launchd/$AGENT_NAME.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$AGENT_NAME.plist"

mkdir -p "$REPO_ROOT/data/logs/launchd" "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
perl -0pi -e "s#__REPO_ROOT__#$REPO_ROOT#g" "$PLIST_DST"

# Edit the copied plist locally before loading it.
# Replace REPLACE_ME values such as INVESTOPEDIA_BOT_DISCORD_WEBHOOK,
# or set INVESTOPEDIA_BOT_NOTIFY_DISCORD=false if you do not want Discord alerts.

plutil -lint "$PLIST_DST"
launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$AGENT_NAME"
launchctl kickstart -k "gui/$(id -u)/$AGENT_NAME"
```

The copied plist in `~/Library/LaunchAgents/` is the live file. Safe values to customize locally include:

- `INVESTOPEDIA_BOT_CANDIDATE_FILE`
- `INVESTOPEDIA_BOT_PORTFOLIO_FILE`
- `INVESTOPEDIA_BOT_PRESET_NAMES`
- `INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN`
- `INVESTOPEDIA_BOT_DISCORD_WEBHOOK`

If you only want a single checkpoint instead of the full weekday schedule, repeat the same copy/load flow with `AGENT_NAME` set to either `com.investopedia.bot.monitor-market.market-open` or `com.investopedia.bot.monitor-market.after-close`.

Reload the recommended LaunchAgent after editing its schedule or environment variables:

```bash
AGENT_NAME="com.investopedia.bot.monitor-market.market-hours"
PLIST_DST="$HOME/Library/LaunchAgents/$AGENT_NAME.plist"
plutil -lint "$PLIST_DST"
launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$AGENT_NAME"
launchctl kickstart -k "gui/$(id -u)/$AGENT_NAME"
```

Disable or unload the recommended LaunchAgent:

```bash
AGENT_NAME="com.investopedia.bot.monitor-market.market-hours"
PLIST_DST="$HOME/Library/LaunchAgents/$AGENT_NAME.plist"
launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl disable "gui/$(id -u)/$AGENT_NAME"
```

Change the schedule later by editing the `StartCalendarInterval` array in the copied plist:

1. open `~/Library/LaunchAgents/com.investopedia.bot.monitor-market.market-hours.plist`
2. add, remove, or adjust weekday `Hour` and `Minute` entries
3. run `plutil -lint` on the edited plist
4. reload it with the `launchctl bootout` and `launchctl bootstrap` commands above

Each scheduled time should have one entry per weekday you want it to run. The repo example already includes Monday through Friday entries for all five recommended checkpoints.

Disable notifications without disabling the LaunchAgent:

- set `INVESTOPEDIA_BOT_NOTIFY=false` in the LaunchAgent `EnvironmentVariables`
- or set `INVESTOPEDIA_BOT_NOTIFY_DISCORD=false` to suppress only Discord delivery
- or set `INVESTOPEDIA_BOT_NOTIFY_LOCAL=false` to suppress only local macOS notifications
- or leave the LaunchAgent enabled and uninstall `terminal-notifier`

Outputs and logs:

- monitor outputs: `data/processed/monitor_market/YYYY-MM-DD/market_monitor.json`
- CSV/text summaries: `data/processed/monitor_market/YYYY-MM-DD/market_monitor.csv`, `market_monitor.txt`, and `market_monitor_brief.txt`
- recommended market-hours logs: `data/logs/launchd/monitor-market.market-hours.out.log` and `monitor-market.market-hours.err.log`
- optional single-run logs: `data/logs/launchd/monitor-market.after-close.out.log`, `monitor-market.after-close.err.log`, `monitor-market.market-open.out.log`, and `monitor-market.market-open.err.log`

### Separate intraday portfolio-review wrapper and LaunchAgent

The repo also includes a separate wrapper at `scripts/review_portfolio_intraday.sh`. It is intentionally not merged into `monitor-market`.

It:

- runs `review-portfolio-intraday`
- defaults the market date using `America/New_York`
- uses the existing portfolio CSV at `data/processed/portfolio/current_positions.csv`
- writes outputs under `data/processed/portfolio_review_intraday/YYYY-MM-DD/`

Run it manually:

```bash
chmod +x scripts/review_portfolio_intraday.sh
./scripts/review_portfolio_intraday.sh
BOT_DATE=2026-03-20 ./scripts/review_portfolio_intraday.sh
PYTHONPATH=src .venv/bin/python -m bot.main review-portfolio-intraday --portfolio-file data/processed/portfolio/current_positions.csv --as-of 2026-03-20 --interval-minutes 15
```

The separate LaunchAgent template is:

- `ops/launchd/com.investopedia.bot.review-portfolio-intraday.market-hours.plist`

Like the monitor templates, it keeps `__REPO_ROOT__` placeholders in the repo. Replace them only in the copied live plist under `~/Library/LaunchAgents/`.

Install the intraday review LaunchAgent:

```bash
REPO_ROOT="$PWD"
AGENT_NAME="com.investopedia.bot.review-portfolio-intraday.market-hours"
PLIST_SRC="$REPO_ROOT/ops/launchd/$AGENT_NAME.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$AGENT_NAME.plist"

mkdir -p "$REPO_ROOT/data/logs/launchd" "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
perl -0pi -e "s#__REPO_ROOT__#$REPO_ROOT#g" "$PLIST_DST"

plutil -lint "$PLIST_DST"
launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$AGENT_NAME"
launchctl kickstart -k "gui/$(id -u)/$AGENT_NAME"
```

Inspect, test, and disable it:

```bash
AGENT_NAME="com.investopedia.bot.review-portfolio-intraday.market-hours"
PLIST_DST="$HOME/Library/LaunchAgents/$AGENT_NAME.plist"

plutil -lint "$PLIST_DST"
launchctl print "gui/$(id -u)/$AGENT_NAME"
launchctl kickstart -k "gui/$(id -u)/$AGENT_NAME"
tail -f "$PWD/data/logs/launchd/review-portfolio-intraday.market-hours.out.log" \
  "$PWD/data/logs/launchd/review-portfolio-intraday.market-hours.err.log"

launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl disable "gui/$(id -u)/$AGENT_NAME"
```

This job is for held-position monitoring only. Keep `monitor-market` as the separate job for buy scans plus portfolio alerts.

## Architecture Notes

- Signal generation should stay separate from execution. Research code can emit candidate trades, while execution modules only format or route orders.
- The data layer normalizes every provider into one daily-bar schema before strategy or backtest code sees it.
- Cache files are plain CSVs so they are easy to inspect, delete, or regenerate.
- Daily bars remain the primary research timeframe. Intraday support is intentionally limited to scheduled held-position monitoring and does not turn the repo into a streaming executor.
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
