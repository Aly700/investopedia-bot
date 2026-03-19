#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
  AS_OF_DATE="$1"
  shift
else
  AS_OF_DATE="${INVESTOPEDIA_BOT_AS_OF:-$(date '+%F')}"
fi

PYTHON_BIN="${INVESTOPEDIA_BOT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
CONFIG_DIR="${INVESTOPEDIA_BOT_CONFIG_DIR:-$REPO_ROOT/config}"
ENV_FILE="${INVESTOPEDIA_BOT_ENV_FILE:-$REPO_ROOT/.env}"
CANDIDATE_FILE="${INVESTOPEDIA_BOT_CANDIDATE_FILE:-$REPO_ROOT/data/raw/candidate_symbols.txt}"
PORTFOLIO_FILE="${INVESTOPEDIA_BOT_PORTFOLIO_FILE:-$REPO_ROOT/data/processed/portfolio/current_positions.json}"
PRESET_NAMES="${INVESTOPEDIA_BOT_PRESET_NAMES:-standard_breakout}"
OUTPUT_BASE="${INVESTOPEDIA_BOT_OUTPUT_BASE:-$REPO_ROOT/data/processed/monitor_market}"
OUTPUT_DIR="$OUTPUT_BASE/$AS_OF_DATE"
MONITOR_TEXT_PATH="$OUTPUT_DIR/market_monitor.txt"
NOTIFY_ENABLED="${INVESTOPEDIA_BOT_NOTIFY:-true}"
NOTIFY_ON_WATCH="${INVESTOPEDIA_BOT_NOTIFY_ON_WATCH:-true}"
NOTIFY_SOUND="${INVESTOPEDIA_BOT_NOTIFY_SOUND:-default}"
NOTIFY_TITLE="${INVESTOPEDIA_BOT_NOTIFY_TITLE:-Investopedia Bot}"
NOTIFY_GROUP="${INVESTOPEDIA_BOT_NOTIFY_GROUP:-investopedia-bot-monitor-market}"
TERMINAL_NOTIFIER_BIN="${INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$CANDIDATE_FILE" ]; then
  echo "Candidate file does not exist: $CANDIDATE_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

alert_count_present() {
  local category="$1"
  if [ ! -f "$MONITOR_TEXT_PATH" ]; then
    return 1
  fi
  grep -Eq "^${category}: [1-9][0-9]*$" "$MONITOR_TEXT_PATH"
}

summarize_alert_counts() {
  local pattern="$1"
  local summary=""

  while IFS= read -r line; do
    if [ -z "$summary" ]; then
      summary="$line"
    else
      summary="$summary; $line"
    fi
  done < <(grep -E "$pattern" "$MONITOR_TEXT_PATH")

  printf '%s\n' "$summary"
}

resolve_terminal_notifier() {
  if [ -n "$TERMINAL_NOTIFIER_BIN" ] && [ -x "$TERMINAL_NOTIFIER_BIN" ]; then
    printf '%s\n' "$TERMINAL_NOTIFIER_BIN"
    return 0
  fi

  if [ -x "/opt/homebrew/bin/terminal-notifier" ]; then
    printf '%s\n' "/opt/homebrew/bin/terminal-notifier"
    return 0
  fi

  if [ -x "/usr/local/bin/terminal-notifier" ]; then
    printf '%s\n' "/usr/local/bin/terminal-notifier"
    return 0
  fi

  command -v terminal-notifier 2>/dev/null || true
}

send_notification() {
  local subtitle="$1"
  local message="$2"
  local notifier_bin

  if ! is_truthy "$NOTIFY_ENABLED"; then
    return 0
  fi

  notifier_bin="$(resolve_terminal_notifier)"
  if [ -z "$notifier_bin" ]; then
    echo "terminal-notifier not found; skipping local notification." >&2
    return 0
  fi

  local args=(
    -title "$NOTIFY_TITLE"
    -subtitle "$subtitle"
    -message "$message"
    -group "$NOTIFY_GROUP"
    -open "file://$MONITOR_TEXT_PATH"
  )

  if [ -n "$NOTIFY_SOUND" ] && [ "$NOTIFY_SOUND" != "none" ]; then
    args+=(-sound "$NOTIFY_SOUND")
  fi

  "$notifier_bin" "${args[@]}" >/dev/null 2>&1 || \
    echo "terminal-notifier returned a non-zero exit status; continuing without failing the monitor run." >&2
}

maybe_notify() {
  if [ ! -f "$MONITOR_TEXT_PATH" ]; then
    echo "Monitor text summary not found at $MONITOR_TEXT_PATH; skipping local notification." >&2
    return 0
  fi

  if alert_count_present "BUY CANDIDATE" || \
    alert_count_present "RAISE STOP" || \
    alert_count_present "EXIT CANDIDATE"; then
    send_notification \
      "Actionable monitor alerts" \
      "$(summarize_alert_counts '^(BUY CANDIDATE|RAISE STOP|EXIT CANDIDATE): ')"
    return 0
  fi

  if is_truthy "$NOTIFY_ON_WATCH" && alert_count_present "WATCH CLOSELY"; then
    send_notification \
      "Watch closely" \
      "$(summarize_alert_counts '^WATCH CLOSELY: ')"
  fi
}

COMMAND=(
  "$PYTHON_BIN"
  -m
  bot.main
  --config-dir
  "$CONFIG_DIR"
  --env-file
  "$ENV_FILE"
  monitor-market
  "$CANDIDATE_FILE"
  --as-of
  "$AS_OF_DATE"
  --preset-names
  "$PRESET_NAMES"
  --output-dir
  "$OUTPUT_DIR"
  --format
  json
)

if [ -n "$PORTFOLIO_FILE" ] && [ -f "$PORTFOLIO_FILE" ]; then
  COMMAND+=(--portfolio-file "$PORTFOLIO_FILE")
elif [ -n "$PORTFOLIO_FILE" ]; then
  echo "Portfolio snapshot not found, continuing without --portfolio-file: $PORTFOLIO_FILE" >&2
fi

if [ "$#" -gt 0 ]; then
  COMMAND+=("$@")
fi

if env PYTHONPATH="$REPO_ROOT/src" "${COMMAND[@]}"; then
  RUN_STATUS=0
else
  RUN_STATUS=$?
fi

if [ "$RUN_STATUS" -eq 0 ]; then
  maybe_notify
fi

exit "$RUN_STATUS"
