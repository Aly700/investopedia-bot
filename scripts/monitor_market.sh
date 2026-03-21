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
MONITOR_BRIEF_PATH="$OUTPUT_DIR/market_monitor_brief.txt"
NOTIFY_ENABLED="${INVESTOPEDIA_BOT_NOTIFY:-true}"
NOTIFY_LOCAL="${INVESTOPEDIA_BOT_NOTIFY_LOCAL:-true}"
NOTIFY_DISCORD="${INVESTOPEDIA_BOT_NOTIFY_DISCORD:-true}"
NOTIFY_ON_WATCH="${INVESTOPEDIA_BOT_NOTIFY_ON_WATCH:-true}"
NOTIFY_SOUND="${INVESTOPEDIA_BOT_NOTIFY_SOUND:-default}"
NOTIFY_TITLE="${INVESTOPEDIA_BOT_NOTIFY_TITLE:-Investopedia Bot}"
NOTIFY_GROUP="${INVESTOPEDIA_BOT_NOTIFY_GROUP:-investopedia-bot-monitor-market}"
TERMINAL_NOTIFIER_BIN="${INVESTOPEDIA_BOT_TERMINAL_NOTIFIER_BIN:-}"
DISCORD_WEBHOOK="${INVESTOPEDIA_BOT_DISCORD_WEBHOOK:-}"

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

summarize_alert_lines() {
  local pattern="$1"
  local max_lines="${2:-5}"

  grep -E "$pattern" "$MONITOR_TEXT_PATH" | head -n "$max_lines" || true
}

brief_available() {
  [ -f "$MONITOR_BRIEF_PATH" ]
}

extract_brief_section() {
  local heading="$1"

  if ! brief_available; then
    return 0
  fi

  awk -v heading="$heading" '
    $0 == heading { in_section = 1; next }
    in_section && $0 == "" { exit }
    in_section { print }
  ' "$MONITOR_BRIEF_PATH"
}

join_nonempty_lines() {
  local separator="$1"
  local summary=""
  local line

  while IFS= read -r line; do
    if [ -z "$line" ]; then
      continue
    fi
    if [ -z "$summary" ]; then
      summary="$line"
    else
      summary="$summary$separator$line"
    fi
  done

  printf '%s\n' "$summary"
}

brief_section_lines() {
  local heading="$1"
  local max_lines="${2:-3}"

  extract_brief_section "$heading" | head -n "$max_lines" || true
}

brief_section_summary() {
  local heading="$1"
  local max_lines="${2:-2}"

  join_nonempty_lines "; " < <(brief_section_lines "$heading" "$max_lines")
}

truncate_message() {
  local message="$1"
  local max_length="${2:-1500}"

  if [ "${#message}" -le "$max_length" ]; then
    printf '%s\n' "$message"
    return 0
  fi

  printf '%s...\n' "${message:0:$((max_length - 3))}"
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

resolve_curl_bin() {
  if [ -x "/usr/bin/curl" ]; then
    printf '%s\n' "/usr/bin/curl"
    return 0
  fi

  command -v curl 2>/dev/null || true
}

build_discord_message() {
  local heading="$1"
  local counts_pattern="$2"
  local detail_pattern="$3"
  local headline
  local best_actions
  local top_buys
  local current_holdings
  local counts
  local details
  local message

  if brief_available; then
    headline="$(brief_section_summary "Headline" 2)"
    best_actions="$(brief_section_lines "Best actions now" 3)"
    top_buys="$(brief_section_lines "Top buy candidates" 3)"
    current_holdings="$(brief_section_lines "Current holdings" 3)"

    message="**$heading**"
    if [ -n "$headline" ]; then
      message="$message"$'\n'"$headline"
    fi
    if [ -n "$best_actions" ] && [ "$best_actions" != "No immediate action is required." ]; then
      message="$message"$'\n\n'"Best actions now"$'\n'"$best_actions"
    fi
    if [ -n "$top_buys" ] && [ "$top_buys" != "No approved buy candidates." ]; then
      message="$message"$'\n\n'"Top buy candidates"$'\n'"$top_buys"
    fi
    if [ -n "$current_holdings" ] && [ "$current_holdings" != "No held positions were reviewed." ]; then
      message="$message"$'\n\n'"Current holdings"$'\n'"$current_holdings"
    fi

    truncate_message "$message"
    return 0
  fi

  counts="$(summarize_alert_counts "$counts_pattern")"
  details="$(summarize_alert_lines "$detail_pattern" 5)"

  message="**$heading**"$'\n'"Date: $AS_OF_DATE"
  if [ -n "$counts" ]; then
    message="$message"$'\n'"$counts"
  fi
  if [ -n "$details" ]; then
    message="$message"$'\n'"$details"
  fi

  truncate_message "$message"
}

build_local_notification_message() {
  local fallback_message="$1"
  local headline
  local primary_line
  local summary

  if ! brief_available; then
    printf '%s\n' "$fallback_message"
    return 0
  fi

  headline="$(brief_section_summary "Headline" 2)"
  primary_line="$(brief_section_lines "Best actions now" 1 | sed 's/^- //')"

  if [ -z "$primary_line" ] || [ "$primary_line" = "No immediate action is required." ]; then
    primary_line="$(brief_section_lines "Lower-priority names" 1 | sed 's/^- //')"
  fi

  if [ -n "$headline" ] && [ -n "$primary_line" ]; then
    summary="$headline"$'\n'"$primary_line"
  elif [ -n "$headline" ]; then
    summary="$headline"
  elif [ -n "$primary_line" ]; then
    summary="$primary_line"
  else
    summary="$fallback_message"
  fi

  truncate_message "$summary" 220
}

send_discord_notification() {
  local heading="$1"
  local counts_pattern="$2"
  local detail_pattern="$3"
  local curl_bin
  local message
  local payload

  if ! is_truthy "$NOTIFY_ENABLED" || ! is_truthy "$NOTIFY_DISCORD"; then
    return 0
  fi

  if [ -z "$DISCORD_WEBHOOK" ]; then
    return 0
  fi

  curl_bin="$(resolve_curl_bin)"
  if [ -z "$curl_bin" ]; then
    echo "curl not found; skipping Discord webhook notification." >&2
    return 0
  fi

  message="$(build_discord_message "$heading" "$counts_pattern" "$detail_pattern")"
  payload="$("$PYTHON_BIN" - "$message" <<'PY'
import json
import sys

print(json.dumps({"content": sys.argv[1]}))
PY
)"

  "$curl_bin" \
    --silent \
    --show-error \
    --fail \
    --header "Content-Type: application/json" \
    --data "$payload" \
    "$DISCORD_WEBHOOK" >/dev/null || \
    echo "Discord webhook delivery failed; continuing without failing the monitor run." >&2
}

send_local_notification() {
  local subtitle="$1"
  local message="$2"
  local notifier_bin
  local open_path="$MONITOR_TEXT_PATH"

  if ! is_truthy "$NOTIFY_ENABLED" || ! is_truthy "$NOTIFY_LOCAL"; then
    return 0
  fi

  if brief_available; then
    open_path="$MONITOR_BRIEF_PATH"
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
    -open "file://$open_path"
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
    send_discord_notification \
      "Actionable monitor alerts" \
      "^(BUY CANDIDATE|RAISE STOP|EXIT CANDIDATE): " \
      "^(BUY CANDIDATE|RAISE STOP|EXIT CANDIDATE) \\|"
    send_local_notification \
      "Actionable monitor alerts" \
      "$(build_local_notification_message "$(summarize_alert_counts '^(BUY CANDIDATE|RAISE STOP|EXIT CANDIDATE): ')")"
    return 0
  fi

  if is_truthy "$NOTIFY_ON_WATCH" && alert_count_present "WATCH CLOSELY"; then
    send_discord_notification \
      "Watch closely" \
      "^WATCH CLOSELY: " \
      "^WATCH CLOSELY \\|"
    send_local_notification \
      "Watch closely" \
      "$(build_local_notification_message "$(summarize_alert_counts '^WATCH CLOSELY: ')")"
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
