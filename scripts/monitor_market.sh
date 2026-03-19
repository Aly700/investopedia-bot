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

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$CANDIDATE_FILE" ]; then
  echo "Candidate file does not exist: $CANDIDATE_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

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

exec env PYTHONPATH="$REPO_ROOT/src" "${COMMAND[@]}"
