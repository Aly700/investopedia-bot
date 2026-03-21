#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env.local" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env.local"
fi

: "${BOT_PYTHON:=.venv/bin/python}"
: "${BOT_MAIN:=src}"
: "${BOT_DATE:=$(TZ=America/New_York date +%F)}"
: "${BOT_PORTFOLIO_FILE:=data/processed/portfolio/current_positions.csv}"
: "${BOT_PRESET:=aggressive_breakout}"
: "${BOT_UNIVERSE_DEFAULT:=data/raw/candidate_symbols_quality_liquid.txt}"
: "${BOT_UNIVERSE_GROWTH:=data/raw/candidate_symbols_growth_momentum.txt}"
: "${BOT_UNIVERSE_BROAD:=data/raw/candidate_symbols_broad_momentum.txt}"

run_bot() {
  (
    cd "$REPO_ROOT"
    PYTHONPATH="$BOT_MAIN" "$BOT_PYTHON" -m bot.main "$@"
  )
}
