#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(
  cd "$REPO_ROOT"
  cat "data/processed/monitor_market/$BOT_DATE/market_monitor.txt"
)
