#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_bot daily-summary \
  "$BOT_UNIVERSE_DEFAULT" \
  --as-of "$BOT_DATE" \
  --preset-names "$BOT_PRESET" \
  --portfolio-file "$BOT_PORTFOLIO_FILE" \
  "$@"
