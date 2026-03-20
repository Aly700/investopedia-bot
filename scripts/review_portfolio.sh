#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_bot review-portfolio \
  --portfolio-file "$BOT_PORTFOLIO_FILE" \
  --as-of "$BOT_DATE" \
  "$@"
