#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/common.sh"

: "${BOT_DATE:=$(TZ=America/New_York date +%F)}"

run_bot \
  review-portfolio-intraday \
  --portfolio-file "$BOT_PORTFOLIO_FILE" \
  --as-of "$BOT_DATE" \
  --interval-minutes 15
