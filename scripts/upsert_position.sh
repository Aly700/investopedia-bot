#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: ./scripts/upsert_position.sh SYMBOL QUANTITY FILL_PRICE STOP [PRESET]" >&2
  exit 1
fi

SYMBOL="$1"
QUANTITY="$2"
FILL_PRICE="$3"
STOP="$4"
PRESET="${5:-$BOT_PRESET}"

run_bot upsert-position \
  "$BOT_PORTFOLIO_FILE" \
  "$SYMBOL" \
  --quantity "$QUANTITY" \
  --average-entry-price "$FILL_PRICE" \
  --current-stop "$STOP" \
  --preset-name "$PRESET" \
  --source investopedia
