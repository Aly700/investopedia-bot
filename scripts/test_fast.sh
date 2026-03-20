#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(
  cd "$REPO_ROOT"
  PYTHONPATH="$BOT_MAIN" "$BOT_PYTHON" -m pytest \
    tests/test_data_layer.py \
    tests/test_daily_summary.py \
    tests/test_risk.py \
    "$@"
)
