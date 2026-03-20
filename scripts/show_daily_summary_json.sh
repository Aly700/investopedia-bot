#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

(
  cd "$REPO_ROOT"
  cat "data/processed/daily_summary/$BOT_DATE/daily_summary.json"
)
