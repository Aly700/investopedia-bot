# Local Paper-Validation Proving Runbook

This runbook is for a serious local paper-mode proving run using the existing paper-validation runtime.

Use it to answer four questions:

1. Did the live stack stay healthy enough to trust?
2. Did the paper-only safety boundary hold?
3. Did pending-order and broker state stay coherent?
4. Is the system ready to move to cloud?

## Scope

Use this runbook with:

- `run-local-paper-validation`
- `run-local-paper-validation-smoke`
- `summarize-local-paper-validation`
- the paper-validation review artifacts under `<validation-root>/archive/validation_review/<YYYY-MM-DD>/`

Do not use this runbook for live trading. This is a paper-only proving workflow.

## Standard Setup

Set one validation root for the main run and keep it stable for the full week:

```bash
export BOT_VALIDATION_ROOT="$PWD/data/local_paper_validation_week1"
export BOT_CANDIDATES="$PWD/data/raw/candidate_symbols_quality_liquid.txt"
export BOT_PORTFOLIO="$PWD/data/processed/portfolio/current_positions.json"
export BOT_WS_URL="wss://<your-market-websocket>"
export BOT_DATE="$(date +%F)"
export BOT_NOTIFY_URL="https://<your-notification-webhook>"
```

If you do not want notifications for the proving run, leave `BOT_NOTIFY_URL` unset and mark notification review as `needs_review`, not `pass`.

Optional launch additions:

- `--portfolio-file "$BOT_PORTFOLIO"` if you want the live cycle to include current holdings
- `--include-intraday-review` only when `--portfolio-file` is also present
- `--notification-webhook-url "$BOT_NOTIFY_URL"` if notifications are in scope

## Standard Commands

Bounded Day-1 smoke run:

```bash
PYTHONPATH=src .venv/bin/python -m bot.main run-local-paper-validation-smoke \
  "$BOT_CANDIDATES" \
  --websocket-url "$BOT_WS_URL" \
  --validation-root "$BOT_VALIDATION_ROOT" \
  --as-of "$BOT_DATE" \
  --format yaml
```

Default smoke-run bound:

- `run-local-paper-validation-smoke` defaults to `--max-iterations 5`
- override `--max-iterations` only if the default is too short or too long for your provider/session

Always-on proving run:

```bash
PYTHONPATH=src .venv/bin/python -m bot.main run-local-paper-validation \
  "$BOT_CANDIDATES" \
  --websocket-url "$BOT_WS_URL" \
  --validation-root "$BOT_VALIDATION_ROOT" \
  --as-of "$BOT_DATE" \
  --format yaml
```

Daily review summary:

```bash
PYTHONPATH=src .venv/bin/python -m bot.main summarize-local-paper-validation \
  --validation-root "$BOT_VALIDATION_ROOT" \
  --as-of "$BOT_DATE" \
  --window-days 1 \
  --format yaml
```

Week-1 rollup:

```bash
PYTHONPATH=src .venv/bin/python -m bot.main summarize-local-paper-validation \
  --validation-root "$BOT_VALIDATION_ROOT" \
  --as-of "$BOT_DATE" \
  --window-days 7 \
  --format yaml
```

Basic reachability checks after the runtime is up:

```bash
curl "http://127.0.0.1:8765/health"
curl "http://127.0.0.1:8766/control/safety"
open "http://127.0.0.1:8780"
```

Adjust host and port only if you started the runtime with non-default binds.

## Artifact Map

Expect these review files under:

```text
<validation-root>/archive/validation_review/<YYYY-MM-DD>/
```

Files:

- `paper_validation_summary.json`
- `paper_validation_checkpoint.json`
- `paper_validation_brief.txt`
- `paper_validation_daily_review.json`
- `paper_validation_daily_review.txt`
- `paper_validation_changes_today.txt`
- `paper_validation_operator_checklist.txt`

Use these as the daily source of truth, not scattered console logs.

## Day-1 Checklist

Run this in order. Do not skip the bounded smoke run.

### 1. Bounded smoke run passed cleanly

Check:

- `run-local-paper-validation-smoke` exits with a clean pass result.
- The command prints a `validation_profile`.
- The command prints `review_outputs`.
- The command prints `smoke_outputs`.
- The review directory for `BOT_DATE` exists.

Pass:

- All seven review files exist for `BOT_DATE`.
- `smoke_run_result.json` and `smoke_run_brief.txt` exist.
- `paper_validation_summary.json` is readable JSON.
- `smoke_run_result.json` shows `passed=true`.

Stop and fix now:

- Smoke run throws.
- Review files are missing.
- Smoke result files are missing.
- `smoke_run_result.json` shows `passed=false`.
- `paper_validation_summary.json` or `paper_validation_daily_review.json` is malformed or obviously stale.

### 2. Live stack is actually running under the paper-validation runtime

Check:

- Start the always-on command in a dedicated terminal or `tmux` session.
- Confirm the process keeps running for at least 15 minutes.
- Confirm the runtime is writing under `BOT_VALIDATION_ROOT`, not normal dev data paths.

Pass:

- `<validation-root>/runtime-root`, `<validation-root>/hot-state`, `<validation-root>/logs`, and `<validation-root>/archive` are all present.
- Active log file exists at `<validation-root>/logs/local_paper_validation.log`.

Stop and fix now:

- Runtime exits unexpectedly.
- Files are being written outside the validation root.

### 3. Paper-only safety state is correct

Check `paper_validation_summary.json`:

- `safety.paper_guardrail_active == true`
- `safety.current_execution_mode == "paper"`
- `safety.execution_submission_enabled == true` unless you intentionally paused submissions
- `safety.live_actions_require_confirmation == true`
- `safety.broker_trading_enabled == false`
- `paper_integrity.paper_only_intact == true`
- `paper_integrity.execution_mode_stayed_paper == true`
- `paper_integrity.broker_target_stayed_paper == true`
- `paper_integrity.live_mode_success_count == 0`
- `paper_integrity.live_trading_enabled_ever == false`

Pass:

- Every item above matches the expected paper profile.

Stop and fix now:

- Any paper/live drift is visible.
- `paper_only_intact` is false and the reason is not a deliberate bounded resilience test in a separate validation root.

### 4. Dashboard and APIs are reachable

Check:

- Open the dashboard URL.
- `curl /health` returns a structured payload.
- `curl /control/safety` returns a structured payload.

Pass:

- Dashboard loads.
- Internal API shows health data.
- Control API shows readable safety state.

Stop and fix now:

- Dashboard is unavailable.
- Internal API or control API is unavailable or lying about current state.

### 5. Notifications work at least once

Check:

- If notifications are configured, confirm at least one notification is actually delivered on Day 1.
- Natural signal is fine.
- If no natural event occurs, run one short bounded warning-path drill in a throwaway validation root with an intentionally bad websocket URL and confirm a warning notification arrives.

Pass:

- At least one notification is observed end-to-end or the operator records that notifications are intentionally not part of this proving run.

Stop and fix now:

- Notifications are expected but delivery fails.
- The runtime says notifications are enabled but nothing can be made to deliver during a controlled warning-path drill.

### 6. Logs are being written and rotation is believable

Check:

- Active log grows during runtime.
- Archived logs, when present, land under `<validation-root>/archive/logs/`.
- The active log remains writable after rotation.

Recommended Day-1 spot check:

- Run one short bounded smoke run in a throwaway validation root with a low `--log-rotate-max-bytes` value.

Pass:

- Active log keeps updating.
- Archived logs appear only in the archive tree.

Stop and fix now:

- No logs are written.
- Rotation breaks logging.
- Archived logs appear in the wrong place.

### 7. Review artifacts are being generated correctly

Check:

- Run `summarize-local-paper-validation --window-days 1`.
- Read `paper_validation_brief.txt`, `paper_validation_daily_review.txt`, `paper_validation_changes_today.txt`, and `paper_validation_operator_checklist.txt`.

Pass:

- Files are readable.
- The text matches the current runtime state.
- The JSON and text are not obviously contradicting each other.

Stop and fix now:

- Review artifacts are missing.
- Review artifacts contradict the dashboard/API or current runtime state.

### 8. Restart sanity check passed

Check:

- Intentionally stop the always-on process once.
- Restart it against the same `BOT_VALIDATION_ROOT`.
- Re-run the daily summary.

Pass:

- Runtime comes back cleanly.
- `last_successful_cycle_at_utc` advances after restart.
- `restart_count` increases.
- No unexpected paper-integrity warning appears.

Stop and fix now:

- Restart loses hot state unexpectedly.
- Restart causes unreadable safety state or stale health.
- Restart produces unexplained pending-order or broker drift.

### 9. No unexpected live-mode or broker-target drift is visible

Check:

- `paper_integrity.integrity_warnings` is empty for the main Day-1 run.
- `control_state_degradation_window_count == 0` on the main Day-1 run.

Pass:

- No live-mode, broker-target, or control-state degradation evidence appears in the main run.

Stop and fix now:

- Any unexplained integrity warning appears in the main run.

## Daily Week-1 Workflow

Run this once per day, preferably after the main session has accumulated enough runtime.

### Daily command sequence

1. Run the daily summary command for `BOT_DATE`.
2. Read:
   - `paper_validation_brief.txt`
   - `paper_validation_daily_review.txt`
   - `paper_validation_changes_today.txt`
   - `paper_validation_operator_checklist.txt`
3. Open the dashboard once and compare it against the review files.
4. Save one short operator note using `ops/paper_validation_daily_review_template.md`.

### Daily checklist

Mark each item `pass`, `warn`, or `fail`.

- Service uptime and restart count look normal for the day.
- `last_successful_cycle_at_utc` is recent and consistent with the runtime session.
- Disconnect warnings and connect failures are low enough that the day was operationally usable.
- No reconnect storm was visible.
- Warning count and error count are explainable.
- Notifications were useful, not silent when needed, and not noisy enough to ignore.
- Pending-order counts and unusual states are coherent.
- Broker sync drift signals, if any, are explained and resolved.
- Failed or rejected control actions are understood.
- `execution_mode` stayed paper.
- `broker_target_stayed_paper` stayed true.
- `live_trading_enabled_ever` stayed false.
- `control_state_readable` stayed true in the main run.
- `paper_only_intact` stayed true in the main run.
- Dashboard, internal API, control API, and review artifacts agreed with each other.
- Manual intervention, if any, was small and explainable.
- No stop condition was triggered.

## Stop Conditions

Stop the proving run and fix the issue before continuing if any of these occur in the main validation root:

| Condition | Evidence | Required action |
| --- | --- | --- |
| Paper/live safety violation | `current_execution_mode != paper`, `live_mode_success_count > 0`, `live_trading_enabled_ever == true`, or `paper_only_intact == false` without an approved bounded test explanation | Stop immediately. Preserve artifacts. Fix before any further run time. |
| Broker-target mismatch | `broker_target_stayed_paper != true` or broker-target integrity warning appears | Stop immediately. Treat as safety-boundary failure. |
| Control-state unreadable and not cleanly recoverable | `control_state_readable == false` in the main run or repeated degradation with no clean recovery | Stop. Fix persistence and fail-closed behavior first. |
| Repeated unresolved broker drift | `broker_sync_drift_count > 0` for two consecutive daily reviews, or one day shows repeated drift with no explanation | Stop. Reconcile state before continuing. |
| Persistent reconnect storm | Daily review shows repeated reconnect anomaly and the service is stale, cycling failures continue, or no stable recovery occurs the same day | Stop. Fix transport stability first. |
| Review layer is missing or lying | Review files missing, malformed, stale, or clearly inconsistent with dashboard/API/runtime state | Stop. The proving run is not reviewable. |
| Dashboard/API/control plane untrustworthy | Dashboard unavailable, `/health` stale or misleading, `/control/safety` inconsistent with current runtime | Stop. Operator truth surfaces must be trustworthy. |
| Pending-order or broker state incoherent | Unusual pending-order states or broker updates accumulate and do not reconcile cleanly | Stop. Fix execution-state coherence first. |

## Evidence To Save Every Day

Keep one dated review directory plus a short operator note.

Minimum retained evidence:

- the full review directory for the day
- one copy of the console output from `summarize-local-paper-validation`
- one screenshot or note if the dashboard disagreed with artifacts
- any notification examples that were useful or misleading
- any manual intervention notes

If a stop condition is triggered, also save:

- the active log tail
- the exact command line used
- the relevant API responses from `/health` and `/control/safety`

## Week-1 Review Flow

At the end of Day 7:

1. Run the 7-day summary.
2. Read all seven daily operator notes.
3. Fill in `ops/paper_validation_week1_review_template.md`.
4. Decide one of:
   - `ready for cloud`
   - `ready after small fixes`
   - `not ready`

### Ready for cloud

Use this only if all of these are true:

- No paper/live safety violation occurred.
- No broker-target mismatch occurred.
- No unresolved control-state degradation remained in the main run.
- Dashboard/API/review artifacts stayed trustworthy enough to operate the system.
- Restart behavior was clean.
- Pending-order and broker state stayed coherent enough that drift was either absent or resolved quickly.
- Alerts were useful enough that an operator would keep them enabled in cloud.

### Ready after small fixes

Use this only if:

- Safety boundaries held, but
- there were small usability or noise problems that do not undermine trust in runtime state.

### Not ready

Use this if:

- Any safety-boundary question remains unresolved.
- The operator could not reliably tell what was true during the run.
- Restart/recovery or broker reconciliation still feels fragile.

## Template Files

Use these fill-in documents:

- `ops/paper_validation_daily_review_template.md`
- `ops/paper_validation_week1_review_template.md`
