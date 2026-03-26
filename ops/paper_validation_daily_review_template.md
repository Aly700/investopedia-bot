# Paper-Validation Daily Review Template

Date:

Validation root:

Review directory:

Operator:

## Daily verdict

Overall: `pass | warn | fail`

Stop condition triggered today: `yes | no`

If yes, which one:

## Service health

- Uptime and restart count: `pass | warn | fail`
- Last successful cycle looked current: `pass | warn | fail`
- Health checkpoint matched runtime reality: `pass | warn | fail`

Evidence:

## Connectivity

- Disconnect behavior acceptable: `pass | warn | fail`
- Connect failures acceptable: `pass | warn | fail`
- Any reconnect storm: `yes | no`

Evidence:

## Safety and control state

- `execution_mode` stayed paper: `pass | fail`
- `broker_target_stayed_paper` stayed true: `pass | fail`
- `live_trading_enabled_ever` stayed false: `pass | fail`
- `paper_only_intact` stayed true: `pass | fail`
- `control_state_readable` stayed true: `pass | fail`
- Any degradation count today:

Evidence:

## Pending orders and broker coherence

- Pending-order state looked coherent: `pass | warn | fail`
- Broker sync drift resolved: `pass | warn | fail`
- Any unusual pending-order states worth follow-up:

Evidence:

## Notifications and operator truth surfaces

- Notifications were useful: `pass | warn | fail`
- Notifications were noisy: `yes | no`
- Dashboard matched daily review artifacts: `pass | warn | fail`
- Internal API and control API matched dashboard/artifacts: `pass | warn | fail`

Evidence:

## Manual intervention

Any manual restart, pause, cancel, replace, or other intervention:

Was it expected:

## Failures and repeated warnings

Top repeated warning(s):

Any failed or rejected control action(s):

Any artifact or review contradiction:

## End-of-day decision

Decision for tomorrow: `continue | continue with caution | stop and fix`

Reason:

Top follow-up item for next session:
