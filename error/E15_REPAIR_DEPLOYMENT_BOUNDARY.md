# E15_REPAIR_DEPLOYMENT_BOUNDARY

## Transition Record

| Field | Value |
|---|---|
| **Timestamp** | 2026-09-17T16:27:03+0530 |
| **Old PID** | 66352 (final restart with all fixes) |
| **Session ID** | training_1789642623 |
| **Git Commit** | d1ee406 (cleanup) |
| **Repair Status** | active (post-repair architecture) |
| **Mode** | in-process wrapper |

## Bugs Fixed During Session

1. **Tracker boxplot None crash** — `core/tracker.py:601-602`: `d.get(metric, 0) or 0` filters None values before boxplot. Also `core/tracker.py:560`: `d.get("total_reward") or 0.0`.

2. **Idle reasoner stale idle_time** — `main.py:67-68`: Recompute `idle_time` after short-idle sleep so long-idle check uses fresh wall-clock time.

3. **Meta-learner factor removal off-by-one** — `core/cuengine/idle.py:609-615`: Build `keep` mask BEFORE removing from `dynamic_factor_names`, so index matches array layout. Also retire from `factor_space` and update `n_factors`.

## Tracker Data Repair
- Restored 266 entries from git (d1ee406)
- Fixed 1059 None values across numeric fields
- Verified 0 None values remaining

## Meta-Learner State Repair
- Reset factor_space to v0 with 19 names matching arrays
- Verified invariant: mu/lambda_/alpha/beta/factor_names all length 19

## Verified Working
- [x] IPC via /tmp/herebot_cmd → /tmp/herebot_out
- [x] PID lock at curiosity_state/training.pid
- [x] Tracker recording with 300+ data points
- [x] idle_long cycle triggering (Deep State Introspection, Offline consolidation, Cross-modal Alignment, Word-State Propagation)
- [x] 0 FactorSpaceInvariantErrors
- [x] 0 Tracker boxplot errors
