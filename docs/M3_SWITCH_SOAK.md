# M3 switching/categorical policy CPU qualification

Status: **the restricted switching/categorical learner passed its own two-hour
local CPU qualification** on October 11, 2026 (Eastern). The measured window was
**7,200.008393 seconds**; the report passed and the launcher exited with code 0.
All 13 original-next-update checkpoints reproduce exactly in a fresh process,
including learner, rollout buffer, final environment and metrics fingerprints.
The [completion receipt](../artifacts/reports/m3-switch-soak.json) binds the result,
source, evidence and passing 460-test regression (237 Node + 223 Python). All 331
tested source/configuration files remain unchanged, so that regression was reused.

This qualifies sustained execution of the sealed restricted learner. The
[engine handoff](M3_SWITCH_ENGINE.md) and its immutable receipt remain unchanged.
Full G2/G3/G4 and public ladder strength remain open. Prime is unnecessary.

## Measured production result

| Measurement | Result |
|---|---:|
| Continuous timed window | 7,200.008393 seconds |
| Primary collect/update cycles | 1,711,484 |
| Primary capped games | 54,767,488 |
| Player decisions | 1,173,453,330 |
| Terminal games | 50,278,791 |
| Nonterminal truncated games | 4,488,697 (8.1959%) |
| Complete-loop throughput | 162,979.439 decisions/s; 7,606.587 games/s |
| Unexpected compilation events after warm-up | 0 |
| Warm-up compilation events | 23 |
| Minute telemetry rows | 120 |
| Exact original-next-update checkpoint audits | 13 |
| Maximum paired-clock host gap | 0.551545 seconds (30-second limit) |
| Peak process RSS | 482,951,168 bytes (460.578125 MiB) |
| Final current process RSS | 298,172,416 bytes (284.359375 MiB) |

The initial boundary is update 8, optimizer step 16, next game 256. The final
boundary is update 1,711,492, optimizer step 3,422,984, next game 54,767,744.
Warm-up and replay audit updates are excluded from primary counters. All minute
and checkpoint boundaries link to these counters. The fresh-process audit also
recomputes the initial eight-update state, validates telemetry and report hashes,
and verifies unchanged source/runtime/reserved-data admission. Admission retains
1,110 predecessor live/archive hash checks plus 3,588 inherited checks and the
earlier runtime/reservation checks; only the declared current-document/registry
live exceptions apply. Historical archives remain byte-bound.

The observed behavior-log range is −16.635532379 to 0, within the declared
categorical support bounds. Mean synchronized cycle time is 4.122122 ms; maximum
is 551.072583 ms. Decision-weighted approximate KL is 0.000036062, entropy 0.082050
and value loss 30.291633. These are numerical diagnostics, not convergence,
critic-calibration or playing-strength results. Process CPU time is 11,548.736066
seconds and can exceed wall time with multithreaded CPU execution.

The run uses local CPU, Python 3.14.3, JAX/jaxlib 0.11.2, NumPy 2.5.4, FP32 and
Threefry. Throughput concerns the restricted twelve-turn profile with per-update
host checks; it is not a full-format battle or isolated hardware benchmark.
Truncations are nonterminal resets, never draws. RSS does not certify device-memory
or system headroom; lower final RSS does not establish a smaller working set.
No strength screen, public server, final reserved-data evaluation or Prime was used.
The production report content hash is
`a3d86d3910cc1d121cccdc16a4428c898ba8ceb9036415c30c3a6581475a7ae7`.

## Frozen subject and plan

The [plan](../contracts/switch-soak-v1/plan.json) binds engine receipt
`af89e56e72553901b4582a3b6910d3ee4ad87c62450a3025db99e5019898774e`.
The scalar/JAX mechanics, observations, four-action categorical law, event-indexed
Showdown chance mapping and PPO update are imported unchanged. This remains the
24-team, two-member starter profile with twelve-turn caps, hidden reserve and speed
variants. Nonterminal cutoffs bootstrap each player's own value, then reset; they
are neither draws nor strength-evaluation outcomes.

Seed 101 is fixed for engineering qualification, without strength selection.
Initialization is the unchanged `learner.start(101)` followed by eight full warm-up
updates. A new versioned checkpoint wrapper extends the earlier 64-update smoke
budget without editing or monkeypatching the sealed learner or checkpoint code.
Updates stop before the declared ten-million-update cap; optimizer and game
counters remain linked and cannot wrap in this budget.

- Production: 7,200 continuous monotonic seconds after warm-up and the initial
  replay audit. No restart or stitching intervals. A 20-second smoke mode has a
  separate directory and cannot qualify the two-hour gate.
- Every primary update synchronizes the unchanged kernels, validates finite
  arrays/metrics, fixed shapes, legal masks/actions, terminal/cutoff accounting,
  categorical log support, and same-player GAE against the sealed NumPy function.
- Actual XLA compilation events are monitored with a real positive control.
  Zero events are allowed after warm-up. Paired wall/monotonic host gaps above
  30 seconds fail, including checkpoint work and the final replay audit.
- Every minute records cumulative counts, four action frequencies, stability
  diagnostics and process memory. Peak process RSS must stay at or below 2 GiB.
  This does not certify device memory, free system memory or policy quality.
- Start, minutes 10–110 and final checkpoints each preserve original next-learner,
  rollout-buffer, final-environment and metric fingerprints. Restored updates must
  match exactly; a later independent process repeats these checks. Audit updates
  do not advance primary counts. Checkpoints are completed-update reset boundaries,
  not live-battle recovery. Partial writes remain evidence and cannot be resumed.

Elapsed throughput includes host validation, bounded aggregation, minute telemetry
and periodic replay work. Warm-up, final replay and final provenance admission are
outside that denominator. RSS includes JAX runtime/compiler allocations. Report
terminal and truncated games separately; the short horizon limits strategic claims.
Only the constructed engineering team roster is used. Final reserved teams remain
unused and admission checks their inherited provenance.

## Execution and completion handoff

Production directory: `artifacts/runs/m3-switch-soak-v1`.
Persistent process session: `pokemon-switch-soak-v1`.
Review/log directory: `artifacts/reviews/m3-switch-soak`.
The timed window began at **2026-10-11 22:46:36 UTC (18:46:36 Eastern)**,
PID 97632, and reached two hours at October 12 00:46:36 UTC (October 11 20:46:36
Eastern). The passing report was written at 00:46:43.577 UTC; the launcher exited
with code 0 at 00:46:43.748 UTC. The process and tmux session are no longer running.
It ran under `caffeinate -i`. The fifteen-minute heartbeat
`finish-pok-mon-switching-qualification` monitored quietly; its final action is to
pause after delivery of the verified result. The prior M1 heartbeat remains paused.
Launch, monitor, exit and first-minute records remain preserved in the review
directory. Do not rerun the historical production command or overwrite evidence.
Keep the qualified Python, plan, tests, registry and sealer source-bound.

```sh
PYTHONPATH=python .venv/bin/python -m pokemon_switch_soak.campaign artifacts/runs/m3-switch-soak-v1
PYTHONPATH=python .venv/bin/python -m pokemon_switch_soak.campaign artifacts/runs/NEW-smoke --smoke
.venv/bin/python scripts/seal-switch-soak.py artifacts/runs/NEW-smoke --audit-smoke
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m3-switch-soak
.venv/bin/python scripts/seal-switch-soak.py artifacts/runs/m3-switch-soak-v1
```

Only a passing production report can seal. Require complete telemetry, exact
checkpoint/telemetry counter linkage, all original-next-update fingerprints,
unchanged start/end admission, the source-bound regression, independent review
dispositions and the verified final preflight. Reuse the regression if its source
manifest still matches; do not rerun unrelated work. An exited process without a
valid passing report is a failure: preserve evidence and diagnose it.

This handoff, README, IMPLEMENTATION_PLAN, ACCEPTANCE_GATES and REVIEW_RECORD
were updated with measured results before writing the completion receipt.
Historical handoffs and predecessor evidence are preserved. The completion
procedure verifies every live and archived source/evidence hash, then reruns the
sealer to require an identical receipt. Its
[post-seal integrity record](../artifacts/reviews/m3-switch-soak-integrity.json)
lives outside the sealed review directory. Pause the quiet follow-up after
reporting completion; no further production run is part of this qualification.

Independent implementation/QA and harness review use the
claude-review skill.
Snapshot reviews do not execute tests. Actual checks and verified dispositions
are retained under the review directory before production starts.

## Measured preflight

The smoke run passes in 20.064911 seconds with 1,417 primary updates, 45,344 games,
1,022,876 decisions, 27,797 terminals and 17,547 explicit truncations. It records
five telemetry rows, four exact original-next-update audits, zero unexpected
compilations (23 warm-up events), a maximum host gap of 0.287367 seconds and
500,318,208 bytes peak process RSS. Current RSS is unavailable inside the sandbox;
that optional reading is not a qualification condition. The separate audit process
recomputes all four saved learner/buffer/final-state/metric fingerprints and the
initial eight-update state exactly. This is a preflight result, not two hours.
All 17 new focused tests pass, plus 36 direct strict-plan input mutations and nine
copied-evidence rejection probes. The complete regression then passes all 460 tests
on the same 331-file source manifest; post-regression admission equals the preflight
admission exactly. The qualification modules, plan, tests and registry are frozen.

## Next gate

Next preregister a separate restricted categorical learning screen
across independent training seeds and frozen opponents. Deal explicitly with
nonterminal cutoffs in evaluation; do not report them as draws. The restricted
learner is not yet connected to the playable full-battle Showdown client. Broader
mechanics, long battles, public deployment, competitive ladder evaluation, beliefs
and search remain subsequent work. Local CPU remains sufficient for this stage.
