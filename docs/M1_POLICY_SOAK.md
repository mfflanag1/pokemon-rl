# M1 selected policy/chance two-hour CPU qualification

Status: **the selected combination passed its own two-hour CPU qualification** on
October 11, 2026. The timed window began at 12:05:19 UTC (08:05 Eastern); the final
report was written at 14:05:21 UTC (10:05 Eastern), and the process exited.
The measured interval was **7,200.006915 seconds**. Full G2/G3 remain open; no
strength evaluation or Prime resources were involved.

The [completion receipt](../artifacts/reports/m1-policy-soak.json) binds the source,
run evidence, review dispositions and passing 235-test regression (65 Node + 170
Python). The tested tree's 206 source files are unchanged, so that regression was
reused. Preflight also passed the repaired ten-second smoke run and a separate
four-checkpoint evidence audit.

The run used `tmux` session `pokemon-policy-soak-v1` with idle-sleep prevention.
The fifteen-minute follow-up `finish-pok-mon-policy-chance-qualification` monitored
normal progress quietly; its final action is to disable itself after reporting the
verified result. Launch and monitor metadata are retained as historical evidence.

## Measured result

| Measurement | Production result |
|---|---:|
| Continuous timed interval | 7,200.006915 seconds |
| Primary collect/update cycles | 8,158,970 |
| Completed fixed-duel games | 522,174,080 |
| Submitted player decisions | 3,272,073,210 |
| Executed moves | 2,871,689,075 |
| Complete-loop throughput | 454,454.176 decisions/s; 72,524.108 games/s |
| Decisions per game | 6.266250 |
| Unexpected compilation events after warm-up | 0 |
| Minute telemetry records | 120 |
| Exact original/restore next-update audits | 13 (start + 11 periodic + final) |
| Maximum paired-clock host gap | 0.918060 seconds; limit 30 seconds |
| Peak process RSS | 496,779,264 bytes (473.765625 MiB) |
| Final current process RSS | 164,970,496 bytes (157.328125 MiB) |

No numerical, legality, completion, capacity or counter-bound failure was detected.
Each compiled entry point (`collect`, `update`, `cycle`) traced once; the real JIT
positive control detected one compilation and its same-shape repeat detected zero.
The saved original next-learner and next-metrics fingerprints reproduce exactly
from all 13 checkpoints in a separate verification process. Checkpoint counters
match the start/end report and the bracketing minute telemetry. Admission verifies
unchanged source/runtime and the reserved final data before and after the run.

The latency histogram gives cycle-time upper bounds of 0.750 ms at p50, 1.219 ms at
p95 and 2.329 ms at p99. Weighted mean approximate KL was 0.000048198, entropy
0.407000 and value loss 0.107368; maximum update ratio error was 0.248576.
The minimum observed `exp(behavior_logp)` was 5.960464122e-8, within the declared
FP32 tolerance for the 2^-24 floor. These are stability diagnostics, not strength
or critic-calibration results. The run used local CPU (Apple M3 Max), JAX/jaxlib
0.11.2, NumPy 2.5.4, Python 3.14.3, FP32 and Threefry.

These counts concern the tiny public duel only. Throughput includes host checks
and periodic evidence work under the observed machine load; it is not a full-format
benchmark or an isolated hardware comparison. RSS does not certify device-memory
or system headroom, and a lower final RSS does not establish a smaller working set.
All 128 final teams remain reserved; no final evaluation episodes were generated.
The [production report](../artifacts/runs/m1-policy-soak-v1/report.json) content hash
is `c2c583a57403c79e3f672e2cbca4433b80d2a96892177df3272a68f90aaa0553`.

## Frozen subject and protocol

The subject is the unchanged binary-policy-1 actor/PPO update with the unchanged
showdown-chance-1 event-addressed sampler in the fixed public Rattata duel. Its
[predecessor receipt](../artifacts/reports/m1-showdown-chance.json) is
`d0c377c6d3e6fe68e27382e68d5e575c092780a0c0056f899eac72ba676e6cc0`.
The earlier 10,000 fresh player-view comparisons remain evidence for this same
selected combination. This qualification adds time under continuous collect/update;
it does not expand mechanics or hidden-information coverage.

The additive `pokemon_soak` package wraps the sealed `pokemon_showdown` executable
and checkpoint code. No predecessor Python, contract, fixture or evidence file is
edited. The [qualification plan](../contracts/soak-v1/plan.json) freezes:

- Seed 73, eight warm-up cycles, then 7,200 seconds of continuous monotonic elapsed
  time. Each cycle completes 64 games and their PPO updates; finite metrics,
  legality, terminal completion and counter bounds are checked at every cycle.
- A real compilation positive control and same-shape repeat control, with exactly
  one trace of each compiled entry point and zero compilation events after warm-up.
- One-minute telemetry: primary cycles/games/decisions/executed moves, compilation
  count, probability/ratio extrema, recent learning diagnostics and process memory.
- A checkpoint before timing, eleven checkpoints at minutes 10 through 110, and
  a final checkpoint. Every original runner's next learner/metric result is compared
  exactly with a restore; fingerprints are retained for later process-independent
  replay checks. Audit cycles do not advance primary counters.
- A fail-fast 30-second host-loop-gap limit using paired monotonic and wall-clock
  readings, including sleep detection when monotonic time pauses. macOS idle sleep is inhibited only for
  the running process by `caffeinate -i`. Suspension or termination cannot silently
  manufacture a successful completed report.

Elapsed throughput includes host validation, bounded aggregation, telemetry,
checkpoint I/O and periodic recovery audits; compile/warm-up and the final replay
are outside the timed window. Cycle time measures synchronized execution plus
validation. Process CPU time is also reported and can exceed wall time because JAX
uses multiple CPU threads. Peak/current process RSS are not certified device-memory
or system headroom. Current RSS may be unavailable under sandbox permissions.

Only the production mode can pass the summary gate. The smoke mode has a separate
10-second target and is explicitly labeled `two_hour_qualification: false`.
The inherited checkpoint format is retained because the learner/sampler/state did
not change; the qualification report separately binds this new harness and plan.

## Run and completion procedure

Production directory: `artifacts/runs/m1-policy-soak-v1`.
Review/log directory: `artifacts/reviews/m1-policy-soak`.
The campaign writes atomic progress/report JSON and periodic durable checkpoints.
Use a new directory for any new run; never overwrite an interrupted campaign.

```sh
# The production campaign is launched once, with the existing pinned environment.
PYTHONPATH=python .venv/bin/python -m pokemon_soak.campaign artifacts/runs/m1-policy-soak-v1
# A smoke run is a different directory and never certifies two hours:
PYTHONPATH=python .venv/bin/python -m pokemon_soak.campaign artifacts/runs/NEW-smoke --smoke
# Bind regression to final source, then seal only a successful production run:
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m1-policy-soak
.venv/bin/python scripts/seal-policy-soak.py artifacts/runs/m1-policy-soak-v1
```

Before sealing, require a passing production report, source/runtime/final-reservation
admission equality, full telemetry coverage, every original next-update fingerprint,
exact checkpoint counters and the source-bound regression. Update this handoff and
current progress documents with the actual measured results, then write the receipt
once. Verify the live and archived hashes and reseal idempotently. Do not modify
sealed evidence afterward. An exit or failed report is a failure, not a completed
two-hour qualification; preserve it for diagnosis.

The claude-review skill is used for
independent gate and harness review. The initial gate delegate failed two malformed-
input exception checks; the rejection and repair evidence are retained. The [verified review dispositions](../artifacts/reviews/m1-policy-soak/review-dispositions.md)
record the repaired timing checks, gate QA and smoke evidence. Production results
are recorded above; the original preflight and both smoke runs remain preserved.

## Remaining gates

This result closes the selected combination's two-hour CPU stability item only.
G2 still needs the declared hidden-state and broader-mechanics coverage.
G3 still needs preregistered evaluation against frozen opponents across training
seeds, critic calibration and appropriate memory/budget evidence. The next policy
experiment must keep final evaluation reserved. Prime remains unnecessary.
