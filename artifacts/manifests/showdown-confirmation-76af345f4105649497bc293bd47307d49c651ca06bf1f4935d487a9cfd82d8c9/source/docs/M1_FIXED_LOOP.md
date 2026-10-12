# Fixed-shape collect/update loop and CPU soak

The new `pokemon_fixed` package compiles complete collection, player-view encoding,
action sampling, simulator transitions, same-player GAE, PPO and Adam into one
fixed-signature CPU executable. It extends the sealed JAX learner without changing
its source or the pinned simulator. Full G2 and G3 remain open; no competitive
strength, full-format, GPU or Prime claim is made.

## Scope and data path

The admitted game remains the public, fixed-set Rattata duel: 64 simultaneous
games, two players, at most four turns. The policy remains linear over 59
player-history features, with two action logits and a value head. The frozen
[plan](../contracts/fixed-loop-v1/plan.json) fixes capacity at 512 decision rows,
two PPO epochs and four 128-row minibatch slots per epoch.

Both views are projected before either choice is submitted. The actor receives
only `ActorInput(features, mask)`, from the same frozen parameter tree for the
whole collection. Targets and terminal scores stay outside this API. Both players'
submitted decisions remain in a KO turn even when the second move is canceled.
Training decisions and executed moves are counted separately.

Buffers use time/game/seat order `[4,64,2]`. Terminal lanes become padding; validity
is separate from the legal action mask. Padded features and numerical inputs are
sanitized before arithmetic, including before the model, so even NaN padding
cannot affect parameter gradients. Losses average valid rows, and empty
minibatches skip Adam completely. Per-player returns use a reverse time scan;
there is no alternation of reward sign or bootstrap from a terminal state.

The shuffle recipe is intentionally new: permute all 512 indices, then stably
partition valid rows first. Shapes remain fixed even when episode lengths vary.
Numerical update parity compares against the sealed learner using this identical
valid sample order; it does not claim the old variable-length permutation has the
same trajectory. Adam uses a constant table of the reference's host-rounded bias
corrections for the first 20,000 steps, then exact FP32 ones. Parameters, moments
and optimizer counters are carried through compiled scans.

## Verification and measurements

**The corrected campaign passed.** The accepted run is
`artifacts/runs/m1-fixed-loop-v2`, campaign hash
`641472102f5c8e12a6914bd4438bc1971585853f9b0d38650d4d6e3a132df0f5`.
Its source, tests and evidence are bound by
[the completion receipt](../artifacts/reports/m1-fixed-loop.json).

| Measurement | Result |
|---|---:|
| Full regression | 206 tests: 63 Node + 143 Python |
| Collector/update parity | 3 seeds, 24 batches, 1,536 games, 9,614 decisions |
| Parity tolerances | absolute 3e-6 + relative 3e-5; worst error 1.98% of its allowed bound |
| Exact disk restores with compared next cycle | 24 parity + 13 soak = 37 |
| Timed soak | 7,200.032 seconds |
| Complete collect/update cycles | 9,885,926 |
| Primary fixed-duel games | 632,699,264 |
| Submitted player decisions | 3,964,679,976; 6.266 per game |
| Executed moves | 3,479,551,680 |
| Operational throughput, including telemetry/checkpoint/audit overhead | 550,648 decisions/s; 87,875 games/s |
| Primary cycle timing, including host validation | 6,838.659 seconds; 579,745 decisions/s |
| Cycle p50 / p95 / p99 histogram upper bounds | 0.673 / 0.791 / 0.982 ms |
| Largest measured cycle latency | 57.043 ms |
| Peak process RSS | 593,854,464 bytes = 566.344 MiB |
| RSS increase from soak start to final audit | 352 KiB |
| Post-warm-up compilation events | 0, after a real positive-control event |

These large game counts are repeated training on one tiny public duel. They add
stability evidence, not mechanics coverage or policy-strength evidence, and do not
forecast full Random Battles throughput. Only fixed-size working buffers, sparse
checkpoints and telemetry are retained; this is not a stored billion-row dataset.
No nonfinite, illegal-action, capacity or unsupported-profile failure was detected.
Initial compilation/setup for the three entry points took 1.719 seconds outside the soak;
each entry point traced once. The separate profiled run used 1,000 cycles and
400,804 decisions, with explicit cProfile overhead.

Acceptance combines comparisons with the sealed collector and learner, poisoned
padding tests, zero-gradient/empty-minibatch tests, fused/separate execution checks,
legal-action and outcome-feature checks, and exact disk recovery. Three parity
seeds each run eight batches; every boundary is freshly loaded and its next cycle
compared byte for byte. First-batch actor and target NPZ exports are separate.

Compilation checks use precompiled callables that reject shape/dtype changes,
entry-point trace counters, and actual JAX compilation/cache-lookup log events.
A real fresh JIT positive control and identical-call negative control must pass
before a zero-new-compilation result can be accepted. Warm-up and compilation are
outside soak timing. A preliminary run exposed an ineffective log matcher and
was interrupted and preserved; it is not evidence of a passing soak.

The operational soak measures complete cycles with host validation and records a
second elapsed-time rate that includes aggregation, telemetry, checkpoint I/O and
periodic resume audits. Start/final checkpoint publication and the final replay
check are outside that interval. The host was not isolated from ordinary OS work
or brief validation commands. The cProfile run is a separate 1,000-cycle host-cost diagnostic;
its rates include profiler overhead. Isolated collection/update timings repeat
fixed inputs and are not end-to-end throughput. Latency quantiles are upper bounds
from a fixed log-spaced histogram, not exact stored per-cycle quantiles.

The host profile attributes most recorded time to `device_get`. This includes
waiting for the compiled cycle to finish and cannot by itself establish that
Python overhead dominates. Separate phase timings help locate work, but compiler
fusion and synchronization mean their sum is not a substitute for measured cycles.

Memory reporting distinguishes process RSS from device allocator statistics. CPU
JAX may provide no device memory statistics; process RSS alone does not certify
15% available system/device headroom. Training buffers, aggregate metrics and
checkpoint state have bounded size. Telemetry is streamed to disk once per minute,
not accumulated in a Python list. There is no per-update history in the checkpoint.

The soak's weighted mean approximate KL was 4.81e-5, entropy 0.407 and value
half-MSE 0.107. The approximate KL minimum of -5.96e-8 is FP32 rounding scale.
These are optimization diagnostics, not critic calibration or strength scores.

One precision limitation became visible at this duration: the minimum stored
`exp(old_logp)` was 1.81e-12. This field is a **nominal softmax probability**, not
an exact finite-grid sampling mass. The inherited FP32 inverse-CDF sampler uses
a uniform grid spaced 2^-23. Enumerating that grid confirms that a constructed
first-action probability this small still selects the zero grid point, giving mass
2^-23 rather than 1.81e-12, conditional on uniform random words. This is a finite
precision issue, separate from the Showdown critical-hit law distinction. The
mirrored last-action case, checked with the actual distribution kernel, has zero
selecting grid points because the first cumulative probability rounds to one.
Stored log probabilities and PPO ratios therefore describe the nominal softmax
law, not the exact finite-grid selection law in these constructed tails. The audit
does not reconstruct a particular rare soak decision or measure its frequency.
See [the measured grid audit](../artifacts/reviews/m1-fixed-loop/policy-grid-audit.json).
The sealed recipe is preserved; a versioned sampler/floor decision is required
before policy-strength promotion. This milestone's parity and stability checks
do not certify arbitrary tail probabilities.

## Recovery and provenance

Checkpoints contain exact FP32 parameter/moment bytes, Adam step, policy version,
next game ID, cumulative decision/move counts, seed, plan and source/runtime hashes.
The random streams are addressed by seed, named domain and game/turn/seat or
version/epoch. The update limit keeps uint32 game IDs and int32 optimizer counters
within range; cumulative decisions use an unrestricted Python integer.

Publication is atomic with fsync and rename. Only complete whole-episode update
boundaries are supported. A collection/update failure discards uncommitted work
and restores the last durable checkpoint; this is rollback, not continuation at
the crash point. Tests reject partial publication, source/runtime/plan drift,
malformed counters and moments. The soak periodically loads a fresh disk checkpoint
and compares its next cycle without counting duplicate audit games as primary work.

The admission gate pins the preceding learner receipt, verifies sealed core files,
checks dependency/runtime versions and keeps the R7 final reservation unchanged.
Admission is checked again at exit. The final seal archives source, tests, review
dispositions, runtime receipts, checkpoints, NPZ exports and campaign logs.
The regression wrapper hashes implementation, tests, contracts and dependency
locks before and after the run; the seal rejects a log from different sources.

## Reproduce and continue

```sh
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m1-fixed-loop
# Existing pinned .venv; no extra packages, network, paid hardware or Prime needed.
npm run jax:fixed-loop -- artifacts/runs/NEW-fixed-loop-run
# The campaign runs a full 7200-second soak. Use a fresh output directory.
.venv/bin/python scripts/seal-fixed-loop.py artifacts/runs/NEW-fixed-loop-run
```

Independent work used the [claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md):
Fable implemented the bounded numerical helper; Codex reviewed it; local checks
verified it before integration. A separate Fable audit covered the loop, recovery
and measurement claims. Findings and executed dispositions are preserved in
[the review record](../artifacts/reviews/m1-fixed-loop/review-dispositions.md).

Next decide the action sampler's tail precision or explicit probability floor,
close the remaining G2 evidence gaps, and decide the exact Showdown finite-word
chance contract before policy promotion. Then preregister a restricted-policy
evaluation against frozen random, damage-greedy and checkpoint opponents, using
multiple training seeds and seat-swapped games. This soak does not establish
learning progress or strategic competence. Persistent games, arbitrary rollout
cuts, live-state checkpoints, richer mechanics and a history transformer remain
separate work. Keep final evaluation sealed; local CPU remains sufficient.
