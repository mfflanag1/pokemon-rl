# F31 — worker timeout diagnosis and observed transport

Status: engineering qualification passed. F30 remains a failed, incomplete comparison. The original timeout was not reproduced and its cause remains unresolved. No replacement learning campaign was launched.

## What happened in F30

All eight workers exceeded the same 60-second response deadline on the first random-opponent midpoint evaluation batch for seed 9311/control, absolute cohort 512. Their last saved responses were natural terminals from the preceding training batch; stderr and partial response buffers were empty. The frozen transport did not retain live worker process/resource measurements or execution phases before cleanup, so the evidence cannot distinguish a shared filesystem/subprocess stall, resource problem or another transient failure.

Static inspection identified synchronous Git source checks during episode start as a possible shared dependency. This is a hypothesis, not a reproduced cause. No provenance check, simulator rule, actor input or learner arithmetic was removed or changed.

The failed run remains preserved by `artifacts/reports/f30-nonlinear-continuation-failure.json`. F31 independently rechecked all 723 live and frozen source files, 5,723 evidence files, archived receipt bytes and the F29 predecessor receipt. Historical documents and sealed evidence remain unchanged.

## Additive implementation

New files under `packages/worker-watch`, `python/pokemon_worker_watch`, `tests/worker-watch` and `contracts/worker-watch-v1` provide:

- Coordinator-only start, team-admission and stream-write phase events on stderr, including timestamps, PID, RSS and dropped-record counts. Diagnostic errors are guarded; queued pipe output is bounded. Filesystem logging still has overhead.
- Complete nonblocking command writes and response reads sharing one 60-second deadline, with zero request retries and unchanged stdout protocol/schema validation.
- Failure evidence with command, previous frame, partial/consumed response, byte counters, wait phase, PID/return code and a bounded process-group snapshot including live RSS and CPU.
- Dedicated worker process groups, descendant termination, bounded waits, descriptor closure and idempotent cleanup. Cleanup and evidence-write failures retain the primary error and are reported separately.
- A new pool adapter whose successful collection logic is AST-identical to the frozen predecessor. Failure handling is hardened independently of the learner.

## Measured verification

All **37 focused tests passed**: 34 Python and three Node, with no skips. Fault cases include blocked stdin, partial responses, successful partial writes, combined write/read deadline consumption, malformed frames, descendant cleanup, cleanup interruptions, evidence-write failures, closed stderr and an undrained stderr pipe. The pinned Showdown build was rebuilt and verified against its existing receipt.

The original and watched transports each executed 16 exact failed-start requests and replayed the eight immediately preceding recorded games (458 submitted actions per transport). Previous terminal frames matched the preserved failure evidence; cold and post-terminal starts matched, including across transports. Both probes passed in about 1.36 and 1.35 seconds respectively. Neither reproduced the timeout; these short fresh-worker probes do not reproduce the preceding long-lived worker history or original machine state.

The fixed CPU comparison covered both arms for all three seeds, starting from their same-arm sealed F29 states with Adam step 2048. Each transport performed one engineering cohort per cell, 6,144 selected rows and 48 optimizer steps; Adam ended at 2056 in each cell. All six comparisons had exact batch, complete ledger/provenance/actor-trace, update-metric, parameter and Adam-state equality. Every watched worker emitted the required phase events with zero dropped records.

| Seed | Arm | Games per transport | Decisions per transport | Terminal / cutoff |
|---|---|---:|---:|---:|
| 7103 | control | 32 | 1,817 | 32 / 0 |
| 7103 | nonlinear | 40 | 2,164 | 40 / 0 |
| 8209 | control | 40 | 2,346 | 40 / 0 |
| 8209 | nonlinear | 40 | 2,007 | 40 / 0 |
| 9311 | control | 32 | 1,644 | 32 / 0 |
| 9311 | nonlinear | 40 | 2,215 | 40 / 0 |
| **Total** | | **224** | **12,193** | **224 / 0** |

Physical execution across both transports was **448 games, 24,386 decisions and 96 optimizer-step executions**, in **61.55 seconds**. The two transports replay the same 224-game schedule; they are not 448 independent training examples. Each collected 8,586 learner rows, selected 6,144 and excluded 2,442, with 12,288 row presentations. Each selected 3,072 self-play and 3,072 tactical rows. Collection times were 10.36 seconds original and 9.34 seconds watched; original always ran first, so these timings do not establish a speedup or overhead estimate.

Runtime: local CPU, Python 3.14.3, JAX/jaxlib 0.11.2, NumPy 2.5.4, Node v25.6.1, FP32. The 733-file executable/configuration inventory and runtime were identical before and after parity execution. Parameter/optimizer equality was asserted in process; the generated engineering policies were not promoted or used as continuations.

## Review and retained limitations

Independent Claude review identified logging failure/backpressure, collector failure-evidence masking and vacuous instrumentation-coverage risks. These were verified, corrected and tested. The delegated transport delivery timed out; its partial candidate and failed status are retained. Local tests and independent Codex QA found two further gaps (live resource fields and interrupted cleanup), which the coordinator corrected. Final QA found no concrete blockers in those fixes. The original failed delegation is not reclassified as a successful delivery.

The evidence supports a short engineering parity and fault-handling qualification only. It does not prove long-duration reliability, explain F30's failure, complete F30, or demonstrate learning improvement. Full G2/G3/G4, held-out-team and public ladder strength remain open. No public server, final reserved evaluation, Prime or GitHub publication was used. The existing visibility hold remains in force.

## Next step

Prepare a separately versioned replacement comparison using this observed transport, all six original F29 continuation states, and the fixed seed/arm budgets. Preserve the failed F30 result, keep a distinct run identity, and require full replay and completion checks. Do not stitch partial F30 cells into a new success claim. F31's engineering plan does not itself authorize that learning campaign.

Evidence: `artifacts/diagnostics/f31-worker-timeout/`. Receipt: `artifacts/reports/f31-worker-watch.json`. Post-seal integrity checks are stored outside the evidence directory.
