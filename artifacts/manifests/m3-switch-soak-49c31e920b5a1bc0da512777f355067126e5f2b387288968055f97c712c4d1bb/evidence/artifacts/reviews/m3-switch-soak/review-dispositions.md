# M3 switching qualification review dispositions

The claude-review workflow was used for independent gate implementation, opposite-model QA, and a frozen harness snapshot review. Snapshot reviews did not execute tests. Source hashes in each request record identify the reviewed versions.

## Harness review

Fable snapshot 20261011-183220-ba05d3d3 found a real final-smoke telemetry bracket edge case. A last telemetry row at 20 seconds is not guaranteed if the host crosses the deadline between completed cycles. The audit now uses the independently restored final checkpoint as the upper bracket when no later telemetry tick exists, and requires exact counters for a telemetry row from the checkpoint's own cycle. Focused tests cover both a valid missing tick and forged equal-time counters.

The review's single-if telemetry warning is retained as a fail-closed limitation: a very slow smoke run can lack scheduled rows and must fail its preflight. We do not fabricate catch-up telemetry with duplicate counters. Production's 60-second interval exceeds its 30-second allowed host gap. The fixed 32-game/2-epoch constants intentionally bind this immutable predecessor; they are not a configurable new learner.

The reviewer did not receive every inherited helper. Coordinator source inspection and prior sealed evidence establish monotonic-first paired clocks, wall/monotonic gap checks, platform-correct RSS units, duplicate-key rejection, a positive actual compilation logger control, and unchanged predecessor admission. New focused tests cover clock suspension/discontinuity and extended checkpoints.

## Gate delegation

The initial Fable candidate passed six coordinator checks but accepted equal floating-point constants where the frozen plan requires integers. Opposite-model Codex QA identified this; the coordinator reproduced seed=101.0 being admitted. The first outcome is rejected and preserved; it was not integrated. The task and tests now explicitly require non-boolean integer plan constants. The suggested relaxation of scope was rejected: this exact qualification intentionally leaves full G2/G3/G4 open and forbids public/final-data/Prime use. A fresh corrected candidate changes only the two primitive checks, with opposite-family QA required before acceptance.

## Verification

Predecessor admission has checked 1,110 live/archive M3-engine hashes (six mutable registry/progress documents exempted live), plus the inherited 3,588 checks and earlier runtime/reservation admission. New checkpoint tests pass all nine cases, including equality with the sealed collect/update results, counter bounds, atomic-write interruption preservation and discarded transactions. The existing compiled Showdown tree verifies against its pinned build receipt; additive Python compile checks pass. Final focused tests, smoke, fresh-process replay audit and source-bound regression are recorded separately before launch.

Final gate candidate 20261011-183417-6d000447 was routed to Codex, then accepted by Fable QA. The coordinator independently verifies unchanged selected input hashes, sole-file ownership and exactly the two intended predicate corrections, executes all 17 focused tests and 36 direct strict-plan mutations, and records an accepted outcome. Controller execution of tests is distinct from the implementer/reviewer claiming to have run them.

The real 20.064911-second smoke passes with 1,417 updates, 45,344 games, 1,022,876 decisions, zero unexpected compilation events and 500,318,208-byte peak RSS. A separate process verifies all four original-next-update checkpoint fingerprints and the initial warm-up state. Nine copied-evidence negatives reject smoke-as-production, unfinished status, runtime drift, missing telemetry, inconsistent counters, forged rollout fingerprint, checkpoint corruption, partial files and reserved-data-use claims. The admitted smoke is byte-unchanged and no completion receipt is written. The pure gate tests also reject altered scopes, timing, categorical support, memory/compilation violations and malformed fields. Full regression is required before launch.

The final source-bound regression passes all 460 tests (237 Node + 223 Python), with all 331 source/configuration files unchanged. Post-regression admission is exactly equal to the passing preflight admission. No sealed predecessor is altered. Production may start once, with this tested tree frozen.


## Production completion

The unchanged production harness reports passed after 7200.008392709307 seconds; launcher exit is 0 and the process has exited. A fresh sealer audit verifies all 13 original-next-update learner/buffer/final-state/metric fingerprints, the initial warm-up state, all 120 telemetry rows and their counter/checkpoint linkage, checksums and unchanged source/runtime/reserved-data admission. All 331 tested source/configuration files still match the passing 460-test regression; it is reused without unrelated test reruns. The pinned compiled Showdown build verifies unchanged. Counts and stability diagnostics are recorded in completion-audit.json and the run report; no strength or full-gate closure is inferred. Current progress documents are updated before sealing; post-seal integrity and repeat-seal evidence are written outside this directory.
