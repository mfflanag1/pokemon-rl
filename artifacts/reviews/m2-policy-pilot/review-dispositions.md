# Verified pilot review dispositions

The independent Fable snapshot review found no blocking issue in the supplied
adapter, inference, campaign and sealer scope. Its strongest observation was that
exact float replay is a local-runtime requirement rather than a portable numerical
contract. The inherited full CPU/runtime binding is retained, an explicit hashed
host-name/architecture label has been added, and the handoff disclaims hardware
attestation and cross-host replay. A separate smoke evidence audit on this host
passed all 48 games, every saved action/value, the report statistics and both
checkpoint next-update fingerprints.

Verified low-severity repairs add a final batch-roster check after the four-turn
loop and an explicit terminal-turn/decision-count assertion, including dropped-first
and dropped-last decision probes. The review's concern about unexecuted regression
counts is resolved only by the source-bound full regression, which is required by
the sealer. No model review is presented as executed test evidence.

The final source-bound regression passed 251 tests (68 Node + 183 Python) and
verified the same 218 implementation/test/configuration files before and after.
An earlier attempt stalled in the unchanged R5 fixture's stdin bridge and was
terminated after a native stack sample showed blocked input. Its failed report
and log are retained separately; the clean rerun passed without source changes.

The main snapshot predates the worker EOF fix. Smoke v1 completed its 48 matches
but failed to exit Node because the parent input pipe was left open. Closing it
after the finish handshake resolved the failure; a real-process lifecycle test was
added. Smoke v2 passed. Smoke v3 failed before admission because sandbox permissions
deny a CPU-brand sysctl. That optional hardware query was replaced by the documented
host-name/architecture hash; smoke v4 and the separate --audit-smoke verification
passed. All four directories remain intact. The repaired source is frozen before
production; no production outcomes informed these fixes.

The initial statistics delivery passed its five tests and received accepting Codex
QA, but independent huge-alpha and finite-calibration-overflow probes exposed two
contract violations. It was rejected. The repaired delivery passes six frozen
tests and exactly matches an independent unchunked 4096-by-1000 bootstrap matrix.
Codex QA returned revise only for hypothetical allocation failure at an unbounded
replication count. Admission now requires exactly 4096 replicates and at most 1000
pairs; the coordinator accepted this bounded workload while retaining the generic
resource-exhaustion limitation. See the immutable delegate-verification.md and both
archived QA/outcome records. No accepting QA verdict is falsely attributed to the
repair, and the repaired module was integrated byte-for-byte.

The bootstrap family has six approximate intervals: score and paired improvement
for each of three training seeds. Diagnostics do not drive selection; the endpoint
and sample count are fixed. The damage-greedy priority tie-break and the shift from
self-play value targets to fixed-opponent calibration are explicit in the handoff.
Full G2/G3 remain open and final data remain reserved. No Prime resources are used.
