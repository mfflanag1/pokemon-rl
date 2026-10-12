# Verified review dispositions before the production run

The initial Fable 5.1 harness/sealer review found a consequential timing weakness:
monotonic time alone can pause during macOS sleep. The coordinator added paired
monotonic/wall-clock gap checks and tests for frozen monotonic, long gaps and
backward wall-clock jumps. Gap is the maximum of both deltas and their absolute
discrepancy; anything above the frozen 30-second limit fails. The report's monotonic
elapsed interval must be between target and target+30 seconds. The final replay is
explicitly outside the timed window and is also gap-checked.

Other verified repairs: the sealer's --audit-smoke path exercises admission,
telemetry, replay roster, persisted original next-state/metric fingerprints,
checkpoint counters and timeline linkage without creating any completion receipt.
History preserves unsigned running stubs and missing reports as incomplete evidence;
completed reports still have their content hashes checked. No interrupted directory
needs removal to make the qualification eligible for sealing.

The focused Fable follow-up review found no blocking issue. Its recommendation to
execute the smoke/audit path was fulfilled: smoke v2 passed 13,587 cycles and
5,448,940 decisions in 10.0093 seconds, with zero unexpected compilation events,
four original/restored replay checks and maximum observed host gap 0.0442 seconds.
The separate smoke audit verified every persisted next-cycle fingerprint. This
establishes a comfortable operational margin against the 30-second fail-fast bound;
it does not qualify two hours. No production strength/performance inference is made.

The review suggested repeating mode checks inside the sealer independently of
validate_summary, because that helper was outside its focused packet. Inspection
confirms the helper explicitly requires mode=qualification, two_hour_qualification
is True, target=7200 and all other scope flags; coordinator negative tests exercise
smoke and promotion rejection. The sealer always calls this source-bound gate in
its non-audit path. Duplicate checks are unnecessary; the helper is a required
validated dependency, not an optional plugin.

The bounded gates.py delegation initially violated its ValueError interface for
huge integer conversion and mixed invalid trace-key sorting. Codex QA findings were
reproduced and the initial delivery was recorded rejected. A separate Fable repair
passed frozen regression probes, got Codex QA accept, and was integrated unchanged
only after input-hash and ownership checks. See delegate-verification.md and both
retained delegate directories. Neither model review is represented as an executed
local check.

Qualification Python, plan and sealer were frozen after smoke v2. Follow-up snapshot
source hashes match that final implementation. Full regression is source-bound by
the unchanged verify-fixed-loop wrapper. The long run's final admission must match
its initial admission before any receipt can be sealed. All predecessor source,
contracts and evidence remain unchanged, with the explicit existing exceptions for
current documentation, package scripts and the additive test-suite registry.

Full G2/G3 remain open. This milestone concerns the selected public-duel combination's
CPU stability only, with no strength evaluation, final reserved-data use, GPU, Prime,
or certified device/system memory headroom.
