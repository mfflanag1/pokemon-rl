# Independent fixed-loop review and verified dispositions

Fable snapshot review `20261010-185932-aff73b5c`, completed successfully in about
189 seconds. Original request, selected-source hashes, prompt, response, execution
record and review are retained here. This was a tool-free source review, not an
executed test run. Full G2/G3 and competitive policy claims remain out of scope.

1. **Compilation logger could be vacuous — confirmed and fixed.** The coordinator
   independently observed the same mismatch in the first campaign log. Installed
   JAX 0.11.2 source (`dispatch.LogElapsedTimeContextManager` and
   `pxla._cached_compilation`) emits completed compile/cache-lookup timing events
   on `jax._src.dispatch`. The handler now counts that actual prefix. A real fresh
   JIT invocation must increment the count; an identical second call must not.
   Both the regression test and campaign execute this positive/negative control;
   acceptance also requires a positive baseline count before zero-new-events can
   be claimed. `m1-fixed-loop-v1` was interrupted before its first minute of soak,
   preserved with failed status, and is not accepted evidence of stability.
2. **cProfile timing overhead — confirmed and labeled.** The profile report now
   explicitly identifies profiler overhead and directs operational throughput
   claims to the unprofiled soak. Isolated collection/update latencies are also
   labeled as fixed-input diagnostics, never summed into an end-to-end claim.
3. **Minimum episode length implicit — aligned and documented.** The maximum hit
   is 58 HP (verified from the pinned integer damage formula), below the 105 HP
   starting total. Every game needs at least two turns. Both the cycle boundary
   and checkpoint validator now require at least 256 decisions per 64 games.
   The review's incidental maximum-hit value of 59 was not adopted.

Independent checks after fixes: all 13 fixed-loop tests passed in 5.822 seconds,
including the real compilation logger positive control (`fixed-tests-v2.log`).
The original 203-test regression log is retained; a fresh full regression is
required for the final seal. The final campaign rechecks source/runtime and
predecessor hashes at admission and exit.

The plan's maximum counter guard was raised from 10 million to 50 million updates
before the accepted soak, because the initial measured cycle cost was much lower
than anticipated. This is a guard, not a training target: stop remains 7200 seconds.
The maximum next-game index is 3.2 billion < uint32 capacity, and Adam step is at
most 400 million < int32 capacity. Cumulative decisions use a Python integer.

No numerical, information-boundary, checkpoint, or unbounded-aggregation blocker
was found in the supplied code. This is limited evidence about this fixed public
duel, not a proof of general simulator or learner correctness.

## Completion sealer audit

Separate bounded snapshot `20261010-190832-86a71b4f` reviewed the newly written
sealer and actual completed parity/profile phases while the soak was still running.
Its original packet and response are retained as `seal-claude-*`. It did not verify
a completed two-hour run; none existed at review time.

- Confirmed that a selected run outside the versioned glob could be omitted from
  the archive. The selected directory is now always included; interrupted v1 must
  be present with failed status. A campaign-history table records all retained run
  statuses. We did not adopt the suggestion that every other run must fail: future
  successful reruns are legitimate and must also be retained accurately.
- Confirmed that an unbound test log could outlive a source edit. Added a regression
  wrapper that captures all implementation/test/contract/configuration file hashes,
  requires them unchanged during execution, and binds the log hash and exit code.
  The sealer recomputes this manifest, so any subsequent covered edit requires a
  fresh full regression. Documentation/output edits do not change executable tests.
- Added explicit checkpoint seed, start/final counter, executed-move delta and
  strictly increasing periodic-boundary checks. Parity boundaries must carry the
  correct seed and version. Telemetry counts must fall within the final totals.
- Added exact parity field and tensor-element coverage checks. Nonempty optimizer
  cases are bounded by the declared 4..8 applied minibatches per cycle; all seven
  metrics must cover the same number. The observed run has 192 optimizer cases.
- The final-reservation flag is required from the revalidated admission result,
  rather than independently asserted by the completion receipt.

Syntax checks passed after these changes. A source-bound full regression and the
actual completion seal remain required before accepting this milestone.

Three additional boundary probes passed in 2.894 seconds (`boundary-tests.log`):
the largest uint32 seed and game IDs above int32 range match the sealed collector;
the last admitted update preserves counters and the next update rejects; genuinely
nonfinite model state fails before publication. The late-counter fixture is
synthetic arithmetic evidence, not a claim of 50 million trained updates.

## Completed soak, final regression and action-tail follow-up

The corrected campaign completed 7200.03185 seconds and passed its frozen gates:
9,885,926 primary cycles, 3,964,679,976 decisions, no detected numerical/legality/
capacity failure, and zero new compilation events after the pre-soak baseline of 330.
It performed 13 fresh disk restores with compared next cycles, in addition to
24 parity restores. Peak RSS was 593,854,464 bytes; final checkpoint size was
13,049 bytes. Final source/runtime/predecessor admission matched the start exactly.

The final source-bound regression passed all 206 tests (63 Node + 143 Python);
`verification.json` binds 168 implementation/test/configuration files and the
regression log, with identical before/after source hashes. The wrapper was rerun
after adding the tail-precision limitation to the sealer. No runtime source or
plan was changed during or after the accepted soak.

The completed soak recorded a minimum nominal `exp(old_logp)` of 1.808e-12.
Installed JAX source inspection and exhaustive enumeration of its 2^23 FP32
uniform grid points confirmed a precision limitation in the inherited actor.
For a constructed first-action probability this small, exactly the zero grid point
selects it. A focused Fable review (`20261010-211010-988ffc81`, retained as
`policy-claude-*`) confirmed the conditional arithmetic and scope, and recommended
checking the mirrored tail. The coordinator executed the actual sealed distribution
kernel with mirrored logits and confirmed zero grid points select the last action
at this nominal mass: its preceding CDF rounds to one. The original reviewed audit
is preserved as `policy-grid-audit-first.json`; the extension is in
`policy-grid-audit.json`.

This is a nominal-versus-realized sampling-law mismatch, not a NaN failure or an
empirical estimate of rare-event frequency. Neither the audit nor review claims
to reconstruct an individual rare soak row. Numerical parity to the unchanged
sealed FP32 recipe and the engineering soak remain passed; exact on-policy tail
sampling and policy strength are not certified. A versioned sampler/floor decision
is explicitly next work before strength promotion, alongside the open G2 gaps.
