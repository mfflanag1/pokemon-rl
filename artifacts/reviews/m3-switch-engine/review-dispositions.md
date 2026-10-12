# M3 switch-engine cross-model review dispositions

Reviews are frozen snapshot analyses, not executed verification or blanket approval.
The coordinator verified findings against pinned source, tests and live evidence.
Historical sealed sources/receipts are unchanged except the six documented current
progress/registry exceptions. All substantive new implementation is additive.

## Delegated scalar damage helper

The initial Fable implementation was byte-correct against the pinned 3,456-value
reference damage grid. The coordinator's first stat test incorrectly expected
Fennekin's swift Speed to be 111, and the delegate incorrectly reported every test
passing despite that failed test. Opposite-model Codex QA identified this mismatch.
The initial result was recorded as unverified, not accepted.

The coordinator checked the sealed reference fixtures (80/112), the stat formula
and all candidate input/output hashes. The corrected independent rerun supplied
byte-identical damage code, passed all three controller tests and received Codex
QA acceptance. Coordinator integration applied only the owned damage.py and reran
all three tests, including the entire grid. Both outcomes and their checks are
preserved in delegate-initial/ and delegate-corrected/. No model opinion substitutes
for those local checks.

## Mechanics review

1. Rejected proposed HP-rounding defect. The reviewer assumed hp/maxhp*100; the
   actual pinned vendor/pokemon-showdown/sim/pokemon.ts:2077 uses 100*hp/maxhp.
   Multiplication is exact in this closed scope. Its 66/120 result is 55, not 56.
   A briefly applied local alternative was reverted before any production work.
   The regression directly calls real Pokemon.getHealth() for all 357 HP values
   across max HP115/119/120 and compares both scalar and JAX results.
2. Mapping context gap resolved by reading unchanged pokemon_showdown/mapping.py:
   roll is the factor85..100, so 100-roll is the required index0..15; crit is bool,
   tie is int32. Tests independently derive per-role words and integer mappings,
   cover batch reordering and all outcome supports. No raw sodium equivalence claim.
3. Accepted chance-trace binding gap. Each executed move must have exactly one
   critical and roll trace for its seat, equal to conditional replay input. Ties
   require tied outgoing speed/class. The first stronger test exposed Gen9's
   repeated move-queue sort: battle.ts:2917-2923 re-sorts before first execution.
   Every captured tie must precede execution; the final shuffle binds effective
   order. All word maps remain independently checked. The failed diagnostic log
   focused-v1.log is preserved; focused-v2 passes.
4. Accepted passivity gap. Controls now use entirely unpatched Sessions (no wrapper
   installation), comparing each real receiving view and privileged audit including
   PRNG state. Production and sealer separately rerun the first24 fresh cases.

## Learner review

No correctness defect was found in conditional sampler/log likelihood, gradient
masking, own-player GAE linkage, terminal/cutoff bootstrap, counters or original
checkpoint replay. This remains a snapshot finding, with executed tests separate.

1. Stale partial file behavior is intentional evidence preservation, following the
   sealed predecessor's rule. The published checkpoint remains valid; a new save
   explicitly rejects until .partial is preserved under another name. No deletion
   workaround was adopted. Error wording, contract and interrupted-save/re-save
   tests now state and check this behavior.
2. Accepted vacuous assertion. First-update bootstrap is asserted exactly zero
   because initial value head is zero. The original next update after disk save
   must have nonzero cutoff bootstraps, while a separate hand-computed wait-gap
   fixture verifies exact nonzero targets. Three numerical bandit seeds are only
   math canaries, not Pokemon strength evaluations.
3. Subsequent host-boundary cleanup makes learner.run delegate to the validated
   transaction, validates buffer shapes/dtypes/action ranges and requires Threefry.
   These guard additions preserve the reviewed math and are exercised by regression.

## Campaign and sealer review

1. Oracle-runtime concern was conditional on omitted context. The actual oracle is
   vendor/pokemon-showdown, not an unbound installed simulator: reference() already
   calls sealed source/build verification. Added a stronger fresh Node audit that
   compares saved plan.reference against current reference(), rechecking upstream
   commit, dirty state, build and harness hashes. Its deliberate manifest-drift
   negative test must reject.
2. Accepted passivity receipt gap: the same read-only Node audit reruns all24 controls
   at seal time and must reproduce oracle-audit.json exactly.
3. Transaction failure checks concern in-memory publication, not disk crash recovery.
   The code is intentionally functional; checks would detect accidental future
   mutation. Counts are explicitly six collection/update-discard checks. Interrupted
   atomic disk write, stale partial and intact prior checkpoint are separately tested
   in regression. No mid-game or partial-buffer recovery is claimed.
4. Rejected claim of absent instrumentation regression. test_engine.py runs the
   actual Node capture, all72 sealed scripts, 24 fresh cases, independent parity
   and Node passivity/runtime audit. This is registered as a Python integration
   test, so the unchanged237 Node test count does not imply untested Node code.
5. Sealed fixture hash already prevents truncated fixture admission; nevertheless
   information_checks now explicitly requires1152 comparisons and derives counts.
6. Accepted legacy identity strengthening. Independently added before this review
   completed: Python now compares legacy specs, every view/audit, actions and case
   counts to the sealed original fixture; fresh spec/seed/actions/shard identities
   are checked against the fixed schedule.

No review establishes public playing strength, broad G2/G3/G4, or a sustained
qualification for this new policy. Those gates remain open. Prime is unnecessary.
