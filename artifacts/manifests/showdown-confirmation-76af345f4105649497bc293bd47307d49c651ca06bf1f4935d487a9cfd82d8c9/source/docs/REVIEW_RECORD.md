# Review and verification record

Planning snapshot: 2026-10-10. This record distinguishes source checks from unexecuted implementation requirements.

## Checks performed

- Confirmed the selected Pokémon workspace had no tracked implementation files or local changes before writing the plan. Created documentation only; copied the prior research handoff into this project.
- Independently inspected Prime's public hosted environment, full-finetuning, models/pricing, algorithms, sandbox, availability and provisioning documentation. Links and limitations are in [the assessment](PRIME_INTELLECT.md).
- Read the `HyperPotatoNeo/pokemon-RL` README to verify its Showdown/verifiers/prime-rl architecture and documented Gen9 OU/Gen1 Random Battle examples. Individual source-page fetches failed; no source-level correctness or test-pass claim is made.
- Ran read-only Prime CLI discovery queries. Hub and GPU availability were unauthorized. The availability command's empty array did not establish stock status. No resources were provisioned.
- Independently checked compute arithmetic: 140 GPU-hours; hypothetical GPU subtotals of $280/$560/$1,120 at $2/$4/$8 per hour; 20% planning reserves produce $336/$672/$1,344. Paired evaluation precision uses pairs as clusters, not independent swapped games.

## Cross-model Prime research

Run: `20261010-105746-63b773a3`, project `Pokemon-83aa7cb62b4e`. Fable supplied a research candidate; Codex supplied independent QA. The candidate is **rejected as a complete research result**, though individually verified facts were retained. This is a quality-record verdict, not an access or permission rejection.

Material issues verified by the coordinator:

1. The candidate's claim that all Prime offerings require LLM/vLLM policies was too broad. The sandbox documentation describes generic isolated compute and custom images; CPU Showdown tests are a plausible use. Current GPU sandbox access is restricted/announced as forthcoming, not a basis for claiming CPU infrastructure is irrelevant.
2. It missed the substantive `HyperPotatoNeo/pokemon-RL` reference. Its existence defeats the candidate's implication that the only credible Showdown-environment lead was an unrelated OpenEnv PR. Neither repository availability nor its README establishes a usable Gen9 Random Battles Hub package.
3. The candidate oversimplified algorithm extensibility. The official algorithms page shows a custom loss hook and a PPO-clip example. However, the QA response also overstated its correction: the documentation explicitly says a new algorithm is a named code class rather than an import-path configuration. Loss hooks and algorithm registration are different interfaces. The final assessment states only the verified extension capability and the continuing JAX/token-architecture mismatch.
4. The hosted-service transition and dedicated full-finetuning beta are supported by the public notice; live enforcement and account access were not tested. No claim of exclusively LoRA-based hosted training was retained.

The decision to use an ordinary GPU instance for our custom learner remains a design recommendation. Actual JAX compatibility, prices and capacity are unverified until a selected instance is benchmarked.

## Plan review

A separate Fable snapshot review completed: `20261010-110500-01b7119c`, project `Pokemon-83aa7cb62b4e`. It found specification gaps, not an implementation failure. A Codex subagent independently checked collection, checkpoint and statistical consistency. The following resolutions were verified by reading the affected contracts and checking the relevant math/source:

- Defined a frozen-actor collection barrier that resolves all included per-player transitions before updates, with explicit overflow/padding and pivot/re-request tests.
- Reconciled boundary-only checkpoint support with its actual acceptance tests. Persistent live games must be saved even at update boundaries; partial work rolls back with accounting. Exact mid-epoch continuation remains optional and separately gated.
- Replaced incompatible seed-based/team-based splitting with one canonical-team-content key. Online generation respects the same split; duplicate teams cannot cross it.
- Made both baseline actor and value head PlayerView-only. No privileged critic inputs enter a shared encoder or deployment search.
- Corrected the single-legal-action test: gradient and entropy vanish, but the scalar PPO surrogate can equal minus the advantage. The verification is algebraic: when legal probability is identically one, its parameter derivative is zero.
- Corrected the throughput projection to deduct setup and apply the declared 25% time allowance explicitly.
- Rejected the reviewer's recalled claim that local Showdown ignores explicit teams. Current `Battle.getTeam` returns a supplied team before generation and `setPlayer` calls it. R1/R4 now require runtime team-hash and seat-swap checks at the eventual pin. No custom rule modification is prescribed. [Source checked](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts)
- Clarified that replay equality normalizes declared nonmechanical timestamps/identifiers while retaining raw evidence.

The review snapshot predates the added `pokemon-RL` paragraph and extensibility clarification; those were checked directly against primary sources. Prime service facts are dated and must be rechecked before use. A local structural check confirmed all relative document links resolve, code fences are balanced and files contain no trailing whitespace. No code tests are represented as passing.

## Not performed at the planning stage

No Showdown revision was selected or cloned; no team corpus generated; no simulator/learner implemented; no training, differential tests, battle evaluations or GPU benchmark executed. Numerical gates in these documents are proposed future requirements. No Prime Hub candidate was installed or smoke-tested. Estimated effort, strength targets and compute budgets are not measured outcomes.

Execution update: R1 selected and built Showdown and ran local reference/replay tests; R2 implemented the versioned Node/Python contract; R3 generated and inventoried 100,000 official whole teams; R4 ran and replayed 1,000 episodes with independent Python validation. See [the R1 guide](R1_REFERENCE.md), [R2 review](R2_REVIEW.md), [R3 review](R3_REVIEW.md) and [R4 review](R4_REVIEW.md) for actual checks and remaining scope. The statements above describe the earlier planning stage only.


## R5 implementation and information boundary — 2026-10-10

R5 adds verified same-player history, fixed actor inputs, request-relative action
bindings and separate immutable training targets. Full regression: 52 Node and
41 Python tests passed. The sealed scan reconstructed 1,000 R4 episodes and
81,947 decisions, with 1,000 adapter pairs, 18 live hidden-state pairs, two visible
HP controls, 14 semantic probes, 90 Node/Python action round trips and all 13 R4
fixtures (334 decisions plus 20 bootstraps). Read [the handoff](R5_FEATURES.md),
[review dispositions](R5_REVIEW.md) and [source audit](R5_SEMANTICS_AUDIT.md).
The implementation review found an absorb-healing source-attribution bug; pinned
source inspection and eight live cases verified the correction. The research
candidate was rejected as a complete specification and recorded honestly in the
cross-model ledger. The scan report is `d0f4c82bbc85df7ed1cf54eafa2d3f825decfa58577179cb1ba64eb2a26b66b5`. All data remains
exploratory/unassigned. R6 is next; this stage required local CPU only.


## R6 mechanics inventory — 2026-10-10

R6 inventories 3,509 conservative entries and 10,710 source sites, covers all
1,172 observed initial IDs, and captures 16 scripted reference fixtures in both
seats with exact replay. All custom-engine support remains blocked. The static
review identified a chained-call ID collision; Python rejected the draft, and
adding source end bytes resolved it. See [the handoff](R6_MECHANICS.md) and
[verified review dispositions](R6_REVIEW.md). The bounded research delegate
timed out without a result; no research approval is claimed. R7 is next.

## R7 dataset boundary and benchmark — 2026-10-10

R7 completes the reference/data foundation with a frozen 90/5/5 ordered-team
policy, an audit of 102,000 exposed team identities, 128 fresh reserved final
teams and 96 new train/development episodes. All episodes replayed and passed
Python semantic validation; all 7,782 actor wires passed strict validation.
Collection with replay took 27.391 seconds on four local CPU workers; Python
validation and export took a separate 267.320 seconds. No final battles were run.

The full suite passed 59 Node and 66 Python tests. Fable's implementation review
findings were independently checked; publication-time source checks, seed
prechecks and explicit policy-boundary validation were strengthened. A bounded
Fable identity implementation passed opposite-model QA and corrected local
acceptance checks; the original routing verdict remains unverified because its
supplied test made a no-op mutation. An initial worker-startup failure was fixed
and retained separately from the accepted benchmark. See the [sprint handoff](R7_SPRINT.md)
and [review dispositions](R7_REVIEW.md) for the exact evidence and limits.

M1/G1 CPU learner canaries are next, followed by an explicitly restricted
simulator slice. All custom-engine support remains blocked by the R6 gate.

## M1/G1 CPU learner and restricted transition slice — 2026-10-10

The three predeclared bandit seeds and three delayed shared-outcome seeds passed
the ≥99% optimal-action target within their fixed budgets. GAE/PPO, masking,
analytic gradients and whole-episode checkpoint rollback checks passed. A fresh
rerun reproduced all six checkpoints and six traces byte for byte. The full
regression suite passed 61 Node and 85 Python tests.

The first explicit fixed-team profile, `gen9-rattata-duel-v1`, has an independent
integer CPU transition specification. It matches 10,000 pinned reference cases,
31,303 transitions and all 32 crit/roll combinations with zero disagreements,
conditional on oracle-recorded semantic chance outcomes. This does not complete
G2 or implement JAX. Public-history parity, sampler validation and the accelerated
port remain next work; final-evaluation teams remain unused.

Two Fable implementation reviews were independently checked and actionable gaps
were addressed. Bounded Fable research received Codex QA and was rejected as a
complete executable specification; verified damage arithmetic was retained while
ambiguous roll reconstruction and an incorrect supplied-team claim were discarded.
See [the handoff](M1_G1.md) and [detailed dispositions](../artifacts/reviews/m1-g1/review-dispositions.md).

## M1/G2 restricted JAX slice

The fixed Rattata duel has a batched CPU JAX implementation, independently addressed
chance sampler and semantic player-view comparison against 1,000 fresh reference
games. The older 10,000-game transition corpus remains intact. Full G2 is open;
see [M1_G2.md](M1_G2.md) for the exact scope and remaining hidden-set/coverage gates.

A bounded Fable reducer implementation received Codex QA. Its runner check omitted
PYTHONPATH, so that delivery is honestly scored rejected; independently executed
checks and local corrections support the integrated code. A separate Fable snapshot
review identified low-severity provenance, imported-state and evidence-interpretation
issues, which were verified and resolved. Original reviews, dispositions and final
checks are archived under `artifacts/reviews/m1-g2`.


## M1 JAX autodiff learner and collector

A bounded Fable numerical-core implementation passed its five declared checks,
received accepting Codex QA, and was independently verified with expanded boundary/
finite-difference tests and fixed-budget numerical campaigns. A separate Fable review
of the observation-only collector and checkpoint boundary found no blocking issue.
Verified low-severity improvements make disk resume evidence explicit, derive report
assertions from actual buffer data and correct the collection-failure test label.
See [M1_JAX_LEARNER.md](M1_JAX_LEARNER.md) and
`artifacts/reviews/m1-jax-learner/review-dispositions.md`. Full G2/G3 remain open.

## M1 fixed-shape loop, CPU soak and profiling

The complete fixed-shape collect/GAE/PPO/Adam cycle passed its two-hour CPU soak:
9,885,926 cycles and 3,964,679,976 submitted decisions in the fixed public duel.
No numerical, legality, capacity or unexpected-compilation failure was detected.
Operational throughput was 550,648 decisions/s; peak process RSS was 566.344 MiB.
This does not establish full-format throughput, policy strength or new mechanic
coverage. Full regression passed 206 tests; 37 disk restores reproduced the next
cycle exactly across parity and soak audits.

Fable's bounded numerical helper received Codex QA and passed independent checks
before integration. Separate source reviews of the loop and completion sealer
identified a vacuous compilation-log matcher and weak evidence bindings. These
were independently verified and corrected, including a real JIT positive control,
source-bound regression logs, checkpoint identity/counter checks and preservation
of the interrupted preliminary run. Long-run diagnostics also exposed very small
nominal policy probabilities; an exhaustive FP32 uniform-grid audit documents the
inherited inverse-CDF sampler's tail precision limit. The sealed recipe remains
unchanged, and a versioned precision/floor decision precedes strength promotion.
See [the handoff](M1_FIXED_LOOP.md) and
[verified dispositions](../artifacts/reviews/m1-fixed-loop/review-dispositions.md).

## M1 binary sampling policy

The new two-action law uses an explicit 2^-24 floor, integer-word selection and
matching log/log1p probabilities. A scalar Fraction oracle verifies 100,002
thresholds, with 600,012 injected boundary selections; three-seed loop qualification
and a 60-second CPU run pass. This does not transfer the old two-hour soak to the
new policy, resolve Showdown chance mapping or promote G2/G3. See
[the handoff](M1_BINARY_POLICY.md) for current results and scope.

Auto routing explored a Codex implementation delegate. It failed ownership checks
and supplied pytest tests despite the unittest/no-new-dependency requirement. The
runner refused QA/adjudication of a failed run; its failed status is retained rather
than recorded as a success. The coordinator salvaged the inspected scalar source,
wrote independent unittest checks and obtained Fable snapshot QA. That review found
no blocking policy issue; the inactive rare-field sentinel difference and pinned
tie-gradient convention are documented.

A focused Fable qualification/sealer review prompted diagnostic bounds and explicit
test-suite identities. Its proposed exponent-range expansion was rejected after
checking IEEE bit patterns: exponent126 admits only exactly .5 in this contract.
Unused inherited schedule fields were removed, the actual checkpoint schedule was
made explicit, and the short campaign was rerun under the cleaned plan. Both runs
and the source-bound final regression are preserved in the completion archive.


## M1 Showdown chance mapping and expanded player views

The finite uint32 maps now match the pinned actual Showdown PRNG for critical
hits, damage rolls and context-aware two-element shuffles. All 100,096 upstream
word probes match (500,480 output comparisons); a separate regeneration is byte
identical. The 10,000-game fresh public-duel campaign passes 180,000 player-view
comparisons, including terminal padding, with zero disagreements. Three short
learner runs use the new chance contract and reproduce the next cycle after disk
restore. Full regression passes 229 tests. See [the handoff](M1_SHOWDOWN_CHANCE.md).

Auto routing selected Fable for the bounded arithmetic implementation. Codex QA
identified a hypothetical mixed-device admission edge; the coordinator accepted
only the admitted CPU execution after independent integer-proof, fixture, ownership
and device checks. The original QA verdict and scoped acceptance are both preserved.

Fable's main snapshot review raised evidence questions that were checked against
the recursive reservation gates, predecessor source, actual actor stream and oracle
regeneration. A direct three-seed predecessor collector comparison now confirms
that replacing only the chance dependency produces the same buffers and terminal
states. All 15,626 captured two-shuffle chains were checked separately. A focused
sealer review found no blocking issue; historical report checksum and delegate
outcome bindings were tightened. The source-bound regression was rerun after the
added check. See [verified dispositions](../artifacts/reviews/m1-showdown-chance/review-dispositions.md).

These results do not establish sodium/raw-seed/word-schedule equivalence or cover
hidden-set and broader mechanics strata. The new binary-policy/chance combination
still requires its own two-hour qualification, then preregistered strength evidence.
Full G2/G3 remain open; final evaluation remains reserved and Prime unnecessary.

## M1 selected policy/chance two-hour CPU qualification

On October 11, 2026 the selected binary-policy-1 / showdown-chance-1 combination
completed its own 7,200.006915-second CPU qualification: 8,158,970 complete cycles,
522,174,080 fixed-duel games and 3,272,073,210 submitted decisions. No numerical,
legality, capacity, completion or unexpected-compilation failure was detected.
Operational throughput was 454,454.176 decisions/s and peak process RSS was
473.765625 MiB. The maximum paired-clock host gap was 0.918060 seconds against
the frozen 30-second bound. These measurements describe the tiny public duel under
the observed local machine load; they do not establish full-format performance,
policy strength or device/system memory headroom.

The production report, source/runtime admission and final reservation checks pass.
All 120 minute telemetry rows and their checkpoint counter linkage were verified.
All 13 saved boundaries reproduce the persisted original next-learner and
next-metrics fingerprints exactly in a separate verification process. The existing
235-test regression (65 Node + 170 Python) remains valid for the unchanged manifest
of 206 tested source files; no unrelated tests were rerun for completion.

Fable's initial bounded summary-gate delivery failed two malformed-input exception
checks identified by Codex QA and reproduced locally. That rejected outcome is
retained. Its repaired delivery received accepting Codex QA and passed independent
ownership, input-hash and regression checks before integration. A separate Fable
harness/sealer review identified the macOS monotonic-clock suspension risk; paired
wall/monotonic gap checks and targeted tests resolved it before the run. Other
verified changes added timed-window bounds, final-replay gap checking, a smoke
evidence audit and preservation of incomplete historical reports. Focused follow-up
review found no blocking issue; its requested smoke audit was executed successfully.

Qualification Python, plan, sealer and all sealed predecessors remained frozen
during the production run. Both smoke campaigns and original review dispositions
are preserved. The [handoff](M1_POLICY_SOAK.md),
[verified review dispositions](../artifacts/reviews/m1-policy-soak/review-dispositions.md)
and [completion receipt](../artifacts/reports/m1-policy-soak.json) record the scope
and evidence. Only the selected combination's two-hour stability item closes.
Full G2/G3 remain open; no strength evaluation or final reserved-data use occurred,
and Prime remains unnecessary.

## M2 restricted-policy development evaluation

The locally preregistered pilot completed all 27 cells and 54,000 pinned Showdown
games. Three fresh training seeds each ran exactly 2,048 updates, with fixed
checkpoints at 0/256/2,048. Final random-opponent scores were 63.65%, 62.35% and
62.30%; all adjusted score and paired-improvement bounds met the declared screen.
The stronger damage-greedy baseline held the final policies near 50%, so this is
learning evidence within the public duel, not general strategic strength.

All 169,024 candidate actions, behavior log probabilities and values were reproduced
from each player's recorded request prefix. Public executed actions, terminal turn,
winner, roster, chance-seed namespace and all nine next-update checkpoint fingerprints
were verified. The full regression passed 251 tests (68 Node + 183 Python) with
218 tested source files unchanged. The earlier regression EOF stall and smoke
failures remain recorded; no sealed predecessor was repaired or bypassed to pass.

Fable's bounded statistics delivery initially failed independent overflow probes
despite accepting Codex QA. It was rejected and repaired. The repaired module
matched an independent complete resampling matrix; coordinator acceptance is scoped
to the frozen allocation budget, with Codex QA's generic resource-exhaustion caveat
retained. A separate Fable harness review found no blocking issue. Verified repairs
tightened terminal/roster checks, made the local replay constraint explicit and fixed
the worker EOF lifecycle. No production outcomes informed these repairs or the
frozen endpoint. See [the handoff](M2_POLICY_PILOT.md),
[verified dispositions](../artifacts/reviews/m2-policy-pilot/review-dispositions.md)
and [completion receipt](../artifacts/reports/m2-policy-pilot.json).

Full G2/G3 remain open. Critic errors/reliability bins are descriptive; no calibration
acceptance or device-memory headroom is certified. All 128 final teams remain
reserved, no final episodes were used, and Prime remains unnecessary.


## M3 switching/hidden-set reference scope — 2026-10-11

The [handoff](M3_SWITCH_REFERENCE.md) and
[receipt](../artifacts/reports/m3-switch-reference.json) record the accepted reference
expansion. All 264 tests pass (75 Node + 189 Python), bound to 235 unchanged tested
source/configuration files. The complete 72-script bundle is generated twice and
matches byte for byte. Python independently checks 1,826 live/wait action masks,
5,426 actor inputs, 1,872 hidden-state invariances, positive own/reveal controls,
all reported coverage counts and 3,456 conditional damage values.

Two Fable snapshot reviews were collected using the claude-review skill. Confirmed
findings prompted a separate two-member actor adapter, negative HP tests, complete
counterfactual view retention/reconstruction, independently counted coverage and
exact roster validation. The final evidence expansion is a locally verified repair,
not a claim of a third snapshot review. Review/source deltas, failure dispositions
and corruption probes are retained under `artifacts/reviews/m3-switch-reference/`.
The first full regression passed; it was rerun after the review repairs to bind the
final tree. Draft generation failures remain preserved. A failed codec delegation
was caused by the coordinator's repeated ownership option; none of its candidate
was integrated, and its unavailable QA/adjudication is explicitly recorded.

Full G2/G3 remain open. No new JAX port or learner has been implemented in this
profile; no final data, strength evaluation or Prime resources were used.

## Playable Showdown baseline — 2026-10-11

Implemented a separate public-view heuristic and native WebSocket client. The
accepted local benchmark has 64 complete games across 32 policy-seat pairs,
60 wins against the weak random baseline, 2,904 exact actor-input/action replays,
and zero choice rejections. Three real loopback-server games exercise login,
challenge acceptance, same-connection rematch, fresh login and terminal agreement.
All 134 network choices used the latest received public turn. Twenty new focused
tests and a passing 284-test full regression (95 Node + 189 Python, 247 unchanged
tested source files) are recorded with the completion receipt.

Fable protocol exploration underwent opposite-model Codex QA and was rejected for
insufficient pin-specific evidence; its useful modern-PM challenge finding was
independently verified in the pinned source. The Fable implementation snapshot
raised four issues. Pinned sendUpdates ordering and an oracle test disprove the
main request-before-public-state concern. A real reversed-endpoint challenge-clear
bug was fixed and verified by a same-connection rematch. Developer-format validation
limits and fatal-popup behavior are documented. The reviewer did not execute tests;
subsequent local corrections and snapshot deltas are explicitly preserved.

The first 64-game production replay found a serialization discrepancy after
Illusion replacement; that run is not qualified. Preserving the prior public HP
fixes it, and the replacement run passes every replay. Historical source, handoffs
and evidence remain unchanged except the six established current-progress/registry
files. No public login, ladder play, learning, final reserved-data use or Prime
resources were involved. Full G2/G3 remain open. See [handoff](PLAYABLE_BASELINE.md)
and `artifacts/reviews/playable-baseline/review-dispositions.md`.

## Bounded Showdown session recovery — 2026-10-11

Added a versioned session wrapper and CLI without modifying sealed baseline or
learner source. It reconnects, reauthenticates and rebuilds already-known active
rooms from player-visible history/current requests, respecting the server's
accepted-choice marker. Three reconnects are allowed by default; errors outside
the declared transport class remain visible failures.

The accepted run `artifacts/runs/showdown-session-v1` completes six real local
games through seven injected disconnects. Both players agree on terminal results;
all 120 attempted actions and actor inputs pass the saved-wire audit. One action
is deliberately lost before transport and reconsidered from a fresh request;
another is confirmed accepted and never resubmitted. Nineteen new focused tests
and the full 303-test regression (114 Node + 189 Python) pass with 254 unchanged
tested source files. Four final evidence corruption probes are rejected.

Fable snapshot review identified a real close-handshake error-classification bug.
The repair is confined to the new layer; two regression cases verify decision and
rejoin sends while the socket is closing. Other findings were checked against the
pin, including actual RoomBattle methods driven by a real hidden-trapping rejection.
The additional code/tests are locally verified review repairs, not a second Fable
approval. Review source deltas and prior drafts remain in the review directory.

Independent deployment research received opposite-model Codex QA. Its packet was
rejected for overstating the availability of registration and labeling technical
limits as policy. The coordinator directly read the official announcement in the
browser, verified dates and pinned code, and wrote a corrected deployment note.
The user still needs to identify public accounts; registry rollout, live protocol
compatibility and public login remain unresolved. No staff contact or public game
occurred. The [handoff](SHOWDOWN_SESSION.md) and [receipt](../artifacts/reports/showdown-session.json)
record local reliability evidence only. Full G2/G3 remain open; no learning,
reserved final data or Prime resources were used.

## Frozen Showdown opponent panel — 2026-10-11

Added a four-policy development matrix, public-view damage/tactical heuristics,
paired bootstrap statistics, exact replay auditor and immutable completion sealer.
The fixed 768-game production run covers 128 admitted development teams and all
six unordered matchups with 64 mirrored pairs each. All 35,711 actor inputs/actions,
outcomes and intervals reproduce exactly. All 338 tests (149 Node + 189 Python)
pass, bound to 266 unchanged source/configuration files. Thirty-five focused tests
include complete-grid statistical negatives, fixed known-answer PRNG/bootstrap
fixtures, baseline driver parity, hidden-reserve invariance, actual pinned
Psych Up/Power Swap events and deliberately corrupted smoke evidence. Four
production-evidence copies fail the sealer for their intended reasons.

The tactical heuristic scores 75/128 against the original baseline and 73/128
against damage. Its score against random is 115/128, below baseline's 118/128.
The per-edge paired intervals are descriptive, not multiplicity-adjusted promotion
or no-regression evidence. No policy was changed after production began. The
qualified network client still uses the original baseline.

Two Fable snapshots identified public-memory and bookkeeping issues. Confirmed
Illusion contamination, selective stage events, active-entry hazard penalties and
bench first-turn eligibility were repaired. The second snapshot's Illusion finding
was independently reproduced with previously known disguise moves/items and fixed
using per-appearance reveals. Actual engine events check boost-copy ordering;
source_manifest coverage is explicitly verified. Remaining heuristic approximations
(including copied-ability persistence) are documented. Snapshot/source deltas and
locally verified repairs are distinguished; no third review is claimed.

The delegated statistics implementation originally failed its own duplicate-ID
error-message assertion and received a rejected adjudication after Codex QA.
A fresh isolated repair changes only that message and adds literal numerical
fixtures; all 17 tests and opposite-model QA pass. Coordinator integration checks
include selected-input hashes, patch scope, local execution and independently
computed Python uint32/index/bootstrap values. Both outcomes remain archived.

Collection wrote a complete report and 768 games. Its optional /usr/bin/time
wrapper then exited nonzero because sandbox policy denied kern.clockrate; this is
preserved as instrumentation failure, not a game failure. Fresh replay exits zero
in 22.145 seconds. The initial corruption-test seed mutation and system-Python
probe mistakes were corrected using the actual sodium seed string and project
virtual environment; no producer evidence was changed.

See [handoff](SHOWDOWN_OPPONENT_PANEL.md),
[receipt](../artifacts/reports/showdown-panel.json) and
`artifacts/reviews/showdown-panel/review-dispositions.md`. Next freeze fresh
confirmation and qualify an additive network adapter. Full G2/G3 remain open;
no learning, public games, final reserved-data evaluation or Prime resources
were used.

## Fresh tactical confirmation and local adapter — 2026-10-11

The fixed production campaign completes 12,000 games across six edges, each with
1,000 mirrored team/seed pairs. All 2,000 official teams pass the frozen development
partition and exact-team exclusion against the earlier panel and both milestones'
fixture pools. A separate process regenerates the teams and replays every game:
all 564,379 player inputs/actions, results, counters and statistical bounds match.
Campaign/replay elapsed times are 321.364/313.062 seconds, both exit zero. No
policy tuning, threshold changes, failed-pair replacement or sample extension occurs.

Tactical scores 60.65% directly against baseline, exceeding the 52% point floor;
the nominal adjusted lower bound is 58.55%, above 50%. Its paired improvements
over baseline against random and damage are +2.20 and +10.05 percentage points,
with lower bounds +0.80 and +7.40, both above the declared −3-point tolerance.
All three conditions pass. The 30,000 shared-pair bootstrap replicates implement
an approximate Bonferroni development screen, not guaranteed coverage, a reserved
final promotion or a human/ladder strength estimate. Thirteen unavailable-choice
attempts are retained: five baseline and eight random switches rejected as trapped.

The additive tactical network adapter completes six actual local-server games
through seven injected disconnects. Its wire auditor reconstructs every one of
179 attempted decisions, including a deliberately lost write and a pending
accepted action that is not resent. Four mutated production traces/reports fail
for their intended reasons. Seven temporary production-evidence copies fail the
sealer at the intended failure/progress/source/game/report/network checks, with
no receipt or archive writes. The full 380-test regression (191 Node + 189 Python)
passes, bound to 283 unchanged tested source/configuration files shared by the
campaign and network admissions. Build and syntax verification pass.

A Fable statistics implementation attempt times out without a completed tested
delivery; none of its partial module is integrated. The runner refuses QA and
adjudication of that incomplete run, so neither is claimed. The coordinator's
implementation receives a separate focused Fable snapshot and seven executed
statistics tests, including a literal independently calculated three-metric
fixture. A separate Fable adapter/pipeline snapshot prompts verified guard,
ordering-test and freshness improvements. Subsequent sealer checks are locally
verified repairs; no third snapshot approval is claimed. Snapshot/source deltas,
failed delegation and the first sandbox-denied loopback preflight are preserved.

The [handoff](SHOWDOWN_TACTICAL_CONFIRMATION.md) and
[receipt](../artifacts/reports/showdown-confirmation.json) qualify the unchanged
tactical heuristic for local development play. The separate 1,000-game real
network soak, public deployment checks, expanded scalar/JAX mechanics and broader
learner remain outstanding. Full G2/G3/G4 remain open. No training, public games,
final evaluation data or Prime resources are used.
