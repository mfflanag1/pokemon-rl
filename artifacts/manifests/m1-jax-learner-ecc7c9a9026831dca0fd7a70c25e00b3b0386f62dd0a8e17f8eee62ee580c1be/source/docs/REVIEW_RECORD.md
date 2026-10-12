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
