# F21 — frozen full-battle architecture comparison

Status: completed and replayed; battle-benefit gate 0/3 seeds.

F21 completed (2026-10-25): evaluated all final F20 flat/shared checkpoints and the F10 rehearsal reference across 6,912 paired-development battle executions in 326.96 seconds on CPU, with zero optimizer updates or checkpoint selection. The shared-versus-flat battle-benefit gate passed in 0/3 seeds. All 27 contrasts remain: 2 improvements, 2 regressions and 23 inconclusive under the registered approximate adjusted bootstrap. Outcomes: 6,908 natural terminals, 4 unresolved cutoffs, 425,793 total attempted decisions and 214,191 learner decisions. All 21 focused tests, exhaustive fresh battle replay and nine immutable checkpoint loads pass. F20 familiar mastery failures remain; no policy is promoted. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer.

F20’s stronger teacher imitation did not establish stronger tactical play in complete battles. Descriptive pooled tactical wins were 53/768 for shared (6.90%), 54/768 for flat (7.03%) and 33/768 for the F10 reference (4.30%). Shared-minus-flat tactical differences by seed were 0.00, +1.95 and -2.34 percentage points; all adjusted intervals include zero. The primary battle-benefit gate has 0/3 passing seeds because neither tactical requirement passes in any seed. Shared’s earlier familiar-context failures also reproduce in seeds 8209 and 9311.

The shared model meets the approximate adjusted-bootstrap improvement criterion against flat on random in two of three seeds, with the third inconclusive. Descriptive random wins are 694/768 shared, 611/768 flat and 685/768 reference; there are three unresolved flat cutoffs and one reference cutoff. Against base-power, wins are 324/768 shared, 320/768 flat and 290/768 reference. No shared-versus-reference comparison establishes an improvement. Flat meets the approximate adjusted-bootstrap regression criterion against reference on random in two seeds. All 27 contrasts are retained: 2 improvements, 2 regressions and 23 inconclusive under the approximate adjusted bootstrap; all conservative Hoeffding intervals contain zero. The pooled totals reuse schedules across seeds and are descriptive, not independent-game confidence calculations.

Production completed in 326.96 seconds on CPU, with 6,908 natural terminals, four unresolved cutoffs and no true ties. All 425,793 attempted decisions and 214,191 learner action records are preserved. The 172 unavailable-choice messages were all advertised switch attempts rejected because the active Pokémon was trapped; refreshed-request handling preserves them. No training or optimizer updates occurred. These results do not establish equivalence, a causal explanation of tactical weakness, or competitive/public strength. F10 remains the reference and no policy is promoted.

F21 falls in the mixed/inconclusive decision branch: retain F10 as the reference and do not claim a tactical winner. Next proposed step: return to full-battle RL with a controlled flat-versus-shared PPO comparison, starting from all three matched F20 final checkpoints and retaining F10 as an external reference. The shared scorer remains an unproven candidate: its matched-size implementation and limited random-opponent signal make RL learnability a testable hypothesis, not an established tactical benefit. First qualify the shared actor/value PPO loss, gradients and checkpoint recovery against the existing learner. Then preregister an equal decision/update budget and fixed-opponent learning curves; use battle improvement as the primary outcome, with familiar tactical cases and held-out self-play value diagnostics as supporting measures. Do not extend behavior cloning merely to improve teacher agreement. Structured observations, history, opponent auxiliary heads and historical-opponent cross-play remain separate candidate interventions. Measure complete-cycle cost before selecting a longer CPU or GPU budget. No new training is launched in F21; local CPU sufficed for this evaluation, and Prime remains unprovisioned.

## Every seed, policy and opponent

Each cell has 256 games from 128 paired schedules. Entries are wins / losses / true ties / unresolved cutoffs. Counts are actual terminal outcomes; administrative cuts are never draws. The same schedule is used across policies and seeds, so pooled counts are descriptive, not independent-game precision estimates.

| Seed | Opponent | Flat | Shared | F10 reference |
| --- | --- | --- | --- | --- |
| 7103 | random | 201 / 53 / 0 / 2 | 240 / 16 / 0 / 0 | 230 / 26 / 0 / 0 |
| 7103 | base_power | 107 / 149 / 0 / 0 | 106 / 150 / 0 / 0 | 102 / 154 / 0 / 0 |
| 7103 | tactical | 14 / 242 / 0 / 0 | 14 / 242 / 0 / 0 | 8 / 248 / 0 / 0 |
| 8209 | random | 207 / 49 / 0 / 0 | 222 / 34 / 0 / 0 | 227 / 28 / 0 / 1 |
| 8209 | base_power | 107 / 149 / 0 / 0 | 105 / 151 / 0 / 0 | 100 / 156 / 0 / 0 |
| 8209 | tactical | 15 / 241 / 0 / 0 | 20 / 236 / 0 / 0 | 15 / 241 / 0 / 0 |
| 9311 | random | 203 / 52 / 0 / 1 | 232 / 24 / 0 / 0 | 228 / 28 / 0 / 0 |
| 9311 | base_power | 106 / 150 / 0 / 0 | 113 / 143 / 0 / 0 | 88 / 168 / 0 / 0 |
| 9311 | tactical | 25 / 231 / 0 / 0 | 19 / 237 / 0 / 0 | 10 / 246 / 0 / 0 |

Per-seat outcome counts, action counts, every likelihood/feature hash and terminal-only value diagnostics are retained in the report and game tape. F20 critics remain constant 0.5; the F10 critic has a different training history. Value metrics condition on each policy’s natural-terminal trajectories and do not establish training-distribution calibration.

## Gates and retained familiar-context failures

| Seed | Battle benefit | Failed battle checks | Flat / shared / F10 familiar mastery |
| --- | --- | --- | --- |
| 7103 | FAIL | tactical_adjusted_lower_positive, tactical_point_gain | PASS / PASS / PASS |
| 8209 | FAIL | tactical_adjusted_lower_positive, tactical_point_gain | PASS / FAIL / PASS |
| 9311 | FAIL | tactical_adjusted_lower_positive, tactical_point_gain | PASS / FAIL / PASS |

F20’s known Tera-variant predictions reproduce exactly. They remain separate diagnostic failures, even if a battle comparison improves. There is no automatic policy promotion, earlier-endpoint substitution or best-seed selection.

## All 27 preregistered contrasts

| Seed | Comparison | Opponent | Cutoff-aware point interval | Adjusted bootstrap interval | Hoeffding interval | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 7103 | flat_vs_reference | base_power | [+1.95, +1.95] pp | [-9.77, +14.06] pp | [-31.08, +34.99] pp | inconclusive |
| 7103 | flat_vs_reference | random | [-11.33, -10.55] pp | [-21.09, -0.98] pp | [-44.36, +22.49] pp | regression |
| 7103 | flat_vs_reference | tactical | [+2.34, +2.34] pp | [-3.52, +9.38] pp | [-30.69, +35.38] pp | inconclusive |
| 7103 | shared_vs_flat | base_power | [-0.39, -0.39] pp | [-12.89, +11.72] pp | [-33.43, +32.65] pp | inconclusive |
| 7103 | shared_vs_flat | random | [+14.45, +15.23] pp | [+6.25, +23.83] pp | [-18.58, +48.27] pp | improvement |
| 7103 | shared_vs_flat | tactical | [+0.00, +0.00] pp | [-6.64, +5.86] pp | [-33.04, +33.04] pp | inconclusive |
| 7103 | shared_vs_reference | base_power | [+1.56, +1.56] pp | [-10.94, +13.67] pp | [-31.47, +34.60] pp | inconclusive |
| 7103 | shared_vs_reference | random | [+3.91, +3.91] pp | [-3.12, +11.72] pp | [-29.13, +36.94] pp | inconclusive |
| 7103 | shared_vs_reference | tactical | [+2.34, +2.34] pp | [-3.12, +8.20] pp | [-30.69, +35.38] pp | inconclusive |
| 8209 | flat_vs_reference | base_power | [+2.73, +2.73] pp | [-7.81, +13.67] pp | [-30.30, +35.77] pp | inconclusive |
| 8209 | flat_vs_reference | random | [-8.20, -7.81] pp | [-18.16, +1.95] pp | [-41.24, +25.22] pp | inconclusive |
| 8209 | flat_vs_reference | tactical | [+0.00, +0.00] pp | [-5.86, +5.86] pp | [-33.04, +33.04] pp | inconclusive |
| 8209 | shared_vs_flat | base_power | [-0.78, -0.78] pp | [-12.50, +10.16] pp | [-33.82, +32.25] pp | inconclusive |
| 8209 | shared_vs_flat | random | [+5.86, +5.86] pp | [-5.47, +16.99] pp | [-27.18, +38.90] pp | inconclusive |
| 8209 | shared_vs_flat | tactical | [+1.95, +1.95] pp | [-4.69, +8.59] pp | [-31.08, +34.99] pp | inconclusive |
| 8209 | shared_vs_reference | base_power | [+1.95, +1.95] pp | [-9.38, +13.28] pp | [-31.08, +34.99] pp | inconclusive |
| 8209 | shared_vs_reference | random | [-2.34, -1.95] pp | [-11.72, +7.03] pp | [-35.38, +31.08] pp | inconclusive |
| 8209 | shared_vs_reference | tactical | [+1.95, +1.95] pp | [-4.69, +8.98] pp | [-31.08, +34.99] pp | inconclusive |
| 9311 | flat_vs_reference | base_power | [+7.03, +7.03] pp | [-5.08, +18.36] pp | [-26.00, +40.07] pp | inconclusive |
| 9311 | flat_vs_reference | random | [-9.77, -9.38] pp | [-19.14, -0.20] pp | [-42.80, +23.66] pp | regression |
| 9311 | flat_vs_reference | tactical | [+5.86, +5.86] pp | [-0.78, +12.50] pp | [-27.18, +38.90] pp | inconclusive |
| 9311 | shared_vs_flat | base_power | [+2.73, +2.73] pp | [-7.42, +12.50] pp | [-30.30, +35.77] pp | inconclusive |
| 9311 | shared_vs_flat | random | [+10.94, +11.33] pp | [+1.17, +21.88] pp | [-22.10, +44.36] pp | improvement |
| 9311 | shared_vs_flat | tactical | [-2.34, -2.34] pp | [-10.55, +5.47] pp | [-35.38, +30.69] pp | inconclusive |
| 9311 | shared_vs_reference | base_power | [+9.77, +9.77] pp | [-1.56, +21.09] pp | [-23.27, +42.80] pp | inconclusive |
| 9311 | shared_vs_reference | random | [+1.56, +1.56] pp | [-6.44, +9.77] pp | [-31.47, +34.60] pp | inconclusive |
| 9311 | shared_vs_reference | tactical | [+3.52, +3.52] pp | [-3.12, +9.77] pp | [-29.52, +36.55] pp | inconclusive |

Both interval procedures use Bonferroni adjustment over 27 contrasts at family-wise alpha 0.05. The paired bootstrap uses 20,000 resamples and has approximate coverage. Whole paired schedules, including both seats, are clusters. The conservative Hoeffding result is retained separately. Inconclusive is not equivalent, non-inferior or proof of no effect. The shared-minus-flat comparison matches F20 training; reference comparisons differ in training history and cannot isolate architecture.

## Collection, cost and verification

Production: 326.96 seconds; collection calls total 271.44 seconds; peak parent RSS 1816.50 MiB. Parent RSS does not sum all Node worker memory. Admission/snapshot startup precedes the timer; subsequent verification is separate. All 6,912 production games are development evaluation, with zero training games or optimizer updates. The independent replay executes the same 6,912 games afresh, separately from the reported production count.

Data roles exclude 16,368 prior training teams and 2,944 prior development teams from the specified F20/F10 histories. The panel uses 256 fresh exact ordered teams; disallowed overlaps are zero. This is not held-out species, set-family or opponent generalization. Unavailable-choice attempts: 172; feedback is retained rather than filtered.

All 21 focused checks pass on the frozen source/runtime tree (9 actor, 8 scope/statistics, 4 live collector). The pinned Showdown build passes. Original flat collection agrees exactly, including two known trapped-switch rejections; shared and flat sampling agree for identical logits and keys, with a non-vacuous changed-counter control. Separate initial/final engineering test suites executed 56 small game runs in total; these are not counted as production or replay evaluation.

Exhaustive replay matches every production chunk, all 214,191 controlled action likelihood/value records, all 425,793 attempted decisions and all outcomes, recovering the same nine frozen checkpoints. Production, verification and workflow exits are zero. Nineteen copied-evidence negatives reject. Source/evidence hashes and identical first/repeat receipts are verified after current completion documents are bound, with logs kept outside the sealed review directory.

[Completion receipt](../artifacts/reports/f21-shared-battle.json). [Original-objective review](OBJECTIVE_REVIEW_2026_10.md).

Production report SHA256: 681e8b746b9b3c293fad340445f49688eb254fb4daeb252724ca294ab80f9378.

## Preserved preregistration


Design recorded before production; all budgets, policies, seeds and gates retained.

F20 improved teacher imitation but passed its descriptive architecture gate in 0/3 seeds. This evaluation tests complete battle outcomes using every final F20 flat/shared checkpoint and the same-seed F10 rehearsal reference. The F10 reference has different training history; only shared versus flat is the matched architecture comparison.

The fixed panel contains 6,912 development games: three seeds (7103, 8209, 9311), three policies and three frozen opponents (random, base-power, tactical), with 128 paired schedules played in both seats per cell. All policies share team/mechanics schedules and action-random roots. Action changes can alter subsequent chance consumption. These are repeated schedules across seeds, not independent evaluation datasets.

There are no optimizer updates, early stopping, checkpoint selection, seed changes or budget extension. F20 endpoints are fixed at 8,192 updates; F10 uses its rehearsal final endpoint at 128 cohorts / 1,024 Adam steps. Fresh exact teams are screened against the admitted F20-relevant corpus/development roles and F10 training/evaluation roles. Opponent identities, format, species and familiar tactical contexts recur. Final reserved data are unused.

Primary battle benefit is shared-minus-flat: tactical cutoff-aware point lower bound at least five percentage points, adjusted bootstrap lower above zero, no detected random/base-power regression, and no shared cutoff rate above 5%. All three seeds must pass. Retain all 27 registered comparisons, including each candidate versus F10. Bootstrap intervals use 20,000 paired-cluster resamples, family-wise alpha 0.05 across 27 contrasts; coverage is approximate. Conservative Hoeffding intervals are also reported. Neither inconclusiveness nor absence of a detected regression establishes equivalence or non-inferiority.

Record wins, losses, true ties, unresolved 120-turn administrative cutoffs, both seats, legal-action traces, rejected-choice feedback, action counts, frozen identities and terminal-only value diagnostics. F20 critics remain constant 0.5. Familiar mastery is rerun unchanged and reported separately; its failed F20 Tera variants are not discarded. No policy promotion follows automatically from this screen.

The observation-only shared actor uses the same finite-word categorical law as F20. Flat inference delegates directly to the sealed actor. The new collector preserves sealed Showdown requests, player projections, frozen opponents and unavailable-choice handling. Preflight tests compare complete flat games with the old collector, check shared batching/worker independence, information boundaries, paired identities and cutoff bounds.

After review and source-bound checks, freeze the new source/plan/tests. Run one production panel and one exhaustive independent replay, recovering the same nine immutable checkpoints and reproducing every attempted action, likelihood, value and outcome. Preserve failures without restarting or stitching. Update current progress documents and the objective audit before sealing, then verify all live/archived hashes and identical repeat receipt. No public server, Prime or GitHub transfer. Full G2/G3/G4 remain open.

