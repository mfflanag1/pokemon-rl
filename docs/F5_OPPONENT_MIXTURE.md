# F5 — controlled opponent-mixture learning screen

Status: production completed; independent audit passed. The exploratory mixture gate did not pass.

F4 learned against random and base-power opponents but won only 5.4–7.5% against
the frozen tactical policy. F5 tests opponent distribution while holding the F1
928-feature, 64-hidden-unit, 14-action network and F2 PPO settings fixed. All
predecessor source and sealed evidence remain unchanged.

## Registered comparison

Three fresh initializations (4103, 5207, 6301), each shared between two arms:

- `self_play`: 100% self-play learner rows.
- `mixture`: 50% self-play, 25% base-power, 25% tactical learner rows.

Each arm gets 128 updates of exactly 1,024 distinct selected learner rows. Two
epochs with 256-row minibatches give 1,024 Adam steps per arm, 6,144 overall,
and 786,432 selected rows overall (1,572,864 row presentations). No F4 warm start.
Weights are fixed throughout collection for an update. Each stratum collects
complete eight-game batches until its row quota is reached, at most 64 games
per stratum per update; failure to fill the quota fails the experiment. Frozen
opponent games alternate learner seats. Self-play retains both players' rows;
frozen-opponent actions never enter the learner loss or same-player linkage.

Compute GAE on complete trajectories before uniformly selecting without
replacement the exact per-stratum quota, using a fixed independent selection
RNG. Excluded rows may supply trajectory context/targets but receive no direct
loss. Shuffle the combined selected batch for each PPO epoch. No leftover rows
carry across updates. Collection overshoot, all game/decision counts, selected
and excluded learner rows, opponent counts, and all timings are reported.
This matches optimization exposure, NOT collected environment interactions,
game counts or wall time. The control uses the same row-selection procedure;
it is not a rerun of F4's variable-size update recipe.
The intervention therefore changes opponent distribution, interaction cost and
within-update trajectory correlation together; it cannot isolate a pure opponent
effect at equal simulator exposure. No opponent identifier is added to the actor
or critic, so values average over the training mixture.

Train and development episodes use unchanged F2 admission of official whole
teams. Training episode/actor RNG namespaces are disjoint by arm, seed and stratum.
The optimizer shuffle intentionally shares `(seed, update)` permutations across
arms; selection RNG includes arm and stratum. Both arms
share initial parameters and the development panel; training game identities
are not paired. Curriculum, architecture changes, search and GPU are excluded.

## Evaluation and decision rule

Fresh development namespace 8522026, 256 team/mechanics pairs, both seats,
all three frozen opponents. For each seed evaluate initialization once and
both final arms: 13,824 evaluation games. No intermediate selection or budget
extension. The same actor RNG is used across endpoints. The official ordered
whole-team partition does not imply species/family-disjoint generalization.

Report all 27 contrasts: each arm versus initialization (18), and mixture
versus self-play (9). Paired team/mechanics-cluster bootstrap, 20,000 resamples,
Bonferroni family 27, alpha .05, seed 8522027; both seats/endpoints remain in
each resampled cluster. Administrative cutoff rewards range [0,1], not draws.
Report cutoff-aware point bounds, approximate bootstrap bounds and conservative
finite-sample Hoeffding bounds. Bootstrap coverage is approximate. Shared teams
across seeds/opponents do not create independent team-level replications.

The exploratory mixture gate requires, for EVERY seed: tactical mixture versus
self-play has positive bootstrap lower bound and conservative point improvement
at least .03; tactical mixture versus initialization has positive bootstrap
lower bound; no mixture-versus-control opponent contrast has an upper bound
below zero; every final evaluation cell has cutoff rate <= .05. Report every
component even if the experiment completes with a failed/inconclusive gate.
The small panel and correction across 27 contrasts can leave modest effects
inconclusive; a failed gate is not evidence that the mixture has no effect.
Absence of a detected regression is not evidence of noninferiority. Terminal-only
value error and mean value are descriptive, conditioned on policy-dependent
terminal outcomes, not calibration certificates.
Evaluation includes the same heuristic identities used in mixture training on
different official whole teams; it does not measure unseen-opponent transfer.

## Validation and preservation

Before production: cross-model design/implementation review, meaningful tests
for learner-only rows and same-player links, exact update budgets, stratum
selection, actor isolation, train/development admission, pairing/cutoffs,
checkpoint recovery, plus a separate bounded preflight. Source/runtime admission
and immutable update checkpoints bind all production evidence. Failure stops
the workflow; no automatic experiment repair/restart.

Completion requires all counters and evaluation summaries recomputed, original
last-update replay for all six seed/arm checkpoints and sampled evaluation replay
for each seed/endpoint/opponent. This bounded replay is not exhaustive independent
replay of every game. Receipt inventories bind source snapshots, original evidence,
tests, review dispositions and this handoff, with identical repeat sealing.

Final reserved data and public servers remain unused; full G2/G3/G4 and public
ladder strength remain open. Prime is unnecessary. GitHub publication remains
held by the existing unresolved visibility choice.

Curriculum is a subsequent separate experiment. Existing C1 switching/Tera
prefixes often execute the choice before learner control; revise decision points
and define held-out scenario families before claiming decision learning or
curriculum transfer. The old 3,000-episode schedule is not executed by F5.

## Pre-production evidence (2026-10-23)

86 focused tests pass (44 Node, 42 Python), bound to 404 source/configuration
files. The pinned Showdown rebuild preserves the compiled-tree hash and receipt.
Independent design, implementation and focused replay-order reviews are preserved
in `artifacts/reviews/f5-opponent-mixture`; the Node adapter also had opposite-model
QA. The first recovery-test failure and its verified fixed-order repair remain
recorded. No production seed was used to tune that repair.

Separate preflight: 96 training plus 72 evaluation games, 166 natural terminals,
2 administrative cutoffs, 14,288 decisions, 4,096 selected / 7,151 collected
learner rows, 3,055 excluded learner rows, and 32 Adam steps in 14.587 seconds.
Both original final-update replays and sampled paired evaluation replays pass.
The preflight is infrastructure evidence only and is excluded from production
learning results. Production uses the registered three seeds and fixed budget.


## Measured production result (2026-10-23)

Experiment completed and its audit passed. The preregistered mixture gate **did not pass**.

**Decision: do not promote this mixture as an improvement.** Tactical wins were
lower for the mixture in every seed (15/19/20 versus self-play's 28/23/23, each
out of 512), but all three tactical difference intervals include zero. Thus no
tactical gain was established; this does not prove that all opponent mixtures
are ineffective. Seed 5207's base-power difference was −8.98 percentage points,
with adjusted approximate bootstrap interval [−17.09, −1.08]. The conservative
Hoeffding interval includes zero, so that regression finding remains specific to
the preregistered exploratory inference method.

Both recipes learned against weak opponents relative to initialization. The
result rejects this specific mixture/budget as a demonstrated advance, not RL
as an approach. Keep the self-play recipe as the control for the next separate
test of decision-point curriculum. Do not increase the budget or change mixture
weights in this completed experiment.

- Duration: 1947.609294 seconds (32.46 minutes), one uninterrupted attempt; no resume.
- Games: 19,232 training and 13,824 evaluation.
- Outcomes: 32,968 natural terminals, 88 administrative cutoffs, 0 true ties.
- Training outcomes: 19,190 terminals / 42 cutoffs; evaluation: 13,778 terminals / 46 cutoffs.
- Decisions: 2,375,863; unavailable-choice refreshes: 1,014.
- Learner rows: 1,138,766 collected, 786,432 selected, 352,334 excluded from direct loss.
- Adam steps: 6,144; row presentations: 1,572,864.
- Parent peak RSS: 746,782,720 bytes (712.2 MiB), excluding eight Node workers.

Wins below are out of 512 games per cell; cutoffs remain unknown outcomes in all improvement estimates. Difference intervals are percentage points for mixture minus self-play. Bootstrap intervals are approximate and adjusted across all 27 contrasts.

| Seed | Opponent | Initial wins | Self-play wins | Mixture wins | Cutoff-aware difference | Adjusted bootstrap interval |
|---|---|---:|---:|---:|---|---|
| 4103 | random | 251 | 447 | 436 | [-2.73, -1.95] | [-9.18, +4.49] |
| 4103 | base_power | 39 | 189 | 188 | [-0.39, -0.20] | [-8.40, +7.62] |
| 4103 | tactical | 4 | 28 | 15 | [-2.54, -2.54] | [-6.25, +1.17] |
| 5207 | random | 239 | 463 | 439 | [-5.08, -4.49] | [-10.55, +1.56] |
| 5207 | base_power | 48 | 217 | 171 | [-8.98, -8.98] | [-17.09, -1.08] |
| 5207 | tactical | 2 | 23 | 19 | [-0.78, -0.78] | [-4.49, +2.63] |
| 6301 | random | 244 | 454 | 445 | [-1.95, -1.37] | [-8.20, +4.88] |
| 6301 | base_power | 46 | 193 | 196 | [+0.39, +0.59] | [-8.20, +8.79] |
| 6301 | tactical | 0 | 23 | 20 | [-0.59, -0.59] | [-4.30, +3.12] |

### Per-seed gate and retained failures

- Seed 4103: did not pass: tactical_vs_control_confidence_positive, tactical_vs_control_practical_gain, tactical_vs_initial_confidence_positive.
- Seed 5207: did not pass: no_detected_opponent_regression, tactical_vs_control_confidence_positive, tactical_vs_control_practical_gain.
- Seed 6301: did not pass: tactical_vs_control_confidence_positive, tactical_vs_control_practical_gain.

### Actual collection exposure and value behaviour

| Seed | Arm | Training games | Train terminals / cutoffs | Collected / selected learner rows | Complete-cycle seconds | Update-call seconds |
|---|---|---:|---|---|---:|---:|
| 4103 | mixture | 4400 | 4394 / 6 | 195629 / 131072 | 194.920 | 1.591 |
| 4103 | self_play | 2032 | 2027 / 5 | 181018 / 131072 | 92.989 | 2.098 |
| 5207 | mixture | 4320 | 4311 / 9 | 199351 / 131072 | 138.497 | 1.150 |
| 5207 | self_play | 2008 | 2001 / 7 | 183815 / 131072 | 85.321 | 1.173 |
| 6301 | mixture | 4464 | 4458 / 6 | 196625 / 131072 | 139.539 | 1.179 |
| 6301 | self_play | 2008 | 1999 / 9 | 182328 / 131072 | 86.740 | 1.216 |

| Seed | Arm | Opponent | Eval cutoffs | Terminal game-mean value MSE | Game-mean predicted value |
|---|---|---|---:|---:|---:|
| 4103 | mixture | base_power | 0 | 0.183568 | 0.301277 |
| 4103 | mixture | random | 1 | 0.186451 | 0.638098 |
| 4103 | mixture | tactical | 0 | 0.066294 | 0.201722 |
| 4103 | self_play | base_power | 1 | 0.193234 | 0.451132 |
| 4103 | self_play | random | 3 | 0.161478 | 0.656830 |
| 4103 | self_play | tactical | 0 | 0.135130 | 0.330825 |
| 5207 | mixture | base_power | 0 | 0.186611 | 0.260503 |
| 5207 | mixture | random | 1 | 0.203012 | 0.589371 |
| 5207 | mixture | tactical | 0 | 0.060478 | 0.177500 |
| 5207 | self_play | base_power | 0 | 0.194361 | 0.465819 |
| 5207 | self_play | random | 2 | 0.139012 | 0.688653 |
| 5207 | self_play | tactical | 0 | 0.142110 | 0.335735 |
| 6301 | mixture | base_power | 0 | 0.186704 | 0.400666 |
| 6301 | mixture | random | 2 | 0.140652 | 0.723031 |
| 6301 | mixture | tactical | 0 | 0.112467 | 0.288574 |
| 6301 | self_play | base_power | 1 | 0.188997 | 0.440756 |
| 6301 | self_play | random | 1 | 0.156321 | 0.655398 |
| 6301 | self_play | tactical | 0 | 0.132807 | 0.325784 |

Initial value output is 0.5, with terminal-only MSE 0.25. The table is descriptive; terminal conditioning and changing game states prevent a calibration or matched-state claim.

Mixture training collected 13,184 games versus 6,048 for self-play, while each
arm used 393,216 selected rows across the three seeds. Recorded training cycles
totalled 472.956 versus 265.051 seconds; these are observed costs with a fixed arm
execution order, not a separate randomized throughput comparison. All 768 cycle
records are present. Update calls took 8.405 seconds combined (1.14% of recorded
training-cycle time), including the first compilation. Compilation was not timed
separately. CPU remains appropriate for this small policy.

The mixture's terminal-only tactical value MSE was lower in all seeds, while its
observed tactical wins were also lower. Better value loss alone did not establish
stronger action selection; changing outcome frequencies and visited states limit
the interpretation of that loss comparison.

All six original final-update replays reproduce the exact model, Adam state, selected batch and metrics. All 27 first-pair evaluation replays reproduce player-input/action/value/outcome records. Every recorded update, counter, checkpoint, row-selection ledger and evaluation summary is bound to the admitted source tree; replay is bounded, not exhaustive.

The 86-test regression is reused on the unchanged 404-file tree. Preflight, failed first recovery test, repair, review dispositions and production evidence are preserved separately. The production run has no failed or replacement attempt. No two-hour stability qualification or GPU comparison is claimed by this experiment.

Known heuristic identities occur in mixture training and evaluation. All reported evaluation games use the development partition; the final reserved partition is unused. Repeated species, team structures and the shared development panel limit generalization claims. A failed mixture gate can reflect regression or uncertainty; inspect its individual components.

Handoff: `docs/F5_OPPONENT_MIXTURE.md`. Completion inventory: [F5 receipt](../artifacts/reports/f5-opponent-mixture.json). Full G2/G3/G4, public ladder strength and curriculum benefit remain open. GitHub publication remains held.
