# F9 v2 — full-battle transfer with supervised rehearsal

Status: completed and exhaustively replayed (2026-10-24); see measured results below.

This small local CPU pilot tests whether a fixed supervised rehearsal term preserves F7's fitted switching and Tera decisions during full-Showdown PPO. It compares plain PPO and PPO plus 0.1 times mean supervised negative log likelihood. Both start from the same seed's sealed F7 step-1024 parameters with fresh Adam moments/counters and the unchanged critic initially predicting 0.5. F8's bandit-trained critic is not imported.

The fixed seeds are 7103, 8209 and 9311. Each arm gets 16 collection/update cohorts, 1,024 selected PPO rows per cohort, two epochs and eight minibatch updates: 128 Adam steps per seed-arm, 768 overall. Complete official training episodes are collected eight at a time until enough rows exist, with a hard 64-game cap per cohort. GAE includes each complete same-player trajectory and its terminal or administrative-cutoff bootstrap before selecting rows without replacement. No incomplete episodes are discarded based on outcomes. All unselected rows and game counts are reported. Both arms share namespaces and shuffle roots, but on-policy episode lengths and subsequent episode cursors may diverge.

Rehearsal uses only F7's sealed 128 fit_train states and labels, on every optimizer step. The PPO and supervised gradients are combined before one global-norm clip and one Adam step. There are 98,304 selected PPO rows, 196,608 PPO row presentations and 49,152 added supervised presentations. Equal PPO budgets do not mean equal total data, compute, or games. The actor receives only the existing 928-feature player view and legal-action mask; context labels and oracle scores do not enter it. The critic learns full-battle targets only; the shared trunk can still be affected by supervised gradients.

Initial policies (once per seed) and both final policies receive the same 32 team/mechanics clusters in both seats against each of random, base_power and tactical: 1,728 evaluation games total. All teams use the development partition; final evaluation data remain reserved. At turn 120, unresolved games are explicit cutoffs with outcome bounds [0,1], never draws. The 27 contrasts are each final arm versus its initial policy and rehearsal versus plain, against all opponents and across all seeds. We report cutoff-aware point bounds, approximate familywise paired-cluster bootstrap intervals (5,000 resamples), conservative Hoeffding intervals and terminal-only value MSE. Small cells can be inconclusive; absence of a detected regression is not proof of equivalence.

Separately, 512 fresh tactical coordinates from the same known scenario families are checked through all seven legal Showdown branches (3,584 branches). Initial and final policy probabilities are evaluated without sampling, for 4,608 prediction rows. For every train/variant context, retention requires greedy accuracy at least 95%, target probability at least 60%, and no more than a five-point probability drop from initialization. Rehearsal benefit additionally requires mean target-probability advantage of at least two points over plain PPO, in every seed. Local HP-retention rewards are used to verify tactical labels, not to evaluate full-battle critic calibration. Fresh coordinates can repeat encoded inputs; overlap with F7's supervised states is reported.

The separate full-battle learning gate per arm/seed requires at least a five-point cutoff-aware gain over initialization against base_power, positive approximate familywise bootstrap lower bound, no detected regression against the other frozen opponents, and final cutoff rates at most 5%. All seeds, arms and opponent contrasts remain in the report regardless of outcome. There is no early stopping, checkpoint selection, budget extension or automatic repair/rerun.

The plan is `contracts/full-transfer-v2/plan.json`. Production used `artifacts/runs/f9-full-transfer-v2`. Before acceptance, every full game, GAE/selection ledger, update, endpoint prediction and six original-next-update checkpoints must replay in a fresh process. The source-bound checks, independent model reviews, production report and replay will be bound in a repeatable completion receipt. Historical source and evidence remain sealed.

This is a development transfer pilot on a small model and known tactical templates. It does not establish public ladder strength or generalization to unseen opponents. Full G2/G3/G4 remain open. No public server, final reserved evaluation or Prime is used. GitHub publication remains on the existing visibility hold.

## Disclosed correction

F9 v1 failed before any training because a newly added guard treated valid unavailable-choice feedback as an error. Episode7 of the initial random-opponent panel reproduced trapped-switch responses in both seats. The sealed Showdown session already supports the refreshed request and keeps the attempted action in the trajectory. V2 removes only the zero-feedback assumption, validates the counter bounds, logs every rejected advertised action with before/after player views, and writes each evaluation chunk immediately. No outcome-based game filtering is allowed. The policy, learner, seeds, budgets, opponents and thresholds stay the same. The original attempt and its failure receipt remain immutable. This is an explicit amendment to the original no-automatic-repair workflow; the already-observed initial prefix means it is not a fresh independent confirmation.

[Preserved v1 failure](../artifacts/reports/f9-full-transfer-failure.json).

## Measured result

Tactical retention passed in all three rehearsal seeds and failed in all three plain-PPO seeds. Rehearsal increased final correct-action probability over plain PPO by 8.81, 12.10 and 9.31 percentage points. Neither arm passed the full-battle learning gate in any seed; all rehearsal-versus-plain win-rate contrasts were inconclusive. Terminal-only value MSE remained around 0.25, with mean values near 0.5.

F9 v2 completed (2026-10-24): the fixed full-battle transfer pilot collected 1,448 training and 1,728 evaluation games in 161.80 seconds on local CPU. The all-seed supervised-rehearsal benefit gate passed. Both arms received 128 new Adam steps per seed (768 total), with 98,304 selected PPO rows and 49,152 extra supervised row presentations in the rehearsal arm. All 20 focused tests pass; fresh replay reproduces every full game, selected batch, update, tactical prediction and six original-next-update checkpoints. This is a small development pilot with known tactical templates, not a public-strength result. Full G2/G3/G4 remain open; public servers, final data and Prime were unused.

Training: 1,432 natural terminals, 16 cutoffs, 154,272 decisions. Evaluation: 1,721 natural terminals, 7 cutoffs, 127,453 decisions. Natural ties: 0 training and 0 evaluation. Total full-battle decisions: 281,725. Of the training decisions, 55,968 were excluded after complete-trajectory GAE. Parent peak RSS: 734.42 MiB, excluding the eight Node workers. Timings include initialization and compilation; replay took 174.44 seconds.

| Seed | Plain retained | Rehearsal retained | Rehearsal mass advantage | Rehearsal benefit | Full-battle learning plain / rehearsal |
|---|---|---|---|---|---|
| 7103 | failed | passed | 8.81 pp | passed | failed / failed |
| 8209 | failed | passed | 12.10 pp | passed | failed / failed |
| 9311 | failed | passed | 9.31 pp | passed | failed / failed |

Wins below are natural wins out of 64 games per cell; cutoffs are shown separately and never counted as wins or draws.

| Seed | Opponent | Initial wins / cutoffs | Plain wins / cutoffs | Rehearsal wins / cutoffs |
|---|---|---|---|---|
| 7103 | random | 41 / 0 | 50 / 0 | 50 / 0 |
| 7103 | base_power | 11 / 0 | 14 / 0 | 16 / 0 |
| 7103 | tactical | 4 / 0 | 1 / 0 | 4 / 0 |
| 8209 | random | 43 / 2 | 48 / 2 | 43 / 2 |
| 8209 | base_power | 13 / 0 | 18 / 0 | 15 / 0 |
| 8209 | tactical | 2 / 0 | 3 / 0 | 4 / 0 |
| 9311 | random | 38 / 0 | 48 / 0 | 51 / 0 |
| 9311 | base_power | 14 / 0 | 18 / 1 | 20 / 0 |
| 9311 | tactical | 2 / 0 | 0 / 0 | 2 / 0 |

All 27 contrasts (probability-point intervals, with unresolved cutoffs retained):

| Seed | Contrast | Opponent | Point bounds | Approx. familywise bootstrap | Hoeffding | Result |
|---|---|---|---|---|---|---|
| 7103 | plain_minus_initial | base_power | [4.69, 4.69] | [-14.06, 22.46] | [-61.38, 70.76] | inconclusive |
| 7103 | plain_minus_initial | random | [14.06, 14.06] | [-7.81, 35.94] | [-52.01, 80.13] | inconclusive |
| 7103 | plain_minus_initial | tactical | [-4.69, -4.69] | [-20.31, 6.25] | [-70.76, 61.38] | inconclusive |
| 7103 | rehearsal_minus_initial | base_power | [7.81, 7.81] | [-11.52, 25.00] | [-58.26, 73.88] | inconclusive |
| 7103 | rehearsal_minus_initial | random | [14.06, 14.06] | [-7.81, 37.50] | [-52.01, 80.13] | inconclusive |
| 7103 | rehearsal_minus_initial | tactical | [0.00, 0.00] | [-12.50, 12.50] | [-66.07, 66.07] | inconclusive |
| 7103 | rehearsal_minus_plain | base_power | [3.12, 3.12] | [-16.21, 21.88] | [-62.95, 69.20] | inconclusive |
| 7103 | rehearsal_minus_plain | random | [0.00, 0.00] | [-20.31, 20.89] | [-66.07, 66.07] | inconclusive |
| 7103 | rehearsal_minus_plain | tactical | [4.69, 4.69] | [-6.25, 15.62] | [-61.38, 70.76] | inconclusive |
| 8209 | plain_minus_initial | base_power | [7.81, 7.81] | [-13.08, 28.12] | [-58.26, 73.88] | inconclusive |
| 8209 | plain_minus_initial | random | [4.69, 10.94] | [-18.75, 32.81] | [-61.38, 77.01] | inconclusive |
| 8209 | plain_minus_initial | tactical | [1.56, 1.56] | [-9.38, 17.19] | [-64.51, 67.63] | inconclusive |
| 8209 | rehearsal_minus_initial | base_power | [3.12, 3.12] | [-21.88, 28.12] | [-62.95, 69.20] | inconclusive |
| 8209 | rehearsal_minus_initial | random | [-3.12, 3.12] | [-28.71, 28.71] | [-69.20, 69.20] | inconclusive |
| 8209 | rehearsal_minus_initial | tactical | [3.12, 3.12] | [-6.25, 17.19] | [-62.95, 69.20] | inconclusive |
| 8209 | rehearsal_minus_plain | base_power | [-4.69, -4.69] | [-25.58, 17.19] | [-70.76, 61.38] | inconclusive |
| 8209 | rehearsal_minus_plain | random | [-10.94, -4.69] | [-35.94, 20.89] | [-77.01, 61.38] | inconclusive |
| 8209 | rehearsal_minus_plain | tactical | [1.56, 1.56] | [-15.62, 15.62] | [-64.51, 67.63] | inconclusive |
| 9311 | plain_minus_initial | base_power | [6.25, 7.81] | [-15.62, 30.27] | [-59.82, 73.88] | inconclusive |
| 9311 | plain_minus_initial | random | [15.62, 15.62] | [-9.38, 42.19] | [-50.45, 81.70] | inconclusive |
| 9311 | plain_minus_initial | tactical | [-3.12, -3.12] | [-10.94, 0.00] | [-69.20, 62.95] | inconclusive |
| 9311 | rehearsal_minus_initial | base_power | [9.38, 9.38] | [-12.50, 31.83] | [-56.70, 75.45] | inconclusive |
| 9311 | rehearsal_minus_initial | random | [20.31, 20.31] | [-1.56, 42.19] | [-45.76, 86.38] | inconclusive |
| 9311 | rehearsal_minus_initial | tactical | [0.00, 0.00] | [-7.81, 7.81] | [-66.07, 66.07] | inconclusive |
| 9311 | rehearsal_minus_plain | base_power | [1.56, 3.12] | [-20.89, 25.00] | [-64.51, 69.20] | inconclusive |
| 9311 | rehearsal_minus_plain | random | [4.69, 4.69] | [-18.75, 26.56] | [-61.38, 70.76] | inconclusive |
| 9311 | rehearsal_minus_plain | tactical | [3.12, 3.12] | [0.00, 10.94] | [-62.95, 69.20] | inconclusive |

Terminal-only game-mean value MSE (initial / plain / rehearsal). These populations can change with policy, so this is not a held-out calibration test:

| Seed | Random | Base power | Tactical |
|---|---|---|---|
| 7103 | 0.2500 / 0.2502 / 0.2531 | 0.2500 / 0.2517 / 0.2484 | 0.2500 / 0.2490 / 0.2463 |
| 8209 | 0.2500 / 0.2464 / 0.2457 | 0.2500 / 0.2501 / 0.2542 | 0.2500 / 0.2517 / 0.2585 |
| 9311 | 0.2500 / 0.2492 / 0.2462 | 0.2500 / 0.2513 / 0.2522 | 0.2500 / 0.2518 / 0.2559 |

Unavailable-choice feedback retained: 116 training and 105 evaluation. Every feedback journal event reproduced in the exhaustive audit; these attempted decisions were not filtered out.

Failed retention contexts and check(s), retaining all cells in the report:

- 7103 plain: train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/4 (drop); train/5 (drop); variant/12 (drop, mass); variant/13 (drop, mass); variant/14 (drop, mass); variant/15 (drop, mass); variant/4 (drop, mass); variant/5 (drop, mass).
- 7103 rehearsal: none.
- 8209 plain: train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/2 (drop); train/3 (drop); train/4 (drop); train/5 (drop); variant/12 (drop, greedy, mass); variant/13 (drop, greedy, mass); variant/14 (drop, greedy, mass); variant/15 (drop, greedy, mass); variant/2 (drop, mass); variant/3 (drop, mass); variant/4 (drop, greedy, mass); variant/5 (drop, greedy, mass).
- 8209 rehearsal: none.
- 9311 plain: train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/6 (drop); train/7 (drop); variant/12 (drop, mass); variant/13 (drop, mass); variant/14 (drop, mass); variant/15 (drop, mass); variant/4 (drop); variant/5 (drop); variant/6 (drop); variant/7 (drop).
- 9311 rehearsal: none.

Tactical input inventory: `{"actual_supervised_batch_verified": true, "fresh_coordinates_not_guaranteed_unseen_inputs": true, "fresh_evaluation_unique_inputs": {"train": 114, "variant": 134}, "overlap_with_supervised_training": {"train": 53, "variant": 0}, "same_templates_and_known_variant_species": true, "supervised_training_unique_inputs": 82}`. These are known scenario families; fresh simulator coordinates do not imply unseen encoded states.

The initial delegated learner delivery was rejected because it used unavailable pytest under a unittest checker. Its tests were ported locally and the unchanged learner was checked in staging before integration; the independent review and final source-bound tests are preserved. The unchanged sealed simulator and predecessor sources were admitted before and after both execution and replay. The corrected v2 attempt kept the original seeds and budgets; v1 failed before training and remains sealed separately. No v2 restart, budget extension or seed selection occurred.

[Completion receipt](../artifacts/reports/f9-full-transfer-v2.json) · [Production report](../artifacts/runs/f9-full-transfer-v2/report.json) · [Exhaustive replay](../artifacts/reviews/f9-full-transfer-v2/production-audit/audit.json).

## Next step

Use this fixed rehearsal setting in a longer preregistered full-battle learning comparison, preserving the plain-PPO control and all seeds. The next gate is improved battle outcomes and value behavior, not another local tactical-fit diagnostic. Current evidence does not justify dropping the control, claiming stronger play, using final evaluation data or provisioning Prime.
