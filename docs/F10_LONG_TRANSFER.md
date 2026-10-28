# F10 — longer full-battle PPO with supervised rehearsal

Status: completed and exhaustively replayed (2026-10-24); see measured results below.

F9 preserved the known tactical decisions with rehearsal but did not establish full-battle improvement. F10 tests whether the same fixed setting improves full-battle results at eight times the pilot's training budget. It is a new development experiment after seeing F9, not an independent confirmation of a previously unselected hyperparameter.

## Fixed comparison

Three seeds (7103, 8209, 9311); plain PPO and PPO plus supervised rehearsal at coefficient 0.1. Both arms start from the same seed's sealed F7 step-1024 checkpoint with fresh Adam and initial critic 0.5. The network, full six-Pokémon observation/action interface, reward, complete-trajectory GAE and joint optimizer are unchanged from F9. Rehearsal uses only the sealed 128-row F7 fit_train batch.

Each arm receives 128 collection/update cohorts, 1,024 selected PPO rows per cohort, two epochs and four minibatches per epoch: 1,024 Adam steps per arm, 6,144 total; 786,432 selected rows and 1,572,864 PPO row presentations. Rehearsal adds 393,216 supervised row presentations. Whole eight-game batches are collected to reach 1,024 rows (hard cap 64 games/cohort). Extra rows are reported and excluded only after complete-trajectory GAE. Training games and total compute can differ by arm. No intermediate endpoint selection, early stopping, restart, seed selection or budget extension.

Fresh train and development mechanics namespaces are declared in [the immutable plan](../contracts/long-transfer-v1/plan.json). Each endpoint plays 256 paired team/mechanics clusters in both seats against random, base_power and tactical opponents: 512 games per opponent, 13,824 evaluation games total. Initial evaluation is shared between the arms within a seed. All 27 contrasts remain in the result. No final reserved data or public server is used.

## Registered interpretation

Primary full-battle gate, per arm and seed: base_power cutoff-aware point gain at least five percentage points and approximate familywise bootstrap lower bound above zero; no detected regression against any initial opponent; final cutoff rate at most 5%. The paired cluster bootstrap uses 20,000 resamples, family 27 and alpha 0.05. Its coverage is approximate; distribution-free Hoeffding bounds are reported too. Cutoffs remain unresolved intervals, never silently counted as draws or discarded.

Secondary tactical checks reuse F9's thresholds: greedy accuracy at least 95%, each context's correct-action mass at least 0.6, drop at most 0.05, and mean rehearsal advantage over plain at least 0.02. Fresh scenario coordinates still use known families and can duplicate encoded states; actual input overlap is measured. The 512 states are scored against every legal oracle branch (3,584 checks), with 4,608 endpoint predictions; oracle labels never enter the new PPO updates.

Value diagnostics compare each endpoint's critic with a constant 0.5 prediction on the **same natural-terminal decisions**. Equal-game and decision-weighted MSE, tie-aware baseline, improvement over baseline, prediction spread and ten fixed reliability bins are descriptive. Cutoffs have no outcome target and are counted separately. Undefined all-cutoff summaries are null. Decisions within a game are correlated, and terminal populations change with the policy: these numbers do not establish out-of-distribution calibration or a causal critic-learning effect.

## Verification and execution

Production used `python -m pokemon_long_transfer.campaign --run artifacts/runs/f10-long-transfer-v1` via the single persistent launcher in `artifacts/reviews/f10-long-transfer/run.py`. It is followed by a fresh-process exact replay of every full training/evaluation game, full-trajectory GAE batch, row selection, optimizer metric/state, tactical branch and endpoint value/policy decision. All 48 periodic checkpoints and six before-last-to-original-next-update recoveries are checked. The advertised-but-unavailable choice journal is retained and exactly replayed using F9's corrected collector.

Preflight uses separate mechanics coordinates and verifies one complete cohort, both joint objectives, checkpoint recovery and the final evaluation chunk. Focused tests and a Showdown build are bound to the exact production source/runtime. Independent cross-model review and verified dispositions precede launch. Production, replay and workflow must all exit zero; a completed experiment can still fail every learning gate. After measured documentation, the receipt is sealed, every live and archived hash checked and sealing repeated to identical bytes.

Full G2/G3/G4 and public ladder strength remain open. Local CPU only; Prime is unnecessary. GitHub preservation remains on the existing unresolved visibility hold.

## Measured result

F10 completed (2026-10-24): the fixed longer full-battle comparison collected 12,256 training and 13,824 evaluation games in 1051.30 seconds on local CPU. All-seed full-battle learning gates under the registered approximate-bootstrap development test: plain passed, rehearsal passed; secondary all-seed tactical-retention benefit passed. Both arms received 1,024 new Adam steps per seed (6,144 total), with 786,432 selected PPO rows and 393,216 extra supervised row presentations in the rehearsal arm. All 28 focused tests pass; fresh replay reproduces every full game, batch, update, tactical prediction, 48 periodic checkpoints and six original-next-update recoveries. This is development evaluation against known opponents and scenario families. Full G2/G3/G4 remain open; public servers, final reserved data and Prime were unused. GitHub remains on the existing visibility hold.

Training: 12,233 natural terminals, 23 cutoffs, 1,075,511 decisions. Evaluation: 13,812 natural terminals, 12 cutoffs, 942,831 decisions. Natural ties: 0 training and 0 evaluation. Total full-battle decisions: 2,018,342. Of the training decisions, 289,079 were excluded after complete-trajectory GAE. Parent peak RSS: 1288.92 MiB, excluding the eight Node workers. Timings include initialization and compilation; replay took 974.66 seconds.

| Seed | Plain retained | Rehearsal retained | Rehearsal mass advantage | Rehearsal benefit | Full-battle learning plain / rehearsal |
|---|---|---|---|---|---|
| 7103 | failed | passed | 21.39 pp | passed | passed / passed |
| 8209 | failed | passed | 19.46 pp | passed | passed / passed |
| 9311 | failed | passed | 21.39 pp | passed | passed / passed |

Wins below are natural wins out of 512 games per cell; cutoffs are shown separately and never counted as wins or draws.

| Seed | Opponent | Initial wins / cutoffs | Plain wins / cutoffs | Rehearsal wins / cutoffs |
|---|---|---|---|---|
| 7103 | random | 359 / 5 | 451 / 0 | 472 / 1 |
| 7103 | base_power | 84 / 0 | 195 / 0 | 188 / 0 |
| 7103 | tactical | 9 / 0 | 21 / 0 | 25 / 0 |
| 8209 | random | 339 / 3 | 466 / 0 | 460 / 0 |
| 8209 | base_power | 84 / 0 | 220 / 0 | 211 / 0 |
| 8209 | tactical | 5 / 0 | 17 / 0 | 20 / 0 |
| 9311 | random | 328 / 2 | 458 / 0 | 454 / 0 |
| 9311 | base_power | 72 / 1 | 201 / 0 | 182 / 0 |
| 9311 | tactical | 6 / 0 | 30 / 0 | 20 / 0 |

All 27 contrasts (probability-point intervals, with unresolved cutoffs retained):

| Seed | Contrast | Opponent | Point bounds | Approx. familywise bootstrap | Hoeffding | Result |
|---|---|---|---|---|---|---|
| 7103 | plain_minus_initial | base_power | [21.68, 21.68] | [13.28, 29.69] | [-1.68, 45.04] | improvement |
| 7103 | plain_minus_initial | random | [16.99, 17.97] | [8.79, 25.78] | [-6.37, 41.33] | improvement |
| 7103 | plain_minus_initial | tactical | [2.34, 2.34] | [-0.98, 5.66] | [-21.02, 25.70] | inconclusive |
| 7103 | rehearsal_minus_initial | base_power | [20.31, 20.31] | [12.50, 28.22] | [-3.05, 43.67] | improvement |
| 7103 | rehearsal_minus_initial | random | [21.09, 22.27] | [13.28, 30.27] | [-2.27, 45.63] | improvement |
| 7103 | rehearsal_minus_initial | tactical | [3.12, 3.12] | [0.00, 6.64] | [-20.23, 26.48] | inconclusive |
| 7103 | rehearsal_minus_plain | base_power | [-1.37, -1.37] | [-9.96, 7.23] | [-24.73, 21.99] | inconclusive |
| 7103 | rehearsal_minus_plain | random | [4.10, 4.30] | [-2.15, 10.35] | [-19.26, 27.66] | inconclusive |
| 7103 | rehearsal_minus_plain | tactical | [0.78, 0.78] | [-2.73, 4.49] | [-22.58, 24.14] | inconclusive |
| 8209 | plain_minus_initial | base_power | [26.56, 26.56] | [18.46, 35.35] | [3.20, 49.92] | improvement |
| 8209 | plain_minus_initial | random | [24.22, 24.80] | [16.41, 32.23] | [0.86, 48.16] | improvement |
| 8209 | plain_minus_initial | tactical | [2.34, 2.34] | [-0.39, 5.47] | [-21.02, 25.70] | inconclusive |
| 8209 | rehearsal_minus_initial | base_power | [24.80, 24.80] | [16.99, 33.01] | [1.44, 48.16] | improvement |
| 8209 | rehearsal_minus_initial | random | [23.05, 23.63] | [15.23, 31.54] | [-0.31, 46.99] | improvement |
| 8209 | rehearsal_minus_initial | tactical | [2.93, 2.93] | [-0.20, 6.25] | [-20.43, 26.29] | inconclusive |
| 8209 | rehearsal_minus_plain | base_power | [-1.76, -1.76] | [-10.74, 6.93] | [-25.12, 21.60] | inconclusive |
| 8209 | rehearsal_minus_plain | random | [-1.17, -1.17] | [-6.25, 3.91] | [-24.53, 22.19] | inconclusive |
| 8209 | rehearsal_minus_plain | tactical | [0.59, 0.59] | [-3.02, 4.10] | [-22.77, 23.95] | inconclusive |
| 9311 | plain_minus_initial | base_power | [25.00, 25.20] | [16.80, 33.30] | [1.64, 48.56] | improvement |
| 9311 | plain_minus_initial | random | [25.00, 25.39] | [17.87, 32.62] | [1.64, 48.75] | improvement |
| 9311 | plain_minus_initial | tactical | [4.69, 4.69] | [1.17, 8.59] | [-18.67, 28.05] | improvement |
| 9311 | rehearsal_minus_initial | base_power | [21.29, 21.48] | [13.28, 29.30] | [-2.07, 44.84] | improvement |
| 9311 | rehearsal_minus_initial | random | [24.22, 24.61] | [17.09, 31.84] | [0.86, 47.97] | improvement |
| 9311 | rehearsal_minus_initial | tactical | [2.73, 2.73] | [-0.20, 5.86] | [-20.63, 26.09] | inconclusive |
| 9311 | rehearsal_minus_plain | base_power | [-3.71, -3.71] | [-11.91, 4.59] | [-27.07, 19.65] | inconclusive |
| 9311 | rehearsal_minus_plain | random | [-0.78, -0.78] | [-6.25, 4.69] | [-24.14, 22.58] | inconclusive |
| 9311 | rehearsal_minus_plain | tactical | [-1.95, -1.95] | [-6.45, 2.15] | [-25.31, 21.41] | inconclusive |

Terminal-only game-mean value MSE (initial / plain / rehearsal). These populations can change with policy, so this is not a held-out calibration test:

| Seed | Random | Base power | Tactical |
|---|---|---|---|
| 7103 | 0.2500 / 0.1630 / 0.1723 | 0.2500 / 0.2051 / 0.1986 | 0.2500 / 0.1600 / 0.1474 |
| 8209 | 0.2500 / 0.1731 / 0.1569 | 0.2500 / 0.2093 / 0.2055 | 0.2500 / 0.1707 / 0.1758 |
| 9311 | 0.2500 / 0.1656 / 0.1844 | 0.2500 / 0.1985 / 0.2072 | 0.2500 / 0.1587 / 0.1632 |

Unavailable-choice feedback retained: 531 training and 556 evaluation. Every feedback journal event reproduced in the exhaustive audit; these attempted decisions were not filtered out.

Failed retention contexts and check(s), retaining all cells in the report:

- 7103 plain: train/0 (drop); train/1 (drop); train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/2 (drop); train/3 (drop); train/4 (drop); train/5 (drop); train/6 (drop); train/7 (drop); variant/0 (drop); variant/1 (drop); variant/12 (drop); variant/13 (drop); variant/14 (drop); variant/15 (drop); variant/2 (drop); variant/3 (drop); variant/4 (drop, mass); variant/5 (drop, mass); variant/6 (drop, mass); variant/7 (drop, mass).
- 7103 rehearsal: none.
- 8209 plain: train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/2 (drop); train/3 (drop); train/4 (drop, mass); train/5 (drop, mass); variant/10 (drop); variant/11 (drop); variant/12 (drop); variant/13 (drop); variant/14 (drop); variant/15 (drop); variant/2 (drop, greedy, mass); variant/3 (drop, greedy, mass); variant/4 (drop, greedy, mass); variant/5 (drop, greedy, mass); variant/8 (drop); variant/9 (drop).
- 8209 rehearsal: none.
- 9311 plain: train/0 (drop); train/1 (drop); train/12 (drop); train/13 (drop); train/14 (drop); train/15 (drop); train/2 (drop); train/3 (drop); train/4 (drop); train/5 (drop); train/6 (drop); train/7 (drop); variant/0 (drop, mass); variant/1 (drop); variant/12 (drop, mass); variant/13 (drop, mass); variant/14 (drop, mass); variant/15 (drop, mass); variant/2 (drop, greedy); variant/3 (drop, greedy); variant/4 (drop); variant/5 (drop); variant/6 (drop); variant/7 (drop).
- 9311 rehearsal: none.

Tactical input inventory: `{"actual_supervised_batch_verified": true, "fresh_coordinates_not_guaranteed_unseen_inputs": true, "fresh_evaluation_unique_inputs": {"train": 118, "variant": 122}, "overlap_with_supervised_training": {"train": 56, "variant": 0}, "same_templates_and_known_variant_species": true, "supervised_training_unique_inputs": 82}`. These are known scenario families; fresh simulator coordinates do not imply unseen encoded states.

The additive value diagnostics were delegated, independently reviewed and verified with analytical unittest cases. The F10 campaign, audit, preregistration and sealer received cross-model review before source freeze. All sealed predecessor evidence was admitted before and after execution and replay. No restart, budget extension, endpoint selection or seed selection occurred.

Same-terminal-population value diagnostics (positive MSE advantage beats constant 0.5; decisions within games are correlated):

| Seed | Endpoint | Opponent | Terminals / cutoffs | Game MSE / baseline / advantage | Decision MSE / baseline / advantage | Mean / SD prediction |
|---|---|---|---|---|---|---|
| 7103 | initial | base_power | 512 / 0 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 7103 | initial | random | 507 / 5 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 7103 | initial | tactical | 512 / 0 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 7103 | plain | base_power | 512 / 0 | 0.205145 / 0.250000 / 0.044855 | 0.208913 / 0.250000 / 0.041087 | 0.469314 / 0.135232 |
| 7103 | plain | random | 512 / 0 | 0.163003 / 0.250000 / 0.086997 | 0.168724 / 0.250000 / 0.081276 | 0.639841 / 0.121757 |
| 7103 | plain | tactical | 512 / 0 | 0.160031 / 0.250000 / 0.089969 | 0.164113 / 0.250000 / 0.085887 | 0.389030 / 0.127583 |
| 7103 | rehearsal | base_power | 512 / 0 | 0.198607 / 0.250000 / 0.051393 | 0.204627 / 0.250000 / 0.045373 | 0.443848 / 0.140876 |
| 7103 | rehearsal | random | 511 / 1 | 0.172317 / 0.250000 / 0.077683 | 0.177366 / 0.250000 / 0.072634 | 0.615583 / 0.132689 |
| 7103 | rehearsal | tactical | 512 / 0 | 0.147406 / 0.250000 / 0.102594 | 0.154984 / 0.250000 / 0.095016 | 0.360169 / 0.130660 |
| 8209 | initial | base_power | 512 / 0 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 8209 | initial | random | 509 / 3 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 8209 | initial | tactical | 512 / 0 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 8209 | plain | base_power | 512 / 0 | 0.209263 / 0.250000 / 0.040737 | 0.214397 / 0.250000 / 0.035603 | 0.473384 / 0.120846 |
| 8209 | plain | random | 512 / 0 | 0.173071 / 0.250000 / 0.076929 | 0.177302 / 0.250000 / 0.072698 | 0.610330 / 0.111527 |
| 8209 | plain | tactical | 512 / 0 | 0.170662 / 0.250000 / 0.079338 | 0.177336 / 0.250000 / 0.072664 | 0.398336 / 0.114079 |
| 8209 | rehearsal | base_power | 512 / 0 | 0.205513 / 0.250000 / 0.044487 | 0.212438 / 0.250000 / 0.037562 | 0.486270 / 0.127296 |
| 8209 | rehearsal | random | 512 / 0 | 0.156913 / 0.250000 / 0.093087 | 0.163474 / 0.250000 / 0.086526 | 0.647200 / 0.110913 |
| 8209 | rehearsal | tactical | 512 / 0 | 0.175814 / 0.250000 / 0.074186 | 0.180485 / 0.250000 / 0.069515 | 0.409006 / 0.122633 |
| 9311 | initial | base_power | 511 / 1 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 9311 | initial | random | 510 / 2 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 9311 | initial | tactical | 512 / 0 | 0.250000 / 0.250000 / 0.000000 | 0.250000 / 0.250000 / 0.000000 | 0.500000 / 0.000000 |
| 9311 | plain | base_power | 512 / 0 | 0.198516 / 0.250000 / 0.051484 | 0.203421 / 0.250000 / 0.046579 | 0.474315 / 0.145015 |
| 9311 | plain | random | 512 / 0 | 0.165572 / 0.250000 / 0.084428 | 0.169234 / 0.250000 / 0.080766 | 0.628447 / 0.135844 |
| 9311 | plain | tactical | 512 / 0 | 0.158730 / 0.250000 / 0.091270 | 0.164077 / 0.250000 / 0.085923 | 0.388512 / 0.132352 |
| 9311 | rehearsal | base_power | 512 / 0 | 0.207160 / 0.250000 / 0.042840 | 0.211734 / 0.250000 / 0.038266 | 0.453885 / 0.109778 |
| 9311 | rehearsal | random | 512 / 0 | 0.184367 / 0.250000 / 0.065633 | 0.189316 / 0.250000 / 0.060684 | 0.588520 / 0.111198 |
| 9311 | rehearsal | tactical | 512 / 0 | 0.163193 / 0.250000 / 0.086807 | 0.166477 / 0.250000 / 0.083523 | 0.395424 / 0.102021 |

All ten reliability bins per endpoint/opponent are retained in the report and receipt. Undefined metrics remain null. These are descriptive terminal-conditioned results, not independent-decision confidence estimates or generalization evidence.

[Completion receipt](../artifacts/reports/f10-long-transfer.json) · [Production report](../artifacts/runs/f10-long-transfer-v1/report.json) · [Exhaustive replay](../artifacts/reviews/f10-long-transfer/production-audit/audit.json).

## Interpretation and remaining bottleneck

Both recipes passed the registered approximate-bootstrap full-battle learning gate in all three seeds. That is development evidence against the declared opponent panel; conservative Hoeffding intervals still include zero in some cells. Rehearsal retained the known tactical decisions in 3/3 seeds, versus 0/3 for plain PPO. All nine rehearsal-minus-plain full-battle contrasts remain inconclusive. Neither superiority nor equivalence is established; base-power point differences favor plain PPO in every seed.

The following pooled rates are descriptive counts across the three seeds, not additional statistical tests. Denominators include unresolved cutoffs; those are not counted as draws or wins. All individual seed results and cutoff-aware intervals remain above.

| Opponent | Initial wins / games | Plain wins / games | Rehearsal wins / games |
|---|---|---|---|
| random | 1026/1536 (66.80%) | 1375/1536 (89.52%) | 1386/1536 (90.23%) |
| base_power | 240/1536 (15.62%) | 616/1536 (40.10%) | 581/1536 (37.83%) |
| tactical | 20/1536 (1.30%) | 68/1536 (4.43%) | 65/1536 (4.23%) |

The critic reduces terminal-only MSE relative to 0.5 in every final cell, but this is a weak reference. A **post-hoc descriptive check**, separate from every registered gate, fits a constant to each cell's same terminal outcomes. With equal-game and decision weighting computed separately, that fitted constant beats the critic on all six random and all six tactical final cells; the critic beats it on all six base-power cells. The fitted constant uses evaluation outcomes and is not an independently trained or held-out predictor. This check, prediction compression and the policy-dependent terminal populations prevent a claim of calibrated values against these frozen opponents. The critic was trained under self-play, so changed opponent policies also limit what these errors imply about its training-distribution values. Details and review dispositions are preserved in `artifacts/reviews/f10-long-transfer/interpretation-review/`.

The 531 training and 556 evaluation unavailable-choice events are advertised actions rejected by Showdown and followed by refreshed requests. The corrected collector refreshes requests and retains attempted on-policy learner decisions; it does not silently filter them. Every event is journaled and included in the exact replay. The obsolete zero-feedback guard belongs to the preserved failed F9 attempt and is not the F10 collection path.

The next useful question is which full-battle decisions cause the large gap to the frozen tactical opponent, and whether broader decision supervision transfers beyond the known scenario families. A bounded player-visible decision-quality audit can guide that experiment while keeping these F10 endpoints and final reserved data fixed. Simply repeating F5's failed opponent mixture or increasing this completed run's budget is not justified by these results. No new experiment is executed by this handoff.

Recorded training cycles totalled 417.643 seconds, of which collection took 398.656 seconds and optimizer calls including compilation took 9.571 seconds (2.29% of cycle time). Compilation was not timed separately. Parent peak RSS was 1,288.92 MiB and excludes the eight Node workers. This fixed-budget learning comparison is not a new two-hour stability qualification or GPU benchmark.
