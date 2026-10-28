# F12: full-battle curriculum comparison

F12 completed (2026-10-24): the fixed curriculum comparison collected 12,848 training and 6,912 evaluation games in 694.16 seconds on local CPU. The preregistered curriculum-benefit gate did not pass (0/3 seeds). Under the registered approximate-bootstrap test, both arms regressed against base-power in every seed; curriculum also regressed against the control on random/base-power in seed 9311. Neither new arm is promoted over the prior F10 baseline. Both arms received 384 Adam steps per seed and identical PPO/rehearsal budgets; only the starting-state mixture differed, with both training against the same frozen tactical opponent. All 17 focused tests pass. Fresh independent replay reproduces every training cohort, selected batch, optimizer update, all 6,912 evaluation games, 18 stage checkpoints and six original-next-update recoveries. Full G2/G3/G4 and public ladder/generalization claims remain open. Final reserved data, public servers and Prime were unused; GitHub remains on the existing visibility hold.

Experiment completion and verified replay do not imply a successful learning gate. All three seeds and all 27 paired contrasts remain in the report. The receipt is `artifacts/reports/f12-curriculum.json`; production is `artifacts/runs/f12-curriculum-v1`.

## Fixed experiment

The control uses fresh official train starts; the curriculum progressively replaces 102, 205, or 256 of 1,024 selected PPO rows with train-only scenario suffixes in three 16-cohort stages. Both arms start from the same seed's sealed F10 rehearsal final parameters with Adam reset, retain coefficient 0.1 F7 fit_train supervision, and face tactical-v1 throughout. This isolates starting-state distribution from opponent identity; it is not continued F10 self-play. Features, architecture, action codec, terminal reward and cutoff bootstrap are unchanged.

Each of three seeds 7103, 8209 and 9311 receives 48 cohorts per arm, two PPO epochs and eight Adam steps per cohort. Complete trajectories precede same-player GAE and uniform stratified sampling without replacement. Replayed prefix actions enter neither learner rows nor targets. Fresh actor keys continue from each original simulator/public-history prefix. All 65 admitted scenario starts are eligible for the fixed weighted-family sampler, with overlapping family memberships. Reset preflight found all starts between turns 1 and 13. No early stop, checkpoint selection, resampling or budget extension occurred.

## What this result establishes

There were no detected improvements among the 27 registered approximate-bootstrap contrasts: 11 regressions and 16 inconclusive results. All tactical contrasts were inconclusive, with fewer observed wins after training in every seed for both arms. Curriculum-versus-control differences were inconclusive in seven cells and regressed in two (random and base-power, seed 9311). Lack of significance does not establish equivalence. Neither F12 arm is promoted; retain the prior F10 baseline.

Training won 372 of 12,848 games (2.90%). On tactical evaluation, voluntary-switch tags increased from 1,673/21,276 learner decisions (7.86%) initially to 4,004/20,606 (19.43%) for official starts and 4,615/20,638 (22.36%) for curriculum. Full-HP Recover choices fell from 94 to 62 and 55, respectively. These pooled diagnostics are descriptive; they do not replace the registered per-seed gates or establish why the policies deteriorated.

The critic learned very low values against the hard tactical opponent, while strongly underpredicting outcomes against random opponents. Final random-opponent decision-weighted MSE was 0.487–0.586, worse than both the constant-0.5 predictor and same-cell fitted constants in every final cell. Low tactical MSE mostly accompanies rare wins; it is not evidence of stronger play.

Next bounded work: retain F10, audit the new switch behavior and return/value targets on train-only examples, then preregister a curriculum revision emphasizing winnable non-opening continuations. A useful next experiment must separate sparse winning signal, critic/credit-assignment effects and scenario diversity rather than simply extend this failed budget. These are hypotheses motivated by the observations, not identified causes. No follow-on experiment has been run.

## Counts and resource measurements

- Training: 12,848 games, 12,841 natural terminals, 7 cutoffs, 644,286 suffix decisions; 341,073 learner rows collected.
- Evaluation: 6,912 games, 6,897 natural terminals, 15 cutoffs, 499,318 decisions.
- Selected PPO rows 294,912, PPO row presentations 589,824, supervised row presentations 294,912, Adam steps 2,304; selected prefix rows 27,024. Excluded collected learner rows: 46,161.
- Prefix replay cost: 4,523 historical actions, excluded from PPO and suffix decision counts.
- Training collection 329.31 s; updates including compilation 4.31 s; end-to-end production 694.16 s. Compilation was not isolated as a separate timer. Parent peak RSS 1,015.1 MiB excludes Node workers.
- Unavailable-choice feedback retained: 182 training and 393 evaluation attempts. These are legitimate request refreshes, not silently dropped games.

All 65 scenario starts were actually sampled across 1,728 prefix games, with per-scenario reuse from 4 to 96 times. Of these, 1,172 (67.82%) began on turn 1 and 1,112 had an empty action prefix. The reset distribution is predominantly opening states; the on-policy continuations still run through complete battles. Collected prefix learner rows 44,600; 27,024 selected and 17,576 excluded. Fresh-start learner rows 296,473; 267,888 selected and 28,585 excluded.

The row/update/rehearsal budgets match; simulator games and replay work differ. The report retains arm-specific timing and game totals.

## Fresh full-battle results

Each endpoint/opponent/seed cell contains 256 games from 128 paired official development team/mechanics clusters, both learner seats. Cells share pairs across endpoints. Counts below are observed wins; true ties and unresolved cutoffs remain separate in the report.

| Seed | Opponent | Initial wins | Official-start wins | Curriculum wins | Curriculum − official point interval (pp) | Adjusted bootstrap interval (pp) | Contrast |
|---|---|---:|---:|---:|---|---|---|
| 7103 | base_power | 79 | 28 | 21 | [-3.12, -1.95] | [-10.55, 5.86] | inconclusive |
| 7103 | random | 228 | 181 | 191 | [3.12, 5.08] | [-8.20, 16.41] | inconclusive |
| 7103 | tactical | 10 | 5 | 4 | [-0.39, -0.39] | [-3.91, 3.52] | inconclusive |
| 8209 | base_power | 87 | 52 | 52 | [0.00, 0.00] | [-9.77, 10.16] | inconclusive |
| 8209 | random | 225 | 200 | 203 | [0.39, 1.56] | [-9.38, 11.72] | inconclusive |
| 8209 | tactical | 13 | 4 | 3 | [-0.39, -0.39] | [-3.91, 2.73] | inconclusive |
| 9311 | base_power | 88 | 56 | 22 | [-13.28, -13.28] | [-21.88, -3.91] | regression |
| 9311 | random | 224 | 208 | 147 | [-24.61, -23.44] | [-36.33, -11.33] | regression |
| 9311 | tactical | 9 | 2 | 3 | [0.39, 0.39] | [-1.56, 2.73] | inconclusive |

The primary per-seed curriculum benefit requires a tactical gain of at least 5 percentage points over both the control and initialization, positive approximate familywise bootstrap lower bounds for both, no detected random/base-power regression against either, and curriculum cutoff rates at most 5%. All three seeds are required. A failed gate does not establish equivalence or prove absence of benefit.

- Seed 7103: NOT PASSED; unmet criteria: no_detected_other_opponent_regression, tactical_vs_control_bootstrap_positive, tactical_vs_control_five_points, tactical_vs_initial_bootstrap_positive, tactical_vs_initial_five_points.
- Seed 8209: NOT PASSED; unmet criteria: no_detected_other_opponent_regression, tactical_vs_control_bootstrap_positive, tactical_vs_control_five_points, tactical_vs_initial_bootstrap_positive, tactical_vs_initial_five_points.
- Seed 9311: NOT PASSED; unmet criteria: no_detected_other_opponent_regression, tactical_vs_control_bootstrap_positive, tactical_vs_control_five_points, tactical_vs_initial_bootstrap_positive, tactical_vs_initial_five_points.

## All registered contrasts

Intervals account for cutoffs as unresolved rewards in [0, 1], not draws. Bootstrap intervals are approximate with a 27-cell family adjustment and 20,000 resamples; the conservative bounded Hoeffding intervals are also retained. These are development estimates, not finite-sample certificates.

| Seed | Comparison | Opponent | Point interval (pp) | Approximate bootstrap (pp) | Hoeffding (pp) | Result |
|---|---|---|---|---|---|---|
| 7103 | curriculum_minus_initial | base_power | [-22.66, -21.88] | [-32.81, -11.72] | [-55.69, 11.16] | regression |
| 7103 | curriculum_minus_initial | random | [-14.84, -13.28] | [-25.00, -3.12] | [-47.88, 19.75] | regression |
| 7103 | curriculum_minus_initial | tactical | [-2.34, -2.34] | [-7.81, 2.34] | [-35.38, 30.69] | inconclusive |
| 7103 | curriculum_minus_official | base_power | [-3.12, -1.95] | [-10.55, 5.86] | [-36.16, 31.08] | inconclusive |
| 7103 | curriculum_minus_official | random | [3.12, 5.08] | [-8.20, 16.41] | [-29.91, 38.11] | inconclusive |
| 7103 | curriculum_minus_official | tactical | [-0.39, -0.39] | [-3.91, 3.52] | [-33.43, 32.65] | inconclusive |
| 7103 | official_minus_initial | base_power | [-19.92, -19.53] | [-30.86, -8.80] | [-52.96, 13.50] | regression |
| 7103 | official_minus_initial | random | [-18.75, -17.58] | [-29.30, -7.23] | [-51.79, 15.46] | regression |
| 7103 | official_minus_initial | tactical | [-1.95, -1.95] | [-7.81, 2.73] | [-34.99, 31.08] | inconclusive |
| 8209 | curriculum_minus_initial | base_power | [-13.67, -13.67] | [-26.17, -1.56] | [-46.71, 19.36] | regression |
| 8209 | curriculum_minus_initial | random | [-8.59, -8.20] | [-16.80, 0.00] | [-41.63, 24.83] | inconclusive |
| 8209 | curriculum_minus_initial | tactical | [-3.91, -3.91] | [-8.98, 0.39] | [-36.94, 29.13] | inconclusive |
| 8209 | curriculum_minus_official | base_power | [0.00, 0.00] | [-9.77, 10.16] | [-33.04, 33.04] | inconclusive |
| 8209 | curriculum_minus_official | random | [0.39, 1.56] | [-9.38, 11.72] | [-32.65, 34.60] | inconclusive |
| 8209 | curriculum_minus_official | tactical | [-0.39, -0.39] | [-3.91, 2.73] | [-33.43, 32.65] | inconclusive |
| 8209 | official_minus_initial | base_power | [-13.67, -13.67] | [-25.00, -1.56] | [-46.71, 19.36] | regression |
| 8209 | official_minus_initial | random | [-9.77, -8.98] | [-19.14, 0.39] | [-42.80, 24.05] | inconclusive |
| 8209 | official_minus_initial | tactical | [-3.52, -3.52] | [-8.59, 1.17] | [-36.55, 29.52] | inconclusive |
| 9311 | curriculum_minus_initial | base_power | [-25.78, -25.78] | [-35.94, -15.83] | [-58.82, 7.25] | regression |
| 9311 | curriculum_minus_initial | random | [-30.08, -29.69] | [-41.41, -18.36] | [-63.11, 3.35] | regression |
| 9311 | curriculum_minus_initial | tactical | [-2.34, -2.34] | [-6.25, 1.36] | [-35.38, 30.69] | inconclusive |
| 9311 | curriculum_minus_official | base_power | [-13.28, -13.28] | [-21.88, -3.91] | [-46.32, 19.75] | regression |
| 9311 | curriculum_minus_official | random | [-24.61, -23.44] | [-36.33, -11.33] | [-57.65, 9.60] | regression |
| 9311 | curriculum_minus_official | tactical | [0.39, 0.39] | [-1.56, 2.73] | [-32.65, 33.43] | inconclusive |
| 9311 | official_minus_initial | base_power | [-12.50, -12.50] | [-23.05, -2.73] | [-45.54, 20.54] | regression |
| 9311 | official_minus_initial | random | [-6.25, -5.47] | [-15.03, 3.52] | [-39.29, 27.57] | inconclusive |
| 9311 | official_minus_initial | tactical | [-2.73, -2.73] | [-7.03, 1.17] | [-35.77, 30.30] | inconclusive |

## Terminal-only value behavior

Cutoffs do not enter these targets. Each learner prediction is compared with its eventual terminal reward. Decision-weighted results follow. Constant 0.5 has MSE 0.25 on non-tied terminal games. The fitted comparator is a post-hoc constant fitted to the same cell; it is not an independent baseline.

| Seed | Endpoint | Opponent | Terminal decisions | Mean prediction | Mean target | Critic MSE | Constant 0.5 MSE | Same-cell fitted constant MSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 7103 | curriculum | base_power | 8404 | 0.0440 | 0.0955 | 0.0831 | 0.2500 | 0.0864 |
| 7103 | curriculum | random | 13571 | 0.1436 | 0.7358 | 0.5365 | 0.2500 | 0.1944 |
| 7103 | curriculum | tactical | 6844 | 0.0342 | 0.0165 | 0.0163 | 0.2500 | 0.0162 |
| 7103 | initial | base_power | 8442 | 0.4276 | 0.3153 | 0.2022 | 0.2500 | 0.2159 |
| 7103 | initial | random | 10715 | 0.6145 | 0.8580 | 0.1774 | 0.2500 | 0.1219 |
| 7103 | initial | tactical | 7224 | 0.3543 | 0.0436 | 0.1489 | 0.2500 | 0.0417 |
| 7103 | official | base_power | 8773 | 0.0348 | 0.1230 | 0.1112 | 0.2500 | 0.1079 |
| 7103 | official | random | 14508 | 0.0903 | 0.6893 | 0.5694 | 0.2500 | 0.2141 |
| 7103 | official | tactical | 7034 | 0.0268 | 0.0330 | 0.0321 | 0.2500 | 0.0319 |
| 8209 | curriculum | base_power | 8631 | 0.0669 | 0.2250 | 0.1894 | 0.2500 | 0.1744 |
| 8209 | curriculum | random | 12299 | 0.2119 | 0.7707 | 0.4871 | 0.2500 | 0.1767 |
| 8209 | curriculum | tactical | 6747 | 0.0470 | 0.0122 | 0.0132 | 0.2500 | 0.0120 |
| 8209 | initial | base_power | 8430 | 0.4715 | 0.3507 | 0.2031 | 0.2500 | 0.2277 |
| 8209 | initial | random | 10561 | 0.6407 | 0.8442 | 0.1747 | 0.2500 | 0.1315 |
| 8209 | initial | tactical | 7185 | 0.4064 | 0.0575 | 0.1763 | 0.2500 | 0.0542 |
| 8209 | official | base_power | 8491 | 0.0548 | 0.2192 | 0.1852 | 0.2500 | 0.1711 |
| 8209 | official | random | 12677 | 0.1792 | 0.7497 | 0.5122 | 0.2500 | 0.1876 |
| 8209 | official | tactical | 6812 | 0.0389 | 0.0195 | 0.0190 | 0.2500 | 0.0191 |
| 9311 | curriculum | base_power | 8484 | 0.0347 | 0.0959 | 0.0895 | 0.2500 | 0.0867 |
| 9311 | curriculum | random | 15162 | 0.0592 | 0.5679 | 0.5006 | 0.2500 | 0.2454 |
| 9311 | curriculum | tactical | 7047 | 0.0323 | 0.0163 | 0.0162 | 0.2500 | 0.0161 |
| 9311 | initial | base_power | 8286 | 0.4378 | 0.3567 | 0.2119 | 0.2500 | 0.2295 |
| 9311 | initial | random | 11118 | 0.5814 | 0.8545 | 0.1944 | 0.2500 | 0.1244 |
| 9311 | initial | tactical | 6867 | 0.3861 | 0.0441 | 0.1634 | 0.2500 | 0.0422 |
| 9311 | official | base_power | 8369 | 0.0561 | 0.2300 | 0.1983 | 0.2500 | 0.1771 |
| 9311 | official | random | 12347 | 0.1425 | 0.7917 | 0.5856 | 0.2500 | 0.1649 |
| 9311 | official | tactical | 6760 | 0.0449 | 0.0121 | 0.0126 | 0.2500 | 0.0120 |

## Full-battle decision diagnostics

These public-state heuristics are descriptive proxies, not ground-truth error labels, independent tests or expected counterfactual regret. Full-HP Recover counts include composite Tera+Recover choices; Tera effects cannot be separated from the move here.

| Seed | Endpoint | Tactical learner decisions | Heuristic disagreements | Full-HP Recover | Of those with Tera |
|---|---|---:|---:|---:|---:|
| 7103 | curriculum | 6844 | 5219 | 17 | 0 |
| 7103 | initial | 7224 | 5009 | 31 | 2 |
| 7103 | official | 7034 | 5326 | 19 | 2 |
| 8209 | curriculum | 6747 | 4932 | 24 | 0 |
| 8209 | initial | 7185 | 4988 | 30 | 0 |
| 8209 | official | 6812 | 5066 | 25 | 2 |
| 9311 | curriculum | 7047 | 5444 | 14 | 2 |
| 9311 | initial | 6867 | 4832 | 33 | 2 |
| 9311 | official | 6760 | 4964 | 18 | 2 |

## Verification and limits

Source/runtime admission binds 482 source/configuration files. All 17 focused tests passed, including real 65-entry overlap sampling, actual suffix reset exclusion, fresh-game parity, complete returns before selection, real original-next-update recovery, report/launcher/log/runtime corruption negatives and existing unavailable-choice handling. The pinned Showdown build passed. Two independent Fable snapshot reviews were adjudicated against code and executed checks; the first scheduler candidate was rejected after a real schema mismatch, the corrected candidate was independently reviewed and tested.

The full independent replay regenerated all training cohorts and 6,912 evaluation games; verified every policy/value output and outcome, complete GAE/batch fingerprints, all 288 updates, 18 periodic checkpoints and six original-next-update recoveries; and recomputed all budgets, aggregate summaries, 27 contrasts, decision/value diagnostics and gates. Receipt sealing requires clean artifact-bound production/verification exits, source-bound passing tests and measured documentation. Post-seal integrity and identical repeat sealing are recorded outside the sealed review directory.

This is an adaptively motivated development ablation using recurring train-only prefixes and known frozen opponents. Fresh official development games do not establish held-out scenario-family, novel-opponent or public ladder generalization. The fixed bank includes many early-game hidden-information tags; that is not a direct belief-inference test. Broad mechanics/generalization, value calibration and full G2/G3/G4 remain open. No final reserved evaluation, public server, Prime or GitHub publication occurred.
