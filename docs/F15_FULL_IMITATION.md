# F15 — full-battle tactical imitation and RL continuation

Status: completed and exactly replayed; imitation, pipeline and incremental-RL benefit gates each failed in all three seeds. F10 remains the reference.

F14 did not establish an entropy-removal benefit. This stage tests a different training signal: demonstrations of the frozen public-view tactical policy throughout complete official battles, followed by a matched RL continuation. The teacher is a heuristic, not an optimal-action oracle. Its own state distribution may transfer poorly to a learned policy, which is measured rather than assumed away.

## Fixed data and learner design

The registration is `contracts/full-imitation-v1/plan.json`. Generate exactly 512 official R7 train-partition games and 128 separately seeded development games with tactical-v1 playing both seats. Use each seat's own projected request/history. Retain all outcomes and attempted actions; only accepted choices become supervised labels. No win filtering, branch search or outcome-conditioned row selection. Every game ends naturally or at the existing 120-turn administrative boundary; cutoffs are retained.

Each of seeds 7103, 8209 and 9311 starts from its same-seed F10 rehearsal checkpoint with reset Adam. Run 2,048 BC steps using the unchanged NLL/Adam kernel and network. Every 128-row batch contains 64 new demonstration rows plus 64 existing F7 train rows. Sample teacher games uniformly, then decisions within each selected game. This prevents long games from dominating the training distribution. The value-head weights receive no BC gradient; changing shared features can still change value predictions.

For each seed, run two 32-cohort RL arms: original F10 and the new imitation endpoint, both with reset Adam. Each cohort collects complete official train games against the frozen tactical opponent, alternates learner seat, computes complete-trajectory GAE and selects 1,024 rows. Two epochs use four 256-row minibatches each. Both arms use the same original 128-row F7 rehearsal batch at coefficient .1. All other PPO settings stay sealed. Optimizer and selected-row budgets match between RL arms; on-policy game counts may differ. The imitation RL arm also starts with a value head fitted to the pre-BC trunk; early value error can affect its advantages and bootstraps. First-cohort terminal-only value error is reported for both arms, and their comparison measures the full initialization pipeline rather than an isolated policy-head effect. The imitation arm has additional earlier BC compute, so this is not a total-compute-matched comparison.

## Evaluation and decisions

Evaluate F10, imitation-only, F10-plus-RL and imitation-plus-RL at fixed endpoints. Each plays 64 fresh development team/mechanics pairs in both seats against random, base-power and tactical: 4,608 evaluation games. Same coordinates and actor keys are paired across endpoints. Exact current train/development whole-team hashes must be disjoint, as must teacher-probe and battle-evaluation teams. These are newly generated draws within the established R7 partitions, not unseen-species/mechanics or public-ladder tests. No final reserved data are accessed.

At every endpoint also measure NLL, target-action probability and greedy agreement on all accepted decisions from the 128 separate development teacher games, with equal-game and decision weighting. Those labels never enter an update. Repeat the 512 already sealed F10 tactical probes to assess retention of all 32 known context cells; these are explicitly familiar templates.

Retain all 45 preregistered full-battle contrasts. Per seed, imitation benefit requires at least five percentage points tactical gain over F10, a positive adjusted-bootstrap lower bound, no detected regression against other opponents, at most 5% cutoff per cell and retention of all 32 contexts. Pipeline benefit requires the same conditions for imitation-plus-RL versus both F10 and the matched F10-plus-RL control. Incremental RL benefit requires those conditions versus imitation-only. Each overall gate requires all three seeds. Teacher agreement is descriptive, never a playing-strength gate. All cutoff outcomes remain unresolved bounds rather than draws; paired-cluster bootstrap coverage is approximate with 64 pairs, and Hoeffding intervals are retained.

## Verification and limits

Build the pinned simulator and run source-bound tests before freezing. Production runs once, followed by exhaustive fresh replay of both teacher datasets, every BC/RL batch and update, all full-battle outcomes and policy/value decisions, every probe probability and all saved states. Replay loads nine original before-last checkpoints and reproduces their next updates. Scientific records must match exactly. Failures are preserved without automatic repair/restart, seed selection, checkpoint selection or budget extension.

Record measured results in this handoff and the four current progress documents before sealing; verify every live/archive source/evidence hash and byte-identical resealing. Historical evidence stays unchanged. F10 remains the reference pending broader evidence; full G2/G3/G4 and public ladder strength remain open. Local CPU only, no Prime. The existing unresolved GitHub visibility hold remains in effect.

## Interpretation and next decision

Broader imitation increased agreement with the teacher but did not pass the registered full-battle learning gate. Pooled descriptive wins (384 games per endpoint/opponent, shared team coordinates across seeds) were:

| Endpoint | Random | Base-power | Tactical |
| --- | ---: | ---: | ---: |
| F10 initial | 348 | 132 | 13 |
| Imitation only | 360 | 197 | 19 |
| F10 plus RL | 294 | 58 | 3 |
| Imitation plus RL | 352 | 171 | 14 |

These pooled counts are descriptive, not a separate significance test. Every imitation-versus-F10 contrast remains inconclusive under the registered adjusted-bootstrap test. Imitation-plus-RL beats the deteriorated RL control against base-power in all three seeds and against random in seed 8209, while control RL regresses against base-power and random in seeds 7103 and 8209. Neither RL continuation establishes tactical improvement. All final endpoints retain all 32 familiar tactical contexts, showing that passing that small template check is insufficient for full-battle transfer.

A post-hoc descriptive breakdown of the already recorded teacher-probe predictions found 201 teacher-Tera labels among 6,378 decisions. These 201 rows span 121 of the 128 probe games. Initial policies had 17–22% greedy agreement on these labels; every imitation endpoint fell to 0/201, despite about 46% agreement on ordinary-move labels. RL partially recovered Tera agreement to 1.5–6.5%. This means the greedy choice did not match the teacher's exact Tera action; it does not mean the sampling distribution assigned zero Tera probability, or that every teacher choice was optimal. The breakdown adds no samples, training updates or new acceptance gate. It is preserved in `artifacts/reviews/f15-full-imitation/probe-breakdown.json`.

RL also shifts value predictions toward the overwhelmingly losing tactical-training distribution. Terminal-only random-opponent game MSE rises from roughly 0.15–0.17 at imitation to 0.50–0.63 after its RL continuation, worse than the constant-0.5 baseline of 0.25. Low tactical-opponent MSE alone does not establish broadly useful calibration. The imitation arm's first-cohort value MSE is higher than the control in all three seeds, consistent with the preregistered shared-trunk/value-head caveat; different on-policy trajectories prevent isolating that cause here.

Recommended next stage: a matched, preregistered imitation comparison that increases training exposure to rare Tera decisions while retaining ordinary-move and switch coverage, followed separately by a controlled test of broad demonstration rehearsal during PPO. Keep the same untouched final evaluation reserve. Diagnose teacher fit and observation/representation limits before expanding compute; the compact hashed features and 64-unit shared trunk are plausible constraints, not proven causes. No further experiment or policy promotion is part of this completed run.

## Measured F15 results

Production completed in 416.28 seconds on local CPU. Combined BC/RL update time, including compilation and batch validation/sampling, was 25.98 seconds. Peak parent-process RSS was 1249.62 MiB; Node worker memory is not included in that figure.

Teacher: 512 games, 26,278 accepted demonstration rows, 26,279 attempted decisions, 1 unavailable/rejected choices, 512 natural terminals and 0 administrative cutoffs.

Probe: 128 games, 6,378 accepted demonstration rows, 6,378 attempted decisions, 0 unavailable/rejected choices, 128 natural terminals and 0 administrative cutoffs.

RL training: 8,440 games, 412,647 total decisions, 8,440 natural terminals, 0 cutoffs, 451 learner wins, 7,989 losses and 0 true ties.

RL evaluation: 4,608 games, 292,350 total decisions, 4,596 natural terminals, 12 cutoffs, 1,961 learner wins, 2,635 losses and 0 true ties.

Across all seeds: 6,144 BC updates / 786,432 row presentations (393,216 new and 393,216 old). RL selected 196,608 rows from 216,976 collected learner rows; two epochs produced 393,216 PPO row presentations plus 196,608 rehearsal presentations over 1,536 Adam updates. Every RL arm received 256 Adam updates per seed. Unavailable-choice attempts remain in the ledgers: training 55, evaluation 51.

Data inventory: 9,728 unique current training teams, 256 teacher-probe teams and 128 evaluation teams. Exact current train/development and teacher-probe/evaluation team intersections are zero. Teams and mechanics may recur within their role; no claim of global novelty across historical experiments.

### Preregistered gates

| Seed | Imitation benefit | Pipeline benefit | Incremental RL benefit |
| --- | --- | --- | --- |
| 7103 | FAIL | FAIL | FAIL |
| 8209 | FAIL | FAIL | FAIL |
| 9311 | FAIL | FAIL | FAIL |

All-seed gates: imitation_benefit = False, incremental_rl_benefit = False, pipeline_benefit = False.

### Every endpoint and opponent

Wins are shown out of 128 games per cell. Cutoffs are unresolved, not draws. Value MSE averages terminal games equally.

| Seed | Endpoint | Opponent | Wins | Losses | True ties | Cutoffs | Value MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7103 | initial | random | 116 | 12 | 0 | 0 | 0.1649 |
| 7103 | initial | base_power | 47 | 81 | 0 | 0 | 0.1866 |
| 7103 | initial | tactical | 2 | 126 | 0 | 0 | 0.1444 |
| 7103 | imitation | random | 120 | 8 | 0 | 0 | 0.1512 |
| 7103 | imitation | base_power | 65 | 62 | 0 | 1 | 0.2122 |
| 7103 | imitation | tactical | 4 | 124 | 0 | 0 | 0.1856 |
| 7103 | control_rl | random | 94 | 32 | 0 | 2 | 0.5332 |
| 7103 | control_rl | base_power | 14 | 114 | 0 | 0 | 0.0959 |
| 7103 | control_rl | tactical | 0 | 128 | 0 | 0 | 0.0016 |
| 7103 | imitation_rl | random | 115 | 12 | 0 | 1 | 0.4959 |
| 7103 | imitation_rl | base_power | 64 | 64 | 0 | 0 | 0.3495 |
| 7103 | imitation_rl | tactical | 4 | 124 | 0 | 0 | 0.0281 |
| 8209 | initial | random | 118 | 8 | 0 | 2 | 0.1598 |
| 8209 | initial | base_power | 44 | 83 | 0 | 1 | 0.2139 |
| 8209 | initial | tactical | 6 | 122 | 0 | 0 | 0.1682 |
| 8209 | imitation | random | 121 | 7 | 0 | 0 | 0.1696 |
| 8209 | imitation | base_power | 67 | 59 | 0 | 2 | 0.2223 |
| 8209 | imitation | tactical | 9 | 119 | 0 | 0 | 0.2211 |
| 8209 | control_rl | random | 96 | 31 | 0 | 1 | 0.5668 |
| 8209 | control_rl | base_power | 21 | 107 | 0 | 0 | 0.1433 |
| 8209 | control_rl | tactical | 3 | 125 | 0 | 0 | 0.0228 |
| 8209 | imitation_rl | random | 119 | 9 | 0 | 0 | 0.6335 |
| 8209 | imitation_rl | base_power | 53 | 75 | 0 | 0 | 0.3253 |
| 8209 | imitation_rl | tactical | 5 | 123 | 0 | 0 | 0.0377 |
| 9311 | initial | random | 114 | 13 | 0 | 1 | 0.1809 |
| 9311 | initial | base_power | 41 | 87 | 0 | 0 | 0.2019 |
| 9311 | initial | tactical | 5 | 123 | 0 | 0 | 0.1618 |
| 9311 | imitation | random | 119 | 8 | 0 | 1 | 0.1572 |
| 9311 | imitation | base_power | 65 | 63 | 0 | 0 | 0.2124 |
| 9311 | imitation | tactical | 6 | 122 | 0 | 0 | 0.2286 |
| 9311 | control_rl | random | 104 | 24 | 0 | 0 | 0.5675 |
| 9311 | control_rl | base_power | 23 | 105 | 0 | 0 | 0.1502 |
| 9311 | control_rl | tactical | 0 | 128 | 0 | 0 | 0.0040 |
| 9311 | imitation_rl | random | 118 | 10 | 0 | 0 | 0.5448 |
| 9311 | imitation_rl | base_power | 54 | 74 | 0 | 0 | 0.2985 |
| 9311 | imitation_rl | tactical | 5 | 123 | 0 | 0 | 0.0372 |

### Teacher agreement and familiar-context retention

Teacher-probe NLL, greedy accuracy and target mass below weight games equally. All accepted probe decisions are used; no probe row enters updates. The 32 retained contexts are previously known templates.

| Seed | Endpoint | NLL | Accuracy | Target mass | Retained contexts |
| --- | --- | ---: | ---: | ---: | ---: |
| 7103 | initial | 1.7522 | 24.80% | 24.73% | reference |
| 7103 | imitation | 1.3312 | 43.02% | 35.69% | 32/32 |
| 7103 | control_rl | 1.8884 | 21.48% | 21.66% | 32/32 |
| 7103 | imitation_rl | 1.4936 | 38.38% | 30.20% | 32/32 |
| 8209 | initial | 1.8573 | 25.68% | 24.64% | reference |
| 8209 | imitation | 1.3295 | 43.33% | 35.59% | 32/32 |
| 8209 | control_rl | 1.9679 | 21.10% | 21.57% | 32/32 |
| 8209 | imitation_rl | 1.4472 | 39.41% | 30.74% | 32/32 |
| 9311 | initial | 1.9283 | 24.81% | 23.81% | reference |
| 9311 | imitation | 1.3278 | 43.36% | 35.73% | 32/32 |
| 9311 | control_rl | 1.9169 | 24.66% | 22.12% | 32/32 |
| 9311 | imitation_rl | 1.4528 | 37.41% | 31.85% | 32/32 |

### First-cohort value diagnostics

These use pre-update value predictions in each RL arm's first collected cohort, scored only against eventual terminal outcomes. Different on-policy trajectories make these descriptive comparisons, not an isolated causal value-head test.

| Seed | Arm | Terminal games | Cutoffs | Mean prediction | Mean outcome | Game MSE | Decision MSE | Constant-0.5 MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7103 | control_rl | 48 | 0 | 0.3491 | 0.0235 | 0.1361 | 0.1400 | 0.2500 |
| 7103 | imitation_rl | 48 | 0 | 0.4205 | 0.0884 | 0.1891 | 0.1885 | 0.2500 |
| 8209 | control_rl | 48 | 0 | 0.4059 | 0.0450 | 0.1696 | 0.1731 | 0.2500 |
| 8209 | imitation_rl | 40 | 0 | 0.4764 | 0.0980 | 0.2332 | 0.2386 | 0.2500 |
| 9311 | control_rl | 48 | 0 | 0.3815 | 0.0192 | 0.1583 | 0.1582 | 0.2500 |
| 9311 | imitation_rl | 48 | 0 | 0.4831 | 0.1174 | 0.2255 | 0.2284 | 0.2500 |

### All 45 registered contrasts

Differences and intervals are percentage points, later endpoint minus comparator. Adjusted percentile-bootstrap intervals have approximate coverage with 64 paired team/mechanics clusters; finite-sample Hoeffding bounds are also retained. A cell-level improvement alone does not pass the composite retention/regression gate.

| Seed | Comparison | Opponent | Cutoff-aware difference | Adjusted bootstrap | Hoeffding | Cell result |
| --- | --- | --- | --- | --- | --- | --- |
| 7103 | control_rl_vs_initial | base_power | -25.78 to -25.78 | -42.97 to -10.16 | -74.18 to +22.62 | regression |
| 7103 | control_rl_vs_initial | random | -17.19 to -15.62 | -30.38 to -3.12 | -65.59 to +32.77 | regression |
| 7103 | control_rl_vs_initial | tactical | -1.56 to -1.56 | -6.25 to +0.00 | -49.96 to +46.84 | inconclusive |
| 7103 | imitation_rl_vs_control_rl | base_power | +39.06 to +39.06 | +23.44 to +55.47 | -9.34 to +87.46 | improvement |
| 7103 | imitation_rl_vs_control_rl | random | +14.84 to +17.19 | -0.69 to +32.03 | -33.55 to +65.59 | inconclusive |
| 7103 | imitation_rl_vs_control_rl | tactical | +3.12 to +3.12 | +0.00 to +8.59 | -45.27 to +51.52 | inconclusive |
| 7103 | imitation_rl_vs_imitation | base_power | -1.56 to -0.78 | -20.23 to +17.88 | -49.96 to +47.62 | inconclusive |
| 7103 | imitation_rl_vs_imitation | random | -3.91 to -3.12 | -14.84 to +6.94 | -52.30 to +45.27 | inconclusive |
| 7103 | imitation_rl_vs_imitation | tactical | +0.00 to +0.00 | -7.03 to +7.03 | -48.40 to +48.40 | inconclusive |
| 7103 | imitation_rl_vs_initial | base_power | +13.28 to +13.28 | -4.69 to +30.47 | -35.12 to +61.68 | inconclusive |
| 7103 | imitation_rl_vs_initial | random | -0.78 to +0.00 | -13.19 to +10.94 | -49.18 to +48.40 | inconclusive |
| 7103 | imitation_rl_vs_initial | tactical | +1.56 to +1.56 | -4.69 to +7.81 | -46.84 to +49.96 | inconclusive |
| 7103 | imitation_vs_initial | base_power | +14.06 to +14.84 | -3.91 to +32.03 | -34.34 to +63.24 | inconclusive |
| 7103 | imitation_vs_initial | random | +3.12 to +3.12 | -7.03 to +13.28 | -45.27 to +51.52 | inconclusive |
| 7103 | imitation_vs_initial | tactical | +1.56 to +1.56 | -4.69 to +7.81 | -46.84 to +49.96 | inconclusive |
| 8209 | control_rl_vs_initial | base_power | -18.75 to -17.97 | -35.94 to -0.78 | -67.15 to +30.43 | regression |
| 8209 | control_rl_vs_initial | random | -18.75 to -16.41 | -31.25 to -4.69 | -67.15 to +31.99 | regression |
| 8209 | control_rl_vs_initial | tactical | -2.34 to -2.34 | -10.16 to +5.47 | -50.74 to +46.05 | inconclusive |
| 8209 | imitation_rl_vs_control_rl | base_power | +25.00 to +25.00 | +7.81 to +42.19 | -23.40 to +73.40 | improvement |
| 8209 | imitation_rl_vs_control_rl | random | +17.19 to +17.97 | +3.12 to +32.03 | -31.21 to +66.37 | improvement |
| 8209 | imitation_rl_vs_control_rl | tactical | +1.56 to +1.56 | -5.47 to +9.38 | -46.84 to +49.96 | inconclusive |
| 8209 | imitation_rl_vs_imitation | base_power | -12.50 to -10.94 | -29.69 to +4.69 | -60.90 to +37.46 | inconclusive |
| 8209 | imitation_rl_vs_imitation | random | -1.56 to -1.56 | -10.16 to +7.73 | -49.96 to +46.84 | inconclusive |
| 8209 | imitation_rl_vs_imitation | tactical | -3.12 to -3.12 | -12.50 to +6.25 | -51.52 to +45.27 | inconclusive |
| 8209 | imitation_rl_vs_initial | base_power | +6.25 to +7.03 | -10.85 to +24.22 | -42.15 to +55.43 | inconclusive |
| 8209 | imitation_rl_vs_initial | random | -0.78 to +0.78 | -10.16 to +11.72 | -49.18 to +49.18 | inconclusive |
| 8209 | imitation_rl_vs_initial | tactical | -0.78 to -0.78 | -9.38 to +7.81 | -49.18 to +47.62 | inconclusive |
| 8209 | imitation_vs_initial | base_power | +17.19 to +19.53 | +0.00 to +36.63 | -31.21 to +67.93 | inconclusive |
| 8209 | imitation_vs_initial | random | +0.78 to +2.34 | -7.03 to +10.16 | -47.62 to +50.74 | inconclusive |
| 8209 | imitation_vs_initial | tactical | +2.34 to +2.34 | -7.03 to +11.63 | -46.05 to +50.74 | inconclusive |
| 9311 | control_rl_vs_initial | base_power | -14.06 to -14.06 | -32.73 to +2.34 | -62.46 to +34.34 | inconclusive |
| 9311 | control_rl_vs_initial | random | -8.59 to -7.81 | -22.66 to +5.47 | -56.99 to +40.59 | inconclusive |
| 9311 | control_rl_vs_initial | tactical | -3.91 to -3.91 | -10.16 to +0.00 | -52.30 to +44.49 | inconclusive |
| 9311 | imitation_rl_vs_control_rl | base_power | +24.22 to +24.22 | +6.25 to +41.41 | -24.18 to +72.62 | improvement |
| 9311 | imitation_rl_vs_control_rl | random | +10.94 to +10.94 | -2.34 to +25.78 | -37.46 to +59.34 | inconclusive |
| 9311 | imitation_rl_vs_control_rl | tactical | +3.91 to +3.91 | +0.00 to +10.85 | -44.49 to +52.30 | inconclusive |
| 9311 | imitation_rl_vs_imitation | base_power | -8.59 to -8.59 | -27.26 to +9.29 | -56.99 to +39.80 | inconclusive |
| 9311 | imitation_rl_vs_imitation | random | -1.56 to -0.78 | -10.94 to +8.59 | -49.96 to +47.62 | inconclusive |
| 9311 | imitation_rl_vs_imitation | tactical | -0.78 to -0.78 | -7.81 to +6.25 | -49.18 to +47.62 | inconclusive |
| 9311 | imitation_rl_vs_initial | base_power | +10.16 to +10.16 | -9.38 to +28.91 | -38.24 to +58.55 | inconclusive |
| 9311 | imitation_rl_vs_initial | random | +2.34 to +3.12 | -8.59 to +14.84 | -46.05 to +51.52 | inconclusive |
| 9311 | imitation_rl_vs_initial | tactical | +0.00 to +0.00 | -6.25 to +6.25 | -48.40 to +48.40 | inconclusive |
| 9311 | imitation_vs_initial | base_power | +18.75 to +18.75 | +0.00 to +37.50 | -29.65 to +67.15 | inconclusive |
| 9311 | imitation_vs_initial | random | +3.12 to +4.69 | -7.81 to +15.62 | -45.27 to +53.09 | inconclusive |
| 9311 | imitation_vs_initial | tactical | +0.78 to +0.78 | -4.69 to +6.25 | -47.62 to +49.18 | inconclusive |

Cell counts: improvement 4, inconclusive 37, regression 4.

## Executed verification

The pinned Showdown build passed. All 16 focused tests pass against 519 unchanged source/configuration files. Independent Fable review covered design and implementation; opposite-model Codex QA found a portability issue in the delegated sampler. Its original delivery was recorded as rejected, the useful implementation was corrected locally, all eight sampler tests passed, and a separate Fable review confirmed the correction. Review limitations and dispositions are preserved, including the strict probability diagnostic's explicit failure behavior.

A fresh separate process regenerated all 640 teacher/probe games and datasets, all 6,144 BC updates, all 192 RL cohorts / 1,536 optimizer updates, all 4,608 evaluation games, all 12 teacher-probe endpoints and all 6,144 familiar tactical predictions. It compared 21 checkpoints and loaded nine original before-last checkpoints to reproduce their next updates. Every scientific record matches exactly. Both launcher stages and the outer workflow exited 0. Ten copied in-memory evidence corruptions were rejected without altering production.

The five current progress documents are written before the completion receipt. Live/archive source and evidence SHA256 checks, archived receipt byte equality and identical repeat sealing are recorded outside the sealed review directory. Receipt: [F15 completion](../artifacts/reports/f15-full-imitation.json). Integrity and final completion records: `artifacts/reviews/f15-full-imitation-integrity-first.json`, `artifacts/reviews/f15-full-imitation-integrity-repeat.json`, `artifacts/reviews/f15-full-imitation-completion.json`. No new sustained two-hour qualification, GPU benchmark or public-strength test is claimed.
