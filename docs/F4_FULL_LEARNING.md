# F4 — Fixed-budget full-Showdown learning screen

Completed 2026-10-22. The fixed production campaign and bounded completion audit passed. The preregistered exploratory learning gate **passed**. All three seeds and all nine comparisons are retained.

## Measured production result

The uninterrupted run completed **12,288 training games and 18,432 development evaluation games** in **2,058.675 seconds** (34.31 minutes). It recorded **2,367,153 decisions**, **9,570 Adam steps**, **30,626 natural terminals** and **94 administrative cutoffs**. 1,012 unavailable-choice rejections were handled through refreshed requests and retained in the decision ledger; these are not skipped games.

Training had 12,254 terminals and 34 cutoffs; evaluation had 18,372 terminals and 60 cutoffs. There were no natural ties. Per-seed training counts and complete-cycle times are:

| Seed | Terminals | Cutoffs | Decisions | Adam steps | Complete-cycle seconds |
|---|---:|---:|---:|---:|---:|
| 1103 | 4,081 | 15 | 344,404 | 3,230 | 201.217 |
| 2207 | 4,087 | 9 | 338,924 | 3,180 | 190.932 |
| 3301 | 4,086 | 10 | 335,670 | 3,160 | 196.300 |

Observed wins below use all 1,024 games as the denominator, with cutoffs separately listed. Improvement intervals use win/tie rewards and unresolved-cutoff bounds, as preregistered. CI endpoints are percentage points.

| Seed | Opponent | Wins before → after / 1,024 | Cutoffs before → after | Cutoff-aware point gain | Approximate simultaneous CI | Hoeffding CI |
|---|---|---:|---:|---:|---:|---:|
| 1103 | random | 530 → 951 | 14 → 2 | [39.75, 41.31] | [34.67, 46.29] | [24.58, 56.47] |
| 1103 | base_power | 107 → 484 | 0 → 0 | [36.82, 36.82] | [31.84, 41.70] | [21.65, 51.98] |
| 1103 | tactical | 6 → 60 | 0 → 0 | [5.27, 5.27] | [3.32, 7.42] | [-9.89, 20.44] |
| 2207 | random | 513 → 940 | 19 → 5 | [39.84, 42.19] | [34.92, 47.07] | [24.68, 57.35] |
| 2207 | base_power | 81 → 496 | 1 → 0 | [40.43, 40.53] | [35.74, 45.31] | [25.27, 55.69] |
| 2207 | tactical | 11 → 77 | 1 → 0 | [6.35, 6.45] | [4.00, 9.08] | [-8.82, 21.61] |
| 3301 | random | 513 → 929 | 15 → 1 | [39.16, 40.72] | [34.08, 45.70] | [24.00, 55.89] |
| 3301 | base_power | 84 → 467 | 2 → 0 | [37.21, 37.40] | [32.32, 42.43] | [22.04, 52.57] |
| 3301 | tactical | 8 → 55 | 0 → 0 | [4.59, 4.59] | [2.54, 6.74] | [-10.57, 19.75] |

| Seed | Base-power positive bound | Base-power ≥5-point gain | No detected regression | Final cutoffs ≤5% | Gate |
|---|---|---|---|---|---|
| 1103 | yes | yes | yes | yes | pass |
| 2207 | yes | yes | yes | yes | pass |
| 3301 | yes | yes | yes | yes | pass |

## Value behavior and training cost

Terminal-game mean value MSE starts at 0.25 because initialization predicts 0.5. The following final values are descriptive and condition on reaching a natural terminal; they do not establish calibration or generalization.

| Seed | Random MSE | Base-power MSE | Tactical MSE | Final mean value, random / base / tactical |
|---|---:|---:|---:|---|
| 1103 | 0.1508 | 0.1874 | 0.1120 | 0.6780 / 0.4334 / 0.2816 |
| 2207 | 0.1386 | 0.1931 | 0.1221 | 0.6996 / 0.4500 / 0.2972 |
| 3301 | 0.1339 | 0.1854 | 0.1298 | 0.7173 / 0.4697 / 0.3181 |

Training complete-cycle time sums to **588.448 seconds**, at **1,731.7 decisions/second**. Optimizer calls account for **13.828 seconds** (2.35% of that time). These include first-use compilation; the remaining campaign time includes evaluation, startup and report processing. Peak parent RSS was **849.2 MiB**; Node worker RSS is excluded. Every expected cohort timing record is present. This is not a two-hour soak or a CPU/GPU comparison.

## Completion verification

All 64 focused tests pass (31 Node, 33 Python), bound to 392 source/configuration files. The independent reviews and verified fixes are recorded in [review dispositions](../artifacts/reviews/f4-full-learning/review-dispositions.md). The separate preflight used 64 training and 96 evaluation games: 158 terminals, two cutoffs and 14,530 decisions. Those games and verification replays are excluded from production counts.

The completion audit verified all committed hashes, counters, partitions and evaluation summaries. Fresh replay exactly matched 24 original training games and their three next updates, including model, both Adam moments, optimizer counters and metrics, plus 36 original evaluation games with their policy/value decisions and outcomes. The earlier F2/F3 sealed sources and evidence remain unchanged. No production failure, restart, resampling or budget extension occurred.

Results: [production report](../artifacts/runs/f4-full-learning-v1/report.json), [bounded replay audit](../artifacts/reviews/f4-full-learning/production-audit/audit.json), [timing/metric diagnostics](../artifacts/reviews/f4-full-learning/diagnostics.json), [completion receipt](../artifacts/reports/f4-full-learning.json).

The next controlled experiment should address the gap against tactical opponents through a fixed training-opponent mixture and targeted scenarios, with a matched-budget control. This result does not identify the representation, training distribution or budget as the sole bottleneck. No follow-on experiment is included in F4.

## Fixed experiment

The executable registration is `contracts/full-learning-v1/plan.json`. Three fresh initializations, seeds **1103, 2207 and 3301**, each receive **4,096 training games**: **12,288 total**. Training remains official-team self-play. We freeze policy parameters for each eight-game cohort, update with unchanged F2 PPO, and atomically commit the parameters, both Adam moments, step counter and cohort evidence. Eight persistent Node workers share fixed-shape player-only CPU inference. No curriculum, opponent-mixture change, larger network, search or GPU intervention is included.

Each initial and final policy plays **512 paired development matchups against each of random, base-power and the frozen tactical heuristic**. Both seat orientations are retained, giving **1,024 games per cell**, **18,432 evaluation games**, and **30,720 total experimental games**. The new evaluation namespace is seed 8422026. No intermediate checkpoint selection, seed removal, opponent removal, resampling, optional stopping or budget expansion is allowed. Preflight uses its own namespace and is excluded from these counts.

The frozen `tactical-v1` heuristic uses its own private request and protocol history available to that seat. It receives no opposing private request or simulator state. Advice appears only in coordinator metadata and never in either neural player's observation. It is evaluated against the learned policy, not used to generate training targets. An untranslatable or mask-illegal heuristic choice fails the experiment. Ordinary Showdown unavailable-choice responses follow F2's request-refresh path and are recorded.

## Evaluation and interpretation

The sampling unit is the team/mechanics pair, retaining both seats and both policy endpoints in one cluster. A natural terminal has its observed reward. An administrative cutoff has unknown eventual reward in **[0,1]**, rather than being counted as a draw. Pair-average improvement therefore has lower and upper endpoints.

We use 20,000 paired cluster bootstrap resamples, seed 8422027. Each resample uses identical indices for the lower and upper endpoint means. The reported percentile interval uses tail probability 0.05/(2×9) for the nine seed/opponent comparisons. Its simultaneous coverage is **approximate**, not a finite-sample guarantee. We also report the more conservative bounded Hoeffding interval separately; it does not replace the preregistered gate after observing results. Paired resampling and percentile intervals follow the general construction documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); the cutoff bounds and experiment-specific gate are defined here and in the executable plan.

The exploratory learning gate requires every seed to meet all of:

- Base-power improvement has a positive bootstrap lower confidence bound.
- Its cutoff-aware point lower improvement is at least five percentage points.
- No opponent cell has a bootstrap upper bound below zero.
- Every final cell has a cutoff rate no greater than 5%.

Absence of a detected regression is not a noninferiority result. A completed experiment may fail this learning gate. All seeds, cells and conservative intervals must be reported regardless. Terminal-only value MSE is averaged by game and conditions on policy-dependent terminal outcomes; it is not a general calibration certificate. We report value means, terminal/cutoff counts, natural ties and update finiteness alongside wins.

The same development teams recur across seeds and opponents. Agreement across cells is not independent team-level replication. R7 separates exact ordered whole-team identities; it does not enforce species or scenario-family disjointness. No final reserved evaluation data is decoded. This screen measures learning under the current distribution and opponents, not public ladder generalization.

## Recovery, checks and evidence

Only complete cohorts enter a checkpoint. An explicit resume requires the same executable inventory, plan and runtime; an uncommitted cohort may replay with identical coordinates, never replacement games. Completed runs cannot be extended. Complete-cycle timing includes collection, update, admissions and durable writes, with missing telemetry after an interrupted commit reported rather than imputed. Parent peak RSS excludes the eight Node workers and is not a sustained memory qualification.

The new adapter is additive over frozen F2. F2/F3 source, reports, receipts and historical handoffs are preserved. Progress prose is excluded from executable admission. Focused regression includes real-engine player isolation, tactical equivalence, forced/wait requests, cutoff advice, F3 collection parity, paired statistics, source admission, interrupted-cohort recovery and a corruption negative for completion verification.

The completion audit checks all stored cohort, counter, checkpoint, partition, evaluation-summary and hash bindings. Its explicitly bounded independent replay covers the original last eight training games and ensuing update per seed, plus the first paired evaluation matchup for every endpoint/opponent/seed: **24 training replays and 36 evaluation replays**. Model parameters, Adam moments, step count, update metrics and sampled policy/value/actions/outcomes must match the original. This does not claim exhaustive independent replay of all 30,720 games.

Production run: `artifacts/runs/f4-full-learning-v1`. Reviews, preflight, launcher and audit evidence: `artifacts/reviews/f4-full-learning/`. Completion receipt: `artifacts/reports/f4-full-learning.json`. Final documentation is copied into the receipt's evidence inventory before sealing; repeat sealing must produce identical bytes.

Full G2/G3/G4 and public ladder strength remain open. No public server or Prime resources. GitHub remains under the existing unresolved visibility hold.
