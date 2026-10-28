# F28 — collection-batch comparison

Status: fixed production and exhaustive replay passed; completion sealing follows the measured documentation below.

F27 reproduced every sampled original update and found variable actor-gradient directions across game subsets. That supports testing collection size, without establishing noise as the cause of weak learning. This experiment compares two ways of spending the same PPO row/update budget.

| Per seed/arm | Small | Large |
|---|---:|---:|
| Selected rows per cohort | 1,024 | 8,192 |
| Self-play / tactical rows | 512 / 512 | 4,096 / 4,096 |
| Cohorts | 128 | 16 |
| Adam steps per cohort | 8 | 64 |
| Total selected rows | 131,072 | 131,072 |
| Total presentations | 262,144 | 262,144 |
| Total Adam steps | 1,024 | 1,024 |
| Evaluation cohorts | 0, 64, 128 | 0, 8, 16 |
| Checkpoint cohorts | 0, 63, 64, 127, 128 | 0, 7, 8, 15, 16 |

Both arms start from each same-seed F25 final matchup checkpoint (7103, 8209, 9311), with identical Adam reset. Those are experimental hosts, not promoted winners. The 60,806-parameter model, 1,278-channel player encoder, legal action law, reward/GAE, two epochs, 256-row minibatches, learning rate, entropy/value coefficients and frozen tactical opponent remain unchanged. Complete games are collected under a fixed policy through both streams, full same-player GAE is calculated before uniform quota selection, and opponent-controlled rows never enter PPO. The cap is 64 games per stream for small and 512 for large, in eight-game chunks. Failure preserves evidence; no automatic cap increase, restart or resampling.

Rows, presentations and optimizer steps match. Simulator games and compute are measured separately. Large batches refresh the collection policy eight times less often and execute 64 successive updates against frozen old likelihoods/GAE targets, so this tests a joint collection-size/refresh intervention. It cannot attribute any benefit solely to variance reduction. Log first-minibatch and final-epoch pre-update minibatch KL/clip metrics by seed/arm; these are descriptive, not full-batch post-update divergences or stopping rules.

The endpoints align at 0, 65,536 and 131,072 selected rows. Each evaluates 128 paired official development schedules in both seats against random, base-power and tactical, plus 32 separate self-play diagnostic games. Common schedules and initialization make endpoint-0 panels duplicate deterministic checks; contrasts share baselines and are dependent. All 27 registered final contrasts remain in the approximate family adjustment. The midpoint is descriptive. Base-power learning and large-minus-small gates require a cutoff-aware gain of at least five percentage points, adjusted bootstrap lower bound above zero, no detected random/base-power regression and at most 5% cutoffs. Tactical adaptation is secondary because tactical is trained on. All three seeds are required for all-seed claims; failure/inconclusive results remain in the report. Terminal-only values and familiar tactical probes are descriptive. No checkpoint selection or promotion.

The fixed total is 432 cohorts, 6,144 optimizer steps, 786,432 selected rows, 1,572,864 presentations, 13,824 main evaluation games, 576 self-play diagnostic games, 30 checkpoint comparisons and 12 original-next-update recoveries. Training game count is measured rather than fixed. All admitted exact-team training/development history through F26 is enforced; F27 added no games. New training/evaluation namespaces are reserved for F28. Final reserved data remains untouched.

Qualification requires focused small-optimizer bitwise parity, large row coverage/frozen-target checks, large original-next-update checkpoint recovery, arm-counter/metadata rejection, gates and collector quota/cap/complete-game tests. Sealed F25/F26/F27 checks and receipts are admitted unchanged. A separate engineering-only preflight replays one complete cohort for all six seed/arm combinations with fresh train-only namespaces, without outcome selection. Production then requires exhaustive fresh game/update replay, copied-evidence negatives, measured documentation, first sealing, all live/archive hashes and identical repeat sealing. Persistent helpers and continuation instructions are under artifacts/reviews/f28-batch-collection.

Fresh development games against familiar heuristics and recurring official generation do not establish held-out-team or public-ladder generalization. Full G2/G3/G4 and public strength remain open. No public server, Prime or GitHub publication; the existing visibility hold remains pending. Scope ends with this comparison or preserved failure, without another automatic experiment.

The statistical implementation is the unchanged, source-pinned `pokemon_full_learning.statistics.compare`: resample 128 paired schedules with both seat games kept in each cluster, 20,000 bootstrap replicates, percentile tails 0.05/(2×27) and 1−0.05/(2×27). Cutoffs contribute unresolved score bounds [0,1]; contrasts use lower/upper difference bounds. Improvement requires adjusted lower >0; regression requires adjusted upper <0. Only about 18.5 replicates occupy each extreme tail, so the bootstrap is approximate; bounded Hoeffding intervals are also reported. No detected regression is not a noninferiority result. Common schedules and duplicated initialization panels do not make the 27 contrasts independent.

## Qualification result

All15 focused checks passed against702 source/configuration files; the pinned build exited0. Independent implementation review found no blocking defect after local verification of its findings. The engineering preflight completed in124.72 seconds:832 distinct train-only games and27,648 selected rows across all three hosts and both batch sizes; each game/selection/update was replayed exactly, including216 original Adam steps and216 replay steps. Small strata used8–16 self-play and24 tactical games (cap64); large used72–80 self-play and168 tactical games (cap512). All six post-update checkpoints reloaded exactly. Preflight teams enter training history, and source/runtime maps match the qualification. These are engineering checks, not an evaluation of strength.

## Measured execution

# F28 measured results

Production completed in 1752.59 seconds on CPU. Replay/receipt verification remains a separate requirement.

- training: 25360 games, 25355 natural terminals, 5 cutoffs, 1361794 attempted decisions.
- evaluation: 13824 games, 13823 natural terminals, 1 cutoffs, 823705 attempted decisions.
- selfplay_diagnostics: 576 games, 576 natural terminals, 0 cutoffs, 33838 attempted decisions.

All-seed learning: {'large': False, 'small': False}; all-seed base-power batch benefit: False; trained-on tactical adaptation benefit: False. Contrasts: {'inconclusive': 27}.

| Seed | Arm | Base-power wins: initial → final /256 | Tactical wins: initial → final /256 | Learning gate | Familiar final mastery |
|---|---|---:|---:|---|---|
| 7103 | small | 131 → 128 | 28 → 20 | False | False |
| 7103 | large | 131 → 122 | 28 → 20 | False | False |
| 8209 | small | 143 → 142 | 17 → 26 | False | False |
| 8209 | large | 143 → 144 | 17 → 27 | False | False |
| 9311 | small | 139 → 135 | 25 → 20 | False | False |
| 9311 | large | 139 → 138 | 25 → 18 | False | False |

Peak parent RSS 1920.2 MiB (workers excluded). Collection 1620.23s; optimizer calls including compile 19.69s. Finite gradients/likelihood ratios are engineering telemetry, not trust-region or calibration guarantees.

Both arms use the same F25 final matchup host and reset Adam. Selected-row and optimizer budgets match; games and compute are measured, not matched. Larger collection also changes policy refresh and within-cohort likelihood staleness. Endpoint0 repeats are deterministic and contrasts are dependent; all27 stay in the approximate family adjustment. Tactical is trained on; its adaptation gate is reported separately. Fresh development schedules against recurring generation and known heuristics, not public ladder or held-out-team generalization. All seeds/27 contrasts retained; intervals and cutoff-aware gates remain in measured-review.json. No promotion, final reserved data, public server, Prime or GitHub publishing.

