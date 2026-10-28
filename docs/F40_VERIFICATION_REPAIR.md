# F40 — verified F39 production; baseline learning gate not met

Status: both F40 stages, cleanup, receipt creation and the independent postcheck passed on **October 28, 2026, at 6:14 p.m. EDT**. Independent cross-model claims review completed with no material numerical finding; its two wording/evidence-scope observations are resolved below. **None of the three baseline primary gates passed.** All twelve contrasts are inconclusive under the registered adjusted bootstrap; no regression was detected. This does not establish either no benefit or noninferiority.

F40 completed sampled verification of preserved F39 production. It added no training cohorts, selected training rows or training optimizer steps. F39's original workflow remains failed: its verifier rejected the generic lifecycle parent package at startup. Its source, evidence, failure receipt and absent success paths remain unchanged.

## Scope and qualification

The additive adapter invokes the unchanged F39 verifier and temporarily replaces only its module-inventory callable. Exact exceptions are `pokemon_teacher_runtime` and `pokemon_teacher_runtime.lifecycle`; other matching names remain forbidden. Module file/origin paths and live/frozen SHA-256 hashes must match the original admission. The original callable is restored afterward. Training, reporting arithmetic, seeds, schedules and budgets are unchanged.

Qualification passed 31 focused checks, compilation and pinned-build verification. Final independent implementation QA found no code blocker. Earlier cross-model review used an earlier source snapshot; the final-source checks and QA are separately recorded. Before/after module snapshots passed, but do not audit arbitrary subprocess execution or transient imports.

F40 executed **66 fresh sampled replays**: 18 training games (first complete game per stream at absolute cohorts 512 and 767 for each seed), plus 48 evaluation games (first paired seats per opponent at both endpoints for each seed). It verified nine checkpoint states and six original saved-batch next updates, recomputing 48 optimizer steps without continuing training. Reporting checked all saved ledgers and evaluation records and recomputed all twelve contrasts. Replay coverage is sampled, not exhaustive.

## Inherited training and evaluation

All seeds—7103, 8209 and 9311—continued their F34 states, preserving parameters, Adam moments/step 4,096 and stream cursors. Each added 256 cohorts and 2,048 optimizer steps, ending at absolute cohort 768 and Adam step 6,144. Each cohort selected 256 self-play, 256 frozen same-seed F32 and 512 tactical learner rows. No teacher or demonstration data were used.

| Verified F39 increment | Count |
|---|---:|
| Cohorts | 768 |
| Optimizer steps | 6,144 |
| Selected learner rows | 786,432 |
| Row presentations | 1,572,864 |
| Training games | 36,768 |
| Main evaluation games | 4,608 |
| Checkpoint comparisons verified by F40 | 9 |
| Original-next-update recoveries verified by F40 | 6 |

“Decisions” below are decision attempts, including rejected choices; they are not all accepted battle actions. Self-play is part of training, not an additional diagnostic panel.

| Inherited activity | Games | Decision attempts | Terminals | Cutoffs | True ties |
|---|---:|---:|---:|---:|---:|
| Self-play training | 6,144 | 327,156 | 6,144 | 0 | 0 |
| Frozen F32 training | 12,200 | 633,238 | 12,200 | 0 | 0 |
| Tactical training | 18,424 | 905,520 | 18,424 | 0 | 0 |
| **All training** | **36,768** | **1,865,914** | **36,768** | **0** | **0** |
| Initial + final evaluation | 4,608 | 216,997 | 4,608 | 0 | 0 |

All 41,376 inherited games reached natural terminals. Evaluation contained 768 games per seed per endpoint: 512 baseline-v1, 128 tactical, 64 base-power and 64 random. The 66 F40 replay executions are additional verification work, not additional independent evaluation evidence.

| Seed | Stream | Collected learner rows | Selected | Excluded |
|---|---|---:|---:|---:|
| 7103 | self_play | 106,796 | 65,536 | 41,260 |
| 7103 | frozen_f32 | 101,152 | 65,536 | 35,616 |
| 7103 | tactical | 152,924 | 131,072 | 21,852 |
| 8209 | self_play | 112,324 | 65,536 | 46,788 |
| 8209 | frozen_f32 | 107,295 | 65,536 | 41,759 |
| 8209 | tactical | 152,876 | 131,072 | 21,804 |
| 9311 | self_play | 108,036 | 65,536 | 42,500 |
| 9311 | frozen_f32 | 107,679 | 65,536 | 42,143 |
| 9311 | tactical | 150,935 | 131,072 | 19,863 |
| All | self_play | 327,156 | 196,608 | 130,548 |
| All | frozen_f32 | 316,126 | 196,608 | 119,518 |
| All | tactical | 456,735 | 393,216 | 63,519 |
| **All** | **All streams** | **1,100,017** | **786,432** | **313,585** |

Training recorded 695 rejected attempts, including 531 learner-controlled attempts; 377 selected rows corresponded to rejected learner attempts. Collected learner rows therefore include controlled rejected attempts. Excluded rows are the fixed-selection remainder, not a count of illegal actions. All rejection attribution and row inventories were checked.

## Learning and opponent results

The primary gate requires, separately for every seed: cutoff-aware lower point gain of at least **5 percentage points** against baseline-v1, adjusted-bootstrap lower bound above zero, and final baseline cutoff rate at most 5%. All three seeds failed the gain and adjusted-lower-bound conditions; all passed the cutoff condition with zero cutoffs. Overall primary success is false.

| Seed | Baseline wins, initial → final / 512 | Gain (percentage points) | ≥5-point gain | Adjusted lower >0 | Cutoff condition | Primary |
|---|---:|---:|---|---|---|---|
| 7103 | 252 → 269 | +3.3203 | Fail | Fail | Pass | Fail |
| 8209 | 236 → 235 | -0.1953 | Fail | Fail | Pass | Fail |
| 9311 | 248 → 252 | +0.7812 | Fail | Fail | Pass | Fail |

All twelve dependent paired-schedule contrasts are retained below. Intervals and gains are **percentage points**, rounded for display; the summary retains full precision. The bootstrap uses 20,000 paired-seat-cluster resamples, one family of twelve contrasts, and nominal family alpha 0.05. Its familywise coverage is approximate/exploratory. Bounded Hoeffding intervals are also retained; neither form establishes public or held-out performance.

| Seed | Opponent | Wins initial → final / games per endpoint | Cutoff-aware point bounds | Adjusted bootstrap interval | Bounded Hoeffding interval | Diagnostic |
|---|---|---:|---:|---:|---:|---|
| 7103 | baseline-v1 | 252 → 269 / 512 | [+3.320, +3.320] | [-4.102, +10.547] | [-18.642, +25.282] | inconclusive |
| 7103 | tactical | 37 → 43 / 128 | [+4.688, +4.688] | [-7.031, +16.406] | [-39.236, +48.611] | inconclusive |
| 7103 | base_power | 56 → 51 / 64 | [-7.812, -7.812] | [-23.438, +4.688] | [-69.930, +54.305] | inconclusive |
| 7103 | random | 64 → 62 / 64 | [-3.125, -3.125] | [-10.938, +0.000] | [-65.243, +58.993] | inconclusive |
| 8209 | baseline-v1 | 236 → 235 / 512 | [-0.195, -0.195] | [-7.031, +6.250] | [-22.157, +21.767] | inconclusive |
| 8209 | tactical | 44 → 50 / 128 | [+4.688, +4.688] | [-11.719, +21.094] | [-39.236, +48.611] | inconclusive |
| 8209 | base_power | 50 → 57 / 64 | [+10.938, +10.938] | [-4.688, +26.562] | [-51.180, +73.055] | inconclusive |
| 8209 | random | 61 → 62 / 64 | [+1.562, +1.562] | [-6.250, +9.375] | [-60.555, +63.680] | inconclusive |
| 9311 | baseline-v1 | 248 → 252 / 512 | [+0.781, +0.781] | [-6.445, +8.203] | [-21.181, +22.743] | inconclusive |
| 9311 | tactical | 51 → 56 / 128 | [+3.906, +3.906] | [-9.375, +17.188] | [-40.018, +47.830] | inconclusive |
| 9311 | base_power | 56 → 52 / 64 | [-6.250, -6.250] | [-21.875, +6.250] | [-68.368, +55.868] | inconclusive |
| 9311 | random | 64 → 62 / 64 | [-3.125, -3.125] | [-10.938, +0.000] | [-65.243, +58.993] | inconclusive |

All adjusted intervals include or touch zero. Point losses remain visible: seed 8209 against baseline-v1; seeds 7103 and 9311 against base-power and random. No contrast met the regression diagnostic, which requires an adjusted upper bound strictly below zero. Tactical gains concern a trained-on opponent. Baseline-v1 was excluded from this training but is historically familiar.

Cutoffs would retain unresolved score bounds rather than be scored as draws. None occurred here, so every cutoff-aware point interval collapses to a point. Sampling uncertainty remains wide; zero observed cutoffs does not remove it.

## Terminal-only value and optimizer diagnostics

Value summaries average within each game first, then weight terminal games equally. MSE compares recorded actor values with the terminal reward for the actor's side; administrative cutoffs are excluded rather than assigned terminal targets. These are descriptive diagnostics, not calibration or strength certificates.

| Seed | Evaluation opponent | Mean value initial → final | Value MSE initial → final |
|---|---|---:|---:|
| 7103 | baseline-v1 | 0.425961 → 0.377653 | 0.175132 → 0.204131 |
| 7103 | tactical | 0.344782 → 0.304165 | 0.153481 → 0.142010 |
| 7103 | base_power | 0.598322 → 0.506843 | 0.193672 → 0.231962 |
| 7103 | random | 0.741175 → 0.649896 | 0.122519 → 0.182684 |
| 8209 | baseline-v1 | 0.411928 → 0.518656 | 0.169081 → 0.180370 |
| 8209 | tactical | 0.374376 → 0.479839 | 0.161440 → 0.164028 |
| 8209 | base_power | 0.576356 → 0.680678 | 0.177448 → 0.121115 |
| 8209 | random | 0.710775 → 0.807088 | 0.139039 → 0.095978 |
| 9311 | baseline-v1 | 0.429357 → 0.448073 | 0.173838 → 0.168371 |
| 9311 | tactical | 0.439357 → 0.414845 | 0.154245 → 0.152337 |
| 9311 | base_power | 0.611292 → 0.573845 | 0.190421 → 0.189084 |
| 9311 | random | 0.777644 → 0.762161 | 0.095363 → 0.106169 |

| Seed | Training terminal mean value | Training terminal value MSE |
|---|---:|---:|
| 7103 | 0.448647 | 0.172899 |
| 8209 | 0.435523 | 0.171220 |
| 9311 | 0.449016 | 0.172916 |

Baseline value MSE rose for seeds 7103 and 8209 and fell slightly for 9311. Opponent-specific MSE changes are mixed. Training value averages combine changing policies and opponent streams, so they do not measure endpoint improvement.

| Seed | First-minibatch mean approximate KL | First-minibatch mean clip fraction | Final-epoch mean approximate KL | Final-epoch mean clip fraction |
|---|---:|---:|---:|---:|
| 7103 | -2.252252e-10 | 0.000000 | 0.001045035 | 0.007331848 |
| 8209 | -3.392068e-10 | 0.000000 | 0.001015332 | 0.007339478 |
| 9311 | -2.547771e-10 | 0.000000 | 0.000880551 | 0.006061554 |

Each seed contributes 256 first-minibatch observations and 1,024 final-epoch minibatch observations. These are **pre-update minibatch statistics**, not full-batch post-update KL. Tiny signed first-minibatch estimates are reported as recorded. They do not establish optimization stability or identify a cause for the learning result.

## Measured costs and integrity

| Measurement | Seconds / memory | Scope |
|---|---:|---|
| F39 production wall time on CPU | 2404.115924 s | 40.07 minutes; not summed CPU-core time |
| F39 collection | 2315.235421 s | Training plus both evaluation endpoints; simulation, encoding, inference, transport and bookkeeping |
| F39 learner updates including compilation | 47.480471 s | Across all seeds |
| F39 internal saved-update replay | 0.354956 s | Original production recovery work |
| F39 peak parent RSS | 865.75 MiB | Parent high-water mark; excludes workers, GPU and total machine memory |
| F40 full launcher | 50.170788 s | Includes stages, integrity work, receipt and cleanup |
| F40 sampled-verification stage | 23.164700 s | Wrapper wall time |
| F40 verifier report | 22.417886 s | Nested within sampled-verification stage |
| F40 measured-results stage | 2.789183 s | Wrapper wall time |

These scopes overlap and must not be summed as independent costs. Per-seed update times including compilation were 16.529361 s, 15.493691 s and 15.457419 s for 7103, 8209 and 9311. F40 reports no separate parent-memory peak; the 865.75 MiB figure belongs to inherited F39 production.

The outer postcheck passed: receipt content/byte linkage, complete inventories, original-failure preservation, semantics and cleanup. It verified 2,096 original source files and 5,789 original evidence files with their frozen/archive copies, plus 98 F40 evidence files; completion records 31 F40 admitted source files. Both stages and the driver/launcher exited zero with cleanup passed. The independent cross-model claims review found no material numerical error. The README now states that the fresh replay execution reused the original verifier; it does not claim an independent reimplementation. The CPU label is corroborated by the original admitted runtime and F40 before/after runtime records, all recording `backend: cpu`.

## Evidence and publication

- [F40 receipt](../artifacts/reports/f40-online-verification.json): SHA-256 `a7b53036219d3d63505121d0cca73743d3c85c5ff985dd3bc855091043c29a2d`.
- [Measured summary](../artifacts/reviews/f40-online-verification/measured-results-v1/summary.json): SHA-256 `303f147afac5019233d1a6ffa1391fec0f7890812d158a70951ad57fd17582a8`.
- [Independent postcheck](../artifacts/reviews/f40-online-verification-postcheck.json), [completion](../artifacts/reviews/f40-online-verification/completion.json), and [outer launcher exit](../artifacts/reviews/f40-online-verification-launcher-exit.json).
- [Preserved F39 failure receipt](../artifacts/reports/f39-online-continuation-failure.json): SHA-256 `0ebefe24c86c9f2b2c534fdcd1516e805cbf8e893c4fd5adf2e1ffc99cbf6d61`.
- [Qualification](../artifacts/reviews/f40-online-verification-preparation/qualification.json), [recorded runtime](../artifacts/reviews/f40-online-verification/workflow-result.json), and [claims-review dispositions](../artifacts/reviews/f40-online-verification-claims-review.json).
- [Compact claims evidence](../artifacts/reviews/f40-online-verification-claims-evidence.json) retains full-precision summary values and source bindings; raw artifacts remain local.

The user chose to keep GitHub public and sync code/docs. The first source sync reached public `main` at `a5cfd1cc4321820fa9a73abf5269c54e91a9115b`; this handoff and the README carry the verified result in the subsequent code/docs update. Sync receipts remain local. No new raw run archives were published.

Baseline-v1 is historically familiar; tactical and frozen F32 were trained on. Recurring official generation and development schedules do not establish independent replication, held-out-team or public strength. Full G2/G3/G4 remain open. No promotion, public server, final reserved data, cloud training/spending or Prime is authorized. F35/F36/F37 remain failed; their partial teacher data were not used. F40 does not complete the separate blocked F35 goal, and no next experiment is authorized.
