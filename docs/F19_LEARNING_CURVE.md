# F19 — fixed-corpus learning curve

Status: experiment completed; descriptive learning gate 0/3 seeds.

F19 completed (2026-10-25): continued all three sealed F18 expanded checkpoints from 2,048 to 8,192 updates, preserving Adam state, the 106,149-row corpus, model and sampling. The run added 18,432 updates / 2,359,296 row presentations and 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,017 attempted decisions). Voluntary-switch training agreement changed from 14.67–16.51% to 59.47–63.74%; fresh-probe agreement changed from 3.85–8.12% to 8.40–12.96%. The descriptive learning gate passed in 0/3 seeds. These are teacher-imitation metrics. All 16 focused tests, exhaustive fresh replay, 15 checkpoint comparisons and six original-next-update recoveries passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer.

The added optimization budget substantially improved voluntary-switch fitting: game-weighted training agreement rose from 14.67–16.51% at step 2,048 to 59.47–63.74% at step 8,192. The same-seed fresh-probe point gains were only 2.85–4.84 percentage points, ending at 8.40–12.96% agreement. All three adjusted switching intervals include zero, and every one of the 12 primary approximate-bootstrap and conservative intervals contains zero (one Tera bootstrap interval touches zero). No statistically resolved improvement or regression is established by these registered intervals. Every seed failed both the 25% final switching threshold and 10-point gain threshold, while all other family-preservation checks and all 32 known-context retention checks passed. The ordinary-move point estimate decreased by 1.49 points for seed 9311; this remains within the descriptive tolerance and statistically inconclusive.

Fresh-probe voluntary-switch NLL worsened in every seed despite higher point accuracy: 2.835 to 3.096, 2.822 to 3.208 and 2.871 to 3.173. These descriptive log-loss changes indicate a poorer probability fit on fresh switching decisions; they do not establish the cause of the gap or a calibrated battle-value result.

All 1,422 voluntary-switch training rows had been sampled by step 8,192, with 65,536 draws per seed and at least 5/6/6 presentations per row for seeds 7103/8209/9311. Missing exposure to those rows does not explain the remaining fitting error. Longer training improves fitting here, but this budget does not close the fresh-state transfer gap or establish the best remedy. Capacity, representation, teacher ambiguity, data diversity and their interactions remain candidates. The prior F17 flat semantic-feature addition did not pass its own screen. This experiment did not train or evaluate an on-policy RL continuation, and teacher agreement does not measure battle strength.

Next proposed step: compare the existing flat policy with a shared action-scoring architecture that reuses parameters across Pokémon slots and legal move/switch choices. Use the same public information, frozen corpus, family sampling and preregistered update budget, with a matched control, fresh development probe and all three seeds. First verify action/slot remapping and hidden-information isolation. This tests a structural generalization hypothesis; F19 does not prove the architecture is the cause or that the new design will work. Do not expand data and architecture together, add more optimization after seeing results, promote a policy, or start PPO automatically. Local CPU remains sufficient for the bounded comparison.

## Fixed design and limits

Fixed continuation on the frozen F18 expanded corpus. Resume all three sealed step2048 expanded checkpoints including Adam moments/counter. Keep original928-feature model, action law/NLL/optimizer, balanced32/16/8/8 teacher sampling and64 old rehearsal rows; resume sampling index2048, never reset Adam or namespaces. Record endpoints2048/4096/8192 without early stopping or selection. Primary comparison is8192 minus2048 only, across3 seeds and4 families. Intermediate4096 metrics are descriptive, with no inferential contrasts. Collect256 fresh development teacher games, exclude exact teams used in F15/F16/F18 development and all training. No new training games. All games/accepted decisions retained, including cutoffs; minimum support failure aborts without resampling. Gate requires25% final voluntary agreement and10-point gain, no other family drop over5points, all32 known contexts retained; all3 seeds required. Teacher imitation is not battle strength. Size/architecture/sampling remain fixed while optimization budget and example repetitions increase. No automatic follow-on, PPO or promotion. Final reserved data/public server/Prime remain unused; GitHub stays held.

Continue the sealed F18 expanded checkpoint and both Adam moments/counter at step 2,048; new sampling indices are 2,048–8,191. The optimization, data, feature representation and family quotas stay fixed. Endpoints 2,048/4,096/8,192 are all reported; only 8,192 minus 2,048 is primary. The intermediate endpoint cannot be selected. Added updates total 18,432, compared with 24,576 accumulated updates including F18. Added teacher and rehearsal presentations each total 1,179,648. No new training games.

The probe excludes exact teams from all 2,048 training games and F15/F16/F18 development games. F17 reused F16 games. Known species, mechanics and the tactical teacher recur, so fresh teams do not establish unseen-opponent or final reserved-data generalization. Both players remain in one game cluster. All accepted choices remain, including any cutoff prefixes; rejected attempts remain evidence but are excluded from supervised rows. Teacher agreement is not an optimal-action label or learned-policy battle win rate.

## Collection and compute

The 256 fresh teacher games contain 13,015 accepted rows, 2 rejected attempts, 256 natural terminals and 0 cutoffs. The frozen training corpus retains all 106,149 rows. New exact-team counts: 512; training teams excluded: 4096; prior development teams excluded: 1408; disallowed overlap: 0.

Measured production: 68.17 seconds on local CPU, peak parent RSS 2487.06 MiB. Probe collection: 8.34 seconds; updates including compilation: 20.16 seconds. Timing excludes initial predecessor admission/source snapshot and subsequent verification; component times exclude some diagnostics/I/O. No GPU/Prime scaling inference or compiler-event count is claimed.

## All seed gates

| Seed | Gate | Failed checks | Known contexts at 2048 / 4096 / 8192 |
| --- | --- | --- | --- |
| 7103 | FAIL | voluntary_accuracy, voluntary_gain | PASS / PASS / PASS |
| 8209 | FAIL | voluntary_accuracy, voluntary_gain | PASS / PASS / PASS |
| 9311 | FAIL | voluntary_accuracy, voluntary_gain | PASS / PASS / PASS |

## Complete learning curves

Every metric below is equally weighted by game within the action family. Decision-weighted metrics and total metrics are retained in the report. Training and probe use different populations.

| Seed | Step | Family | Train accuracy | Train target mass | Train NLL | Probe accuracy | Probe target mass | Probe NLL |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7103 | 2048 | normal_move | 38.48% | 26.73% | 1.5426 | 35.84% | 25.90% | 1.5843 |
| 7103 | 2048 | tera_move | 52.03% | 25.80% | 1.5312 | 35.12% | 20.13% | 1.8133 |
| 7103 | 2048 | voluntary_switch | 16.51% | 13.89% | 2.2884 | 8.12% | 8.85% | 2.8347 |
| 7103 | 2048 | forced_switch | 44.53% | 42.61% | 1.0593 | 42.69% | 41.95% | 1.0865 |
| 7103 | 4096 | normal_move | 39.07% | 28.86% | 1.4943 | 35.32% | 27.37% | 1.5728 |
| 7103 | 4096 | tera_move | 66.90% | 36.87% | 1.2018 | 37.60% | 25.24% | 1.7279 |
| 7103 | 4096 | voluntary_switch | 37.63% | 22.42% | 1.8262 | 11.40% | 10.47% | 2.8588 |
| 7103 | 4096 | forced_switch | 46.25% | 43.20% | 1.0606 | 43.80% | 42.15% | 1.1029 |
| 7103 | 8192 | normal_move | 42.27% | 32.22% | 1.4025 | 36.79% | 29.68% | 1.5337 |
| 7103 | 8192 | tera_move | 74.32% | 45.84% | 0.9825 | 38.22% | 26.56% | 1.8608 |
| 7103 | 8192 | voluntary_switch | 62.92% | 37.28% | 1.2890 | 12.96% | 11.31% | 3.0957 |
| 7103 | 8192 | forced_switch | 47.73% | 44.33% | 1.0635 | 44.01% | 42.52% | 1.1358 |
| 8209 | 2048 | normal_move | 37.72% | 26.60% | 1.5413 | 36.18% | 25.83% | 1.5828 |
| 8209 | 2048 | tera_move | 52.05% | 25.51% | 1.5333 | 31.82% | 19.45% | 1.8315 |
| 8209 | 2048 | voluntary_switch | 16.09% | 14.02% | 2.3160 | 6.27% | 8.56% | 2.8225 |
| 8209 | 2048 | forced_switch | 44.91% | 42.63% | 1.0541 | 42.83% | 41.78% | 1.0877 |
| 8209 | 4096 | normal_move | 38.83% | 28.80% | 1.4918 | 35.08% | 27.24% | 1.5748 |
| 8209 | 4096 | tera_move | 65.05% | 35.78% | 1.2320 | 39.46% | 24.26% | 1.7514 |
| 8209 | 4096 | voluntary_switch | 37.56% | 22.81% | 1.8172 | 9.12% | 9.79% | 2.8944 |
| 8209 | 4096 | forced_switch | 46.61% | 43.22% | 1.0556 | 42.76% | 42.09% | 1.1010 |
| 8209 | 8192 | normal_move | 42.38% | 32.13% | 1.3966 | 36.41% | 29.53% | 1.5296 |
| 8209 | 8192 | tera_move | 74.70% | 45.86% | 0.9748 | 39.26% | 26.70% | 1.8289 |
| 8209 | 8192 | voluntary_switch | 59.47% | 35.53% | 1.3413 | 9.12% | 10.01% | 3.2078 |
| 8209 | 8192 | forced_switch | 48.10% | 44.24% | 1.0530 | 43.00% | 42.23% | 1.1353 |
| 9311 | 2048 | normal_move | 39.49% | 26.66% | 1.5279 | 37.79% | 26.02% | 1.5619 |
| 9311 | 2048 | tera_move | 46.05% | 22.72% | 1.6383 | 30.37% | 17.87% | 1.8934 |
| 9311 | 2048 | voluntary_switch | 14.67% | 13.03% | 2.3089 | 3.85% | 7.77% | 2.8713 |
| 9311 | 2048 | forced_switch | 44.92% | 42.66% | 1.0524 | 41.71% | 41.80% | 1.0814 |
| 9311 | 4096 | normal_move | 40.88% | 29.87% | 1.4451 | 37.70% | 28.42% | 1.5166 |
| 9311 | 4096 | tera_move | 59.86% | 32.71% | 1.3328 | 33.47% | 22.33% | 1.8377 |
| 9311 | 4096 | voluntary_switch | 34.65% | 21.36% | 1.8909 | 7.98% | 9.33% | 2.9653 |
| 9311 | 4096 | forced_switch | 46.35% | 43.36% | 1.0569 | 43.22% | 42.21% | 1.1011 |
| 9311 | 8192 | normal_move | 41.88% | 31.83% | 1.4052 | 36.30% | 29.37% | 1.5357 |
| 9311 | 8192 | tera_move | 72.98% | 44.18% | 1.0243 | 36.98% | 26.37% | 1.8607 |
| 9311 | 8192 | voluntary_switch | 63.74% | 36.81% | 1.2831 | 8.40% | 10.11% | 3.1729 |
| 9311 | 8192 | forced_switch | 48.33% | 44.47% | 1.0548 | 44.41% | 42.51% | 1.1334 |

## All 12 primary contrasts

| Seed | Family | Games | Rows | 8192 minus 2048 | Adjusted bootstrap interval | Hoeffding interval |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 7103 | normal_move | 256 | 10127 | +0.95 pp | [-0.89, +2.61] pp | [-21.01, +22.91] pp |
| 7103 | tera_move | 242 | 383 | +3.10 pp | [-4.75, +10.54] pp | [-19.49, +25.69] pp |
| 7103 | voluntary_switch | 117 | 150 | +4.84 pp | [-1.65, +12.25] pp | [-27.64, +37.33] pp |
| 7103 | forced_switch | 256 | 2355 | +1.32 pp | [-1.33, +3.95] pp | [-20.64, +23.28] pp |
| 8209 | normal_move | 256 | 10127 | +0.23 pp | [-1.66, +2.08] pp | [-21.73, +22.20] pp |
| 8209 | tera_move | 242 | 383 | +7.44 pp | [+0.00, +14.46] pp | [-15.15, +30.03] pp |
| 8209 | voluntary_switch | 117 | 150 | +2.85 pp | [-3.50, +10.20] pp | [-29.64, +35.34] pp |
| 8209 | forced_switch | 256 | 2355 | +0.17 pp | [-2.53, +2.93] pp | [-21.80, +22.13] pp |
| 9311 | normal_move | 256 | 10127 | -1.49 pp | [-3.51, +0.51] pp | [-23.46, +20.47] pp |
| 9311 | tera_move | 242 | 383 | +6.61 pp | [-1.24, +14.17] pp | [-15.98, +29.20] pp |
| 9311 | voluntary_switch | 117 | 150 | +4.56 pp | [-1.00, +11.40] pp | [-27.93, +37.04] pp |
| 9311 | forced_switch | 256 | 2355 | +2.70 pp | [-0.0048, +5.3446] pp | [-19.26, +24.66] pp |

The 5,000-resample whole-game paired bootstrap is Bonferroni-adjusted across all 12 seed/family primary contrasts; its finite-sample coverage is approximate. Conservative Hoeffding intervals assume independent sampled games. Per-game IDs and differences are preserved. The descriptive operating gate is separate from these intervals and from any battle-strength gate. No inference is assigned to the intermediate 4,096 endpoint.

## Unique exposure and repetitions

| Seed | Step | Family | Unique rows seen / total | Draws | Minimum / maximum repetitions |
| --- | ---: | --- | ---: | ---: | --- |
| 7103 | 2048 | normal_move | 43563 / 82651 | 65536 | 0 / 7 |
| 7103 | 2048 | tera_move | 3206 / 3207 | 32768 | 0 / 30 |
| 7103 | 2048 | voluntary_switch | 1417 / 1422 | 16384 | 0 / 29 |
| 7103 | 2048 | forced_switch | 10914 / 18869 | 16384 | 0 / 7 |
| 7103 | 2048 | rehearsal | 128 / 128 | 131072 | 931 / 1117 |
| 7103 | 4096 | normal_move | 62263 / 82651 | 131072 | 0 / 12 |
| 7103 | 4096 | tera_move | 3207 / 3207 | 65536 | 3 / 55 |
| 7103 | 4096 | voluntary_switch | 1422 / 1422 | 32768 | 1 / 52 |
| 7103 | 4096 | forced_switch | 15466 / 18869 | 32768 | 0 / 10 |
| 7103 | 4096 | rehearsal | 128 / 128 | 262144 | 1929 / 2178 |
| 7103 | 8192 | normal_move | 75574 / 82651 | 262144 | 0 / 18 |
| 7103 | 8192 | tera_move | 3207 / 3207 | 131072 | 16 / 93 |
| 7103 | 8192 | voluntary_switch | 1422 / 1422 | 65536 | 5 / 101 |
| 7103 | 8192 | forced_switch | 18179 / 18869 | 65536 | 0 / 17 |
| 7103 | 8192 | rehearsal | 128 / 128 | 524288 | 3950 / 4251 |
| 8209 | 2048 | normal_move | 43411 / 82651 | 65536 | 0 / 8 |
| 8209 | 2048 | tera_move | 3207 / 3207 | 32768 | 1 / 37 |
| 8209 | 2048 | voluntary_switch | 1418 / 1422 | 16384 | 0 / 28 |
| 8209 | 2048 | forced_switch | 10886 / 18869 | 16384 | 0 / 7 |
| 8209 | 2048 | rehearsal | 128 / 128 | 131072 | 923 / 1104 |
| 8209 | 4096 | normal_move | 62116 / 82651 | 131072 | 0 / 11 |
| 8209 | 4096 | tera_move | 3207 / 3207 | 65536 | 5 / 55 |
| 8209 | 4096 | voluntary_switch | 1422 / 1422 | 32768 | 3 / 53 |
| 8209 | 4096 | forced_switch | 15401 / 18869 | 32768 | 0 / 9 |
| 8209 | 4096 | rehearsal | 128 / 128 | 262144 | 1949 / 2190 |
| 8209 | 8192 | normal_move | 75544 / 82651 | 262144 | 0 / 17 |
| 8209 | 8192 | tera_move | 3207 / 3207 | 131072 | 17 / 92 |
| 8209 | 8192 | voluntary_switch | 1422 / 1422 | 65536 | 6 / 95 |
| 8209 | 8192 | forced_switch | 18208 / 18869 | 65536 | 0 / 14 |
| 8209 | 8192 | rehearsal | 128 / 128 | 524288 | 3938 / 4268 |
| 9311 | 2048 | normal_move | 43306 / 82651 | 65536 | 0 / 8 |
| 9311 | 2048 | tera_move | 3206 / 3207 | 32768 | 0 / 33 |
| 9311 | 2048 | voluntary_switch | 1415 / 1422 | 16384 | 0 / 35 |
| 9311 | 2048 | forced_switch | 10830 / 18869 | 16384 | 0 / 7 |
| 9311 | 2048 | rehearsal | 128 / 128 | 131072 | 929 / 1114 |
| 9311 | 4096 | normal_move | 61888 / 82651 | 131072 | 0 / 11 |
| 9311 | 4096 | tera_move | 3207 / 3207 | 65536 | 5 / 52 |
| 9311 | 4096 | voluntary_switch | 1422 / 1422 | 32768 | 1 / 55 |
| 9311 | 4096 | forced_switch | 15389 / 18869 | 32768 | 0 / 11 |
| 9311 | 4096 | rehearsal | 128 / 128 | 262144 | 1935 / 2163 |
| 9311 | 8192 | normal_move | 75519 / 82651 | 262144 | 0 / 16 |
| 9311 | 8192 | tera_move | 3207 / 3207 | 131072 | 17 / 95 |
| 9311 | 8192 | voluntary_switch | 1422 / 1422 | 65536 | 6 / 99 |
| 9311 | 8192 | forced_switch | 18163 / 18869 | 65536 | 0 / 15 |
| 9311 | 8192 | rehearsal | 128 / 128 | 524288 | 3917 / 4273 |

Exposure is cumulative from the original F18 update indices. Duplicate sampled indices count with multiplicity. Rehearsal remains the same 128 rows; no probe example enters training.

## Verification and remaining scope

All 16 focused tests pass on the final source/runtime inventory: eight exposure checks, four scope/endpoint checks and four continuation checks. Pinned Showdown build passed. Nine actual-corpus sampler draws match the sealed legacy sampler. Fresh independent replay regenerates all 256 probe games, all 18,432 new updates, all input arrays and every train/probe/tactical endpoint prediction. Fifteen checkpoint comparisons and six recoveries of the original next update pass. All three 2,048 baselines reproduce the F18 train probabilities, metrics and tactical diagnostics exactly. Production, verification and workflow exits are zero. Fifteen copied-evidence corruptions reject. Value-head weights remain unchanged; the shared trunk changes, so value predictions may change. No critic-calibration or win-rate benefit is claimed.

Five current documents are bound before sealing. Independent live/archive/source/evidence hash checks and identical repeat sealing are recorded outside the sealed review directory. The F10 policy remains the reference and full G2/G3/G4 remain open. No final reserved-data use, public server, Prime or GitHub publication.

[Completion receipt](../artifacts/reports/f19-learning-curve.json).

Production report SHA256: `f85d73a6ab15e942089d39f50344db4cb31c36ccd281cb37175084cea8a7c4c1`.

## Preserved prelaunch design


Design recorded before production; all seeds, budgets and gates were retained.

F18 expanded training from 512 to 2,048 teacher games but reached only 14.7–16.5% voluntary-switch agreement on the larger training corpus at 2,048 updates. F19 tests whether a larger fixed optimization budget improves fitting and transfer. It does not assume that more fitting will improve generalization.

For seeds 7103, 8209 and 9311, resume each sealed F18 expanded checkpoint at update 2,048 with its exact parameters, Adam moments and counter. Reuse the frozen 106,149 accepted training rows, the original 928-feature model, unchanged NLL/action distribution/Adam settings, 32/16/8/8 teacher family sampling and 64 old rehearsal rows. Continue the original sampling indices from 2,048 through 8,191. No reset, retraining from initialization or new training corpus.

Report fixed endpoints 2,048, 4,096 and 8,192. The primary comparison is 8,192 versus 2,048 across all three seeds and four action families (12 contrasts); 4,096 is descriptive and cannot be selected for the gate. Added budget: 18,432 updates and 2,359,296 row presentations. Accumulated budget includes the 6,144 sealed predecessor updates, giving 24,576 total across three seeds. No early stopping, resampling or post-result extension.

Collect 256 fresh development teacher games, seed 1025261901, in the existing development partition. Their exact teams must be disjoint from all training and F15/F16/F18 development teams. F17 reused F16 games. Preserve all natural outcomes, cutoff prefixes and rejected-attempt evidence; only accepted choices become diagnostic rows. Probe labels and outcomes never enter training; the existing teacher labels define the fixed family quotas. No reserved final evaluation data or learned-policy battle-strength evaluation.

Report fit and fresh-probe accuracy, target mass and NLL for every family at every endpoint; known-context retention; paired whole-game uncertainty on the primary comparison; and counts of unique training examples seen and repetitions at each endpoint. Both sides stay in one game cluster. Use the existing 5,000-resample approximate Bonferroni bootstrap and conservative Hoeffding bounds for all 12 primary contrasts. Intermediate results receive no confidence claims.

Keep the descriptive gate fixed: final voluntary-switch game-weighted agreement at least 25%, improvement over step 2,048 at least 10 percentage points, no other action-family drop greater than 5 points, and retention of all 32 familiar tactical contexts. All three seeds must pass. Minimum probe support is unchanged (Tera 64 rows/32 games; voluntary switches 32 rows/16 games). Shortfall or production failure preserves all evidence and stops without a rerun.

Freeze source/runtime/plan after checks and review. Replay all 256 new games, all 18,432 new updates and all endpoint probabilities, compare 15 checkpoints (steps 2,048, 4,095, 4,096, 8,191 and 8,192 per seed) and recover six original next updates (indices 4,095 and 8,191, producing the 4,096 and 8,192 endpoints). Verify baseline training probabilities/tactical diagnostics against F18, strict checkpoint metadata and non-reset optimizer state. Update only current progress documents and the new F19 handoff before sealing; preserve all predecessor evidence and historical handoffs. F10 remains the reference; full G2/G3/G4 and public strength remain open. Prime is unnecessary and GitHub publication remains held.

