# F20 — shared action-scoring comparison

Status: experiment completed; descriptive architecture gate 0/3 seeds.

F20 completed (2026-10-25): trained fresh flat and shared action-scoring models on the same frozen 2,048-game corpus, with identical sampled batches and 8,192 updates per arm/seed. All three seeds completed 49,152 updates / 6,291,456 row presentations in 161.97 seconds on CPU. Collected 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,087 attempted decisions). Final voluntary-switch training agreement was 58.10–63.13% for flat and 21.12–23.47% for shared; fresh-probe agreement was 11.67–13.39% and 21.41–25.50%. The descriptive architecture gate passed in 0/3 seeds. All 28 focused tests, exhaustive replay, 30 checkpoint comparisons and 12 original-next-update recoveries passed. These are teacher-imitation results, not battle win rates. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer.

The shared architecture has higher final fresh-probe point agreement in all 12 seed/action-family comparisons, despite generally lower training fit. Voluntary-switch agreement is 21.41–25.50% for shared versus 11.67–13.39% for flat; Tera agreement is 55.02–57.53% versus 35.36–36.19%. Normal-move agreement is 39.70–40.76% versus 35.87–36.89%, and forced-switch agreement is 47.15–47.82% versus 43.67–44.99%. All 12 shared probe NLLs are lower than their matched flat NLLs. Under the preregistered approximate adjusted bootstrap there are 9 improvement contrasts, 0 regressions and 3 inconclusive contrasts. All 12 conservative Hoeffding intervals include zero. Both interval procedures use Bonferroni adjustment across the 12 primary contrasts (family-wise alpha 0.05). This is encouraging development evidence for the selected configuration; it is not a claim of universal architecture superiority, causal explanation of F19, or stronger battle play.

The all-seed descriptive architecture gate has 0/3 passing seeds. Seed 7103 fails both switching thresholds; seed 8209 passes the switching thresholds but fails familiar-context mastery; seed 9311 fails both switching thresholds and familiar mastery. All other action-family preservation checks pass. Shared mastery passed in every seed at step 2,048 but fails at the preregistered final step for seeds 8209 and 9311; the earlier endpoint is not selected as a replacement. The final flat models pass familiar mastery in all three seeds.

The known-context failures are substantive greedy-action changes, not merely close probability thresholds. In each affected seed, variant contexts 12–15 (64 oracle rows) choose Tera action 4 instead of oracle target ordinary action 0. Mean target mass is about 27.5% for seed 8209 and 48.5% for seed 9311, with 0% greedy correctness. A read-only exact-input check confirms these variant rows do not occur in the 128-row rehearsal batch. They are familiar, previously exposed development variants of rehearsed skills; they are not an untouched final benchmark or simple identical-row memorization tests. The oracle is a fixed tactical diagnostic, not a proof of the best full-battle action. No new data or training was used for this failure audit.

Production measured 161.97 seconds on CPU. Shared update computation including compilation took 60.65 seconds versus 19.35 seconds for flat, about 3.13 times as long at equal update/row budgets. The experiment does not match FLOPs or optimize each architecture's hyperparameters. Value heads remain untrained and no critic claim is made. No policy is promoted over F10; final reserved data remain unused.

Next proposed step: preregister a full-battle development evaluation of these frozen 8,192-update flat and shared checkpoints, alongside the existing F10 reference, against fixed random, base-power and tactical opponents. Keep all three seeds and identical paired battle schedules, report wins/losses and unresolved cutoffs separately, and carry the known Tera-variant failures into the assessment. Make no additional optimizer updates or checkpoint selection during this screen. It will test whether better teacher imitation translates into battle strength before choosing another training intervention; F20 does not establish that translation or authorize promotion. Local CPU remains sufficient for a bounded evaluation; no public ladder or final reserved evaluation.

## Design and interpretation limits

Both models start fresh with zero Adam moments and receive exactly the same 24,576 selected batches across three seeds. Each batch is applied once to each architecture. Neither inherits F10 or F19 weights. The 8,192 endpoint is the only primary comparison; 0 and 2,048 are descriptive. Training corpus, 928 public features, action law, teacher quotas, rehearsal and optimizer settings stay fixed. The shared architecture changes pooling, parameter sharing and depth; initialization functions and FLOPs are not matched. Total parameters are 60,431 flat versus 60,479 shared; policy-only counts are 60,366 versus 60,318. This tests the selected configurations at a fixed update budget, not all possible architectures or a proven cause of F19’s transfer gap.

The 6,291,456 presentations comprise 3,145,728 teacher and 3,145,728 rehearsal draws. Half the loss remains on 128 heavily repeated rehearsal rows by design. The familiar 32-context mastery check includes previously exposed variants of rehearsed skills; not every variant is an exact rehearsal input. It is not an untouched benchmark or pretrained retention test. Only final shared mastery enters the gate. No model-specific learning-rate or clipping tuning occurred.

Fresh exact teams are excluded from all training and F15/F16/F18/F19 development roles; F17 reused F16 games. Known species, mechanics and the heuristic teacher recur. Teacher labels are demonstrations, not proven optimal actions. No learned-policy battle-strength evaluation or final reserved-data use occurred. All accepted choices, including any cutoff prefixes, are included; rejected attempts stay in transcripts and do not become diagnostic rows.

## Collection and compute

Probe: 256 games, 13,087 accepted rows, 0 rejected attempts, 256 natural terminals and 0 cutoffs. Exact teams: 512 new; 4096 training and 1920 prior development excluded; disallowed overlap 0. Frozen training retains 106,149 rows from 2,048 games, plus 128 rehearsal rows.

Measured production: 161.97 seconds on local CPU, peak parent RSS 2567.73 MiB. Probe collection 8.66 seconds. Update computation including compilation: flat 19.35 seconds; shared 60.65 seconds. This excludes initial predecessor admission/source snapshot and subsequent verification. Shared and flat receive equal updates/presentations, not equal FLOPs or wall time. Component timings exclude some selection, diagnostics and I/O and do not sum to total wall time.

## Every seed and gate

| Seed | Gate | Failed checks | Flat mastery at 0 / 2048 / 8192 | Shared mastery at 0 / 2048 / 8192 |
| --- | --- | --- | --- | --- |
| 7103 | FAIL | voluntary_accuracy, voluntary_gain | FAIL / PASS / PASS | FAIL / PASS / PASS |
| 8209 | FAIL | known_contexts_mastered | FAIL / PASS / PASS | FAIL / PASS / FAIL |
| 9311 | FAIL | known_contexts_mastered, voluntary_accuracy, voluntary_gain | FAIL / PASS / PASS | FAIL / PASS / FAIL |

## All action families and endpoints

Accuracy, target probability mass and NLL are equally weighted by game within family. The report also retains decision-weighted and total metrics.

| Seed | Arm | Step | Family | Fit accuracy | Fit mass | Fit NLL | Probe accuracy | Probe mass | Probe NLL |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7103 | flat | 0 | normal_move | 19.72% | 15.69% | 1.9883 | 19.90% | 15.87% | 1.9842 |
| 7103 | flat | 0 | tera_move | 11.71% | 9.87% | 2.3626 | 13.39% | 9.88% | 2.3665 |
| 7103 | flat | 0 | voluntary_switch | 7.72% | 12.16% | 2.1958 | 14.47% | 12.87% | 2.1514 |
| 7103 | flat | 0 | forced_switch | 41.50% | 41.71% | 1.0358 | 39.52% | 41.30% | 1.0445 |
| 7103 | flat | 2048 | normal_move | 39.08% | 27.57% | 1.5291 | 35.65% | 26.72% | 1.5747 |
| 7103 | flat | 2048 | tera_move | 53.21% | 27.79% | 1.4721 | 33.47% | 20.93% | 1.8013 |
| 7103 | flat | 2048 | voluntary_switch | 17.95% | 14.05% | 2.2642 | 5.20% | 8.52% | 2.8287 |
| 7103 | flat | 2048 | forced_switch | 44.38% | 42.53% | 1.0621 | 42.45% | 41.65% | 1.0946 |
| 7103 | flat | 8192 | normal_move | 41.73% | 32.13% | 1.4144 | 35.87% | 29.99% | 1.5313 |
| 7103 | flat | 8192 | tera_move | 73.75% | 45.66% | 0.9921 | 35.98% | 25.27% | 1.8949 |
| 7103 | flat | 8192 | voluntary_switch | 62.11% | 37.07% | 1.3041 | 13.39% | 11.50% | 3.1479 |
| 7103 | flat | 8192 | forced_switch | 47.31% | 44.16% | 1.0688 | 44.04% | 42.16% | 1.1559 |
| 7103 | shared | 0 | normal_move | 14.30% | 15.57% | 1.9968 | 14.77% | 15.73% | 1.9928 |
| 7103 | shared | 0 | tera_move | 26.68% | 9.90% | 2.3586 | 26.78% | 9.87% | 2.3646 |
| 7103 | shared | 0 | voluntary_switch | 0.00% | 12.19% | 2.1911 | 0.00% | 12.87% | 2.1464 |
| 7103 | shared | 0 | forced_switch | 41.32% | 41.72% | 1.0346 | 41.14% | 41.35% | 1.0414 |
| 7103 | shared | 2048 | normal_move | 38.54% | 29.83% | 1.4399 | 39.80% | 30.08% | 1.4376 |
| 7103 | shared | 2048 | tera_move | 47.46% | 26.17% | 1.5598 | 46.23% | 25.14% | 1.6217 |
| 7103 | shared | 2048 | voluntary_switch | 9.18% | 12.55% | 2.3336 | 9.70% | 12.63% | 2.4115 |
| 7103 | shared | 2048 | forced_switch | 45.66% | 42.19% | 1.0342 | 44.89% | 41.86% | 1.0375 |
| 7103 | shared | 8192 | normal_move | 40.75% | 32.31% | 1.4069 | 40.61% | 32.27% | 1.4184 |
| 7103 | shared | 8192 | tera_move | 64.05% | 38.08% | 1.2084 | 57.32% | 35.47% | 1.3220 |
| 7103 | shared | 8192 | voluntary_switch | 22.87% | 18.72% | 2.0158 | 21.41% | 18.68% | 2.2040 |
| 7103 | shared | 8192 | forced_switch | 48.10% | 43.06% | 1.0113 | 47.18% | 42.56% | 1.0192 |
| 8209 | flat | 0 | normal_move | 15.01% | 15.50% | 2.0014 | 14.63% | 15.64% | 1.9982 |
| 8209 | flat | 0 | tera_move | 6.37% | 9.69% | 2.3813 | 6.28% | 9.65% | 2.3885 |
| 8209 | flat | 0 | voluntary_switch | 13.61% | 12.42% | 2.1729 | 15.16% | 13.18% | 2.1245 |
| 8209 | flat | 0 | forced_switch | 41.30% | 41.71% | 1.0356 | 41.67% | 41.37% | 1.0417 |
| 8209 | flat | 2048 | normal_move | 39.62% | 27.98% | 1.5065 | 37.17% | 27.35% | 1.5468 |
| 8209 | flat | 2048 | tera_move | 53.00% | 27.77% | 1.4701 | 31.80% | 21.35% | 1.7886 |
| 8209 | flat | 2048 | voluntary_switch | 17.62% | 13.96% | 2.3202 | 2.61% | 8.52% | 2.8650 |
| 8209 | flat | 2048 | forced_switch | 44.79% | 42.62% | 1.0615 | 42.76% | 41.82% | 1.0871 |
| 8209 | flat | 8192 | normal_move | 42.11% | 32.49% | 1.3922 | 36.71% | 30.62% | 1.4943 |
| 8209 | flat | 8192 | tera_move | 73.34% | 45.53% | 0.9940 | 35.36% | 25.16% | 1.9209 |
| 8209 | flat | 8192 | voluntary_switch | 58.10% | 34.66% | 1.3875 | 12.10% | 10.64% | 3.2233 |
| 8209 | flat | 8192 | forced_switch | 47.60% | 44.09% | 1.0710 | 43.67% | 42.29% | 1.1375 |
| 8209 | shared | 0 | normal_move | 3.63% | 15.47% | 2.0046 | 3.56% | 15.62% | 2.0008 |
| 8209 | shared | 0 | tera_move | 21.01% | 9.91% | 2.3578 | 21.34% | 9.88% | 2.3641 |
| 8209 | shared | 0 | voluntary_switch | 10.42% | 12.29% | 2.1827 | 14.18% | 12.99% | 2.1374 |
| 8209 | shared | 0 | forced_switch | 41.15% | 41.72% | 1.0347 | 39.82% | 41.35% | 1.0417 |
| 8209 | shared | 2048 | normal_move | 37.55% | 30.43% | 1.4568 | 39.50% | 30.74% | 1.4530 |
| 8209 | shared | 2048 | tera_move | 50.13% | 28.30% | 1.4984 | 47.28% | 27.01% | 1.5677 |
| 8209 | shared | 2048 | voluntary_switch | 9.17% | 12.50% | 2.4072 | 9.34% | 12.73% | 2.4878 |
| 8209 | shared | 2048 | forced_switch | 46.00% | 42.24% | 1.0322 | 44.62% | 41.81% | 1.0388 |
| 8209 | shared | 8192 | normal_move | 39.03% | 31.40% | 1.4534 | 39.70% | 31.57% | 1.4559 |
| 8209 | shared | 8192 | tera_move | 65.25% | 38.92% | 1.1583 | 57.53% | 36.08% | 1.2997 |
| 8209 | shared | 8192 | voluntary_switch | 23.47% | 19.05% | 1.9926 | 25.50% | 19.19% | 2.1300 |
| 8209 | shared | 8192 | forced_switch | 48.06% | 43.17% | 1.0091 | 47.82% | 42.74% | 1.0166 |
| 9311 | flat | 0 | normal_move | 16.95% | 15.56% | 1.9953 | 18.11% | 15.73% | 1.9906 |
| 9311 | flat | 0 | tera_move | 2.26% | 9.52% | 2.3981 | 1.26% | 9.48% | 2.4054 |
| 9311 | flat | 0 | voluntary_switch | 14.27% | 12.42% | 2.1716 | 18.97% | 13.16% | 2.1259 |
| 9311 | flat | 0 | forced_switch | 41.90% | 41.73% | 1.0350 | 40.61% | 41.33% | 1.0425 |
| 9311 | flat | 2048 | normal_move | 39.98% | 28.04% | 1.4986 | 37.59% | 27.36% | 1.5383 |
| 9311 | flat | 2048 | tera_move | 50.26% | 26.50% | 1.5192 | 33.89% | 20.80% | 1.8134 |
| 9311 | flat | 2048 | voluntary_switch | 17.66% | 14.17% | 2.2635 | 2.87% | 8.92% | 2.7683 |
| 9311 | flat | 2048 | forced_switch | 44.45% | 42.68% | 1.0607 | 43.01% | 42.03% | 1.0800 |
| 9311 | flat | 8192 | normal_move | 41.93% | 32.09% | 1.4086 | 36.89% | 30.16% | 1.5162 |
| 9311 | flat | 8192 | tera_move | 72.57% | 45.18% | 1.0142 | 36.19% | 26.19% | 1.8924 |
| 9311 | flat | 8192 | voluntary_switch | 63.13% | 37.10% | 1.2971 | 11.67% | 10.95% | 3.1686 |
| 9311 | flat | 8192 | forced_switch | 47.61% | 44.22% | 1.0714 | 44.99% | 42.62% | 1.1343 |
| 9311 | shared | 0 | normal_move | 20.86% | 15.63% | 1.9925 | 20.79% | 15.79% | 1.9884 |
| 9311 | shared | 0 | tera_move | 33.77% | 9.95% | 2.3539 | 32.85% | 9.92% | 2.3597 |
| 9311 | shared | 0 | voluntary_switch | 0.00% | 12.12% | 2.1971 | 0.00% | 12.80% | 2.1524 |
| 9311 | shared | 0 | forced_switch | 41.14% | 41.72% | 1.0346 | 40.89% | 41.35% | 1.0414 |
| 9311 | shared | 2048 | normal_move | 39.59% | 30.45% | 1.4095 | 41.44% | 30.63% | 1.4091 |
| 9311 | shared | 2048 | tera_move | 45.74% | 24.88% | 1.6011 | 43.51% | 23.97% | 1.6537 |
| 9311 | shared | 2048 | voluntary_switch | 4.99% | 10.77% | 2.4829 | 5.17% | 10.72% | 2.5597 |
| 9311 | shared | 2048 | forced_switch | 46.00% | 42.15% | 1.0341 | 44.51% | 41.79% | 1.0394 |
| 9311 | shared | 8192 | normal_move | 40.18% | 33.23% | 1.3991 | 40.76% | 33.42% | 1.3999 |
| 9311 | shared | 8192 | tera_move | 61.40% | 37.07% | 1.2440 | 55.02% | 34.61% | 1.3712 |
| 9311 | shared | 8192 | voluntary_switch | 21.12% | 18.16% | 2.0695 | 21.41% | 17.82% | 2.2965 |
| 9311 | shared | 8192 | forced_switch | 48.13% | 43.20% | 1.0073 | 47.15% | 42.82% | 1.0131 |

## All 12 primary shared-minus-flat contrasts

| Seed | Family | Games | Rows | Difference | Adjusted bootstrap interval | Hoeffding interval |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 7103 | normal_move | 256 | 10187 | +4.74 pp | [+2.5328, +6.9541] pp | [-17.22, +26.71] pp |
| 7103 | tera_move | 239 | 398 | +21.34 pp | [+12.3431, +30.1255] pp | [-1.39, +44.07] pp |
| 7103 | voluntary_switch | 116 | 169 | +8.02 pp | [-3.3326, +20.2961] pp | [-24.61, +40.64] pp |
| 7103 | forced_switch | 256 | 2333 | +3.13 pp | [+0.1193, +6.4992] pp | [-18.83, +25.10] pp |
| 8209 | normal_move | 256 | 10187 | +2.99 pp | [+0.7923, +5.1822] pp | [-18.97, +24.95] pp |
| 8209 | tera_move | 239 | 398 | +22.18 pp | [+13.8075, +30.7531] pp | [-0.55, +44.91] pp |
| 8209 | voluntary_switch | 116 | 169 | +13.41 pp | [+2.6041, +25.2348] pp | [-19.22, +46.03] pp |
| 8209 | forced_switch | 256 | 2333 | +4.15 pp | [+0.9351, +7.3983] pp | [-17.81, +26.11] pp |
| 9311 | normal_move | 256 | 10187 | +3.87 pp | [+1.7952, +5.9226] pp | [-18.09, +25.83] pp |
| 9311 | tera_move | 239 | 398 | +18.83 pp | [+10.5470, +26.3598] pp | [-3.90, +41.56] pp |
| 9311 | voluntary_switch | 116 | 169 | +9.74 pp | [-0.8358, +19.8949] pp | [-22.88, +42.37] pp |
| 9311 | forced_switch | 256 | 2333 | +2.16 pp | [-0.9657, +5.4626] pp | [-19.80, +24.12] pp |

Intervals use 5,000 whole-game paired bootstrap resamples with Bonferroni adjustment across 12 primary contrasts; coverage is approximate. Conservative Hoeffding bounds also use Bonferroni adjustment across the 12 primary contrasts at family-wise alpha 0.05 and assume independent sampled games. Both sides stay in the game cluster. Per-game IDs/differences are retained. The support floor is not a power guarantee; a descriptive gate can pass while an interval is inconclusive. No significance or equivalence follows from a point threshold alone.

## Matched cumulative exposure

Both arms have exactly these same selected rows/counts in each seed. Duplicate indices count with multiplicity.

| Seed | Step | Family | Unique rows seen / total | Draws per arm | Min / max repetitions |
| --- | ---: | --- | ---: | ---: | --- |
| 7103 | 2048 | normal_move | 43563 / 82651 | 65536 | 0 / 7 |
| 7103 | 2048 | tera_move | 3206 / 3207 | 32768 | 0 / 30 |
| 7103 | 2048 | voluntary_switch | 1417 / 1422 | 16384 | 0 / 29 |
| 7103 | 2048 | forced_switch | 10914 / 18869 | 16384 | 0 / 7 |
| 7103 | 2048 | rehearsal | 128 / 128 | 131072 | 931 / 1117 |
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
| 9311 | 8192 | normal_move | 75519 / 82651 | 262144 | 0 / 16 |
| 9311 | 8192 | tera_move | 3207 / 3207 | 131072 | 17 / 95 |
| 9311 | 8192 | voluntary_switch | 1422 / 1422 | 65536 | 6 / 99 |
| 9311 | 8192 | forced_switch | 18163 / 18869 | 65536 | 0 / 15 |
| 9311 | 8192 | rehearsal | 128 / 128 | 524288 | 3917 / 4273 |

## Verification

All 28 focused tests pass on the final source/runtime tree: 12 shared-network checks, 8 learning/checkpoint checks and 8 scope/remapping/information checks. Pinned Showdown build passed. The flat non-invariance positive control confirms slot-remapping tests are non-vacuous. Shared logits/value and legal probabilities obey remapping within 1e-5 FP32 tolerance; the finite-word tree does not promise exact permutation-identical random draws. All 9 shared endpoint train-row symmetry checks passed.

Fresh independent replay regenerates every one of 256 probe games and all 24,576 matched batches / 49,152 updates, all 18 endpoint prediction sets, 30 checkpoints and 12 original next updates. Each seed’s first flat update matches the sealed implementation exactly. All state arrays, reported gradient norms/metrics and optimizer counters are finite and valid; value-head weights remain at zero, so no critic learning/calibration claim is made. Production, verification and workflow exits are zero. Seventeen copied-evidence negatives reject. Every live/archive source/evidence hash and identical repeat receipt are verified outside the sealed review directory after completion documents are bound.

[Completion receipt](../artifacts/reports/f20-shared-policy.json).

Production report SHA256: `ad8ebb341bcebc3cd79b445ea5193f05d9d1c5f14992236173db53438f4e1ff6`.

## Preserved prelaunch design


Design recorded before production; all seeds, budgets and gates retained.

F19 improved switching fit to 59–64% but fresh-probe agreement remained 8–13%, with worse probe log-loss. F20 tests a structural generalization hypothesis. It does not assume architecture caused the gap or that a shared scorer will solve it.

Train two fresh models from deterministic seeds7103/8209/9311, 8,192 updates each (49,152 total), on the same frozen 106,149 teacher rows from 2,048 games and the same128 rehearsal rows. Both use exactly the same batches, teacher quotas32/16/8/8 plus64 rehearsal rows, sampler indices0..8191, masked finite-word action law, NLL and Adam settings. No F10/F19 learned weights initialize either arm: retaining a pretrained flat control would confound this new-architecture comparison. F10 stays the external reference. The 0 and2,048 endpoints are descriptive; only final8,192 shared-minus-flat is primary. No early stopping or checkpoint selection.

The flat model is the existing928->64->14 network (60,431 parameters). The shared model reads exactly the same928 public features, using shared32-wide embeddings of12 Pokémon blocks and4 move blocks, invariant own/opponent means and active summaries, and a shared195->281->1 scorer over14 legal-action candidates. Normal and Tera moves share candidate embeddings with separate kind flags. Switches use destination embeddings. Global context and a candidate feed the same scorer regardless of slot. Padding is masked out of summaries. It has60,479 parameters (48 more,0.08%); policy-only counts60366 and60318. Width281 was chosen by parameter count before any result. Parameter count and update budgets are approximately/exactly matched respectively; FLOPs, initialization functions and depth are not. Measure both arms’ compute, and do not attribute any effect exclusively to one component of the architecture.

Collect256 fresh development teacher games (seed1025262001), excluding all training and F15/F16/F18/F19 development exact teams. F17 reused F16 games. No new training corpus or final reserved data. Preserve all games/outcomes/rejected attempts; only accepted choices enter diagnostic rows. Require existing minimum Tera and voluntary-switch support, without resampling. Same species, mechanics and teacher can recur.

Report all seeds, action families, training/probe accuracy, target mass and NLL at0/2048/8192, exact exposure counts, and absolute mastery of32 familiar tactical contexts (at least95%greedy correctness and60%target mass). This is mastery learned from rehearsal, not retention of a pretrained model. The descriptive final gate requires25%voluntary-switch agreement,10percentage-point advantage over the matched flat arm, no other family drop over5points and all32contexts mastered. All3seeds must pass. Retain all12primary whole-game paired contrasts,5,000-resample adjusted bootstrap and conservative Hoeffding intervals; seed1025262002. No win-rate or optimal-action claim follows from teacher agreement.

The fixed rehearsal share deliberately preserves the previous training objective while architecture changes. Total presentations comprise 3,145,728 teacher and 3,145,728 rehearsal draws; these include heavy repetition of the same 128 rehearsal rows. Familiar-context results are a rehearsed diagnostic, not unseen-context competence. Only the final shared model must pass the mastery gate in each seed; the flat model and earlier endpoints are reported descriptively. This is a comparison of sample efficiency at the fixed budget, not a causal diagnosis of F19. The inherited learning rate, gradient clipping and initialization scale may favor one architecture; neither arm receives per-architecture tuning. Parameter matching does not match FLOPs.

The support floor is an admission threshold, not a power guarantee. At the minimum 16 voluntary-switch games, the registered Hoeffding half-width before clipping is approximately 87.85 percentage points (12 contrasts, alpha 0.05). A 10-point descriptive advantage can therefore coexist with an inconclusive interval. No significance or equivalence claim follows from that gate alone.

Before production, verify own/opponent/move-slot permutations and target/mask remapping, original action-law probabilities within FP32 tolerance, public-only input admission and unchanged old flat updates. Check both architectures’ original-next-update recovery, strict state schema and finite gradients. Run a bounded train-only performance preflight; no development results used to tune design. Freeze reviewed sources/plan/tests before the sole production run. Fresh replay must reconstruct all256games,49,152updates,all18endpoint prediction sets,30checkpoints and12original next updates. Failure preserves evidence without restarting or stitching. Update current progress docs before sealing and verify every live/archive hash plus identical repeat receipt. FullG2/G3/G4 remain open; no public server, final reserved data, Prime or GitHub transfer.

Prelaunch engineering checks completed: 28 focused tests, pinned Showdown build and predecessor admission passed. A separate train-only engineering model (seed 1025262098, 8,192 updates) checked slot symmetry at trained weight scales; maximum logit/probability errors were 1.91e-6/8.35e-7 under the unchanged 1e-5 tolerance. Its weights and results are excluded from the production comparison. An earlier 34-update-per-arm feasibility check measured about 790 flat and 328 shared updates/s. These engineering updates are additional to the 49,152 preregistered production updates.

