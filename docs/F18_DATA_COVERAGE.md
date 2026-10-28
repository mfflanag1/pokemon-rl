# F18 — controlled teacher-corpus expansion

Status: experiment completed; descriptive learning gate 0/3 seeds.

F18 completed (2026-10-25): compared the original 512 teacher games with an expanded 2,048-game corpus, using the unchanged 928-feature model and 2,048 updates per arm/seed. Collected 1,536 additional training games and 256 fresh development teacher games (1,792 natural terminals, 0 cutoffs, 93,300 attempted decisions). All three seeds and both arms completed 12,288 updates / 1,572,864 row presentations. The descriptive learning gate passed in 0/3 seeds. Fresh-probe voluntary-switch agreement was 5.03–7.41% for original and 4.63–5.16% for expanded. These are teacher-imitation metrics, not battle win rates. All 21 focused checks, exhaustive replay, 18 checkpoint comparisons, six original-next-update recoveries and three exact F16 control matches passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer.

Next proposed step: measure a preregistered learning curve on the now-frozen expanded corpus, keeping the original model, sampling and optimizer fixed and comparing the existing 2,048-update budget with an 8,192-update endpoint on fresh development games. Expanded-corpus fit agreement was only 14.7–16.5%, so insufficient fitting remains plausible; this does not establish that extra updates will improve transfer. Track both fitting and fresh-probe agreement, preserve all seeds and other action families, and do not collect another training corpus or promote a policy from F18.

## Fixed comparison and limits

This is a preregistered development diagnostic selected after F17. It compares the original 512 complete teacher games with their union with 1,536 fresh training games. Corpus size and diversity change together. The original 928-feature model, same-seed F10 initialization with reset Adam, sealed action distribution and likelihood loss, 2,048 updates, 32/16/8/8 family quotas and 64 rehearsal rows remain fixed. The 256 fresh development games use exact teams separated from training and F15/F16 development; familiar mechanics, species and the teacher recur. No games were filtered by outcomes or labels. All three seeds must meet the descriptive gate. Paired game-level intervals across all 12 family contrasts are diagnostics, not ladder-strength evidence. Insufficient probe support would fail the run without resampling. Each original-arm final state must reproduce its sealed F16 checkpoint exactly.

The expanded corpus is the complete union of the original 512 games and all 1,536 new games. Original IDs remain 0–511; new IDs become 512–2047. Both arms retain identical action-family quotas, 64 old rehearsal rows, initialization and total updates. Twelve actual-corpus sampler draws match the sealed legacy sampler exactly. All original-arm final parameters and both Adam moments/counters reproduce F16 exactly. Added corpus size and diversity change together; repetition per example changes while presentations stay fixed. No outcome-based game filtering or post-result budget expansion occurred.

New teacher rollouts and diagnostic metadata use the player projection. Only the original feature/mask/target model inputs enter learning. All fresh exact teams are separated from original/additional training and F15/F16 development teams. F17 collected no new games and reused the F16 probe. Known mechanics, species and the tactical teacher recur, so this is development evidence rather than final reserved-data or public-ladder generalization. Both sides of a game stay together for paired statistical diagnostics. Teacher agreement does not identify the optimal move.

The expanded corpus increased voluntary-switch training examples from 363 to 1,422 and unique destination/opponent pairs from 357 to 1,407. Fresh-probe pair coverage increased from 1/170 to 6/170 rows. The expanded arm fits its own voluntary-switch examples at 14.7–16.5%, compared with 63.1–72.6% for the original arm on its smaller corpus. At this fixed update budget, more data did not meet the switching threshold; the experiment does not establish that broader data is ineffective with a different fitting budget. All three switching-change intervals include zero. Ordinary-move point agreement increased in all seeds; two approximate adjusted bootstrap intervals exclude zero, while the conservative bounds remain wider.

## Collection and resources

| Corpus | Games | Accepted decisions | Natural terminals | Cutoffs | Rejected choices |
| --- | ---: | ---: | ---: | ---: | ---: |
| New training | 1,536 | 79,871 | 1,536 | 0 | 2 |
| Fresh development | 256 | 13,427 | 256 | 0 | 0 |

Original training: 512 sealed games / 26,278 rows. Expanded training: 2,048 games / 106,149 rows. All accepted rows, including any cutoff prefixes, remain included; teacher collection itself is not a learned-policy strength evaluation.

Measured production phase: 88.87 seconds on local CPU, peak parent RSS 2929.44 MiB. This excludes initial predecessor admission/source snapshot and subsequent verification. Additional collection: 40.71 s (1961.88 accepted decisions/s). Probe collection: 8.14 s (1648.59 decisions/s). Update computation including compilation: 13.48 s (911.45 updates/s). These component timings exclude some I/O and diagnostics and do not sum to total wall time. All 786,432 teacher and 786,432 rehearsal presentations are retained.

Two training switch attempts (games 616 and 781) were rejected by Showdown because the active Pokémon was trapped. Both games continued to natural terminals. Rejected attempts remain in the transcript but do not become supervised rows; all 79,871 accepted added-training rows and all 13,427 probe rows are retained. No choice was silently treated as accepted.

## Switching coverage

| Corpus | Voluntary rows | Games with voluntary switches | Unique destination/foe pairs | Fresh probe destination/foe rows seen |
| --- | ---: | ---: | ---: | ---: |
| original | 363 | 257 | 357 | 1/170 |
| expanded | 1,422 | 969 | 1,407 | 6/170 |

Species-pair coverage is measured within the voluntary-switch family. Related type matchups can recur across different species pairs; novelty counts do not establish causal bottlenecks or independent situations.

## All seed gates and paired switching changes

| Seed | Gate | Failed checks | Expanded minus original | Adjusted bootstrap interval | Hoeffding interval |
| --- | --- | --- | ---: | --- | --- |
| 7103 | FAIL | voluntary_accuracy, voluntary_gain | -1.59 pp | [-8.81, +4.76] pp | [-32.89, +29.72] pp |
| 8209 | FAIL | voluntary_accuracy, voluntary_gain | -2.78 pp | [-9.52, +3.86] pp | [-34.08, +28.53] pp |
| 9311 | FAIL | voluntary_accuracy, voluntary_gain | +0.13 pp | [-6.61, +6.75] pp | [-31.17, +31.44] pp |

All 12 seed/family contrasts are retained in the production report with per-game differences and IDs. Bootstrap intervals use 5,000 whole-game paired resamples and a Bonferroni adjustment across 12 contrasts; finite-sample coverage is approximate. Hoeffding bounds assume independent sampled games and apply to game-level differences. The descriptive operating thresholds are not a statistical strength gate.

## All action-family results

| Seed | Family | Initial probe | Original fit | Original probe | Expanded fit | Expanded probe | Paired change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7103 | normal_move | 23.65% | 43.44% | 34.84% | 38.48% | 36.66% | +1.82 pp |
| 7103 | tera_move | 28.57% | 83.10% | 34.90% | 52.03% | 35.31% | +0.41 pp |
| 7103 | voluntary_switch | 0.00% | 70.59% | 6.22% | 16.51% | 4.63% | -1.59 pp |
| 7103 | forced_switch | 41.97% | 49.14% | 42.93% | 44.53% | 42.40% | -0.54 pp |
| 8209 | normal_move | 23.98% | 43.71% | 34.38% | 37.72% | 36.72% | +2.33 pp |
| 8209 | tera_move | 21.43% | 82.38% | 35.92% | 52.05% | 34.90% | -1.02 pp |
| 8209 | voluntary_switch | 0.00% | 63.08% | 7.41% | 16.09% | 4.63% | -2.78 pp |
| 8209 | forced_switch | 41.76% | 49.22% | 43.27% | 44.91% | 41.92% | -1.34 pp |
| 9311 | normal_move | 23.54% | 42.13% | 32.91% | 39.49% | 38.04% | +5.13 pp |
| 9311 | tera_move | 20.00% | 81.36% | 32.86% | 46.05% | 32.24% | -0.61 pp |
| 9311 | voluntary_switch | 0.00% | 72.64% | 5.03% | 14.67% | 5.16% | +0.13 pp |
| 9311 | forced_switch | 42.62% | 49.35% | 42.82% | 44.92% | 43.40% | +0.57 pp |

Agreement is equally weighted by game within each family. Fit columns use each arm’s own training corpus and thus have different populations. Decision-weighted metrics, target probability mass and NLL for all families/endpoints are retained in the report.

## Retention and verification

| Seed | Original known-context retention | Expanded known-context retention |
| --- | --- | --- |
| 7103 | PASS | PASS |
| 8209 | PASS | PASS |
| 9311 | PASS | PASS |

Value-head weights remain unchanged; shared-trunk value predictions can change. No value-calibration or win-rate improvement is claimed. All 21 tests pass on the final source/runtime tree, including immutable sampling, probe support negatives, checkpoint inventory, original-next-update recovery, paired-game clustering and live collection parity. Pinned Showdown build passed. Fresh replay regenerates every one of the 1,792 collected games and all datasets/updates/endpoint probabilities, checks 18 checkpoints and recovers six original next updates. Production, verification and workflow exits are zero. Thirteen copied-evidence negatives reject. Documentation is bound before independent all-hash verification and identical resealing.

[Completion receipt](../artifacts/reports/f18-data-coverage.json).

Production report SHA256: `96e53b63502d07fcd112d93af7ead56b3da9704b0dbdd94fbb53c336fd01ddc1`.

## Preserved prelaunch design


Design recorded before production; all seeds, budgets and gates were retained.

F17's semantic feature block failed all three descriptive gates. F18 tests the next candidate explanation: insufficient breadth of training demonstrations. This is a controlled corpus-expansion experiment, not evidence that data coverage is the sole bottleneck.

Use the original 512 F15 teacher games as control. The expanded arm adds exactly 1,536 official training-partition games with seed 1025261801, for 2,048 total. Both arms retain all accepted teacher decisions and all terminal/cutoff outcomes. No game selection by outcome, switching count or subsequent performance. Whole-team R7 partitions remain enforced.

Hold the original 928-feature network, F10 initialization, Adam/NLL/action distribution, 2,048 updates per arm and seed, 32/16/8/8 action-family quotas and 64 old rehearsal rows fixed. Seeds: 7103, 8209, 9311. Total: 12,288 updates and 1,572,864 row presentations. An immutable prepared sampler must match the sealed sampler's indices and batches exactly; it only avoids repeated corpus validation. The original-arm final learner/optimizer must match the sealed F16 balanced checkpoint exactly.

Collect exactly 256 fresh development teacher games (seed 1025261802). Their exact teams must be disjoint from all training and F15/F16 development teams. No final reserved data. Keep natural probe frequencies, all families and all three seeds; no resampling if support is insufficient. Report game-weighted and decision-weighted agreement, probability mass and NLL, fit/probe gaps, switching coverage, natural terminals/cutoffs and collection/update throughput. Bootstrap paired differences by whole game, keeping both players together; report approximate Bonferroni intervals across 12 seed/family contrasts and conservative Hoeffding bounds. Teacher agreement is not optimal action truth, and these diagnostics do not measure battle win rate.

The descriptive per-seed gate requires voluntary-switch agreement >=25%, gain over original >=10 percentage points, no >5-point drop in any other family, and retention of all 32 familiar tactical contexts. Overall passage requires all three seeds. Minimum fresh probe support: Tera 64 rows/32 games, voluntary switching 32 rows/16 games. Size and diversity change together; holding presentations fixed reduces repetition per example.

Freeze sources/runtime/plan after checks and review. Replay every new game, dataset, update and endpoint prediction; compare 18 checkpoints and recover six original next updates. Seal only after clean process exits, source-bound tests, copied-evidence rejection checks and measured documentation. Preserve F17 and all predecessors. F10 remains the reference; no automatic follow-on training, public server, Prime or GitHub publication. GitHub visibility remains held.

