# F39 — teacher-free online PPO continuation

Status: **failed at independent verifier startup** on 2026-10-28 at 21:12:25 UTC. All three production seeds completed training and evaluation, but the verifier rejected its own generic lifecycle wrapper before any independent replay, recovery or checkpoint comparison. Measured reporting and the success receipt were never started. This is our verifier/launcher integration defect. The original run is preserved; no automatic repair or rerun occurred. The separate F35 teacher-assisted native goal remains blocked.

The single launcher started at 20:32:18 UTC with launcher PID 27516, driver PID 27517 and production PID 27522. Production exited 0; verifier PID 26777, workflow and outer driver exited 1. Cleanup records contain no remaining owned/observed PIDs, unknown PIDs, errors or process-table error; polling cannot prove absence of every transient or reparented descendant. The approved cutoff was 2026-10-29 00:13:21 UTC (8:13:21 p.m. EDT), and did not cause this failure.

## Question and design

Does another fixed block of the existing local PPO training improve each of the three F34 policies against baseline-v1? All final F34 states for seeds 7103, 8209 and 9311 continue with their exact parameters, Adam moments and step 4096, and per-stream episode cursors. There is no teacher, demonstration loss, new input representation or optimizer change.

Each seed receives 256 new cohorts of 1,024 selected rows: 256 self-play, 256 against its frozen same-seed F32 policy, and 512 against tactical. Two epochs with minibatches of 256 produce 2,048 new optimizer steps per seed. The complete screen therefore requires 768 cohorts, 6,144 new steps, 786,432 selected rows and 1,572,864 presentations. Final coordinates are absolute cohort 768 and Adam step 6144. Collected games and excluded rows are measured rather than equated with these fixed row budgets.

Fresh common initial/final panels evaluate each seed against baseline-v1 (512 games per endpoint), tactical (128), base-power (64) and random (64): 4,608 evaluation games in total. The plan is `contracts/online-continuation-v1/plan.json`. Baseline-v1 is historically familiar but excluded from this continuation's training. Tactical is trained on; self-play and the frozen F32 opponent are separate training streams.

The primary gate for each seed requires a cutoff-aware lower point improvement of at least five percentage points, a positive multiplicity-adjusted paired-cluster bootstrap lower bound, and a final baseline cutoff rate of at most five percent. All three seeds must pass to call the primary screen successful. All twelve dependent opponent/seed contrasts, regressions, and unresolved cutoff bounds remain in the report. Execution completion can coexist with failed learning gates. This is development evaluation with recurring official generation, not independent architecture replication, held-out-team evaluation or public ladder strength.

## Qualification and verification

The discarded engineering preflight loaded all three original F34 states and ran 144 local training games plus six baseline games. It selected 3,072 rows, applied 24 new optimizer steps, reproduced those 24 steps from saved originals, and passed six checkpoint comparisons. Two actual unavailable-choice rejections were retained and accounted for. It took 22.66 seconds on CPU; peak parent RSS was 730 MiB (worker and total-machine memory excluded). The small-sample campaign projection is approximately 56 minutes, excluding verification; game lengths and runtime may vary. None of these games, states or samples enter the scientific run.

F39 reuses unchanged qualified F34 PPO/checkpoint components and the F35 local collector, including recorded rejection feedback. F35/F36/F37 remain failed and none of their partial teacher labels or games are reused. F38's native revival incompatibility remains unresolved; the teacher branch stays paused.

The single bounded driver was designed to run production, sampled independent verification, measured results, and receipt creation. The planned independent verification would check every training ledger and evaluation schedule; fresh replays would cover 18 training games and 48 evaluation games, plus six original-next-update recoveries and nine checkpoint comparisons. None of that independent stage work executed. The planned replay scope was sampled, not exhaustive. Pre-update minibatch KL/clip statistics are descriptive and are not full-batch post-update KL. Value error is reported only for natural terminals; administrative cutoffs remain unresolved and are not draws or fabricated value targets.

Independent completion checks must validate receipt content, archived receipt byte equality, live/frozen input hashes, all evidence hashes and complete stage/output inventories, fixed counts, all seeds and contrasts, successful cleanup, and completion/driver linkage. Post-seal checks stay outside the sealed review directory. No rerun, extension, stitching, seed selection or dropped cell is authorized after scientific launch. A failure preserves actual partial counts and unstarted work.

Final admission passed 49 focused checks and an independent audit of all 2,096 live/frozen input files. Admission SHA256: `2921e7aaed8dfba6f4826ba4e329027172760d5818c944f9fb518bb67e3d937e`. The audit directly compared all three inherited parameter and Adam arrays and cursors with their original F34 checkpoints. Cross-model findings and verified dispositions are preserved in the review directory.

The F39 heartbeat is **paused** after failure preservation and independent byte checks. The teacher-assisted heartbeat remains paused. Production completed; independent scientific verification failed before its first check; measured reporting never started. The 49 checks and standalone fresh-process module check did not exercise the real lifecycle-wrapper/guard composition. The mocked driver tests inspected that wrapper command without launching it. Qualification versions and historical limitations remain preserved.

## Paths and limits

- Production: `artifacts/runs/f39-online-continuation-v1`
- Review and stage records: `artifacts/reviews/f39-online-continuation`
- Success receipt: `artifacts/reports/f39-online-continuation.json` — not created
- Success postcheck: `artifacts/reviews/f39-online-continuation-postcheck.json` — not run on this failure

Full G2/G3/G4 and public strength remain open. No promotion, public server, final reserved data, cloud spending or Prime is authorized. GitHub remains held by `artifacts/repository-backups/visibility-hold.json`; no push, upload, uploader resume or visibility change.

## Preserved production measurements and provisional results

These are saved production records reconciled by a separate read-only audit of all 768 update ledgers and 4,608 evaluation traces. That audit found no arithmetic or inventory discrepancies. It did not replay games, recover updates, run the scientific verifier or recompute bootstrap intervals. The complete comparison remains unverified.

Production took **2,404.12 seconds (40.07 minutes) of wall time on CPU**, including 2,315.24 seconds of collection and 47.48 seconds of updates including compilation. Internal sampled recovery updates took 0.355 seconds. Stage wall time was 2,405.49 seconds; workflow wall time 2,406.47 seconds; outer launcher/driver wall time 2,407.03 seconds. These are elapsed times, not CPU-seconds. Peak parent RSS was **865.75 MiB**, excluding workers and total-machine memory.

Each seed completed 256 NEW cohorts, 2,048 NEW steps, 262,144 selected rows and 524,288 presentations. Aggregate totals are 768 NEW cohorts, 6,144 NEW steps, 786,432 selected rows and 1,572,864 presentations. Each cumulative final checkpoint is absolute cohort 768 / Adam step 6,144, continuing from cohort 512 / Adam step 4,096. Production internally reported nine checkpoint comparisons and six original-next-update recoveries; the independent stage completed **zero** of each.

| Training stream (all seeds) | Games | Decisions | Collected learner rows | Selected | Excluded |
|---|---:|---:|---:|---:|---:|
| self_play | 6,144 | 327,156 | 327,156 | 196,608 | 130,548 |
| frozen_f32 | 12,200 | 633,238 | 316,126 | 196,608 | 119,518 |
| tactical | 18,424 | 905,520 | 456,735 | 393,216 | 63,519 |

Training totals: **36,768 games / 1,865,914 decision attempts**, including 6,144 self-play games / 327,156 attempts. Evaluation: **4,608 games / 216,997 decision attempts**. All **41,376** games ended naturally, with **zero cutoffs or ties**. The row pool contained 1,100,017 eligible learner rows; 313,585 were excluded by fixed selection. Training retained 695 rejected attempts in logs, 531 on the controlled side and 377 selected rows; evaluation retained 52 rejected attempts, 33 on the controlled side. Decision attempts therefore include rejected choices rather than only accepted simulator actions.

| Seed | Self-play collected / selected / excluded | Frozen F32 collected / selected / excluded | Tactical collected / selected / excluded |
|---|---:|---:|---:|
| 7103 | 106,796 / 65,536 / 41,260 | 101,152 / 65,536 / 35,616 | 152,924 / 131,072 / 21,852 |
| 8209 | 112,324 / 65,536 / 46,788 | 107,295 / 65,536 / 41,759 | 152,876 / 131,072 / 21,804 |
| 9311 | 108,036 / 65,536 / 42,500 | 107,679 / 65,536 / 42,143 | 150,935 / 131,072 / 19,863 |

All three provisional primary gates are **false**: no baseline point gain reaches five percentage points and no adjusted lower bound is positive. All final baseline cutoff rates are zero. The twelve dependent paired-schedule contrasts are retained below; intervals are the stored approximate familywise bootstrap intervals, in percentage points. Every interval includes or touches zero. Several raw scores declined, but the stored analysis detects no adjusted regression; this is not a demonstration of noninferiority.

| Seed | Opponent | Initial wins | Final wins | Games per endpoint | Gain (pp) | Adjusted interval (pp) | Terminal-only value MSE, initial → final |
|---|---|---:|---:|---:|---:|---|---|
| 7103 | baseline-v1 | 252 | 269 | 512 | +3.32 | [-4.10, +10.55] | 0.17513 → 0.20413 |
| 7103 | tactical | 37 | 43 | 128 | +4.69 | [-7.03, +16.41] | 0.15348 → 0.14201 |
| 7103 | base_power | 56 | 51 | 64 | -7.81 | [-23.44, +4.69] | 0.19367 → 0.23196 |
| 7103 | random | 64 | 62 | 64 | -3.12 | [-10.94, +0.00] | 0.12252 → 0.18268 |
| 8209 | baseline-v1 | 236 | 235 | 512 | -0.20 | [-7.03, +6.25] | 0.16908 → 0.18037 |
| 8209 | tactical | 44 | 50 | 128 | +4.69 | [-11.72, +21.09] | 0.16144 → 0.16403 |
| 8209 | base_power | 50 | 57 | 64 | +10.94 | [-4.69, +26.56] | 0.17745 → 0.12111 |
| 8209 | random | 61 | 62 | 64 | +1.56 | [-6.25, +9.38] | 0.13904 → 0.09598 |
| 9311 | baseline-v1 | 248 | 252 | 512 | +0.78 | [-6.45, +8.20] | 0.17384 → 0.16837 |
| 9311 | tactical | 51 | 56 | 128 | +3.91 | [-9.38, +17.19] | 0.15425 → 0.15234 |
| 9311 | base_power | 56 | 52 | 64 | -6.25 | [-21.88, +6.25] | 0.19042 → 0.18908 |
| 9311 | random | 64 | 62 | 64 | -3.12 | [-10.94, +0.00] | 0.09536 → 0.10617 |

With zero observed cutoffs, the recorded lower/upper cutoff-aware point gains coincide; no unknown terminal outcomes were assigned fabricated rewards. Sampling uncertainty still spans zero. Baseline terminal-only value MSE worsened for seeds 7103 and 8209 and improved slightly for 9311. Per-game predicted values, all endpoint decisions, conservative Hoeffding bounds and exact unrounded statistics remain in the failure receipt and static audit.

Descriptive pre-update minibatch diagnostics: first-minibatch approximate KL mean −2.731e−10 (range −1.946e−9 to 1.230e−9), clip fraction 0 across 768 first minibatches. Across 3,072 final-epoch minibatches, mean approximate KL was 0.000980306 (maximum 0.0104155), mean clip fraction 0.00691096 (maximum 0.109375). These are not full-batch post-update KL measurements.

## Failure preservation and remaining work

The admitted verifier bans the entire `pokemon_teacher_runtime` prefix, although the driver intentionally uses `pokemon_teacher_runtime.lifecycle` as its generic process-safety wrapper. Loading the wrapper installs its docstring-only parent package in `sys.modules`, causing the guard to fail before validating production inputs. An additive import-only probe reproduces this exact inventory difference without running any scientific stage. A bounded independent Claude review supported the diagnosis; cleanup scope, per-seed versus aggregate counters and historical check claims were verified and narrowed.

No production seed is unstarted. All independent verification work remains unstarted: **18 training-game replays, 48 evaluation-game replays, nine checkpoint comparisons and six original-next-update recoveries**. The measured-results stage, completion record and success receipt do not exist. No result is promoted; a future repair or verification run requires separate authorization.

Failure receipt content hash: `334004b882d63164222601d1afe6da5a129135dbb04d01a582b49c2bfa86b808`. The archive binds **2,096** admitted live/frozen inputs and **5,789** evidence files; archived receipt bytes match the live failure receipt. The independent failure postcheck **passed**, verifying all 2,096 live inputs plus both frozen input copies, all 5,789 live/archive evidence files and complete inventories, receipt content/byte equality, raw ledger/evaluation counts, failure/cleanup linkage and absent later outputs. Postcheck SHA256: `07a912dc69cbadab72cf43de5da28052aac79485c5bde8a8559304b64fc77d17`. This certifies preservation and static accounting, not scientific completion.

- [Failure receipt](../artifacts/reports/f39-online-continuation-failure.json)
- [Reviewed diagnosis](../artifacts/reviews/f39-online-continuation-failure-v1/diagnosis-reviewed.md)
- [Static accounting audit](../artifacts/reviews/f39-failure-audit-v1/report.json)
- [Independent failure postcheck](../artifacts/reviews/f39-online-continuation-failure-postcheck.json)

Baseline-v1 remains historically familiar; tactical and frozen F32 were trained on. Development schedules are not independent replication or held-out-team/public-strength evidence. Full G2/G3/G4 remain open. The original F35 teacher-assisted goal remains blocked. No automatic repair, new experiment, public server, reserved data, cloud spending, Prime or GitHub upload is authorized.
