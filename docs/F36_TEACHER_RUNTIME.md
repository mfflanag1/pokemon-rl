# F36 — teacher-assisted comparison after runtime repair

Status: **failed during teacher acquisition** at 2026-10-28 02:21:07 UTC. Workflow and driver exited 1 after 1,869.78 and 1,870.28 wall seconds respectively (about 31.2 minutes on local CPU). Qualification had passed, but this comparison did not reach learner training or evaluation. The original scientific goal remains incomplete; no repair or new attempt followed this failure.

The user asked “finish, whats issuye” after being told that F35 had failed and further work required a separately recorded repair/replacement. This authorizes one fresh full attempt after qualification. It does not erase F35, authorize partial-data stitching or change scientific budgets. The original deadline remains 2026-10-28 07:36:08 UTC (3:36:08 a.m. EDT); the four-hour checkpoint remains 03:36:08 UTC (11:36:08 p.m. EDT) if work is still active.

## Failure being repaired

F35's supervisor could not inspect a previously tracked Node worker during process churn. It treated that inconclusive observation as a fatal stage error and terminated acquisition. Process-group cleanup then raised PermissionError while unwinding StageTerminated. The underlying OS causes are unresolved. The preserved failure contains 43 terminal teacher games and 985 accepted labels; no scientific learner training or evaluation occurred. See [F35 failure receipt](../artifacts/reports/f35-teacher-assisted-failure.json) and [verified diagnosis](../artifacts/reviews/f35-teacher-assisted-lifecycle-diagnosis.md).

The teacher creates a short-lived ProcessPoolExecutor child for each decision. Qualification therefore must cover real worker/search churn as well as synthetic stable trees.

## Narrow runtime change

New files under `python/pokemon_teacher_runtime/` provide the repaired lifecycle and external JSON transport. Observation distinguishes live, gone and unknown. Unknown observations retain evidence and are retried without aborting healthy science; they do not authorize signals or count as confirmed absence. Cleanup remains bounded and fails if ownership or disappearance cannot be established. PID reuse never authorizes a signal to the new process. Cleanup errors must not replace an existing timeout or termination exception.

An explicit stage overlay installs the new external JsonWorker before importing unchanged F35 stages. It fails on stale imported bindings. The local Showdown collector, teacher adapter/acceptance feedback, learner arithmetic, feature representation, optimizer, demonstrations, reporting and sampled replay algorithms remain the frozen F35 implementations. Their F35 schema strings identify reused protocols; they do not identify this replacement execution as the failed F35 run.

The replacement uses a new contract/admission, output directories, launcher/driver and receipt. All 1,097 F35-admitted files and the failed F35 evidence remain unchanged. The unchanged 67 focused checks, six-cell discarded preflight, unavailable-choice fixtures, runtime/build qualification and independent implementation review remain applicable to those algorithms. Only changed runtime/wiring behavior receives new qualification.

## Fixed comparison retained

All scientific fields of `contracts/teacher-assisted-v1/plan.json` are retained exactly. The new `contracts/teacher-runtime-v1/plan.json` adds execution metadata binding the failed predecessor and new runtime qualification.

- Seeds 7103, 8209, 9311; control and assisted arms start from same-seed F34 final parameters with Adam reset.
- 480 fresh teacher-executed acquisition games: 384 train and 96 diagnostic probe. No F35 partial game or label is reused. Same train/probe schedules, 100 ms search and 4,500-second acquisition cap.
- Each seed/arm: 256 cohorts × 1,024 PPO rows; stream quotas 256 self-play, 256 frozen F32, 512 tactical. Total 1,536 cohorts, 12,288 updates, 1,572,864 selected PPO rows and 3,145,728 PPO presentations.
- Assisted-only coefficient 0.1 masked offline cross-entropy, 256 independently sampled demo rows per update, 1,572,864 demo presentations. No architecture change, extra optimizer steps or demo critic targets.
- All 8,064 fixed evaluation games: 6,912 local and 1,152 teacher, shared initialization and both final arms. All 45 paired contrasts and cutoff bounds retained.
- Verification remains 114 sampled game replays, 12 original-next-update recoveries and 18 checkpoint comparisons. Teacher actions are replayed; stochastic native search is not reproduced.

The original schedules intentionally recur, so this is a replacement execution, not an independent replication. Foul Play is trained on; baseline-v1 is historically familiar but excluded from F35/F36 training. No held-out-team, unseen-opponent or public-strength claim is established by this development comparison.

## Qualification and launch conditions

Before launch, require independent opposite-model review and verified dispositions for the isolated runtime repair; focused lifecycle/transport/wiring tests; 64 fixed discarded engineering worker cycles with actual teacher choice calls; bounded startup and active-choice interruption checks; and one existing discarded engineering game replay with exact recorded-action/label parity. These engineering calls are not new scientific games or training labels. Retain any engineering failures and corrections.

Freeze source/admission only after these checks pass and the old source/evidence bindings still match. Launch exactly one persistent driver. No F36 output may already exist, no automatic scientific restart, no dropped cell, no outcome-dependent resizing and no extension of the original deadline. The driver records terminal exits and cleanup evidence even on startup/capture failure.

## Artifacts and closure

- Review: `artifacts/reviews/f36-teacher-runtime/`.
- Acquisition: `artifacts/runs/f36-teacher-data-v1`.
- Training/local evaluation: `artifacts/runs/f36-teacher-runtime-v1`.
- Teacher evaluation: `artifacts/runs/f36-teacher-evaluation-v1`.
- Success receipt, only after complete valid execution: `artifacts/reports/f36-teacher-runtime.json`.
- Postcheck stays outside the review directory and independently verifies all receipt-bound hashes, fixed counts/cells/stages, predecessor/runtime linkage and archived receipt bytes.

A valid completed comparison can have failed or inconclusive learning gates. On a failed stage, preserve the actual evidence and stop the attempt. At the deadline stop launching work and report the actual incomplete result. Full G2/G3/G4 and public strength remain open. No public server, final reserved evaluation, cloud training/spending, Prime or GitHub publication; the existing visibility hold remains in force.

## Measured runtime qualification

Admission SHA256: `f75e3d6236ed6c8fcf3e7ffb799079da82f12787141bce9804b4d5c37c2500f1`, covering 2,182 live/frozen files, including all 1,097 unchanged F35 inputs. There are 43 unique focused checks: 26 runtime/transport, 3 additional EOF/constructor and 14 wiring. All 43 passed together; the final timeout-inventory refinement passed the 26 affected checks again, while the remaining 17 code/tests stayed unchanged. Final real-worker qualification passed 64 completed teacher choices, 128 graceful normal worker exits, startup/active-choice interruption and one recorded engineering replay. It took 58.53 seconds of wall time.

The initial isolated implementation timed out and remains unaccepted. Its partial code was independently corrected locally. All engineering attempts are retained: v1's four stages passed but final relative-path report packaging failed; v2 passed after that correction; v3 passed with the final cleanup-inventory fix. Across the three engineering attempts, 192 completed search choices and 3 recorded-game replays were used only for runtime qualification; none are scientific labels or evaluation results.

Independent wiring/runtime reviews and verified dispositions are preserved in the F36 review directory. The original OS cause remains unresolved; the fix handles inconclusive observations and preserves original failures. Unknown descendants are never assumed gone or signalled without ownership proof. Polling cannot certify descendants that were never observed. Successful final checks establish bounded tested behavior, not universal absence of operating-system failures.

The existing heartbeat was reused under “Finish Maxcalibur F36 qualified replacement”; no duplicate automation or goal was created. The independent prelaunch record verified all live/frozen inputs, exact scientific plan and complete qualification lineage before launch. The attempt failed before the four-hour checkpoint. Its original goal clock was never reset or extended.


## Preserved F36 failure

The teacher response failed `label_candidate` in `train/8209/058-0`, game `f2:train:1850182552:58:base`. At turn 8, request 12 offered Zekrom only forced Outrage: action 0 was the sole legal action. Validation raised before the response was recorded as a step, submitted to Showdown or accepted as a label. Cleanup preserved the original exception.

The exact failed response was not persisted. Plain Outrage maps to slot 0 in the current request, so using the full roster's old slot 3 is not supported as the cause. Source inspection finds a missing per-request Tera availability check in the choice translation path. A switch mismatch is a weaker possibility because trapped status reaches the search state. These are hypotheses, not recorded choices or a proven native-engine defect. See the [bounded diagnosis](../artifacts/reviews/f36-teacher-runtime-failure-diagnosis.md).

| Acquisition partition | Completed | Failed partial | Never started | Completed-game labels |
|---|---:|---:|---:|---:|
| Train, seed 7103 | 128 | 0 | 0 | 2,845 |
| Train, seed 8209 | 116 | 1 | 11 | 2,745 |
| Train, seed 9311 | 0 | 0 | 128 | 0 |
| Probe, all three seeds | 0 | 0 | 96 | 0 |
| Total | 244 | 1 | 235 | 5,590 |

All 244 completed games ended naturally; none reached cutoff. They contain 11,809 submitted decisions: 5,590 teacher and 6,219 frozen-host decisions. There were 11,806 accepted submissions and three rejected host attempts; no recorded teacher submissions were rejected. The failed game preserves ten earlier accepted teacher labels and 19 accepted decisions, with no outcome. Across all started games, 5,600 labels and 11,828 recorded decisions survive. The invalid final response is excluded. Its earlier validation failure also explains why the runtime's illegal-action counter remained zero.

Completed labels comprise 3,923 ordinary moves, 220 Tera moves and 1,447 switches. The partial game adds seven ordinary moves and three switches. The teacher's 224 wins versus 20 host wins describe acquisition only; they are not evaluation or learning results. All 122 completed seat pairs, the failed descriptor and all 235 unstarted descriptors remain in the inventory.

Acquisition took 1,798.15 internal wall seconds and 1,799.03 supervised wall seconds. Completed teacher-choice calls account for 1,614.86 seconds, plus 2.96 seconds for saved choices in the failed game. The final failed reply's individual duration was not saved. The workflow took 1,869.78 seconds, about 31.2 minutes. These are measured wall times on local CPU, not CPU-seconds; peak memory was not measured. No scientific learner update or evaluation cost was incurred.

All six control/assisted cells for seeds 7103, 8209 and 9311 remain unstarted: zero cohorts, optimizer steps, selected PPO rows and demonstration presentations. All 8,064 planned evaluations, 45 contrasts, 114 sampled replays, 12 original-next-update recoveries and 18 checkpoint comparisons remain unexecuted. Teacher learning gates, regressions, cutoff-aware improvement and terminal-only value behavior are not assessed.

The supervisor observed a natural failed exit: no wait error, deadline timeout, cleanup signals, remaining tracked identities or unresolved identities. The independent [cleanup audit](../artifacts/reviews/f36-teacher-runtime-failure-cleanup-check.json) found all 2,089 queried historical PIDs absent. The 2,088 teacher-stage retirement records include three repeated zombie placeholders, not extra verified processes. Polling cannot establish absence of descendants never observed.

The [failure receipt](../artifacts/reports/f36-teacher-runtime-failure.json) binds all 2,182 admitted live/frozen files and 2,833 preserved evidence files, including all 1,717 actual run files. Receipt SHA256: `2747842eb29a654b014b4ba8a850b8cc80dcc81dfc14b161805b9b126f85bcf3`. The independent [postcheck](../artifacts/reviews/f36-teacher-runtime-failure-postcheck.json) passed: identical receipt bytes, all source/evidence hashes, every accepted label's NPZ row and request mask, transcript acknowledgments and the complete schedule inventory. It performed no new simulator replay, search, inference or feature re-encoding. Passed preservation checks do not complete the experiment.

Two outer audit-helper corrections are retained: accounting first misread a submission-attempt counter as accepted-only; the first integrity checker expected identical schema tags for failure and exit records. Separate corrected helpers passed without changing experiment code, data, receipt or archived evidence. An independent Codex reviewer verified the strict checker correction and executed it read-only. Cross-model diagnosis review confirmed the central source-level findings; missing-context concerns were checked locally and citations clarified.

The heartbeat is **paused** after failure reporting. The scientific goal remains incomplete; no repair, resumption or replacement was launched. F35 remains failed and none of its partial data was reused. Full G2/G3/G4 and public strength remain open. The GitHub visibility hold remains unchanged.
