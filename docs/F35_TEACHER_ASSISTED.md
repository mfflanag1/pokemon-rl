# F35 — eight-hour teacher-assisted RL goal

Status: fixed F35 execution failed during teacher acquisition at 2026-10-28T00:27:35Z, before PPO training or scientific evaluation. The supervisor lost inspectability of a previously tracked simulator worker and terminated the stage; transport cleanup then raised a permission error. The low-level OS causes remain unresolved. All frozen evidence is preserved; the comparison is incomplete and no learning gate was assessed.

## Objective and clock

Test whether fresh demonstrations from the pinned Foul Play agent improve complete-battle performance of the existing F34 structured/history policy beyond an equal-budget PPO-only control. Retain every seed and both arms, including failures and inconclusive results. This is one development comparison, not a promise of stronger play.

- Start: 2026-10-27 23:36:08 UTC (7:36:08 p.m. EDT October 27).
- Four-hour checkpoint, if still active: 2026-10-28 03:36:08 UTC (11:36:08 p.m. EDT October 27).
- Deadline: 2026-10-28 07:36:08 UTC (3:36:08 a.m. EDT October 28).

Allow up to two hours for focused implementation, qualification and independent review; target approximately four and a half hours for the fixed acquisition/training/evaluation, leaving verification and reporting time. These are estimates from F34, not guarantees or permission to truncate cells. Stop launching work at the deadline. If the fixed comparison cannot finish, preserve the actual partial outcome without changing budgets or claiming completion.

## Why this direction

F34 added structured identity and public history, but none of its three learning gates passed. Foul Play wins remained 11/192 before and 13/192 after, pooled descriptively; all external contrasts were inconclusive. More ordinary PPO alone has not yet supplied convincing evidence. F35 tests an explicit source of stronger action supervision while keeping the architecture fixed.

[Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835) and [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087) motivate teacher guidance and demonstration assistance. Their results do not establish benefit here. An offline hard-action auxiliary loss is not a replication of on-policy teacher distillation. Earlier local F15/F16 imitation results also caution that better label fit need not improve battles.

## Chosen design

Use all seeds 7103, 8209 and 9311. Both arms start from the same-seed F34 final checkpoint at `artifacts/runs/f34-policy-reset-v1/<seed>/checkpoints/512`, with identical parameters and Adam reset. The 261,559-parameter, 4,224-feature policy and critic remain structurally unchanged.

Add one fixed-weight masked hard-action demonstration cross-entropy term (coefficient 0.1) to each treatment PPO minibatch. Control uses the original PPO update. Use a separate 256-row demonstration minibatch per treatment update, sampled game-uniformly and then accepted-decision-uniformly. The dataset and sampling schedule are fixed before learning. No critic target comes from demonstrations; shared-trunk gradients can nevertheless change values. Qualify exact original-update parity when assistance is zero. Extra treatment computation is measured; equal PPO steps do not make compute equal.

The independent review favored this over a standalone behavior-cloning warm start because it preserves the shared initial checkpoint and matched optimizer-step count. No warm start, coefficient sweep, probe-based selection or insertion of demonstration rows into PPO is permitted.

Fixed budgets:

| Item | Budget |
|---|---:|
| Cohorts per seed and arm | 256 |
| Selected PPO rows per cohort | 1,024 |
| Stream quotas | 256 self-play + 256 frozen F32 + 512 tactical |
| PPO epochs / minibatch | 2 / 256 |
| Total cohorts / optimizer steps | 1,536 / 12,288 |
| Total selected PPO rows / presentations | 1,572,864 / 3,145,728 |
| Treatment demonstration presentations | 1,572,864 |

Both arms use the same stream definitions and deterministic schedule namespaces. Complete games precede GAE and fixed row selection. Different policies generate different games, costs and eligible row counts; equal row/update budgets do not imply equal compute or trajectories. Retain stream-level collected, selected and excluded counts. Frozen F32 supplies the same opponent identities as F34; no F34 evaluation transcript enters training.

## Fresh demonstrations and partition integrity

Acquire 384 new explicitly train-partition teacher-executed games: 128 against each frozen F34 host, balanced seats. Collect 96 separate development probe games: 32 per host. The probe is diagnostic only, excluded from training and checkpoint/hyperparameter selection. Dataset acquisition cap is 75 minutes; incomplete scheduled acquisition is a blocker, not permission to drop or replace games. Shared demonstration data across learners means the three results are conditional on one corpus.

Use the existing pinned Foul Play/runtime/public prior/search budget. A new split-aware worker is required because the F34 external worker fixes development. Encode only the teacher's own observable projection and history immediately before its action. Record request IDs, legal masks, commands, accepted/rejected submissions and outcomes. Labels are accepted teacher actions only. Moves 0–3, Tera 4–7 and roster switches 8–13 retain request-relative mapping. Wait frames provide no labels and are counted. An unmappable active choice fails the fixed execution, rather than silently omitting a difficult state. Preserve rejected attempts and continue the same game with its refreshed request; do not replace difficult games.

The F35 subprocess adds exact request-bound acceptance feedback around the unchanged adapter. It snapshots `last_selected_move` before a teacher choice, restores it when that choice is rejected, and refuses further ingestion/choice until feedback arrives. The request counter still prevents duplicate decisions. This prevents a rejected recommendation from becoming the teacher's last executed selection. Qualify stale/duplicate/missing feedback and rejection recovery explicitly before launch.

Teacher advice on learner-executed states is out of scope: the current adapter mutates last-selected-move bookkeeping when queried, so naive DAgger would corrupt subsequent state. Teacher-executed trajectories avoid that interface hazard. Unseeded wall-clock Rust search is recorded, not claimed reproducible by rerunning the search.

Enforce existing ordered whole-team hash partitions, exact train/development admission and the five F34 quarantine keys. No lead rotation, hash salt change, relabeling, final-reserve access or use of F34 evaluation data. Retain the historical F33 exposure limitation. Fresh development panels must be disjoint from teacher probe teams at the declared exact-team boundary; this is not held-out-species or unordered-team generalization.

## Fresh fixed evaluation and decision rule

Evaluate the shared initialization once and each final arm, using identical fresh paired-seat schedules within each seed. Fixed budgets per seed/state:

| Opponent | Paired schedules | Games | Interpretation |
|---|---:|---:|---|
| Pinned baseline-v1 / public-matchup-heuristic-v1 | 256 | 512 | Historically known, excluded from F35 demos and PPO |
| Pinned Foul Play | 64 | 128 | Trained-on teacher adaptation |
| Tactical | 64 | 128 | Trained-on familiar regression check |
| Base-power | 32 | 64 | Familiar regression check |
| Random | 32 | 64 | Familiar regression check |

Total: 8,064 games across three states and three seeds, including 1,152 Foul Play games. No intermediate scientific evaluation or best-checkpoint selection.

Primary endpoint: final treatment minus final control against baseline-v1 per seed. Gate: cutoff-aware lower point gain at least 5 percentage points, family-adjusted paired-seat-cluster bootstrap lower bound above zero, and both final-arm cutoff rates at most 5%. Overall primary claim requires all three seeds. Three teacher-adaptation contrasts are secondary in the same six-contrast family (20,000 resamples, fixed seed 1027273503). Common team pairs and mechanics seeds pair outcomes across arms; the two seats form each bootstrap cluster. Unseeded teacher-search randomness remains. Report unresolved cutoff bounds with cutoffs in [0,1] and bounded Hoeffding intervals; do not score cutoffs as losses, ties or omitted games. Retain every contrast.

Initialization-to-final changes, tactical/base-power/random retention and terminal-only value behavior are diagnostics; report every opponent regression and do not equate an undetected regression with noninferiority. The other 39 contrasts have a separate diagnostic multiplicity family and do not override the six prespecified primary/teacher contrasts. Even 512 games per cell may not resolve a five-point benefit; failure of the all-seed conjunction does not establish no benefit. Averages across three seeds are descriptive, not an unseen-seed population certificate.

## Evidence and stopping rules

Focused qualification covers off-policy loss separation, masks/accepted-action alignment, observation privacy, exact zero-assistance parity, stream budgets, split/quarantine checks, checkpoint round trips and original-next-update recovery. Include an explicit unavailable-choice fixture, because F34's 66 sampled replays contained no rejected attempts. Add compact rejection attribution for the new run; distinguish rejected attempts, accepted decisions and selected PPO rows.

Review implementation independently before launch. Sample fresh game/action replay and original updates; external replay uses recorded accepted actions, not recreated search. Bind runtime/source/input/evidence hashes and independently check the final receipt. No unrelated regression or exhaustive predecessor replay. Report teacher acquisition and label coverage, all seeds/arms, terminal/cutoff counts, policy/value behavior, stream rows, costs, memory and useful rows per wall hour. PPO KL/clip metrics are pre-update minibatch descriptions, not full-batch post-update KL.

Any failed scientific stage stops the fixed execution. No scientific restart, stitch, resampling, dropped seed/cell, outcome-dependent budget change or substitute easier comparison. Engineering fixes before freeze are allowed with retained failures and focused requalification. At the four-hour checkpoint report measured progress and feasibility without adjusting the frozen design to results. At completion or deadline report the supported decision and pause the heartbeat. Do not mark an incomplete scientific objective complete merely because time expired.

Full G2/G3/G4 and public strength remain open. No architecture search, simulator rewrite, deeper search, public play, final reserved evaluation, cloud training/spending or Prime. GitHub remains held pending an explicit visibility choice; no push, upload, uploader resume or visibility change.

## Planned artifacts

- Contract: `contracts/teacher-assisted-v1/` (frozen at launch admission).
- Implementation: `python/pokemon_teacher_assisted/`, `packages/teacher-assisted/`, focused new tests/scripts only.
- Demonstrations: `artifacts/runs/f35-teacher-data-v1`.
- Learning/evaluation: `artifacts/runs/f35-teacher-assisted-v1`.
- Reviews and qualification: `artifacts/reviews/f35-teacher-assisted/`.
- Continuation: `artifacts/reviews/f35-teacher-assisted/CONTINUATION.md`.
- Final receipt: `artifacts/reports/f35-teacher-assisted.json` only if the complete fixed workflow and verification succeed; otherwise preserve a clearly separate failure/partial report.

All frozen predecessors and unrelated working-tree edits remain untouched.

## Launch admission — 2026-10-28T00:21:08Z

The fixed F35 driver launched with launcher PID 33824 and driver PID 33825; these are historical identities and both have since exited with failure. The source admission binds 1,097 files with input hash `264086bf50ec4ec488c4959f3462d5dc7be43015483b6f8fa1789533677c9659`. All 870 F34-bound files remain unchanged. Before launch, 67 focused checks, the six-cell discarded engineering preflight, one recorded-action teacher replay, pinned build/runtime checks and independent code-review dispositions passed. The complete 1,584-pair declared team schedule has no train/probe/evaluation identity intersection. No scientific outcome is available at launch.

The four-hour checkpoint remains 2026-10-28T03:36:08Z and deadline 2026-10-28T07:36:08Z. Scientific stages are owned by the single driver; no manual duplicate/restart is permitted. Current process and closure instructions are in `artifacts/reviews/f35-teacher-assisted/CONTINUATION.md`.


## Failed execution — 2026-10-28T00:27:35Z

The declared fixed schedule passed. Teacher acquisition then stopped after 318.44 seconds of stage wall time (317.87 seconds in the acquisition failure ledger); the workflow took 386.28 seconds. Neither the acquisition cap nor the goal deadline was reached. The supervisor recorded `Cannot obtain process birth identity: 37541` while refreshing a previously known Node worker. It terminated acquisition during the next descriptor's Node hello. The frozen worker transport raised `PermissionError` during process-group cleanup, masking cooperative termination. This identifies the recorded causal sequence, but does not establish why the OS identity or group-signal operation failed.

Preserved acquisition totals:

| Measure | Actual |
|---|---:|
| Planned teacher games | 480: 384 train and 96 probe |
| Completed games | 43, all train host 7103 |
| Recorded natural terminals / cutoffs | 43 / 0 |
| Interrupted descriptor, before battle start | 1, with no outcome or recorded decisions |
| Never-started descriptors | 436 |
| Combined accepted decisions | 2,085: 985 teacher and 1,100 frozen host |
| Accepted teacher labels | 985: 718 ordinary moves, 41 Tera, 226 switches |
| Teacher wait frames / rejected submissions | 207 / 0 |
| Scientific PPO games, selected rows and optimizer steps | 0 |
| Scientific evaluation games | 0 |

The completed record contains 41 teacher wins and 2 frozen-host wins. This is one incomplete acquisition prefix, with only 21 complete seat pairs and one unpaired game; it is not a teacher evaluation panel or a learning result. All three planned seeds and both arms remain in the failed comparison disposition as not started. There is no estimate of assistance benefit, value drift, opponent regression or a learning gate. The preserved partial labels are not an admitted completed corpus and cannot be used to stitch or salvage this fixed attempt.

The supervisor's captured-process cleanup reports no remaining owned identities; a later read-only check found none of the named launch/stage/worker PIDs present. This does not prove global absence of wholly unobserved descendants. Static accounting checked all completed artifact hashes and 985 label/acceptance links, with 1,097 live and frozen source files unchanged. These checks do not constitute fresh simulator replay or outcome revalidation. The 67 prelaunch qualification checks and discarded engineering runs remain separate evidence.

Failure diagnosis: `artifacts/reviews/f35-teacher-assisted-lifecycle-diagnosis.md`. Complete 480-descriptor disposition and static accounting: `artifacts/reviews/f35-teacher-assisted-partial-accounting.json`. The separate [failure receipt](../artifacts/reports/f35-teacher-assisted-failure.json) and [independent postcheck](../artifacts/reviews/f35-teacher-assisted-failure-postcheck.json) are complete. Receipt and archive bytes match; all 1,097 live/frozen source hashes and 542 live/archived evidence hashes passed, including all 309 partial-run files. The independent checker recomputed the 43 outcomes, 985 accepted label links and full 480-descriptor disposition without running new games. A bounded cross-model review supports the recorded failure sequence, with verified limitations in `artifacts/reviews/f35-teacher-assisted-failure/review-dispositions.md`. The heartbeat is PAUSED after failure reporting. The success receipt remains absent. The scientific goal has not been achieved; no repair or scientific restart is authorized by this attempt's frozen failure policy.

The native scientific goal was marked **BLOCKED** after the third consecutive blocker audit at 2026-10-28T00:42:25Z. Failure preservation is complete; the learning comparison is not. The heartbeat remains paused.
