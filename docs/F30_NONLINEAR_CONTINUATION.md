# F30 — Sustained nonlinear-policy continuation

Status: production failed; partial evidence preserved. No completed comparison, exhaustive replay or learning gate result.

F29's nonlinear actor improved over initialization in all three seeds, but its final tactical advantage over the trained control passed in only two. Its tactical wins at start/midpoint/final were 21/36/72, 29/43/59 and 26/37/56 out of 256 games. These observations motivate a bounded continuation; they do not establish that more training will keep improving strength. The original objective remains a strong full-battle policy before search.

## Fixed continuation

- Retain every F29 final seed (7103, 8209, 9311) and both control/nonlinear arms. Load the exact same-arm cohort-256 parameters, Adam moments, step 2,048 and next-episode cursor. No reset or reinitialization. The two arms already differ at continuation start.
- Continue absolute cohorts 256 through 1,023: 768 additional cohorts per seed/arm. Cumulative endpoint is cohort 1,024, four times F29's total cohort budget. Preserve model, inputs, PPO/GAE, terminal reward, learning rate, two epochs, 256-row minibatches and 50/50 self-play/tactical selection.
- Each new cohort selects 1,024 rows (512 from each stream) and performs eight optimizer steps. Across six runs: 4,608 additional cohorts, 36,864 new steps, 4,718,592 selected rows and 9,437,184 row presentations. Each final learner has 8,192 cumulative optimizer steps and 1,048,576 cumulative selected rows. Counts exclude F29's already-completed work.
- Preserve F29 training, selection and optimizer namespaces with absolute cohort indices and continued episode cursors, preventing counter or schedule reuse. Engineering preflight uses a separate namespace and is excluded from learning evidence.
- Fixed evaluation at absolute cohorts 256, 512 and 1,024 (cumulative selected rows 262,144; 524,288; 1,048,576). Reevaluate all six starting checkpoints on the same fresh F30 schedules used at later endpoints. Main panels use 128 paired schedules in both seats per random, base-power and tactical opponent: 13,824 games. Separate self-play diagnostics add 576 physical games, with both-seat terminal-only value reporting. Final reserved data remains unused.
- Thirty checkpoint comparisons at cohorts 256, 511, 512, 1,023 and 1,024; twelve original-next-update recoveries before updates 512 and 1,024. Fresh exhaustive replay must reproduce every game, selection, optimizer update, policy/value prediction, outcome and checkpoint state.

The versioned contract is `contracts/nonlinear-continuation-v1/plan.json`. No early stopping, extension, resampling, seed selection, cell dropping or automatic repair/restart after failure.

## Registered questions and interpretation

Primary: does each arm improve against the trained-on tactical opponent from its own F30 continuation-start evaluation to its fixed final endpoint? Also report final nonlinear-minus-control advantage. Use the inherited cutoff-aware five-percentage-point threshold, positive lower endpoint of the registered approximate familywise paired bootstrap, no detected random/base-power regression and final cutoff rates at most 5%. Require all three seeds for all-seed claims. The same 27 dependent comparisons include secondary base-power learning/benefit; retain all intervals and unresolved cutoff bounds. Midpoint curves, familiar mastery and value summaries remain descriptive.

Final nonlinear-versus-control advantage includes differences inherited from F29. It is not an independent architecture replication or a difference-in-differences estimate of the incremental benefit of additional training. Within-arm continuation gains answer the duration question. The nonlinear arm's capacity and shared-trunk feedback remain inseparable here. Value error comparisons follow policy-dependent trajectories, so they cannot alone diagnose critic failure.

These are fresh development games against familiar heuristics and recurring official generation. Tactical is trained on; gains establish adaptation, not unseen-opponent, held-out-team or public ladder generalization. Full G2/G3/G4 and public strength remain open. No promotion, public server, final reserved evaluation or Prime. GitHub remains under the unresolved visibility hold.

## Execution and evidence

Production will use `artifacts/runs/f30-nonlinear-continuation-v1`; review/replay use `artifacts/reviews/f30-nonlinear-continuation`. One persistent driver will run the fixed production, exhaustive fresh replay, measured documentation, copied-evidence rejection checks, first seal, all source/evidence hash checks and identical repeat sealing. On failure it preserves evidence and stops. Post-seal verification belongs outside the sealed review directory.

Reuse the sealed F29 learner/collector qualification and earlier evidence after checking all source hashes. Qualify only the new continuation boundaries and run all six original collect/update engineering preflights. Before launch, independently review implementation and qualification evidence, bind all helper hashes, and verify source/runtime admission. No unrelated regression or predecessor campaign rerun.

## Qualification results

All13 source-bound continuation tests passed across723 executable/configuration files. The unchanged F29 learner/collector checks are admitted through its sealed receipt. The pinned Showdown build passed. All six same-seed/arm original collect/update preflights passed in 103.46 seconds: 208 engineering games, 6,144 selected rows and48 new optimizer steps, with each Adam counter advancing2048→2056. Each collection matched the sealed F29 collector exactly on the same engineering schedule; each update and checkpoint round-trip matched. These engineering games do not establish learning.

Independent test delivery and opposite-model QA, harness review, verified finding dispositions and direct receipt-level checks of all six predecessor checkpoint pointers/files are preserved. The review's suspected missing Adam receipt binding was already covered by full predecessor evidence admission; completion additionally asserts baseline and final panel/checkpoint parameter hashes. All source/runtime and helper hashes are bound before launch.

## Failed production — 2026-10-26

F30 stopped during production after 5,476.65 seconds of launcher elapsed time (91.28 minutes). All eight simulator workers hit the 60-second response deadline while starting the first random-opponent midpoint evaluation batch for seed 9311/control at cohort 512. The underlying cause remains unresolved. The partial run preserves 3,328 of 4,608 new cohorts, 26,624 optimizer steps, 3,407,872 selected rows, 128,144 training games, 9,984 opponent-evaluation games and 416 self-play diagnostic games. No exhaustive replay or successful completion seal ran, so no completed comparison or all-seed learning result is claimed. Original run evidence and frozen code remain unchanged; no restart or repair was performed. Full G2/G3/G4 and public strength remain open. No promotion, public server, final reserved data, Prime or GitHub publication.

The four seed/arm runs for 7103 and 8209 reached cohort 1,024 and collected their scheduled panels. Seed 9311/control saved cohort 512 after 256 new cohorts; its midpoint evaluation failed before the first eight-game batch returned. Seed 9311/nonlinear never started. Completed cells are retained as partial evidence and are not selected into a substitute comparison.

Persisted game records contain 128,127 natural training terminals and 17 training cutoffs; 9,980 natural opponent-evaluation terminals and four cutoffs; all 416 diagnostic games terminated naturally. Decision counts are 6,683,117 training, 556,169 opponent evaluation and 23,775 self-play diagnostic. The failed eight-game batch is excluded. The completed report never existed, so collection/update timing, final peak memory and all-seed gates are unavailable.

The eight failure payloads identify `start` requests for paired schedules 0–3 against random, with empty partial responses and empty worker stderr. This establishes a response-timeout failure, not its root cause. The process exited 1 and the driver stopped before replay. No repair, increased timeout, automatic resume or replacement run was performed.

Failure-only evidence: [inventory and diagnosis](../artifacts/reviews/f30-nonlinear-continuation-failure/summary.json), [failure receipt](../artifacts/reports/f30-nonlinear-continuation-failure.json). The receipt binds the original evidence and unchanged 723 live/frozen source files; it does not certify learning, replay, or successful experiment completion.
