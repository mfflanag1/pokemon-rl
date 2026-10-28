# F22 — matched full-battle PPO after shared-policy qualification

Status: completed and independently verified on 2026-10-25. Flat learning 0/3, shared learning 2/3, shared tactical-benefit 0/3; neither all-seed learning gate passed. No policy promotion.

F21 found no established tactical advantage for the shared scorer. F22 tests whether the two matched F20 initialized policies learn battle wins under the same full-game PPO budget. It returns to the original Jaxcalibur-inspired objective of a strong policy before search. The original research assessment and course correction remain in [the objective review](OBJECTIVE_REVIEW_2026_10.md).

The [frozen plan](../contracts/shared-ppo-v1/plan.json) retains seeds 7103, 8209 and 9311. Each flat/shared arm starts from its own F20 final 8,192-update weights; Adam is reset. Both value heads initially predict 0.5. No extra imitation or rehearsal data enters training. F10's same-seed rehearsal/final checkpoint remains a frozen external reference with a different training history.

Each arm receives 128 cohorts of 1,024 selected learner decisions, two epochs and eight Adam steps per cohort: 1,024 optimizer steps per arm/seed, 6,144 overall. Complete official train-partition self-play games supply same-player GAE before row selection. Collection stops at complete-game boundaries, with a fixed 64-game cap per cohort. Games and compute can differ across architectures. Learning rate 0.0003, clip 0.2, entropy coefficient 0.01, value coefficient 0.5 on half-MSE, gamma 1 and lambda 0.95 match the existing full-battle learner.

At 0, 64 and 128 cohorts, each candidate plays 128 paired schedules in both seats against random, base-power and tactical opponents. F10 plays the same panel once per seed. The full fixed panel comprises 16,128 games. Midpoint curves are descriptive; 45 final contrasts use paired-cluster percentile bootstrap with the preregistered family correction, plus Hoeffding bounds and explicit cutoff intervals. Reused schedules across seeds mean pooled rates are descriptive. Administrative cutoffs are not draws.

The per-arm learning gate requires at least five percentage points of cutoff-aware improvement over its own initialization against base-power, an adjusted bootstrap lower bound above zero, no detected opponent regression and final cutoff rates at most 5%. The architecture-benefit gate requires the same improvement threshold for final shared versus flat against tactical, without detected random/base-power regressions. All three seeds must pass for an all-seed claim. These gates do not authorize policy promotion.

Each candidate endpoint also plays 32 separate development self-play games, 576 overall. Both actors are frozen; these games produce no training rows. Both-seat terminal-only value MSE, calibration bins and constant-0.5 comparisons are descriptive. The diagnostic team identities are separated from the main evaluation panel and admitted historical train/development identities. Natural-terminal and cutoff counts remain separate. This is development evidence against known heuristics and a recurring official team generator, not final held-out-team or ladder generalization.

Qualification covers exact flat PPO loss/gradient/eight-step parity, independent shared loss arithmetic and finite-difference gradients, policy/value learning, invalid padding and malformed states, full-game train/development separation, checkpoint identity and original-next-update recovery. Fresh verification must reproduce every production game, selected batch, optimizer update and endpoint, with all 66 checkpoint comparisons and 12 original-next-update recoveries. Production source/plan/seed/budget are frozen at launch; failure is preserved without automatic repair or restart.

Unavailable-choice feedback follows the sealed F9 v2 collector: an advertised action rejected as trapped remains an attempted action, followed by the refreshed player request. Same-player linkage includes that feedback transition and records the original likelihood; it does not substitute a later accepted action. Before/after feedback is journaled. Such attempts are counted, never filtered to manufacture a clean panel. Checkpoint recovery means deterministic next-update replay from the original saved state, not an injected crash/restart test.

Full G2/G3/G4 and public strength remain open. Final reserved evaluation, public servers and Prime are unused. GitHub remains on the existing visibility hold.

## Measured fixed-run result

Measured production duration: 1184.68 seconds on local CPU. Collection calls: 1084.88 seconds; optimizer calls including compilation: 21.35 seconds. Peak parent-process RSS: 1910.1 MiB (worker memory is not included).

| Role | Games | Natural terminals | Cutoffs | Attempted decisions |
| --- | ---: | ---: | ---: | ---: |
| Training | 15,056 | 15,044 | 12 | 978,470 |
| Frozen opponents | 16,128 | 16,113 | 15 | 978,014 |
| Separate self-play diagnostics | 576 | 576 | 0 | 35,843 |

All six learners completed 128 cohorts and 1,024 optimizer steps each: 6,144 steps, 786,432 selected PPO rows and 1,572,864 row presentations overall. There were 381 unavailable-choice responses across the three roles, preserved with their refreshed requests. Actual game counts and compute differed, while selected-row/update budgets matched.

Descriptive pooled wins out of 768 games per opponent at each endpoint (the three seeds reuse schedules):

| Policy | Cohorts | Random wins | Base-power wins | Tactical wins | Cutoffs across opponents |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | 0 | 605 | 305 | 50 | 6 |
| flat | 64 | 703 | 343 | 50 | 3 |
| flat | 128 | 708 | 343 | 42 | 2 |
| shared | 0 | 711 | 259 | 41 | 0 |
| shared | 64 | 711 | 355 | 59 | 1 |
| shared | 128 | 724 | 355 | 55 | 2 |
| reference | reference | 677 | 236 | 31 | 1 |

| Seed | Flat learning | Shared learning | Shared tactical benefit over flat |
| --- | --- | --- | --- |
| 7103 | fail | pass | fail |
| 8209 | fail | fail | fail |
| 9311 | fail | pass | fail |

All 45 final contrasts are retained: {'inconclusive': 35, 'improvement': 10} under the registered approximate adjusted bootstrap. Midpoint curves are descriptive and did not select a checkpoint. Experiment completion is separate from learning-gate passage.

Both-seat terminal-only self-play value MSE, with natural-terminal seat views counted explicitly:

| Seed | Arm | Initial MSE | Midpoint MSE | Final MSE | Final constant-0.5 MSE | Final terminal seat views |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7103 | flat | 0.250000 | 0.238488 | 0.217980 | 0.250000 | 64 |
| 7103 | shared | 0.250000 | 0.237074 | 0.229987 | 0.250000 | 64 |
| 8209 | flat | 0.250000 | 0.242485 | 0.225955 | 0.250000 | 64 |
| 8209 | shared | 0.250000 | 0.240624 | 0.226363 | 0.250000 | 64 |
| 9311 | flat | 0.250000 | 0.234284 | 0.222787 | 0.250000 | 64 |
| 9311 | shared | 0.250000 | 0.240198 | 0.231792 | 0.250000 | 64 |

The two seat views of a self-play game are dependent. These value errors condition on natural termination; cutoff games contribute no terminal target. They describe each current policy’s self-play distribution, not calibrated predictions against arbitrary opponents. Full bin summaries, prediction spreads, decision-weighted errors and frozen-opponent value diagnostics remain in the report.

Familiar diagnostic mastery (initial → midpoint → final):

- 7103, flat: pass → pass → pass.
- 7103, shared: pass → fail → fail.
- 8209, flat: pass → pass → pass.
- 8209, shared: fail → fail → fail.
- 9311, flat: pass → pass → pass.
- 9311, shared: fail → fail → fail.

These finite familiar-context diagnostics support interpretation; they are not battle-strength or promotion gates. F10 remains the external reference until a separate decision is made. No public-strength, full G2/G3/G4, final reserved-evaluation or Prime claim is made.

## Interpretation and next decision

This is real full-battle reinforcement learning with measurable but modest terminal-value learning, but it has not established the tactical strength needed for the original objective. The flat policy’s descriptive pooled random wins rose from 605/768 to 708/768; shared base-power wins rose from 259/768 to 355/768. Shared started below flat against base-power (259 versus 305 initial wins), so its larger within-arm gain also reflects more initial headroom. Their final base-power results are close (355 versus 343), and do not establish shared superiority. Flat tactical wins declined descriptively from 50 to 42; those intervals include zero, so this is not a detected regression under the registered test. All 45 final contrasts are retained: 10 improvements, 0 detected regressions and 35 inconclusive under the preregistered approximate bootstrap. Every tactical contrast and every final shared-versus-flat contrast remains inconclusive. The all-seed gates fail; lack of a detected regression or superiority does not establish equivalence. Shared's familiar mastery falls from 1/3 to 0/3, whereas flat retains 3/3. No candidate is promoted and no favorable seed or midpoint is selected.

The self-play diagnostic critic errors improve from the constant 0.25 baseline to 0.2180–0.232 in all six final policies. This is a small descriptive development check with 32 physical games per endpoint/seed, not a statistical calibration guarantee or evidence that the critic predicts a stronger opponent correctly. All 576 diagnostic games ended naturally. Known heuristic identities, generator families and species recur; the final reserved evaluation remains untouched.

The next proposed intervention is a public matchup-feature ablation in full-battle RL; event history should be a later, separate intervention. Use flat as the initial host because it retained familiar mastery in all three seeds and the head-to-head comparison established no shared advantage. This is an experimental design choice, not policy promotion. First qualify an additive, zero-initialized feature adapter that preserves every original policy/value prediction. Both treatment and zero-masked control must have the same augmented parameter inventory and start from all three F22 final flat checkpoints with the same optimizer-reset rule and fixed budget. The control receives zero in the new channels; treatment receives strictly player-visible matchup information. Validate unknown-information handling and input isolation before training. Keep F10 as an external reference and battle outcomes as the primary measure.

This tests a hypothesis motivated by the current lossy representation and the original Jaxcalibur design; F22 does not establish representation as the cause of weak tactical play. Do not add imitation merely to raise agreement, select a favorable seed or midpoint, or extend this run. Historical-opponent cross-play and stronger local opponents remain separate robustness work. Search remains deferred until there is a useful base policy. The next experiment requires its own frozen plan; none is launched by F22. Each learner saw only 2,392–2,648 training games in F22. This remains a small learning screen, not a sustained competitive training campaign. After adapter qualification, the next plan should include a meaningfully longer preregistered full-battle learning curve, without changing opponents, representation and optimization settings together.

## Throughput and stability

A post-run audit reconciles all 3,970 collection calls and 6,144 finite optimizer metric records. Training collection used 405.83 seconds: 212.27 seconds in worker barriers, 177.91 in combined encoding/inference/synchronization, and 9.05 in coordinator handling. Frozen-opponent evaluation used 659.22 seconds and separate self-play diagnostics 19.83. Optimizer calls, including compilation, used 21.35 seconds across all six learners. These buckets do not isolate raw simulator or device compute: barriers include IPC/serialization/waiting, and encoding and inference are combined. No GPU speedup is inferred. This workload calls for collector profiling before allocating Prime resources or attributing the bottleneck to the learner.

Training collected 978,470 rows, selected 786,432 after full-trajectory GAE, and excluded 192,038 according to the frozen sampling rule. First-minibatch approximate KL values across all 768 cohorts were between -1.60e-9 and 2.98e-9, consistent with FP32 rounding at the current rollout policy. Gradient norms were finite, from 0.0451 to 0.3207. Unavailable-choice totals reconcile to 381: 132 training, 222 opponent evaluation and 27 self-play diagnostic attempts. They remain in the original/replay tapes and journals. No restart, budget extension, extra seed, final reserved data, public server or Prime use occurred. GitHub remains held.

A separate post-run audit of all 6,144 updates reports approximate KL by minibatch step. At the eighth step, sample approximate KL ranges from 7.61843e-05 to 0.0105076, with mean 0.00223743; all step summaries are retained in the review evidence. Clip fraction was not recorded. First-minibatch agreement alone is not a bound on subsequent policy drift.

## Completion verification

Production and the separate exhaustive replay both exited with code 0. Replay took 1155.78 measured seconds, separately from the production duration. Every production game, action/value record, selected batch, update metric and parameter fingerprint reproduced. All 66 checkpoint parameter, Adam and counter states match. The replay loaded the original states after 63 and 127 completed cohorts in each arm/seed, then reproduced each next cohort’s collection and eight Adam steps: 12 checkpoint resumptions overall. This is deterministic original-state recovery, not an injected crash test. Scientific reports are byte-identical; timing fields are reported separately.

All 14 focused qualification groups passed against the frozen 604-file source/configuration inventory. The pinned Showdown build passed. The sealer baseline accepted the evidence and rejected all 19 copied corruptions, including missing seeds/arms/endpoints, wrong counts, forged gates, incorrect replay hashes, stale test bindings and failed launcher/build records. Numerical learner qualification and the implementation/result reviews are preserved in the review directory; model reviews were static, while the coordinator executed the checks and replay. The result review's headroom, descriptive decline, critic and control-initialization clarifications are reflected above.

Completion is bound by the [receipt](../artifacts/reports/f22-shared-ppo.json). Production report SHA256: d801dccb9f25b9c2bd1c172480c305c5d6d77ae9c2411b5463e7fa92f3d909e7. Final receipt integrity and identical resealing records are stored outside the sealed review directory. Historical handoffs and sealed predecessors remain unchanged. Full G2/G3/G4, competitive strength, public play and final reserved evaluation remain open; GitHub remains on its existing visibility hold.
