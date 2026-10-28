# F26: fixed-opponent exposure in complete-battle PPO

Status: fixed production and exhaustive replay passed; completion sealing follows the measured documentation below.

F25 completed and replayed all 60,832 training games but passed neither learning nor feature-benefit gates in any seed. Tactical wins remained 16–37 of 256. That motivates testing the training distribution; it does not establish its cause or rule out larger models and longer useful training. The original objective remains a strong independent Gen9 Random Battles policy, then scalable collection and information-safe search. F26 tests actual full-battle return while reusing the reference simulator.

F5 already tested an opponent mixture from random initialization (50% self-play,25% base-power,25% tactical,128cohorts) and did not establish a benefit; its tactical win counts were lower in all seeds and one base-power contrast regressed. F26 is a conditional retest after substantial training and the added public matchup representation, using a50/50self-play/tactical mixture. The rationale is that the stronger host may receive a more useful learning signal, not that opponent exposure is a new or established solution. No comparison across F5 and F26 identifies which intervening change mattered; the contemporary self-play control is essential. The F5 sealed result is preserved.

Both arms start from all three sealed F25 final matchup checkpoints (7103, 8209, 9311), preserving weights and resetting copied Adam state identically. This is a common experimental host, not a claim that F25 matchup won. Features, model, action sampler, rewards, PPO and the 120-turn cutoff remain unchanged. The 60,806-parameter policy still lacks an event sequence, explicit beliefs and opponent-prediction heads; this experiment does not resolve those gaps.

The self arm selects 1,024 current-policy self-play rows per cohort. The mixed arm collects complete games in two separately namespaced streams and selects exactly 512 self-play rows plus 512 rows against the frozen tactical heuristic. The current policy is frozen across both collections. Full same-player GAE precedes selection. Only the learner's actions, likelihoods and values enter tactical-game PPO; opponent actions stay in the audit trace. Training seats alternate by episode parity, and a monotone cursor records disjoint per-stream episode ranges. The eight-game collection chunks have a fixed cap of 64 games per stream. Stream-cap exhaustion stops the experiment. Opponent composition changes the actor's experience and critic's target distribution together.

Each arm/seed runs 128 cohorts, two epochs of four 256-row minibatches: 768 cohorts, 6,144 updates, 786,432 selected rows and 1,572,864 row presentations overall. Budgets match rows/updates, not games, decisions or wall time. Midpoint 64 is descriptive; there is no early stopping, endpoint selection, seed dropping, rerun, or extension. Main evaluation at 0/64/128 uses 128 paired schedules in both seats against random, base-power and tactical: 13,824 games. Another 576 self-play games provide terminal-only value diagnostics. Checkpoints at 0/63/64/127/128 give 30 checkpoint comparisons and 12 original-next-update recoveries during replay.

The 27 registered final contrasts are self-versus-initial, mixed-versus-initial and mixed-versus-self, for each of three seeds and three opponents. All retain paired seat clusters, conservative unresolved-cutoff bounds, and the existing approximate family-adjusted bootstrap; conservative Hoeffding intervals are also reported. The learning and mixture-benefit gates use base-power improvement of at least five percentage points, positive adjusted lower bound, no detected random/base-power regression, and final cutoff rates no greater than five percent. No detected regression is not equivalence. Separately named tactical-adaptation gates apply to the trained-on heuristic. A tactical gain is not guaranteed and is not evidence of opponent generalization. Random and base-power remain familiar development opponents. All-seed claims require all three seeds; familiar tactical probes, pooled rates and value diagnostics remain descriptive.

New train/development namespaces and seeds are frozen in `contracts/opponent-mix-v1/plan.json`. Admission verifies the sealed F25 receipt and all source/evidence hashes, then incorporates its complete training, main evaluation and self-play team identities alongside previously admitted history. New development schedules exclude those exact identities. This is not an exhaustive species/set-family holdout or an inventory of every engineering exposure. Final evaluation data remain reserved.

The additive Node session admits the unchanged frozen tactical advisor on train-partition teams. It reuses F25's terminal identity and public-type projection; advice stays outside both player observations. The sealed F25 adapter continues to reject train tactical advice. F25's documented terminal-state and nominal-type limitations remain. The new Python collector excludes all frozen-opponent rows and checks action, likelihood, feature, mask, value and side consistency against the current-policy actor trace.

Qualification includes live same-action parity with F25, train advisor/player-view isolation, rejected final/injected inputs, learner-only rows, independent merged-trajectory GAE, both-seat terminal/cutoff handling, exact deterministic quotas, all three immutable checkpoint loads, optimizer reset/recovery, paired-cutoff/regression gate fixtures and a pinned simulator build. The source-bound F25 88-check evidence is reused for unchanged mechanics/features/learner code. Independent design and implementation review dispositions are retained with the new review.

Production and exhaustive fresh replay must reproduce every update, action/value decision and outcome, all checkpoints and original-next-update recoveries. Completion updates these measured results and current progress documents before first sealing, checks every live/archived source and evidence digest, then requires an identical repeat receipt. Failure preserves evidence without repair or restart. This comparison authorizes no promotion or subsequent experiment.

Full G2/G3/G4 and public strength remain open. No public server, final reserved data or Prime resources. GitHub remains on its existing visibility hold; no publication or upload is performed.

The row/actor-record comparison is an internal consistency check; it is not an independent model inference. The engineering preflight independently recollects from the original unchanged host and reproduces selected-batch and update fingerprints. Production requires exhaustive fresh inference/replay. Initial self-play games can be shared across arms; the same namespace and checkpoint do not force identical selected rows or subsequent trajectories.

Final qualification binds683source/configuration files. The pinned build passes, and the18 focused checks include both frozen development-advisor and train player-view parity. The engineering preflight admitted all predecessor hashes and reproduced both selected batches and original updates exactly; it is not production evidence. See `artifacts/reviews/f26-opponent-mix/CONTINUATION.md` for the persistent production/replay/sealing workflow.

## Measured execution

# F26 measured results

Production completed in 1804.41 seconds on CPU. Replay/receipt verification remains a separate requirement.

- training: 22192 games, 22188 natural terminals, 4 cutoffs, 1270682 attempted decisions.
- evaluation: 13824 games, 13817 natural terminals, 7 cutoffs, 797965 attempted decisions.
- selfplay_diagnostics: 576 games, 576 natural terminals, 0 cutoffs, 36532 attempted decisions.

All-seed learning: {'mixed': False, 'self': False}; all-seed base-power mixture benefit: False; trained-on tactical adaptation benefit: False. Contrasts: {'inconclusive': 26, 'regression': 1}.

| Seed | Arm | Base-power wins: initial → final /256 | Tactical wins: initial → final /256 | Learning gate | Familiar final mastery |
|---|---|---:|---:|---|---|
| 7103 | self | 146 → 140 | 14 → 19 | False | False |
| 7103 | mixed | 146 → 145 | 14 → 19 | False | False |
| 8209 | self | 145 → 148 | 30 → 23 | False | False |
| 8209 | mixed | 145 → 142 | 30 → 23 | False | False |
| 9311 | self | 137 → 142 | 31 → 22 | False | False |
| 9311 | mixed | 137 → 126 | 31 → 10 | False | False |

Peak parent RSS 1867.8 MiB (workers excluded). Collection 1671.44s; optimizer calls including compile 21.89s. Finite gradients/likelihood ratios are engineering telemetry, not trust-region or calibration guarantees.

Both arms use the same F25 final matchup host and reset Adam. Tactical is trained on; its adaptation gate is reported separately. Fresh development schedules against recurring generation and known heuristics, not public ladder or held-out-team generalization. All seeds/27 contrasts retained; intervals and cutoff-aware gates remain in measured-review.json. No promotion, final reserved data, public server, Prime or GitHub publishing.

