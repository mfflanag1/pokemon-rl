# M6 — live-game learner continuation

Status: the fixed CPU engineering canary passed. The immutable completion receipt is `artifacts/reports/m6-live-learner.json`; sustained qualification and playing strength remain o|pen.

The admitted game remains the restricted two-member starter-team profile with two moves per member, switching and hidden speed sets. This additive version integrates the sealed M5 60-turn simulator with a new 1,187-feature player-history encoder, the sealed four-action categorical sampling law, and a sigmoid value head. It does not load M3/M4 weights or reinterpret their checkpoints. All M3–M5 source, handoffs and evidence remain immutable.

## Collection and learning contract

Each fixed 32-game cohort occupies five 12-turn windows. Games that remain live carry every state field, PP, hidden set, active member, history event, game ID and chance address into the next update. The actor receives only the exact M5 PlayerView. Features flatten own state, own PP, opponent public state, context and all 128 history events, then divide by 128; the legal-action mask stays separate.

A window ends only after faint replacements are resolved, at a fresh joint request at turn 13, 25, 37, 49 or 61. Terminal lanes freeze. A new cohort is allocated only in the transaction following the fifth window. At turn 61, an unfinished game is an administrative cutoff, with a collecting-policy bootstrap; it is never relabeled as a terminal result or draw.

PPO uses a shared 32-unit tanh trunk, bounded sigmoid value, two full-batch Adam epochs, learning rate .001, clip .2, entropy coefficient .01, value coefficient .5, gamma 1 and lambda .95. Same-player GAE skips forced-wait and padding rows. It stops at each window and uses the frozen collecting policy's afterstate value. The next window uses the updated policy; no GAE trace or gradient crosses a window. Newly terminal games pay one win reward and one loss-zero terminal row, once. Empty windows advance the window cursor without changing parameters, Adam moments/step or update count.

Checkpoints support completed-window/update boundaries, including live and empty cohorts. They bind parameters, optimizer state, counters, seed, IDs and every environment/history array. Partial rollouts are unsupported. Production stops after ten windows per seed. A separately labeled, discarded endpoint recovery probe may compute the next window; that resulting session cannot be saved or admitted as production input.

## Measured production result

The fixed run completed on 2026-10-13 in **11.283 seconds** (launcher exit 0; 11.672 seconds including process startup). This includes generation-time recovery probes and scalar checks, so it is not a pure training-throughput benchmark.

| Seed | Games | Decisions | Optimizer updates | Terminal after turn 12 | Cutoffs |
|---|---:|---:|---:|---:|---:|
|431|64|1,799|4|33|0|
|641|64|2,128|5|48|0|
|853|64|2, 133|5|52|0|
|Total|192|6,060|14|133|0|

All 192 production games reached a terminal outcome. Thirty collection windows produced 14 optimizer updates (28 Adam steps); 16 empty windows skipped Adam entirely. There were 1,385 decisions after turn 12 and 402 with histories longer than 32 events. Scalar replay covered 13,440 padded frames and 26,880 player views, including 346 forced-wait rows, with zero feature, mask or final-state mismatches.

Collected values stayed in **[0.494599879, 0.500599921]**. Targets stayed in **[0.000000000, 1.000000000]**, within the admitted FP32 tolerance, and the largest gradient norm was **0.084259011**. These observations establish short-run finite behavior only.

All 21 checkpoint pairs reproduced the original next-session, buffer, final environment and metric fingerprints. Generation used 42 discarded checkpoint audit transactions (576 replay game-start events and 15,506 decisions, excluded from production counts). Each full acceptance audit independently performs 51 discarded transactions: 30 trajectory replays and 21 checkpoint replays. Endpoint audit sessions are excluded from production checkpoint admission.

The source-bound regression passed **527 tests**: 240 Node, 236 inherited Python, 21 bounded-value, 13 continuation-simulator and 17 new live-learner tests, covering 392 source/configuration files. Eight copied-evidence corruptions were rejected. The M3, M4 and M5 receipts passed 2,712 live/archive hash checks before production. Initial failed tests and the superseded regression whose source tree changed are preserved; they are not acceptance evidence. Two fixed preflights are separate from the single production run.

## Verification and evidence

The fixed canary uses seeds 431, 641 and 853, two production cohorts per seed. Scalar replay checks every padded collection frame, both player encodings/masks and the final environment. Checkpoints at windows 0, 1, 2, 3, 4, 5 and 10 compare original and resumed next-session, buffer, final-state and metric fingerprints. Production counts exclude these discarded recovery probes.

The test suite covers analytic NumPy/JAX network-gradient parity in the smooth actor region, hidden-opponent perturbations in both seats, public/late-history sensitivity, batch permutation, hand-calculated forced-wait GAE, single terminal rewards, live cutoff bootstrap, empty-window Adam immutability, cohort refill, transaction rollback, incompatible/corrupt checkpoints, endpoint budget rejection and review/launcher evidence gates. The categorical rare-mass implementation is inherited; the NumPy gradient oracle does not claim to cover that floor region.

The independent Codex design and implementation reviews and their dispositions are stored under `artifacts/reviews/m6-live-learner`. Claude's documented Opus fallback also hit the weekly quota; cross-family review is unavailable, not passed. Static reviewer output is not test execution.

Fresh Showdown campaigns are not rerun here: simulator mechanics rely on the sealed M5 144-game, unpatched-reference evidence. This milestone independently replays M6 collections against that scalar simulator. It does not expand the admitted mechanics or establish general simulator completeness.

## Scope and next step

This is a CPU engineering canary, not a strength screen or sustained qualification. Full G2/G3/G4 and public ladder strength remain open. Final reserved data, public servers and Prime are unused. The next step is the new policy/continuation combination's own sustained CPU qualification before any separately designed learning screen.

GitHub publication remains on the existing unresolved visibility hold; no new files or run data were pushed. Older README/progress documents are sealed source evidence and were preserved; this handoff records the new milestone.
