# F27 — original-batch policy-gradient audit

Status: completed and independently reproduced on 25 October 2026. This is a local diagnostic, not a new learning or strength result.

## Measured result

All 18 original cohorts reproduce exactly: **520 previously recorded training-game executions, 29,586 decisions and 18,432 selected rows**. Every game ended naturally; cutoffs are zero. The audit reproduces 144 original optimizer steps and 12 complete next checkpoints, and computes 288 additional diagnostic clone steps. It creates no new trajectories, candidate checkpoint, evaluation result or strength gain.

The original optimizer moves in an actor descent direction on **144/144** measured minibatches and reduces their actor loss on **144/144**. All 18 complete batches improve their frozen-target actor surrogate. Component-gradient reconstruction differs by at most **1.40e-8** per element. The largest original gradient norm is 0.17594, below the norm-clipping threshold of 1. These observations support correct local optimization on this sample; they do not prove policy improvement away from the sampled data or exclude every learner defect.

Removing the current shared-trunk value gradient changes the original eight-step actor-surrogate improvement by **−0.31% to +5.57%** across anchors. Removing the current entropy gradient changes it by **−0.73% to +3.46%**. These percentages are relative changes in the local loss improvement, not win-rate percentage points. Mean shared actor/value gradient cosine is **−0.00168**. This sample does not show large, systematic critic-gradient opposition. It cannot establish that value sharing or entropy is harmless over longer training.

Different physical-game groups give less consistent actor directions: **42/72** half comparisons have negative cosine, with mean **−0.0697** and median **−0.0299**. The repeated splits overlap and the 18 anchors are correlated snapshots; these counts have no independent-binomial confidence interpretation. Differences in game/phase composition and sparse outcomes can contribute. This is a reason to test collection-batch size, not proof that noise is the sole bottleneck.

| Seed | Arm | Mean game-half cosine over 12 splits | Negative splits /12 | Tactical wins at cohorts 0 /63 /127 |
| --- | --- | ---: | ---: | --- |
| 7103 | self | +0.0477 | 4 | not applicable |
| 7103 | mixed | +0.0624 | 6 | 3 / 3 / 2 |
| 8209 | self | -0.0294 | 5 | not applicable |
| 8209 | mixed | -0.2767 | 10 | 2 / 4 / 2 |
| 9311 | self | -0.0570 | 7 | not applicable |
| 9311 | mixed | -0.1649 | 10 | 3 / 2 / 2 |

Every sampled mixed batch contains **24 tactical games and 2–4 tactical wins**; all those games and wins contribute selected rows. Thus none of these nine batches supports the proposed zero-win explanation. Tactical positive-advantage fractions are 10.55–22.27%, and between-game advantage variance shares are 59.05–79.02%. Both signs and within-game variation matter; terminal rewards do not imply identical advantages throughout a game. These are conditional training diagnostics, not development win rates.

## Next bounded comparison

The best-supported next hypothesis is that collecting more games under each unchanged policy before an update may produce more consistent learning. Preregister a comparison of the current collection-batch size against a larger on-policy collection batch, holding the model, 50/50 self-play/tactical row mixture, loss, minibatch size, total selected rows, row presentations and optimizer-step budget fixed. Report differing game/collection costs and changed policy-refresh cadence explicitly. Keep all three hosts/seeds, fresh development battle outcomes and regressions as primary evidence. Do not pool stale trajectories with action-only importance ratios or simultaneously change game weighting, representation or entropy.

This remains a proposed intervention, not a diagnosed cure or an automatically launched campaign. F27 supplies no basis for policy promotion. Representation, state coverage, long-horizon credit and opponent generalization remain unresolved; a larger batch must demonstrate better complete-battle outcomes to be useful. The original goal remains a strong full-battle RL policy before search.

## Execution and verification

Production took **95.85 measured seconds on local CPU**; parent peak RSS was **795.28 MiB**, excluding Node workers. Collection was 22.44s and diagnostic computations including compilation were 13.67s. These components exclude other startup, checks and I/O. Independent update verification took 15.36s. This is not a two-hour qualification or a hardware-scaling benchmark.

All 13 focused tests pass, including four completion-corruption negatives within one test. The pinned Showdown build passes. Qualification binds 692 source/configuration files. Production freshly reconstructs the selected games and compares the entire original ledgers; the independent verifier reloads the original-bound selected batches and reruns every original/diagnostic update and summary. No unrelated regression or full predecessor campaign is repeated.

The delegated design received independent Codex QA; unsupported noise/sample-size claims were rejected. Two Fable implementation reviews received source-backed dispositions and added mixed-stream/normalization tests. Static opinions are not treated as executed checks. The [completion receipt](../artifacts/reports/f27-gradient-audit.json) binds measured evidence and document snapshots; post-seal integrity and identical-repeat records stay outside the sealed review directory.

## All fixed anchors

Relative columns compare the eight-step actor-surrogate improvement to the original joint update. All are local frozen-batch diagnostics.

| Seed | Arm | Cohort | Joint surrogate improvement | Block current shared value: relative change | Omit current entropy: relative change | Mean game-half cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7103 | self | 0 | 0.005142 | +2.10% | +0.40% | +0.2560 |
| 7103 | self | 63 | 0.003922 | +0.98% | +0.86% | +0.0857 |
| 7103 | self | 127 | 0.002585 | +0.87% | +0.89% | -0.1987 |
| 7103 | mixed | 0 | 0.005736 | +5.57% | -0.73% | +0.3777 |
| 7103 | mixed | 63 | 0.004981 | +0.76% | -0.30% | +0.1381 |
| 7103 | mixed | 127 | 0.002474 | +0.37% | -0.64% | -0.3285 |
| 8209 | self | 0 | 0.005623 | +2.59% | -0.52% | +0.1110 |
| 8209 | self | 63 | 0.002886 | -0.31% | +1.27% | +0.0031 |
| 8209 | self | 127 | 0.002401 | +0.95% | -0.57% | -0.2023 |
| 8209 | mixed | 0 | 0.005601 | +3.74% | -0.12% | -0.3549 |
| 8209 | mixed | 63 | 0.001904 | +0.76% | +3.46% | -0.0777 |
| 8209 | mixed | 127 | 0.001577 | +0.54% | -0.44% | -0.3976 |
| 9311 | self | 0 | 0.005081 | +2.39% | +2.30% | -0.1522 |
| 9311 | self | 63 | 0.002915 | +0.96% | +1.00% | -0.0766 |
| 9311 | self | 127 | 0.002530 | +0.14% | +0.50% | +0.0578 |
| 9311 | mixed | 0 | 0.004512 | +4.72% | +0.11% | -0.3645 |
| 9311 | mixed | 63 | 0.002334 | +0.70% | +1.16% | -0.1299 |
| 9311 | mixed | 127 | 0.001688 | +0.49% | +2.46% | -0.0004 |

## Registered diagnostic and limits

F26's opponent mixture reduced tactical value error without demonstrating stronger play. F13 already checked returns and component updates on older policies, and F14's entropy comparison was inconclusive. F27 asks a narrower question about the actual F26 updates: how do the actor, value and entropy gradients interact, and how consistent are actor gradients across different games in each batch?

The [fixed plan](../contracts/gradient-audit-v1/plan.json) selects original cohorts 0, 63 and 127 in both F26 arms and all three seeds. It reconstructs 520 already-recorded training games, 29,586 decisions and 18,432 selected rows. No new games, new evaluation panel, training continuation or policy promotion is included. All 520 original games ended naturally.

Every reconstructed game, action/value trace, selection and batch fingerprint must equal sealed F26. The original eight-step updates must reproduce all metrics and final parameter hashes; the 12 available next checkpoints must also match parameters, Adam moments and counters. The audit then independently reruns all updates from saved, original-bound batches. This reuses F26's existing full replay evidence rather than repeating that entire campaign.

Each anchor has three independent eight-step clones: the original joint objective; a variant removing only the current shared-trunk value gradient; and a variant removing only the current entropy gradient. All use the original frozen targets, old likelihoods, minibatch order and inherited Adam moments. Cohort 0 starts with zero moments; cohorts 63 and 127 retain prior component history. Removal happens before global clipping and Adam, so changes are nonadditive and can affect more than the directly edited gradient entries. Clone parameters are fingerprinted for reproduction, not saved as candidate policies.

Geometry is measured along all 144 original minibatch updates, with coefficient-weighted components and separate shared/policy/value parameter groups. Actual Adam motion and same-minibatch actor-loss changes accompany gradient cosines. Each anchor also has four game-disjoint actor-gradient comparisons at its starting parameters. Both seats of a physical game stay together, and game splits are stratified by stream. Row-weighted halves can have different row counts. The 72 overlapping splits are descriptive agreement statistics, not independent replications, formal SNR estimates or causal diagnoses. Game/seat/stream identities are checked against each selected actor record.

There is no learning gate or automatic follow-on experiment. Same-batch surrogate gains, low value error or gradient conflict do not establish battle improvements. Report every anchor and seed. Any proposed remedy still needs its own controlled full-battle comparison. Full G2/G3/G4 and public strength remain open. No reserved final data, public server or Prime use; GitHub remains under the existing visibility hold.

Implementation tests cover independent objective arithmetic, finite-difference gradients, component/head isolation, invalid padding, exact original updates, inherited optimizer history, physical-game grouping, selected-row identity and corrupted completion evidence. Review dispositions, the source-bound 13-test qualification and measured results are preserved before sealing.

Production: `artifacts/runs/f27-gradient-audit-v1`. Review: `artifacts/reviews/f27-gradient-audit`. Receipt: `artifacts/reports/f27-gradient-audit.json` after successful execution and verification.
