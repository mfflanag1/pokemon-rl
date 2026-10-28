# F13 — credit assignment and material-advantaged continuations

## Measured results

F13 completed (2026-10-24): the credit-assignment and continuation canary collected 4,536 training and 864 evaluation continuations in 203.34 seconds on local CPU; all ended naturally. The registered lambda=1 benefit gate did not pass (0/3 seeds); all 18 contrasts were inconclusive. No new policy is promoted; F10 remains the reference. Independent return calculations agree with sealed GAE, and fixed-state probes confirm that F12 increased switching propensity. The new public-state-selected bank has 48 midgame/endgame starts from 192 official train-partition teacher games. Both arms received equal selected-row and optimizer-step budgets. All 15 focused tests and exhaustive replay pass, including six original-next-update recoveries. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold.

Production report: `artifacts/runs/f13-credit-v1/report.json`. Receipt: `artifacts/reports/f13-credit.json`. Elapsed time is measured inside the campaign, including diagnostics, teacher generation, collection and updates; it is not a two-hour stability qualification. Parent peak RSS was 981.84 MiB. All 384 optimizer steps, including compilation, took 1.661 seconds; this measurement does not justify moving this small learner to a GPU.

| Activity | Games | Natural terminals | Cutoffs | Decisions |
|---|---:|---:|---:|---:|
| Full-battle diagnostics | 288 | 288 | 0 | 14,169 |
| Training suffixes | 4,536 | 4,536 | 0 | 113,434 |
| Evaluation suffixes | 864 | 864 | 0 | 22,203 |

The 192 teacher games also ended naturally. Training collected 59,265 learner rows and selected 49,152 after complete-return calculation; two epochs produced 98,304 PPO row presentations, 49,152 rehearsal presentations and 384 Adam steps. The 137,757 training-prefix replay decisions were excluded from learner rows. Thirty cloned gradient steps were diagnostic only. Training won 1,448/4,536 (31.92%). Four official trapped-switch feedback responses occurred in diagnostics; the existing request-refresh handling completed those games. No unavailable-choice responses occurred in training or evaluation.

The 48 starts covered turns 8–55 (median 11), with 12 per family/role and one per teacher episode. Fit/probe teacher episodes and exact whole-team hashes did not overlap in this generated bank. Both roles still come from the training partition; species/mechanics and previously encountered teams may recur elsewhere. This is not reserved held-out-team evaluation. Material advantage was selected before outcomes, and all selected states remained in the result.

### Continuation wins

Each entry is wins out of 48 suffixes: 12 fixed states with four actor repeats. All 864 suffixes ended naturally; there were no ties. The independent units for the registered intervals are the 12 scenarios, not 48 independent starting states.

| Seed | Family | F10 initial | GAE λ=.95 | λ=1 | Seed benefit gate |
|---|---|---:|---:|---:|---|
| 7103 | midgame | 13 | 7 | 8 | not passed |
| 7103 | endgame | 16 | 15 | 14 | not passed |
| 8209 | midgame | 10 | 10 | 11 | not passed |
| 8209 | endgame | 12 | 13 | 13 | not passed |
| 9311 | midgame | 11 | 8 | 10 | not passed |
| 9311 | endgame | 12 | 13 | 12 | not passed |

| Seed | Family | Contrast | Point gain (pp) | Adjusted bootstrap interval (pp) | Verdict |
|---|---|---|---:|---|---|
| 7103 | endgame | gae95_vs_initial | -2.08 | [-12.50, +8.33] | inconclusive |
| 7103 | endgame | lambda1_vs_gae95 | -2.08 | [-14.58, +12.50] | inconclusive |
| 7103 | endgame | lambda1_vs_initial | -4.17 | [-16.67, +8.33] | inconclusive |
| 7103 | midgame | gae95_vs_initial | -12.50 | [-35.42, +0.00] | inconclusive |
| 7103 | midgame | lambda1_vs_gae95 | +2.08 | [-8.33, +12.50] | inconclusive |
| 7103 | midgame | lambda1_vs_initial | -10.42 | [-35.42, +6.25] | inconclusive |
| 8209 | endgame | gae95_vs_initial | +2.08 | [-10.42, +14.58] | inconclusive |
| 8209 | endgame | lambda1_vs_gae95 | +0.00 | [-27.08, +18.75] | inconclusive |
| 8209 | endgame | lambda1_vs_initial | +2.08 | [-16.67, +14.58] | inconclusive |
| 8209 | midgame | gae95_vs_initial | +0.00 | [-14.58, +14.58] | inconclusive |
| 8209 | midgame | lambda1_vs_gae95 | +2.08 | [-12.50, +16.67] | inconclusive |
| 8209 | midgame | lambda1_vs_initial | +2.08 | [-16.67, +20.83] | inconclusive |
| 9311 | endgame | gae95_vs_initial | +2.08 | [-8.33, +12.50] | inconclusive |
| 9311 | endgame | lambda1_vs_gae95 | -2.08 | [-12.50, +8.33] | inconclusive |
| 9311 | endgame | lambda1_vs_initial | +0.00 | [-8.33, +8.33] | inconclusive |
| 9311 | midgame | gae95_vs_initial | -6.25 | [-22.92, +8.33] | inconclusive |
| 9311 | midgame | lambda1_vs_gae95 | +4.17 | [-12.50, +22.92] | inconclusive |
| 9311 | midgame | lambda1_vs_initial | -2.08 | [-22.92, +18.75] | inconclusive |

All reported Hoeffding intervals include zero. The tiny-cluster percentile bootstrap is exploratory and does not guarantee familywise coverage. Failure to establish a difference does not establish equality. No checkpoint/seed was selected or discarded.

### What the diagnostics establish

The independently computed returns matched the sealed implementation in every diagnostic and training cohort. The largest discrepancy across the nine diagnostic cohorts was 1.467203671e-07. This excludes a detected implementation mismatch on these trajectories, not every possible learner defect.

| Seed | Same fixed observations | F10 switch probability | F12 official | F12 curriculum |
|---|---:|---:|---:|---:|
| 7103 | 526 | 9.27% | 33.74% | 34.82% |
| 8209 | 568 | 6.19% | 21.68% | 26.07% |
| 9311 | 565 | 8.45% | 23.09% | 38.22% |

These are probabilities on identical F10-visited move-phase observations with both move and switch options, so the increase is a policy change on those states rather than solely a change in visited-state composition. They are not automatically incorrect-switch labels.

| Endpoint | Losing-chain rows | Positive GAE advantages | Losing-chain voluntary switches | Positive GAE switch advantages |
|---|---:|---:|---:|---:|
| initial | 2314 | 1 | 131 | 0 |
| official | 2441 | 40 | 442 | 5 |
| curriculum | 2531 | 81 | 539 | 18 |

All λ=1 terminal-loss advantages were nonpositive. Most GAE terminal-loss advantages were also nonpositive, including almost all losing switches. The evidence does not support the simple explanation that widespread positive credit for losing switches alone caused F12's change.

| Seed | Target | Actor Δswitch (pp) | Value | Entropy | Rehearsal | Joint |
|---|---|---:|---:|---:|---:|---:|
| 7103 | gae95 | -0.2520 | +0.1046 | +0.7440 | -0.0307 | +0.0569 |
| 7103 | lambda1 | -0.2847 | +0.1025 | +0.7440 | -0.0307 | -0.0102 |
| 8209 | gae95 | -0.0052 | +0.0133 | +0.3328 | -0.0618 | -0.0286 |
| 8209 | lambda1 | +0.0034 | +0.0133 | +0.3328 | -0.0618 | -0.0215 |
| 9311 | gae95 | +0.3040 | +0.1247 | +0.4496 | -0.0163 | +0.2409 |
| 9311 | lambda1 | +0.3403 | +0.1246 | +0.4496 | -0.0163 | +0.2886 |

Each component used one cloned zero-Adam step on the same sampled rows and fixed observations. Entropy alone increased switch probability in all three seeds, as did the shared value update; the actor and joint effects varied. Adam deltas are nonadditive. These local probes nominate hypotheses for a controlled ablation; they do not identify the cause of the accumulated F12 changes.

### Terminal-only value behavior

Descriptive decision-weighted MSE uses only naturally completed suffixes. Constant 0.5 has MSE 0.25 in every cell. The fitted constant uses the outcome rate of the same cell and is an explanatory comparator, not a deployable model or independent evaluation. Policies visit different states and have different outcomes, so lower MSE across endpoints alone does not establish better calibration.

| Seed | Endpoint | Family | Mean prediction | Mean terminal target | Model MSE | Same-cell constant MSE |
|---|---|---|---:|---:|---:|---:|
| 7103 | initial | endgame | 0.4456 | 0.4015 | 0.2087 | 0.2403 |
| 7103 | initial | midgame | 0.4272 | 0.2393 | 0.1912 | 0.1820 |
| 7103 | gae95 | endgame | 0.2875 | 0.4186 | 0.2259 | 0.2434 |
| 7103 | gae95 | midgame | 0.2605 | 0.1027 | 0.0972 | 0.0921 |
| 7103 | lambda1 | endgame | 0.2922 | 0.3272 | 0.1748 | 0.2201 |
| 7103 | lambda1 | midgame | 0.2503 | 0.1199 | 0.1059 | 0.1056 |
| 8209 | initial | endgame | 0.5604 | 0.3779 | 0.2143 | 0.2351 |
| 8209 | initial | midgame | 0.5059 | 0.1967 | 0.2113 | 0.1580 |
| 8209 | gae95 | endgame | 0.4438 | 0.3546 | 0.2044 | 0.2288 |
| 8209 | gae95 | midgame | 0.3832 | 0.2590 | 0.1633 | 0.1919 |
| 8209 | lambda1 | endgame | 0.3759 | 0.2851 | 0.1723 | 0.2038 |
| 8209 | lambda1 | midgame | 0.3737 | 0.2772 | 0.1663 | 0.2003 |
| 9311 | initial | endgame | 0.5082 | 0.3823 | 0.2006 | 0.2361 |
| 9311 | initial | midgame | 0.4564 | 0.2892 | 0.1983 | 0.2056 |
| 9311 | gae95 | endgame | 0.4071 | 0.4329 | 0.1955 | 0.2455 |
| 9311 | gae95 | midgame | 0.2908 | 0.2174 | 0.1359 | 0.1702 |
| 9311 | lambda1 | endgame | 0.3983 | 0.3716 | 0.1837 | 0.2335 |
| 9311 | lambda1 | midgame | 0.3156 | 0.2068 | 0.1183 | 0.1640 |

### Completion and next decision

Production and fresh replay both exited 0. Replay reproduced all 192 teacher games, 288 diagnostic games, 4,536 training suffixes, 864 evaluation suffixes, 192 update cohorts and all optimizer metrics/parameters. It checked three initial and six final checkpoints, then loaded the six original before-last checkpoints and reproduced their final updates. Fifteen source-bound tests pass; the independent return implementation and campaign design received cross-model review. All ten copied-evidence corruption negatives passed before sealing. Receipt and archived source/evidence integrity are checked separately, with repeat sealing required to produce identical bytes.

Retain F10. The new starts produce winning training trajectories, but changing λ alone did not establish a learning benefit. The next bounded experiment should isolate entropy regularization, tracking switch probability, tactical retention and continuation outcomes under matched budgets. A full-battle development comparison must precede any policy promotion. The value/shared-encoder interaction remains a separate hypothesis, not a diagnosed cause. No additional experiment has been launched.


Status: production and exhaustive replay completed; learning benefit gate not passed (0/3 seeds). F10 retained.

F12 increased voluntary switching and failed to establish tactical improvement. F13 starts from the retained F10 rehearsal endpoints. It tests a narrower question: does reducing critic bootstrapping help learning in later, more favorable positions? This stage cannot promote a full-battle policy.

## Registered design

The frozen specification is `contracts/credit-audit-v1/plan.json`. Three fixed seeds, both arms, all evaluation cells and any failures must remain in the result.

1. Collect 288 fresh official train-partition diagnostic games: 32 against tactical for each seed and each of F10 initialization, F12 official and F12 curriculum. Independently reconstruct every linked return. Compare switching probabilities for all endpoints on identical observations visited by F10, with both move and switch choices legal. Record 30 cloned single-step component probes; these are local diagnostics, not policy training or an additive attribution of F12's behavior.
2. Generate exactly 192 teacher games using the frozen public-view tactical policy at both seats. Before seeing the outcome, select 12 fit and 12 probe states for each of midgame and endgame, from separate teacher episodes. Select joint move states at turns 8–119 where own remaining Pokémon are at least the publicly inferred opponent count. Unknown opponent slots count alive. One state per teacher game, no outcome filtering and no quota extension. Material advantage does not prove a position is winnable.
3. Train two equal-budget continuations from each seed's F10 parameters with reset Adam. Both keep the same policy, observation, reward, tactical opponent and sealed F7 rehearsal. Each receives 32 cohorts of 256 selected rows and two full-batch optimizer steps per cohort. The sole learner difference is λ=.95 versus λ=1. Complete suffix returns are computed before row subsampling; teacher prefix actions and probe rows never enter updates.
4. Evaluate initialization and both final policies on all 24 probe states with four fresh actor repeats each: 864 evaluation continuations. Pair within state/repeat, resample the 12 scenario clusters per family and retain all 18 contrasts. Cutoffs receive worst-case outcome bounds. The registered benefit requires both comparisons (λ=1 versus initialization and versus λ=.95) to improve by at least five points with an adjusted bootstrap lower bound above zero, in both families for all three seeds. Small-sample bootstrap intervals remain approximate; distribution-free intervals are also reported.

## Credit calculation

With γ=1 and zero intermediate rewards, a linked target is `(1−λ) V(next) + λ target(next)`. The last row uses the actual terminal reward, or the recorded bootstrap for an administrative cutoff. Independent NumPy calculations are compared with the unchanged sealed float32 GAE implementation at a tolerance of 2e-6.

At λ=1, every row in a naturally completed game receives that game's outcome. A losing terminal chain therefore cannot have a positive advantage under a bounded nonnegative critic. A positive λ=.95 advantage in a losing game can arise from the critic's estimates; it does not by itself demonstrate an implementation bug. Cutoff targets remain bootstrapped even at λ=1 and must not be described as observed terminal returns.

## Verification and limits

The source, runtime and plan freeze before production. Fresh verification repeats every teacher game, diagnostic decision, continuation, selected batch, optimizer update and evaluation result. It checks all initial/final checkpoints and resumes the six original saved before-last checkpoints for the final updates. All scientific records must match exactly. Source-bound tests and the pinned simulator build precede freezing.

Fit and probe teacher episodes are disjoint but all teams come from the training partition and may recur across roles. This is a conditional continuation canary, not held-out-team, full-battle or public-ladder generalization. F10 remains the reference. Full G2/G3/G4 remain open. Final reserved evaluation, public servers and Prime are unused. The unresolved GitHub visibility hold remains in effect.

Preflight numerical check: 108 long chains, including 36 factual F12 trajectories, produced a maximum absolute return discrepancy of 3.8819435266290725e-07. The tolerance remains 2e-6. Fable reviewed the design; the independent return calculator received separate cross-model QA. Review findings and verified dispositions are preserved in `artifacts/reviews/f13-credit`.
