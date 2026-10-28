# F14 — entropy ablation on fixed continuation scenarios

## Measured results

F14 completed (2026-10-24): the matched-budget entropy ablation collected 4,512 training and 864 evaluation continuations in 169.41 seconds on local CPU. The preregistered no-entropy benefit gate passed in 0/3 seeds; the descriptive two-point switch-reduction threshold was met in 0/3. All 18 contrasts are retained (0 improvements, 0 regressions, 18 inconclusive under the registered approximate-bootstrap test). Training/evaluation natural terminals: 4,512/864; cutoffs: 0/0. All 14 focused tests and exhaustive replay pass, including six original-next-update recoveries. This adaptively reused development bank does not establish fresh-team or full-battle generalization. F10 remains the reference. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold.

Production: `artifacts/runs/f14-entropy-v1/report.json`. Receipt: `artifacts/reports/f14-entropy.json`. This is a small continuation experiment, not a sustained stability qualification. All three seeds, both arms and all evaluation cells remain in the result.

Parent peak RSS was 800.75 MiB. The 384 Adam steps, including compilation, took 2.716 seconds. Training collected 59,543 learner rows, selected 49,152, and produced 98,304 PPO and 49,152 supervised row presentations. The 139,246 training-prefix replay decisions were excluded from updates. Unavailable-choice responses: 0.

| Activity | Games | Natural terminals | Cutoffs | Wins | Ties | Decisions |
|---|---:|---:|---:|---:|---:|---:|
| training | 4,512 | 4,512 | 0 | 1,518 | 0 | 113,944 |
| evaluation | 864 | 864 | 0 | 193 | 0 | 22,504 |

## Continuation wins and retention

Each outcome entry is wins / natural terminals / cutoffs, out of 48 suffixes per cell. Twelve scenario clusters with four actor repeats form each family. The primary gate also requires retention across all 32 known tactical contexts.

| Seed | Family | Initial | Entropy .01 | Entropy 0 | Seed benefit |
|---|---|---|---|---|---|
| 7103 | midgame | 8 / 48 / 0 | 8 / 48 / 0 | 10 / 48 / 0 | not passed |
| 7103 | endgame | 12 / 48 / 0 | 13 / 48 / 0 | 12 / 48 / 0 | not passed |
| 8209 | midgame | 8 / 48 / 0 | 9 / 48 / 0 | 8 / 48 / 0 | not passed |
| 8209 | endgame | 14 / 48 / 0 | 11 / 48 / 0 | 11 / 48 / 0 | not passed |
| 9311 | midgame | 13 / 48 / 0 | 11 / 48 / 0 | 10 / 48 / 0 | not passed |
| 9311 | endgame | 11 / 48 / 0 | 9 / 48 / 0 | 15 / 48 / 0 | not passed |

| Seed | Entropy .01 retention | Entropy 0 retention | .01 passing contexts | 0 passing contexts |
|---|---|---|---:|---:|
| 7103 | True | True | 32/32 | 32/32 |
| 8209 | True | True | 32/32 | 32/32 |
| 9311 | True | True | 32/32 | 32/32 |

## Fixed-state behavior

These are the same initial-policy observations evaluated under each endpoint, restricted to move-phase states with both move and switch choices. Probability changes are descriptive, not error labels. The mechanism threshold is standard minus no-entropy switch probability >=2 percentage points.

| Seed | Rows | Initial switch % | .01 switch % | 0 switch % | Reduction pp | Threshold |
|---|---:|---:|---:|---:|---:|---|
| 7103 | 807 | 5.57 | 6.74 | 6.54 | +0.20 | False |
| 8209 | 889 | 5.54 | 4.16 | 3.82 | +0.34 | False |
| 9311 | 804 | 5.79 | 5.06 | 8.56 | -3.50 | False |

| Seed | Endpoint | Entropy (nats) | Entropy / log legal actions |
|---|---|---:|---:|
| 7103 | initial | 1.5244 | 0.8314 |
| 7103 | standard | 1.5601 | 0.8512 |
| 7103 | no_entropy | 1.5408 | 0.8421 |
| 8209 | initial | 1.4516 | 0.8044 |
| 8209 | standard | 1.3843 | 0.7688 |
| 8209 | no_entropy | 1.3856 | 0.7669 |
| 9311 | initial | 1.4623 | 0.8087 |
| 9311 | standard | 1.4232 | 0.7846 |
| 9311 | no_entropy | 1.4865 | 0.8166 |

## All preregistered contrasts

Intervals are percentage points. Both bounds are shown for cutoff-aware point estimates; no unresolved outcome is converted to a draw. Small-cluster adjusted-bootstrap intervals do not guarantee familywise coverage; Hoeffding intervals are retained as a conservative comparator. An inconclusive contrast establishes neither equality nor superiority.

| Seed | Family | Contrast | Point bounds | Adjusted bootstrap | Hoeffding | Verdict |
|---|---|---|---|---|---|---|
| 7103 | endgame | no_entropy_vs_initial | [+0.00, +0.00] | [-8.33, +8.33] | [-100.00, +100.00] | inconclusive |
| 7103 | endgame | no_entropy_vs_standard | [-2.08, -2.08] | [-19.22, +12.50] | [-100.00, +100.00] | inconclusive |
| 7103 | endgame | standard_vs_initial | [+2.08, +2.08] | [-14.58, +20.83] | [-100.00, +100.00] | inconclusive |
| 7103 | midgame | no_entropy_vs_initial | [+4.17, +4.17] | [-10.42, +22.92] | [-100.00, +100.00] | inconclusive |
| 7103 | midgame | no_entropy_vs_standard | [+4.17, +4.17] | [-8.33, +16.67] | [-100.00, +100.00] | inconclusive |
| 7103 | midgame | standard_vs_initial | [+0.00, +0.00] | [-12.50, +12.50] | [-100.00, +100.00] | inconclusive |
| 8209 | endgame | no_entropy_vs_initial | [-6.25, -6.25] | [-22.92, +0.00] | [-100.00, +98.47] | inconclusive |
| 8209 | endgame | no_entropy_vs_standard | [+0.00, +0.00] | [+0.00, +0.00] | [-100.00, +100.00] | inconclusive |
| 8209 | endgame | standard_vs_initial | [-6.25, -6.25] | [-22.92, +0.00] | [-100.00, +98.47] | inconclusive |
| 8209 | midgame | no_entropy_vs_initial | [+0.00, +0.00] | [-12.50, +12.50] | [-100.00, +100.00] | inconclusive |
| 8209 | midgame | no_entropy_vs_standard | [-2.08, -2.08] | [-12.50, +8.33] | [-100.00, +100.00] | inconclusive |
| 8209 | midgame | standard_vs_initial | [+2.08, +2.08] | [-12.50, +19.22] | [-100.00, +100.00] | inconclusive |
| 9311 | endgame | no_entropy_vs_initial | [+8.33, +8.33] | [+0.00, +18.75] | [-96.38, +100.00] | inconclusive |
| 9311 | endgame | no_entropy_vs_standard | [+12.50, +12.50] | [+0.00, +37.50] | [-92.22, +100.00] | inconclusive |
| 9311 | endgame | standard_vs_initial | [-4.17, -4.17] | [-20.83, +8.33] | [-100.00, +100.00] | inconclusive |
| 9311 | midgame | no_entropy_vs_initial | [-6.25, -6.25] | [-35.42, +29.17] | [-100.00, +98.47] | inconclusive |
| 9311 | midgame | no_entropy_vs_standard | [-2.08, -2.08] | [-12.50, +8.33] | [-100.00, +100.00] | inconclusive |
| 9311 | midgame | standard_vs_initial | [-4.17, -4.17] | [-27.08, +25.47] | [-100.00, +100.00] | inconclusive |

## Terminal-only value behavior

Only natural terminal outcomes enter these descriptive targets. Decision weighting and game weighting are both reported. Endpoints visit different states and induce different outcomes; lower MSE across endpoints alone is not evidence of better calibration. Cutoff states are excluded.

| Seed | Endpoint | Family | Mean value | Mean terminal target | Decision MSE | Game MSE | Constant .5 decision MSE |
|---|---|---|---:|---:|---:|---:|---:|
| 7103 | initial | endgame | 0.4470 | 0.3471 | 0.1909 | 0.1746 | 0.2500 |
| 7103 | initial | midgame | 0.4234 | 0.1034 | 0.1746 | 0.1596 | 0.2500 |
| 7103 | standard | endgame | 0.3247 | 0.4377 | 0.2069 | 0.1563 | 0.2500 |
| 7103 | standard | midgame | 0.2697 | 0.1207 | 0.1170 | 0.1100 | 0.2500 |
| 7103 | no_entropy | endgame | 0.3223 | 0.3651 | 0.1827 | 0.1581 | 0.2500 |
| 7103 | no_entropy | midgame | 0.2747 | 0.1580 | 0.1284 | 0.1245 | 0.2500 |
| 8209 | initial | endgame | 0.5554 | 0.4318 | 0.2193 | 0.2315 | 0.2500 |
| 8209 | initial | midgame | 0.5045 | 0.1601 | 0.2255 | 0.2154 | 0.2500 |
| 8209 | standard | endgame | 0.4578 | 0.3460 | 0.1826 | 0.1870 | 0.2500 |
| 8209 | standard | midgame | 0.3821 | 0.2132 | 0.1660 | 0.1426 | 0.2500 |
| 8209 | no_entropy | endgame | 0.4691 | 0.3986 | 0.1813 | 0.1858 | 0.2500 |
| 8209 | no_entropy | midgame | 0.3523 | 0.1045 | 0.1407 | 0.1321 | 0.2500 |
| 9311 | initial | endgame | 0.5315 | 0.3674 | 0.1982 | 0.2189 | 0.2500 |
| 9311 | initial | midgame | 0.4636 | 0.2819 | 0.1679 | 0.1668 | 0.2500 |
| 9311 | standard | endgame | 0.4286 | 0.2885 | 0.1776 | 0.1814 | 0.2500 |
| 9311 | standard | midgame | 0.3684 | 0.2029 | 0.1527 | 0.1341 | 0.2500 |
| 9311 | no_entropy | endgame | 0.4148 | 0.4420 | 0.1864 | 0.1835 | 0.2500 |
| 9311 | no_entropy | midgame | 0.3174 | 0.1698 | 0.1429 | 0.1262 | 0.2500 |

## Verification and next decision

Production and exhaustive fresh replay exited 0. Replay reproduced all suffix decisions, selected batches, 192 cohorts, 384 optimizer steps, 4,608 tactical predictions, three fixed observation sets and nine behavior endpoints. It compared three initial and six final checkpoints and loaded the six original before-last checkpoints to reproduce the next updates. The 14-test regression binds all 506 source/configuration files and the CPU FP32 runtime. The pinned Showdown build passed. All ten copied-evidence corruption negatives reject. Independent review findings and checked dispositions are preserved in the review directory. A separate read-only arithmetic audit recomputed all nine saved behavioral matrices and all 4,608 tactical predictions/cell summaries, with maximum behavioral discrepancy below 1e-12.

Retain F10. Neither the primary benefit gate nor the secondary switch-reduction threshold passed in any seed. Removing entropy changes the policy, but does not consistently lower switching or improve continuation outcomes here: switching fell by 0.20 and 0.34 percentage points in seeds 7103/8209 and rose by 3.50 points in seed 9311. All six final endpoints retained all 32 known tactical contexts. This does not establish equivalence, prove entropy harmless in other regimes, or identify the cause of F12's behavior.

The next proposed step is broader tactical imitation on actual full-battle observations, followed by a matched RL fine-tuning comparison against retained F10. Use public-view teacher decisions as demonstrations rather than optimal-action labels; partition new development teams/scenarios before fitting, preserve all seeds, and measure complete-battle transfer plus tactical retention. The narrow template retention results show that those known skills survived, while broader play remains weak. This proposal is a hypothesis to test, not a measured improvement. No further experiment has been launched.

Five progress documents are updated before sealing. Post-seal receipt/source/evidence checks and byte-identical repeat sealing are recorded outside the sealed review directory. Historical handoffs and sealed predecessors are preserved.

## Registered design (preserved preregistration)

## Question and fixed design

F13 found that a cloned entropy-only step increased switching probability, but that observation did not establish the cause of accumulated policy changes. F14 tests entropy coefficient .01 versus 0, with all other learner settings fixed: gamma 1, lambda .95, Adam learning rate .0003, clipping .2, value coefficient .5, rehearsal coefficient .1 and unchanged actor/observations. The .01 control calls the sealed update kernel exactly. Each seed starts from its F10 rehearsal endpoint with reset Adam; no F13 endpoint is selected.

The versioned plan is `contracts/entropy-ablation-v1/plan.json`. Seeds 7103, 8209 and 9311 each run both arms for 32 cohorts of 256 selected rows and two full-batch epochs. Total budgets are 49,152 selected PPO rows, 98,304 PPO presentations, 49,152 supervised presentations and 384 Adam steps. Collection uses complete suffix returns before subsampling. Arms have equal optimizer/selected-row budgets; on-policy game lengths and cursors may differ.

Reuse all 48 sealed F13 starts: 12 fit and 12 probe states in each of midgame and endgame, from official training-partition games. Teacher episodes and exact whole-team hashes are disjoint across the bank's roles. This bank was adaptively chosen after seeing F13; it is not fresh held-out data. Fresh F14 actor keys are shared between arms, while each restored simulator prefix retains its existing chance state. No teacher generation or broader campaign is repeated.

Evaluate initialization and both final policies at all 24 probe states with four actor repeats: 864 evaluation suffixes. Retain all three seeds and 18 contrasts. Pair outcomes by scenario/repeat and resample the 12 scenario clusters per family. Cutoffs receive worst-case bounds, never draw scores. The adjusted percentile bootstrap is approximate and has no guaranteed familywise coverage with these small clusters; conservative Hoeffding intervals are also reported.

Per-seed benefit requires removal of entropy to improve over both control and initialization by at least five points, with adjusted-bootstrap lower bound above zero, in both families, while retaining all 32 known tactical contexts. Overall benefit requires all three seeds. Tactical retention evaluates the 512 already sealed F10 one-decision probes at all nine endpoints; thresholds are greedy accuracy >=.95, target mass >=.60 and mass decline <=.05 in every context. These known templates are not new generalization evidence.

Fixed-state behavior uses every initial-policy probe observation in move phase with both a move and switch legal. Evaluate the same features and masks at all three endpoints and preserve probability arrays. Report conditional switch probability, entropy and entropy normalized by legal-action count. The secondary descriptive threshold is at least two points lower switching probability for the no-entropy arm; it cannot establish stronger play. Probe rows never enter optimizer batches.

## Verification and limits

Run source-bound tests and the pinned simulator build before freezing sources, runtime and plan. Production runs once; a fresh exhaustive replay must reproduce every suffix decision, target, selected batch, update, tactical prediction and fixed-state probability. It checks three initial and six final checkpoints and resumes six original before-last checkpoints to reproduce the next update. Preserve failures without automatic restart or budget changes. Record measured terminal outcomes and terminal-only value behavior for all endpoints, including regressions.

No full-battle or public-strength policy promotion follows this canary. Retain F10 pending a broader development comparison. Full G2/G3/G4 remain open. Final reserved data, public servers and Prime are unused; the GitHub visibility hold remains unresolved.

