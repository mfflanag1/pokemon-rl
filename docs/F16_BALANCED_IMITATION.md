# F16 — balanced full-battle demonstrations

Status: experiment completed; coverage and strength gates failed (0/3 seeds each).

F16 completed (2026-10-24): balanced full-battle imitation reused 512 sealed teacher games and executed 6,144 new BC updates, 128 fresh teacher-probe games and 6,912 full-battle evaluation games in 314.59 seconds on local CPU. Both the descriptive coverage screen and the strength gate failed in all three seeds. Tera agreement improved to 39–43%, but ordinary-move agreement fell to 33–34% and voluntary-switch agreement reached only 3–7%. Tactical-opponent wins for initial/uniform/balanced were 40/49/38 out of 768 each (descriptive totals). All 27 registered contrasts remain: 3 improvements favoring uniform over initial, 1 balanced-versus-uniform regression against random, and 23 inconclusive results. Evaluation retains 6,907 natural terminals and five unresolved cutoffs. All 16 focused tests and exhaustive fresh replay pass, including 12 checkpoint comparisons and three original-next-update recoveries. F10 remains the reference; no PPO follow-on or policy promotion. Full G2/G3/G4 and public strength remain open. Final reserved data, public servers and Prime were unused; GitHub remains held.

## Interpretation and next step

Both the descriptive coverage screen and the full-battle strength gate failed in all three seeds. Compared with the sealed uniform controls at the same 2,048-step-per-seed accumulated BC budget, balanced sampling raised fresh-probe greedy Tera agreement from the uniform control's 0% to 39.34–42.62%, with target mass above 24%. Ordinary-move accuracy fell from about 44% to 32.55–34.11%, exceeding the five-point preservation tolerance. Voluntary-switch agreement was only 3.27–7.37%, below the fixed 25% threshold. Forced-switch preservation and all 32 known tactical contexts passed. Initial Tera agreement was 27.46%, 24.18% and 20.08%; balanced gains over initial were 13.52, 15.16 and 22.54 percentage points. This supplies context for the uniform control's near-absent Tera imitation. Teacher agreement is against a heuristic, not optimal labels.

Tactical full-battle wins for initial/uniform/balanced were 14/18/12 (7103), 8/11/8 (8209), and 18/20/18 (9311), each out of 256. The pooled descriptive totals are 40/49/38 out of 768; no pooled inferential test is claimed because the teams and mechanics clusters recur across seeds. No balanced-versus-initial or balanced-versus-uniform tactical strength benefit was established. Across 27 registered approximate adjusted-bootstrap contrasts there were three improvements, one regression and 23 inconclusive results. The three improvements favor uniform over initial (random 7103/8209 and base_power 8209). Balanced versus uniform regressed against random in seed 8209: cutoff-aware difference −9.77 to −9.38 percentage points, approximate adjusted bootstrap −16.80 to −2.34 points. This is one registered cell, not proof of a universal regression. Conservative Hoeffding intervals are retained. The adjusted bootstrap tail probability is 0.05/(2×27), roughly 18.5 of 20,000 resamples per tail. Improvement/inconclusive classifications near zero are sensitive to Monte Carlo resolution; seed 9311 uniform-versus-initial base-power has a lower bound exactly zero. The registered results are retained without rerunning the bootstrap or changing the statistics seed.

The completed production reused 512 sealed teacher training games (26,278 accepted decisions) and executed 6,144 new BC updates, 786,432 row presentations, 128 fresh teacher-probe games and 6,912 full-battle evaluation games. Probe: 128 natural terminals, no cutoff, 6,533 decisions. Evaluation: 6,907 natural terminals, five unresolved cutoffs, 435,299 decisions. Production elapsed 314.59 seconds on CPU; update time including compile/sampling/validation 32.54 seconds; peak parent RSS 1,082.48 MiB excludes worker memory. No new PPO updates occurred; the uniform checkpoints reuse sealed F15 training and replay evidence.

After production, a descriptive fixed-checkpoint diagnostic evaluated both arms on the entire existing 512-game fit corpus. There were no new games or updates and no changes to gates or selection. Balanced voluntary-switch fit accuracy was 63.08–72.64%, versus 3.27–7.37% on the fresh probe; Tera fit accuracy was 81.36–83.10%, versus 39.34–42.62% fresh. All are equal-game family metrics. Common ordinary moves also show a fit-to-probe gap: balanced 42.13–43.71% to 32.55–34.11%, uniform 55.86–56.54% to 44.29–44.40%. The voluntary-switch probe contains only 82 decisions from 57 games; these correlated, game-weighted observations do not justify a simple independent-binomial confidence interval. The gap supports investigating transfer from the sampled training states; it does not identify whether representation, dataset diversity, optimization or label ambiguity causes it. Ordinary moves also lose fit accuracy, consistent with a tradeoff under this sampling change but not a proven mechanism. A claim that the model cannot fit rare choices at all would be contradicted by these results.

Next proposed bounded step: audit voluntary-switch representation and observable-state label ambiguity, then preregister a controlled data-diversity or encoding change based on that audit. Do not launch another PPO continuation or promote the balanced policy on this evidence. F10 remains the reference. This is fresh development-game confirmation of an adaptively chosen hypothesis; exact teams exclude the training partition and F15 development panels but species, mechanics and known opponents recur. It does not establish novelty against every older experiment or public/generalization strength. Full G2/G3/G4 remain open. Final reserved data, public servers and Prime were unused. GitHub remains held pending the existing visibility choice.


## Verification and evidence

Fresh production and verification stages and the outer workflow exited 0. Exact replay passed all 6,912 evaluation games, 128 teacher-probe games, 6,144 new updates and 4,608 tactical predictions; compared 12 saved checkpoints, recovered three original next updates, verified nine teacher-probe endpoints and three sealed uniform-control loads. The reused uniform-control training retains its historical F15 replay proof. All 16 focused tests passed on the frozen 530-file source/configuration inventory; the pinned build passed. Ten copied-evidence corruptions were rejected. Independent review and verified dispositions are preserved in the review directory.

Completion receipt: [f16-balanced-imitation.json](../artifacts/reports/f16-balanced-imitation.json). Sealing binds the actual reports, source snapshots, reviews and five current documents; external integrity and identical-reseal records are stored outside this sealed review directory. Production report SHA256: `2b659ed2493a0fe3b2a0965e8827b3771dc3af216d04ed8759aa4d1273505212`.

The value-head weights were unchanged during BC, but the shared trunk changed, so value predictions can change. Terminal-only value metrics for every endpoint/opponent remain below; no value-calibration improvement or playing-strength claim follows from imitation agreement.

## Measured results

Production completed in 314.59 seconds on local CPU. Update time including compilation, sampling and validation was 32.54 seconds. Peak parent RSS was 1082.48 MiB; this excludes Node worker RSS.

Reused 512 sealed training games / 26,278 accepted decisions. Only the balanced arm trained anew: 6,144 BC updates / 786,432 row presentations (393,216 full-battle teacher and 393,216 old rehearsal). Historical uniform controls have the same 2,048-step-per-seed budget; no new control updates or PPO updates occurred.

Fresh teacher probe: 128 games, 6,533 accepted rows, 6,533 attempted decisions, 0 rejected/unavailable choices, 128 natural terminals and 0 cutoffs.

Full-battle evaluation: 6,912 games / 435,299 total decisions, 6,907 natural terminals, 5 unresolved cutoffs, 3,154 wins, 3,753 losses and 0 true ties. 189 unavailable-choice attempts are preserved.

Team inventory: 1024 fit teams, 256 fresh teacher-probe teams, 256 fresh battle-evaluation teams; 384 F15 development teams excluded. Exact role intersections are zero. This does not establish novelty across every older experiment, species or mechanic.

### Fixed family exposure and fresh probe support

| Family | Training presentations | Probe decisions | Probe games |
| --- | ---: | ---: | ---: |
| normal_move | 196,608 | 5,091 | 128 |
| tera_move | 98,304 | 195 | 122 |
| voluntary_switch | 49,152 | 82 | 57 |
| forced_switch | 49,152 | 1,165 | 128 |

### Registered gates

| Seed | Coverage screen | Strength gate | Failed coverage checks |
| --- | --- | --- | --- |
| 7103 | FAIL | FAIL | normal_move_preserved, voluntary_switch_accuracy |
| 8209 | FAIL | FAIL | normal_move_preserved, voluntary_switch_accuracy |
| 9311 | FAIL | FAIL | normal_move_preserved, voluntary_switch_accuracy |

Coverage is a descriptive threshold screen, not a confidence-qualified strength result. All-seed gates: {'coverage_screen': False, 'strength_gate': False}.

### All endpoint outcomes

Each cell contains 256 games. Value MSE weights terminal games equally; cutoffs are excluded from value targets and remain unresolved in outcome comparisons.

| Seed | Endpoint | Opponent | Wins | Losses | Cutoffs | Value MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 7103 | initial | random | 222 | 34 | 0 | 0.1794 |
| 7103 | initial | base_power | 88 | 168 | 0 | 0.1944 |
| 7103 | initial | tactical | 14 | 242 | 0 | 0.1425 |
| 7103 | uniform | random | 244 | 12 | 0 | 0.1556 |
| 7103 | uniform | base_power | 115 | 141 | 0 | 0.2163 |
| 7103 | uniform | tactical | 18 | 238 | 0 | 0.1867 |
| 7103 | balanced | random | 233 | 23 | 0 | 0.2163 |
| 7103 | balanced | base_power | 90 | 166 | 0 | 0.1972 |
| 7103 | balanced | tactical | 12 | 244 | 0 | 0.1344 |
| 8209 | initial | random | 224 | 32 | 0 | 0.1628 |
| 8209 | initial | base_power | 94 | 162 | 0 | 0.2142 |
| 8209 | initial | tactical | 8 | 248 | 0 | 0.1685 |
| 8209 | uniform | random | 244 | 11 | 1 | 0.1700 |
| 8209 | uniform | base_power | 129 | 127 | 0 | 0.2290 |
| 8209 | uniform | tactical | 11 | 245 | 0 | 0.2270 |
| 8209 | balanced | random | 220 | 36 | 0 | 0.2275 |
| 8209 | balanced | base_power | 102 | 154 | 0 | 0.2158 |
| 8209 | balanced | tactical | 8 | 248 | 0 | 0.1435 |
| 9311 | initial | random | 230 | 23 | 3 | 0.1842 |
| 9311 | initial | base_power | 102 | 154 | 0 | 0.2110 |
| 9311 | initial | tactical | 18 | 238 | 0 | 0.1662 |
| 9311 | uniform | random | 236 | 19 | 1 | 0.1609 |
| 9311 | uniform | base_power | 131 | 125 | 0 | 0.2268 |
| 9311 | uniform | tactical | 20 | 236 | 0 | 0.2284 |
| 9311 | balanced | random | 221 | 35 | 0 | 0.2001 |
| 9311 | balanced | base_power | 102 | 154 | 0 | 0.2151 |
| 9311 | balanced | tactical | 18 | 238 | 0 | 0.2119 |

### All teacher-agreement endpoints and families

NLL, accuracy and target probability here weight games equally within each family. Evaluation keeps its natural distribution. Rows and per-decision summaries remain in report.json.

| Seed | Endpoint | Family | NLL | Greedy accuracy | Target probability |
| --- | --- | --- | ---: | ---: | ---: |
| 7103 | initial | all | 1.7003 | 25.83% | 25.42% |
| 7103 | initial | normal_move | 1.7679 | 22.15% | 22.17% |
| 7103 | initial | tera_move | 2.0541 | 27.46% | 17.65% |
| 7103 | initial | voluntary_switch | 4.1134 | 0.00% | 2.24% |
| 7103 | initial | forced_switch | 1.2030 | 42.61% | 41.50% |
| 7103 | uniform | all | 1.3347 | 41.91% | 35.70% |
| 7103 | uniform | normal_move | 1.2448 | 44.29% | 36.13% |
| 7103 | uniform | tera_move | 3.4266 | 0.00% | 4.27% |
| 7103 | uniform | voluntary_switch | 4.9441 | 0.00% | 1.06% |
| 7103 | uniform | forced_switch | 1.0891 | 42.49% | 41.59% |
| 7103 | balanced | all | 1.5583 | 35.61% | 29.34% |
| 7103 | balanced | normal_move | 1.6041 | 33.93% | 26.77% |
| 7103 | balanced | tera_move | 1.8608 | 40.98% | 24.74% |
| 7103 | balanced | voluntary_switch | 3.1717 | 7.37% | 9.00% |
| 7103 | balanced | forced_switch | 1.2039 | 43.32% | 41.56% |
| 8209 | initial | all | 1.7818 | 27.38% | 25.60% |
| 8209 | initial | normal_move | 1.8249 | 23.75% | 22.01% |
| 8209 | initial | tera_move | 2.2104 | 24.18% | 18.07% |
| 8209 | initial | voluntary_switch | 4.4252 | 0.00% | 2.72% |
| 8209 | initial | forced_switch | 1.3738 | 43.71% | 42.30% |
| 8209 | uniform | all | 1.3357 | 41.90% | 35.55% |
| 8209 | uniform | normal_move | 1.2469 | 44.34% | 35.96% |
| 8209 | uniform | tera_move | 3.3934 | 0.00% | 4.50% |
| 8209 | uniform | voluntary_switch | 5.0246 | 0.00% | 0.89% |
| 8209 | uniform | forced_switch | 1.0917 | 42.34% | 41.49% |
| 8209 | balanced | all | 1.5373 | 35.00% | 29.52% |
| 8209 | balanced | normal_move | 1.5775 | 34.11% | 27.14% |
| 8209 | balanced | tera_move | 1.8490 | 39.34% | 24.82% |
| 8209 | balanced | voluntary_switch | 3.2905 | 3.27% | 7.04% |
| 8209 | balanced | forced_switch | 1.2018 | 40.12% | 41.19% |
| 9311 | initial | all | 1.8456 | 26.77% | 24.70% |
| 9311 | initial | normal_move | 1.9382 | 23.38% | 21.04% |
| 9311 | initial | tera_move | 2.4014 | 20.08% | 17.96% |
| 9311 | initial | voluntary_switch | 4.3058 | 0.00% | 2.45% |
| 9311 | initial | forced_switch | 1.2058 | 42.66% | 41.74% |
| 9311 | uniform | all | 1.3293 | 42.08% | 35.78% |
| 9311 | uniform | normal_move | 1.2412 | 44.40% | 36.21% |
| 9311 | uniform | tera_move | 3.4586 | 0.00% | 4.23% |
| 9311 | uniform | voluntary_switch | 4.8467 | 0.00% | 1.07% |
| 9311 | uniform | forced_switch | 1.0804 | 42.93% | 41.71% |
| 9311 | balanced | all | 1.5743 | 34.25% | 28.78% |
| 9311 | balanced | normal_move | 1.6256 | 32.55% | 26.07% |
| 9311 | balanced | tera_move | 1.8532 | 42.62% | 24.36% |
| 9311 | balanced | voluntary_switch | 3.1399 | 5.61% | 8.36% |
| 9311 | balanced | forced_switch | 1.2076 | 41.30% | 41.50% |

### All 27 full-battle contrasts

Later endpoint minus comparator, in percentage points. Adjusted paired-cluster percentile-bootstrap coverage is approximate with 128 clusters; the finite-sample Hoeffding interval remains alongside it.

| Seed | Comparison | Opponent | Cutoff-aware difference | Adjusted bootstrap | Hoeffding | Cell result |
| --- | --- | --- | --- | --- | --- | --- |
| 7103 | balanced_vs_initial | base_power | +0.78 to +0.78 | -12.89 to +14.06 | -32.25 to +33.82 | inconclusive |
| 7103 | balanced_vs_initial | random | +4.30 to +4.30 | -3.91 to +12.69 | -28.74 to +37.33 | inconclusive |
| 7103 | balanced_vs_initial | tactical | -0.78 to -0.78 | -6.64 to +5.47 | -33.82 to +32.25 | inconclusive |
| 7103 | balanced_vs_uniform | base_power | -9.77 to -9.77 | -22.66 to +2.53 | -42.80 to +23.27 | inconclusive |
| 7103 | balanced_vs_uniform | random | -4.30 to -4.30 | -11.33 to +1.95 | -37.33 to +28.74 | inconclusive |
| 7103 | balanced_vs_uniform | tactical | -2.34 to -2.34 | -9.77 to +4.69 | -35.38 to +30.69 | inconclusive |
| 7103 | uniform_vs_initial | base_power | +10.55 to +10.55 | -1.95 to +23.05 | -22.49 to +43.58 | inconclusive |
| 7103 | uniform_vs_initial | random | +8.59 to +8.59 | +1.17 to +16.41 | -24.44 to +41.63 | improvement |
| 7103 | uniform_vs_initial | tactical | +1.56 to +1.56 | -4.69 to +8.20 | -31.47 to +34.60 | inconclusive |
| 8209 | balanced_vs_initial | base_power | +3.12 to +3.12 | -9.95 to +16.80 | -29.91 to +36.16 | inconclusive |
| 8209 | balanced_vs_initial | random | -1.56 to -1.56 | -9.38 to +6.25 | -34.60 to +31.47 | inconclusive |
| 8209 | balanced_vs_initial | tactical | +0.00 to +0.00 | -5.08 to +4.69 | -33.04 to +33.04 | inconclusive |
| 8209 | balanced_vs_uniform | base_power | -10.55 to -10.55 | -24.61 to +4.09 | -43.58 to +22.49 | inconclusive |
| 8209 | balanced_vs_uniform | random | -9.77 to -9.38 | -16.80 to -2.34 | -42.80 to +23.66 | regression |
| 8209 | balanced_vs_uniform | tactical | -1.17 to -1.17 | -6.25 to +3.70 | -34.21 to +31.86 | inconclusive |
| 8209 | uniform_vs_initial | base_power | +13.67 to +13.67 | +0.78 to +26.56 | -19.36 to +46.71 | improvement |
| 8209 | uniform_vs_initial | random | +7.81 to +8.20 | +0.20 to +16.02 | -25.22 to +41.24 | improvement |
| 8209 | uniform_vs_initial | tactical | +1.17 to +1.17 | -4.30 to +6.25 | -31.86 to +34.21 | inconclusive |
| 9311 | balanced_vs_initial | base_power | +0.00 to +0.00 | -12.11 to +12.11 | -33.04 to +33.04 | inconclusive |
| 9311 | balanced_vs_initial | random | -4.69 to -3.52 | -14.84 to +6.25 | -37.72 to +29.52 | inconclusive |
| 9311 | balanced_vs_initial | tactical | +0.00 to +0.00 | -6.44 to +6.44 | -33.04 to +33.04 | inconclusive |
| 9311 | balanced_vs_uniform | base_power | -11.33 to -11.33 | -23.63 to +1.17 | -44.36 to +21.71 | inconclusive |
| 9311 | balanced_vs_uniform | random | -6.25 to -5.86 | -15.23 to +2.92 | -39.29 to +27.18 | inconclusive |
| 9311 | balanced_vs_uniform | tactical | -0.78 to -0.78 | -7.42 to +5.86 | -33.82 to +32.25 | inconclusive |
| 9311 | uniform_vs_initial | base_power | +11.33 to +11.33 | +0.00 to +22.84 | -21.71 to +44.36 | inconclusive |
| 9311 | uniform_vs_initial | random | +1.17 to +2.73 | -6.64 to +11.33 | -31.86 to +35.77 | inconclusive |
| 9311 | uniform_vs_initial | tactical | +0.78 to +0.78 | -6.44 to +7.81 | -32.25 to +33.82 | inconclusive |

Cell counts: {'inconclusive': 23, 'improvement': 3, 'regression': 1}.

Bootstrap resolution note: the registered adjusted percentile tails use approximately 18.5 of 20,000 resamples per tail. Cell classifications near zero are Monte Carlo-sensitive; seed 9311 uniform-versus-initial base-power has a lower bound exactly zero. No resampling or statistics-seed change was performed.

## Post-hoc fixed-checkpoint transfer diagnostic

Post-hoc descriptive full-corpus fit versus preregistered fresh-probe comparison. Same fixed checkpoints, natural row distribution; no sample selection, optimizer update, new simulator game or acceptance gate.

All rows are retained; metrics weight games equally within each family. This is diagnostic, not a new acceptance gate or a selected-checkpoint comparison.

| Seed | Endpoint | Family | Training-corpus accuracy | Fresh-probe accuracy |
| --- | --- | --- | ---: | ---: |
| 7103 | uniform | normal_move | 56.54% | 44.29% |
| 7103 | uniform | tera_move | 0.20% | 0.00% |
| 7103 | uniform | voluntary_switch | 0.00% | 0.00% |
| 7103 | uniform | forced_switch | 54.76% | 42.49% |
| 7103 | balanced | normal_move | 43.44% | 33.93% |
| 7103 | balanced | tera_move | 83.10% | 40.98% |
| 7103 | balanced | voluntary_switch | 70.59% | 7.37% |
| 7103 | balanced | forced_switch | 49.14% | 43.32% |
| 8209 | uniform | normal_move | 55.86% | 44.34% |
| 8209 | uniform | tera_move | 0.31% | 0.00% |
| 8209 | uniform | voluntary_switch | 0.00% | 0.00% |
| 8209 | uniform | forced_switch | 54.15% | 42.34% |
| 8209 | balanced | normal_move | 43.71% | 34.11% |
| 8209 | balanced | tera_move | 82.38% | 39.34% |
| 8209 | balanced | voluntary_switch | 63.08% | 3.27% |
| 8209 | balanced | forced_switch | 49.22% | 40.12% |
| 9311 | uniform | normal_move | 56.53% | 44.40% |
| 9311 | uniform | tera_move | 0.51% | 0.00% |
| 9311 | uniform | voluntary_switch | 0.00% | 0.00% |
| 9311 | uniform | forced_switch | 55.45% | 42.93% |
| 9311 | balanced | normal_move | 42.13% | 32.55% |
| 9311 | balanced | tera_move | 81.36% | 42.62% |
| 9311 | balanced | voluntary_switch | 72.64% | 5.61% |
| 9311 | balanced | forced_switch | 49.35% | 41.30% |

## Preserved preregistration

Original design recorded before production; budgets, seeds and gates remained fixed.

F15 increased overall teacher agreement but suppressed greedy agreement on rare teacher-Tera decisions. F16 tests action-family sampling while holding the network, likelihood/Adam update, training corpus, original F10 initialization and accumulated training budget fixed. The tactical teacher remains a heuristic; agreement is not optimal play.

The new arm receives 2,048 BC steps per seed (7103, 8209, 9311). Each batch has 32 ordinary moves, 16 Tera moves, eight voluntary switches and eight forced/revival replacements from the same sealed 512-game F15 train corpus, plus the same 64 old F7 samples used by F15. Within each family, sample games uniformly then decisions uniformly; shuffle the 64 selected rows. Family definitions use only the acting player's request phase and teacher action. The uniform control reuses F15's sealed 2,048-step imitation checkpoint; its training is not rerun. The F10 initial endpoint is also retained. No PPO continuation is part of this experiment.

Generate 128 fresh, unbalanced development teacher games for descriptive agreement diagnostics. Evaluate all three endpoints against random, base-power and tactical across 128 new paired team/mechanics clusters in both seats: 6,912 games. Exact teacher-probe and battle-evaluation teams must be mutually disjoint and separate from F15's probe/evaluation teams and the training partition. All 512 familiar F10 tactical probes repeat at each endpoint. Final reserved data remain untouched; mechanics/species and opponent identities recur.

The fixed contract is `contracts/balanced-imitation-v1/plan.json`. Keep all 27 paired full-battle contrasts, approximate adjusted-bootstrap intervals and conservative Hoeffding intervals. Cutoffs remain unresolved, never draws. The separate coverage screen is a descriptive threshold check: Tera equal-game accuracy at least 25%, gain over uniform at least ten points, target mass at least 15%; voluntary-switch accuracy at least 25%; normal-move and forced-switch accuracy losses no more than five points; all 32 known contexts retained. Required fresh probe support is fixed before generation, with no resampling if it fails. This screen is not a statistical strength claim.

The strength gate requires at least five points tactical improvement over both uniform and F10 initial with positive adjusted-bootstrap lower bounds, no detected other-opponent regression, no cell above 5% cutoffs and known-context retention. Each overall gate requires all three seeds. No seed/checkpoint selection, budget extension or automatic follow-on experiment.

Freeze source/runtime/plan after focused tests and cross-model review. Run once, then fresh-replay every new teacher game, BC update, full evaluation outcome and policy/value prediction. Compare 12 saved checkpoints, recover three original next updates and verify three sealed uniform-control loads. The historical F15 control-training replay remains the bound evidence for its training. Completion requires clean launcher exits, corruption checks, measured current documentation, full live/archive hash verification and identical resealing. F10 remains the reference pending broader evidence; full G2/G3/G4 and public strength remain open. Local CPU only, no Prime or public server. The unresolved GitHub visibility hold remains in effect.
