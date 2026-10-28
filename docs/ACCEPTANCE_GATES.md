# Proposed execution gates and experiment ladder

> Publication note (2026-10-28): the status chronology below ends at F30 and is historical. See the [current project status](../README.md) and [F39 failure handoff](F39_ONLINE_CONTINUATION.md). Frozen scientific originals are preserved separately.

F30 stopped during production after 5,476.65 seconds of launcher elapsed time (91.28 minutes). All eight simulator workers hit the 60-second response deadline while starting the first random-opponent midpoint evaluation batch for seed 9311/control at cohort 512. The underlying cause remains unresolved. The partial run preserves 3,328 of 4,608 new cohorts, 26,624 optimizer steps, 3,407,872 selected rows, 128,144 training games, 9,984 opponent-evaluation games and 416 self-play diagnostic games. No exhaustive replay or successful completion seal ran, so no completed comparison or all-seed learning result is claimed. Original run evidence and frozen code remain unchanged; no restart or repair was performed. Full G2/G3/G4 and public strength remain open. No promotion, public server, final reserved data, Prime or GitHub publication. See [F30 handoff](F30_NONLINEAR_CONTINUATION.md).

F29 completed its fixed full-battle comparison in 3016.79 measured CPU seconds: 57,232 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'control': False, 'nonlinear': True}; all-seed tactical feature benefit: False; secondary base-power benefit: True. All 27 contrasts retained: {'improvement': 11, 'inconclusive': 16}. The nonlinear actor passed its source-bound qualification checks (with sealed F25/F26/F27/F28 evidence reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. The arms match selected rows, updates and refresh frequency, with identical initial predictions. Added nonlinear actor capacity can also alter shared-trunk gradients. Tactical is trained on: its adaptation gate is primary, and base-power benefit secondary. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F29 handoff](F29_NONLINEAR_POLICY.md).

F28 completed its fixed full-battle comparison in 1752.59 measured CPU seconds: 25,360 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'large': False, 'small': False}; all-seed base-power batch benefit: False; trained-on tactical adaptation benefit: False. All 27 contrasts retained: {'inconclusive': 27}. The batch-size wrapper passed 15 focused qualification checks (with sealed F25/F26/F27 evidence reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. The arms match rows and optimizer steps while differing in collection size and policy-refresh frequency. Tactical is trained on; adaptation is reported separately. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F28 handoff](F28_BATCH_COLLECTION.md).

F27 completed its fixed original-batch gradient audit in 95.85s on local CPU: 18 cohorts across all seeds/arms, 520 replayed training games, 29,586 decisions, 144 exact original steps and 12 next-checkpoint matches. All 13 focused tests and independent update replay pass. Every sampled joint update improves its actor surrogate; current critic/entropy removal has modest local effects. Game-half gradients disagree in 42/72 comparisons, but every sampled tactical batch has 2–4 wins, so zero-win batches are not the observed explanation. These are correlated diagnostics, not a strength gain or a causal noise diagnosis. Next proposed comparison: larger on-policy collection batches with matched selected-row/optimizer budgets and unchanged model/opponent mixture. No new campaign or promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F27 handoff](F27_GRADIENT_AUDIT.md).

F26 completed its fixed full-battle comparison in 1804.41 measured CPU seconds: 22,192 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'mixed': False, 'self': False}; all-seed base-power mixture benefit: False; trained-on tactical adaptation benefit: False. All 27 contrasts retained: {'inconclusive': 26, 'regression': 1}. The opponent adapter passed 18 focused qualification checks (with sealed F25 checks reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. Tactical is a trained-on heuristic; adaptation is reported separately. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F26 handoff](F26_OPPONENT_MIX.md).

F25 completed its fixed full-battle comparison in 3549.77 measured CPU seconds: 60,832 training games, 16,128 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'matchup': False, 'zero': False}; all-seed matchup benefit: False. All 45 contrasts retained: {'improvement': 8, 'inconclusive': 37}. The corrected dynamic view passed 88 qualification checks and the pinned build; exhaustive fresh replay reproduces all updates and games, 48 checkpoints and 12 original-next-update recoveries. F23/F24 failure evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F25 handoff](F25_TERMINAL_MATCHUP.md).

F24 stopped before training after 68 qualification checks/build passed. The fixed run committed 656 reference games (all natural terminals; 43,625 decisions), then rejected a terminal Illusion observation with both actual Zoroark and its benched Mimikyu disguise marked active. A synthetic reducer test reproduces this pre-existing terminal identity/HP mapping defect in the older and new views. Training games and optimizer updates are zero; no learning result or successful completion receipt is claimed. Frozen source/evidence remain preserved and the monitor is paused. Next is a separately versioned own-identity/terminal reconciliation repair before another fixed comparison. Full G2/G3/G4 remain open; no final reserved data, public server, Prime or GitHub publication. See [F24 handoff](F24_DYNAMIC_MATCHUP.md).

F23 production failed before training:31 focused checks/build passed, but Double Shock produced a valid public typeless component (`???/Fighting`) rejected by the new matchup encoder.200 reference games were committed (all natural terminals;16,036 decisions); zero training games or optimizer updates occurred. The exact failure was reproduced for diagnosis; frozen source/evidence are preserved, with no automatic repair/restart or completed-screen/strength claim. Next proposed step is a separately versioned dynamic-type repair and qualification before a new fixed comparison, including the confirmed stale public type override after switching. See [F23 handoff](F23_MATCHUP_PPO.md). Full G2/G3/G4 and public strength remain open; Prime and final reserved data remain unused; GitHub visibility hold persists.


F22 completed (2026-10-25): qualified shared-policy/value PPO and ran the matched full-battle comparison. All three seeds completed 15,056 training games, 16,128 frozen-opponent games and 576 separate self-play diagnostic games in 1184.68 measured seconds on CPU. Learning gates passed in 0/3 flat seeds and 2/3 shared seeds; shared tactical benefit passed in 0/3. Neither all-seed learning gate passed, and no architecture is promoted. All 45 final contrasts are retained: 10 improvements, no detected regressions and 35 inconclusive results under the registered approximate bootstrap. All 14 focused qualification groups, exhaustive fresh replay, 66 checkpoint comparisons, 12 original-next-update recoveries and 19 copied-evidence rejection checks pass. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer. See the [F22 handoff](F22_SHARED_PPO.md) and [completion receipt](../artifacts/reports/f22-shared-ppo.json).

Next proposed step: qualify a function-preserving public matchup-feature adapter on the flat policy, then preregister a longer full-battle RL ablation against an identical augmented model with those new inputs zero-masked. Use all three final flat checkpoints, equal initialization and optimizer-reset rules, matched budgets, and F10 as the external reference. Flat is the experimental host because it retained familiar mastery; this is not policy promotion or a claim that representation caused tactical weakness. Keep event history and opponent-pool changes as separate interventions. F22 trained each learner on only 2,392–2,648 games, so sustained learning remains untested. Training collection used 405.83 seconds versus 21.35 seconds in optimizer calls; profile the collector before choosing GPU or Prime resources.

F21 completed (2026-10-25): evaluated all final F20 flat/shared checkpoints and the F10 rehearsal reference across 6,912 paired-development battle executions in 326.96 seconds on CPU, with zero optimizer updates or checkpoint selection. The shared-versus-flat battle-benefit gate passed in 0/3 seeds. All 27 contrasts remain: 2 improvements, 2 regressions and 23 inconclusive under the registered approximate adjusted bootstrap. Outcomes: 6,908 natural terminals, 4 unresolved cutoffs, 425,793 total attempted decisions and 214,191 learner decisions. All 21 focused tests, exhaustive fresh battle replay and nine immutable checkpoint loads pass. F20 familiar mastery failures remain; no policy is promoted. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer. See the [F21 handoff](F21_SHARED_BATTLE.md) and [original-objective review](OBJECTIVE_REVIEW_2026_10.md).

F21 falls in the mixed/inconclusive decision branch: retain F10 as the reference and do not claim a tactical winner. Next proposed step: return to full-battle RL with a controlled flat-versus-shared PPO comparison, starting from all three matched F20 final checkpoints and retaining F10 as an external reference. The shared scorer remains an unproven candidate: its matched-size implementation and limited random-opponent signal make RL learnability a testable hypothesis, not an established tactical benefit. First qualify the shared actor/value PPO loss, gradients and checkpoint recovery against the existing learner. Then preregister an equal decision/update budget and fixed-opponent learning curves; use battle improvement as the primary outcome, with familiar tactical cases and held-out self-play value diagnostics as supporting measures. Do not extend behavior cloning merely to improve teacher agreement. Structured observations, history, opponent auxiliary heads and historical-opponent cross-play remain separate candidate interventions. Measure complete-cycle cost before selecting a longer CPU or GPU budget. No new training is launched in F21; local CPU sufficed for this evaluation, and Prime remains unprovisioned.

F20 completed (2026-10-25): trained fresh flat and shared action-scoring models on the same frozen 2,048-game corpus, with identical sampled batches and 8,192 updates per arm/seed. All three seeds completed 49,152 updates / 6,291,456 row presentations in 161.97 seconds on CPU. Collected 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,087 attempted decisions). Final voluntary-switch training agreement was 58.10–63.13% for flat and 21.12–23.47% for shared; fresh-probe agreement was 11.67–13.39% and 21.41–25.50%. The descriptive architecture gate passed in 0/3 seeds. All 28 focused tests, exhaustive replay, 30 checkpoint comparisons and 12 original-next-update recoveries passed. These are teacher-imitation results, not battle win rates. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F20 handoff](F20_SHARED_POLICY.md).

Next proposed step: preregister a full-battle development evaluation of these frozen 8,192-update flat and shared checkpoints, alongside the existing F10 reference, against fixed random, base-power and tactical opponents. Keep all three seeds and identical paired battle schedules, report wins/losses and unresolved cutoffs separately, and carry the known Tera-variant failures into the assessment. Make no additional optimizer updates or checkpoint selection during this screen. It will test whether better teacher imitation translates into battle strength before choosing another training intervention; F20 does not establish that translation or authorize promotion. Local CPU remains sufficient for a bounded evaluation; no public ladder or final reserved evaluation.

F19 completed (2026-10-25): continued all three sealed F18 expanded checkpoints from 2,048 to 8,192 updates, preserving Adam state, the 106,149-row corpus, model and sampling. The run added 18,432 updates / 2,359,296 row presentations and 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,017 attempted decisions). Voluntary-switch training agreement changed from 14.67–16.51% to 59.47–63.74%; fresh-probe agreement changed from 3.85–8.12% to 8.40–12.96%. The descriptive learning gate passed in 0/3 seeds. These are teacher-imitation metrics. All 16 focused tests, exhaustive fresh replay, 15 checkpoint comparisons and six original-next-update recoveries passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F19 handoff](F19_LEARNING_CURVE.md).

Next proposed step: compare the existing flat policy with a shared action-scoring architecture that reuses parameters across Pokémon slots and legal move/switch choices. Use the same public information, frozen corpus, family sampling and preregistered update budget, with a matched control, fresh development probe and all three seeds. First verify action/slot remapping and hidden-information isolation. This tests a structural generalization hypothesis; F19 does not prove the architecture is the cause or that the new design will work. Do not expand data and architecture together, add more optimization after seeing results, promote a policy, or start PPO automatically. Local CPU remains sufficient for the bounded comparison.

F18 completed (2026-10-25): compared the original 512 teacher games with an expanded 2,048-game corpus, using the unchanged 928-feature model and 2,048 updates per arm/seed. Collected 1,536 additional training games and 256 fresh development teacher games (1,792 natural terminals, 0 cutoffs, 93,300 attempted decisions). All three seeds and both arms completed 12,288 updates / 1,572,864 row presentations. The descriptive learning gate passed in 0/3 seeds. Fresh-probe voluntary-switch agreement was 5.03–7.41% for original and 4.63–5.16% for expanded. These are teacher-imitation metrics, not battle win rates. All 21 focused checks, exhaustive replay, 18 checkpoint comparisons, six original-next-update recoveries and three exact F16 control matches passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F18 handoff](F18_DATA_COVERAGE.md).

Next proposed step: measure a preregistered learning curve on the now-frozen expanded corpus, keeping the original model, sampling and optimizer fixed and comparing the existing 2,048-update budget with an 8,192-update endpoint on fresh development games. Expanded-corpus fit agreement was only 14.7–16.5%, so insufficient fitting remains plausible; this does not establish that extra updates will improve transfer. Track both fitting and fresh-probe agreement, preserve all seeds and other action families, and do not collect another training corpus or promote a policy from F18.

F17 completed (2026-10-25): replayed all 640 existing teacher games / 32,811 accepted decisions, then ran a fixed two-arm semantic-input canary with 12,288 BC updates and 1,572,864 row presentations. The descriptive representation gate passed in 0/3 seeds. Voluntary-switch agreement on the reused development probe was 3.27–7.37% for the zero-input control and 0.88–4.74% with explicit public type/move metadata. No new games or full-battle strength test occurred. All 28 focused checks and exhaustive update/prediction replay pass, including six original-next-update recoveries and 18 checkpoint comparisons. F10 remains the reference; full G2/G3/G4 and public strength remain open. Final reserved data, public servers and Prime were unused; GitHub remains held. See the [F17 handoff](F17_SWITCH_REPRESENTATION.md).

Next proposed step: test a larger, more diverse full-battle teacher corpus against the original corpus while keeping the original 928-feature model, balanced sampling and update budget fixed. Use a fresh development probe and preserve final reserved data. This tests data coverage separately; broader data is a candidate remedy, not an established one. No policy promotion or PPO continuation follows from this canary.

F16 completed (2026-10-24): balanced full-battle imitation reused 512 sealed teacher games and executed 6,144 new BC updates, 128 fresh teacher-probe games and 6,912 full-battle evaluation games in 314.59 seconds on local CPU. Both the descriptive coverage screen and the strength gate failed in all three seeds. Tera agreement improved to 39–43%, but ordinary-move agreement fell to 33–34% and voluntary-switch agreement reached only 3–7%. Tactical-opponent wins for initial/uniform/balanced were 40/49/38 out of 768 each (descriptive totals). All 27 registered contrasts remain: 3 improvements favoring uniform over initial, 1 balanced-versus-uniform regression against random, and 23 inconclusive results. Evaluation retains 6,907 natural terminals and five unresolved cutoffs. All 16 focused tests and exhaustive fresh replay pass, including 12 checkpoint comparisons and three original-next-update recoveries. F10 remains the reference; no PPO follow-on or policy promotion. Full G2/G3/G4 and public strength remain open. Final reserved data, public servers and Prime were unused; GitHub remains held. See the [F16 handoff](F16_BALANCED_IMITATION.md).

F16 coverage failed on ordinary-move preservation and voluntary-switch accuracy in every seed; Tera thresholds, forced-switch preservation and known-context retention passed. The strength gate passed in 0/3 seeds. The single detected balanced-versus-uniform regression is random opponent / seed 8209: cutoff-aware −9.77 to −9.38 percentage points, approximate adjusted-bootstrap interval −16.80 to −2.34. Other balanced comparisons are inconclusive; this is neither equivalence nor universal regression. Retain conservative Hoeffding intervals and all cutoffs. Full G2/G3/G4 remain open.

F15 completed (2026-10-24): full-battle tactical demonstrations plus matched RL continuations collected 512 training teacher games, 128 separate teacher-probe games, 8,440 RL training games and 4,608 evaluation games in 416.28 seconds on local CPU. All three preregistered benefit gates failed in every seed (0/3 each). Teacher agreement improved from about 25% to 43% after imitation, then fell to 37–39% after RL; tactical-opponent strength remains unestablished. All 16 focused tests and exhaustive fresh replay pass, including nine original-next-update recoveries. The 45 contrasts contain 4 improvements, 4 regressions and 37 inconclusive cells under the registered approximate-bootstrap test. Evaluation retains 12 unresolved cutoffs. F10 remains the reference. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F15 handoff](F15_FULL_IMITATION.md).

F14 completed (2026-10-24): the matched-budget entropy ablation collected 4,512 training and 864 evaluation continuations in 169.41 seconds on local CPU. The preregistered no-entropy benefit gate passed in 0/3 seeds; the descriptive two-point switch-reduction threshold was met in 0/3. All 18 contrasts are retained (0 improvements, 0 regressions, 18 inconclusive under the registered approximate-bootstrap test). Training/evaluation natural terminals: 4,512/864; cutoffs: 0/0. All 14 focused tests and exhaustive replay pass, including six original-next-update recoveries. This adaptively reused development bank does not establish fresh-team or full-battle generalization. F10 remains the reference. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F14 handoff](F14_ENTROPY_ABLATION.md).

F13 completed (2026-10-24): the credit-assignment and continuation canary collected 4,536 training and 864 evaluation continuations in 203.34 seconds on local CPU; all ended naturally. The registered lambda=1 benefit gate did not pass (0/3 seeds); all 18 contrasts were inconclusive. No new policy is promoted; F10 remains the reference. Independent return calculations agree with sealed GAE, and fixed-state probes confirm that F12 increased switching propensity. The new public-state-selected bank has 48 midgame/endgame starts from 192 official train-partition teacher games. Both arms received equal selected-row and optimizer-step budgets. All 15 focused tests and exhaustive replay pass, including six original-next-update recoveries. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F13 handoff](F13_CREDIT_ASSIGNMENT.md).

F12 completed (2026-10-24): the fixed curriculum comparison collected 12,848 training and 6,912 evaluation games in 694.16 seconds on local CPU. The preregistered curriculum-benefit gate did not pass (0/3 seeds). Under the registered approximate-bootstrap test, both arms regressed against base-power in every seed; curriculum also regressed against the control on random/base-power in seed 9311. Neither new arm is promoted over the prior F10 baseline. Both arms received 384 Adam steps per seed and identical PPO/rehearsal budgets; only the starting-state mixture differed, with both training against the same frozen tactical opponent. All 17 focused tests pass. Fresh independent replay reproduces every training cohort, selected batch, optimizer update, all 6,912 evaluation games, 18 stage checkpoints and six original-next-update recoveries. Full G2/G3/G4 and public ladder/generalization claims remain open. Final reserved data, public servers and Prime were unused; GitHub remains on the existing visibility hold. See the [F12 handoff](F12_CURRICULUM_LEARNING.md).

F11 completed (2026-10-24): audited all 3,072 final F10 tactical-opponent games and 78,744 learner decisions. There were 54,900 heuristic disagreements; these are screening signals, not automatic error labels. Among 62 fixed single-action alternatives, 7 changed a loss to a win, 3 worsened a win to a loss, and 52 left the terminal reward unchanged; all ended naturally. Built 65 executable train-only starts from 96 fresh official train games, meeting all seven family quotas with verified unsolved handoffs and full-battle continuations. All 35 focused tests and independent replay pass. No optimizer updates or new playing-strength claim. Next is a preregistered matched-budget curriculum ablation; full G2/G3/G4, public strength and generalization remain open. Final data, public servers and Prime were unused; GitHub stays on the existing visibility hold. See the [F11 handoff](F11_BATTLE_DECISIONS.md).

F10 completed (2026-10-24): the fixed longer full-battle comparison collected 12,256 training and 13,824 evaluation games in 1051.30 seconds on local CPU. All-seed full-battle learning gates under the registered approximate-bootstrap development test: plain passed, rehearsal passed; secondary all-seed tactical-retention benefit passed. Both arms received 1,024 new Adam steps per seed (6,144 total), with 786,432 selected PPO rows and 393,216 extra supervised row presentations in the rehearsal arm. All 28 focused tests pass; fresh replay reproduces every full game, batch, update, tactical prediction, 48 periodic checkpoints and six original-next-update recoveries. This is development evaluation against known opponents and scenario families. Full G2/G3/G4 remain open; public servers, final reserved data and Prime were unused. GitHub remains on the existing visibility hold. See the [F10 handoff](F10_LONG_TRANSFER.md).

Both arms improved against random and base-power opponents. Rehearsal retained the known tactical skills in 3/3 seeds versus plain PPO's 0/3, but all nine rehearsal-minus-plain full-battle contrasts remain inconclusive: superiority and equivalence are unestablished. Pooled tactical-opponent wins remain only 4.43% for plain PPO and 4.23% for rehearsal. The critic beats constant 0.5, but a post-hoc constant fitted to each same evaluation cell beats it on random and tactical opponents; calibration and broader transfer remain open.


F9 v2 completed (2026-10-24): the fixed full-battle transfer pilot collected 1,448 training and 1,728 evaluation games in 161.80 seconds on local CPU. The all-seed supervised-rehearsal benefit gate passed. Both arms received 128 new Adam steps per seed (768 total), with 98,304 selected PPO rows and 49,152 extra supervised row presentations in the rehearsal arm. All 20 focused tests pass; fresh replay reproduces every full game, selected batch, update, tactical prediction and six original-next-update checkpoints. This is a small development pilot with known tactical templates, not a public-strength result. Full G2/G3/G4 remain open; public servers, final data and Prime were unused. See the [F9 handoff](F9_FULL_TRANSFER_V2.md).

Tactical retention passed in all three rehearsal seeds and failed in all three plain-PPO seeds. Rehearsal increased final correct-action probability over plain PPO by 8.81, 12.10 and 9.31 percentage points. Neither arm passed the full-battle learning gate in any seed; all rehearsal-versus-plain win-rate contrasts were inconclusive. Terminal-only value MSE remained around 0.25, with mean values near 0.5.

F9 v1 failed before training (2026-10-24): the new admission guard incorrectly disallowed legitimate Showdown unavailable-choice feedback. A bounded replay of the original initial-policy prefix reproduced two trapped-switch responses in development episode7, both seat orientations; the sealed collector refreshed requests correctly. No optimizer update or learning result exists. The failed source, run and diagnostic are preserved; a separate corrected version keeps the same seeds and training budget. See [F9 v1 handoff](F9_FULL_TRANSFER.md).

**F8 completed (2026-10-24):** The [matched-budget PPO comparison](F8_DECISION_RETENTION.md) collected 49,152 training and 6,144 evaluation decisions in 320.98 seconds on local CPU. The supervised-initialization retention gate **failed** (0/3 seeds), due to probability drops in variant contexts. All three supervised arms nevertheless retained **100% greedy accuracy**, versus **25% for fresh arms**, and passed the final probability-advantage gate. Improvement remains incomplete across families. Both arms received 128 new Adam steps per seed; the supervised arm also had F7's prior supervision. All 25 focused tests, every simulator decision/branch and all PPO/checkpoint replays pass. These are local one-decision HP-retention cases with repeated development inputs; no full battles or playing-strength result. Full G2/G3/G4 remain open. Final data, public servers and Prime were unused.

**F7 completed (2026-10-24):** The [supervised fit diagnostic](F7_DECISION_FIT.md) reached **100% final argmax accuracy on all 512 states in all three seeds**, but **all preregistered probability gates failed**. Minimum training-context target probabilities were 86.31%, 89.77% and 89.81%, below the 95% requirement. The unchanged network can express these finite greedy choices; reliable stochastic mastery and the cause of F6's RL failure remain unresolved. The fixed run used 3,072 Adam steps and 393,216 row presentations in 3.38 seconds on CPU, with no new games. All 15 focused tests and exhaustive update/checkpoint replay pass. Full G2/G3/G4 remain open; final data, public servers and Prime were unused. The gate is failed even though every argmax accuracy subgate passes; no threshold is relaxed. Variant observations have no encoded overlap with training, but are known development scenarios.

**F6 completed (2026-10-24):** The [decision-point canary](F6_DECISION_CANARY.md) completed 24,576 one-decision training cases and 1,536 evaluation cases in 142.19 seconds on local CPU. The preregistered local learning gate **failed in all three seeds**. The final policies chose Tera almost regardless of context; correct-switch probability fell below 1%. These cases optimize one-turn HP retention, not battle wins. All 96 separate frozen-policy continuation checks completed: 96 natural terminals and 0 cutoffs. All 70 focused tests pass; every training/evaluation decision, PPO update and continuation game reproduced in fresh replay. This does not establish curriculum transfer or public strength; full G2/G3/G4 remain open. Final data and Prime were unused.

**F5 completed (2026-10-23):** [Opponent-mixture comparison](F5_OPPONENT_MIXTURE.md) completed **19,232 training + 13,824 evaluation games** in 1947.61 seconds (32.5 minutes) on local CPU. The preregistered exploratory mixture gate **did not pass**; all three seeds, both arms and 27 contrasts are retained. Each seed–arm received 131,072 selected learner rows and 1,024 Adam steps (786,432 rows / 6,144 steps overall). No tactical improvement over self-play was established; observed tactical wins were lower in every mixture seed. One base-power comparison showed a regression under the approximate bootstrap test. The mixture required 13,184 training games versus 6,048 for self-play and is not promoted. There were 32,968 natural terminals and 88 cutoffs. All 86 focused tests pass; bounded original-update and evaluation replay passes. This measures fresh development games against known heuristic identities, not public ladder or unseen-opponent strength. Full G2/G3/G4 remain open; final data and Prime were unused.

**F4 completed (2026-10-22):** [Full-Showdown learning screen](F4_FULL_LEARNING.md) completed **12,288 training + 18,432 evaluation games** in 2,058.68 seconds (34.3 minutes) on local CPU. All three seeds and nine initial/final opponent comparisons are retained; the preregistered exploratory learning gate **passed**. There were 30,626 natural terminals, 94 explicit cutoffs, 2,367,153 decisions and 9,570 Adam steps. All 64 focused tests pass; bounded replay exactly reproduced the original last cohort/update for every seed and the first paired matchup in every endpoint/opponent cell. This is development learning evidence, not a public-strength result; full G2/G3/G4 remain open. Final data and Prime were unused.

F3 performance update (2026-10-22): [CPU/Metal throughput comparison](F3_THROUGHPUT.md) completed. Eight persistent CPU workers with batched player-only inference achieved 1.87× warm collection throughput (48 matched games: 5.105 → 2.727 seconds). On the same isolated Python/JAX runtime, Metal took 4.841 seconds versus CPU 3.088 seconds, 57% longer. Actual Metal execution and bounded numerical/sampler checks pass. All 47 focused tests pass; actions/outcomes match and next updates agree within tolerance, not byte-for-byte. This is a fixed-policy throughput benchmark, not a new learning screen or strength claim. Keep the current CPU runtime and reuse workers. F2 evidence is preserved; full G2/G3/G4 remain open.

F2 execution update (2026-10-22): [full-Showdown training](F2_FULL_TRAINING.md) completed the fixed three-seed replacement pilot: 768 training + 768 development games, 1,516 natural terminals / 20 cutoffs, 157,983 decisions, 798 Adam steps, 207.45 seconds on local CPU. Observed wins increased in all six seed/opponent cells, but every preregistered gate is inconclusive; value error improved against random and worsened against base power. All 42 focused tests pass; replaying each final cohort exactly reproduced model/optimizer state. The first F2 attempt and its own-Illusion wait-request repair are preserved separately. The completion inventory verifies source/evidence hashes and counters. M7 remains failed; full G2/G3/G4 and public ladder strength remain open. Final data and Prime were unused.

Contribution for the execution plan, 2026-10-10. All numbers below are proposed engineering acceptance criteria, not facts recovered from Jaxcalibur or guarantees of completeness/convergence. A gate combines an explicit supported scope, executable checks, and logged evidence. Passing a statistical screen licenses the next experiment; it does not prove global correctness or optimality.

## Playable reference baseline — local adapter evidence

The [playable baseline](PLAYABLE_BASELINE.md) passes 64 complete full-format pinned
Showdown development games, exact replay of 2,904 actor decisions, and three real
local-server WebSocket matches with a same-connection rematch and fresh login.
Its heuristic scores 60/64 against a weak random policy; this is descriptive
functional evidence, not a learning gate, Elo estimate or strong-opponent result.
The client uses only own requests/public events. Targeted tests cover unseen reserve
invariance, real hidden trapping, Revival Blessing, request order, duplicate/stale
requests, ranked names, Tera memory and Illusion replacement. Public-account login,
browser interaction, extended unattended operation and ladder evaluation remain untested.
No custom simulator scope was expanded. Full G2/G3 remain open and final data stay
reserved. The original quantitative gates below continue to apply.

The additive [recovery client](SHOWDOWN_SESSION.md) passes six real local-server
games with seven injected disconnects, including an accepted pending action and a
lost outgoing write. All 120 attempted decisions and their player inputs pass the
wire audit. The source-bound regression passes 303 tests (114 Node + 189 Python),
including nineteen new recovery/server-method tests. Recovery is bounded, in memory, restricted to known room IDs and the
stateless heuristic, and depends on the pinned combined request/sentchoice frame.
This establishes no process-restart, live-public-server, recurrent-policy or
strength guarantee. See [deployment requirements](SHOWDOWN_DEPLOYMENT.md).

## Frozen opponent panel — development evidence

The [four-policy panel](SHOWDOWN_OPPONENT_PANEL.md) passes 768 complete pinned
Showdown games, 35,711 exact actor-input/action replays and a source-bound
338-test regression (149 Node + 189 Python; 266 unchanged tested files). Each
of six matchups has 64 paired team/seed clusters and swapped policy seats.
Every admitted team is independently regenerated and content-partition checked.
Tactical's direct score is 58.59% against baseline (descriptive paired interval
52.34–65.62%) and 57.03% against damage (50.78–63.28%). Its point score against
random is 2.34 points lower than baseline's, so no universal dominance or panel
non-regression is certified. These unadjusted development intervals authorize
neither a strength promotion nor a ladder claim.

The new heuristics approximate public damage/threat information; they are not a
complete mechanics model or trained full-format policy. Existing network behavior
is unchanged in that milestone. Fresh confirmation and the additive adapter are
recorded below. G2/G3/G4 requirements below
remain open outside previously declared slices. Final data stay reserved; no GPU
or Prime resources were used.

## Tactical confirmation — passed local development screen

The [frozen confirmation](SHOWDOWN_TACTICAL_CONFIRMATION.md) completes 1,000
mirrored team/seed pairs for each of six matchups: 12,000 games on 2,000 fresh
development teams. No pair was replaced and no policy was tuned during the run.
Fresh simulator replay reproduces all 564,379 actor inputs/actions and the complete
report, including outcomes, counters, intervals and the confirmation decision.
Tactical scores 60.65% against baseline; the approximate nominal adjusted lower
bound is 58.55%, passing both the 52% point floor and strict 50% lower-bound rule.
Relative to baseline, tactical scores +2.20 percentage points against random and
+10.05 against damage, with respective lower bounds +0.80 and +7.40 points. Both
pass the declared strict −3-point tolerance. Shared whole-pair bootstrap sampling
uses 30,000 replicates and three Bonferroni allocations; this is an approximate
development screen, not guaranteed coverage or the larger final promotion budget.

The opt-in tactical network client completes six local-server games through seven
injected disconnects. All 179 attempted actions and their player-visible inputs
pass the saved-wire reconstruction. All 380 source-bound tests pass. This qualifies
the frozen heuristic for continued local play and evaluation. That six-game
result did not meet the separate network-volume prerequisite; the subsequent
1,000-game soak is recorded below. Full-format learned-policy requirements and
public deployment checks remain open. G2/G3/G4 remain open outside declared slices; reserved
final data are unused and no Prime resources are needed.

## Local adapter volume prerequisite — passed in declared scope

The [1,000-game local network soak](SHOWDOWN_NETWORK_SOAK.md) passes on the frozen
tactical client and pinned loopback server: 20 fresh login sessions, 980 rematches,
500 games in each tactical seat, and 20 each of before-choice, lost-write and
accepted-choice disconnects. All 46,024 decisions reconstruct from each player's
own received history; exact server team receipts, results, request IDs and counters
agree. There are no campaign failures, known-forbidden actions or unavailable-choice
retries. All 426 tests pass; sampled client RSS peaks at 445.063 MiB under the
768 MiB bound. No team or game is replaced, and no official set values are edited.

This closes the named local adapter soak prerequisite only. Full G4 still requires
the complete intended-format engine and learned-policy work; full G2/G3 remain
open. The run does not validate public login, public-server conditions, process
restart, unlimited sessions, continuous memory peaks or a playing-strength gain.
The official generator's sets use an exact-pool local admission rule because the
normal Random Battle path bypasses constructed-team legality. This does not relax
any simulator battle rule or final-data partition. Prime remains unnecessary.

## Dependency order and safe parallelism

The critical path is: pinned format/data contract → request/observation semantics → reference parity harness → restricted JAX engine → correct small PPO learner → measured learning → expanded mechanics → complete target-format evaluation → search. Keep action IDs, observation schemas, phase names, and checkpoint versions stable early.

Three independent workstreams are safe after the interfaces are agreed:

- **Simulator:** mechanics, reference scenarios, request timing, supported-feature manifest, JAX throughput. Owns `FullState` and transitions.
- **Policy/learner:** small transformer, per-player rollout buffer, PPO/GAE, auxiliary heads, checkpoint/resume. Initially develops against tiny scripted environments and frozen observation fixtures. Owns learner internals, not simulator truth extraction.
- **Data/evaluation:** pinned generator export, sampled-team validation, frozen team seeds, baseline policies, scoring and paired intervals. Owns fixtures and evaluation reports, not engine mechanics.

A separate observation/adapter owner approves interface changes spanning these streams. Belief-model design can run in parallel once observation fixtures exist; connecting beliefs to production action features waits for leakage tests. Search literature/toy-game experiments can run in parallel, but integration waits for a useful validated policy and simulator. Do not launch a large training run while known mechanics divergences remain in its sampled scope.

## Coverage stages: call each game what it is

**Stage A: synthetic canaries.** One-step bandits, deterministic delayed reward, asymmetric forced replacements, and toy hidden-information games. These test algorithms, not Pokémon strength.

**Stage B: restricted Pokémon.** Explicit fixed teams/sets and a small mechanics whitelist, first 1v1 then 3v3/6v6. Use reference battles with those exact teams and documented rules. Do not call a whitelist-filtered training distribution unmodified Gen 9 Random Battles. Excluded features must be impossible to encounter or must fail loudly before sample collection; never implement unsupported effects as silent no-ops.

**Stage C: expanding mechanics.** Add complete mechanic families in batches, with interactions: status/residual damage; stat changes; switching/pivots/trapping; item/ability interactions; field effects; Tera; copying/transformation and other edge cases. Re-run prior fixtures and a newly sampled interaction suite after each batch. Keep the reduced-format baseline checkpoint and compare transfer against training from scratch at the same budget.

**Stage D: intended complete format.** Pin the exact official generator, rules, simulator revision, and data export. Audit all generated sets *and the transitive mechanics they can invoke*; move copying or random move effects can extend beyond listed sets. A closed generator manifest plus a passing harness is evidence of supported scope, not proof that every interaction is correct. Record unknowns and observed unsupported effects as blockers to an unqualified full-format claim. Default train/eval sampling may only include fully supported matches; if exclusions remain, label the result restricted and report rejection rates/distribution shift.

## Quantitative gate ledger

### G0 — Contracts and testability (before learning)

Deliver a reproducible version manifest; phase diagram; observable action-mask contract; full-state/observation separation; deterministic seed protocol; supported-feature manifest; saved reference fixtures; checkpoint schema.

Acceptance: every phase has a fixture for normal entry, terminal exit, cancellation/re-request if applicable, and both player roles. Every action ID round-trips between network, simulator, and adapter. All displayed/consumed observations must be reconstructible from the same player's protocol history and private information. Unknown enum values and unsupported mechanics fail visibly. Zero tolerated silent fallbacks that alter battle rules.

### G1 — Learner canaries (before Pokémon scale)

Status 2026-10-10: passed for the documented whole-episode NumPy CPU canaries;
see [M1/G1 evidence and limitations](M1_G1.md). Cut arithmetic is hand-tested;
end-to-end persistent-game cuts and live-state checkpointing are not yet supported.

Run on CPU/FP32 where practical, using a transparent reference implementation.

- GAE/returns/log-ratios agree with hand-computed short trajectories within absolute 1e-5; terminal and buffer-cut cases are separate. No terminal bootstrap; rollout truncation bootstraps from the correct perspective.
- Same weights/data before an update give PPO ratios of 1 within 1e-5. Single-legal-action decisions have actor gradient/entropy zero within 1e-7, while a forced switch with multiple candidates remains trainable. The scalar surrogate need not be zero (it can equal minus the advantage); alternatively mask those rows out of the actor objective explicitly.
- One-step four-action canary, with exploration penalties disabled: all three learner seeds put at least 99% probability on the known rewarding action within 100,000 sampled decisions. This is a diagnostic target; failure triggers learner investigation rather than immediately expanding training.
- Deterministic delayed-reward canary learns both player perspectives without winner/loser sign inversion. Every example with a known action forbidden by the observable mask produces zero probability for that action. No NaN/Inf in sampled logits, probabilities, losses, gradients, or optimizer state.
- The boundary-checkpoint resume tests below pass before jobs longer than two hours. Mid-collection/mid-epoch tests become required only if those continuation modes are implemented.

### G2 — Restricted engine fidelity and information safety

Acceptance within the declared scope:

- Zero unexplained disagreements in legal request structure, observation/reveal timing, HP/status/PP changes, winner, and phase transitions across all hand-authored fixtures and at least 10,000 independently seeded reference battles. Adapt the battle count to CPU cost, but record the actual count and do not substitute correlated transitions for independent games in a failure-rate claim.
- Deterministic submechanics and forced random outcomes should compare exactly after normalization. For chance mechanics, either enumerate finite outcome distributions or run predeclared probability checks with a family-wise error policy. Same seed alone is not a parity contract if the engines consume randomness differently; inject/check semantically identified chance outcomes.
- At least 1,000 paired hidden-state mutations, stratified over unrevealed species/sets/items/abilities/Tera and request phases: normalized player observations must be identical whenever that player's information is identical. Deterministic inference logits must then match within 1e-6 FP32 (or exactly for byte-identical inputs and deterministic backend). This includes damage features, memory, masks, and cached values.
- For each covered mechanics family, include adversarial boundary cases: HP/KO thresholds, move PP exhaustion, simultaneous fainting, interrupted turns, forced switches, stale requests, and legal-information changes. Random volume cannot replace these targeted tests.

Interpretation: zero failures in 10,000 independent battles implies an approximate 95% upper bound of 0.03% for the tested distribution's per-battle failure probability, *not* for an untested distribution or all mechanics. A discovered divergence blocks training on affected states until resolved or explicitly excluded.

Current restricted progress: the [Showdown-chance campaign](M1_SHOWDOWN_CHANCE.md)
passes exact finite-word mapping checks and 10,000 fresh public-duel player-view
games with zero disagreements. This closes that slice’s fresh coverage target;
full G2 stays open for hidden-set and broader mechanics strata. It does not certify
Showdown seed/PRNG-stream equivalence. The binary-policy/new-chance combination
has now passed its own [two-hour CPU qualification](M1_POLICY_SOAK.md); that adds
stability evidence without expanding the fidelity or information-safety scope.

### G3 — Stable restricted-policy pilot

Engineering progress: the selected binary-policy/Showdown-chance combination
passed its own [two-hour CPU qualification](M1_POLICY_SOAK.md): 7,200.007 seconds,
8,158,970 complete cycles and 3,272,073,210 decisions in the public Rattata duel.
There were zero unexpected compilations after warm-up, 120 telemetry records and
13 exact original-next-update checkpoint audits. Peak process RSS was 473.766 MiB;
the unchanged source-bound regression passed 235 tests. This closes the selected
combination's two-hour stability item. **Full G3 remains open:** G2 coverage is
incomplete, and frozen-opponent evaluation across training seeds, critic calibration
and appropriate memory/budget evidence remain. Process RSS does not certify
device-memory headroom. The [binary policy](M1_BINARY_POLICY.md)'s explicit floor
and exact dyadic sampling resolve the old tail-law mismatch in this two-action
profile; neither that fix nor the soak establishes strategic strength.

Development evaluation update (2026-10-11): the [restricted-policy pilot](M2_POLICY_PILOT.md)
passes the preregistered random-opponent learning screen for all three fresh training
seeds. Final scores are 63.65%, 62.35% and 62.30%; adjusted score lower bounds exceed
55% and adjusted paired improvement bounds exceed zero. All 54,000 real Showdown
games and 169,024 candidate decisions are audited; 251 tests pass. This supplies
frozen-panel, multi-seed learning evidence for the public duel only. Scores against
damage-greedy remain about 50%; calibration diagnostics are not a calibration gate
pass. **Full G3 remains open**, as do G2's hidden-set and wider mechanics requirements
and appropriate memory/budget evidence. No final evaluation data were used.

Use the smallest practical feed-forward/history-token transformer first; defer recurrence and specialized token MLPs. Measure complete collect/update cycles, not isolated kernel throughput. Proposed pilot: three training seeds, a modest predeclared decision budget, and checkpoint evaluation every fixed budget increment.

Hard gates: zero unsupported effects or observation-mask bugs in training scope; zero NaN/Inf; no unexpected recompilation after warm-up at fixed shapes; a two-hour soak without OOM; peak allocated memory leaves roughly 15% headroom where the selected runtime permits it. Throughput gate is budget-derived: measured decisions/s must project the next experiment within its allocated GPU hours with at least 25% overhead allowance. Report decisions/game as well as games/s.

Learning gate: reliable improvement over initialization against a frozen baseline panel, plus a predeclared starter target such as point score ≥60% against random legal play and lower 95% confidence bound >55%. Include a damage-greedy baseline and old checkpoints as diagnostics. Beating random alone does not authorize a claim of strategic competence. Analyze all three training seeds; a single successful seed is insufficient to conclude the learner is stable. Record KL/clip fraction, entropy, legal-action probability floor, critic calibration, phase-specific losses, battle length, and timeouts.

### G4 — Full intended-format readiness

Expand G2's parity/hidden-info checks to all supported format families and the pinned official team distribution. Run at least 10,000 fresh independent reference battles after the last mechanics/schema change, and an adapter soak of at least 1,000 private/local evaluation games. Require zero unexplained divergences, zero known-forbidden actions, zero unsupported mechanics, and zero server-visible phase/request corruption. Distinguish legitimate hidden-legality rejections from client mistakes.

A production-length run is authorized technically only after a short full-format learner smoke test and resume test pass, and its measured complete-cycle throughput/memory predict the planned budget. Save learning curves against fixed opponents; current self-play score near 50% is not a learning metric.

### G5 — Search integration (after a useful policy)

First baseline: world-averaged one-step lookahead versus an observation-only opponent policy. Then simultaneous deeper search. Before Pokémon integration, test 2×2 games and rock-paper-scissors against exact solutions: at the declared rollout budget, mean-strategy exploitability ≤0.02 on normalized [-1,1] payoffs across a fixed test suite and at least 20 seeds is a proposed target. Report the whole distribution and worst case; do not elevate toy-game success into a Pokémon equilibrium claim.

Require exact pre-reveal hidden-information invariance for any opponent policy component explicitly intended to use only that opponent's information. Test returned action distributions, not only network inputs, because optimized search backups can leak secrets. Root mixed-strategy sampling is the default candidate; argmax is an ablation.

Promote a search configuration only after a predeclared meaningful gain (for example ≥2 percentage points in direct head-to-head score) with a paired lower confidence bound above 50%, no material regression against the fixed opponent panel, and p99 decision latency under the adapter's specified time budget. Use measured latency bounds chosen from the target deployment; do not invent a universal seconds threshold. Search timeouts must return a legal base-policy action and be counted.

## Phase-aware reject/re-request semantics

A policy's action mask represents what the player is allowed to infer from its request, not all legality knowable to the omniscient simulator. A choice may be admissible under that request but rejected when hidden mechanics are checked. Masking it using hidden truth can disclose information.

Implement and test three classes:

1. **Known-invalid choice:** forbidden by the observable mask, stale request ID, malformed mapping, or wrong phase. This is an implementation bug; quarantine/log the affected sample. Do not learn from accidental protocol failures.
2. **Rule-defined hidden-legality rejection:** reproduce the reference rejection message and knowledge update, issue the appropriate new request, retain commitments exactly as the protocol requires, and consume no turn/PP/chance/reward unless the reference does. The attempted choice and resulting information-bearing re-request form a zero-reward transition in that player's decision trajectory; do not silently erase useful feedback or reset GAE as if the game ended.
3. **Transport retry:** replay idempotently against the request ID; do not create a new learner decision or change the opponent's commitment merely because a message was resent.

Fixtures must cover repeated rejected choices, simultaneous submissions, one player awaiting re-request while the other has committed, terminal events before an anticipated request, and a hidden restriction becoming known. Test whether each feedback event is public or private. Keep the rejection/phase state in checkpoints. Retry-count safeguards should surface a bug or explicit failure, not secretly adjudicate a match as lost.

## Resume and on-policy state

Start with synchronous collection: a frozen actor snapshot collects a buffer; PPO updates that buffer; the next collection uses the new actor. Multiple GPUs, async actors, and stale-policy correction can wait.

A continuation checkpoint includes learner parameters, optimizer/scaler state, schedules and update counters; behavior/actor parameters and version; live simulator states; phase/request IDs and pending actions; observations/private memory; all PRNG streams; partially filled buffers with old log-probabilities/values/masks; collection cursor; minibatch permutation and epoch/minibatch cursor (plus accumulated gradients if mid-accumulation saving is supported); and data/schema/source versions. Save atomically with checksums. A manifest must make partial writes distinguishable from valid checkpoints.

Initial resume gate: support completed-update checkpoints at the documented collection barrier, and on a deterministic CPU canary reproduce the next samples and updates from a restore. Live environment/history/request state is required even at that boundary when games persist. Also simulate a crash during collection and during an update: roll back atomically to the last complete checkpoint, discard intervening work and verify loss/re-execution accounting. This is rollback recovery, not exact continuation at the crash point. GPU checks compare counters, actor versions, masks, next PRNG keys and trajectory identities; declare any numerical tolerance without allowing missing or repeated committed data.

Optional extended gate: only after adding exact mid-collection, pre-update-buffer or mid-epoch checkpoint modes, interrupt at each supported point and reproduce the uninterrupted canary's continuation with its complete state above. Prefer bit identity for deterministic operations. A mode that resets unfinished games must be labeled non-exact and report lost samples; it does not satisfy exact continuation. Never stitch a partial buffer to a different behavior-policy version. Raw histories are preferable to unversioned learned recurrent state initially. If recurrence is later added, specify hidden-state recomputation/burn-in across policy updates and restores.

## Lean experiment ladder and statistical promotion

Run one decision-relevant question at a time:

1. Tiny canaries and PPO correctness, no architecture search.
2. Restricted policy baseline, then one exploration/GAE screen (lambda and zero-avoidance settings); keep the best stable cheap configuration.
3. Set-prediction auxiliary head, opponent-action head, then both. Require calibration/hidden-state safety in addition to win-rate gain.
4. Snapshot opponent pool versus current-self-play only; measure cyclic regressions against frozen checkpoints.
5. Expand mechanics and distribution. Repeat the useful baseline before adding model capacity.
6. Correlated generator-based belief candidates versus no belief input; only then learned conditional proposals.
7. One-step search, deeper simultaneous search, and world-count/rollouts/depth tradeoffs at equal inference time. Then test sampled mixture versus argmax and exact HP versus bins.
8. Only after gains survive: specialist MLPs, delayed matchup features, recurrence, distillation, or larger networks. Each competes at both matched sampled decisions and matched GPU-hours; these answer different questions.

Use three independent training seeds for pilot selection and preferably five for consequential comparisons. Match team seeds and swap seats. A team-seed pair is one statistical cluster; do not count both games as independent. Direct head-to-head pilot: 1,000 pairs (2,000 games); promotion: 5,000 pairs (10,000 games), extending by a predeclared plan if precision is inadequate. Worst-case normal-approximation 95% half-widths for bounded pair scores are about 3.1 and 1.4 percentage points, respectively. Actual precision depends on within-pair covariance; bootstrap complete pairs, and account for training seeds separately. These counts describe game-level comparisons: balance evaluation across training seeds and report per-seed results; no quantity of games from one checkpoint replaces independent training runs. If a claim depends on run-to-run reliability, budget evaluation for each seed and estimate the training-seed variation explicitly.

For comparisons against a fixed opponent panel, compute per-pair score differences between candidate and baseline and bootstrap those paired differences, stratified by opponent. Their variance differs from direct head-to-head score variance; do not reuse the same precision claim. Estimate sample size from a pilot using n ≈ (1.96+0.84)^2 s_delta^2 / desired_gain^2 for a rough two-sided 5%, 80%-power plan, then freeze that plan. Correct for multiple comparisons or reserve untouched confirmation seeds. Avoid repeatedly checking ordinary intervals and stopping when favorable.

Common RNG seeds are not automatically common chance outcomes after policies take different actions. If event-indexed random coupling is used, validate it; otherwise pair the teams and seats while allowing independent chance, and state that choice. Preserve full score tables, confidence intervals, generator/simulator/model versions, illegal-action/rejection categories, turn counts, and latency distributions in each promotion report.

## M3 expanded reference-scope evidence (2026-10-11)

The [new reference handoff](M3_SWITCH_REFERENCE.md) supplies a versioned 24-team,
two-member profile and four-action Node/Python codec. It passes 72 natural scripted
fixtures with opposite-order replay, 1,872 hidden-state equal-view comparisons,
576 own-variant positive controls, 24 different-reserve reveal pairs, 5,426 actor
reconstructions, and 3,456 conditional damage values. A separate process regenerates
the entire fixture bundle exactly; the source-bound regression passes 264 tests.
Counterfactual raw views are retained and independently encoded in Python, not
merely compared through producer hashes. Coverage is independently recounted.

This is a reference prerequisite only. It does not close G2/G3 or certify a new
simulator, chance sampler, fixed-shape actor, categorical policy or learner. The
72 full scripts are not an exhaustive 24×24 game campaign. Hidden move sets are
fixed by species; reserve identity and speed choices provide the current hidden-set
strata. Every claimed exclusion applies only to the exact closed roster. Before
training: pass conditional damage and state/player-view differential tests, declare
and verify the chance law, implement phase-aware collection and resolve the handling
of twelve-turn truncations. No strategic strength, GPU readiness or final-data
release follows from this milestone. Prime remains unnecessary.


## M3 switching/hidden-set engine — restricted engineering smoke (2026-10-11)

The [engine handoff](M3_SWITCH_ENGINE.md) records zero scalar/JAX/reference differences
across 10,000 fresh games, 136,635 states, 273,270 player views and 241,384 decisions,
plus all 72 sealed scripts. The 24 fresh unpatched controls are re-executed at seal time
against the currently verified pinned oracle. All 1,872 hidden-state equalities,
576 own-speed controls, 24 reserve-reveal pairs, 357 exhaustive HP-display values and
3,456 conditional damage values pass; JAX private-state mutation tests also pass.

The categorical sampler covers all 16 masks and exact conditional finite-word tails.
Autodiff PPO agrees with the NumPy categorical reference; three numerical bandit
canaries pass. The primary smoke completes 192 updates, 6,144 games and 140,983 decisions,
with finite valid-only updates, own-player GAE, six exact original-next-update replays,
nine disk loads and six injected in-memory discard checks. Interrupted atomic file
writes are tested separately. All 443 tests pass on 320 unchanged tested files.
Eleven copied-evidence negatives reject without changing production or writing receipts.

These results admit this fixed restricted collect/update loop for its own sustained
CPU qualification. They do not pass a learning-strength screen or close full
G2/G3/G4. The 2,675 training cutoffs and 7,814 reference cutoffs are nonterminal resets,
not draws or evaluation outcomes. Public ladder strength remains unqualified;
full battles need additional horizon/history and mechanics coverage. Final reserved
data remain unused; Prime is unnecessary.


## M3 switching/categorical sustained CPU gate — passed (2026-10-11 Eastern)

The [versioned two-hour plan](../contracts/switch-soak-v1/plan.json) passes on the
sealed M3 kernels with fresh engineering seed 101 and eight warm-up updates.
The production window measures 7,200.008393 seconds, 1,711,484 primary updates,
54,767,488 capped games and 1,173,453,330 decisions. All per-update legality,
finite-state and independent same-player GAE checks pass. There are zero
post-warm-up compilation events; linked counters remain within their fixed bounds.
The maximum paired-clock gap is 0.551545 seconds against the 30-second limit;
peak process RSS is 460.578125 MiB against the 2 GiB cap. All 120 minute records,
13 exact original-next-update restores, initial warm-up identity and final
source/runtime/reserved-data admission pass in a separate verification process.
The launcher exits with code 0. All 331 tested files still match the passing
460-test regression, which is reused. See the [handoff](M3_SWITCH_SOAK.md) and
[completion receipt](../artifacts/reports/m3-switch-soak.json).

This closes the restricted sustained CPU-execution item only. It closes no strength,
broad-mechanics, live-game-checkpoint, device-headroom, public-deployment or full
G2/G3/G4 gate. The 4,488,697 twelve-turn cutoffs are nonterminal resets, not draws;
50,278,791 games terminate. Numerical learning diagnostics do not establish
convergence or critic calibration. The playable client and restricted JAX learner
remain separate. No final reserved teams, public server or Prime resources are used.


## M3 restricted learning gate — did not pass (2026-10-11 Eastern)

The [M3 categorical learning screen](M3_SWITCH_SCREEN.md) completed all
**31,104 pinned Showdown games** after fresh three-seed training (24,576 updates,
786,432 games, 18,615,320 decisions). The preregistered conservative learning gate
**did not pass**. Evaluation has 12,837 terminal games and 18,267
explicit unresolved cutoffs; no cutoff is scored as a draw. All 476 tests pass.
See the [receipt](../artifacts/reports/m3-switch-screen.json). Full G2/G3/G4 and public
ladder strength remain open; final data stay reserved and Prime is unnecessary.

Three fixed seeds and all 27 cells are retained. The actor, chance law and PPO kernels
remain sealed. The primary gate uses conservative unresolved-outcome bounds and
complete seat-pair uncertainty; it does not pool seeds or treat cutoffs as draws.
The source-bound 476-test regression covers 347 unchanged files; nine original-next-
update checkpoints and all recorded action/value outputs are exact. The receipt
requires a fresh complete Showdown replay of every evaluation game. Review snapshots,
verified dispositions, preflight and nine corruption probes are preserved. A completed
experiment may fail its learning gate. No public-ladder or full G2/G3/G4 promotion follows.
