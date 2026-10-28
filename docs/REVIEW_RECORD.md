# Review and verification record

> Publication note (2026-10-28): the status chronology below ends at F30 and is historical. See the [current project status](../README.md) and [F39 failure handoff](F39_ONLINE_CONTINUATION.md). Frozen scientific originals are preserved separately.

F30 stopped during production after 5,476.65 seconds of launcher elapsed time (91.28 minutes). All eight simulator workers hit the 60-second response deadline while starting the first random-opponent midpoint evaluation batch for seed 9311/control at cohort 512. The underlying cause remains unresolved. The partial run preserves 3,328 of 4,608 new cohorts, 26,624 optimizer steps, 3,407,872 selected rows, 128,144 training games, 9,984 opponent-evaluation games and 416 self-play diagnostic games. No exhaustive replay or successful completion seal ran, so no completed comparison or all-seed learning result is claimed. Original run evidence and frozen code remain unchanged; no restart or repair was performed. Full G2/G3/G4 and public strength remain open. No promotion, public server, final reserved data, Prime or GitHub publication. See [F30 handoff](F30_NONLINEAR_CONTINUATION.md).

F29 completed its fixed full-battle comparison in 3016.79 measured CPU seconds: 57,232 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'control': False, 'nonlinear': True}; all-seed tactical feature benefit: False; secondary base-power benefit: True. All 27 contrasts retained: {'improvement': 11, 'inconclusive': 16}. The nonlinear actor passed its source-bound qualification checks (with sealed F25/F26/F27/F28 evidence reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. The arms match selected rows, updates and refresh frequency, with identical initial predictions. Added nonlinear actor capacity can also alter shared-trunk gradients. Tactical is trained on: its adaptation gate is primary, and base-power benefit secondary. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F29 handoff](F29_NONLINEAR_POLICY.md).

F28 completed its fixed full-battle comparison in 1752.59 measured CPU seconds: 25,360 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'large': False, 'small': False}; all-seed base-power batch benefit: False; trained-on tactical adaptation benefit: False. All 27 contrasts retained: {'inconclusive': 27}. The batch-size wrapper passed 15 focused qualification checks (with sealed F25/F26/F27 evidence reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. The arms match rows and optimizer steps while differing in collection size and policy-refresh frequency. Tactical is trained on; adaptation is reported separately. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F28 handoff](F28_BATCH_COLLECTION.md).

F27 completed its fixed original-batch gradient audit in 95.85s on local CPU: 18 cohorts across all seeds/arms, 520 replayed training games, 29,586 decisions, 144 exact original steps and 12 next-checkpoint matches. All 13 focused tests and independent update replay pass. Every sampled joint update improves its actor surrogate; current critic/entropy removal has modest local effects. Game-half gradients disagree in 42/72 comparisons, but every sampled tactical batch has 2–4 wins, so zero-win batches are not the observed explanation. These are correlated diagnostics, not a strength gain or a causal noise diagnosis. Next proposed comparison: larger on-policy collection batches with matched selected-row/optimizer budgets and unchanged model/opponent mixture. No new campaign or promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. Qualification binds 692 files; two static implementation reviews and independently checked design QA are preserved. See [F27 handoff](F27_GRADIENT_AUDIT.md).

F26 completed its fixed full-battle comparison in 1804.41 measured CPU seconds: 22,192 training games, 13,824 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'mixed': False, 'self': False}; all-seed base-power mixture benefit: False; trained-on tactical adaptation benefit: False. All 27 contrasts retained: {'inconclusive': 26, 'regression': 1}. The opponent adapter passed 18 focused qualification checks (with sealed F25 checks reused) and the pinned build; exhaustive fresh replay reproduces all updates and games, 30 checkpoints and 12 original-next-update recoveries. Tactical is a trained-on heuristic; adaptation is reported separately. Predecessor evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F26 handoff](F26_OPPONENT_MIX.md).

F25 completed its fixed full-battle comparison in 3549.77 measured CPU seconds: 60,832 training games, 16,128 opponent games and 576 separate self-play diagnostic games. All-seed learning: {'matchup': False, 'zero': False}; all-seed matchup benefit: False. All 45 contrasts retained: {'improvement': 8, 'inconclusive': 37}. The corrected dynamic view passed 88 qualification checks and the pinned build; exhaustive fresh replay reproduces all updates and games, 48 checkpoints and 12 original-next-update recoveries. F23/F24 failure evidence stays immutable. No promotion; full G2/G3/G4 and public strength remain open. No reserved data, public server, Prime or GitHub publication. See [F25 handoff](F25_TERMINAL_MATCHUP.md).

F24 stopped before training after 68 qualification checks/build passed. The fixed run committed 656 reference games (all natural terminals; 43,625 decisions), then rejected a terminal Illusion observation with both actual Zoroark and its benched Mimikyu disguise marked active. A synthetic reducer test reproduces this pre-existing terminal identity/HP mapping defect in the older and new views. Training games and optimizer updates are zero; no learning result or successful completion receipt is claimed. Frozen source/evidence remain preserved and the monitor is paused. Next is a separately versioned own-identity/terminal reconciliation repair before another fixed comparison. Full G2/G3/G4 remain open; no final reserved data, public server, Prime or GitHub publication. See [F24 handoff](F24_DYNAMIC_MATCHUP.md).

F23 production failed before training:31 focused checks/build passed, but Double Shock produced a valid public typeless component (`???/Fighting`) rejected by the new matchup encoder.200 reference games were committed (all natural terminals;16,036 decisions); zero training games or optimizer updates occurred. The exact failure was reproduced for diagnosis; frozen source/evidence are preserved, with no automatic repair/restart or completed-screen/strength claim. Next proposed step is a separately versioned dynamic-type repair and qualification before a new fixed comparison, including the confirmed stale public type override after switching. See [F23 handoff](F23_MATCHUP_PPO.md). Full G2/G3/G4 and public strength remain open; Prime and final reserved data remain unused; GitHub visibility hold persists.


F22 completed (2026-10-25): qualified shared-policy/value PPO and ran the matched full-battle comparison. All three seeds completed 15,056 training games, 16,128 frozen-opponent games and 576 separate self-play diagnostic games in 1184.68 measured seconds on CPU. Learning gates passed in 0/3 flat seeds and 2/3 shared seeds; shared tactical benefit passed in 0/3. Neither all-seed learning gate passed, and no architecture is promoted. All 45 final contrasts are retained: 10 improvements, no detected regressions and 35 inconclusive results under the registered approximate bootstrap. All 14 focused qualification groups, exhaustive fresh replay, 66 checkpoint comparisons, 12 original-next-update recoveries and 19 copied-evidence rejection checks pass. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer. See the [F22 handoff](F22_SHARED_PPO.md) and [completion receipt](../artifacts/reports/f22-shared-ppo.json).

Next proposed step: qualify a function-preserving public matchup-feature adapter on the flat policy, then preregister a longer full-battle RL ablation against an identical augmented model with those new inputs zero-masked. Use all three final flat checkpoints, equal initialization and optimizer-reset rules, matched budgets, and F10 as the external reference. Flat is the experimental host because it retained familiar mastery; this is not policy promotion or a claim that representation caused tactical weakness. Keep event history and opponent-pool changes as separate interventions. F22 trained each learner on only 2,392–2,648 games, so sustained learning remains untested. Training collection used 405.83 seconds versus 21.35 seconds in optimizer calls; profile the collector before choosing GPU or Prime resources.

F21 completed (2026-10-25): evaluated all final F20 flat/shared checkpoints and the F10 rehearsal reference across 6,912 paired-development battle executions in 326.96 seconds on CPU, with zero optimizer updates or checkpoint selection. The shared-versus-flat battle-benefit gate passed in 0/3 seeds. All 27 contrasts remain: 2 improvements, 2 regressions and 23 inconclusive under the registered approximate adjusted bootstrap. Outcomes: 6,908 natural terminals, 4 unresolved cutoffs, 425,793 total attempted decisions and 214,191 learner decisions. All 21 focused tests, exhaustive fresh battle replay and nine immutable checkpoint loads pass. F20 familiar mastery failures remain; no policy is promoted. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer. See the [F21 handoff](F21_SHARED_BATTLE.md) and [original-objective review](OBJECTIVE_REVIEW_2026_10.md).

F21 retains independent implementation review, exact flat-collector and random-word parity checks, non-vacuous trapped-switch replay, 21 source-bound tests and the pinned build. Nineteen copied-evidence corruptions reject. The independent objective audit received opposite-model QA; its uncorrected draft was rejected and the final audit corrects its unsupported claims. Completion documents are bound before all-hash checks and identical resealing.

F20 completed (2026-10-25): trained fresh flat and shared action-scoring models on the same frozen 2,048-game corpus, with identical sampled batches and 8,192 updates per arm/seed. All three seeds completed 49,152 updates / 6,291,456 row presentations in 161.97 seconds on CPU. Collected 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,087 attempted decisions). Final voluntary-switch training agreement was 58.10–63.13% for flat and 21.12–23.47% for shared; fresh-probe agreement was 11.67–13.39% and 21.41–25.50%. The descriptive architecture gate passed in 0/3 seeds. All 28 focused tests, exhaustive replay, 30 checkpoint comparisons and 12 original-next-update recoveries passed. These are teacher-imitation results, not battle win rates. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F20 handoff](F20_SHARED_POLICY.md).

F20 retains independent design, implementation and measured-result reviews, exact delegated-network patch, opposite-model QA and locally verified outcomes. All 28 source-bound tests and pinned Showdown build pass. Seventeen copied-evidence corruptions reject. The experiment preserves both fresh arms, all three seeds and all 12 primary contrasts. Reviewed completion documents are bound before independent all-hash verification and identical resealing; no predecessor was modified.

F19 completed (2026-10-25): continued all three sealed F18 expanded checkpoints from 2,048 to 8,192 updates, preserving Adam state, the 106,149-row corpus, model and sampling. The run added 18,432 updates / 2,359,296 row presentations and 256 fresh development teacher games (256 natural terminals, 0 cutoffs, 13,017 attempted decisions). Voluntary-switch training agreement changed from 14.67–16.51% to 59.47–63.74%; fresh-probe agreement changed from 3.85–8.12% to 8.40–12.96%. The descriptive learning gate passed in 0/3 seeds. These are teacher-imitation metrics. All 16 focused tests, exhaustive fresh replay, 15 checkpoint comparisons and six original-next-update recoveries passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F19 handoff](F19_LEARNING_CURVE.md).

F19 retains independent design and implementation review, opposite-model exposure-module QA and local verification. All 16 source-bound tests and the pinned Showdown build pass. Fifteen copied-evidence corruptions reject. Exact predecessor baselines, continuing sampler indices, all updates and every endpoint prediction reproduce. Completion documents are bound before independent all-hash verification and identical resealing; no predecessor was modified.

F18 completed (2026-10-25): compared the original 512 teacher games with an expanded 2,048-game corpus, using the unchanged 928-feature model and 2,048 updates per arm/seed. Collected 1,536 additional training games and 256 fresh development teacher games (1,792 natural terminals, 0 cutoffs, 93,300 attempted decisions). All three seeds and both arms completed 12,288 updates / 1,572,864 row presentations. The descriptive learning gate passed in 0/3 seeds. Fresh-probe voluntary-switch agreement was 5.03–7.41% for original and 4.63–5.16% for expanded. These are teacher-imitation metrics, not battle win rates. All 21 focused checks, exhaustive replay, 18 checkpoint comparisons, six original-next-update recoveries and three exact F16 control matches passed. F10 remains the reference; full G2/G3/G4 and public strength remain open. No final reserved data, public server, Prime or GitHub transfer. See the [F18 handoff](F18_DATA_COVERAGE.md).

F18 retains reviewed exact-law sampler delegation, opposite-model QA, independent design and implementation reviews and verified dispositions. The strict new checkpoint inventory guard was corrected during preflight after its valid-load test rejected a trunk-key typo; that failure is preserved. All final source-bound checks pass. Thirteen copied-evidence corruptions are rejected; the five current completion documents are updated before sealing. No predecessor was modified.

F17 completed (2026-10-25): replayed all 640 existing teacher games / 32,811 accepted decisions, then ran a fixed two-arm semantic-input canary with 12,288 BC updates and 1,572,864 row presentations. The descriptive representation gate passed in 0/3 seeds. Voluntary-switch agreement on the reused development probe was 3.27–7.37% for the zero-input control and 0.88–4.74% with explicit public type/move metadata. No new games or full-battle strength test occurred. All 28 focused checks and exhaustive update/prediction replay pass, including six original-next-update recoveries and 18 checkpoint comparisons. F10 remains the reference; full G2/G3/G4 and public strength remain open. Final reserved data, public servers and Prime were unused; GitHub remains held. See the [F17 handoff](F17_SWITCH_REPRESENTATION.md).

F17 preserves the accepted delegated coverage implementation, opposite-model QA, three bounded design/correction reviews and verified dispositions. Preflight failures in cosmetic/typeless metadata, rehearsal source/duplicate assumptions and legacy checkpoint loading were caught before production and are preserved. The strict F17 reader includes all semantic parameters/moments and rejects missing blocks. Eleven copied-evidence corruptions are rejected; all completion documents are updated before sealing.

F16 completed (2026-10-24): balanced full-battle imitation reused 512 sealed teacher games and executed 6,144 new BC updates, 128 fresh teacher-probe games and 6,912 full-battle evaluation games in 314.59 seconds on local CPU. Both the descriptive coverage screen and the strength gate failed in all three seeds. Tera agreement improved to 39–43%, but ordinary-move agreement fell to 33–34% and voluntary-switch agreement reached only 3–7%. Tactical-opponent wins for initial/uniform/balanced were 40/49/38 out of 768 each (descriptive totals). All 27 registered contrasts remain: 3 improvements favoring uniform over initial, 1 balanced-versus-uniform regression against random, and 23 inconclusive results. Evaluation retains 6,907 natural terminals and five unresolved cutoffs. All 16 focused tests and exhaustive fresh replay pass, including 12 checkpoint comparisons and three original-next-update recoveries. F10 remains the reference; no PPO follow-on or policy promotion. Full G2/G3/G4 and public strength remain open. Final reserved data, public servers and Prime were unused; GitHub remains held. See the [F16 handoff](F16_BALANCED_IMITATION.md).

F16 preserves the accepted Codex sampler, opposite-model Fable QA, independently checked integration, prelaunch design review and measured-result review with dispositions. Source-bound checks cover 530 source/configuration files and 16 tests; the pinned Showdown build passed. Fresh replay reproduces all new games, updates, policy/value decisions and 4,608 known tactical predictions, verifies three sealed uniform-control loads, compares 12 checkpoints and recovers three original next updates. Ten copied-evidence corruptions are rejected. Measured current documents are written before first sealing; external integrity and repeat-seal records stay outside the sealed review tree. The initial approval-review rejection did not execute or modify predecessors; its safe resolution is preserved. No sealed predecessor was edited by F16.

F15 completed (2026-10-24): full-battle tactical demonstrations plus matched RL continuations collected 512 training teacher games, 128 separate teacher-probe games, 8,440 RL training games and 4,608 evaluation games in 416.28 seconds on local CPU. All three preregistered benefit gates failed in every seed (0/3 each). Teacher agreement improved from about 25% to 43% after imitation, then fell to 37–39% after RL; tactical-opponent strength remains unestablished. All 16 focused tests and exhaustive fresh replay pass, including nine original-next-update recoveries. The 45 contrasts contain 4 improvements, 4 regressions and 37 inconclusive cells under the registered approximate-bootstrap test. Evaluation retains 12 unresolved cutoffs. F10 remains the reference. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F15 handoff](F15_FULL_IMITATION.md).

F14 completed (2026-10-24): the matched-budget entropy ablation collected 4,512 training and 864 evaluation continuations in 169.41 seconds on local CPU. The preregistered no-entropy benefit gate passed in 0/3 seeds; the descriptive two-point switch-reduction threshold was met in 0/3. All 18 contrasts are retained (0 improvements, 0 regressions, 18 inconclusive under the registered approximate-bootstrap test). Training/evaluation natural terminals: 4,512/864; cutoffs: 0/0. All 14 focused tests and exhaustive replay pass, including six original-next-update recoveries. This adaptively reused development bank does not establish fresh-team or full-battle generalization. F10 remains the reference. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F14 handoff](F14_ENTROPY_ABLATION.md).

F13 completed (2026-10-24): the credit-assignment and continuation canary collected 4,536 training and 864 evaluation continuations in 203.34 seconds on local CPU; all ended naturally. The registered lambda=1 benefit gate did not pass (0/3 seeds); all 18 contrasts were inconclusive. No new policy is promoted; F10 remains the reference. Independent return calculations agree with sealed GAE, and fixed-state probes confirm that F12 increased switching propensity. The new public-state-selected bank has 48 midgame/endgame starts from 192 official train-partition teacher games. Both arms received equal selected-row and optimizer-step budgets. All 15 focused tests and exhaustive replay pass, including six original-next-update recoveries. Full G2/G3/G4 and public strength remain open; final reserved data, public servers and Prime were unused. GitHub remains on the existing visibility hold. See the [F13 handoff](F13_CREDIT_ASSIGNMENT.md).

F12 completed (2026-10-24): the fixed curriculum comparison collected 12,848 training and 6,912 evaluation games in 694.16 seconds on local CPU. The preregistered curriculum-benefit gate did not pass (0/3 seeds). Under the registered approximate-bootstrap test, both arms regressed against base-power in every seed; curriculum also regressed against the control on random/base-power in seed 9311. Neither new arm is promoted over the prior F10 baseline. Both arms received 384 Adam steps per seed and identical PPO/rehearsal budgets; only the starting-state mixture differed, with both training against the same frozen tactical opponent. All 17 focused tests pass. Fresh independent replay reproduces every training cohort, selected batch, optimizer update, all 6,912 evaluation games, 18 stage checkpoints and six original-next-update recoveries. Full G2/G3/G4 and public ladder/generalization claims remain open. Final reserved data, public servers and Prime were unused; GitHub remains on the existing visibility hold. See the [F12 handoff](F12_CURRICULUM_LEARNING.md).

F11 completed (2026-10-24): audited all 3,072 final F10 tactical-opponent games and 78,744 learner decisions. There were 54,900 heuristic disagreements; these are screening signals, not automatic error labels. Among 62 fixed single-action alternatives, 7 changed a loss to a win, 3 worsened a win to a loss, and 52 left the terminal reward unchanged; all ended naturally. Built 65 executable train-only starts from 96 fresh official train games, meeting all seven family quotas with verified unsolved handoffs and full-battle continuations. All 35 focused tests and independent replay pass. No optimizer updates or new playing-strength claim. Next is a preregistered matched-budget curriculum ablation; full G2/G3/G4, public strength and generalization remain open. Final data, public servers and Prime were unused; GitHub stays on the existing visibility hold. See the [F11 handoff](F11_BATTLE_DECISIONS.md).

F11 preserves two completed cross-model reviews, the timed-out partial delegate record, 35 executed tests with command/log/source hashes, all 62 fresh alternative replays, all 96 train-game replays and 65 bank resets. Documentation is updated before sealing; integrity and repeat-seal records stay outside the sealed review directory.

F10 completed (2026-10-24): the fixed longer full-battle comparison collected 12,256 training and 13,824 evaluation games in 1051.30 seconds on local CPU. All-seed full-battle learning gates under the registered approximate-bootstrap development test: plain passed, rehearsal passed; secondary all-seed tactical-retention benefit passed. Both arms received 1,024 new Adam steps per seed (6,144 total), with 786,432 selected PPO rows and 393,216 extra supervised row presentations in the rehearsal arm. All 28 focused tests pass; fresh replay reproduces every full game, batch, update, tactical prediction, 48 periodic checkpoints and six original-next-update recoveries. This is development evaluation against known opponents and scenario families. Full G2/G3/G4 remain open; public servers, final reserved data and Prime were unused. GitHub remains on the existing visibility hold. See the [F10 handoff](F10_LONG_TRANSFER.md).

Both arms improved against random and base-power opponents. Rehearsal retained the known tactical skills in 3/3 seeds versus plain PPO's 0/3, but all nine rehearsal-minus-plain full-battle contrasts remain inconclusive: superiority and equivalence are unestablished. Pooled tactical-opponent wins remain only 4.43% for plain PPO and 4.23% for rehearsal. The critic beats constant 0.5, but a post-hoc constant fitted to each same evaluation cell beats it on random and tactical opponents; calibration and broader transfer remain open.


F9 v2 completed (2026-10-24): the fixed full-battle transfer pilot collected 1,448 training and 1,728 evaluation games in 161.80 seconds on local CPU. The all-seed supervised-rehearsal benefit gate passed. Both arms received 128 new Adam steps per seed (768 total), with 98,304 selected PPO rows and 49,152 extra supervised row presentations in the rehearsal arm. All 20 focused tests pass; fresh replay reproduces every full game, selected batch, update, tactical prediction and six original-next-update checkpoints. This is a small development pilot with known tactical templates, not a public-strength result. Full G2/G3/G4 remain open; public servers, final data and Prime were unused. See the [F9 handoff](F9_FULL_TRANSFER_V2.md).

Tactical retention passed in all three rehearsal seeds and failed in all three plain-PPO seeds. Rehearsal increased final correct-action probability over plain PPO by 8.81, 12.10 and 9.31 percentage points. Neither arm passed the full-battle learning gate in any seed; all rehearsal-versus-plain win-rate contrasts were inconclusive. Terminal-only value MSE remained around 0.25, with mean values near 0.5.

F9 v1 failed before training (2026-10-24): the new admission guard incorrectly disallowed legitimate Showdown unavailable-choice feedback. A bounded replay of the original initial-policy prefix reproduced two trapped-switch responses in development episode7, both seat orientations; the sealed collector refreshed requests correctly. No optimizer update or learning result exists. The failed source, run and diagnostic are preserved; a separate corrected version keeps the same seeds and training budget. See [F9 v1 handoff](F9_FULL_TRANSFER.md).

**F8 completed (2026-10-24):** The [matched-budget PPO comparison](F8_DECISION_RETENTION.md) collected 49,152 training and 6,144 evaluation decisions in 320.98 seconds on local CPU. The supervised-initialization retention gate **failed** (0/3 seeds), due to probability drops in variant contexts. All three supervised arms nevertheless retained **100% greedy accuracy**, versus **25% for fresh arms**, and passed the final probability-advantage gate. Improvement remains incomplete across families. Both arms received 128 new Adam steps per seed; the supervised arm also had F7's prior supervision. All 25 focused tests, every simulator decision/branch and all PPO/checkpoint replays pass. These are local one-decision HP-retention cases with repeated development inputs; no full battles or playing-strength result. Full G2/G3/G4 remain open. Final data, public servers and Prime were unused.

**F7 completed (2026-10-24):** The [supervised fit diagnostic](F7_DECISION_FIT.md) reached **100% final argmax accuracy on all 512 states in all three seeds**, but **all preregistered probability gates failed**. Minimum training-context target probabilities were 86.31%, 89.77% and 89.81%, below the 95% requirement. The unchanged network can express these finite greedy choices; reliable stochastic mastery and the cause of F6's RL failure remain unresolved. The fixed run used 3,072 Adam steps and 393,216 row presentations in 3.38 seconds on CPU, with no new games. All 15 focused tests and exhaustive update/checkpoint replay pass. Full G2/G3/G4 remain open; final data, public servers and Prime were unused. Fable static review found no material defect; primary-result and tie-semantics notes were resolved. Codex metrics delegation received Fable QA and independent local checks. Original-next-update recovery passes for every seed. Historical handoffs and sealed evidence are preserved.

**F6 completed (2026-10-24):** The [decision-point canary](F6_DECISION_CANARY.md) completed 24,576 one-decision training cases and 1,536 evaluation cases in 142.19 seconds on local CPU. The preregistered local learning gate **failed in all three seeds**. The final policies chose Tera almost regardless of context; correct-switch probability fell below 1%. These cases optimize one-turn HP retention, not battle wins. All 96 separate frozen-policy continuation checks completed: 96 natural terminals and 0 cutoffs. All 70 focused tests pass; every training/evaluation decision, PPO update and continuation game reproduced in fresh replay. This does not establish curriculum transfer or public strength; full G2/G3/G4 remain open. Final data and Prime were unused.

**F5 completed (2026-10-23):** [Opponent-mixture comparison](F5_OPPONENT_MIXTURE.md) completed **19,232 training + 13,824 evaluation games** in 1947.61 seconds (32.5 minutes) on local CPU. The preregistered exploratory mixture gate **did not pass**; all three seeds, both arms and 27 contrasts are retained. Each seed–arm received 131,072 selected learner rows and 1,024 Adam steps (786,432 rows / 6,144 steps overall). No tactical improvement over self-play was established; observed tactical wins were lower in every mixture seed. One base-power comparison showed a regression under the approximate bootstrap test. The mixture required 13,184 training games versus 6,048 for self-play and is not promoted. There were 32,968 natural terminals and 88 cutoffs. All 86 focused tests pass; bounded original-update and evaluation replay passes. This measures fresh development games against known heuristic identities, not public ladder or unseen-opponent strength. Full G2/G3/G4 remain open; final data and Prime were unused.

**F4 completed (2026-10-22):** [Full-Showdown learning screen](F4_FULL_LEARNING.md) completed **12,288 training + 18,432 evaluation games** in 2,058.68 seconds (34.3 minutes) on local CPU. All three seeds and nine initial/final opponent comparisons are retained; the preregistered exploratory learning gate **passed**. There were 30,626 natural terminals, 94 explicit cutoffs, 2,367,153 decisions and 9,570 Adam steps. All 64 focused tests pass; bounded replay exactly reproduced the original last cohort/update for every seed and the first paired matchup in every endpoint/opponent cell. This is development learning evidence, not a public-strength result; full G2/G3/G4 remain open. Final data and Prime were unused.

F3 performance update (2026-10-22): [CPU/Metal throughput comparison](F3_THROUGHPUT.md) completed. Eight persistent CPU workers with batched player-only inference achieved 1.87× warm collection throughput (48 matched games: 5.105 → 2.727 seconds). On the same isolated Python/JAX runtime, Metal took 4.841 seconds versus CPU 3.088 seconds, 57% longer. Actual Metal execution and bounded numerical/sampler checks pass. All 47 focused tests pass; actions/outcomes match and next updates agree within tolerance, not byte-for-byte. This is a fixed-policy throughput benchmark, not a new learning screen or strength claim. Keep the current CPU runtime and reuse workers. F2 evidence is preserved; full G2/G3/G4 remain open.

F2 execution update (2026-10-22): [full-Showdown training](F2_FULL_TRAINING.md) completed the fixed three-seed replacement pilot: 768 training + 768 development games, 1,516 natural terminals / 20 cutoffs, 157,983 decisions, 798 Adam steps, 207.45 seconds on local CPU. Observed wins increased in all six seed/opponent cells, but every preregistered gate is inconclusive; value error improved against random and worsened against base power. All 42 focused tests pass; replaying each final cohort exactly reproduced model/optimizer state. The first F2 attempt and its own-Illusion wait-request repair are preserved separately. The completion inventory verifies source/evidence hashes and counters. M7 remains failed; full G2/G3/G4 and public ladder strength remain open. Final data and Prime were unused.

Planning snapshot: 2026-10-10. This record distinguishes source checks from unexecuted implementation requirements.

## Checks performed

- Confirmed the selected Pokémon workspace had no tracked implementation files or local changes before writing the plan. Created documentation only; copied the prior research handoff into this project.
- Independently inspected Prime's public hosted environment, full-finetuning, models/pricing, algorithms, sandbox, availability and provisioning documentation. Links and limitations are in [the assessment](PRIME_INTELLECT.md).
- Read the `HyperPotatoNeo/pokemon-RL` README to verify its Showdown/verifiers/prime-rl architecture and documented Gen9 OU/Gen1 Random Battle examples. Individual source-page fetches failed; no source-level correctness or test-pass claim is made.
- Ran read-only Prime CLI discovery queries. Hub and GPU availability were unauthorized. The availability command's empty array did not establish stock status. No resources were provisioned.
- Independently checked compute arithmetic: 140 GPU-hours; hypothetical GPU subtotals of $280/$560/$1,120 at $2/$4/$8 per hour; 20% planning reserves produce $336/$672/$1,344. Paired evaluation precision uses pairs as clusters, not independent swapped games.

## Cross-model Prime research

Run: `20261010-105746-63b773a3`, project `Pokemon-83aa7cb62b4e`. Fable supplied a research candidate; Codex supplied independent QA. The candidate is **rejected as a complete research result**, though individually verified facts were retained. This is a quality-record verdict, not an access or permission rejection.

Material issues verified by the coordinator:

1. The candidate's claim that all Prime offerings require LLM/vLLM policies was too broad. The sandbox documentation describes generic isolated compute and custom images; CPU Showdown tests are a plausible use. Current GPU sandbox access is restricted/announced as forthcoming, not a basis for claiming CPU infrastructure is irrelevant.
2. It missed the substantive `HyperPotatoNeo/pokemon-RL` reference. Its existence defeats the candidate's implication that the only credible Showdown-environment lead was an unrelated OpenEnv PR. Neither repository availability nor its README establishes a usable Gen9 Random Battles Hub package.
3. The candidate oversimplified algorithm extensibility. The official algorithms page shows a custom loss hook and a PPO-clip example. However, the QA response also overstated its correction: the documentation explicitly says a new algorithm is a named code class rather than an import-path configuration. Loss hooks and algorithm registration are different interfaces. The final assessment states only the verified extension capability and the continuing JAX/token-architecture mismatch.
4. The hosted-service transition and dedicated full-finetuning beta are supported by the public notice; live enforcement and account access were not tested. No claim of exclusively LoRA-based hosted training was retained.

The decision to use an ordinary GPU instance for our custom learner remains a design recommendation. Actual JAX compatibility, prices and capacity are unverified until a selected instance is benchmarked.

## Plan review

A separate Fable snapshot review completed: `20261010-110500-01b7119c`, project `Pokemon-83aa7cb62b4e`. It found specification gaps, not an implementation failure. A Codex subagent independently checked collection, checkpoint and statistical consistency. The following resolutions were verified by reading the affected contracts and checking the relevant math/source:

- Defined a frozen-actor collection barrier that resolves all included per-player transitions before updates, with explicit overflow/padding and pivot/re-request tests.
- Reconciled boundary-only checkpoint support with its actual acceptance tests. Persistent live games must be saved even at update boundaries; partial work rolls back with accounting. Exact mid-epoch continuation remains optional and separately gated.
- Replaced incompatible seed-based/team-based splitting with one canonical-team-content key. Online generation respects the same split; duplicate teams cannot cross it.
- Made both baseline actor and value head PlayerView-only. No privileged critic inputs enter a shared encoder or deployment search.
- Corrected the single-legal-action test: gradient and entropy vanish, but the scalar PPO surrogate can equal minus the advantage. The verification is algebraic: when legal probability is identically one, its parameter derivative is zero.
- Corrected the throughput projection to deduct setup and apply the declared 25% time allowance explicitly.
- Rejected the reviewer's recalled claim that local Showdown ignores explicit teams. Current `Battle.getTeam` returns a supplied team before generation and `setPlayer` calls it. R1/R4 now require runtime team-hash and seat-swap checks at the eventual pin. No custom rule modification is prescribed. [Source checked](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts)
- Clarified that replay equality normalizes declared nonmechanical timestamps/identifiers while retaining raw evidence.

The review snapshot predates the added `pokemon-RL` paragraph and extensibility clarification; those were checked directly against primary sources. Prime service facts are dated and must be rechecked before use. A local structural check confirmed all relative document links resolve, code fences are balanced and files contain no trailing whitespace. No code tests are represented as passing.

## Not performed at the planning stage

No Showdown revision was selected or cloned; no team corpus generated; no simulator/learner implemented; no training, differential tests, battle evaluations or GPU benchmark executed. Numerical gates in these documents are proposed future requirements. No Prime Hub candidate was installed or smoke-tested. Estimated effort, strength targets and compute budgets are not measured outcomes.

Execution update: R1 selected and built Showdown and ran local reference/replay tests; R2 implemented the versioned Node/Python contract; R3 generated and inventoried 100,000 official whole teams; R4 ran and replayed 1,000 episodes with independent Python validation. See [the R1 guide](R1_REFERENCE.md), [R2 review](R2_REVIEW.md), [R3 review](R3_REVIEW.md) and [R4 review](R4_REVIEW.md) for actual checks and remaining scope. The statements above describe the earlier planning stage only.


## R5 implementation and information boundary — 2026-10-10

R5 adds verified same-player history, fixed actor inputs, request-relative action
bindings and separate immutable training targets. Full regression: 52 Node and
41 Python tests passed. The sealed scan reconstructed 1,000 R4 episodes and
81,947 decisions, with 1,000 adapter pairs, 18 live hidden-state pairs, two visible
HP controls, 14 semantic probes, 90 Node/Python action round trips and all 13 R4
fixtures (334 decisions plus 20 bootstraps). Read [the handoff](R5_FEATURES.md),
[review dispositions](R5_REVIEW.md) and [source audit](R5_SEMANTICS_AUDIT.md).
The implementation review found an absorb-healing source-attribution bug; pinned
source inspection and eight live cases verified the correction. The research
candidate was rejected as a complete specification and recorded honestly in the
cross-model ledger. The scan report is `d0f4c82bbc85df7ed1cf54eafa2d3f825decfa58577179cb1ba64eb2a26b66b5`. All data remains
exploratory/unassigned. R6 is next; this stage required local CPU only.


## R6 mechanics inventory — 2026-10-10

R6 inventories 3,509 conservative entries and 10,710 source sites, covers all
1,172 observed initial IDs, and captures 16 scripted reference fixtures in both
seats with exact replay. All custom-engine support remains blocked. The static
review identified a chained-call ID collision; Python rejected the draft, and
adding source end bytes resolved it. See [the handoff](R6_MECHANICS.md) and
[verified review dispositions](R6_REVIEW.md). The bounded research delegate
timed out without a result; no research approval is claimed. R7 is next.

## R7 dataset boundary and benchmark — 2026-10-10

R7 completes the reference/data foundation with a frozen 90/5/5 ordered-team
policy, an audit of 102,000 exposed team identities, 128 fresh reserved final
teams and 96 new train/development episodes. All episodes replayed and passed
Python semantic validation; all 7,782 actor wires passed strict validation.
Collection with replay took 27.391 seconds on four local CPU workers; Python
validation and export took a separate 267.320 seconds. No final battles were run.

The full suite passed 59 Node and 66 Python tests. Fable's implementation review
findings were independently checked; publication-time source checks, seed
prechecks and explicit policy-boundary validation were strengthened. A bounded
Fable identity implementation passed opposite-model QA and corrected local
acceptance checks; the original routing verdict remains unverified because its
supplied test made a no-op mutation. An initial worker-startup failure was fixed
and retained separately from the accepted benchmark. See the [sprint handoff](R7_SPRINT.md)
and [review dispositions](R7_REVIEW.md) for the exact evidence and limits.

M1/G1 CPU learner canaries are next, followed by an explicitly restricted
simulator slice. All custom-engine support remains blocked by the R6 gate.

## M1/G1 CPU learner and restricted transition slice — 2026-10-10

The three predeclared bandit seeds and three delayed shared-outcome seeds passed
the ≥99% optimal-action target within their fixed budgets. GAE/PPO, masking,
analytic gradients and whole-episode checkpoint rollback checks passed. A fresh
rerun reproduced all six checkpoints and six traces byte for byte. The full
regression suite passed 61 Node and 85 Python tests.

The first explicit fixed-team profile, `gen9-rattata-duel-v1`, has an independent
integer CPU transition specification. It matches 10,000 pinned reference cases,
31,303 transitions and all 32 crit/roll combinations with zero disagreements,
conditional on oracle-recorded semantic chance outcomes. This does not complete
G2 or implement JAX. Public-history parity, sampler validation and the accelerated
port remain next work; final-evaluation teams remain unused.

Two Fable implementation reviews were independently checked and actionable gaps
were addressed. Bounded Fable research received Codex QA and was rejected as a
complete executable specification; verified damage arithmetic was retained while
ambiguous roll reconstruction and an incorrect supplied-team claim were discarded.
See [the handoff](M1_G1.md) and [detailed dispositions](../artifacts/reviews/m1-g1/review-dispositions.md).

## M1/G2 restricted JAX slice

The fixed Rattata duel has a batched CPU JAX implementation, independently addressed
chance sampler and semantic player-view comparison against 1,000 fresh reference
games. The older 10,000-game transition corpus remains intact. Full G2 is open;
see [M1_G2.md](M1_G2.md) for the exact scope and remaining hidden-set/coverage gates.

A bounded Fable reducer implementation received Codex QA. Its runner check omitted
PYTHONPATH, so that delivery is honestly scored rejected; independently executed
checks and local corrections support the integrated code. A separate Fable snapshot
review identified low-severity provenance, imported-state and evidence-interpretation
issues, which were verified and resolved. Original reviews, dispositions and final
checks are archived under `artifacts/reviews/m1-g2`.


## M1 JAX autodiff learner and collector

A bounded Fable numerical-core implementation passed its five declared checks,
received accepting Codex QA, and was independently verified with expanded boundary/
finite-difference tests and fixed-budget numerical campaigns. A separate Fable review
of the observation-only collector and checkpoint boundary found no blocking issue.
Verified low-severity improvements make disk resume evidence explicit, derive report
assertions from actual buffer data and correct the collection-failure test label.
See [M1_JAX_LEARNER.md](M1_JAX_LEARNER.md) and
`artifacts/reviews/m1-jax-learner/review-dispositions.md`. Full G2/G3 remain open.

## M1 fixed-shape loop, CPU soak and profiling

The complete fixed-shape collect/GAE/PPO/Adam cycle passed its two-hour CPU soak:
9,885,926 cycles and 3,964,679,976 submitted decisions in the fixed public duel.
No numerical, legality, capacity or unexpected-compilation failure was detected.
Operational throughput was 550,648 decisions/s; peak process RSS was 566.344 MiB.
This does not establish full-format throughput, policy strength or new mechanic
coverage. Full regression passed 206 tests; 37 disk restores reproduced the next
cycle exactly across parity and soak audits.

Fable's bounded numerical helper received Codex QA and passed independent checks
before integration. Separate source reviews of the loop and completion sealer
identified a vacuous compilation-log matcher and weak evidence bindings. These
were independently verified and corrected, including a real JIT positive control,
source-bound regression logs, checkpoint identity/counter checks and preservation
of the interrupted preliminary run. Long-run diagnostics also exposed very small
nominal policy probabilities; an exhaustive FP32 uniform-grid audit documents the
inherited inverse-CDF sampler's tail precision limit. The sealed recipe remains
unchanged, and a versioned precision/floor decision precedes strength promotion.
See [the handoff](M1_FIXED_LOOP.md) and
[verified dispositions](../artifacts/reviews/m1-fixed-loop/review-dispositions.md).

## M1 binary sampling policy

The new two-action law uses an explicit 2^-24 floor, integer-word selection and
matching log/log1p probabilities. A scalar Fraction oracle verifies 100,002
thresholds, with 600,012 injected boundary selections; three-seed loop qualification
and a 60-second CPU run pass. This does not transfer the old two-hour soak to the
new policy, resolve Showdown chance mapping or promote G2/G3. See
[the handoff](M1_BINARY_POLICY.md) for current results and scope.

Auto routing explored a Codex implementation delegate. It failed ownership checks
and supplied pytest tests despite the unittest/no-new-dependency requirement. The
runner refused QA/adjudication of a failed run; its failed status is retained rather
than recorded as a success. The coordinator salvaged the inspected scalar source,
wrote independent unittest checks and obtained Fable snapshot QA. That review found
no blocking policy issue; the inactive rare-field sentinel difference and pinned
tie-gradient convention are documented.

A focused Fable qualification/sealer review prompted diagnostic bounds and explicit
test-suite identities. Its proposed exponent-range expansion was rejected after
checking IEEE bit patterns: exponent126 admits only exactly .5 in this contract.
Unused inherited schedule fields were removed, the actual checkpoint schedule was
made explicit, and the short campaign was rerun under the cleaned plan. Both runs
and the source-bound final regression are preserved in the completion archive.


## M1 Showdown chance mapping and expanded player views

The finite uint32 maps now match the pinned actual Showdown PRNG for critical
hits, damage rolls and context-aware two-element shuffles. All 100,096 upstream
word probes match (500,480 output comparisons); a separate regeneration is byte
identical. The 10,000-game fresh public-duel campaign passes 180,000 player-view
comparisons, including terminal padding, with zero disagreements. Three short
learner runs use the new chance contract and reproduce the next cycle after disk
restore. Full regression passes 229 tests. See [the handoff](M1_SHOWDOWN_CHANCE.md).

Auto routing selected Fable for the bounded arithmetic implementation. Codex QA
identified a hypothetical mixed-device admission edge; the coordinator accepted
only the admitted CPU execution after independent integer-proof, fixture, ownership
and device checks. The original QA verdict and scoped acceptance are both preserved.

Fable's main snapshot review raised evidence questions that were checked against
the recursive reservation gates, predecessor source, actual actor stream and oracle
regeneration. A direct three-seed predecessor collector comparison now confirms
that replacing only the chance dependency produces the same buffers and terminal
states. All 15,626 captured two-shuffle chains were checked separately. A focused
sealer review found no blocking issue; historical report checksum and delegate
outcome bindings were tightened. The source-bound regression was rerun after the
added check. See [verified dispositions](../artifacts/reviews/m1-showdown-chance/review-dispositions.md).

These results do not establish sodium/raw-seed/word-schedule equivalence or cover
hidden-set and broader mechanics strata. The new binary-policy/chance combination
still requires its own two-hour qualification, then preregistered strength evidence.
Full G2/G3 remain open; final evaluation remains reserved and Prime unnecessary.

## M1 selected policy/chance two-hour CPU qualification

On October 11, 2026 the selected binary-policy-1 / showdown-chance-1 combination
completed its own 7,200.006915-second CPU qualification: 8,158,970 complete cycles,
522,174,080 fixed-duel games and 3,272,073,210 submitted decisions. No numerical,
legality, capacity, completion or unexpected-compilation failure was detected.
Operational throughput was 454,454.176 decisions/s and peak process RSS was
473.765625 MiB. The maximum paired-clock host gap was 0.918060 seconds against
the frozen 30-second bound. These measurements describe the tiny public duel under
the observed local machine load; they do not establish full-format performance,
policy strength or device/system memory headroom.

The production report, source/runtime admission and final reservation checks pass.
All 120 minute telemetry rows and their checkpoint counter linkage were verified.
All 13 saved boundaries reproduce the persisted original next-learner and
next-metrics fingerprints exactly in a separate verification process. The existing
235-test regression (65 Node + 170 Python) remains valid for the unchanged manifest
of 206 tested source files; no unrelated tests were rerun for completion.

Fable's initial bounded summary-gate delivery failed two malformed-input exception
checks identified by Codex QA and reproduced locally. That rejected outcome is
retained. Its repaired delivery received accepting Codex QA and passed independent
ownership, input-hash and regression checks before integration. A separate Fable
harness/sealer review identified the macOS monotonic-clock suspension risk; paired
wall/monotonic gap checks and targeted tests resolved it before the run. Other
verified changes added timed-window bounds, final-replay gap checking, a smoke
evidence audit and preservation of incomplete historical reports. Focused follow-up
review found no blocking issue; its requested smoke audit was executed successfully.

Qualification Python, plan, sealer and all sealed predecessors remained frozen
during the production run. Both smoke campaigns and original review dispositions
are preserved. The [handoff](M1_POLICY_SOAK.md),
[verified review dispositions](../artifacts/reviews/m1-policy-soak/review-dispositions.md)
and [completion receipt](../artifacts/reports/m1-policy-soak.json) record the scope
and evidence. Only the selected combination's two-hour stability item closes.
Full G2/G3 remain open; no strength evaluation or final reserved-data use occurred,
and Prime remains unnecessary.

## M2 restricted-policy development evaluation

The locally preregistered pilot completed all 27 cells and 54,000 pinned Showdown
games. Three fresh training seeds each ran exactly 2,048 updates, with fixed
checkpoints at 0/256/2,048. Final random-opponent scores were 63.65%, 62.35% and
62.30%; all adjusted score and paired-improvement bounds met the declared screen.
The stronger damage-greedy baseline held the final policies near 50%, so this is
learning evidence within the public duel, not general strategic strength.

All 169,024 candidate actions, behavior log probabilities and values were reproduced
from each player's recorded request prefix. Public executed actions, terminal turn,
winner, roster, chance-seed namespace and all nine next-update checkpoint fingerprints
were verified. The full regression passed 251 tests (68 Node + 183 Python) with
218 tested source files unchanged. The earlier regression EOF stall and smoke
failures remain recorded; no sealed predecessor was repaired or bypassed to pass.

Fable's bounded statistics delivery initially failed independent overflow probes
despite accepting Codex QA. It was rejected and repaired. The repaired module
matched an independent complete resampling matrix; coordinator acceptance is scoped
to the frozen allocation budget, with Codex QA's generic resource-exhaustion caveat
retained. A separate Fable harness review found no blocking issue. Verified repairs
tightened terminal/roster checks, made the local replay constraint explicit and fixed
the worker EOF lifecycle. No production outcomes informed these repairs or the
frozen endpoint. See [the handoff](M2_POLICY_PILOT.md),
[verified dispositions](../artifacts/reviews/m2-policy-pilot/review-dispositions.md)
and [completion receipt](../artifacts/reports/m2-policy-pilot.json).

Full G2/G3 remain open. Critic errors/reliability bins are descriptive; no calibration
acceptance or device-memory headroom is certified. All 128 final teams remain
reserved, no final episodes were used, and Prime remains unnecessary.


## M3 switching/hidden-set reference scope — 2026-10-11

The [handoff](M3_SWITCH_REFERENCE.md) and
[receipt](../artifacts/reports/m3-switch-reference.json) record the accepted reference
expansion. All 264 tests pass (75 Node + 189 Python), bound to 235 unchanged tested
source/configuration files. The complete 72-script bundle is generated twice and
matches byte for byte. Python independently checks 1,826 live/wait action masks,
5,426 actor inputs, 1,872 hidden-state invariances, positive own/reveal controls,
all reported coverage counts and 3,456 conditional damage values.

Two Fable snapshot reviews were collected using the claude-review skill. Confirmed
findings prompted a separate two-member actor adapter, negative HP tests, complete
counterfactual view retention/reconstruction, independently counted coverage and
exact roster validation. The final evidence expansion is a locally verified repair,
not a claim of a third snapshot review. Review/source deltas, failure dispositions
and corruption probes are retained under `artifacts/reviews/m3-switch-reference/`.
The first full regression passed; it was rerun after the review repairs to bind the
final tree. Draft generation failures remain preserved. A failed codec delegation
was caused by the coordinator's repeated ownership option; none of its candidate
was integrated, and its unavailable QA/adjudication is explicitly recorded.

Full G2/G3 remain open. No new JAX port or learner has been implemented in this
profile; no final data, strength evaluation or Prime resources were used.

## Playable Showdown baseline — 2026-10-11

Implemented a separate public-view heuristic and native WebSocket client. The
accepted local benchmark has 64 complete games across 32 policy-seat pairs,
60 wins against the weak random baseline, 2,904 exact actor-input/action replays,
and zero choice rejections. Three real loopback-server games exercise login,
challenge acceptance, same-connection rematch, fresh login and terminal agreement.
All 134 network choices used the latest received public turn. Twenty new focused
tests and a passing 284-test full regression (95 Node + 189 Python, 247 unchanged
tested source files) are recorded with the completion receipt.

Fable protocol exploration underwent opposite-model Codex QA and was rejected for
insufficient pin-specific evidence; its useful modern-PM challenge finding was
independently verified in the pinned source. The Fable implementation snapshot
raised four issues. Pinned sendUpdates ordering and an oracle test disprove the
main request-before-public-state concern. A real reversed-endpoint challenge-clear
bug was fixed and verified by a same-connection rematch. Developer-format validation
limits and fatal-popup behavior are documented. The reviewer did not execute tests;
subsequent local corrections and snapshot deltas are explicitly preserved.

The first 64-game production replay found a serialization discrepancy after
Illusion replacement; that run is not qualified. Preserving the prior public HP
fixes it, and the replacement run passes every replay. Historical source, handoffs
and evidence remain unchanged except the six established current-progress/registry
files. No public login, ladder play, learning, final reserved-data use or Prime
resources were involved. Full G2/G3 remain open. See [handoff](PLAYABLE_BASELINE.md)
and `artifacts/reviews/playable-baseline/review-dispositions.md`.

## Bounded Showdown session recovery — 2026-10-11

Added a versioned session wrapper and CLI without modifying sealed baseline or
learner source. It reconnects, reauthenticates and rebuilds already-known active
rooms from player-visible history/current requests, respecting the server's
accepted-choice marker. Three reconnects are allowed by default; errors outside
the declared transport class remain visible failures.

The accepted run `artifacts/runs/showdown-session-v1` completes six real local
games through seven injected disconnects. Both players agree on terminal results;
all 120 attempted actions and actor inputs pass the saved-wire audit. One action
is deliberately lost before transport and reconsidered from a fresh request;
another is confirmed accepted and never resubmitted. Nineteen new focused tests
and the full 303-test regression (114 Node + 189 Python) pass with 254 unchanged
tested source files. Four final evidence corruption probes are rejected.

Fable snapshot review identified a real close-handshake error-classification bug.
The repair is confined to the new layer; two regression cases verify decision and
rejoin sends while the socket is closing. Other findings were checked against the
pin, including actual RoomBattle methods driven by a real hidden-trapping rejection.
The additional code/tests are locally verified review repairs, not a second Fable
approval. Review source deltas and prior drafts remain in the review directory.

Independent deployment research received opposite-model Codex QA. Its packet was
rejected for overstating the availability of registration and labeling technical
limits as policy. The coordinator directly read the official announcement in the
browser, verified dates and pinned code, and wrote a corrected deployment note.
The user still needs to identify public accounts; registry rollout, live protocol
compatibility and public login remain unresolved. No staff contact or public game
occurred. The [handoff](SHOWDOWN_SESSION.md) and [receipt](../artifacts/reports/showdown-session.json)
record local reliability evidence only. Full G2/G3 remain open; no learning,
reserved final data or Prime resources were used.

## Frozen Showdown opponent panel — 2026-10-11

Added a four-policy development matrix, public-view damage/tactical heuristics,
paired bootstrap statistics, exact replay auditor and immutable completion sealer.
The fixed 768-game production run covers 128 admitted development teams and all
six unordered matchups with 64 mirrored pairs each. All 35,711 actor inputs/actions,
outcomes and intervals reproduce exactly. All 338 tests (149 Node + 189 Python)
pass, bound to 266 unchanged source/configuration files. Thirty-five focused tests
include complete-grid statistical negatives, fixed known-answer PRNG/bootstrap
fixtures, baseline driver parity, hidden-reserve invariance, actual pinned
Psych Up/Power Swap events and deliberately corrupted smoke evidence. Four
production-evidence copies fail the sealer for their intended reasons.

The tactical heuristic scores 75/128 against the original baseline and 73/128
against damage. Its score against random is 115/128, below baseline's 118/128.
The per-edge paired intervals are descriptive, not multiplicity-adjusted promotion
or no-regression evidence. No policy was changed after production began. The
qualified network client still uses the original baseline.

Two Fable snapshots identified public-memory and bookkeeping issues. Confirmed
Illusion contamination, selective stage events, active-entry hazard penalties and
bench first-turn eligibility were repaired. The second snapshot's Illusion finding
was independently reproduced with previously known disguise moves/items and fixed
using per-appearance reveals. Actual engine events check boost-copy ordering;
source_manifest coverage is explicitly verified. Remaining heuristic approximations
(including copied-ability persistence) are documented. Snapshot/source deltas and
locally verified repairs are distinguished; no third review is claimed.

The delegated statistics implementation originally failed its own duplicate-ID
error-message assertion and received a rejected adjudication after Codex QA.
A fresh isolated repair changes only that message and adds literal numerical
fixtures; all 17 tests and opposite-model QA pass. Coordinator integration checks
include selected-input hashes, patch scope, local execution and independently
computed Python uint32/index/bootstrap values. Both outcomes remain archived.

Collection wrote a complete report and 768 games. Its optional /usr/bin/time
wrapper then exited nonzero because sandbox policy denied kern.clockrate; this is
preserved as instrumentation failure, not a game failure. Fresh replay exits zero
in 22.145 seconds. The initial corruption-test seed mutation and system-Python
probe mistakes were corrected using the actual sodium seed string and project
virtual environment; no producer evidence was changed.

See [handoff](SHOWDOWN_OPPONENT_PANEL.md),
[receipt](../artifacts/reports/showdown-panel.json) and
`artifacts/reviews/showdown-panel/review-dispositions.md`. Next freeze fresh
confirmation and qualify an additive network adapter. Full G2/G3 remain open;
no learning, public games, final reserved-data evaluation or Prime resources
were used.

## Fresh tactical confirmation and local adapter — 2026-10-11

The fixed production campaign completes 12,000 games across six edges, each with
1,000 mirrored team/seed pairs. All 2,000 official teams pass the frozen development
partition and exact-team exclusion against the earlier panel and both milestones'
fixture pools. A separate process regenerates the teams and replays every game:
all 564,379 player inputs/actions, results, counters and statistical bounds match.
Campaign/replay elapsed times are 321.364/313.062 seconds, both exit zero. No
policy tuning, threshold changes, failed-pair replacement or sample extension occurs.

Tactical scores 60.65% directly against baseline, exceeding the 52% point floor;
the nominal adjusted lower bound is 58.55%, above 50%. Its paired improvements
over baseline against random and damage are +2.20 and +10.05 percentage points,
with lower bounds +0.80 and +7.40, both above the declared −3-point tolerance.
All three conditions pass. The 30,000 shared-pair bootstrap replicates implement
an approximate Bonferroni development screen, not guaranteed coverage, a reserved
final promotion or a human/ladder strength estimate. Thirteen unavailable-choice
attempts are retained: five baseline and eight random switches rejected as trapped.

The additive tactical network adapter completes six actual local-server games
through seven injected disconnects. Its wire auditor reconstructs every one of
179 attempted decisions, including a deliberately lost write and a pending
accepted action that is not resent. Four mutated production traces/reports fail
for their intended reasons. Seven temporary production-evidence copies fail the
sealer at the intended failure/progress/source/game/report/network checks, with
no receipt or archive writes. The full 380-test regression (191 Node + 189 Python)
passes, bound to 283 unchanged tested source/configuration files shared by the
campaign and network admissions. Build and syntax verification pass.

A Fable statistics implementation attempt times out without a completed tested
delivery; none of its partial module is integrated. The runner refuses QA and
adjudication of that incomplete run, so neither is claimed. The coordinator's
implementation receives a separate focused Fable snapshot and seven executed
statistics tests, including a literal independently calculated three-metric
fixture. A separate Fable adapter/pipeline snapshot prompts verified guard,
ordering-test and freshness improvements. Subsequent sealer checks are locally
verified repairs; no third snapshot approval is claimed. Snapshot/source deltas,
failed delegation and the first sandbox-denied loopback preflight are preserved.

The [handoff](SHOWDOWN_TACTICAL_CONFIRMATION.md) and
[receipt](../artifacts/reports/showdown-confirmation.json) qualify the unchanged
tactical heuristic for local development play. The separate 1,000-game real
network soak, public deployment checks, expanded scalar/JAX mechanics and broader
learner remain outstanding. Full G2/G3/G4 remain open. No training, public games,
final evaluation data or Prime resources are used.


## Frozen tactical client: 1,000-game local network soak — 2026-10-11

The fixed production run completes 1,000 actual loopback-server games in 169.097
seconds including setup. Twenty login sessions contain 980 rematches, 500 tactical
games in each seat and 60 planned disconnects: 20 before choice, 20 lost writes
and 20 after accepted pending choices. All expected recoveries pass; accepted
choices are not resent and lost writes are reconsidered. Neither policy nor the
sealed recovering client is modified. No failed game is replaced or sample added.

A separate process regenerates the 1,000 fresh official development teams and
reconstructs both players from their received wire histories: all 22,928 tactical
and 23,096 baseline decisions, exact original own-team receipts, results, request
IDs and counters agree. There are zero unavailable-choice retries or campaign
failures. Peak sampled client RSS/heap are 445.063/127.693 MiB; tactical local
request-to-attempt p99 is 2.941 ms. Sampling does not certify continuous peaks,
server/device memory or public-network latency. Server-copy verification matches
all 1,523 unchanged upstream files, with only the three declared configuration,
exact-pool format and private receipt endpoint overrides.

Preflights exposed two distinct integration issues: unregistered identities cannot
use the ordinary `/showteam` chat command, and 164 of the 1,000 official random
teams fail constructed-team legality. The normal Random Battle path bypasses
that validation. The final loopback format instead accepts only exact regenerated
development teams and preserves all set values and resolved battle rules. No
rejected team is edited or replaced. The private test endpoint returns only the
requesting participant's original roster and is ignored by both policies. The v3
preflight audits 324 decisions across eight games and six faults; its real trace
supports deliberate corruption tests. Earlier failures and 41 preflight evidence
files are preserved. Serialization probes distinguish initial move-name
canonicalization from actual mutation; all 1,000 canonical team/receipt roundtrips
and mutation-free admissions pass.

The full regression passes 426 tests (237 Node + 189 Python), bound to 298 unchanged
source/configuration files and the production admission. Build verification passes.
Eight copied-evidence negatives fail for failure marker, progress, source binding,
game hash, report binding, local plugin, exact format source and forged saved audit.
The sealer repeats the entire wire audit instead of trusting its saved result;
negative cases cannot write receipts or archives.

Fable's first isolated scheduler delivery fails one negative-zero test and is
rejected after opposite-model Codex QA. A fresh minimal repair passes all 23 checks,
receives Codex QA, is independently verified and integrated, and is recorded as
accepted. Three Fable snapshots review harness/codec, held-action recovery and
exact-pool admission. Findings are checked against executed tests and pinned source;
reviewers do not execute tests. Source deltas and all dispositions are retained.
The final format snapshot's implementation is unchanged afterward. An initial
inspection mistakenly counted an undefined onBegin property as an active hook;
value/type checks resolve it without any source change.

See the [handoff](SHOWDOWN_NETWORK_SOAK.md) and
[completion receipt](../artifacts/reports/showdown-soak.json). This satisfies the
local adapter volume prerequisite only. Full G2/G3/G4, public deployment and ladder
strength remain open. No training, public game/login, final reserved-data evaluation
or Prime resource is used. Next return to the M3 switching/hidden-set scalar/JAX
transition and observation parity work, then its categorical learner.


## M3 switching/hidden-set scalar/JAX and categorical learner — 2026-10-11

The fixed production campaign completes 10,000 fresh games with zero differences
across 136,635 state boundaries, 273,270 player views, 241,384 decisions, 234,307 events
and 332,815 chance draws, plus the 72 sealed reference scripts. The 24 completely
unpatched passivity controls are rerun by the sealer against the pinned source/build.
All 1,872 hidden equalities, 576 private-speed positive controls and 24 reserve-reveal
pairs pass. Independent compact encoding uses only each receiving player's wire.
Additional JAX private-state mutations, the 3,456-value damage grid and all 357 actual
Showdown HP-display values pass.

The four-action categorical loop completes all three fixed 64-update seeds:
6,144 primary games, 140,983 decisions, 3,469 terminals and 2,675 explicit nonterminal
cutoffs. Every buffer passes an independent NumPy same-player GAE check. Six
original in-memory next updates match fresh disk restores exactly, all nine boundary
checkpoints load, and six injected collect/update failures leave prior state intact.
The regression separately tests interrupted atomic writes, stale partial preservation,
checkpoint corruption, categorical gradients, all masks/tails and three bandit seeds.
All 443 source-bound tests pass (237 Node + 206 Python), with 320 tested files unchanged.
The learner smoke takes 3.700 seconds including its internal audits; process peak
RSS 835.219 MiB also includes the preceding exhaustive parity work, so neither
measurement is a sustained learner qualification. Eleven copied-evidence negatives
reject failure markers, source/runtime/plan drift, incomplete status, broken report
bindings, reserved-data use, missing updates, broken counters and forged original
next-update audits. Production hashes are unchanged; no negative writes a receipt.

Independent damage implementation used Fable and opposite-model Codex QA. An initial
coordinator test wrongly expected Fennekin swift Speed 111; sealed evidence establishes 112.
The delegate also inaccurately reported all checks passing. The first outcome is
preserved as unverified; a corrected, byte-identical rerun passes all checks and is
independently verified before acceptance. Three Fable snapshots cover mechanics,
learner and campaign/sealer evidence. The proposed HP rounding bug is rejected
against pinned source and exhaustive real getHealth tests. Accepted fixes bind
critical/roll/effective final queue-tie traces, use truly unpatched controls, strengthen
legacy/fresh identity checks and re-execute oracle/passivity admission at seal time.
The stale partial-file behavior is intentionally fail-closed evidence preservation.
Snapshot reviews did not execute tests. All findings, source deltas, failures,
outcomes and 17 original preflight files are preserved in the review directory.

See the [handoff](M3_SWITCH_ENGINE.md), [receipt](../artifacts/reports/m3-switch-engine.json)
and [review dispositions](../artifacts/reviews/m3-switch-engine/review-dispositions.md).
This completes the restricted engine and learner smoke only. Its own sustained
CPU qualification and preregistered strength screen are next. Full G2/G3/G4,
public deployment, competitive ladder strength and search remain open. No public
server, final reserved-data evaluation or Prime resources are used.


## M3 switching/categorical sustained qualification preflight — 2026-10-11

The [new handoff](M3_SWITCH_SOAK.md) records the frozen two-hour plan and safe launch
and completion procedure. Additive checkpoint tests demonstrate exact equality
with the unchanged sealed kernels, extended bounded counters, exact disk replay,
partial-write preservation and transaction discard. All 17 focused tests pass.
A 20.064911-second real smoke completes 1,417 updates, 45,344 games and 1,022,876
decisions with zero unexpected compilation events. Four original next-update
learner/buffer/final-state/metric fingerprints and the initial warm-up state reproduce
in a separate process. Nine altered-evidence probes reject without changing the
smoke or writing a receipt; 36 strict-plan mutations also reject.

Cross-model gate implementation initially passed its checks but admitted float-valued
integer plan constants. Opposite-model QA and local reproduction exposed this;
the initial candidate is recorded as rejected. A fresh corrected Codex candidate
passes the stronger checks, Fable QA and coordinator verification before acceptance.
The Fable harness snapshot identified a final-smoke telemetry bracket edge case;
the final restored checkpoint now supplies the upper bracket when the final tick
is absent, with dedicated equality and rejection tests. Slow smoke telemetry remains
fail-closed. No snapshot reviewer executed tests. See the preserved
[dispositions](../artifacts/reviews/m3-switch-soak/review-dispositions.md) and outcomes.
The pinned compiled Showdown build and additive Python compilation verify unchanged.
The complete source-bound regression passes 460 tests (237 Node + 223 Python), with
all 331 tested source/configuration files unchanged. Post-regression admission
exactly matches the passing preflight. Production started at 18:46:36 Eastern on
October 11 in `pokemon-switch-soak-v1`, with idle-sleep prevention and quiet
fifteen-minute follow-up. The completed production result is recorded below.


## M3 switching/categorical sustained qualification completion — 2026-10-11 Eastern

The production report passes after 7,200.008392709307 continuous seconds and the
launcher exits with code 0. Primary work totals 1,711,484 updates, 54,767,488 capped
games and 1,173,453,330 decisions: 50,278,791 terminal games and 4,488,697 explicit
nonterminal truncations. All 120 minute telemetry records link to the initial and
final counters. There are zero unexpected compilations, maximum paired-clock gap
0.551545143 seconds and peak process RSS 482,951,168 bytes (460.578125 MiB).
The positive compile control detects one event and zero on same-shape repeat.

A fresh invocation of the unchanged sealer audit reproduces all 13 original next
learner, rollout-buffer, final-environment and metric fingerprints; it also
recomputes the initial eight-update state and checks checkpoint/telemetry timing
and counter brackets. Full admission matches the launch and passing preflight,
including predecessor archives, runtime and reserved-data evidence. All 331 tested
source/configuration files and the passing log hash remain unchanged, so the
460-test regression (237 Node + 223 Python) is reused. The pinned compiled Showdown
build still verifies against its original receipt. No qualification code, plan,
test, registry, sealer or sealed predecessor was edited to finish this run.

The five current progress documents are updated before creating the immutable
[completion receipt](../artifacts/reports/m3-switch-soak.json). Subsequent full
live/archive hash verification and identical resealing are retained outside the
sealed review directory in the
[integrity record](../artifacts/reviews/m3-switch-soak-integrity.json).
This is restricted CPU stability evidence only. The aggregate value loss and
other numerical diagnostics are not a convergence or critic-calibration screen.
Full G2/G3/G4, public deployment and competitive ladder strength remain open;
no strength screen, final reserved-data evaluation or Prime resources were used.


## M3 screen verification and evidence (2026-10-11 Eastern)

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
