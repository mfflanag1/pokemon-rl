# F11 — full-battle decision audit and broader train scenarios

Completed 2026-10-24 on local CPU. The fixed audit replayed every final F10 game against tactical-v1: **3,072 games, 149,566 total decisions and 78,744 learner decisions**. All games ended naturally; 24 advertised-but-unavailable attempts were retained. All three seeds and both arms remain in the census. Production completed in 98.887 seconds, excluding preflight and the independent replay pass. This is an adaptive development audit, not a new learning experiment or strength result.

The policy disagreed with the frozen public-information tactical heuristic on **54,900 decisions (69.72%)**. Disagreement is a screening signal, not an error label. In **62 fixed single-action alternatives**, **7** changed a recorded loss into a win, **3** changed a win into a loss, and **52** left the terminal reward unchanged; no branch ended at a cutoff. These are single conditional simulator paths, not expected action regret or an optimal-policy comparison. The same original state and opponent action are retained at the intervention; the learner then resumes its own frozen stochastic policy. Changes to a move and its Tera flag are one composite action, so their effects are not separately identified. The cases share some F10 team/mechanics settings across seeds and arms and are not independent population samples.

The resulting broader bank contains **65 train-only decision starts**, from **96 fresh official training-partition games** (5,211 decisions, 96 natural terminals, zero cutoffs; five learner wins). Every scenario is stopped before the target choice, retains the full action space and has a verified complete-battle continuation. All seven family quotas were met: four entries per seed/family, 12 per family across seeds and 84 overlapping memberships in total. Some starts cover multiple families, hence 65 unique starts rather than 84. There were **zero optimizer updates**; improved policy performance from this bank remains untested.

## What the audit found

- **Immediate move choice remains weak.** There are 10,247 chosen damaging actions whose public heuristic damage estimate trails another legal move by at least 0.20 expected HP fraction. This proxy is incomplete mechanics reasoning and is not ground truth. Damage-to-damage disagreements total 20,054, including choices of move slot or Tera variant.
- **Recovery and setup timing need broader contexts.** The learner chose Recover at full HP 348 times, including 35 actions also spending Tera. A concrete seed 7103 plain case uses Tera+Recover with full-HP Hydrapple against Gastrodon; substituting Fickle Beam changes that single path from loss to win. A defensive Tera decision can have value separately from recovery, so counts alone are not proof every bundled action is inferior.
- **Type matchups and Tera interact.** In seed 7103 rehearsal, Zoroark chose Tera+Focus Blast against Galarian Slowbro; the Dark Pulse alternative won that path. Two Lunala cases chose Moonblast against Gholdengo; Moongeist Beam alternatives won their respective paths. These establish useful training examples, not universal action rules.
- **Replacement choice matters.** One selected faint-replacement alternative changed a loss into a win. Across the complete census, 8,842 forced-replacement choices disagree with the heuristic.
- **Blind teacher imitation is not justified.** Three substitutions made the outcome worse. None of the 12 selected hazard-tagged alternatives improved terminal reward, and voluntary-switch alternatives had no improvements and one worsening. Keep these families as coverage, without claiming this audit demonstrated a hazard or voluntary-switch remedy.

| Seed | Arm | Wins | Learner decisions | Heuristic disagreements |
|---|---|---:|---:|---:|
| 7103 | plain | 21/512 | 13,031 | 9,128 |
| 7103 | rehearsal | 25/512 | 13,056 | 9,084 |
| 8209 | plain | 17/512 | 13,330 | 9,325 |
| 8209 | rehearsal | 20/512 | 13,158 | 9,148 |
| 9311 | plain | 30/512 | 12,983 | 8,988 |
| 9311 | rehearsal | 20/512 | 13,186 | 9,227 |

The pooled 133/3,072 wins are descriptive reuse of F10 endpoints. They are not a new independent evaluation or a basis for an improved strength claim.

## Decision-family coverage

Tags overlap. Branch columns refer to family memberships, so they sum beyond 62. The fixed selection was first two qualifying disagreements per family/cell, in saved game/decision order, at most one selected point per game. Neither selection nor the train-bank selector receives the outcome. Every selected branch is retained.

| Family | Census disagreements tagged | Branch better / worse / same | Train-bank memberships |
|---|---:|---:|---:|
| damage_choice | 10,247 | 3 / 2 / 7 | 12 |
| forced_replacement | 8,842 | 1 / 0 / 11 | 12 |
| voluntary_switch | 5,458 | 0 / 1 / 11 | 12 |
| tera_timing | 3,216 | 3 / 1 / 8 | 12 |
| status_setup_recovery | 16,462 | 3 / 0 / 9 | 12 |
| hazards | 4,122 | 0 / 0 / 12 | 12 |
| hidden_information | 9,857 | 3 / 1 / 8 | 12 |

The hidden-information tag means the active opponent has not publicly revealed a damaging move. It is not a belief-learning or hidden-set inference benchmark. Of 13,627 tagged decisions, 3,123 (22.92%) are on turn 1; new unrevealed active opponents also trigger it later. Eleven of the 12 selected hidden-information branch memberships occur on turn 1, with one on turn 4. All 12 hidden-information bank memberships occur on turn 1, so this part of the bank has an opening bias. Additional midgame belief scenarios remain a future expansion.

## Broader training scenarios

The bank uses the same-seed F10 rehearsal final policies against tactical-v1 on 32 fresh train games per seed, alternating learner seats. The new training seed namespace is disjoint from F10. Whole teams are generated and admitted by the unchanged official F2 R7 partition gate; no constructed replacement teams, development states, final-reserved data or private opponent sets are copied into training. The audit's development findings influence only family priorities and the proposed recipe. This adaptivity is explicit.

Each scenario stores the original train coordinates, team provenance, exact action prefix, per-seat RNG counters and target public-frame/feature fingerprints. `pokemon_battle_audit.core.reset_scenario(worker, scenario)` regenerates the official game, checks every prefix transition, and returns a live frame **before** the target action. Prefix choices are excluded from learner rows; subsequent play should use normal full-battle rewards, with natural terminals distinct from administrative cutoffs. There is no state injection and no heuristic label supplied to the learner.

- [Seed 7103 scenarios](../artifacts/runs/f11-battle-decisions-v1/training/7103/scenarios.json): 22 unique starts.
- [Seed 8209 scenarios](../artifacts/runs/f11-battle-decisions-v1/training/8209/scenarios.json): 22 unique starts.
- [Seed 9311 scenarios](../artifacts/runs/f11-battle-decisions-v1/training/9311/scenarios.json): 21 unique starts.
- [Curriculum recipe](../artifacts/runs/f11-battle-decisions-v1/curriculum-recipe.json): proposed prefix fractions 10%, 20%, 25%, with the remainder fresh official-start games. These stages have **not** been trained.

Family weights are one plus the count of favorable selected conditional branches, normalized. All families keep positive weight. These weights are a proposal based on development observations, not estimated optimal sampling proportions. Choose a family, then sample uniformly among its bank entries across seeds; overlapping scenarios are eligible under each of their assigned families.

| Family | Proposed family weight |
|---|---:|
| damage_choice | 20% |
| forced_replacement | 10% |
| voluntary_switch | 5% |
| tera_timing | 20% |
| status_setup_recovery | 20% |
| hazards | 5% |
| hidden_information | 20% |

Next register an equal-PPO-row/optimizer-budget curriculum-versus-official-start ablation across all three seeds, with fresh development games against every frozen opponent. Measure full-battle wins, regressions, terminal/cutoff counts, action error proxies and terminal-only value behavior. The small recurring bank can be memorized; improvements on its own starts do not establish transfer. No further training was launched in F11.

## Validation and limits

**35 focused tests pass**: 29 Node and 6 Python, bound to 470 source/configuration files and executed command/log digests. Pinned Showdown builds and verifies unchanged. Tests cover public/private isolation, exact actor seeds and values, legal action/phase diagnostics including Recharge/Struggle, invalid/final admission, tampered trace fingerprints, train-only prefix resets, JSON round trips, and unchanged projection/unavailable-choice behavior.

Production freshly replayed all 3,072 original games, checking every projected feature hash/request/action, frozen opponent choice, outcome and unavailable-choice count. Original F10 learner values were already sealed; F11 additionally resamples exact original action/value at each of 62 intervention targets. The independent verifier reaggregates all 78,744 diagnostic rows and fixed selections, freshly reproduces every alternative continuation and all 96 train games, and resets every one of 65 bank entries. Production also exactly reproduced each full suffix after resetting its scenario. Tests and source snapshots match the final tested tree.

Two completed cross-model reviews and verified dispositions are preserved. A separate implementation delegate timed out with only a partial descriptor; that failure is retained and not counted as successful delivery. Codex completed local testing and inspection, then the focused second review covered the descriptor. No unexecuted reviewer opinion is presented as a test.

Full G2/G3/G4, public ladder strength, expected-regret estimation, critic calibration and broader curriculum transfer remain open. No public server, Prime or final reserved-data use. GitHub remains on its existing visibility hold. All sealed predecessors, including F10, are preserved.

## Evidence and reproduction

- [Fixed plan](../contracts/battle-audit-v1/plan.json)
- [Production report](../artifacts/runs/f11-battle-decisions-v1/report.json)
- [All alternatives](../artifacts/runs/f11-battle-decisions-v1/branch-report.json)
- [Independent verification](../artifacts/reviews/f11-battle-decisions/verification/verification.json)
- [Review dispositions](../artifacts/reviews/f11-battle-decisions/review-dispositions.md)
- [Completion receipt](../artifacts/reports/f11-battle-decisions.json)

Do not overwrite the production directory or sealed inputs. The report was generated with `PYTHONPATH=python .venv/bin/python -m pokemon_battle_audit.campaign --run artifacts/runs/f11-battle-decisions-v1`; that path now refuses reuse. Verification uses `python -m pokemon_battle_audit.verify --run RUN --evidence NEW_DIRECTORY`. Post-seal verification output must stay outside the sealed review directory. Use `python -m pokemon_battle_audit.seal --run RUN --review REVIEW` only against unchanged evidence; a repeat must produce the identical receipt.
