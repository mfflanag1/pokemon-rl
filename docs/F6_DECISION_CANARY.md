# F6 — decision-point learnability canary

Status: completed and exhaustively replayed on 24 October 2026. The fixed local learning gate **failed in all three seeds**.

F5 did not establish a tactical improvement from its opponent mixture. F6 first asks whether the unchanged F1 observation encoder and F2 PPO learner can learn simple switching and Tera decisions. This is a local contextual-bandit diagnostic, not terminal-reward full-battle training and not evidence of curriculum transfer.

## Environment and information

Six-member constructed teams run in the pinned, unpatched Showdown simulator. All teams pass Gen9 Anything Goes legality validation and the frozen R7 train-partition admission. They are not official Random Battle generator teams. Only ordinary actions create the prefix; there is no private state patching. The actor receives only the unchanged `full-showdown-player-v1` player view and 14-action mask. Prefix actions, scenario IDs, expected actions and opponent private data are excluded from actor input and PPO rows.

There are 16 balanced contexts: switching/Tera × Electric/Water threat × two bench orders × both seats. Switching starts with Blissey facing a publicly revealed Thunderbolt or Surf, with Dugtrio and Water Absorb Vaporeon in different reserve slots. The useful switch changes with the threat and order. Tera starts with Ground-Tera Gyarados before activation. Ground Tera blocks Thunderbolt; retaining Water/Flying typing reduces Surf damage. A two-turn ordinary prefix reveals the attack against a bulky Magmar, then brings Gyarados in while the opponent uses a nondamaging stat move. The learner's Tera resource remains available.

Training opponents are Jolteon and Floatzel at level 90. The separate constructed variant panel uses Raichu and Golduck and level-100 teams. All these teams remain in R7 train; "variant" denotes constructed diagnostic development, not the official R7 development split or held-out mechanics families. This small panel was inspected for mechanical validity before learning and supports only narrow variation checks.

## Objective and budget

One sampled learner decision, then the opponent's frozen player-view-only base-power action, resolves one turn. Reward is the exposed own member's post-turn HP fraction divided by its own pre-action HP fraction, clipped to [0,1]. The ratio avoids rewarding sacrifice of an already injured reserve. It ignores future damage, status costs, Tera opportunity cost and battle victory. Therefore a high diagnostic score must not be called stronger full-battle play.

Each row ends this one-decision objective with zero bootstrap; this does **not** declare a natural battle terminal. Prefix decisions are excluded. The unchanged 928-feature, 64-hidden-unit, 14-action network and bounded critic receive unchanged F2 PPO updates (learning rate .0003, two epochs, 256-row valid-padded minibatches, clip .2, entropy .01). There is no imitation target, target-action supervision or oracle reward label: reward comes from observed own HP. The critic predicts local HP retention, not win probability.

The fixed plan is `contracts/decision-canary-v1/plan.json`: seeds 7103, 8209 and 9311; 64 updates ×128 fresh balanced decisions per seed; 8,192 decisions and 128 Adam steps per seed. Total 24,576 training decisions, 384 Adam steps, 49,152 row presentations. All three seeds are retained, regardless of result; there is no seed selection, budget expansion or automatic repair/rerun after failure.

Initial/final evaluation uses eight fresh mechanics realizations per context on both panels: 256 states per endpoint and seed, 1,536 sampled evaluation decisions total. Actor randomness is paired between endpoints. The same panel is shared across seeds, so these are not independent environment-level replications. We report sampled rewards/actions and the exact policy probability on the mechanically verified preferred action. The local gate requires at least +0.15 mean preferred-action probability and final mean ≥0.60 in every family/panel, **plus every individual context's final mean ≥0.60**, for every seed. These are diagnostic thresholds, not statistical significance tests or a strength estimate.

The mechanics probe enumerates every legal first action over 16 mechanics seeds ×16 contexts ×two panels, verifies deterministic prefix replay, and requires the expected action to have uniquely greatest realized local reward. This tests a finite set of chance realizations, not all possible RNG outcomes or full-battle optimality. Conflicting target cases must have distinguishable encoded observations.

After each seed's final policy, 32 separate fresh games (one/context/panel) continue from its first decision under frozen base-power play to natural terminal or turn-120 administrative cutoff. These 96 games test environment continuation only; their wins cannot establish learned full-battle policy strength.

## Verification and next decision

Focused tests cover coordinates/admission, public attack reveals, pre-choice Tera availability, changing correct actions, independently reconstructed HP reward, actor/log-probability isolation, prefix exclusion, zero-bootstrap diagnostic returns, fixed-shape deterministic replay, PPO updates and per-context gates. A fresh audit replays every training and evaluation decision, every PPO update and all continuation games against the pinned engine, including original-next-update and final model/Adam checkpoints. Source/runtime snapshots and predecessor hashes remain bound.

The initial test with one- versus two-row inference shapes found a few-ULP log-probability difference. F6 now fixes inference to eight rows even during replay. The initial failure is preserved. The open-ended delegated design review timed out without a completed result and is not counted as approval. Two focused Fable 5.1 reviews completed; their concrete faint/replacement and report-binding concerns were independently checked and addressed before production. Review packets, dispositions and initial failures are preserved.

Only if the small decision task is learnable should a separately preregistered self-play-versus-targeted-curriculum experiment test full-battle terminal-reward improvement with matched optimization exposure. If the canary fails, inspect representation or learning signal before spending on a larger run. The old C1 3,000-episode schedule remains unexecuted. Full G2/G3/G4, public deployment, ladder strength and #1 remain open. CPU is sufficient; no Prime, final reserved data or public server is used. GitHub publication remains on the existing visibility hold.

## Measured result

Production completed in **142.19 seconds** (2.37 minutes), including startup/JIT, training, evaluations and the separate continuations. Parent peak RSS was **479.3 MiB**, excluding eight Node workers. Compilation was not timed separately. This was not a two-hour qualification.

All three seeds completed 64 updates, 8,192 fresh one-decision rows and 128 Adam steps each: 24,576 training decisions, 384 steps and 49,152 row presentations overall. There were 1,536 initial/final evaluation decisions and 96 additional continuation games. The 26,112 training/evaluation diagnostic boundaries are not natural battle terminals or turn-cap outcomes.

The 96 continuation games produced **96 natural terminals / 0 administrative cutoffs**. Both sides used the frozen base-power policy after the learned first decision; these are continuation smoke results, not a learned-policy win-rate screen.

Preferred-action probability on fresh diagnostic states (initial → final). Each family/panel contains 64 states per endpoint and seed. Minimum is the lowest final mean across its eight contexts. Sampled correct is final count/64.

| Seed | Panel / family | Initial | Final | Minimum context | Sampled correct | Local value MSE, initial → final |
|---|---|---:|---:|---:|---:|---:|
| 7103 | train/switch | 14.9% | 0.8% | 0.8% | 0/64 | 0.1251 → 0.0164 |
| 7103 | train/tera | 14.3% | 47.5% | 3.3% | 32/64 | 0.0798 → 0.0917 |
| 7103 | variant/switch | 14.9% | 0.8% | 0.8% | 0/64 | 0.1077 → 0.0105 |
| 7103 | variant/tera | 14.3% | 47.6% | 3.3% | 31/64 | 0.0986 → 0.0956 |
| 8209 | train/switch | 14.4% | 0.8% | 0.6% | 1/64 | 0.1286 → 0.0077 |
| 8209 | train/tera | 13.8% | 46.5% | 3.5% | 32/64 | 0.0768 → 0.0949 |
| 8209 | variant/switch | 14.4% | 0.8% | 0.6% | 1/64 | 0.0945 → 0.0151 |
| 8209 | variant/tera | 13.9% | 46.6% | 3.6% | 31/64 | 0.0914 → 0.0994 |
| 9311 | train/switch | 14.9% | 0.7% | 0.6% | 1/64 | 0.1236 → 0.0078 |
| 9311 | train/tera | 13.5% | 47.0% | 2.6% | 28/64 | 0.0882 → 0.0805 |
| 9311 | variant/switch | 14.9% | 0.7% | 0.6% | 0/64 | 0.1092 → 0.0252 |
| 9311 | variant/tera | 13.5% | 47.1% | 2.6% | 30/64 | 0.0779 → 0.1139 |

Per-seed preregistered gate results:

- **7103: FAIL** — train/switch: minimum_gain, minimum_final, every_context_minimum; train/tera: minimum_final, every_context_minimum; variant/switch: minimum_gain, minimum_final, every_context_minimum; variant/tera: minimum_final, every_context_minimum.
- **8209: FAIL** — train/switch: minimum_gain, minimum_final, every_context_minimum; train/tera: minimum_final, every_context_minimum; variant/switch: minimum_gain, minimum_final, every_context_minimum; variant/tera: minimum_final, every_context_minimum.
- **9311: FAIL** — train/switch: minimum_gain, minimum_final, every_context_minimum; train/tera: minimum_final, every_context_minimum; variant/switch: minimum_gain, minimum_final, every_context_minimum; variant/tera: minimum_final, every_context_minimum.

All per-context probabilities, sampled outcomes and all seed-level results are retained in the production report. No seed or case was dropped. The separate constructed variant panel changes foe species/level but reuses the same two mechanics families; results do not establish held-out-family, official full-battle or public-ladder generalization.

Verification: 36 Node +34 Python tests passed on 416 source/configuration files. The finite mechanics probe covered 512 contexts and 3,584 legal-action branches; every expected local optimum was unique (minimum observed reward gap 0.1142857). Train/variant ordered-team identities were disjoint and no conflicting preferred actions shared an encoded input/mask. Fresh audit exactly reproduced all 24,576 training decisions, all 1,536 evaluation decisions, all 192 PPO updates and all 96 continuation games, including original-before-last and final parameter/Adam states. The F5 receipt, all 808 live/archive source checks and 3,691 evidence hashes remain intact.

[Production report](../artifacts/runs/f6-decision-canary-v1/report.json) · [Exhaustive audit](../artifacts/reviews/f6-decision-canary/production-audit/audit.json) · [Completion receipt](../artifacts/reports/f6-decision-canary.json).

Next: diagnose the failed contexts and test representation/optimization learnability before a larger curriculum run. This fixed canary will not be extended or rerun to obtain a passing result.

## Failure diagnosis from the saved checkpoints

The final policy put approximately **89–92% probability on `move 1 terastallize` in every training category**, including Blissey switching cases and Water attacks against Ground-Tera Gyarados. Consequently, it scored well in the Electric Tera case but poorly when the context required a different response. Correct-switch probability fell from roughly 14–15% at initialization to below 1%. Mean sampled local reward across both panels rose (7103: 0.605 → 0.770; 8209: 0.611 → 0.770; 9311: 0.636 → 0.761), while the conditional-choice gates failed. This is an observed context-insensitive shortcut, not evidence that switching is unavailable or that the simulator is broken. The mechanics probe confirmed that better legal actions exist.

A post-hoc NumPy forward-pass diagnostic on the saved evaluation panel found 58–64% of final hidden activations above |0.95|, compared with zero initially. This threshold was selected after seeing the failure; the finding does not establish saturation as the cause. Neither a failed small RL budget nor distinguishable input encodings proves that this architecture cannot represent the conditional policy. The lower local value errors likewise do not establish better battle strategy.

The next bounded investigation should first test whether the unchanged network can fit the mechanically verified conditional choices on diagnostic training states and preserve them on the existing variant panel. That would be an explicitly supervised representability test, not new RL evidence. If it can, test the RL optimization/exploration mechanism with one preregistered change at a time. If it cannot, improve action-conditioned observation features before spending on larger full-battle curriculum training. No such follow-up or additional training was performed in F6.

[Saved-policy action probabilities](../artifacts/reviews/f6-decision-canary/observed-policy-behavior.json) · [Post-hoc activation diagnostic](../artifacts/reviews/f6-decision-canary/hidden-activation-diagnostic.json).
