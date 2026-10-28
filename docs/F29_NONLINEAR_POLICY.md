# F29 — Nonlinear public action scorer

Status: fixed production and exhaustive replay passed; completion sealing follows the measured documentation below.

F28 did not establish a battle-strength benefit from its larger collection batch. Tactical value errors improved without a reliable increase in wins. F27 found no local sign error or strong systematic critic/actor conflict at its audited anchors. These results leave representation and long-horizon optimization open; they do not establish either as the cause.

F29 tests a narrow actor-capacity hypothesis. The current policy processes its 25 explicit matchup channels per action through a linear residual. The new branch combines those channels with the existing 64-dimensional public observation trunk through a width-32 tanh layer and a shared scalar action scorer. It uses the same 1,278 input channels and adds 2,912 parameters. The critic head, observation encoder, Showdown engine, opponent mixture, PPO and GAE rules are unchanged. Because the actor shares the old trunk, actor gradients can still change that trunk and affect value predictions. This is a capacity/interaction intervention, not an isolated test of nonlinearity or a remedy already demonstrated by F28.

## Registered comparison

- Seeds: 7103, 8209, 9311. Each starts from its F25 final matchup checkpoint, with reset Adam. No seed or checkpoint selection.
- Arms: `control` disables the new branch; `nonlinear` enables it. Both carry the same 63,718 parameters. New local/context weights have deterministic FP32 initialization; the new output weights are zero. Initial logits, values and seeded action law must match the predecessor and each other before training.
- Each arm/seed: 256 cohorts × 1,024 selected rows. Every cohort selects 512 self-play and 512 tactical controlled-player rows after complete-game GAE, holding the policy fixed through both streams. Eight-game chunks; at most 64 games per stream. Two PPO epochs, 256-row minibatches, eight Adam steps per cohort. No opponent rows or evaluation data enter updates.
- Total: 1,536 cohorts, 12,288 Adam steps, 1,572,864 selected rows and 3,145,728 row presentations. Both arms have equal selected-row/update/refresh budgets. Actual game counts, trajectories and compute can differ and are measured.
- Evaluation at cohorts 0, 128, 256 (0, 131,072, 262,144 selected rows). Random, base-power and tactical opponents each receive 128 common paired schedules in both seats. Total 13,824 opponent games and 576 separate self-play diagnostic games. New development schedules exclude admitted exact-team histories through F28 and F29 engineering preflight; recurring official generation is not held-out-team generalization.
- Checkpoints at 0, 127, 128, 255, 256: 30 comparisons, including 12 original-next-update recoveries before updates 128 and 256. Full original policy/value decisions, outcomes, row selection, optimizer metrics and states must reproduce during fresh exhaustive replay.

## Outcomes fixed before production

The **primary** outcome is trained-on tactical adaptation: each arm's final improvement over its own initialization and nonlinear minus control. Each requires at least five percentage points of cutoff-aware gain, positive lower bound from the registered approximate familywise paired bootstrap, no detected random/base-power regression, and final cutoff rates at most 5%. All three seeds must pass for an all-seed claim.

Base-power learning and nonlinear benefit are secondary, using the same 27 dependent final contrasts and adjusted intervals. Midpoints, familiar mastery probes, terminal-only value MSE against constant 0.5 and self-play diagnostics are descriptive. Cutoffs retain their unresolved reward bounds. No early stopping, budget extension, cherry-picked cells or candidate promotion. Completion can coexist with failed or inconclusive gates.

The fixed contract is `contracts/nonlinear-policy-v1/plan.json`. Source-bound checks must cover function preservation, exact control arithmetic, independent forward/gradient tests, nonlinear interaction and gradient reachability, legal/wait inference, full-game row selection, checkpoint inventory/counters, role boundaries and regression/cutoff gates. Engineering preflight runs all six host/arm combinations and reproduces their original collection/update before production. Pinned simulator build and independent review dispositions must be preserved.

## Execution and boundaries

Production: `artifacts/runs/f29-nonlinear-policy-v1`. Review/replay: `artifacts/reviews/f29-nonlinear-policy`. A single persistent driver will run the fixed comparison, exhaustive replay, measured documents, copied-evidence rejection checks, sealing, every live/archived source/evidence hash, and identical repeat sealing. Failure preserves evidence and stops; no automatic repair, restart or resampling. Post-seal verification stays outside the sealed review directory.

This advances the original objective by testing whether a richer public action representation learns useful full-battle choices. It is still a small model against familiar development heuristics, not a reproduction of Jaxcalibur's transformer-scale self-play result. Full G2/G3/G4 and public strength remain open. No public server, final reserved data or Prime. GitHub remains on the unresolved visibility hold.

## Qualification evidence

The pinned Showdown build passed. All 25 source-bound tests passed on CPU FP32 with 716 executable/configuration files: 7 scope/host/checkpoint/gate checks, 12 learner checks, 3 actor checks and 3 collector checks. The original learner delivery had two invalid-dtype test-fixture failures; both are preserved. A separately reviewed NumPy-fixture repair passed all 12 learner checks without changing learner arithmetic or runtime settings. Independent harness and opposite-model learner review dispositions are preserved in the review directory.

The six host/arm engineering preflights passed in 88.16 seconds. Each collected 32 games and selected 1,024 rows for eight optimizer steps; every collection and original update reproduced exactly. Both arms had identical initial game/action/value ledgers for each seed, and all updated checkpoints round-tripped. These are engineering games and do not establish learning. No final evaluation data was used.

## Measured execution

# F29 measured results

Production completed in 3016.79 seconds on CPU. Replay/receipt verification remains a separate requirement.

- training: 57232 games, 57208 natural terminals, 24 cutoffs, 3075592 attempted decisions.
- evaluation: 13824 games, 13819 natural terminals, 5 cutoffs, 788904 attempted decisions.
- selfplay_diagnostics: 576 games, 574 natural terminals, 2 cutoffs, 36580 attempted decisions.

All-seed learning: {'control': False, 'nonlinear': True}; all-seed tactical feature benefit: False; secondary base-power benefit: True. Contrasts: {'improvement': 11, 'inconclusive': 16}.

| Seed | Arm | Base-power wins: initial → final /256 | Tactical wins: initial → final /256 | Learning gate | Familiar final mastery |
|---|---|---:|---:|---|---|
| 7103 | control | 138 → 136 | 21 → 22 | False | False |
| 7103 | nonlinear | 138 → 195 | 21 → 72 | True | False |
| 8209 | control | 151 → 142 | 29 → 28 | False | False |
| 8209 | nonlinear | 151 → 189 | 29 → 59 | True | False |
| 9311 | control | 139 → 137 | 26 → 34 | False | False |
| 9311 | nonlinear | 139 → 175 | 26 → 56 | True | False |

Peak parent RSS 1986.6 MiB (workers excluded). Collection 2825.35s; optimizer calls including compile 44.52s. Finite gradients/likelihood ratios are engineering telemetry, not trust-region or calibration guarantees.

Both arms use the same F25 final matchup host and reset Adam. Selected-row and optimizer budgets match; games and compute are measured, not matched. The nonlinear arm adds2912 actor parameters and permits feature/context interactions with possible shared-trunk feedback; it does not isolate nonlinearity from capacity. Endpoint0 repeats are deterministic and contrasts are dependent; all27 stay in the approximate family adjustment. Tactical is trained on and is the primary adaptation gate. Base-power learning and benefit are secondary. Fresh development schedules against recurring generation and known heuristics, not public ladder or held-out-team generalization. All seeds/27 contrasts retained; intervals and cutoff-aware gates remain in measured-review.json. No promotion, final reserved data, public server, Prime or GitHub publishing.

