# M3 restricted categorical learning screen

The production run and its action/outcome audit completed, and **all three seeds
failed the preregistered learning gate**. All 27 evaluation cells are retained.
Final verification additionally requires the receipt and integrity record below.
Full G2/G3/G4 and public ladder strength remain open; Prime is unnecessary.

## Measured results

The unchanged learner trained 24,576 updates, 786,432 capped games and 18,615,320
player decisions across seeds 1009/2027/3037. Training produced 265,205 terminal
and 521,227 explicitly truncated games. Training plus boundary replay took
105.715 seconds; complete training/evaluation/action-audit wall time was
5853.584 seconds. Timing concerns this local CPU implementation, including host
checks; it is not full-format or GPU throughput.

All **31,104 pinned Showdown evaluation games** completed the declared horizon:
**12,837 terminal**, **18,267 nonterminal cutoffs**, and
**370,741 candidate decisions**. Each seed contributes 9 cells
and 10,368 games. Nine saved checkpoints reproduce the original next learner,
rollout buffer, final environment and metric fingerprints exactly. Every stored
candidate action, behavior log probability and value is reconstructed from the
original receiving-player protocol prefix.

Primary endpoint versus random legal (1,152 games/576 complete seat pairs per seed):

| Training seed | Secured wins | Final outcome range | Initial outcome range | Observed secured-win change | Adjusted worst-case improvement lower bound | Gate |
|---|---:|---:|---:|---:|---:|---|
| 1009 | 3.99% | 3.99%–89.24% | 12.24%–86.98% | -8.25 pp | -95.88 pp | Fail |
| 2027 | 0.95% | 0.95%–89.32% | 11.72%–87.50% | -10.76 pp | -99.44 pp | Fail |
| 3037 | 0.43% | 0.43%–90.80% | 14.24%–87.59% | -13.80 pp | -100.00 pp | Fail |

The outcome range is [wins/games,(wins+cutoffs)/games], bounding arbitrary binary
assignments to unresolved games. It does not simulate a longer battle. The adjusted
improvement bound compares final wins with initial wins **plus all initial cutoffs**,
then applies the preregistered pair-level Hoeffding radius. All three seeds must
pass; there is no seed pooling or best-checkpoint selection. Secured wins fell
in every seed, each final secured-win fraction is below 55%, and every adjusted
win lower bound is zero. The adjusted improvement bounds are also negative. These
are development results within the constructed 24-team profile; they do not
establish full-format generalization.

All frozen opponent endpoints (each 1,152 games):

| Seed | Opponent | Wins | Losses | Cutoffs | Outcome range |
|---|---|---:|---:|---:|---:|
| 1009 | random | 46 | 124 | 982 | 3.99%–89.24% |
| 1009 | damage-greedy | 21 | 630 | 501 | 1.82%–45.31% |
| 1009 | frozen-initial | 39 | 113 | 1000 | 3.39%–90.19% |
| 2027 | random | 11 | 123 | 1018 | 0.95%–89.32% |
| 2027 | damage-greedy | 17 | 424 | 711 | 1.48%–63.19% |
| 2027 | frozen-initial | 10 | 107 | 1035 | 0.87%–90.71% |
| 3037 | random | 5 | 106 | 1041 | 0.43%–90.80% |
| 3037 | damage-greedy | 10 | 400 | 742 | 0.87%–65.28% |
| 3037 | frozen-initial | 5 | 107 | 1040 | 0.43%–90.71% |

All initial and 2,048-update cells remain in the
[complete report](../artifacts/runs/m3-switch-screen-v1/report.json). No outcome-based
rerun, budget extension, seed selection or parameter change was made. Critic values
and terminal-only value errors are descriptive; no calibration/convergence gate is
claimed, and no critic prediction decides a cutoff outcome.

## Frozen subject and protocol

The [plan](../contracts/switch-screen-v1/README.md) and its JSON contract were frozen
before training. Three fresh seeds each receive exactly 8,192 updates and retain
checkpoints 0/2,048/8,192. The opponents are equal-logit random legal, immediate
damage-greedy with forced switches only, and each seed's frozen initialization.
Each cell covers all 24×24 ordered candidate/opponent team combinations in both
seats. The same team pair and mechanics seed are shared across seats/checkpoints;
diverging actions do not promise identical realized chance events. Candidate and
opponent randomness have separate role addresses and smoke uses a separate namespace.

This uses the unchanged sealed M3 scalar/JAX semantics, 323-feature actor/value
model, categorical finite-word law, Showdown chance mapping and PPO kernels.
The [two-hour qualification](M3_SWITCH_SOAK.md) and historical evidence remain
immutable. Only the new screen modules and current registry/progress documents
were added or updated. Actors receive only their own compact player view and mask;
seed/team/pair metadata addresses RNG and never enters their features.

A twelve-turn cutoff remains nonterminal. The inferential unit is a two-seat pair;
one-sided bounds assume independent per-pair chance/policy streams conditional on
the trained model and fixed team grid. A union bound allocates .05/6 over the six
primary bounds. Deterministic PRNG namespaces are not a proof of independence.
The gate requires each final random-opponent secured-win fraction≥55%, adjusted
win lower bound>50%, and adjusted worst-case improvement lower bound>0.
Unexpected ties fail closed; simultaneous-faint effects and PP exhaustion are
outside this admitted profile. No cutoff becomes a draw or terminal result.

## Verification and artifacts

The [completion receipt](../artifacts/reports/m3-switch-screen.json) and passing
[integrity record](../artifacts/reviews/m3-switch-screen-integrity.json) are required
to establish final verification. The receipt binds the source, immutable predecessors,
report, review and original evidence. Sealing
requires a fresh original-next-update replay for all nine checkpoints, complete
policy/value replay from every original player-view prefix, and **fresh replay of
all 31,104 games** through unpatched Showdown with exact view hashes and complete
final-protocol equality. It then requires an identical reseal and live/archive hash
verification, retained outside the sealed review directory in the integrity record.
The production report alone does not establish that these independent checks passed.

The complete source-bound regression passes **476 tests (240 Node + 236 Python)**
on 347 unchanged implementation/configuration files. The pinned Showdown build
verifies. The final 48-game smoke reproduces both checkpoint updates and all 48 full
reference games. Nine copied-evidence negatives reject smoke promotion, source/
runtime drift, reserved-data use, missing cells, broken counters/checkpoint outputs,
changed action logs, false terminal outcomes and changed view prefixes. An earlier
smoke remains preserved before the final replay/freeze checks were added.

Independent work used the
claude-review skill.
The bounded statistics implementation auto-routed to Codex, received Fable QA, then
passed five local frozen tests plus 6,561 exhaustive small outcome comparisons.
Two Fable harness snapshots and coordinator verification resolved the conditional
damage-helper, horizon and provenance concerns; see
[review dispositions](../artifacts/reviews/m3-switch-screen/review-dispositions.md).
Static reviewer opinions are not represented as executed tests.

Production directory: `artifacts/runs/m3-switch-screen-v1`. The launcher exits 0;
`production.log`, `launch.json`, `exit.json`, the timestamped pre-training source/
runtime freeze and verification records remain under
`artifacts/reviews/m3-switch-screen`. The frozen timestamp is local provenance,
not external timestamp attestation. A quiet fifteen-minute follow-up finishes verification and sealing, then pauses.

```sh
# Always choose a fresh output directory; do not overwrite completed evidence.
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW --smoke
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW --audit-smoke
# Production additionally requires the exact pre-training review admission freeze.
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW-development
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW-development
```

## Next gate

Inspect the recorded action distributions, reward and truncation handling, and
unresolved-outcome counts before designing another experiment or increasing compute.
The current results do not establish the cause of the decline in secured wins. Broader claims require an expanded admitted
horizon/history/PP scope, fresh scalar/JAX/Showdown and information parity, then a
new preregistered evaluation. Do not tune on these outcomes and present a rerun as
confirmation. The restricted actor remains separate from the playable full-battle
client. Public deployment, competitive rank, stronger opponents, calibrated values,
beliefs/search and full G2/G3/G4 remain open. No final reserved teams, public server
or Prime resources were used.
