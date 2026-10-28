# F2 — checkpointed full-Showdown learning pilot

Status (2026-10-22): replacement pilot completed with exit 0; 42 focused tests pass. All three seeds recorded more development wins against both opponents, but every preregistered learning gate is inconclusive. Run: artifacts/runs/f2-full-training-v2. No public strength claim. This additive track makes full-reference learning the next performance experiment. M7 remains failed; its outputs and all sealed predecessors are unchanged.

The local pinned Gen9 Random Battles engine supplies complete six-member games. The new session admits official teams from the frozen R7 **train** partition for updates and **development** partition for evaluation. It refuses final, arbitrary plans and curriculum inputs. Actor inputs reuse the F1 player projection, 928-feature lossy encoder and 64-unit model; missing battle state/history is a known baseline limitation, not resolved by the full mechanics engine.

`contracts/full-training-v1/plan.json` fixes three seeds (1103, 2207, 3301), 256 self-play training games each, eight frozen-policy games per collection cohort, two PPO epochs with 256-row masked minibatches, and a 120-turn boundary limit. This is 768 training games. Every initial and final policy faces random and raw-base-power opponents on 32 development team/mechanics pairs in both seat orientations: 768 evaluation games total. A seat swap reverses teams while the learner retains the same team. The base-power opponent is not a damage calculator or the existing tactical heuristic.

Evaluation takes no gradient steps and produces no learner rows. Initial/final endpoints use the same teams, mechanics seeds and action-randomness keys; all seeds and cells remain in the report. The same development pairs are shared across training seeds and opponents; seed agreement is not independent replication over teams. Training sees no evaluation trajectories. R7 splits exact ordered whole teams, not species or scenario families. The unchanged final reservation is not read or decoded by this workflow.

Natural terminal rewards are win 1/loss 0/true tie 0.5. Administrative cutoffs remain nonterminal and bootstrap training from the collecting policy. Evaluation assigns cutoff outcomes the interval [0,1], never a draw. The report gives paired worst-case improvement bounds and simultaneous 95% Hoeffding intervals over the six seed/opponent cells. The predefined gate requires a positive lower bound in every cell. With only 32 pairs per cell the simultaneous Hoeffding radius is about 58.5 percentage points, so these bounds are wide: an inconclusive gate is expected for modest improvements and does not prove absence of learning. Terminal-only value error is reported descriptively and is selected by each policy's terminal outcomes.

Every complete cohort atomically saves parameters, both Adam moments, optimizer step, next episode, and hashes of committed cohort evidence. Input snapshots bind executable Python/Node, contracts, tests, package locks, simulator build receipt and runtime; mutable documentation is excluded. The original run's frozen rules are not weakened retroactively. Any error retains command/previous-frame evidence, terminates the stage and leaves the last checkpoint intact. Explicit resume repeats only uncommitted work using identical seeds. It never replaces seeds, skips failed games or extends the budget. Live games are not serialized.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python -m pokemon_full_train.campaign --run artifacts/runs/f2-full-training-v2
# Only after an interruption has been diagnosed, on unchanged inputs:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python -m pokemon_full_train.campaign --run artifacts/runs/f2-full-training-v2 --resume
```

The first campaign deliberately measures the existing representation before changing it. Next decisions depend on actual terminal rates, learning curves, value behavior and sustained collection/update timing. Richer history, stronger frozen tactical opponents, historical-policy mixtures, curriculum ablations and parallel collection remain separate work. No public server, final evaluation or Prime is involved; full G2/G3/G4 and public ladder strength remain open. GitHub publication remains on the existing visibility hold.

## Preserved first attempt and narrow repair

The first F2 production attempt (v1) failed during seed1103 episode11, after eight training games were committed. The exact checkpoint replay reproduced missing own HP after an Illusion replacement: the adapter discarded an authoritative wait request. The F2-only repair retains own wait-request updates, preserves terminal public damage/faint updates and known own Tera reveals, and keeps all actions masked while waiting. Real fixtures check the old failure, correct own HP/roster, terminal behavior and opponent isolation. The same failed episode then completed naturally. The replacement initializes all three seeds afresh with identical budgets; v1 remains failed. Detailed evidence and independent review dispositions are in artifacts/reviews/f2-full-training/.

## Measured replacement result

The production replacement completed in **207.45 seconds** on local CPU. It collected **768 training games** (758 terminal, 10 cutoff), **89,657 training decisions**, and **798 optimizer steps**. The fixed evaluation collected **768 games** (758 terminal, 10 cutoff), **68,326 decisions**. Total: 1,536 games and 157,983 decisions; all three seeds and all six opponent cells are retained. No game is labeled a draw merely because it reached the administrative cutoff; there were no true ties.

Training collection plus update took 84.12 seconds, or **1,065.8 decisions/s**; this excludes admission, checkpoint serialization and evaluation. Whole-process throughput, including evaluation and overhead, was **761.5 decisions/s**. Peak RSS was **468.33 MiB**; this is a peak measurement, not a two-hour memory-stability qualification. The fixed 256-row update shape avoids per-cohort update-shape recompilation; no compilation-count telemetry is claimed.

Each endpoint below has 64 scheduled games. Wins are observed counts, not cutoff-imputed scores or confidence bounds.

| Seed | Random wins, initial → trained | Base-power wins, initial → trained | Training terminal / cutoff | Adam steps |
|---|---:|---:|---:|---:|
| 1103 | 35 → 42 | 4 → 7 | 255 / 1 | 274 |
| 2207 | 21 → 47 | 4 → 15 | 253 / 3 | 248 |
| 3301 | 27 → 41 | 2 → 9 | 250 / 6 | 276 |

The cutoff-aware observed improvement intervals (percentage points) are:

| Seed | Random | Base power | Preregistered gates |
|---|---:|---:|---|
| 1103 | +9.38 to +14.06 | +4.69 to +6.25 | Both inconclusive |
| 2207 | +37.50 to +42.19 | +17.19 to +17.19 | Both inconclusive |
| 3301 | +18.75 to +21.88 | +10.94 to +12.50 | Both inconclusive |

Those observed intervals account for unknown cutoff outcomes; they are **not confidence intervals**. All six simultaneous Hoeffding confidence intervals include zero. The same development teams/mechanics recur across training seeds and opponents, so agreement is not independent team-level replication. The policies still lose most games against the simple base-power opponent; these results support further learning experiments, not a ladder-strength claim.

The critic remains near 0.5. Initial terminal-only mean squared value error is 0.25. Final random-opponent MSE is 0.24636 / 0.24418 / 0.24629 for seeds 1103 / 2207 / 3301; against base power it worsens to 0.25358 / 0.25160 / 0.25362. These are game-averaged, terminal-only diagnostics under changed opponent distributions, not a calibrated win predictor or proof of universal value improvement.

All three final checkpoints round-trip. Independent verification reloaded each penultimate checkpoint and replayed its last eight original games (episodes 248–255): all 2,579 recorded decisions, values, actions and outcomes matched, and the original next updates reproduced final model parameters, both Adam moments, step counters and metrics exactly. These 24 games are verification replays, not added experimental samples. The smaller interrupted-run integration test separately verified exact-budget continuation and identical final state after a committed-batch interruption.

The [completion inventory](../artifacts/reports/f2-full-training.json) binds the report, frozen source copies, checkpoints and run evidence. It verifies counters, paired evaluation summaries and hashes; it does not claim independent replay of every production decision. The review directory preserves the preflight, failed v1, exact failure reproduction, repair reviews, dispositions, tests and recovery proof. All 409 predecessor non-Markdown live files and all 433 archived predecessor source files were unchanged at verification.

Next: retain this fixed baseline; improve the player-visible state/history and opponent evaluation panel, then preregister a larger-budget learning comparison with adequate evaluation precision. Test curriculum and opponent mixtures as explicit matched-budget changes. Keep final evaluation reserved. Prime remains unnecessary for this next local development stage.
