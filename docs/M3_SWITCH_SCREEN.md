# M3 restricted categorical learning screen

Status: **training and all 31,104 Showdown evaluation games are complete; the
full action/outcome audit is running**. All 476 source-bound tests pass on 347 files.
The [plan](../contracts/switch-screen-v1/README.md) was frozen before production
started at 2026-10-12 01:16:20 UTC (October 11, 21:16:20 Eastern). The run remains
`artifacts/runs/m3-switch-screen-v1`; the launcher PID is 47191, unified execution
session 12400, and logs are under `artifacts/reviews/m3-switch-screen`. No learning
pass/fail result is available until the complete audit finishes. The existing
[two-hour qualification](M3_SWITCH_SOAK.md) remains unchanged.

Measured training: 24,576 updates, 786,432 capped games and 18,615,320 decisions across
all three seeds, in 105.714630 seconds including nine exact checkpoint replays.
Reference evaluation took 1788.223304 seconds. Its full 31,104-game action audit and
subsequent fresh Showdown replay/sealing are materially slower than training.
A quiet background follow-up completes the remaining verification, updates the
five current progress documents, seals and checks all hashes, repeats sealing,
reports the actual result, then pauses. Historical handoffs and frozen code stay
unchanged. The two qualification heartbeats remain paused.


Three fresh seeds (1009/2027/3037) train 8,192 updates each with the sealed M3
categorical PPO learner. Checkpoints 0/2,048/8,192 face random legal, damage-greedy,
and frozen update-zero opponents. Each cell covers all 576 ordered team pairs in
both seats. The 27 cells contain 31,104 unpatched pinned Showdown games. The model
receives only its own protocol-derived compact view and legal mask.

Nonterminal twelve-turn cutoffs remain unresolved. Report wins/losses/cutoffs and
the outcome range [wins/games,(wins+cutoffs)/games]. The primary per-seed random
opponent gate requires at least 55% secured wins, an adjusted one-sided win lower
bound above 50%, and a positive adjusted worst-case improvement bound over the same
seed's initialization. The latter assigns all initial cutoffs the favorable outcome
and all final cutoffs the unfavorable outcome only to calculate a bound; it does
not label them losses, wins or draws. Complete seat pairs are the uncertainty unit.
The conservative Hoeffding bounds assume independent per-pair chance/policy streams.
All three seeds must pass; there is no seed pooling, selection or retuning.

The audited preflight uses a separate namespace and only 48 games. Its two original
next-update checkpoints replay exactly; every action/log likelihood/value is
recomputed from the stored receiving-player prefix. All 48 games replay from their
original seed on unpatched Showdown with identical views and complete protocols.
Nine copied-evidence negatives reject malformed scope, source/runtime, counters,
checkpoints, actions, outcome labels and view prefixes. The first 48-game smoke is
retained before the additional complete-reference replay/freeze checks; it is not
the final source-bound preflight.

Independent implementation and review used the
[claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md).
The bounded statistics delegate auto-routed to Codex, received Fable QA, then
passed local frozen tests and 6,561 exhaustive outcome-matrix comparisons. Two
Fable harness snapshots and coordinator dispositions are preserved in
`artifacts/reviews/m3-switch-screen`. Reviews are static; actual checks are recorded
separately. Historical handoffs and sealed sources/evidence remain unchanged.

## Execution and evidence

Use a new directory for every run; no outcome-dependent extensions or cell reruns.

```sh
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m3-switch-screen
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW --smoke
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW --audit-smoke
# After freezing admission in the review directory, run the development campaign:
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW-development
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW-development
```

Production must hash-bind a timestamped source/runtime admission freeze before any
training. This is local provenance, not trusted external timestamp attestation.
Checkpoint audits bind seed, learner/Adam state, counters, source/runtime and the
original next learner/buffer/final-environment/metric fingerprints. Only completed
update reset boundaries are supported. All original game tapes are re-audited and
fully re-executed through pinned Showdown before sealing. A completed experiment
can fail its learning gate without being an implementation failure.

Full G2/G3/G4, calibrated critics, long battles, broader mechanics and public ladder
strength remain open. No public server or final reserved-data evaluation is used.
This actor is not connected to the full-battle playable client. Prime is unnecessary.
