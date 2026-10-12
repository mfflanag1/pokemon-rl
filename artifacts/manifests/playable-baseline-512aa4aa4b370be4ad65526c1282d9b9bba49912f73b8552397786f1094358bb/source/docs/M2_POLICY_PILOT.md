# M2 restricted-policy development pilot

Status: **all three training seeds passed the preregistered development learning
screen** on October 11, 2026. The campaign completed 54,000 real pinned Showdown
games, and every one of the 169,024 candidate decisions, log probabilities and
value predictions reproduced exactly from its saved player protocol. All 251 tests
pass (68 Node + 183 Python), bound to 218 unchanged tested source files.

The [completion receipt](../artifacts/reports/m2-policy-pilot.json) binds the run,
source, regression and review evidence. The unchanged binary-policy-1 /
showdown-chance-1 learner is evaluated only in the public fixed Rattata duel.
Full G2/G3 and general Pokémon strength remain uncertified.

## Measured development results

The fixed endpoint is update 2,048. Each table entry uses 1,000 complete seat pairs
(2,000 games); draw count is zero. The score lower bound and improvement interval
below use the preregistered six-interval adjustment, not the nominal diagnostic
intervals also present in the full report.

| Training seed | Initial score vs random | Update 256 | Final score vs random | Adjusted score lower bound | Paired gain, adjusted interval |
|---|---:|---:|---:|---:|---:|
| 101 | 50.10% | 62.55% | 63.65% | 60.80% | +13.55 pp [9.35, 17.45] |
| 211 | 49.95% | 62.75% | 62.35% | 59.55% | +12.40 pp [8.65, 16.35] |
| 307 | 49.55% | 62.35% | 62.30% | 59.45% | +12.75 pp [8.97, 16.45] |

Every final point score exceeds 60%, every adjusted lower score bound exceeds 55%,
and every adjusted improvement interval is above zero. There was no endpoint
selection: seed 211's earlier diagnostic score is higher, but the fixed final
checkpoint remains the evaluated endpoint.

| Training seed | Final vs damage-greedy, nominal 95% interval | Final vs frozen initial, nominal 95% interval | Final critic raw MSE vs random |
|---|---:|---:|---:|
| 101 | 49.45% [47.50, 51.45] | 61.60% [59.55, 63.50] | 0.247037 |
| 211 | 50.45% [48.35, 52.43] | 61.45% [59.32, 63.50] | 0.234324 |
| 307 | 49.55% [47.70, 51.40] | 63.70% [61.60, 65.70] | 0.230400 |

The learner improves against random and its frozen initialization, but does not
demonstrate an advantage over damage-greedy. These are results in the two-move
public duel, not evidence of hidden-information reasoning or full-format strength.
Critic calibration is measured, not certified: final-cell out-of-range prediction
rates range from zero to 0.032%, and full reliability bins/raw versus clipped errors
are retained. No calibration threshold was used to select a checkpoint or tune it.

Training comprised 6,144 updates, 393,216 games and 2,464,188 submitted decisions
across the three seeds. All nine checkpoints reproduce the original next learner
state and metrics exactly. The production run lasted from 14:43:47 to 14:50:14 UTC
(about 386.81 seconds including training, reference evaluation, complete tape audit
and admission checks). Recorded evaluation cells totaled 263.33 seconds. These
are local operational measurements, not a hardware comparison or full-format
throughput projection. No new memory-headroom certification is claimed.

Accepted run: `artifacts/runs/m2-policy-pilot-v1`.
[Report](../artifacts/runs/m2-policy-pilot-v1/report.json) content hash:
`d9438d93079f50266a01bcee4e2ac5f973d77319db4b1d019dfe97cc4ec781a5`.
All four smoke attempts, the failed regression attempt and review dispositions
remain preserved. All 128 final teams remain reserved and no final episodes were
used or generated. Prime remains unnecessary.

## Preregistered experiment

The [plan](../contracts/pilot-v1/plan.json) is written before production training
or evaluation. Admission copies it and the complete implementation manifest into
the run's `preregistration.json` before executing any update. The predecessor is
the [selected combination's two-hour qualification](M1_POLICY_SOAK.md), receipt
`90fd64d35aa7cee6ce6cb6ebb1e33fb0c7817ea8355fd1f336c36d4770cced88`.
The old learner, sampler, simulator, qualification and evidence remain unchanged.

- Fresh training seeds 101, 211 and 307; exactly 2,048 whole-episode updates per
  seed, 64 games per update. Checkpoints at 0, 256 and 2,048; update 2,048 is the
  fixed endpoint, with no checkpoint selection or hyperparameter tuning.
- All nine checkpoints play random legal, damage-greedy and that training seed's
  frozen initial policy in real pinned Showdown. Each cell contains 1,000 pairs,
  swapping the candidate between both seats: 27 cells and 54,000 games.
- Both moves have identical immediate damage in this admitted public set.
  Damage-greedy's declared tie-break favors higher priority, so it always uses
  Quick Attack. It is a deliberately stronger diagnostic than random legal play.
- Both seats in a pair share a Showdown seed. Checkpoints share pair IDs and
  policy random-key addresses. Divergent actions do not promise identical random
  events. Smoke and development use separate mechanics-seed namespaces.
- Average seats first, then bootstrap complete pairs with 4,096 replicates.
  For each training seed, endpoint score versus random must be at least 60%, its
  adjusted lower bound must exceed 55%, and the adjusted paired improvement bound
  over initialization must exceed zero. Bonferroni adjustment covers six intervals
  (three score and three improvement intervals). No pooling across training seeds.
- Other score intervals are nominal 95% diagnostics. Percentile-bootstrap bounds
  are approximate; the adjusted tail has about 17 bootstrap replicates. These are
  development screens, not guaranteed-coverage confirmatory intervals.

Every run records all cells, including a failed learning screen. An implementation
failure invalidates the run and leaves its evidence intact. A repair uses a new
directory and admission. No outcome-dependent continuation, sample-size increase
or selective cell rerun is part of this protocol.

## Actor and evidence boundaries

A local Node worker runs the exact public profile through pinned Showdown. It
exports the existing R1 player channels and accepts only current-request move
choices. Both views are taken before either action is submitted; the first pending
commitment must leave both projections unchanged. Unsupported phases fail.

Python reconstructs each actor's view using the sealed strict protocol reducer,
then calls the unchanged binary actor on 59 observation features and its legal
mask. Each row is independent. Seed/pair/seat metadata address separate evaluation
random streams; they do not enter policy or value features. Actor sampling uses
the existing exact dyadic word selector and matching PPO log probabilities.

Each compressed game record retains both final player protocols, the submitted
action tape, behavior log probability, candidate value and Quick Attack probability.
The audit reconstructs every original live request from its own protocol prefix,
recomputes every actor decision and value exactly, checks public executed moves
against the submitted tape, and verifies terminal winner, turn count and cell/pair
identity. All checkpoint boundaries reproduce the original next learner update.
The sealer recomputes score tables and calibration from the complete tape.

Exact replay is restricted to the admitted local CPU/runtime. The inherited
runtime/package pins and a hashed host-name/architecture label are recorded;
the label is not hardware attestation. Matching source is necessary, and the
actual exact replay check is decisive. No cross-host FP32 portability is promised.

Critic diagnostics compare each candidate decision's value with that game's own
terminal outcome. Raw MSE, clipped Brier, out-of-range rate and ten reliability
bins are reported separately. These repeat game outcomes across decisions and are
descriptive, without decision-level confidence intervals. Values were trained under
self-play; the frozen opponent panel changes that distribution. Neither clipped
Brier nor a passing random-opponent screen establishes a calibrated critic.

All 128 final teams remain reserved; no final episodes are read or generated.
The fixed public teams are the already-admitted restricted profile, not a sample
from the official full-format final partition. Prime remains unnecessary.

## Reproduction and completion

```sh
# Always choose a fresh output directory. Smoke does not count as the pilot.
PYTHONPATH=python .venv/bin/python -m pokemon_pilot.campaign artifacts/runs/NEW-smoke --smoke
.venv/bin/python scripts/seal-policy-pilot.py artifacts/runs/NEW-smoke --audit-smoke
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m2-policy-pilot
PYTHONPATH=python .venv/bin/python -m pokemon_pilot.campaign artifacts/runs/NEW-pilot
.venv/bin/python scripts/seal-policy-pilot.py artifacts/runs/NEW-pilot
```

Do not run the full regression alongside a Node reference campaign: older tests
temporarily mutate vendor sentinels. Freeze implementation, plan and sealer before
production. After the report passes, update current progress documents, seal once,
verify every live/archive hash and repeat sealing to require an identical receipt.
Historical handoffs and sealed predecessor evidence must remain intact.

The first full regression stalled in an unchanged Node/Python fixture while its
Python child waited for EOF. Its failed verification, log and native stack sample
are retained. A clean full rerun passed all 251 tests without a source change;
the underlying transient pipe cause remains unconfirmed.

Cross-model work uses the [claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md).
The initial statistics delegate was rejected after independent overflow probes;
its repair is accepted for the explicitly bounded pilot workload, with the original
QA resource-exhaustion caveat retained. The independent harness review found no
blocking issue. See [verified dispositions](../artifacts/reviews/m2-policy-pilot/review-dispositions.md).

Full G2/G3 remain open despite the passing screen: hidden-set and broader mechanics
coverage, stronger strategic evaluation, critic-calibration acceptance and appropriate
memory/budget evidence remain. Next define an explicit richer restricted profile
with meaningful move/switch choices and hidden-set strata, then establish fixtures,
information safety and fresh reference parity before training in that new scope.
