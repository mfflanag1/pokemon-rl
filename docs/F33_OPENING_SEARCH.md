# F33 — bounded opening lookahead

Status: completed; production and exhaustive fresh replay passed. Learning/search benefit is reported separately below.

F32 completed and was independently verified. Its three nonlinear final hosts won
87/256, 85/256 and 84/256 tactical games. All three beat the corresponding control,
but further-training improvement was inconclusive for seed 8209. F33 changes no
weights and retains all three hosts (7103,8209,9311).

## Scope and fixed budget

The accepted plan is `contracts/opening-search-v1/plan.json`. The comparison is
policy alone versus search at the first joint move request only, then the same
frozen policy for the rest of each complete battle. It contains 1,152 development
games:3 hosts × 3 familiar opponents × 32 paired schedules × 2 seats × 2 arms. It performs
zero optimizer updates. The independent fixed-position value audit uses 4 other
openings per host, every legal root action and 2 continuation-RNG replicates.
All data are retained; cutoff outcomes have unresolved [0,1] score bounds.

Search receives only its own current private request, its player projection and
public messages since battle start. It never receives the live opponent request,
team seeds, team keys, mechanics state or evaluation opponent identity. A separate
worker creates hypothetical battles. Own sets are fitted to private stats and the
entire reconstructed opening must match, including public entry effects. Consumed
items or unsupported forms therefore cause counted fallback, not invented state.

The hypothesis prior generates training-partition official teams and considers
their cyclic lead rotations, rechecking the actual rotated team's partition.
It accepts exactly 2 worlds within 512 team draws or
returns the base policy action. This rotated-team prior is deliberately approximate;
it is not the official conditional posterior or a calibrated belief distribution.
Generated Illusion worlds are admissible only if their actual projected opening
matches. No arbitrary mid-battle state reconstruction is attempted.

Every legal root action is simulated in both worlds. The frozen opponent model
chooses once per world before candidate actions are submitted. Intermediate pivot,
replacement and re-request phases continue under frozen policies, up to 16 requests,
until a fresh joint move boundary at turn 2 or later, or a natural terminal. The
backup is the exact terminal reward or the frozen own-side value. Root actions are
sampled from softmax(log frozen prior + mean backup/0.1). Paired initial chance
seeds do not imply aligned random events across different actions.

## Qualification and interpretation

Engineering admission checks public-input closure, changed-secret invariance,
exact opening reconstruction, unsupported entry effects, legal simultaneous
submission, pivot resolution, fixed-opponent toy backups and mixed-root sampling.
It measures 24 engineering roots (8 per host), requires at least 25% successful searches
and maximum measured root latency <= 5 seconds, and replays six complete engineering
games. These are research feasibility thresholds, not deployment guarantees.

The value audit uses the real local simulator solely to obtain offline outcome
labels. On fixed successor observations it measures terminal-only critic error
and within-opening action ordering. Two continuation replicates share initial
mechanics and are very noisy; no calibration or critic-failure claim follows.
Audit outcomes do not tune search. The comparison reports all 9 dependent paired
contrasts, all three tactical benefit gates, regressions, search coverage and
latency. It is underpowered for small gains from changing only an opening action.
Failure or inconclusive benefit does not authorize extension or another run.

Exhaustive fresh replay must reproduce all actual game decisions, projected frame
hashes, search worlds, action distributions and values. Runtime fields are excluded
from replay equality. Sources/configuration are frozen; final receipt and evidence
hashes are independently checked. Existing F32 and other predecessor files remain
unchanged. G2/G3/G4/G5 and public strength remain open. No promotion, public server,
reserved final evaluation, Prime or GitHub publication is authorized here.

## Review history

The delegated design and opposite-model QA identified opening regeneration as the
defensible first scope, and rejected general mid-battle reconstruction. The initial
design's large hypothesis bank, much larger game budget, argmax default and cutoff
target claim were not adopted. Existing training bootstraps administrative cutoffs;
zero scalar cutoff fields alone do not establish a zero critic target. Complete
projected-opening equality and explicit fallback address entry-effect omissions.

During unsealed engineering, a protocol-parser error treated the start marker as
`|start|` rather than upstream `|start`. It was corrected before qualification.
Two- and four-world feasibility probes at the same engineering opening showed
that four matches exhausted the proposed budget. The fixed two-world pilot plan
was selected before any scientific outcomes were collected. The successful
two-world probe simulated 26 branches in 0.249 seconds; it is not a strength result.

Prelaunch qualification v1 was invalidated after a separate audit found that
the ordered-team partition had been checked before lead rotation only. Of its
34 accepted hypothetical opponents, 2 landed in the final partition and 1 in development
after rotation. No optimizer updates or scientific comparison ran. The original
evidence is preserved. `reserved-exposure.json` records the accepted exposures;
`reserved-exposure-inventory.json` conservatively inventories potential rejected
candidates too. Those exact identities must be excluded from future held-out claims.
The repaired sampler checks the actual rotated team before any hypothetical battle
starts. Qualification v2 and independent repair review are required before launch.
Thus this workflow must not claim that all of its earlier engineering work kept
the final partition untouched, even if the repaired production run is clean.

## Measured result

Production took 469.42 CPU-backed wall seconds; 1,152 main evaluation games and 312 fixed-position audit rollouts; zero optimizer updates. Peak parent RSS was 594.6 MiB (excludes workers).

| Seed | Tactical policy wins | Search wins | Search applied | Tactical benefit gate |
|---|---:|---:|---:|---|
| 7103 | 23/64 | 20/64 | 135/192 | False |
| 8209 | 26/64 | 28/64 | 132/192 | False |
| 9311 | 27/64 | 28/64 | 144/192 | False |

All-seed benefit gate: **False**. All nine opponent contrasts, unresolved cutoff bounds, terminal-only values, fixed-position rankings, fallback reasons and latency measurements are retained in the production report. These opening-only development results do not establish general search strength or public ladder strength.

Receipt: `artifacts/reports/f33-opening-search.json`.
