# Frozen Showdown opponent panel

This milestone measures the playable baseline against two additional deterministic
heuristics and the existing random control. All battles run in the pinned official
Showdown engine with complete generated Gen 9 Random Battle teams. The panel is a
development benchmark, not a full-format learner, public-ladder test or promotion.
The recoverable network client retains its sealed baseline policy.

## Frozen design

The plan in `packages/showdown-panel/plan.json` declares 64 team/seed pairs for each
of the six unordered matchups, with both policy-seat assignments: 768 games total.
The same 128 distinct development-partition teams and 64 mechanics seeds are reused
across edges. Teams remain in their seats while policies swap. Policy random streams
are keyed by policy ID and pair; team generation, mechanics and policy streams are
separate. Common mechanics seeds do not imply the same chance events after policies
choose different actions.

Official whole teams are generated and admitted by the sealed development filter
before play. No sampled final-partition teams enter the benchmark, actors or saved
team pool. The predecessor admission also verifies the reserved-data binding.
Source/runtime admission, the exact plan, assumptions and team pool are saved before
the first game. Source is frozen throughout the production run and replay. A failure
or decision-cap exhaustion invalidates completion; it is never replaced or scored as
a loss. The 24-game smoke namespace is separate and supplies no strength claim.

| Policy | Behavior |
|---|---|
| `baseline-v1` | Unchanged sealed playable heuristic and public reducer. |
| `random-v1` | Unchanged sealed random control. |
| `damage-v1` | Approximate damage ranking from own request and revealed state; forced-switch matchup ranking. |
| `tactical-v1` | Same damage proxy plus healing, setup, status, early hazards and selective voluntary switching. |

New policies use exact own request data and only player-visible opponent events.
Unrevealed moves, items, abilities and reserve identities are not simulator lookups.
Unknown opponent stats use explicit neutral-nature, 31-IV, 85-EV-per-stat species
proxies. Unknown incoming attacks use an 80-power STAB threat proxy; these are
assumptions, not recovered hidden sets. Public moves/items/abilities, boosts, weather,
hazards, Tera and statuses feed the richer reducer. Hidden reserves are not predicted.

Damage ranking is deliberately approximate: dynamic move callbacks, many ability/
item interactions, volatiles, special fields, defensive Tera, transformed stats,
Baton Pass and rare public-history interactions are not complete. Copied abilities
such as Trace can remain over-attributed in memory after switching. The engine itself
still resolves real rules. Approximate policy preferences do not expand the custom
simulator's supported scope or certify a complete public-state model.

## Statistics and audit

Scores are 1/0.5/0 for win/draw/loss, always from the first named policy's perspective,
including the reversed-seat game. Each mirrored pair is one cluster. The report
includes every game, all pair scores and 10,000 deterministic bootstrap resamples
of whole pairs. Each replicate uses the same resampled pair indices across edges.
For B replicates, the interval uses the ceil(B/40)-th smallest and largest values.

Intervals are descriptive, per edge and unadjusted for multiple comparisons. They
do not authorize promotion or a claim about Elo, the live ladder, other generators,
or training-seed reliability. A degenerate interval reflects identical sampled pair
scores, not certainty about the underlying strength. No stopping or tuning occurs
after production outcomes. Future improvements require a new version and fresh
development namespace.

Each game retains its starting teams/seeds, input tape, spectator events and every
player's decision input/action. Actors receive only their own projected channel,
never the spectator stream, opposite private request or terminal team payload. The
auditor regenerates the admitted teams, constructs all expected game specifications
from the plan and reruns every game in a fresh simulator. It compares inputs,
actions and outcomes exactly, excluding only measured latency; then recomputes
score tables, counters and paired intervals. Latencies are validated as finite and
nonnegative, bound to original records, and recounted rather than re-created.

Tests also require exact unchanged-baseline driver parity, actual hidden-reserve
input/action invariance, legal choices and deliberate corruption rejection for
player requests, seeds, policy seats, outcomes/summary scores, intervals and teams.
The source-bound full regression and predecessor hashes are required by the sealer.

## Reproduce

Use new output directories; existing evidence is write-once:

```sh
npm run bot:panel -- artifacts/runs/NEW-panel-smoke --smoke
npm run bot:panel-audit -- artifacts/runs/NEW-panel-smoke
npm run bot:panel -- artifacts/runs/NEW-panel
npm run bot:panel-audit -- artifacts/runs/NEW-panel
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/showdown-panel
.venv/bin/python scripts/seal-showdown-panel.py artifacts/runs/NEW-panel
```

Only the frozen production plan may be sealed. Documentation must record actual
results before the receipt is written. The canonical receipt is
`artifacts/reports/showdown-panel.json`; a new policy/plan needs a new versioned
campaign and sealer rather than overwriting it.

## Measured result — 2026-10-11

The accepted run `artifacts/runs/showdown-panel-v1` completes **768/768 games**
with **128 distinct admitted development teams**. Fresh replay reproduces all
**35,711 actor inputs/actions**, terminal results, pairing and score intervals.
There are zero unavailable-choice retries, failures or cap-truncated games.
The original plan and all 266 tested source/configuration files remain unchanged.

| First policy | Opponent | W–D–L | First-policy score | Descriptive 95% pair interval |
|---|---|---:|---:|---:|
| baseline-v1 | random-v1 | 118–0–10 | 92.19% | 87.50–96.09% |
| baseline-v1 | damage-v1 | 59–0–69 | 46.09% | 39.84–52.34% |
| baseline-v1 | tactical-v1 | 53–0–75 | 41.41% | 34.38–47.66% |
| random-v1 | damage-v1 | 10–0–118 | 7.81% | 3.91–12.50% |
| random-v1 | tactical-v1 | 13–0–115 | 10.16% | 5.47–15.62% |
| damage-v1 | tactical-v1 | 55–0–73 | 42.97% | 36.72–49.22% |

Every row has 128 games in 64 mirrored pairs. Tactical's direct score is therefore
**58.59% against baseline** (descriptive interval **52.34–65.62%**) and **57.03%
against damage** (50.78–63.28%). Damage scores 53.91% against baseline, with an
interval spanning 50%. These are development observations, not adjusted promotion
claims. Tactical scores 89.84% against random versus baseline's 92.19%, a 2.34-point
lower point estimate. No universal dominance or no-regression claim follows.

| Policy | Decisions | Status moves | Voluntary switches | Tera | p99 decision time |
|---|---:|---:|---:|---:|---:|
| baseline-v1 | 8,494 | 8 | 25 | 337 | 0.107 ms |
| random-v1 | 9,822 | 2,321 | 265 | 0 | 0.047 ms |
| damage-v1 | 8,513 | 37 | 0 | 300 | 0.156 ms |
| tactical-v1 | 8,882 | 812 | 110 | 295 | 0.170 ms |

Each policy appears in 384 games. Status use and switching differ markedly, but
several heuristic changes are bundled; these counts do not identify the cause of
the score differences. Latency includes cloning the already-built player inputs
and policy execution, not network delay or complete request processing. Policy
order is fixed, so JIT/warm-up effects are not controlled for latency comparisons.

Fresh replay took **22.145 seconds** with exit code zero. The optional system timing
wrapper around collection printed 20.20 seconds but exited nonzero when sandbox
permissions denied its `kern.clockrate` probe after collection. The campaign itself
wrote the complete report and all 768 games; there is no failure marker, and every
game subsequently replayed. This instrumentation limitation is preserved in
`production-time.log`; it is not a valid peak-memory measurement or a failed game.

The full source-bound regression passes **338 tests: 149 Node + 189 Python**,
including 35 new tests. All 266 tested files are unchanged. Pinned build verification
and syntax checks pass. Four deliberately corrupted production-evidence copies
are rejected by sealing (failure marker, incomplete progress, changed score and
changed frozen source). The tests separately reject actor-input, seed, seat,
interval and team corruptions using fresh smoke games.

Two Fable snapshot reviews and opposite-model Codex QA of the delegated statistics
are preserved in `artifacts/reviews/showdown-panel/`. Confirmed public-memory,
stat-event, hazard and first-turn-move issues were fixed and tested before the
production run. The first statistics delivery was rejected for a failed test; a
fresh repair passed all 17 checks and QA. Review deltas explicitly distinguish
reviewed snapshots from locally verified repairs.

The [completion receipt](../artifacts/reports/showdown-panel.json) binds the frozen
source, predecessor, all games, fresh replay, regression, review and this handoff.

## Next step

Treat `tactical-v1` as a candidate for a controlled upgrade, not an automatic
replacement for the qualified network client. Freeze a fresh development
confirmation plan against all three opponents, with a declared regression tolerance
and sample budget, then qualify an additive network adapter with the same
player-view reconstruction and disconnect tests. Keep the original panel frozen
for comparison. Any policy tuning gets a new version and independent seeds.

Continue the switching/hidden-set scalar/JAX port separately before expanded
learner training. This panel does not supply a full-format trained policy or close
G2/G3. Public account/registration details and live-server qualification remain
outstanding. No training, public-server play, final reserved-data evaluation, GPU
or Prime resources were used. Local CPU remains sufficient for the next step.
