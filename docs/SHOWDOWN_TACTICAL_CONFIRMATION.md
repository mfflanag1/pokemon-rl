# Tactical policy confirmation and local network qualification

This additive milestone tests the already-frozen `tactical-v1` heuristic selected
from the [opponent panel](SHOWDOWN_OPPONENT_PANEL.md). It does not change the four
policy definitions, the earlier recoverable client or any sealed predecessor.
Statistical confirmation and network correctness are separate requirements.

## Predeclared development screen

`packages/showdown-confirmation/plan.json` fixes 1,000 fresh team/seed pairs for all
six unordered matchups: 12,000 games, each team pair played in both policy-seat
assignments. All 2,000 distinct teams must pass the official generator's sealed R7
development filter and have no exact-team overlap with the selected panel, its
smoke pool, or this milestone’s smoke/network pools. A collision fails admission; no replacement is permitted. Mechanics,
policy and team seeds use the new `showdown-tactical-confirmation-v1` namespace.
This is fresh development evidence, not use of the reserved final teams.

The same pair indices are used across opponents. Three comparisons are fixed:

1. Tactical's direct score against baseline: point estimate at least 52%, with a
   one-sided lower bound strictly above 50%.
2. Tactical minus baseline score against random: lower bound strictly above −3
   percentage points.
3. Tactical minus baseline score against damage: lower bound strictly above −3
   percentage points.

The three-point tolerance is a declared engineering choice for this development
screen. It is not a claim that every possible opponent regression is acceptable.
Differences are formed within each paired team/seed cluster before resampling.
30,000 bootstrap replicates use common sampled cluster indices for all three
metrics. Each lower bound is the ceil(30,000 × 0.05 / 3)-th smallest replicate,
the 500th order statistic. This uses nominal Bonferroni allocation for the three
comparisons; percentile bootstrap coverage is approximate, not a finite-sample
guarantee. Six ordinary two-sided edge intervals remain descriptive diagnostics.

Pass requires every declared condition. There is no optional stopping, threshold
relaxation, policy tuning, pair replacement or adaptive sample extension. A failed
screen is retained as a non-promotion result. Common seeds do not force identical
chance events after different actions. No Elo, ladder, full-format learner or
G2/G3 conclusion follows from a pass.

## Additive tactical client

`TacticalBot` wraps the sealed recovering session with an instance-local room
factory. It inserts the exact frozen tactical actor and richer public reducer
when a new battle initializes. The parent still owns authentication, request IDs,
legal choices, timers, accepted-choice suppression, error classification and the
three-reconnect budget. A new connection gets new tactical memory reconstructed
from its received player-visible history; stale requests and reducer state are
never transferred between sockets.

The wrapper depends on the pinned parent's synchronous connection construction
before its first await and its queued message handling. Targeted tests exercise
that boundary, initial tactical decisions, rich-history reconstruction, all earlier
disconnect classes, stale sockets, same-turn replacements and accepted actions.
It is not a generic policy-factory API for arbitrary client versions or stateful
learned actors.

The six real local-server fixtures cover disconnect before choice, deliberately
lost outgoing action, accepted pending action, later turn, forced replacement and
two successive losses. The saved-wire auditor rebuilds tactical memory and request
state per connection and recomputes every attempted action. It checks that accepted
actions are not resent and that the deliberately lost write is reconsidered from
the fresh server request. These games use separate development teams and qualify
functionality, regardless of whether the statistical screen passes.

## Commands

All output directories must be new. Keep the original baseline available with
`npm run bot:recover`. The explicit tactical client is:

```sh
npm run bot:tactical -- --name TacticalBot --accept YourName
```

Use the [local server guide](PLAYABLE_BASELINE.md). Public account details and
[deployment requirements](SHOWDOWN_DEPLOYMENT.md) remain unresolved; no public
server is contacted by this milestone. This CLI has no ladder search option.

```sh
npm run bot:confirm -- artifacts/runs/NEW-confirmation --smoke
npm run bot:confirm-audit -- artifacts/runs/NEW-confirmation
npm run bot:confirm -- artifacts/runs/NEW-production
npm run bot:confirm-audit -- artifacts/runs/NEW-production
npm run bot:tactical-network -- artifacts/runs/NEW-network 18020
npm run bot:tactical-audit -- artifacts/runs/NEW-network
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/showdown-confirmation
.venv/bin/python scripts/seal-showdown-confirmation.py artifacts/runs/NEW-production --network artifacts/runs/NEW-network
```

## Verified engineering result — 2026-10-11

The final source-bound regression passes **380 tests (191 Node + 189 Python)**,
including 42 new tests. All **283 tested source/configuration files** are unchanged.
Build verification and syntax checks pass. The 24-game smoke campaign replays all
1,198 actor inputs/actions; its statistical status is diagnostic only and supplies
no confirmation claim.

The final local run `artifacts/runs/showdown-tactical-network-v1` completes **six
games through seven injected disconnects**. All **179 attempted actions and player
inputs** pass the wire audit. This includes one deliberately lost outgoing write
and one accepted pending choice that is not replayed. Four mutated copies of the
production trace/report are rejected (actor memory, accepted-action resubmission,
decision count and policy identity). The source manifests for campaign, network
and regression match exactly.

Twenty-two adapter tests exercise the inherited recovery cases and tactical-specific
initial decisions, rich-history reconstruction, synchronous startup traffic and
baseline-fallback rejection. Four audit tests use a compressed, source-bound real
preflight trace; nine campaign tests cover exact replay and corrupted evidence;
seven statistics tests include a separately calculated literal three-metric fixture.

The first loopback preflight was blocked by sandbox bind permissions and remains
preserved. The authorized loopback-only retry passed. A Fable implementation
attempt timed out without a complete tested delivery; none of its partial module
was integrated, and the runner refused QA/adjudication on that incomplete run.
The short coordinator implementation received a separate focused Fable review and
actual tests. Another Fable snapshot reviewed the adapter and experiment pipeline.
Review deltas, checked findings and unavailable QA are recorded explicitly.

## Measured confirmation result

The fixed campaign `artifacts/runs/showdown-confirmation-v1` completes all
**12,000 games and 564,379 attempted decisions** in **321.364 seconds** measured
around the complete campaign subprocess. It admits 2,000 distinct development
teams, with zero exact-team overlaps against the declared prior/fixture pools.
There are no failed games or replaced pairs. Policies, thresholds and budgets
remain unchanged throughout production.

| Tactical opponent | Wins / games | Score |
|---|---:|---:|
| Original baseline | 1,213 / 2,000 | 60.65% |
| Damage heuristic | 1,104 / 2,000 | 55.20% |
| Random policy | 1,902 / 2,000 | 95.10% |

The predeclared confirmation decision is **pass**:

| Metric | Estimate | Nominal adjusted lower bound | Required |
|---|---:|---:|---|
| Tactical score against baseline | 60.65% | 58.55% | Point ≥52%; bound >50% |
| Tactical minus baseline against random | +2.20 pp | +0.80 pp | Bound >−3 pp |
| Tactical minus baseline against damage | +10.05 pp | +7.40 pp | Bound >−3 pp |

Baseline's corresponding scores against random and damage are 92.90% and 45.15%.
The earlier panel's apparent random-opponent regression does not recur here.
The lower bounds are the frozen approximate bootstrap screen described above;
they do not establish dominance over other opponents or measure human strength.

Tactical makes 140,772 decisions, including 14,824 status moves, 1,954 voluntary
switches and 4,436 Tera choices. Measured tactical decision computation has median
0.084 ms and p99 0.239 ms; these exclude network transport and do not certify a
public-server deadline. The report retains five baseline and eight random-policy
unavailable-choice retries, all explicitly rejected as trapped switches, and none
for damage or tactical. All attempts remain
in the evidence; they are not scored as failed games or silently discarded.

An independent Node process regenerates all 2,000 teams, checks their partitions,
and replays every game in **313.062 seconds**. All **564,379 actor inputs/actions**,
outcomes, pair/seat/seed assignments, counters, intervals and confirmation decisions
match exactly. Both the campaign and replay subprocesses exit zero. The final
campaign, network qualification and 380-test regression share the same 283-file
source manifest and admitted runtime; historical sealed sources remain intact.
Seven temporary evidence copies are rejected by the real sealer for their
intended failure-marker, collection/replay-progress, frozen-plan, game-digest,
confirmation-report and network-trace violations, without writing a receipt.

The [completion receipt](../artifacts/reports/showdown-confirmation.json) binds
the actual source, documentation, run evidence and review records. The separate
[integrity record](../artifacts/reviews/showdown-confirmation-integrity.json)
records every live/archive hash comparison and byte-identical repeat sealing.

The selected heuristic is available for local play via the explicit tactical CLI.
Next extend the real local-server soak toward the separate 1,000-game adapter
requirement, then continue the switching/hidden-set scalar/JAX port and categorical
learner. A public challenge still needs account details and applicable deployment
requirements. This milestone trains no model, uses no final evaluation data and
does not close full G2/G3/G4 or establish ladder/#1 readiness. Local CPU remains
sufficient; Prime is unnecessary.
