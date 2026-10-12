# M1 Showdown chance mapping and 10,000 fresh player-view games

The new `pokemon_showdown` package integrates the pinned Showdown finite-word
chance maps into the existing binary-policy collect/update loop. The fresh
10,000-game player-view campaign passes with zero disagreements in the fixed
public Rattata duel. **Full G2/G3 remain open. The selected policy/chance combination
still needs its own two-hour qualification. Local CPU is sufficient; Prime remains
unnecessary.** No strength evaluation or final reserved-data use occurred.

The [completion receipt](../artifacts/reports/m1-showdown-chance.json) binds source,
contracts, tests, captured reference data, campaign reports and review evidence.
The existing binary-policy receipt and all predecessor Python packages/contracts
remain unchanged. This is an additive version, not a rewrite of sealed evidence.

## What changed

The pinned Showdown commit is `942cda71d14bf0618bd57991708a2e2734b6daba`.
Its actual `PRNG.random(n)` computes `floor(w*n/2^32)` from a uint32 word.
For the supported divisors 24, 16 and 2, the intermediate product is exact in
JavaScript Number. The JAX implementation uses uint32 arithmetic with x64 disabled.

| Event | Exact mapping |
|---|---|
| Critical hit | `floor(w*24/2^32) == 0`, equivalently `w < 178956971` |
| Damage percentage | `100 - (w >> 28)`, descending from 100 through 85 |
| Two-action tie, input order `[first, 1-first]` | `first XOR (w >> 31)` |

For bucket24, split the word into 16-bit high and low limbs. Compute
`(hi*24 + ((lo*24)>>16)) >> 16`. The largest intermediate is 1,572,863, so the
uint32 calculation cannot overflow. The critical bucket contains exactly
178,956,971 of the 2^32 words. This replaces the earlier nominal 1/24 sampler;
the finite-word mass is `178956971 / 4294967296`, conditional on uniform words.
The damage mapping also replaces the earlier low-bit, ascending mapping.

The new event-addressed sampler uses the `SDW1` domain, game ID, turn and five
roles: critical hits by seat, damage rolls by seat, and a tie coin. The collector
first separates the environment and actor streams. The binary policy, its `DYB1`
sampler, behavior logs, PPO loss and Adam implementation are reused unchanged.
A distinct checkpoint format rejects old chance/policy formats and binds source,
contract, runtime, counters and completed-update state.

This does **not** port Showdown's sodium PRNG or reproduce its seed stream,
consumed-word schedule or whole trajectories from the same seed. The oracle
injects the same words into actual upstream PRNG methods and the new JAX mapping.
Uniformity and independence of pseudorandom words are assumptions of the stated
law, not mathematical facts established by the finite probe campaign.

## Reference capture and information boundary

The capture generates 10,000 new mechanics seeds and a new deterministic action
schedule. Both namespaces differ from the earlier 10,000 transition cases and
1,000 player-view cases. Admission checks their disjointness and pins the actual
upstream source/build manifest. One hundred compressed shards preserve all cases;
no reserved final team or episode is used for training or evaluation.

Raw-word instrumentation calls the original generator exactly once for each
recorded draw. It records role, actor or tie input order, raw word, result and
consumption index. Index gaps include untraced accuracy draws; only the stated
semantic chance roles are recorded. Instrumented and uninstrumented captures agree on the tested
passivity fixtures. The captured slice shuffles tied move actions twice before
execution. Both draws and their input orders are retained, and the final shuffle
must agree with the first executed move. The learner's single canonical tie coin
matches the conditional outcome law under independent uniform words; it does not
imitate that upstream draw schedule.

Each executed move's critical hit and damage roll are decoded again from its raw
words in JAX and supplied to conditional simulator parity. A canceled second move
has no executed damage event. Initial, pending-commitment and post-resolution
snapshots are reduced from each player's reference protocol. The pending snapshot
must equal the prior view: submitting the first action reveals neither that action
nor future chance outcomes to the player view.

The view check compares the existing normalized header/history representation and
its deterministic actor probe. It is not a new general learned-policy equivalence
claim. Separate integration tests check that the actual binary actor and new chance
sampler feed the complete collector and fused update. A three-seed comparison
against the sealed predecessor collector with only its sampler dependency replaced
also matches buffers and terminal states exactly. Both Pokémon sets and HP are
public in this tiny Custom Game profile. Hidden species, sets, items, abilities,
Tera and broader request/mechanics strata remain outside this qualification.

## Recorded results

Campaign: `artifacts/runs/m1-showdown-chance-v1`, hash
`fe2623e4e9ff5f0e4d6e7a83915e9c3cfddd9717225742f2892bc5e483667c93`.

| Check | Result |
|---|---:|
| Actual upstream word probes | 100,096 words: 96 boundary cases + 100,000 deterministic probes |
| Mapping comparisons | 500,480; zero disagreements |
| Fresh reference games | 10,000, in 100 source-bound shards |
| Resolved turns / executed moves | 31,371 / 55,062 |
| Player-view comparisons | 180,000, including terminal padding |
| Comparisons at live phases | 145,484, including 62,742 pending-player checks |
| Raw chance checks | 55,062 critical, 55,062 damage, 31,252 shuffle |
| Player/transition disagreements | 0 |
| Short learner integration | 3 seeds × 8 updates × 64 games = 1,536 games |
| Exact saved-boundary next-cycle comparisons | 3 |
| Source-bound regression | 229 tests: 65 Node + 164 Python |

Pending-player checks are included in the totals, not additional games. Terminal
padding contributes 34,516 comparisons. Saved-boundary replay executes an extra
cycle on both original and restored runners; those audit cycles are excluded from
the 1,536-game training count. Each compiled entry point traced once. The roughly
23-second campaign is a correctness check, not a throughput benchmark or soak.
The complete Node word oracle was separately regenerated with byte-identical output.
An independent inventory checks all 15,626 consecutive shuffle chains.

The [review dispositions](../artifacts/reviews/m1-showdown-chance/review-dispositions.md)
record independent review through the
[claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md).
Fable implemented the bounded integer mapping; Codex checked its proof, source
ownership, upstream fixtures and arbitrary-precision oracle tests before accepting
it for the admitted CPU scope. Opposite-model QA flagged hypothetical mixed-device
admission; GPU and mixed-device execution remain unsupported. This was a scoped
acceptance, not a claim that the reviewer approved accelerator support.

## Reproduction and next work

Use the pinned local dependencies already installed. The campaign intentionally
reads the sealed capture/word paths recorded in its source; preserve those original
artifacts. Capture commands require a fresh output path. The 10,000-game size and
100-game shards are fixed by the contract.

```sh
npm run showdown:words -- artifacts/runs/NEW-showdown-words.json
npm run showdown:reference -- artifacts/runs/NEW-showdown-player
# Recheck the source-pinned original capture and words:
npm run jax:showdown-chance -- artifacts/runs/NEW-showdown-chance
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m1-showdown-chance
.venv/bin/python scripts/seal-showdown-chance.py artifacts/runs/m1-showdown-chance-v1
```

Use the existing receipt for verification; do not rewrite sealed evidence after
completion. A new capture selection, source change or qualification campaign needs
new versioned evidence and a new receipt destination. The seal includes archived
copies; the source-bound regression wrapper is reused unchanged.

Next qualify this exact binary-policy/Showdown-chance combination for two hours on
CPU, with compile controls, finite metrics, memory telemetry and boundary recovery.
The old two-hour run certifies the old policy/chance contract only. Preregister a
restricted frozen-opponent pilot when the required fidelity scope is satisfied;
include random, damage-greedy and frozen-checkpoint opponents across training seeds.
Broader mechanics and hidden-information fixtures remain necessary for full G2,
and strength, calibration and stability evidence remain necessary for G3. Keep
final evaluation reserved. No Prime spending is justified by this milestone.
