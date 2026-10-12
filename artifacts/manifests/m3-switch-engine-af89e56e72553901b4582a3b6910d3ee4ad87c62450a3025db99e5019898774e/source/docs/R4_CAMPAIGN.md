# R4 handoff: request-aware episodes and reference campaign

**R4 is complete.** All **1,000 official episodes** completed, replayed exactly
from serialized input tapes and passed independent Python artifact validation.
The run contains **81,947 decisions and 81,947 same-player transitions** across
**36,074 turns**, with **zero collection failures or validation mismatches**.
Thirteen additional fixtures passed both replay and independent validation.

The [run guide and R5 handoff](R4_EPISODES.md) explain the API and lifecycle rules.
The [machine-readable report](../artifacts/reports/r4-reference.json) binds the
results to manifests, source hashes and verification evidence. The
[review record](R4_REVIEW.md) describes findings, fixes and checks actually run.

## What changed

`packages/episodes/` wraps the pinned R1 oracle with request groups, per-player
commitments, ordered submissions, idempotent retry receipts, versioned observations,
selected mechanics audits and exact replay. The additive
[episode-v1 contract](../contracts/episode-v1/README.md) preserves the R2 schema
and existing R1–R3 artifacts. Pure-Python validators reconstruct the private
protocol channels and verify the exported records without invoking Node.

Hidden-legality feedback is an ordinary recorded transition. A re-request updates
only the affected player, keeps the opponent's committed action, and receives zero
intermediate reward. Each decision points to that same player's next decision,
terminal result or nonterminal bootstrap view. Cuts finish the current request
chain and stop only at a fresh joint boundary. Visible-invalid/stale submissions
fail explicitly; transport retries create no new decisions.

## Measured coverage

These are counts of distinct nonterminal **request groups**, not episodes or
individual submissions. Natural examples use zero-based episode/frame indices.

| Natural group | Count | First example: episode / frame |
|---|---:|---|
| Joint move choices | 36,074 | 0 / 0 |
| Single faint replacement | 8,126 | 0 / 58 |
| Double faint replacement | 197 | 3 / 15 |
| Live switch continuation, including pivots | 1,200 | 0 / 110 |
| Revival target selection | 38 | 21 / 23 |

The 41 unavailable-choice rejections occurred in 31 episodes; **24 preserved an
already committed opponent choice**. There were 500 p1-first and 500 p2-first
submission schedules. Seat outcomes were 537 p1 wins and 463 p2 wins, with no ties.
These are collection diagnostics, not an agent-strength or seat-advantage estimate.
The largest episode contained 245 decisions.

The extra fixture set has two complete natural episodes, two natural budget-cut
episodes and nine explicitly constructed probes. It covers both seats for single
replacements, live switches and revival, plus double replacements, mixed requests
and a tie that cancels an unsubmitted request. Constructed probes use marked
simulator interventions and do **not** establish natural reachability or mid-turn
timing. Mixed requests and ties were not observed naturally in this campaign.
Natural cutoff tests additionally hit pivots, faint replacements and hidden
rejections; they continue to a fresh joint boundary before bootstrapping.

## Provenance and storage

- Showdown: `942cda71d14bf0618bd57991708a2e2734b6daba`.
- Format: pinned `gen9randombattle`, PotD disabled, official whole-team generation.
- Campaign namespace: `r4-reference-2026-10-10-v1`; 1,000 exact indices, no skipped games.
- Campaign ID: `ed2a282a5a409ee3f58779062392ed6a252a86c5d89aba30b010cead25b5797a`.
- R4 manifest ID: `45e7b82cd6aca36bc21ccee5331c8871d510eaf76f8e574292ea2e15d968a7f4`.
- Controller: pinned upstream random baseline, move probability 0.9, Tera disabled.
  Team generation, mechanics and the two action samplers use separate seed streams.
- Split: `unassigned`. These inspected data are exploratory development evidence.

Episode files live in
[`artifacts/runs/r4-reference-1000-v1/`](../artifacts/runs/r4-reference-1000-v1/),
with `index.json`, `manifest.json`, progress checkpoint and `validation/` reports.
Each `episode-NNNNNN.json.gz` holds one complete bundle, including raw evidence.
[Additional fixtures](../artifacts/runs/r4-fixtures-v1/) have separate official and
constructed manifests plus a validated index. Bulk `artifacts/runs/` data are
Git-ignored; preserve or transfer them separately when moving the project.

Compressed episode data total **71,977,245 bytes** (72.0 MB); uncompressed canonical
JSON totals **1,765,827,211 bytes** (1.766 GB). Mean compressed storage is about
72 KB per episode. These are diagnostic records, not compact training samples.

[The retained archive](../artifacts/manifests/r4-ed2a282a5a409ee3f58779062392ed6a252a86c5d89aba30b010cead25b5797a/) includes the complete campaign index, manifests,
per-episode validation results, fixture index, passing regression log, and a source
snapshot with checksums. The snapshot preserves project code, contracts and tests;
upstream Showdown and installed dependencies are rebuilt from their pins. Restore
the recorded runtime before requiring the original manifest identity. Do not
replace current working files merely to inspect an old snapshot.

## Verification and runtime

`npm test` passes **51 Node and 28 Python tests**. The 13 saved fixtures also pass
Node replay and Python validation. Every campaign worker validates and replays its
serialized episode before committing it. Python independently checks file and JSON
hashes, canonical bytes, schema, raw channel projection, request/view/mask linkage,
commitments, actions, rewards, seeds, schedule and summary totals. Python does not
provide a second mechanics simulator.

On this Apple M3 Max (14 logical CPUs, 96 GiB RAM, macOS/Darwin arm64), using eight
workers, generation **including all 1,000 exact replays** took **170.760 seconds**.
Independent validation took **351.303 seconds**. Node was v25.6.1; Python was 3.14.3.
These are observed local wall times: validation overlapped the final regression
checks. They are not isolated benchmarks or RL throughput estimates.

## Limits and next step

The full bundle is privileged: initial teams, RNG, opponent commitments, exact HP,
raw end messages and cross-player phase labels must never enter actor features.
The selected mechanics audit is not a complete state or arbitrary restart format.
Reference replay demonstrates reproducibility of this pinned oracle, not fidelity
of an accelerated simulator or exhaustive mechanics coverage. The random baseline
never activates Tera, so this campaign supplies no natural Tera activation coverage.
No learned policy, PPO training or search was added.

**Next is R5:** typed per-player history loading, explicit known/unknown features,
semantic/tensor action round trips, and paired hidden-state perturbation tests for
both policy and value inputs. Follow the concrete implementation order in the
[R5 handoff](R4_EPISODES.md#r5-handoff). R6 owns systematic mechanics coverage;
R7 owns split/leakage gates and sealed evaluation data. R4 needed only local CPU
work. Prime infrastructure or rented GPUs would add no necessary capability here.
