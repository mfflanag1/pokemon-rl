# R3 handoff: official whole-team corpus and inventory

**R3 is complete.** The final collection contains **100,000 official teams / 600,000
Pokémon**, independently validated in Python, with **1,000 exact seed replays** and
zero generation failures. Every team preserves all six ordered slots and all
thirteen original generated fields. No independently sampled marginal sets were
assembled into teams.

## Data identity and locations

* Format: `gen9randombattle`, PotD disabled, Showdown commit
  `942cda71d14bf0618bd57991708a2e2734b6daba`.
* Corpus ID: `2266d7b8b9ea2a9264c0e283d3b3501daed67f1d984973b0f14971663d14e81a`.
* Seed namespace: `r3-inventory-2026-10-10-v1`; indices 0–99,999 are contiguous.
* Team schema: `1.0.0`; vocabulary: `showdown-dex-ids-v2`; split: `unassigned`.
* [Compact verification report](../artifacts/reports/r3-corpus.json).
* [Corpus index](../artifacts/runs/r3-gen9-random-100k-v2/index.json) and 100 gzip
  shards in `artifacts/runs/r3-gen9-random-100k-v2/`, 1,000 teams per shard.
* [Inventory report](../artifacts/runs/r3-gen9-random-100k-v2/inventory-final/report.json),
  [all frequencies](../artifacts/runs/r3-gen9-random-100k-v2/inventory-final/frequencies.json),
  [coverage](../artifacts/runs/r3-gen9-random-100k-v2/inventory-final/coverage.json),
  joint-set and team-count JSONL exports, and `counts.sqlite` in `inventory-final/`.
* [Exact replay evidence](../artifacts/runs/r3-gen9-random-100k-v2/replay.json).

Raw shards total **55,379,296 compressed bytes** (52.8 MiB), or **376,928,145
canonical JSONL bytes** before compression. Large artifacts are local and ignored
by Git; retain this directory for downstream work. Source/catalog/vocabulary/index
and verification metadata are also preserved under `artifacts/manifests/`, at the
paths recorded in the compact report. Those metadata cannot replace the actual
team files.

The final generation pass took about 125 seconds on this local machine. Analysis
timing and runtime fingerprints are recorded in its report. These are reference
data-pipeline measurements, not simulation or training throughput benchmarks.

## Measured inventory

| Quantity | Observed |
|---|---:|
| Complete teams, unique by exact generated content | 100,000 |
| Pokémon slots | 600,000 |
| Move slots | 2,395,662 |
| Original generator source entries | 509 |
| Base species | 421 |
| Emitted initial forms / display forms | 561 / 561 |
| Moves | 348 |
| Abilities | 203 |
| Held-item values, including the empty-item sentinel | 60 |
| Tera types | 19 |
| Distinct levels | 33, spanning 64–100 |
| Generator roles | 10 |
| Joint configurations | 9,589 |
| Unordered base-species pairs | 87,972 |

Source entry, emitted form and base species are different identities. The joint
configuration includes emitted form, ability, item, sorted moves, Tera, level,
gender and exact EVs/IVs; it ignores move order, nickname, shininess and role.
The separate ordered-team configuration retains team and move order. Both exact
generated-team and ordered-configuration keys had zero duplicate records.

All integer frequencies are retained, including source-conditioned alternatives,
lead frequencies, gender, shininess, levels, forms and team pairs. The 1,500,000
pair observations equal 15 per complete team. Move totals use the actual number of
slots, because some generated Pokémon have fewer than four moves.
There are 1,446 one-move sets and 598,554 four-move sets; 1,515 Pokémon hold no item.

Examples useful for prioritizing later mechanics fixtures:

| Domain | Most frequent values, with Pokémon counts |
|---|---|
| Moves | Knock Off 101,949; Earthquake 97,545; Close Combat 66,788 |
| Items | Leftovers 158,983; Heavy-Duty Boots 110,620; Life Orb 89,052 |
| Abilities | Levitate 26,723; Intimidate 20,692; Pressure 14,655 |
| Tera | Steel 90,347; Fighting 69,304; Water 62,677 |

These are observed counts across 600,000 Pokémon, not independently sampled
probabilities. Whole-team constraints and generator weights affect the distribution.

## Declarative coverage

| Source-conditioned denominator | Observed / declared |
|---|---:|
| Generator entries | 509 / 509 |
| Entry–move alternatives | 3,330 / 3,330 |
| Entry–ability alternatives | 583 / 583 |
| Entry–Tera alternatives | 1,285 / 1,285 |
| Entry–role alternatives | 873 / 873 |
| Templates with a compatible emitted set | 879 / 879 |

Every emitted set matched exactly one template under the recorded role, moves,
ability and Tera criteria in this sample. All 879 templates had a unique match;
there were no unmatched sets or observed alternatives outside the declared unions.
The implementation still records zero/multiple matches explicitly, because
compatibility is not an instrumented trace of template selection. Repeated role
names are never used as template identities.

Items, programmatic forms and EV/IV adjustments have no exhaustive declarative
denominator here. Full declared initial-set coverage does **not** establish full
battle-mechanics coverage: transformations, move calling/copying and subsequent
effects require R6's transitive inventory and targeted fixtures.

## Integration notes

The pilot uncovered a real R2 vocabulary omission: cosmetic forms such as
`florgesblue`. Vocabulary v2 adds 68 species IDs and preserves v1 in
`contracts/battle-v1/vocabulary.v1.json`. Numeric indices can shift; model tensors
and checkpoints must bind the exact vocabulary/hash and migrate by string ID.
The team envelope stays schema 1.0.0. Golden fixtures and the R2 smoke report were
refreshed against v2.

Generation supports checkpointed resume, rejects provenance/configuration changes,
checks committed shards, preserves orphaned shards and never skips a failed seed.
Python validates all schema/hash/ID/seed/accounting invariants and aggregates with
bounded in-flight shards and SQLite. Replay independently regenerates selected
teams through Showdown, covering all shard starts and the final row. The completed
run passed a second same-command integrity check. The full regression suite passed
40 Node and 13 Python tests. See [the run guide](R3_CORPUS.md) and
[checked review findings](R3_REVIEW.md) for exact commands and limits.

## Next: R4

Implement request-aware episode phases and per-player transition linkage on the
existing oracle: normal joint choices, pivots, single/double replacement, hidden
legality feedback and re-requests, and terminal versus artificial truncation.
Preserve pending opponent commitments and reject stale or visibly invalid actions.
The R4 acceptance campaign is at least 1,000 seeded local episodes with exact
normalized replay, plus fixtures for those exceptional phases.

Use this exploratory corpus for development and fixtures. Do not designate its
already-inspected teams as an untouched final benchmark; R7 must reserve independent
evaluation data and enforce content-based partitions. R5 still owns broader actor
information isolation. Policy training and accelerated-simulator fidelity remain
future milestones. R3 required local CPU work; Prime infrastructure is unnecessary
for this step.

R4 execution update: the [1,000-episode campaign is complete](R4_CAMPAIGN.md). The R4 next-step text above records this document’s R3 handoff; the current next ticket is R5.
