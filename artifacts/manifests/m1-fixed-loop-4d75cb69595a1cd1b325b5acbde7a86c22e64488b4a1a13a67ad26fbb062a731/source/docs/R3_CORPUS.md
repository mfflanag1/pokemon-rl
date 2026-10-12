# R3: official whole-team corpus and inventory

This pipeline generates complete teams through the pinned official Showdown
`Teams.generate('gen9randombattle', {seed})` entry point. It preserves all thirteen
generated fields per Pokémon and all six team slots, then wraps the result in the
R2 team contract. It never independently samples Pokémon or reconstructs sets.

The completed 100,000-team run, coverage figures and next-step handoff are in
[R3_INVENTORY.md](R3_INVENTORY.md); checked review findings are in
[R3_REVIEW.md](R3_REVIEW.md).

## Run and resume

Install the local dependencies in [R2_CONTRACT.md](R2_CONTRACT.md), then run:

```sh
npm run corpus -- generate --out artifacts/runs/my-corpus --namespace my-corpus-v1 --count 100000 --shard-size 1000
PYTHONPATH=python .venv/bin/python -m pokemon_reference.corpus artifacts/runs/my-corpus --out artifacts/runs/my-corpus/inventory --workers 4
npm run corpus -- replay --out artifacts/runs/my-corpus --samples 1000
```

Repeat the same generation command to resume or verify a completed run. Count,
shard size, namespace, producer hashes, vocabulary and reference manifest must all
match. A different configuration needs a new directory. A live writer lock prevents
concurrent writes; only a recorded local PID confirmed dead can be recovered.
Recovery itself has an exclusive `.writer-recovery.json` guard. If that guard
survives a crash, further stale-lock recovery stops conservatively (a fresh
exclusive acquisition can still succeed when no writer lock exists). Inspect the
guard's host/PID and remove it manually only after establishing that no writer or
recovery process is active. Corrupt locks and PID reuse also require operator
inspection; they are never guessed safe to delete.
Do not run the R1 integrity tests concurrently with an oracle job: they temporarily
create deliberate source/build corruption sentinels.

The seed at zero-based row `i` is `sodium,` followed by the canonical SHA-256 of
`["r3-team-seeds-v1", namespace, i]`. It is independent of shard completion timing
and never resampled after a failure. A shard enters `progress.json` only after its
complete compressed file is published. Uncommitted rows are regenerated from the
same indices. A completed file left behind before its checkpoint is preserved in
`orphans/` and regenerated. Failures record the phase and committed record count
in `failures.jsonl`; the attempted index and seed are present only for per-team
generation failures and null for shard/checkpoint errors. There is no
skip-and-continue path.

The final `index.json` is published only when every requested row has a committed
shard. Each shard has compressed and uncompressed SHA-256 checksums, byte counts,
start index and row count. Resume checks all committed checksums. This provides
process-interruption recovery; it is not a promise of durability against storage
hardware failure or power loss during filesystem writes.
Compressed bytes can vary with platform/zlib; their checksums protect stored files.
Use canonical uncompressed JSONL hashes for content comparison. Cross-platform
generator reproducibility has not been established.

Python refuses incomplete corpora. It independently checks all artifact schemas,
hashes, canonical bytes, row seeds, manifest references, resolved identifiers,
vocabulary membership and six-member species uniqueness. Validation is parallel
with bounded in-flight shards. SQLite aggregates joint sets and team identities
without retaining the whole corpus in memory. Each analysis uses a fresh output
directory; failed analyses get `failed.json`, never a successful report.

Replay chooses deterministic indices without replacement, including endpoints and
shard starts when the sample budget allows. Each selected team is regenerated
through Showdown and compared against the complete stored artifact, including its
seed, original generator fields and canonical identifiers. Replay also checks all
shard checksums. Full Python validation and sampled official regeneration are
different checks; passing either does not substitute for the other.
Python validates internal consistency against the exported pinned catalog and
vocabulary. The generator emits moves as IDs already, so move resolution is
identity plus membership/uniqueness checks. Rewriting both raw and canonical fields
consistently and recomputing every hash can produce a structurally valid but
non-generated team; validation alone does not authenticate generation. Exact
sampled replay checks generation provenance for the selected rows.

## Files

The corpus directory contains:

* `manifest.json`: the pinned R2 source/build/rule/configuration manifest.
* `catalog.json`: all 509 source entries, their 879 declarative templates, source
  hashes and pinned Dex name resolvers. Templates retain source array index and
  canonical raw-template hash, even when role names repeat.
* `vocabulary.json`: the exact token table bound to this collection.
* `teams-000000.jsonl.gz`, …: ordered, complete R2 team artifacts.
* `progress.json`, `index.json`, optional `failures.jsonl`: collection accounting.
* `replay.json`: sampled regeneration indices and verified artifact/team hashes.

The inventory directory contains `report.json`, `frequencies.json`,
`coverage.json`, `joint-sets.jsonl.gz`, `team-counts.jsonl.gz`, and `counts.sqlite`.
The report binds each derived file by hash and records Python dependencies,
consumer source hashes, validation counts, timing, and worker count.

Large local corpora and SQLite files remain under ignored `artifacts/runs/`.
Keep that directory or copy it to durable storage before removing the checkout.
Compact handoff reports under `artifacts/reports/` do not contain the actual teams.

## Inventory definitions

`pokemon.speciesId` is the **source entry** selected from `sets.json`.
`ids.species` resolves the **emitted initial form**, which can differ because of
battle-only pre-battle forms, cosmetic selection and generator-specific form
consolidation. `ids.base_species` groups those emitted forms by base species.
All three are counted separately, along with raw display names, lead source IDs,
moves, abilities, items, Tera types, levels, roles, gender and shininess. Empty held
items count as `<none>`, independently of unknown vocabulary tokens.

Exact frequencies use integer counts. Pokémon-level marginals have denominator
`6 × team_count`; move counts use their separately summed number of move slots;
lead counts use `team_count`. Unordered base-species pairs count once per team,
with total denominator `15 × team_count`. The sampler is constrained and weighted,
so a uniform distribution over source entries is not the expected baseline.

A joint-set key hashes emitted form, ability, item, sorted move IDs, Tera type,
level, gender and exact EVs/IVs. It ignores move order, nickname, shiny appearance
and generator role. Team counts separately retain the exact generator output hash
and an ordered configuration key that preserves team order and move order while
omitting nickname, shiny appearance and role. These are inventory definitions,
not the final R7 split-equivalence policy.

Coverage compares source-conditioned observed moves, abilities, Tera types and
roles against the union of declared alternatives. Template coverage reports both
compatibility and unique matches using emitted role/moves/ability/Tera. An emitted
set may match multiple overlapping templates; compatibility is not a trace of which
template the generator selected. No claimed template frequency silently assigns an
ambiguous set to one template.

Items, emitted display forms, stat adjustments and other programmatic choices have
no exhaustive declarative denominator here. Unobserved declared alternatives are
gaps for investigation, not proof of unreachability or data loss. An initial-team
corpus does not cover dynamic moves, transformations or other battle-time effects;
R6 owns that broader mechanics inventory.

## Vocabulary revision found during R3

The independent pilot caught `florgesblue` missing from R2's token vocabulary.
`Dex.species.all()` does not enumerate every cosmetic species that
`Dex.species.get()` can synthesize. Vocabulary revision `showdown-dex-ids-v2` adds
all declared cosmetic forms: **68 additional species IDs**. The original table is
preserved at `contracts/battle-v1/vocabulary.v1.json`.

Species token indices can change between these revisions. Always bind model/data
tokens to the exact vocabulary version and hash; do not reuse v1 numeric tensors
with v2 without remapping by string ID. Team artifact shape remains schema 1.0.0;
its species IDs are strings, and the manifest records the changed table hash.

## Boundary for subsequent training

This is an exploratory inventory corpus, marked `split: unassigned`. It is not a
sealed benchmark or evidence of trained playing strength. Do not later promote
these inspected teams into an untouched final evaluation set. R7 must establish
the split policy and reserve independent final-evaluation data before pairing.
The next implementation ticket is R4: request-aware episode phases, transition
linkage and the larger reference battle fixture suite.

R4 execution update: the [1,000-episode campaign is complete](R4_CAMPAIGN.md). The R4 next-step text above records this document’s R3 handoff; the current next ticket is R5.
