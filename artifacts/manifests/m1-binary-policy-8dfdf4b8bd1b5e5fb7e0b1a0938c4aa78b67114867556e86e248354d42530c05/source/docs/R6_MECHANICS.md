# R6: mechanics inventory and fixture handoff

R6 adds a source-bound conservative mechanics manifest, a fixture catalog and
an explicit custom-engine support gate. Local CPU is sufficient. This is
inventory and test design; the project still has no accelerated simulator or
trained policy.

## What is covered

The pinned Showdown commit remains `942cda71d14bf0618bd57991708a2e2734b6daba`,
with the same Gen9 Random Battle profile, disabled PotD and private configuration.
The active Dex mod is `base`, generation 9. Python consumes the Node inventory;
it does not interpret TypeScript.

| Domain | Observed initial IDs | Conservative inventory |
|---|---:|---:|
| Moves | 348 | 954 |
| Abilities | 203 | 321 |
| Items, including no held item | 60 | 584 |
| Forms, including cosmetic aliases | 561 | 1,586 |
| Global conditions | — | 35 |
| Core source modules | — | 29 |
| Total | 1,172 | 3,509 |

Every observed initial entry is checked against the sealed R3 frequency inventory.
It has a source file/hash and byte/line span, handler ID, implementation ticket,
fixture families and `unsupported` custom-engine status. The table is an ID count,
not a count of independent mechanics. Embedded conditions and callback paths are
indexed under their owning moves, abilities or items; they are additional
obligations, not absent simply because there are only 35 global conditions.

Source analysis records **10,710** calls, constructors and selected identity-field
assignments across 35 files, including the generator. Known move/form/ability/item/
condition sinks have full-domain bounds. Ordinary calls and constructors retain
an enclosing pinned-runtime obligation so callback aliases are not silently
ignored. Loaded callback-valued table properties must exactly match the AST
index, including nested arrays. Every recorded site is classified; **zero
unclassified sites does not mean a resolved call graph or a reachability proof**.
The source and build pin covers the remaining runtime/library dependencies.

## Reachability and generator boundaries

The conservative envelope deliberately contains the entire pinned base Dex,
including legacy and nonstandard candidates. It is an over-approximation for
planning and rejecting unsupported behavior. It does not claim that all 3,509
entries can occur in an official Gen9 random battle, nor that every pair can
co-occur. This version certifies no global unreachability exclusions. Tightening
it requires a separately checked proof or analysis, not frequency thresholds.

Initial move/ability candidates come from all 509 declarative source entries.
Initial emitted forms are enumerated by running the pinned `getForme` method
with every alternative of its at-most-one sampling call for this fixed profile;
changes to that sampling structure fail, and other random methods/direct
PRNG access are forbidden during enumeration. Item candidates inspect literals in
both singles item selectors, all `requiredItems`, and the five Gluttony berries.
That item list is conservative, not an exhaustive branch-feasibility proof; the
support envelope still includes all pinned items.

The explicit indirect audit includes Sleep Talk, Transform/Imposter, Trace,
Dancer, Magic Bounce, Struggle, item restoration/transfer and form changes.
Metronome, Assist, Copycat, Mimic, Sketch, Instruct, Me First, Mirror Move,
Nature Power and ability-changing moves are retained even where an initial root
is not established. The sampled absence of a move is never used to prove it
unreachable. Active-move mutations and newly created effect objects are source
obligations; R6 does not pretend they form a finite catalog of object values.

## Fixture catalog and evidence

Ten fixture families map to `ENG-MOVE`, `ENG-ABILITY`, `ENG-ITEM`, `ENG-FORM`,
`ENG-CONDITION`, `ENG-EVENT`, `ENG-CALL`, `ENG-COPY`, `ENG-TERA` and `ENG-CORE`.
Each entry has declarative parity obligations and a separate obligation for each
callback path. Required dimensions include positive/failure cases, both seats,
cleanup, immunity/suppression/targeting interactions and random branches.

Sixteen new scripted reference cases exercise eight mechanisms in both seats:

| Mechanism | Assertion |
|---|---|
| Sleep Talk | Sleeping user invokes its eligible Tackle slot |
| Imposter | Copies target moves/ability; five PP; retains Ditto HP/level |
| Magic Bounce | Spore reflects to its source; target stays awake |
| Dancer | Both Pokemon receive Quiver Dance boosts; external call recorded |
| Trace | Copies Intimidate while base ability remains Trace |
| Ogerpon Tera | Wellspring Tera form and Embody Aspect ability appear |
| Recycle | Consumed Chesto Berry is restored after Rest |
| Zero to Hero | Palafin switches through a pivot and returns in Hero form |

All sixteen pass semantic assertions and match on a second initialization/tape
replay. Seeds, fixed teams, choice tapes, normalized protocol and selected state
checkpoints are saved. Only timestamp protocol fields are normalized. Checkpoints
are diagnostic, not portable restart states; RNG checkpoint seeds are retained,
but there is no per-draw trace or statistical distribution test yet.

These are constructed fixed-team examples, not official generator witnesses.
R4's 13 phase fixtures and R5's 40 observer probes are hash-linked as inherited
evidence with their original scope. A reference case does not certify all
members of its family. **All differential checks remain planned.**

## Fail-closed use

`python/pokemon_reference/mechanics.py` independently verifies canonical hashes,
source fragments, initial corpus coverage, callback/pathway links, fixture/catalog
links and producer hashes. Source paths are confined to their declared roots.

`SupportGate.require(keys)` rejects unknown mechanics, empty profiles and every
known unsupported entry with its owning implementation ticket.
`require_profile()` rejects the complete envelope. R6 does not expose a JSON flag
that can authorize engine execution: `implemented-tested` is invalid in this
version. The Showdown oracle remains usable through the existing reference path.
An accelerated engine will need a separate implementation/evidence contract;
a deliberately reduced environment must have a new profile ID.

## Reproduce and inspect

```sh
npm run mechanics:setup
npm run mechanics -- /tmp/pokemon-r6-check
PYTHONPATH=python .venv/bin/python -m pokemon_reference.mechanics /tmp/pokemon-r6-check
npm test
```

The TypeScript parser is separately locked at 5.9.3 under `tools/mechanics-deps`.
The root package lock and R1–R5 producer/consumer implementations are unchanged.
Generation refuses to overwrite a different artifact at an existing destination.
Use a fresh path when changing the producer.

- Implementation: `packages/mechanics/`; independent consumer: `python/pokemon_reference/mechanics.py`.
- Contract: `contracts/mechanics-v1/README.md`.
- Current saved run: `artifacts/runs/r6-mechanics-v4/` (manifest, catalog, fixtures, report).
- Review and final verification: `artifacts/reviews/r6/`; dispositions in [R6_REVIEW.md](R6_REVIEW.md).
- Sealed completion: `artifacts/reports/r6-mechanics.json`; matching content-addressed source/evidence archive under `artifacts/manifests/r6-<completion-hash>/`.
- Final manifest hash: `1dabde6ef062af4c37b1f739a5d3e372b8ab98630ee1da4dd6e3467e37309905`.

Draft runs v1–v3 are retained for audit only. Use v4; the independent validator
rejects stale producers and the draft with duplicate pathway IDs.

## Final verification

The full regression suite passed: **56 Node tests and 51 Python tests**. A fresh
generation after the suite produced byte-identical manifest, catalog, fixtures
and report files. Independent Python validation checked all 3,509 entries,
10,710 pathways and 16 reference cases. Every source hash bound into the R5
completion report is unchanged. Review findings and the rejected draft are
retained rather than treated as successful results.

## Next work

**R7: deterministic dataset partitions, provenance/leakage gates and a sprint
benchmark report.** Split whole teams before pairing, keep mirrored perspectives
together, seal final evaluation data, verify actor loaders and measure local
throughput/size/failure/replay rates. No Prime Intellect compute is needed yet.

Before any custom-engine acceleration, implement the catalog tickets and
per-entry/interaction differential fixtures. A smaller justified envelope may
reduce that workload, but R6 does not silently assume it. Training can first use
the pinned reference environment once R7 establishes the data/evaluation boundary.
