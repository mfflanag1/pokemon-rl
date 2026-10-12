# R3 review and verification record

2026-10-10. Scope: the pinned official Gen9 Random Battle team factory, sharded
collection, independent Python validation and inventory, and sampled exact replay.
The source pin remains `942cda71d14bf0618bd57991708a2e2734b6daba`.
Measured corpus results are in [R3_INVENTORY.md](R3_INVENTORY.md).

## Source audit and pilot discovery

The delegated source audit received opposite-model Codex QA and was adjudicated
against the complete local source and runtime evidence. Its full proposed
specification was rejected; verified facts were retained. The immutable disposition
is [R3_SEMANTICS_AUDIT.md](R3_SEMANTICS_AUDIT.md). In particular, role strings are
not template identities, declarative alternatives do not imply uniform sampling,
and source entry, emitted form and base species must be counted separately.

Independent Python validation of a 1,000-team pilot rejected `florgesblue`, absent
from the original R2 vocabulary. We verified that Dex synthesizes cosmetic forms
outside `species.all()`. Vocabulary v2 adds all 68 missing cosmetic IDs; v1 remains
preserved, and numeric token migration is documented. The corpus regression checks
all exported species-name resolutions against v2. R2 fixtures and its ten-episode
smoke report were regenerated after this change.

## Independent implementation review

Fable snapshot review:
`20261010-123215-97763f31`, project `Pokemon-83aa7cb62b4e`.
The reviewer reported non-blocking findings. Coordinator verification:

1. **Move resolution:** the reviewer suggested a name-to-ID resolver and questioned
   independent validation. The pinned `randomSet` returns move IDs directly from
   the selected move set (`teams.ts:1622–1631`); the R3 check requires equality,
   uniqueness and vocabulary membership. An identity resolver would add no
   independence. We retained the check and clarified its limits: schema/hash/ID
   validation cannot prove that a consistently rewritten artifact came from a
   generator seed. Sampled all-field oracle replay supplies that separate evidence.
2. **Failure attribution:** confirmed that non-generation errors could retain the
   preceding attempted index. We added explicit phases and reset attempted index
   and seed outside generation. We also found that assigning in-memory progress
   before atomic checkpoint publication could overstate the committed count.
   Progress now advances only after successful publication. A checkpoint collision
   regression verifies the reported durable prefix and exact recovery through the
   orphan shard.
3. **Compressed reproducibility:** confirmed that gzip bytes can depend on zlib
   and platform. Documentation distinguishes stored-file checksums from canonical
   JSONL content hashes; no cross-platform replay claim is made.
4. **Inventory accounting:** added display-form, source-conditioned, lead, template
   match and variable move-slot denominator checks. A fixed 24 moves per team would
   be wrong; the full sample contains sets with fewer than four moves.
5. **Zero lead counts:** replaced zero increments for non-leads with an explicit
   lead branch. The regression requires strictly positive stored lead counts.

An additional coordinator check found a race between two stale-lock recoveries.
Recovery now acquires a separate exclusive guard before inspecting/removing a
dead owner's lock. Fresh writer acquisition always uses exclusive creation.

Focused Fable follow-up: `20261010-124037-b4071d3e`. Static trace review found no
correctness counterexample for the tested recovery scenarios and verified the
checkpoint accounting correction. Its robustness notes were checked:

* A leftover recovery guard blocks later recovery, but a fresh acquisition with no
  writer lock remains safe. This behavior is now documented and tested.
* A writer exiting during recovery may cause a conservative filesystem error;
  retry is safe. A reused PID, remote host, corrupt lock or permission error never
  authorizes automatic deletion. Operator inspection may be required.
* Lock locality assumes a local directory and host/PID namespace. Shared network
  storage coordination and power-loss durability are not claimed.

The final producer generated a fresh 100,000-team directory after these fixes.
Every shard's compressed and uncompressed content matched the earlier run using
the same seeds. The final corpus ID differs because producer source hashes changed.

The final `npm test` run passed **40 Node tests and 13 Python tests** (7 contract,
6 corpus). `npm run contract:smoke` independently validated 1,677 artifacts from
10 episodes / 823 decisions / 359 turns, including two natural hidden-trapping
rejections; the 71,886,183 canonical JSONL bytes matched exactly across languages.
The completed full corpus also passed the same-command resume/integrity check.
The final small follow-up streams hashes of inventory output files instead of
reading the SQLite file into memory. All six corpus Python tests passed again;
the full inventory was rerun into `inventory-final/` with the final consumer hash.

## Verification boundaries

Tests exercise generation failure before and within a shard, recovery without seed
skips, an orphan file, checkpoint write failure, live/dead writer locks, recovery
guard behavior, wrong resume configuration, corrupt gzip, missing/reordered shards,
and rehashed invalid seeds/identifier mappings. Independent Python serial and
parallel analyses produce identical logical output files. All source files used by
the Python inventory are fingerprinted at load and checked again before publishing.

The full inventory checks every record; replay regenerates a deterministic sample
including all shard starts and the final row. Neither this sample nor complete
declarative coverage establishes dynamic battle-mechanics support, information
isolation for future model features, a train/evaluation partition, or policy strength.
The data remain exploratory and `unassigned`. R4–R7 remain separate acceptance gates.
