Implemented `python/pokemon_reference/dataset_identity.py` (the only file touched).

## What it does

**`canonical_team(payload)`**
- Requires the exact R2 `team` field set (`generator_seed`, `source_team_hash`, `pokemon`, `ids`), then runs `validate_schema(payload, 'team')` with `ValidationError`/`TypeError`/`ValueError` wrapped into `ValueError`.
- Independently re-checks exact field sets for each `ids[i]`, `pokemon[i]`, `evs`, `ivs` (unknown fields like `nature` raise explicitly naming them; bools are rejected as ints via `type(x) is int`), plus level/gender/stat ranges and move list shape.
- Pairs `ids[i]` ↔ `pokemon[i]` by position only (no Dex lookup, no `speciesId`/`source_team_hash` cross-checks — the tests edit `level` without recomputing the source hash and set `speciesId` to bookkeeping text, so those must not participate).
- Emits an ordered list of `{species, ability, item, moves, tera_type, level, gender, evs, ivs}`; `base_species`, `name`, `shiny`, `role`, `speciesId`, seed and source hash are excluded. Output is `deepcopy`'d so it shares no containers with the input; the input is never mutated.

**`team_key`** = `hash_json([policy['identity_version'], canonical_team(payload)])`.

**`partition`** = `int(hash_json([hash_version, salt, team]), 16)` as a full unsigned 256-bit integer; assignment is `lower * 2**256 <= digest * denominator < upper * 2**256` — pure integer arithmetic, no floats, no modulo. Matches the test's expected formula exactly.

**`pair_partition`** raises `ValueError` if the two teams land in different splits.

**Policy validation** (`_validate_policy`): exact field set; pins `schema_version`, `identity_version`, `hash_version`, `profile`, `showdown_commit`, `holdout_unit`, `ignored_fields`; non-empty string salt; `denominator == 10000`; exactly three ranges `train [0,9000)`, `development [9000,9500)`, `final [9500,10000)` with integer bounds. Any deviation (e.g. `denominator=0`, `upper=9001`, future versions) raises `ValueError`.

## Limitations / judgment calls
- I did not run the tests (no execution tool available); the controller's acceptance run is the verification.
- `holdout_unit` and `ignored_fields` are pinned exactly in addition to the versions/profile/commit. That's stricter than the minimum the task asked for — I chose it because those fields define the identity semantics, so a change there without an `identity_version` bump is an inconsistent policy. If the controller wants those left free-form, that's a one-line removal from `PINNED_POLICY`.
- `base_species` is excluded from the canonical team per the stated field list ("species, ability, item, moves, tera_type"); it's redundant with the resolved `species` anyway.
