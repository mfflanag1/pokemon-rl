Both owned files are written; no further tool calls are needed.

## Result

**Created (only these two paths):**

- `python/pokemon_switch/contract.py` — standard-library-only codec bound to `contracts/switch-v1/profile.json` (loaded and validated at runtime, not duplicated).
  - `mask(request) -> (bool, bool, bool, bool)`; `command(request, index) -> "move 1" | "move 2" | "switch <current 1-based position>"`. Also `Codec`, `decode`, `load_profile`, `parse_profile`.
  - Members are located by exact ident suffix `"<side.id>: A|B"`, so request reordering after a switch never moves identities; switch commands carry the member's *current* position.
  - Request kinds: `wait:true` (empty mask; `side` optional but validated if present), `forceSwitch:[true]` (bench-living only, both move bits off), singles move request (one `active` slot, exactly two moves whose IDs match the active member's profile set in declared order, `pp`/`maxpp` as non-bool ints with `0 ≤ pp ≤ maxpp`, `disabled` a real bool).
  - Rejects: `canTerastallize`/`trapped`/`maybeTrapped`/`maybeDisabled`/Mega/Z/Dynamax flags, `reviving`/`commanding`/`terastallized`, `teamPreview`/`maxChosenTeamSize`, extra active or forceSwitch dimensions, Struggle, unknown/misordered move IDs or wrong move count, side-pokemon move set drift, wrong `side.id`, ident/side mismatch, duplicate or unknown names, duplicate species, details not exactly `"<Species>, L50, M"`, ≠1 active, fainted active in a move request, held items, forceSwitch with no living bench, non-`True` `wait`.
  - `command` fully decodes the request *before* touching the index; `bool`, out-of-range, and non-int indices raise `ContractError`; a valid index on a masked-off action raises `IllegalActionError` (a subclass).
- `tests/switch-reference/test_contract.py` — `unittest`, with a `sys.path` shim to `python/`. Covers: four-boolean shape; stable identities after reorder (`[B, A]` → index 2 still means A → `"switch 2"`, index 3 illegal); forced-switch and wait handling (both orderings, exact-`true` strictness); PP zero and disabled masking; malformed PP/disabled (bool/str/float/negative/over-max/`"hidden"`/missing); Struggle, unknown/misordered/other-species moves and wrong counts; every out-of-scope flag above; side/ident/details malformations; bool/invalid index rejection with the explicit check that `True` is not coerced to `"move 2"`; rejection before command generation (asserted as `ContractError` but *not* `IllegalActionError`); request non-mutation; profile parsing and drift rejection.

**Not touched:** `profile.json`, `README.md`, `__init__.py`, `.git`.

## Limitations

- **I did not run the tests** — this session has no shell tool. I desk-checked both files line by line and fixed two defects that surfaced (a fixture binding species to list position instead of original name, which would have broken the reorder test; and `ctx.exception` being read outside a `subTest`). I cannot claim the suite passes; the controller's run is the actual check.
- Request shapes follow stock Showdown's `getRequestData` layout from memory (no fixtures were present to verify against). If the golden file carries something I treat as out-of-scope that is actually stock (e.g. a non-empty `item`, a `details` form other than `"<Species>, L50, M"`), the strictness should be relaxed there rather than in the identity logic.
- Two judgment calls where the brief left room: a `forceSwitch` with no living bench is rejected as unreachable rather than returning an all-false mask; and `wait` requests with a `side` still validate it. Unknown top-level/pokemon keys are tolerated; only enumerated out-of-scope flags are rejected.
- Stale `rqid` and extra submission fields (README §"Reject...") are a submission-layer concern and are intentionally not part of this codec's API.
