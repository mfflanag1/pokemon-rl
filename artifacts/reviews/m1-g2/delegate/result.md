The implementation is complete; no further reads or edits are needed. I traced the six supplied tests against the code by hand (not by execution):

- `test_initial_and_no_mutation` → header `[1,0,0,105,105,56,48,0,0,1,1]`, 8 padded rows; the reducer only reads the input.
- `test_exact_hp_and_reveal` → opponent Quick Attack reveals `opponent_quickattack_seen=1`, own PP stays 56/48, history row `[1,1,1,0,34,71]`.
- `test_crit` → own PP 55/48 matches the fixture, row `[1,0,0,1,58,47]`.
- The rejection tests hit, respectively: extra top-level key, request side ≠ carrier side, unknown `-status` command, damage without an executing move, turn 1→3 skip, `|win|P1` with no knockout, and request PP ≠ reconstructed PP.

## Result

Created `python/pokemon_jax/views.py` (stdlib `json` only, no new dependencies). `reduce_view(view)` returns `{'header': [11 ints], 'history': [8 × [6 ints]]}` per the contract, raising `ValueError` on everything else.

Invariants implemented:
- **Carrier**: exactly the keys `side,request_id,turn,request,protocol`; side ∈ {p1,p2}; `request_id` nonnegative non-bool int equal to the count of `|request|` lines; outer `turn` equal to the last protocol turn; protocol a list of strings.
- **Protocol**: initial switches (exactly one per side, `Rattata, L50, M`, `105/105`, before turn 1; later switches rejected); turns progress 1,2,… ≤ 4; move must target the opponent, be Tackle/Quick Attack, by/against a living Pokémon, and have exactly one damage line with ≤1 preceding `-crit`; damage strings `N/105` (1–105) or `0 fnt`; non-decreasing HP rejected; `faint` only for the just-KO'd side; `win|P1/P2` requires the loser fainted, no duplicate, nothing but metadata afterward; KO without a win line rejected; >8 hits rejected; extra fields on event lines (e.g. `[miss]`, `[from]`) and unknown commands rejected; listed metadata ignored.
- **History**: `hp_lost = pre_hit_hp − post_hit_hp` (so KO rows are capped at pre-hit HP); only *executed* moves consume/reveal PP (a KO-cancelled move has no line and is never revealed).
- **Request**: live views require own `side.id`, one active Pokémon with condition equal to public HP, two enabled moves tackle/quickattack with PP equal to reconstructed PP and maxpp 56/48, no `canTerastallize`/`teamPreview`/`forceSwitch`/`wait`. Terminal views accept `None` or a stale own request (PP/HP ignored) but still reject another side's request; masks become `[0,0]`.
- **Side isolation**: embedded `|request|` payloads are only checked for side identity (must be the carrier's side); their contents are never treated as events. Both seats work symmetrically via relative actor/winner mapping.

## Limitations

- I did not run the tests (no execution tool available); the pass claims above are from manual tracing only.
- Judgment calls where the contract is silent: a damage line with zero HP lost is rejected as impossible for these moves; a view with no `|turn|` line yet is rejected; `|tie` and turn-limit endings without a KO are rejected as unsupported (the contract's winner rule only describes KO wins). If the oracle capture emits a pre-turn-1 projection or a turn-limit result, these would need loosening.
- Only `python/pokemon_jax/views.py` was created; no other files were touched.
