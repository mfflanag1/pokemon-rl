# Independent scalar continuation implementation

Own only `python/pokemon_continuation/scalar.py`. You are not alone in the project;
the coordinator owns the JAX port, reference harness and tests. Do not alter any
other file, revert others' edits or recursively delegate.

Implement the original integer switching state machine under the additive
continuation plan. Reuse the sealed pure damage/stats functions and roster
definitions; do not mutate original modules or monkeypatch global constants.
Export `SPECIES, ROSTER, ROSTER_IDS, require, stats, damage, max_pp, CAPACITY,
MAX_TURNS, MAX_GROUPS, MAX_EVENTS`, and `initial, masks, step, view, percent`.
Same dict state/events/player fields as M3, but full history padded to128 in
the player view. Permit transitions through turn60 inclusive; reject turn61
submissions. Window boundaries at13/25/37/49 do not change game state.
Terminal/phase semantics, speed order, chance validation, move PP, switch/forced
switch order, hidden information and exact public HP rounding remain identical.
State/action/chance inputs must not be mutated. Preserve complete history and
fail on overflow instead of dropping rows. Use the fixed JSON plan or explicit
matching constants. No NumPy/JAX imports or file edits outside the owned module.

The coordinator's tests exercise >32 history, five windows, old-engine prefix
parity, rejection of horizon/illegal action/chance, hidden reserve/speed invariance
and delayed post-turn12 outcomes. Passing these does not prove all-state parity;
fresh Showdown and independent JAX comparisons will follow locally.
