Both spot checks match the fixture (fennekin→bulbasaur hp 37: 120, roll 1 → 116; fennekin→squirtle: 30, 29), so the formula ordering and type handling are confirmed on physical/special, STAB/non-STAB, ¼/½/×2 cases. Nothing further is needed from the workspace.

## Result

Implemented `python/pokemon_switch_engine/damage.py` (only owned path touched). Stdlib-free, pure, deterministic.

**API**
- `stats(species, swift=False)` → new `[hp, atk, def, spa, spd, spe]` list; L50, IV31, EV0, Serious; `swift=True` adds Speed EV252 (`+63`).
- `max_pp(species)` → new `[move0, move1]` list (40/40/24 and 56).
- `damage(attacker, defender, hp, move, crit, roll)` → int.

**Formula (as specified in the contract, verified by hand against the fixture)**
1. Signature-type move with `3*hp <= maxhp` → offensive stat × 1.5 via `(v*6144+2047)//4096` (power not boosted).
2. `floor(floor(22*power*attack/defense)/50) + 2`.
3. Crit: `*3//2`. 4. Roll: `*(100-roll)//100`. 5. STAB via the same 6144/4096 rounding (no Normal STAB). 6. Type multipliers applied per defending type, flooring each halving (so Grass vs Grass/Poison = ¼). 7. `max(1, …)`.

**Validation** — species must be one of the three lowercase IDs (`ValueError`); `swift`/`crit` must be actual `bool` (`TypeError`); `hp`/`move`/`roll` must be non-bool `int` (`TypeError`) within `1..maxhp` / `0..1` / `0..15` (`ValueError`). This covers every rejection case in `test_rejects_bad_profile_arguments`.

**Hand-verified against fixture** (not a substitute for the test run): bulbasaur→bulbasaur hp 39 move 0 rolls 0/4/12/13 (11/10/10/9) and crit rolls 0/1 (17/16); fennekin→bulbasaur hp 37 rolls 0/1 (120/116); fennekin→squirtle hp 37 rolls 0/1 (30/29); stat expectations in `test_stats_and_pp` (120/69/…/65, 119/…/95, 115/…/111) computed by hand match.

**Limitations**
- I did not execute the test suite or the "provided check" (no execution tool available in this task); the coordinator should run `python tests/switch-engine/test_damage.py`. The 3456-value claim rests on the hand-checked samples above plus the formula derivation, not an exhaustive run.
- Ability threshold is `3*hp <= maxhp` using the attacker's **non-swift** max HP (swift only changes Speed, so HP is identical either way).
