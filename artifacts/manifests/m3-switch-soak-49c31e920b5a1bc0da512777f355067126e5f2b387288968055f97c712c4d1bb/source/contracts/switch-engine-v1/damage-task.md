# Independent scalar damage implementation

Implement only `python/pokemon_switch_engine/damage.py`, dependency-free Python.
The existing tests/fixture and package initializer are read-only inputs. You are
not alone in this project; do not edit or revert anything outside this ownership.
No delegation from this task. The coordinator owns integration and the JAX port.

Public API: stats(species, swift=False) -> new list [hp,atk,def,spa,spd,spe];
max_pp(species) -> new list for two moves; damage(attacker,defender,hp,move,crit,roll)
-> integer. Species are lowercase IDs bulbasaur/squirtle/fennekin. swift and crit
are actual bools, hp/move/roll are actual ints (no bools). hp is in 1..maxhp,
move is 0/1, roll is Showdown's bounded draw 0..15 (damage factor 100-roll).
Reject unsupported arguments. Level 50, Serious, IV31, EV0 except swift Speed EV252.
Base stats in HP/Atk/Def/SpA/SpD/Spe order: Bulbasaur 45/49/49/65/65/45,
Squirtle 44/48/65/50/64/43, Fennekin 40/45/40/62/60/60.
Move0: Vine Whip physical45 Grass, Water Gun special40 Water, Incinerate special60
Fire; move1 is physical40 Normal (Tackle or Scratch). PP boosted by 8/5:
move0 40/40/24; move1 56. Bulbasaur is Grass/Poison, Squirtle Water, Fennekin Fire.

The exact closed-profile Gen9 formula: matching signature ability at 3*hp<=maxhp
multiplies the offensive stat by 1.5 with Pokémon fixed-point rounding
(value*6144+2047)//4096; do not boost base power. At level50 compute
floor(floor(22*power*attack/defense)/50)+2. If critical multiply by3//2.
Multiply by (100-roll)//100 with floor. Apply signature STAB via the same
6144/4096 rounding. Then type multipliers, flooring each resistance halving,
and minimum1. No Normal STAB. Grass vs Bulbasaur is 1/4, not1/2; all other
matchups follow these fixed types. The 3,456-value fixture is extracted unchanged
from the sealed real Showdown reference grid, including both seats and threshold
boundaries. Derive the formula, do not make a lookup table from the fixture.

Acceptance: all supplied tests pass, no dependency beyond stdlib, no changes to
inputs/tests, pure deterministic functions, concise implementation. Run the
provided check; the coordinator will perform opposite-model QA and integration.
