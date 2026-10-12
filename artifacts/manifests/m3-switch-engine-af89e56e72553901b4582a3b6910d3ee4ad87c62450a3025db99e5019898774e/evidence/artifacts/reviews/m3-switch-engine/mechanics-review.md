**Provisional verdict: findings.** Most important: the public-HP percent rule in scalar/JAX/README (`ceil(100*HP/maxHP)`) is exact-rational, but the pinned Showdown oracle computes `Math.ceil((hp/maxhp)*100)` in IEEE doubles. For the one reachable exact-integer ratio in this profile that is not float-stable, Bulbasaur at 66/120 HP, Showdown emits `56/100` while `scalar.percent`/`engine._percent` yield 55. Current 72+24-game parity simply never hit that HP value.

## Findings

**1. Medium — public HP percent false parity at 66/120 (`scalar.py:76-78`, `engine.py:141-143`, README L26).**
Evidence: `0.55` as a double is 0.55000000000000004440…; ×100 rounds to 55.000000000000007 (ulp near 55 is 7.1e-15, value is above the midpoint), so `Math.ceil` returns 56. Showdown's `getHealth` uses exactly this float path under HP Percentage Mod. The scalar/JAX integer ceil returns 55. Enumerating all `hp/maxhp` with `100·hp/maxhp` integral for max ∈ {115,119,120}, 66/120 is the only divergent case; all other ratios are far enough from integers that float error cannot change the ceiling. Effect: opponent `opponent[.,1]` and history `HP_after` differ from the real wire → learned policy sees an observation the production oracle never produces; any production run reaching 66/120 fails `compare(actual, expected, 'Reference view …')`.
Verification: add a unit test asserting `percent(66,120)==56` against a Node one-liner `Math.ceil(66/120*100)`; or implement the Showdown float rule verbatim (compute in float64 in scalar; in JAX int32/FP32-only this is nontrivial — a lookup/exception for the single pair is defensible given the closed roster).

**2. Low/Medium — `chance.py:16` roll mapping unverifiable; sign convention risk.**
`Chance.roll` must be 0..15 (`engine.step` rejects otherwise, `damage` uses `100-roll`). `sample` returns `100-mapped['roll']`, so correctness depends on `pokemon_showdown.mapping.kernel` returning the *factor* 85..100, not the raw `word>>>28`. The mapping module is outside the packet; if it returns 0..15, every sampled batch is rejected (or, worse, silently wrong if validation is bypassed in a batched trainer). Also `crit`/`tie0` dtypes (bool vs uint32) are unverified. Missing evidence, not a confirmed bug.

**3. Low — independent checker does not bind `chance` to the recorded draws (`parity.py:48-51`).**
The checker validates word→result mapping per draw but never checks `chance.crit[side]`/`chance.roll[side]` equal the corresponding draw results, nor that `chance.tie` equals the tie-draw result when one exists (`reference.mjs:80` sets `tie` from `events[0]` actor, not from the draw). A tie-ordering bug in scalar would be caught only probabilistically through event order. Suggest: per step, assert draw roles ⊇ one crit+roll per executed move, and `tie` draw result == `chance.tie` when speeds/classes tie.

**4. Low — passivity control is weaker than labelled (`reference.mjs:109-110`).**
`instrument:false` still installs all monkey-patches (`random`, `randomChance`, `shuffle`, `getDamage`, `switchIn`); only word logging is disabled. "Passive instrumentation changed battle" therefore tests the logger, not the wrappers. Real passivity evidence is the 72 legacy `expected` frame+audit (incl. PRNG seed) comparisons — adequate for legacy, absent for fresh games. Consider a truly unpatched Session for the 24-game control.

**Verified as consistent (no finding):** switch-before-move with outgoing speed (`scalar.py:49-53`, `engine.py:86-90`); +32 swift speed offset for all three species; `(22·P·A)//D//50+2`, crit 3//2, roll, STAB `+2047` rounding, per-type halving; Overgrow/Torrent/Blaze `3·hp≤max` on stat; fainted actor skipped; replacement before turn increment; 26-event reachable bound; encoder opponent `/100` leak guard.

## Scope limitations
Snapshot only; no tests executed. `pokemon_showdown.mapping`, `reference-node/*`, `pokemon_switch.*`, Showdown source at the pinned commit, and the fixture bundle were not available — Showdown float behaviour is asserted from the known `getHealth` implementation and IEEE arithmetic.

**Verdict: findings** (one concrete mechanics/false-parity defect, three evidence gaps).
