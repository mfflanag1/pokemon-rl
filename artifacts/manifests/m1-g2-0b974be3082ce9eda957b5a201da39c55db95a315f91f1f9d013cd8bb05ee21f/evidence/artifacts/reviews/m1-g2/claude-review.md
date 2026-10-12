**Provisional verdict: findings (low severity only; no correctness/leakage bug found in reviewed scope).** Most important: the Python admission's `provenance()` does not hash the test oracle or the profile, so the "source drift" gate cannot detect edits to what the JAX port is being checked against.

## Findings

**1. Low — Provenance omits oracle/profile/sealing code** (`campaign.py:23-26`)
`provenance()` hashes `python/pokemon_jax/*.py`, `contracts/jax-slice-v1/*`, `reference.mjs`, `requirements-jax.txt`. It omits `python/pokemon_slice/engine.py` (the CPU oracle whose `damage`/`PROFILE` the port is compared to), `contracts/slice-v1/profile.json`, and `pokemon_reference`/`pokemon_learner` (`hash_json`, `wire`, `digest`, `verify_seal`). An edit to any of these changes results without changing `source_hashes`. `runtime` also omits `jax.config.x64_enabled`, which affects dtype promotion in `rollout` (`campaign.py:119`). Verification: add these paths to `provenance()`; record x64 flag.

**2. Low — `validate_state` import-rejection is weaker than its docstring claims** (`engine.py:84-108`)
Turn-sequence checks only enforce `t ∈ {last_turn, last_turn+1}` and `t ≤ turn`; a non-terminal turn with a single hit is not rejected. Example: history turns `[1,2,2,3]`, live, `turn=3`, `count=4` satisfies `count==2*(turn-1)` and all per-row checks if HP/PP replay consistently. Unreachable from the kernels, but the comment "reject unsupported imported states" overstates. Verification: construct such a state and call `step`; add "each non-final turn has exactly two hits" check.

**3. Low — Family-size/power claim for the Showdown population** (`statistics.py:36-41`, `campaign.py:107`, `campaign.json:13-14`)
`fresh_showdown_first_turn` is fed `turns=ones`, so 105 per-turn cells (turns 2–4) are vacuous (`size=0 → p=1`) yet counted in `family_test_count=1184`. Turn-1 cells exactly duplicate the global cells. With n≈1000 (≈500 valid ties) and Bonferroni cutoff ≈8.4e-6, rare cells (e.g. `crit-both-tie`, p=1/1152, expected ≈0.4) cannot fail. "passed" is non-detection at very low power; the `scope` string says "finite checks" but not this. Verification: report per-cell minimum attainable p-value or achievable power; avoid implying 592 informative tests per population.

**4. Low — Leakage tests are weaker than their names** (`test_engine.py:27-34, 64-68`; `campaign.py:56-58`)
`test_cancel_does_not_reveal_choice` compares only seat 0's view; seat 1 (whose move was canceled) is the one whose own-PP/history could leak. `test_rng_and_future_cannot_change_current_view` cannot fail (views don't depend on `sample`). `compare_views` actor "independence" check rebuilds `PlayerView` from arrays just asserted equal — tautological. Verification: add seat-1 comparison in the cancel test.

## Verified OK (reasoned from packet)
- Damage: base 26/39 (atk 76, def 55), crit before roll, STAB `(x*6144+2047)//4096` round-half-down; matches oracle and Showdown order.
- KO cancellation: second actor's PP/history/events masked by `live`; `turn` frozen at terminal; validate relations consistent with 7-hit bound.
- Chance mapping: `LIMIT24=4294967280=⌊2³²/24⌋·24` exact; `&15`, `&1` exact; Showdown roll `100-random(16)` ∈ [85,100]; `random(16)` hook does not collide with `randomChance(1,24)`.
- Reducer/engine agree on `hp_lost` capping, `seen` from executed moves, terminal masks, turn at terminal, padding -1.
- Cell probabilities (1/384, 23/384, 1/48, 1/32, 530/576, 1/1152) and 583+9=592 count check out.

## Scope limitations
Not in packet: `pokemon_reference.*`, `pokemon_learner.checkpoint`, `reference-node/*.mjs`, `PlayerProjection`, the sealed reference JSONs, old campaign. Showdown PRNG bias (~2⁻²⁷) and `fold_in` collision behavior assumed negligible. I did not run tests.
