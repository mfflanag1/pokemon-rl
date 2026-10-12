**Provisional verdict: findings (none blocking the stated node tests; several unverified claims and gaps).** Most important: the Python adapter and the 72/1152/1008/864/48 fixture figures are not covered by the seven stated tests, and there is no negative check that opponent HP is always masked.

## Findings

**1. Medium — Fixture/information counts are asserted in code but not run by any stated test.**
`tests/switch-reference/session.test.mjs` never calls `generate()` or `information()` (fixtures.mjs:29–102). The 1152/1008/864/48 numbers and the 72-game opposite-order replay (fixtures.mjs:59,64–68) exist only as runtime asserts; no `summary.json` is in the packet. Treat them as unverified until the generator run output (and its hash) is supplied. Also, the probe loop asserts `g.phase==='joint_move'` and that original B is switchable after one Tackle/Scratch turn (fixtures.mjs:51) — reasonable, but a turn‑1 crit KO would abort the whole run rather than skip a probe.

**2. Medium — `python/pokemon_switch/views.py` reuses frozen M1 codec helpers for a two-mon switch profile.**
`request_phase`, `admissible_actions`, `action_mask` (views.py:2,19) were built for the one-mon Rattata duel. Nothing in the packet shows they accept two‑member `side.pokemon`, `forceSwitch` requests, or the four-action switch mapping; a mismatch would silently produce wrong masks in the "diagnostic" R5 tensors. Verify with a Python test replaying `golden[].mask` against `action_mask(view.request)` for all three phases.

**3. Low/Medium — Leak checks are positive-only.**
`coverage.masked_hp` (fixtures.mjs:94) only proves some `/100` opponent line exists. Add a negative invariant over every view: all `|switch|`/`|-damage|` lines for the *other* side use `/100`, none expose `x/y` with y≠100, and `request.side.id===view.side`. Also assert `|-heal|`, `|-status|`, `|-boost|` never appear (currently checked only via audit, not via the player channel).

**4. Low — Chance contract omits accuracy-roll PRNG consumption.**
profile.json:27 lists crits/damage/ties/switch ordering. In Showdown, 100-accuracy moves still call `randomChance(accuracy,100)`, which advances the PRNG. Not a current correctness issue, but a future "raw-stream parity" port would diverge; verify against commit `942cda71…` and add to the contract.

**5. Low — `own_variant_observations` is tautological.**
fixtures.mjs:45–47 compares the request's own `stats.spe` to the same simulator mon's `storedStats.spe`. It demonstrates nothing about information flow; don't cite it as a positive "own private speed is an input" result. A real test would show the two `plain`/`swift` own-variant views hash differently at turn 1.

**6. Low — Damage oracle minor hygiene.**
damage.mjs:21 uses global `Dex.getActiveMove`, not `battle.dex`; equivalent for gen9 here but should be pinned. `battle.random` is replaced and not restored (acceptable since the session is closed). The three-HP threshold check (hp=⌊max/3⌋−1, ⌊max/3⌋, +1) correctly brackets `hp <= maxhp/3` for both divisible and non-divisible maxhp; 18×3×2×2×16=3456 is arithmetically consistent.

**7. Note — Fixture file bundles privileged data with views.**
`snapshots[].audit` and `comparisons[].opponent_own_hash` sit alongside `golden` views in the same gz. README says to treat audit as privileged; downstream loaders must enforce this (views.py:7 does reject extra keys, which is good).

## Reviewed and found consistent
Request binding (group, request_id, view_hash, exact-key check; stale/duplicate/extra rejected, partial-write `failed` latch), A/B mapping by ident suffix with 1-based current position, wait→all-false, forceSwitch→only non-active non-fainted, pending-choice leak assert between the two writes, truncation only at `joint_move` with turn>12 (replacement at turn 13 may still resolve, matching the stated cutoff), 1152/1008/864 key arithmetic.

## Scope limitations
No execution; Showdown source not in packet; Python codec and regression explicitly incomplete; `engine.mjs`, `canonical.mjs`, `PlayerHistory` not reviewed.
