**Provisional verdict: no blocking findings in reviewed scope; two validation gaps worth closing before the 10 000-case run.** Most important: `verify.py` never ties the captured reference back to the pinned Showdown commit in `profile.json`.

## Independent checks (hand-computed, not executed)

- **Damage**: `tr(2·50/5+2)=22`; `22·40·76/55 = 1216`; `/50 → 24`; `+2 → 26`. Crit `tr(26·1.5)=39`. Roll via `tr(base·roll/100)`, then STAB via Showdown `modify` = `tr((v·6144+2047)/4096)`. Yields 33/39 (no crit, roll 85/100) and 49/58 (crit): 39·6144+2047 = 241663 < 59·4096 = 241664 → 58. Python `engine.py:21-24` matches the excerpted `battle-actions.ts:1709,1722,1742,1746,1783` order (crit before roll before STAB). No type mod, burn, item or ability hook in this profile.
- **Four-turn bound**: min hit 33, 3·33 = 99 < 105 ≤ 4·33, so every surviving Pokémon has ≤6 HP entering turn 4 and the first mover KOs. Min PP 48 ≫ 4; PP exhaustion/Struggle unreachable. `engine.py:48,50,63` encode exactly this.
- **Order**: QA priority 1 > Tackle 0 (`moves.ts:14483,18819`); equal Spe 92 → `speedSort` shuffle. `engine.py:51-54` and `reference.mjs:38` (`tie_first = events[0].actor`) agree. KO cancels second move with no PP deduction; `engine.py:61-62` enforces exact event consumption.
- **Capture self-validation**: `reference.mjs:29-32` asserts roll captured exactly once in [85,100], crit boolean, `current===null` on entry — a mis-hooked `random`/`randomizer` would throw rather than silently record. `snapshot()` asserts `canTerastallize===undefined` every request, so Terastal Clause is runtime-checked.

## Findings

1. **Medium — Showdown commit pin not cross-verified.** `profile.json` pins `showdown_commit: 942cda7…`, `plan.json` records `reference: describe()`, but `verify.py:12-13` only checks `plan_hash`, `profile==PROFILE`, and hashes of `reference.mjs`/`profile.json`. Nothing asserts `plan['reference']` (or vendor file digests) equals the profile's pinned commit. A capture against a different vendor checkout would pass verification. *Verify*: add `require(plan['reference'][commit_key]==PROFILE['showdown_commit'])` and confirm `describe()` includes the commit.

2. **Low — Winner derived from player display name.** `reference.mjs:15` parses `b.winner.slice(1)`; this depends on names `P1`/`P2` set at line 21. Correct today, but a rename silently breaks seat indexing. *Verify*: derive from `b.sides.findIndex(s=>s.name===b.winner)` or the `|win|` message.

3. **Low — Verifier trusts the tape's provenance.** `verify.py` checks internal consistency only; it does not recompute `seed` from `seedFor` or `actions` from `hashJSON([...])`, nor check `plan['count']==10000`. Acceptable for conditional parity, but the "independent" claim should state that action/seed selection is taken from the fixture unverified.

4. **Note — Hard-coded constants.** `engine.py:21,30,38` embed 76/55/105/(56,48) rather than deriving from `PROFILE`. Frame equality would catch drift, but any profile edit requires manual edits here; consider asserting `PROFILE['initial_hp']==105` and `initial_pp==[56,48]` at import.

## Scope limitations

- Could not run tests or inspect `engine.mjs`, `provenance.mjs`, `canonical.mjs`, or `pokemon_reference.corpus`; `seedFor`, `describe`, `verify_seal` behavior unverified.
- Showdown `randomizer`/`randomChance` implementations not in excerpt; relied on the capture's runtime assertions.
- Terastal Clause semantics taken from the runtime `canTerastallize` check, not source.

## Verdict

Findings (non-blocking): rounding, order, KO cancellation, and the 4-turn bound check out; close finding 1 before treating the campaign as pinned-reference evidence.
