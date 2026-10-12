**Provisional verdict: findings.** Most important: `-heal` source attribution mis-credits absorb-type abilities to the attacker.

## Findings

**1. Medium — `public_history.py:157-167`: heal `[of]` attribution inverts ability ownership.**
The code treats `[of] X` on `-damage`/`-heal` as "X owns the `[from]` item/ability". For `damage()` this matches battle.ts (Rough Skin/Iron Barbs/Aftermath pass the holder as source). For `heal()`, battle.ts's default branch emits `[of] source` where `source` defaults to `this.event.source` — the *attacker* in TryHit. Real logs look like `|-heal|p2a: Lanturn|100/100|[from] ability: Volt Absorb|[of] p1a: Rotom`. Here `owner = subject('p1a: Rotom')`, so Rotom's `last_seen_ability` becomes Volt Absorb and Lanturn's stays unknown. Affects Water/Volt Absorb, Dry Skin, Earth Eater (both own public row and opponent rows). Not a hidden-information leak (public line, wrong interpretation), but the comment at line 155 ("Pinned battle.ts damage/heal emitters identify the effect owner this way") overclaims. No test or live probe covers an absorb heal. *Verify:* read the pinned `sim/battle.ts` `heal()` default branch; add a live probe (Water move into Water Absorb) and assert `last_seen_ability` lands on the defender.

**2. Low (research validity) — `feature_campaign.py:88-99`, `R5_DESIGN.md:46-47`: the 1,000 pairs are tautological.**
`project_player_views` reads only `decisions[i].side/view` and `bootstrap_views`; the carrier preserves exactly those for the viewer. The pair cannot fail except by code change, so "1,000 paired information-boundary perturbations" is satisfied by a test that exercises adapter plumbing, not hidden-state invariance. The report labels this honestly (`adapter_invariant_pairs`, limitations), but the acceptance wording should say so explicitly. Substantive live invariance evidence is ~18 pairs, and 7/9 per viewer mutate state that cannot emit protocol at that instant (bench `pokemon[1]` stats/species/item, PRNG seed with no rolls, uncommitted vs committed choice). Only `hidden_hp_bucket` (197 vs 199 of 400 → both ceil to 50%) tests a genuine observable-boundary case. Limitation, not bug.

**3. Low — `public_history.py:81-83`: silently-ignored structured lines are not surfaced.**
Lines with a structured `kind` but a non-active subject (e.g. `|-curestatus|p2: Nick|slp` from Heal Bell, `|-damage|p2: Nick|…`) update no row and are *not* counted in `unstructured_events`, so `global.unstructured_events` under-reports unmodelled information. They remain in the byte window, so design claim "never silently treated as known" holds; the counter semantics just differ from what a reader would assume.

## Checked, no issues found
- Transactional append: all mutable state (`prefix`, `own_max`, deep-copied `public`, `request`) is staged and committed only after encode; idempotent replay returns the cached actor; stale/foreign/mask/extra-key views rejected (tests 94-112).
- Leakage: actor tensors derive only from the verified request (owner-checked, line 107-109) and delta lines; `split`/`win`/`tie` rejected; opponent HP forced to `/100`; `-transform` copies only the target's public `apparent_species`; `replace` edits only the current encounter.
- `ActionCodec`/`validate_tensors` binding↔mask consistency; wait/pass have no policy indices.
- `-sethp` loop bounds and `[from]` break; `_bytes_rows` padding/overflow invariants match validator.
- Fixture `toString()` claim for active target ident (`p1a: Name`) is consistent with Showdown and asserted at `player-feature-fixtures.mjs:84`.

## Scope limitations
No files/tests executed; battle.ts not in packet (Finding 1 relies on knowledge of the emitter, so treat as high-confidence but verify against the pinned revision). R4 projection correctness, vocabulary coverage (`recharge`/`struggle` ids), and schema files not reviewed.
