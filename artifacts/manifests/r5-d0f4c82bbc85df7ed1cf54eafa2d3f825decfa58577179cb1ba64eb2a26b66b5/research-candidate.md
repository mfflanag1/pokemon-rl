I have all five supplied files; nothing further is needed. Below is the audit, with every consequential claim tied to the supplied lines. Citation form: `src:pokemon.ts:NNN` = original line NNN quoted in `artifacts/reference/r5-information-source.txt`; `abilities.ts`, `SIM-PROTOCOL.md` likewise; `R5:NN`, `R4:NN`, `contract.py:NN`, `schema.json:NN` are file lines.

## A. Source facts the encoder must honour

| # | Fact | Evidence |
|---|---|---|
| F1 | Protocol `POKEMON` ident for an Illusion user is the **disguise's** nickname (`this.illusion.fullname`). | `pokemon.ts:532-533` |
| F2 | Illusion DETAILS = disguise's species/gender/shiny, but **level is the user's own** unless `illusionlevelmod`; `tera:` suffix and HP/status are the **user's own**. | `pokemon.ts:544-553`, `536-541` |
| F3 | Illusion disguises as the **last non-fainted party slot to the right** (may be an already-active mon); no disguise if user is last. | `abilities.ts:2057-2070` |
| F4 | `replace` + `-end … Illusion` are emitted only on End when **not** being called back; switch-out leaves no `replace`. `onFaint` clears illusion **silently** (no `replace` in excerpt). | `abilities.ts:2077-2091` |
| F5 | `replace` = "everything you thought you knew about the previous Pokémon is now wrong"; new ident carries the Illusion user's nickname. Only `switch/replace/drag/detailschange` change the mon at a position. | `SIM-PROTOCOL:284-291`, `221-225` |
| F6 | Identical `POKEMON|DETAILS` cannot be told apart by the viewer. | `SIM-PROTOCOL:260-264` |
| F7 | `transformInto` copies species, weight, types, **storedStats (not baseStoredStats, not HP)**, moveSlots (pp 5, `virtual`), boosts, crit volatiles, ability; **not item, HP, status, level**. Fails if either side has `illusion`. | `pokemon.ts:1273`, `1289-1352` |
| F8 | `-transform` is added with a **Pokémon object** as 3rd arg plus optional `[from] EFFECT`; doc says `SPECIES`. | `pokemon.ts:1344-1346` vs `SIM-PROTOCOL:533-536` |
| F9 | Opponent HP (gen≥7): `ceil(100*hp/maxhp)`, 100→99 if not full; `"N/100"`; fainted `"0 fnt"`; status appended. Own side gets `hp/maxhp`. `reportExactHP` would make shared = secret. | `pokemon.ts:2060-2104`, `SIM-PROTOCOL:266-268` |
| F10 | Request `stats` = `baseStoredStats` (5 stats, no HP); `moves` = current `moves` (not `baseMoves`) with `hiddenpower…`/`return…`/`frustration…` suffixes; `baseAbility` and (gen>6) `ability`; `condition` = exact secret HP. | `pokemon.ts:1153-1190`; `schema.json:393-493` |
| F11 | `[from] EFFECT|[of] SOURCE` "affect the message or animation … do not affect anything else"; doc gives **no ownership rule** for EFFECT when `[of]` absent. | `SIM-PROTOCOL:232-237` |
| F12 | `-item P I [from]` = I held by P (changed or revealed); `-item P I` = switch-in long-term (Air Balloon only); `-enditem P I [from]` = now none, **`[silent]` when ownership changed**; Sticky Barb never announces; `-ability P A [from]` = **changed** ability; Skill Swap sends nothing; `-endability` = suppressed, not removed. | `SIM-PROTOCOL:483-531` |
| F13 | `-damage` with HP 0: STATUS "should be ignored"; `fnt` not to be relied on. | `SIM-PROTOCOL:346-347` |
| F14 | View validator already enforces viewer-only requests, request_count/turn/last-request boundary; `known_id` `value:''` → `<none>` idx 1, unknown → idx 0, OOV → raises. | `contract.py:165-186`, `211-224` |

## B. Counterexamples and exact handling requirements

**Illusion / `replace`**
- C1 (nickname aliasing): Zoroark-as-"Garchomp" reveals an item via `-item|p2a: Garchomp|…` (F1). Later real Garchomp switches in with the identical `POKEMON|DETAILS` (F6, F3 — the disguise *is* a real party member). Requirement: key revealed evidence to the **appearance index**, never to nickname/species string (`R5:29` says this; make it a tested invariant). The real-Garchomp appearance must start with item/ability/moves = unknown.
- C2 (no `replace` on switch-out): per F4 a disguised mon that switches out never gets corrected. Requirement: an appearance that ended without `replace` must carry `identity_revealed=false` forever; the encoder must not retroactively "fix" it even when a later `replace` of a *different* appearance shows the same nickname (`R5:28-29`).
- C3 (`replace` keeps HP/status/tera): `replace|…|HP STATUS` is the same mon (F2, F5); HP is not reset. Requirement: on `replace`, update apparent species/details of the current appearance, set `replaced=true`, keep accumulated moves/items (they *were* Zoroark's), keep HP from the `replace` line (same denominator), and retain the pre-replace details as historical (either as a field or guaranteed in the byte window), so "apparent" vs "revealed-by-replace" is distinguishable (`R5:30`).
- C4 (level mismatch is public but must not be exploited by the encoder): F2 — a disguised "Garchomp, L78" vs an earlier real "Garchomp, L80" is detectable publicly. Requirement: do **not** add an Illusion detector; keep raw details string in the appearance so a learner may notice. Whether `illusionlevelmod` is in `gen9randombattle` is **not in evidence**.
- C5 (silent faint clearing): F4 `onFaint` nulls illusion with no `replace`. The `-damage … 0 fnt` line (F13) likely shows the disguise ident while any later `faint` line shows the true name (inference: `toString` F1 after null). Requirement: never count distinct species seen as roster/remaining count; count only `faint` events; treat species-of-fainted as apparent. Message ordering is **unresolved** (needs a fixture, test T11).
- C6 (Illusion blocks Transform): F7 line 1273 — no Illusion+Transform compound state exists; failure message form is **not in excerpt**.

**Transform**
- C7 (`-transform` 3rd field is an ident, not species): F8. Requirement: parse 3rd field as `pXa: Nick`; resolve display species **only** via the viewer's own current public appearance of that slot (or own request if the slot is the viewer's), never via Dex. If unresolvable, leave in bytes and mark apparent species unknown-after-transform.
- C8 (no copying into opponent record): when opponent Ditto transforms into the viewer's mon, the viewer *knows by mechanics* Ditto now has its moves/ability. Requirement (`R5:30-32`): opponent appearance moves/ability/item stay as last-observed/unknown; only display species/types-as-display may change. State explicitly in the design: **own request data is never used to populate opponent appearance fields**.
- C9 (own Transform skews request fields): F7+F10 — after own Ditto transforms, request `moves` = copied moves (pp 5), `ability` = copied, `baseAbility` = original, `stats` = **Ditto's untransformed baseStoredStats**, `details` unchanged-or-not (**not in evidence**). Requirement: label own `stats` as base; encode both `ability` and `baseAbility` (`R5:20-21` says only "ability"); do not derive effective stats.
- C10 (perturbation-test trap): changing opponent truth moves/ability in a paired test where the viewer's own mon has Transformed/been Imposter'd **changes the viewer's private request**, so it is not a valid "unchanged private history" pair (`R4:128-132`). Exclude or detect such pairs.
- C11 (no `-ability` guaranteed after Transform): `setAbility(…, true, true)` message behaviour **not in excerpt**. Requirement: never assume a follow-up `-ability`; boosts copied (F7) with no evidenced message → leave boosts in bytes.

**`[from]` / `[of]` attribution**
- C12: `-item|p2a: X|Choice Scarf|[from] ability: Frisk|[of] p1a: Mine` — Frisk belongs to the `[of]` mon, not X; `-transform|p2a: Ditto|p1a: Mine|[from] ability: Imposter` — Imposter belongs to POKEMON. Doc gives no rule (F11). Requirement: derive `known_id` item/ability **only** from the explicit doc semantics: `-ability P A[ [from]]` → A is P's current ability (`SIM-PROTOCOL:514,522`); `-item P I[ [from]]` → I is P's current item (`485,490`); `-enditem P I` → P item = none (`497,505`). Do **not** derive ownership from `[from] ability:/item:` or from `[of]`; retain in bytes (`R5:35-36`).
- C13 (`[silent]` enditem ordering): Trick emits silent `-enditem` then `-item`. Requirement: process lines in order; none → new item; both "last-observed".

**HP percent precision**
- C14: shown `N/100` means true fraction ∈ ((N−1)/100, N/100] for N≤98 or N=100; **N=99 covers (98/100, 1) exclusive** due to the clamp (F9). `1/100` = alive at ≤1%. Requirement: store `(numerator, denominator, fainted_flag, status)` verbatim; no float normalisation, no integer HP, no floor assumption (`R5:32-33`). `0 fnt`: no denominator, status ignored (F13). Own `condition` keeps exact `hp/maxhp` (schema `404` pattern; colour suffixes `y/r/g` are out of pattern — fine for gen9, but a `/48` string would validate).
- C15: whether `reportExactHP`/HP Percentage Mod is set for `gen9randombattle` is **not in evidence**; encoder must accept any denominator on opponent lines.

**Own baseStoredStats** — covered by C9; additionally, whether `baseStoredStats` changes on `formeChange`/`setSpecies` is **not in excerpt**. Requirement: name the field `base_stored_stats`, never "stats".

**Item/ability reveal certainty**
- C16: `<none>` (`contract.py:217`) is legitimate only after a `-enditem` for that appearance; absence of messages ≠ none (Sticky Barb silent, F12). After `-ability … [from]`, prior ability is historical; Skill Swap changes with no message, so "current ability" is never certain; `-endability` → add `suppressed` flag, keep last value (`R5:33-34` is correct but should enumerate these).

**Repeated switches**
- C17: X in → Y in → X in = 3 appearances; the two X's may be the same mon, or a disguise (F3/F6). Requirement: no merging; `drag` and `switch` both open appearances (`R5:26-27`), flag which. Recommendation (public-only, not a merge): expose `same_ident_details_as_prev_appearance` computed from strings, so HP continuity isn't wholly lost (`R5:41-42` limitation).
- C18: Window truncation (`R5:27`) must keep `truncated=true` plus counts of appearances dropped, so ≥7 appearances (possible via Illusion aliasing) is not mistaken for roster size.

## C. Current-vs-historical evidence traps
1. Pre-`replace` species/moves/items observed → attach to the same appearance, mark pre-replace details historical (C3).
2. `-formechange` (temporary) vs `detailschange` (permanent) (`SIM-PROTOCOL:274-282`): the former is display state that may not persist; encode as `display_species` with a `permanent` flag.
3. Last HP STATUS line per position (`switch/drag/replace/-damage/-heal/-sethp`) is current; a `-status`/`-curestatus` also updates status (`353-363`).
4. Ability/item "revealed" are last-observed (F12); Transform/Trick/Knock Off/Skill Swap alter without full disclosure.
5. Own request vs. protocol: requests are authoritative for own current data (`R5:22-23`); own `stats` are not current combat stats (C9).
6. Request count ≠ turn (`R4:64-65`); boundary validation at `contract.py:174-186`.

## D. Bounded adversarial tests (fixtures; label as adapter-noninterference, not live hidden-state, per `R5:49-51`)
- T1 Illusion replace in place: `switch|p2a: Garchomp|Garchomp, L78|100/100` → `move|p2a: Garchomp|Knock Off` → `-damage|p2a: Garchomp|57/100` → `replace|p2a: Zoroark|Zoroark, L78|57/100` → `-end|p2a: Zoroark|Illusion`. Assert: appearance count 1; `apparent_species=zoroark`, `replaced=true`; `knockoff` attached; HP (57,100) unchanged; pre-replace `garchomp` retained as historical; no roster assertion.
- T2 Switch-out without replace then real Garchomp enters: assert two appearances, second has empty reveals, first `identity_revealed=false`.
- T3 Nickname aliasing (C1): item revealed on disguised "X", later real "X" → item unknown on later appearance.
- T4 HP boundaries: opponent `1/100`, `99/100` (from 399/400), `100/100`, `50/100` (from 199/400 and 197/400 — must be byte-identical), `0 fnt` (status ignored). Own `399/400`, `0 fnt`.
- T5 Paired perturbations (≥1,000, `R5:46-47`): vary opponent hp within ceil bucket, unrevealed item/ability/moves, Illusion-vs-real before reveal, RNG/pending choice, group label; assert byte-identical actor input + mask. Exclude pairs invalidated by C10.
- T6 Transform both directions: `-transform|p1a: Ditto|p2a: Garchomp|[from] ability: Imposter` and `-transform|p2a: Ditto|p1a: Mine|[from] ability: Imposter`. Assert opponent record unchanged except display species (resolved by slot), HP unchanged; own request shows copied `moves` pp 5, `ability≠baseAbility`, `stats` unchanged — encoder labels base.
- T7 Attribution: the five lines in C12/C13 plus `-ability|p2a: X|Intimidate|[from] ability: Trace`, `-endability|p2a: X`, `-enditem|p2a: X|Sitrus Berry|[eat]`. Assert only doc-sanctioned fields change.
- T8 Repeated/drag switches beyond window: assert truncation flags and no merge; both `switch` and `drag` open appearances.
- T9 Viewer mirror: same episode as p1 and p2 views; assert own/opponent classification follows `viewer` prefix only (`contract.py:168-169`), never species.
- T10 Idempotent/stale delivery (`R5:12`): same view twice → unchanged; `request_id` mismatch → rejected, history unchanged.
- T11 Evidence-gathering (no assertion yet): Illusion user fainting from non-hit damage; record `-damage 0 fnt` vs `faint` ident order.
- T12 OOV own move (`return102`-style, F10) through `vocabulary_index` (`contract.py:221-223`): decide fail-loud vs dedicated OOV id; do not map own known move to `<unknown>`. Reachability in gen9 **unresolved**.

## E. Missing evidence vs. spec defects
- Spec gaps (R5_DESIGN.md, text only — no code supplied, so no code bugs identified): `-transform` ident-vs-species (C7); own `ability`+`baseAbility` (C9); `<none>` rule (C16); `[from]/[of]` rule (C12); `0 fnt`/ceil semantics (C14); own-request-never-fills-opponent rule (C8); Struggle requests have 1 move (`schema.json:368-369`) so "four advertised moves" (`R5:16-17`) needs `<none>` padding.
- Not in supplied evidence: `battle.add` Pokémon serialisation (F8 is high-confidence inference from `toString`); `setSpecies`/`setAbility` message emission; `details` after Transform; Transform revert on switch; `baseStoredStats` under forme change; `gen9randombattle` rules (Species Clause, `illusionlevelmod`, `reportExactHP`, Team Preview `-*` `SIM-PROTOCOL:215-219`); faint ordering (T11); whether `[silent]` lines reach the viewer's log.

## Limitations
I read only the five supplied files; no code was executed and no fixtures were generated. R5 implementation code was not supplied, so findings are specification-level requirements and tests, not verified defects. Inferences are marked as such; everything else cites the lines above.
