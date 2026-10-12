Verdict: **revise**

The candidate is strong and mostly aligned with the conservative appearance-based design, but two consequential semantic errors and several overclaims prevent acceptance.

- **Incorrect HP interval for `100/100`.** C14 says `N=100` represents `((N−1)/100, N/100]`. Under `pokemon.ts:2077-2081`, every non-full value that rounds to 100 is changed to 99. Therefore, in percentage mode, `100/100` means **exactly full HP**, not `(99%,100%]`. `99/100` represents `(98%,100%)`, excluding full. The handling requirement to retain the raw numerator and denominator is correct, but the interval explanation and a corresponding boundary assertion must be fixed.

- **`-transform` field interpretation is not established.** C7 mandates parsing the third field as a Pokémon ident. The implementation passes a Pokémon object (`pokemon.ts:1344-1346`), but the supplied excerpts do not define `battle.add` serialization. The protocol contract explicitly documents the field as `SPECIES` (`SIM-PROTOCOL.md:533-536`). This must be classified as **missing/conflicting evidence**, not an exact requirement. A compliant encoder should retain the raw field and only interpret ident-versus-species after a pinned serialization fixture or source excerpt establishes it.

- **T6 inherits the unsupported Transform assumption.** Resolving the target “by slot” and updating display species is actionable only if the emitted field is proven to be an ident. The bounded test should accept the pinned observed serialization form, then test conservative handling; it should not bake in C7 before that evidence exists.

- **Item/ability certainty is stated too absolutely.** C16 says current ability is “never certain” because Skill Swap may be silent. The cited exception concerns Skill Swap between allies in Double or Triple Battles (`SIM-PROTOCOL.md:516-518`), while the supplied contract pins `gen9randombattle` singles (`schema.json`, `reference_manifest.format`). An explicit `-ability` establishes the ability at that event (`SIM-PROTOCOL.md:512-523`). The safe requirement is: record the value and observation time as certain-at-observation, but do not promise persistence after later potentially changing events.

- **Likewise, `<none>` needs temporal qualification.** `-enditem` establishes no item at that event (`SIM-PROTOCOL.md:494-505`), but later ownership-changing events could invalidate a current-state claim. The encoder should distinguish `last_observed_none` from guaranteed-current none unless subsequent public history preserves that conclusion.

- **Illusion faint handling exceeds the evidence.** C5 correctly labels the damage/faint name transition as an inference, but “count only `faint` events” is not supported by the supplied excerpts because the `faint` command semantics and ordering are absent. T11 is the correct disposition: unresolved fixture gathering. The roster-count recommendation should be softened to “do not infer distinct roster members from apparent species or nicknames.”

- **“Same denominator” on `replace` is not directly established.** The same underlying Pokémon and HP are supported by `pokemon.ts:544-553` and `abilities.ts:2077-2083`, but the exact serialized `replace` health field depends on omitted `battle.add` serialization. Require HP continuity at the semantic level while retaining the emitted condition verbatim.

- **The review occasionally blurs requirements and optional recommendations.** `same_ident_details_as_prev_appearance`, dropped-appearance counts, and retaining structured pre-replace details are useful recommendations, but R5 only strictly requires preserved history, bounded windows, and truncation flags. They should not be presented as source-mandated behavior.

What is solid and should remain:

- Appearance-scoped evidence and no merging across repeated switches.
- No private truth used to identify opponent roster slots.
- `replace` updates the current appearance rather than opening a new one.
- Transform does not copy HP, item, status, level, or `baseStoredStats`; own request `stats` are `baseStoredStats` (`pokemon.ts:1153-1180`, `1270-1352`).
- `[from]` and `[of]` must not independently assign item or ability ownership (`SIM-PROTOCOL.md:232-237`).
- Opponent HP should be stored in its displayed fraction form.
- Paired perturbation tests must exclude cases where Transform changes the viewer’s private request.
- The tests are clearly described as adapter fixtures rather than claims of live execution.

Required revision: correct the HP mathematics, downgrade `-transform` serialization to unresolved evidence, qualify certainty temporally, and separate optional encoder recommendations from requirements compelled by the pinned sources.