# R3 delegated source audit: verified disposition

2026-10-10. Evidence for the cross-model quality ledger. Preserve this record after
adjudication; later run results belong in R3_REVIEW.md.

Research run:
`local-review-archive/20261010-122038-87adf3a1`.
Fable audited selected pinned generator and Dex source plus a hash-labeled source
table excerpt. Codex performed the required opposite-model QA. The complete
proposed specification is **rejected**, although several verified facts were useful.

Verified independently at Showdown commit
`942cda71d14bf0618bd57991708a2e2734b6daba`:

1. `randomSet` in `data/random-battles/gen9/teams.ts:1498` reads the source template
   using the original Dex species ID. Its return object at `:1624` preserves this
   as `speciesId` while emitting `getForme`'s potentially different display form.
   The R1 `generateTeam` and R2 `teamArtifact` preserve that complete original
   object. The Python pilot verified source-entry membership on all 6,000 Pokémon.
2. The full pinned source JSON has **509 entries and 879 templates**. Six entries
   contain repeated role strings: Forretress, Grumpig, Luvdisc, Necrozma,
   Necrozma-Dusk-Mane and Tatsugiri. Role is not a template identifier. The R3
   catalog preserves original template index and raw-template hash, and coverage
   explicitly distinguishes compatible matches from unique matches.
3. The research's claim that `Dex.species.all()` is a superset of emitted forms was
   false. Independent Python validation of the first 1,000 teams rejected a real
   `florgesblue` because its token was absent. Expanding each Dex species'
   `cosmeticFormes` adds 68 IDs. A regression now checks every resolved cosmetic
   name against the revised vocabulary. The original v1 table remains preserved.
4. At `teams.ts:690`, `addMove` removes the selected move from the pool via
   `fastPop`; `fastPop` at `:291` throws for a negative index. Consequently the
   research's requiredMove-as-an-extra-source claim is not valid for successful
   generation through this branch. Coverage still reports outside-declaration
   observations explicitly rather than hiding them or assuming they cannot occur.
5. The generator's set filtering, weighted base pool, type/weakness restrictions,
   lead handling and other team constraints do not justify uniform final inclusion
   probabilities or guaranteed global reachability for every declared alternative.
   Unobserved alternatives are labeled for investigation, not called unreachable.
6. `randomSet` returns **13 fields**, not the research's stated 12. Schema validation
   and the pilot preserve name/species/speciesId/gender/shiny/level/moves/ability/
   evs/ivs/item/teraType/role. Items, forms and EV/IV adjustments are programmatic.

The bounded source excerpt did not prove full-table counts or all form behavior;
those claims were checked against the local complete source and runtime. No web
master branch was used as pinned evidence. No source review alone is treated as
a proof of initial-set or dynamic battle reachability.
