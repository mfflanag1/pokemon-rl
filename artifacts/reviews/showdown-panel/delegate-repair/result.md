## Result

Both owned files were changed; nothing else was touched.

### `packages/showdown-panel/statistics.mjs` (1 line, line 205)
- Duplicate-ID message changed from `policyIds contains duplicate id "p"` → `policyIds contains duplicate policy id "p"`, so the existing assertion `expectRejected(..., ['p','p'], /duplicate policy id/)` at test line 451 now matches. No algorithm, validation ordering, or other message was altered.

### `tests/showdown-panel/statistics.test.mjs` (new section inserted before "determinism")
Three literal known-answer tests, using only the coordinator-supplied figures (no second bootstrap implementation; the existing `oracleSummary` test is untouched and still runs):
1. **`mulberry32(0)` first 15 words** — frozen as `MULBERRY32_SEED0_WORDS`, `deepEqual` against the generator.
2. **`uniformIndex(next, 3)` on that stream** — frozen as `SEED0_INDICES_MOD3`; also asserts every word is `< 2^32 − 1` (so no rejections, exactly 15 words consumed) and `word % 3` equals the frozen index.
3. **`summarizeMatrix` known answer** — 6 explicit records (both mirrors per pair; pair 1 is split 1/0 across mirrors), `pairs: 3, bootstrapSamples: 5, seed: 0`. Asserts wins/draws/losses = 3/0/3, `pair_scores` `[0, 0.5, 1]`, and `interval95` **exactly** `[1/3, 2/3]`, with the per-replicate derivation documented in comments.

### Consistency check I performed (by hand, not by running code)
- Each supplied word's digit-sum mod 3 matches the supplied index (all 15), and max word 3880206403 < 4294967295, so the rejection claim holds.
- Walking the indices over `[0, 0.5, 1]` gives means `[2/3, 1/3, 2/3, 1/3, 1/2]`; with k = ⌈5/40⌉ = 1 the interval is `[1/3, 2/3]`. The implementation computes `sum / pairs` as `2/3` and `1/3` via the same IEEE divisions as the literal `2 / 3` / `1 / 3`, so exact `deepEqual` is appropriate.

### Limitations
- I have no shell tool in this session, so I did **not** run the test suite; the controller's acceptance run is the verification.
- I could not independently recompute the mulberry32 words themselves (the `Math.imul` steps are impractical by hand); I verified only their internal consistency with the stated indices/interval. If word 1 (1144304738) were wrong, tests 1–2 would fail while test 3 would still pass only if the mod-3 indices happen to coincide.
