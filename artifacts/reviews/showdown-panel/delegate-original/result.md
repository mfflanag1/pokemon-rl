Both owned files are now written; nothing further depends on another tool result, so there is nothing more to request.

## Result

**Files changed (only the two owned paths):**
- `packages/showdown-panel/statistics.mjs`
- `tests/showdown-panel/statistics.test.mjs`

### Implementation (`statistics.mjs`)
- `summarizeMatrix(records, {policyIds, pairs, bootstrapSamples=10000, seed=73291})` — pure, dependency-free; returns one entry per edge in lexicographic `(i, j)` policyIds-index order with `{a,b,games,pairs,wins,draws,losses,mean_score,pair_scores,interval95:[low,high]}`.
- **Validation:** options (`policyIds` ≥2 unique non-empty strings; `pairs`/`bootstrapSamples` positive integers; `seed` integer in `[0, 2³²−1]`); records (object shape, known policy ids, `a≠b`, integer `pair` in range, `mirror` ∈ {0,1}, `score` ∈ {0,0.5,1} — NaN/±Infinity/0.3/strings rejected); exact count, duplicate cell (including same game in reversed orientation), and explicit missing-cell check. All throw `MatrixValidationError`. Inputs never mutated (options copied, records read only).
- **Orientation:** a record with `a` later than `b` in `policyIds` is normalised to canonical order with `1−score` (lossless since score is defined as `a`'s outcome). This is a design choice I made under ambiguity — documented in the header.
- **Bootstrap:** whole-pair resampling (pair score = average of both mirrors), a single index vector per replicate shared across all edges, `mulberry32` uint32 PRNG written out explicitly with unbiased rejection sampling, documented draw order.
- **Quantiles:** `k = ⌈B/40⌉` computed in integer arithmetic; `low` = k-th smallest, `high` = k-th largest (symmetric nearest-rank; documented as differing from plain ⌈0.975B⌉ by ≤1 rank, chosen so complementing scores mirrors the interval exactly).
- Header documents: descriptive-only intervals, per-edge (no simultaneous/multiplicity coverage), no promotion interpretation, when degenerate intervals occur and what they do/don't mean.
- Extra named exports for testability: `mulberry32`, `uniformIndex`, `percentileInterval95`, `MatrixValidationError`.

### Tests (`node --test`, `node:test` + `node:assert/strict`, hand-derived expectations)
Exact W/D/L + `mean_score` + `pair_scores` accounting and edge ordering; mirror-1 perspective and pair-as-unit (6 wins/6 losses but every pair 0.5 ⇒ degenerate `[0.5,0.5]`); single-pair degeneracy; `[0,1]` case at default B; shared indices (identical edge ⇒ identical interval, complemented edge ⇒ exactly mirrored interval); an in-test oracle re-deriving the whole procedure from the documented spec (canonical float-state mulberry32, counting-based order statistics) on a nontrivial 3-edge × 5-pair × 300-replicate case; determinism/defaults/fresh outputs; PRNG reproducibility and seed sensitivity; `uniformIndex` rejection boundary (`2³² mod 3 = 1`, hand-checked); `percentileInterval95` known answers for B = 5, 40, 41, 80, 81, 10000; reversed/mixed orientation equivalence; ~35 malformed-grid cases and ~25 invalid-option cases; deep-frozen input preservation.

## Limitations
- **I had no execution tool, so nothing has been run.** All expected values were derived by hand; the controller's acceptance run is the first execution. The highest-risk items if something is off: the quantile rank arithmetic (double-checked by hand), and the oracle/implementation PRNG equivalence (float-state vs uint32-state forms — I reasoned these are identical mod 2³² via ToInt32/ToUint32 wrapping).
- Two assertions are probabilistic rather than exact (`[0,1]` with B=10000 and several "non-degenerate interval" checks); for a working PRNG their failure probability is astronomically small, but they are not hand-exact.
- The reversed-orientation acceptance rule is my interpretation of "unordered pair"; if the runner's convention differs (e.g. reversed records should be rejected outright), that is a one-line change plus one test edit.
- No existing repo test harness was present in the isolated workspace (`scripts/test.mjs` absent), so I cannot confirm how the project discovers tests; the file is runnable directly with `node --test tests/showdown-panel/statistics.test.mjs`.
