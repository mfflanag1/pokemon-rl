**Static review only** — I have no tools and did not execute anything; all conclusions come from reading the supplied text. The `checks.json`/stderr showing 5 tests passing is treated as unverified evidence (it's consistent with my reading, but I can't confirm it ran).

## Findings

**Scope / ownership**
- Patch touches only `python/pokemon_switch_screen/statistics.py`. No new files, no test/contract edits visible. ✔

**Input validation vs. contract**
- Shape: `ndim != 2 or shape[1:] != (2,)` rejects `[]`, `[1,0]`; N bounded 2..100000 rejects `[[1,0]]` and the 100001-row case. ✔
- Dtype: `kind not in "iu"` rejects bool (`b`), float (`f`), complex (`c`), object (`O`). ✔
- Values: `np.isin(array, (-1,0,1)).all()` rejects `[[2,0],[0,1]]`. ✔ (Unsigned dtypes simply can't hold -1; harmless.)
- Ragged input: modern NumPy raises ValueError from `np.asarray`, which is caught and re-raised. ✔
- Mismatched shapes rejected after both validated. ✔
- `alpha`: bool/np.bool_ rejected first; `'0.05'` fails `numbers.Real`; `10**1000` is a Real int → `math.isfinite` raises OverflowError → caught → ValueError; nan/inf/0/1/-1 fail `isfinite and 0<alpha<1`. ✔ Minor: `Decimal` is rejected (not `numbers.Real`); contract says "finite real", so this is defensible.
- `family_size`: bool rejected, `1.0` and `'6'` rejected (not int/np.integer), 0/-1/10001 out of range. ✔

**Formulas (hand-traced against test 1)**
- candidate `[[1,-1],[0,1]]`: win mean 2/4=0.5, not-loss 3/4=0.75; baseline `[[-1,0],[1,0]]`: 0.25 / 0.5. ✔
- secured = mean(candwin−basewin) = (1+0−1+1)/4 = 0.25; worst = mean(candwin − base≠0) = (0+0−1+1)/4 = 0; best = mean(cand≠0 − basewin) = (1+1−1+1)/4 = 0.5. ✔
- `astype(np.int8) - bool_array` yields int8 in {−1,0,1}; `.mean()` promotes to float64. ✔
- Mean over the full [N,2] array equals the mean of complete pair means, satisfying "means of complete two-seat pairs". Cutoffs are never recoded as draws; no bootstrap/p-values. ✔

**Radius**
- `sqrt((log(family) − log(alpha)) / (2N))` computed as a difference of logs, not `log(family/alpha)`, so `alpha=5e-324, family=10000` stays finite (log(5e-324) ≈ −744.4, radius ≈ 11.2, then clamped to 0 / −1). Test 3 reduces to `sqrt(log(120)/(2·576))` as expected. ✔
- `adjusted_improvement_lower = max(-1, worst − 2·radius)` — correct for range-2 Hoeffding. ✔

**Output**
- Exact 14 keys present, all Python `int`/`float`, all finite (radius finite whenever alpha is a positive finite float; clamps bound the adjusted values). `json.dumps(..., allow_nan=False)` will succeed. ✔
- `test_symmetry_and_extremes`: seat reversal doesn't change means, so `r == s` holds exactly (same float ops on same multisets). ✔

**Nits (non-blocking)**
- `result.md` says the author "did not run or claim acceptance checks," yet a `checks.json` with a passing run is attached; presumably harness-generated. Not a defect, just note the provenance is the harness, not the author.
- `float(max(0.0, ...))` wrappers are redundant but harmless.

## Verdict: **accept**

Implementation matches the contract's shape/dtype/value/parameter rejection rules, the cutoff-bound formulas, the pair-count Hoeffding radius with underflow-safe logs, and the output key set; it stays within the owned file. The supplied test transcript (5/5 OK) is consistent with my static trace of every test case.
