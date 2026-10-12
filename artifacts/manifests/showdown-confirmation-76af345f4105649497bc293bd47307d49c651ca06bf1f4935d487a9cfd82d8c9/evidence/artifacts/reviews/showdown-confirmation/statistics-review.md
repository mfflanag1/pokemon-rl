**Provisional verdict: no concrete bugs found in reviewed scope.** Most important point: I hand-verified the literal seed-0 fixture arithmetic against the stated index sequence and all three orientation/delta expectations hold; the only residual risk is low-severity floating-point fragility in the Bonferroni rank computation.

## Findings

**1. Low — `statistics.mjs:19` rank uses floating-point `ceil(samples*alpha/comparisons)`.**
Evidence: `Math.ceil(samples * alpha / comparisons)`. For the three cases that matter I checked rounding by hand: B=30000 → 30000·0.05 rounds to exactly 1500, /3 = 500; B=60 → 3/3 = 1; B=5 → 0.25/3 → ceil 1. All correct. But other B values could produce `k + ε` and overshoot by one rank (the panel module explicitly avoids this at line 175 "without floating-point"). Not a bug for the frozen plan; a fragility if B is ever changed.
Verification: `for B in 1..100000 assert Math.ceil(B*.05/3) === Math.ceil(B/60)` (integer form `Math.floor((B+59)/60)`).

**2. Orientation and deltas — verified correct.** `statistics.mjs:40-42`: canonical scores are `a`'s view (summarizeMatrix contract, lines 21-28), IDS order baseline(0), random(1), damage(2), tactical(3). Direct = `1-score(0,3)`; random delta = `(1-score(1,3)) - score(0,1)`; damage delta = `(1-score(2,3)) - score(0,2)`. Matches plan metrics. Fixture recomputation with `edgePairs` and indices `[2,1,1][0,0,2][2,2,0][0,2,0][1,0,2]`: direct replicates {2/3,1/3,2/3,1/3,1/2} → min 1/3; random deltas [.5,0,-.5] → {-1/6,1/6,-1/6,1/6,0} → min -1/6; damage deltas [-1,0,1] → {1/3,-1/3,1/3,-1/3,0} → min -1/3. All match `decision-fixture.test.mjs:21`. Float equality holds (−0.5/3 and −1/6 are both correctly rounded, dyadic sums exact).

**3. Thresholds — consistent with plan.** `lower > threshold` strict (line 55), `mean >= minimum` inclusive (line 57); tests at `statistics.test.mjs:18-23` pin both boundaries. With 1000 pairs, means are multiples of 1/4000, so a replicate exactly at .5 or −30/1000 compares equal to the literal and fails, as intended.

**4. Sort — correct.** `Float64Array.prototype.sort()` with no comparator sorts numerically (unlike `Array`), so line 54 is right; in-place mutation is of an internal buffer only.

**5. Validation — adequate.** `Array.from` turns holes into `undefined` → rejected; `j > i` rejects reversed/foreign edges; six unique canonical keys among 4 ids imply completeness (C(4,2)=6). Guard at line 20 cannot fail for valid alpha (ceil of a positive number ≥ 1); harmless.

**6. Nit — test name overclaims.** `statistics.test.mjs:12` says "without treating games as independent" but asserts nothing about resampling unit; whole-pair resampling is only verified indirectly via the fixture. Not a correctness issue.

**7. Note — `confirm` does not check `pairs === plan.pairs`.** Caller must enforce 1000 pairs; module accepts any positive count.

## Scope limitations
- Cannot execute; the claim that seed-0 mulberry32 indices are `[2,1,1,0,0,2,2,2,0,0,2,0,1,0,2]` is taken as given (stated as independently established). Everything downstream of that sequence is verified by hand.
- Fresh confirmation run, freshness/overlap admission, and game-generation code are out of scope.
- Claims stay appropriately approximate: interpretation strings (line 61, plan line 50) disclaim finite-sample/final guarantees.

## Verdict
No findings of concrete bugs in reviewed scope; one low-severity floating-point fragility (Finding 1) worth hardening to integer arithmetic before B is ever varied.
