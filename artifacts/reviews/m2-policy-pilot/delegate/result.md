## Result

Wrote `python/pokemon_pilot/statistics.py` (the only path touched) by copying the candidate's behavior and applying a bounded repair for the two rejection causes.

**Repairs**

1. **Huge-integer alpha → `ValueError`** (`_validate_alpha`): `float(10**1000)` raised `OverflowError`; the conversion is now wrapped and re-raised as `ValueError`. The same guard was added to the `astype(np.float64)` coercion in `_as_numeric_array` for consistency.
2. **Calibration overflow fails closed** (`calibration`): the unclipped `(vals - outs) ** 2` / mean is computed under `np.errstate(over="raise", invalid="raise")`, with `FloatingPointError` mapped to `ValueError`. A new `_finite_float` helper additionally rejects any `inf`/`nan` scalar before it reaches the result dict, so `calibration([1e308],[0])` raises instead of returning `Infinity`. Raw MSE remains unclipped; clipped Brier, out-of-range rate and ten-bin semantics are unchanged.

**Preserved**: pair-is-unit averaging before resampling, identical index rows for candidate means and paired differences, complete-pair units, `np.random.default_rng(seed)`, 64-replicate chunks, all validation rules, public API (`paired_summary`, `calibration`), and JSON-native output shape. `_finite_float` is also applied to bootstrap outputs (`mean`, `lower`, `upper`), which is a no-op for admitted score data but enforces the "finite or ValueError" guarantee. No new dependencies.

**Limitations**

- I did not run the frozen tests (no execution tool available); the controller's acceptance run is the verification. Mapping to tests: test 6 is covered by repairs 1 and 2; tests 1–5 exercise unchanged candidate logic.
- Verification suggestion: `python -I -m pytest tests/pilot/test_statistics.py -q`, plus a spot check that `calibration([-1.,0.,.5,1.,2.],[0.,0.,1.,1.,1.])['raw_mse']` is still `0.45` (confirms unclipped semantics survived the `errstate` wrapping) and that no `RuntimeWarning` is emitted on that path.
- Out of scope / unresolved: a very large `replicates` can still raise `MemoryError` from `np.empty`, which is not a `ValueError`; the task did not ask for a bound on replicates and none was in the frozen tests, so I left it as-is.
