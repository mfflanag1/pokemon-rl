Verdict: **revise**

Concrete finding:

- `replicates` accepts any positive integer, but extremely large values can make `np.empty(replicates, ...)` raise `OverflowError`, `ValueError`, or `MemoryError`. In particular, `MemoryError` violates the stated guarantee that admitted inputs either produce finite JSON-native output or raise `ValueError`. The result’s own limitations section acknowledges this gap. Add bounded validation or translate allocation failures to `ValueError`.

Otherwise, the supplied evidence supports the requested behavior:

- All six frozen tests pass.
- Huge integer `alpha` is correctly translated to `ValueError`.
- Calibration overflow is rejected as `ValueError`.
- Raw MSE remains unclipped while Brier remains clipped.
- Pair-level resampling, shared bootstrap indices, fixed RNG, and 64-replicate chunks are preserved.
- The patch shows only `python/pokemon_pilot/statistics.py` changed.