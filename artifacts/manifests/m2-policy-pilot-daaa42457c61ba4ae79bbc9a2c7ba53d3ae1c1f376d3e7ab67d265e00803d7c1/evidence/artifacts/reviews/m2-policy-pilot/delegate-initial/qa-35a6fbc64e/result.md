Verdict: **accept**

No blocking findings.

- `paired_summary` correctly averages seats before resampling, bootstraps complete pairs, and computes paired candidate-minus-baseline differences using the same sampled indices.
- Bootstrap generation is deterministic and limits index allocation to 64 replicates per chunk.
- Validation rejects malformed shapes, booleans, nonfinite values, invalid scores, invalid seeds, replicate counts, and alpha values with `ValueError`.
- Quantiles explicitly use NumPy’s linear method.
- Returned values are JSON-native.
- `calibration` correctly distinguishes raw MSE from clipped Brier score, assigns clipped predictions to ten bins, handles the upper endpoint in bin 9, and returns `None` for empty-bin means.
- The patch changes only `python/pokemon_pilot/statistics.py` and adds no dependency beyond NumPy.
- The supplied isolated-source test command completed successfully: all five frozen tests passed.

Minor non-blocking note: accepting NumPy integer scalars for `seed` and `replicates` is a reasonable interpretation of “integer.”