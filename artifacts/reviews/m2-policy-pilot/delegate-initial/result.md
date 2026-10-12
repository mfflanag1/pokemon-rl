Implementation is complete; no further tool calls are needed. I have no execution tool in this environment, so I have **not** run the tests — the controller's acceptance check will do that. Below is what I wrote and a hand-trace against the frozen tests.

## Result

Only `python/pokemon_pilot/statistics.py` was changed. No new dependencies (numpy + stdlib only, no file IO, no JAX).

### `paired_summary(candidate, baseline=None, *, seed, replicates, alpha)`
- **Validation** (all `ValueError`): candidate coerced via `np.asarray`, dtype kind must be `i/u/f` (rejects bool/object/complex), non-empty, finite, shape `[N,2]` with `N>=2`, entries exactly in `{0, .5, 1}` via `np.isin`. Baseline, if given, same checks plus shape equality. `seed`: non-bool int in `[0, 2**32-1]`. `replicates`: non-bool int `>=1`. `alpha`: non-bool real, finite, `0<alpha<1`.
- **Pair clustering**: `pair_means = cand.mean(axis=1)` *before* any resampling; `pair_diffs = pair_means - base.mean(axis=1)`.
- **Bootstrap**: `np.random.default_rng(seed)`, `rng.integers(0, N, size=(chunk, N))` with chunks of ≤64 replicates (bounded index memory); the same index rows are applied to both `pair_means` and `pair_diffs`.
- **Interval**: `np.quantile(samples, [alpha/2, 1-alpha/2], method="linear")`.
- **Output**: JSON-native `pairs, games(=2N), wins, draws, losses, mean, lower, upper` plus `improvement={mean,lower,upper}` when baseline supplied; all `int`/`float`.

### `calibration(values, outcomes)`
- Both non-empty, 1-D, finite real numeric, equal length; outcomes in `{0,.5,1}`.
- `raw_mse` on **unclipped** predictions; `clipped_brier` after `np.clip(vals,0,1)`; `out_of_range_rate` = fraction strictly outside `[0,1]`.
- 10 bins, `bin = min(floor(clipped*10), 9)`; each `{index, count, mean_prediction, mean_outcome}` with `None` means for empty bins.

### Hand-trace of the five frozen tests (reasoning, not executed)
1. `tile([0,1])` → all pair means 0.5 → every replicate 0.5 → `(mean,lower,upper)=(.5,.5,.5)`, 20 wins/20 losses, 20 pairs/40 games.
2. Pair means `{0,1,.5}` vary → `lower<upper`; diffs all 0 → `improvement={0.,0.,0.}`; same seed → identical dict for `x` vs `x.copy()`.
3. ones vs zeros → diffs all 1 → `{1.,1.,1.}`; `full(.5)` → 24 draws, mean .5.
4. Each malformed input hits a distinct `ValueError` branch (empty, 1-D, N<2, wrong width, NaN, 0.1, bool dtype, seed −1/True, replicates 0/True, alpha 0/1/NaN, baseline shape mismatch).
5. `raw_mse=2.25/5=.45`, `clipped_brier=.25/5=.05`, `out_of_range_rate=.4`, bins 0 and 9 each count 2, bin 5 count 1, bin 1 means `None`; the four bad inputs raise.

## Limitations / unresolved
- **Not executed**: no shell/test tool was available to me, so passing is inferred from the trace above, not verified.
- **`mean_prediction` interpretation**: the spec says "clip predictions only for Brier and 10 bins"; I took that to mean the bin's `mean_prediction` is the mean of the *clipped* predictions (so it lies within the bin's range). The frozen test only checks that an empty bin yields `None`, so either reading passes; flag if raw-value means were intended.
- `seed`/`replicates` accept `numpy.integer` as well as Python `int` (both rejecting bool/`np.bool_`); the spec didn't say whether numpy scalars should be allowed.
