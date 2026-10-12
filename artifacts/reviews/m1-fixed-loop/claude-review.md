**Provisional verdict: findings (no numerical correctness blocker located; one telemetry claim is unverified and likely vacuous).**

## Findings

**1. Medium–High — `unexpected_compilations` counter may never observe real XLA compiles (python/pokemon_fixed/campaign.py:48-53, 190, 208, 230; tests/fixed-loop/test_campaign.py:31-35).**
`CompileLog` only matches messages starting with `'Compiling jit('` on logger `jax._src.interpreters.pxla`. The real `jax_log_compiles` message in `pxla.lower_sharding_computation` is formatted as `"Compiling %s with global shapes and types ..."` with `fun_name` = the Python function name (e.g. `wrapped`), not `jit(wrapped)` — and in some JAX versions it reads `"Compiling %s for with global shapes..."`. If the prefix never matches, `baseline == count == 0` for the whole campaign and the "no unexpected compilation" guarantees in `profile`/`soak` are vacuously satisfied. The unit test only feeds synthetic strings it invented itself; it never asserts that a genuine first-call `jax.jit(f)(x)` increments the counter. **Verification:** add a positive control after warmup (fresh `jax.jit(lambda x: x+1)(jnp.ones(3))` must raise `count` by ≥1, then re-baseline), or match on the logger record rather than a string prefix. Until then the `unexpected_compilations: 0` fields should not be cited as evidence. (I cannot confirm the exact message for the installed JAX version from the packet.)

**2. Medium — Profile-phase latency/throughput is measured with cProfile enabled (campaign.py:172-176).**
`Aggregate.add` timings for the 1000 profile cycles are taken inside `profiler.enable()`, so `complete_cycle_seconds`, `decisions_per_cycle_second` and the latency quantiles in `profile.json` include cProfile's per-call host overhead. `host_profiler_enabled: True` is recorded, but nothing labels these numbers as profiler-inflated; the soak numbers (campaign.py:206) are the clean ones. **Verification:** either time outside the profiler, or add an explicit `timing_scope` caveat to the profile report.

**3. Low — Checkpoint invariants rely on an engine-derived bound that is only implicit (checkpoint.py:108-109).**
`4*updates ≤ step ≤ 8*updates` and `256*updates ≤ decisions` assume every game lasts ≥2 turns. That holds (max hit = `damage(crit, 100)` = 59 < 105, engine.py:40-42), but the bound is undocumented and `check_report` uses the weaker `128 ≤ decisions` (loop.py:236), so a checkpoint written from a passing report could in principle fail to reload. Not reachable with the current engine; document or align the two bounds.

**Reviewed and found consistent (no finding):** padding sanitisation before `forward` and inside `masked_loss` (loop.py:143; math.py:61-64), exact-zero padded cotangents (test_math.py:30, test_loop.py:84-89); Adam skipped and step not incremented on empty minibatches (loop.py:158-162); actor sees only pre-step `view_kernel` output, with `terminal/win` zero-checked (loop.py:109-116, 184); time-major `(4,128)` GAE scan is per (game, seat) column with `ended` masking next value/advantage, matching `_gae_kernel`; `next_index = i+128` and `max_link < 512`; both seats sampled from the same pre-step state with frozen parameters; RNG domains ENV/ACT/SHF disjoint and game ids monotone; counter bounds `8·1e7 < 2³¹`, `64·1e7 < 2³²`; hex `<f4` round-trip with `state(runner)==b` is byte-exact; bias table threshold (step 17320 < 20000); bounded `Aggregate`; soak stop and accounting checks.

## Scope limitations
`Comparisons`, `checked_file`, `previous_checks`, `pokemon_slice`, `pokemon_learner.core`, and the predecessor report were not supplied. Tests were not executed; the "10 tests passed" claim is unverified here. Exact JAX version/log format unknown.

## Verdict
Findings. No blocker to the numerical/checkpoint claims from the supplied code; the compile-telemetry claim (Finding 1) needs a positive control before the soak report can cite it as evidence.
