Implementation is complete; nothing further is needed from other tools (I have no shell in this environment, so I did not run the tests — the controller will).

## Result

Created `python/pokemon_fixed/math.py` (only file touched). It imports the sealed kernels (`distribution_kernel`, `_adam_kernel`) from `pokemon_jax_learner.core` without modifying them.

**`masked_loss(logits, values, mask, actions, old_logp, advantages, returns, valid, clip=.2, value_coef=.5, entropy_coef=.01)` → `(loss, metrics)`**
- Sanitizes all eight inputs with `jnp.where(valid, …)` before any arithmetic: zeros for logits/values/targets, all‑true mask, action 0. NaN payloads and `-1` actions on padded rows never reach `exp`/`log`/gather.
- Replicates `core._loss` exactly on valid rows: same `distribution_kernel`, same inclusive `active` branch with `stop_gradient(clipped)`, half‑MSE, entropy bonus, stop‑gradient targets, host‑rounded `F32(1±clip)` constants.
- Reductions are `sum(where(valid, rows, 0)) / max(count, 1)` and `max(where(valid, error, 0))` (error ≥ 0, so padding never raises the max). All seven reference keys returned.
- Invalid-row gradients are exactly 0 because both selections (input sanitization, per‑row re‑masking) have select‑type cotangents; all-invalid gives count=1 → every metric 0 and zero gradients.

**`bias_correction(step)`** → shape‑(2,) FP32. Table `np.array([[1-.9**s, 1-.999**s] …], np.float32)` for s=1..20000 (the same Python‑double → FP32 rounding path the test and sealed `Adam.apply` use), gather by `clip(step-1, 0, 19999)`, `where(step > 20000, ones, table)`. Steps > 20000 are exactly 1.0 in FP32 (0.999**step < 2⁻²⁵ for step ≥ 17323), so the ones branch is consistent with host rounding. Non‑positive steps clamp to the step‑1 entry (documented; outside contract).

**`adam_step(parameters, gradients, m, v, step, lr=.003, max_norm=1.)`** → `(new_p, new_m, new_v, step+1, preclip_norm)`. `new_step = int32(step)+1`, bias from `bias_correction(new_step)`, then sealed `_adam_kernel` with `jnp.asarray(lr/max_norm, float32)` (same host rounding as `F32(self.lr)`). Fresh dicts, no mutation, no host checks → jit/scan compatible.

## Limitations / unverified
- I could not execute `tests/fixed-loop/test_math.py` (no shell tool); correctness is by inspection against `core.py`. Specific things the acceptance run will confirm: (a) reduction‑order drift between `jnp.mean` over compacted rows vs. masked sum over 16 padded rows stays within atol 3e‑6/rtol 3e‑5; (b) the borderline step 17320 table entry matches bit‑for‑bit (it should — identical Python expression and rounding path on the same host).
- `max_ratio_error` still propagates NaN from a *valid* row (reference behaviour preserved); only invalid rows are sanitized, by design.
- `clip`/`value_coef`/`entropy_coef` are expected as Python floats; passing traced FP32 scalars would compute `1±clip` in FP32 rather than host‑rounded double→FP32 (documented in the docstring).
