Implementation complete. I've reviewed the file against the contract and the test harness line by line; I have no execution tool in this environment, so I did not run the parity tests — the controller's acceptance run is the verification.

## Result

Created `python/pokemon_jax_learner/core.py` (only file touched). Public API: `require`, `finite`, `distribution`, `distribution_kernel`, `linked_gae`, `loss_kernel`, `ppo_loss_grad`, `Adam`.

**Numerics (all JAX, FP32, CPU):**
- `distribution_kernel` — pure masked softmax; row maxima under `stop_gradient` and every mask applied via `jnp.where` (select), so illegal logits get exactly-zero probability, zero log sentinel, and exactly-zero gradient (no `-inf`/NaN leakage, no max-tie splitting). `distribution` = host checks + jitted kernel + output finiteness check.
- `linked_gae` — vectorized host checks replicating all reference `require`s (forward/boundary linkage, terminal ⇒ `next=-1 & bootstrap=0`, internal link ⇒ same trajectory & bootstrap 0), then a jitted `lax.scan` in reverse over a full-advantage carry (needed since links jump to arbitrary forward `j`, not `i+1`). `F32(gamma)` and `F32(gamma*lam)` rounded on host like the reference.
- `loss_kernel` / `_loss` — reference arithmetic order; targets stop-gradient; seven metrics. PPO clip boundary uses the explicit inclusive branch `active = where(A>=0, ratio<=1+clip, ratio>=1-clip)` selecting `ratio` vs `stop_gradient(clipped)`: forward value is provably identical to `min(ratio·A, clipped·A)` and the gradient matches the reference's `active` mask (including equality at ties) rather than autodiff's averaged tie gradient. Clip bounds are host-rounded `F32(1±clip)` when given Python scalars.
- `ppo_loss_grad` — full reference checks (FP32 vectors, action index range, known-forbidden action, coefficients, finiteness), then `jit(value_and_grad(argnums=(0,1), has_aux=True))`; returns Python-float metrics plus JAX gradient arrays; rejects nonfinite loss/ratio/gradients.
- `Adam` — fields `lr, max_norm, step, m, v`; keys/shape/dtype/state-shape/finiteness checked before anything mutates; `float32(1-.9**step)`, `float32(1-.999**step)` computed on host and passed into a single jitted kernel doing FP32 global-norm clipping and the update in reference order; output finiteness is checked *before* assigning, so a bad step leaves `step`, `m`, `v` and the parameter mapping untouched. Returns pre-clip norm as a float.

## Limitations / notes
- **Not executed**: no shell tool was available, so the supplied `tests/jax-learner/test_core.py` were not run by me. The code was checked by inspection against each test case (including the trajectory-mismatch GAE rejection, forbidden-action rejection, FP64-logits rejection, and the NaN-gradient no-mutation case).
- Expected tolerable deviations (per contract): Adam's global norm is an FP32 JAX reduction vs the reference's FP64 — ~1e-7 relative, within the 3e-6/3e-5 test tolerances. Reference raises `FloatingPointError` on `exp` overflow; this port raises `ValueError` via the nonfinite-output check.
- `_loss`'s clip constants are only bit-identical to the reference's `F32(1±clip)` when `clip` is a Python/NumPy scalar (always true via `ppo_loss_grad`); if a caller traces `clip` through `loss_kernel` under their own `jit`, the bounds are computed in FP32 arithmetic instead.
- JIT recompiles per distinct batch shape / parameter-tree shape (standard JAX behavior); no global `jax.config` mutation is done, so CPU/x64-off is assumed from the environment rather than enforced.
