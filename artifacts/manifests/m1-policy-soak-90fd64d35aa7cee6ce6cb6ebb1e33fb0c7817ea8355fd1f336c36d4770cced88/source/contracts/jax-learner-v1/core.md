# Core API and parity v1

Implement ONLY `python/pokemon_jax_learner/core.py`, using JAX 0.11.2 and NumPy
for host validation (no NumPy numerical learner fallback). Do not edit sealed
`pokemon_learner` or `pokemon_jax`. CPU only, x64 disabled, FP32 arrays.

Match `pokemon_learner.core` math and public argument signatures:
- `distribution(logits, mask)` checked wrapper returns JAX arrays (p, logp).
  Expose `distribution_kernel(logits,mask)` pure JAX for a compiled actor.
- `linked_gae(rewards,values,next_index,bootstrap,terminal,trajectory,*,gamma=1.,lam=.95)`
  checked wrapper, JAX reverse scan. Same-player forward linkage; no alternating
  signs. Terminals next=-1/bootstrap0; cuts next=-1 use own bootstrap; internal
  bootstrap0 and same trajectory. n>0. Return FP32 JAX arrays.
- `loss_kernel(logits,values,mask,actions,old_logp,advantages,returns,clip,value_coef,entropy_coef)`
  pure differentiable JAX, return (scalar_loss, metrics_dict) with the same seven
  metrics as NumPy. Value half MSE; no value clipping, advantage normalization,
  exploration penalty or optimizer weight decay. All targets stop-gradient.
- `ppo_loss_grad` same checked signature as NumPy; use JAX value_and_grad, return
  metrics (Python floats), grad_logits, grad_values. May expose compiled kernel.
  IMPORTANT exact PPO clip boundaries: NumPy includes equality on the unclipped
  side (`A>=0: ratio<=1+clip`, `A<0: ratio>=1-clip`). Naive min/clip autodiff
  averages gradients at ties; choose an explicit inclusive branch to match reference.
  Illegal probabilities/logits gradients exactly zero, finite log sentinel zero.
- `Adam(parameters,lr=.03,max_norm=1.)` fields lr,max_norm,step,m,v and apply method
  matching reference. State m/v JAX arrays, Python integer step. apply mutates the
  *mapping* by replacing immutable JAX arrays, returns pre-clip norm float. Check
  shape/dtype/key/finiteness before mutation, no partially applied bad update.
  All numerical updates/gradient clipping JAX; FP32 global norm may differ from
  NumPy's FP64 norm reduction within declared tolerance. Compute bias-correction
  scalar constants at host as float32(1-.9**step), float32(1-.999**step), matching
  the reference, then pass to JIT. No common jnp.power FP32 cancellation variant.

Checks must reject illegal actions, empty masks, invalid indices/links, nonfinite
input/output, invalid coefficients or optimizer state; don't silently coerce policy
FP64 or wrong-shaped data. Raw GAE numeric lists may convert to FP32 as in reference.
Expose `require` if convenient. Tests import only public API above.
