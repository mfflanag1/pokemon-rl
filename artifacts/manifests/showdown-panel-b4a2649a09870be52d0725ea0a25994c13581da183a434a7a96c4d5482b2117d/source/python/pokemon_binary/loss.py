"""Valid-row PPO with the binary-policy-1 law; sealed loss arithmetic retained."""
import jax
import jax.numpy as jnp
from .policy import distribution_kernel

def masked_loss(logits, values, mask, actions, old_logp, advantages, returns, valid,
                clip=.2, value_coef=.5, entropy_coef=.01):
    """Padded PPO loss -> (scalar_loss, metrics) over the rows flagged by `valid`.

    Shapes (n rows, A actions): `logits` (n, A) FP32, `values` (n,) FP32, `mask` (n, A) bool,
    `actions` (n,) integer, `old_logp`/`advantages`/`returns` (n,) FP32, `valid` (n,) bool.
    `clip`, `value_coef`, `entropy_coef` are Python floats (host-rounded to FP32 like the
    reference's `F32(1 - clip)` / `F32(1 + clip)`); treat them as static under `jax.jit`.

    Reduction and clipping semantics on valid rows follow the sealed
    `pokemon_jax_learner.core._loss`; distribution is binary-policy-1:
    mean clipped actor loss with the explicit inclusive `active` branch (equality on the
    unclipped side keeps the gradient), `value_coef` * half-MSE value loss, minus
    `entropy_coef` * mean entropy; `old_logp`, `advantages`, `returns` are stop-gradient.
    Metrics are the reference's seven keys: `loss`, `actor_loss`, `value_loss`, `entropy`,
    `max_ratio_error` (max over valid rows), `clip_fraction`, `approx_kl` (valid-row means).

    Invalid rows may hold any transport payload (NaN logits/values/targets, action -1,
    all-false mask). Every input is sanitized with `jnp.where(valid, ...)` to neutral finite
    values (zeros; all-true mask; action 0) *before* any arithmetic, and every per-row term is
    re-masked before its reduction, so invalid rows contribute neither value nor gradient:
    both selections have exact-zero cotangents, giving exactly 0 gradient entries on padded
    rows. With no valid rows the divisor is 1, so all seven metrics and both gradients are
    exactly zero. Differentiable in `logits` and `values`; no host checks.
    """
    valid = jnp.asarray(valid, bool)
    valid_rows = valid[:, None]
    zero = jnp.float32(0)
    # Sanitize *all* inputs on invalid rows before any arithmetic touches them.
    logits = jnp.where(valid_rows, logits, zero)
    mask = jnp.where(valid_rows, mask, True)
    actions = jnp.where(valid, actions, 0).astype(jnp.int32)
    values, old_logp, advantages, returns = [jnp.where(valid, a, zero) for a in (values, old_logp, advantages, returns)]
    old_logp, advantages, returns = [jax.lax.stop_gradient(a) for a in (old_logp, advantages, returns)]
    low, high, clip, value_coef, entropy_coef = [
        jnp.asarray(c, jnp.float32) for c in (1 - clip, 1 + clip, clip, value_coef, entropy_coef)]

    count = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), jnp.float32(1))

    def valid_mean(rows):
        return jnp.sum(jnp.where(valid, rows, zero)) / count

    p, logp = distribution_kernel(logits, mask)
    n = logits.shape[0]
    log_ratio = logp[jnp.arange(n), actions] - old_logp
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, low, high)
    # Reference inclusive branch: equality on the unclipped side keeps the gradient.
    active = jnp.where(advantages >= 0, ratio <= high, ratio >= low)
    surrogate = jnp.where(active, ratio, jax.lax.stop_gradient(clipped)) * advantages
    actor_loss = -valid_mean(surrogate)
    entropy = valid_mean(-jnp.sum(p * logp, axis=1))
    value_loss = jnp.float32(.5) * valid_mean(jnp.square(values - returns))
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
    error = jnp.abs(ratio - jnp.float32(1))
    metrics = {'loss': loss, 'actor_loss': actor_loss, 'value_loss': value_loss, 'entropy': entropy,
               'max_ratio_error': jnp.max(jnp.where(valid, error, zero)),
               'clip_fraction': valid_mean((error > clip).astype(jnp.float32)),
               'approx_kl': valid_mean((ratio - jnp.float32(1)) - log_ratio)}
    return loss, metrics


