"""JAX autodiff/JIT port of the sealed FP32 PPO/GAE reference (`pokemon_learner.core`).

Host-side validation is NumPy; every numerical result (distribution, GAE scan, loss,
gradients, Adam update and gradient clipping) is computed by JAX on CPU in FP32.
Nothing here is a NumPy numerical fallback. x64 is assumed disabled; inputs must
already be FP32 (no silent coercion of policy data).
"""
import jax
import jax.numpy as jnp
import numpy as np

F32 = np.float32

__all__ = ['require', 'finite', 'distribution', 'distribution_kernel', 'linked_gae',
           'loss_kernel', 'ppo_loss_grad', 'Adam']


# ----------------------------------------------------------------------------- host checks

def require(condition, message):
    if not condition: raise ValueError(message)


def finite(*arrays):
    require(all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays), 'Nonfinite learner data')


def _check_policy(logits, mask):
    """Reference `distribution` input checks; returns host NumPy views (no dtype coercion)."""
    logits, mask = np.asarray(logits), np.asarray(mask)
    require(logits.dtype == F32 and logits.ndim == 2 and mask.shape == logits.shape and mask.dtype == np.bool_, 'Policy shape/dtype')
    require(len(logits) > 0 and bool(np.all(mask.any(axis=1))), 'Empty legal-action mask')
    finite(logits)
    return logits, mask


# ----------------------------------------------------------------------------- distribution

def distribution_kernel(logits, mask):
    """Pure JAX masked softmax: (probabilities, log-probabilities), shape (n, actions).

    Illegal entries have exactly zero probability, a finite log sentinel of zero and an
    exactly zero gradient (all masking is `select`, and the row maximum carries no gradient).
    """
    maxima = jax.lax.stop_gradient(jnp.max(jnp.where(mask, logits, -jnp.inf), axis=1, keepdims=True))
    shifted = jnp.where(mask, logits - maxima, jnp.float32(0))
    exp = jnp.where(mask, jnp.exp(shifted), jnp.float32(0))
    denominator = jnp.sum(exp, axis=1, keepdims=True)
    logp = jnp.where(mask, shifted - jnp.log(denominator), jnp.float32(0))
    return exp / denominator, logp


_distribution_compiled = jax.jit(distribution_kernel)


def distribution(logits, mask):
    """Checked wrapper over `distribution_kernel`; returns FP32 JAX arrays (p, logp)."""
    logits, mask = _check_policy(logits, mask)
    probabilities, logp = _distribution_compiled(logits, mask)
    finite(probabilities, logp)
    return probabilities, logp


# ----------------------------------------------------------------------------- linked GAE

def _gae_kernel(rewards, values, next_index, bootstrap, terminal, gamma, gamma_lam):
    n = rewards.shape[0]

    def step(advantages, i):
        j = next_index[i]
        linked = j >= 0
        j_safe = jnp.where(linked, j, 0)
        v_next = jnp.where(terminal[i], jnp.float32(0), jnp.where(linked, values[j_safe], bootstrap[i]))
        a_next = jnp.where(linked, advantages[j_safe], jnp.float32(0))
        delta = rewards[i] + gamma * v_next - values[i]
        return advantages.at[i].set(delta + gamma_lam * a_next), None

    advantages, _ = jax.lax.scan(step, jnp.zeros(n, jnp.float32), jnp.arange(n - 1, -1, -1))
    return advantages, advantages + values


_gae_compiled = jax.jit(_gae_kernel)


def linked_gae(rewards, values, next_index, bootstrap, terminal, trajectory, *, gamma=1., lam=.95):
    """Same-player forward-linked GAE (reference semantics) via a JAX reverse scan.

    next=-1 + terminal: no bootstrap (bootstrap must be 0). next=-1 + nonterminal: buffer
    cut using that decision's own bootstrap. Internal links must stay inside one
    trajectory (player AND episode) and carry bootstrap 0. No alternating sign rule.
    """
    rewards, values, bootstrap = [np.asarray(a, dtype=F32) for a in (rewards, values, bootstrap)]
    next_index, terminal, trajectory = [np.asarray(a) for a in (next_index, terminal, trajectory)]
    n = len(rewards)
    require(n > 0 and all(a.shape == (n,) for a in (rewards, values, bootstrap, next_index, terminal, trajectory)), 'GAE shapes')
    require(next_index.dtype.kind in 'iu' and terminal.dtype == np.bool_, 'GAE index/terminal types')
    require(0 <= gamma <= 1 and 0 <= lam <= 1, 'GAE discounts')
    finite(rewards, values, bootstrap)
    i = np.arange(n)
    j = next_index.astype(np.int64)
    require(bool(np.all((j == -1) | ((j > i) & (j < n)))), 'GAE linkage must be forward or boundary')
    linked = j >= 0
    require(bool(np.all(~terminal | (~linked & (bootstrap == 0)))), 'Terminal must not bootstrap')
    require(bool(np.all(trajectory[i[linked]] == trajectory[j[linked]])), 'Cross-player/episode linkage')
    require(bool(np.all(bootstrap[linked] == 0)), 'Internal decision cannot have boundary bootstrap')
    advantages, returns = _gae_compiled(rewards, values, j.astype(np.int32), bootstrap, terminal,
                                        F32(gamma), F32(gamma * lam))
    finite(advantages, returns)
    return advantages, returns


# ----------------------------------------------------------------------------- PPO loss

def _loss(logits, values, mask, actions, old_logp, advantages, returns, low, high, clip, value_coef, entropy_coef):
    """Pure differentiable loss; `low/high/clip/value_coef/entropy_coef` are FP32 scalars."""
    old_logp, advantages, returns = [jax.lax.stop_gradient(a) for a in (old_logp, advantages, returns)]
    p, logp = distribution_kernel(logits, mask)
    n = logits.shape[0]
    rows = jnp.arange(n)
    log_ratio = logp[rows, actions] - old_logp
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, low, high)
    # Explicit inclusive branch (reference `active`): equality on the unclipped side keeps
    # the gradient, instead of autodiff's averaged gradient at min/clip ties. The selected
    # value equals min(ratio*A, clipped*A) exactly.
    active = jnp.where(advantages >= 0, ratio <= high, ratio >= low)
    surrogate = jnp.where(active, ratio, jax.lax.stop_gradient(clipped)) * advantages
    actor_loss = -jnp.mean(surrogate)
    entropy_rows = -jnp.sum(p * logp, axis=1)
    entropy = jnp.mean(entropy_rows)
    value_loss = jnp.float32(.5) * jnp.mean(jnp.square(values - returns))
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
    error = jnp.abs(ratio - jnp.float32(1))
    metrics = {'loss': loss, 'actor_loss': actor_loss, 'value_loss': value_loss, 'entropy': entropy,
               'max_ratio_error': jnp.max(error), 'clip_fraction': jnp.mean(error > clip),
               'approx_kl': jnp.mean((ratio - jnp.float32(1)) - log_ratio)}
    return loss, metrics


def _loss_constants(clip, value_coef, entropy_coef):
    # Host-rounded like the reference's F32(1-clip)/F32(1+clip) when given Python scalars.
    return tuple(jnp.asarray(c, jnp.float32) for c in (1 - clip, 1 + clip, clip, value_coef, entropy_coef))


def loss_kernel(logits, values, mask, actions, old_logp, advantages, returns, clip, value_coef, entropy_coef):
    """Pure JAX PPO loss -> (scalar_loss, metrics) with the reference's seven metrics.

    Mean clipped actor loss + value_coef * half-MSE - entropy_coef * entropy. No value
    clipping, advantage normalization, exploration penalty or weight decay. All targets
    (old_logp, advantages, returns) are stop-gradient. Differentiable in logits and values.
    """
    return _loss(logits, values, mask, actions, old_logp, advantages, returns, *_loss_constants(clip, value_coef, entropy_coef))


_loss_grad_compiled = jax.jit(jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True))


def ppo_loss_grad(logits, values, mask, actions, old_logp, advantages, returns, *, clip=.2, value_coef=.5, entropy_coef=0.):
    """Checked PPO step: (metrics as Python floats, grad_logits, grad_values) via JIT value_and_grad."""
    logits, mask = _check_policy(logits, mask)
    n, actions_n = logits.shape
    values, old_logp, advantages, returns = [np.asarray(a) for a in (values, old_logp, advantages, returns)]
    actions = np.asarray(actions)
    require(all(a.shape == (n,) and a.dtype == F32 for a in (values, old_logp, advantages, returns)), 'PPO FP32 vectors')
    require(actions.shape == (n,) and actions.dtype.kind in 'iu' and bool(np.all((actions >= 0) & (actions < actions_n))), 'PPO action indices')
    require(bool(np.all(mask[np.arange(n), actions])), 'Known-forbidden action')
    require(0 < clip < 1 and value_coef >= 0 and entropy_coef >= 0, 'PPO coefficients')
    finite(values, old_logp, advantages, returns)
    constants = (F32(1 - clip), F32(1 + clip), F32(clip), F32(value_coef), F32(entropy_coef))
    (loss, metrics), (grad_logits, grad_values) = _loss_grad_compiled(
        logits, values, mask, actions.astype(np.int32), old_logp, advantages, returns, *constants)
    # max_ratio_error is nonfinite iff any ratio is (reference checks `ratio` directly).
    finite(loss, metrics['max_ratio_error'], grad_logits, grad_values)
    return {k: float(v) for k, v in metrics.items()}, grad_logits, grad_values


# ----------------------------------------------------------------------------- Adam

def _adam_kernel(parameters, gradients, m, v, lr, max_norm, bias1, bias2):
    """Global-norm clip + Adam moment/parameter update. Returns (parameters, m, v, pre-clip norm)."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in gradients.values()))
    scale = jnp.minimum(jnp.float32(1), max_norm / jnp.maximum(norm, jnp.float32(1e-30)))
    new_parameters, new_m, new_v = {}, {}, {}
    for k in gradients:
        g = gradients[k] * scale
        new_m[k] = jnp.float32(.9) * m[k] + jnp.float32(.1) * g
        new_v[k] = jnp.float32(.999) * v[k] + jnp.float32(.001) * jnp.square(g)
        m_hat = new_m[k] / bias1
        v_hat = new_v[k] / bias2
        new_parameters[k] = parameters[k] - lr * m_hat / (jnp.sqrt(v_hat) + jnp.float32(1e-8))
    return new_parameters, new_m, new_v, norm


_adam_compiled = jax.jit(_adam_kernel)


class Adam:
    """Reference Adam (beta 0.9/0.999, eps 1e-8, bias-corrected) with global-norm clipping.

    `m`/`v` are dicts of FP32 JAX arrays, `step` a Python int. `apply` replaces the entries
    of the parameter mapping with new immutable JAX arrays and returns the pre-clip global
    norm (FP32 reduction; may differ from the reference's FP64 norm within tolerance).
    """

    def __init__(self, parameters, *, lr=.03, max_norm=1.):
        require(lr > 0 and max_norm > 0, 'Optimizer configuration')
        self.lr, self.max_norm, self.step = lr, max_norm, 0
        self.m = {k: jnp.zeros_like(v) for k, v in parameters.items()}
        self.v = {k: jnp.zeros_like(v) for k, v in parameters.items()}

    def apply(self, parameters, gradients):
        require(set(parameters) == set(gradients) == set(self.m) == set(self.v), 'Optimizer keys')
        for k in parameters:
            p, g = parameters[k], gradients[k]
            require(getattr(p, 'dtype', None) == F32 and getattr(g, 'dtype', None) == F32 and p.shape == g.shape, 'Optimizer FP32 shapes')
            require(self.m[k].dtype == F32 and self.v[k].dtype == F32 and self.m[k].shape == p.shape and self.v[k].shape == p.shape, 'Optimizer state shapes')
        finite(*parameters.values(), *gradients.values(), *self.m.values(), *self.v.values())
        step = self.step + 1
        # Host-rounded bias-correction constants, matching the reference F32(1 - beta ** step).
        bias1, bias2 = F32(1 - .9 ** step), F32(1 - .999 ** step)
        new_parameters, new_m, new_v, norm = _adam_compiled(
            dict(parameters), dict(gradients), self.m, self.v, F32(self.lr), F32(self.max_norm), bias1, bias2)
        # Validate the complete update before mutating anything: no partially applied bad step.
        finite(norm, *new_parameters.values(), *new_m.values(), *new_v.values())
        self.step = step
        self.m, self.v = new_m, new_v
        for k in parameters:
            parameters[k] = new_parameters[k]
        return float(norm)
