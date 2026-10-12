"""Transparent FP32 PPO/GAE reference with analytic tabular gradients.

No autodiff, exploration penalty, advantage normalization or privileged inputs.
The table is a canary model, not the future Pokemon policy architecture.
"""
import numpy as np

F32 = np.float32


def require(condition, message):
    if not condition: raise ValueError(message)


def finite(*arrays):
    require(all(np.all(np.isfinite(a)) for a in arrays), 'Nonfinite learner data')


def linked_gae(rewards, values, next_index, bootstrap, terminal, trajectory, *, gamma=1., lam=.95):
    """Next indices are same-player, same-episode decisions, ordered forward.

    next=-1 + terminal means no bootstrap; next=-1 + nonterminal is a buffer cut.
    At a cut bootstrap is that player's frozen-actor next value, never an opponent's.
    trajectory must distinguish both player AND episode; no alternating sign rule.
    """
    rewards, values, bootstrap = [np.asarray(a, dtype=F32) for a in (rewards, values, bootstrap)]
    next_index, terminal, trajectory = [np.asarray(a) for a in (next_index, terminal, trajectory)]
    n = len(rewards)
    require(n > 0 and all(a.shape == (n,) for a in (rewards, values, bootstrap, next_index, terminal, trajectory)), 'GAE shapes')
    require(next_index.dtype.kind in 'iu' and terminal.dtype == np.bool_, 'GAE index/terminal types')
    require(0 <= gamma <= 1 and 0 <= lam <= 1, 'GAE discounts')
    finite(rewards, values, bootstrap)
    advantages = np.zeros(n, dtype=F32)
    for i in range(n-1, -1, -1):
        j = int(next_index[i])
        require(j == -1 or i < j < n, 'GAE linkage must be forward or boundary')
        if terminal[i]:
            require(j == -1 and bootstrap[i] == 0, 'Terminal must not bootstrap')
            v_next, a_next = F32(0), F32(0)
        elif j == -1:
            v_next, a_next = bootstrap[i], F32(0)
        else:
            require(trajectory[i] == trajectory[j], 'Cross-player/episode linkage')
            require(bootstrap[i] == 0, 'Internal decision cannot have boundary bootstrap')
            v_next, a_next = values[j], advantages[j]
        delta = rewards[i] + F32(gamma) * v_next - values[i]
        advantages[i] = delta + F32(gamma * lam) * a_next
    returns = advantages + values
    finite(advantages, returns)
    return advantages, returns


def distribution(logits, mask):
    logits, mask = np.asarray(logits), np.asarray(mask)
    require(logits.dtype == F32 and logits.ndim == 2 and mask.shape == logits.shape and mask.dtype == np.bool_, 'Policy shape/dtype')
    require(len(logits) > 0 and np.all(mask.any(axis=1)), 'Empty legal-action mask')
    finite(logits)
    maxima = np.max(np.where(mask, logits, F32(-np.inf)), axis=1, keepdims=True)
    # Illegal entries have exactly zero probability and a finite log sentinel.
    shifted = np.where(mask, logits - maxima, F32(0))
    exp = np.where(mask, np.exp(shifted), F32(0))
    denominator = exp.sum(axis=1, keepdims=True, dtype=F32)
    logp = np.where(mask, shifted - np.log(denominator), F32(0))
    probabilities = exp / denominator
    finite(probabilities, logp)
    return probabilities, logp


def ppo_loss_grad(logits, values, mask, actions, old_logp, advantages, returns, *, clip=.2, value_coef=.5, entropy_coef=0.):
    """Mean clipped actor loss + coefficient * half-MSE - coefficient * entropy."""
    p, logp = distribution(logits, mask)
    n, actions_n = p.shape
    values, old_logp, advantages, returns = [np.asarray(a) for a in (values, old_logp, advantages, returns)]
    actions = np.asarray(actions)
    require(all(a.shape == (n,) and a.dtype == F32 for a in (values, old_logp, advantages, returns)), 'PPO FP32 vectors')
    require(actions.shape == (n,) and actions.dtype.kind in 'iu' and np.all((actions >= 0) & (actions < actions_n)), 'PPO action indices')
    require(np.all(mask[np.arange(n), actions]), 'Known-forbidden action')
    require(0 < clip < 1 and value_coef >= 0 and entropy_coef >= 0, 'PPO coefficients')
    finite(values, old_logp, advantages, returns)
    log_ratio = logp[np.arange(n), actions] - old_logp
    with np.errstate(over='raise', invalid='raise'):
        ratio = np.exp(log_ratio)
    clipped = np.clip(ratio, F32(1-clip), F32(1+clip))
    actor_loss = -np.minimum(ratio * advantages, clipped * advantages).mean(dtype=F32)
    entropy_rows = -(p * logp).sum(axis=1, dtype=F32)
    entropy = entropy_rows.mean(dtype=F32)
    value_loss = F32(.5) * np.square(values - returns).mean(dtype=F32)
    loss = actor_loss + F32(value_coef) * value_loss - F32(entropy_coef) * entropy
    active = np.where(advantages >= 0, ratio <= F32(1+clip), ratio >= F32(1-clip))
    score = -p.copy(); score[np.arange(n), actions] += F32(1)
    grad_logits = (-advantages * ratio * active / F32(n))[:, None] * score
    grad_logits += F32(entropy_coef / n) * p * (logp + entropy_rows[:, None])
    grad_values = F32(value_coef / n) * (values - returns)
    finite(loss, ratio, grad_logits, grad_values)
    return {'loss': float(loss), 'actor_loss': float(actor_loss), 'value_loss': float(value_loss),
            'entropy': float(entropy), 'max_ratio_error': float(np.max(np.abs(ratio-F32(1)))),
            'clip_fraction': float(np.mean(np.abs(ratio-F32(1)) > clip)),
            'approx_kl': float(((ratio-F32(1))-log_ratio).mean())}, grad_logits, grad_values


class Adam:
    def __init__(self, parameters, *, lr=.03, max_norm=1.):
        require(lr > 0 and max_norm > 0, 'Optimizer configuration')
        self.lr, self.max_norm, self.step = lr, max_norm, 0
        self.m = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.v = {k: np.zeros_like(v) for k, v in parameters.items()}

    def apply(self, parameters, gradients):
        require(set(parameters) == set(gradients) == set(self.m) == set(self.v), 'Optimizer keys')
        for k in parameters:
            require(parameters[k].dtype == F32 and gradients[k].dtype == F32 and parameters[k].shape == gradients[k].shape, 'Optimizer FP32 shapes')
        finite(*parameters.values(), *gradients.values(), *self.m.values(), *self.v.values())
        norm = float(np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in gradients.values())))
        scale = F32(min(1., self.max_norm / max(norm, 1e-30)))
        self.step += 1
        for k, gradient in gradients.items():
            g = gradient * scale
            self.m[k] = F32(.9) * self.m[k] + F32(.1) * g
            self.v[k] = F32(.999) * self.v[k] + F32(.001) * np.square(g)
            m = self.m[k] / F32(1 - .9 ** self.step)
            v = self.v[k] / F32(1 - .999 ** self.step)
            parameters[k] -= F32(self.lr) * m / (np.sqrt(v) + F32(1e-8))
        finite(*parameters.values(), *self.m.values(), *self.v.values())
        return norm
