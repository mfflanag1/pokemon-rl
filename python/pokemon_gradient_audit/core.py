"""Loss-component geometry and cloned updates; no policy promotion or new targets."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.math import adam_step
from pokemon_full.policy import distribution
from pokemon_full_train.learning import PPO
from pokemon_matchup.learning import objective, update, validate
from pokemon_matchup.network import forward, check_state

COMPONENTS = ('actor', 'value', 'entropy')
GROUPS = {'shared': ('w', 'b'), 'policy': ('pw', 'pb', 'matchup_policy'),
          'value': ('vw', 'vb', 'matchup_value')}
VARIANTS = ('joint', 'block_current_shared_value', 'omit_current_entropy')


def component(p, b, name):
    _, m = objective(p, b)
    if name == 'actor':
        return m['actor_loss']
    if name == 'value':
        return PPO['value_coefficient'] * m['value_loss']
    if name == 'entropy':
        return -PPO['entropy'] * m['entropy']
    raise ValueError('Unknown loss component')


@jax.jit
def gradients(p, b):
    return ({name: jax.grad(partial(component, name=name))(p, b) for name in COMPONENTS},
            jax.grad(lambda q: objective(q, b)[0])(p))


def intervention_gradient(joint, parts, variant):
    if variant == 'joint':
        return joint
    if variant == 'block_current_shared_value':
        return {k: g - parts['value'][k] if k in GROUPS['shared'] else g
                for k, g in joint.items()}
    if variant == 'omit_current_entropy':
        return {k: g - parts['entropy'][k] for k, g in joint.items()}
    raise ValueError('Unknown diagnostic variant')


@partial(jax.jit, static_argnames=('variant',))
def changed_kernel(p, opt, b, variant):
    parts, joint = gradients(p, b)
    g = intervention_gradient(joint, parts, variant)
    q, m, v, step, norm = adam_step(p, g, *opt, lr=PPO['learning_rate'], max_norm=1.)
    return q, (m, v, step), norm


def vector(g, keys):
    return np.concatenate([np.asarray(g[k], dtype=np.float64).reshape(-1) for k in keys])


def geometry(p, b):
    parts, joint = gradients(p, b)
    error = max(float(np.max(np.abs(np.asarray(joint[k]) -
                sum(np.asarray(parts[c][k]) for c in COMPONENTS)))) for k in joint)
    if error > 2e-6:
        raise ValueError('Loss gradient decomposition mismatch')
    result = {'decomposition_max_abs_error': error, 'groups': {}}
    for group, keys in GROUPS.items():
        vecs = {c: vector(g, keys) for c, g in parts.items()}
        norms = {c: float(np.linalg.norm(v)) for c, v in vecs.items()}
        pairs = {}
        for other in ['value', 'entropy']:
            dot = float(vecs['actor'] @ vecs[other])
            denom = norms['actor'] * norms[other]
            pairs[other] = {'dot': dot, 'cosine': dot / denom if denom > 1e-30 else None,
                            'norm_ratio_to_actor': norms[other] / norms['actor']
                            if norms['actor'] > 1e-30 else None}
        result['groups'][group] = {'norms': norms, 'actor_alignment': pairs}
    return result


def update_motion(before, after, b):
    parts, _ = gradients(before, b)
    delta = {k: np.asarray(after[k], np.float64)-np.asarray(before[k], np.float64) for k in before}
    result = {}
    for group, keys in {**GROUPS, 'all': tuple(sorted(before))}.items():
        a, d = vector(parts['actor'], keys), vector(delta, keys)
        denom = float(np.linalg.norm(a)*np.linalg.norm(d))
        dot = float(a@d)
        result[group] = {'actor_gradient_dot_update': dot,
                         'cosine_with_actor_descent': -dot/denom if denom>1e-30 else None,
                         'update_norm':float(np.linalg.norm(d))}
    result['actual_minibatch_actor_loss_change'] = float(component(after,b,'actor')-component(before,b,'actor'))
    return result


@jax.jit
def predictions(p, b):
    logits, values = forward(p, b['features'])
    probs, logs = distribution(logits, b['mask'])
    return probs, logs, values


def measure(p, b, reference):
    _, metrics = objective(p, b)
    probs, logs, values = predictions(p, b)
    probs, logs, values = map(np.asarray, (probs, logs, values))
    old_probs, old_logs, _ = map(np.asarray, reference)
    has_choice = np.asarray(b['mask']).sum(1) > 1
    kl = np.sum(old_probs.astype(np.float64) * (old_logs - logs), axis=1)
    x = np.asarray(b['features'])
    hidden = np.tanh(x[:, :928] @ np.asarray(p['w']) + np.asarray(p['b']))
    return {**{k: float(v) for k, v in metrics.items()},
            'old_policy_kl': float(kl.mean()),
            'choice_row_switch_probability': float(probs[has_choice, 8:].sum(1).mean())
            if has_choice.any() else None,
            'mean_value': float(values.mean()),
            'value_target_mse': float(np.mean((values - np.asarray(b['returns'])) ** 2)),
            'hidden_abs_gt_099_fraction': float(np.mean(np.abs(hidden) > .99))}


def audit_batch(p, opt, batch, seed):
    """Three independent eight-step clones, identical frozen targets and Adam history.

    Component gradients are measured along the original joint path. The other
    paths remove only new gradient contributions; inherited Adam moments remain.
    These local frozen-batch counterfactuals do not estimate battle improvement.
    """
    b = validate({**batch, 'valid': np.ones(len(batch['actions']), bool)})
    if len(batch['actions']) != 1024:
        raise ValueError('Expected one complete selected F26 cohort')
    check_state(p, opt)
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(1024) for _ in range(2)]
    reference = predictions(p, b)
    initial = measure(p, b, reference)
    traces = []
    results = {}
    states = {}
    for variant in VARIANTS:
        q, qo = p, opt
        metrics = []
        for order in orders:
            for first in range(0, 1024, 256):
                part = {k: v[order[first:first + 256]] for k, v in b.items()}
                if variant == 'joint':
                    trace = geometry(q, part)
                    before = q
                    q, qo, m = update(q, qo, part)
                    trace['actual_adam_motion'] = update_motion(before, q, part)
                    traces.append(trace)
                    metrics.append(m)
                else:
                    q, qo, norm = changed_kernel(q, qo, part, variant)
                    if not np.isfinite(float(norm)):
                        raise ValueError('Nonfinite diagnostic update')
        check_state(q, qo)
        if int(qo[2]) != int(opt[2]) + 8:
            raise ValueError('Diagnostic counter mismatch')
        end = measure(q, b, reference)
        results[variant] = {'end': end, 'actor_surrogate_improvement': initial['actor_loss'] - end['actor_loss'],
                            'original_metrics': metrics if variant == 'joint' else None}
        states[variant] = (q, qo)
    advantages = np.asarray(batch['advantages'])
    result = {'initial': initial, 'variants': results, 'original_path_gradient_geometry': traces,
              'rows': {'selected': len(advantages), 'positive_advantages': int((advantages > 0).sum()),
                       'mean_advantage': float(advantages.mean()), 'std_advantage': float(advantages.std()),
                       'mean_abs_advantage': float(np.abs(advantages).mean()),
                       'single_legal_action_rows': int((np.asarray(batch['mask']).sum(1) == 1).sum())}}
    return result, states
