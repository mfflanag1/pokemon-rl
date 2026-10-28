"""Independent numerical and boundary checks; no games or stored artifacts."""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full import policy
from pokemon_full_train import learning as sealed
from pokemon_shared import network
from pokemon_shared_ppo import learning as learner


def equal(a, b, exact=True):
    aa, ta = jax.tree.flatten(a)
    bb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb):
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-7)


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Malformed input accepted')


def zeros(p):
    z = jax.tree.map(jnp.zeros_like, p)
    return z, z, jnp.int32(0)


def batch(p, architecture, n):
    rng = np.random.default_rng(27)
    x = rng.normal(0, .3, (n, 928)).astype(np.float32)
    mons = x[:, :768].reshape(n, 12, 64)
    mons[:, :, 0] = 1
    mons[:, :, 1] = 0
    mons[:, 0, 1] = mons[:, 6, 1] = 1
    mask = rng.random((n, 14)) > .3
    mask[:, 0] = True
    actions = np.array([rng.choice(np.flatnonzero(row)) for row in mask], np.int32)
    fn = policy.forward if architecture == 'flat' else network.forward
    logits, _ = fn(p, jnp.asarray(x))
    _, logs = policy.distribution(logits, jnp.asarray(mask))
    # Both signs, with both clipped and unclipped branches for each sign.
    ratios = np.resize(np.array([1.4, .6, 1.05, .95], np.float32), n)
    advantages = np.resize(np.array([.8, -.7, -.4, .6], np.float32), n)
    old = np.asarray(logs)[np.arange(n), actions] - np.log(ratios)
    return dict(features=x, mask=mask, actions=actions, old_logp=old,
                advantages=advantages, returns=rng.uniform(.05, .95, n).astype(np.float32),
                valid=np.ones(n, bool))


def asjax(b):
    return {k: jnp.asarray(v) for k, v in b.items()}


def flat_parity():
    p = policy.initialize(3)
    b = asjax(batch(p, 'flat', 16))
    equal(learner.objective(p, b, 'flat'), sealed.objective(p, b))
    equal(jax.grad(lambda q: learner.objective(q, b, 'flat')[0])(p),
          jax.grad(lambda q: sealed.objective(q, b)[0])(p))
    cohort = batch(p, 'flat', 1024)
    del cohort['valid']
    q, opt, records = learner.optimize_batch(p, zeros(p), cohort, 91, 'flat')
    m, v, step = zeros(p)
    rng = np.random.default_rng(91)
    expected = []
    for _ in range(2):
        order = rng.permutation(1024)
        for first in range(0, 1024, 256):
            ids = order[first:first + 256]
            part = {k: jnp.asarray(a[ids]) for k, a in cohort.items()}
            part['valid'] = jnp.ones(256, bool)
            p, m, v, step, metrics = sealed.update(p, m, v, step, part)
            expected.append({k: float(x) for k, x in metrics.items()})
    assert len(records) == 8
    equal((q, opt, records), (p, (m, v, step), expected))


def shared_math():
    p = network.initialize(4)
    b = asjax(batch(p, 'shared', 12))
    logits, values = network.forward(p, b['features'])
    np.testing.assert_array_equal(values, .5)
    assert np.ptp(np.asarray(logits)) > .001
    # Independent output-level NumPy arithmetic, using the actual finite-word
    # probabilities/logs rather than substituting an ordinary softmax.
    probs, logs = map(np.asarray, policy.distribution(logits, b['mask']))
    values = np.asarray(values)
    ratio = np.exp(logs[np.arange(12), np.asarray(b['actions'])] - np.asarray(b['old_logp']))
    adv = np.asarray(b['advantages'])
    actor = -np.minimum(ratio * adv, np.clip(ratio, .8, 1.2) * adv).mean()
    value = .5 * np.mean((values - np.asarray(b['returns']))**2)
    entropy = -np.sum(probs * logs, axis=1).mean()
    loss, metrics = learner.objective(p, b, 'shared')
    expected = dict(loss=actor + .5*value - .01*entropy, actor_loss=actor,
                    value_loss=value, entropy=entropy,
                    approx_kl=np.mean(ratio - 1 - np.log(ratio)))
    for k in expected:
        np.testing.assert_allclose(metrics[k], expected[k], rtol=3e-6, atol=2e-7)
    # Nonzero value weights make the encoder derivative exercise both heads.
    p = {**p, 'value_w': jnp.linspace(-.05, .08, 160)}
    f = lambda q: learner.objective(q, b, 'shared')[0]
    grads = jax.grad(f)(p)
    rng = np.random.default_rng(123)
    for name in ('score_w', 'value_w', 'context_w', 'mon_w', 'move_w'):
        direction = rng.normal(size=p[name].shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        delta = .003
        plus = {**p, name: p[name] + delta*direction}
        minus = {**p, name: p[name] - delta*direction}
        finite = float((f(plus) - f(minus)) / (2*delta))
        analytic = float(jnp.sum(grads[name] * direction))
        np.testing.assert_allclose(finite, analytic, rtol=.06, atol=2e-5)
    assert float(jnp.linalg.norm(grads['score_w'])) > 0
    assert float(jnp.linalg.norm(grads['value_w'])) > 0
    q, _, _, counter, _ = learner.update(p, *zeros(p), b, 'shared')
    assert int(counter) == 1
    assert not np.array_equal(q['score_w'], p['score_w'])
    assert not np.array_equal(q['value_w'], p['value_w'])
    outputs = np.asarray(network.forward(q, b['features'])[1])
    assert ((outputs > 0) & (outputs < 1)).all()
    # Initialized constant .5 value head learns toward a uniform bounded target.
    initial = network.initialize(4)
    target = {**b, 'returns': jnp.full(12, .9)}
    trained, *_ = learner.update(initial, *zeros(initial), target, 'shared')
    assert np.mean(np.asarray(network.forward(trained, b['features'])[1])) > .5
    for key in ('old_logp', 'advantages', 'returns'):
        g = jax.grad(lambda a: learner.objective(p, {**b, key: a}, 'shared')[0])(b[key])
        np.testing.assert_array_equal(g, np.zeros(12))


def padding_and_guards():
    for architecture, p in [('flat', policy.initialize(2)), ('shared', network.initialize(2))]:
        clean = batch(p, architecture, 4)
        padded = {k: np.concatenate([a, a]) for k, a in clean.items()}
        padded['valid'][4:] = False
        for k in ('features', 'old_logp', 'advantages', 'returns'):
            padded[k][4:] = np.nan
        padded['actions'][4:] = -999
        padded['mask'][4:] = False
        a, b = asjax(clean), asjax(padded)
        equal(learner.objective(p, a, architecture), learner.objective(p, b, architecture), False)
        equal(jax.grad(lambda q: learner.objective(q, a, architecture)[0])(p),
              jax.grad(lambda q: learner.objective(q, b, architecture)[0])(p), False)
        learner.update(p, *zeros(p), b, architecture)
        empty = {**b, 'valid': jnp.zeros(8, bool)}
        loss, metrics = learner.objective(p, empty, architecture)
        assert float(loss) == 0 and all(float(x) == 0 for x in metrics.values())
        equal(jax.grad(lambda q: learner.objective(q, empty, architecture)[0])(p), zeros(p)[0])
        opt = zeros(p)
        for bad in (None, (), (opt[0],), (None, opt[1], opt[2]),
                    (opt[0], opt[1], np.int64(0)), (opt[0], opt[1], np.int32(-1)),
                    (opt[0], opt[1], np.int32(2**31-1))):
            rejected(lambda: learner.optimize_batch(p, bad, {}, 1, architecture))
        key = next(iter(p))
        for badp in ({}, {**p, key: jnp.full_like(p[key], jnp.nan)},
                     {**p, key: jnp.zeros((1,), jnp.float32)}):
            rejected(lambda: learner.update(badp, *opt, a, architecture))
        negative = {**opt[1], key: -jnp.ones_like(opt[1][key])}
        rejected(lambda: learner.update(p, opt[0], negative, opt[2], a, architecture))
        for bad in ({k: v for k, v in a.items() if k != 'returns'},
                    {**a, 'actions': jnp.full(4, 14, jnp.int32)},
                    {**a, 'returns': jnp.full(4, 1.5)},
                    {**a, 'features': jnp.full((4, 928), jnp.nan)},
                    {**a, 'mask': jnp.zeros((4, 14), bool)}):
            rejected(lambda: learner.update(p, *opt, bad, architecture))
        cohort = batch(p, architecture, 1024)
        del cohort['valid']
        rejected(lambda: learner.optimize_batch(p, (opt[0], opt[1], np.int32(2**31-8)), cohort, 1, architecture))
        rejected(lambda: learner.optimize_batch(p, opt, {**cohort, 'extra': 1}, 1, architecture))
    rejected(lambda: learner.objective({}, {}, 'unknown'))


if __name__ == '__main__':
    flat_parity()
    shared_math()
    padding_and_guards()
    print('shared PPO learner checks passed')
