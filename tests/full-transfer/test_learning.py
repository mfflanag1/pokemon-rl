"""Synthetic boundary and numerical checks for the additive F9 learner."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from unittest.mock import patch

from pokemon_transfer import learning
from pokemon_full_train import learning as ppo
from pokemon_fit import learning as fit
from pokemon_full.policy import initialize, forward, distribution
from pokemon_fixed.math import adam_step


def make_inputs():
    parameters = initialize(19)
    rng = np.random.default_rng(47)
    x = rng.uniform(0, 1, (1024, 928)).astype(np.float32)
    mask = np.zeros((1024, 14), bool)
    mask[:, [0, 2, 3, 8, 10]] = True
    actions = np.resize(np.array([0, 2, 3, 8, 10], np.int32), 1024)
    logits, _ = forward(parameters, jnp.asarray(x))
    _, logs = distribution(logits, jnp.asarray(mask))
    batch = {'features': x, 'mask': mask, 'actions': actions,
             'old_logp': np.asarray(logs)[np.arange(1024), actions].copy(),
             'advantages': rng.uniform(-.5, .5, 1024).astype(np.float32),
             'returns': rng.uniform(0, 1, 1024).astype(np.float32)}
    supervised = {'features': x[:128].copy(), 'mask': mask[:128].copy(),
                  'targets': np.full(128, 2, np.int32)}
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return parameters, (zeros, zeros, jnp.int32(0)), batch, supervised


def minibatch(batch):
    return {**{k: jnp.asarray(a[:256]) for k, a in batch.items()},
            'valid': jnp.ones(256, bool)}


def assert_tree(a, b, exact=False):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if exact:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=1e-6, rtol=0)


def test_zero_coefficient_matches_sealed_update(inputs):
    parameters, optimizer, batch, supervised = inputs
    part = minibatch(batch)
    # Exercise nonzero moments and a nonzero counter as well as a fresh state.
    for _ in range(2):
        expected = ppo.update(parameters, *optimizer, part)
        actual = learning.update(parameters, *optimizer, part, supervised, 0.)
        assert_tree(actual[:4], expected[:4])
        for key, value in expected[4].items():
            np.testing.assert_allclose(actual[4][key], value, atol=1e-6, rtol=0)
        np.testing.assert_allclose(actual[4]['total_loss'], actual[4]['loss'], atol=1e-6, rtol=0)
        parameters, optimizer = expected[0], expected[1:4]


def test_joint_gradient_and_single_clipped_step(inputs):
    parameters, optimizer, batch, supervised = inputs
    part, coefficient = minibatch(batch), .7
    total, metrics = learning.objective(parameters, part, supervised, coefficient)
    loss, original = ppo.objective(parameters, part)
    nll = fit.objective(parameters, supervised)
    np.testing.assert_allclose(total, loss + coefficient * nll, atol=1e-6, rtol=0)
    assert_tree({k: metrics[k] for k in original}, original)
    np.testing.assert_allclose(metrics['rehearsal_nll'], nll, atol=1e-6, rtol=0)
    policy_gradient = jax.grad(lambda p: ppo.objective(p, part)[0])(parameters)
    ce_gradient = jax.grad(fit.objective)(parameters, supervised)
    for key in ['vw', 'vb']:
        np.testing.assert_array_equal(ce_gradient[key], np.zeros_like(ce_gradient[key]))
    assert np.linalg.norm(np.asarray(ce_gradient['pw'])) > 0
    summed = jax.tree_util.tree_map(lambda a, b: a + coefficient * b,
                                    policy_gradient, ce_gradient)
    joint = jax.grad(lambda p: learning.objective(p, part, supervised, coefficient)[0])(parameters)
    assert_tree(joint, summed)
    expected = adam_step(parameters, joint, *optimizer,
                         lr=ppo.PPO['learning_rate'], max_norm=1.)
    actual = learning.update(parameters, *optimizer, part, supervised, coefficient)
    assert_tree(actual[:4], expected[:4])
    np.testing.assert_allclose(actual[4]['gradient_norm'], expected[4], atol=1e-6, rtol=0)


def test_optimize_is_exactly_repeatable_eight_steps(inputs):
    parameters, optimizer, batch, supervised = inputs
    first = learning.optimize_batch(parameters, optimizer, batch, supervised, 51, .3)
    second = learning.optimize_batch(parameters, optimizer, batch, supervised, 51, .3)
    assert_tree(first, second, exact=True)
    assert len(first[2]) == 8
    assert int(first[1][2]) == 8
    assert all(type(v) is float and np.isfinite(v) for row in first[2] for v in row.values())
    # Independently reproduce shuffle order and all-valid minibatch construction.
    rng = np.random.default_rng(51)
    p, (m, v, step) = parameters, optimizer
    records = []
    for _ in range(2):
        order = rng.permutation(1024)
        for start in range(0, 1024, 256):
            part = {k: jnp.asarray(a[order[start:start + 256]]) for k, a in batch.items()}
            part['valid'] = jnp.ones(256, bool)
            p, m, v, step, metrics = learning.update(p, m, v, step, part, supervised, np.float32(.3))
            records.append({k: float(a) for k, a in metrics.items()})
    assert_tree(first, (p, (m, v, step), records), exact=True)


class Learning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inputs=make_inputs()
    def test_zero_coefficient(self):test_zero_coefficient_matches_sealed_update(self.inputs)
    def test_joint_gradient(self):test_joint_gradient_and_single_clipped_step(self.inputs)
    def test_eight_steps(self):test_optimize_is_exactly_repeatable_eight_steps(self.inputs)
    def test_bad_coefficients(self):
        p,opt,b,s=self.inputs
        for c in [-1.,np.nan,np.inf,-np.inf,[0.],True,1e100]:
            with self.subTest(c=c),patch.object(learning,'update',side_effect=AssertionError('Update reached')):
                with self.assertRaises(ValueError):learning.optimize_batch(p,opt,b,s,0,c)
    def test_invalid_ppo(self):
        p,opt,source,s=self.inputs
        cases=[('extra','extra'),('features','shape'),('features','dtype'),('features','nan'),('mask','dtype'),('mask','empty'),('actions','illegal'),('actions','range'),('actions','dtype'),('old_logp','nan'),('old_logp','positive'),('advantages','nan'),('returns','nan'),('returns','range')]
        for field,kind in cases:
            b={k:a.copy() for k,a in source.items()}
            if kind=='extra':b[field]=np.zeros(1024)
            elif kind=='shape':b[field]=b[field][:-1]
            elif kind=='dtype':b[field]=b[field].astype(np.float64)
            elif kind=='nan':b[field].flat[0]=np.nan
            elif kind=='empty':b[field][0]=False
            elif kind=='illegal':b[field][0]=1
            elif kind=='positive':b[field][0]=.1
            else:b[field][0]=14 if field=='actions' else 1.1
            with self.subTest(field=field,kind=kind),patch.object(learning,'update',side_effect=AssertionError('Update reached')):
                with self.assertRaises(ValueError):learning.optimize_batch(p,opt,b,s,0,.1)
    def test_invalid_supervised(self):
        p,opt,b,source=self.inputs
        for kind in ['illegal','range','nan','extra','dtype','empty']:
            s={k:a.copy() for k,a in source.items()}
            if kind in ['illegal','range']:s['targets'][0]=1 if kind=='illegal' else 14
            elif kind=='nan':s['features'][0,0]=np.nan
            elif kind=='extra':s['labels']=s['targets']
            elif kind=='dtype':s['targets']=s['targets'].astype(np.int64)
            else:s['mask'][0]=False
            with self.subTest(kind=kind),patch.object(learning,'update',side_effect=AssertionError('Update reached')):
                with self.assertRaises(ValueError):learning.optimize_batch(p,opt,b,s,0,.1)

if __name__=='__main__':unittest.main()
