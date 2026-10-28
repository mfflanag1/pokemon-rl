import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize, distribution
from pokemon_matchup.network import extend, forward
from pokemon_matchup.learning import objective, optimize_batch, validate
from pokemon_full.learner import parameter_hash
from pokemon_shared.common import same_state
from pokemon_gradient_audit.core import (component, gradients, intervention_gradient,
    audit_batch, geometry, changed_kernel, GROUPS)
from pokemon_gradient_audit.campaign import validate_plan, anchors, PLAN
from pokemon_diversity.common import read


def fixture(n=8):
    p, opt = extend(initialize(1729))
    p = {**p, 'vw': jnp.linspace(-.2, .3, 64), 'vb': jnp.float32(.1),
         'matchup_policy': jnp.linspace(-.1, .2, 25),
         'matchup_value': jnp.linspace(-.03, .03, 350)}
    rng = np.random.default_rng(29)
    x = rng.normal(0, .2, (n, 1278)).astype(np.float32)
    mask = np.ones((n, 14), bool)
    mask[:, 12:] = False
    actions = np.arange(n, dtype=np.int32) % 12
    logits, _ = forward(p, jnp.asarray(x))
    _, logs = distribution(logits, jnp.asarray(mask))
    b = dict(features=x, mask=mask, actions=actions,
             old_logp=np.asarray(logs)[np.arange(n), actions],
             advantages=np.linspace(-.4, .7, n, dtype=np.float32),
             returns=np.linspace(.1, .9, n, dtype=np.float32))
    return p, opt, b


class AuditTests(unittest.TestCase):
    def test_independent_loss_arithmetic(self):
        p, _, batch = fixture()
        b = validate({**batch, 'valid': np.ones(8, bool)})
        x = batch['features'].astype(np.float64)
        h = np.tanh(x[:, :928] @ np.asarray(p['w']) + np.asarray(p['b']))
        logits = h @ np.asarray(p['pw']) + np.asarray(p['pb']) + x[:, 928:].reshape(-1, 14, 25) @ np.asarray(p['matchup_policy'])
        logits[~batch['mask']] = -np.inf
        prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
        value = 1 / (1 + np.exp(-(h @ np.asarray(p['vw']) + np.asarray(p['vb']) + x[:, 928:] @ np.asarray(p['matchup_value']))))
        ratio = prob[np.arange(8), batch['actions']] / np.exp(batch['old_logp'])
        actor = -np.minimum(ratio * batch['advantages'], np.clip(ratio, .8, 1.2) * batch['advantages']).mean()
        ent = .01 * np.sum(np.where(prob > 0, prob * np.log(np.maximum(prob, 1e-100)), 0), 1).mean()
        expected = {'actor': actor, 'value': .25 * np.mean((value-batch['returns'])**2), 'entropy': ent}
        for name, value in expected.items():
            self.assertAlmostEqual(float(component(p, b, name)), value, delta=2e-6)

    def test_decomposition_and_disjoint_heads(self):
        p, _, batch = fixture()
        b = validate({**batch, 'valid': np.ones(8, bool)})
        parts, joint = gradients(p, b)
        self.assertLess(geometry(p, b)['decomposition_max_abs_error'], 2e-6)
        for k in GROUPS['policy']:
            np.testing.assert_array_equal(parts['value'][k], np.zeros_like(parts['value'][k]))
        for name in ['actor', 'entropy']:
            for k in GROUPS['value']:
                np.testing.assert_array_equal(parts[name][k], np.zeros_like(parts[name][k]))
        changed = intervention_gradient(joint, parts, 'block_current_shared_value')
        self.assertGreater(float(np.linalg.norm(np.asarray(parts['value']['w']))), 1e-5)
        for k in joint:
            expected = joint[k] - parts['value'][k] if k in GROUPS['shared'] else joint[k]
            np.testing.assert_array_equal(changed[k], expected)

    def test_component_directional_derivatives(self):
        p, _, batch = fixture()
        b = validate({**batch, 'valid': np.ones(8, bool)})
        parts, _ = gradients(p, b)
        rng = np.random.default_rng(199)
        direction = {k: jnp.asarray(rng.normal(size=v.shape).astype(np.float32)) for k,v in p.items()}
        norm = np.sqrt(sum(float(jnp.sum(v*v)) for v in direction.values()))
        direction = {k: v/norm for k,v in direction.items()}
        eps = .01
        for name in ['actor', 'value', 'entropy']:
            left = {k: p[k]-eps*direction[k] for k in p}
            right = {k: p[k]+eps*direction[k] for k in p}
            numeric = (float(component(right,b,name))-float(component(left,b,name)))/(2*eps)
            analytic = sum(float(jnp.sum(parts[name][k]*direction[k])) for k in p)
            self.assertAlmostEqual(numeric, analytic, delta=1e-5)

    def test_invalid_padding_is_inert(self):
        p, _, batch = fixture()
        full = validate({**batch, 'valid': np.ones(8, bool)})
        padded = {k: np.concatenate([v, np.zeros_like(v)]) for k,v in batch.items()}
        padded['features'][8:] = np.nan
        padded['actions'][8:] = -1
        b = validate({**padded, 'valid': np.arange(16)<8})
        parts_a, _ = gradients(p, full); parts_b, _ = gradients(p, b)
        for name in parts_a:
            for k in p:
                np.testing.assert_allclose(parts_a[name][k], parts_b[name][k], atol=1e-7, rtol=1e-5)

    def test_joint_clone_preserves_original_and_inputs(self):
        p, opt, batch = fixture(1024)
        before = parameter_hash(p)
        expected, eo, metrics = optimize_batch(p, opt, batch, 301, 'matchup')
        result, states = audit_batch(p, opt, batch, 301)
        same_state(*states['joint'], expected, eo)
        self.assertEqual(result['variants']['joint']['original_metrics'], metrics)
        self.assertEqual(before, parameter_hash(p))
        self.assertEqual(int(opt[2]), 0)
        self.assertEqual(len(result['original_path_gradient_geometry']), 8)
        for _, state in states.items():
            self.assertEqual(int(state[1][2]), 8)

    def test_adam_history_not_reset(self):
        p, opt, batch = fixture(1024)
        p, opt, _ = optimize_batch(p, opt, batch, 91, 'matchup')
        b = validate({k: v[:256] for k,v in {**batch, 'valid': np.ones(1024, bool)}.items()})
        q, qo, _ = changed_kernel(p, opt, b, 'block_current_shared_value')
        zeros = {k: jnp.zeros_like(v) for k,v in p.items()}
        other, _, _ = changed_kernel(p, (zeros, zeros, jnp.int32(0)), b, 'block_current_shared_value')
        self.assertEqual(int(qo[2]), 9)
        self.assertNotEqual(parameter_hash(q), parameter_hash(other))

    def test_fixed_scope_and_anchor_grid(self):
        plan = read(PLAN)
        validate_plan(plan)
        self.assertEqual(len(anchors(plan)), 18)
        self.assertEqual(len(set(anchors(plan))), 18)
        for change in [{'cohort_indices':[127]}, {'prime_used':True}, {'seeds':[7103]}, {'evaluation_games':8}]:
            with self.assertRaises(ValueError):
                validate_plan({**plan, **change})


if __name__ == '__main__':
    unittest.main()
