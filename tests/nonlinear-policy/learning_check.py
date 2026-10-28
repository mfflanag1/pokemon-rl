"""F29 nonlinear-policy network/learner checks (12 unittest cases).

Run with the existing .venv Python; the runner places candidate/python before the source
project's python for inherited predecessors. This file only prepends its own candidate
python directory and never edits the source project.
"""
import os
import sys
import unittest

_CANDIDATE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
if _CANDIDATE not in sys.path:
    sys.path.insert(0, _CANDIDATE)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pokemon_full.policy import distribution  # noqa: E402
from pokemon_shared.learning import FLAT_SHAPES  # noqa: E402
from pokemon_matchup import network as matchup_network  # noqa: E402
from pokemon_matchup import learning as matchup_learning  # noqa: E402
from pokemon_nonlinear_policy import network, learning  # noqa: E402

CH, N_EXTRA, N_FEATURES = 25, 350, 1278
NEW = ['local_w', 'context_w', 'residual_b', 'residual_out']
OLD = list(matchup_network.SHAPES)
METRICS = {'loss', 'actor_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm'}


# --------------------------------------------------------------------------- fixtures

def matchup_parameters(seed):
    """Sealed-shape F28 predecessor with every head active (nonzero matchup residuals)."""
    rng = np.random.default_rng(seed)
    flat = {'w': rng.normal(0, .025, (928, 64)), 'b': rng.normal(0, .01, 64),
            'pw': rng.normal(0, .025, (64, 14)), 'pb': rng.normal(0, .01, 14),
            'vw': rng.normal(0, .025, 64), 'vb': np.float32(.05)}
    flat = {k: jnp.asarray(np.asarray(v, np.float32)) for k, v in flat.items()}
    p, opt = matchup_network.extend(flat)
    p = {**p, 'matchup_policy': jnp.asarray(rng.normal(0, .05, CH).astype(np.float32)),
         'matchup_value': jnp.asarray(rng.normal(0, .02, N_EXTRA).astype(np.float32))}
    matchup_network.check_state(p, opt)
    return p, opt


def make_batch(rng, n, p=None):
    """Synthetic public rows; old_logp from the given policy keeps ratio exactly 1."""
    base = rng.normal(0, 1, (n, 928)).astype(np.float32)
    mask = rng.random((n, 14)) < .6
    mask[np.arange(n), rng.integers(0, 14, n)] = True
    extra = rng.uniform(0, 1, (n, 14, CH)).astype(np.float32)
    extra[~mask] = 0
    features = np.concatenate([base, extra.reshape(n, N_EXTRA)], 1).astype(np.float32)
    actions = np.asarray([rng.choice(np.flatnonzero(m)) for m in mask], np.int32)
    if p is None:
        old_logp = rng.uniform(-3, -.1, n).astype(np.float32)
    else:
        logits, _ = matchup_network.forward(p, jnp.asarray(features))
        _, logs = distribution(logits, jnp.asarray(mask))
        old_logp = np.asarray(logs)[np.arange(n), actions].astype(np.float32)
    return {'features': features, 'mask': mask, 'actions': actions, 'old_logp': old_logp,
            'advantages': rng.normal(0, 1, n).astype(np.float32),
            'returns': rng.uniform(0, 1, n).astype(np.float32)}


def with_valid(batch, valid=None):
    n = len(batch['actions'])
    return {**batch, 'valid': np.ones(n, bool) if valid is None else valid}


def activated(q, rng, scale=1.):
    """Nonzero residual output/bias so the branch and its internal gradients are live."""
    return {**q, 'residual_out': jnp.asarray(rng.normal(0, .5 * scale, 32).astype(np.float32)),
            'residual_b': jnp.asarray(rng.normal(0, .1 * scale, 32).astype(np.float32)),
            'local_w': jnp.asarray((np.asarray(q['local_w']) * scale).astype(np.float32)),
            'context_w': jnp.asarray((np.asarray(q['context_w']) * scale).astype(np.float32))}


def np_forward(p, x, arm):
    """Independent float64 NumPy forward for both arms."""
    P = {k: np.asarray(v, np.float64) for k, v in p.items()}
    x = np.asarray(x, np.float64)
    h = np.tanh(x[:, :928] @ P['w'] + P['b'])
    extra = x[:, 928:]
    rows = extra.reshape(-1, 14, CH)
    logits = h @ P['pw'] + P['pb'] + rows @ P['matchup_policy']
    value = 1 / (1 + np.exp(-(h @ P['vw'] + P['vb'] + extra @ P['matchup_value'])))
    if arm == 'nonlinear':
        pre = rows @ P['local_w'] + (h @ P['context_w'])[:, None, :] + P['residual_b']
        logits = logits + np.tanh(pre) @ P['residual_out']
    return logits, value


def masked_softmax(logits, mask):
    z = np.where(mask, np.asarray(logits, np.float64), -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.where(mask, np.exp(z), 0)
    return e / e.sum(1, keepdims=True)


def count(shapes):
    return sum(int(np.prod(s)) for s in shapes.values())


def close(a, b, rtol, atol):
    return np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- tests

class NonlinearPolicyChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p, cls.popt = matchup_parameters(11)
        cls.q, cls.qopt = network.extend(cls.p, 29)
        cls.rng = np.random.default_rng(1234)
        cls.batch = make_batch(np.random.default_rng(5), 64, cls.p)

    # 1
    def test_api_inventory_and_parameter_counts(self):
        self.assertEqual(network.NEW_SHAPES, {'local_w': (25, 32), 'context_w': (64, 32),
                                              'residual_b': (32,), 'residual_out': (32,)})
        self.assertEqual(network.SHAPES, {**matchup_network.SHAPES, **network.NEW_SHAPES})
        self.assertEqual(count(network.NEW_SHAPES), 2912)
        self.assertEqual(count(network.SHAPES), 63718)
        self.assertEqual(count(matchup_network.SHAPES), 63718 - 2912)
        self.assertEqual(network.ARMS, ('control', 'nonlinear'))
        for arm in network.ARMS:  # both arms share one full inventory
            self.assertEqual(set(self.q), set(network.SHAPES))
            logits, value = network.forward(self.q, jnp.asarray(self.batch['features']), arm)
            self.assertEqual(logits.shape, (64, 14)); self.assertEqual(value.shape, (64,))
        for name in ['objective', 'kernel', 'update', 'optimize_batch']:
            self.assertTrue(callable(getattr(learning, name)))
        self.assertIs(learning.validate, matchup_learning.validate)

    # 2
    def test_extend_three_seeds_preserve_parameters_and_outputs(self):
        x = jnp.asarray(self.batch['features'])
        old_logits, old_value = matchup_network.forward(self.p, x)
        previous = None
        for seed in [0, 7, 2**32 - 1]:
            q, opt = network.extend(self.p, seed)
            q2, _ = network.extend(self.p, seed)
            for k in OLD:
                self.assertTrue(np.array_equal(np.asarray(q[k]), np.asarray(self.p[k])), k)
                self.assertEqual(np.asarray(q[k]).dtype, np.float32)
            ref = np.random.default_rng(seed)
            self.assertTrue(np.array_equal(np.asarray(q['local_w']), ref.normal(0, .025, (25, 32)).astype(np.float32)))
            self.assertTrue(np.array_equal(np.asarray(q['context_w']), ref.normal(0, .025, (64, 32)).astype(np.float32)))
            self.assertTrue(np.all(np.asarray(q['residual_b']) == 0))
            self.assertTrue(np.all(np.asarray(q['residual_out']) == 0))
            self.assertTrue(np.any(np.asarray(q['local_w']) != 0) and np.any(np.asarray(q['context_w']) != 0))
            for k in network.SHAPES:
                self.assertTrue(np.array_equal(np.asarray(q[k]), np.asarray(q2[k])), k)
                self.assertTrue(np.all(np.asarray(opt[0][k]) == 0) and np.all(np.asarray(opt[1][k]) == 0))
                self.assertEqual(np.asarray(opt[0][k]).dtype, np.float32)
            self.assertEqual(np.asarray(opt[2]).dtype, np.int32); self.assertEqual(int(opt[2]), 0)
            if previous is not None:
                self.assertFalse(np.array_equal(np.asarray(q['local_w']), previous))
            previous = np.asarray(q['local_w'])
            for arm in network.ARMS:  # exact function preservation
                logits, value = network.forward(q, x, arm)
                self.assertTrue(np.array_equal(np.asarray(logits), np.asarray(old_logits)), arm)
                self.assertTrue(np.array_equal(np.asarray(value), np.asarray(old_value)), arm)
            # Under jit the zero residual is still an exact-zero addend; the tolerance only
            # covers XLA fusion/reduction-order differences of a separately compiled graph.
            jl, jv = jax.jit(network.forward, static_argnames='arm')(q, x, 'nonlinear')
            self.assertTrue(close(jl, old_logits, 0., 1e-6))
            self.assertTrue(close(jv, old_value, 0., 1e-7))

    # 3
    def test_extend_rejects_invalid_predecessor_and_seed(self):
        p = self.p
        bad = [{k: v for k, v in p.items() if k != 'matchup_value'},
               {**p, 'extra_key': jnp.zeros(1, jnp.float32)},
               {**p, **{k: jnp.zeros(1, jnp.float32) for k in NEW}},  # already extended
               {**p, 'w': jnp.zeros((928, 32), jnp.float32)},
               {**p, 'pb': np.zeros(14, np.float64)},  # NumPy: jnp truncates to float32 under x64-disabled
               {**p, 'vb': jnp.asarray(np.float32('nan'))},
               {**p, 'matchup_policy': jnp.asarray(np.full(CH, np.inf, np.float32))},
               list(p.items())]
        for i, bp in enumerate(bad):
            with self.assertRaises(ValueError, msg=i):
                network.extend(bp, 1)
        for seed in [True, False, np.bool_(True), -1, 2**32, 1.0, '1', None, np.float32(2)]:
            with self.assertRaises(ValueError, msg=repr(seed)):
                network.extend(p, seed)
        for seed in [np.uint32(3), np.int64(4), 2**32 - 1]:
            network.extend(p, seed)

    # 4
    def test_check_state_rejects_invalid_inventory_and_counters(self):
        q, (m, v, step) = self.q, self.qopt
        network.check_state(q, (m, v, step))
        network.check_state(q, (m, v, jnp.int32(2**31 - 1)))
        bad_states = [
            (q, (m, v)), (q, [m, v, step, step]), (q, None),
            ({k: q[k] for k in OLD}, (m, v, step)),
            (q, ({k: m[k] for k in OLD}, v, step)),
            (q, (m, {**v, 'bogus': jnp.zeros(1, jnp.float32)}, step)),
            ({**q, 'local_w': jnp.zeros((32, 25), jnp.float32)}, (m, v, step)),
            ({**q, 'residual_out': np.zeros(32, np.float64)}, (m, v, step)),  # NumPy: jnp truncates to float32
            ({**q, 'context_w': jnp.asarray(np.full((64, 32), np.nan, np.float32))}, (m, v, step)),
            (q, ({**m, 'w': jnp.asarray(np.full((928, 64), np.inf, np.float32))}, v, step)),
            (q, (m, {**v, 'residual_b': jnp.full(32, -1e-9, jnp.float32)}, step)),
            (q, (m, v, np.int64(0))), (q, (m, v, jnp.int32(-1))), (q, (m, v, jnp.zeros(1, jnp.int32))),
            (q, (m, v, 0)), (q, (m, v, jnp.float32(0))),
            (list(q.items()), (m, v, step)), (q, (list(m.items()), v, step)),
        ]
        for i, (pp, oo) in enumerate(bad_states):
            with self.assertRaises(ValueError, msg=i):
                network.check_state(pp, oo)

    # 5
    def test_independent_numpy_forward_both_arms_and_masked_distribution(self):
        q = activated(self.q, np.random.default_rng(3))
        network.check_state(q, self.qopt)
        x = self.batch['features']; mask = self.batch['mask']
        ctrl = network.forward(q, jnp.asarray(x), 'control')
        nonl = network.forward(q, jnp.asarray(x), 'nonlinear')
        rc = np_forward(q, x, 'control'); rn = np_forward(q, x, 'nonlinear')
        self.assertTrue(close(ctrl[0], rc[0], 1e-5, 1e-5)); self.assertTrue(close(ctrl[1], rc[1], 1e-5, 1e-6))
        self.assertTrue(close(nonl[0], rn[0], 1e-5, 1e-5)); self.assertTrue(close(nonl[1], rn[1], 1e-5, 1e-6))
        # The activated branch really changes logits (and only logits).
        delta = np.asarray(nonl[0]) - np.asarray(ctrl[0])
        self.assertGreater(np.abs(delta).max(), 1e-2)
        self.assertTrue(np.array_equal(np.asarray(nonl[1]), np.asarray(ctrl[1])))
        self.assertTrue(close(delta, rn[0] - rc[0], 1e-4, 1e-5))
        # Control equals the sealed matchup forward exactly.
        ml, mv = matchup_network.forward({k: q[k] for k in OLD}, jnp.asarray(x))
        self.assertTrue(np.array_equal(np.asarray(ctrl[0]), np.asarray(ml)))
        self.assertTrue(np.array_equal(np.asarray(ctrl[1]), np.asarray(mv)))
        # Action softmax under legal masks for both arms.
        for logits in (ctrl[0], nonl[0]):
            probs, logs = distribution(logits, jnp.asarray(mask))
            probs, logs = np.asarray(probs), np.asarray(logs)
            self.assertTrue(np.all(probs[~mask] == 0) and np.all(logs[~mask] == 0))
            self.assertTrue(np.all(probs >= 0) and np.isfinite(logs).all())
            self.assertLess(np.abs(probs.sum(1) - 1).max(), 1e-5)
            self.assertTrue(close(np.exp(logs[mask]), probs[mask], 1e-4, 1e-6))
            self.assertTrue(close(probs, masked_softmax(logits, mask), 1e-2, 1e-3))

    # 6
    def test_control_update_parity_with_sealed_matchup_one_and_eight(self):
        p, popt, q, qopt = self.p, self.popt, self.q, self.qopt
        part = with_valid(self.batch)
        # Inactive branch: exact zero gradients for all four new leaves.
        (_, _), g = jax.value_and_grad(learning.objective, has_aux=True)(q, {k: jnp.asarray(v) for k, v in part.items()}, 'control')
        for k in NEW:
            self.assertTrue(np.all(np.asarray(g[k]) == 0), k)
        p1, po1, m1 = matchup_learning.update(p, popt, part)
        q1, qo1, n1 = learning.update(q, qopt, part, 'control')
        self._assert_parity(p1, po1, m1, q1, qo1, n1)
        # Eight updates through the fixed cohort schedule, same RNG order.
        cohort = make_batch(np.random.default_rng(77), 1024, p)
        p8, po8, r8 = matchup_learning.optimize_batch(p, popt, cohort, 2024, 'matchup')
        q8, qo8, s8 = learning.optimize_batch(q, qopt, cohort, 2024, 'control')
        self.assertEqual(len(r8), 8); self.assertEqual(len(s8), 8)
        self.assertEqual(int(po8[2]), 8); self.assertEqual(int(qo8[2]), 8)
        for a, b in zip(r8, s8):
            self.assertEqual(set(a), METRICS); self.assertEqual(set(b), METRICS)
            for k in METRICS:
                self.assertTrue(np.isclose(a[k], b[k], rtol=1e-5, atol=1e-6), (k, a[k], b[k]))
        self._assert_parity(p8, po8, r8[-1], q8, qo8, s8[-1])
        # Inactive branch frozen across all eight updates, moments exactly zero.
        for k in NEW:
            self.assertTrue(np.array_equal(np.asarray(q8[k]), np.asarray(q[k])), k)
            self.assertTrue(np.all(np.asarray(qo8[0][k]) == 0) and np.all(np.asarray(qo8[1][k]) == 0), k)

    def _assert_parity(self, p, po, m, q, qo, n):
        # Identical FP32 math on identical inputs; tolerance covers only XLA fusion/reduction
        # order differences between two separately compiled kernels (a few ULPs).
        for k in OLD:
            self.assertTrue(close(q[k], p[k], 1e-5, 1e-7), ('param', k))
            self.assertTrue(close(qo[0][k], po[0][k], 1e-4, 1e-9), ('m', k))
            self.assertTrue(close(qo[1][k], po[1][k], 1e-4, 1e-12), ('v', k))
        self.assertEqual(int(qo[2]), int(po[2]))
        self.assertEqual(set(m), METRICS); self.assertEqual(set(n), METRICS)
        for k in METRICS:
            self.assertTrue(np.isclose(n[k], m[k], rtol=1e-5, atol=1e-6), (k, n[k], m[k]))
        for k in NEW:
            self.assertTrue(np.all(np.asarray(qo[0][k]) == 0) and np.all(np.asarray(qo[1][k]) == 0), k)

    # 7
    def test_first_step_zero_output_gradients_and_second_step_reachability(self):
        q, qopt = self.q, self.qopt
        part = with_valid(self.batch)
        b = {k: jnp.asarray(v) for k, v in part.items()}
        grad = lambda params, arm: jax.grad(learning.objective, has_aux=True)(params, b, arm)[0]
        gn, gc = grad(q, 'nonlinear'), grad(q, 'control')
        # residual_out is zero: internal leaves and the trunk see no branch gradient yet,
        # while residual_out itself already receives a nonzero gradient.
        for k in ['local_w', 'context_w', 'residual_b']:
            self.assertTrue(np.all(np.asarray(gn[k]) == 0), k)
        self.assertGreater(np.abs(np.asarray(gn['residual_out'])).max(), 0)
        for k in OLD:
            self.assertTrue(close(gn[k], gc[k], 1e-5, 1e-8), k)
        q1, qo1, _ = learning.update(q, qopt, part, 'nonlinear')
        self.assertFalse(np.array_equal(np.asarray(q1['residual_out']), np.asarray(q['residual_out'])))
        for k in ['local_w', 'context_w', 'residual_b']:
            self.assertTrue(np.array_equal(np.asarray(q1[k]), np.asarray(q[k])), k)
            self.assertTrue(np.all(np.asarray(qo1[0][k]) == 0) and np.all(np.asarray(qo1[1][k]) == 0), k)
        self.assertTrue(np.any(np.asarray(qo1[0]['residual_out']) != 0))
        # Second step: nonzero residual_out makes every internal leaf reachable.
        g2 = grad(q1, 'nonlinear')
        for k in NEW:
            self.assertGreater(np.abs(np.asarray(g2[k])).max(), 0, k)
        q2, _, _ = learning.update(q1, qo1, part, 'nonlinear')
        for k in NEW:
            self.assertFalse(np.array_equal(np.asarray(q2[k]), np.asarray(q1[k])), k)
        # Shared trunk receives nonlinear actor gradients once the branch is live.
        self.assertFalse(close(g2['w'], grad(q1, 'control')['w'], 1e-6, 1e-9))

    # 8
    def test_finite_differences_activated_branch_and_shared_trunk(self):
        rng = np.random.default_rng(8)
        q = activated(self.q, rng)
        x = self.batch['features'][:32]
        C = rng.normal(0, 1, (32, 14)); D = rng.normal(0, 1, 32)

        def surrogate(params):  # smooth function of forward logits/value, no distribution
            logits, value = network.forward(params, jnp.asarray(x), 'nonlinear')
            return jnp.sum(logits * jnp.asarray(C, jnp.float32)) + jnp.sum(value * jnp.asarray(D, jnp.float32))

        def reference(params):
            logits, value = np_forward(params, x, 'nonlinear')
            return float(np.sum(logits * C) + np.sum(value * D))

        g = jax.grad(surrogate)(q)
        P = {k: np.asarray(v, np.float64) for k, v in q.items()}
        eps = 1e-5
        for k in ['local_w', 'context_w', 'residual_out', 'residual_b', 'w', 'b', 'pw', 'matchup_policy']:
            u = rng.normal(0, 1, P[k].shape)
            plus = {**P, k: P[k] + eps * u}; minus = {**P, k: P[k] - eps * u}
            fd = (reference(plus) - reference(minus)) / (2 * eps)
            an = float(np.sum(np.asarray(g[k], np.float64) * u))
            self.assertGreater(abs(fd), 1e-3, k)
            self.assertLessEqual(abs(fd - an), 2e-3 * max(1., abs(fd)), (k, fd, an))
        # No new value branch: the value output is independent of the new leaves.
        gv = jax.grad(lambda params: jnp.sum(network.forward(params, jnp.asarray(x), 'nonlinear')[1] * jnp.asarray(D, jnp.float32)))(q)
        for k in NEW:
            self.assertTrue(np.all(np.asarray(gv[k]) == 0), k)

    # 9
    def test_cross_difference_interaction_beyond_linear_residual(self):
        rng = np.random.default_rng(9)
        q = activated(self.q, rng, scale=8.)
        base = rng.normal(0, 1, (2, 928)).astype(np.float32)
        rows = rng.uniform(0, 1, (2, 14, CH)).astype(np.float32)
        # Four combinations of (context i, action-row features j).
        xs = np.stack([np.concatenate([base[i], rows[j].reshape(-1)]) for i in (0, 1) for j in (0, 1)]).astype(np.float32)
        for arm, expect_interaction in (('control', False), ('nonlinear', True)):
            logits, _ = network.forward(q, jnp.asarray(xs), arm)
            L = np.asarray(logits, np.float64)
            cross = L[3] - L[2] - L[1] + L[0]  # f(h1,e1)-f(h1,e0)-f(h0,e1)+f(h0,e0)
            if expect_interaction:
                self.assertGreater(np.abs(cross).max(), 1e-2)
                ref, _ = np_forward(q, xs, arm)
                self.assertTrue(close(cross, ref[3] - ref[2] - ref[1] + ref[0], 1e-3, 1e-4))
            else:  # additive h@pw + e@matchup_policy has exactly zero interaction
                self.assertLess(np.abs(cross).max(), 1e-5)

    # 10
    def test_all_invalid_padding_objective_and_gradients_zero(self):
        n = 16
        garbage = {'features': np.full((n, N_FEATURES), np.nan, np.float32),
                   'mask': np.zeros((n, 14), bool), 'actions': np.full(n, -1, np.int32),
                   'old_logp': np.full(n, np.inf, np.float32), 'advantages': np.full(n, np.nan, np.float32),
                   'returns': np.full(n, 5., np.float32), 'valid': np.zeros(n, bool)}
        b = {k: jnp.asarray(v) for k, v in garbage.items()}
        for arm in network.ARMS:
            for params in (self.q, activated(self.q, np.random.default_rng(10))):
                (loss, metrics), g = jax.value_and_grad(learning.objective, has_aux=True)(params, b, arm)
                self.assertEqual(float(loss), 0.)
                self.assertEqual(set(metrics), METRICS - {'gradient_norm'})
                for k, v in metrics.items():
                    self.assertEqual(float(v), 0., (arm, k))
                for k in network.SHAPES:
                    a = np.asarray(g[k])
                    self.assertTrue(np.isfinite(a).all() and np.all(a == 0), (arm, k))
                # Host path: validation only inspects live rows; the update is a no-op step.
                q1, qo1, m1 = learning.update(params, self.qopt, garbage, arm)
                self.assertEqual(int(qo1[2]), 1)
                self.assertEqual(m1['gradient_norm'], 0.)
                for k in network.SHAPES:
                    self.assertTrue(np.array_equal(np.asarray(q1[k]), np.asarray(params[k])), (arm, k))
        # Mixed padding: results equal the compacted valid subset.
        live = with_valid(self.batch)
        half = {k: np.concatenate([v, garbage[k][:n]]) for k, v in live.items()}
        for arm in network.ARMS:
            (l1, _), g1 = jax.value_and_grad(learning.objective, has_aux=True)(self.q, {k: jnp.asarray(v) for k, v in live.items()}, arm)
            (l2, _), g2 = jax.value_and_grad(learning.objective, has_aux=True)(self.q, {k: jnp.asarray(v) for k, v in half.items()}, arm)
            self.assertTrue(np.isclose(float(l1), float(l2), rtol=1e-5, atol=1e-7))
            for k in network.SHAPES:
                self.assertTrue(close(g1[k], g2[k], 1e-4, 1e-8), (arm, k))

    # 11
    def test_validation_rejections(self):
        q, qopt = self.q, self.qopt
        good = with_valid(self.batch)
        x = jnp.asarray(good['features'])
        for arm in ['matchup', 'zero', 'Control', None, 0, ('control',)]:
            with self.assertRaises(ValueError, msg=repr(arm)):
                network.forward(q, x, arm)
            with self.assertRaises(ValueError, msg=repr(arm)):
                learning.objective(q, {k: jnp.asarray(v) for k, v in good.items()}, arm)
            with self.assertRaises((ValueError, TypeError), msg=repr(arm)):
                learning.kernel(q, *qopt, {k: jnp.asarray(v) for k, v in good.items()}, arm)
            with self.assertRaises(ValueError, msg=repr(arm)):
                learning.update(q, qopt, good, arm)
            with self.assertRaises(ValueError, msg=repr(arm)):
                learning.optimize_batch(q, qopt, make_batch(np.random.default_rng(1), 1024), 0, arm)

        def corrupt(**changes):
            b = {k: np.array(v, copy=True) for k, v in good.items()}
            for k, f in changes.items():
                f(b[k]) if callable(f) else b.__setitem__(k, f)
            return b

        def set_illegal(actions):  # action outside its own legal mask
            m = good['mask'][0]; actions[0] = np.flatnonzero(~m)[0] if (~m).any() else 14

        bad = [corrupt(actions=set_illegal),
               corrupt(actions=lambda a: a.__setitem__(1, 14)),
               corrupt(actions=lambda a: a.__setitem__(1, -1)),
               corrupt(old_logp=lambda a: a.__setitem__(2, .5)),
               corrupt(returns=lambda a: a.__setitem__(2, 1.5)),
               corrupt(returns=lambda a: a.__setitem__(2, -.5)),
               corrupt(advantages=lambda a: a.__setitem__(3, np.nan)),
               corrupt(features=lambda a: a.__setitem__((4, 1000), np.inf)),
               corrupt(features=lambda a: a.__setitem__((4, 10), np.nan)),
               corrupt(mask=lambda a: a.__setitem__(5, False)),
               corrupt(features=good['features'][:, :928]),
               corrupt(features=good['features'].astype(np.float64)),
               corrupt(actions=good['actions'].astype(np.int64)),
               corrupt(valid=good['valid'].astype(np.int32)),
               {k: v for k, v in good.items() if k != 'valid'},
               {k: v for k, v in good.items() if k != 'features'},
               {**good, 'extra': np.zeros(64)}]
        for i, b in enumerate(bad):
            for arm in network.ARMS:
                with self.assertRaises(ValueError, msg=(i, arm)):
                    learning.update(q, qopt, b, arm)
        with self.assertRaises(ValueError):
            learning.update(q, (qopt[0], qopt[1], jnp.int32(2**31 - 1)), good, 'control')
        with self.assertRaises(ValueError):
            learning.update({k: q[k] for k in OLD}, qopt, good, 'control')
        # Control may never move the inactive leaves, even with inherited nonzero moments.
        hot = ({**qopt[0], 'residual_out': jnp.full(32, 1e-3, jnp.float32)}, qopt[1], qopt[2])
        with self.assertRaises(ValueError):
            learning.update(q, hot, good, 'control')
        learning.update(q, hot, good, 'nonlinear')
        cohort = make_batch(np.random.default_rng(2), 1024)
        with self.assertRaises(ValueError):
            learning.optimize_batch(q, qopt, {k: v[:1000] for k, v in cohort.items()}, 0, 'control')
        with self.assertRaises(ValueError):
            learning.optimize_batch(q, qopt, with_valid(cohort), 0, 'control')
        with self.assertRaises(ValueError):
            learning.optimize_batch(q, qopt, {k: v for k, v in cohort.items() if k != 'returns'}, 0, 'control')
        with self.assertRaises(ValueError):
            learning.optimize_batch(q, (qopt[0], qopt[1], jnp.int32(2**31 - 8)), cohort, 0, 'control')
        learning.update(q, (qopt[0], qopt[1], jnp.int32(2**31 - 2)), good, 'control')

    # 12
    def test_optimize_batch_schedule_rng_order_and_arm_behaviour(self):
        q, qopt = self.q, self.qopt
        cohort = make_batch(np.random.default_rng(12), 1024, self.p)
        for arm in network.ARMS:
            qa, oa, ra = learning.optimize_batch(q, qopt, cohort, 99, arm)
            qb, ob, rb = learning.optimize_batch(q, qopt, cohort, 99, arm)
            self.assertEqual(len(ra), 8); self.assertEqual(int(oa[2]), 8)
            self.assertEqual(ra, rb)
            for k in network.SHAPES:
                self.assertTrue(np.array_equal(np.asarray(qa[k]), np.asarray(qb[k])), (arm, k))
                self.assertTrue(np.isfinite(np.asarray(qa[k])).all())
            # Manual replay: two epochs of one seeded permutation stream, 256-row minibatches.
            rng = np.random.default_rng(99); qm, om = q, qopt; replay = []
            for _ in range(2):
                order = rng.permutation(1024)
                for first in range(0, 1024, 256):
                    ids = order[first:first + 256]
                    qm, om, mm = learning.update(qm, om, with_valid({k: v[ids] for k, v in cohort.items()}), arm)
                    replay.append(mm)
            self.assertEqual(replay, ra)
            for k in network.SHAPES:
                self.assertTrue(np.array_equal(np.asarray(qm[k]), np.asarray(qa[k])), (arm, k))
            if arm == 'control':
                for k in NEW:
                    self.assertTrue(np.array_equal(np.asarray(qa[k]), np.asarray(q[k])), k)
            else:
                for k in NEW:  # residual_out moves at step 1, the rest from step 2 on
                    self.assertFalse(np.array_equal(np.asarray(qa[k]), np.asarray(q[k])), k)
                self.assertNotEqual(ra, learning.optimize_batch(q, qopt, cohort, 99, 'control')[2])
        # A different seed changes the minibatch order and therefore the trajectory.
        _, _, rc = learning.optimize_batch(q, qopt, cohort, 100, 'control')
        _, _, r99 = learning.optimize_batch(q, qopt, cohort, 99, 'control')
        self.assertNotEqual(rc, r99)


if __name__ == '__main__':
    unittest.main(verbosity=2)
