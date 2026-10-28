"""Twelve behavioral checks for the public-feature shared JAX network."""

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_shared.network import SHAPES, forward, initialize


class NetworkCheck(unittest.TestCase):
    def setUp(self):
        self.p = initialize(17)
        rng = np.random.default_rng(91)
        self.x = rng.normal(size=(3, 928)).astype(np.float32)
        mons = self.x[:, :768].reshape(3, 12, 64)
        mons[:, :, 0] = 1
        mons[:, :, 1] = 0
        mons[:, 2, 1] = 1
        mons[:, 10, 1] = 1

    def close(self, a, b):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)

    def permute(self, own=None, opponent=None, moves=None):
        x = self.x.copy()
        mons = x[:, :768].reshape(3, 12, 64)
        source = mons.copy()
        if own is not None:
            mons[:, :6] = source[:, own]
        if opponent is not None:
            mons[:, 6:] = source[:, 6 + np.asarray(opponent)]
        if moves is not None:
            x[:, 768:864] = self.x[:, 768:864].reshape(3, 4, 24)[:, moves].reshape(3, 96)
        return x

    def permutation_check(self, own=None, opponent=None, moves=None):
        p = dict(self.p, value_w=jnp.linspace(-.2, .3, 160))
        before, value = forward(p, self.x)
        changed = self.permute(own, opponent, moves)
        self.assertFalse(np.array_equal(changed, self.x))
        after, new_value = forward(p, changed)
        order = np.arange(14)
        if own is not None:
            order[8:] = 8 + np.asarray(own)
        if moves is not None:
            order[:4] = moves
            order[4:8] = 4 + np.asarray(moves)
        self.assertGreater(float(jnp.ptp(before)), 1e-5)
        self.close(after, before[:, order])
        self.close(value, new_value)

    def test_01_initialization_schema(self):
        expected = {"mon_w": (64, 32), "mon_b": (32,),
                    "move_w": (24, 32), "move_b": (32,),
                    "context_w": (64, 32), "context_b": (32,),
                    "score_w": (195, 281), "score_b": (281,),
                    "logit_w": (281,), "logit_b": (),
                    "value_w": (160,), "value_b": ()}
        self.assertEqual(SHAPES, expected)
        self.assertEqual(set(self.p), set(expected))
        self.assertEqual(sum(v.size for v in self.p.values()), 60479)
        self.assertEqual(sum(v.size for k, v in self.p.items()
                             if not k.startswith("value_")), 60318)
        again = initialize(17)
        names = ["mon_w", "move_w", "context_w", "score_w", "logit_w"]
        keys = jax.random.split(jax.random.key(17), 5)
        for name, shape in expected.items():
            self.assertIsInstance(self.p[name], jax.Array)
            self.assertEqual(self.p[name].shape, shape)
            self.assertEqual(self.p[name].dtype, jnp.float32)
            np.testing.assert_array_equal(self.p[name], again[name])
            if name not in names:
                np.testing.assert_array_equal(self.p[name], np.zeros(shape))
        for name, key in zip(names, keys):
            np.testing.assert_array_equal(self.p[name],
                jax.random.normal(key, expected[name], dtype=jnp.float32) * jnp.float32(.025))
        self.assertFalse(np.array_equal(self.p["mon_w"], initialize(18)["mon_w"]))

    def test_02_output_shapes_and_bounds(self):
        for n in (1, 3):
            logits, values = forward(self.p, self.x[:n])
            self.assertEqual(logits.shape, (n, 14))
            self.assertEqual(values.shape, (n,))
            self.assertTrue(np.isfinite(logits).all() and np.isfinite(values).all())
            self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_03_zero_weights(self):
        p = {k: jnp.zeros(s, jnp.float32) for k, s in SHAPES.items()}
        logits, values = forward(p, self.x)
        np.testing.assert_array_equal(logits, np.zeros((3, 14)))
        np.testing.assert_array_equal(values, np.full(3, .5))

    def test_04_own_permutation(self):
        self.permutation_check(own=[4, 2, 0, 5, 1, 3])

    def test_05_opponent_permutation(self):
        self.permutation_check(opponent=[5, 3, 1, 0, 4, 2])

    def test_06_move_permutation(self):
        self.permutation_check(moves=[2, 0, 3, 1])

    def test_07_combined_permutation(self):
        self.permutation_check(own=[5, 0, 3, 1, 2, 4],
                               opponent=[2, 4, 0, 5, 3, 1], moves=[3, 2, 0, 1])

    def test_08_absent_padding_bias(self):
        p = dict(self.p, mon_b=jnp.full((32,), .7),
                 value_w=jnp.linspace(-.1, .4, 160))
        x = self.x.copy()
        mons = x[:, :768].reshape(3, 12, 64)
        mons[:, 6:, 0] = 0
        baseline = forward(p, x)
        mons[:, 6:, 1:] = 19
        changed = forward(p, x)
        for a, b in zip(baseline, changed):
            self.close(a, b)
        # With no opponents, opponent mean/active must be zero even with bias.
        q = {k: jnp.zeros(s, jnp.float32) for k, s in SHAPES.items()}
        q.update(mon_b=p["mon_b"], value_w=jnp.zeros(160).at[32:64].set(1)
                 .at[96:128].set(1), score_w=jnp.zeros((195, 281))
                 .at[32:64, 0].set(1).at[96:128, 0].set(1),
                 logit_w=jnp.zeros(281).at[0].set(1))
        logits, values = forward(q, x)
        self.close(logits, np.zeros((3, 14)))
        self.close(values, np.full(3, .5))

    def test_09_active_selection(self):
        p = {k: jnp.zeros(s, jnp.float32) for k, s in SHAPES.items()}
        p.update(mon_w=jnp.zeros((64, 32)).at[2, 0].set(1),
                 score_w=jnp.zeros((195, 281)).at[64, 0].set(1)
                 .at[96, 0].set(2), logit_w=jnp.zeros(281).at[0].set(1),
                 value_w=jnp.zeros(160).at[64].set(1).at[96].set(-1))
        x = np.zeros((1, 928), np.float32)
        mons = x[:, :768].reshape(1, 12, 64)
        mons[:, :, 0] = 1
        mons[0, :, 2] = np.linspace(-.6, .8, 12)
        mons[0, [2, 10], 1] = 1
        for own, opp in ((2, 10), (4, 7)):
            mons[:, :, 1] = 0
            mons[0, [own, opp], 1] = 1
            a, b = np.tanh(mons[0, [own, opp], 2])
            logits, values = forward(p, x)
            self.close(logits, np.full((1, 14), np.tanh(a + 2*b)))
            self.close(values, [1 / (1 + np.exp(-(a-b)))])

    def test_10_hand_scoring(self):
        p = {k: jnp.zeros(s, jnp.float32) for k, s in SHAPES.items()}
        p.update(mon_w=jnp.zeros((64, 32)).at[2, 0].set(.7),
                 move_w=jnp.zeros((24, 32)).at[0, 0].set(-.4),
                 context_w=jnp.zeros((64, 32)).at[0, 0].set(.3),
                 score_w=jnp.zeros((195, 281)).at[0, 0].set(.2)
                 .at[128, 0].set(.4).at[160, 0].set(.8)
                 .at[192:195, 0].set(jnp.array([-.3, .1, .6])),
                 score_b=jnp.zeros(281).at[0].set(.15),
                 logit_w=jnp.zeros(281).at[0].set(1.7), logit_b=jnp.array(-.2),
                 value_w=jnp.zeros(160).at[0].set(.3).at[128].set(-.2),
                 value_b=jnp.array(.1))
        mon = np.tanh(.7 * self.x[:, :768].reshape(3, 12, 64)[:, :, 2])
        move = np.tanh(-.4 * self.x[:, 768:864].reshape(3, 4, 24)[:, :, 0])
        context = np.tanh(.3 * self.x[:, 864])
        candidates = np.concatenate((move, move, mon[:, :6]), axis=1)
        expected = 1.7 * np.tanh(.2*mon[:, :6].mean(1)[:, None]
                    + .4*context[:, None] + .8*candidates
                    + np.array([-.3]*4 + [.1]*4 + [.6]*6) + .15) - .2
        logits, values = forward(p, self.x)
        self.close(logits, expected)
        self.close(values, 1/(1+np.exp(-(.3*mon[:, :6].mean(1)-.2*context+.1))))
        self.assertGreater(float(np.ptp(expected)), .1)

    def test_11_policy_gradients(self):
        def loss(p, x):
            logits, _ = forward(p, x)
            return jnp.sum(logits * jnp.arange(1, 15))
        grads, input_grad = jax.grad(loss, argnums=(0, 1))(self.p, jnp.asarray(self.x))
        for name, gradient in grads.items():
            self.assertTrue(np.isfinite(gradient).all(), name)
            if name.startswith("value_"):
                np.testing.assert_array_equal(gradient, np.zeros(SHAPES[name]))
            else:
                self.assertGreater(float(jnp.linalg.norm(gradient)), 0, name)
        self.assertTrue(np.isfinite(input_grad).all())
        self.assertGreater(float(jnp.linalg.norm(input_grad)), 0)

    def test_12_no_mutation(self):
        original = self.x.copy()
        parameters = {k: np.array(v, copy=True) for k, v in self.p.items()}
        forward(self.p, self.x)
        np.testing.assert_array_equal(self.x, original)
        self.assertEqual(set(self.p), set(parameters))
        for name in parameters:
            np.testing.assert_array_equal(self.p[name], parameters[name])


if __name__ == "__main__":
    unittest.main()
