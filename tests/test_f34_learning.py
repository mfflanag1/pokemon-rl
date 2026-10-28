"""Focused structured learner qualification; synthetic rows are never training data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full.policy import distribution
from pokemon_nonlinear_policy.network import forward as host_forward
from pokemon_nonlinear_policy.actor import infer as host_infer
from pokemon_structured_policy import network, learning, actor, checkpoint
from pokemon_structured_policy.features import (
    CATEGORICAL_SIZES, FIELD_SHAPES, FIELD_SLICES, FEATURES, empty_history)


def packed_fixture(seed=34, n=8):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, FEATURES), np.float32)
    x[:, :1278] = rng.normal(0, .15, (n, 1278))
    def field(name):
        return x[:, FIELD_SLICES[name]].reshape((n, *FIELD_SHAPES[name]))
    for name, domain in learning.CATEGORY_FIELDS.items():
        a = field(name)
        a[:] = rng.integers(2, CATEGORICAL_SIZES[domain], size=a.shape)
    mon = field('mon_numeric')
    mon[:] = rng.uniform(0, .5, mon.shape)
    mon[:, :, 0] = 1
    mon[:, :6, 1], mon[:, 6:, 1] = 1, 0
    mon[:, :, 2] = 0
    mon[:, 0, 2], mon[:, 6, 2] = 1, 1
    field('action_numeric')[:] = rng.uniform(0, 1, (n, 14, 8))
    for i, domain in enumerate(network.EVENT_FIELDS):
        field('event_ids')[:, :, i] = rng.integers(0, CATEGORICAL_SIZES[domain], (n, 32))
    field('event_numeric')[:] = rng.uniform(0, .5, (n, 32, 4))
    field('event_mask')[:, :6] = 1
    field('public_move_counts')[:, :, 15] = np.log1p(3)
    return x


def fresh_batch(parameters, seed=34, n=8):
    rng = np.random.default_rng(seed)
    x = packed_fixture(seed, n)
    mask = rng.random((n, 14)) > .4
    mask[:, 0] = True
    actions = np.asarray([rng.choice(np.flatnonzero(m)) for m in mask], np.int32)
    logits, _ = network.forward(parameters, jnp.asarray(x))
    _, logs = distribution(logits, jnp.asarray(mask))
    return dict(features=x, mask=mask, actions=actions,
        old_logp=np.asarray(logs)[np.arange(n), actions].astype(np.float32),
        advantages=rng.normal(0, .5, n).astype(np.float32),
        returns=rng.uniform(0, 1, n).astype(np.float32), valid=np.ones(n, bool))


def player_fixture():
    own = [{'ident': 'Own' + str(i), 'species': 'Pikachu', 'level': 80,
            'condition': '100/100', 'active': i == 0,
            'moves': ['thunderbolt', 'protect', 'surf', 'nastyplot'],
            'ability': 'static', 'item': 'leftovers', 'tera': None, 'boosts': {},
            'types': ['Electric'], 'tera_type': 'Electric',
            'stats': {k: 100 for k in ['atk', 'def', 'spa', 'spd', 'spe']}, 'reviving': False}
           for i in range(6)]
    foe = {'ident': 'Foe', 'species': 'Raichu', 'level': 80, 'condition': '80/100',
           'active': True, 'moves': ['thunderbolt'], 'ability': None, 'item': None,
           'tera': None, 'boosts': {}, 'types': None}
    active = [dict(id=m, pp=8, maxpp=16, disabled=False, base_power=p, type=t, category=c)
              for m, p, t, c in [('thunderbolt', 90, 'Electric', 'Special'),
                  ('protect', 0, 'Normal', 'Status'), ('surf', 90, 'Water', 'Special'),
                  ('nastyplot', 0, 'Dark', 'Status')]]
    return dict(schema_version='full-showdown-player-v2', side='p1', request_id=1, turn=1,
                phase='move', own=own, opponent=[foe] + [None] * 5, active_moves=active,
                weather=None, terrain=None, side_conditions={'own': {}, 'opponent': {}},
                mask=[True] * 8 + [False] + [True] * 5)


class StructuredLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hosts = {seed: checkpoint.load_f32(seed) for seed in (7103, 8209, 9311)}
        cls.parameters, cls.optimizer = network.extend(cls.hosts[7103][0], 34007103)

    def assert_state_equal(self, p, opt, q, qo):
        for left, right in [(p, q), (opt[0], qo[0]), (opt[1], qo[1])]:
            self.assertEqual(set(left), set(right))
            for key in left:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        self.assertEqual(int(opt[2]), int(qo[2]))

    def test_real_f32_initial_predictions_and_sampler_match_all_seeds(self):
        x = packed_fixture(n=8)
        mask = np.random.default_rng(35).random((8, 14)) > .5
        mask[:6, 0] = True
        mask[6:] = False
        seeds = np.arange(8, dtype=np.uint32)
        counters = seeds * 3
        self.assertEqual(network.PARAMETER_COUNT, 261559)
        for seed, (host, _, _) in self.hosts.items():
            p, opt = network.extend(host, seed)
            expected = host_forward(host, jnp.asarray(x[:, :1278]), 'nonlinear')
            actual = network.forward(p, jnp.asarray(x))
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            expected = host_infer(host, x[:, :1278], mask, seeds, counters, 'nonlinear')
            actual = actor.infer(p, x, mask, seeds, counters)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(int(opt[2]), 0)
            self.assertTrue(all(np.count_nonzero(a) == 0 for group in opt[:2] for a in group.values()))

    def test_batched_player_history_actor_handles_wait_and_padding(self):
        player = player_fixture()
        waiting = {**copy.deepcopy(player), 'phase': 'wait', 'mask': [False] * 14, 'active_moves': []}
        output = actor.batched_actor(self.parameters, [player, waiting], [empty_history('p1')] * 2,
            [71, 72], [0, 0], 4, self.hosts[7103][1])
        self.assertEqual(output[0][0].shape, (FEATURES,))
        self.assertTrue(output[0][1][output[0][2]])
        self.assertEqual(output[1][2:4], (-1, 0.))
        self.assertTrue(all(0 <= row[4] <= 1 for row in output))
        with self.assertRaisesRegex(ValueError, 'Foreign'):
            actor.batched_actor(self.parameters, [player], [empty_history('p2')], [71], [0], 1, self.hosts[7103][1])
        with self.assertRaises(ValueError):
            actor.batched_actor(self.parameters, [player], [empty_history('p1')], [True], [0], 1, self.hosts[7103][1])

    def test_fresh_updates_activate_internal_branch_and_recovery_is_exact(self):
        p, opt = self.parameters, self.optimizer
        first = fresh_batch(p, 41)
        q, qo, metrics = learning.update(p, opt, first)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertGreater(float(np.linalg.norm(q['sp_policy_out'])), 0)
        self.assertGreater(float(np.linalg.norm(q['sp_value_out'])), 0)
        np.testing.assert_array_equal(p['sp_species'], q['sp_species'])
        self.assertFalse(np.array_equal(p['w'], q['w']))
        second = fresh_batch(q, 42)
        gradients = jax.grad(lambda params: learning.objective(params, learning.validate(second))[0])(q)
        self.assertGreater(float(np.linalg.norm(gradients['sp_species'])), 1e-10)
        self.assertGreater(float(np.linalg.norm(gradients['sp_history_w'])), 1e-10)
        metadata = dict(input_hash='a' * 64, seed=7103, cohorts=0, next_episode=8)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint.save(tmp, q, qo, metadata)
            restored, ro, saved = checkpoint.load(tmp, metadata)
            self.assert_state_equal(q, qo, restored, ro)
            self.assertEqual(saved['optimizer_step'], 1)
            final, fo, m1 = learning.update(q, qo, second)
            resumed, so, m2 = learning.update(restored, ro, second)
            self.assert_state_equal(final, fo, resumed, so)
            self.assertEqual(m1, m2)
            self.assertFalse(np.array_equal(q['sp_species'], final['sp_species']))
            with self.assertRaisesRegex(ValueError, 'run identity'):
                checkpoint.load(tmp, {**metadata, 'seed': 8209})
            pointer = json.loads((Path(tmp) / 'latest.json').read_text())
            path = Path(tmp) / pointer['file']
            with path.open('ab') as file:
                file.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'digest'):
                checkpoint.load(tmp)

    def test_event_branch_affects_actor_and_value_after_activation(self):
        p = {**self.parameters, 'sp_policy_out': jnp.linspace(-.4, .4, network.ACTION_WIDTH),
             'sp_value_out': jnp.linspace(-.4, .4, network.CONTEXT_WIDTH)}
        x = packed_fixture(n=8)
        altered = x.copy()
        altered[:, FIELD_SLICES['event_mask']] = 0
        before = network.forward(p, jnp.asarray(x))
        after = network.forward(p, jnp.asarray(altered))
        self.assertGreater(float(np.linalg.norm(before[0] - after[0])), 1e-5)
        self.assertGreater(float(np.linalg.norm(before[1] - after[1])), 1e-6)

    def test_move_count_context_is_a_weighted_average_even_below_unit_weight(self):
        low = packed_fixture(n=2)
        counts = low[:, FIELD_SLICES['public_move_counts']].reshape(2, 2, -1)
        counts[:] = 0
        counts[:, :, 15] = .2
        high = low.copy()
        high[:, FIELD_SLICES['public_move_counts']] *= 3
        hidden = jnp.zeros((2, 64), jnp.float32)
        _, a = network.structured_context(self.parameters, jnp.asarray(low), hidden)
        _, b = network.structured_context(self.parameters, jnp.asarray(high), hidden)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        low[:, FIELD_SLICES['public_move_counts']] = 0
        _, empty = network.structured_context(self.parameters, jnp.asarray(low), hidden)
        self.assertTrue(np.isfinite(np.asarray(empty)).all())

    def test_categorical_batch_mask_and_target_validation(self):
        original = fresh_batch(self.parameters, 51)
        for field, bad in [('species', CATEGORICAL_SIZES['species']), ('moves', -1),
                           ('ability', .5), ('event_mask', 2)]:
            batch = copy.deepcopy(original)
            batch['features'][0, FIELD_SLICES[field].start] = bad
            with self.assertRaises(ValueError):
                learning.validate(batch)
        for key, bad in [('old_logp', 1.), ('returns', 1.1), ('actions', 14)]:
            batch = copy.deepcopy(original)
            batch[key][0] = bad
            with self.assertRaises(ValueError):
                learning.validate(batch)
        batch = copy.deepcopy(original)
        batch['mask'][0] = False
        with self.assertRaises(ValueError):
            learning.validate(batch)
        batch = copy.deepcopy(original)
        batch['features'] = batch['features'].astype(np.float64)
        with self.assertRaises(ValueError):
            learning.validate(batch)

    def test_padding_is_sanitized_and_empty_update_does_not_advance(self):
        batch = fresh_batch(self.parameters, 61)
        batch['valid'][:] = False
        for key in ['features', 'old_logp', 'advantages', 'returns']:
            batch[key][:] = np.nan
        batch['mask'][:] = False
        batch['actions'][:] = -1
        p, opt, metrics = learning.update(self.parameters, self.optimizer, batch)
        self.assert_state_equal(p, opt, self.parameters, self.optimizer)
        self.assertTrue(all(v == 0 for v in metrics.values()))
        batch = fresh_batch(self.parameters, 62)
        batch['valid'][-1] = False
        for key in ['features', 'old_logp', 'advantages', 'returns']:
            batch[key][-1] = np.nan
        batch['mask'][-1] = False
        batch['actions'][-1] = -1
        b = learning.validate(batch)
        value, grads = jax.value_and_grad(lambda p: learning.objective(p, b)[0])(self.parameters)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in grads.values()))


if __name__ == '__main__':
    unittest.main()
