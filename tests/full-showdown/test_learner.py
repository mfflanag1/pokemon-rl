import copy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode, FEATURES
from pokemon_full.policy import initialize, act, distribution, select, update, finite_parameters
from pokemon_full.learner import build_batch, finish_rows, parameter_hash
from pokemon_full.learner import collect_episode
from pokemon_full.client import NodeWorker


def player():
    mon = {'ident': 'Pikachu', 'species': 'Pikachu', 'level': 80, 'condition': '100/200',
           'active': True, 'moves': ['tackle'], 'ability': 'static', 'item': 'lightball',
           'tera': None, 'tera_type': 'Water', 'stats': {'atk': 100}, 'reviving': False, 'boosts': {}}
    return {'schema_version': 'full-showdown-player-v1', 'side': 'p1', 'request_id': 1, 'turn': 1,
            'phase': 'move', 'own': [copy.deepcopy(mon) for _ in range(6)], 'opponent': [None]*6,
            'active_moves': [{'id': 'tackle', 'pp': 35, 'maxpp': 35, 'disabled': False,
                              'base_power': 40, 'type': 'Normal', 'category': 'Physical'}],
            'weather': None, 'terrain': None, 'side_conditions': {'own': {}, 'opponent': {}},
            'mask': [True]*14}


def row(value=.5, trajectory=0):
    return {'features': np.zeros(FEATURES, np.float32), 'mask': np.array([True]*14),
            'action': 0, 'old_logp': -np.log(14), 'value': value, 'reward': 0.,
            'next_index': -1, 'bootstrap': 0., 'terminal': False, 'trajectory': trajectory}


class Features(unittest.TestCase):
    def test_only_player_projection_and_public_sensitivity(self):
        p = player(); x, mask = encode(p)
        self.assertEqual(x.shape, (FEATURES,)); self.assertEqual(mask.shape, (14,))
        with self.assertRaisesRegex(ValueError, 'projection'):
            encode({'players': {'p1': p}, 'provenance': {'hidden_team': 'secret'}})
        with self.assertRaisesRegex(ValueError, 'projection'):
            encode({**p, 'hidden_team': 'secret'})
        q = copy.deepcopy(p); q['opponent'][0] = {'ident': 'Charizard', 'species': 'Charizard',
            'level': 80, 'condition': '100/100', 'active': True, 'moves': [], 'ability': None,
            'item': None, 'tera': None, 'boosts': {}, 'types': None}
        self.assertFalse(np.array_equal(x, encode(q)[0]))
        r = copy.deepcopy(q); r['opponent'][0]['condition'] = '50/100'
        self.assertFalse(np.array_equal(encode(q)[0], encode(r)[0]))
        r['opponent'][0]['stats'] = {'atk': 999}
        with self.assertRaisesRegex(ValueError, 'public fields'):
            encode(r)

    def test_wait_mask_never_selects_action(self):
        p = player(); p['phase'] = 'wait'; p['mask'] = [False]*14
        x, mask = encode(p)
        actions, logs, values = act(initialize(31), jnp.asarray(x[None]), jnp.asarray(mask[None]), jax.random.split(jax.random.key(2), 1))
        self.assertEqual(int(actions[0]), -1); self.assertEqual(float(logs[0]), 0.)
        self.assertTrue(0 <= float(values[0]) <= 1)


class Links(unittest.TestCase):
    def test_forced_switch_links_skip_other_player_waits(self):
        rows = [row(.2, 0), row(.7, 1), row(.3, 0), row(.4, 0), row(.6, 1)]
        rows[0]['next_index'] = 2; rows[2]['next_index'] = 3; rows[1]['next_index'] = 4
        finish_rows(rows, {'p1': 3, 'p2': 4}, {'kind': 'terminal', 'rewards': {'p1': 1., 'p2': 0.}}, [99, 99])
        batch = build_batch(rows)
        expected = np.array([.1 + .95*(.1 + .95*.6), -.1 + .95*(-.6), .1+.95*.6, .6, -.6])
        np.testing.assert_allclose(np.asarray(batch['advantages']), expected, atol=2e-7)
        self.assertEqual(sum(r['reward'] for r in rows), 1.)
        finish_rows(rows, {'p1': 3, 'p2': 4}, {'kind': 'terminal', 'rewards': {'p1': 1., 'p2': 0.}}, [99, 99])
        self.assertEqual(sum(r['reward'] for r in rows), 1.)

    def test_cutoff_bootstraps_without_terminal_or_draw_reward(self):
        rows = [row(.2, 0), row(.7, 1)]
        finish_rows(rows, {'p1': 0, 'p2': 1}, {'kind': 'administrative_cutoff', 'rewards': {'p1': 0, 'p2': 0}}, [.8, .4])
        batch = build_batch(rows)
        np.testing.assert_allclose(batch['returns'], [.8, .4], atol=1e-7)
        self.assertTrue(all(not r['terminal'] and r['reward'] == 0 for r in rows))
        rows[0]['next_index'] = 1; rows[0]['bootstrap'] = 0
        with self.assertRaisesRegex(ValueError, 'Cross-player'):
            build_batch(rows)
        self.assertIsNone(build_batch([]))


class Policy(unittest.TestCase):
    def test_tree_law_masks_forced_actions_and_rare_tails(self):
        logits = jnp.zeros((3, 14), jnp.float32).at[1, 0].set(1000)
        mask = jnp.ones((3, 14), bool).at[2].set(False).at[2, 13].set(True)
        p, logs = distribution(logits, mask)
        np.testing.assert_allclose(np.asarray(p).sum(1), 1., atol=1e-6)
        np.testing.assert_allclose(np.exp(np.asarray(logs)[mask]), np.asarray(p)[mask], rtol=3e-6)
        self.assertTrue(np.all(np.asarray(p[1]) > 0))
        words = jnp.zeros((3, 15, 2), jnp.uint32)
        actions = np.asarray(select(logits, mask, words))
        self.assertEqual(actions[2], 13)
        self.assertTrue(np.asarray(mask)[np.arange(3), actions].all())

    def test_real_gradient_update_changes_parameters_and_stays_finite(self):
        params = initialize(71)
        x, mask = encode(player())
        features = jnp.asarray(np.stack([x, x*.5, x, x*.5]))
        masks = jnp.asarray(np.stack([mask]*4))
        actions, logs, values = act(params, features, masks, jax.random.split(jax.random.key(19), 4))
        batch = {'features': features, 'mask': masks, 'actions': actions, 'old_logp': logs,
                 'advantages': jnp.array([.5, -.5, .5, -.5], jnp.float32),
                 'returns': jnp.array([1., 0., 1., 0.], jnp.float32)}
        zeros = {k: jnp.zeros_like(v) for k, v in params.items()}
        new, m, v, step, metrics = update(params, zeros, zeros, jnp.int32(0), batch)
        finite_parameters(new)
        self.assertNotEqual(parameter_hash(params), parameter_hash(new))
        self.assertEqual(int(step), 1); self.assertGreater(float(metrics['gradient_norm']), 0.)
        self.assertTrue(all(np.isfinite(float(v)) for v in metrics.values()))


class WireIntegration(unittest.TestCase):
    def test_real_recharge_projection_encodes_and_forces_only_request_move_one(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(['node', 'tests/full-showdown/recharge-fixture.mjs'],
                                cwd=root, check=True, capture_output=True, text=True, timeout=15)
        tape = json.loads(result.stdout)
        p = tape['recharge']['players']['p1']
        self.assertEqual(p['active_moves'][0]['id'], 'recharge')
        self.assertIsNone(p['active_moves'][0]['pp'])
        features, mask = encode(p)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(np.flatnonzero(mask), [0])
        actions, logs, values = act(initialize(73), jnp.asarray(features[None]),
                                   jnp.asarray(mask[None]), jax.random.split(jax.random.key(73), 1))
        self.assertEqual(int(actions[0]), 0)
        self.assertEqual(float(logs[0]), 0.)
        self.assertTrue(np.isfinite(np.asarray(values)).all())
        self.assertEqual(tape['resumed']['players']['p1']['active_moves'][0]['id'], 'hyperbeam')

    def test_real_trapping_replies_retain_rejected_row_and_same_player_link(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(['node', 'tests/full-showdown/trapped-fixture.mjs'],
                                cwd=root, check=True, capture_output=True, text=True, timeout=15)
        tape = json.loads(result.stdout)
        calls = []
        case = self

        class ReferenceTapeWorker:
            def request(self, operation, **arguments):
                calls.append(operation)
                if operation == 'start':
                    case.assertEqual(len(calls), 1)
                    return copy.deepcopy(tape['initial'])
                if operation == 'step':
                    step = calls.count('step') - 1
                    case.assertEqual(arguments, tape['commands'][step])
                    return copy.deepcopy(tape[['rejected', 'continued'][step]])
                case.assertEqual(operation, 'cutoff')
                return copy.deepcopy(tape['cutoff'])

        # Script only action selection; the observations, request IDs, rejected
        # feedback and continuation above all came from the real pinned engine.
        selected = iter([[9, 0], [0, -1], [0, 0]])

        def scripted_predict(parameters, frame, run_seed, counters):
            encoded = [encode(frame['players'][side]) for side in ['p1', 'p2']]
            features = np.stack([item[0] for item in encoded])
            masks = np.stack([item[1] for item in encoded])
            actions = np.asarray(next(selected))
            for index, action in enumerate(actions):
                self.assertTrue(masks[index, action] if action >= 0 else not masks[index].any())
            logs = -np.log(np.maximum(masks.sum(axis=1), 1)).astype(np.float32)
            return features, masks, actions, logs, np.full(2, .5, np.float32)

        rows = []
        with patch('pokemon_full.learner.predict', side_effect=scripted_predict):
            report = collect_episode(ReferenceTapeWorker(), {}, 73, 0, rows, max_turns=2)
        self.assertEqual(calls, ['start', 'step', 'step', 'cutoff'])
        self.assertEqual(report['unavailable_choices'], 1)
        self.assertEqual(report['decisions'], 3)
        self.assertEqual([r['trajectory'] for r in rows], [0, 1, 0])
        self.assertEqual([r['next_index'] for r in rows], [2, -1, -1])
        self.assertEqual([r['action'] for r in rows], [9, 0, 0])
        self.assertTrue(rows[0]['mask'][9])
        self.assertFalse(rows[2]['mask'][9])
        self.assertTrue(all(not r['terminal'] and r['reward'] == 0 for r in rows))
        np.testing.assert_allclose(build_batch(rows)['returns'], .5, atol=1e-7)

    def test_pinned_jsonl_one_turn_collection_and_cutoff(self):
        rows = []
        with NodeWorker() as worker:
            report = collect_episode(worker, initialize(73), 73, 0, rows, max_turns=2)
        self.assertEqual(report['outcome']['kind'], 'administrative_cutoff')
        self.assertGreaterEqual(report['decisions'], 2)
        self.assertEqual({r['trajectory'] for r in rows}, {0, 1})
        self.assertTrue(all(not r['terminal'] and r['reward'] == 0 for r in rows))
        self.assertTrue(np.all(np.isfinite(np.asarray(build_batch(rows)['returns']))))

    def test_optional_curriculum_prefix_is_not_a_training_decision(self):
        with NodeWorker() as worker:
            frame = worker.request('start', run_seed=73, episode_index=0, curriculum=True)
            self.assertGreaterEqual(frame['submissions'], 3)
            self.assertFalse(frame['provenance']['scenario']['prefix_actions_trainable'])
            self.assertTrue(frame['provenance']['prefix_evidence'])
            for side in ['p1', 'p2']:
                features, _ = encode(frame['players'][side])
                self.assertEqual(features.shape, (FEATURES,))


if __name__ == '__main__':
    unittest.main()
