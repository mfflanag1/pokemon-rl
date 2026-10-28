"""F34 collector linkage checks with synthetic transport/actor outputs only.

No battles, model optimization, external opponent search, or scientific data are
launched. The real collector, episode admission, counters, and cleanup execute.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from pokemon_structured_policy import collector, learning
from pokemon_structured_policy.features import FEATURES
from pokemon_full.policy import distribution
from pokemon_nonlinear_policy.learning import validate as predecessor_validate
from pokemon_full_train.storage import save_checkpoint

SIDES = ('p1', 'p2')


class SyntheticWorker:
    """Three controlled boundaries, including a waiting p2, then a terminal."""
    def __init__(self, evidence):
        self.requests = []
        self.close_calls = 0
        self.index = 0
        self.last = None

    def frame(self):
        terminal = self.index == 3
        players = {}
        for side in SIDES:
            waiting = terminal or (self.index == 1 and side == 'p2')
            players[side] = {'side': side, 'request_id': self.index + 1,
                'turn': self.index + 1, 'phase': 'wait' if waiting else 'move',
                'mask': [not waiting] * 14}
        return {'players': players, 'observable_history': {side: {'side': side} for side in SIDES},
            'provenance': {'teams': [{'partition': 'train'}, {'partition': 'train'}]},
            'outcome': {'kind': 'terminal', 'winner': 'Full p1', 'turns': 3,
                        'rewards': {'p1': 1., 'p2': 0.}} if terminal else None,
            'submitted': []}

    def request(self, op, **payload):
        self.requests.append((op, copy.deepcopy(payload)))
        if op == 'start':
            self.index = 0
        elif op == 'step':
            self.index += 1
        else:
            raise AssertionError('Unexpected synthetic operation: ' + op)
        self.last = self.frame()
        if op == 'step':
            self.last['submitted'] = [{'side': side, 'accepted': True}
                                      for side in payload['actions']]
        return self.last

    def close(self):
        self.close_calls += 1


class CollectorAdmission(unittest.TestCase):
    def test_archived_foe_rows_excluded_and_sampling_counter_advances_both_seats(self):
        for learner_side in SIDES:
            with self.subTest(learner_side=learner_side), tempfile.TemporaryDirectory() as temp:
                workers, frozen_calls, structured_calls = [], [], []
                foe = 'p2' if learner_side == 'p1' else 'p1'

                def worker_factory(evidence):
                    worker = SyntheticWorker(evidence)
                    workers.append(worker)
                    return worker

                def structured_actor(parameters, players, histories, seeds, counters, capacity, catalog):
                    outputs = []
                    for player, history, seed, counter in zip(players, histories, seeds, counters):
                        self.assertEqual(player['side'], history['side'])
                        live = any(player['mask'])
                        side_index = SIDES.index(player['side'])
                        structured_calls.append((player['side'], player['request_id'], seed, counter, live))
                        x = np.array([side_index, player['request_id'], counter, 99], np.float32)
                        action = (0 if player['side'] == learner_side else 1) if live else -1
                        outputs.append((x, np.asarray(player['mask'], bool), action,
                                        -.25 if live else 0., .4))
                    return outputs

                def frozen_actor(parameters, players, seeds, counters, capacity, architecture, catalog):
                    self.assertEqual(architecture, 'nonlinear')
                    outputs = []
                    for player, seed, counter in zip(players, seeds, counters):
                        self.assertEqual(player['side'], foe)
                        live = any(player['mask'])
                        action = 2 + counter % 2 if live else -1
                        frozen_calls.append((player['request_id'], seed, counter, live, action))
                        outputs.append((None, np.asarray(player['mask'], bool), action, -1., .8))
                    return outputs

                spec = dict(seed=34001, episode=17, max_turns=20, split='train',
                            swap=False, opponent='frozen_f32', learner_side=learner_side)
                with patch.object(collector, 'Worker', side_effect=worker_factory), \
                     patch.object(collector, 'batched_actor', side_effect=structured_actor), \
                     patch.object(collector, 'frozen_actor', side_effect=frozen_actor):
                    with collector.Pool(temp, 1, {}, frozen_parameters={'frozen': True}) as pool:
                        results, profile = pool.collect({}, [spec])
                    self.assertTrue(pool.closed)
                    self.assertTrue(pool.executor._shutdown)
                    pool.close()  # inherited cleanup is idempotent

                self.assertEqual(workers[0].close_calls, 1)
                starts = [payload for op, payload in workers[0].requests if op == 'start']
                self.assertIsNone(starts[0]['tactical_side'])
                rows, record = results[0]
                self.assertEqual(record['opponent'], 'frozen_f32')
                self.assertEqual(record['learner_side'], learner_side)
                self.assertEqual(len(rows), 3 if learner_side == 'p1' else 2)
                self.assertTrue(all(row['trajectory'] == SIDES.index(learner_side) for row in rows))
                self.assertTrue(all(row['features'][0] == SIDES.index(learner_side) for row in rows))
                self.assertTrue(all(row['action'] == 0 and row['old_logp'] == -.25 for row in rows))
                self.assertEqual([r['next_index'] for r in rows], list(range(1, len(rows))) + [-1])
                self.assertTrue(rows[-1]['terminal'])
                self.assertEqual(rows[-1]['reward'], 1. if learner_side == 'p1' else 0.)
                self.assertTrue(all(row['side'] == learner_side for row in record['actor_trace']))
                self.assertEqual(record['decisions'], 5)
                self.assertEqual(profile['worker_requests'], 4)

                live_calls = [row for row in frozen_calls if row[3]]
                self.assertEqual([row[2] for row in live_calls], list(range(len(live_calls))))
                expected_actions = [row[4] for row in live_calls]
                observed_actions = [payload['actions'][foe] for op, payload in workers[0].requests
                                    if op == 'step' and foe in payload['actions']]
                self.assertEqual(observed_actions, expected_actions)
                self.assertEqual([row[2] for row in record['trace'] if row[0] == foe], expected_actions)
                self.assertTrue(all(row[3] is None for row in record['trace'] if row[0] == foe))
                for request_id, seed, counter, live, _ in frozen_calls:
                    matching = [row for row in structured_calls if row[:2] == (foe, request_id)]
                    self.assertEqual(matching, [(foe, request_id, seed, counter, live)])
                if foe == 'p2':
                    self.assertEqual([row[2] for row in frozen_calls], [0, 1, 1, 2])
                else:
                    self.assertEqual([row[2] for row in frozen_calls], [0, 1, 2, 3])

    def test_matchup_block_finiteness_retains_predecessor_admission(self):
        x = np.zeros((2, FEATURES), np.float32)
        x[:, 928:1278] = 1e6  # predecessor checked finiteness, not a numeric range
        batch = dict(features=x, mask=np.ones((2, 14), bool), actions=np.zeros(2, np.int32),
            old_logp=np.zeros(2, np.float32), advantages=np.zeros(2, np.float32),
            returns=np.ones(2, np.float32), valid=np.ones(2, bool))
        learning.validate(batch)
        predecessor_validate({**batch, 'features': x[:, :1278]})
        for column in [928, 1277]:
            bad = x.copy()
            bad[0, column] = np.nan
            with self.assertRaises(ValueError):learning.validate({**batch, 'features': bad})
            with self.assertRaises(ValueError):predecessor_validate({**batch, 'features': bad[:, :1278]})

    def test_masked_distribution_has_finite_entropy_including_waiting_rows(self):
        logits = jnp.tile(jnp.linspace(-1000, 1000, 14), (4, 1))
        mask = np.array([[True] * 14, [True] + [False] * 13,
                         [False] * 14, [i % 2 == 0 for i in range(14)]], bool)
        probs, logs = map(np.asarray, distribution(logits, jnp.asarray(mask)))
        self.assertTrue(np.isfinite(probs).all() and np.isfinite(logs).all())
        self.assertTrue(np.isfinite((probs * logs).sum(1)).all())
        np.testing.assert_array_equal(probs[~mask], 0)
        np.testing.assert_array_equal(logs[~mask], 0)
        np.testing.assert_allclose(probs.sum(1), [1, 1, 0, 1], atol=1e-6)

    def test_storage_injects_actual_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temp:
            arrays = {'test': np.zeros(1, np.float32)}
            pointer = save_checkpoint(temp, arrays, (arrays, arrays, np.int32(17)),
                                      {'cohorts': 0, 'optimizer_step': 999})
            with np.load(Path(temp) / pointer['file'], allow_pickle=False) as saved:
                metadata = json.loads(saved['metadata'].tobytes())
            self.assertEqual(metadata['optimizer_step'], 17)


if __name__ == '__main__':
    unittest.main()
