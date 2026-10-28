"""Bounded F32 scope checks; real sealed F29 inputs are read, never mutated."""
import copy
import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np

from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, canonical, digest
from pokemon_observed_continuation import common as c
from pokemon_observed_continuation.campaign import training_regimes, training_specs


def games(reward, terminal=True, opponent='tactical'):
    """128 independent schedules, each observed in both learner seats."""
    return [dict(episode=i, swap=swap, learner_side='p2' if swap else 'p1',
                 opponent=opponent,
                 provenance={'teams': [{'team_key': str(i), 'partition': 'development'}]},
                 decisions=1, trace=[], value_mean=.5, terminal_value_mse=.25,
                 outcome={'kind': 'terminal' if terminal else 'administrative_cutoff',
                          'rewards': {'p1': 1-reward if swap else reward,
                                      'p2': reward if swap else 1-reward}})
            for i in range(128) for swap in [False, True]]


@lru_cache(maxsize=6)
def start(seed, arm):
    return c.initial(None, seed, arm)


class Scope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = c.read(c.PLAN)
        cls.prior_plan = c.read(ROOT/'contracts/nonlinear-policy-v1/plan.json')

    def panels(self):
        return {a: {e: {o: games(0, opponent=o) for o in self.plan['opponents']}
                    for e in ['262144', '1048576']} for a in self.plan['arms']}

    def test_exact_same_seed_same_arm_predecessor_state(self):
        prior = c.read(ROOT/self.plan['f29_receipt'])
        run = ROOT/prior['run']
        report = c.read(run/'report.json')
        h = digest(run/'inputs/manifest.json')
        for seed in self.plan['seeds']:
            for arm in self.plan['arms']:
                with self.subTest(seed=seed, arm=arm):
                    p, opt, identity = start(seed, arm)
                    summary = report['seeds'][str(seed)]['arms'][arm]
                    cursor = summary['final_next_episode']
                    folder = run/str(seed)/arm/'checkpoints/256'
                    q, qo = c.load(folder, h, seed, arm, 256, cursor)
                    for actual, expected in [(p, q), (opt[0], qo[0]), (opt[1], qo[1])]:
                        self.assertEqual(set(actual), set(expected))
                        for key in expected:
                            np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(int(opt[2]), 2048)
                    self.assertEqual(int(qo[2]), 2048)
                    self.assertEqual(identity['next_episode'], cursor)
                    self.assertEqual(identity['parameter_hash'], summary['endpoints']['262144']['parameter_hash'])
                    self.assertEqual(identity['adam_m_sha256'], fingerprint(qo[0]))
                    self.assertEqual(identity['adam_v_sha256'], fingerprint(qo[1]))
                    self.assertEqual(identity['predecessor_receipt_sha256'], digest(ROOT/self.plan['f29_receipt']))
                    self.assertEqual(identity['predecessor_input_hash'], h)
                    self.assertEqual(identity['checkpoint_pointer_sha256'], digest(folder/'latest.json'))
                    self.assertEqual(identity['checkpoint_sha256'], c.read(folder/'latest.json')['sha256'])
                    self.assertEqual(identity['source_checkpoint'], str(folder.relative_to(ROOT)))
                    self.assertEqual((identity['seed'], identity['arm'], identity['cohort'], identity['optimizer_step']),
                                     (seed, arm, 256, 2048))
                    self.assertIs(identity['optimizer_reset'], False)

    def test_initial_rejects_wrong_source_seed_and_arm(self):
        prior = c.read(ROOT/self.plan['f29_receipt'])
        bad = copy.deepcopy(prior); bad['status'] = 'forged'
        for previous, seed, arm in [(bad, 7103, 'control'), (None, -1, 'control'),
                                    (None, 7103, 'unknown')]:
            with self.subTest(seed=seed, arm=arm), self.assertRaises(ValueError):
                c.initial(previous, seed, arm)

    def test_absolute_continuation_and_additional_budgets(self):
        c.validate_plan(self.plan)
        for arm in self.plan['arms']:
            b = c.budget(arm)
            indices = list(range(b['start_cohort'], b['cohorts']))
            self.assertEqual((indices[0], indices[-1], len(indices)), (256, 1023, 768))
            self.assertEqual(b['additional_cohorts'], 768)
            self.assertEqual(len(indices)*b['selected_rows'], 786432)
            self.assertEqual(2*len(indices)*b['selected_rows'], 1572864)
            self.assertEqual(len(indices)*b['optimizer_steps_per_cohort'], 6144)
            self.assertEqual(2048+len(indices)*8, 8192)
            self.assertEqual([x*b['selected_rows'] for x in b['endpoints']], [262144, 524288, 1048576])
            self.assertEqual(training_regimes(arm), [(None, 512), ('tactical', 512)])
            self.assertEqual(b['maximum_games_per_stream'], 64)

    def test_global_counts_and_recovery_boundaries(self):
        counts = self.plan['expected_counts']
        self.assertEqual(counts, dict(cohorts=4608, optimizer_steps=36864,
            selected_ppo_rows=4718592, ppo_row_presentations=9437184,
            evaluation_games=13824, selfplay_diagnostic_games=576,
            checkpoint_comparisons=30, original_next_update_recoveries=12))
        for arm in self.plan['arms']:
            b = c.budget(arm)
            self.assertEqual(b['checkpoint_cohorts'], [256, 511, 512, 1023, 1024])
            self.assertEqual(b['recovery_indices'], [511, 1023])
            self.assertTrue(set(b['recovery_indices']) <= set(b['checkpoint_cohorts']))
            self.assertEqual([8*i for i in b['recovery_indices']], [4088, 8184])
        self.assertEqual(counts['evaluation_games'], 6*3*3*128*2)
        self.assertEqual(counts['selfplay_diagnostic_games'], 6*3*16*2)

    def test_scope_mutations_rejected(self):
        for key, value in [('seeds', [7103]), ('arm_budgets', {}), ('start_cohort', 0),
                ('final_cohort', 512), ('ppo', {}), ('training_seed_namespace', 'reset'),
                ('optimizer_seed_namespace', 'reset'), ('evaluation_seed', 1),
                ('statistics', {}), ('expected_counts', {}), ('promotion_authorized', True),
                ('final_reserved_data_used', True), ('public_server_used', True), ('prime_used', True)]:
            p = copy.deepcopy(self.plan); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_plan(p)

    def test_checkpoint_metadata_and_recovery_roundtrips(self):
        p, opt, identity = start(7103, 'control')
        cursor = identity['next_episode']
        with tempfile.TemporaryDirectory() as directory:
            for cohort in [256, 511, 512, 1023, 1024]:
                # Synthetic counters test serialization, not a claim of trained states.
                state = (opt[0], opt[1], np.int32(8*cohort))
                folder = Path(directory)/str(cohort)
                c.save(folder, p, state, 'input', 7103, 'control', cohort, cursor)
                q, qo = c.load(folder, 'input', 7103, 'control', cohort, cursor)
                c.same_state(p, state, q, qo)
                self.assertEqual(c.metadata('input', 7103, 'control', cohort, cursor),
                    dict(input_hash='input', seed=7103, architecture='control', cohorts=cohort,
                         next_episode=cursor, selected_rows=1024*cohort, optimizer_step=8*cohort))
                ptr = c.read(folder/'latest.json')
                with np.load(folder/ptr['file'], allow_pickle=False) as z:
                    self.assertEqual(json.loads(z['metadata'].tobytes()),
                                     c.metadata('input', 7103, 'control', cohort, cursor))
            with self.assertRaises(ValueError):
                c.save(Path(directory)/'bad', p, opt, 'input', 7103, 'control', 512, cursor)

    def test_checkpoint_wrong_identity_and_hash_rejected(self):
        p, opt, identity = start(7103, 'control'); cursor = identity['next_episode']
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); c.save(d, p, opt, 'h', 7103, 'control', 256, cursor)
            for args in [('wrong', 7103, 'control', 256, cursor), ('h', 7103, 'nonlinear', 256, cursor),
                         ('h', 7103, 'control', 256, cursor+8), ('h', 8209, 'control', 256, cursor),
                         ('h', 7103, 'control', 512, cursor)]:
                with self.subTest(args=args), self.assertRaises(ValueError): c.load(d, *args)
            ptr = c.read(d/'latest.json'); ptr['sha256'] = '0'*64
            (d/'latest.json').write_bytes(canonical(ptr))
            with self.assertRaises(ValueError): c.load(d, 'h', 7103, 'control', 256, cursor)

    def test_checkpoint_invalid_numpy_dtype_rejected(self):
        p, opt, identity = start(7103, 'control'); cursor = identity['next_episode']
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); c.save(d, p, opt, 'h', 7103, 'control', 256, cursor)
            ptr = c.read(d/'latest.json'); path = d/ptr['file']
            with np.load(path, allow_pickle=False) as z: original = {k: z[k].copy() for k in z.files}
            for prefix in ['p', 'm', 'v']:
                arrays = dict(original); key = prefix+'__'+next(iter(p))
                arrays[key] = np.asarray(arrays[key], dtype=np.float64)
                with path.open('wb') as f: np.savez_compressed(f, **arrays)
                ptr['sha256'] = digest(path); (d/'latest.json').write_bytes(canonical(ptr))
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    c.load(d, 'h', 7103, 'control', 256, cursor)

    def test_immutable_namespaces_and_absolute_optimizer_schedule(self):
        for key in ['training_seed_namespace', 'optimizer_seed_namespace', 'ppo']:
            self.assertEqual(self.plan[key], self.prior_plan[key])
        for seed in self.plan['seeds']:
            ns = self.plan['optimizer_seed_namespace']
            old = {derived_seed(ns, seed, i) for i in range(256)}
            new = {derived_seed(ns, seed, i) for i in range(256, 1024)}
            self.assertEqual(len(new), 768); self.assertFalse(old & new)
            for arm in self.plan['arms']:
                cursor = start(seed, arm)[2]['next_episode']
                for opponent in [None, 'tactical']:
                    first = training_specs(self.plan, seed, cursor, opponent)
                    second = training_specs(self.plan, seed, cursor+8, opponent)
                    self.assertEqual(first, training_specs(self.prior_plan, seed, cursor, opponent))
                    self.assertEqual([j['episode'] for j in first], list(range(cursor, cursor+8)))
                    self.assertTrue(all(j['split'] == 'train' for j in first+second))
                    self.assertFalse({canonical(j) for j in first} & {canonical(j) for j in second})
                self.assertNotEqual(training_specs(self.plan, seed, cursor, None)[0]['seed'],
                                    training_specs(self.plan, seed, cursor, 'tactical')[0]['seed'])

    def test_fresh_shared_development_schedules(self):
        for seed in self.plan['seeds']:
            main_seeds = set()
            for opponent in self.plan['opponents']:
                jobs = c.evaluation_specs(self.plan, seed, opponent, 0, 4)
                self.assertEqual(jobs, c.evaluation_specs(self.plan, seed, opponent, 0, 4))
                self.assertTrue(all(j['split'] == 'development' for j in jobs))
                old = c.evaluation_specs(self.prior_plan, seed, opponent, 0, 4)
                self.assertNotEqual(jobs[0]['seed'], old[0]['seed'])
                main_seeds.update(j['seed'] for j in jobs)
            diag = c.diagnostic_specs(self.plan, seed, 0, 4)
            self.assertTrue(all(j['split'] == 'development' for j in diag))
            self.assertFalse(main_seeds & {j['seed'] for j in diag})
            self.assertNotEqual(diag[0]['seed'], c.diagnostic_specs(self.prior_plan, seed, 0, 4)[0]['seed'])

    def test_analysis_links_continuation_baseline_and_tactical_gate(self):
        panels = self.panels()
        panels['nonlinear']['1048576']['tactical'] = games(1)
        result = c.analysis(panels, self.plan)
        self.assertEqual(set(result['comparisons']), set(self.plan['comparisons']))
        self.assertEqual(sum(map(len, result['comparisons'].values())), 9)
        self.assertTrue(result['learning']['nonlinear']['passed'])
        self.assertFalse(result['learning']['control']['passed'])
        self.assertTrue(result['feature_benefit']['passed'])
        self.assertFalse(result['base_power_benefit']['passed'])
        panels['nonlinear']['262144']['tactical'] = games(1)
        result = c.analysis(panels, self.plan)
        self.assertFalse(result['learning']['nonlinear']['passed'])
        self.assertTrue(result['feature_benefit']['passed'])

    def test_secondary_base_power_and_cutoff_gates(self):
        panels = self.panels()
        for o in ['tactical', 'base_power']:
            panels['nonlinear']['1048576'][o] = games(1, opponent=o)
        result = c.analysis(panels, self.plan)
        self.assertTrue(result['base_power_learning']['nonlinear']['passed'])
        self.assertTrue(result['base_power_benefit']['passed'])
        for o in self.plan['opponents']:
            panels['nonlinear']['1048576'][o] = games(1, False, o)
        result = c.analysis(panels, self.plan)
        for key in ['feature_benefit', 'base_power_benefit']:
            self.assertFalse(result[key]['passed'])
            self.assertFalse(result[key]['checks']['cutoff_rates'])

    def test_regression_and_pair_corruption_rejection(self):
        panels = self.panels()
        for arm in self.plan['arms']:
            for endpoint in panels[arm]: panels[arm][endpoint]['random'] = games(1, opponent='random')
        panels['nonlinear']['1048576']['tactical'] = games(1)
        self.assertTrue(c.analysis(panels, self.plan)['feature_benefit']['passed'])
        panels['nonlinear']['1048576']['random'] = games(0, opponent='random')
        result = c.analysis(panels, self.plan)['feature_benefit']
        self.assertFalse(result['passed']); self.assertFalse(result['checks']['no_detected_regression'])
        panels['nonlinear']['1048576']['random'][0]['swap'] = True
        with self.assertRaises(ValueError): c.analysis(panels, self.plan)


if __name__ == '__main__':
    unittest.main()
