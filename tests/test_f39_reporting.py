"""Synthetic F39 report and mutation tests; no simulator, optimizer or teacher."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pokemon_full_train.campaign import value_summary
from pokemon_full_train.learning import derived_seed
from pokemon_online_continuation import reporting as r
from pokemon_online_continuation.common import NAMESPACES, evaluation_specs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def game(spec, score=0, cutoff=False, rejected=False):
    side = spec['learner_side']
    own = 2 if spec['opponent'] is None else 1
    train = spec['split'] == 'train'
    attempts = []
    if rejected:
        attempts = [{'side': side, 'controlled': True, 'action': 0, 'request_id': 1,
            'actor_counter': 0, 'learner_row_index': 0 if train else None,
            'selected_for_update': False if train else None, 'feedback': ['[Unavailable choice] fixture'],
            'refreshed_request_id': 2, 'refreshed_mask': [True] + [False] * 13}]
    return {k: spec[k] for k in ('episode', 'split', 'swap', 'learner_side', 'opponent')} | {
        'provenance': {'episode': spec['episode'], 'swap': spec['swap'], 'opponent': spec['opponent']},
        'outcome': {'kind': 'administrative_cutoff' if cutoff else 'terminal',
            'rewards': None if cutoff else {side: score, 'p2' if side == 'p1' else 'p1': 1-score}},
        'decisions': 2, 'value_mean': .5, 'terminal_value_mse': None if cutoff else (.5-score)**2,
        'unavailable_choices': int(rejected), 'rejected_attempts': attempts,
        'accepted_decisions': 2-int(rejected), 'accepted_controlled_decisions': own-int(rejected),
        'eligible_learner_rows': own if train else 0}


def panels(score, cutoff=False):
    return {o: [game({'episode': i, 'split': 'development', 'swap': swap,
        'learner_side': 'p2' if swap else 'p1', 'opponent': o}, score, cutoff)
        for i in range(8) for swap in (False, True)] for o in r.OPPONENTS}


def ledger_fixture(base):
    seed = 7103
    plan = {'initial_cohorts_per_seed': 512, 'final_cohorts_per_seed': 514,
        'cohorts_per_seed': 2, 'sampled_update_cohorts': [512], 'epochs': 2, 'minibatch_rows': 1,
        'selected_rows_per_cohort': 4, 'stream_selected_rows': {'self_play': 1, 'frozen_f32': 1, 'tactical': 2},
        'optimizer_steps_per_cohort': 8}
    initial_cursors = dict.fromkeys(r.STREAMS, 100)
    cursors, all_games, stream_counts = dict(initial_cursors), [], {s: {} for s in r.STREAMS}
    for cohort in range(512, 514):
        games, strata = [], []
        for label, opponent in r.STREAMS.items():
            first = cursors[label]
            current = [game({'episode': first+i, 'split': 'train', 'swap': False,
                'learner_side': 'p2' if (first+i)%2 else 'p1', 'opponent': opponent}, rejected=i == 0)
                for i in range(2)]
            n = sum(g['eligible_learner_rows'] for g in current)
            quota = plan['stream_selected_rows'][label]
            selection_seed = derived_seed('f39-row-selection-v1', seed, label)
            indices = np.random.default_rng(derived_seed('f9-row-selection', selection_seed, cohort)).choice(n, quota, replace=False).tolist()
            selection = {'collected_rows': n, 'selected_rows': quota, 'excluded_rows': n-quota,
                         'selected_indices': indices}
            current[0]['rejected_attempts'][0]['selected_for_update'] = 0 in indices
            rejection = {'rejected_attempts': 1, 'rejected_controlled_attempts': 1,
                         'rejected_selected_rows': int(0 in indices)}
            strata.append({'stream': label, 'opponent': opponent, 'first_episode': first,
                'next_episode': first+2, 'games': 2, 'selection': selection,
                'selection_seed': selection_seed, 'rejection_selection': rejection})
            sc = stream_counts[label]
            for k, v in {'games': 2, 'collected_rows': n, 'selected_rows': quota, 'excluded_rows': n-quota, **rejection}.items():
                sc[k] = sc.get(k, 0) + v
            games.extend(current)
            cursors[label] += 2
        metrics = [{'loss': 0., 'actor_loss': 0., 'value_loss': .1, 'entropy': .2,
                    'approx_kl': i/100, 'clip_fraction': i/10, 'gradient_norm': .1} for i in range(8)]
        update = {'cohort': cohort, 'optimizer_step': (cohort+1)*8,
            'optimizer_seed': derived_seed('f39-optimizer-v1', seed, cohort),
            'parameter_hash_before': str(cohort), 'state_hash_before': 'state'+str(cohort),
            'parameter_hash_after': str(cohort+1), 'state_hash_after': 'state'+str(cohort+1),
            'sampled_update_verified': cohort == 512, 'update_seconds_including_compile': .1,
            'sampled_replay_update_seconds': .1 if cohort == 512 else 0., 'ledger': {'strata': strata, 'games': games},
            'metrics': metrics}
        write(base / str(seed) / f'update-{cohort:04d}.json', update)
        all_games.extend(games)
    row = {'initial_parameter_hash': '512', 'final_parameter_hash': '514',
        'initial_state_hash': 'state512', 'final_state_hash': 'state514',
        'initial_stream_cursors': initial_cursors, 'stream_cursors': cursors,
        'identity': {'f34_metadata': {'stream_cursors': initial_cursors}},
        'initial_optimizer_step': 4096, 'final_optimizer_step': 4112, 'optimizer_steps': 4112,
        'continuation_optimizer_steps': 16, 'cumulative_selected_rows': 514*4,
        'cumulative_row_presentations': 514*8, 'training_unavailable_choices': 6,
        'training': value_summary(all_games), 'stream_counts': stream_counts}
    return seed, plan, row


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'seed': 1027283903, 'resamples': 64, 'family_cells': 12, 'alpha': .05,
                         'practical_improvement': .05, 'max_final_cutoff_rate': .05}

    def test_primary_gate_and_regression_flag_are_separate(self):
        initial, final = panels(0), panels(1)
        initial['random'], final['random'] = panels(1)['random'], panels(0)['random']
        result = r.analyze(initial, final, self.settings)
        self.assertTrue(result['primary_gate']['passed'])
        self.assertTrue(result['any_detected_regression'])
        self.assertEqual(result['detected_regressions'], ['random'])
        self.assertEqual(len(result['contrasts']), 4)
        changed = {**self.settings, 'family_cells': 3}
        with self.assertRaisesRegex(ValueError, 'twelve'):
            r.analyze(initial, final, changed)

    def test_cutoffs_stay_unresolved_and_values_terminal_only(self):
        result = r.analyze(panels(0, True), panels(0, True), self.settings)
        self.assertEqual(result['contrasts']['baseline-v1']['cutoff_aware_point_interval'], [-1., 1.])
        self.assertFalse(result['primary_gate']['passed'])
        mixed = [panels(0, True)['baseline-v1'][0], panels(1)['baseline-v1'][1]]
        mixed[0]['value_mean'] = .999
        summary = r.outcome_summary(mixed)
        self.assertEqual(summary['terminal_game_mean_predicted_value'], .5)
        self.assertEqual(summary['terminal_game_mean_value_mse'], .25)
        self.assertEqual(summary['true_ties'], 0)
        self.assertEqual(summary['cutoff_rate'], .5)

    def test_contrast_values_exclude_cutoffs_without_changing_score_arithmetic(self):
        initial, final = panels(0), panels(1)
        for panel in (initial, final):
            for games in panel.values():
                games[0]['outcome'] = {'kind': 'administrative_cutoff', 'rewards': None}
                games[0]['terminal_value_mse'] = None
                games[0]['value_mean'] = .999
        raw = r.compare(initial['baseline-v1'], final['baseline-v1'], self.settings)
        result = r.analyze(initial, final, self.settings)
        contrast = result['contrasts']['baseline-v1']
        for endpoint in ('initial', 'final'):
            self.assertNotIn('game_mean_predicted_value', contrast[endpoint])
            self.assertEqual(contrast[endpoint]['terminal_game_mean_predicted_value'], .5)
            self.assertEqual(contrast[endpoint]['terminal_game_mean_value_mse'], .25)
        for key in ('cutoff_aware_point_interval', 'approximate_familywise_bootstrap_interval',
                    'bounded_hoeffding_interval', 'pairs', 'bootstrap_resamples', 'gate'):
            self.assertEqual(contrast[key], raw[key])
        self.assertEqual(result['primary_gate'], r.gate(raw, self.settings))

    def test_every_game_value_trace_is_validated_and_empty_summary_is_defined(self):
        games = panels(1)['baseline-v1'][:2]
        for g in games:
            g['actor_trace'] = [{'side': g['learner_side'], 'value': .5}]
        self.assertEqual(r.outcome_summary(games)['games'], 2)
        games[0]['actor_trace'][0]['value'] = .25
        with self.assertRaisesRegex(ValueError, 'mean linkage'):
            r.outcome_summary(games)
        empty = r.outcome_summary([])
        self.assertEqual(empty['games'], 0)
        self.assertIsNone(empty['terminal_game_mean_predicted_value'])

    def test_initial_cutoff_rate_does_not_add_unregistered_gate(self):
        c = {'cutoff_aware_point_interval': [.5, 1.], 'approximate_familywise_bootstrap_interval': [.2, 1.],
             'initial': {'cutoffs': 100, 'games': 100}, 'final': {'cutoffs': 0, 'games': 100}}
        self.assertTrue(r.gate(c, self.settings)['passed'])
        c['final']['cutoffs'] = 6
        self.assertFalse(r.gate(c, self.settings)['passed'])

    def test_pair_identity_and_nonfinite_values_fail(self):
        final = panels(1)
        final['baseline-v1'][0]['provenance']['episode'] = 999
        with self.assertRaisesRegex(ValueError, 'identity'):
            r.analyze(panels(0), final, self.settings)
        g = panels(0)['baseline-v1'][0]
        g['terminal_value_mse'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'MSE'):
            r.outcome_summary([g])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.seed, self.plan, self.row = ledger_fixture(self.base)
        self.path = self.base / str(self.seed) / 'update-0512.json'

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self, fn):
        data = json.loads(self.path.read_text())
        fn(data)
        write(self.path, data)

    def test_every_update_rows_rejections_and_preupdate_statistics(self):
        result = r.update_accounting(self.base, self.seed, self.plan, self.row)
        self.assertEqual(result['counts']['cohorts'], 2)
        self.assertAlmostEqual(result['update_seconds_including_compile'], .2)
        self.assertAlmostEqual(result['sampled_replay_update_seconds'], .1)
        self.assertEqual(result['counts']['selected_rows'], 8)
        self.assertEqual(result['totals']['collected_rows'], 16)
        self.assertEqual(result['totals']['excluded_rows'], 8)
        self.assertEqual(result['totals']['rejected_attempts'], 6)
        diagnostics = result['optimizer_diagnostics']
        self.assertEqual(diagnostics['first_minibatch_per_cohort']['approx_kl']['n'], 2)
        self.assertAlmostEqual(diagnostics['last_epoch_minibatches']['approx_kl']['mean'], .055)

    def test_mutations_in_each_independent_chain_fail(self):
        original = json.loads(self.path.read_text())
        cases = [
            (lambda x: x.update(state_hash_before='wrong'), 'state chain'),
            (lambda x: x.update(update_seconds_including_compile=-1), 'cost scope'),
            (lambda x: x.update(optimizer_step=8), 'coordinates'),
            (lambda x: x['ledger']['strata'][0]['selection'].update(excluded_rows=99), 'row accounting'),
            (lambda x: x['ledger']['strata'][0].update(first_episode=0), 'cursor'),
            (lambda x: x['ledger']['strata'][0]['rejection_selection'].update(rejected_selected_rows=99), 'rejection aggregate'),
            (lambda x: x['ledger']['games'][0]['rejected_attempts'][0].update(learner_row_index=999), 'learner-row'),
            (lambda x: x['ledger']['games'][0]['rejected_attempts'][0].update(selected_for_update=None), 'selected-row'),
            (lambda x: x['metrics'][0].update(demo_nll=.1), 'metric inventory'),
        ]
        for fn, message in cases:
            with self.subTest(message=message):
                write(self.path, original)
                self.mutate(fn)
                with self.assertRaisesRegex(ValueError, message):
                    r.update_accounting(self.base, self.seed, self.plan, self.row)

    def test_missing_or_extra_cohorts_fail(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, 'update inventory'):
            r.update_accounting(self.base, self.seed, self.plan, self.row)

    def test_rejected_row_offsets_use_complete_games(self):
        current = [game({'episode': i, 'split': 'train', 'swap': False,
            'learner_side': 'p1', 'opponent': 'tactical'}, rejected=True) for i in range(2)]
        current[1]['rejected_attempts'][0]['selected_for_update'] = True
        result = r.rejection_accounting(current, {'selected_indices': [1], 'collected_rows': 2})
        self.assertEqual(result['rejected_selected_rows'], 1)
        current[0]['rejected_attempts'][0]['selected_for_update'] = True
        with self.assertRaisesRegex(ValueError, 'selected-row'):
            r.rejection_accounting(current, {'selected_indices': [1], 'collected_rows': 2})

    def test_exact_local_evaluation_schedule_and_extra_chunks(self):
        plan = {'namespaces': NAMESPACES, 'max_turns': 240, 'evaluation_pairs': dict.fromkeys(r.OPPONENTS, 4)}
        folder = self.base / str(self.seed) / 'initial'
        summary = {}
        for opponent in r.OPPONENTS:
            specs = evaluation_specs(plan, self.seed, opponent, 0, 4)
            games = [game(s) for s in specs]
            write(folder / f'{opponent}-000.json', {'specs': specs, 'games': games})
            summary[opponent] = value_summary(games)
        write(folder / 'summary.json', summary)
        self.assertEqual(sum(map(len, r.local_panels(self.base, self.seed, 'initial', plan).values())), 32)
        write(folder / 'unknown-000.json', {})
        with self.assertRaisesRegex(ValueError, 'chunk inventory'):
            r.local_panels(self.base, self.seed, 'initial', plan)

    def test_duplicate_sample_coordinates_rejected(self):
        records = []
        for seed in r.SEEDS:
            records += [{'kind': 'fresh_training', 'seed': seed, 'cohort': c, 'stream': s}
                        for c in (512, 767) for s in r.STREAMS]
            records += [{'kind': 'fresh_update', 'seed': seed, 'cohort': c} for c in (512, 767)]
            records += [{'kind': 'checkpoint', 'seed': seed, 'cohort': c} for c in (512, 767, 768)]
            records += [{'kind': 'fresh_evaluation_pair', 'seed': seed, 'endpoint': e, 'opponent': o}
                        for e in r.ENDPOINTS for o in r.OPPONENTS]
        counts = {'training_games': 18, 'evaluation_games': 48, 'game_executions': 66,
            'sampled_updates': 6, 'optimizer_steps_recomputed': 48, 'checkpoint_comparisons': 9}
        verified = {'counts': counts, 'expected_counts': counts, 'records': records,
            'coverage': {'exhaustive': False,
                'teacher_search_module_inventory': {'before': [], 'after': [], 'forbidden_prefixes': ['fp']},
                'teacher_search_execution_scope': 'Synthetic before/after inventory fixture.'}}
        r.verify_coverage(verified, self.plan)
        records[1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(ValueError, 'coverage coordinates'):
            r.verify_coverage(verified, self.plan)


if __name__ == '__main__':
    unittest.main()
