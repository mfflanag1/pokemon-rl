"""Synthetic evidence fixtures only: no games, checkpoint loads, or optimization."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from pokemon_full_train.campaign import value_summary
from pokemon_full_train.storage import digest
from pokemon_structured_policy.common import SEEDS, evaluation_specs, learning_analysis
from pokemon_structured_policy import reporting


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def game(spec, reward=0, cutoff=False, value=.5):
    side = spec['learner_side']
    rewards = {side: reward, 'p2' if side == 'p1' else 'p1': 1 - reward}
    return {k: spec[k] for k in ('episode', 'split', 'swap', 'learner_side', 'opponent')} | {
        'outcome': {'kind': 'administrative_cutoff' if cutoff else 'terminal',
                    'rewards': None if cutoff else rewards},
        'provenance': {'fixture': [spec['episode'], spec['swap'], spec['opponent']]},
        'decisions': 2, 'unavailable_choices': 0, 'value_mean': value,
        'terminal_value_mse': None if cutoff else (value - reward) ** 2}


def fixture(base):
    training, initial, final, verify = [base / p for p in ('training', 'initial', 'final', 'verify')]
    plan = {'seeds': list(SEEDS), 'max_turns': 240, 'cohorts_per_seed': 2,
        'selected_rows_per_cohort': 4, 'stream_selected_rows': {'self_play': 1, 'frozen_f32': 1, 'tactical': 2},
        'minibatch_rows': 1, 'epochs': 2, 'optimizer_steps_per_cohort': 8,
        'evaluation_pairs': {'tactical': 4, 'base_power': 4, 'random': 4},
        'sampled_update_cohorts': [0, 1], 'public_server_used': False, 'final_reserved_data_used': False,
        'statistics': {'seed': 1027, 'resamples': 16, 'alpha': .05, 'family_cells': 12,
                       'practical_improvement': .05, 'max_final_cutoff_rate': .05},
        'expected_counts': {'cohorts': 6, 'optimizer_steps': 48, 'selected_rows': 24,
            'row_presentations': 48, 'evaluation_games': 144, 'checkpoint_comparisons': 9,
            'original_next_update_recoveries': 6}}
    external_plan = {'seeds': list(SEEDS), 'pairs_per_seed': 1, 'endpoints': ['initial', 'final'],
        'expected_games_per_endpoint': 6, 'expected_total_games': 12,
        'search_time_ms': 100, 'parallelism': 1, 'search_threads': 1,
        'public_server_used': False, 'final_reserved_data_used': False}
    frozen_plan = training / 'inputs/source/contracts/fixture-external.json'
    write(frozen_plan, external_plan)
    plan_hash = digest(frozen_plan)
    write(training / 'inputs/manifest.json', {'plan': plan, 'source_files': {'contracts/fixture-external.json': plan_hash}})
    input_hash = digest(training / 'inputs/manifest.json')
    seeds = {}
    external_reports = {e: {'status': 'passed', 'input_hash': input_hash, 'plan': external_plan,
        'plan_sha256': plan_hash, 'endpoint': e, 'seeds': {}, 'elapsed_seconds': 3.}
        for e in reporting.ENDPOINTS}
    records = []
    for seed in SEEDS:
        all_games = []
        for cohort in range(2):
            strata, games = [], []
            for stream, opponent in reporting.STREAMS.items():
                current = [game({'episode': cohort * 2 + i, 'split': 'train', 'swap': False,
                    'learner_side': 'p2' if i else 'p1', 'opponent': opponent}) for i in range(2)]
                selected = plan['stream_selected_rows'][stream]
                collected = 4 if stream == 'self_play' else 2
                strata.append({'stream': stream, 'opponent': opponent, 'first_episode': cohort * 2,
                    'next_episode': cohort * 2 + 2, 'games': 2, 'selection': {'collected_rows': collected,
                    'selected_rows': selected, 'excluded_rows': collected - selected,
                    'selected_indices': list(range(selected))}})
                games.extend(current)
                records.append({'kind': 'fresh_training', 'seed': seed, 'cohort': cohort, 'stream': stream})
            all_games.extend(games)
            update = {'cohort': cohort, 'optimizer_step': (cohort + 1) * 8,
                'parameter_hash_before': f'{seed}-{cohort}', 'parameter_hash_after': f'{seed}-{cohort + 1}',
                'ledger': {'strata': strata, 'games': games},
                'metrics': [{'approx_kl': i / 100, 'clip_fraction': i / 10} for i in range(8)]}
            write(training / str(seed) / f'update-{cohort:04d}.json', update)
            records.append({'kind': 'fresh_update', 'seed': seed, 'cohort': cohort})
        panels = {}
        for endpoint in reporting.ENDPOINTS:
            panels[endpoint] = {}
            for opponent in plan['evaluation_pairs']:
                specs = evaluation_specs(plan, seed, opponent, 0, 4)
                reward = int((endpoint == 'final') != (opponent == 'random'))
                games = [game(spec, reward) for spec in specs]
                panels[endpoint][opponent] = games
                write(training / str(seed) / endpoint / f'{opponent}-000.json', {'specs': specs, 'games': games})
                records.append({'kind': 'fresh_evaluation_pair', 'seed': seed, 'endpoint': endpoint, 'opponent': opponent})
        row = {'seed': seed, 'identity': {'parameter_hash': f'host-{seed}'},
            'initial_parameter_hash': f'{seed}-0', 'final_parameter_hash': f'{seed}-2',
            'optimizer_steps': 16, 'stream_cursors': dict.fromkeys(reporting.STREAMS, 4),
            'training': value_summary(all_games),
            'initial': {o: value_summary(gs) for o, gs in panels['initial'].items()},
            'final': {o: value_summary(gs) for o, gs in panels['final'].items()},
            'analysis': learning_analysis(panels['initial'], panels['final'], plan['statistics'])}
        seeds[str(seed)] = row
        write(training / str(seed) / 'result.json', row)
        for endpoint, folder in [('initial', initial), ('final', final)]:
            games = []
            for swap in (False, True):
                g = game({'episode': 0, 'swap': swap, 'learner_side': 'p2' if swap else 'p1',
                    'split': 'development', 'opponent': 'foul_play'}, int(endpoint == 'final'),
                    cutoff=endpoint == 'final' and not swap,
                    value=999. if endpoint == 'final' and not swap else .75)
                raw = {'status': 'completed', 'outcome': g['outcome'], 'provenance': g['provenance'],
                    'search_time_ms': 100, 'parallelism': 1, 'search_threads': 1,
                    'foul_play_decisions': 1, 'steps': [{'actions': {'p1': 0, 'p2': 0},
                        'external_wall_seconds': .3, 'external': {'action': 0, 'seconds': .25,
                            'search_time_ms': 100, 'sampled_worlds': 4, 'per_world_ms': 50,
                            'configured_search_total_ms': 200}}],
                    'rejected_actions': {'p1': 0, 'p2': 0}, 'illegal_actions': {'p1': 0, 'p2': 0}, 'seconds': 1.}
                path = folder / str(seed) / f'000-{int(swap)}' / 'result.json'
                write(path, raw)
                g.update({k: raw[k] for k in ('seconds', 'foul_play_decisions', 'rejected_actions', 'illegal_actions')})
                g.update(game_record=str(path), game_record_sha256=digest(path))
                games.append(g)
                records.append({'kind': 'recorded_external_actions', 'seed': seed, 'endpoint': endpoint, 'swap': swap})
            ext_row = {'seed': seed, 'parameter_hash': f'host-{seed}' if endpoint == 'initial' else f'{seed}-2',
                'games': games, 'summary': value_summary(games)}
            external_reports[endpoint]['seeds'][str(seed)] = ext_row
            write(folder / str(seed) / 'result.json', ext_row)
    write(training / 'report.json', {'status': 'passed', 'input_hash': input_hash, 'plan': plan,
        'counts': plan['expected_counts'] | {'training_games': 36}, 'seeds': seeds,
        'all_seed_learning_passed': False, 'elapsed_seconds': 10., 'collection_seconds': 6.,
        'update_seconds_including_compile': 2., 'sampled_replay_update_seconds': 1., 'peak_parent_rss_mib': 128.,
        'profiles': [{'collection_seconds': 2.}, {'collection_seconds': 4.}],
        'public_server_used': False, 'final_reserved_data_used': False})
    for endpoint, folder in [('initial', initial), ('final', final)]:
        write(folder / 'report.json', external_reports[endpoint])
    counts = {'training_games': 18, 'evaluation_games': 36, 'external_action_replay_games': 12,
        'sampled_updates': 6, 'optimizer_steps_recomputed': 48, 'game_executions': 66}
    write(verify / 'report.json', {'status': 'passed', 'input_hash': input_hash, 'counts': counts,
        'expected_counts': counts, 'records': records, 'elapsed_seconds': 2.,
        'input_files': {str(p): digest(p) for p in (training / 'report.json',
            training / 'inputs/manifest.json', initial / 'report.json', final / 'report.json')},
        'coverage': {'exhaustive': False, 'external_search_reexecuted': False}})
    return training, initial, final, verify


class ReportingFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='f34-reporting-synthetic-')
        self.base = Path(self.temp.name).resolve()
        self.inputs = fixture(self.base)
        self.output = self.base / 'summary'

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, path, mutate):
        value = json.loads(path.read_text())
        mutate(value)
        write(path, value)

    def test_complete_synthetic_evidence_accounting_and_cli_contract(self):
        result = reporting.run(*self.inputs, self.output)
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['all_seed_learning_passed'])
        self.assertEqual(result['training_totals']['games'], 36)
        self.assertEqual(result['training_totals']['decisions'], 72)
        self.assertEqual(result['training_totals']['collected_rows'], 48)
        self.assertEqual(result['training_totals']['selected_rows'], 24)
        self.assertEqual(result['training_totals']['excluded_rows'], 24)
        self.assertEqual(result['training_totals']['non_learner_decisions'], 24)
        for row in result['seeds'].values():
            self.assertEqual(row['detected_regressions'], ['random'])
            self.assertEqual(row['external']['final']['outcomes']['cutoffs'], 1)
            self.assertEqual(row['external']['final']['outcomes']['true_ties'], 0)
            self.assertEqual(row['external']['final']['outcomes']['terminal_game_mean_predicted_value'], .75)
            self.assertEqual(row['external']['final']['outcomes']['terminal_game_mean_value_mse'], .0625)
            self.assertEqual(row['external']['contrast']['cutoff_aware_point_interval'], [.5, 1.])
            diagnostic = row['training']['optimizer_diagnostics']
            self.assertEqual(diagnostic['first_minibatch_per_cohort']['approx_kl']['n'], 2)
            self.assertEqual(diagnostic['last_epoch_minibatches']['approx_kl']['n'], 8)
            self.assertAlmostEqual(diagnostic['last_epoch_minibatches']['approx_kl']['mean'], .055)
        self.assertEqual(result['costs']['component_elapsed_seconds_sum'], 18.)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text()), result)
        prose = (self.output / 'results.md').read_text()
        self.assertIn('not passed', prose)
        self.assertIn('not total machine or GPU memory', prose)
        cli = subprocess.run([sys.executable, '-m', 'pokemon_structured_policy.reporting',
            '--training-run', str(self.inputs[0]), '--external-initial', str(self.inputs[1]),
            '--external-final', str(self.inputs[2]), '--verification', str(self.inputs[3]),
            '--output', str(self.base / 'cli-summary')], capture_output=True, text=True)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads((self.base / 'cli-summary/summary.json').read_text()), result)

    def test_incomplete_or_mismatched_reports_rejected(self):
        path = self.inputs[1] / 'report.json'
        original = json.loads(path.read_text())
        for change in (lambda r: r.update(status='failed'), lambda r: r.update(input_hash='wrong'),
                       lambda r: r['seeds'].pop(str(SEEDS[-1]))):
            with self.subTest(change=change):
                write(path, original)
                self.mutate(path, change)
                with self.assertRaises(ValueError):
                    reporting.run(*self.inputs, self.output)
                self.assertFalse(self.output.exists())

    def test_row_accounting_rejected(self):
        self.mutate(self.inputs[0] / str(SEEDS[0]) / 'update-0000.json',
            lambda r: r['ledger']['strata'][0]['selection'].update(excluded_rows=0))
        with self.assertRaisesRegex(ValueError, 'row accounting'):
            reporting.run(*self.inputs, self.output)

    def test_missing_update_rejected(self):
        (self.inputs[0] / str(SEEDS[0]) / 'update-0001.json').unlink()
        with self.assertRaisesRegex(ValueError, 'update inventory'):
            reporting.run(*self.inputs, self.output)

    def test_duplicate_verification_cell_rejected(self):
        def duplicate(r):
            cells = [v for v in r['records'] if v['kind'] == 'fresh_training' and v['seed'] == SEEDS[0]]
            cells[1].update(cells[0])
        self.mutate(self.inputs[3] / 'report.json', duplicate)
        with self.assertRaisesRegex(ValueError, 'coverage coordinates'):
            reporting.run(*self.inputs, self.output)

    def test_external_record_drift_rejected(self):
        self.mutate(self.inputs[1] / str(SEEDS[0]) / '000-0/result.json', lambda r: r.update(seconds=2.))
        with self.assertRaisesRegex(ValueError, 'external game digest'):
            reporting.run(*self.inputs, self.output)

    def test_verification_receipt_cannot_be_transplanted_or_stale(self):
        path = self.inputs[3] / 'report.json'
        original = json.loads(path.read_text())
        self.mutate(path, lambda r: r['input_files'].pop(str(self.inputs[0] / 'report.json')))
        with self.assertRaisesRegex(ValueError, 'verification evidence paths'):
            reporting.run(*self.inputs, self.output)
        write(path, original)
        self.mutate(self.inputs[0] / 'report.json', lambda r: r.update(peak_parent_rss_mib=256.))
        with self.assertRaisesRegex(ValueError, 'verification input digest'):
            reporting.run(*self.inputs, self.output)

    def test_output_cannot_replace_original_evidence(self):
        with self.assertRaisesRegex(ValueError, 'output must be separate'):
            reporting.run(*self.inputs, self.inputs[0] / 'summary')
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            reporting.run(*self.inputs, self.output)


if __name__ == '__main__':
    unittest.main()
