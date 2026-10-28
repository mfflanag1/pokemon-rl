"""Focused F39 completion semantics and copied-evidence integrity negatives."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('f39_completion_checker', ROOT / 'scripts/check-online-continuation.py')
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
PLAN = json.loads((ROOT / checker.PLAN).read_text())


def value(games):
    return {'games': games, 'terminals': games, 'cutoffs': 0, 'decisions': games * 10,
        'learner_wins': games // 2, 'learner_losses': games - games // 2, 'true_ties': 0,
        'terminal_game_mean_value_mse': .25, 'terminal_game_mean_predicted_value': .5, 'cutoff_rate': 0.}


def fixture():
    """A completed no-improvement screen: completion must not demand learning success."""
    manifest = {'schema_version': 'f39-inputs-v1', 'plan': deepcopy(PLAN),
        'runtime': {'backend': 'cpu', 'x64': False}, 'qualification': {'status': 'passed'},
        'predecessor': {'path': 'artifacts/reports/f34-policy-reset.json', 'sha256': checker.F34_SHA}}
    verification = {'status': 'passed', 'input_hash': 'input', 'counts': deepcopy(checker.VERIFICATION),
        'expected_counts': deepcopy(checker.VERIFICATION), 'records': [], 'input_files': {},
        'coverage': {'exhaustive': False, 'teacher_search_module_inventory': {'before': [], 'after': [],
            'forbidden_prefixes': ['pokemon_teacher_assisted.teacher']},
            'teacher_search_execution_scope': 'Python module snapshots only; source inspection separately. Not an OS audit.'}, 'elapsed_seconds': 3.}
    training = {'schema_version': 'f39-online-continuation-run-v1', 'status': 'passed',
        'input_hash': 'input', 'plan': deepcopy(PLAN), 'seeds': {}, 'counts': deepcopy(checker.COUNTS),
        'all_seed_learning_passed': False, 'elapsed_seconds': 30., 'collection_seconds': 15.,
        'update_seconds_including_compile': 8., 'sampled_replay_update_seconds': 2., 'peak_parent_rss_mib': 100.}
    summary = {'schema_version': 'f39-measured-summary-v1', 'status': 'passed', 'input_hash': 'input',
        'statistics': deepcopy(checker.STATISTICS), 'seeds': {}, 'counts': deepcopy(checker.COUNTS),
        'primary_contrasts': 3, 'regression_diagnostic_contrasts': 9, 'total_contrasts': 12,
        'promotion_authorized': False, 'all_seed_primary_passed': False, 'any_detected_regression': False,
        'verification': {'counts': deepcopy(verification['counts']), 'coverage': deepcopy(verification['coverage'])},
        'costs': {k: training[k] for k in ('elapsed_seconds', 'collection_seconds', 'update_seconds_including_compile',
            'sampled_replay_update_seconds', 'peak_parent_rss_mib')}, 'regression_diagnostics': [], 'input_files': {}}
    aggregate = Counter()
    for seed in checker.SEEDS:
        records = verification['records']
        records.extend({'kind': 'checkpoint', 'seed': int(seed), 'cohort': c} for c in (512, 767, 768))
        records.extend({'kind': 'fresh_update', 'seed': int(seed), 'cohort': c} for c in (512, 767))
        records.extend({'kind': 'fresh_training', 'seed': int(seed), 'cohort': c, 'stream': s}
                       for c in (512, 767) for s in checker.STREAM_ROWS)
        records.extend({'kind': 'fresh_evaluation_pair', 'seed': int(seed), 'endpoint': e, 'opponent': o}
                       for e in ('initial', 'final') for o in checker.GAMES)
        streams = {}
        for stream, selected in checker.STREAM_ROWS.items():
            streams[stream] = {'games': 256, 'decisions': 2560, 'terminals': 256, 'cutoffs': 0, 'true_ties': 0,
                'collected_rows': selected + 256, 'selected_rows': selected, 'excluded_rows': 256,
                'non_learner_decisions': 0, 'rejected_attempts': 0, 'rejected_controlled_attempts': 0,
                'rejected_selected_rows': 0}
            streams[stream]['decisions'] = streams[stream]['collected_rows']
        totals = Counter()
        for cell in streams.values():
            totals.update(cell)
        aggregate.update(totals)
        outcomes = value(totals['games']); outcomes['decisions'] = totals['decisions']
        diagnostic = {'n': 256, 'mean': 0., 'median': 0., 'p95': 0., 'min': 0., 'max': 0., 'sum': 0.}
        account = {'counts': {**checker.SEED_COUNTS, 'training_games': 768}, 'streams': streams,
            'totals': dict(totals), 'outcomes': outcomes, 'optimizer_diagnostics': {
                'first_minibatch_per_cohort': {m: deepcopy(diagnostic) for m in ('approx_kl', 'clip_fraction')},
                'last_epoch_minibatches': {m: {**diagnostic, 'n': 1024} for m in ('approx_kl', 'clip_fraction')}}}
        endpoints = {e: {o: value(n) for o, n in checker.GAMES.items()} for e in ('initial', 'final')}
        contrasts = {}
        for opponent, n in checker.GAMES.items():
            radius = math.sqrt(2 * math.log(2 * 12 / .05) / (n // 2))
            contrasts[opponent] = {'initial': deepcopy(endpoints['initial'][opponent]),
                'final': deepcopy(endpoints['final'][opponent]), 'pairs': n // 2, 'bootstrap_resamples': 20000,
                'cutoff_aware_point_interval': [0., 0.], 'approximate_familywise_bootstrap_interval': [-.1, .1],
                'bounded_hoeffding_interval': [max(-1., -radius), min(1., radius)], 'gate': 'inconclusive'}
        analysis = {'contrasts': contrasts, 'detected_regressions': [], 'any_detected_regression': False,
            'primary_gate': {'checks': {'cutoff_aware_lower_point_gain_at_least_five_points': False,
                'adjusted_lower_positive': False, 'final_cutoff_rate_admitted': True}, 'passed': False}}
        summary['seeds'][seed] = {'training': account, 'analysis': analysis, 'endpoints': endpoints}
        summary['regression_diagnostics'].extend({'seed': int(seed), 'opponent': o, **deepcopy(c)}
            for o, c in contrasts.items() if o != 'baseline-v1')
        training['seeds'][seed] = {'seed': int(seed), 'identity': {'adam_reset': False,
            'f34_metadata': {'stream_cursors': {s: 10 for s in streams}}, 'f34_parameter_hash': 'p', 'state_hash': 'a'},
            'initial_optimizer_step': 4096, 'final_optimizer_step': 6144, 'optimizer_steps': 6144,
            'continuation_optimizer_steps': 2048, 'cumulative_selected_rows': 786432, 'cumulative_row_presentations': 1572864,
            'initial_stream_cursors': {s: 10 for s in streams}, 'stream_cursors': {s: 266 for s in streams},
            'initial_parameter_hash': 'p', 'initial_state_hash': 'a', 'analysis': deepcopy(analysis),
            'initial': deepcopy(endpoints['initial']), 'final': deepcopy(endpoints['final'])}
    summary['training_totals'] = dict(aggregate)
    summary['counts']['training_games'] = training['counts']['training_games'] = 2304
    exits, cleanups, launches = {}, {}, {}
    for i, (name, command) in enumerate(checker.expected_commands().items()):
        launch = {'schema_version': 'f39-stage-v1', 'stage': name, 'command': command, 'pid': 20 + i, 'started_utc': 'now'}
        launches[name] = launch
        exits[name] = {**launch, 'exit_code': 0, 'deadline_timeout': False, 'leaked_after_success': False,
            'wait_error': None, 'cleanup_status': 'passed', 'elapsed_seconds': 10.}
        cleanups[name] = {'status': 'passed', 'stage_was_running': False, 'root_pid': 20 + i, 'returncode': 0,
            'unknown_overflow': 0, 'table_error': None, **{k: [] for k in ('remaining_pids', 'errors',
                'identity_changed_skipped', 'live_descendants_at_cleanup', 'initial_live_pids', 'signals',
                'unresolved_unknown_pids', 'reused_pids', 'handle_signals')}}
    receipt = {'schema_version': 'f39-online-continuation-receipt-v1', 'status': 'completed',
        'input_hash': 'input', 'output_paths': deepcopy(checker.OUTPUT_PATHS),
        'summary': checker.OUTPUT_PATHS['results'] + '/summary.json', 'stages': list(exits.values()),
        'teacher_used': False, 'demonstrations_used': False, 'public_server_used': False,
        'final_reserved_data_used': False, 'promotion_authorized': False, 'learning_success_separate_from_execution': True}
    completion = {'schema_version': 'f39-completion-v1', 'status': 'passed', 'input_hash': 'input', 'receipt': checker.RECEIPT}
    workflow_exit = {**completion, 'schema_version': 'f39-workflow-exit-v1', 'exit_code': 0}
    driver_exit = {'schema_version': 'f39-driver-exit-v1', 'exit_code': 0, 'error': None,
        'cleanup': {**deepcopy(cleanups['training']), 'root_pid': 2}}
    return [receipt, manifest, summary, training, verification, completion, workflow_exit, driver_exit, exits, cleanups, launches]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(checker.canonical(data) + b'\n')


def sealed_fixture(root):
    records = fixture()
    receipt, manifest, summary, training, verification, completion, workflow_exit, driver_exit, exits, cleanups, launches = records
    review = root / checker.REVIEW
    write(root / 'artifacts/reports/f34-policy-reset.json', {'fixture': 'predecessor'})
    manifest['predecessor']['sha256'] = checker.sha(root / 'artifacts/reports/f34-policy-reset.json')
    write(root / checker.PLAN, PLAN)
    sources = {checker.PLAN: checker.sha(root / checker.PLAN)}
    manifest['source_files'] = sources
    manifest_path = review / 'admission/manifest.json'
    write(manifest_path, manifest)
    write(review / 'admission/source' / checker.PLAN, PLAN)
    receipt['input_manifest'] = str(manifest_path.relative_to(root))
    receipt['source_files'] = sources
    input_hash = checker.sha(manifest_path)
    for record in (receipt, summary, training, verification, completion, workflow_exit):
        record['input_hash'] = input_hash
    train = root / checker.OUTPUT_PATHS['training']
    write(train / 'inputs/manifest.json', manifest)
    write(train / 'inputs/source' / checker.PLAN, PLAN)
    write(train / 'report.json', training)
    input_files = {str(train / name): checker.sha(train / name) for name in ('inputs/manifest.json', 'report.json')}
    verification['input_files'] = input_files
    verify = root / checker.OUTPUT_PATHS['verification'] / 'report.json'
    write(verify, verification)
    summary['input_files'] = {**input_files, str(verify): checker.sha(verify)}
    write(root / receipt['summary'], summary)
    receipt['summary_sha256'] = checker.sha(root / receipt['summary'])
    for name in checker.STAGES:
        write(review / (name + '-exit.json'), exits[name])
        write(review / (name + '-launch.json'), launches[name])
        write(review / (name + '-cleanup.json'), cleanups[name])
        (review / (name + '.log')).write_text('completed\n')
    for name in ('authorization.json', 'qualification.json', 'prelaunch-independent.json'):
        write(review / name, {'status': 'passed'})
    write(review / 'launcher-claim.json', {'pid': 1})
    write(review / 'driver-launch.json', {'launcher_pid': 1, 'driver_pid': 2})
    write(review / 'workflow-claim.json', {'pid': 2})
    inventory = {}
    for relative in checker.OUTPUT_PATHS.values():
        inventory.update({str(p.relative_to(root)): checker.sha(p) for p in (root / relative).rglob('*') if p.is_file()})
    for path in review.glob('*'):
        if path.is_file():
            inventory[str(path.relative_to(root))] = checker.sha(path)
    receipt['evidence_files'] = inventory
    receipt['content_hash'] = hashlib.sha256(checker.canonical(receipt)).hexdigest()
    write(root / checker.RECEIPT, receipt)
    write(review / 'f39-online-continuation-receipt.json', receipt)
    for record in (completion, workflow_exit):
        record.update(receipt_sha256=checker.sha(root / checker.RECEIPT), source_files_verified=len(sources), evidence_files_verified=len(inventory))
    for name, record in [('completion.json', completion), ('workflow-exit.json', workflow_exit), ('driver-exit.json', driver_exit)]:
        write(review / name, record)
    return records


class CompletionTests(unittest.TestCase):
    def test_failed_learning_is_valid_completion(self):
        checker.validate_completion(*fixture())

    def test_each_missing_seed_is_rejected(self):
        for seed in checker.SEEDS:
            with self.subTest(seed=seed):
                records = fixture(); del records[2]['seeds'][seed]
                with self.assertRaisesRegex(ValueError, 'three seed'):
                    checker.validate_completion(*records)

    def test_cutoffs_cannot_be_terminal_targets(self):
        records = fixture()
        cell = records[2]['seeds']['7103']['endpoints']['final']['random']
        cell.update(terminals=0, cutoffs=64, learner_wins=0, learner_losses=0, cutoff_rate=1.)
        with self.assertRaisesRegex(ValueError, 'terminal-only'):
            checker.validate_completion(*records)

    def test_tampered_fixed_counts_rows_adam_and_gate_fail(self):
        cases = [lambda r: r[2]['counts'].__setitem__('selected_rows', 1),
            lambda r: r[2]['seeds']['7103']['training']['streams']['self_play'].__setitem__('excluded_rows', 1),
            lambda r: r[3]['seeds']['7103']['identity'].__setitem__('adam_reset', True),
            lambda r: r[2].__setitem__('all_seed_primary_passed', True),
            lambda r: r[2]['seeds']['7103']['analysis']['contrasts']['baseline-v1'].__setitem__('cutoff_aware_point_interval', [.1, .1]),
            lambda r: r[2]['regression_diagnostics'].pop(),
            lambda r: r[4]['records'].pop(),
            lambda r: r[4]['coverage']['teacher_search_module_inventory']['after'].append('pokemon_teacher_assisted.teacher'),
            lambda r: r[7].__setitem__('exit_code', 1)]
        for change in cases:
            with self.subTest(change=change):
                records = fixture(); change(records)
                with self.assertRaises(ValueError):
                    checker.validate_completion(*records)

    def test_unresolved_or_leaked_cleanup_fails(self):
        for key, value_ in [('remaining_pids', [9]), ('unresolved_unknown_pids', [9]),
                            ('handle_signals', ['TERM']), ('unknown_overflow', 1), ('table_error', 'failed')]:
            with self.subTest(key=key):
                records = fixture(); records[9]['training'][key] = value_
                with self.assertRaises(ValueError):
                    checker.validate_completion(*records)

    def test_outer_driver_cleanup_or_launcher_error_fails(self):
        mutations = [lambda r: r[7].__setitem__('error', 'launcher failure'),
            lambda r: r[7].__setitem__('cleanup', None),
            lambda r: r[7]['cleanup'].__setitem__('unresolved_unknown_pids', [999]),
            lambda r: r[7]['cleanup'].__setitem__('live_descendants_at_cleanup', [999]),
            lambda r: r[7]['cleanup'].__setitem__('returncode', 1)]
        for change in mutations:
            with self.subTest(change=change):
                records = fixture(); change(records)
                with self.assertRaises(ValueError):
                    checker.validate_completion(*records)

    def test_empty_launch_is_not_complete_evidence(self):
        records = fixture(); records[10]['training'] = {}
        with self.assertRaisesRegex(ValueError, 'launch/exit'):
            checker.validate_completion(*records)

    def test_wrong_teacher_wrapper_fails(self):
        records = fixture()
        records[0]['stages'][0]['command'].insert(4, 'pokemon_teacher_legality.stage')
        with self.assertRaises(ValueError):
            checker.validate_completion(*records)

    def test_unsafe_paths_fail(self):
        for path in ('../secret', '/tmp/file', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                checker.bound_path(path)

    def test_full_bytes_and_exclusive_postcheck(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(checker, 'ROOT', Path(tmp)):
            records = sealed_fixture(Path(tmp))
            with patch.object(checker, 'F34_SHA', records[1]['predecessor']['sha256']):
                output = Path(tmp) / 'postcheck.json'
                checker.main(output)
                self.assertEqual(json.loads(output.read_text())['status'], 'passed')
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    checker.main(output)
                with self.assertRaisesRegex(ValueError, 'outside sealed'):
                    checker.main(Path(tmp) / checker.REVIEW / 'postcheck.json')

    def test_mutated_source_archive_or_evidence_fails(self):
        for relative in (checker.PLAN, checker.REVIEW + '/admission/source/' + checker.PLAN,
                         checker.OUTPUT_PATHS['training'] + '/report.json',
                         checker.REVIEW + '/f39-online-continuation-receipt.json'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp, patch.object(checker, 'ROOT', Path(tmp)):
                records = sealed_fixture(Path(tmp)); path = Path(tmp) / relative
                path.write_bytes(path.read_bytes() + b' ')
                with patch.object(checker, 'F34_SHA', records[1]['predecessor']['sha256']), self.assertRaises(ValueError):
                    checker.main(Path(tmp) / 'postcheck.json')

    def test_unlisted_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(checker, 'ROOT', Path(tmp)):
            records = sealed_fixture(Path(tmp))
            (Path(tmp) / checker.OUTPUT_PATHS['training'] / 'unlisted.json').write_text('{}')
            with patch.object(checker, 'F34_SHA', records[1]['predecessor']['sha256']), self.assertRaisesRegex(ValueError, 'inventory'):
                checker.main(Path(tmp) / 'postcheck.json')


if __name__ == '__main__':
    unittest.main()
