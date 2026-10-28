"""F40 additive receipt, preservation and supervisor negatives; no scientific runs."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


checker = load('f40_completion_checker', 'scripts/check-online-verification.py')
runner = load('f40_runner_for_tests', 'scripts/run-online-verification.py')
f39 = load('f40_original_fixture', 'tests/test_f39_completion.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(checker.canonical(value) + b'\n')


def content(value):
    return hashlib.sha256(checker.canonical({k: v for k, v in value.items() if k != 'content_hash'})).hexdigest()


def fixture(root):
    """Small byte inventories plus the complete three-seed/twelve-cell semantic fixture."""
    records = f39.fixture()
    _, original, summary, training, verified, _, _, _, exits, cleanups, launches = records
    paths = {'pokemon_teacher_runtime': 'python/pokemon_teacher_runtime/__init__.py',
             'pokemon_teacher_runtime.lifecycle': 'python/pokemon_teacher_runtime/lifecycle.py'}
    for path in paths.values():
        write(root / path, {'fixture': path})
    sources = {path: checker.sha(root / path) for path in paths.values()}
    original['source_files'] = sources
    write(root / checker.ORIGINAL_ADMISSION, original)
    admission_sha = checker.sha(root / checker.ORIGINAL_ADMISSION)
    write(root / checker.TRAIN / 'inputs/manifest.json', original)
    old_review = str(Path(checker.ORIGINAL_ADMISSION).parents[1])
    for archive in (old_review + '/admission/source', checker.TRAIN + '/inputs/source'):
        for path in sources:
            destination = root / archive / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / path).read_bytes())
    training['input_hash'] = verified['input_hash'] = summary['input_hash'] = admission_sha
    write(root / checker.TRAIN / 'report.json', training)
    original_roots = [old_review, checker.TRAIN]
    evidence = {str(p.relative_to(root)): checker.sha(p) for base in original_roots
                for p in (root / base).rglob('*') if p.is_file()}
    failure_dir = 'artifacts/reviews/f39-online-continuation-failure-v1'
    for path in evidence:
        destination = root / failure_dir / 'evidence' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / path).read_bytes())
    failure = {'status': 'failed', 'completed_comparison': False, 'scientific_goal_completed': False,
        'admission_manifest': checker.ORIGINAL_ADMISSION, 'admission_sha256': admission_sha,
        'source_files': sources, 'source_files_count': len(sources), 'evidence_files': evidence,
        'evidence_files_count': len(evidence), 'source_archive': old_review + '/admission/source',
        'production_source_archive': checker.TRAIN + '/inputs/source', 'evidence_archive': failure_dir + '/evidence',
        'archived_receipt_path': failure_dir + '/receipt.json', 'inventory_roots': original_roots,
        'inventory_exclusions': [], 'inventory_seal_outputs_excluded': [],
        'required_outputs_absent': ['artifacts/reports/f39-online-continuation.json', old_review + '/measured-results-v1'],
        'training_exit': exits['training'], 'verification_exit': {'exit_code': 1},
        'workflow_exit': {'exit_code': 1}, 'driver_exit_summary': {'exit_code': 1}}
    failure['content_hash'] = content(failure)
    write(root / checker.FAILURE, failure); write(root / failure['archived_receipt_path'], failure)
    failure_sha = checker.sha(root / checker.FAILURE)
    prep = 'artifacts/reviews/f40-online-verification-preparation'
    policy = {'exact_runtime_exceptions': paths, 'forbidden_prefixes': checker.FORBIDDEN_PREFIXES,
              'exception_sha256': {name: sources[path] for name, path in paths.items()}}
    prep_values = {'qualification': {'status': 'passed'}, 'guard_policy': policy,
        'authorization': {'verification_only': True, 'no_new_training': True, 'deadline_utc': checker.DEADLINE}}
    bindings = {}
    for name, value in prep_values.items():
        path = prep + '/' + name + '.json'; write(root / path, value)
        bindings[name] = {'path': path, 'sha256': checker.sha(root / path)}
    new_sources = {b['path']: b['sha256'] for b in bindings.values()}
    for path in new_sources:
        destination = root / checker.REVIEW / 'admission/source' / path
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes((root / path).read_bytes())
    admission = {'schema_version': 'f40-verification-admission-v1', 'deadline_utc': checker.DEADLINE,
        'original_failure_receipt': {'path': checker.FAILURE, 'sha256': failure_sha},
        'original_admission': {'path': checker.ORIGINAL_ADMISSION, 'sha256': admission_sha},
        'source_files': new_sources, **bindings}
    write(root / checker.ADMISSION, admission)
    inputs = {str(root / checker.TRAIN / p): checker.sha(root / checker.TRAIN / p)
              for p in ('report.json', 'inputs/manifest.json')}
    verified['input_files'] = inputs
    verified['coverage']['teacher_search_module_inventory']['forbidden_prefixes'] = checker.FORBIDDEN_PREFIXES
    snapshot = {'forbidden': [], 'allowed_runtime_modules': [{'name': 'pokemon_teacher_runtime',
        'path': paths['pokemon_teacher_runtime'], 'sha256': sources[paths['pokemon_teacher_runtime']]}],
        'forbidden_prefixes': checker.FORBIDDEN_PREFIXES, 'exact_runtime_exceptions': paths,
        'original_admission_sha256': admission_sha}
    verified['f40_runtime_policy'] = {'schema_version': 'f40-runtime-policy-v1', 'snapshots': [snapshot, deepcopy(snapshot)],
        'production_repeated': False, 'original_failure_preserved': True}
    write(root / checker.VERIFY / 'report.json', verified)
    summary['verification']['coverage'] = deepcopy(verified['coverage'])
    summary['input_files'] = {**inputs, str(root / checker.VERIFY / 'report.json'): checker.sha(root / checker.VERIFY / 'report.json')}
    write(root / checker.RESULT / 'summary.json', summary)
    stages = []
    for i, name in enumerate(checker.STAGES):
        launch = {'schema_version': 'f40-stage-v1', 'stage': name, 'command': checker.expected_commands()[name],
            'pid': 21 + i, 'started_utc': '2026-10-28T22:00:00+00:00'}
        row = {**exits[name], **launch, 'ended_utc': '2026-10-28T22:00:10+00:00'}
        stages.append(row)
        for suffix, data in [('-exit.json', row), ('-launch.json', launch), ('-cleanup.json', {**cleanups[name], 'root_pid': launch['pid']})]:
            write(root / checker.REVIEW / (name + suffix), data)
        (root / checker.REVIEW / (name + '.log')).write_text('fixture\n')
    review = root / checker.REVIEW
    write(review / 'launcher-claim.json', {'pid': 1})
    write(review / 'workflow-claim.json', {'pid': 2})
    write(review / 'driver-launch.json', {'launcher_pid': 1, 'driver_pid': 2,
        'command': [str(root / '.venv/bin/python'), str(root / 'scripts/run-online-verification.py'), '--worker']})
    write(review / 'driver-exit.json', {'schema_version': 'f40-driver-exit-v1', 'exit_code': 0, 'error': None,
        'ended_unix': 1793224820., 'cleanup': {**cleanups['training'], 'root_pid': 2}})
    (review / 'driver.log').write_text('fixture\n')
    preserved = {'failure_receipt_sha256': failure_sha, 'source_files_verified': len(sources), 'evidence_files_verified': len(evidence)}
    new_admission_sha = checker.sha(root / checker.ADMISSION)
    write(review / 'workflow-result.json', {'status': 'passed', 'stages': stages, 'admission_sha256': new_admission_sha,
        'runtime_before': original['runtime'], 'runtime_after': original['runtime'],
        'preservation_before': preserved, 'preservation_after': preserved})
    receipt = {'schema_version': 'f40-online-verification-receipt-v1', 'status': 'completed',
        'admission_manifest': checker.ADMISSION, 'admission_sha256': new_admission_sha,
        'source_files': new_sources, 'original_failure_receipt': admission['original_failure_receipt'],
        'original_workflow_status': 'failed', 'deadline_utc': checker.DEADLINE,
        'output_paths': {'inherited_training': checker.TRAIN, 'verification': checker.VERIFY, 'results': checker.RESULT},
        'new_work': {'training_cohorts': 0, 'selected_training_rows': 0, 'training_optimizer_steps': 0,
                     'fresh_replay_games': 66, 'recomputed_optimizer_steps': 48},
        'learning_success_separate_from_execution': True, 'teacher_used': False, 'demonstrations_used': False,
        'public_server_used': False, 'final_reserved_data_used': False, 'promotion_authorized': False, 'stages': stages,
        'inherited_training': {'report': checker.TRAIN + '/report.json', 'sha256': checker.sha(root / checker.TRAIN / 'report.json'),
            'original_stage': failure['training_exit'], 'counts': training['counts'], 'seeds': list(checker.old.SEEDS)},
        'summary': checker.RESULT + '/summary.json', 'summary_sha256': checker.sha(root / checker.RESULT / 'summary.json')}
    reseal(root, receipt)
    return {'FAILURE_SHA': failure_sha, 'ORIGINAL_ADMISSION_SHA': admission_sha,
            'SOURCE_COUNT': len(sources), 'EVIDENCE_COUNT': len(evidence)}, receipt


def reseal(root, receipt):
    receipt['evidence_files'] = {checker.REVIEW + '/' + name: checker.sha(path)
        for name, path in checker.files(root / checker.REVIEW).items() if checker.REVIEW + '/' + name not in checker.SEAL_EXCLUSIONS}
    receipt['content_hash'] = content(receipt)
    write(root / checker.RECEIPT, receipt)
    write(root / checker.REVIEW / 'f40-online-verification-receipt.json', receipt)
    write(root / checker.REVIEW / 'completion.json', {'schema_version': 'f40-completion-v1', 'status': 'passed',
        'receipt': checker.RECEIPT, 'receipt_sha256': checker.sha(root / checker.RECEIPT),
        'admission_sha256': receipt['admission_sha256'], 'source_files_verified': len(receipt['source_files']),
        'evidence_files_verified': len(receipt['evidence_files'])})


class F40CompletionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(checker, 'ROOT', self.root))
        self.stack.enter_context(patch.object(checker, 'REQUIRED_SOURCES', set()))
        constants, self.receipt = fixture(self.root)
        for key, value in constants.items():
            self.stack.enter_context(patch.object(checker, key, value))

    def mutate(self, relative, fn, seal=True):
        path = self.root / relative; data = checker.read(path); fn(data); write(path, data)
        if seal:
            reseal(self.root, self.receipt)

    def test_completed_failed_learning_still_passes(self):
        _, summary = checker.verify_completion()
        self.assertFalse(summary['all_seed_primary_passed'])
        self.assertEqual(summary['total_contrasts'], 12)

    def test_exclusive_outer_postcheck(self):
        checker.main(self.root / checker.POSTCHECK)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            checker.main(self.root / checker.POSTCHECK)
        with self.assertRaisesRegex(ValueError, 'exclusive outer'):
            checker.main(self.root / checker.REVIEW / 'postcheck.json')

    def test_original_source_live_and_both_archives_tampering_fails(self):
        for prefix in ('', 'artifacts/reviews/f39-online-continuation/admission/source/', checker.TRAIN + '/inputs/source/'):
            path = self.root / (prefix + 'python/pokemon_teacher_runtime/__init__.py')
            saved = path.read_bytes(); path.write_bytes(saved + b' ')
            with self.assertRaises(ValueError): checker.verify_completion()
            path.write_bytes(saved)

    def test_original_evidence_archive_and_receipt_tampering_fails(self):
        failure = checker.read(self.root / checker.FAILURE)
        for relative in (checker.TRAIN + '/report.json', failure['evidence_archive'] + '/' + checker.TRAIN + '/report.json',
                         failure['archived_receipt_path'], checker.FAILURE):
            path = self.root / relative; saved = path.read_bytes(); path.write_bytes(saved + b' ')
            with self.assertRaises(ValueError): checker.verify_completion()
            path.write_bytes(saved)

    def test_unlisted_original_and_new_output_rejected(self):
        for base in (checker.TRAIN, checker.REVIEW, 'artifacts/reviews/f39-online-continuation/admission/source'):
            path = self.root / base / 'unlisted.json'; write(path, {})
            with self.assertRaisesRegex(ValueError, 'inventory'): checker.verify_completion()
            path.unlink()

    def test_original_success_cannot_be_fabricated(self):
        write(self.root / 'artifacts/reports/f39-online-continuation.json', {})
        with self.assertRaisesRegex(ValueError, 'remain absent'): checker.verify_completion()

    def test_new_receipt_archive_and_completion_linkage(self):
        self.mutate(checker.REVIEW + '/completion.json', lambda d: d.__setitem__('receipt_sha256', 'bad'), seal=False)
        with self.assertRaisesRegex(ValueError, 'completion receipt'): checker.verify_completion()
        reseal(self.root, self.receipt)
        path = self.root / checker.REVIEW / 'f40-online-verification-receipt.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'archived receipt'): checker.verify_completion()

    def test_unresolved_outer_cleanup_rejected_even_after_reseal(self):
        self.mutate(checker.REVIEW + '/driver-exit.json', lambda d: d['cleanup'].__setitem__('unresolved_unknown_pids', [9]))
        with self.assertRaisesRegex(ValueError, 'cleanup'): checker.verify_completion()

    def test_scope_counts_and_third_stage_rejected(self):
        self.receipt['new_work']['selected_training_rows'] = 1; reseal(self.root, self.receipt)
        with self.assertRaisesRegex(ValueError, 'work counts'): checker.verify_completion()
        self.receipt['new_work']['selected_training_rows'] = 0
        self.receipt['stages'].append({'stage': 'training'}); reseal(self.root, self.receipt)
        with self.assertRaisesRegex(ValueError, 'two ordered'): checker.verify_completion()

    def test_policy_prefix_weakening_rejected(self):
        verification = checker.read(self.root / checker.VERIFY / 'report.json')
        policy = {'exact_runtime_exceptions': verification['f40_runtime_policy']['snapshots'][0]['exact_runtime_exceptions'],
                  'forbidden_prefixes': ['pokemon_teacher_assisted.teacher']}
        with self.assertRaisesRegex(ValueError, 'exact lifecycle'):
            checker.validate_policy(verification, policy, checker.read(self.root / checker.ORIGINAL_ADMISSION))

    def test_policy_extra_runtime_child_or_wrong_hash_rejected(self):
        verification = checker.read(self.root / checker.VERIFY / 'report.json')
        original = checker.read(self.root / checker.ORIGINAL_ADMISSION)
        policy = {'exact_runtime_exceptions': verification['f40_runtime_policy']['snapshots'][0]['exact_runtime_exceptions'],
                  'forbidden_prefixes': checker.FORBIDDEN_PREFIXES,
                  'exception_sha256': {name: original['source_files'][path] for name, path in
                      verification['f40_runtime_policy']['snapshots'][0]['exact_runtime_exceptions'].items()}}
        for mutation in (lambda d: d['allowed_runtime_modules'][0].__setitem__('sha256', 'bad'),
                         lambda d: d['allowed_runtime_modules'].append({'name': 'pokemon_teacher_runtime.transport'})):
            changed = deepcopy(verification); mutation(changed['f40_runtime_policy']['snapshots'][0])
            with self.assertRaises(ValueError): checker.validate_policy(changed, policy, original)

    def test_missing_seed_contrast_and_recomputed_steps_rejected(self):
        training = checker.read(self.root / checker.TRAIN / 'report.json')
        verification = checker.read(self.root / checker.VERIFY / 'report.json')
        summary = checker.read(self.root / checker.RESULT / 'summary.json')
        original = checker.read(self.root / checker.ORIGINAL_ADMISSION)
        for mutate in (lambda s, v: s['seeds'].pop('8209'), lambda s, v: s['regression_diagnostics'].pop(),
                       lambda s, v: v['counts'].__setitem__('optimizer_steps_recomputed', 49),
                       lambda s, v: v['records'].pop()):
            s, v = deepcopy(summary), deepcopy(verification); mutate(s, v)
            with self.assertRaises(ValueError): checker.validate_reports(training, v, s, original)

    def test_unsafe_paths_and_unbound_report_inputs_rejected(self):
        for value in ('../outside', '/tmp/outside', ''):
            with self.assertRaises(ValueError): checker.bound(value)
        verification = checker.read(self.root / checker.VERIFY / 'report.json')
        verification['input_files'][str(self.root / checker.TRAIN / 'not-in-inventory.json')] = 'bad'
        with self.assertRaises((ValueError, FileNotFoundError)):
            checker.validate_inputs(verification, self.receipt['evidence_files'], {})

    def test_stale_policy_exception_hash_rejected(self):
        verification = checker.read(self.root / checker.VERIFY / 'report.json')
        original = checker.read(self.root / checker.ORIGINAL_ADMISSION)
        manifest = checker.read(self.root / checker.ADMISSION)
        policy = checker.read(self.root / manifest['guard_policy']['path'])
        policy['exception_sha256']['pokemon_teacher_runtime'] = 'stale'
        with self.assertRaisesRegex(ValueError, 'policy SHA'):
            checker.validate_policy(verification, policy, original)

    def test_outer_validation_precedes_any_success_receipt_write(self):
        for path in (checker.RECEIPT, checker.REVIEW + '/f40-online-verification-receipt.json',
                     checker.REVIEW + '/completion.json'):
            (self.root / path).unlink()
        self.mutate(checker.REVIEW + '/driver-exit.json',
                    lambda d: d['cleanup'].__setitem__('unresolved_unknown_pids', [9]), seal=False)
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'REVIEW', self.root / checker.REVIEW), \
             patch.object(runner, 'check', checker), patch.object(runner, 'deadline'):
            with self.assertRaisesRegex(ValueError, 'cleanup'):
                runner.seal_success()
        self.assertFalse((self.root / checker.RECEIPT).exists())
        self.assertFalse((self.root / checker.REVIEW / 'completion.json').exists())

    def test_failed_seal_binds_partial_success_and_invalidates_it(self):
        partial = {path: checker.sha(self.root / path) for path in (checker.RECEIPT,
            checker.REVIEW + '/f40-online-verification-receipt.json', checker.REVIEW + '/completion.json')}
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'REVIEW', self.root / checker.REVIEW), \
             patch.object(runner, 'check', checker):
            runner.seal_failure(OSError('fixture late seal failure'))
        failure = checker.read(self.root / checker.FAILURE_RECEIPT)
        self.assertEqual(failure['partial_success_artifacts'], partial)
        self.assertFalse(failure['scientific_completion'])
        self.assertFalse(failure['candidate_success_authoritative'])
        self.assertEqual((self.root / checker.FAILURE_RECEIPT).read_bytes(),
                         (self.root / checker.REVIEW / 'f40-online-verification-failure-receipt.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'failure evidence'):
            checker.verify_completion()

    def test_workflow_failure_rejects_otherwise_valid_candidate(self):
        write(self.root / checker.REVIEW / 'workflow-failure.json', {'status': 'failed'})
        reseal(self.root, self.receipt)
        with self.assertRaisesRegex(ValueError, 'failure evidence'):
            checker.verify_completion()

    def test_stage_nonzero_missing_stage_wrong_command_and_cleanup_rejected(self):
        stages = deepcopy(self.receipt['stages'])
        groups = [{name: checker.read(self.root / checker.REVIEW / (name + suffix)) for name in checker.STAGES}
                  for suffix in ('-exit.json', '-cleanup.json', '-launch.json')]
        for case in ('missing', 'nonzero', 'command', 'cleanup', 'late'):
            rows, (exits, cleanups, launches) = deepcopy(stages), deepcopy(groups)
            name = checker.STAGES[0]
            if case == 'missing':
                exits.pop(name)
            elif case == 'nonzero':
                rows[0]['exit_code'] = exits[name]['exit_code'] = 1
            elif case == 'command':
                for row in (rows[0], exits[name], launches[name]):
                    row['command'][4] = 'pokemon_online_continuation.campaign'
            elif case == 'cleanup':
                cleanups[name]['remaining_pids'] = [99]
            else:
                rows[0]['ended_utc'] = exits[name]['ended_utc'] = '2026-10-29T00:13:22+00:00'
            with self.subTest(case=case), self.assertRaises(ValueError):
                checker.validate_stages(rows, exits, cleanups, launches)

    def test_existing_outputs_block_second_attempt_before_spawn(self):
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'REVIEW', self.root / checker.REVIEW), \
             patch.object(runner, 'check', checker), patch.object(runner, 'deadline'), \
             patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                runner.main()
            spawn.assert_not_called()

    def test_failed_stage_records_actual_exit_without_retry(self):
        attempt = self.root / 'attempt'; attempt.mkdir()
        name = checker.STAGES[0]
        class Child:
            pid = 22
            def poll(self): return 7
        cleaned = {'status': 'passed', 'root_pid': 22, 'returncode': 7,
                   'live_descendants_at_cleanup': [], 'remaining_pids': []}
        with patch.object(runner, 'REVIEW', attempt), patch.object(runner, 'check', checker), \
             patch.object(runner, 'deadline'), patch.object(runner, 'cleanup', return_value=cleaned), \
             patch.object(runner.subprocess, 'Popen', return_value=Child()) as spawn:
            with self.assertRaisesRegex(ValueError, 'failed new stage'):
                runner.stage(name, checker.expected_commands()[name], attempt / 'missing-report.json')
            self.assertEqual(checker.read(attempt / (name + '-exit.json'))['exit_code'], 7)
            with self.assertRaises(FileExistsError):
                runner.stage(name, checker.expected_commands()[name], attempt / 'missing-report.json')
            self.assertEqual(spawn.call_count, 1)

    def test_deadline_and_exact_commands_prevent_new_training(self):
        with patch.object(runner.time, 'time', return_value=runner.STOP), self.assertRaises(TimeoutError):
            runner.deadline()
        self.assertEqual(tuple(checker.expected_commands()), checker.STAGES)
        self.assertTrue(all('pokemon_online_continuation.campaign' not in command for command in checker.expected_commands().values()))


if __name__ == '__main__':
    unittest.main()
