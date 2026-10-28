"""F37 workflow and journal integrity checks; no game, search or learner execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_script('f37_checker_tests', 'check-teacher-legality.py')
admitter = load_script('f37_admitter_tests', 'admit-teacher-legality.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def deadline_approval():
    return {'schema_version': 'f37-deadline-authorization-v1', 'new_attempt': 'F37', 'approved': True,
        'approved_deadline_utc': '2026-10-28T18:00:00Z', 'original_deadline_utc': '2026-10-28T07:36:08Z',
        'original_start_utc': '2026-10-27T23:36:08Z', 'original_checkpoint_utc': '2026-10-28T03:36:08Z',
        'clock_reset': False, 'scientific_seeds_or_budgets_changed': False,
        'single_fresh_comparison_only': True, 'github_publication_authorized': False}


def selection_audit(forced):
    if forced:
        return {'schema_version': 'f37-forced-request-v1', 'action': 0, 'legal_action_count': 1,
            'native_search_invoked': False, 'selection_rng_draws': 0,
            'reason': 'Exactly one authoritative legal action.'}
    return {'schema': 'f37-legal-selection-v1', 'outcome': 'selected', 'decision': 'move0',
        'decision_action': 0, 'mask_true_count': 2, 'threshold_fraction': .75, 'threshold': .75,
        'worlds': [{'position': i, 'index': i, 'chance': .5, 'total_visits': 10., 'option_count': 1} for i in range(2)],
        'candidates': [{'choice': 'move0', 'insertion_index': 0, 'mass': 1., 'action': 0, 'allowed': True, 'reason': None}],
        'aggregated_count': 1, 'aggregated_mass': 1., 'legal_count': 1, 'highest_legal_mass': 1.,
        'retained': [{'choice': 'move0', 'action': 0, 'mass': 1.}], 'retained_count': 1,
        'retained_mass': 1., 'excluded_count': 0, 'excluded_mass': 0.}


def game_fixture(folder):
    masks = [[True] + [False] * 13, [False] * 14, [True, True] + [False] * 12]
    responses = []
    for index, mask in enumerate(masks):
        if not any(mask):
            responses.append({'waiting': True})
            continue
        forced = sum(mask) == 1
        responses.append({'action': 0, 'decision': 'move0', 'request_id': index + 1, 'teacher_policy': checker.TEACHER_POLICY,
            'search_time_ms': 100, 'sampled_worlds': 0 if forced else 2,
            'per_world_ms': 0 if forced else 100, 'configured_search_total_ms': 0 if forced else 200,
            'selection': {'kind': 'forced-request' if forced else 'native-legal-root',
                'legal_mask': mask, 'request_id': index + 1, 'search_calls': 0 if forced else 1,
                'audit': selection_audit(forced)}})
    steps, frames, journal, labels = [], [], [], []
    previous = []
    for index, (mask, response) in enumerate(zip(masks, responses)):
        external = {'synthetic_input': index}
        frames.append({'external': external, 'frame': {'players': {'p2': {'request_id': index + 1,
            'mask': mask}}, 'submitted': previous}})
        submitted = [] if not any(mask) else [{'side': 'p2', 'action': 0, 'request_id': index + 1, 'accepted': True}]
        step = {'index': index, 'actions': {} if not any(mask) else {'p2': 0},
                'request_ids': {} if not any(mask) else {'p2': index + 1},
                'external': response, 'submitted': submitted}
        if submitted:
            step['teacher_feedback'] = {'acknowledged': True, 'request_id': index + 1, 'accepted': True}
            labels.append({'step_index': index, 'submission_index': 0, 'action': 0, 'request_id': index + 1})
        steps.append(step); previous = submitted
        journal.append({'schema_version': 'f37-teacher-response-v1', 'status': 'response',
            'step_index': index, 'request_id': index + 1, 'legal_mask': mask,
            'external_input_sha256': checker.json_hash(external), 'raw_response': response})
    frames.append({'frame': {'submitted': previous}})
    record = {'status': 'completed', 'teacher_policy': checker.TEACHER_POLICY,
        'teacher_response_journal': 'teacher-responses.jsonl', 'external_side': 'p2', 'steps': steps,
        'capture_demonstrations': True, 'accepted_labels': 2,
        'teacher_policy_counts': dict(forced_request=1, native_legal_root=1, waiting=1, errors=0)}
    write(folder / 'result.json', record); write(folder / 'labels.json', {'labels': labels})
    (folder / 'transcript.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in frames))
    (folder / 'teacher-responses.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in journal))
    return record, frames, journal, labels


class JournalTests(unittest.TestCase):
    def test_complete_journal_joins_forced_native_waiting_and_accepted_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); game_fixture(folder)
            self.assertEqual(checker.validate_teacher_game(folder, capture_demonstrations=True),
                dict(games=1, responses=3, forced_request=1, native_legal_root=1, waiting=1, errors=0))

    def test_missing_extra_error_or_altered_journal_rows_fail(self):
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[-1])),
            lambda rows: rows[0].update(status='error'), lambda rows: rows[0].update(request_id=99),
            lambda rows: rows[0].update(external_input_sha256='0' * 64),
            lambda rows: rows[0]['raw_response'].update(action=1)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary); _, _, rows, _=game_fixture(folder);mutate(rows)
                (folder/'teacher-responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                with self.assertRaises(ValueError): checker.validate_teacher_game(folder,capture_demonstrations=True)

    def test_invalid_policy_forced_search_or_native_fallback_fails(self):
        mutations = [lambda r: r['steps'][0]['external'].update(teacher_policy='old-policy'),
            lambda r: r['steps'][0]['external']['selection'].update(search_calls=1),
            lambda r: r['steps'][0]['external'].update(sampled_worlds=1),
            lambda r: r['steps'][2]['external']['selection'].update(kind='fallback'),
            lambda r: r['steps'][2]['external']['selection'].update(search_calls=2)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary); record, _, rows, _=game_fixture(folder);mutate(record)
                for row, step in zip(rows,record['steps']):row['raw_response']=step['external']
                write(folder/'result.json',record)
                (folder/'teacher-responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                with self.assertRaises(ValueError):checker.validate_teacher_game(folder,capture_demonstrations=True)

    def test_counter_or_unaccepted_label_is_rejected(self):
        for change in ('counts','labels','rejection'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary); record, frames, rows, labels=game_fixture(folder)
                if change=='counts':record['teacher_policy_counts']['forced_request']=2
                elif change=='labels':labels.append({'step_index':1,'submission_index':0,'action':0,'request_id':2})
                else:
                    record['steps'][0]['submitted'][0]['accepted']=False
                    frames[1]['frame']['submitted']=record['steps'][0]['submitted']
                write(folder/'result.json',record);write(folder/'labels.json',{'labels':labels})
                (folder/'transcript.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in frames))
                with self.assertRaises(ValueError):checker.validate_teacher_game(folder,capture_demonstrations=True)

    def test_selection_audit_rejects_changed_threshold_mass_decision_and_forced_metadata(self):
        mutations = [lambda a: a.update(threshold=.8), lambda a: a.update(decision='foreign'),
            lambda a: a.update(aggregated_mass=2.), lambda a: a.update(excluded_count=1),
            lambda a: a['retained'][0].update(action=2),
            lambda a: a['candidates'][0].update(allowed=False, reason='illegal_action'),
            lambda a: a['worlds'][0].update(total_visits=0)]
        for mutate in mutations:
            response = {'action': 0, 'decision': 'move0', 'sampled_worlds': 2, 'selection': {'audit': selection_audit(False)}}
            mutate(response['selection']['audit'])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                checker.validate_selection_audit(response, [0, 1])
        response = {'action': 0, 'selection': {'audit': selection_audit(True)}}
        response['selection']['audit']['selection_rng_draws'] = 1
        with self.assertRaises(ValueError): checker.validate_selection_audit(response, [0])

    def test_feedback_acknowledgement_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); record, _, _, _ = game_fixture(folder)
            record['steps'][0].pop('teacher_feedback'); write(folder / 'result.json', record)
            with self.assertRaisesRegex(ValueError, 'acknowledgement'):
                checker.validate_teacher_game(folder, capture_demonstrations=True)

    def test_waiting_frame_rejects_teacher_submission_or_feedback(self):
        for change in ('submission', 'feedback', 'null_feedback'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary); record, frames, _, _ = game_fixture(folder)
                step = record['steps'][1]
                if change == 'submission':
                    step['submitted'] = [{'side': 'p2', 'action': 0, 'request_id': 2, 'accepted': True}]
                    frames[2]['frame']['submitted'] = step['submitted']
                else:
                    step['teacher_feedback'] = None if change == 'null_feedback' else {'acknowledged': True}
                write(folder / 'result.json', record)
                (folder / 'transcript.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in frames))
                with self.assertRaisesRegex(ValueError, 'Waiting teacher submission/feedback'):
                    checker.validate_teacher_game(folder, capture_demonstrations=True)

    def test_waiting_teacher_allows_only_matching_learner_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); record, frames, _, _ = game_fixture(folder)
            frames[1]['frame']['players']['p1'] = {'request_id': 9, 'mask': [True] + [False] * 13}
            step = record['steps'][1]
            step.update(actions={'p1': 0}, request_ids={'p1': 9},
                submitted=[{'side': 'p1', 'action': 0, 'request_id': 9, 'accepted': True}])
            frames[2]['frame']['submitted'] = step['submitted']
            write(folder / 'result.json', record)
            (folder / 'transcript.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in frames))
            self.assertEqual(checker.validate_teacher_game(folder, capture_demonstrations=True)['waiting'], 1)

    def test_all_submission_sides_counts_and_action_request_ids_match(self):
        frame = {'players': {side: {'request_id': i + 1, 'mask': [True] + [False] * 13}
            for i, side in enumerate(('p1', 'p2'))}}
        original = {'actions': {'p1': 0, 'p2': 0}, 'request_ids': {'p1': 1, 'p2': 2},
            'submitted': [{'side': side, 'action': 0, 'request_id': i + 1, 'accepted': True}
                for i, side in enumerate(('p1', 'p2'))]}
        checker.validate_submissions(frame, original)
        mutations = [lambda s: s['submitted'].pop(),
            lambda s: s['submitted'].append(copy.deepcopy(s['submitted'][0])),
            lambda s: s['submitted'][1].update(side='p1'),
            lambda s: s['submitted'][1].update(side='p3'),
            lambda s: s['submitted'][1].update(action=1),
            lambda s: s['submitted'][1].update(request_id=99),
            lambda s: s['submitted'][1].update(accepted=1),
            lambda s: s['request_ids'].pop('p1'),
            lambda s: s['request_ids'].update(p1=99)]
        for mutate in mutations:
            step = copy.deepcopy(original); mutate(step)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                checker.validate_submissions(frame, step)

    def test_role_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);game_fixture(folder)
            with self.assertRaises(ValueError):checker.validate_teacher_game(folder,capture_demonstrations=False)

    def test_full_inventory_is_required_without_reusing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            write(folder/'report.json',{'status':'passed','games':[]})
            with self.assertRaises(ValueError):checker.validate_teacher_journals(folder)
            write(folder/'report.json',{'status':'passed','games':[{'directory':'same'}]*480})
            with mock.patch.object(checker,'validate_teacher_game',return_value=dict(games=1,responses=0,forced_request=0,native_legal_root=0,waiting=0,errors=0)):
                with self.assertRaisesRegex(ValueError,'reuse'):checker.validate_teacher_journals(folder)


class BindingTests(unittest.TestCase):
    def test_actual_cli_plan_bytes_and_only_approved_deadline_change_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = {'seeds': [7103, 8209, 9311], 'deadline_utc': checker.ORIGINAL_DEADLINE,
                'start_utc': '2026-10-27T23:36:08Z', 'checkpoint_utc': '2026-10-28T03:36:08Z',
                'cohorts_per_seed_arm': 256}
            approval = checker.BINDINGS['deadline_authorization']; write(root / approval, deadline_approval())
            approved_hash = checker.sha(root / approval)
            plan = {**original, 'deadline_utc': checker.APPROVED_DEADLINE,
                'execution': {'deadline_authorization': {'path': approval, 'sha256': approved_hash}}}
            write(root / 'contracts/teacher-assisted-v1/plan.json', original); write(root / checker.PLAN_RELATIVE, plan)
            manifest = {'plan': plan, 'source_files': {checker.PLAN_RELATIVE: checker.sha(root / checker.PLAN_RELATIVE),
                approval: approved_hash}}
            with mock.patch.object(checker, 'ROOT', root):
                checker.verify_plan_binding(manifest)
                changed = copy.deepcopy(manifest); changed['source_files'].pop(approval)
                with self.assertRaisesRegex(ValueError, 'authorization is not bound'): checker.verify_plan_binding(changed)
                changed = copy.deepcopy(manifest); changed['source_files'].pop(checker.PLAN_RELATIVE)
                with self.assertRaisesRegex(ValueError, 'CLI plan is not bound'): checker.verify_plan_binding(changed)
                for field, value in [('deadline_utc', checker.ORIGINAL_DEADLINE),
                        ('deadline_utc', '2027-01-01T00:00:00Z'), ('start_utc', '2026-10-28T11:00:00Z'),
                        ('checkpoint_utc', '2026-10-28T15:00:00Z'), ('cohorts_per_seed_arm', 128), ('seeds', [7103])]:
                    changed = copy.deepcopy(manifest); changed['plan'][field] = value
                    write(root / checker.PLAN_RELATIVE, changed['plan'])
                    changed['source_files'][checker.PLAN_RELATIVE] = checker.sha(root / checker.PLAN_RELATIVE)
                    with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                        checker.verify_plan_binding(changed)

    def test_deadline_authorization_rejects_reset_budgets_false_approval_and_other_cutoff(self):
        plan = {'deadline_utc': checker.APPROVED_DEADLINE, 'start_utc': '2026-10-27T23:36:08Z',
            'checkpoint_utc': '2026-10-28T03:36:08Z'}
        checker.validate_deadline_authorization(deadline_approval(), plan)
        for key, value in [('approved', False), ('approved', 1), ('clock_reset', True),
                ('scientific_seeds_or_budgets_changed', True), ('single_fresh_comparison_only', False),
                ('approved_deadline_utc', '2026-10-28T19:00:00Z'), ('github_publication_authorized', True)]:
            approval = deadline_approval(); approval[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checker.validate_deadline_authorization(approval, plan)

    def test_previous_qualification_uses_archived_sources_and_unchanged_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); relative = checker.REVIEW + '/legality-qualification-v1.json'
            archive = checker.REVIEW + '/qualification-source-v1'
            required = checker.NEW_SOURCES - {'python/pokemon_teacher_legality/deadline.py', 'tests/test_f37_deadline.py'}
            sources = {}
            for name in required:
                path = root / archive / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('old qualified bytes')
                sources[name] = checker.sha(path)
                live = root / name; live.parent.mkdir(parents=True, exist_ok=True); live.write_text('new deadline-only revision')
            old_log = checker.REVIEW + '/old.log'; (root / old_log).write_text('passed old checks')
            write(root / relative, {'status': 'passed', 'source_files': sources,
                'evidence_files': {old_log: checker.sha(root / old_log)}})
            old_hash = checker.sha(root / relative)
            write(root / archive / 'manifest.json', {'schema_version': 'f37-qualified-source-archive-v1',
                'qualification_sha256': old_hash, 'source_files': sources})
            evidence = {relative: old_hash, archive + '/manifest.json': checker.sha(root / archive / 'manifest.json'),
                **{archive + '/' + path: expected for path, expected in sources.items()}}
            record = {'previous_qualification': {'path': relative, 'sha256': old_hash, 'source_archive': archive},
                'evidence_files': evidence}
            with mock.patch.object(checker, 'ROOT', root):
                checker.verify_previous_qualification(record)
                altered = copy.deepcopy(record); altered['evidence_files'].pop(relative)
                with self.assertRaisesRegex(ValueError, 'evidence not bound'): checker.verify_previous_qualification(altered)
                (root / old_log).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'evidence drift'): checker.verify_previous_qualification(record)
                (root / old_log).write_text('passed old checks')
                (root / archive / sorted(required)[0]).write_text('changed')
                with self.assertRaisesRegex(ValueError, 'archived qualified source mismatch'):
                    checker.verify_previous_qualification(record)

    def test_deadline_installs_before_frozen_common_imports(self):
        for filename in ('run-teacher-legality.py', 'admit-teacher-legality.py'):
            tree = ast.parse((ROOT / 'scripts' / filename).read_text())
            calls = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'install_deadline']
            imports = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == 'pokemon_teacher_assisted.common']
            self.assertEqual(len(calls), 1); self.assertEqual(len(imports), 1)
            self.assertLess(calls[0], imports[0])

    def test_execution_requires_explicit_authorization_and_non_circular_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sources={p:'a'*64 for p in checker.NEW_SOURCES|{'tests/test_f37_workflow.py'}}
            source_record={'status':'passed','source_files':sources,'evidence_files':{'review.md':'b'*64},'independent_reviews':{'review':'review.md'}}
            records={'failed_predecessor':{'status':'failed'},'runtime_qualification':source_record,
                'deadline_authorization':deadline_approval(),
                'legality_qualification':copy.deepcopy(source_record),
                'authorization':{'schema_version':'f37-authorization-v1','new_attempt':'F37','scientific_launch_authorized_after_qualification':True,
                    'original_deadline_utc':'2026-10-28T07:36:08Z','github_publication_authorized':False}}
            execution={'schema_version':'f37-teacher-legality-execution-v1',
                'teacher_policy':{'policy_id':checker.TEACHER_POLICY,'native_search_budget_unchanged':True,'teacher_history_acceptance_unchanged':True},
                **{key:{'path':p,'sha256':'a'*64} for key,p in checker.BINDINGS.items()}}
            manifest={'plan':{'deadline_utc':checker.APPROVED_DEADLINE,'start_utc':'2026-10-27T23:36:08Z',
                'checkpoint_utc':'2026-10-28T03:36:08Z','execution':execution},'source_files':sources}
            for key,p in checker.BINDINGS.items():write(root/p,records[key])
            def hashes(p):return 'b'*64 if str(p).endswith('review.md') else 'a'*64
            with mock.patch.object(checker,'ROOT',root),mock.patch.object(checker,'verify_plan_binding'),mock.patch.object(checker,'verify_previous_qualification'),mock.patch.object(checker,'sha',side_effect=hashes):
                checker.verify_execution(manifest,archives=False)
                self.assertEqual(len(checker.NEW_SOURCES), 16)
                for required in sorted(checker.NEW_SOURCES):
                    changed = copy.deepcopy(manifest); changed['source_files'].pop(required)
                    with self.subTest(source=required), self.assertRaisesRegex(ValueError, 'inventory is incomplete'):
                        checker.verify_execution(changed, archives=False)
                    altered = copy.deepcopy(records['legality_qualification']); altered['source_files'].pop(required)
                    write(root/checker.BINDINGS['legality_qualification'], altered)
                    with self.subTest(qualification=required), self.assertRaisesRegex(ValueError, 'required code/tests'):
                        checker.verify_execution(manifest, archives=False)
                    write(root/checker.BINDINGS['legality_qualification'], records['legality_qualification'])
                records['authorization']['scientific_launch_authorized_after_qualification']=False
                write(root/checker.BINDINGS['authorization'],records['authorization'])
                with self.assertRaisesRegex(ValueError,'authorization'):checker.verify_execution(manifest,archives=False)
                records['authorization']['scientific_launch_authorized_after_qualification']=True
                write(root/checker.BINDINGS['authorization'],records['authorization'])
                records['legality_qualification']['source_files'][checker.PLAN_RELATIVE]='a'*64
                write(root/checker.BINDINGS['legality_qualification'],records['legality_qualification'])
                with self.assertRaisesRegex(ValueError,'Circular'):checker.verify_execution(manifest,archives=False)

    def test_predecessor_archive_and_complete_source_inventory_required(self):
        record={'schema_version':'f36-failure-receipt-v1','status':'failed','completed_comparison':False,
            'archive_receipt':'archive/receipt.json','source_archive':'archive/source','evidence_archive':'archive/evidence',
            'source_files':{},'evidence_files':{}}
        record['content_hash']=checker.json_hash(record)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);write(root/record['archive_receipt'],record);write(root/checker.BINDINGS['failed_predecessor'],record)
            with mock.patch.object(checker,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'complete source inventory'):checker.verify_predecessor(record)
                (root/record['archive_receipt']).write_text('changed')
                with self.assertRaisesRegex(ValueError,'archived receipt'):checker.verify_predecessor(record)

    def test_new_paths_and_frozen_scientific_completion_are_reused(self):
        self.assertIs(checker.validate_completion,checker._frozen.validate_completion)
        for command in checker.expected_commands().values():
            self.assertIn('pokemon_teacher_runtime.lifecycle',command)
            self.assertIn('pokemon_teacher_legality.stage',command)
            self.assertTrue(any(x.startswith('pokemon_teacher_assisted.') for x in command))
        self.assertTrue(all('f37-' in p for p in checker.OUTPUT_PATHS.values()))
        self.assertEqual(checker.BINDINGS['runtime_qualification'],'artifacts/reviews/f36-teacher-runtime/runtime-qualification-v1.json')

    def test_admission_does_not_run_without_explicit_freeze(self):
        with self.assertRaisesRegex(ValueError,'--freeze'):admitter.main()


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch('pokemon_teacher_legality.deadline.install_deadline'):
            cls.driver = load_script('f37_driver_tests', 'run-teacher-legality.py')

    def test_initial_capture_failure_still_records_cleanup_and_original_failure(self):
        driver=self.driver
        with tempfile.TemporaryDirectory() as temporary:
            review=Path(temporary);child=mock.Mock(pid=424242,returncode=None);child.poll.side_effect=lambda:child.returncode
            def terminate(*args,**kwargs):
                child.returncode=143
                return {'status':'passed','root_pid':child.pid,'returncode':143,'live_descendants_at_cleanup':[],'remaining_pids':[],'errors':[]}
            with mock.patch.object(driver,'REVIEW',review),mock.patch.object(driver,'before_deadline'), \
                    mock.patch.object(driver.subprocess,'Popen',return_value=child), \
                    mock.patch.object(driver,'capture_tree',side_effect=RuntimeError('synthetic capture error')), \
                    mock.patch.object(driver,'terminate_tree',side_effect=terminate):
                with self.assertRaises(RuntimeError):driver.stage('synthetic',['-m','pokemon_teacher_assisted.schedules'],review/'report.json')
            result=json.loads((review/'synthetic-exit.json').read_text())
            self.assertEqual(result['exit_code'],143);self.assertIn('synthetic capture error',result['wait_error'])
            self.assertIn('pokemon_teacher_legality.stage',result['command'])

    def test_journal_gate_blocks_training_after_acquisition(self):
        driver=self.driver
        with tempfile.TemporaryDirectory() as temporary:
            review=Path(temporary);paths={key:review/key for key in ['SCHEDULES','DEMOS','TRAIN','EXTERNAL','VERIFY','RESULT','RECEIPT']}
            calls=[]
            def stage(name,*args):calls.append(name);return {'stage':name}
            with mock.patch.object(driver,'REVIEW',review),mock.patch.multiple(driver,**paths), \
                    mock.patch.object(driver,'before_deadline'),mock.patch.object(driver,'digest',return_value='a'*64), \
                    mock.patch.object(driver,'verify_inputs',return_value={}),mock.patch.object(driver,'verify_execution',return_value={}), \
                    mock.patch.object(driver,'stage',side_effect=stage), \
                    mock.patch.object(driver.CHECKER,'validate_teacher_journals',side_effect=ValueError('missing raw reply')):
                with self.assertRaisesRegex(ValueError,'missing raw reply'):driver.main()
            self.assertEqual(calls,['fixed-schedules','teacher-acquisition'])
            self.assertEqual(json.loads((review/'workflow-exit.json').read_text())['exit_code'],1)


if __name__=='__main__':unittest.main()
