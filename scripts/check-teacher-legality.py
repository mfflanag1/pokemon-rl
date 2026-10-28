#!/usr/bin/env python3
"""F37 byte/path/runtime verification with frozen F35 scientific completion checks."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('f35_frozen_completion_checker', ROOT / 'scripts/check-teacher-assisted.py')
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)
validate_completion, STAGES, sha = _frozen.validate_completion, _frozen.STAGES, _frozen.sha
REVIEW = 'artifacts/reviews/f37-teacher-legality'
PLAN_RELATIVE = 'contracts/teacher-legality-v1/plan.json'
OUTPUT_PATHS = {
    'schedules': REVIEW + '/fixed-schedules-v1',
    'demonstrations': 'artifacts/runs/f37-teacher-data-v1',
    'training': 'artifacts/runs/f37-teacher-legality-v1',
    'external_evaluation': 'artifacts/runs/f37-teacher-evaluation-v1',
    'verification': REVIEW + '/sampled-verification-v1',
    'results': REVIEW + '/measured-results-v1',
}
TEACHER_POLICY = 'f37-authoritative-root-legality-v1'
APPROVED_DEADLINE = '2026-10-28T18:00:00Z'
ORIGINAL_DEADLINE = '2026-10-28T07:36:08Z'
NEW_SOURCES = {
    'python/pokemon_teacher_legality/' + name + '.py'
    for name in ('__init__', 'stage', 'external', 'teacher', 'selection', 'qualification', 'deadline')
} | {'scripts/' + name + '-teacher-legality.py' for name in ('run', 'launch', 'check', 'admit')} | {
    'tests/test_f37_' + name + '.py' for name in ('selection', 'teacher', 'qualification', 'workflow', 'deadline')}
BINDINGS = {
    'failed_predecessor': 'artifacts/reports/f36-teacher-runtime-failure.json',
    'runtime_qualification': 'artifacts/reviews/f36-teacher-runtime/runtime-qualification-v1.json',
    'legality_qualification': REVIEW + '/legality-qualification-v2.json',
    'deadline_authorization': REVIEW + '/deadline-authorization-v1.json',
    'authorization': REVIEW + '/authorization.json',
}



def bound_path(relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid receipt path')
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Receipt path outside project')
    return path


def expected_commands():
    review = ROOT / REVIEW
    admission = review / 'admission'
    paths = {k: ROOT / v for k, v in OUTPUT_PATHS.items()}
    tails = {
        'fixed-schedules': ['schedules', '--admission', admission, '--output', paths['schedules']],
        'teacher-acquisition': ['demonstrations', '--admission', admission, '--plan',
            ROOT / PLAN_RELATIVE, '--output', paths['demonstrations']],
        'training': ['campaign', '--admission', admission, '--demonstrations', paths['demonstrations'],
            '--run', paths['training']],
        'external-evaluation': ['external_evaluation', '--admission', admission, '--demonstrations',
            paths['demonstrations'], '--training-run', paths['training'], '--output', paths['external_evaluation']],
        'sampled-verification': ['verification', '--training-run', paths['training'], '--external-run',
            paths['external_evaluation'], '--demonstrations', paths['demonstrations'], '--output', paths['verification']],
        'measured-results': ['reporting', '--training-run', paths['training'], '--external-run',
            paths['external_evaluation'], '--demonstrations', paths['demonstrations'],
            '--verification', paths['verification'], '--output', paths['results']],
    }
    prefix = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle',
              '--', 'pokemon_teacher_legality.stage']
    return {k: prefix + ['pokemon_teacher_assisted.' + v[0], *map(str, v[1:])]
            for k, v in tails.items()}


def validate_runtime(receipt, manifest, completion, workflow_exit, driver_exit):
    expected = [(receipt, 'f37-teacher-legality-receipt-v1'),
        (completion, 'f37-completion-v1'), (workflow_exit, 'f37-workflow-exit-v1'),
        (driver_exit, 'f37-driver-exit-v1')]
    for record, schema in expected:
        if record.get('schema_version') != schema:
            raise ValueError('F37 record schema: ' + schema)
    if type(driver_exit.get('exit_code')) is not int or driver_exit['exit_code'] != 0:
        raise ValueError('F37 outer driver did not exit cleanly')
    if (receipt['output_paths'] != OUTPUT_PATHS or receipt['summary'] != OUTPUT_PATHS['results'] + '/summary.json'
            or completion['receipt'] != 'artifacts/reports/f37-teacher-legality.json'
            or workflow_exit['receipt'] != completion['receipt']):
        raise ValueError('F37 output path inventory')
    execution = manifest['plan']['execution']
    if execution.get('schema_version') != 'f37-teacher-legality-execution-v1' or receipt['execution'] != execution:
        raise ValueError('F37 execution metadata linkage')
    for name, relative in BINDINGS.items():
        entry = execution[name]
        if entry['path'] != relative or receipt['evidence_files'].get(relative) != entry['sha256']:
            raise ValueError('F37 runtime/predecessor evidence linkage: ' + name)
    commands = expected_commands()
    for entry in receipt['stages']:
        if entry.get('schema_version') != 'f37-stage-v1':
            raise ValueError('F37 stage record schema')
        if entry.get('stage') not in commands or entry.get('command') != commands[entry['stage']]:
            raise ValueError('F37 stage wrapper command/paths')


def validate_runtime_cleanup(cleanup):
    for key in ('unresolved_unknown_pids', 'reused_pids', 'handle_signals'):
        if cleanup.get(key) != []:
            raise ValueError('F37 cleanup unresolved or signalled: ' + key)
    if type(cleanup.get('unknown_overflow')) is not int or cleanup['unknown_overflow'] != 0:
        raise ValueError('F37 cleanup unknown identity overflow')
    if 'table_error' not in cleanup or cleanup['table_error'] is not None:
        raise ValueError('F37 cleanup process-table state is unresolved')


def validate_deadline_authorization(record, plan):
    expected = {'schema_version': 'f37-deadline-authorization-v1', 'new_attempt': 'F37',
        'approved': True, 'approved_deadline_utc': APPROVED_DEADLINE,
        'original_deadline_utc': ORIGINAL_DEADLINE, 'original_start_utc': '2026-10-27T23:36:08Z',
        'original_checkpoint_utc': '2026-10-28T03:36:08Z', 'clock_reset': False,
        'scientific_seeds_or_budgets_changed': False, 'single_fresh_comparison_only': True,
        'github_publication_authorized': False}
    if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in expected.items()):
        raise ValueError('F37 approved deadline authorization mismatch')
    if (plan.get('deadline_utc') != APPROVED_DEADLINE
            or plan.get('start_utc') != record['original_start_utc']
            or plan.get('checkpoint_utc') != record['original_checkpoint_utc']):
        raise ValueError('F37 approved deadline/start/checkpoint mismatch')


def verify_plan_binding(manifest):
    plan_path = bound_path(PLAN_RELATIVE)
    if manifest['source_files'].get(PLAN_RELATIVE) != sha(plan_path):
        raise ValueError('F37 CLI plan is not bound by admission')
    if json.loads(plan_path.read_text()) != manifest['plan']:
        raise ValueError('F37 CLI plan differs from admission plan')
    original_plan = json.loads((ROOT / 'contracts/teacher-assisted-v1/plan.json').read_text())
    normalized = {k: v for k, v in manifest['plan'].items() if k != 'execution'}
    normalized['deadline_utc'] = ORIGINAL_DEADLINE
    if normalized != original_plan:
        raise ValueError('F37 changed scientific plan beyond approved deadline')
    relative = BINDINGS['deadline_authorization']
    entry = manifest['plan'].get('execution', {}).get('deadline_authorization', {})
    expected = sha(bound_path(relative))
    if (entry != {'path': relative, 'sha256': expected}
            or manifest['source_files'].get(relative) != expected):
        raise ValueError('F37 deadline authorization is not bound by admission')
    validate_deadline_authorization(json.loads(bound_path(relative).read_text()), manifest['plan'])


def verify_predecessor(record):
    if (record.get('schema_version') != 'f36-failure-receipt-v1'
            or record.get('status') != 'failed' or record.get('completed_comparison') is not False):
        raise ValueError('F36 failed attempt was not preserved')
    content = record.get('content_hash')
    encoded = json.dumps({k: v for k, v in record.items() if k != 'content_hash'},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != content:
        raise ValueError('F36 failure content hash')
    if bound_path(record['archive_receipt']).read_bytes() != bound_path(BINDINGS['failed_predecessor']).read_bytes():
        raise ValueError('F36 archived receipt bytes')
    if len(record['source_files']) != 2182:
        raise ValueError('F36 complete source inventory')
    for kind, archive_key in [('source_files', 'source_archive'), ('evidence_files', 'evidence_archive')]:
        archive = bound_path(record[archive_key])
        for relative, expected in record[kind].items():
            if sha(bound_path(relative)) != expected or sha(archive / relative) != expected:
                raise ValueError('Preserved F36 live/archive mismatch: ' + relative)
    return {'source_files': len(record['source_files']), 'evidence_files': len(record['evidence_files'])}


def verify_previous_qualification(record):
    previous = record.get('previous_qualification', {})
    relative = REVIEW + '/legality-qualification-v1.json'
    archive_relative = REVIEW + '/qualification-source-v1'
    if previous.get('path') != relative or previous.get('source_archive') != archive_relative:
        raise ValueError('F37 original qualification archive identity')
    old_path, archive = bound_path(relative), bound_path(archive_relative)
    old_hash = sha(old_path)
    if previous.get('sha256') != old_hash:
        raise ValueError('F37 original qualification hash')
    old = json.loads(old_path.read_text())
    frozen = json.loads((archive / 'manifest.json').read_text())
    required = NEW_SOURCES - {'python/pokemon_teacher_legality/deadline.py', 'tests/test_f37_deadline.py'}
    if (old.get('status') != 'passed' or set(old.get('source_files', {})) != required
            or frozen.get('schema_version') != 'f37-qualified-source-archive-v1'
            or frozen.get('qualification_sha256') != old_hash or frozen.get('source_files') != old['source_files']):
        raise ValueError('F37 original qualification/source archive linkage')
    evidence = record.get('evidence_files', {})
    required_evidence = {relative: old_hash, archive_relative + '/manifest.json': sha(archive / 'manifest.json')}
    for source, expected in old['source_files'].items():
        archived = archive_relative + '/' + source
        if sha(bound_path(archived)) != expected:
            raise ValueError('F37 archived qualified source mismatch: ' + source)
        required_evidence[archived] = expected
    if any(evidence.get(path) != expected for path, expected in required_evidence.items()):
        raise ValueError('F37 original qualification/archive evidence not bound')
    for path, expected in old.get('evidence_files', {}).items():
        if sha(bound_path(path)) != expected:
            raise ValueError('F37 original qualification evidence drift: ' + path)


def verify_execution(manifest, *, archives=True):
    verify_plan_binding(manifest)
    execution = manifest['plan']['execution']
    if execution.get('schema_version') != 'f37-teacher-legality-execution-v1':
        raise ValueError('F37 execution schema')
    policy = execution.get('teacher_policy', {})
    if (policy.get('policy_id') != TEACHER_POLICY or policy.get('native_search_budget_unchanged') is not True
            or policy.get('teacher_history_acceptance_unchanged') is not True):
        raise ValueError('Explicit adapted-teacher policy binding')
    if not NEW_SOURCES.issubset(manifest['source_files']):
        raise ValueError('F37 source inventory is incomplete')
    records = {}
    for name, relative in BINDINGS.items():
        entry = execution[name]
        if entry.get('path') != relative or entry.get('sha256') != sha(bound_path(relative)):
            raise ValueError('F37 evidence binding: ' + name)
        record = json.loads(bound_path(relative).read_text())
        records[name] = record
        if name == 'authorization':
            if (record.get('schema_version') != 'f37-authorization-v1' or record.get('new_attempt') != 'F37'
                    or record.get('scientific_launch_authorized_after_qualification') is not True
                    or record.get('original_deadline_utc') != ORIGINAL_DEADLINE
                    or record.get('github_publication_authorized') is not False):
                raise ValueError('F37 explicit authorization/deadline')
        elif name == 'deadline_authorization':
            validate_deadline_authorization(record, manifest['plan'])
        elif name.endswith('qualification'):
            qualified = record.get('source_files')
            if record.get('status') != 'passed' or not isinstance(qualified, dict) or not qualified:
                raise ValueError('Qualification status/source inventory: ' + name)
            if PLAN_RELATIVE in qualified or relative in qualified:
                raise ValueError('Circular qualification/plan source binding')
            for source, expected in qualified.items():
                if manifest['source_files'].get(source) != expected or sha(bound_path(source)) != expected:
                    raise ValueError('Qualification source drift: ' + source)
            for evidence, expected in record.get('evidence_files', {}).items():
                if sha(bound_path(evidence)) != expected:
                    raise ValueError('Qualification evidence drift: ' + evidence)
            if name == 'legality_qualification':
                verify_previous_qualification(record)
                if not NEW_SOURCES.issubset(qualified):
                    raise ValueError('F37 qualification omits required code/tests')
                reviews = record.get('independent_reviews', {})
                if not isinstance(reviews, dict) or not reviews:
                    raise ValueError('Missing independent legality reviews')
                if any(path not in record.get('evidence_files', {}) for path in reviews.values()):
                    raise ValueError('Independent review bytes are not bound')
    if archives:
        verify_predecessor(records['failed_predecessor'])
    return execution


def verify_runtime_evidence(receipt, manifest):
    execution = verify_execution(manifest)
    for name, relative in BINDINGS.items():
        if receipt['evidence_files'].get(relative) != execution[name]['sha256']:
            raise ValueError('F37 receipt evidence linkage: ' + name)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_selection_audit(response, legal):
    selection = response['selection']
    audit = selection.get('audit')
    action = response['action']
    if len(legal) == 1:
        expected = {'schema_version': 'f37-forced-request-v1', 'action': action,
            'legal_action_count': 1, 'native_search_invoked': False,
            'selection_rng_draws': 0, 'reason': 'Exactly one authoritative legal action.'}
        if audit != expected:
            raise ValueError('Forced-request selection audit mismatch')
        return
    if (not isinstance(audit, dict) or audit.get('schema') != 'f37-legal-selection-v1'
            or audit.get('outcome') != 'selected' or audit.get('decision') != response.get('decision')
            or audit.get('decision_action') != action or audit.get('mask_true_count') != len(legal)
            or audit.get('threshold_fraction') != .75 or 'failure' in audit):
        raise ValueError('Native legal-root selection audit identity')
    worlds, candidates = audit.get('worlds'), audit.get('candidates')
    if (not isinstance(worlds, list) or len(worlds) != response['sampled_worlds']
            or not isinstance(candidates, list) or not candidates):
        raise ValueError('Native legal-root world/candidate inventory')
    for position, world in enumerate(worlds):
        if (world.get('position') != position or type(world.get('index')) is not int
                or type(world.get('option_count')) is not int or world['option_count'] < 1):
            raise ValueError('Native legal-root world identity')
        for field in ('chance', 'total_visits'):
            value = world.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (field == 'total_visits' and value == 0):
                raise ValueError('Native legal-root world mass/visits')
    choices, indices, legal_candidates = set(), set(), []
    for candidate in candidates:
        choice, mass, index = candidate.get('choice'), candidate.get('mass'), candidate.get('insertion_index')
        if (not isinstance(choice, str) or not choice or choice in choices or type(index) is not int
                or index in indices or type(mass) not in (int, float) or not math.isfinite(mass) or mass < 0
                or type(candidate.get('allowed')) is not bool):
            raise ValueError('Native legal-root candidate mass/identity')
        choices.add(choice); indices.add(index)
        mapped, reason = candidate.get('action'), candidate.get('reason')
        if type(mapped) is int and mapped in legal:
            legal_candidates.append(candidate)
        elif type(mapped) is int and 0 <= mapped < 14:
            if reason != 'illegal_action' or candidate['allowed']:
                raise ValueError('Illegal native candidate retained')
        elif type(mapped) is int:
            if reason != 'action_out_of_range' or candidate['allowed']:
                raise ValueError('Out-of-range native candidate retained')
        elif mapped is not None or reason not in ('unmappable_choice', 'boolean_action', 'non_integer_action') or candidate['allowed']:
            raise ValueError('Unmappable native candidate retained')
    if (indices != set(range(len(candidates))) or not legal_candidates
            or candidates != sorted(candidates, key=lambda c: (-c['mass'], c['insertion_index']))):
        raise ValueError('Native candidate ordering/legal inventory')
    highest = legal_candidates[0]['mass']; threshold = highest * .75
    retained = []
    for candidate in legal_candidates:
        mass = candidate['mass']
        reason = 'non_positive_mass' if mass <= 0 else 'below_threshold' if mass < threshold else None
        if candidate['reason'] != reason or candidate['allowed'] is not (reason is None):
            raise ValueError('Native threshold must follow authoritative legality')
        if reason is None:
            retained.append({key: candidate[key] for key in ('choice', 'action', 'mass')})
    if not retained or audit.get('retained') != retained:
        raise ValueError('Native legal-root retained population')
    selected = [c for c in retained if c['choice'] == response['decision'] and c['action'] == action]
    if len(selected) != 1:
        raise ValueError('Native decision is outside retained legal population')
    expected = {'aggregated_count': len(candidates), 'aggregated_mass': sum(c['mass'] for c in candidates),
        'legal_count': len(legal_candidates), 'highest_legal_mass': highest, 'threshold': threshold,
        'retained_count': len(retained), 'retained_mass': sum(c['mass'] for c in retained),
        'excluded_count': len(candidates) - len(retained),
        'excluded_mass': sum(c['mass'] for c in candidates if not c['allowed'])}
    for field, value in expected.items():
        if type(audit.get(field)) not in (int, float) or audit[field] != value:
            raise ValueError('Native legal-root aggregate linkage: ' + field)


def validate_submissions(frame, step):
    actions, request_ids, submissions = step.get('actions'), step.get('request_ids'), step.get('submitted')
    if (not isinstance(actions, dict) or not set(actions).issubset({'p1', 'p2'})
            or not isinstance(request_ids, dict) or set(request_ids) != set(actions)
            or not isinstance(submissions, list) or len(submissions) != len(actions)):
        raise ValueError('Action/submission inventory mismatch')
    seen = set()
    for submission in submissions:
        if not isinstance(submission, dict):
            raise ValueError('Invalid submission record')
        side = submission.get('side')
        if side not in actions or side in seen:
            raise ValueError('Duplicate or extraneous submission side')
        seen.add(side)
        player = frame['players'].get(side)
        if not isinstance(player, dict):
            raise ValueError('Submitted side missing player projection')
        mask, action = player.get('mask'), actions[side]
        if (not isinstance(mask, list) or len(mask) != 14 or any(type(bit) is not bool for bit in mask)
                or type(action) is not int or not 0 <= action < 14 or not mask[action]
                or type(request_ids[side]) is not int or request_ids[side] != player.get('request_id')
                or type(submission.get('action')) is not int or submission['action'] != action
                or type(submission.get('request_id')) is not int or submission['request_id'] != request_ids[side]
                or type(submission.get('accepted')) is not bool):
            raise ValueError('Submitted action/request does not match authoritative projection')


def validate_teacher_game(folder, *, capture_demonstrations):
    record = json.loads((folder / 'result.json').read_text())
    if (record.get('status') != 'completed' or record.get('teacher_policy') != TEACHER_POLICY
            or record.get('teacher_response_journal') != 'teacher-responses.jsonl'):
        raise ValueError('Missing adapted-teacher record/journal identity')
    journal = [json.loads(row) for row in (folder / 'teacher-responses.jsonl').read_text().splitlines()]
    transcript = [json.loads(row) for row in (folder / 'transcript.jsonl').read_text().splitlines()]
    steps = record['steps']
    if len(journal) != len(steps) or len(transcript) < len(steps) + 1:
        raise ValueError('Incomplete raw teacher response journal')
    counts = dict(forced_request=0, native_legal_root=0, waiting=0, errors=0)
    accepted_teacher = []
    teacher = record['external_side']
    for index, (row, step) in enumerate(zip(journal, steps)):
        frame = transcript[index]['frame']; player = frame['players'][teacher]
        if len(player['mask']) != 14 or any(type(bit) is not bool for bit in player['mask']):
            raise ValueError('Authoritative legal mask must contain 14 booleans')
        if (row.get('schema_version') != 'f37-teacher-response-v1' or row.get('status') != 'response'
                or row.get('step_index') != index or step['index'] != index
                or row.get('request_id') != player['request_id'] or row.get('legal_mask') != player['mask']
                or row.get('external_input_sha256') != json_hash(transcript[index]['external'])
                or row.get('raw_response') != step['external']):
            raise ValueError('Teacher journal/frame/step identity mismatch')
        if step.get('submitted') != transcript[index + 1]['frame'].get('submitted'):
            raise ValueError('Teacher submission/transcript mismatch')
        response = row['raw_response']; legal = [i for i, allowed in enumerate(player['mask']) if allowed]
        if not legal:
            if response != {'waiting': True} or teacher in step['actions']:
                raise ValueError('Waiting teacher response mismatch')
            if ('teacher_feedback' in step or any(isinstance(sub, dict) and sub.get('side') == teacher
                    for sub in step['submitted'])):
                raise ValueError('Waiting teacher submission/feedback is forbidden')
        validate_submissions(frame, step)
        if not legal:
            counts['waiting'] += 1
            continue
        action = response.get('action'); selection = response.get('selection', {})
        if (type(action) is not int or action not in legal or response.get('request_id') != player['request_id']
                or response.get('teacher_policy') != TEACHER_POLICY or selection.get('legal_mask') != player['mask']
                or selection.get('request_id') != player['request_id'] or step['actions'].get(teacher) != action):
            raise ValueError('Teacher response authoritative legality mismatch')
        forced = len(legal) == 1
        expected_kind = 'forced-request' if forced else 'native-legal-root'
        if (selection.get('kind') != expected_kind or type(selection.get('search_calls')) is not int
                or selection['search_calls'] != (0 if forced else 1)):
            raise ValueError('Teacher forced/native selection identity')
        if response.get('search_time_ms') != 100:
            raise ValueError('Teacher configured search budget drift')
        if forced and any(response.get(field) != 0 for field in ['sampled_worlds', 'per_world_ms', 'configured_search_total_ms']):
            raise ValueError('Forced teacher request performed native search')
        if not forced:
            if (any(type(response.get(field)) is not int or response[field] <= 0 for field in
                    ('sampled_worlds', 'per_world_ms', 'configured_search_total_ms'))
                    or response['configured_search_total_ms'] != response['sampled_worlds'] * response['per_world_ms']):
                raise ValueError('Native teacher configured search arithmetic')
        validate_selection_audit(response, legal)
        counts['forced_request' if forced else 'native_legal_root'] += 1
        submissions = [(i, s) for i, s in enumerate(step['submitted']) if s['side'] == teacher]
        if len(submissions) != 1:
            raise ValueError('Teacher submission inventory')
        sub_index, submission = submissions[0]
        if (submission['action'] != action or submission['request_id'] != player['request_id']
                or type(submission['accepted']) is not bool):
            raise ValueError('Teacher submission decision mismatch')
        if step.get('teacher_feedback') != {'acknowledged': True, 'request_id': player['request_id'],
                'accepted': submission['accepted']}:
            raise ValueError('Teacher feedback/submission acknowledgement mismatch')
        if submission['accepted']:
            accepted_teacher.append((index, sub_index, action, player['request_id']))
    if (record.get('teacher_policy_counts') != counts
            or any(type(value) is not int for value in record['teacher_policy_counts'].values())):
        raise ValueError('Teacher per-game policy counters')
    if record.get('capture_demonstrations') is not capture_demonstrations:
        raise ValueError('Teacher corpus/evaluation role mismatch')
    if capture_demonstrations:
        labels = json.loads((folder / 'labels.json').read_text())
        actual = [(x['step_index'], x['submission_index'], x['action'], x['request_id']) for x in labels['labels']]
        if actual != accepted_teacher or record['accepted_labels'] != len(actual):
            raise ValueError('Raw teacher reply/accepted-label linkage')
    return {'games': 1, 'responses': len(journal), **counts}


def validate_teacher_journals(demonstrations, external=None):
    demonstrations = Path(demonstrations)
    data = json.loads((demonstrations / 'report.json').read_text())
    if data.get('status') != 'passed' or len(data.get('games', [])) != 480:
        raise ValueError('Incomplete teacher acquisition inventory')
    groups = {'acquisition': [(demonstrations / game['directory'], True) for game in data['games']]}
    if external is not None:
        panel = json.loads((Path(external) / 'report.json').read_text())
        if panel.get('status') != 'passed':
            raise ValueError('Incomplete external evaluation')
        groups['evaluation'] = [(bound_path(game['game_record']).parent, False)
            for result in panel['seeds'].values() for endpoint in result.values() for game in endpoint['games']]
        if len(groups['evaluation']) != 1152:
            raise ValueError('External evaluation journal inventory')
    result = {'teacher_policy': TEACHER_POLICY, 'groups': {}}
    seen = set()
    for name, games in groups.items():
        counts = dict(games=0, responses=0, forced_request=0, native_legal_root=0, waiting=0, errors=0)
        for folder, capture in games:
            key = str(folder.resolve())
            expected_root = demonstrations if capture else Path(external)
            if not folder.resolve().is_relative_to(expected_root.resolve()) or key in seen:
                raise ValueError('Teacher journal path reuse/escape')
            seen.add(key)
            for field, value in validate_teacher_game(folder, capture_demonstrations=capture).items():
                counts[field] += value
        result['groups'][name] = counts
    return result


def main(output):
    receipt_path = ROOT / 'artifacts/reports/f37-teacher-legality.json'
    review = ROOT / 'artifacts/reviews/f37-teacher-legality'
    body = json.loads(receipt_path.read_text())
    if body.get('schema_version') != 'f37-teacher-legality-receipt-v1':
        raise ValueError('F37 receipt schema')
    recorded = body.pop('content_hash')
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != recorded or body['status'] != 'completed':
        raise ValueError('Receipt content identity/status')
    if receipt_path.read_bytes() != (review / 'f37-teacher-legality-receipt.json').read_bytes():
        raise ValueError('Archived receipt bytes differ')
    if body['input_manifest'] != 'artifacts/reviews/f37-teacher-legality/admission/manifest.json':
        raise ValueError('F37 admission path')
    manifest_path = ROOT / body['input_manifest']
    if sha(manifest_path) != body['input_hash']:
        raise ValueError('Admission manifest identity')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'f35-inputs-v1':
        raise ValueError('Preserved admission schema')
    verify_plan_binding(manifest)
    if manifest['source_files'] != body['source_files']:
        raise ValueError('Source inventory differs from admitted manifest')
    for rel, expected in body['source_files'].items():
        if sha(bound_path(rel)) != expected or sha(manifest_path.parent / 'source' / rel) != expected:
            raise ValueError('Live/admitted source mismatch: ' + rel)
    for rel, expected in body['evidence_files'].items():
        if sha(bound_path(rel)) != expected:
            raise ValueError('Evidence mismatch: ' + rel)
    completion_records = {}
    for name in ('completion.json', 'workflow-exit.json'):
        record = json.loads((review / name).read_text())
        completion_records[name] = record
        if record['status'] != 'passed' or record['receipt_sha256'] != sha(receipt_path) or record['input_hash'] != body['input_hash']:
            raise ValueError('Completion linkage')
        if name == 'workflow-exit.json' and record['exit_code'] != 0:
            raise ValueError('Workflow exit failure')
    if sha(bound_path(body['summary'])) != body['summary_sha256']:
        raise ValueError('Summary identity')
    summary = json.loads(bound_path(body['summary']).read_text())
    if summary['status'] != 'passed' or summary['input_hash'] != body['input_hash'] or set(summary['seeds']) != {'7103', '8209', '9311'}:
        raise ValueError('Summary completion/seed inventory')
    stage_exits, stage_cleanups = {}, {}
    for name in STAGES:
        for suffix in ('.log', '-launch.json', '-exit.json', '-cleanup.json'):
            path = review / (name + suffix)
            if body['evidence_files'].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError('Required stage evidence missing: ' + name + suffix)
        stage_exits[name] = json.loads((review / (name + '-exit.json')).read_text())
        stage_cleanups[name] = json.loads((review / (name + '-cleanup.json')).read_text())
        validate_runtime_cleanup(stage_cleanups[name])
        launch = json.loads((review / (name + '-launch.json')).read_text())
        if any(stage_exits[name].get(key) != value for key, value in launch.items()):
            raise ValueError('Stage launch/exit mismatch: ' + name)
    driver_exit = json.loads((review / 'driver-exit.json').read_text())
    validate_runtime(body, manifest, completion_records['completion.json'],
                     completion_records['workflow-exit.json'], driver_exit)
    verify_runtime_evidence(body, manifest)
    audit = validate_teacher_journals(bound_path(OUTPUT_PATHS['demonstrations']), bound_path(OUTPUT_PATHS['external_evaluation']))
    if body.get('teacher_policy_audit') != audit:
        raise ValueError('Teacher journal/policy count receipt linkage')
    for key, relative in OUTPUT_PATHS.items():
        folder = bound_path(relative)
        required_report = folder / ('summary.json' if key == 'results' else 'report.json')
        if not required_report.is_file():
            raise ValueError('Missing F37 stage report: ' + key)
        report = json.loads(required_report.read_text())
        if report.get('status') != 'passed' or report.get('input_hash') != body['input_hash']:
            raise ValueError('F37 stage report status/input: ' + key)
        for path in folder.rglob('*'):
            if path.is_file() and body['evidence_files'].get(str(path.relative_to(ROOT))) != sha(path):
                raise ValueError('Unbound F37 output: ' + str(path))
    validate_completion(body, manifest, summary, completion_records['completion.json'],
                        completion_records['workflow-exit.json'], stage_exits, stage_cleanups)
    result = {'schema_version': 'f37-postcheck-v1', 'status': 'passed', 'receipt_sha256': sha(receipt_path), 'content_hash': recorded,
        'live_and_archived_sources': len(body['source_files']), 'evidence_files': len(body['evidence_files']),
        'fixed_counts_cells_stages_verified': True,
        'scope': 'All receipt-listed source/evidence bytes, archived receipt equality, successful stages, fixed counts, complete cells/contrasts and completion/exit linkage.'}
    output = Path(output).resolve()
    if output.is_relative_to(review.resolve()):
        raise ValueError('Postcheck must remain outside original review directory')
    with output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
