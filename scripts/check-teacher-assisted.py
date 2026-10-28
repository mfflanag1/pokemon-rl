#!/usr/bin/env python3
"""Independent read-only F35 receipt/source/evidence verification after driver exit."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {'7103', '8209', '9311'}
ARMS = {'control', 'assisted'}
ENDPOINTS = {'initial', *ARMS}
GAMES = {'baseline-v1': 512, 'tactical': 128, 'base_power': 64, 'random': 64, 'foul_play': 128}
STAGES = ('fixed-schedules', 'teacher-acquisition', 'training', 'external-evaluation',
          'sampled-verification', 'measured-results')
COUNTS = {'cohorts': 1536, 'optimizer_steps': 12288, 'selected_rows': 1572864,
    'row_presentations': 3145728, 'demonstration_presentations': 1572864,
    'evaluation_games': 6912, 'external_evaluation_games': 1152, 'all_evaluation_games': 8064,
    'checkpoint_comparisons': 18, 'original_next_update_recoveries': 12,
    'teacher_train_games': 384, 'teacher_probe_games': 96}
VERIFICATION = {'training_games': 36, 'evaluation_games': 48,
    'external_action_replay_games': 18, 'teacher_action_replay_games': 12,
    'game_executions': 114, 'sampled_updates': 12, 'optimizer_steps_recomputed': 96,
    'checkpoint_comparisons': 18}
ARM_COUNTS = {'cohorts': 256, 'optimizer_steps': 2048, 'selected_rows': 262144,
              'row_presentations': 524288}
STREAM_ROWS = {'self_play': 65536, 'frozen_f32': 65536, 'tactical': 131072}
CONTRASTS = {'control_vs_initial': ('initial', 'control'),
    'assisted_vs_initial': ('initial', 'assisted'), 'assisted_vs_control': ('control', 'assisted')}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def counts_match(actual, expected, label, *, exact=False):
    if exact:
        require(set(actual) == set(expected), label + ' inventory')
    for key, value in expected.items():
        require(integer(actual[key], label + '/' + key) == value, label + '/' + key)


def value_cell(cell, games, label):
    require(integer(cell['games'], label + '/games', 1) == games, label + '/games')
    for key in ('terminals', 'cutoffs', 'decisions', 'learner_wins', 'learner_losses', 'true_ties'):
        integer(cell[key], label + '/' + key)
    require(cell['terminals'] + cell['cutoffs'] == games, label + '/outcomes')
    require(sum(cell[k] for k in ('learner_wins', 'learner_losses', 'true_ties')) == cell['terminals'],
            label + '/terminal rewards')
    mse = cell['terminal_game_mean_value_mse']
    require((mse is None) == (cell['terminals'] == 0), label + '/terminal-only value')
    for name, value in [('mean value', cell['game_mean_predicted_value']), ('terminal MSE', mse)]:
        if value is not None:
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                    label + '/' + name)
    require(cell['game_mean_predicted_value'] is not None, label + '/missing mean value')


def validate_completion(receipt, manifest, summary, completion, workflow_exit, stage_exits, stage_cleanups):
    """Validate fixed completion semantics without replaying games or importing JAX.

    Byte identity is checked separately by main. A failed learning gate is valid
    experiment completion; only its consistency with the recorded bounds matters.
    """
    try:
        _validate_completion(receipt, manifest, summary, completion, workflow_exit, stage_exits, stage_cleanups)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Incomplete or malformed F35 completion: ' + str(exc)) from exc


def _validate_completion(receipt, manifest, summary, completion, workflow_exit, stage_exits, stage_cleanups):
    require(receipt['status'] == 'completed' and summary['status'] == 'passed', 'Completion status')
    plan = manifest['plan']
    require(plan['schema_version'] == 'f35-teacher-assisted-plan-v1', 'Plan schema')
    require(plan['seeds'] == [7103, 8209, 9311] and plan['arms'] == ['control', 'assisted'], 'Plan cells')
    counts_match(plan['expected_counts'], COUNTS, 'Plan counts', exact=True)
    require(summary['input_hash'] == receipt['input_hash'], 'Summary input linkage')
    for record in (completion, workflow_exit):
        require(record['status'] == 'passed' and record['input_hash'] == receipt['input_hash'], 'Completion linkage')
    require(type(workflow_exit['exit_code']) is int and workflow_exit['exit_code'] == 0, 'Workflow exit')
    require([r['stage'] for r in receipt['stages']] == list(STAGES), 'Required ordered stages')
    require(set(stage_exits) == set(STAGES), 'Stage exit inventory')
    require(set(stage_cleanups) == set(STAGES), 'Stage cleanup inventory')
    for entry in receipt['stages']:
        name = entry['stage']
        require(entry == stage_exits[name], 'Stage receipt/exit mismatch: ' + name)
        require(type(entry['exit_code']) is int and entry['exit_code'] == 0
                and entry['deadline_timeout'] is False, 'Unsuccessful stage: ' + name)
        require(entry['cleanup_status'] == 'passed' and entry['leaked_after_success'] is False
                and entry['wait_error'] is None, 'Unsuccessful stage cleanup: ' + name)
        cleanup = stage_cleanups[name]
        require(cleanup['status'] == entry['cleanup_status']
                and integer(cleanup['root_pid'], 'Cleanup root PID', 1) == integer(entry['pid'], 'Stage PID', 1)
                and type(cleanup['returncode']) is int and cleanup['returncode'] == entry['exit_code'],
                'Stage cleanup linkage: ' + name)
        require(cleanup['stage_was_running'] is False, 'Stage was still running during successful cleanup: ' + name)
        for key in ('remaining_pids', 'errors', 'identity_changed_skipped',
                    'live_descendants_at_cleanup', 'initial_live_pids', 'signals'):
            require(cleanup[key] == [], 'Nonempty cleanup ' + key + ': ' + name)
    for k in ('cohorts', 'optimizer_steps', 'selected_rows', 'row_presentations',
              'demonstration_presentations', 'evaluation_games', 'checkpoint_comparisons',
              'original_next_update_recoveries'):
        counts_match(summary['counts'], {k: COUNTS[k]}, 'Training counts')
    require(summary['extra_demonstration_presentations'] == COUNTS['demonstration_presentations'], 'Demo presentation linkage')
    counts_match(summary['external_counts'], {'games': 1152}, 'External counts')
    require(summary['counts']['evaluation_games'] + summary['external_counts']['games'] == 8064, 'All evaluation games')
    counts_match(summary['verification'], VERIFICATION, 'Verification counts', exact=True)
    require(set(summary['teacher_roles']) == {'train', 'probe'}, 'Teacher role inventory')
    for role, expected in [('train', 384), ('probe', 96)]:
        cell = summary['teacher_roles'][role]
        counts_match(cell, {'games': expected}, 'Teacher ' + role)
        require(integer(cell['accepted_labels'], 'Teacher labels', 1) >= expected, 'Every teacher game labeled')
        require(integer(cell['terminal_games'], 'Teacher terminals') + integer(cell['cutoff_games'], 'Teacher cutoffs') == expected,
                'Teacher outcome totals')
    require(set(summary['seeds']) == SEEDS, 'Seed inventory')
    families = {'six primary/teacher contrasts': 0, '39 diagnostic contrasts': 0}
    total_contrasts, total_games, training_games = 0, 0, 0
    external_totals = {'terminals': 0, 'cutoffs': 0, 'decisions': 0}
    for seed, result in summary['seeds'].items():
        require(set(result['endpoints']) == ENDPOINTS and set(result['accounting']) == ARMS, 'Seed endpoint/arm inventory')
        for endpoint, cells in result['endpoints'].items():
            require(set(cells) == set(GAMES), 'Opponent inventory')
            for opponent, expected in GAMES.items():
                value_cell(cells[opponent], expected, seed + '/' + endpoint + '/' + opponent)
                total_games += expected
            for key in external_totals:
                external_totals[key] += cells['foul_play'][key]
        for arm, cell in result['accounting'].items():
            counts_match(cell['counts'], ARM_COUNTS, 'Arm counts', exact=True)
            require(set(cell['streams']) == set(STREAM_ROWS), 'Stream inventory')
            for stream, rows in STREAM_ROWS.items():
                stats = cell['streams'][stream]
                counts_match(stats, {'selected_rows': rows}, 'Stream selected rows')
                require(integer(stats['collected_rows'], 'Collected rows') - rows == integer(stats['excluded_rows'], 'Excluded rows'),
                        'Stream row accounting')
                integer(stats['games'], 'Stream games', 1)
            games = sum(s['games'] for s in cell['streams'].values())
            value_cell(cell['training'], games, seed + '/' + arm + '/training')
            training_games += games
        contrasts = result['analysis']['contrasts']
        require(set(contrasts) == set(GAMES), 'Contrast opponent inventory')
        for opponent, cells in contrasts.items():
            require(set(cells) == set(CONTRASTS), 'Contrast endpoint inventory')
            for name, (before, after) in CONTRASTS.items():
                contrast = cells[name]
                require(contrast['before_endpoint'] == before and contrast['after_endpoint'] == after, 'Contrast direction')
                require(contrast['initial'] == result['endpoints'][before][opponent]
                        and contrast['final'] == result['endpoints'][after][opponent], 'Contrast endpoint linkage')
                counts_match(contrast, {'pairs': GAMES[opponent] // 2}, 'Contrast pairs')
                family = 'six primary/teacher contrasts' if name == 'assisted_vs_control' and opponent in ('baseline-v1', 'foul_play') else '39 diagnostic contrasts'
                require(contrast['multiplicity_family'] == family, 'Contrast multiplicity family')
                families[family] += 1
                total_contrasts += 1
        for opponent, name in [('baseline-v1', 'primary_gate'), ('foul_play', 'teacher_adaptation_gate')]:
            contrast = contrasts[opponent]['assisted_vs_control']
            checks = {'practical_gain': contrast['cutoff_aware_point_interval'][0] >= .05,
                'adjusted_lower_positive': contrast['approximate_familywise_bootstrap_interval'][0] > 0,
                'both_final_cutoffs_admitted': all(result['endpoints'][a][opponent]['cutoffs'] / GAMES[opponent] <= .05 for a in ARMS)}
            gate = result['analysis'][name]
            require(gate['checks'] == checks and gate['passed'] is all(checks.values()), 'Learning gate consistency')
    require(total_contrasts == summary['contrasts'] == 45 and families == {'six primary/teacher contrasts': 6, '39 diagnostic contrasts': 39}, 'Contrast totals')
    require(total_games == 8064, 'Endpoint evaluation totals')
    counts_match(summary['counts'], {'training_games': training_games}, 'Training game linkage')
    counts_match(summary['external_counts'], external_totals, 'External outcome linkage')
    for field, gate in [('all_seed_primary_passed', 'primary_gate'), ('all_seed_teacher_adaptation_passed', 'teacher_adaptation_gate')]:
        require(summary[field] is all(s['analysis'][gate]['passed'] for s in summary['seeds'].values()), 'All-seed gate consistency')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    receipt_path = ROOT / 'artifacts/reports/f35-teacher-assisted.json'
    review = ROOT / 'artifacts/reviews/f35-teacher-assisted'
    body = json.loads(receipt_path.read_text())
    recorded = body.pop('content_hash')
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != recorded or body['status'] != 'completed':
        raise ValueError('Receipt content identity/status')
    if receipt_path.read_bytes() != (review / 'f35-teacher-assisted-receipt.json').read_bytes():
        raise ValueError('Archived receipt bytes differ')
    manifest_path = ROOT / body['input_manifest']
    if sha(manifest_path) != body['input_hash']:
        raise ValueError('Admission manifest identity')
    manifest = json.loads(manifest_path.read_text())
    if manifest['source_files'] != body['source_files']:
        raise ValueError('Source inventory differs from admitted manifest')
    for rel, expected in body['source_files'].items():
        if sha(ROOT / rel) != expected or sha(manifest_path.parent / 'source' / rel) != expected:
            raise ValueError('Live/admitted source mismatch: ' + rel)
    for rel, expected in body['evidence_files'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('Evidence mismatch: ' + rel)
    completion_records = {}
    for name in ('completion.json', 'workflow-exit.json'):
        record = json.loads((review / name).read_text())
        completion_records[name] = record
        if record['status'] != 'passed' or record['receipt_sha256'] != sha(receipt_path) or record['input_hash'] != body['input_hash']:
            raise ValueError('Completion linkage')
        if name == 'workflow-exit.json' and record['exit_code'] != 0:
            raise ValueError('Workflow exit failure')
    if sha(ROOT / body['summary']) != body['summary_sha256']:
        raise ValueError('Summary identity')
    summary = json.loads((ROOT / body['summary']).read_text())
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
        launch = json.loads((review / (name + '-launch.json')).read_text())
        if any(stage_exits[name].get(key) != value for key, value in launch.items()):
            raise ValueError('Stage launch/exit mismatch: ' + name)
    validate_completion(body, manifest, summary, completion_records['completion.json'],
                        completion_records['workflow-exit.json'], stage_exits, stage_cleanups)
    result = {'status': 'passed', 'receipt_sha256': sha(receipt_path), 'content_hash': recorded,
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
