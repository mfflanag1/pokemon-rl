#!/usr/bin/env python3
"""Read-only F39 completion verification; never replay or start scientific work."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = 'artifacts/reviews/f39-online-continuation'
RECEIPT = 'artifacts/reports/f39-online-continuation.json'
PLAN = 'contracts/online-continuation-v1/plan.json'
OUTPUT_PATHS = {'training': 'artifacts/runs/f39-online-continuation-v1',
    'verification': REVIEW + '/sampled-verification-v1', 'results': REVIEW + '/measured-results-v1'}
STAGES = ('training', 'sampled-verification', 'measured-results')
SEEDS = ('7103', '8209', '9311')
GAMES = {'baseline-v1': 512, 'tactical': 128, 'base_power': 64, 'random': 64}
STREAM_ROWS = {'self_play': 65536, 'frozen_f32': 65536, 'tactical': 131072}
COUNTS = {'cohorts': 768, 'optimizer_steps': 6144, 'selected_rows': 786432,
    'row_presentations': 1572864, 'evaluation_games': 4608, 'checkpoint_comparisons': 9,
    'original_next_update_recoveries': 6}
SEED_COUNTS = {'cohorts': 256, 'optimizer_steps': 2048, 'selected_rows': 262144,
    'row_presentations': 524288}
VERIFICATION = {'training_games': 18, 'evaluation_games': 48, 'game_executions': 66,
    'sampled_updates': 6, 'optimizer_steps_recomputed': 48, 'checkpoint_comparisons': 9}
STATISTICS = {'seed': 1027283903, 'resamples': 20000, 'alpha': .05, 'family_cells': 12,
    'practical_improvement': .05, 'max_final_cutoff_rate': .05}
F34_SHA = 'e88ffc1eeacf74cfa1afd5c3b112335fe60bfbd115d70648104eb1cd255d076a'


def require(condition, label):
    if not condition:
        raise ValueError('F39 completion: ' + label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_path(relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and '..' not in Path(relative).parts, 'unsafe receipt path')
    path = ROOT / relative
    require(path.resolve().is_relative_to(ROOT.resolve()), 'path escaped project')
    return path


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return float(value)


def counts_match(actual, expected, label, exact=False):
    require(isinstance(actual, dict), label)
    if exact:
        require(set(actual) == set(expected), label + ' inventory')
    for key, expected_value in expected.items():
        require(integer(actual[key], label + '/' + key) == expected_value, label + '/' + key)


def value_cell(cell, games, label):
    counts_match(cell, {'games': games}, label)
    for key in ('terminals', 'cutoffs', 'decisions', 'learner_wins', 'learner_losses', 'true_ties'):
        integer(cell[key], label + '/' + key)
    require(cell['decisions'] >= games and cell['terminals'] + cell['cutoffs'] == games,
            label + ' outcomes/decisions')
    require(sum(cell[k] for k in ('learner_wins', 'learner_losses', 'true_ties')) == cell['terminals'],
            label + ' terminal rewards')
    for key in ('terminal_game_mean_value_mse', 'terminal_game_mean_predicted_value'):
        value = cell[key]
        require((value is None) == (cell['terminals'] == 0), label + '/' + key + ' terminal-only')
        if value is not None:
            require(0 <= number(value, label + '/' + key) <= 1, label + '/' + key)
    require(cell['cutoff_rate'] == cell['cutoffs'] / games, label + '/cutoff rate')


def expected_commands():
    admission = str(ROOT / REVIEW / 'admission')
    paths = {k: str(ROOT / v) for k, v in OUTPUT_PATHS.items()}
    prefix = [str(ROOT / '.venv/bin/python'), '-m', 'pokemon_teacher_runtime.lifecycle', '--']
    return {
        'training': prefix + ['pokemon_online_continuation.campaign', '--admission', admission,
            '--run', paths['training']],
        'sampled-verification': prefix + ['pokemon_online_continuation.verification', '--training-run',
            paths['training'], '--output', paths['verification']],
        'measured-results': prefix + ['pokemon_online_continuation.reporting', '--training-run',
            paths['training'], '--verification', paths['verification'], '--output', paths['results']]}


def validate_stages(receipt, exits, cleanups, launches):
    require([r['stage'] for r in receipt['stages']] == list(STAGES), 'ordered stage inventory')
    require(all(set(records) == set(STAGES) for records in (exits, cleanups, launches)), 'stage evidence inventory')
    commands = expected_commands()
    for entry in receipt['stages']:
        name = entry['stage']
        expected_launch = {k: entry[k] for k in ('schema_version', 'stage', 'command', 'pid', 'started_utc')}
        require(entry == exits[name] and launches[name] == expected_launch,
                'stage launch/exit linkage ' + name)
        require(entry['schema_version'] == 'f39-stage-v1' and entry['command'] == commands[name],
                'stage command ' + name)
        require(type(entry['exit_code']) is int and entry['exit_code'] == 0
                and entry['deadline_timeout'] is False and entry['leaked_after_success'] is False
                and entry['wait_error'] is None and entry['cleanup_status'] == 'passed', 'failed stage ' + name)
        require(number(entry['elapsed_seconds'], 'stage elapsed') >= 0, 'negative stage elapsed')
        validate_cleanup(cleanups[name], entry['pid'], name)


def validate_cleanup(cleanup, pid, label):
    require(cleanup['status'] == 'passed' and cleanup['stage_was_running'] is False
            and integer(cleanup['root_pid'], 'cleanup PID', 1) == integer(pid, 'owner PID', 1)
            and type(cleanup['returncode']) is int and cleanup['returncode'] == 0, 'cleanup linkage ' + label)
    for key in ('remaining_pids', 'errors', 'identity_changed_skipped', 'live_descendants_at_cleanup',
                'initial_live_pids', 'signals', 'unresolved_unknown_pids', 'reused_pids', 'handle_signals'):
        require(cleanup[key] == [], 'nonempty cleanup ' + label + '/' + key)
    require(type(cleanup['unknown_overflow']) is int and cleanup['unknown_overflow'] == 0
            and cleanup['table_error'] is None, 'unresolved cleanup identity ' + label)


def validate_coverage(report):
    counts_match(report['counts'], VERIFICATION, 'verification counts', True)
    require(report['expected_counts'] == VERIFICATION, 'verification expected counts')
    wanted = {'checkpoint': {('cohort',): {(512,), (767,), (768,)}},
        'fresh_update': {('cohort',): {(512,), (767,)}},
        'fresh_training': {('cohort', 'stream'): {(c, s) for c in (512, 767) for s in STREAM_ROWS}},
        'fresh_evaluation_pair': {('endpoint', 'opponent'): {(e, o) for e in ('initial', 'final') for o in GAMES}}}
    require({str(r['seed']) for r in report['records']} == set(SEEDS), 'verification seed inventory')
    for seed in SEEDS:
        records = [r for r in report['records'] if str(r['seed']) == seed]
        require(Counter(r['kind'] for r in records) == {k: len(next(iter(v.values()))) for k, v in wanted.items()},
                'verification record counts')
        for kind, shapes in wanted.items():
            fields, coordinates = next(iter(shapes.items()))
            require({tuple(r[k] for k in fields) for r in records if r['kind'] == kind} == coordinates,
                    'verification coordinates ' + kind)
    inventory = report['coverage']['teacher_search_module_inventory']
    require(report['coverage']['exhaustive'] is False and inventory['before'] == [] and inventory['after'] == []
            and isinstance(inventory['forbidden_prefixes'], list) and inventory['forbidden_prefixes']
            and all(isinstance(p, str) and p for p in inventory['forbidden_prefixes']), 'sampled teacher-module coverage')
    require(isinstance(report['coverage']['teacher_search_execution_scope'], str)
            and report['coverage']['teacher_search_execution_scope'], 'teacher inventory scope')


def validate_summary(summary, training, verification):
    counts_match(summary['counts'], COUNTS, 'campaign counts')
    require(summary['counts'] == training['counts'] and set(summary['counts']) == set(COUNTS) | {'training_games'},
            'campaign count linkage')
    require(set(summary['seeds']) == set(training['seeds']) == set(SEEDS), 'all three seed cells')
    require(summary['statistics'] == STATISTICS, 'fixed statistics')
    counts_match(summary, {'primary_contrasts': 3, 'regression_diagnostic_contrasts': 9,
        'total_contrasts': 12}, 'contrast counts')
    require(summary['promotion_authorized'] is False, 'promotion not authorized')
    validate_coverage(verification)
    require(summary['verification'] == {'counts': verification['counts'], 'coverage': verification['coverage']},
            'verification summary linkage')
    aggregate, training_games, all_gates, any_regression = Counter(), 0, [], False
    diagnostics_expected = []
    for seed, row in summary['seeds'].items():
        source = training['seeds'][seed]
        require(source['seed'] == int(seed) and source['identity']['adam_reset'] is False, 'seed/Adam identity')
        counts_match(source, {'initial_optimizer_step': 4096, 'final_optimizer_step': 6144,
            'optimizer_steps': 6144, 'continuation_optimizer_steps': 2048,
            'cumulative_selected_rows': 786432, 'cumulative_row_presentations': 1572864}, 'seed cumulative counts')
        require(source['initial_stream_cursors'] == source['identity']['f34_metadata']['stream_cursors'], 'inherited cursors')
        require(source['initial_parameter_hash'] == source['identity']['f34_parameter_hash']
                and source['initial_state_hash'] == source['identity']['state_hash'], 'inherited parameter/Adam identity')
        account = row['training']
        counts_match(account['counts'], SEED_COUNTS, 'seed update counts')
        require(set(account['counts']) == set(SEED_COUNTS) | {'training_games'}, 'seed count inventory')
        require(set(account['streams']) == set(STREAM_ROWS), 'stream inventory')
        stream_totals = Counter()
        for stream, selected in STREAM_ROWS.items():
            cell = account['streams'][stream]
            counts_match(cell, {'selected_rows': selected}, 'stream selected rows')
            require(integer(cell['collected_rows'], 'collected rows') == selected + integer(cell['excluded_rows'], 'excluded rows'),
                    'stream selected/excluded linkage')
            for key, value in cell.items():
                integer(value, 'stream/' + key)
            require(cell['terminals'] + cell['cutoffs'] == cell['games']
                    and cell['collected_rows'] + cell['non_learner_decisions'] == cell['decisions']
                    and cell['rejected_selected_rows'] <= cell['rejected_controlled_attempts'] <= cell['rejected_attempts'],
                    'stream outcome/rejection counts')
            require(source['stream_cursors'][stream] - source['initial_stream_cursors'][stream] == cell['games'],
                    'stream cursor/game count')
            stream_totals.update(cell)
        require(dict(stream_totals) == account['totals'], 'seed stream totals')
        aggregate.update(stream_totals)
        n = integer(stream_totals['games'], 'training games', 1)
        require(account['counts']['training_games'] == n, 'training game count')
        training_games += n
        value_cell(account['outcomes'], n, seed + '/training')
        for key in ('games', 'terminals', 'cutoffs', 'decisions', 'true_ties'):
            require(account['outcomes'][key] == stream_totals[key], 'training outcome total')
        diagnostics = account['optimizer_diagnostics']
        for label, expected_n in [('first_minibatch_per_cohort', 256), ('last_epoch_minibatches', 1024)]:
            for metric in ('approx_kl', 'clip_fraction'):
                stats = diagnostics[label][metric]
                counts_match(stats, {'n': expected_n}, 'minibatch statistic count')
                for key in ('mean', 'median', 'p95', 'min', 'max', 'sum'):
                    number(stats[key], 'minibatch statistic')
        require(set(row['endpoints']) == {'initial', 'final'}, 'endpoint inventory')
        for endpoint, cells in row['endpoints'].items():
            require(set(cells) == set(GAMES), 'opponent inventory')
            for opponent, n_games in GAMES.items():
                value_cell(cells[opponent], n_games, seed + '/' + endpoint + '/' + opponent)
        analysis = row['analysis']
        diagnostics_expected.extend({'seed': int(seed), 'opponent': o, **c}
            for o in GAMES if o != 'baseline-v1' for c in [analysis['contrasts'][o]])
        require(analysis == source['analysis'] and set(analysis['contrasts']) == set(GAMES), 'analysis linkage')
        regressions = []
        for opponent, contrast in analysis['contrasts'].items():
            n_games = GAMES[opponent]
            require(integer(contrast['pairs'], 'paired schedule count') == n_games // 2
                    and contrast['bootstrap_resamples'] == 20000, 'contrast fixed coordinates')
            for endpoint in ('initial', 'final'):
                cell = row['endpoints'][endpoint][opponent]
                raw = contrast[endpoint]
                require(raw == cell, 'contrast terminal-only endpoint summary')
                for key, value in source[endpoint][opponent].items():
                    if key in cell:
                        require(value == cell[key], 'contrast original endpoint linkage/' + key)
            a, b = contrast['initial'], contrast['final']
            delta = b['learner_wins'] + .5 * b['true_ties'] - a['learner_wins'] - .5 * a['true_ties']
            point = [(delta - a['cutoffs']) / n_games, (delta + b['cutoffs']) / n_games]
            require(contrast['cutoff_aware_point_interval'] == point, 'cutoff-aware point interval')
            for key in ('approximate_familywise_bootstrap_interval', 'bounded_hoeffding_interval'):
                bounds = contrast[key]
                require(len(bounds) == 2 and -1 <= number(bounds[0], 'interval lower') <=
                        number(bounds[1], 'interval upper') <= 1, 'interval order/range')
            radius = math.sqrt(2 * math.log(2 * 12 / .05) / contrast['pairs'])
            wanted = [max(-1., point[0] - radius), min(1., point[1] + radius)]
            require(all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
                        for x, y in zip(contrast['bounded_hoeffding_interval'], wanted)), 'bounded interval')
            lo, hi = contrast['approximate_familywise_bootstrap_interval']
            label = 'improvement' if lo > 0 else 'regression' if hi < 0 else 'inconclusive'
            require(contrast['gate'] == label, 'contrast gate')
            if label == 'regression':
                regressions.append(opponent)
        require(set(analysis['detected_regressions']) == set(regressions)
                and len(analysis['detected_regressions']) == len(regressions)
                and analysis['any_detected_regression'] is bool(regressions), 'regression flags')
        primary = analysis['contrasts']['baseline-v1']
        checks = {'cutoff_aware_lower_point_gain_at_least_five_points': primary['cutoff_aware_point_interval'][0] >= .05,
            'adjusted_lower_positive': primary['approximate_familywise_bootstrap_interval'][0] > 0,
            'final_cutoff_rate_admitted': primary['final']['cutoffs'] / 512 <= .05}
        require(analysis['primary_gate'] == {'checks': checks, 'passed': all(checks.values())}, 'primary gate')
        all_gates.append(all(checks.values()))
        any_regression |= bool(regressions)
    require(summary['regression_diagnostics'] == diagnostics_expected, 'all nine regression diagnostics')
    require(training_games == summary['counts']['training_games'] and dict(aggregate) == summary['training_totals'],
            'all-seed training totals')
    require(summary['all_seed_primary_passed'] is all(all_gates)
            and training['all_seed_learning_passed'] is all(all_gates)
            and summary['any_detected_regression'] is any_regression, 'all-seed gates')
    for key in ('elapsed_seconds', 'collection_seconds', 'update_seconds_including_compile',
                'sampled_replay_update_seconds', 'peak_parent_rss_mib'):
        require(number(summary['costs'][key], 'measured cost') >= 0 and summary['costs'][key] == training[key],
                'measured cost linkage')


def validate_completion(receipt, manifest, summary, training, verification, completion,
                        workflow_exit, driver_exit, stage_exits, stage_cleanups, stage_launches):
    """Pure semantic checks; failed learning gates still permit valid completion."""
    try:
        expected = [(receipt, 'f39-online-continuation-receipt-v1', 'completed'),
            (manifest, 'f39-inputs-v1', None), (training, 'f39-online-continuation-run-v1', 'passed'),
            (summary, 'f39-measured-summary-v1', 'passed'), (completion, 'f39-completion-v1', 'passed'),
            (workflow_exit, 'f39-workflow-exit-v1', 'passed'), (driver_exit, 'f39-driver-exit-v1', None)]
        for record, schema, status in expected:
            require(record['schema_version'] == schema, 'schema/' + schema)
            if status is not None:
                require(record['status'] == status, 'status/' + schema)
        require(verification['status'] == 'passed', 'verification status')
        plan = manifest['plan']
        require(plan['schema_version'] == 'f39-online-continuation-plan-v1' and plan['seeds'] == [7103, 8209, 9311]
                and plan['expected_counts'] == COUNTS and plan['statistics'] == STATISTICS, 'fixed plan')
        require(training['plan'] == plan, 'training plan linkage')
        require(manifest['runtime']['backend'] == 'cpu' and manifest['runtime']['x64'] is False
                and manifest['qualification']['status'] == 'passed', 'qualified CPU FP32 admission')
        require(manifest['predecessor'] == {'path': 'artifacts/reports/f34-policy-reset.json', 'sha256': F34_SHA},
                'F34 predecessor')
        for record in (summary, training, verification, completion, workflow_exit):
            require(record['input_hash'] == receipt['input_hash'], 'input hash linkage')
        for record in (workflow_exit, driver_exit):
            require(type(record['exit_code']) is int and record['exit_code'] == 0, 'outer workflow/driver exit')
        require(driver_exit['error'] is None, 'outer launcher error')
        validate_cleanup(driver_exit['cleanup'], driver_exit['cleanup']['root_pid'], 'outer driver')
        for record in (completion, workflow_exit):
            require(record['receipt'] == RECEIPT, 'completion receipt path')
        require(receipt['output_paths'] == OUTPUT_PATHS and receipt['summary'] == OUTPUT_PATHS['results'] + '/summary.json',
                'output path inventory')
        for key in ('teacher_used', 'demonstrations_used', 'public_server_used', 'final_reserved_data_used', 'promotion_authorized'):
            require(receipt[key] is False, 'receipt scope/' + key)
        require(receipt['learning_success_separate_from_execution'] is True, 'learning/execution distinction')
        validate_stages(receipt, stage_exits, stage_cleanups, stage_launches)
        validate_summary(summary, training, verification)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Incomplete or malformed F39 completion: ' + str(exc)) from exc


def main(output):
    output = Path(output).resolve()
    review = bound_path(REVIEW)
    require(not output.is_relative_to(review.resolve()) and not output.is_relative_to(bound_path(OUTPUT_PATHS['training']).resolve()),
            'postcheck must be outside sealed evidence')
    require(not output.exists(), 'postcheck output already exists')
    receipt_path = bound_path(RECEIPT)
    receipt = read(receipt_path)
    content = {k: v for k, v in receipt.items() if k != 'content_hash'}
    require(hashlib.sha256(canonical(content)).hexdigest() == receipt['content_hash'], 'receipt content hash')
    require(receipt_path.read_bytes() == (review / 'f39-online-continuation-receipt.json').read_bytes(), 'archived receipt bytes')
    manifest_path = bound_path(receipt['input_manifest'])
    require(receipt['input_manifest'] == REVIEW + '/admission/manifest.json'
            and sha(manifest_path) == receipt['input_hash'], 'admission identity')
    manifest = read(manifest_path)
    require(manifest['source_files'] == receipt['source_files'] and manifest['source_files'], 'source inventory')
    for relative, expected in receipt['source_files'].items():
        require(sha(bound_path(relative)) == expected and sha(manifest_path.parent / 'source' / relative) == expected,
                'live/frozen source hash: ' + relative)
    require(set(str(p.relative_to(manifest_path.parent / 'source')) for p in (manifest_path.parent / 'source').rglob('*') if p.is_file())
            == set(receipt['source_files']), 'complete archived source inventory')
    require(receipt['source_files'].get(PLAN) == sha(bound_path(PLAN)) and read(bound_path(PLAN)) == manifest['plan'], 'plan byte/content binding')
    require(sha(bound_path(manifest['predecessor']['path'])) == F34_SHA, 'predecessor receipt bytes')
    evidence = receipt['evidence_files']
    for relative, expected in evidence.items():
        require(sha(bound_path(relative)) == expected, 'evidence hash: ' + relative)
    require(receipt['output_paths'] == OUTPUT_PATHS, 'fixed output paths')
    for relative in OUTPUT_PATHS.values():
        actual = {str(p.relative_to(ROOT)) for p in bound_path(relative).rglob('*') if p.is_file()}
        listed = {p for p in evidence if p.startswith(relative + '/')}
        require(actual and actual == listed, 'complete output inventory: ' + relative)
    require((bound_path(OUTPUT_PATHS['training']) / 'inputs/manifest.json').read_bytes() == manifest_path.read_bytes(),
            'training admission copy bytes')
    stage_exits, stage_cleanups, stage_launches = {}, {}, {}
    for name in STAGES:
        for suffix in ('.log', '-launch.json', '-exit.json', '-cleanup.json'):
            relative = REVIEW + '/' + name + suffix
            require(evidence.get(relative) == sha(bound_path(relative)), 'required stage evidence')
        stage_exits[name] = read(review / (name + '-exit.json'))
        stage_cleanups[name] = read(review / (name + '-cleanup.json'))
        stage_launches[name] = read(review / (name + '-launch.json'))
    for name in ('authorization.json', 'qualification.json', 'prelaunch-independent.json',
                 'launcher-claim.json', 'driver-launch.json', 'workflow-claim.json'):
        relative = REVIEW + '/' + name
        require(evidence.get(relative) == sha(bound_path(relative)), 'required launch/admission evidence')
    for name in ('qualification.json', 'prelaunch-independent.json'):
        require(read(review / name)['status'] == 'passed', 'prelaunch admission did not pass')
    driver_launch, workflow_claim = read(review / 'driver-launch.json'), read(review / 'workflow-claim.json')
    require(driver_launch['driver_pid'] == workflow_claim['pid'] and driver_launch['launcher_pid'] == read(review / 'launcher-claim.json')['pid'],
            'single launcher/driver ownership')
    completion, workflow_exit, driver_exit = [read(review / n) for n in ('completion.json', 'workflow-exit.json', 'driver-exit.json')]
    require(driver_exit['cleanup']['root_pid'] == driver_launch['driver_pid'], 'outer cleanup owner')
    for record in (completion, workflow_exit):
        require(record['receipt_sha256'] == sha(receipt_path)
                and record['source_files_verified'] == len(receipt['source_files'])
                and record['evidence_files_verified'] == len(evidence), 'completion byte/count linkage')
    require(sha(bound_path(receipt['summary'])) == receipt['summary_sha256'], 'summary SHA')
    summary = read(bound_path(receipt['summary']))
    training = read(bound_path(OUTPUT_PATHS['training']) / 'report.json')
    verification = read(bound_path(OUTPUT_PATHS['verification']) / 'report.json')
    required_inputs = {str(bound_path(OUTPUT_PATHS['training']) / p) for p in ('report.json', 'inputs/manifest.json')}
    for record in (summary, verification):
        require(required_inputs.issubset(record['input_files']), 'missing original training input bindings')
        for absolute, expected in record['input_files'].items():
            path = Path(absolute)
            require(path.is_absolute() and any(path.resolve().is_relative_to(bound_path(OUTPUT_PATHS[key]).resolve())
                    for key in ('training', 'verification')), 'bound input path outside original science evidence')
            relative = str(path.relative_to(ROOT))
            require(evidence.get(relative) == expected and sha(path) == expected, 'analysis/replay input evidence linkage')
    require(str(bound_path(OUTPUT_PATHS['verification']) / 'report.json') in summary['input_files'],
            'missing verification report binding')
    validate_completion(receipt, manifest, summary, training, verification, completion, workflow_exit,
                        driver_exit, stage_exits, stage_cleanups, stage_launches)
    result = {'schema_version': 'f39-postcheck-v1', 'status': 'passed', 'receipt_sha256': sha(receipt_path),
        'content_hash': receipt['content_hash'], 'live_and_frozen_sources': len(receipt['source_files']),
        'evidence_files': len(evidence), 'counts': summary['counts'], 'total_contrasts': summary['total_contrasts'],
        'all_seed_primary_passed': summary['all_seed_primary_passed'], 'any_detected_regression': summary['any_detected_regression'],
        'scope': 'Receipt and archived bytes, all admitted live/frozen source hashes and live evidence hashes, full output inventories, stage ownership/cleanup, counts, cells, cutoff bounds, gates and completion/driver linkage. No scientific replay launched.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
