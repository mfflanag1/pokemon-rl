"""Aggregate completed F34 evidence without running games or model updates.

The coordinator owns the final evidence receipt. This module checks report and
ledger linkage, recomputes the prespecified comparisons, and reports measured
costs with their actual scope. It never selects checkpoints or drops seeds.
"""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from pokemon_full_train.campaign import value_summary
from pokemon_full_learning.statistics import compare
from pokemon_full_train.storage import ROOT, atomic_json, digest
from .common import SEEDS, read, learning_analysis, evaluation_specs

STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
ENDPOINTS = ('initial', 'final')


def require(condition, message):
    if not condition:
        raise ValueError('F34 reporting: ' + message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return float(value)


def measurements(values):
    a = np.asarray(values, np.float64)
    require(a.ndim == 1 and np.isfinite(a).all(), 'nonfinite measurements')
    if not len(a):
        return {'n': 0, 'mean': None, 'median': None, 'p95': None, 'min': None, 'max': None, 'sum': 0.}
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'min': float(a.min()),
            'max': float(a.max()), 'sum': float(a.sum())}


def outcome_summary(games):
    for game in games:
        kind = game['outcome']['kind']
        require(kind in ('terminal', 'administrative_cutoff'), 'unfinished game')
        integer(game['decisions'], 'game decisions')
        finite(game['value_mean'], 'predicted value')
        if kind == 'terminal':
            require(finite(game['terminal_value_mse'], 'terminal value MSE') >= 0, 'negative value MSE')
            rewards = game['outcome']['rewards']
            require(set(rewards) == {'p1', 'p2'} and set(rewards.values()) <= {0, .5, 1}
                    and sum(rewards.values()) == 1, 'terminal rewards')
    summary = value_summary(games)
    # The existing helper also averages predictions across cutoffs. Keep value
    # diagnostics terminal-only here; cutoff scores are intervals, never draws.
    del summary['game_mean_predicted_value']
    terminal = [g for g in games if g['outcome']['kind'] == 'terminal']
    summary['terminal_game_mean_predicted_value'] = value_summary(terminal)['game_mean_predicted_value']
    summary['cutoff_rate'] = summary['cutoffs'] / len(games) if games else None
    return summary


def contrast(initial, final, settings):
    result = compare(initial, final, settings)
    result['initial'], result['final'] = outcome_summary(initial), outcome_summary(final)
    return result


def _completed(folder, input_hash=None):
    report = read(folder / 'report.json')
    require(report.get('status') == 'passed', 'report must pass: ' + str(folder))
    if input_hash is not None:
        require(report.get('input_hash') == input_hash, 'input identity: ' + str(folder))
    return report


def _panel_games(base, seed, endpoint, opponent, pairs, plan):
    folder = base / str(seed) / endpoint
    expected = [folder / f'{opponent}-{first:03d}.json' for first in range(0, pairs, 4)]
    require(sorted(folder.glob(f'{opponent}-*.json')) == expected, 'local panel chunk inventory')
    games = []
    for first, path in zip(range(0, pairs, 4), expected):
        chunk = read(path)
        specs = evaluation_specs(plan, seed, opponent, first, min(4, pairs - first))
        require(chunk['specs'] == specs and len(chunk['games']) == len(specs), 'local schedule/count')
        for game, spec in zip(chunk['games'], specs):
            require(all(game[k] == spec[k] for k in ('episode', 'swap', 'learner_side', 'opponent', 'split')),
                    'local game coordinates')
        games.extend(chunk['games'])
    require(len(games) == 2 * pairs, 'local panel game count')
    return games


def _training(base, seed, plan, seed_report):
    cohorts = integer(plan['cohorts_per_seed'], 'cohorts', 1)
    expected = [base / str(seed) / f'update-{i:04d}.json' for i in range(cohorts)]
    require(sorted((base / str(seed)).glob('update-*.json')) == expected, 'update inventory')
    by_stream = {label: Counter() for label in STREAMS}
    cursors = {label: 0 for label in STREAMS}
    first_metrics, last_metrics, all_games = [], [], []
    minibatches_per_epoch = plan['selected_rows_per_cohort'] // plan['minibatch_rows']
    last_epoch_start = (plan['epochs'] - 1) * minibatches_per_epoch
    previous_hash = seed_report['initial_parameter_hash']
    for cohort, path in enumerate(expected):
        update = read(path)
        require(update['cohort'] == cohort and update['optimizer_step'] == (cohort + 1) * plan['optimizer_steps_per_cohort'],
                'update coordinates')
        require(update['parameter_hash_before'] == previous_hash, 'update parameter chain')
        previous_hash = update['parameter_hash_after']
        metrics = update['metrics']
        require(len(metrics) == plan['optimizer_steps_per_cohort'], 'minibatch metrics count')
        for metric in metrics:
            for name, value in metric.items():
                finite(value, 'optimizer metric/' + name)
            require(0 <= metric['clip_fraction'] <= 1, 'clip fraction')
        first_metrics.append(metrics[0])
        last_metrics.extend(metrics[last_epoch_start:])
        ledger = update['ledger']
        require(len(ledger['strata']) == len(STREAMS), 'training strata count')
        strata = {s['stream']: s for s in ledger['strata']}
        require(set(strata) == set(STREAMS), 'training strata inventory')
        used = 0
        for label, opponent in STREAMS.items():
            stratum = strata[label]
            count = integer(stratum['games'], 'stratum game count', 1)
            selected = stratum['selection']
            collected = integer(selected['collected_rows'], 'collected learner rows')
            chosen = integer(selected['selected_rows'], 'selected learner rows')
            excluded = integer(selected['excluded_rows'], 'excluded learner rows')
            require(chosen == plan['stream_selected_rows'][label] and collected == chosen + excluded,
                    'stratum row accounting')
            require(stratum['opponent'] == opponent and stratum['first_episode'] == cursors[label]
                    and stratum['next_episode'] == cursors[label] + count, 'stratum cursor/opponent')
            indices = selected['selected_indices']
            require(len(indices) == chosen and len(set(indices)) == chosen
                    and all(type(i) is int and 0 <= i < collected for i in indices), 'selected row inventory')
            games = ledger['games'][used:used + count]
            require(len(games) == count, 'stratum games missing')
            for index, game in enumerate(games):
                require(game['opponent'] == opponent and game['split'] == 'train'
                        and game['episode'] == cursors[label] + index and game['swap'] is False
                        and game['learner_side'] == ('p2' if game['episode'] % 2 else 'p1'),
                        'training game schedule')
            summary = outcome_summary(games)
            decisions = summary['decisions']
            require(collected <= decisions and (label != 'self_play' or collected == decisions),
                    'collected rows versus decisions')
            by_stream[label].update(games=count, decisions=decisions,
                terminals=summary['terminals'], cutoffs=summary['cutoffs'], true_ties=summary['true_ties'],
                collected_rows=collected, selected_rows=chosen, excluded_rows=excluded,
                non_learner_decisions=decisions - collected,
                unavailable_choices=sum(integer(g['unavailable_choices'], 'unavailable choices') for g in games))
            used += count
            cursors[label] += count
            all_games.extend(games)
        require(used == len(ledger['games']), 'unassigned training games')
    require(previous_hash == seed_report['final_parameter_hash'], 'final update parameter hash')
    require(cursors == seed_report['stream_cursors'], 'final stream cursors')
    require(seed_report['optimizer_steps'] == cohorts * plan['optimizer_steps_per_cohort'], 'seed optimizer steps')
    require(value_summary(all_games) == seed_report['training'], 'training summary linkage')
    totals = Counter()
    for counts in by_stream.values():
        totals.update(counts)
    return {'streams': {k: dict(v) for k, v in by_stream.items()}, 'totals': dict(totals),
        'optimizer_diagnostics': {
            'first_minibatch_per_cohort': {k: measurements([m[k] for m in first_metrics])
                                         for k in ('approx_kl', 'clip_fraction')},
            'last_epoch_minibatches': {k: measurements([m[k] for m in last_metrics])
                                      for k in ('approx_kl', 'clip_fraction')},
            'scope': 'Pre-update minibatch diagnostics across all cohorts, not full-batch post-update KL; no adaptive stopping or post-hoc gate.'}}


def _external(folder, seed, endpoint, report, expected_hash):
    seed_report = report['seeds'][str(seed)]
    require(read(folder / str(seed) / 'result.json') == seed_report, 'external seed report')
    require(seed_report['seed'] == seed and seed_report['parameter_hash'] == expected_hash, 'external policy identity')
    games, plan = seed_report['games'], report['plan']
    require(len(games) == 2 * plan['pairs_per_seed'], 'external seed game count')
    require(value_summary(games) == seed_report['summary'], 'external summary linkage')
    wall, search, nominal_ms, game_seconds = [], [], Counter(), []
    rejected, illegal = Counter(), Counter()
    for i, game in enumerate(games):
        episode, swap = i // 2, bool(i % 2)
        require(game['episode'] == episode and game['swap'] is swap and game['opponent'] == 'foul_play'
                and game['learner_side'] == ('p2' if swap else 'p1') and game['split'] == 'development',
                'external game coordinates')
        path = folder / str(seed) / f'{episode:03d}-{int(swap)}' / 'result.json'
        require((ROOT / game['game_record']).resolve() == path.resolve(), 'external game path')
        require(digest(path) == game['game_record_sha256'], 'external game digest')
        raw = read(path)
        require(raw['status'] == 'completed' and raw['outcome'] == game['outcome']
                and raw['provenance'] == game['provenance'], 'external game completion')
        require(raw['search_time_ms'] == plan['search_time_ms'] and raw['parallelism'] == plan['parallelism']
                and raw['search_threads'] == plan['search_threads'], 'external fixed search configuration')
        require(sum(len(s['actions']) for s in raw['steps']) == game['decisions'], 'external decision total')
        live = [s for s in raw['steps'] if 'action' in s['external']]
        require(len(live) == raw['foul_play_decisions'] == game['foul_play_decisions'], 'external search count')
        for step in live:
            w = finite(step['external_wall_seconds'], 'external request latency')
            s = finite(step['external']['seconds'], 'external search latency')
            require(w >= 0 and s >= 0, 'negative external latency')
            wall.append(w); search.append(s)
            decision = step['external']
            allocation = decision['sampled_worlds'] * decision['per_world_ms']
            require(decision['search_time_ms'] == plan['search_time_ms']
                    and decision['sampled_worlds'] in (2, 4)
                    and decision['per_world_ms'] == (plan['search_time_ms'] if decision['sampled_worlds'] == 2
                                                      else plan['search_time_ms'] // 2)
                    and allocation == decision['configured_search_total_ms'], 'external nominal allocation')
            nominal_ms[str(allocation)] += 1
        for field, total in [('rejected_actions', rejected), ('illegal_actions', illegal)]:
            require(raw[field] == game[field] and set(raw[field]) == {'p1', 'p2'}, 'external action accounting')
            for side, count in raw[field].items():
                total[side] += integer(count, field)
        require(sum(raw['illegal_actions'].values()) == 0, 'illegal external action in completed panel')
        seconds = finite(raw['seconds'], 'external game seconds')
        require(seconds >= 0 and seconds == game['seconds'], 'external game timing linkage')
        game_seconds.append(seconds)
    return games, {'outcomes': outcome_summary(games),
        'foul_play_request_wall_seconds': measurements(wall), 'foul_play_search_seconds': measurements(search),
        'nominal_engine_ms_per_decision_counts': dict(nominal_ms), 'game_seconds': measurements(game_seconds),
        'rejected_actions': dict(rejected), 'illegal_actions': dict(illegal)}


def _verify_coverage(report, plan):
    n, samples, opponents = len(SEEDS), len(plan['sampled_update_cohorts']), len(plan['evaluation_pairs'])
    expected = {'training_games': n * samples * len(STREAMS), 'evaluation_games': n * 2 * opponents * 2,
        'external_action_replay_games': n * 2 * 2, 'sampled_updates': n * samples,
        'optimizer_steps_recomputed': n * samples * plan['optimizer_steps_per_cohort']}
    expected['game_executions'] = sum(expected[k] for k in ('training_games', 'evaluation_games', 'external_action_replay_games'))
    require(report['counts'] == expected == report['expected_counts'], 'verification count inventory')
    require({r['seed'] for r in report['records']} == set(SEEDS), 'verification seed coverage')
    for seed in SEEDS:
        records = [r for r in report['records'] if r['seed'] == seed]
        counts = Counter(r['kind'] for r in records)
        require(counts == {'fresh_training': samples * len(STREAMS), 'fresh_update': samples,
                          'fresh_evaluation_pair': 2 * opponents, 'recorded_external_actions': 4},
                'verification per-seed record counts')
        expected_keys = {
            'fresh_training': {(c, stream) for c in plan['sampled_update_cohorts'] for stream in STREAMS},
            'fresh_update': {(c,) for c in plan['sampled_update_cohorts']},
            'fresh_evaluation_pair': {(e, o) for e in ENDPOINTS for o in plan['evaluation_pairs']},
            'recorded_external_actions': {(e, swap) for e in ENDPOINTS for swap in (False, True)}}
        fields = {'fresh_training': ('cohort', 'stream'), 'fresh_update': ('cohort',),
                  'fresh_evaluation_pair': ('endpoint', 'opponent'),
                  'recorded_external_actions': ('endpoint', 'swap')}
        for kind, keys in expected_keys.items():
            require({tuple(r[k] for k in fields[kind]) for r in records if r['kind'] == kind} == keys,
                    'verification per-seed coverage coordinates')
    return expected


def _render(summary):
    def number(value, digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    lines = ['# F34 measured results', '',
        'All three seeds and both endpoints are included. Execution/verification completion and the learning gate are separate.', '',
        f"All-seed local learning gate: **{'passed' if summary['all_seed_learning_passed'] else 'not passed'}**.", '',
        '| Seed | Opponent | Initial W/L/T; cutoffs | Final W/L/T; cutoffs | Gain bounds | Adjusted bootstrap interval | Result |',
        '|---|---|---|---|---|---|---|']
    for seed, row in summary['seeds'].items():
        for opponent, comparison in {**row['local']['contrasts'], 'foul_play': row['external']['contrast']}.items():
            a, b = comparison['initial'], comparison['final']
            score = lambda x: f"{x['learner_wins']}/{x['learner_losses']}/{x['true_ties']}; {x['cutoffs']}"
            interval = lambda x: f'[{x[0]:+.3f}, {x[1]:+.3f}]'
            lines.append(f"| {seed} | {opponent} | {score(a)} | {score(b)} | {interval(comparison['cutoff_aware_point_interval'])} | {interval(comparison['approximate_familywise_bootstrap_interval'])} | {comparison['gate']} |")
    lines += ['', 'Cutoffs retain [0, 1] score uncertainty and are not counted as ties. Intervals use the training plan’s paired-cluster bootstrap settings and 12-cell family when specified. Foul Play transfer is descriptive, with no separate promotion gate.', '',
        '| Seed | Stream | Games | Decisions | Terminal / cutoff | Collected learner rows | Selected | Excluded by selection |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed, row in summary['seeds'].items():
        for stream, counts in row['training']['streams'].items():
            lines.append(f"| {seed} | {stream} | {counts['games']} | {counts['decisions']} | {counts['terminals']} / {counts['cutoffs']} | {counts['collected_rows']} | {counts['selected_rows']} | {counts['excluded_rows']} |")
    lines += ['', 'Collected rows are eligible learner decisions after complete-trajectory processing. Excluded rows here are removed by row selection; frozen/tactical opponent decisions never enter that pool.', '',
        '| Seed | First-minibatch mean KL / clip | Last-epoch mean KL / clip | Local learning gate | Detected regressions |',
        '|---|---:|---:|---|---|']
    for seed, row in summary['seeds'].items():
        diagnostics = row['training']['optimizer_diagnostics']
        first, last = diagnostics['first_minibatch_per_cohort'], diagnostics['last_epoch_minibatches']
        lines.append(f"| {seed} | {number(first['approx_kl']['mean'])} / {number(first['clip_fraction']['mean'])} | {number(last['approx_kl']['mean'])} / {number(last['clip_fraction']['mean'])} | {'passed' if row['local']['passed'] else 'not passed'} | {', '.join(row['detected_regressions']) or 'none detected'} |")
    lines += ['', 'KL and clipping values are measured before each minibatch update; they are not full-batch post-update KL or additional success gates.', '',
        '| Seed | Opponent | Initial terminal value MSE | Final terminal value MSE |', '|---|---|---:|---:|']
    for seed, row in summary['seeds'].items():
        for opponent, comparison in {**row['local']['contrasts'], 'foul_play': row['external']['contrast']}.items():
            lines.append(f"| {seed} | {opponent} | {number(comparison['initial']['terminal_game_mean_value_mse'])} | {number(comparison['final']['terminal_game_mean_value_mse'])} |")
    c = summary['costs']
    lines += ['', 'Value error is an equal-game average over terminal games only. Cutoff positions have no terminal outcome label.', '',
        f"Training run: {c['training_elapsed_seconds']:.2f}s; collection including local evaluation: {c['collection_seconds_including_local_evaluation']:.2f}s; optimizer calls including compilation: {c['update_seconds_including_compile']:.2f}s; in-run sampled update replay: {c['sampled_replay_update_seconds']:.2f}s.", '',
        f"External initial/final panels: {c['external_initial_seconds']:.2f}s / {c['external_final_seconds']:.2f}s. Independent sampled verification: {c['verification_seconds']:.2f}s. Component time sum: {c['component_elapsed_seconds_sum']:.2f}s; this is not a measurement of task elapsed time if components overlap.", '',
        f"Selected learner rows per training-run wall hour: {c['selected_rows_per_training_run_hour']:.1f}. Peak training-parent RSS: {c['peak_parent_rss_mib']:.1f} MiB; this excludes child processes and is not total machine or GPU memory.", '',
        '| Seed | Endpoint | Foul Play decisions | Mean request latency | P95 request latency | Rejected / illegal actions |',
        '|---|---|---:|---:|---:|---:|']
    for seed, row in summary['seeds'].items():
        for endpoint in ENDPOINTS:
            e = row['external'][endpoint]
            latency = e['foul_play_request_wall_seconds']
            lines.append(f"| {seed} | {endpoint} | {latency['n']} | {number(latency['mean'])}s | {number(latency['p95'])}s | {sum(e['rejected_actions'].values())} / {sum(e['illegal_actions'].values())} |")
    lines += ['', *[paragraph for paragraph in summary['limitations']], '',
        'Verification was sampled: ' + json.dumps(summary['verification']['counts'], sort_keys=True) + '.', '']
    return '\n'.join(lines)


def run(training_run, external_initial, external_final, verification, output):
    """Write summary.json/results.md from four completed, matching evidence roots."""
    training_run, external_initial, external_final, verification, output = map(
        lambda p: Path(p).resolve(), (training_run, external_initial, external_final, verification, output))
    inputs = (training_run, external_initial, external_final, verification)
    require(all(output != p and p not in output.parents and output not in p.parents for p in inputs),
            'output must be separate from original evidence directories')
    require(not output.exists(), 'output already exists')
    training = _completed(training_run)
    input_hash = training['input_hash']
    admission = read(training_run / 'inputs/manifest.json')
    require(digest(training_run / 'inputs/manifest.json') == input_hash and admission['plan'] == training['plan'],
            'training admission binding')
    plan = training['plan']
    require(plan['seeds'] == list(SEEDS) and set(training['seeds']) == {str(s) for s in SEEDS}, 'training seed inventory')
    require(set(plan['stream_selected_rows']) == set(STREAMS), 'training stream plan')
    for name in ('cohorts_per_seed', 'selected_rows_per_cohort', 'minibatch_rows', 'epochs', 'optimizer_steps_per_cohort'):
        integer(plan[name], 'plan/' + name, 1)
    for rows in plan['stream_selected_rows'].values():
        integer(rows, 'stream row budget', 1)
    samples = plan['sampled_update_cohorts']
    require(len(samples) > 0 and len(set(samples)) == len(samples)
            and all(type(c) is int and 0 <= c < plan['cohorts_per_seed'] for c in samples), 'sampled cohort inventory')
    require(set(plan['evaluation_pairs']) == {'tactical', 'base_power', 'random'}, 'local opponent inventory')
    require(all(source.get(k) is False for source in (training, plan)
                for k in ('public_server_used', 'final_reserved_data_used')), 'local-only data role')
    require(sum(plan['stream_selected_rows'].values()) == plan['selected_rows_per_cohort'], 'cohort row budget')
    require(plan['selected_rows_per_cohort'] % plan['minibatch_rows'] == 0 and
            plan['optimizer_steps_per_cohort'] == plan['epochs'] * (plan['selected_rows_per_cohort'] // plan['minibatch_rows']),
            'optimizer budget')
    require(all(type(n) is int and n > 0 and n % 4 == 0 for n in plan['evaluation_pairs'].values()), 'local pair chunks')
    cohorts = len(SEEDS) * plan['cohorts_per_seed']
    expected_counts = {'cohorts': cohorts, 'optimizer_steps': cohorts * plan['optimizer_steps_per_cohort'],
        'selected_rows': cohorts * plan['selected_rows_per_cohort'],
        'row_presentations': cohorts * plan['selected_rows_per_cohort'] * plan['epochs'],
        'evaluation_games': len(SEEDS) * 4 * sum(plan['evaluation_pairs'].values()),
        'checkpoint_comparisons': len(SEEDS) * (2 + sum(c > 0 for c in samples)),
        'original_next_update_recoveries': len(SEEDS) * len(samples)}
    require(plan['expected_counts'] == expected_counts and
            all(training['counts'].get(k) == v for k, v in expected_counts.items()), 'training fixed counts')
    externals = {endpoint: _completed(folder, input_hash) for endpoint, folder in
                 [('initial', external_initial), ('final', external_final)]}
    a, b = externals['initial'], externals['final']
    require(a['plan'] == b['plan'] and a['plan_sha256'] == b['plan_sha256'], 'external endpoint plans')
    external_plan = a['plan']
    require(external_plan['seeds'] == list(SEEDS), 'external planned seeds')
    integer(external_plan['pairs_per_seed'], 'external pairs', 1)
    require(external_plan['endpoints'] == list(ENDPOINTS)
            and external_plan['expected_games_per_endpoint'] == len(SEEDS) * 2 * external_plan['pairs_per_seed']
            and external_plan['expected_total_games'] == 2 * external_plan['expected_games_per_endpoint'],
            'external planned counts')
    require(all(external_plan.get(k) is False for k in ('public_server_used', 'final_reserved_data_used')),
            'external-only data role')
    require(external_plan['search_time_ms'] == 100 and external_plan['parallelism'] == 1
            and external_plan['search_threads'] == 1, 'prespecified external search budget')
    matching = [rel for rel, sha in admission['source_files'].items() if sha == a['plan_sha256']]
    require(bool(matching), 'external plan absent from admission')
    for rel in matching:
        frozen = training_run / 'inputs/source' / rel
        require(digest(frozen) == a['plan_sha256'] and read(frozen) == external_plan, 'external plan admission binding')
    for endpoint, report in externals.items():
        require(report['endpoint'] == endpoint and set(report['seeds']) == {str(s) for s in SEEDS}, 'external endpoint inventory')
    verified = _completed(verification, input_hash)
    verification_counts = _verify_coverage(verified, plan)
    verified_files = verified['input_files']
    required_files = [training_run / 'report.json', training_run / 'inputs/manifest.json',
                      external_initial / 'report.json', external_final / 'report.json']
    require(all(str(path) in verified_files for path in required_files), 'verification evidence paths')
    for path, expected in verified_files.items():
        require(digest(Path(path)) == expected, 'verification input digest: ' + path)
    result = {'schema_version': 'f34-measured-summary-v1', 'status': 'passed', 'input_hash': input_hash,
        'scope': 'Completed fixed development screen; all seeds retained; no public ladder strength claim.',
        'statistics': copy.deepcopy(plan['statistics']), 'seeds': {},
        'verification': {'counts': verification_counts, 'coverage': verified['coverage']},
        'sources': {k: str(v) for k, v in zip(('training', 'external_initial', 'external_final', 'verification'), inputs)}}
    totals = Counter()
    for seed in SEEDS:
        seed_report = training['seeds'][str(seed)]
        require(read(training_run / str(seed) / 'result.json') == seed_report and seed_report['seed'] == seed,
                'training seed report linkage')
        collected = _training(training_run, seed, plan, seed_report)
        totals.update(collected['totals'])
        panels = {endpoint: {opponent: _panel_games(training_run, seed, endpoint, opponent, pairs, plan)
                  for opponent, pairs in plan['evaluation_pairs'].items()} for endpoint in ENDPOINTS}
        for endpoint in ENDPOINTS:
            require({o: value_summary(gs) for o, gs in panels[endpoint].items()} == seed_report[endpoint],
                    'local endpoint summary linkage')
        analysis = learning_analysis(panels['initial'], panels['final'], plan['statistics'])
        require(analysis == seed_report['analysis'], 'local learning analysis linkage')
        local = {'checks': analysis['checks'], 'passed': analysis['passed'],
            'contrasts': {o: contrast(panels['initial'][o], panels['final'][o], plan['statistics'])
                          for o in plan['evaluation_pairs']}}
        external_games, external_summary = {}, {}
        for endpoint, folder in [('initial', external_initial), ('final', external_final)]:
            wanted_hash = seed_report['identity']['parameter_hash'] if endpoint == 'initial' else seed_report['final_parameter_hash']
            games, summary = _external(folder, seed, endpoint, externals[endpoint], wanted_hash)
            external_games[endpoint], external_summary[endpoint] = games, summary
        external_summary['contrast'] = contrast(external_games['initial'], external_games['final'], plan['statistics'])
        regressions = [o for o, c in local['contrasts'].items() if c['gate'] == 'regression']
        if external_summary['contrast']['gate'] == 'regression':
            regressions.append('foul_play')
        result['seeds'][str(seed)] = {'training': collected, 'local': local,
            'external': external_summary, 'detected_regressions': regressions}
    require(totals['games'] == training['counts']['training_games'], 'total training games')
    require(totals['selected_rows'] == training['counts']['selected_rows'], 'total selected rows')
    result['training_totals'] = dict(totals)
    result['all_seed_learning_passed'] = all(s['local']['passed'] for s in result['seeds'].values())
    require(result['all_seed_learning_passed'] == training['all_seed_learning_passed'], 'all-seed learning gate')
    timings = {key: finite(training[key], key) for key in ('elapsed_seconds', 'collection_seconds',
               'update_seconds_including_compile', 'sampled_replay_update_seconds', 'peak_parent_rss_mib')}
    require(all(v >= 0 for v in timings.values()) and timings['elapsed_seconds'] > 0, 'negative training cost')
    profile_collection = sum(finite(p['collection_seconds'], 'profile collection') for p in training['profiles'])
    require(math.isclose(profile_collection, timings['collection_seconds'], rel_tol=1e-10, abs_tol=1e-7), 'collection profile linkage')
    ext_initial = finite(a['elapsed_seconds'], 'initial external time')
    ext_final = finite(b['elapsed_seconds'], 'final external time')
    verify_seconds = finite(verified['elapsed_seconds'], 'verification time')
    require(min(ext_initial, ext_final, verify_seconds) >= 0, 'negative component time')
    result['costs'] = {'training_elapsed_seconds': timings['elapsed_seconds'],
        'collection_seconds_including_local_evaluation': timings['collection_seconds'],
        'update_seconds_including_compile': timings['update_seconds_including_compile'],
        'sampled_replay_update_seconds': timings['sampled_replay_update_seconds'],
        'external_initial_seconds': ext_initial, 'external_final_seconds': ext_final,
        'verification_seconds': verify_seconds,
        'component_elapsed_seconds_sum': timings['elapsed_seconds'] + ext_initial + ext_final + verify_seconds,
        'selected_rows_per_training_run_hour': totals['selected_rows'] * 3600 / timings['elapsed_seconds'],
        'peak_parent_rss_mib': timings['peak_parent_rss_mib'],
        'memory_scope': 'Training parent process high-water RSS only; excludes children and GPU/total-machine memory.',
        'collection_scope': 'Training plus local endpoint evaluation; includes simulation, encoding, inference, transport and bookkeeping.'}
    result['limitations'] = [
        'Representation, additional training, optimizer reset and opponent mixture changed together. This evaluates the candidate package and does not isolate the effect of architecture or history.',
        'Tactical is a training opponent. Foul Play is held out from training during this screen, but uses an approximate search engine and a fixed small search budget. No local rate is converted into ladder Elo.',
        'Foul Play Python sampling is seeded; native Rust MCTS is unseeded and wall-clock stopped. Paired teams and seats do not imply identical opponent actions across endpoints. External verification replays recorded actions and does not reproduce search.',
        'Intervals are the prespecified approximate family-adjusted paired-cluster bootstrap, with bounded Hoeffding intervals also retained in summary.json. Inconclusive comparisons do not establish no effect.',
        'Terminal value MSE averages games equally and excludes cutoff games. Final reserved data and public servers were not used; this is not a held-out-team or public-strength certificate.']
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'summary.json', result)
    (output / 'results.md').write_text(_render(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('training-run', 'external-initial', 'external-final', 'verification', 'output'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.training_run, args.external_initial, args.external_final, args.verification, args.output)
    print({'status': summary['status'], 'all_seed_learning_passed': summary['all_seed_learning_passed']})
