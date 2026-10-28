"""Independent F39 ledger accounting and fixed, cutoff-aware comparisons.

All seeds and twelve contrasts are retained. Regression flags do not redefine
the three baseline gates, and no result authorizes promotion or extra training.
"""
import argparse
from collections import Counter
import math
from pathlib import Path

import numpy as np

from pokemon_full_learning.statistics import compare
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import atomic_json, digest
from .common import SEEDS, deadline, evaluation_specs, read, verify_inputs

STREAMS = {'self_play': None, 'frozen_f32': 'frozen_f32', 'tactical': 'tactical'}
OPPONENTS = ('baseline-v1', 'tactical', 'base_power', 'random')
ENDPOINTS = ('initial', 'final')


def require(condition, label):
    if not condition:
        raise ValueError('F39 reporting: ' + label)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return float(value)


def measurements(values):
    a = np.asarray(values, np.float64)
    require(a.ndim == 1 and np.isfinite(a).all(), 'nonfinite measurements')
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
        'median': float(np.median(a)) if len(a) else None,
        'p95': float(np.quantile(a, .95)) if len(a) else None,
        'min': float(a.min()) if len(a) else None, 'max': float(a.max()) if len(a) else None,
        'sum': float(a.sum())}


def outcome_summary(games):
    for game in games:
        kind = game['outcome']['kind']
        require(kind in ('terminal', 'administrative_cutoff'), 'unfinished game')
        integer(game['decisions'], 'game decisions', 1)
        require(0 <= finite(game['value_mean'], 'predicted value') <= 1, 'predicted value range')
        if kind == 'terminal':
            require(0 <= finite(game['terminal_value_mse'], 'terminal value MSE') <= 1, 'value MSE range')
            rewards = game['outcome']['rewards']
            require(set(rewards) == {'p1', 'p2'} and set(rewards.values()) <= {0, .5, 1}
                    and sum(rewards.values()) == 1, 'terminal rewards')
        else:
            require(game['terminal_value_mse'] is None and game['outcome']['rewards'] is None,
                    'cutoff is not a terminal target or draw')
        if 'actor_trace' in game:
            values = [r['value'] for r in game['actor_trace']]
            require(values and all(0 <= finite(v, 'actor value') <= 1 for v in values), 'actor value inventory')
            require(float(np.mean(values)) == game['value_mean'], 'actor/value mean linkage')
            if kind == 'terminal':
                mse = float(np.mean([(r['value'] - game['outcome']['rewards'][r['side']]) ** 2
                                     for r in game['actor_trace']]))
                require(mse == game['terminal_value_mse'], 'actor/terminal value MSE linkage')
    summary = value_summary(games)
    del summary['game_mean_predicted_value']
    terminal = [g for g in games if g['outcome']['kind'] == 'terminal']
    summary['terminal_game_mean_predicted_value'] = value_summary(terminal)['game_mean_predicted_value']
    summary['cutoff_rate'] = summary['cutoffs'] / len(games) if games else None
    return summary


def gate(contrast, settings):
    checks = {
        'cutoff_aware_lower_point_gain_at_least_five_points':
            contrast['cutoff_aware_point_interval'][0] >= settings['practical_improvement'],
        'adjusted_lower_positive': contrast['approximate_familywise_bootstrap_interval'][0] > 0,
        'final_cutoff_rate_admitted': contrast['final']['cutoffs'] / contrast['final']['games']
            <= settings['max_final_cutoff_rate']}
    return {'checks': checks, 'passed': all(checks.values())}


def analyze(initial, final, settings):
    require(set(initial) == set(final) == set(OPPONENTS), 'endpoint opponent inventory')
    require(settings['family_cells'] == 12, 'single twelve-contrast multiplicity family')
    contrasts = {o: compare(initial[o], final[o], settings) for o in OPPONENTS}
    for opponent, contrast in contrasts.items():
        contrast['initial'] = outcome_summary(initial[opponent])
        contrast['final'] = outcome_summary(final[opponent])
    regressions = [o for o in OPPONENTS if contrasts[o]['gate'] == 'regression']
    return {'contrasts': contrasts, 'primary_gate': gate(contrasts['baseline-v1'], settings),
        'detected_regressions': regressions, 'any_detected_regression': bool(regressions)}


def rejection_accounting(games, selection=None):
    """Recompute rejected-row attribution from each complete game's local offset."""
    counts, offset = Counter(rejected_attempts=0, rejected_controlled_attempts=0, rejected_selected_rows=0), 0
    selected = set(selection['selected_indices']) if selection is not None else set()
    for game in games:
        decisions = integer(game['decisions'], 'game decisions', 1)
        eligible = integer(game['eligible_learner_rows'], 'eligible learner rows')
        accepted = integer(game['accepted_decisions'], 'accepted decisions')
        controlled = integer(game['accepted_controlled_decisions'], 'accepted controlled decisions')
        attempts = game['rejected_attempts']
        require(len(attempts) == integer(game['unavailable_choices'], 'unavailable choices'),
                'rejection attempt count')
        require(accepted + len(attempts) == decisions and controlled <= accepted,
                'accepted/rejected decision linkage')
        training = game['split'] == 'train'
        rejected_controlled = 0
        seen_requests, seen_rows = set(), set()
        for attempt in attempts:
            side, request, row = attempt['side'], attempt['request_id'], attempt['learner_row_index']
            require(side in ('p1', 'p2') and type(attempt['controlled']) is bool,
                    'rejection side/control')
            is_controlled = game['opponent'] is None or side == game['learner_side']
            require(attempt['controlled'] is is_controlled, 'rejection controlled-side linkage')
            integer(request, 'rejection request')
            require((side, request) not in seen_requests, 'duplicate rejected request')
            seen_requests.add((side, request))
            integer(attempt['actor_counter'], 'rejection actor counter')
            require(type(attempt['action']) is int and 0 <= attempt['action'] < 14,
                    'rejected action range')
            require(integer(attempt['refreshed_request_id'], 'refreshed request') > request,
                    'rejection request refresh')
            mask = attempt['refreshed_mask']
            require(len(mask) == 14 and all(type(x) is bool for x in mask) and any(mask),
                    'refreshed actionable mask')
            feedback = attempt['feedback']
            require(isinstance(feedback, list) and feedback and all(isinstance(x, str)
                    and x.startswith('[Unavailable choice]') for x in feedback), 'rejection feedback')
            if training and is_controlled:
                require(type(row) is int and 0 <= row < eligible and row not in seen_rows,
                        'rejected learner-row linkage')
                seen_rows.add(row)
            else:
                require(row is None, 'nonlearner rejection has a learner row')
            wanted = offset + row in selected if row is not None else False
            if training:
                require(selection is not None and attempt['selected_for_update'] is wanted,
                        'rejected selected-row linkage')
            else:
                require(attempt['selected_for_update'] is None, 'evaluation rejection selection')
            counts['rejected_attempts'] += 1
            counts['rejected_controlled_attempts'] += int(is_controlled)
            counts['rejected_selected_rows'] += int(wanted)
            rejected_controlled += int(is_controlled)
        require(eligible == (controlled + rejected_controlled if training else 0),
                'eligible learner-row/controlled-attempt linkage')
        if game['opponent'] is None:
            require(controlled == accepted and eligible == decisions, 'self-play row accounting')
        offset += eligible
    if selection is not None:
        require(offset == selection['collected_rows'], 'complete-game collected row accounting')
    return dict(counts)


def local_panels(base, seed, endpoint, plan, reader=read):
    folder = base / str(seed) / endpoint
    wanted_paths = {folder / f'{o}-{first:03d}.json' for o, pairs in plan['evaluation_pairs'].items()
                    for first in range(0, pairs, 4)}
    require(set(folder.glob('*.json')) == wanted_paths | {folder / 'summary.json'},
            'evaluation chunk inventory')
    panels = {}
    for opponent, pairs in plan['evaluation_pairs'].items():
        games = []
        for first in range(0, pairs, 4):
            chunk = reader(folder / f'{opponent}-{first:03d}.json')
            wanted = evaluation_specs(plan, seed, opponent, first, min(4, pairs - first))
            require(chunk['specs'] == wanted and len(chunk['games']) == len(wanted), 'evaluation schedule/count')
            for game, spec in zip(chunk['games'], wanted):
                require(all(game[k] == spec[k] for k in ('episode', 'swap', 'learner_side', 'opponent', 'split')),
                        'evaluation game coordinates')
            rejection_accounting(chunk['games'])
            outcome_summary(chunk['games'])
            games.extend(chunk['games'])
        require(len(games) == 2 * pairs, 'evaluation panel count')
        panels[opponent] = games
    require(reader(folder / 'summary.json') == {o: value_summary(gs) for o, gs in panels.items()},
            'evaluation summary linkage')
    return panels


def update_accounting(base, seed, plan, seed_report, reader=read):
    first, end = plan['initial_cohorts_per_seed'], plan['final_cohorts_per_seed']
    expected = [base / str(seed) / f'update-{i:04d}.json' for i in range(first, end)]
    require(sorted((base / str(seed)).glob('update-*.json')) == expected, 'update inventory')
    cursors = dict(seed_report['initial_stream_cursors'])
    require(set(cursors) == set(STREAMS), 'initial stream cursor inventory')
    require(cursors == seed_report['identity']['f34_metadata']['stream_cursors'], 'inherited cursor identity')
    by_stream, simple_streams = {s: Counter() for s in STREAMS}, {s: Counter() for s in STREAMS}
    first_metrics, last_metrics, all_games = [], [], []
    counts = Counter()
    optimizer_seconds = recovery_seconds = 0.
    previous_parameter, previous_state = seed_report['initial_parameter_hash'], seed_report['initial_state_hash']
    per_epoch = plan['selected_rows_per_cohort'] // plan['minibatch_rows']
    for cohort, path in zip(range(first, end), expected):
        update = reader(path)
        require(update['cohort'] == cohort and update['optimizer_step'] == (cohort + 1) * 8,
                'absolute update coordinates')
        require(update['optimizer_seed'] == derived_seed('f39-optimizer-v1', seed, cohort), 'optimizer schedule')
        require(update['parameter_hash_before'] == previous_parameter and update['state_hash_before'] == previous_state,
                'parameter/Adam state chain')
        previous_parameter, previous_state = update['parameter_hash_after'], update['state_hash_after']
        sampled = cohort in plan['sampled_update_cohorts']
        require(update['sampled_update_verified'] is sampled, 'original recovery coverage')
        update_time = finite(update['update_seconds_including_compile'], 'update measured seconds')
        recovery_time = finite(update['sampled_replay_update_seconds'], 'recovery measured seconds')
        require(update_time >= 0 and recovery_time >= 0 and (sampled or recovery_time == 0), 'update cost scope')
        optimizer_seconds += update_time
        recovery_seconds += recovery_time
        metrics, ledger = update['metrics'], update['ledger']
        require(len(metrics) == plan['optimizer_steps_per_cohort'], 'optimizer metric count')
        for metric in metrics:
            require(set(metric) == {'loss', 'actor_loss', 'value_loss', 'entropy', 'approx_kl',
                    'clip_fraction', 'gradient_norm'}, 'original PPO metric inventory')
            for key, value in metric.items():
                finite(value, 'optimizer metric/' + key)
            require(0 <= metric['clip_fraction'] <= 1 and metric['gradient_norm'] >= 0, 'optimizer metric range')
        first_metrics.append(metrics[0])
        last_metrics.extend(metrics[(plan['epochs'] - 1) * per_epoch:])
        require(len(ledger['strata']) == 3, 'stratum count')
        require([s['stream'] for s in ledger['strata']] == list(STREAMS), 'stratum order/inventory')
        used = 0
        for stratum in ledger['strata']:
            label, selected = stratum['stream'], stratum['selection']
            n = integer(stratum['games'], 'stratum games', 1)
            require(stratum['opponent'] == STREAMS[label] and stratum['first_episode'] == cursors[label]
                    and stratum['next_episode'] == cursors[label] + n, 'stratum cursor/opponent')
            collected = integer(selected['collected_rows'], 'collected rows', 1)
            chosen = integer(selected['selected_rows'], 'selected rows', 1)
            excluded = integer(selected['excluded_rows'], 'excluded rows')
            require(chosen == plan['stream_selected_rows'][label] and collected == chosen + excluded,
                    'selected/excluded row accounting')
            indices = selected['selected_indices']
            require(len(indices) == chosen and len(set(indices)) == chosen and all(type(i) is int
                    and 0 <= i < collected for i in indices), 'selected row inventory')
            require(stratum['selection_seed'] == derived_seed('f39-row-selection-v1', seed, label), 'row selection seed')
            wanted_indices = np.random.default_rng(derived_seed('f9-row-selection',
                derived_seed('f39-row-selection-v1', seed, label), cohort)).choice(collected, chosen, replace=False).tolist()
            require(indices == wanted_indices, 'fixed row selection schedule')
            games = ledger['games'][used:used + n]
            require(len(games) == n, 'missing stratum games')
            for index, game in enumerate(games):
                episode = cursors[label] + index
                require(game['episode'] == episode and game['split'] == 'train' and game['swap'] is False
                    and game['opponent'] == STREAMS[label]
                    and game['learner_side'] == ('p2' if episode % 2 else 'p1'), 'training game schedule')
            rejected = rejection_accounting(games, selected)
            require(rejected == stratum['rejection_selection'], 'rejection aggregate linkage')
            summary = outcome_summary(games)
            by_stream[label].update(games=n, decisions=summary['decisions'], terminals=summary['terminals'],
                cutoffs=summary['cutoffs'], true_ties=summary['true_ties'], collected_rows=collected,
                selected_rows=chosen, excluded_rows=excluded,
                non_learner_decisions=summary['decisions'] - collected)
            by_stream[label].update(rejected)
            simple_streams[label].update(games=n, collected_rows=collected, selected_rows=chosen, excluded_rows=excluded)
            simple_streams[label].update(rejected)
            used += n
            cursors[label] += n
            all_games.extend(games)
        require(used == len(ledger['games']), 'unassigned training games')
        counts.update(cohorts=1, optimizer_steps=len(metrics), selected_rows=plan['selected_rows_per_cohort'],
            row_presentations=plan['selected_rows_per_cohort'] * plan['epochs'], training_games=used)
    require(previous_parameter == seed_report['final_parameter_hash'] and previous_state == seed_report['final_state_hash'],
            'final parameter/Adam identity')
    require(cursors == seed_report['stream_cursors'], 'final stream cursors')
    require(seed_report['initial_optimizer_step'] == first * 8
            and seed_report['final_optimizer_step'] == seed_report['optimizer_steps'] == end * 8,
            'cumulative optimizer counters')
    require(seed_report['continuation_optimizer_steps'] == (end - first) * 8
            and seed_report['cumulative_selected_rows'] == end * plan['selected_rows_per_cohort']
            and seed_report['cumulative_row_presentations'] == end * plan['selected_rows_per_cohort'] * plan['epochs'],
            'declared cumulative row/update counters')
    require(seed_report['training_unavailable_choices'] == sum(g['unavailable_choices'] for g in all_games),
            'training rejection summary linkage')
    require(value_summary(all_games) == seed_report['training'], 'training summary linkage')
    require({s: dict(v) for s, v in simple_streams.items()} == seed_report['stream_counts'], 'stream summary linkage')
    totals = Counter()
    for value in by_stream.values():
        totals.update(value)
    return {'counts': dict(counts), 'update_seconds_including_compile': optimizer_seconds,
        'sampled_replay_update_seconds': recovery_seconds, 'streams': {s: dict(v) for s, v in by_stream.items()},
        'totals': dict(totals), 'outcomes': outcome_summary(all_games),
        'optimizer_diagnostics': {'first_minibatch_per_cohort': {k: measurements([m[k] for m in first_metrics])
            for k in ('approx_kl', 'clip_fraction')}, 'last_epoch_minibatches': {k: measurements([m[k] for m in last_metrics])
            for k in ('approx_kl', 'clip_fraction')},
            'scope': 'Pre-update minibatch statistics; final-epoch means are not full-batch post-update KL. Descriptive only.'}}


def verify_coverage(verified, plan):
    expected = {'training_games': 18, 'evaluation_games': 48, 'game_executions': 66,
        'sampled_updates': 6, 'optimizer_steps_recomputed': 48, 'checkpoint_comparisons': 9}
    require(verified['counts'] == verified['expected_counts'] == expected, 'verification counts')
    require({r['seed'] for r in verified['records']} == set(SEEDS), 'verification seeds')
    wanted = {'fresh_training': {(c, s) for c in (512, 767) for s in STREAMS},
        'fresh_update': {(c,) for c in (512, 767)},
        'fresh_evaluation_pair': {(e, o) for e in ENDPOINTS for o in OPPONENTS},
        'checkpoint': {(c,) for c in (512, 767, 768)}}
    fields = {'fresh_training': ('cohort', 'stream'), 'fresh_update': ('cohort',),
        'fresh_evaluation_pair': ('endpoint', 'opponent'), 'checkpoint': ('cohort',)}
    for seed in SEEDS:
        records = [r for r in verified['records'] if r['seed'] == seed]
        require(Counter(r['kind'] for r in records) == {k: len(v) for k, v in wanted.items()}, 'verification record counts')
        for kind, keys in wanted.items():
            require({tuple(r[k] for k in fields[kind]) for r in records if r['kind'] == kind} == keys,
                    'verification coverage coordinates')
    scope = verified['coverage']
    modules = scope['teacher_search_module_inventory']
    require(scope['exhaustive'] is False and modules['before'] == modules['after'] == []
            and isinstance(modules['forbidden_prefixes'], list) and modules['forbidden_prefixes']
            and all(isinstance(p, str) and p for p in modules['forbidden_prefixes'])
            and isinstance(scope['teacher_search_execution_scope'], str)
            and scope['teacher_search_execution_scope'], 'verification scope')
    return expected


def run(training_run, verification, output):
    training_run, verification, output = map(lambda p: Path(p).resolve(), (training_run, verification, output))
    require(all(output != p and p not in output.parents and output not in p.parents
                for p in (training_run, verification)), 'output must be separate from original evidence')
    require(not output.exists(), 'output already exists')
    inventory = {}
    def bound(path):
        path = Path(path).resolve()
        observed = digest(path)
        require(str(path) not in inventory or inventory[str(path)] == observed, 'input changed during read')
        inventory[str(path)] = observed
        value = read(path)
        require(digest(path) == observed, 'input changed during read')
        return value
    manifest = bound(training_run / 'inputs/manifest.json')
    plan, input_hash = manifest['plan'], digest(training_run / 'inputs/manifest.json')
    deadline(plan)
    admitted = verify_inputs(training_run / 'inputs')
    require(admitted['plan'] == plan, 'input admission plan')
    training, verified = bound(training_run / 'report.json'), bound(verification / 'report.json')
    require(all(r.get('status') == 'passed' and r.get('input_hash') == input_hash
                for r in (training, verified)), 'complete report/input identity')
    require(training['plan'] == plan and set(training['seeds']) == {str(s) for s in SEEDS}, 'all-seed plan inventory')
    verification_counts = verify_coverage(verified, plan)
    required_files = [training_run / 'report.json', training_run / 'inputs/manifest.json']
    require(all(str(p) in verified['input_files'] for p in required_files), 'verification evidence paths')
    for path, expected in verified['input_files'].items():
        require(digest(path) == expected, 'verification input digest: ' + path)
    totals, counts, seeds = Counter(), Counter(), {}
    for seed in SEEDS:
        deadline(plan)
        row = bound(training_run / str(seed) / 'result.json')
        require(row == training['seeds'][str(seed)] and row['seed'] == seed, 'seed report linkage')
        collected = update_accounting(training_run, seed, plan, row, bound)
        totals.update(collected['totals']); counts.update(collected['counts'])
        panels = {e: local_panels(training_run, seed, e, plan, bound) for e in ENDPOINTS}
        for endpoint in ENDPOINTS:
            require({o: value_summary(gs) for o, gs in panels[endpoint].items()} == row[endpoint],
                    'endpoint summary linkage')
            counts['evaluation_games'] += sum(map(len, panels[endpoint].values()))
        analysis = analyze(panels['initial'], panels['final'], plan['statistics'])
        require(analysis == row['analysis'], 'analysis linkage')
        seeds[str(seed)] = {'training': collected, 'analysis': analysis,
            'endpoints': {e: {o: outcome_summary(gs) for o, gs in p.items()} for e, p in panels.items()}}
    counts.update(checkpoint_comparisons=9, original_next_update_recoveries=6)
    require(dict(counts) == training['counts'], 'complete campaign counts')
    require(all(counts[k] == v for k, v in plan['expected_counts'].items()), 'fixed budget totals')
    primary = all(s['analysis']['primary_gate']['passed'] for s in seeds.values())
    require(training['all_seed_learning_passed'] is primary, 'all-seed primary gate linkage')
    timings = {k: finite(training[k], k) for k in ('elapsed_seconds', 'collection_seconds',
        'update_seconds_including_compile', 'sampled_replay_update_seconds', 'peak_parent_rss_mib')}
    require(all(v >= 0 for v in timings.values()) and timings['elapsed_seconds'] > 0, 'nonpositive cost')
    collection = sum(finite(p['collection_seconds'], 'collection profile') for p in training['profiles'])
    require(all(p['collection_seconds'] >= 0 for p in training['profiles']) and
            math.isclose(collection, timings['collection_seconds'], rel_tol=1e-10, abs_tol=1e-7), 'collection cost linkage')
    for field in ('update_seconds_including_compile', 'sampled_replay_update_seconds'):
        require(math.isclose(sum(s['training'][field] for s in seeds.values()), timings[field],
                rel_tol=1e-10, abs_tol=1e-7), 'independent update cost linkage/' + field)
    verify_seconds = finite(verified['elapsed_seconds'], 'verification elapsed')
    require(verify_seconds >= 0, 'verification negative cost')
    for path, expected in inventory.items():
        require(digest(path) == expected, 'input changed before report')
    result = {'schema_version': 'f39-measured-summary-v1', 'status': 'passed', 'input_hash': input_hash,
        'statistics': plan['statistics'], 'counts': dict(counts), 'training_totals': dict(totals),
        'seeds': seeds, 'all_seed_primary_passed': primary,
        'any_detected_regression': any(s['analysis']['any_detected_regression'] for s in seeds.values()),
        'primary_contrasts': 3, 'regression_diagnostic_contrasts': 9, 'total_contrasts': 12,
        'regression_diagnostics': [{'seed': int(seed), 'opponent': opponent, **contrast}
            for seed, row in seeds.items() for opponent, contrast in row['analysis']['contrasts'].items()
            if opponent != 'baseline-v1'],
        'promotion_authorized': False, 'verification': {'counts': verification_counts, 'coverage': verified['coverage']},
        'input_files': inventory, 'costs': {**timings, 'verification_seconds': verify_seconds,
            'component_elapsed_seconds_sum': timings['elapsed_seconds'] + verify_seconds,
            'selected_rows_per_training_run_hour': counts['selected_rows'] * 3600 / timings['elapsed_seconds'],
            'memory_scope': 'Training parent high-water RSS only; excludes child processes and GPU/total machine memory.',
            'collection_scope': 'Training and both local evaluation endpoints; simulation, encoding, inference, transport and bookkeeping.'},
        'limitations': ['Fixed development screen with recurring local opponents/teams; no public or held-out-team strength claim.',
            'Baseline-v1 is excluded from training. Tactical and frozen F32 are training opponents; no teacher or demonstrations used.',
            'All twelve contrasts use the same approximate family-adjusted paired-seat-cluster bootstrap; bounded Hoeffding intervals are retained.',
            'Cutoffs retain unresolved score bounds. Value diagnostics average terminal games equally and exclude cutoff games.',
            'Failure of the primary conjunction does not establish no benefit. Absence of detected regression does not establish noninferiority.',
            'Independent replay is sampled, not exhaustive. No budget extension or promotion follows automatically.']}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'summary.json', result)
    lines = ['# F39 measured online continuation', '',
        f"Execution and verification passed. All three baseline primary gates passed: {primary}. Any detected regression: {result['any_detected_regression']}.", '',
        '| Seed | Opponent | Initial wins / games | Final wins / games | Cutoff-aware gain | Adjusted interval |',
        '|---|---|---:|---:|---|---|']
    for seed, row in seeds.items():
        for opponent, c in row['analysis']['contrasts'].items():
            a, b = c['initial'], c['final']
            lines.append(f"| {seed} | {opponent} | {a['learner_wins']}/{a['games']} | {b['learner_wins']}/{b['games']} | {c['cutoff_aware_point_interval']} | {c['approximate_familywise_bootstrap_interval']} |")
    lines += ['', *result['limitations'], '', 'All counts, cutoff rates, terminal-only value metrics, rejection attribution and measured costs appear in summary.json. Parent RSS is not total machine or GPU memory.']
    (output / 'results.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training-run', 'verification', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.training_run, args.verification, args.output)
    print({'status': result['status'], 'all_seed_primary_passed': result['all_seed_primary_passed']}, flush=True)
