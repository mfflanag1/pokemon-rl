"""Complete F35 accounting and paired treatment-control development estimates."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from pokemon_full_learning.statistics import compare
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.storage import ROOT, atomic_json, digest
from .common import ARMS, SEEDS, admit_games, evaluation_specs, read, verify_inputs


def local_panels(run, seed, endpoint, plan):
    base = run / str(seed) / ('initial' if endpoint == 'initial' else endpoint + '/final')
    panels = {}
    for opponent, pairs in plan['evaluation_pairs'].items():
        games = []
        for first in range(0, pairs, 4):
            chunk = read(base / f'{opponent}-{first:03d}.json')
            wanted = evaluation_specs(plan, seed, opponent, first, min(4, pairs - first))
            if chunk['specs'] != wanted or len(chunk['games']) != len(wanted):
                raise ValueError('F35 evaluation chunk schedule')
            admit_games(chunk['games'], 'development', set(plan['quarantined_team_keys']))
            games.extend(chunk['games'])
        panels[opponent] = games
    return panels


def gate(contrast, settings):
    checks = {'practical_gain': contrast['cutoff_aware_point_interval'][0] >= settings['practical_improvement'],
        'adjusted_lower_positive': contrast['approximate_familywise_bootstrap_interval'][0] > 0,
        'both_final_cutoffs_admitted': all(contrast[name]['cutoffs'] / contrast[name]['games']
            <= settings['max_final_cutoff_rate'] for name in ('initial', 'final'))}
    return {'checks': checks, 'passed': all(checks.values())}


def analyze(panels, settings):
    if set(panels) != {'initial', *ARMS} or any(set(v) != set(panels['initial']) for v in panels.values()):
        raise ValueError('F35 endpoint/opponent inventory')
    diagnostics = {**settings, 'family_cells': settings['diagnostic_family_cells']}
    contrasts, regressions = {}, []
    for opponent in panels['initial']:
        contrasts[opponent] = {}
        for name, before, after in [('control_vs_initial', 'initial', 'control'),
                                     ('assisted_vs_initial', 'initial', 'assisted'),
                                     ('assisted_vs_control', 'control', 'assisted')]:
            primary_family = name == 'assisted_vs_control' and opponent in ('baseline-v1', 'foul_play')
            result = compare(panels[before][opponent], panels[after][opponent], settings if primary_family else diagnostics)
            result['multiplicity_family'] = 'six primary/teacher contrasts' if primary_family else '39 diagnostic contrasts'
            result['before_endpoint'], result['after_endpoint'] = before, after
            contrasts[opponent][name] = result
            if result['gate'] == 'regression':
                regressions.append({'opponent': opponent, 'contrast': name,
                    'point_interval': result['cutoff_aware_point_interval'],
                    'adjusted_interval': result['approximate_familywise_bootstrap_interval']})
    return {'contrasts': contrasts,
        'primary_gate': gate(contrasts['baseline-v1']['assisted_vs_control'], settings),
        'teacher_adaptation_gate': gate(contrasts['foul_play']['assisted_vs_control'], settings),
        'detected_regressions': regressions,
        'limitation': 'Failure of the conjunction does not establish no benefit; absence of detected regression is not noninferiority.'}


def update_accounting(run, seed, arm, plan):
    counts, streams, first, last = Counter(), {}, [], []
    games = []
    for cohort in range(plan['cohorts_per_seed_arm']):
        update = read(run / str(seed) / arm / f'update-{cohort:04d}.json')
        if update['cohort'] != cohort or update['arm'] != arm or update['optimizer_step'] != (cohort + 1) * 8:
            raise ValueError('F35 update coordinates')
        metrics, ledger = update['metrics'], update['ledger']
        if len(metrics) != 8 or {s['stream'] for s in ledger['strata']} != set(plan['stream_selected_rows']):
            raise ValueError('F35 optimizer/stream inventory')
        if len(ledger['strata']) != 3:
            raise ValueError('Duplicate F35 stream')
        for stratum in ledger['strata']:
            selection, label = stratum['selection'], stratum['stream']
            if (selection['selected_rows'] != plan['stream_selected_rows'][label]
                    or selection['collected_rows'] - selection['selected_rows'] != selection['excluded_rows']
                    or len(selection['selected_indices']) != selection['selected_rows']
                    or len(set(selection['selected_indices'])) != selection['selected_rows']):
                raise ValueError('F35 fixed selection accounting')
            stream = streams.setdefault(label, Counter())
            stream.update({k: selection[k] for k in ('collected_rows', 'selected_rows', 'excluded_rows')})
            stream.update(stratum['rejection_selection'])
            stream['games'] += stratum['games']
        if arm == 'assisted' and any('demo_nll' not in m for m in metrics):
            raise ValueError('Missing demonstration-update metrics')
        if arm == 'control' and any('demo_nll' in m for m in metrics):
            raise ValueError('Control received demonstration loss')
        first.append(metrics[0])
        last.extend(metrics[4:])
        games.extend(ledger['games'])
        counts.update(cohorts=1, optimizer_steps=len(metrics), selected_rows=1024, row_presentations=2048)
    fields = ('approx_kl', 'clip_fraction')
    description = {'scope': 'Pre-update minibatch statistics; final-epoch averages are not full-batch post-update KL.',
        'first_minibatch_mean': {k: float(np.mean([m[k] for m in first])) for k in fields},
        'final_epoch_minibatch_mean': {k: float(np.mean([m[k] for m in last])) for k in fields}}
    return {'counts': dict(counts), 'streams': {k: dict(v) for k, v in streams.items()},
        'training': value_summary(games), 'training_by_stream': {label: value_summary([
            g for g in games if g['opponent'] == opponent]) for label, opponent in
            [('self_play', None), ('frozen_f32', 'frozen_f32'), ('tactical', 'tactical')]},
        'unavailable_choices': sum(g['unavailable_choices'] for g in games), 'optimizer_diagnostics': description}


def run(training_run, external_run, demonstrations, verification, output):
    training_run, external_run, demonstrations, verification, output = map(lambda p: Path(p).resolve(),
        (training_run, external_run, demonstrations, verification, output))
    admitted = verify_inputs(training_run / 'inputs')
    plan, input_hash = admitted['plan'], digest(training_run / 'inputs/manifest.json')
    training, external, teacher, verified = [read(p / 'report.json') for p in
        (training_run, external_run, demonstrations, verification)]
    if any(r.get('status') != 'passed' or r.get('input_hash') != input_hash for r in (training, external, teacher, verified)):
        raise ValueError('F35 complete report/input admission')
    if digest(demonstrations / 'report.json') != training['demonstration_manifest_sha256']:
        raise ValueError('F35 corpus identity drift')
    if (teacher['roles']['train']['games'] != 384 or teacher['roles']['probe']['games'] != 96
            or external['counts']['games'] != 1152 or training['counts']['evaluation_games'] != 6912):
        raise ValueError('F35 prescribed game totals')
    output.mkdir(parents=True, exist_ok=False)
    started, seeds, contrasts = time.perf_counter(), {}, 0
    try:
        for seed in SEEDS:
            panels = {endpoint: local_panels(training_run, seed, endpoint, plan) for endpoint in ('initial', *ARMS)}
            for endpoint in panels:
                ext = external['seeds'][str(seed)][endpoint]
                if len(ext['games']) != 128:
                    raise ValueError('Missing F35 teacher-adaptation cell')
                panels[endpoint]['foul_play'] = ext['games']
            analysis = analyze(panels, plan['statistics'])
            contrasts += sum(len(c) for c in analysis['contrasts'].values())
            cells = {arm: update_accounting(training_run, seed, arm, plan) for arm in ARMS}
            for arm, cell in cells.items():
                original = training['seeds'][str(seed)]['arms'][arm]
                if cell['training'] != original['training'] or cell['streams'] != original['stream_counts']:
                    raise ValueError('Independent F35 update accounting differs from campaign report')
            seeds[str(seed)] = {'analysis': analysis, 'accounting': cells,
                'endpoints': {e: {o: value_summary(gs) for o, gs in ps.items()} for e, ps in panels.items()},
                'teacher_local_probe_diagnostics': {'initial': training['seeds'][str(seed)]['initial_demonstration_probes'],
                    **{a: training['seeds'][str(seed)]['arms'][a]['demonstration_probes'] for a in ARMS}}}
        if contrasts != 45:
            raise ValueError('F35 complete contrast inventory')
        durations = {'teacher_acquisition': teacher['elapsed_seconds'], 'training_and_local_evaluation': training['elapsed_seconds'],
            'external_evaluation': external['elapsed_seconds'], 'verification': verified['elapsed_seconds'],
            'collection': training['collection_seconds'], 'optimizer_including_compile': training['update_seconds_including_compile'],
            'checkpoint_recovery_updates': training['sampled_replay_update_seconds'],
            'demonstration_probes': training['demonstration_probe_seconds']}
        scientific_wall = sum(durations[k] for k in ('teacher_acquisition', 'training_and_local_evaluation', 'external_evaluation'))
        result = {'status': 'passed', 'input_hash': input_hash, 'seeds': seeds, 'contrasts': contrasts,
            'all_seed_primary_passed': all(s['analysis']['primary_gate']['passed'] for s in seeds.values()),
            'all_seed_teacher_adaptation_passed': all(s['analysis']['teacher_adaptation_gate']['passed'] for s in seeds.values()),
            'counts': training['counts'], 'external_counts': external['counts'], 'teacher_roles': teacher['roles'],
            'seconds': durations, 'selected_ppo_rows_per_training_wall_hour': 1572864 / training['elapsed_seconds'] * 3600,
            'selected_ppo_rows_per_scientific_wall_hour': 1572864 / scientific_wall * 3600,
            'teacher_accepted_labels_per_acquisition_hour': sum(r['accepted_labels'] for r in teacher['roles'].values()) / teacher['elapsed_seconds'] * 3600,
            'peak_training_parent_rss_mib': training['peak_parent_rss_mib'], 'memory_scope': 'Training parent process peak only; worker/search memory excluded.',
            'extra_demonstration_presentations': 1572864,
            'verification': verified['counts'], 'claims': plan['claims'], 'elapsed_seconds': time.perf_counter() - started}
        atomic_json(output / 'summary.json', result)
        lines = ['# F35 measured development comparison', '',
            f"Execution and verification passed. All-seed primary gate: {result['all_seed_primary_passed']}; teacher-adaptation gate: {result['all_seed_teacher_adaptation_passed']}.", '',
            '| Seed | Opponent | Initial wins | Control wins | Assisted wins | Assisted − control bounds | Primary/teacher gate |',
            '|---|---|---:|---:|---:|---|---|']
        for seed, s in seeds.items():
            for opponent in ('baseline-v1', 'foul_play'):
                c = s['analysis']['contrasts'][opponent]['assisted_vs_control']
                scores = [s['endpoints'][e][opponent] for e in ('initial', *ARMS)]
                shown = [f"{v['learner_wins']}/{v['games']}" for v in scores]
                outcome = s['analysis']['primary_gate' if opponent == 'baseline-v1' else 'teacher_adaptation_gate']['passed']
                lines.append(f"| {seed} | {opponent} | {' | '.join(shown)} | {c['cutoff_aware_point_interval']} | {outcome} |")
        lines += ['', 'Intervals, cutoff counts, terminal-only value errors, all 45 contrasts and per-stream accounting are in summary.json. A failed conjunction is not proof of no benefit.', '',
            'Baseline-v1 is historically known and excluded from this intervention. Foul Play is trained on. This is a fixed development screen, not a public-strength or held-out-team result.']
        (output / 'results.md').write_text('\n'.join(lines) + '\n')
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'completed_seeds': list(seeds),
            'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training-run', 'external-run', 'demonstrations', 'verification', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    report = run(a.training_run, a.external_run, a.demonstrations, a.verification, a.output)
    print({'status': report['status'], 'all_seed_primary_passed': report['all_seed_primary_passed']}, flush=True)
