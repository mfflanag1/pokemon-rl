"""Acquire the complete frozen F35 train/probe teacher corpus exactly once."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, digest
from pokemon_structured_policy.actor import batched_actor
from .external import run_game


def schedule(plan):
    from .common import teacher_specs
    result = []
    for role in ('train', 'probe'):
        for seed in plan['seeds']:
            for spec in teacher_specs(plan, seed, role):
                game_id = f"f2:{spec['split']}:{spec['run_seed']}:{spec['episode_index']}:{'swap' if spec['swap'] else 'base'}"
                result.append({'game_id': game_id, 'host_seed': seed, 'role': role, **spec})
    if len({row['game_id'] for row in result}) != len(result):
        raise ValueError('Teacher schedule game identity collision')
    return result


def admit_game(record, spec, quarantine):
    if (record['status'] != 'completed' or record['game_id'] != spec['game_id']
            or record['split'] != spec['split'] or record['external_side'] != spec['external_side']
            or record['outcome']['kind'] not in ('terminal', 'administrative_cutoff')):
        raise ValueError('Teacher game completion/schedule mismatch')
    provenance = record['provenance']
    if (provenance['split'] != spec['split'] or provenance['swap'] != spec['swap']
            or len(provenance['teams']) != 2
            or any(t['partition'] != spec['split'] or t['team_key'] in quarantine for t in provenance['teams'])):
        raise ValueError('Teacher game partition/quarantine violation')
    if record['accepted_labels'] < 1:
        raise ValueError('Teacher scheduled game has no accepted labels; game cannot be omitted')


def acquire(output, plan_path, admission):
    from .common import before_deadline, read, verify_inputs, load_initial
    before_deadline()
    output, plan_path, admission = Path(output), Path(plan_path), Path(admission)
    frozen = verify_inputs(admission)
    plan, plan_hash, input_hash = read(plan_path), digest(plan_path), digest(admission / 'manifest.json')
    if plan != frozen['plan'] or frozen['source_files'].get(str(plan_path.resolve().relative_to(ROOT))) != plan_hash:
        raise ValueError('Teacher plan does not match source admission')
    scheduled = schedule(plan)
    output.mkdir(parents=True, exist_ok=False)
    started, start_epoch = time.perf_counter(), time.time()
    deadline = min(start_epoch + plan['teacher_data']['acquisition_cap_seconds'],
                   datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')).timestamp())
    games, identities, current = [], {}, None
    atomic_json(output / 'schedule.json', {'schema_version': 'f35-teacher-data-schedule-v1',
        'input_hash': input_hash, 'plan_sha256': plan_hash, 'schedule': scheduled,
        'started_at': datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
        'acquisition_deadline_epoch': deadline})
    try:
        hosts = {}
        for seed in plan['seeds']:
            parameters, _optimizer, catalog, _frozen, identity = load_initial(seed)
            hosts[seed] = parameters, catalog
            identities[str(seed)] = identity
        for current in scheduled:
            before_deadline()
            if time.time() >= deadline:
                raise TimeoutError('Fixed 75-minute teacher acquisition cap reached')
            seed, role = current['host_seed'], current['role']
            parameters, catalog = hosts[seed]
            learner_side = 'p1' if current['external_side'] == 'p2' else 'p2'
            actor_seed = derived_seed('f35-teacher-host-actor-v1', role, seed)
            action_seed = derived_seed('actor', actor_seed, current['episode_index'], current['swap'], learner_side)
            decisions, phases = [], Counter()
            def decide(player, _index, history):
                out = batched_actor(parameters, [player], [history], [action_seed], [len(decisions)], 2, catalog)[0]
                decisions.append({'request_id': player['request_id'], 'action': out[2],
                                  'log_probability': out[3], 'value': out[4]})
                phases[player['phase']] += 1
                return out[2]
            directory = f"{role}/{seed}/{current['episode_index']:03d}-{int(current['swap'])}"
            folder = output / directory
            args = {key: current[key] for key in ('run_seed', 'episode_index', 'swap', 'external_side', 'split', 'fp_seed')}
            record = run_game(folder, **args, search_time_ms=plan['teacher_data']['search_time_ms'],
                max_turns=plan['max_turns'], learner=decide, learner_uses_history=True,
                purpose=f'Preregistered F35 teacher-executed {role} acquisition against frozen F34 host {seed}',
                capture_demonstrations=True, deadline_epoch=deadline)
            admit_game(record, current, set(plan['quarantined_team_keys']))
            atomic_json(folder / 'learner.json', {'parameter_hash': parameter_hash(parameters),
                'action_seed': action_seed, 'decisions': decisions, 'phases': dict(phases),
                'scope': 'Frozen opposing host decisions; these are not demonstration labels or training transitions.'})
            with np.load(folder / 'demonstrations.npz', allow_pickle=False) as data:
                actions = data['actions']
                coverage = {'ordinary_moves': int(np.sum(actions < 4)),
                    'tera_moves': int(np.sum((actions >= 4) & (actions < 8))),
                    'switches': int(np.sum(actions >= 8)),
                    'by_action': {str(i): int(np.sum(actions == i)) for i in range(14)}}
            games.append({**current, 'directory': directory, 'provenance': record['provenance'],
                'outcome': record['outcome'], 'accepted_labels': record['accepted_labels'],
                'decisions': record['decisions'], 'foul_play_decisions': record['foul_play_decisions'],
                'rejected_actions': record['rejected_actions'], 'illegal_actions': record['illegal_actions'],
                'seconds': record['seconds'], 'foul_play_seconds': record['total_foul_play_seconds'],
                'coverage': coverage, 'files': {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}})
            atomic_json(output / 'progress.json', {'stage': 'teacher acquisition', 'completed_games': len(games),
                'scheduled_games': len(scheduled), 'completed_labels': sum(g['accepted_labels'] for g in games),
                'last_game': current, 'elapsed_seconds': time.perf_counter() - started})
        if digest(plan_path) != plan_hash or digest(admission / 'manifest.json') != input_hash:
            raise ValueError('Teacher input changed during acquisition')
        verify_inputs(admission)
        report = {'schema_version': 'f35-teacher-data-v1', 'status': 'passed', 'input_hash': input_hash,
            'plan_sha256': plan_hash, 'schedule': scheduled, 'games': games, 'host_identities': identities,
            'elapsed_seconds': time.perf_counter() - started,
            'roles': {role: {'games': sum(g['role'] == role for g in games),
                'accepted_labels': sum(g['accepted_labels'] for g in games if g['role'] == role),
                'terminal_games': sum(g['role'] == role and g['outcome']['kind'] == 'terminal' for g in games),
                'cutoff_games': sum(g['role'] == role and g['outcome']['kind'] == 'administrative_cutoff' for g in games)}
                for role in ('train', 'probe')},
            'scope': 'Fresh teacher-executed accepted-action supervision; shared corpus, no PPO targets; probe excluded from training.'}
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'schema_version': 'f35-teacher-data-failure-v1',
            'status': 'failed', 'error': repr(exc), 'input_hash': input_hash, 'plan_sha256': plan_hash,
            'schedule': scheduled, 'games': games, 'inflight_game': current,
            'elapsed_seconds': time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--admission', required=True, type=Path)
    args = parser.parse_args()
    report = acquire(args.output, args.plan, args.admission)
    print({'status': report['status'], 'games': len(report['games']), 'seconds': report['elapsed_seconds']}, flush=True)


if __name__ == '__main__':
    main()
