"""Fixed local Foul Play panels; no training and no adaptive game replacement."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
from pokemon_external_benchmark.engineering import run_game
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, digest
from pokemon_full_train.campaign import value_summary
from pokemon_nonlinear_policy.actor import batched_actor as host_actor
from .actor import batched_actor
from .checkpoint import load_f32, load
from .common import read, SEEDS, before_deadline, verify_inputs


def panel(run, plan_path, endpoint, training_run=None, admission_path=None):
    before_deadline()
    run, plan_path = Path(run), Path(plan_path)
    plan_hash = digest(plan_path)
    plan = read(plan_path)
    if admission_path is None:
        raise ValueError('External panel requires frozen source admission')
    admission_path = Path(admission_path)
    admission = verify_inputs(admission_path)
    input_hash = digest(admission_path / 'manifest.json')
    relative_plan = str(plan_path.resolve().relative_to(ROOT))
    if admission['source_files'].get(relative_plan) != plan_hash:
        raise ValueError('External plan is not in the frozen source admission')
    if plan['seeds'] != list(SEEDS) or endpoint not in ('initial', 'final'):
        raise ValueError('External panel seed/endpoint inventory')
    if endpoint == 'final' and training_run is None:
        raise ValueError('Final policy requires completed fixed training run')
    if endpoint == 'initial' and training_run is not None:
        raise ValueError('Initial external panel uses frozen F32 hosts')
    run.mkdir(parents=True, exist_ok=False)
    begun, summaries = time.perf_counter(), {}
    try:
        for seed in plan['seeds']:
            host, catalog, identity = load_f32(seed)
            if endpoint == 'initial':
                parameters = host
            else:
                training = Path(training_run)
                admission = verify_inputs(training / 'inputs')
                report = read(training / 'report.json')
                if report['status'] != 'passed' or report['input_hash'] != digest(training / 'inputs/manifest.json'):
                    raise ValueError('Final training completion admission')
                if report['input_hash'] != input_hash:
                    raise ValueError('External and training source admissions differ')
                cohorts = admission['plan']['cohorts_per_seed']
                parameters, optimizer, metadata = load(training / str(seed) / 'checkpoints' / str(cohorts),
                    {'input_hash': report['input_hash'], 'seed': seed, 'cohorts': cohorts})
                if int(optimizer[2]) != cohorts * 8:
                    raise ValueError('Final external checkpoint step')
                if parameter_hash(parameters) != report['seeds'][str(seed)]['final_parameter_hash']:
                    raise ValueError('Final external checkpoint hash')
            games = []
            for episode in range(plan['pairs_per_seed']):
                for swap in (False, True):
                    before_deadline()
                    side = 'p2' if swap else 'p1'
                    external_side = 'p1' if swap else 'p2'
                    actor_seed = derived_seed('f34-evaluation-actor-v1', seed, 'foul_play')
                    action_seed = derived_seed('actor', actor_seed, episode, swap, side)
                    values, actions, phases = [], [], Counter()
                    def decide(player, _index, history):
                        counter = len(actions)
                        if endpoint == 'initial':
                            out = host_actor(parameters, [player], [action_seed], [counter],
                                             2, 'nonlinear', catalog)[0]
                        else:
                            out = batched_actor(parameters, [player], [history], [action_seed],
                                                [counter], 2, catalog)[0]
                        values.append(out[4])
                        actions.append({'request_id': player['request_id'], 'action': out[2],
                            'log_probability': out[3], 'value': out[4]})
                        phases[player['phase']] += 1
                        return out[2]
                    folder = run / str(seed) / f'{episode:03d}-{int(swap)}'
                    evidence = run_game(folder,
                        run_seed=derived_seed('f34-evaluation-teams-v1', seed, 'foul_play'),
                        episode_index=episode, swap=swap, external_side=external_side,
                        search_time_ms=plan['search_time_ms'],
                        fp_seed=derived_seed('f34-external-sampling-v1', seed, episode, swap),
                        max_turns=plan['max_turns'], learner=decide, learner_uses_history=True,
                        purpose='preregistered F34 held-out-opponent development panel')
                    outcome = evidence['outcome']
                    if evidence['status'] != 'completed' or outcome['kind'] not in ('terminal', 'administrative_cutoff'):
                        raise ValueError('External game did not complete')
                    quarantine = set(plan['quarantined_team_keys'])
                    if any(t['partition'] != 'development' or t['team_key'] in quarantine
                           for t in evidence['provenance']['teams']):
                        raise ValueError('External team admission')
                    terminal = outcome['kind'] == 'terminal'
                    reward = outcome['rewards'][side] if terminal else None
                    game = {'episode': episode, 'split': 'development', 'swap': swap,
                        'learner_side': side, 'opponent': 'foul_play', 'provenance': evidence['provenance'],
                        'outcome': outcome, 'decisions': sum(len(s['actions']) for s in evidence['steps']),
                        'learner_decisions': len(actions), 'learner_phases': dict(phases),
                        'value_mean': float(np.mean(values)) if values else None,
                        'terminal_value_mse': float(np.mean([(v - reward)**2 for v in values]))
                            if terminal and values else None,
                        'seconds': evidence['seconds'], 'foul_play_decisions': evidence['foul_play_decisions'],
                        'rejected_actions': evidence['rejected_actions'],
                        'illegal_actions': evidence['illegal_actions'],
                        'game_record': str((folder / 'result.json').relative_to(ROOT)),
                        'game_record_sha256': digest(folder / 'result.json')}
                    atomic_json(folder / 'learner.json', {'parameter_hash': parameter_hash(parameters),
                        'action_seed': action_seed, 'decisions': actions})
                    games.append(game)
                    atomic_json(run / 'progress.json', {'stage': 'external evaluation', 'endpoint': endpoint,
                        'seed': seed, 'completed_games_for_seed': len(games),
                        'elapsed_seconds': time.perf_counter() - begun})
            summary = {'seed': seed, 'identity': identity, 'parameter_hash': parameter_hash(parameters),
                'games': games, 'summary': value_summary(games)}
            atomic_json(run / str(seed) / 'result.json', summary)
            summaries[str(seed)] = summary
        if digest(plan_path) != plan_hash:
            raise ValueError('External plan changed during evaluation')
        verify_inputs(admission_path)
        if digest(admission_path / 'manifest.json') != input_hash:
            raise ValueError('External source admission changed during evaluation')
        report = {'status': 'passed', 'endpoint': endpoint, 'plan': plan,
            'plan_sha256': plan_hash, 'input_hash': input_hash, 'seeds': summaries,
            'elapsed_seconds': time.perf_counter() - begun,
            'scope': 'Pinned local search opponent at a small fixed wall-time budget. Rust MCTS is stochastic; no ladder Elo claim.'}
        atomic_json(run / 'report.json', report)
        print({'status': 'passed', 'endpoint': endpoint, 'seconds': report['elapsed_seconds']}, flush=True)
        return report
    except BaseException as exc:
        atomic_json(run / 'failure.json', {'status': 'failed', 'error': repr(exc), 'endpoint': endpoint,
            'completed_seeds': sorted(summaries), 'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--endpoint', choices=['initial', 'final'], required=True)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    panel(args.run, args.plan, args.endpoint, args.training_run, args.admission)
