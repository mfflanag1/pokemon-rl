"""Fixed fresh F35 teacher-adaptation panels for shared initial and final arms."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from pokemon_full.learner import parameter_hash
from pokemon_full_train.campaign import value_summary
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, digest
from pokemon_structured_policy.actor import batched_actor
from pokemon_structured_policy.checkpoint import load
from .common import (ARMS, SEEDS, DEADLINE, before_deadline, evaluation_specs,
                     load_initial, read, verify_inputs)
from .external import run_game


def parameters_for(training_run, report, seed, endpoint):
    initial, _, catalog, _, identity = load_initial(seed)
    if endpoint == 'initial':
        return initial, catalog, identity
    if endpoint not in ARMS:
        raise ValueError('Unknown F35 external endpoint')
    cohorts = report['plan']['cohorts_per_seed_arm']
    p, opt, _ = load(training_run / str(seed) / endpoint / 'checkpoints' / str(cohorts),
        {'input_hash': report['input_hash'], 'seed': seed, 'cohorts': cohorts, 'arm': endpoint})
    expected = report['seeds'][str(seed)]['arms'][endpoint]
    if int(opt[2]) != cohorts * 8 or parameter_hash(p) != expected['final_parameter_hash']:
        raise ValueError('F35 final checkpoint identity')
    return p, catalog, identity


def run(output, training_run, admission, demonstrations):
    output, training_run, admission, demonstrations = map(
        lambda p: Path(p).resolve(), (output, training_run, admission, demonstrations))
    admitted = verify_inputs(admission)
    plan = admitted['plan']
    training = read(training_run / 'report.json')
    input_hash = digest(admission / 'manifest.json')
    if training['status'] != 'passed' or training['input_hash'] != input_hash:
        raise ValueError('F35 training completion admission')
    teacher = read(demonstrations / 'report.json')
    if (teacher['status'] != 'passed' or teacher['input_hash'] != input_hash
            or digest(demonstrations / 'report.json') != training['demonstration_manifest_sha256']):
        raise ValueError('F35 demonstration/probe identity')
    excluded = set(plan['quarantined_team_keys']) | {t['team_key'] for g in teacher['games']
        if g['role'] == 'probe' for t in g['provenance']['teams']}
    output.mkdir(parents=True, exist_ok=False)
    started, results, counts = time.perf_counter(), {}, Counter()
    try:
        for seed in SEEDS:
            results[str(seed)] = {}
            for endpoint in ('initial', *ARMS):
                before_deadline()
                parameters, catalog, identity = parameters_for(training_run, training, seed, endpoint)
                wanted_hash = parameter_hash(parameters)
                games = []
                specs = evaluation_specs(plan, seed, 'foul_play', 0, plan['external_pairs_per_seed_state'])
                for spec in specs:
                    before_deadline()
                    episode, swap, side = spec['episode'], spec['swap'], spec['learner_side']
                    action_seed = derived_seed('actor', spec['actor_seed'], episode, swap, side)
                    actions, phases = [], Counter()

                    def decide(player, _index, history):
                        out = batched_actor(parameters, [player], [history], [action_seed],
                            [len(actions)], 2, catalog)[0]
                        actions.append({'request_id': player['request_id'], 'action': out[2],
                            'log_probability': out[3], 'value': out[4]})
                        phases[player['phase']] += 1
                        return out[2]

                    folder = output / str(seed) / endpoint / f'{episode:03d}-{int(swap)}'
                    evidence = run_game(folder, run_seed=spec['seed'], episode_index=episode, swap=swap,
                        external_side='p1' if side == 'p2' else 'p2', split='development',
                        search_time_ms=plan['teacher_data']['search_time_ms'],
                        fp_seed=derived_seed('f35-external-sampling-v1', seed, episode, swap),
                        max_turns=plan['max_turns'], learner=decide, learner_uses_history=True,
                        purpose='fixed F35 trained-on-teacher development panel',
                        capture_demonstrations=False, deadline_epoch=DEADLINE)
                    if evidence['status'] != 'completed':
                        raise ValueError('External game incomplete')
                    if any(t['partition'] != 'development' or t['team_key'] in excluded
                           for t in evidence['provenance']['teams']):
                        raise ValueError('External team admission')
                    outcome, values = evidence['outcome'], [a['value'] for a in actions]
                    terminal = outcome['kind'] == 'terminal'
                    reward = outcome['rewards'][side] if terminal else None
                    if not values:
                        raise ValueError('External learner never acted')
                    game = {'episode': episode, 'split': 'development', 'swap': swap,
                        'learner_side': side, 'opponent': 'foul_play', 'provenance': evidence['provenance'],
                        'outcome': outcome, 'decisions': evidence['decisions'],
                        'learner_decisions': len(actions), 'learner_phases': dict(phases),
                        'value_mean': float(np.mean(values)),
                        'terminal_value_mse': float(np.mean([(v - reward) ** 2 for v in values])) if terminal else None,
                        'seconds': evidence['seconds'], 'foul_play_decisions': evidence['foul_play_decisions'],
                        'rejected_actions': evidence['rejected_actions'], 'illegal_actions': evidence['illegal_actions'],
                        'game_record': str((folder / 'result.json').relative_to(ROOT)),
                        'game_record_sha256': digest(folder / 'result.json')}
                    atomic_json(folder / 'learner.json', {'parameter_hash': wanted_hash,
                        'action_seed': action_seed, 'decisions': actions})
                    games.append(game)
                    counts.update(games=1, decisions=game['decisions'], learner_decisions=len(actions),
                        teacher_decisions=evidence['foul_play_decisions'],
                        terminals=int(terminal), cutoffs=int(not terminal))
                    atomic_json(output / 'progress.json', {'seed': seed, 'endpoint': endpoint,
                        'games': counts['games'], 'elapsed_seconds': time.perf_counter() - started})
                result = {'seed': seed, 'endpoint': endpoint, 'identity': identity,
                    'parameter_hash': wanted_hash, 'games': games, 'summary': value_summary(games)}
                atomic_json(output / str(seed) / endpoint / 'result.json', result)
                results[str(seed)][endpoint] = result
        if counts['games'] != plan['expected_counts']['external_evaluation_games']:
            raise ValueError('F35 external game count')
        verify_inputs(admission)
        report = {'status': 'passed', 'input_hash': input_hash, 'plan': plan, 'seeds': results,
            'counts': dict(counts), 'elapsed_seconds': time.perf_counter() - started,
            'scope': 'Fresh development games against the demonstration teacher; adaptation, not unseen-opponent strength.',
            'teacher_search_reproducible': False}
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'counts': dict(counts), 'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output, args.training_run, args.admission, args.demonstrations)
    print({'status': report['status'], 'counts': report['counts']}, flush=True)
