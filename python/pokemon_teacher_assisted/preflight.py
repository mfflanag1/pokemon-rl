"""Discarded engineering-only collect/update comparison from every F34 host."""
import argparse
from pathlib import Path
import time

import numpy as np
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import atomic_json, canonical, digest, runtime
from pokemon_structured_policy.actor import batched_actor
from pokemon_structured_policy.checkpoint import save, load
from .campaign import collect_cohort, optimize
from .collector import Pool
from .common import SEEDS, ARMS, DEADLINE, before_deadline, load_initial, read, state_hash, evaluation_specs
from .data import Dataset
from .external import run_game


def run(output, plan_path):
    before_deadline()
    output, plan_path = Path(output), Path(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    plan = read(plan_path)
    started, cells, teachers, completed = time.perf_counter(), [], [], []
    try:
        teacher_parameters, _, catalog, _, _ = load_initial(SEEDS[0])
        for side in ('p1', 'p2'):
            before_deadline()
            learner_side = 'p1' if side == 'p2' else 'p2'
            actor_seed = derived_seed('f35-engineering-teacher-host-v1', learner_side)
            decisions = []
            def decide(player, _index, history):
                result = batched_actor(teacher_parameters, [player], [history], [actor_seed],
                    [len(decisions)], 2, catalog)[0]
                decisions.append(result[2])
                return result[2]
            folder = output / ('teacher-' + side)
            teacher = run_game(folder, run_seed=derived_seed('f35-engineering-teacher-teams-v1', SEEDS[0]),
                episode_index=0, swap=side == 'p1', external_side=side, split='train',
                search_time_ms=100, fp_seed=derived_seed('f35-engineering-teacher-search-v1', side),
                max_turns=plan['max_turns'], learner=decide, learner_uses_history=True,
                purpose='Discarded F35 original collect/update engineering preflight; never scientific corpus',
                capture_demonstrations=True, deadline_epoch=DEADLINE)
            if teacher['status'] != 'completed' or teacher['accepted_labels'] <= 0:
                raise ValueError('Engineering teacher game incomplete or labels absent')
            with np.load(folder / 'demonstrations.npz', allow_pickle=False) as arrays:
                batch = {k: arrays[k].copy() for k in ('features', 'mask', 'actions')}
            teachers.append((teacher['game_id'], batch))
            completed.append({'side': side, 'game_id': teacher['game_id'], 'outcome': teacher['outcome'],
                'accepted_labels': teacher['accepted_labels'], 'decisions': teacher['decisions'],
                'rejected_actions': teacher['rejected_actions'], 'seconds': teacher['seconds'],
                'files': {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})
        # Real accepted-action labels, strictly validated in memory. No fake
        # scientific manifest or demo corpus enters the subsequent fixed run.
        dataset = Dataset(teachers, split='train')
        initial_hash = digest(plan_path)
        for seed in SEEDS:
            parameters, optimizer, catalog, frozen, identity = load_initial(seed)
            initial_parameters = parameter_hash(parameters)
            initial_state = state_hash(parameters, optimizer)
            expected_batch = expected_ledger = None
            engineering_seed = derived_seed('f35-engineering-collect-v1', seed)
            if engineering_seed in SEEDS:
                raise ValueError('Engineering/scientific coordinate collision')
            for arm in ARMS:
                before_deadline()
                folder = output / str(seed) / arm
                folder.mkdir(parents=True)
                with Pool(folder / 'workers', plan['workers'], catalog, frozen) as pool:
                    batch, ledger, profiles = collect_cohort(pool, parameters, plan, engineering_seed, 0,
                        {key: 0 for key in plan['stream_selected_rows']}, set(plan['quarantined_team_keys']))
                    if arm == 'control':
                        expected_batch, expected_ledger = ledger['batch_sha256'], canonical(ledger)
                        # The new opponent path is exercised on distinct paired
                        # engineering development games without updating from it.
                        specs = evaluation_specs(plan, engineering_seed, 'baseline-v1', 0, 1)
                        evaluation, eval_profile = pool.collect(parameters, specs)
                        if any(rows for rows, _ in evaluation) or any(
                                game['opponent'] != 'baseline-v1' for _, game in evaluation):
                            raise ValueError('Engineering baseline path leaked learner rows')
                        atomic_json(folder / 'baseline-engineering.json', {'specs': specs,
                            'games': [game for _, game in evaluation], 'profile': eval_profile})
                    elif ledger['batch_sha256'] != expected_batch or canonical(ledger) != expected_ledger:
                        raise ValueError('Same initial arms diverged before update')
                if parameter_hash(parameters) != initial_parameters or state_hash(parameters, optimizer) != initial_state:
                    raise ValueError('Collection mutated initial state')
                metadata = {'input_hash': initial_hash, 'seed': seed, 'arm': arm, 'cohorts': 0,
                    'next_episode': 0, 'purpose': 'discarded F35 engineering preflight'}
                save(folder / 'checkpoint-0', parameters, optimizer, metadata)
                q, qo, _ = load(folder / 'checkpoint-0', metadata)
                if state_hash(q, qo) != initial_state:
                    raise ValueError('Engineering initial checkpoint mismatch')
                optimizer_seed = derived_seed('f35-engineering-optimizer-v1', seed)
                demo_seed = derived_seed('f35-engineering-demonstrations-v1', seed)
                update_start = time.perf_counter()
                trained, updated, metrics = optimize(arm, parameters, optimizer, batch,
                    optimizer_seed, dataset, demo_seed, 0)
                update_seconds = time.perf_counter() - update_start
                replay_start = time.perf_counter()
                recovered, ro, replay_metrics = optimize(arm, q, qo, batch,
                    optimizer_seed, dataset, demo_seed, 0)
                replay_seconds = time.perf_counter() - replay_start
                if state_hash(trained, updated) != state_hash(recovered, ro) or metrics != replay_metrics:
                    raise ValueError('Engineering original-next-update mismatch')
                if int(updated[2]) != 8 or len(metrics) != 8 or parameter_hash(trained) == initial_parameters:
                    raise ValueError('Engineering update count or parameter motion')
                if len(batch['actions']) != 1024 or [x['selection']['selected_rows'] for x in ledger['strata']] != [256, 256, 512]:
                    raise ValueError('Engineering selected row budget')
                final_metadata = {**metadata, 'cohorts': 1}
                save(folder / 'checkpoint-1', trained, updated, final_metadata)
                q2, qo2, _ = load(folder / 'checkpoint-1', final_metadata)
                if state_hash(q2, qo2) != state_hash(trained, updated):
                    raise ValueError('Engineering final checkpoint mismatch')
                np.savez_compressed(folder / 'batch.npz', **batch)
                record = {'seed': seed, 'arm': arm, 'engineering_seed': engineering_seed,
                    'identity': identity, 'initial_parameter_hash': initial_parameters,
                    'initial_state_hash': initial_state, 'final_state_hash': state_hash(trained, updated),
                    'ledger': ledger, 'metrics': metrics, 'profiles': profiles, 'optimizer_steps': 8,
                    'selected_rows': 1024, 'checkpoint_comparisons': 2, 'original_next_update_recoveries': 1,
                    'update_seconds_including_compile': update_seconds, 'recovery_seconds': replay_seconds,
                    'initial_predictions_preserved': True, 'batch_identity_between_arms': expected_batch,
                    'original_next_update_identical': True}
                atomic_json(folder / 'result.json', record)
                cells.append(record)
                print({'seed': seed, 'arm': arm, 'status': 'passed',
                    'games': len(ledger['games']), 'elapsed_seconds': time.perf_counter()-started}, flush=True)
        result = {'status': 'passed', 'schema_version': 'f35-engineering-preflight-v1',
            'runtime': runtime(), 'teacher_games': completed, 'cells': cells,
            'counts': {'teacher_games': len(completed), 'local_training_games': sum(len(c['ledger']['games']) for c in cells),
                'local_evaluation_games': 2 * len(SEEDS), 'selected_rows': 1024 * len(cells),
                'optimizer_steps': 8 * len(cells), 'physical_optimizer_steps_including_recovery': 16 * len(cells),
                'checkpoint_comparisons': 2 * len(cells), 'original_next_update_recoveries': len(cells)},
            'elapsed_seconds': time.perf_counter() - started,
            'scope': 'Discarded engineering data/state; matched initial collection and exact original next update for all six seed/arm combinations; not a learning result.'}
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'teacher_games': completed, 'completed_cells': [{'seed': c['seed'], 'arm': c['arm']} for c in cells],
            'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.output, args.plan)
    print({'status': report['status'], 'counts': report['counts'], 'seconds': report['elapsed_seconds']}, flush=True)
