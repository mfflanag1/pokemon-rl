"""Discarded engineering-only original-state collect/update qualification for F39."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, digest, runtime
from pokemon_structured_policy.checkpoint import save, load
from pokemon_structured_policy.learning import optimize_batch
from pokemon_teacher_assisted.collector import Pool
from .campaign import collect_cohort, peak_parent_rss_mib
from .common import (SEEDS, NAMESPACES, deadline, load_initial, read, state_hash,
                     evaluation_specs, admit_games, validate_plan)

QUALIFICATION = 'artifacts/reviews/f35-teacher-assisted/qualification-worker-campaign-v3.json'


def retained_rejection_fixture():
    """Reuse the recorded real unavailable-choice fixture only with unchanged inputs."""
    report = read(ROOT / QUALIFICATION)
    fixture = report.get('explicit_unavailable_choice_fixture', {})
    if report.get('status') != 'passed' or fixture.get('passed') is not True:
        raise ValueError('Retained rejection qualification did not pass')
    if fixture.get('evidence_log') not in report['evidence_files']:
        raise ValueError('Retained rejection fixture evidence missing')
    checked = {}
    for group in ('source_files', 'evidence_files'):
        for rel, expected in report[group].items():
            if digest(ROOT / rel) != expected:
                raise ValueError('Retained rejection fixture input drift: ' + rel)
            checked[rel] = expected
    return {'path': QUALIFICATION, 'sha256': digest(ROOT / QUALIFICATION),
        'status': 'passed', 'fixture': fixture, 'verified_files': checked,
        'scope': 'Previously executed real rejection fixture; unchanged bytes verified, fixture not rerun.'}


def run(output, plan_path):
    output, plan_path = Path(output), Path(plan_path)
    scientific_plan = validate_plan(read(plan_path))
    deadline(scientific_plan)
    plan = copy.deepcopy(scientific_plan)
    plan['namespaces'] = {key: value.replace('f39-', 'f39-discarded-engineering-')
                          for key, value in NAMESPACES.items()}
    output.mkdir(parents=True, exist_ok=False)
    started, cells = time.perf_counter(), []
    try:
        rejection_fixture = retained_rejection_fixture()
        initial_hash = digest(plan_path)
        for seed in SEEDS:
            deadline(plan)
            parameters, optimizer, catalog, frozen, identity = load_initial(seed)
            initial_parameters = parameter_hash(parameters)
            initial_state = state_hash(parameters, optimizer)
            cursors = dict(identity['f34_metadata']['stream_cursors'])
            initial_cursors = dict(cursors)
            folder = output / str(seed)
            folder.mkdir()
            metadata = {'input_hash': initial_hash, 'seed': seed, 'cohorts': 512,
                'next_episode': sum(cursors.values()), 'stream_cursors': dict(cursors),
                'selected_rows': 524288, 'purpose': 'discarded F39 engineering preflight', 'adam_reset': False}
            save(folder / 'checkpoint-512', parameters, optimizer, metadata)
            recovered, ro, _ = load(folder / 'checkpoint-512', metadata)
            if state_hash(recovered, ro) != initial_state or int(ro[2]) != 4096:
                raise ValueError('Engineering original checkpoint/Adam mismatch')
            with Pool(folder / 'workers', plan['workers'], catalog, frozen) as pool:
                batch, ledger, profiles = collect_cohort(pool, parameters, plan, seed, 512,
                    cursors, set(plan['quarantined_team_keys']))
                specs = evaluation_specs(plan, seed, 'baseline-v1', 0, 1)
                evaluation, eval_profile = pool.collect(parameters, specs)
                if len(evaluation) != 2 or any(rows for rows, _ in evaluation):
                    raise ValueError('Engineering baseline evaluation inventory/row leakage')
                games = [game for _, game in evaluation]
                admit_games(games, 'development', set(plan['quarantined_team_keys']))
                if any(game[key] != spec[key] for spec, game in zip(specs, games)
                       for key in ('episode', 'swap', 'learner_side', 'opponent')):
                    raise ValueError('Engineering baseline schedule identity')
                atomic_json(folder / 'baseline-engineering.json',
                    {'specs': specs, 'games': games, 'profile': eval_profile})
            if parameter_hash(parameters) != initial_parameters or state_hash(parameters, optimizer) != initial_state:
                raise ValueError('Collection mutated original F34 state')
            deadline(plan)
            optimizer_seed = derived_seed(plan['namespaces']['optimizer'], seed, 512)
            update_start = time.perf_counter()
            trained, updated, metrics = optimize_batch(parameters, optimizer, batch, optimizer_seed)
            update_seconds = time.perf_counter() - update_start
            deadline(plan)
            replay_start = time.perf_counter()
            recovered, ro, replay_metrics = optimize_batch(recovered, ro, batch, optimizer_seed)
            replay_seconds = time.perf_counter() - replay_start
            if state_hash(trained, updated) != state_hash(recovered, ro) or metrics != replay_metrics:
                raise ValueError('Engineering original-next-update mismatch')
            if int(updated[2]) != 4104 or len(metrics) != 8 or parameter_hash(trained) == initial_parameters:
                raise ValueError('Engineering cumulative update count or parameter motion')
            if len(batch['actions']) != 1024 or [x['selection']['selected_rows'] for x in ledger['strata']] != [256, 256, 512]:
                raise ValueError('Engineering selected row budget')
            final_metadata = {**metadata, 'cohorts': 513, 'next_episode': sum(cursors.values()),
                'stream_cursors': dict(cursors), 'selected_rows': 525312}
            save(folder / 'checkpoint-513', trained, updated, final_metadata)
            q2, qo2, _ = load(folder / 'checkpoint-513', final_metadata)
            if state_hash(q2, qo2) != state_hash(trained, updated):
                raise ValueError('Engineering final checkpoint mismatch')
            np.savez_compressed(folder / 'batch.npz', **batch)
            record = {'seed': seed, 'identity': identity, 'initial_parameter_hash': initial_parameters,
                'initial_state_hash': initial_state, 'final_state_hash': state_hash(trained, updated),
                'initial_stream_cursors': initial_cursors, 'final_stream_cursors': dict(cursors),
                'initial_optimizer_step': 4096, 'final_optimizer_step': int(updated[2]),
                'ledger': ledger, 'metrics': metrics, 'profiles': profiles, 'evaluation_profile': eval_profile,
                'optimizer_seed': optimizer_seed, 'optimizer_steps': 8, 'selected_rows': 1024,
                'checkpoint_comparisons': 2, 'original_next_update_recoveries': 1,
                'update_seconds_including_compile': update_seconds, 'recovery_seconds': replay_seconds,
                'original_next_update_identical': True, 'adam_reset': False}
            atomic_json(folder / 'result.json', record)
            cells.append(record)
            print({'seed': seed, 'status': 'passed', 'games': len(ledger['games']),
                   'elapsed_seconds': time.perf_counter() - started}, flush=True)
        collection_seconds = sum(p['collection_seconds'] for cell in cells for p in cell['profiles'])
        evaluation_seconds = sum(cell['evaluation_profile']['collection_seconds'] for cell in cells)
        optimizer_seconds = sum(cell['update_seconds_including_compile'] for cell in cells)
        estimate = (collection_seconds + optimizer_seconds) * 256 + evaluation_seconds * 768
        elapsed = time.perf_counter() - started
        result = {'status': 'passed', 'schema_version': 'f39-engineering-preflight-v1',
            'runtime': runtime(), 'plan_sha256': initial_hash, 'engineering_namespaces': plan['namespaces'],
            'scientific_namespaces': scientific_plan['namespaces'], 'rejection_fixture': rejection_fixture,
            'cells': cells, 'counts': {'local_training_games': sum(len(c['ledger']['games']) for c in cells),
                'local_evaluation_games': 6, 'selected_rows': 3072, 'optimizer_steps': 24,
                'physical_optimizer_steps_including_recovery': 48, 'checkpoint_comparisons': 6,
                'original_next_update_recoveries': 3},
            'elapsed_seconds': elapsed, 'collection_seconds': collection_seconds,
            'update_seconds_including_compile': optimizer_seconds, 'evaluation_seconds': evaluation_seconds,
            'peak_parent_rss_mib': peak_parent_rss_mib(),
            'memory_scope': 'Parent process high-water RSS; excludes workers and total machine memory.',
            'estimated_campaign_seconds': estimate,
            'estimation': 'Conservative small-sample projection: 3 collect/update cells x256 plus six engineering baseline games x768; includes first-call compilation scaled across cohorts; excludes independent verification and can vary with battle lengths.',
            'scope': 'Discarded engineering data/states, every original F34 host and Adam state; not a scientific learning run.'}
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'completed_cells': [cell['seed'] for cell in cells], 'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.output, args.plan)
    print({'status': report['status'], 'counts': report['counts'], 'seconds': report['elapsed_seconds'],
           'estimated_campaign_seconds': report['estimated_campaign_seconds']}, flush=True)
