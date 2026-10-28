"""Bounded engineering qualification; none of these games enter the screen."""
import argparse
from pathlib import Path
import time

import numpy as np
from pokemon_full_train.storage import ROOT, atomic_json
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_fit.data import fingerprint
from pokemon_transfer.common import select
from pokemon_worker_watch.collector import Pool as OriginalPool
from .checkpoint import load_f32, save, load
from .collector import Pool
from .common import SEEDS, state_hash, before_deadline
from .network import extend, PARAMETER_COUNT
from .learning import optimize_batch


def jobs(seed, opponent, first=0, count=8):
    return [dict(seed=derived_seed('f34-engineering-teams-v1', seed, opponent),
        episode=i, max_turns=240, split='train', swap=False, opponent=opponent,
        learner_side='p2' if i % 2 else 'p1',
        actor_seed=derived_seed('f34-engineering-actor-v1', seed, opponent))
        for i in range(first, first + count)]


def run(destination):
    before_deadline()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    results = []
    try:
        for seed in SEEDS:
            host, catalog, identity = load_f32(seed)
            p, opt = extend(host, derived_seed('f34-structured-initialization-v1', seed))
            base = destination / str(seed)
            base.mkdir()
            # Sidecar collection must preserve the original engine and actor path
            # at initialization. Compare actual complete games, not just tensors.
            parity_jobs = jobs(seed, None, count=2)
            with OriginalPool(base / 'original', 8, 'nonlinear', catalog) as old:
                expected, _ = old.collect(host, parity_jobs)
            with Pool(base / 'structured', 8, catalog, host) as pool:
                actual, _ = pool.collect(p, parity_jobs)
                for (_, a), (_, b) in zip(expected, actual):
                    for key in ('outcome', 'provenance', 'decisions', 'phases', 'unavailable_choices'):
                        if a[key] != b[key]:
                            raise ValueError('Initial complete-game parity: ' + key)
                    aa = [(r['side'], r['request_id'], r['action'], r['log_probability'], r['value'])
                          for r in a['actor_trace']]
                    bb = [(r['side'], r['request_id'], r['action'], r['log_probability'], r['value'])
                          for r in b['actor_trace']]
                    if aa != bb:
                        raise ValueError('Initial actor decision/value parity')
                collected, strata, batches, profiles = [], [], [], []
                for opponent, quota in ((None, 256), ('frozen_f32', 256), ('tactical', 512)):
                    stream, first = [], 100
                    while sum(len(rows) for rows, _ in stream) < quota:
                        if first >= 164:
                            raise ValueError('Engineering complete-game collection cap')
                        chunk, profile = pool.collect(p, jobs(seed, opponent, first))
                        stream.extend(chunk)
                        profiles.append(profile)
                        first += 8
                    selected, selection = select(stream, derived_seed('f34-engineering-selection', seed, opponent), 0, quota)
                    batches.append(selected)
                    collected.extend(game for _, game in stream)
                    strata.append({'opponent': opponent, 'selection': selection,
                        'games': len(stream), 'first_episode': 100, 'next_episode': first})
                batch = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
                if len(batch['actions']) != 1024:
                    raise ValueError('Engineering cohort row budget')
                np.savez_compressed(base / 'selected-batch.npz', **batch)
                metadata = {'input_hash': fingerprint(batch), 'seed': seed, 'cohorts': 0, 'next_episode': 0}
                save(base / 'before', p, opt, metadata)
                restored, restored_opt, _ = load(base / 'before', metadata)
                if state_hash(p, opt) != state_hash(restored, restored_opt):
                    raise ValueError('Checkpoint round trip')
                optimizer_seed = derived_seed('f34-engineering-update', seed)
                started = time.perf_counter()
                q, qo, metrics = optimize_batch(p, opt, batch, optimizer_seed)
                update_seconds = time.perf_counter() - started
                r, ro, replay_metrics = optimize_batch(restored, restored_opt, batch, optimizer_seed)
                if state_hash(q, qo) != state_hash(r, ro) or metrics != replay_metrics:
                    raise ValueError('Original-next-update replay')
                if int(qo[2]) != 8 or parameter_hash(q) == parameter_hash(p):
                    raise ValueError('Update counter/non-vacuity')
                save(base / 'after', q, qo, {**metadata, 'cohorts': 1})
                row = {'seed': seed, 'identity': identity, 'initial_parameter_hash': parameter_hash(p),
                    'final_parameter_hash': parameter_hash(q), 'parameter_count': PARAMETER_COUNT,
                    'parity_games': len(actual), 'training_games': len(collected),
                    'training_decisions': sum(g['decisions'] for g in collected),
                    'terminal_games': sum(g['outcome']['kind'] == 'terminal' for g in collected),
                    'cutoff_games': sum(g['outcome']['kind'] == 'administrative_cutoff' for g in collected),
                    'strata': strata, 'profiles': profiles,
                    'collection_seconds': sum(x['collection_seconds'] for x in profiles),
                    'update_seconds_including_compile': update_seconds, 'metrics': metrics,
                    'batch_sha256': fingerprint(batch), 'original_next_update_exact': True,
                    'initial_complete_game_parity_exact': True}
                atomic_json(base / 'result.json', row)
                results.append(row)
                atomic_json(destination / 'progress.json', {'seeds_finished': len(results), 'status': 'running'})
        report = {'status': 'passed', 'purpose': 'engineering only; discarded states and disjoint schedules',
            'seeds': results, 'elapsed_seconds': time.perf_counter() - begun}
        atomic_json(destination / 'report.json', report)
        print({'status': 'passed', 'seconds': report['elapsed_seconds'],
            'collection_seconds': [x['collection_seconds'] for x in results],
            'update_seconds': [x['update_seconds_including_compile'] for x in results]}, flush=True)
        return report
    except BaseException as exc:
        atomic_json(destination / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'seeds_finished': len(results), 'elapsed_seconds': time.perf_counter() - begun})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
