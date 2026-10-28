"""Reconstruct fixed original F26 cohorts, verify their updates, and audit clones."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import resource
import time
import numpy as np
from pokemon_fit.data import fingerprint
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, canonical, digest, atomic_json, freeze, admission
from pokemon_diversity.common import read, receipt
from pokemon_opponent_mix.common import load, same_state
from pokemon_opponent_mix.collector import Pool
from pokemon_opponent_mix.campaign import collect_update
from pokemon_credit.campaign import same
from .core import audit_batch, VARIANTS
from .signal import audit_signal

PLAN = ROOT / 'contracts/gradient-audit-v1/plan.json'


def validate_plan(plan):
    if plan != read(PLAN) or plan['seeds'] != [7103, 8209, 9311] or plan['arms'] != ['self', 'mixed']:
        raise ValueError('Audit plan/seed/arm mismatch')
    if plan['cohort_indices'] != [0, 63, 127] or plan['variants'] != list(VARIANTS) or plan['workers'] != 8:
        raise ValueError('Audit anchors/variants mismatch')
    if plan['game_cluster_splits_per_cohort'] != 4 or plan['game_cluster_namespace'] != 'f27-game-halves':
        raise ValueError('Fixed game-cluster diagnostics')
    if any(plan[k] for k in ['new_trajectory_sampling', 'evaluation_games', 'promotion_authorized',
                            'final_reserved_data_used', 'public_server_used', 'prime_used']):
        raise ValueError('Scope violation')


def predecessor(plan):
    r = receipt(ROOT / plan['predecessor_receipt'], plan['predecessor_sha256'], 'f26-opponent-mix')
    if r['status'] != 'passed' or r['run'] != plan['predecessor_run']:
        raise ValueError('Predecessor completion/run mismatch')
    base = ROOT / plan['predecessor_run']
    original_plan = read(base / 'inputs/manifest.json')['plan']
    same(original_plan, read(ROOT / 'contracts/opponent-mix-v1/plan.json'), 'F26 plan')
    return base, original_plan, digest(base / 'inputs/manifest.json')


def anchors(plan):
    return [(seed, arm, index) for seed in plan['seeds'] for arm in plan['arms']
            for index in plan['cohort_indices']]


def original_state(base, h, seed, arm, index):
    original = read(base / str(seed) / arm / f'update-{index:03d}.json')
    p, opt = load(base / str(seed) / arm / 'checkpoints' / str(index), h, seed, arm,
                  index, original['ledger']['first_episode'])
    if parameter_hash(p) != original['parameter_hash_before']:
        raise ValueError('Original pre-update parameters differ')
    return p, opt, original


def verify_original(base, h, seed, arm, index, original, result, states):
    q, qo = states['joint']
    if parameter_hash(q) != original['parameter_hash_after'] or int(qo[2]) != original['optimizer_step']:
        raise ValueError('Original update/counter mismatch')
    same(result['variants']['joint']['original_metrics'], original['metrics'], 'Original optimizer metrics')
    if index in [63, 127]:
        expected_p, expected_o = load(base / str(seed) / arm / 'checkpoints' / str(index + 1),
            h, seed, arm, index + 1, original['ledger']['next_episode'])
        same_state(q, qo, expected_p, expected_o)
    return index in [63, 127]


def state_fingerprint(state):
    p, opt = state
    arrays = {pre + '__' + k: np.asarray(v) for pre, group in zip(['p', 'm', 'v'], [p, opt[0], opt[1]])
              for k, v in group.items()}
    arrays['step'] = np.asarray(opt[2])
    return fingerprint(arrays)


def result_record(seed, arm, index, original, batch, result, states):
    return {'seed': seed, 'arm': arm, 'cohort_index': index,
            'original_update_sha256': digest(ROOT / 'artifacts/runs/f26-opponent-mix-v1' /
                                             str(seed) / arm / f'update-{index:03d}.json'),
            'batch_sha256': fingerprint(batch),
            'original_ledger_sha256': hashlib.sha256(canonical(original['ledger'])).hexdigest(),
            'state_fingerprints': {v: state_fingerprint(s) for v, s in states.items()}, 'diagnostics': result}


def campaign(run):
    plan = read(PLAN)
    validate_plan(plan)
    base, original_plan, h = predecessor(plan)
    run.mkdir(parents=True, exist_ok=False)
    manifest = freeze(run, plan)
    catalog = read(ROOT / original_plan['catalog'])
    if digest(ROOT / original_plan['catalog']) != original_plan['catalog_sha256']:
        raise ValueError('Catalog mismatch')
    started = time.perf_counter()
    counts = Counter()
    records = []
    collection_seconds = diagnostic_seconds = 0.
    for seed, arm, index in anchors(plan):
        folder = run / str(seed) / arm / str(index)
        p, opt, original = original_state(base, h, seed, arm, index)
        with Pool(folder / 'workers', 8, 'matchup', catalog) as pool:
            batch, ledger, profiles = collect_update(pool, p, original_plan, seed, index,
                original['ledger']['first_episode'], arm)
        same(ledger, original['ledger'], 'Every original game/decision/selection')
        if fingerprint(batch) != original['ledger']['selection']['selected_batch_sha256']:
            raise ValueError('Original selected batch mismatch')
        with (folder / 'batch.npz').open('xb') as out:
            np.savez_compressed(out, **batch)
        collection_seconds += sum(p['collection_seconds'] for p in profiles)
        t0 = time.perf_counter()
        result, states = audit_batch(p, opt, batch, derived_seed(original_plan['optimizer_seed_namespace'], seed, index))
        result['game_signal'] = audit_signal(p, batch, ledger, derived_seed('f27-game-halves', seed, arm, index))
        diagnostic_seconds += time.perf_counter() - t0
        matched = verify_original(base, h, seed, arm, index, original, result, states)
        record = result_record(seed, arm, index, original, batch, result, states)
        atomic_json(folder / 'result.json', record)
        records.append(record)
        counts.update(cohorts=1, selected_rows=1024, original_optimizer_steps=8,
                      diagnostic_optimizer_steps=16, gradient_minibatches=8, next_checkpoint_comparisons=int(matched))
        for game in ledger['games']:
            if game['split'] != 'train':
                raise ValueError('Non-training trajectory in audit')
            counts.update(replayed_games=1, replayed_decisions=game['decisions'])
            counts['natural_terminals' if game['outcome']['kind'] == 'terminal' else 'cutoffs'] += 1
        atomic_json(run / 'progress.json', {'cohorts': counts['cohorts'], 'seed': seed, 'arm': arm, 'index': index})
        print({'cohort': counts['cohorts'], 'seed': seed, 'arm': arm, 'index': index}, flush=True)
    counts.setdefault('cutoffs', 0)
    if dict(counts) != plan['expected_counts']:
        raise ValueError('Audit inventory mismatch: ' + str(dict(counts)))
    admission(run, manifest)
    predecessor(plan)
    report = {'status': 'passed', 'plan': plan, 'input_hash': digest(run / 'inputs/manifest.json'),
              'counts': dict(counts), 'records': records, 'elapsed_seconds': time.perf_counter() - started,
              'collection_seconds': collection_seconds, 'diagnostic_seconds_including_compile': diagnostic_seconds,
              'peak_parent_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              'new_games': 0, 'strength_evaluation': False, 'promotion_authorized': False}
    atomic_json(run / 'report.json', report)
    print({'status': 'passed', 'seconds': report['elapsed_seconds'], 'counts': dict(counts)}, flush=True)
    return report


def verify(run, output):
    if output.exists():
        raise ValueError('Verification output already exists')
    report = read(run / 'report.json')
    plan = read(PLAN)
    validate_plan(plan)
    same(report['plan'], plan, 'Audit report plan')
    same(report['counts'], plan['expected_counts'], 'Audit report counts')
    manifest = read(run / 'inputs/manifest.json')
    if digest(run / 'inputs/manifest.json') != report['input_hash'] or report['status'] != 'passed':
        raise ValueError('Audit report admission')
    admission(run, manifest)
    base, original_plan, h = predecessor(plan)
    records = []
    for seed, arm, index in anchors(plan):
        folder = run / str(seed) / arm / str(index)
        p, opt, original = original_state(base, h, seed, arm, index)
        with np.load(folder / 'batch.npz', allow_pickle=False) as z:
            batch = {k: z[k] for k in z.files}
        if fingerprint(batch) != original['ledger']['selection']['selected_batch_sha256']:
            raise ValueError('Saved batch/original mismatch')
        result, states = audit_batch(p, opt, batch, derived_seed(original_plan['optimizer_seed_namespace'], seed, index))
        result['game_signal'] = audit_signal(p, batch, original['ledger'], derived_seed('f27-game-halves', seed, arm, index))
        verify_original(base, h, seed, arm, index, original, result, states)
        record = result_record(seed, arm, index, original, batch, result, states)
        same(record, read(folder / 'result.json'), 'Diagnostic reproduction')
        records.append(record)
    same(records, report['records'], 'Complete diagnostic inventory')
    admission(run, manifest)
    result = {'status': 'passed', 'report_sha256': digest(run / 'report.json'),
              'input_hash': report['input_hash'], 'cohorts': len(records),
              'scope': 'All saved batches bound to original F26 fingerprints; all original and diagnostic updates rerun. Original games were freshly reconstructed during production and exactly compared with sealed F26 ledgers.'}
    atomic_json(output, result)
    print(result, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.run.resolve(), args.verify.resolve())
    else:
        campaign(args.run.resolve())
