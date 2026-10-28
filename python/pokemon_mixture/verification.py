"""Audit all F5 bindings; bounded original-update and evaluation decision replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from pokemon_full.policy import initialize
from pokemon_full.learner import parameter_hash
from pokemon_full_train.storage import atomic_json, canonical, digest, admission, load_checkpoint
from pokemon_full_train.learning import derived_seed
from .collector import Pool
from .campaign import (read, validate_plan, checkpoint_dir, validate_checkpoint, summarize_arm,
                       optimizer_hashes, evaluation_specs, check_games)
from .learning import collect_update, optimize_batch, array_hash
from .statistics import contrasts, mixture_gate


def audit(run, output):
    if not __debug__:
        raise RuntimeError('F5 audit requires enabled assertions; remove -O/PYTHONOPTIMIZE')
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest = read(run/'inputs/manifest.json'); admission(run, manifest)
    report = read(run/'report.json'); plan = report['plan']; validate_plan(plan)
    assert report['status'] == 'completed' and plan == manifest['plan']
    assert report['input_hash'] == digest(run/'inputs/manifest.json')
    assert [r['seed'] for r in report['seeds']] == plan['seeds']
    totals = dict(training_games=0, evaluation_games=0, decisions=0, terminals=0, cutoffs=0,
                  true_ties=0, unavailable_choices=0, optimizer_steps=0,
                  collected_learner_rows=0, selected_learner_rows=0, excluded_learner_rows=0)
    replays = []; recomputed = []

    def count(games, role):
        totals[role+'_games'] += len(games)
        for g in games:
            totals['decisions'] += g['decisions']; totals['unavailable_choices'] += g['unavailable_choices']
            terminal = g['outcome']['kind'] == 'terminal'
            totals['terminals' if terminal else 'cutoffs'] += 1
            totals['true_ties'] += int(terminal and g['outcome']['rewards']['p1'] == .5)

    def panel(pool, parameters, seed, endpoint):
        panels = {}; samples = []
        for opponent in plan['opponents']:
            games = []
            for first in range(0, plan['evaluation_pairs_per_opponent'], plan['evaluation_chunk_pairs']):
                p = run/f'seed-{seed}/evaluation/{endpoint}-{opponent}-{first:04d}.json'
                body = read(p); sha = body.pop('content_sha256')
                assert hashlib.sha256(canonical(body)).hexdigest() == sha
                assert body['input_hash'] == report['input_hash']
                assert body['parameter_hash'] == parameter_hash(parameters)
                assert body['coordinates'] == dict(seed=seed, endpoint=endpoint, opponent=opponent,
                    first_pair=first, pairs=plan['evaluation_chunk_pairs'])
                check_games(body['games'], split='development', first=first,
                            count=plan['evaluation_chunk_pairs'], opponent=opponent)
                assert all(g['learner_rows'] == 0 for g in body['games'])
                for g in body['games']:
                    assert g['learner_decisions'] == sum(t[0] == g['learner_side'] for t in g['trace'])
                games.extend(body['games'])
            replayed, _ = pool.collect(parameters, evaluation_specs(plan, seed, opponent, 0, 1))
            assert all(not rows for rows, _ in replayed)
            assert [g for _, g in replayed] == games[:2], 'Original evaluation decision replay mismatch'
            samples.append({'endpoint': endpoint, 'opponent': opponent, 'pair': 0, 'games': 2})
            count(games, 'evaluation'); panels[opponent] = games
        return panels, samples

    with Pool(output/'replay-transport', 8) as pool:
        for result in report['seeds']:
            seed = result['seed']; initial = initialize(seed)
            assert parameter_hash(initial) == result['initial_parameter_hash']
            before, samples = panel(pool, initial, seed, 'initial')
            finals, arms = {}, {}
            for arm in plan['arms']:
                final, opt, meta = validate_checkpoint(run, seed, arm, plan, report['input_hash'])
                assert meta['cohorts'] == plan['updates_per_arm']
                summary = summarize_arm(run, seed, arm, final, opt, meta)
                assert summary == result['arms'][arm]
                arms[arm] = summary
                totals['optimizer_steps'] += int(opt[2])
                for key in ['collected_learner_rows', 'selected_learner_rows', 'excluded_learner_rows']:
                    totals[key] += summary[key]
                updates = [read(run/ref['file']) for ref in meta['cohort_records']]
                games = [g for c in updates for s in c['strata'].values() for g in s['episodes']]
                count(games, 'training')
                for p in (run/f'seed-{seed}/arms/{arm}/telemetry').glob('*.json'):
                    t = read(p)
                    assert 0 <= t['update_index'] < len(updates) and p.stem == f'{t["update_index"]:04d}'
                    assert t['full_cycle_seconds'] > 0 and t['selected_rows'] == plan['selected_rows_per_update']
                    assert t['optimizer_steps'] == updates[t['update_index']]['optimizer_after']['step']
                last = updates[-1]; pointer = last['input_checkpoint']
                original = checkpoint_dir(run, seed, arm)/pointer['file']
                assert digest(original) == pointer['sha256']
                with tempfile.TemporaryDirectory() as d:
                    shutil.copyfile(original, Path(d)/original.name)
                    atomic_json(Path(d)/'latest.json', pointer)
                    prior, oldopt, oldmeta = load_checkpoint(d, report['input_hash'])
                assert oldmeta['cohort_records'] == meta['cohort_records'][:-1]
                assert oldmeta['cohorts'] == last['update_index']
                assert parameter_hash(prior) == last['parameter_hash_before']
                assert optimizer_hashes(oldopt) == last['optimizer_before']
                batch, strata, selection, cursors = collect_update(pool, prior, plan, seed, arm,
                    last['update_index'], oldmeta['next_episodes'])
                assert cursors == meta['next_episodes'] and selection == last['selection']
                for s in strata:
                    for key in ['first_episode', 'specs', 'episodes']:
                        assert strata[s][key] == last['strata'][s][key], 'Original training decision replay mismatch'
                assert array_hash(batch) == last['selected_batch_sha256']
                actual, actualopt, metrics = optimize_batch(prior, oldopt, batch,
                    derived_seed('f5-update', seed, last['update_index']))
                assert parameter_hash(actual) == parameter_hash(final)
                assert optimizer_hashes(actualopt) == optimizer_hashes(opt) and metrics == last['metrics']
                finalpanel, samples_arm = panel(pool, final, seed, arm)
                finals[arm] = finalpanel; samples.extend(samples_arm)
                replays.append({'seed': seed, 'arm': arm, 'original_next_update_exact': True,
                    'games_replayed': sum(len(s['episodes']) for s in strata.values()),
                    'model_sha256': parameter_hash(actual), 'optimizer': optimizer_hashes(actualopt),
                    'selected_batch_sha256': array_hash(batch)})
            computed = contrasts(before, finals, plan['statistics'])
            assert computed == result['contrasts']
            recomputed.append({**result, 'arms': arms, 'contrasts': computed})
            replays.append({'seed': seed, 'evaluation_samples': samples})
    exposure = len(plan['seeds'])*len(plan['arms'])*plan['updates_per_arm']*plan['selected_rows_per_update']
    assert totals['selected_learner_rows'] == exposure
    assert totals['collected_learner_rows'] == exposure+totals['excluded_learner_rows']
    assert totals['optimizer_steps'] == 2*exposure//256
    assert totals['evaluation_games'] == len(plan['seeds'])*3*3*2*plan['evaluation_pairs_per_opponent']
    assert mixture_gate(recomputed, plan['statistics']) == report['learning_gate']
    admission(run, manifest)
    body = {'status': 'passed', 'report_sha256': digest(run/'report.json'), 'input_hash': report['input_hash'],
        'totals': totals, 'replays': replays, 'learning_gate': report['learning_gate'],
        'scope': 'All update/checkpoint/counter/selection/evaluation-summary bindings audited. Exact original last-update decision, selected-batch, model/Adam/metric replay for every seed/arm; first paired evaluation replay per seed/endpoint/opponent. Bounded replay, not exhaustive independent replay of every game or update.'}
    atomic_json(output/'audit.json', body)
    return body


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.output)['totals']), flush=True)
