"""Audit all F4 records and replay a declared, bounded sample of original games."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full.learner import parameter_hash
from pokemon_full_train.storage import (atomic_json, canonical, digest, admission,
                                       load_checkpoint)
from pokemon_full_train.learning import optimize, derived_seed
from pokemon_full_train.campaign import value_summary
from pokemon_throughput.collector import merge_rows
from .collector import Pool
from .campaign import validate_plan, validate_checkpoint, check_games, evaluation_specs
from .statistics import compare, learning_gate


def audit(run, output):
    if not __debug__:
        raise RuntimeError('F4 audit requires enabled assertions; remove -O/PYTHONOPTIMIZE')
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest = json.loads((run/'inputs/manifest.json').read_text())
    admission(run, manifest)
    report = json.loads((run/'report.json').read_text())
    plan = report['plan']; validate_plan(plan)
    assert report['status'] == 'completed' and plan == manifest['plan']
    assert report['input_hash'] == digest(run/'inputs/manifest.json')
    assert [r['seed'] for r in report['seeds']] == plan['seeds']
    totals = dict(training_games=0, evaluation_games=0, decisions=0, terminals=0,
                  cutoffs=0, unavailable_choices=0, optimizer_steps=0)
    replays = []

    def count(games, kind):
        totals[kind+'_games'] += len(games)
        for g in games:
            totals['decisions'] += g['decisions']
            totals['unavailable_choices'] += g['unavailable_choices']
            totals['terminals' if g['outcome']['kind']=='terminal' else 'cutoffs'] += 1

    with Pool(output/'replay-transport',8) as pool:
        for result in report['seeds']:
            seed = result['seed']; initial = initialize(seed)
            final, opt, meta = validate_checkpoint(run,seed,plan,report['input_hash'])
            assert meta['next_episode'] == plan['train_games_per_seed']
            assert parameter_hash(initial) == result['initial_parameter_hash']
            assert parameter_hash(final) == result['final_parameter_hash']
            assert int(opt[2]) == result['optimizer_steps']
            totals['optimizer_steps'] += int(opt[2])
            cohorts = [json.loads((run/r['file']).read_text()) for r in meta['cohort_records']]
            games = [g for c in cohorts for g in c['episodes']]
            assert value_summary(games) == result['training']
            assert sum(c['update_seconds'] for c in cohorts) == result['training_update_seconds']
            assert sum(c['collection_profile']['collection_seconds'] for c in cohorts) == result['training_collection_seconds']
            for c in cohorts:
                assert all(np.isfinite(v) for m in c['metrics'] for v in m.values())
            telemetry = [json.loads(p.read_text()) for p in sorted((run/f'seed-{seed}/cycle-telemetry').glob('*.json'))]
            assert len(telemetry) == result['complete_cycle_records']
            assert len(cohorts) == result['expected_cycle_records']
            assert sum(t['full_cycle_seconds'] for t in telemetry) == result['recorded_complete_cycle_seconds']
            for t in telemetry:
                c = cohorts[t['first_episode']//8]
                assert t['first_episode'] == c['first_episode'] and t['decisions'] == c['decisions']
                assert t['full_cycle_seconds'] > 0
            count(games,'training')

            # Recover the exact original penultimate checkpoint. No alternate
            # game, seed, policy or update is substituted if replay fails.
            directory = run/f'seed-{seed}/checkpoints'
            last = cohorts[-1]; pointer = last['input_checkpoint']
            assert Path(pointer['file']).name == pointer['file']
            original = directory/pointer['file']
            assert digest(original) == pointer['sha256']
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp)/original.name; shutil.copyfile(original,target)
                atomic_json(Path(temp)/'latest.json',pointer)
                p, oldopt, prior = load_checkpoint(temp, report['input_hash'])
            assert prior['cohort_records'] == meta['cohort_records'][:-1]
            assert prior['next_episode'] == last['first_episode']
            assert parameter_hash(p) == last['parameter_hash_before']
            collected,_ = pool.collect(p,[dict(seed=seed,episode=g['episode'],max_turns=plan['max_turns']) for g in last['episodes']])
            assert [g for _,g in collected] == last['episodes'], 'Original training decision replay mismatch'
            rows = merge_rows(collected)
            actual, actualopt, metrics = optimize(p,oldopt,rows,derived_seed('f4-update',seed,last['first_episode']))
            assert parameter_hash(actual) == parameter_hash(final)
            assert metrics == last['metrics'] and int(actualopt[2]) == int(opt[2])
            assert all(parameter_hash(a)==parameter_hash(b) for a,b in zip(actualopt[:2],opt[:2]))
            sampled = []
            for opponent in plan['opponents']:
                ends = []
                for endpoint, parameters in [('initial',initial),('final',final)]:
                    games = []
                    for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
                        path = run/f'seed-{seed}/evaluation/{endpoint}-{opponent}-{first:04d}.json'
                        body = json.loads(path.read_text()); sha = body.pop('content_sha256')
                        assert hashlib.sha256(canonical(body)).hexdigest() == sha
                        assert body['input_hash'] == report['input_hash']
                        assert body['parameter_hash'] == parameter_hash(parameters)
                        assert body['coordinates'] == dict(seed=seed,endpoint=endpoint,opponent=opponent,
                            first_pair=first,pairs=plan['evaluation_chunk_pairs'])
                        check_games(body['games'],split='development',first=first,
                            count=plan['evaluation_chunk_pairs'],opponent=opponent)
                        games.extend(body['games'])
                    replayed,_ = pool.collect(parameters,evaluation_specs(plan,seed,opponent,0,1))
                    assert all(not rows for rows,_ in replayed)
                    assert [g for _,g in replayed] == games[:2], 'Original evaluation decision replay mismatch'
                    sampled.append({'endpoint':endpoint,'opponent':opponent,'pair':0,'games':2})
                    count(games,'evaluation'); ends.append(games)
                assert compare(*ends,plan['statistics']) == result['evaluation'][opponent]
            replays.append({'seed':seed,'training_episodes':[g['episode'] for g in last['episodes']],
                'training_decisions':len(rows),'original_next_update_exact':True,
                'model_sha256':parameter_hash(actual),'adam_m_sha256':parameter_hash(actualopt[0]),
                'adam_v_sha256':parameter_hash(actualopt[1]),'evaluation_samples':sampled})
    assert totals['training_games'] == len(plan['seeds'])*plan['train_games_per_seed']
    assert totals['evaluation_games'] == len(plan['seeds'])*2*3*2*plan['evaluation_pairs_per_opponent']
    assert learning_gate(report['seeds'],plan['statistics']) == report['learning_gate']
    admission(run,manifest)
    body = {'status':'passed','report_sha256':digest(run/'report.json'),'input_hash':report['input_hash'],
        'totals':totals,'replays':replays,'learning_gate':report['learning_gate'],
        'scope':'All cohort, counter, checkpoint and evaluation-summary bindings audited. Exact original last-eight-game/update replay per seed; first paired matchup replay per endpoint/opponent/seed. This is a bounded decision replay, not exhaustive independent replay of every game.'}
    atomic_json(output/'audit.json',body)
    return body


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run,args.output)['totals']),flush=True)
