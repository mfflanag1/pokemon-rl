"""Matched-work F3 benchmark. No policy selection or expanded learning screen."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import shutil
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import collect, optimize
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest, source_paths, runtime, load_checkpoint
from pokemon_full_train.transport import Worker
from .collector import Pool, merge_rows


def compare_results(reference, candidate):
    if len(reference) != len(candidate):
        raise AssertionError('Game count mismatch')
    errors = {'value':0., 'old_logp':0., 'bootstrap':0.}
    for (ar, a), (br, b) in zip(reference, candidate):
        for key in set(a)-{'trace','value_mean','terminal_value_mse'}:
            if a[key] != b[key]:
                raise AssertionError(f'Game identity/outcome mismatch: {key}')
        if len(a['trace']) != len(b['trace']) or len(ar) != len(br):
            raise AssertionError('Decision/row count mismatch')
        for x,y in zip(a['trace'],b['trace']):
            if x[:3] != y[:3] or x[4] != y[4]:
                raise AssertionError('Action/request/features mismatch')
            if x[3] is None or y[3] is None:
                if x[3] != y[3]: raise AssertionError('Controlled seat mismatch')
            else:
                np.testing.assert_allclose(x[3],y[3],rtol=3e-6,atol=3e-6)
        for key in ['value_mean','terminal_value_mse']:
            if a[key] is None or b[key] is None:
                if a[key] != b[key]: raise AssertionError('Value availability mismatch')
            else:
                np.testing.assert_allclose(a[key],b[key],rtol=3e-6,atol=3e-6)
        for x,y in zip(ar,br):
            for key in x:
                if key in errors:
                    errors[key] = max(errors[key],abs(x[key]-y[key]))
                    np.testing.assert_allclose(x[key],y[key],rtol=3e-6,atol=3e-6)
                else:
                    np.testing.assert_array_equal(x[key],y[key],err_msg=key)
    return errors


def optimizer(parameters):
    zeros = {k:jnp.zeros_like(v) for k,v in parameters.items()}
    return zeros, dict(zeros), jnp.int32(0)


def compare_update(parameters, reference, candidate):
    a,ao,am = optimize(parameters,optimizer(parameters),merge_rows(reference),7341)
    b,bo,bm = optimize(parameters,optimizer(parameters),merge_rows(candidate),7341)
    errors = {}
    for name, left,right in [('parameters',a,b),('adam_m',ao[0],bo[0]),('adam_v',ao[1],bo[1])]:
        errors[name] = max(float(np.max(np.abs(np.asarray(left[k])-np.asarray(right[k])))) for k in left)
        for key in left:
            np.testing.assert_allclose(left[key],right[key],rtol=3e-5,atol=3e-6)
    if int(ao[2]) != int(bo[2]):
        raise AssertionError('Optimizer counter mismatch')
    for left,right in zip(am,bm):
        for key in left:
            np.testing.assert_allclose(left[key],right[key],rtol=3e-5,atol=3e-6)
    return {**errors,'steps':int(ao[2]),'reference_hash':parameter_hash(a),'candidate_hash':parameter_hash(b)}


def specs_from_plan(plan):
    jobs = [dict(seed=plan['training_seed'],episode=i,max_turns=plan['max_turns'])
            for i in range(plan['training_games'])]
    for opponent in ['random','base_power']:
        for episode in range(plan['development_pairs_per_opponent']):
            for swap in [False,True]:
                jobs.append(dict(seed=8412026,episode=episode,max_turns=plan['max_turns'],split='development',
                    opponent=opponent,swap=swap,learner_side='p2' if swap else 'p1',actor_seed=7342))
    return jobs


def run(run, plan):
    run = Path(run).resolve()
    run.mkdir(parents=True,exist_ok=False)
    source = {str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    for rel in source:
        target = run/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,target)
    f2 = ROOT/'artifacts/runs/f2-full-training-v2'
    input_hash = digest(f2/'inputs/manifest.json')
    parameters,_,_ = load_checkpoint(f2/'seed-2207/checkpoints',input_hash)
    pointer = json.loads((f2/'seed-2207/checkpoints/latest.json').read_text())
    shutil.copyfile(f2/'seed-2207/checkpoints'/pointer['file'],run/'policy.npz')
    admission = {'runtime':runtime(),'sources':source,'plan':plan,'policy':pointer,'parameter_hash':parameter_hash(parameters)}
    atomic_json(run/'inputs.json',admission)
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled:
        raise ValueError('CPU benchmark requires original CPU FP32 runtime')
    records, references, candidates = [], None, {}
    jobs = specs_from_plan(plan)
    warm = [{**jobs[i],'max_turns':2} for i in range(2)]
    configs = ['serial', *[f'batch-{n}' for n in plan['workers']]]
    try:
        for repetition in range(plan['repetitions']):
            order = configs[repetition:]+configs[:repetition]
            for config in order:
                evidence = run/f'repeat-{repetition}/{config}'
                started = time.perf_counter()
                count = 1 if config == 'serial' else int(config.split('-')[1])
                with Worker(evidence) if config == 'serial' else Pool(evidence,count) as worker:
                    startup = time.perf_counter()-started
                    t0 = time.perf_counter()
                    if config == 'serial':
                        for spec in warm: collect(worker,parameters,**spec)
                    else:
                        worker.collect(parameters,warm)
                    warm_seconds = time.perf_counter()-t0
                    t0 = time.perf_counter()
                    if config == 'serial':
                        results = [collect(worker,parameters,**spec) for spec in jobs]
                        profile = {}
                    else:
                        results,profile = worker.collect(parameters,jobs)
                    elapsed = time.perf_counter()-t0
                if references is None:
                    if config != 'serial': raise AssertionError('Serial reference must be first')
                    references = results
                errors = compare_results(references,results)
                if config not in candidates:
                    candidates[config] = results[:plan['update_parity_games']]
                decisions = sum(report['decisions'] for _,report in results)
                body = {'config':config,'repetition':repetition,'workers':count,'games':len(jobs),
                    'decisions':decisions,'terminals':sum(r['outcome']['kind']=='terminal' for _,r in results),
                    'collection_seconds':elapsed,'decisions_per_second':decisions/elapsed,
                    'startup_seconds':startup,'warmup_seconds':warm_seconds,'profile':profile,
                    'max_float_errors':errors,'games_evidence':[r for _,r in results]}
                atomic_json(evidence/'result.json',body)
                records.append({k:v for k,v in body.items() if k != 'games_evidence'})
                atomic_json(run/'progress.json',{'status':'running','completed_cells':len(records),'cells':len(configs)*plan['repetitions']})
                print(json.dumps(records[-1]),flush=True)
        updates = {config:compare_update(parameters,candidates['serial'],candidate)
                   for config,candidate in candidates.items() if config != 'serial'}
        with Worker(run/'profile') as worker:
            profiler = cProfile.Profile()
            profiler.enable()
            for spec in jobs[:plan['profile_games']]: collect(worker,parameters,**spec)
            profiler.disable()
            profiler.dump_stats(str(run/'serial-profile.pstats'))
            with open(run/'serial-profile.txt','w') as out:
                pstats.Stats(profiler,stream=out).strip_dirs().sort_stats('cumtime').print_stats(35)
        for rel,sha in source.items():
            if digest(ROOT/rel) != sha or digest(run/'source'/rel) != sha:
                raise ValueError(f'Source drift: {rel}')
        summary = {}
        serial_median = float(np.median([r['collection_seconds'] for r in records if r['config']=='serial']))
        for config in configs:
            selected = [r for r in records if r['config']==config]
            median = float(np.median([r['collection_seconds'] for r in selected]))
            summary[config] = {'median_seconds':median,'speedup_vs_serial':serial_median/median,
                'median_decisions_per_second':selected[0]['decisions']/median,
                'min_seconds':min(r['collection_seconds'] for r in selected),'max_seconds':max(r['collection_seconds'] for r in selected)}
        report = {'status':'completed','scope':'fixed-policy throughput and update parity, no learning/strength screen',
            'inputs_sha256':digest(run/'inputs.json'),'records':records,'summary':summary,'update_parity':updates,
            'games_per_cell':len(jobs),'measured_game_executions':len(jobs)*len(records),
            'unique_game_specs':len(jobs),'warmup_executions':2*len(records),'profile_replays':plan['profile_games'],
            'all_actions_requests_features_outcomes_identical':True,
            'limitations':['Short local throughput comparison, not sustained qualification.',
                'Repeated identical games are timing repetitions, not independent learning evidence.',
                'Worker barrier includes serialization, simulator, IPC and Python decoding; not pure simulator time.',
                'Warm collection excludes worker startup, warm-up/compilation, evidence writes and optimizer checks.',
                'Batch float outputs and next updates are tolerance-equivalent, not necessarily byte-identical.']}
        atomic_json(run/'report.json',report)
        return report
    except BaseException as exc:
        atomic_json(run/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--plan',default=str(ROOT/'contracts/throughput-v1/plan.json'))
    args = parser.parse_args()
    run(args.run,json.loads(Path(args.plan).read_text()))
