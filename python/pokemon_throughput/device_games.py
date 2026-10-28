"""Full-game comparison on a matched Python/JAX runtime after probe admission."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.storage import ROOT, atomic_json, digest, runtime
from .benchmark import specs_from_plan, compare_results
from .collector import Pool


def run(output, backend, cpu_run, reference=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_threefry_partitionable',True)
    run=Path(cpu_run)
    inputs=json.loads((run/'inputs.json').read_text())
    plan=inputs['plan']
    report={'status':'running','runtime':runtime(),'cpu_inputs_sha256':digest(run/'inputs.json'),
            'records':[],'workers':8,'sources':inputs['sources']}
    try:
        if jax.default_backend().lower()!=backend.lower() or jax.config.x64_enabled:
            raise ValueError('Wrong device or precision')
        if digest(run/'policy.npz')!=inputs['policy']['sha256']: raise ValueError('Policy drift')
        for rel,sha in inputs['sources'].items():
            if digest(ROOT/rel)!=sha: raise ValueError(f'Source drift: {rel}')
        with np.load(run/'policy.npz',allow_pickle=False) as arrays:
            parameters={k[3:]:jnp.asarray(arrays[k]) for k in arrays.files if k.startswith('p__')}
        jobs=specs_from_plan(plan)
        # No optimizer steps here; every repetition uses precisely the same weights.
        baseline=json.loads((run/'repeat-0/serial/result.json').read_text())['games_evidence']
        reference_reports=[([],r) for r in baseline]
        for repetition in range(plan['repetitions']):
            t0=time.perf_counter()
            with Pool(out/f'repeat-{repetition}',8) as pool:
                startup=time.perf_counter()-t0
                t0=time.perf_counter()
                pool.collect(parameters,[{**jobs[i],'max_turns':2} for i in range(2)])
                warm=time.perf_counter()-t0
                t0=time.perf_counter()
                results,profile=pool.collect(parameters,jobs)
                seconds=time.perf_counter()-t0
            compare_results(reference_reports,[([],r) for _,r in results])
            body={'repetition':repetition,'seconds':seconds,'startup_seconds':startup,'warmup_seconds':warm,
                  'profile':profile,'decisions':sum(r['decisions'] for _,r in results),'games':[r for _,r in results]}
            atomic_json(out/f'repeat-{repetition}.json',body)
            report['records'].append({k:v for k,v in body.items() if k!='games'})
            atomic_json(out/'report.json',report)
            print(json.dumps(report['records'][-1]),flush=True)
        for rel,sha in inputs['sources'].items():
            if digest(ROOT/rel)!=sha: raise ValueError(f'Source drift: {rel}')
        report.update(status='completed',median_seconds=float(np.median([r['seconds'] for r in report['records']])),
                      exact_actions_requests_features_outcomes=True,games_per_repetition=len(jobs))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(out/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=['cpu','METAL'],required=True)
    parser.add_argument('--cpu-run',required=True)
    args=parser.parse_args()
    run(args.output,args.backend,args.cpu_run)
