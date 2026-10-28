"""Fresh replay of every F6 decision and every unchanged-PPO update."""
import argparse
from collections import Counter
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full_train.learning import derived_seed, optimize
from pokemon_full_train.storage import admission, atomic_json, digest, load_checkpoint
from .collect import Pool, specs_for, evaluation_specs
from .campaign import summarize, learning_gate


def equal_tree(actual, expected):
    a, at = jax.tree.flatten(actual); b, bt = jax.tree.flatten(expected)
    if at != bt or any(not np.array_equal(x,y) for x,y in zip(a,b)):
        raise ValueError('Checkpoint parameters/Adam state mismatch')


def require_equal(actual, expected, where):
    if actual != expected:
        raise ValueError(f'Fresh replay mismatch: {where}')


def verify_seed_summary(base, summary, optimizer_step, tails):
    require_equal(optimizer_step, summary['optimizer_steps'], 'reported optimizer count')
    require_equal(dict(Counter(r['continuation']['frame']['outcome']['kind'] for r in tails)),
                  summary['continuation_outcomes'], 'reported continuation outcomes')
    for endpoint in ['initial', 'final']:
        require_equal(json.loads((base/endpoint/'latest.json').read_text()),
                      summary[f'{endpoint}_checkpoint'], f'reported {endpoint} checkpoint')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((run/'report.json').read_text());manifest=json.loads((run/'inputs/manifest.json').read_text())
    if report['status']!='completed' or report['input_hash']!=digest(run/'inputs/manifest.json') or report['plan']!=manifest['plan']:
        raise ValueError('Invalid completed run binding')
    admission(run,manifest);plan=report['plan'];counts={'training_decisions':0,'evaluation_decisions':0,'continuation_games':0,'updates':0}
    with Pool(out/'workers',plan['workers']) as pool:
        for seed in plan['seeds']:
            p=initialize(seed);z=jax.tree.map(jnp.zeros_like,p);opt=(z,z,jnp.int32(0));base=run/str(seed)
            stored_p,stored_opt,_=load_checkpoint(base/'initial',report['input_hash']);equal_tree((p,opt),(stored_p,stored_opt))
            specs=evaluation_specs(plan['evaluation_seed'],plan['evaluation_repeats'])
            _,records=pool.collect(p,specs,derived_seed('f6-eval',seed))
            require_equal(records,json.loads((base/'initial-evaluation.json').read_text()),f'{seed}/initial')
            initial=summarize(p,records);counts['evaluation_decisions']+=len(records)
            for step in range(plan['updates_per_seed']):
                if step==plan['updates_per_seed']-1:
                    stored_p,stored_opt,meta=load_checkpoint(base/'before-last',report['input_hash'])
                    equal_tree((p,opt),(stored_p,stored_opt))
                    if meta['cohorts']!=step:raise ValueError('Wrong original-next-update checkpoint')
                    p,opt=stored_p,stored_opt
                rows,records=pool.collect(p,specs_for(seed,step,plan['rows_per_update']),derived_seed('f6-train-actor',seed))
                stored=json.loads((base/f'update-{step:03d}.json').read_text())
                require_equal(records,stored['records'],f'{seed}/update-{step}')
                p,opt,metrics=optimize(p,opt,rows,derived_seed('f6-update',seed,step))
                require_equal(metrics,stored['metrics'],f'{seed}/metrics-{step}')
                require_equal(int(opt[2]),stored['optimizer_steps'],'optimizer counter')
                counts['training_decisions']+=len(records);counts['updates']+=1
            stored_p,stored_opt,_=load_checkpoint(base/'final',report['input_hash']);equal_tree((p,opt),(stored_p,stored_opt))
            _,records=pool.collect(p,specs,derived_seed('f6-eval',seed))
            require_equal(records,json.loads((base/'final-evaluation.json').read_text()),f'{seed}/final')
            final=summarize(p,records);counts['evaluation_decisions']+=len(records)
            for k,v in {'initial':initial,'final':final,'gate':learning_gate(initial,final,plan)}.items():
                require_equal(v,report['seeds'][str(seed)][k],f'{seed}/{k}')
            _,records=pool.collect(p,evaluation_specs(derived_seed('f6-continuation',seed),1),
                derived_seed('f6-continuation-actor',seed),continuation=True)
            require_equal(records,json.loads((base/'continuations.json').read_text()),f'{seed}/continuations')
            verify_seed_summary(base,report['seeds'][str(seed)],int(opt[2]),records)
            counts['continuation_games']+=len(records)
    admission(run,manifest)
    for k in ['training_decisions','evaluation_decisions','continuation_games']:require_equal(counts[k],report[k],k)
    result={'status':'passed','report_sha256':digest(run/'report.json'),'input_hash':report['input_hash'],**counts,
        'scope':'fresh pinned Showdown replay of every local training/evaluation decision and all continuation games; every PPO update; exact original-next-update and final model/Adam fingerprints'}
    atomic_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
