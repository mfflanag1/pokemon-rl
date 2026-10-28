import argparse
import json
from pathlib import Path
import resource
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full_train.storage import ROOT, atomic_json, freeze, admission, digest, save_checkpoint
from .data import load_data, predecessor, require, fingerprint
from .learning import model_batch, checked_step, predictions
from .metrics import summarize, gates


def evaluate(parameters, splits):
    summaries={}
    for name,split in splits.items():
        # Equal 128-row inference shape for every panel and all replays.
        probs=[];logs=[]
        for first in range(0,len(split['targets']),128):
            p,l=predictions(parameters,jnp.asarray(split['features'][first:first+128]),jnp.asarray(split['mask'][first:first+128]))
            probs.append(np.asarray(p));logs.append(np.asarray(l))
        p=np.concatenate(probs);l=np.concatenate(logs);targets=split['targets']
        require(np.all(p[~split['mask']]==0),'Probability on unavailable action')
        summaries[name]={**summarize(p,targets,split['contexts']),
            'nll':float(-l[np.arange(len(targets)),targets].mean()),
            'prediction_fingerprint':fingerprint({'probabilities':p,'logps':l})}
    return summaries


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run=Path(args.run);run.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'contracts/decision-fit-v1/plan.json').read_text())
    require(plan['seeds']==[7103,8209,9311] and plan['evaluation_steps']==[0,128,1024] and
            plan['optimizer_steps']==1024 and plan['batch_rows']==128 and plan['learning_rate']==.0003 and plan['max_gradient_norm']==1.,'Plan/kernel mismatch')
    manifest=freeze(run,plan);input_hash=digest(run/'inputs/manifest.json');started=time.perf_counter()
    try:
        predecessor(plan,all_files=True)
        splits,inventory=load_data(plan);batch=model_batch(splits['fit_train'])
        shutil.copyfile(ROOT/plan['dataset'],run/'inputs/probe.json')
        shutil.copyfile(ROOT/plan['predecessor_receipt'],run/'inputs/predecessor-receipt.json')
        atomic_json(run/'inputs/dataset.json',inventory)
        report={'status':'running','input_hash':input_hash,'plan':plan,'dataset':inventory,'seeds':{}}
        for seed in plan['seeds']:
            base=run/str(seed);base.mkdir();p=initialize(seed);z=jax.tree.map(jnp.zeros_like,p);opt=(z,z,jnp.int32(0))
            metadata={'seed':seed,'input_hash':input_hash};evaluations={};checkpoints={};metrics=[]
            for step in range(plan['optimizer_steps']+1):
                if step in plan['evaluation_steps']:
                    evaluations[str(step)]=evaluate(p,splits)
                    checkpoints[str(step)]=save_checkpoint(base/f'step-{step}',p,opt,{**metadata,'cohorts':step})
                    atomic_json(base/f'evaluation-{step}.json',evaluations[str(step)])
                if step==plan['optimizer_steps']:break
                if step==plan['optimizer_steps']-1:
                    save_checkpoint(base/'before-last',p,opt,{**metadata,'cohorts':step})
                p,opt,m=checked_step(p,opt,batch);metrics.append(m)
            require(np.array_equal(p['vw'],np.zeros_like(p['vw'])) and float(p['vb'])==0,'Value head changed')
            atomic_json(base/'updates.json',metrics)
            report['seeds'][str(seed)]={'evaluations':evaluations,'checkpoints':checkpoints,
                'optimizer_steps':int(opt[2]),'gates':gates(evaluations[str(plan['optimizer_steps'])],plan),
                'value_head_unchanged':True}
            admission(run,manifest)
            atomic_json(run/'progress.json',{'completed_seeds':list(report['seeds']),'elapsed':time.perf_counter()-started})
        predecessor(plan,all_files=True)
        report.update(status='completed',primary_fit_passed=all(s['gates']['fit_train']['passed'] for s in report['seeds'].values()),
            seconds=time.perf_counter()-started,optimizer_steps=3*1024,
            training_row_presentations=3*1024*128,evaluation_predictions=3*3*512,new_games=0,
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(run/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['plan','dataset','seeds']}),flush=True)
    except BaseException as e:
        atomic_json(run/'failure.json',{'error':repr(e),'elapsed':time.perf_counter()-started});raise


if __name__=='__main__':main()
