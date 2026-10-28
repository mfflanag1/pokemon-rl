"""Replay every supervised update, prediction, and saved checkpoint from F7."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full_train.storage import ROOT, admission, atomic_json, digest, load_checkpoint
from .data import load_data, predecessor, require
from .learning import checked_step, model_batch
from .campaign import evaluate
from .metrics import gates


def equal_tree(a, b):
    av,at=jax.tree.flatten(a);bv,bt=jax.tree.flatten(b)
    require(at==bt and all(np.array_equal(x,y) for x,y in zip(av,bv)), 'Checkpoint model/Adam mismatch')


def checkpoint(directory, p, opt, seed, step, input_hash):
    loaded_p,loaded_opt,meta=load_checkpoint(directory,input_hash)
    require(meta=={'seed':seed,'cohorts':step,'optimizer_step':step,'input_hash':input_hash},'Checkpoint metadata mismatch')
    equal_tree((p,opt),(loaded_p,loaded_opt))
    return loaded_p,loaded_opt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((run/'report.json').read_text());manifest=json.loads((run/'inputs/manifest.json').read_text());plan=manifest['plan']
    require(report['status']=='completed' and not (run/'failure.json').exists(),'Run did not complete')
    require(report['input_hash']==digest(run/'inputs/manifest.json'),'Input hash mismatch')
    require(report['plan']==plan==json.loads((ROOT/'contracts/decision-fit-v1/plan.json').read_text()),'Plan mismatch')
    require(plan['seeds']==[7103,8209,9311] and plan['evaluation_steps']==[0,128,1024] and plan['optimizer_steps']==1024,'Wrong fixed experiment')
    admission(run,manifest);predecessor(plan,all_files=True)
    require((run/'inputs/probe.json').read_bytes()==(ROOT/plan['dataset']).read_bytes(),'Archived probe mismatch')
    require((run/'inputs/predecessor-receipt.json').read_bytes()==(ROOT/plan['predecessor_receipt']).read_bytes(),'Archived predecessor mismatch')
    splits,inventory=load_data(plan);batch=model_batch(splits['fit_train'])
    require(inventory==report['dataset']==json.loads((run/'inputs/dataset.json').read_text()),'Dataset inventory mismatch')
    require(set(report['seeds'])=={str(s) for s in plan['seeds']},'Incomplete seed set')
    counts={'optimizer_steps':0,'training_row_presentations':0,'evaluation_predictions':0,'new_games':0}
    for seed in plan['seeds']:
        base=run/str(seed);summary=report['seeds'][str(seed)];p=initialize(seed);z=jax.tree.map(jnp.zeros_like,p);opt=(z,z,jnp.int32(0))
        stored_metrics=json.loads((base/'updates.json').read_text());evaluations={}
        require(len(stored_metrics)==1024,'Wrong update count')
        require(set(summary['evaluations'])==set(summary['checkpoints'])=={'0','128','1024'},'Wrong checkpoint set')
        for step in range(1025):
            if step in plan['evaluation_steps']:
                checkpoint(base/f'step-{step}',p,opt,seed,step,report['input_hash'])
                require(json.loads((base/f'step-{step}/latest.json').read_text())==summary['checkpoints'][str(step)],'Reported checkpoint pointer mismatch')
                evaluations[str(step)]=evaluate(p,splits)
                require(evaluations[str(step)]==summary['evaluations'][str(step)]==json.loads((base/f'evaluation-{step}.json').read_text()),'Prediction/summary replay mismatch')
                counts['evaluation_predictions']+=sum(len(s['targets']) for s in splits.values())
            if step==1024:break
            if step==1023:
                # Resume the actual saved state for the original next update.
                p,opt=checkpoint(base/'before-last',p,opt,seed,step,report['input_hash'])
            p,opt,metrics=checked_step(p,opt,batch)
            require(metrics==stored_metrics[step],f'Update metric replay mismatch at {seed}/{step}')
            counts['optimizer_steps']+=1;counts['training_row_presentations']+=len(batch['targets'])
        require(int(opt[2])==summary['optimizer_steps']==1024,'Reported optimizer count mismatch')
        require(summary['gates']==gates(evaluations['1024'],plan),'Gate mismatch')
        require(summary['value_head_unchanged'] is True and np.all(np.asarray(p['vw'])==0) and float(p['vb'])==0,'Value head mismatch')
    for k,v in counts.items():require(v==report[k],f'Reported {k} mismatch')
    primary=all(s['gates']['fit_train']['passed'] for s in report['seeds'].values())
    require(report['primary_fit_passed'] is primary,'Primary all-seed gate mismatch')
    require(np.isfinite(report['seconds']) and report['seconds']>0 and report['parent_peak_rss_bytes']>0,'Invalid telemetry')
    admission(run,manifest);predecessor(plan,all_files=True)
    result={'status':'passed','primary_fit_passed':primary,'input_hash':report['input_hash'],'report_sha256':digest(run/'report.json'),**counts,
        'scope':'all supervised updates and metrics; all 4608 prediction fingerprints and summaries; exact initial, intermediate, original-next-update and final model/Adam states; source/runtime and sealed F6 admission; no new games'}
    atomic_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
