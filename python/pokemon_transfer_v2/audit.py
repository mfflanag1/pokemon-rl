"""Fresh exact replay of every full battle, selected batch, update and probe."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT,admission,atomic_json,digest,load_checkpoint
from pokemon_full_train.campaign import value_summary
from .collector import Pool
from pokemon_decision.collect import Pool as DecisionPool
from pokemon_retention.common import probe,input_inventory
from pokemon_fit.audit import equal_tree
from pokemon_fit.data import fingerprint,require
from .common import read,predecessor,initialize,collect_update,evaluation,probe_specs,tactical,summarize,verified_inventory
from pokemon_transfer.learning import optimize_batch
from .campaign import counts


def checkpoint(directory,p,opt,seed,cohorts,cursor,h):
    a,b,meta=load_checkpoint(directory,h)
    require(meta=={'seed':seed,'input_hash':h,'cohorts':cohorts,'next_episode':cursor,'optimizer_step':cohorts*8},'Checkpoint counters')
    equal_tree((a,b),(p,opt));return a,b


def verify_eval(path,p,pool,plan,seed,oracle):
    saved=read(path);games,_=evaluation(pool,p,plan,seed);predictions=tactical(p,oracle)
    require(saved['parameter_hash']==parameter_hash(p),'Evaluation policy binding')
    require(saved['games']==games,'Full reference evaluation replay mismatch')
    require(saved['tactical']==predictions,'Tactical prediction mismatch')
    from pokemon_full_learning.campaign import evaluation_specs
    folder=path.parent/('initial-chunks' if path.name.startswith('initial') else 'final-chunks')
    require(len(list(folder.glob('*.json')))==24,'Missing evaluation chunk evidence')
    for opponent in plan['opponents']:
        for first in range(0,32,4):
            chunk=read(folder/f'{opponent}-{first:04d}.json')
            require(chunk=={'specs':evaluation_specs(plan,seed,opponent,first,4),'games':games[opponent][first*2:(first+4)*2],'row_counts':[0]*8},'Per-chunk evaluation evidence mismatch')
    return games,predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    r=read(run/'report.json');m=read(run/'inputs/manifest.json');plan=m['plan'];h=digest(run/'inputs/manifest.json')
    require(r['status']=='completed' and not (run/'failure.json').exists(),'Incomplete production')
    require(r['input_hash']==h and r['plan']==plan==read(ROOT/'contracts/full-transfer-v2/plan.json'),'Input/plan binding')
    require(plan['seeds']==[7103,8209,9311] and plan['arms']==['plain','rehearsal'] and plan['updates_per_arm']==16,'Unregistered budget')
    admission(run,m);batch,inventory=predecessor(plan)
    require(r['supervised_inventory']==inventory and r['supervised_batch_sha256']==fingerprint(batch),'Supervised data binding')
    for key in ['predecessor_receipt','retention_receipt','failed_attempt_receipt']:require((ROOT/plan[key]).read_bytes()==(run/'inputs'/f'{key}.json').read_bytes(),'Archived receipt mismatch')
    with np.load(run/'inputs/supervised.npz',allow_pickle=False) as data:require(set(data.files)==set(batch) and all(np.array_equal(data[k],batch[k]) for k in batch),'Archived supervised input')
    with DecisionPool(out/'probe-workers',plan['workers']) as pool:oracle=probe(pool,probe_specs(plan))
    require(oracle==read(run/'mechanics-probe.json'),'Fresh all-action probe mismatch')
    require(verified_inventory(oracle,batch,plan)==r['tactical_input_inventory'],'Probe overlap mismatch')
    require(set(r['seeds'])==set(map(str,plan['seeds'])),'Missing seed')
    train_all=[];eval_all=[];recovered=0;steps=0;selected_total=0;presentations=0;rehearsed=0;predictions=0
    with Pool(out/'workers',plan['workers']) as pool:
        for seed in plan['seeds']:
            base=run/str(seed);p,opt=initialize(seed,plan);checkpoint(base/'initial',p,opt,seed,0,0,h)
            initial,initial_probe=verify_eval(base/'initial-evaluation.json',p,pool,plan,seed,oracle)
            predictions+=len(initial_probe['predictions'])
            eval_all.extend(g for games in initial.values() for g in games);finals={};final_probes={}
            require(set(r['seeds'][str(seed)]['arms'])==set(plan['arms']),'Missing arm')
            for arm in plan['arms']:
                p,opt=initialize(seed,plan);cursor=0;training=[];summary=r['seeds'][str(seed)]['arms'][arm]
                require(len(list((base/arm).glob('update-*.json')))==plan['updates_per_arm'],'Wrong stored update count')
                for index in range(plan['updates_per_arm']):
                    if index==plan['updates_per_arm']-1:
                        p,opt=checkpoint(base/arm/'before-last',p,opt,seed,index,cursor,h);recovered+=1
                    saved=read(base/arm/f'update-{index:03d}.json');before=parameter_hash(p)
                    selected,ledger,_=collect_update(pool,p,plan,seed,index,cursor);cursor=ledger['next_episode']
                    require(saved['ledger']==ledger,'Whole-game/GAE/selection replay mismatch')
                    p,opt,metrics=optimize_batch(p,opt,selected,batch,derived_seed('f9-update',seed,index),plan['coefficients'][arm])
                    require(saved['parameter_hash_before']==before and saved['parameter_hash_after']==parameter_hash(p),'Update parameter binding')
                    require(saved['optimizer_step']==int(opt[2])==(index+1)*8 and saved['metrics']==metrics,'Update metric/counter mismatch')
                    require(len(metrics)==8,'Wrong minibatch step count');steps+=len(metrics);selected_total+=len(selected['actions']);training.extend(ledger['games'])
                    presentations+=len(metrics)*256
                    if plan['coefficients'][arm]>0:rehearsed+=len(metrics)*len(batch['targets'])
                    atomic_json(out/'progress.json',{'seed':seed,'arm':arm,'updates':index+1,'elapsed':time.perf_counter()-start})
                checkpoint(base/arm/'final',p,opt,seed,16,cursor,h)
                require(read(base/arm/'final/latest.json')==summary['final_checkpoint'],'Final pointer mismatch')
                require(summary['training']==value_summary(training) and summary['optimizer_steps']==128 and summary['selected_rows']==16384,'Training totals mismatch')
                require(summary['excluded_rows']==sum(g['decisions'] for g in training)-16384,'Excluded rows mismatch')
                for key in ['training_seconds','collection_seconds','update_seconds_including_compile']:require(np.isfinite(summary[key]) and summary[key]>0,'Invalid arm telemetry')
                final,final_probe=verify_eval(base/arm/'final-evaluation.json',p,pool,plan,seed,oracle)
                predictions+=len(final_probe['predictions'])
                train_all.extend(training);eval_all.extend(g for games in final.values() for g in games);finals[arm]=final;final_probes[arm]=final_probe
            require(r['seeds'][str(seed)]['analysis']==summarize(initial,finals,initial_probe,final_probes,plan),'Recomputed comparison mismatch')
    total=counts(train_all,eval_all,{'optimizer_steps':steps,'selected_ppo_rows':selected_total,'ppo_row_presentations':presentations,
        'rehearsal_row_presentations':rehearsed,'tactical_predictions':predictions,'mechanics_probe_branches':sum(len(o['scores']) for o in oracle['records'])})
    require(total==r['counts'] and steps==total['optimizer_steps']==768 and selected_total==total['selected_ppo_rows']==98304,'Global counters mismatch')
    require(total['evaluation']['games']==1728 and recovered==6,'Evaluation/recovery counts')
    feedback=0
    for worker in range(plan['workers']):
        source=run/'workers'/f'feedback-{worker}.jsonl'
        original=[json.loads(line) for line in source.read_text().splitlines()]
        replayed=[json.loads(line) for line in (out/'workers'/f'feedback-{worker}.jsonl').read_text().splitlines()]
        require(original==replayed,'Rejected-feedback journal replay mismatch');feedback+=len(original)
        for event in original:
            choice=event['submitted'];side=choice['side']
            require(choice['accepted'] is False and all(text.startswith('[Unavailable choice]') for text in choice['feedback']),'Wrong feedback classification')
            require(event['before']['players'][side]['mask'][choice['action']],'Rejected action was not advertised')
    require(feedback==total['training']['unavailable_choices']+total['evaluation']['unavailable_choices'],'Feedback counter mismatch')
    require(r['all_seed_rehearsal_benefit'] is all(s['analysis']['rehearsal_benefit_passed'] for s in r['seeds'].values()),'Primary result mismatch')
    require(r['seconds']>0 and r['parent_peak_rss_bytes']>0,'Telemetry missing');admission(run,m);predecessor(plan)
    result={'status':'passed','input_hash':h,'report_sha256':digest(run/'report.json'),'counts':total,'original_next_update_checkpoints':recovered,'unavailable_feedback_events_verified':feedback,
        'all_seed_rehearsal_benefit':r['all_seed_rehearsal_benefit'],'seconds':time.perf_counter()-start,
        'scope':'fresh exact unpatched Showdown replay of every train/evaluation game and tactical oracle branch; every complete-GAE batch/selected row, joint update/metric, checkpoint/Adam state and endpoint prediction; paired statistics and gates recomputed; source/runtime and F6/F7/F8 admission'}
    atomic_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
