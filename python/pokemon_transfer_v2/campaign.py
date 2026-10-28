"""One fixed full-battle transfer run, with no restart or endpoint selection."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint
from pokemon_full_train.campaign import value_summary
from .collector import Pool
from pokemon_decision.collect import Pool as DecisionPool
from pokemon_retention.common import probe,input_inventory
from pokemon_fit.data import fingerprint,require
from .common import read,predecessor,initialize,collect_update,evaluation,probe_specs,tactical,summarize,verified_inventory
from pokemon_transfer.learning import optimize_batch


def counts(train,evaluations,measured):
    return {'training':{**value_summary(train),'unavailable_choices':sum(g['unavailable_choices'] for g in train)},
        'evaluation':{**value_summary(evaluations),'unavailable_choices':sum(g['unavailable_choices'] for g in evaluations)},**measured}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    plan=read(ROOT/'contracts/full-transfer-v2/plan.json');batch,inventory=predecessor(plan)
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json');start=time.perf_counter()
    report={'status':'running','plan':plan,'input_hash':h,'supervised_inventory':inventory,'supervised_batch_sha256':fingerprint(batch),'seeds':{}}
    train_all=[];eval_all=[]
    measured={k:0 for k in ['selected_ppo_rows','ppo_row_presentations','rehearsal_row_presentations','optimizer_steps','tactical_predictions','mechanics_probe_branches']}
    try:
        for key in ['predecessor_receipt','retention_receipt','failed_attempt_receipt']:shutil.copyfile(ROOT/plan[key],run/'inputs'/f'{key}.json')
        with open(run/'inputs/supervised.npz','xb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in batch.items()})
        with DecisionPool(run/'probe-workers',plan['workers']) as pool:
            oracle=probe(pool,probe_specs(plan));atomic_json(run/'mechanics-probe.json',oracle)
        report['tactical_input_inventory']=verified_inventory(oracle,batch,plan)
        measured['mechanics_probe_branches']=sum(len(o['scores']) for o in oracle['records'])
        with Pool(run/'workers',plan['workers']) as pool:
            for seed in plan['seeds']:
                base=run/str(seed);p,opt=initialize(seed,plan);meta={'seed':seed,'input_hash':h,'cohorts':0,'next_episode':0}
                save_checkpoint(base/'initial',p,opt,meta)
                initial,profile=evaluation(pool,p,plan,seed,base/'initial-chunks');initial_probe=tactical(p,oracle)
                measured['tactical_predictions']+=len(initial_probe['predictions'])
                atomic_json(base/'initial-evaluation.json',{'games':initial,'profiles':profile,'tactical':initial_probe,'parameter_hash':parameter_hash(p)})
                eval_all.extend(g for games in initial.values() for g in games);arms={};finals={};final_probes={}
                for arm in plan['arms']:
                    p,opt=initialize(seed,plan);cursor=0;training=[];t0=time.perf_counter();collection_s=update_s=0.
                    for index in range(plan['updates_per_arm']):
                        if index==plan['updates_per_arm']-1:save_checkpoint(base/arm/'before-last',p,opt,{**meta,'cohorts':index,'next_episode':cursor})
                        before=parameter_hash(p);selected,ledger,profiles=collect_update(pool,p,plan,seed,index,cursor);cursor=ledger['next_episode']
                        collection_s+=sum(v['collection_seconds'] for v in profiles);t=time.perf_counter()
                        p,opt,metrics=optimize_batch(p,opt,selected,batch,derived_seed('f9-update',seed,index),plan['coefficients'][arm]);update_elapsed=time.perf_counter()-t;update_s+=update_elapsed
                        require(len(metrics)==8 and int(opt[2])==(index+1)*8,'Update counter mismatch')
                        measured['optimizer_steps']+=len(metrics);measured['selected_ppo_rows']+=len(selected['actions'])
                        measured['ppo_row_presentations']+=len(metrics)*plan['minibatch_rows']
                        if plan['coefficients'][arm]>0:measured['rehearsal_row_presentations']+=len(metrics)*len(batch['targets'])
                        atomic_json(base/arm/f'update-{index:03d}.json',{'ledger':ledger,'profiles':profiles,'update_seconds':update_elapsed,
                            'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2]),'metrics':metrics})
                        training.extend(ledger['games']);atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'updates':index+1,'games':cursor,'elapsed':time.perf_counter()-start})
                    ck=save_checkpoint(base/arm/'final',p,opt,{**meta,'cohorts':plan['updates_per_arm'],'next_episode':cursor})
                    training_seconds=time.perf_counter()-t0
                    final,profile=evaluation(pool,p,plan,seed,base/arm/'final-chunks');final_probe=tactical(p,oracle)
                    measured['tactical_predictions']+=len(final_probe['predictions'])
                    atomic_json(base/arm/'final-evaluation.json',{'games':final,'profiles':profile,'tactical':final_probe,'parameter_hash':parameter_hash(p)})
                    train_all.extend(training);eval_all.extend(g for games in final.values() for g in games);finals[arm]=final;final_probes[arm]=final_probe
                    arms[arm]={'training':value_summary(training),'optimizer_steps':int(opt[2]),'selected_rows':16384,
                        'excluded_rows':sum(g['decisions'] for g in training)-16384,'final_checkpoint':ck,'training_seconds':training_seconds,
                        'collection_seconds':collection_s,'update_seconds_including_compile':update_s}
                    admission(run,manifest)
                report['seeds'][str(seed)]={'arms':arms,'analysis':summarize(initial,finals,initial_probe,final_probes,plan)}
                atomic_json(run/'progress.json',{'stage':'seed-completed','seed':seed,'elapsed':time.perf_counter()-start})
        admission(run,manifest);predecessor(plan)
        report.update(status='completed',seconds=time.perf_counter()-start,parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            counts=counts(train_all,eval_all,measured),all_seed_rehearsal_benefit=all(s['analysis']['rehearsal_benefit_passed'] for s in report['seeds'].values()))
        require(report['counts']['evaluation']['games']==1728,'Wrong evaluation count')
        atomic_json(run/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k in ['status','seconds','counts','all_seed_rehearsal_benefit']}),flush=True)
    except BaseException as e:
        atomic_json(run/'failure.json',{'error':repr(e),'elapsed':time.perf_counter()-start});raise


if __name__=='__main__':main()
