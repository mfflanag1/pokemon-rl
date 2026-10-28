"""Run one frozen F12 ablation; preserve every seed, arm, update and endpoint."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import time
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_transfer.learning import optimize_batch
from pokemon_full_train.campaign import value_summary
from .collector import Pool
from .common import read,validate_plan,admit,initialize,collect_update,evaluation,summarize


def run_campaign(run):
    plan=read(ROOT/'contracts/curriculum-learning-v1/plan.json');validate_plan(plan)
    bank,weights,supervised=admit(plan)
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json');begun=time.perf_counter()
    report={'status':'running','input_hash':h,'plan':plan,'seeds':{}}
    train_all=[];eval_all=[];counts=Counter();collection_seconds=update_seconds=0.
    with Pool(run/'workers',plan['workers'],bank) as pool:
        for seed in plan['seeds']:
            p,opt=initialize(seed,plan);base=run/str(seed)
            meta={'seed':seed,'input_hash':h,'cohorts':0,'cursors':{'official':0,'prefix':0}}
            save_checkpoint(base/'initial',p,opt,meta)
            initial,profiles=evaluation(pool,p,plan,seed)
            atomic_json(base/'initial-evaluation.json',{'games':initial,'profiles':profiles,'parameter_hash':parameter_hash(p)})
            eval_all.extend(g for games in initial.values() for g in games)
            finals={};arms={}
            for arm in plan['arms']:
                p,opt=initialize(seed,plan);cursors={'official':0,'prefix':0};training=[];arm_started=time.perf_counter()
                arm_collection=arm_updates=0.
                for index in range(plan['updates_per_arm']):
                    if index==plan['updates_per_arm']-1:save_checkpoint(base/arm/'before-last',p,opt,{**meta,'cohorts':index,'cursors':cursors})
                    before=parameter_hash(p)
                    batch,ledger,profiles=collect_update(pool,p,plan,seed,arm,index,cursors,bank,weights)
                    cursors=ledger['next'];seconds=sum(v['collection_seconds'] for v in profiles);arm_collection+=seconds;collection_seconds+=seconds
                    t=time.perf_counter()
                    p,opt,metrics=optimize_batch(p,opt,batch,supervised,derived_seed('f12-update',seed,index),plan['rehearsal_coefficient'])
                    elapsed=time.perf_counter()-t;arm_updates+=elapsed;update_seconds+=elapsed
                    if len(metrics)!=8 or int(opt[2])!=(index+1)*8:raise ValueError('Optimizer budget mismatch')
                    atomic_json(base/arm/f'update-{index:03d}.json',{'ledger':ledger,'profiles':profiles,'metrics':metrics,
                        'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2]),'update_seconds':elapsed})
                    counts['optimizer_steps']+=8;counts['selected_ppo_rows']+=1024;counts['ppo_row_presentations']+=2048
                    counts['supervised_row_presentations']+=1024;counts['selected_prefix_rows']+=ledger['quotas']['prefix']
                    games=[g for group in ledger['games'].values() for g in group];training.extend(games)
                    if (index+1)%plan['checkpoint_every_updates']==0:
                        save_checkpoint(base/arm/f'checkpoint-{index+1:03d}',p,opt,{**meta,'cohorts':index+1,'cursors':cursors})
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'updates':index+1,'cursors':cursors,'elapsed_seconds':time.perf_counter()-begun})
                save_checkpoint(base/arm/'final',p,opt,{**meta,'cohorts':48,'cursors':cursors})
                final,profiles=evaluation(pool,p,plan,seed)
                atomic_json(base/arm/'final-evaluation.json',{'games':final,'profiles':profiles,'parameter_hash':parameter_hash(p)})
                finals[arm]=final;eval_all.extend(g for games in final.values() for g in games);train_all.extend(training)
                arms[arm]={'training':value_summary(training),'prefix_games':sum(g['prefix']['scenario_hash'] is not None for g in training),
                    'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in training),
                    'learner_rows':sum(g['learner_rows'] for g in training),'collection_seconds':arm_collection,
                    'update_seconds_including_compile':arm_updates,'arm_seconds_including_evaluation':time.perf_counter()-arm_started}
                admission(run,manifest)
            report['seeds'][str(seed)]={'arms':arms,'analysis':summarize(initial,finals,plan)}
            atomic_json(run/'progress.json',{'stage':'seed_complete','seed':seed,'elapsed_seconds':time.perf_counter()-begun})
    counts['evaluation_games']=len(eval_all)
    if dict(counts)!=plan['expected_counts']:raise ValueError('Final budget mismatch')
    admission(run,manifest);admit(plan)
    report.update(status='passed',elapsed_seconds=time.perf_counter()-begun,counts=dict(counts),
        training=value_summary(train_all),evaluation=value_summary(eval_all),
        training_unavailable_choices=sum(g['unavailable_choices'] for g in train_all),
        evaluation_unavailable_choices=sum(g['unavailable_choices'] for g in eval_all),
        learner_rows_collected=sum(g['learner_rows'] for g in train_all),prefix_replay_decisions=sum(g['prefix']['decisions'] for g in train_all),
        collection_seconds=collection_seconds,update_seconds_including_compile=update_seconds,
        peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        all_seed_curriculum_benefit=all(s['analysis']['curriculum_benefit']['passed'] for s in report['seeds'].values()),
        final_reserved_data_used=False,public_server_used=False,prime_used=False,interpretation=plan['interpretation'])
    atomic_json(run/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    report=run_campaign(p.parse_args().run);print(report['status'],report['elapsed_seconds'],report['all_seed_curriculum_benefit'],flush=True)
