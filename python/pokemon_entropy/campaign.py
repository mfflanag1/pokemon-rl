"""A single frozen entropy ablation and exhaustive fresh replay of the same budget."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import time
import numpy as np
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint,load_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_full_train.campaign import value_summary
from pokemon_credit.collector import Pool
from pokemon_credit.common import specs
from pokemon_credit.campaign import collect_update,record,same
from pokemon_curriculum.verify import state_equal
from pokemon_fit.data import fingerprint
from pokemon_fit.learning import predictions
from pokemon_transfer.common import tactical
from pokemon_long_transfer.value import summarize_value
from .common import PLAN,read,validate_plan,admit,initialize,analysis
from .learning import optimize
from .behavior import describe

def array_record(path,arrays,original=None):
    if original:
        with np.load(original,allow_pickle=False) as data:
            if set(data.files)!=set(arrays) or any(not np.array_equal(data[k],v) for k,v in arrays.items()):raise ValueError('Array replay mismatch: '+str(original))
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'xb') as out:np.savez_compressed(out,**arrays)
    return fingerprint(arrays)

def evaluation(pool,p,plan,bank,seed):
    requests=specs(bank,plan,seed,0,24*plan['evaluation_repeats'],'probe')
    if any(pool.workers[0].bank[s['scenario_hash']]['origin']['role']!='probe' for s in requests):raise ValueError('Fit entered evaluation')
    results,_=pool.collect(p,requests);games=[g for _,g in results]
    if Counter(g['prefix']['scenario_hash'] for g in games)!=Counter({s['content_hash']:4 for s in bank if s['origin']['role']=='probe'}):raise ValueError('Probe coverage mismatch')
    grouped={f:[g for g in games if pool.workers[0].bank[g['prefix']['scenario_hash']]['families']==[f]] for f in plan['families']}
    # These arrays are observation-only diagnostics; no evaluation rows return
    # to collect_update, and this function never returns rewards/targets as data.
    rows=[row for new,_ in results for row in new if row['phase']=='move' and row['mask'][:8].any() and row['mask'][8:].any()]
    if not rows:raise ValueError('No switch-opportunity probe states')
    probe={'features':np.stack([r['features'] for r in rows]),'mask':np.stack([r['mask'] for r in rows])}
    return grouped,probe

def behavior(p,probe,path,original=None):
    probabilities,_=predictions(p,probe['features'],probe['mask']);probabilities=np.asarray(probabilities)
    h=array_record(path,{'probabilities':probabilities},original)
    return {**describe(probabilities,probe['mask'],['move']*len(probabilities)),
        'probability_fingerprint':h,'observation_fingerprint':fingerprint(probe)}

def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);bank,supervised,oracle=admit(plan)
    if original:
        original_manifest=read(original/'inputs/manifest.json');admission(original,original_manifest);same(original_manifest['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,original_manifest,'source/runtime manifest')
    atomic_json(run/'inputs/bank.json',bank);atomic_json(run/'inputs/tactical-oracle.json',oracle)
    begun=time.perf_counter();counts=Counter();audited=Counter();all_train=[];all_eval=[];seeds={};recovered=0;update_seconds=0.
    with Pool(run/'workers',8,{s['content_hash']:s for s in bank}) as pool:
        for seed in plan['seeds']:
            data_seed=derived_seed('f14-continuation',seed);p,opt=initialize(seed);base=run/str(seed);obase=original/str(seed) if original else None
            meta={'seed':seed,'input_hash':h,'cohorts':0,'cursors':{'fit':0}}
            save_checkpoint(base/'initial',p,opt,meta)
            if original:
                state_equal(p,opt,obase/'initial',h,seed,0,{'fit':0});audited['initial_checkpoints']+=1
            initial,probe=evaluation(pool,p,plan,bank,data_seed)
            array_record(base/'fixed-probe.npz',probe,obase/'fixed-probe.npz' if original else None)
            audited['fixed_observation_sets']+=1
            initial_tactical=tactical(p,oracle);counts['tactical_predictions']+=len(initial_tactical['predictions'])
            behaviors={'initial':behavior(p,probe,base/'behavior-initial.npz',obase/'behavior-initial.npz' if original else None)}
            audited['behavior_endpoints']+=1
            record(base/'initial-evaluation.json',{'games':initial,'tactical':initial_tactical,'behavior':behaviors['initial'],'parameter_hash':parameter_hash(p)},obase/'initial-evaluation.json' if original else None)
            all_eval.extend(g for group in initial.values() for g in group);arms={};finals={};final_tactical={}
            for arm,coefficient in plan['arms'].items():
                p,opt=initialize(seed);cursor=0;training=[]
                for index in range(plan['cohorts']):
                    if index==plan['cohorts']-1:
                        save_checkpoint(base/arm/'before-last',p,opt,{**meta,'cohorts':index,'cursors':{'fit':cursor}})
                        if original:
                            state_equal(p,opt,obase/arm/'before-last',h,seed,index,{'fit':cursor});p,opt,_=load_checkpoint(obase/arm/'before-last',h);recovered+=1
                    before=parameter_hash(p);selected,ledger=collect_update(pool,p,plan,bank,data_seed,index,cursor,.95)
                    cursor=ledger['cursor_after'];t=time.perf_counter();p,opt,metrics=optimize(p,opt,selected,supervised,coefficient);update_seconds+=time.perf_counter()-t
                    if int(opt[2])!=2*(index+1) or len(metrics)!=2:raise ValueError('Optimizer count mismatch')
                    value={'ledger':ledger,'metrics':metrics,'entropy_coefficient':coefficient,'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])}
                    record(base/arm/f'update-{index:03d}.json',value,obase/arm/f'update-{index:03d}.json' if original else None)
                    audited['updates']+=1
                    training+=ledger['games'];counts['selected_ppo_rows']+=len(selected['actions']);counts['optimizer_steps']+=len(metrics)
                    counts['ppo_row_presentations']+=len(metrics)*len(selected['actions']);counts['supervised_row_presentations']+=len(metrics)*len(supervised['targets'])
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'cohorts':index+1,'games':cursor,'elapsed_seconds':time.perf_counter()-begun})
                save_checkpoint(base/arm/'final',p,opt,{**meta,'cohorts':32,'cursors':{'fit':cursor}})
                if original:
                    state_equal(p,opt,obase/arm/'final',h,seed,32,{'fit':cursor});audited['final_checkpoints']+=1
                final,_=evaluation(pool,p,plan,bank,data_seed);finals[arm]=final
                final_tactical[arm]=tactical(p,oracle);counts['tactical_predictions']+=len(final_tactical[arm]['predictions'])
                behaviors[arm]=behavior(p,probe,base/arm/'behavior-final.npz',obase/arm/'behavior-final.npz' if original else None)
                audited['behavior_endpoints']+=1
                record(base/arm/'final-evaluation.json',{'games':final,'tactical':final_tactical[arm],'behavior':behaviors[arm],'parameter_hash':parameter_hash(p)},obase/arm/'final-evaluation.json' if original else None)
                all_train+=training;all_eval.extend(g for group in final.values() for g in group)
                arms[arm]={'training':value_summary(training),'learner_rows_collected':sum(g['learner_rows'] for g in training),
                    'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in training),'evaluation':{f:value_summary(g) for f,g in final.items()},
                    'terminal_value':{f:summarize_value(g) for f,g in final.items()}}
            seeds[str(seed)]={'initial':{f:value_summary(g) for f,g in initial.items()},'initial_value':{f:summarize_value(g) for f,g in initial.items()},
                'arms':arms,'behavior':behaviors,'analysis':analysis(initial,finals,initial_tactical,final_tactical,behaviors,plan)}
    counts['evaluation_games']=len(all_eval)
    if dict(counts)!=plan['expected']:raise ValueError('Registered budget mismatch '+str(dict(counts)))
    report={'status':'passed','input_hash':h,'plan':plan,'counts':dict(counts),'seeds':seeds,
        'bank_sha256':digest(ROOT/plan['bank']),'oracle_sha256':digest(ROOT/plan['oracle']),
        'training':value_summary(all_train),'evaluation':value_summary(all_eval),
        'learner_rows_collected':sum(g['learner_rows'] for g in all_train),'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in all_train),
        'unavailable_choices':sum(g['unavailable_choices'] for g in all_train+all_eval),
        'all_seed_entropy_benefit':all(s['analysis']['entropy_benefit'] for s in seeds.values()),
        'all_seed_switch_reduction':all(s['analysis']['descriptive_switch_reduction_threshold_met'] for s in seeds.values()),
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,'interpretation':plan['interpretation']}
    record(run/'scientific.json',report,original/'scientific.json' if original else None)
    admission(run,manifest);admit(plan)
    if original:admission(original,original_manifest)
    report.update(elapsed_seconds=time.perf_counter()-begun,update_seconds_including_compile=update_seconds,
        peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic_json(run/'report.json',report)
    if original:
        if recovered!=6 or dict(audited)!={'initial_checkpoints':3,'final_checkpoints':6,'updates':192,'fixed_observation_sets':3,'behavior_endpoints':9}:raise ValueError('Missing measured replay checks')
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,'report_sha256':digest(original/'report.json'),
            'scientific_sha256':digest(original/'scientific.json'),'checks':{**dict(counts),'original_next_update_recoveries':recovered,**dict(audited)},
            'all_seed_entropy_benefit':report['all_seed_entropy_benefit']})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path)
    a=p.parse_args();r=campaign(a.run.resolve(),a.original.resolve() if a.original else None)
    print(r['status'],r['elapsed_seconds'],r['all_seed_entropy_benefit'],flush=True)
