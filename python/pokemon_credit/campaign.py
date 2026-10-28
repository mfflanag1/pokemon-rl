"""One registered F13 campaign; fresh verification re-executes every decision/update."""
import argparse
from collections import Counter
from pathlib import Path
import time
import resource
import numpy as np
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint,load_checkpoint,canonical
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_full_train.campaign import value_summary
from pokemon_throughput.collector import merge_rows
from pokemon_fit.data import fingerprint
from pokemon_curriculum.verify import state_equal
from .common import read,admit,initialize,endpoint,specs,batch,diagnostics,outcome_summary
from .collector import Pool
from .bank import generate
from .learning import optimize,switch_probability,gradient_probe
from .statistics import compare

PLAN=ROOT/'contracts/credit-audit-v1/plan.json'

def validate_plan(plan):
    if plan!=read(PLAN):raise ValueError('Plan differs from registered file')
    if (plan['seeds']!=[7103,8209,9311] or plan['cohorts']!=32 or plan['selected_rows']!=256 or
        plan['arms']!={'gae95':.95,'lambda1':1.} or plan['epochs']!=2 or plan['rehearsal_coefficient']!=.1 or
        plan['states_per_family_role']!=12 or plan['evaluation_repeats']!=4 or plan['teacher_games']!=192):raise ValueError('Unregistered budget')
    if any(plan[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):raise ValueError('Reserved scope')

def same(a,b,label):
    if canonical(a)!=canonical(b):raise ValueError('Fresh replay mismatch: '+label)

def record(path,value,original=None):
    if original is not None:same(value,read(original),str(original))
    atomic_json(path,value)

def diagnostic_specs(plan,seed):
    return [dict(seed=derived_seed('f13-diagnostic-game',seed),episode=i,split='train',swap=False,
        max_turns=plan['max_turns'],opponent='tactical',learner_side=['p1','p2'][i%2],
        actor_seed=derived_seed('f13-diagnostic-actor',seed)) for i in range(plan['diagnostic_games_per_endpoint_seed'])]

def diagnostic_stage(run,plan,supervised,original):
    result={};all_games=[]
    with Pool(run/'diagnostic-workers',8,{}) as pool:
        for seed in plan['seeds']:
            rows_by={};entry={}
            for name in plan['diagnostic_endpoints']:
                rows_games,_=pool.collect(endpoint(seed,name,plan),diagnostic_specs(plan,seed));rows=merge_rows(rows_games)
                games=[g for _,g in rows_games];rows_by[name]=rows;all_games+=games
                entry[name]={'games':games,'outcomes':value_summary(games),'targets':diagnostics(rows),
                    'diagnostic_counts':dict(sum((Counter(g['diagnostics']) for g in games),Counter()))}
            rows=rows_by['initial'];probe_rows=[r for r in rows if r['phase']=='move' and r['mask'][:8].any() and r['mask'][8:].any()]
            if len(rows)<256 or not probe_rows:raise ValueError('Insufficient registered diagnostic rows')
            probe={'features':np.stack([r['features'] for r in probe_rows]),'mask':np.stack([r['mask'] for r in probe_rows])}
            entry['fixed_state_switch_probability']={'rows':len(probe_rows),'probe_hash':fingerprint(probe),
                'probabilities':{name:float(switch_probability(endpoint(seed,name,plan),probe['features'],probe['mask']).mean()) for name in plan['diagnostic_endpoints']}}
            ids=np.random.default_rng(derived_seed('f13-gradient-rows',seed)).choice(len(rows),256,replace=False)
            probes={}
            for arm,lam in plan['arms'].items():
                full,_=batch(rows,lam);part={k:a[ids].copy() for k,a in full.items()}
                probes[arm]={'selected_indices':ids.tolist(),'batch_hash':fingerprint(part),
                    'components':gradient_probe(endpoint(seed,'initial',plan),part,supervised,probe)}
            entry['local_gradient_probes']=probes
            record(run/f'diagnostic-{seed}.json',entry,original/f'diagnostic-{seed}.json' if original else None)
            result[str(seed)]=entry
            atomic_json(run/'progress.json',{'stage':'diagnostics','seed':seed})
    return result,all_games

def bank_summary(bank,games):
    groups={}
    for role in ['fit','probe']:
        selected=[s for s in bank if s['origin']['role']==role]
        groups[role]={'states':len(selected),'teacher_episodes':[s['origin']['teacher_episode'] for s in selected],
            'families':dict(Counter(s['families'][0] for s in selected)),
            'turns':[s['origin']['turn'] for s in selected],
            'teams':sorted({t['team_key'] for s in selected for t in s['provenance']['teams']})}
    if set(groups['fit']['teacher_episodes'])&set(groups['probe']['teacher_episodes']):raise ValueError('Fit/probe episode overlap')
    return {'roles':groups,'shared_team_hashes':sorted(set(groups['fit']['teams'])&set(groups['probe']['teams'])),
        'teacher_games':len(games),'teacher_terminals':sum(g['outcome']['kind']=='terminal' for g in games),
        'teacher_cutoffs':sum(g['outcome']['kind']!='terminal' for g in games),
        'interpretation':'Public material advantage is a selection criterion, not proof of winnability. Probe teacher episodes are disjoint; train-partition teams can recur.'}

def collect_update(pool,p,plan,bank,seed,index,cursor,lam):
    results=[];first=cursor
    while sum(len(rows) for rows,_ in results)<256:
        if cursor-first>=plan['maximum_games_per_cohort']:raise ValueError('Insufficient rows within frozen cohort game cap')
        requests=specs(bank,plan,seed,cursor,plan['chunk_games'],'fit')
        if any(pool.workers[0].bank[s['scenario_hash']]['origin']['role']!='fit' for s in requests):raise ValueError('Probe entered training')
        current,_=pool.collect(p,requests);results+=current;cursor+=len(current)
    rows=merge_rows(results);full,error=batch(rows,lam)
    ids=np.random.default_rng(derived_seed('f13-selected',seed,index)).choice(len(rows),256,replace=False)
    selected={k:v[ids].copy() for k,v in full.items()};games=[g for _,g in results]
    ledger={'games':games,'cursor_before':first,'cursor_after':cursor,'learner_rows':len(rows),
        'rows_hash':fingerprint({str(i):np.frombuffer(canonical({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()}),dtype=np.uint8) for i,row in enumerate(rows)}),'full_batch_hash':fingerprint(full),'selected_indices':ids.tolist(),
        'selected_batch_hash':fingerprint(selected),'target_parity_error':error,
        'selected_families':dict(Counter(bank_family(pool,results,rows,i) for i in ids))}
    return selected,ledger

def bank_family(pool,results,rows,index):
    # merge_rows labels each complete game as trajectory 2*ordinal + learner seat.
    game=results[rows[index]['trajectory']//2][1]
    return pool.workers[0].bank[game['prefix']['scenario_hash']]['families'][0]

def evaluation(pool,p,plan,bank,seed):
    requests=specs(bank,plan,seed,0,24*plan['evaluation_repeats'],'probe')
    if any(pool.workers[0].bank[s['scenario_hash']]['origin']['role']!='probe' for s in requests):raise ValueError('Fit entered evaluation')
    results,_=pool.collect(p,requests)
    # Prefix transport uses the train partition. These probe rows are discarded,
    # and can never be forwarded to collect_update or an optimizer.
    games=[g for _,g in results]
    if Counter(g['prefix']['scenario_hash'] for g in games)!=Counter({s['content_hash']:4 for s in bank if s['origin']['role']=='probe'}):raise ValueError('Probe coverage mismatch')
    return {f:[g for g in games if pool.workers[0].bank[g['prefix']['scenario_hash']]['families']==[f]] for f in plan['families']}

def analysis(initial,finals,plan):
    contrasts={}
    for family in plan['families']:
        contrasts[family]={name:compare(a[family],b[family],plan['statistics']) for name,a,b in [
            ('gae95_vs_initial',initial,finals['gae95']),('lambda1_vs_initial',initial,finals['lambda1']),('lambda1_vs_gae95',finals['gae95'],finals['lambda1'])]}
    passed=all(contrasts[f][c]['gate']=='improvement' and contrasts[f][c]['point_interval'][0]>=plan['statistics']['practical_gain']
        for f in plan['families'] for c in ['lambda1_vs_initial','lambda1_vs_gae95'])
    return {'contrasts':contrasts,'lambda1_benefit':passed}

def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);supervised=admit(plan)
    if original:
        original_manifest=read(original/'inputs/manifest.json');admission(original,original_manifest)
        same(original_manifest['plan'],plan,'frozen plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,original_manifest,'source/runtime manifest')
    begun=time.perf_counter();counts=Counter();diagnostic,diagnostic_games=diagnostic_stage(run,plan,supervised,original)
    counts['diagnostic_games']=len(diagnostic_games);counts['virtual_gradient_steps']=sum(len(probe['components']) for entry in diagnostic.values() for probe in entry['local_gradient_probes'].values())
    bank,teacher=generate(plan,run/'bank')
    if original:
        same(bank,read(original/'bank/scenarios.json'),'fresh teacher bank');same(teacher,read(original/'bank/teacher-games.json'),'fresh teacher trajectories')
    bank_report=bank_summary(bank,teacher);counts['teacher_games']=len(teacher)
    all_train=[];all_eval=[];seeds={};recovered=0;update_seconds=0.
    with Pool(run/'learning-workers',8,{s['content_hash']:s for s in bank}) as pool:
        for seed in plan['seeds']:
            p,opt=initialize(seed);base=run/str(seed);obase=original/str(seed) if original else None
            meta={'seed':seed,'input_hash':h,'cohorts':0,'cursors':{'fit':0}}
            save_checkpoint(base/'initial',p,opt,meta)
            if original:state_equal(p,opt,obase/'initial',h,seed,0,{'fit':0})
            initial=evaluation(pool,p,plan,bank,seed)
            record(base/'initial-evaluation.json',{'games':initial,'parameter_hash':parameter_hash(p)},obase/'initial-evaluation.json' if original else None)
            all_eval.extend(g for group in initial.values() for g in group);arms={};finals={}
            for arm,lam in plan['arms'].items():
                p,opt=initialize(seed);cursor=0;training=[]
                for index in range(plan['cohorts']):
                    if index==plan['cohorts']-1:
                        save_checkpoint(base/arm/'before-last',p,opt,{**meta,'cohorts':index,'cursors':{'fit':cursor}})
                        if original:
                            state_equal(p,opt,obase/arm/'before-last',h,seed,index,{'fit':cursor})
                            p,opt,_=load_checkpoint(obase/arm/'before-last',h);recovered+=1
                    before=parameter_hash(p);selected,ledger=collect_update(pool,p,plan,bank,seed,index,cursor,lam)
                    cursor=ledger['cursor_after'];t=time.perf_counter();p,opt,metrics=optimize(p,opt,selected,supervised);update_seconds+=time.perf_counter()-t
                    if int(opt[2])!=2*(index+1):raise ValueError('Optimizer count mismatch')
                    value={'ledger':ledger,'metrics':metrics,'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])}
                    record(base/arm/f'update-{index:03d}.json',value,obase/arm/f'update-{index:03d}.json' if original else None)
                    training+=ledger['games'];counts['selected_ppo_rows']+=256;counts['optimizer_steps']+=2
                    counts['ppo_row_presentations']+=512;counts['supervised_row_presentations']+=256
                    atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'cohorts':index+1,'games':cursor,'elapsed_seconds':time.perf_counter()-begun})
                save_checkpoint(base/arm/'final',p,opt,{**meta,'cohorts':32,'cursors':{'fit':cursor}})
                if original:state_equal(p,opt,obase/arm/'final',h,seed,32,{'fit':cursor})
                final=evaluation(pool,p,plan,bank,seed);finals[arm]=final
                record(base/arm/'final-evaluation.json',{'games':final,'parameter_hash':parameter_hash(p)},obase/arm/'final-evaluation.json' if original else None)
                all_train+=training;all_eval.extend(g for group in final.values() for g in group)
                arms[arm]={'training':value_summary(training),'learner_rows_collected':sum(g['learner_rows'] for g in training),
                    'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in training),
                    'evaluation':{f:value_summary(g) for f,g in final.items()}}
            seeds[str(seed)]={'initial':{f:value_summary(g) for f,g in initial.items()},'arms':arms,'analysis':analysis(initial,finals,plan)}
    counts['evaluation_games']=len(all_eval)
    if dict(counts)!=plan['expected']:raise ValueError('Registered budget mismatch '+str(dict(counts)))
    report={'status':'passed','input_hash':h,'plan':plan,'counts':dict(counts),'bank':bank_report,'seeds':seeds,
        'diagnostic_games':value_summary(diagnostic_games),'training':value_summary(all_train),'evaluation':value_summary(all_eval),
        'learner_rows_collected':sum(g['learner_rows'] for g in all_train),
        'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in all_train),
        'unavailable_choices':sum(g['unavailable_choices'] for g in all_train+all_eval+diagnostic_games),
        'all_seed_lambda1_benefit':all(s['analysis']['lambda1_benefit'] for s in seeds.values()),
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,
        'interpretation':plan['interpretation']}
    record(run/'scientific.json',report,original/'scientific.json' if original else None)
    admission(run,manifest);admit(plan)
    if original:admission(original,original_manifest)
    report.update(elapsed_seconds=time.perf_counter()-begun,update_seconds_including_compile=update_seconds,
        peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic_json(run/'report.json',report)
    if original:
        if recovered!=6:raise ValueError('Missing original-next-update recoveries')
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,'report_sha256':digest(original/'report.json'),
            'scientific_sha256':digest(original/'scientific.json'),'checks':{**dict(counts),'original_next_update_recoveries':recovered,
            'initial_checkpoints':3,'final_checkpoints':6,'updates':192},'all_seed_lambda1_benefit':report['all_seed_lambda1_benefit']})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path)
    a=p.parse_args();r=campaign(a.run.resolve(),a.original.resolve() if a.original else None)
    print(r['status'],r['elapsed_seconds'],r['all_seed_lambda1_benefit'],flush=True)
