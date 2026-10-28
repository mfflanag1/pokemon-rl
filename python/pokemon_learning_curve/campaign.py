"""One fixed continuation; fresh probe and exhaustive original-next-update replay."""
import argparse,time,resource
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import ROOT,freeze,admission,digest,atomic_json,save_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_balanced.common import initialize
from pokemon_balanced.learning import probe,support
from pokemon_balanced.sampling import classify,select as legacy_select
from pokemon_diversity.sampling import PreparedSampler
from pokemon_diversity.learning import bc_step
from pokemon_diversity.collection import generate
from pokemon_diversity.common import load_strict_checkpoint,team_keys
from pokemon_diversity.analysis import paired_accuracy,gate
from pokemon_switch_audit.coverage import summarize as coverage
from pokemon_transfer.common import tactical,retention
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from pokemon_curriculum.verify import state_equal
from .common import PLAN,read,validate_plan,admit,baseline,probe_roles
from .exposure import Exposure


def run(path,original=None):
    plan=read(PLAN);validate_plan(plan);old,oracle,data,rows,train_keys,past=admit(plan)
    if original:
        om=read(original/'inputs/manifest.json');admission(original,om);same(om['plan'],plan,'plan')
    path.mkdir(parents=True,exist_ok=False);manifest=freeze(path,plan);h=digest(path/'inputs/manifest.json')
    if original:same(manifest,om,'manifest')
    def evidence(rel,value):record(path/rel,value,original/rel if original else None)
    for name,arrays in [('teacher',data),('old-supervised',old)]:
        array_record(path/'inputs'/f'{name}.npz',arrays,original/'inputs'/f'{name}.npz' if original else None)
    evidence('inputs/tactical-oracle.json',oracle)
    begun=time.perf_counter();t=begun
    fresh,games,probe_rows,inventory=generate(plan['probe_seed'],256,'development',plan,path/'probe',original/'probe' if original else None)
    collection_seconds=time.perf_counter()-t;roles=probe_roles(train_keys,past,team_keys(games));probe_support=support(fresh,plan)
    cov=coverage(rows,probe_rows);evidence('coverage.json',cov);evidence('team-admission.json',roles);evidence('probe-support.json',probe_support)
    sampler=PreparedSampler(data);labels=classify(data);sampler_parity=0
    for seed in plan['seeds']:
        for index in [2048,4095,8191]:
            key=derived_seed('f16-balanced-sample',seed,index);a,ai=sampler.sample(key);b,bi=legacy_select(data,key)
            if not np.array_equal(ai,bi) or any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Sampler continuation law drift')
            sampler_parity+=1
    updates=0;comparisons=0;recoveries=0;baseline_matches=0;endpoints_evaluated=0;update_seconds=0.;value_heads_unchanged=True;results={}
    for seed in plan['seeds']:
        p,opt,binding=baseline(seed,plan);evidence(str(Path(str(seed))/'predecessor-checkpoint.json'),binding)
        base=path/str(seed);obase=original/str(seed) if original else None;predecessor=ROOT/plan['predecessor_run']/str(seed)/'expanded'
        initial_p,_=initialize(seed);initial_t=tactical(initial_p,oracle);evidence(str(Path(str(seed))/'retention-reference.json'),initial_t)
        value_head={k:np.asarray(p[k]).copy() for k in ['vw','vb']};tracker=Exposure(labels);endpoints={};probabilities={}
        for index in range(2048):
            previous=read(predecessor/'updates'/f'{index:04d}.json')
            if previous['optimizer_step']!=index+1:raise ValueError('Historical update-index discontinuity')
            tracker.add(previous['teacher_indices'],previous['old_indices'],previous['optimizer_step'])
        def checkpoint(label,p,opt,step,recover=False):
            nonlocal comparisons,recoveries
            cursors={'bc':step};save_checkpoint(base/label,p,opt,{'seed':seed,'input_hash':h,'cohorts':step,'cursors':cursors})
            if original:
                load_strict_checkpoint(obase/label,h);state_equal(p,opt,obase/label,h,seed,step,cursors);comparisons+=1
                if recover:p,opt,_=load_strict_checkpoint(obase/label,h);recoveries+=1
            return p,opt
        def evaluate(step,p):
            nonlocal baseline_matches,endpoints_evaluated
            label=str(step);summary={};folder=base/label
            for name,dataset in [('train',data),('probe',fresh)]:
                pr,metrics=probe(p,dataset);summary[name]=metrics
                array_record(folder/(name+'-probabilities.npz'),{'probabilities':pr},obase/label/(name+'-probabilities.npz') if original else None)
                if name=='probe':probabilities[label]=pr
                if step==2048 and name=='train':
                    with np.load(predecessor/'train-probabilities.npz',allow_pickle=False) as z:
                        if set(z.files)!={'probabilities'} or not np.array_equal(pr,z['probabilities']):raise ValueError('Baseline train predictions differ from F18')
                    same(metrics,read(predecessor/'summary.json')['train'],'F18 baseline training metrics')
            tac=tactical(p,oracle);evidence(str((folder/'tactical.json').relative_to(path)),tac)
            if step==2048:
                same(tac,read(predecessor/'tactical.json'),'F18 baseline tactical predictions');baseline_matches+=1
            summary.update(parameter_hash=parameter_hash(p),retention=retention(initial_t,tac,plan['tactical_thresholds']),exposure=tracker.summary())
            if summary['exposure']['updates']!=step or summary['exposure']['teacher_draws']!=step*64 or summary['exposure']['rehearsal_draws']!=step*64:raise ValueError('Cumulative exposure accounting mismatch')
            evidence(str((folder/'summary.json').relative_to(path)),summary);endpoints[label]=summary;endpoints_evaluated+=1
        checkpoint('2048',p,opt,2048);evaluate(2048,p)
        for index in range(2048,8192):
            if index in [4095,8191]:p,opt=checkpoint('before-'+str(index+1),p,opt,index,True)
            before=parameter_hash(p);t=time.perf_counter();p,opt,item=bc_step(p,opt,sampler,old,seed,index);update_seconds+=time.perf_counter()-t
            quota=np.bincount(labels[item['teacher_indices']],minlength=4).tolist()
            value_heads_unchanged=value_heads_unchanged and all(np.array_equal(np.asarray(p[k]),value_head[k]) for k in value_head)
            if int(opt[2])!=index+1 or quota!=plan['teacher_quota'] or not value_heads_unchanged:raise ValueError('Global step/quota/value-head drift')
            tracker.add(item['teacher_indices'],item['old_indices'],int(opt[2]))
            evidence(str(Path(str(seed))/'updates'/f'{index:04d}.json'),{**item,'before':before,'after':parameter_hash(p),'family_counts':quota,'optimizer_step':int(opt[2])});updates+=1
            if index+1 in [4096,8192]:checkpoint(str(index+1),p,opt,index+1);evaluate(index+1,p)
            if (index+1)%1024==0:atomic_json(path/'progress.json',{'seed':seed,'global_step':index+1,'new_updates':updates,'elapsed_seconds':time.perf_counter()-begun})
        results[str(seed)]={'endpoints':endpoints,
            'gate':gate({'original':endpoints['2048'],'expanded':endpoints['8192']},plan),
            'comparisons':paired_accuracy(probabilities['2048'],probabilities['8192'],fresh,plan['statistics'],seed)}
        evidence(str(Path(str(seed))/'result.json'),results[str(seed)])
    if updates!=18432 or baseline_matches!=3 or endpoints_evaluated!=9 or sampler_parity!=9:raise ValueError('Fixed continuation count mismatch')
    scientific={'status':'passed','plan':plan,'input_hash':h,'new_updates':updates,'new_presentations':updates*128,
        'accumulated_updates':24576,'new_games':256,'new_training_games':0,'training_rows':len(data['targets']),
        'probe':inventory,'probe_support':probe_support,'roles':roles,'coverage':cov,'seeds':results,
        'all_seed_gate':all(s['gate']['passed'] for s in results.values()),'baseline_matches':baseline_matches,
        'endpoints_evaluated':endpoints_evaluated,'sampler_parity_draws':sampler_parity,'value_head_weights_unchanged':value_heads_unchanged,
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    evidence('scientific.json',scientific);admission(path,manifest);admit(plan)
    if original:admission(original,om)
    report={**scientific,'elapsed_seconds':time.perf_counter()-begun,'probe_collection_seconds':collection_seconds,
        'update_seconds_including_compile':update_seconds,'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    atomic_json(path/'report.json',report)
    if original:
        if comparisons!=15 or recoveries!=6:raise ValueError('Incomplete checkpoint recovery')
        atomic_json(path/'verification.json',{'status':'passed','report_sha256':digest(original/'report.json'),
            'scientific_sha256':digest(original/'scientific.json'),'input_hash':h,'new_games':256,'new_updates':18432,
            'new_presentations':2359296,'checkpoint_comparisons':comparisons,'original_next_update_recoveries':recoveries,
            'baseline_matches':baseline_matches,'endpoints_evaluated':endpoints_evaluated,'all_seed_gate':report['all_seed_gate']})
    print({'status':'passed','elapsed_seconds':report['elapsed_seconds'],'all_seed_gate':report['all_seed_gate']},flush=True)
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--original',type=Path)
    a=parser.parse_args();run(a.run.resolve(),a.original.resolve() if a.original else None)
