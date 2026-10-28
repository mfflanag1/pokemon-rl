"""One fixed two-arm campaign and exhaustive fresh original-next-update replay."""
import argparse,time,resource
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import ROOT,freeze,admission,digest,atomic_json
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_diversity.sampling import PreparedSampler
from pokemon_diversity.learning import bc_step as legacy_step
from pokemon_balanced.sampling import classify,select as legacy_select
from pokemon_balanced.learning import support
from pokemon_diversity.collection import generate
from pokemon_diversity.common import team_keys
from pokemon_diversity.analysis import paired_accuracy
from pokemon_learning_curve.exposure import Exposure
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from .common import PLAN,read,validate_plan,admit,probe_roles,save,load,same_state,gate
from .learning import initial,selected_batch,step,probe,tactical,mastery,value_keys
from .symmetry import verify

def run(path,original=None):
    plan=read(PLAN);validate_plan(plan);old,oracle,data,rows,train_keys,past=admit(plan)
    if original:
        om=read(original/'inputs/manifest.json');admission(original,om);same(om['plan'],plan,'plan')
    path.mkdir(parents=True,exist_ok=False);manifest=freeze(path,plan);h=digest(path/'inputs/manifest.json')
    if original:same(manifest,om,'manifest')
    def evidence(rel,obj):record(path/rel,obj,original/rel if original else None)
    for name,arrays in [('teacher',data),('old-supervised',old)]:array_record(path/'inputs'/f'{name}.npz',arrays,original/'inputs'/f'{name}.npz' if original else None)
    evidence('inputs/tactical-oracle.json',oracle)
    begun=time.perf_counter();t=begun;fresh,games,_,inventory=generate(plan['probe_seed'],256,'development',plan,path/'probe',original/'probe' if original else None)
    collection_seconds=time.perf_counter()-t;roles=probe_roles(train_keys,past,team_keys(games));ps=support(fresh,plan);evidence('team-admission.json',roles);evidence('probe-support.json',ps)
    sampler=PreparedSampler(data);labels=classify(data);sampler_parity=0
    for seed in plan['seeds']:
        for index in [0,2047,8191]:
            key=derived_seed('f16-balanced-sample',seed,index);a,ai=sampler.sample(key);b,bi=legacy_select(data,key)
            if not np.array_equal(ai,bi) or any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Sampler law changed')
            sampler_parity+=1
    updates=0;comparisons=0;recoveries=0;endpoints_evaluated=0;flat_reference=0;timing={a:0. for a in plan['arms']};results={}
    for seed in plan['seeds']:
        states={a:initial(seed,a) for a in plan['arms']};base=path/str(seed);obase=original/str(seed) if original else None;tracker=Exposure(labels);arms={a:{'endpoints':{}} for a in plan['arms']};probabilities={}
        value_heads={a:{k:np.asarray(states[a][0][k]).copy() for k in value_keys(a)} for a in plan['arms']}
        for arm,(p,opt) in states.items():
            count=sum(v.size for v in p.values());policy_count=count-sum(p[k].size for k in value_keys(arm))
            if count!=plan['parameter_counts'][arm] or policy_count!=plan['policy_parameter_counts'][arm]:raise ValueError('Architecture parameter budget')
            evidence(str(Path(str(seed))/arm/'initialization.json'),{'parameter_hash':parameter_hash(p),'optimizer_step':int(opt[2]),'parameters':count,'policy_parameters':policy_count,'fresh_initialization':True})
        def checkpoint(label,step_number,recover=False):
            nonlocal comparisons,recoveries
            for arm,(p,opt) in states.items():
                save(base/arm/label,p,opt,h,seed,arm,step_number)
                if original:
                    q,qo=load(obase/arm/label,h,seed,arm,step_number);same_state(p,opt,q,qo);comparisons+=1
                    if recover:states[arm]=(q,qo);recoveries+=1
        def evaluate(n):
            nonlocal endpoints_evaluated
            for arm,(p,_) in states.items():
                folder=base/arm/str(n);summary={}
                for split,dataset in [('train',data),('probe',fresh)]:
                    pr,metrics=probe(p,dataset,arm);summary[split]=metrics
                    array_record(folder/(split+'-probabilities.npz'),{'probabilities':pr},obase/arm/str(n)/(split+'-probabilities.npz') if original else None)
                    if split=='probe' and n==8192:probabilities[arm]=pr
                tac=tactical(p,oracle,arm);evidence(str((folder/'tactical.json').relative_to(path)),tac)
                summary.update(parameter_hash=parameter_hash(p),mastery=mastery(tac,plan['mastery_thresholds']),exposure=tracker.summary())
                if arm=='shared':summary['permutation_check']=verify(p,data)
                if summary['exposure']['updates']!=n or summary['exposure']['teacher_draws']!=n*64 or summary['exposure']['rehearsal_draws']!=n*64:raise ValueError('Exposure count')
                evidence(str((folder/'summary.json').relative_to(path)),summary);arms[arm]['endpoints'][str(n)]=summary;endpoints_evaluated+=1
        checkpoint('0',0);evaluate(0)
        for index in range(8192):
            if index in [2047,8191]:checkpoint('before-'+str(index+1),index,True)
            batch,selection=selected_batch(sampler,old,seed,index);quota=np.bincount(labels[selection['teacher_indices']],minlength=4).tolist()
            if quota!=plan['teacher_quota']:raise ValueError('Family quota')
            item={'selection':selection,'family_counts':quota,'arms':{}}
            for arm,(p,opt) in states.items():
                reference=legacy_step(p,opt,sampler,old,seed,index) if arm=='flat' and index==0 else None
                before=parameter_hash(p);t=time.perf_counter();p,opt,metrics=step(p,opt,batch,arm);timing[arm]+=time.perf_counter()-t
                if int(opt[2])!=index+1 or any(not np.array_equal(np.asarray(p[k]),v) for k,v in value_heads[arm].items()):raise ValueError('Optimizer/value-head drift')
                if reference:
                    same_state(p,opt,reference[0],reference[1]);same(metrics,reference[2]['metrics'],'flat metrics')
                    same(selection,{k:reference[2][k] for k in selection},'flat selection');flat_reference+=1
                states[arm]=(p,opt);updates+=1;item['arms'][arm]={'before':before,'after':parameter_hash(p),'optimizer_step':int(opt[2]),'metrics':metrics}
            tracker.add(selection['teacher_indices'],selection['old_indices'],index+1)
            evidence(str(Path(str(seed))/'updates'/f'{index:04d}.json'),item)
            if index+1 in [2048,8192]:checkpoint(str(index+1),index+1);evaluate(index+1)
            if (index+1)%512==0:atomic_json(path/'progress.json',{'seed':seed,'step_per_arm':index+1,'updates':updates,'elapsed_seconds':time.perf_counter()-begun})
        results[str(seed)]={'arms':arms,'gate':gate(arms,plan),'comparisons':paired_accuracy(probabilities['flat'],probabilities['shared'],fresh,plan['statistics'],seed)}
        evidence(str(Path(str(seed))/'result.json'),results[str(seed)])
    if updates!=49152 or flat_reference!=3 or endpoints_evaluated!=18 or sampler_parity!=9:raise ValueError('Fixed comparison inventory')
    scientific={'status':'passed','plan':plan,'input_hash':h,'updates':updates,'presentations':updates*128,'new_games':256,'new_training_games':0,'training_rows':len(data['targets']),'probe':inventory,'probe_support':ps,'roles':roles,'seeds':results,'all_seed_gate':all(s['gate']['passed'] for s in results.values()),'flat_reference_updates':flat_reference,'endpoints_evaluated':endpoints_evaluated,'sampler_parity_draws':sampler_parity,'matched_batches':24576,'value_head_weights_unchanged':True,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    evidence('scientific.json',scientific);admission(path,manifest);admit(plan)
    if original:admission(original,om)
    report={**scientific,'elapsed_seconds':time.perf_counter()-begun,'probe_collection_seconds':collection_seconds,'update_seconds_by_arm_including_compile':timing,'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2};atomic_json(path/'report.json',report)
    if original:
        if comparisons!=30 or recoveries!=12:raise ValueError('Incomplete checkpoint recovery')
        atomic_json(path/'verification.json',{'status':'passed','report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),'input_hash':h,'new_games':256,'updates':49152,'presentations':6291456,'matched_batches':24576,'checkpoint_comparisons':comparisons,'original_next_update_recoveries':recoveries,'flat_reference_updates':flat_reference,'endpoints_evaluated':18,'all_seed_gate':report['all_seed_gate']})
    print({'status':'passed','elapsed_seconds':report['elapsed_seconds'],'all_seed_gate':report['all_seed_gate']},flush=True);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path);a=p.parse_args();run(a.run.resolve(),a.original.resolve() if a.original else None)
