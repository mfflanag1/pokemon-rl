"""Fixed F18 corpus comparison and exhaustive fresh simulator/update replay."""
import argparse
from pathlib import Path
import resource
import time
import numpy as np
from pokemon_full_train.storage import ROOT,freeze,admission,digest,atomic_json,save_checkpoint,load_checkpoint
from pokemon_full.learner import parameter_hash
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from pokemon_balanced.learning import probe,support
from pokemon_balanced.sampling import classify,select as legacy_select
from pokemon_full_train.learning import derived_seed
from pokemon_transfer.common import tactical,retention
from pokemon_switch_audit.coverage import summarize as coverage
from pokemon_curriculum.verify import state_equal
from .common import PLAN,read,validate_plan,admit,initialize,team_keys,compare_control,load_strict_checkpoint
from .collection import generate,combine
from .sampling import PreparedSampler
from .learning import bc_step
from .analysis import gate,paired_accuracy,team_admission


def campaign(run, original=None):
    plan=read(PLAN); validate_plan(plan)
    old,oracle,base_data,base_rows,original_keys,past=admit(plan)
    if original:
        om=read(original/'inputs/manifest.json'); admission(original,om); same(om['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False)
    manifest=freeze(run,plan); h=digest(run/'inputs/manifest.json')
    if original: same(manifest,om,'manifest')
    def evidence(rel,value):
        record(run/rel,value,original/rel if original else None)
    for name,d in [('original-teacher',base_data),('old-supervised',old)]:
        array_record(run/'inputs'/f'{name}.npz',d,original/'inputs'/f'{name}.npz' if original else None)
    evidence('inputs/tactical-oracle.json',oracle)
    begun=time.perf_counter(); t=begun
    additional,games,added_rows,added_inventory=generate(plan['teacher_seed'],plan['additional_games'],'train',plan,run/'additional',original/'additional' if original else None)
    additional_seconds=time.perf_counter()-t
    atomic_json(run/'progress.json',{'stage':'additional-collected','games':len(games),'rows':len(added_rows),'elapsed_seconds':time.perf_counter()-begun})
    t=time.perf_counter()
    fresh,probe_games,probe_rows,probe_inventory=generate(plan['probe_seed'],plan['probe_games'],'development',plan,run/'probe',original/'probe' if original else None)
    probe_seconds=time.perf_counter()-t
    roles=team_admission(original_keys,team_keys(games),team_keys(probe_games),past)
    probe_support=support(fresh,plan)
    expanded,expanded_rows=combine(base_data,additional,base_rows,added_rows)
    cov={arm:coverage(rows,probe_rows) for arm,rows in [('original',base_rows),('expanded',expanded_rows)]}
    evidence('coverage.json',cov); evidence('team-admission.json',roles); evidence('probe-support.json',probe_support)
    array_record(run/'inputs/expanded-teacher.npz',expanded,original/'inputs/expanded-teacher.npz' if original else None)
    datasets={'original':base_data,'expanded':expanded}
    samplers={arm:PreparedSampler(data) for arm,data in datasets.items()}
    labels={arm:classify(data) for arm,data in datasets.items()}
    sampler_parity=0
    for arm in plan['arms']:
        for seed in plan['seeds']:
            for step in [0,2047]:
                key=derived_seed('f16-balanced-sample',seed,step)
                actual,ai=samplers[arm].sample(key);expected,ei=legacy_select(datasets[arm],key)
                if not np.array_equal(ai,ei) or any(not np.array_equal(actual[k],expected[k]) for k in actual):
                    raise ValueError('Prepared sampler differs on production corpus')
                sampler_parity+=1
    updates=0;comparisons=0;recoveries=0;controls=0;update_seconds=0.;results={};value_heads_unchanged=True
    for seed in plan['seeds']:
        ip,io=initialize(seed); initial_t=tactical(ip,oracle)
        initial_prob,initial_metrics=probe(ip,fresh)
        array_record(run/str(seed)/'initial-probe.npz',{'probabilities':initial_prob},original/str(seed)/'initial-probe.npz' if original else None)
        arms={};probabilities={}
        for arm in plan['arms']:
            p,opt=initialize(seed);base=run/str(seed)/arm; obase=original/str(seed)/arm if original else None
            def checkpoint(label,p,opt,step,recover=False):
                nonlocal comparisons,recoveries
                cursors={'bc':step,'arm':arm}
                save_checkpoint(base/label,p,opt,{'seed':seed,'input_hash':h,'cohorts':step,'cursors':cursors})
                if original:
                    load_strict_checkpoint(obase/label,h)
                    state_equal(p,opt,obase/label,h,seed,step,cursors);comparisons+=1
                    if recover:
                        p,opt,_=load_strict_checkpoint(obase/label,h);recoveries+=1
                return p,opt
            checkpoint('initial',p,opt,0)
            for step in range(plan['bc_steps']):
                if step==plan['bc_steps']-1: p,opt=checkpoint('before-last',p,opt,step,True)
                before=parameter_hash(p); t=time.perf_counter()
                p,opt,item=bc_step(p,opt,samplers[arm],old,seed,step)
                update_seconds+=time.perf_counter()-t
                quota=np.bincount(labels[arm][item['teacher_indices']],minlength=4).tolist()
                value_heads_unchanged=value_heads_unchanged and all(np.array_equal(np.asarray(p[k]),np.asarray(ip[k])) for k in ['vw','vb'])
                if quota!=plan['teacher_quota'] or int(opt[2])!=step+1 or not value_heads_unchanged:
                    raise ValueError('Family quota/optimizer/value-head mismatch')
                evidence(str((base/'updates'/f'{step:04d}.json').relative_to(run)),{**item,'family_counts':quota,'before':before,'after':parameter_hash(p),'optimizer_step':int(opt[2])})
                updates+=1
                if (step+1)%512==0:atomic_json(run/'progress.json',{'stage':'training','seed':seed,'arm':arm,'updates':updates,'elapsed_seconds':time.perf_counter()-begun})
            checkpoint('final',p,opt,plan['bc_steps'])
            if arm=='original': controls+=int(compare_control(p,opt,seed,plan))
            summary={}
            for name,data in [('train',datasets[arm]),('probe',fresh)]:
                pr,metrics=probe(p,data);summary[name]=metrics
                array_record(base/(name+'-probabilities.npz'),{'probabilities':pr},obase/(name+'-probabilities.npz') if original else None)
                if name=='probe':probabilities[arm]=pr
            tact=tactical(p,oracle);evidence(str((base/'tactical.json').relative_to(run)),tact)
            summary.update(retention=retention(initial_t,tact,plan['tactical_thresholds']),parameter_hash=parameter_hash(p))
            evidence(str((base/'summary.json').relative_to(run)),summary);arms[arm]=summary
        results[str(seed)]={'initial_probe':initial_metrics,'arms':arms,'gate':gate(arms,plan),
            'comparisons':paired_accuracy(probabilities['original'],probabilities['expanded'],fresh,plan['statistics'],seed)}
        evidence(str(Path(str(seed))/'result.json'),results[str(seed)])
    if updates!=plan['expected_updates'] or controls!=3 or added_inventory['games']+probe_inventory['games']!=plan['expected_new_games']:
        raise ValueError('Fixed count mismatch')
    scientific={'status':'passed','plan':plan,'input_hash':h,'updates':updates,'presentations':updates*128,
        'new_games':plan['expected_new_games'],'original_training_games':512,'expanded_training_games':2048,
        'additional':added_inventory,'probe':probe_inventory,'probe_support':probe_support,'roles':roles,
        'coverage':cov,'seeds':results,'all_seed_gate':all(s['gate']['passed'] for s in results.values()),
        'sampler_parity_draws':sampler_parity,'sealed_original_controls_matched':controls,'value_head_weights_unchanged':value_heads_unchanged,
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    evidence('scientific.json',scientific);admission(run,manifest);admit(plan)
    if original:admission(original,om)
    report={**scientific,'elapsed_seconds':time.perf_counter()-begun,'additional_collection_seconds':additional_seconds,
        'probe_collection_seconds':probe_seconds,'update_seconds_including_compile':update_seconds,
        'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    atomic_json(run/'report.json',report)
    if original:
        if comparisons!=18 or recoveries!=6:raise ValueError('Incomplete checkpoint replay')
        atomic_json(run/'verification.json',{'status':'passed','report_sha256':digest(original/'report.json'),
            'scientific_sha256':digest(original/'scientific.json'),'input_hash':h,'new_games':1792,
            'updates':updates,'presentations':updates*128,'checkpoint_comparisons':comparisons,
            'original_next_update_recoveries':recoveries,'sealed_original_controls_matched':controls,
            'all_seed_gate':report['all_seed_gate']})
    print({'status':'passed','elapsed_seconds':report['elapsed_seconds'],'all_seed_gate':report['all_seed_gate']},flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--original',type=Path)
    args=parser.parse_args();campaign(args.run.resolve(),args.original.resolve() if args.original else None)
