"""One F16 balancing screen, then exact fresh replay without resampling."""
import argparse
from collections import Counter
from pathlib import Path
import time,resource
import numpy as np
from pokemon_full_train.storage import ROOT,freeze,admission,digest,atomic_json,save_checkpoint,load_checkpoint
from pokemon_full_train.campaign import value_summary
from pokemon_full.learner import parameter_hash
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from pokemon_curriculum.verify import state_equal
from pokemon_transfer_v2.collector import Pool
from pokemon_transfer_v2.common import evaluation
from pokemon_transfer.common import tactical
from pokemon_long_transfer.value import summarize_value
from pokemon_imitation.teacher import generate
from pokemon_fit.data import fingerprint
from .common import PLAN,read,validate_plan,admit,initialize,uniform_control,team_keys,analysis
from .learning import bc_step,probe,support,NAMES
from .sampling import classify

def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);old,oracle,data,fit_keys,past_dev=admit(plan)
    if original:
        om=read(original/'inputs/manifest.json');admission(original,om);same(om['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,om,'manifest')
    start=time.perf_counter();update_time=0.;counts=Counter();audits=Counter();all_eval=[];seeds={};presented=Counter()
    labels=classify(data)
    array_record(run/'inputs/teacher.npz',data,original/'inputs/teacher.npz' if original else None)
    array_record(run/'inputs/old-supervised.npz',old,original/'inputs/old-supervised.npz' if original else None)
    atomic_json(run/'inputs/tactical-oracle.json',oracle)
    fresh,probe_games,inventory=generate(plan['probe_seed'],plan['probe_games'],'development',plan,run/'probe',original/'probe' if original else None)
    probe_support=support(fresh,plan);fresh_keys=team_keys(probe_games)
    if fresh_keys&(fit_keys|past_dev):raise ValueError('Fresh probe overlaps fit or F15 development')
    counts['new_teacher_probe_games']=len(probe_games);audits['teacher_datasets']+=1
    record(run/'probe-support.json',probe_support,original/'probe-support.json' if original else None)
    with Pool(run/'eval-workers',plan['workers']) as pool:
        for seed in plan['seeds']:
            base=run/str(seed);obase=original/str(seed) if original else None;panels={};tacticals={};probes={}
            def checkpoint(label,p,opt,steps,recovery=False):
                cursors={'bc':steps};save_checkpoint(base/label,p,opt,{'seed':seed,'input_hash':h,'cohorts':steps,'cursors':cursors})
                if original:
                    state_equal(p,opt,obase/label,h,seed,steps,cursors);audits['checkpoint_comparisons']+=1
                    if recovery:p,opt,_=load_checkpoint(obase/label,h);audits['original_next_update_recoveries']+=1
                return p,opt
            def evaluate(label,p):
                panel,_=evaluation(pool,p,plan,seed);games=[g for group in panel.values() for g in group];keys=team_keys(games)
                if keys&(fresh_keys|fit_keys|past_dev):raise ValueError('Fresh evaluation team role overlap')
                probabilities,metrics=probe(p,fresh)
                ph=array_record(base/label/'teacher-probabilities.npz',{'probabilities':probabilities},obase/label/'teacher-probabilities.npz' if original else None)
                tp=tactical(p,oracle);counts['tactical_predictions']+=len(tp['predictions']);audits['teacher_probe_endpoints']+=1
                record(base/label/'evaluation.json',{'games':panel,'teacher_probe':metrics,'teacher_probabilities_fingerprint':ph,'tactical':tp,'parameter_hash':parameter_hash(p)},obase/label/'evaluation.json' if original else None)
                panels[label]=panel;probes[label]=metrics;tacticals[label]=tp;all_eval.extend(games)
                atomic_json(run/'progress.json',{'stage':'evaluated','seed':seed,'endpoint':label,'elapsed_seconds':time.perf_counter()-start})
            p,opt=initialize(seed);checkpoint('initial',p,opt,0);evaluate('initial',p)
            up,uo=uniform_control(seed,plan);audits['sealed_uniform_control_loads']+=1
            checkpoint('uniform',up,uo,plan['bc_steps']);evaluate('uniform',up)
            for step in range(plan['bc_steps']):
                if step==plan['bc_steps']-1:p,opt=checkpoint('before-last',p,opt,step,True)
                before=parameter_hash(p);t=time.perf_counter();p,opt,item=bc_step(p,opt,data,old,seed,step);update_time+=time.perf_counter()-t
                quota=np.bincount(labels[item['teacher_indices']],minlength=4).tolist()
                if quota!=plan['teacher_quota'] or int(opt[2])!=step+1:raise ValueError('Sampling/optimizer budget drift')
                record(base/f'bc/update-{step:04d}.json',{**item,'family_counts':quota,'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])},obase/f'bc/update-{step:04d}.json' if original else None)
                counts['bc_steps']+=1;counts['bc_row_presentations']+=128;counts['teacher_row_presentations']+=64;counts['old_row_presentations']+=64
                presented.update(dict(zip(NAMES,quota)))
                if (step+1)%256==0:atomic_json(run/'progress.json',{'stage':'balanced-training','seed':seed,'steps':step+1,'elapsed_seconds':time.perf_counter()-start})
            checkpoint('balanced',p,opt,plan['bc_steps']);evaluate('balanced',p)
            seeds[str(seed)]={'panels':{e:{o:value_summary(gs) for o,gs in panel.items()} for e,panel in panels.items()},'terminal_value':{e:{o:summarize_value(gs) for o,gs in panel.items()} for e,panel in panels.items()},'teacher_probe':probes,'analysis':analysis(panels,tacticals,probes,plan)}
    counts['evaluation_games']=len(all_eval)
    if dict(counts)!=plan['expected']:raise ValueError('Count mismatch '+str(counts))
    if dict(presented)!=dict(zip(NAMES,[q*plan['bc_steps']*len(plan['seeds']) for q in plan['teacher_quota']])):raise ValueError('Family presentation mismatch')
    keys=team_keys(all_eval)
    report={'status':'passed','input_hash':h,'plan':plan,'counts':dict(counts),'reused_training_teacher_games':512,'reused_training_teacher_rows':len(data['targets']),'reused_training_dataset_fingerprint':fingerprint(data),'family_presentations':dict(presented),'probe':inventory,'probe_support':probe_support,'evaluation':value_summary(all_eval),'evaluation_unavailable_choices':sum(g['unavailable_choices'] for g in all_eval),'data_inventory':{'fit_teams':len(fit_keys),'fresh_probe_teams':len(fresh_keys),'fresh_evaluation_teams':len(keys),'f15_development_teams_excluded':len(past_dev),'role_overlap':0},'seeds':seeds,'all_seed_gates':{k:all(s['analysis'][k] for s in seeds.values()) for k in ['coverage_screen','strength_gate']},'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,'interpretation':plan['interpretation']}
    record(run/'scientific.json',report,original/'scientific.json' if original else None)
    admission(run,manifest);admit(plan)
    if original:admission(original,om)
    report.update(elapsed_seconds=time.perf_counter()-start,update_seconds_including_compile=update_time,peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic_json(run/'report.json',report)
    if original:
        expected={'teacher_datasets':1,'teacher_probe_endpoints':9,'sealed_uniform_control_loads':3,'checkpoint_comparisons':12,'original_next_update_recoveries':3}
        if dict(audits)!=expected:raise ValueError('Incomplete replay counts '+str(audits))
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,'report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),'checks':{**dict(counts),**dict(audits)},'all_seed_gates':report['all_seed_gates']})
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--original',type=Path);args=a.parse_args()
    r=campaign(args.run.resolve(),args.original.resolve() if args.original else None);print(r['status'],r['elapsed_seconds'],r['all_seed_gates'],flush=True)
