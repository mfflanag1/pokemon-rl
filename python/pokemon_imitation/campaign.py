"""One registered full-battle imitation/RL screen and exact fresh replay."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import time
import numpy as np
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint,load_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.campaign import value_summary
from pokemon_full.learner import parameter_hash
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from pokemon_curriculum.verify import state_equal
from pokemon_mixture.collector import Pool as TrainPool
from pokemon_transfer_v2.collector import Pool as EvalPool
from pokemon_transfer_v2.common import evaluation
from pokemon_transfer.common import tactical
from pokemon_transfer.learning import optimize_batch
from pokemon_long_transfer.value import summarize_value
from .common import PLAN,read,validate_plan,admit,initialize,team_keys,analysis
from .teacher import generate
from .learning import bc_step,reset_optimizer,teacher_probe
from .collection import collect_update

def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);old,oracle=admit(plan)
    if original:
        original_manifest=read(original/'inputs/manifest.json');admission(original,original_manifest);same(original_manifest['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,original_manifest,'manifest')
    begun=time.perf_counter();counts=Counter();audits=Counter();all_train=[];all_eval=[];seeds={};update_seconds=0.
    teacher,teacher_games,teacher_inventory=generate(plan['teacher_seed'],plan['teacher_games'],'train',plan,run/'teacher',original/'teacher' if original else None)
    audits['teacher_datasets']+=1
    probe,probe_games,probe_inventory=generate(plan['probe_seed'],plan['probe_games'],'development',plan,run/'probe',original/'probe' if original else None)
    counts['teacher_games']=len(teacher_games)+len(probe_games);audits['teacher_datasets']+=1
    if team_keys(teacher_games)&team_keys(probe_games):raise ValueError('Teacher train/probe overlap')
    record(run/'teacher-inventory.json',{'fit':teacher_inventory,'probe':probe_inventory},original/'teacher-inventory.json' if original else None)
    array_record(run/'inputs/old-supervised.npz',old,original/'inputs/old-supervised.npz' if original else None)
    atomic_json(run/'inputs/tactical-oracle.json',oracle)
    with TrainPool(run/'train-workers',8) as train_pool,EvalPool(run/'eval-workers',8) as eval_pool:
        for seed in plan['seeds']:
            base=run/str(seed);obase=original/str(seed) if original else None
            meta={'seed':seed,'input_hash':h};panels={};tacticals={};probes={};arms={}
            p,opt=initialize(seed)
            def checkpoint(label,p,opt,cohorts,cursors,recovery=False):
                save_checkpoint(base/label,p,opt,{**meta,'cohorts':cohorts,'cursors':cursors})
                if original:
                    state_equal(p,opt,obase/label,h,seed,cohorts,cursors);audits['checkpoint_comparisons']+=1
                    if recovery:
                        p,opt,_=load_checkpoint(obase/label,h);audits['original_next_update_recoveries']+=1
                return p,opt
            def evaluate(label,p):
                panel,_=evaluation(eval_pool,p,plan,seed)
                # Same evaluation team coordinates across all seeds/endpoints: check
                # the first initial panel before any BC or RL, and every panel again.
                if team_keys(probe_games)&team_keys([g for gs in panel.values() for g in gs]):raise ValueError('Teacher probe/battle evaluation overlap')
                probs,metrics=teacher_probe(p,probe)
                probability_hash=array_record(base/label/'teacher-probabilities.npz',{'probabilities':probs},obase/label/'teacher-probabilities.npz' if original else None)
                tp=tactical(p,oracle);counts['tactical_predictions']+=len(tp['predictions']);audits['teacher_probe_endpoints']+=1
                record(base/label/'evaluation.json',{'games':panel,'tactical':tp,'teacher_probe':metrics,'teacher_probabilities_fingerprint':probability_hash,'parameter_hash':parameter_hash(p)},obase/label/'evaluation.json' if original else None)
                panels[label]=panel;tacticals[label]=tp;probes[label]=metrics
                all_eval.extend(g for group in panel.values() for g in group)
                atomic_json(run/'progress.json',{'stage':'evaluated','seed':seed,'endpoint':label,'elapsed_seconds':time.perf_counter()-begun})
            checkpoint('initial',p,opt,0,{'fit':0});evaluate('initial',p)
            for step in range(plan['bc_steps']):
                if step==plan['bc_steps']-1:p,opt=checkpoint('bc-before-last',p,opt,step,{'bc':step},True)
                before=parameter_hash(p);t=time.perf_counter();p,opt,item=bc_step(p,opt,teacher,old,seed,step);update_seconds+=time.perf_counter()-t
                if int(opt[2])!=step+1:raise ValueError('BC step count')
                record(base/f'bc/update-{step:04d}.json',{**item,'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])},obase/f'bc/update-{step:04d}.json' if original else None)
                counts['bc_steps']+=1;counts['bc_row_presentations']+=128;counts['new_teacher_row_presentations']+=64;counts['old_bc_row_presentations']+=64
                if (step+1)%256==0:atomic_json(run/'progress.json',{'stage':'imitation','seed':seed,'steps':step+1,'elapsed_seconds':time.perf_counter()-begun})
            checkpoint('imitation',p,opt,plan['bc_steps'],{'bc':plan['bc_steps']});evaluate('imitation',p);imitation=p
            for arm in ['control_rl','imitation_rl']:
                p=initialize(seed)[0] if arm=='control_rl' else imitation;opt=reset_optimizer(p);cursor=0;training=[]
                for index in range(plan['rl_cohorts']):
                    if index==plan['rl_cohorts']-1:p,opt=checkpoint(arm+'/before-last',p,opt,index,{'fit':cursor},True)
                    before=parameter_hash(p);selected,ledger=collect_update(train_pool,p,plan,seed,index,cursor);cursor=ledger['cursor_after']
                    t=time.perf_counter();p,opt,metrics=optimize_batch(p,opt,selected,old,derived_seed('f15-rl-optimizer',seed,index),.1);update_seconds+=time.perf_counter()-t
                    if len(metrics)!=8 or int(opt[2])!=8*(index+1):raise ValueError('RL step count')
                    record(base/arm/f'update-{index:03d}.json',{'ledger':ledger,'metrics':metrics,'parameter_hash_before':before,'parameter_hash_after':parameter_hash(p),'optimizer_step':int(opt[2])},obase/arm/f'update-{index:03d}.json' if original else None)
                    audits['rl_cohorts']+=1;counts['ppo_selected_rows']+=1024;counts['ppo_row_presentations']+=2048;counts['rl_rehearsal_row_presentations']+=1024;counts['rl_optimizer_steps']+=len(metrics)
                    if index==0:first_cohort_value=summarize_value(ledger['games'])
                    training+=ledger['games'];atomic_json(run/'progress.json',{'stage':'rl','seed':seed,'arm':arm,'cohorts':index+1,'games':cursor,'elapsed_seconds':time.perf_counter()-begun})
                checkpoint(arm+'/final',p,opt,plan['rl_cohorts'],{'fit':cursor});evaluate(arm,p)
                all_train+=training;arms[arm]={'first_cohort_terminal_value':first_cohort_value,'training':value_summary(training),'collected_learner_rows':sum(g['learner_rows'] for g in training)}
            seeds[str(seed)]={'panels':{e:{o:value_summary(gs) for o,gs in panel.items()} for e,panel in panels.items()},
                'terminal_value':{e:{o:summarize_value(gs) for o,gs in panel.items()} for e,panel in panels.items()},
                'teacher_probe':probes,'arms':arms,'analysis':analysis(panels,tacticals,plan)}
    counts['evaluation_games']=len(all_eval)
    if dict(counts)!=plan['expected']:raise ValueError('Registered count mismatch '+str(counts))
    train_teams=team_keys(teacher_games+all_train);probe_teams=team_keys(probe_games);eval_teams=team_keys(all_eval)
    if train_teams&(probe_teams|eval_teams) or probe_teams&eval_teams:raise ValueError('Data role team overlap')
    inventory={'train_unique_teams':len(train_teams),'teacher_probe_unique_teams':len(probe_teams),'battle_eval_unique_teams':len(eval_teams),'train_development_overlap':0,'teacher_probe_battle_eval_overlap':0,
        'fit_dataset_fingerprint':teacher_inventory['fingerprint'],'probe_dataset_fingerprint':probe_inventory['fingerprint']}
    report={'status':'passed','input_hash':h,'plan':plan,'counts':dict(counts),'data_inventory':inventory,'teacher':teacher_inventory,'probe':probe_inventory,'training':value_summary(all_train),'evaluation':value_summary(all_eval),
        'training_unavailable_choices':sum(g['unavailable_choices'] for g in all_train),'evaluation_unavailable_choices':sum(g['unavailable_choices'] for g in all_eval),
        'collected_learner_rows':sum(g['learner_rows'] for g in all_train),'seeds':seeds,'all_seed_gates':{k:all(s['analysis'][k] for s in seeds.values()) for k in ['imitation_benefit','pipeline_benefit','incremental_rl_benefit']},
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,'interpretation':plan['interpretation']}
    record(run/'scientific.json',report,original/'scientific.json' if original else None)
    admission(run,manifest);admit(plan)
    if original:admission(original,original_manifest)
    report.update(elapsed_seconds=time.perf_counter()-begun,update_seconds_including_compile=update_seconds,peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic_json(run/'report.json',report)
    if original:
        expected={'teacher_datasets':2,'teacher_probe_endpoints':12,'rl_cohorts':192,'checkpoint_comparisons':21,'original_next_update_recoveries':9}
        if dict(audits)!=expected:raise ValueError('Incomplete measured verification counts')
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,'report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),'checks':{**dict(counts),**dict(audits)},'all_seed_gates':report['all_seed_gates']})
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path)
    a=p.parse_args();r=campaign(a.run.resolve(),a.original.resolve() if a.original else None);print(r['status'],r['elapsed_seconds'],r['all_seed_gates'],flush=True)
