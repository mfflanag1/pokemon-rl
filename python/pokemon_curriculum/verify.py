"""Fresh exhaustive training/evaluation replay and original-final-update recovery."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import ROOT,admission,atomic_json,digest,load_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_full_train.campaign import value_summary
from pokemon_transfer.learning import optimize_batch
from .collector import Pool
from .common import read,validate_plan,admit,initialize,collect_update,evaluation,summarize


def same(a,b,label):
    # JSON reload converts tuple index pairs to lists.
    from pokemon_full_train.storage import canonical
    if canonical(a)!=canonical(b):raise ValueError('Replay mismatch: '+label)


def state_equal(p,opt,path,h,seed,cohorts,cursors):
    q,other,meta=load_checkpoint(path,h)
    if parameter_hash(p)!=parameter_hash(q) or int(opt[2])!=int(other[2]):raise ValueError('Checkpoint parameter/step mismatch')
    for a,b in [(opt[0],other[0]),(opt[1],other[1])]:
        if any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Checkpoint optimizer mismatch')
    same({k:meta[k] for k in ['seed','cohorts','cursors']},
         {'seed':seed,'cohorts':cohorts,'cursors':cursors},'checkpoint metadata')
    return meta


def verify_totals(report,training,evaluation,analyses):
    same(report['training'],value_summary(training),'aggregate training summary')
    same(report['evaluation'],value_summary(evaluation),'aggregate evaluation summary')
    for key,actual in {
        'training_unavailable_choices':sum(g['unavailable_choices'] for g in training),
        'evaluation_unavailable_choices':sum(g['unavailable_choices'] for g in evaluation),
        'learner_rows_collected':sum(g['learner_rows'] for g in training),
        'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in training),
        'all_seed_curriculum_benefit':all(a['curriculum_benefit']['passed'] for a in analyses.values()),
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}.items():
        same(report[key],actual,'report '+key)


def verify(run,evidence):
    manifest=read(run/'inputs/manifest.json');plan=manifest['plan'];validate_plan(plan);admission(run,manifest)
    bank,weights,supervised=admit(plan);h=digest(run/'inputs/manifest.json');report=read(run/'report.json')
    if report['status']!='passed' or report['input_hash']!=h or report['plan']!=plan or report['interpretation']!=plan['interpretation']:raise ValueError('Bad production report')
    counts=Counter();fresh_analysis={};train_all=[];eval_all=[]
    if set(report['seeds'])!={str(s) for s in plan['seeds']}:raise ValueError('Wrong report seeds')
    with Pool(evidence/'workers',8,bank) as pool:
        for seed in plan['seeds']:
            p,opt=initialize(seed,plan);base=run/str(seed)
            state_equal(p,opt,base/'initial',h,seed,0,{'official':0,'prefix':0})
            initial,_=evaluation(pool,p,plan,seed)
            initial_record=read(base/'initial-evaluation.json')
            same(initial,initial_record['games'],'initial evaluation')
            if initial_record['parameter_hash']!=parameter_hash(p):raise ValueError('Initial evaluation parameter binding')
            eval_all.extend(g for group in initial.values() for g in group)
            if set(report['seeds'][str(seed)]['arms'])!=set(plan['arms']):raise ValueError('Wrong report arms')
            counts['evaluation_games']+=sum(map(len,initial.values()));finals={}
            for arm in plan['arms']:
                p,opt=initialize(seed,plan);cursors={'official':0,'prefix':0};training=[]
                for index in range(48):
                    original=read(base/arm/f'update-{index:03d}.json')
                    if parameter_hash(p)!=original['parameter_hash_before']:raise ValueError('Update initial parameters')
                    if index==47:
                        meta=state_equal(p,opt,base/arm/'before-last',h,seed,index,cursors)
                        same(meta['cursors'],cursors,'checkpoint cursors')
                        # Resume from the saved predecessor, not only the replay's live state.
                        p,opt,_=load_checkpoint(base/arm/'before-last',h)
                        counts['original_next_update_recoveries']+=1
                    batch,ledger,_=collect_update(pool,p,plan,seed,arm,index,cursors,bank,weights)
                    same(ledger,original['ledger'],'complete cohort and selected batch')
                    training.extend(g for group in ledger['games'].values() for g in group)
                    cursors=ledger['next'];counts['training_games']+=sum(map(len,ledger['games'].values()))
                    counts['prefix_replay_decisions']+=sum(g['prefix']['decisions'] for group in ledger['games'].values() for g in group)
                    p,opt,metrics=optimize_batch(p,opt,batch,supervised,derived_seed('f12-update',seed,index),plan['rehearsal_coefficient'])
                    same(metrics,original['metrics'],'all optimizer metrics')
                    if parameter_hash(p)!=original['parameter_hash_after'] or int(opt[2])!=original['optimizer_step']:raise ValueError('Updated state mismatch')
                    counts['updates']+=1;counts['optimizer_steps']+=8;counts['selected_ppo_rows']+=1024
                    counts['selected_prefix_rows']+=ledger['quotas']['prefix']
                    counts['ppo_row_presentations']+=2048;counts['supervised_row_presentations']+=1024
                    if (index+1)%16==0:
                        state_equal(p,opt,base/arm/f'checkpoint-{index+1:03d}',h,seed,index+1,cursors);counts['stage_checkpoints']+=1
                    atomic_json(evidence/'progress.json',{'seed':seed,'arm':arm,'updates':index+1})
                state_equal(p,opt,base/arm/'final',h,seed,48,cursors)
                final,_=evaluation(pool,p,plan,seed)
                final_record=read(base/arm/'final-evaluation.json')
                same(final,final_record['games'],'final evaluation')
                if final_record['parameter_hash']!=parameter_hash(p):raise ValueError('Final evaluation parameter binding')
                summary=report['seeds'][str(seed)]['arms'][arm]
                same(summary['training'],value_summary(training),'arm training summary')
                same({k:summary[k] for k in ['prefix_games','prefix_replay_decisions','learner_rows']},
                    {'prefix_games':sum(g['prefix']['scenario_hash'] is not None for g in training),
                     'prefix_replay_decisions':sum(g['prefix']['decisions'] for g in training),
                     'learner_rows':sum(g['learner_rows'] for g in training)},'arm row accounting')
                train_all.extend(training);eval_all.extend(g for group in final.values() for g in group)
                finals[arm]=final;counts['evaluation_games']+=sum(map(len,final.values()))
            fresh_analysis[str(seed)]=summarize(initial,finals,plan)
            same(fresh_analysis[str(seed)],report['seeds'][str(seed)]['analysis'],'all registered statistics')
    same(report['counts'],plan['expected_counts'],'registered budgets')
    for key in plan['expected_counts']:
        if counts[key]!=plan['expected_counts'][key]:raise ValueError('Verification counts mismatch')
    if counts['training_games']!=report['training']['games'] or counts['prefix_replay_decisions']!=report['prefix_replay_decisions']:raise ValueError('Training count mismatch')
    verify_totals(report,train_all,eval_all,fresh_analysis)
    if counts['original_next_update_recoveries']!=6 or counts['stage_checkpoints']!=18:raise ValueError('Missing checkpoint checks')
    admission(run,manifest);admit(plan)
    result={'status':'passed','report_sha256':digest(run/'report.json'),'input_hash':h,'checks':dict(counts),
        'all_seed_curriculum_benefit':all(a['curriculum_benefit']['passed'] for a in fresh_analysis.values()),
        'scope':'All original training cohorts, suffix policy outputs, same-player GAE, selected rows, Adam updates, stage checkpoints and all evaluation outputs/outcomes replayed; six original-final-update recoveries. Prefix actions excluded from every learner batch.'}
    atomic_json(evidence/'verification.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();print(verify(a.run,a.evidence),flush=True)
