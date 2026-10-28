"""Full fresh simulator and optimizer replay for the fixed F8 comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
from pokemon_full_train.learning import optimize,derived_seed
from pokemon_full_train.storage import ROOT,admission,atomic_json,digest,load_checkpoint
from pokemon_decision.collect import Pool
from pokemon_fit.data import require
from pokemon_fit.audit import equal_tree
from .common import admit_predecessor,initial_state,training_specs,eval_specs,pair,probe,summarize,input_inventory
from .metrics import compare


def check_checkpoint(base,name,p,opt,seed,arm,step,h):
    loaded,loaded_opt,meta=load_checkpoint(base/name,h)
    require(meta=={'seed':seed,'arm':arm,'cohorts':step,'input_hash':h,'optimizer_step':step*2},'Checkpoint metadata mismatch')
    equal_tree((p,opt),(loaded,loaded_opt))
    return loaded,loaded_opt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    r=json.loads((run/'report.json').read_text());m=json.loads((run/'inputs/manifest.json').read_text());plan=m['plan'];h=digest(run/'inputs/manifest.json')
    require(r['status']=='completed' and not (run/'failure.json').exists() and r['input_hash']==h,'Incomplete run binding')
    require(r['plan']==plan==json.loads((ROOT/'contracts/decision-retention-v1/plan.json').read_text()),'Plan binding')
    require(plan['seeds']==[7103,8209,9311] and plan['arms']==['fresh','supervised'] and plan['updates_per_arm']==64 and plan['rows_per_update']==128 and plan['evaluation_repeats']==16,'Wrong fixed budget')
    require(set(r['seeds'])=={str(s) for s in plan['seeds']},'Missing seed')
    admission(run,m);admit_predecessor(plan)
    require((run/'inputs/predecessor-receipt.json').read_bytes()==(ROOT/plan['predecessor_receipt']).read_bytes(),'Archived F7 receipt mismatch')
    counts={'training_decisions':0,'evaluation_decisions':0,'optimizer_steps':0,'training_row_presentations':0,'mechanics_probe_branches':0,'continuation_games':0}
    comparisons={}
    with Pool(out/'workers',plan['workers']) as pool:
        oracle=probe(pool,eval_specs(plan));atomic_json(out/'mechanics-probe.json',oracle)
        require(oracle==json.loads((run/'mechanics-probe.json').read_text()),'Mechanics branch replay mismatch')
        counts['mechanics_probe_branches']=oracle['branches'];require(input_inventory(oracle,plan)==r['input_inventory'],'Input inventory mismatch')
        for seed in plan['seeds']:
            arms={};stored=r['seeds'][str(seed)];require(set(stored['arms'])==set(plan['arms']),'Missing arm')
            for arm in plan['arms']:
                base=run/str(seed)/arm;p,opt=initial_state(seed,arm,plan);summary=stored['arms'][arm]
                check_checkpoint(base,'initial',p,opt,seed,arm,0,h)
                _,initial_records=pool.collect(p,eval_specs(plan),derived_seed('f8-eval',seed))
                require(initial_records==json.loads((base/'initial-evaluation.json').read_text()),'Initial evaluation replay mismatch')
                initial=summarize(p,initial_records,oracle);counts['evaluation_decisions']+=len(initial_records)
                for step in range(plan['updates_per_arm']):
                    if step==63:p,opt=check_checkpoint(base,'before-last',p,opt,seed,arm,step,h)
                    rows,records=pool.collect(p,training_specs(seed,step,plan),derived_seed('f8-train-actor',seed))
                    saved=json.loads((base/f'update-{step:03d}.json').read_text());require(records==saved['records'],'Training trace mismatch')
                    if arm=='supervised':pair(records,json.loads((run/str(seed)/'fresh'/f'update-{step:03d}.json').read_text())['records'])
                    p,opt,metrics=optimize(p,opt,rows,derived_seed('f8-update',seed,step))
                    require(metrics==saved['metrics'] and int(opt[2])==saved['optimizer_steps']==2*(step+1),'Update metric/counter mismatch')
                    counts['training_decisions']+=len(rows);counts['optimizer_steps']+=len(metrics);counts['training_row_presentations']+=2*len(rows)
                check_checkpoint(base,'final',p,opt,seed,arm,64,h)
                _,final_records=pool.collect(p,eval_specs(plan),derived_seed('f8-eval',seed));pair(final_records,initial_records)
                require(final_records==json.loads((base/'final-evaluation.json').read_text()),'Final evaluation replay mismatch')
                final=summarize(p,final_records,oracle);counts['evaluation_decisions']+=len(final_records)
                require(initial==summary['initial'] and final==summary['final'],'Summary mismatch')
                require(summary['optimizer_steps']==int(opt[2])==128,'Arm optimizer count mismatch')
                for name in ['initial','final']:require(summary[name+'_checkpoint']==json.loads((base/name/'latest.json').read_text()),'Reported pointer mismatch')
                outcomes=dict(Counter('ongoing' if rec['natural_outcome'] is None else rec['natural_outcome']['kind'] for rec in initial_records+final_records))
                require(outcomes==summary['evaluation_natural_outcomes'],'Reported evaluation outcomes mismatch')
                arms[arm]={'initial':initial,'final':final}
                atomic_json(out/'progress.json',{'seed':seed,'arm':arm,**counts})
            comparisons[str(seed)]=compare({k:v['initial'] for k,v in arms.items()},{k:v['final'] for k,v in arms.items()},plan)
            require(comparisons[str(seed)]==stored['comparison'],'Comparison mismatch')
    require(r['primary_retention_passed'] is all(c['retained'] for c in comparisons.values()),'Primary conjunction mismatch')
    for k,v in counts.items():require(v==r[k],f'Reported count mismatch: {k}')
    admission(run,m);admit_predecessor(plan)
    result={'status':'passed','input_hash':h,'report_sha256':digest(run/'report.json'),**counts,
            'primary_retention_passed':r['primary_retention_passed'],
            'scope':'fresh pinned Showdown replay of all 3584 legal evaluation branches, all 49152 training and 6144 evaluation decisions; every PPO/Adam update; exact initialization, original-next-update and final checkpoints; pairing, labels, metrics, source/runtime and F7/F6 admission'}
    atomic_json(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
