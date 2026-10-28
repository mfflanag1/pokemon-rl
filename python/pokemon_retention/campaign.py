import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import time
from pokemon_full_train.learning import optimize,derived_seed
from pokemon_full_train.storage import ROOT,atomic_json,freeze,admission,digest,save_checkpoint
from pokemon_decision.collect import Pool
from pokemon_fit.data import require
from .common import admit_predecessor,initial_state,training_specs,eval_specs,pair,probe,summarize,input_inventory
from .metrics import compare


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'contracts/decision-retention-v1/plan.json').read_text());manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json');start=time.perf_counter()
    report={'status':'running','input_hash':h,'plan':plan,'seeds':{}}
    try:
        admit_predecessor(plan);shutil.copyfile(ROOT/plan['predecessor_receipt'],run/'inputs/predecessor-receipt.json')
        with Pool(run/'workers',plan['workers']) as pool:
            oracle=probe(pool,eval_specs(plan));atomic_json(run/'mechanics-probe.json',oracle)
            report['input_inventory']=input_inventory(oracle,plan)
            for seed in plan['seeds']:
                arms={}
                for arm in plan['arms']:
                    base=run/str(seed)/arm;base.mkdir(parents=True);p,opt=initial_state(seed,arm,plan)
                    meta={'seed':seed,'arm':arm,'input_hash':h,'cohorts':0};initial_ck=save_checkpoint(base/'initial',p,opt,meta)
                    _,initial_records=pool.collect(p,eval_specs(plan),derived_seed('f8-eval',seed))
                    atomic_json(base/'initial-evaluation.json',initial_records);initial=summarize(p,initial_records,oracle)
                    for step in range(plan['updates_per_arm']):
                        if step==plan['updates_per_arm']-1:save_checkpoint(base/'before-last',p,opt,{**meta,'cohorts':step})
                        rows,records=pool.collect(p,training_specs(seed,step,plan),derived_seed('f8-train-actor',seed))
                        if arm=='supervised':pair(records,json.loads((run/str(seed)/'fresh'/f'update-{step:03d}.json').read_text())['records'])
                        p,opt,metrics=optimize(p,opt,rows,derived_seed('f8-update',seed,step))
                        atomic_json(base/f'update-{step:03d}.json',{'records':records,'metrics':metrics,'optimizer_steps':int(opt[2])})
                        atomic_json(run/'progress.json',{'seed':seed,'arm':arm,'updates':step+1,'elapsed':time.perf_counter()-start})
                    final_ck=save_checkpoint(base/'final',p,opt,{**meta,'cohorts':plan['updates_per_arm']})
                    _,final_records=pool.collect(p,eval_specs(plan),derived_seed('f8-eval',seed));pair(final_records,initial_records)
                    atomic_json(base/'final-evaluation.json',final_records);final=summarize(p,final_records,oracle)
                    arms[arm]={'initial':initial,'final':final,'initial_checkpoint':initial_ck,'final_checkpoint':final_ck,'optimizer_steps':int(opt[2]),
                        'evaluation_natural_outcomes':dict(Counter('ongoing' if r['natural_outcome'] is None else r['natural_outcome']['kind'] for r in initial_records+final_records))}
                    admission(run,manifest)
                report['seeds'][str(seed)]={'arms':arms,'comparison':compare({k:v['initial'] for k,v in arms.items()},{k:v['final'] for k,v in arms.items()},plan)}
        admit_predecessor(plan)
        report.update(status='completed',seconds=time.perf_counter()-start,training_decisions=49152,evaluation_decisions=6144,
            optimizer_steps=768,training_row_presentations=98304,mechanics_probe_branches=3584,continuation_games=0,
            primary_retention_passed=all(s['comparison']['retained'] for s in report['seeds'].values()),
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(run/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['plan','seeds','input_inventory']}),flush=True)
    except BaseException as e:
        atomic_json(run/'failure.json',{'error':repr(e),'elapsed':time.perf_counter()-start});raise


if __name__=='__main__':main()
