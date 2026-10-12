"""Run the predeclared three-seed CPU learning diagnostics into a fresh directory."""
import argparse
import json
from pathlib import Path
import platform
import time
from .trainer import Trainer
from .core import require
from .checkpoint import ROOT, wire, digest, provenance, save


def write(path, value):
    with path.open('xb') as f:f.write(wire(value)+b'\n')


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'contracts/learner-v1/canaries.json').read_text())
    sources=provenance();write(out/'plan.json',{'plan':plan,'plan_hash':digest(plan),'provenance':sources})
    results=[];started=time.monotonic_ns()
    try:
        for task in ['bandit','delayed']:
            for seed in plan['seeds']:
                t=Trainer(task,seed,plan);trace=[];first=None;began=time.monotonic_ns()
                for _ in range(plan[task]['max_updates']):
                    b=t.collect();metrics=t.update(b);probs=t.probabilities()
                    if min(probs)>=plan[task]['required_probability'] and first is None:first=t.decisions
                    trace.append({'update':t.updates,'decisions':t.decisions,'probabilities':probs,'metrics':metrics})
                require(t.decisions<=plan[task]['sample_budget'],'Decision budget exceeded')
                passed=min(t.probabilities())>=plan[task]['required_probability']
                checkpoint_hash=save(out/f'{task}-{seed}.checkpoint.json',t)
                record={'task':task,'seed':seed,'decisions':t.decisions,'updates':t.updates,
                    'probabilities':t.probabilities(),'first_threshold_decisions':first,'passed':passed,
                    'elapsed_ms':(time.monotonic_ns()-began)//1_000_000,'checkpoint_hash':checkpoint_hash}
                write(out/f'{task}-{seed}.trace.json',trace);results.append(record);print(json.dumps(record),flush=True)
        require(provenance()==sources,'Canary source/runtime changed')
        report={'version':'g1-learning-report-1','plan_hash':digest(plan),'provenance':sources,'results':results,
            'status':'passed' if all(r['passed'] for r in results) else 'failed',
            'elapsed_ms':(time.monotonic_ns()-started)//1_000_000,'hardware':platform.platform(),
            'scope':'Synthetic tabular CPU learner only. No Pokemon strength, JAX, GPU or final data.'}
        report['content_hash']=digest(report);write(out/'report.json',report)
        require(report['status']=='passed','Learning threshold failed; investigate before expanding')
        return report
    except BaseException as e:
        write(out/'failure.json',{'error':str(e),'completed_runs':len(results)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');run(p.parse_args().out)
