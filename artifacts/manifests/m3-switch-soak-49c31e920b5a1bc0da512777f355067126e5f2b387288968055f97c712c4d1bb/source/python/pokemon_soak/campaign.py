"""Two-hour CPU stability qualification of the unchanged selected policy/chance loop."""
import argparse
from datetime import datetime,timezone
import json
import logging
import math
import os
from pathlib import Path
import signal
import time
import traceback
import jax
import numpy as np
from pokemon_fixed.campaign import Aggregate,CompileLog,compile_positive_control,memory,identical
from pokemon_showdown.campaign import admission as selected_admission,sealed
from pokemon_showdown.loop import ROOT,Executables,check_report
from pokemon_showdown.checkpoint import Runner,save,load,state
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file
from .gates import validate_summary

PLAN_PATH=ROOT/'contracts/soak-v1/plan.json'
PREDECESSOR='d0c377c6d3e6fe68e27382e68d5e575c092780a0c0056f899eac72ba676e6cc0'


def read(path):return json.loads(path.read_bytes())
def utc():return datetime.now(timezone.utc).isoformat()


def write(path,doc):
    """Publish complete JSON to readers; partial writes remain distinct."""
    temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:f.write(wire(doc)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def admission():
    plan=read(PLAN_PATH);require(plan['predecessor']==PREDECESSOR,'Soak predecessor')
    prior=sealed(read(ROOT/'artifacts/reports/m1-showdown-chance.json'))
    require(prior['content_hash']==PREDECESSOR,'Selected completion receipt')
    for path,h in prior['source_files'].items():
        if not path.startswith('docs/') and path not in ('README.md','package.json','scripts/test.mjs'):
            checked_file(ROOT,path,h)
    for path,h in prior['evidence_files'].items():checked_file(ROOT,path,h)
    selected=selected_admission()
    require(selected==read(ROOT/prior['run']/'admission.json'),'Selected runtime/contract drift')
    paths=[*sorted((ROOT/'python/pokemon_soak').glob('*.py')),PLAN_PATH,ROOT/'scripts/seal-policy-soak.py']
    return {'plan':plan,'predecessor':PREDECESSOR,'selected':selected,
            'qualification_sources':{str(p.relative_to(ROOT)):file_hash(p) for p in paths}}


def fingerprint(tree):
    leaves,structure=jax.tree.flatten(tree)
    return digest({'tree':str(structure),'leaves':[{'shape':list(np.asarray(x).shape),'dtype':np.asarray(x).dtype.str,
                  'hex':np.asarray(x).tobytes().hex()} for x in leaves]})


def counters(r):return {'updates':int(r.learner.version),'next_game':int(r.learner.next_game),
                        'decisions':r.decisions,'executed_moves':r.executed_moves}


def replay(path,r,exe):
    save(path,r);restored=load(path,exe)
    require(state(r)==state(restored),'Saved boundary differs')
    expected,metrics=exe(r.learner,r.seed);metrics=check_report(metrics)
    actual=restored.run();identical(expected,restored.learner);identical(metrics,actual)
    return {'file':path.name,'sha256':file_hash(path),'seed':r.seed,'counters':counters(r),
            'next_learner_fingerprint':fingerprint(expected),'next_metrics_fingerprint':fingerprint(metrics),
            'original_vs_restored_exact':True}


def check_diagnostics(r):
    probability=float(r['minimum_behavior_probability'])
    ratio=float(np.max(r['metrics']['max_ratio_error']))
    require(math.isfinite(probability) and 2**-24*(1-3e-6)<=probability<=1+3e-6,'Behavior law support')
    require(math.isfinite(ratio) and 0<=ratio<=2**24+16,'Update ratio support')
    return probability,ratio


def clock_pair():return time.monotonic(),time.time()


def check_gap(previous,current,limit):
    """Detect suspension even when the host monotonic clock pauses during sleep."""
    mono=current[0]-previous[0];wall=current[1]-previous[1]
    gap=max(mono,wall,abs(wall-mono))
    require(mono>=0 and math.isfinite(gap) and gap<=limit,'Host pause or wall-clock discontinuity exceeded limit')
    return gap


def run_soak(out,exe,compilation,plan,mode):
    smoke=mode=='smoke'
    seconds=plan['smoke']['seconds'] if smoke else plan['qualification_seconds']
    telemetry=plan['smoke']['telemetry_seconds'] if smoke else plan['telemetry_seconds']
    checkpoint=plan['smoke']['checkpoint_seconds'] if smoke else plan['checkpoint_seconds']
    r=Runner(plan['seed'],exe)
    for _ in range(plan['warmup_cycles']):check_diagnostics(r.run())
    replays=[replay(out/'soak-start.json',r,exe)];initial=counters(r);initial_memory=memory()
    write(out/'replays.json',replays)
    baseline=compilation.count;traces=dict(exe.traces)
    a=Aggregate();previous=clock_pair();beg=previous[0];cpu=time.process_time()
    next_log=telemetry;next_checkpoint=checkpoint;periodic=rows=0;max_gap=0.;min_probability=1.;max_ratio=0.
    print(json.dumps({'phase':'soak','mode':mode,'target_seconds':seconds,'started_at':utc(),'pid':os.getpid()}),flush=True)
    with (out/'telemetry.jsonl').open('x') as stream:
        while time.monotonic()-beg<seconds:
            cycle_start=time.perf_counter();report=r.run();cycle_time=time.perf_counter()-cycle_start
            probability,ratio=check_diagnostics(report);min_probability=min(min_probability,probability);max_ratio=max(max_ratio,ratio)
            a.add(report,cycle_time);now=clock_pair();elapsed=now[0]-beg
            max_gap=max(max_gap,check_gap(previous,now,plan['maximum_host_gap_seconds']));previous=now
            require(compilation.count==baseline and exe.traces==traces,'Unexpected compilation during soak')
            if elapsed>=next_log:
                row={'scheduled_seconds':next_log,'elapsed_seconds':elapsed,'cycles':a.cycles,
                     'games':a.games,'decisions':a.decisions,'executed_moves':a.moves,'counters':counters(r),
                     'minimum_exp_behavior_logp':min_probability,'maximum_update_ratio_error':max_ratio,
                     'maximum_host_gap_seconds':max_gap,'unexpected_compilations':compilation.count-baseline,
                     'memory':memory(),'recent_metrics':{k:float(np.mean(v)) for k,v in report['metrics'].items()}}
                stream.write(wire(row).decode()+'\n');stream.flush();write(out/'progress.json',row)
                print(json.dumps({'elapsed_seconds':round(elapsed,2),'cycles':a.cycles,'decisions':a.decisions,
                                 'peak_rss_bytes':row['memory']['peak_process_rss_bytes']}),flush=True)
                rows+=1;next_log+=telemetry
            if elapsed>=next_checkpoint and next_checkpoint<seconds:
                periodic+=1;replays.append(replay(out/f'soak-boundary-{periodic:02d}.json',r,exe))
                write(out/'replays.json',replays);next_checkpoint+=checkpoint
        now=clock_pair();max_gap=max(max_gap,check_gap(previous,now,plan['maximum_host_gap_seconds']));previous=now
    elapsed=time.monotonic()-beg;cpu=time.process_time()-cpu
    require(seconds<=elapsed<=seconds+plan['maximum_host_gap_seconds'],'Timed window overshoot')
    replays.append(replay(out/'soak-final.json',r,exe));write(out/'replays.json',replays)
    max_gap=max(max_gap,check_gap(previous,clock_pair(),plan['maximum_host_gap_seconds']))
    require(compilation.count==baseline and exe.traces==traces,'Unexpected final replay compilation')
    final=counters(r);require(final['updates']-initial['updates']==a.cycles,'Cycle counter mismatch')
    summary=a.report()
    return {'target_seconds':seconds,'elapsed_seconds':elapsed,'process_cpu_seconds':cpu,
            'cycles':a.cycles,'games':a.games,'decisions':a.decisions,'executed_moves':a.moves,
            'cycle_seconds':a.wall,'decisions_per_elapsed_second':a.decisions/elapsed,
            'minimum_exp_behavior_logp':min_probability,'maximum_update_ratio_error':max_ratio,
            'maximum_host_gap_seconds':max_gap,'unexpected_compilations':compilation.count-baseline,
            'warmup_compilation_count':baseline,'traces':traces,'telemetry_rows':rows,
            'periodic_replays':periodic,'replays':len(replays),'device_memory_headroom_certified':False,
            'initial_counters':initial,'final_counters':final,'initial_memory':initial_memory,
            'final_memory':memory(),'peak_process_rss_bytes':memory()['peak_process_rss_bytes'],
            'latency_histogram':summary['latency_histogram'],
            'latency_quantile_upper_bounds_seconds':summary['latency_quantile_upper_bounds_seconds'],
            'diagnostics':summary['diagnostics'],
            'timing_scope':plan['timing']}


def main(directory,smoke=False):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Output must be under artifacts/runs')
    out.mkdir(parents=True,exist_ok=False);mode='smoke' if smoke else 'qualification'
    result={'status':'running','mode':mode,'two_hour_qualification':False,'full_g2':'open','full_g3':'open',
            'prime_used':False,'strength_evaluation':False,'started_at':utc(),'pid':os.getpid()}
    write(out/'report.json',result)
    logger=logging.getLogger('jax._src.dispatch');compilation=CompileLog();old=logger.propagate
    logger.addHandler(compilation);logger.propagate=False
    previous_signals={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(signum,frame):raise InterruptedError(f'Qualification interrupted by signal {signum}')
    for s in previous_signals:signal.signal(s,stop)
    try:
        admitted=admission();write(out/'admission.json',admitted);result['admission_hash']=digest(admitted)
        with jax.log_compiles(True):
            result['compile_control']=compile_positive_control(compilation)
            t=time.perf_counter();exe=Executables();exe.diagnostics();result['compile_seconds']=time.perf_counter()-t
            result['soak']=run_soak(out,exe,compilation,admitted['plan'],mode)
        require(admission()==admitted,'End admission/source/runtime/final reservation drift')
        result['final_reservation_unchanged']=True;result['status']='passed'
        result['two_hour_qualification']=not smoke
        if not smoke:validate_summary(result,admitted['plan'])
    except BaseException as e:
        result.update(status='failed',two_hour_qualification=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:
        logger.removeHandler(compilation);logger.propagate=old
        for s,value in previous_signals.items():signal.signal(s,value)
        result['ended_at']=utc();result['content_hash']=digest(result);write(out/'report.json',result)
        print(json.dumps({'status':result['status'],'mode':mode,'report':str(out/'report.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();main(args.directory,args.smoke)
