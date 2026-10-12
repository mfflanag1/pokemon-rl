"""Continuous CPU qualification of the unchanged switching/categorical kernels."""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import time
import traceback
import jax
import numpy as np
from pokemon_fixed.campaign import CompileLog,compile_positive_control,memory
from pokemon_soak.campaign import write,utc,clock_pair,check_gap
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_switch_engine import learner
from pokemon_switch_engine.scalar import require
from .provenance import ROOT,PLAN,admission,binding
from . import checkpoint


class Aggregate:
    """Bounded host accounting; no trajectory retention across updates."""
    def __init__(self):
        self.cycles=self.decisions=self.terminal=self.truncated=0
        self.actions=np.zeros(4,np.int64);self.seconds=0.;self.low=0.;self.high=-100.
        self.minimum_latency=float('inf');self.maximum_latency=0.;self.metric_sums={}

    def add(self,result,seconds):
        _,buffer,final,metrics=result;valid=np.asarray(buffer.valid);actions=np.asarray(buffer.actions)[valid]
        self.cycles+=1;self.decisions+=len(actions);self.terminal+=int(np.sum(np.asarray(final.winner)!=-1))
        self.truncated+=int(np.sum(np.asarray(final.winner)==-1));self.actions+=np.bincount(actions,minlength=4)
        self.seconds+=seconds;self.minimum_latency=min(self.minimum_latency,seconds);self.maximum_latency=max(self.maximum_latency,seconds)
        logp=np.asarray(buffer.old_logp)[valid];self.low=min(self.low,float(logp.min()));self.high=max(self.high,float(logp.max()))
        for k,v in metrics.items():self.metric_sums[k]=self.metric_sums.get(k,0.)+float(np.mean(np.asarray(v)))*len(actions)

    def counts(self):
        return dict(cycles=self.cycles,games=32*self.cycles,decisions=self.decisions,
                    terminal=self.terminal,truncated=self.truncated,actions=self.actions.tolist())


def replay(path,state,bind,scheduled,elapsed):
    checkpoint.save(path,state,bind);restored=checkpoint.load(path,bind)
    require(checkpoint.fingerprint(state)==checkpoint.fingerprint(restored),'Saved state differs')
    original=checkpoint.result_fingerprints(checkpoint.transaction(state))
    repeated=checkpoint.result_fingerprints(checkpoint.transaction(restored))
    require(original==repeated,'Original next-update replay mismatch')
    return {'file':path.name,'sha256':file_hash(path),'counters':checkpoint.counters(state),
        'state':checkpoint.fingerprint(state),'next':original,'original_vs_restored_exact':True,
        'scheduled_seconds':scheduled,'elapsed_seconds':elapsed}


def run(out,admitted,mode,compilation):
    bind=binding(admitted,mode);smoke=mode=='smoke'
    seconds=PLAN['smoke']['seconds'] if smoke else PLAN['qualification_seconds']
    telemetry=PLAN['smoke']['telemetry_seconds'] if smoke else PLAN['telemetry_seconds']
    interval=PLAN['smoke']['checkpoint_seconds'] if smoke else PLAN['checkpoint_seconds']
    state=learner.start(PLAN['seed'])
    for _ in range(PLAN['warmup_cycles']):state=checkpoint.transaction(state)[0]
    replays=[replay(out/'start.json',state,bind,0,0)];write(out/'replays.json',replays)
    initial=checkpoint.counters(state);initial_memory=memory();baseline=compilation.count
    a=Aggregate();previous=clock_pair();beg=previous[0];cpu=time.process_time()
    next_log=telemetry;next_checkpoint=interval;periodic=rows=0;maxgap=0.
    print(json.dumps({'phase':'timed','mode':mode,'target_seconds':seconds,'started_at':utc(),'pid':os.getpid()}),flush=True)
    with (out/'telemetry.jsonl').open('x') as stream:
        while time.monotonic()-beg<seconds:
            start=time.perf_counter();result=checkpoint.transaction(state);duration=time.perf_counter()-start
            state=result[0];a.add(result,duration);now=clock_pair();elapsed=now[0]-beg
            maxgap=max(maxgap,check_gap(previous,now,PLAN['maximum_host_gap_seconds']));previous=now
            require(compilation.count==baseline,'Unexpected compilation after warm-up')
            if elapsed>=next_log:
                mem=memory();require(mem['peak_process_rss_bytes']<=PLAN['maximum_peak_rss_bytes'],'Peak process RSS budget')
                row={**a.counts(),'scheduled_seconds':next_log,'elapsed_seconds':elapsed,'counters':checkpoint.counters(state),
                    'maximum_host_gap_seconds':maxgap,'unexpected_compilations':compilation.count-baseline,
                    'minimum_logp':a.low,'maximum_logp':a.high,'memory':mem,
                    'recent_metrics':{k:float(np.mean(np.asarray(v))) for k,v in result[3].items()}}
                stream.write(wire(row).decode()+'\n');stream.flush();os.fsync(stream.fileno());write(out/'progress.json',row)
                print(json.dumps({'elapsed_seconds':round(elapsed,2),**a.counts(),'peak_rss_bytes':mem['peak_process_rss_bytes']}),flush=True)
                rows+=1;next_log+=telemetry
            if elapsed>=next_checkpoint and next_checkpoint<seconds:
                periodic+=1;replays.append(replay(out/f'boundary-{periodic:02d}.json',state,bind,next_checkpoint,elapsed))
                write(out/'replays.json',replays);next_checkpoint+=interval
        now=clock_pair();maxgap=max(maxgap,check_gap(previous,now,PLAN['maximum_host_gap_seconds']));previous=now
    elapsed=time.monotonic()-beg;cpu=time.process_time()-cpu
    require(seconds<=elapsed<=seconds+PLAN['maximum_host_gap_seconds'],'Timed window overshoot')
    replays.append(replay(out/'final.json',state,bind,seconds,elapsed));write(out/'replays.json',replays)
    maxgap=max(maxgap,check_gap(previous,clock_pair(),PLAN['maximum_host_gap_seconds']))
    require(compilation.count==baseline,'Unexpected compilation in final replay')
    final_memory=memory();require(final_memory['peak_process_rss_bytes']<=PLAN['maximum_peak_rss_bytes'],'Final RSS budget')
    return {**a.counts(),'elapsed_seconds':elapsed,'process_cpu_seconds':cpu,'cycle_seconds':a.seconds,
        'maximum_host_gap_seconds':maxgap,'unexpected_compilations':compilation.count-baseline,'warmup_compilation_count':baseline,
        'minimum_logp':a.low,'maximum_logp':a.high,'peak_process_rss_bytes':final_memory['peak_process_rss_bytes'],
        'initial_memory':initial_memory,'final_memory':final_memory,'initial_counters':initial,'final_counters':checkpoint.counters(state),
        'telemetry_rows':rows,'replays':len(replays),'decisions_per_elapsed_second':a.decisions/elapsed,
        'cycle_latency_seconds':{'min':a.minimum_latency,'max':a.maximum_latency,'mean':a.seconds/a.cycles},
        'decision_weighted_metrics':{k:v/a.decisions for k,v in a.metric_sums.items()},'device_memory_headroom_certified':False}


def main(directory,smoke=False):
    from .gates import validate
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run directory')
    out.mkdir(parents=True,exist_ok=False);mode='smoke' if smoke else 'qualification'
    result={'status':'running','mode':mode,'two_hour_qualification':False,'started_at':utc(),'pid':os.getpid()}
    for key in ('full_g2','full_g3','full_g4','public_ladder_strength','strength_evaluation','final_evaluation_used','public_server_used','prime_used'):result[key]=PLAN[key]
    write(out/'report.json',result)
    logger=logging.getLogger('jax._src.dispatch');compilation=CompileLog();old=logger.propagate
    logger.addHandler(compilation);logger.propagate=False
    signals={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(signum,frame):raise InterruptedError(f'Qualification interrupted by signal {signum}')
    for s in signals:signal.signal(s,stop)
    try:
        admitted=admission();write(out/'admission.json',admitted);result['admission_hash']=digest(admitted)
        with jax.log_compiles(True):
            result['compile_control']=compile_positive_control(compilation)
            result['soak']=run(out,admitted,mode,compilation)
        require(admission()==admitted,'End source/runtime/reserved-data admission drift')
        result.update(status='passed',two_hour_qualification=not smoke,final_reservation_unchanged=True)
        rows=[json.loads(line) for line in (out/'telemetry.jsonl').read_text().splitlines()]
        result['telemetry_sha256']=file_hash(out/'telemetry.jsonl');result['replays_sha256']=file_hash(out/'replays.json')
        validate(result,PLAN,rows)
    except BaseException as error:
        result.update(status='failed',two_hour_qualification=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        logger.removeHandler(compilation);logger.propagate=old
        for s,v in signals.items():signal.signal(s,v)
        result['ended_at']=utc();result['content_hash']=digest(result);write(out/'report.json',result)
        print(json.dumps({'status':result['status'],'mode':mode,'report':str(out/'report.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    main(args.directory,args.smoke)
