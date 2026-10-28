"""Single uninterrupted two-hour CPU qualification; preflight is explicitly separate."""
import argparse
import logging
import os
from pathlib import Path
import signal
import time
import traceback
import jax
from pokemon_fixed.campaign import CompileLog, compile_positive_control, memory
from pokemon_soak.campaign import clock_pair, check_gap
from pokemon_switch_engine.scalar import require
from . import learner as l, checkpoint as cp, evidence as e


def run(directory, preflight=False):
    out = Path(directory).resolve(); require(out.is_relative_to(l.ROOT/'artifacts/runs'), 'Run directory')
    out.mkdir(exist_ok=False); q = l.PLAN['qualification']; conf = {**q, **(l.PLAN['preflight'] if preflight else {})}
    admitted = e.admission(); e.once(out/'admission.json', admitted)
    result = {'format':'live-training-qualification-1','status':'running','mode':'preflight' if preflight else 'qualification',
        'started_at':e.utc(),'pid':os.getpid(),'admission_hash':e.digest(admitted),**l.PLAN['scope']}
    e.write(out/'report.json',result)
    logger = logging.getLogger('jax._src.dispatch'); compilation = CompileLog(); old = logger.propagate
    logger.addHandler(compilation); logger.propagate = False
    signals = {s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(signum, frame): raise InterruptedError(f'Interrupted by signal {signum}')
    for s in signals: signal.signal(s,stop)
    try:
        with jax.log_compiles(True):
            result['compile_control'] = compile_positive_control(compilation)
            state = l.start(conf['seed']); warm = e.Totals()
            for _ in range(q['warmup_windows']):
                r = l.transaction(state,conf['seed']); warm.add(e.window_summary(state,r)); state = r[0]
            result['warmup'] = warm.document()
            replays = [e.replay(out,'start.json',state,admitted,scalar=True)]
            result['initial_fingerprint'] = cp.fingerprint(state); initial = e.counters(state)
            baseline = compilation.count; initial_memory = memory(); totals = e.Totals()
            start = previous = clock_pair(); cpu_start = time.process_time(); next_log = conf['telemetry_seconds']
            next_cp = conf['checkpoint_seconds']; max_gap = 0.; rows = 0; training_seconds = 0.; replay_seconds = 0.
            memory_baseline = None; last_memory = initial_memory
            with (out/'telemetry.jsonl').open('x') as stream:
                while True:
                    before_time = time.perf_counter(); r = l.transaction(state,conf['seed'])
                    training_seconds += time.perf_counter()-before_time
                    totals.add(e.window_summary(state,r)); state = r[0]
                    if int(state.windows)==q['warmup_windows']+1:
                        when=time.perf_counter();replays.append(e.replay(out,'post-refill.json',state,admitted))
                        replay_seconds+=time.perf_counter()-when
                    now = clock_pair(); elapsed = now[0]-start[0]
                    max_gap = max(max_gap,check_gap(previous,now,q['maximum_host_gap_seconds'])); previous = now
                    require(compilation.count == baseline,'Unexpected compilation after warm-up')
                    if elapsed >= next_log:
                        last_memory = memory(); peak = last_memory['peak_process_rss_bytes']
                        require(peak <= q['maximum_peak_rss_bytes'],'Peak RSS exceeds budget')
                        if memory_baseline is None: memory_baseline = peak
                        require(peak-memory_baseline <= q['maximum_rss_growth_bytes'],'RSS growth exceeds budget')
                        row = {'elapsed_seconds':elapsed,'totals':totals.document(),'counters':e.counters(state),
                            'memory':last_memory,'unexpected_compilations':compilation.count-baseline,'maximum_host_gap_seconds':max_gap}
                        stream.write(e.wire(row).decode()+'\n');stream.flush();os.fsync(stream.fileno())
                        e.write(out/'progress.json',row);print(e.wire({'elapsed_seconds':elapsed,'totals':row['totals']}).decode(),flush=True)
                        rows += 1; next_log += conf['telemetry_seconds']
                    if elapsed >= next_cp and next_cp < conf['seconds']:
                        when = time.perf_counter(); replays.append(e.replay(out,f'boundary-{len(replays)}.json',state,admitted))
                        replay_seconds += time.perf_counter()-when; next_cp += conf['checkpoint_seconds']
                        e.write(out/'replays.json',replays)
                    if elapsed >= conf['seconds'] and int(state.windows)%5 == 0: break
            elapsed = time.monotonic()-start[0]; cpu_seconds = time.process_time()-cpu_start
            require(conf['seconds'] <= elapsed <= conf['seconds']+q['maximum_host_gap_seconds'],'Duration outside admitted bounds')
            replays.append(e.replay(out,'final.json',state,admitted,scalar=True)); e.write(out/'replays.json',replays)
            last_memory = memory(); require(last_memory['peak_process_rss_bytes'] <= q['maximum_peak_rss_bytes'],'Final RSS')
            require(memory_baseline is not None and last_memory['peak_process_rss_bytes']-memory_baseline<=q['maximum_rss_growth_bytes'],'Final RSS growth')
            require(compilation.count == baseline,'Compilation during final recovery probe')
            e.closed_counts(totals,initial,e.counters(state))
            require(all(totals.counts[k]>0 for k in ('updates','post12_decisions','terminal_after12','history_over32_decisions')),'Nonvacuous sustained continuation')
            require(e.admission() == admitted,'Source/runtime/reservation drift')
            result.update(status='passed',two_hour_qualification=not preflight,elapsed_seconds=elapsed,
                training_transaction_seconds=training_seconds,timed_replay_seconds=replay_seconds,process_cpu_seconds=cpu_seconds,
                totals=totals.document(),initial_counters=initial,final_counters=e.counters(state),
                initial_memory=initial_memory,final_memory=last_memory,telemetry_rows=rows,
                unexpected_compilations=compilation.count-baseline,warmup_compilations=baseline,maximum_host_gap_seconds=max_gap,
                checkpoint_count=len(replays),discarded_checkpoint_transactions=2*len(replays),
                final_fingerprint=cp.fingerprint(state),telemetry_sha256=e.file_hash(out/'telemetry.jsonl'),
                replays_sha256=e.file_hash(out/'replays.json'))
    except BaseException as error:
        result.update(status='failed',two_hour_qualification=False,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        result['ended_at']=e.utc();result['content_hash']=e.digest(result);e.write(out/'report.json',result)
        logger.removeHandler(compilation);logger.propagate=old
        for s,v in signals.items():signal.signal(s,v)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    run(a.directory,a.preflight)
