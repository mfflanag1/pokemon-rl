"""Frozen-plan parity, fixed-shape CPU profiling and a two-hour complete-cycle soak."""
import argparse
import cProfile
import io
import json
import logging
import os
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax_learner.campaign import admission_checks as previous_checks, Comparisons
from pokemon_jax_learner.collector import collect as reference_collect, array_hash
from pokemon_jax_learner.model import model_loss_grad, predict_kernel, ActorInput
from pokemon_jax_learner.core import Adam
from pokemon_jax.engine import validate_state
from pokemon_learner.checkpoint import digest, wire, file_hash
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file
from .loop import ROOT, plan, start, Executables, permutation, check_report
from .checkpoint import Runner, provenance, state, save, load


def write(path, value):
    Path(path).write_bytes(wire(value) + b'\n')


def admission():
    p = plan()
    previous = json.loads((ROOT / 'artifacts/reports/m1-jax-learner.json').read_bytes())
    require(previous['content_hash'] == p['predecessor'] == digest({k:v for k,v in previous.items() if k != 'content_hash'}), 'Predecessor receipt')
    # Immutable predecessors; documentation, test runner and additive new package may evolve.
    prefixes = ('python/pokemon_jax_learner/', 'python/pokemon_jax/', 'python/pokemon_slice/',
                'python/pokemon_learner/', 'python/pokemon_reference/', 'contracts/jax-learner-v1/',
                'contracts/jax-slice-v1/', 'contracts/slice-v1/', 'contracts/learner-v1/')
    for path, sha in previous['source_files'].items():
        if path.startswith(prefixes) or path in ('requirements-jax.txt','requirements-learner.txt','requirements-contract.txt','scripts/seal-jax-learner.py'):
            checked_file(ROOT, path, sha)
    return {'plan': p, 'provenance': provenance(), 'predecessor': p['predecessor'], 'prior_gates': previous_checks()}


class CompileLog(logging.Handler):
    """Count actual XLA compilation messages, in addition to entry-point tracing."""
    def __init__(self):
        super().__init__(); self.count = 0
    def emit(self, record):
        # JAX 0.11.2 emits this from dispatch.LogElapsedTimeContextManager, including
        # compile-or-cache retrieval. Actual first-call/repeated-call controls test it.
        if record.getMessage().startswith('Finished XLA compilation of '): self.count += 1


def compile_positive_control(counter):
    fn=jax.jit(lambda x:jnp.sin(x)+jnp.float32(.125))
    x=np.array([.1,.3,.7],np.float32);before=counter.count
    jax.block_until_ready(fn(x));after=counter.count
    require(after>before,'Compilation logger failed positive control')
    jax.block_until_ready(fn(x))
    require(counter.count==after,'Repeated control unexpectedly compiled')
    return {'first_call_compilation_events':after-before,'same_shape_repeat_events':counter.count-after,
            'logger':'jax._src.dispatch','message_prefix':'Finished XLA compilation of '}


def memory():
    # ru_maxrss is process peak RSS (bytes on macOS, KiB on Linux); not device allocation.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': peak *= 1024
    try:
        current = int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True,
                                              stderr=subprocess.DEVNULL).strip()) * 1024
    except (OSError, ValueError, subprocess.CalledProcessError): current = None
    device = jax.devices()[0].memory_stats()
    return {'peak_process_rss_bytes': int(peak), 'current_process_rss_bytes': current,
            'device_memory_stats': device, 'rss_scope': 'this process, including JAX runtime/compiler; system free memory not certified'}


class Aggregate:
    """Constant-memory latency histogram and weighted scalar diagnostics."""
    def __init__(self):
        self.cycles = self.decisions = self.games = self.moves = self.weight = 0
        self.wall = self.latency_min = self.latency_max = 0.
        self.edges = np.geomspace(1e-5, 10, 257)
        self.hist = np.zeros(258, dtype=np.int64)
        self.sums = {}; self.maxima = {}; self.minima = {}
        self.probability_min = 1.

    def add(self, report, elapsed):
        self.cycles += 1; self.decisions += int(report['decisions']); self.games += 64
        self.moves += int(report['executed_moves']); self.wall += elapsed
        self.latency_min = elapsed if self.cycles == 1 else min(elapsed, self.latency_min)
        self.latency_max = max(elapsed, self.latency_max)
        self.hist[np.searchsorted(self.edges, elapsed)] += 1
        self.probability_min = min(self.probability_min, float(report['minimum_behavior_probability']))
        count = np.asarray(report['metrics']['valid_count']); self.weight += int(count.sum())
        for k in ('loss','actor_loss','value_loss','entropy','approx_kl','clip_fraction','gradient_norm','max_ratio_error'):
            values = np.asarray(report['metrics'][k]); selected = values[count > 0]
            self.sums[k] = self.sums.get(k,0.) + float(np.sum(values * count))
            self.maxima[k] = max(self.maxima.get(k,-float('inf')),float(selected.max()))
            self.minima[k] = min(self.minima.get(k,float('inf')),float(selected.min()))

    def report(self):
        quantiles = {}
        for q in (.5,.95,.99):
            i = np.searchsorted(np.cumsum(self.hist), np.ceil(q * self.cycles))
            quantiles[str(q)] = float(self.edges[i]) if i < len(self.edges) else None
        return {'cycles':self.cycles, 'games':self.games, 'decisions':self.decisions, 'executed_moves':self.moves,
                'complete_cycle_seconds':self.wall, 'decisions_per_cycle_second':self.decisions/self.wall,
                'games_per_cycle_second':self.games/self.wall, 'decisions_per_game':self.decisions/self.games,
                'latency_min_seconds':self.latency_min, 'latency_max_seconds':self.latency_max,
                'latency_quantile_upper_bounds_seconds':quantiles,
                'latency_histogram':{'edges':self.edges.tolist(),'counts':self.hist.tolist()},
                'diagnostics':{k:{'valid_weighted_mean':self.sums[k]/self.weight,'min_minibatch':self.minima[k],
                                  'max_minibatch':self.maxima[k]} for k in self.sums},
                'minimum_sampled_behavior_probability':self.probability_min,
                'diagnostic_scope':'one supported joint-action phase, minibatch values before their respective updates; probability minimum over sampled actions'}


def identical(a,b):
    require(jax.tree.structure(a)==jax.tree.structure(b),'Resume tree structure')
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        require(np.array_equal(np.asarray(x),np.asarray(y)),'Resume array/diagnostic mismatch')


def parity(executable, out):
    p=executable.plan; comparisons=Comparisons(p); games=decisions=resumes=0
    for seed in p['parity_seeds']:
        runner=Runner(seed,executable)
        for batch in range(p['parity_batches_per_seed']):
            l=runner.learner
            b,terminal=executable.compiled['collect'](l.parameters,l.next_game,jnp.uint32(seed))
            validate_state(terminal); valid=np.asarray(b.valid)
            ref=reference_collect(l.parameters,int(l.version),np.arange(int(l.next_game),int(l.next_game)+64,dtype=np.uint32),environment_seed=seed,action_seed=seed)
            for name in b.actor._fields:
                expected=getattr(ref.actor,name);actual=np.asarray(getattr(b.actor,name))[valid]
                if expected.dtype.kind in 'iub':require(np.array_equal(expected,actual),'Discrete actor parity: '+name)
                comparisons.check('actor/'+name,expected,actual)
            for name in ('rewards','terminal','advantages','returns'): comparisons.check('targets/'+name,getattr(ref.targets,name),np.asarray(getattr(b.targets,name))[valid])
            mapping=np.cumsum(valid)-1;nxt=np.asarray(b.targets.next_index)[valid]
            require(np.array_equal(np.where(nxt>=0,mapping[np.maximum(nxt,0)],-1),ref.targets.next_index),'Compacted own-player linkage')
            _,logp,values=predict_kernel(l.parameters,ActorInput(b.actor.features,b.actor.mask))
            selected=np.asarray(logp)[np.arange(512),np.maximum(np.asarray(b.actor.actions),0)]
            comparisons.check('behavior/old_logp',selected[valid],np.asarray(b.actor.old_logp)[valid])
            comparisons.check('behavior/old_values',np.asarray(values)[valid],np.asarray(b.actor.old_values)[valid])
            pars=dict(l.parameters); opt=Adam(pars,lr=p['learning_rate'],max_norm=p['max_gradient_norm'])
            opt.m,opt.v,opt.step=dict(l.m),dict(l.v),int(l.step)
            fixed,fixed_metrics=executable.compiled['update'](l,b,jnp.uint32(seed))
            for epoch in range(2):
                order=np.asarray(permutation(jnp.uint32(seed),l.version,jnp.int32(epoch),b.valid))
                for mb,ix in enumerate(order.reshape(4,128)):
                    ix=ix[valid[ix]]
                    if not len(ix):continue
                    (_,metrics),gradients=model_loss_grad(pars,b.actor.features[ix],b.actor.mask[ix],b.actor.actions[ix],
                        b.actor.old_logp[ix],b.targets.advantages[ix],b.targets.returns[ix],p['ppo_clip'],p['value_coefficient'],p['entropy_coefficient'])
                    opt.apply(pars,gradients)
                    for name in metrics:comparisons.check('update/'+name,metrics[name],fixed_metrics[name][epoch,mb])
            for label,a,c in [('parameters',pars,fixed.parameters),('m',opt.m,fixed.m),('v',opt.v,fixed.v)]:
                for name in a:comparisons.check(label+'/'+name,a[name],c[name])
            require(opt.step==int(fixed.step),'Applied optimizer steps')
            if batch==0:
                directory=out/f'first-buffer-{seed}';directory.mkdir()
                np.savez(directory/'actor.npz',**{k:np.asarray(v) for k,v in b.actor._asdict().items()})
                np.savez(directory/'targets.npz',**{k:np.asarray(v) for k,v in b.targets._asdict().items()},valid=valid)
                write(directory/'manifest.json',{'capacity':512,'valid':int(valid.sum()),'actor_fields':list(b.actor._fields),
                    'actor_has_targets':False,'files':{f:file_hash(directory/f) for f in ('actor.npz','targets.npz')}})
            boundary=out/f'parity-{seed}-{batch}.checkpoint.json'
            save(boundary,runner)
            restored=load(boundary,executable)
            report=runner.run();again=restored.run();identical(report,again);identical(runner.learner,restored.learner)
            require(state(runner)==state(restored),'Resume complete state');resumes+=1
            for x,y in zip(jax.tree.leaves(fixed),jax.tree.leaves(runner.learner)):comparisons.check('fused_vs_separate',x,y)
            games+=64;decisions+=int(report['decisions'])
    return {'status':'passed','games':games,'decisions':decisions,'disk_resumes_with_compared_next_cycle':resumes,
            'comparisons':comparisons.rows,'atol':p['atol'],'rtol':p['rtol'],
            'shuffle_scope':'new fixed-capacity shuffle; compact reference uses the identical resulting valid order, not the prior variable-length permutation'}


def profile(ex, out, compile_log):
    runner=Runner(61,ex)
    for _ in range(ex.plan['warmup_cycles']):runner.run()
    baseline=compile_log.count;traces=dict(ex.traces);aggregate=Aggregate();profiler=cProfile.Profile()
    profiler.enable()
    for _ in range(ex.plan['profile_cycles']):
        t=time.perf_counter();r=runner.run();aggregate.add(r,time.perf_counter()-t)
    profiler.disable();profiler.dump_stats(str(out/'host-profile.pstats'))
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    (out/'host-profile.txt').write_text(stream.getvalue())
    phases={}
    for name in ('collect','update'):
        samples=[]
        b,_=ex.compiled['collect'](runner.learner.parameters,runner.learner.next_game,jnp.uint32(runner.seed))
        for _ in range(100):
            t=time.perf_counter()
            value=ex.compiled[name](runner.learner.parameters,runner.learner.next_game,jnp.uint32(runner.seed)) if name=='collect' else ex.compiled[name](runner.learner,b,jnp.uint32(runner.seed))
            jax.block_until_ready(value);samples.append(time.perf_counter()-t)
        phases[name]={'repetitions':100,'median_seconds':float(np.median(samples)),
                      'p95_seconds':float(np.quantile(samples,.95)),
                      'scope':'isolated repeated phase on a fixed input; not complete-cycle throughput'}
    require(compile_log.count==baseline and ex.traces==traces,'Unexpected profile compilation')
    return {'status':'passed',**aggregate.report(),'warmup_cycles':ex.plan['warmup_cycles'],
            'host_profiler_enabled':True,'memory':memory(),'isolated_phases':phases,
            'timing_scope':'cProfile is enabled and adds host overhead; use the unprofiled soak for operational throughput',
            'warmup_compilation_count':baseline,
            'unexpected_compilations':compile_log.count-baseline,'entry_traces':dict(ex.traces)}


def soak(ex,out,compile_log):
    p=ex.plan;runner=Runner(p['soak_seed'],ex)
    for _ in range(p['warmup_cycles']):runner.run()
    save(out/'soak-start.checkpoint.json',runner)
    baseline=compile_log.count;traces=dict(ex.traces);initial_memory=memory()
    initial_counters={'updates':int(runner.learner.version),'decisions':runner.decisions,'next_game':int(runner.learner.next_game)}
    aggregate=Aggregate();start_wall=time.monotonic();start_cpu=time.process_time()
    next_telemetry=p['telemetry_seconds'];next_checkpoint=p['checkpoint_seconds'];resumes=0
    with (out/'telemetry.jsonl').open('x') as log:
        while time.monotonic()-start_wall < p['soak_seconds']:
            t=time.perf_counter();r=runner.run();aggregate.add(r,time.perf_counter()-t)
            elapsed=time.monotonic()-start_wall
            require(ex.traces==traces and compile_log.count==baseline,'Unexpected soak recompilation')
            if elapsed>=next_telemetry:
                line={'elapsed_seconds':elapsed,'updates':int(runner.learner.version),'soak_cycles':aggregate.cycles,
                      'decisions':aggregate.decisions,'memory':memory(),
                      'recent_metrics':{k:float(np.mean(v)) for k,v in r['metrics'].items()}}
                log.write(wire(line).decode()+'\n');log.flush()
                write(out/'progress.json',line)
                print(json.dumps({'soak_elapsed_seconds':round(elapsed,2),'cycles':aggregate.cycles,
                                  'decisions':aggregate.decisions,'peak_rss_bytes':line['memory']['peak_process_rss_bytes']}),flush=True)
                next_telemetry+=p['telemetry_seconds']
            if elapsed>=next_checkpoint:
                path=out/f'soak-boundary-{resumes+1}.checkpoint.json';save(path,runner)
                restored=load(path,ex)
                # Audit the NEXT update without counting a duplicate as primary work.
                expected,expected_report=ex(runner.learner,runner.seed);check_report(expected_report)
                restored_report=restored.run();identical(expected,restored.learner);identical(expected_report,restored_report)
                resumes+=1;next_checkpoint+=p['checkpoint_seconds']
    wall=time.monotonic()-start_wall;cpu=time.process_time()-start_cpu
    save(out/'soak-final.checkpoint.json',runner)
    restored=load(out/'soak-final.checkpoint.json',ex)
    expected,expected_report=ex(runner.learner,runner.seed);check_report(expected_report)
    identical(expected_report,restored.run());identical(expected,restored.learner);resumes+=1
    require(compile_log.count==baseline and ex.traces==traces,'Unexpected final-resume compilation')
    require(int(runner.learner.version)==initial_counters['updates']+aggregate.cycles and
            runner.decisions==initial_counters['decisions']+aggregate.decisions,'Soak accounting')
    return {'status':'passed',**aggregate.report(),'target_seconds':p['soak_seconds'],'elapsed_seconds':wall,
            'process_cpu_seconds':cpu,'decisions_per_elapsed_second':aggregate.decisions/wall,
            'games_per_elapsed_second':aggregate.games/wall,'initial_memory':initial_memory,'final_memory':memory(),
            'warmup_cycles':p['warmup_cycles'],'initial_counters':initial_counters,
            'final_counters':{'updates':int(runner.learner.version),'optimizer_steps':int(runner.learner.step),
                              'decisions':runner.decisions,'next_game':int(runner.learner.next_game)},
            'unexpected_compilations':compile_log.count-baseline,'entry_traces':dict(ex.traces),
            'warmup_compilation_count':baseline,
            'disk_resumes_with_compared_next_cycle':resumes,
            'timing_scope':'elapsed includes host validation, aggregation, telemetry, checkpoint I/O and audit replay; cycle sum excludes those except validation; warmup/compile excluded',
            'device_memory_headroom_certified':False,'full_g2':'open','full_g3':'open'}


def main(output):
    out=Path(output).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Output must be under artifacts/runs')
    out.mkdir(parents=True,exist_ok=False);admitted=admission();write(out/'admission.json',admitted)
    compilation=CompileLog();logger=logging.getLogger('jax._src.dispatch');logger.addHandler(compilation)
    jax.config.update('jax_log_compiles',True)
    result={'status':'failed','full_g2':'open','full_g3':'open','admission_hash':digest(admitted)}
    try:
        t=time.perf_counter();ex=Executables();ex.diagnostics()
        result['compile']={'seconds':time.perf_counter()-t,'entry_traces':dict(ex.traces),
                           'cycle_memory_analysis':str(ex.compiled['cycle'].memory_analysis()),
                           'cycle_cost_analysis':ex.compiled['cycle'].cost_analysis(),
                           'logging_positive_control':compile_positive_control(compilation)}
        result['parity']=parity(ex,out);write(out/'parity.json',result['parity'])
        result['profile']=profile(ex,out,compilation);write(out/'profile.json',result['profile'])
        print(json.dumps({'parity':'passed','profile':'passed','soak_seconds':ex.plan['soak_seconds']}),flush=True)
        result['soak']=soak(ex,out,compilation);write(out/'soak.json',result['soak'])
        require(admission()==admitted,'Source/runtime/predecessor changed during campaign')
        result['status']='passed'
    except Exception as error:
        result['error']=repr(error);result['traceback']=traceback.format_exc();raise
    finally:
        logger.removeHandler(compilation)
        result['content_hash']=digest(result);write(out/'report.json',result)
        print(json.dumps({'report':str(out/'report.json'),'status':result['status']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');main(parser.parse_args().output)
