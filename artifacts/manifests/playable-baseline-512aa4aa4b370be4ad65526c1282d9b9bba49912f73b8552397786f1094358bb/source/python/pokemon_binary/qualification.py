"""Short CPU qualification of binary-policy-1; never a full G2/G3 promotion."""
import argparse
import json
import logging
from pathlib import Path
import resource
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.campaign import admission as prior_admission, CompileLog, compile_positive_control
from pokemon_jax_learner.collector import collect as reference_collect
from pokemon_learner.checkpoint import digest, wire, file_hash
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file
from .loop import ROOT, Executables, start, plan, objective, check_report
from .checkpoint import Runner, provenance, save, load, state
from .policy import threshold_words, select_words, act
from .reference import dyadic_threshold


def write(path, data):Path(path).write_bytes(wire(data)+b'\n')


def admission():
    p=plan();contract=json.loads((ROOT/p['policy_contract']).read_bytes())
    receipt=json.loads((ROOT/'artifacts/reports/m1-fixed-loop.json').read_bytes())
    require(receipt['content_hash']==contract['predecessor']==p['predecessor']==digest(
        {k:v for k,v in receipt.items() if k!='content_hash'}),'Predecessor receipt')
    for path,sha in receipt['source_files'].items():
        if path.startswith(('python/','contracts/','tests/fixed-loop/')) or path in (
            'scripts/seal-fixed-loop.py','scripts/verify-fixed-loop.py','requirements-jax.txt',
            'requirements-learner.txt','requirements-contract.txt'):
            checked_file(ROOT,path,sha)
    return {'contract':contract,'plan':p,'provenance':provenance(),
            'predecessor':receipt['content_hash'],'prior':prior_admission()}


def exact(a,b):
    require(jax.tree.structure(a)==jax.tree.structure(b),'Replay structure')
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        require(np.array_equal(x,y),'Replay bytes differ')


def compare(a,b):
    x,y=np.asarray(a),np.asarray(b)
    require(x.shape==y.shape,'Parity shape')
    if x.dtype.kind in 'f':
        error=np.abs(x.astype(float)-y.astype(float));limit=3e-6+3e-5*np.abs(y.astype(float))
        fraction=float(np.max(error/limit,initial=0))
        require(fraction<=1,'Parity tolerance');return fraction
    require(np.array_equal(x,y),'Discrete parity');return 0.


def sampling_audit(count):
    rng=np.random.default_rng(8251)
    bits=((rng.integers(103,126,count,dtype=np.uint32)<<23)|rng.integers(0,2**23,count,dtype=np.uint32))
    rare=np.r_[bits.view(np.float32),np.float32(.5),np.float32(2**-24)].astype(np.float32)
    expected=np.array([dyadic_threshold(r) for r in rare],np.uint32)
    actual=np.asarray(jax.jit(threshold_words)(rare))
    require(np.array_equal(actual,expected),'Exact threshold oracle')
    # Strict lexicographic comparison across high-word carries, for both action orderings.
    words=[];results=[]
    for hi,lo in expected:
        t=(int(hi)<<32)|int(lo)
        for w in (t-1,t,t+1):words.append([w>>32,w&0xffffffff]);results.append(w<t)
    masses=np.repeat(rare,3);mask=np.ones((len(masses),2),bool)
    for ra in (0,1):
        actions=select_words(masses,np.full(len(masses),ra,np.int32),mask,np.array(words,np.uint32))
        want=np.where(results,ra,1-ra)
        require(np.array_equal(actions,want),'Threshold edge selection')
    return {'status':'passed','rare_mass_cases':len(rare),'integer_thresholds_exact':True,
            'injected_action_checks':2*len(words),'uniform_word_assumption':True,
            'sampling_frequency_estimated':False,'minimum_mass':2**-24,
            'proof':'r=m*2^(e-150), e=103..126; r*2^64=m*2^(e-86), shift17..40. Every integer below this threshold selects rare; its complement selects common.'}


def parity(exe,p):
    fraction=0.;decisions=0;batches=0
    for seed in p['parity_seeds']:
        l=start(seed)
        for _ in range(p['parity_batches_per_seed']):
            b,_=exe.compiled['collect'](l.parameters,l.next_game,jnp.uint32(seed));valid=np.asarray(b.valid)
            ids=np.arange(int(l.next_game),int(l.next_game)+64,dtype=np.uint32)
            ref=reference_collect(l.parameters,int(l.version),ids,environment_seed=seed,action_seed=seed,actor_fn=act)
            for name in b.actor._fields:
                fraction=max(fraction,compare(np.asarray(getattr(b.actor,name))[valid],getattr(ref.actor,name)))
            for name in ('rewards','terminal','advantages','returns'):
                fraction=max(fraction,compare(np.asarray(getattr(b.targets,name))[valid],getattr(ref.targets,name)))
            compact=np.full(512,-1,np.int32);compact[valid]=np.arange(valid.sum())
            nxt=np.asarray(b.targets.next_index)[valid]
            compare(np.where(nxt<0,-1,compact[np.maximum(nxt,0)]),ref.targets.next_index)
            _,metrics=objective(l.parameters,b.actor,b.targets,b.valid,p)
            require(float(metrics['max_ratio_error'])<=1e-5,'Same policy ratio')
            separate,_=exe.compiled['update'](l,b,jnp.uint32(seed))
            fused,report=exe(l,seed);check_report(report)
            for a,c in zip(jax.tree.leaves(fused),jax.tree.leaves(separate)):fraction=max(fraction,compare(a,c))
            l=fused;decisions+=int(valid.sum());batches+=1
    return {'status':'passed','batches':batches,'games':64*batches,'decisions':decisions,
            'max_fraction_of_tolerance':fraction,'atol':3e-6,'rtol':3e-5,
            'reference':'sealed Python-control collector with explicit new actor_fn; fused vs split update',
            'old_policy_trajectory_equivalence_claimed':False}


def replay(path,runner,exe):
    save(path,runner);restored=load(path,exe)
    a=exe(runner.learner,runner.seed);b=exe(restored.learner,restored.seed)
    exact(a,b);require(state(runner)==state(restored),'Checkpoint body differs')
    return {'file':path.name,'sha256':file_hash(path),'version':int(runner.learner.version),
            'next_cycle_exact':True}


def main(directory):
    out=Path(directory).resolve()
    require(out.is_relative_to(ROOT/'artifacts/runs'),'Campaign output location')
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'running','full_g2':'open','full_g3':'open','prime_used':False,
            'new_two_hour_soak':False,'strength_evaluation':False}
    log=logging.getLogger('jax._src.dispatch');counter=CompileLog();old=log.propagate
    log.addHandler(counter);log.propagate=False
    try:
        admitted=admission();write(out/'admission.json',admitted)
        result['admission_hash']=digest(admitted);q=admitted['contract']['qualification'];p=admitted['plan']
        result['sampling']=sampling_audit(q['word_probe_rows'])
        with jax.log_compiles(True):
            control=compile_positive_control(counter)
            t=time.perf_counter();exe=Executables();exe.diagnostics()
            result['compile_seconds']=time.perf_counter()-t
            result['parity']=parity(exe,p)
            result['seed_runs']=[];result['replays']=[]
            for seed in q['seeds']:
                runner=Runner(seed,exe)
                for _ in range(q['updates_per_seed']):runner.run()
                result['seed_runs'].append({'seed':seed,'updates':int(runner.learner.version),
                    'games':int(runner.learner.next_game),'decisions':runner.decisions,'executed_moves':runner.executed_moves})
                result['replays'].append(replay(out/f'seed-{seed}.json',runner,exe))
            runner=Runner(q['soak_seed'],exe)
            for _ in range(p['warmup_cycles']):runner.run()
            result['replays'].append(replay(out/'soak-start.json',runner,exe))
            baseline=counter.count;initial=state(runner);beg=time.perf_counter();cycles=0;cycle_seconds=0.;next_log=10
            minimum=1.;maximum_ratio=0.;rows=[]
            with (out/'telemetry.jsonl').open('w') as stream:
                while time.perf_counter()-beg<q['soak_seconds']:
                    t=time.perf_counter();r=runner.run();cycle_seconds+=time.perf_counter()-t;cycles+=1
                    minimum=min(minimum,float(r['minimum_behavior_probability']))
                    maximum_ratio=max(maximum_ratio,float(np.max(r['metrics']['max_ratio_error'])))
                    require(counter.count==baseline,'Unexpected compilation')
                    elapsed=time.perf_counter()-beg
                    if elapsed>=next_log:
                        row={'elapsed_seconds':elapsed,'cycles':cycles,'decisions':runner.decisions-initial['decisions']}
                        stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row);next_log+=10
                elapsed=time.perf_counter()-beg
            result['soak']={'status':'passed','target_seconds':q['soak_seconds'],'elapsed_seconds':elapsed,
                'cycles':cycles,'games':cycles*64,'decisions':runner.decisions-initial['decisions'],
                'executed_moves':runner.executed_moves-initial['executed_moves'],
                'cycle_seconds':cycle_seconds,'decisions_per_second':(runner.decisions-initial['decisions'])/elapsed,
                'minimum_exp_behavior_logp':minimum,'maximum_update_ratio_error':maximum_ratio,
                'unexpected_compilations':counter.count-baseline,'warmup_compilation_count':baseline,
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'device_memory_headroom_certified':False,'telemetry_rows':len(rows)}
            result['replays'].append(replay(out/'soak-final.json',runner,exe))
            result['compile_control']=control;result['traces']=exe.traces
        require(admission()==admitted,'Source/runtime/final reservation changed')
        result['final_reservation_unchanged']=True;result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        log.removeHandler(counter);log.propagate=old
        result['content_hash']=digest(result);write(out/'report.json',result)
    print(json.dumps({'status':result['status'],'content_hash':result['content_hash'],'soak':result['soak']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
