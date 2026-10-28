"""Fail-closed actual-model Metal probe, also runnable as a matching CPU control."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import forward
from pokemon_binary.policy import threshold_words, select_words
from pokemon_full_train.learning import update
from pokemon_full_train.storage import ROOT, atomic_json, digest, runtime
from .collector import infer


def probe(output, backend):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    # Make the PRNG lowering choice explicit across JAX versions.
    jax.config.update('jax_threefry_partitionable',True)
    report = {'status':'running','requested_backend':backend,'steps':[],
              'jax_metal':importlib.metadata.version('jax-metal') if backend == 'METAL' else None}
    def record(stage, **data):
        report['steps'].append({'stage':stage,**data})
        atomic_json(output/'report.json',report)
        print(stage,flush=True)
    try:
        report['runtime'] = runtime()
        report['devices'] = [str(d) for d in jax.devices()]
        if jax.default_backend().lower() != backend.lower() or jax.config.x64_enabled:
            raise ValueError('Requested backend was not selected or x64 enabled')
        record('device_admitted')
        rare = np.asarray([2.**(-k) for k in range(1,25)],np.float32)
        # Integer boundary fixtures are generated on the host, independently of JAX.
        thresholds = np.asarray([[int(float(r)*2**64)>>32,int(float(r)*2**64)&0xffffffff] for r in rare],np.uint32)
        rare = np.repeat(rare,3)
        words = np.asarray([[((int(h)<<32)+int(l)+delta)>>32,((int(h)<<32)+int(l)+delta)&0xffffffff]
                            for h,l in thresholds for delta in [-1,0,1]],np.uint32)
        actual_threshold = np.asarray(jax.jit(threshold_words)(jnp.asarray(rare)))
        np.testing.assert_array_equal(actual_threshold,np.repeat(thresholds,3,axis=0))
        decisions = np.asarray(jax.jit(select_words)(jnp.asarray(rare),jnp.zeros(len(rare),jnp.int32),
            jnp.ones((len(rare),2),bool),jnp.asarray(words)))
        np.testing.assert_array_equal(decisions,np.tile([0,1,1],24))
        record('integer-threshold-boundaries',cases=len(rare))
        directory = ROOT/'artifacts/runs/f2-full-training-v2/seed-2207/checkpoints'
        pointer = json.loads((directory/'latest.json').read_text())
        path = directory/pointer['file']
        if digest(path) != pointer['sha256']: raise ValueError('Checkpoint digest mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            parameters = {k[3:]:jnp.asarray(arrays[k]) for k in arrays.files if k.startswith('p__')}
        report['checkpoint_sha256'] = pointer['sha256']
        rng = np.random.default_rng(73109)
        outputs = {}
        for size in [1,16,256]:
            x = rng.normal(0,.2,(size,928)).astype(np.float32)
            mask = rng.random((size,14)) > .4
            mask[0] = False
            if size > 1:
                mask[1] = False; mask[1,13] = True
            seeds = rng.integers(0,2**32,size,dtype=np.uint32)
            counters = np.arange(size,dtype=np.uint32)
            t0 = time.perf_counter()
            logits,values = map(np.asarray,jax.jit(forward)(parameters,jnp.asarray(x)))
            record(f'forward-{size}',seconds=time.perf_counter()-t0)
            t0 = time.perf_counter()
            action,logp,value = map(np.asarray,infer(parameters,x,mask,seeds,counters))
            record(f'actor-{size}',cold_seconds=time.perf_counter()-t0)
            timings=[]
            for _ in range(5):
                t0=time.perf_counter()
                result=infer(parameters,x,mask,seeds,counters)
                jax.block_until_ready(result)
                timings.append(time.perf_counter()-t0)
            record(f'actor-warm-{size}',median_seconds=float(np.median(timings)))
            for name, array in [('logits',logits),('forward_value',values),('action',action),('logp',logp),('value',value)]:
                if not np.isfinite(array).all(): raise ValueError('Nonfinite probe output')
                outputs[f'{size}_{name}']=array
        batch={'features':x,'mask':mask,'actions':np.maximum(action,0),'old_logp':logp,
               'advantages':rng.normal(0,.2,256).astype(np.float32),'returns':rng.uniform(0,1,256).astype(np.float32),
               'valid':mask.any(1)}
        zeros={k:jnp.zeros_like(v) for k,v in parameters.items()}
        t0=time.perf_counter()
        p,m,v,step,metrics=update(parameters,zeros,zeros,jnp.int32(0),{k:jnp.asarray(v) for k,v in batch.items()})
        jax.block_until_ready((p,m,v,step,metrics))
        record('update',seconds=time.perf_counter()-t0)
        for prefix,state in [('p',p),('m',m),('v',v),('metric',metrics)]:
            for key,value in state.items():
                value=np.asarray(value)
                if not np.isfinite(value).all(): raise ValueError('Nonfinite update')
                outputs[f'{prefix}_{key}']=value
        np.savez_compressed(output/'outputs.npz',**outputs)
        report['status']='probe_completed_pending_cross_device_parity'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=['cpu','METAL'],required=True)
    args=parser.parse_args()
    probe(args.output,args.backend)
