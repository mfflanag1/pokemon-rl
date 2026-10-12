"""Frozen-plan autodiff parity, original-budget canaries, and collector/resume campaign."""
from dataclasses import fields
import json
from pathlib import Path
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner import core as reference
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file,ActorDataset
from pokemon_reference.corpus import read_json,verify_seal
from . import core
from .checkpoint import ROOT,provenance,save,load,state
from .canaries import Canary,table_loss_grad
from .collector import array_hash,policy_hash
from .trainer import Trainer

PREDECESSOR='0b974be3082ce9eda957b5a201da39c55db95a315f91f1f9d013cd8bb05ee21f'


def admission_checks():
    g2=json.loads((ROOT/'artifacts/reports/m1-g2.json').read_bytes())
    require(g2['content_hash']==PREDECESSOR==digest({k:v for k,v in g2.items() if k!='content_hash'}),'G2 predecessor seal')
    prefixes=('python/pokemon_jax/','python/pokemon_slice/','python/pokemon_learner/','python/pokemon_reference/','packages/reference-node/','packages/slice/','packages/jax-slice/','contracts/slice-v1/','contracts/jax-slice-v1/','contracts/learner-v1/')
    for p,h in g2['source_files'].items():
        if p.startswith(prefixes) or p in ('requirements-learner.txt','requirements-jax.txt','requirements-contract.txt','package-lock.json','scripts/seal-jax.py'):checked_file(ROOT,p,h)
    r7=verify_seal(read_json(ROOT/'artifacts/reports/r7-sprint.json'))
    data=ActorDataset(ROOT/r7['dataset_directory'],expected_hash=r7['dataset_hash'],purpose='training')
    checked_file(data.directory,'teams/final.json',data.document['collection_files']['teams/final.json'])
    require(data.document['validation']['final_episodes']==0,'Final evaluation opened')
    runtime=provenance()['runtime'];require(runtime['backend']=='cpu' and runtime['x64_enabled'] is False,'CPU FP32 runtime')
    for p in ('requirements-learner.txt','requirements-jax.txt','requirements-contract.txt'):
        for line in (ROOT/p).read_text().splitlines():
            if '==' in line:
                name,version=line.strip().split('==');require(runtime['packages'][name]==version,'Dependency pin drift')
    return {'previous_g2':PREDECESSOR,'final_reservation_unchanged':True,'previous_r7':r7['content_hash']}


class Comparisons:
    def __init__(self,plan):self.plan=plan;self.rows={}
    def check(self,label,expected,actual):
        a=np.asarray(expected);b=np.asarray(actual)
        require(a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),'Parity shapes/finiteness: '+label)
        error=np.abs(a.astype(float)-b.astype(float));bound=self.plan['atol']+self.plan['rtol']*np.abs(a.astype(float))
        row=self.rows.setdefault(label,{'comparisons':0,'elements':0,'max_absolute_error':0.,'max_fraction_of_tolerance':0.})
        row['comparisons']+=1;row['elements']+=a.size
        row['max_absolute_error']=max(row['max_absolute_error'],float(np.max(error,initial=0)))
        row['max_fraction_of_tolerance']=max(row['max_fraction_of_tolerance'],float(np.max(error/bound,initial=0)))
        require(np.all(error<=bound),'Numerical parity failed: '+label)


def math_parity(plan):
    checks=Comparisons(plan);cases=0
    for seed in plan['parity_seeds']:
        rng=np.random.default_rng(seed)
        for case in range(plan['parity_cases_per_seed']):
            n=(1,7,31,64)[case%4];width=(2,4,8)[case%3]
            logits=rng.normal(0,2,(n,width)).astype(np.float32);mask=rng.random((n,width))>.3;mask[:,0]=True
            actions=np.array([rng.choice(np.flatnonzero(row)) for row in mask],np.int32)
            if case%7==0:mask[:]=False;mask[np.arange(n),actions]=True
            logits[~mask]=1e20
            p,logp=reference.distribution(logits,mask);pj,lj=core.distribution(logits,mask)
            checks.check('probabilities',p,pj);checks.check('log_probabilities',logp,lj)
            require(np.all(np.asarray(pj)[~mask]==0),'Forbidden probability')
            ratios=rng.choice([.4,.65,.9,1.,1.1,1.35,1.8],n).astype(np.float32)
            old=(logp[np.arange(n),actions]-np.log(ratios)).astype(np.float32)
            values,adv,returns=[rng.normal(size=n).astype(np.float32) for _ in range(3)];adv[::9]=0
            entropy=.13 if case%2 else 0.
            args=(logits,values,mask,actions,old,adv,returns)
            mr,gr,vr=reference.ppo_loss_grad(*args,entropy_coef=entropy);mj,gj,vj=core.ppo_loss_grad(*args,entropy_coef=entropy)
            for k in mr:checks.check('metric/'+k,mr[k],mj[k])
            checks.check('logit_gradients',gr,gj);checks.check('value_gradients',vr,vj)
            require(np.all(np.asarray(gj)[~mask]==0),'Forbidden logit gradient')
            # Interleaved same-player trajectories with independent terminal and cut boundaries.
            length=31;trajectory=rng.integers(0,6,length,dtype=np.int32);nxt=np.full(length,-1,np.int32)
            terminal=np.zeros(length,bool);boot=np.zeros(length,np.float32)
            for t in np.unique(trajectory):
                ix=np.flatnonzero(trajectory==t);nxt[ix[:-1]]=ix[1:]
                terminal[ix[-1]]=t%2==0
                if not terminal[ix[-1]]:boot[ix[-1]]=rng.normal()
            rewards=rng.normal(size=length).astype(np.float32);v=rng.normal(size=length).astype(np.float32)
            kw={'gamma':(0.,.9,1.)[case%3],'lam':(0.,.5,.95,1.)[case%4]}
            for k,a,b in zip(('gae_advantages','gae_returns'),reference.linked_gae(rewards,v,nxt,boot,terminal,trajectory,**kw),core.linked_gae(rewards,v,nxt,boot,terminal,trajectory,**kw)):checks.check(k,a,b)
            cases+=1
        parameters={k:rng.normal(size=shape).astype(np.float32) for k,shape in [('w',(7,4)),('b',(4,)),('v',(3,))]}
        jparameters={k:jnp.asarray(v) for k,v in parameters.items()};ar=reference.Adam(parameters,lr=.003,max_norm=.5);aj=core.Adam(jparameters,lr=.003,max_norm=.5)
        for step in range(plan['adam_steps']):
            gradients={k:(rng.normal(size=v.shape)*(0.,.01,10.,1e-9)[step%4]).astype(np.float32) for k,v in parameters.items()}
            checks.check('adam_norm',ar.apply(parameters,gradients),aj.apply(jparameters,gradients))
            for label,x,y in [('adam_parameters',parameters,jparameters),('adam_m',ar.m,aj.m),('adam_v',ar.v,aj.v)]:
                for k in x:checks.check(label,x[k],y[k])
        parameters={'logits':rng.normal(0,.1,(6,4)).astype(np.float32),'values':rng.normal(0,.1,6).astype(np.float32)}
        jparameters={k:jnp.asarray(v) for k,v in parameters.items()};ar=reference.Adam(parameters,lr=.003);aj=core.Adam(jparameters,lr=.003)
        states=rng.integers(0,6,64,dtype=np.int32);mask=np.ones((64,4),bool);mask[::2,3]=False;actions=np.array([rng.choice(np.flatnonzero(m)) for m in mask],np.int32)
        old=reference.distribution(parameters['logits'][states],mask)[1][np.arange(64),actions]
        adv=rng.normal(size=64).astype(np.float32);returns=rng.normal(size=64).astype(np.float32)
        for _ in range(plan['table_update_steps']):
            mr,gl,gv=reference.ppo_loss_grad(parameters['logits'][states],parameters['values'][states],mask,actions,old,adv,returns,entropy_coef=.01)
            gradient={k:np.zeros_like(v) for k,v in parameters.items()};np.add.at(gradient['logits'],states,gl);np.add.at(gradient['values'],states,gv)
            (_,mj),gj=table_loss_grad(jparameters,jnp.asarray(states),jnp.asarray(mask),jnp.asarray(actions),jnp.asarray(old),jnp.asarray(adv),jnp.asarray(returns),.2,.5,.01)
            for k in gradient:checks.check('table_parameter_gradients',gradient[k],gj[k])
            for k in mr:checks.check('table_metric/'+k,mr[k],mj[k])
            ar.apply(parameters,gradient);aj.apply(jparameters,gj)
            for k in parameters:checks.check('table_updates',parameters[k],jparameters[k])
    return {'status':'passed','ppo_cases':cases,'gae_cases':cases,'adam_steps':len(plan['parity_seeds'])*plan['adam_steps'],
            'table_update_steps':len(plan['parity_seeds'])*plan['table_update_steps'],'atol':plan['atol'],'rtol':plan['rtol'],'comparisons':checks.rows}


def canaries(plan,out):
    frozen=json.loads((ROOT/plan['canary_plan']).read_text());results=[]
    for task in ('bandit','delayed'):
        for seed in frozen['seeds']:
            trainer=Canary(task,seed,frozen);trace=[];first=None
            for _ in range(frozen[task]['max_updates']):
                metrics=trainer.update(trainer.collect());probabilities=trainer.probabilities()
                require(all(np.isfinite(x) for row in metrics for x in row.values()),'Canary nonfinite metrics')
                if first is None and min(probabilities)>=frozen[task]['required_probability']:first=trainer.decisions
                trace.append({'updates':trainer.updates,'decisions':trainer.decisions,'probabilities':probabilities,'max_ratio_error':max(m['max_ratio_error'] for m in metrics)})
            require(trainer.decisions==frozen[task]['sample_budget'] and min(probabilities)>=frozen[task]['required_probability'],'Canary learning gate')
            result={'task':task,'seed':seed,'decisions':trainer.decisions,'episodes':trainer.episodes,'updates':trainer.updates,
                    'probabilities':probabilities,'first_threshold_decisions':first,'parameters_hash':digest({k:array_hash(v) for k,v in trainer.parameters.items()})}
            (out/f'canary-{task}-{seed}.json').write_bytes(wire({'result':result,'trace':trace})+b'\n');results.append(result)
            print(json.dumps(result),flush=True)
    return {'status':'passed','mode':plan['canary_mode'],'plan_hash':digest(frozen),'results':results}


def export_buffer(out,buffer):
    out.mkdir(exist_ok=False)
    for name,obj in [('actor',buffer.actor),('targets',buffer.targets)]:
        np.savez_compressed(out/f'{name}.npz',**{f.name:getattr(obj,f.name) for f in fields(obj)})
    metadata={'format':'jax-learner-buffer-1','version':buffer.version,'policy_hash':buffer.policy_hash,'collection_hash':buffer.content_hash,
              'games':buffer.games,'decisions':len(buffer.actor.actions),
              'actor_fields':[f.name for f in fields(buffer.actor)],
              'actor_has_targets':bool(set(f.name for f in fields(buffer.actor)) & {'rewards','terminal','advantages','returns','bootstrap','winner','chance'}),
              'files':{name:file_hash(out/name) for name in ('actor.npz','targets.npz')}}
    (out/'manifest.json').write_bytes(wire(metadata)+b'\n')


def collectors(plan,out):
    reports=[]
    for seed in plan['collector_seeds']:
        trainer=Trainer(seed,plan);replay=None;rows=[];seed_dir=out/f'collector-{seed}';seed_dir.mkdir()
        for update in range(plan['collector_updates']):
            if update:
                boundary=seed_dir/('boundary.checkpoint.json' if update==1 else f'boundary-{update}.checkpoint.json')
                if update>1:save(boundary,trainer)
                replay=load(boundary,expected_plan=plan)
            start=time.perf_counter();buffer=trainer.collect();collected=time.perf_counter();metrics=trainer.update(buffer);ended=time.perf_counter()
            if update==0:
                export_buffer(seed_dir/'first-buffer',buffer);save(seed_dir/'boundary.checkpoint.json',trainer)
                replay=load(seed_dir/'boundary.checkpoint.json',expected_plan=plan)
            else:
                other=replay.collect();require(other.content_hash==buffer.content_hash,'Restored collection mismatch')
                require(replay.update(other)==metrics,'Restored optimizer diagnostics mismatch')
            require(state(trainer)==state(replay),'Restored boundary state mismatch')
            rows.append({'update':update+1,'games':buffer.games,'decisions':len(buffer.actor.actions),'collection_hash':buffer.content_hash,
                         'terminal_reward_sum':float(np.sum(buffer.targets.rewards)),
                         'actor_boundary_check':bool(set(f.name for f in fields(buffer.actor))=={'features','mask','actions','old_logp','old_values','game_id','side','turn'} and
                            np.all(buffer.actor.features[:,1:3]==0) and np.all(buffer.actor.mask.any(axis=1))),
                         'collect_seconds':collected-start,'update_seconds':ended-collected,'complete_cycle_seconds':ended-start,
                         'behavior_max_ratio_error':metrics['behavior_max_ratio_error'],'optimizer_minibatches':len(metrics['minibatches']),
                         'max_gradient_norm':max(m['gradient_norm_before_clip'] for m in metrics['minibatches'])})
        save(seed_dir/'final.checkpoint.json',trainer)
        require(state(load(seed_dir/'final.checkpoint.json',expected_plan=plan))==state(trainer),'Final checkpoint reload')
        report={'seed':seed,'games':trainer.games,'decisions':trainer.decisions,'updates':trainer.updates,'resume_compared_updates':len(rows)-1,
                'disk_resumes_with_compared_update':len(rows)-1,
                'terminal_score_pair_sum':sum(row['terminal_reward_sum'] for row in rows),'final_policy_hash':policy_hash(trainer.parameters),'trace':rows}
        (seed_dir/'report.json').write_bytes(wire(report)+b'\n');reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='trace'}),flush=True)
    require(all(row['actor_boundary_check'] for s in reports for row in s['trace']),'Actor audit gate')
    return {'status':'passed','seeds':reports,'all_buffers_observation_only':all(row['actor_boundary_check'] for s in reports for row in s['trace']),'recovery_mode':'whole-episode completed-update only',
        'timing_scope':'Wall time from host collection through final update synchronization; includes views, inference, sampling, environment, transfer, linkage, GAE, shuffling and optimization. Excludes verification replay, serialization and checkpoints. Includes compilation/cache warm-up; variable final minibatch and compact GAE shapes are not a G3 steady-state benchmark.'}


def run(directory):
    out=Path(directory);out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'contracts/jax-learner-v1/plan.json').read_text())
    bound=provenance();admission={'format':'m1-jax-learner-admission-1','plan':plan,'provenance':bound,'predecessors':admission_checks()}
    (out/'admission.json').write_bytes(wire(admission)+b'\n');start=time.perf_counter()
    report={'format':'m1-jax-learner-results-1','admission_hash':digest(admission),'full_g2':'open','g3':'open'}
    try:
        report['parity']=math_parity(plan);print(json.dumps({'parity':'passed','ppo_cases':report['parity']['ppo_cases']}),flush=True)
        report['canaries']=canaries(plan,out)
        report['collector']=collectors(plan,out)
        require(provenance()==bound,'Campaign source drift')
        report['status']='passed'
    except Exception as e:
        report['status']='failed';report['error']=str(e);raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start;report['content_hash']=digest(report)
        (out/'report.json').write_bytes(wire(report)+b'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');run(parser.parse_args().directory)
