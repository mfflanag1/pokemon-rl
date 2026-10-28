"""Frozen, player-only inference and one-joint-turn hypothetical lookahead."""
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full_train.storage import ROOT, canonical, digest
from pokemon_full_train.learning import derived_seed
from pokemon_full.learner import parameter_hash
from pokemon_full.policy import distribution
from pokemon_nonlinear_policy.network import forward
from pokemon_nonlinear_policy.actor import batched_actor
from pokemon_nonlinear_policy.common import load
from pokemon_terminal_matchup.features import encode, validate_catalog
from pokemon_worker_watch.transport import Worker

COMMAND = ('node','packages/opening-search/worker.mjs')
SIDES = ('p1','p2')
RECEIPT_SHA = '40a0e75ca66238d9dc2ee81ae8261e326d2638d75bbe9fa60119c1088d986bfd'


def load_frozen(seed):
    receipt_path=ROOT/'artifacts/reports/f32-observed-continuation.json'
    if digest(receipt_path)!=RECEIPT_SHA or seed not in (7103,8209,9311):
        raise ValueError('F32 receipt/seed admission')
    receipt=json.loads(receipt_path.read_text());run=ROOT/receipt['run']
    report=json.loads((run/'report.json').read_text())
    if digest(run/'report.json')!=receipt['report_sha256'] or report['status']!='passed':
        raise ValueError('F32 report binding')
    arm=report['seeds'][str(seed)]['arms']['nonlinear']
    folder=run/str(seed)/'nonlinear/checkpoints/1024'
    parameters,opt=load(folder,receipt['input_hash'],seed,'nonlinear',1024,arm['final_next_episode'])
    if int(opt[2])!=8192 or parameter_hash(parameters)!=arm['endpoints']['1048576']['parameter_hash']:
        raise ValueError('F32 final state binding')
    plan=json.loads((ROOT/'contracts/observed-continuation-v1/plan.json').read_text())
    catalog_path=ROOT/plan['catalog']
    if digest(catalog_path)!=plan['catalog_sha256']:raise ValueError('Public catalog binding')
    catalog=json.loads(catalog_path.read_text());validate_catalog(catalog)
    return parameters,catalog,{'receipt_sha256':RECEIPT_SHA,'parameter_hash':parameter_hash(parameters),
        'pointer':str((folder/'latest.json').relative_to(ROOT)),
        'pointer_sha256':digest(folder/'latest.json'),**json.loads((folder/'latest.json').read_text())}


@jax.jit
def predict(parameters,x,mask):
    logits,values=forward(parameters,x,'nonlinear')
    probabilities,_=distribution(logits,mask)
    return probabilities,values


def predictions(parameters,players,catalog):
    rows=[encode(p,catalog,'matchup') for p in players]
    if not rows:return []
    out=[]
    for first in range(0,len(rows),128):
        batch=rows[first:first+128]
        x=np.zeros((128,1278),np.float32);m=np.zeros((128,14),bool)
        for i,(a,b) in enumerate(batch):x[i],m[i]=a,b
        probs,values=map(np.asarray,predict(parameters,jnp.asarray(x),jnp.asarray(m)))
        if not np.isfinite(probs).all() or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError('Invalid actor prediction')
        out.extend((probs[i].copy(),float(values[i])) for i in range(len(batch)))
    return out


def sample(probabilities,seed):
    p=np.asarray(probabilities,np.float64)
    if p.shape!=(14,) or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:
        raise ValueError('Invalid root distribution')
    return int(np.random.default_rng(seed).choice(14,p=p/p.sum()))


def frozen_actions(parameters,players,catalog,seeds,counters):
    outputs=[]
    for first in range(0,len(players),128):
        last=first+128
        outputs.extend(batched_actor(parameters,players[first:last],seeds[first:last],
            counters[first:last],128,'nonlinear',catalog))
    return outputs


def root_distribution(prior,scores,mask,temperature):
    prior=np.asarray(prior,np.float64);scores=np.asarray(scores,np.float64);mask=np.asarray(mask,bool)
    if prior.shape!=(14,) or scores.shape!=(14,) or mask.shape!=(14,) or not mask.any() or temperature<=0:
        raise ValueError('Root shape/temperature')
    if not np.isfinite(scores[mask]).all() or np.any((scores[mask]<0)|(scores[mask]>1)):
        raise ValueError('Root value range')
    if not np.isfinite(prior).all() or np.any(prior<0) or np.any(prior[~mask]) or prior[mask].sum()<=0:
        raise ValueError('Root prior')
    weights=np.zeros(14,np.float64)
    logits=np.log(np.maximum(prior[mask],np.finfo(np.float64).tiny))+scores[mask]/temperature
    weights[mask]=np.exp(logits-logits.max());weights/=weights.sum()
    return weights


def joint(frame,turn=2):
    return all(frame['players'][s]['phase']=='move' and any(frame['players'][s]['mask'])
        and frame['players'][s]['turn']>=turn for s in SIDES)


class Search:
    def __init__(self,worker,parameters,catalog,settings):
        self.worker,self.p,self.catalog,self.settings=worker,parameters,catalog,settings

    def decide(self,input,search_seed,prior,fallback):
        started=time.perf_counter();side=input['side'];foe='p2' if side=='p1' else 'p1'
        result={'input_sha256':hashlib.sha256(canonical(input)).hexdigest(),'search_seed':search_seed,
            'fallback_action':fallback,'action':fallback,'probabilities':list(map(float,prior)),
            'applied':False,'reason':None,'branches':[]}
        hypotheses=self.worker.request('hypotheses',input=input,search_seed=search_seed,
            budget={'worlds':self.settings['worlds'],'attempts':self.settings['hypothesis_attempts']})
        result['hypotheses']=hypotheses
        if not hypotheses['worlds']:
            result.update(reason=hypotheses['reason'],seconds=time.perf_counter()-started);return result
        roots=self.worker.request('roots',worlds=hypotheses['worlds'])
        # All opponent choices are drawn before any candidate action is submitted.
        opponent_outputs=frozen_actions(self.p,[r['players'][foe] for r in roots],self.catalog,
            [derived_seed('f33-opponent-root',search_seed,w) for w in range(len(roots))],[0]*len(roots))
        opponent_actions=[out[2] for out in opponent_outputs]
        result['opponent_actions']=opponent_actions
        branches=self.worker.request('branches',opponent_actions=opponent_actions)
        result['attempted_branches']=len(branches)
        counters={b['index']:0 for b in branches}
        for round_index in range(self.settings['continuation_requests']+1):
            pending=[b for b in branches if b['frame']['outcome'] is None and not joint(b['frame'])]
            if not pending:break
            if round_index==self.settings['continuation_requests']:
                result.update(reason='Branch request budget exhausted',seconds=time.perf_counter()-started)
                return result
            players=[b['frame']['players'][s] for b in pending for s in SIDES]
            action_seeds=[derived_seed('f33-resolution',search_seed,b['world'],b['action'],round_index,s)
                for b in pending for s in SIDES]
            outputs=frozen_actions(self.p,players,self.catalog,action_seeds,[0]*len(players));steps=[]
            for i,b in enumerate(pending):
                actions={};ids={}
                for j,s in enumerate(SIDES):
                    player=players[2*i+j]
                    if not any(player['mask']):continue
                    actions[s]=outputs[2*i+j][2]
                    ids[s]=player['request_id']
                steps.append({'index':b['index'],'actions':actions,'request_ids':ids})
                counters[b['index']]+=len(actions)
            updated=self.worker.request('branch_steps',steps=steps)
            for b in updated:branches[b['index']]=b
        leaf_values=predictions(self.p,[b['frame']['players'][side] for b in branches],self.catalog)
        totals={a:[] for a,m in enumerate(input['player']['mask']) if m}
        for b,(_,value) in zip(branches,leaf_values):
            outcome=b['frame']['outcome'];score=outcome['rewards'][side] if outcome else value
            totals[b['action']].append(score)
            result['branches'].append({**b,'score':score,'predicted_value':value,
                'resolution_decisions':counters[b['index']]})
        if any(len(v)!=self.settings['worlds'] for v in totals.values()):raise ValueError('Incomplete action/world grid')
        scores=np.zeros(14)
        for a,values in totals.items():scores[a]=np.mean(values)
        probabilities=root_distribution(prior,scores,input['player']['mask'],self.settings['temperature'])
        action=sample(probabilities,derived_seed('f33-root-sample',search_seed))
        if not input['player']['mask'][action]:raise ValueError('Search selected illegal action')
        result.update(applied=True,action=action,probabilities=probabilities.tolist(),scores=scores.tolist(),
            seconds=time.perf_counter()-started)
        return result
