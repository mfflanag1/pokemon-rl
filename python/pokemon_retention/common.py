"""F8 source admission, paired schedules, initialization and evaluation."""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import initialize,forward,distribution
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT,canonical,digest,load_checkpoint
from pokemon_decision.collect import evaluation_specs,target_action,retained_hp
from pokemon_fit.data import require,fingerprint,predecessor as f6_admission


def admit_predecessor(plan):
    path=ROOT/plan['predecessor_receipt'];r=json.loads(path.read_text());body=dict(r);h=body.pop('content_hash')
    require(h==plan['predecessor_content_hash']==hashlib.sha256(canonical(body)).hexdigest(),'F7 content binding')
    require(path.read_bytes()==(ROOT/'artifacts/manifests'/f'f7-decision-fit-{h}'/'receipt.json').read_bytes(),'F7 receipt archive')
    for rel,sha in r['source_files'].items():
        require(digest(ROOT/rel)==sha and digest(ROOT/r['source_snapshot_root']/rel)==sha,f'F7 source drift: {rel}')
    for rel,sha in r['evidence_files'].items():require(digest(ROOT/rel)==sha,f'F7 evidence drift: {rel}')
    require(r['run']==plan['predecessor_run'],'F7 run binding')
    f6_admission(json.loads((ROOT/'contracts/decision-fit-v1/plan.json').read_text()),all_files=True)
    return r


def initial_state(seed,arm,plan):
    require(seed in plan['seeds'] and arm in ['fresh','supervised'],'Unknown initialization')
    if arm=='fresh':p=initialize(seed)
    else:
        run=ROOT/plan['predecessor_run'];h=digest(run/'inputs/manifest.json')
        p,old,meta=load_checkpoint(run/str(seed)/plan['predecessor_checkpoint'],h)
        require(meta=={'seed':seed,'input_hash':h,'cohorts':1024,'optimizer_step':1024},'Wrong supervised checkpoint')
        require(int(old[2])==1024,'Wrong supervised budget')
    require(np.all(np.asarray(p['vw'])==0) and float(p['vb'])==0.,'Initial critic not 0.5')
    z=jax.tree.map(jnp.zeros_like,p)
    return p,(z,z,jnp.int32(0))


def training_specs(seed,update,plan):
    return [{'context':i%16,'panel':'train','episode':update*plan['rows_per_update']+i,
             'run_seed':derived_seed(plan['training_seed_namespace'],seed)} for i in range(plan['rows_per_update'])]


def eval_specs(plan):return evaluation_specs(plan['evaluation_seed'],plan['evaluation_repeats'])


def pair(records,reference):
    require(len(records)==len(reference),'Pair length mismatch')
    for a,b in zip(records,reference):
        for field in ['spec','actor_seed','side','player','provenance']:
            require(a[field]==b[field],f'Paired pre-action {field} mismatch')


def probe(pool,specs):
    """Branch every legal action; no policy parameters or optimizer enter here."""
    records=[]
    for offset in range(0,len(specs),len(pool.workers)):
        chunk=specs[offset:offset+len(pool.workers)];frames=pool.requests('start',chunk)
        sides=['p2' if s['context']%2 else 'p1' for s in chunk]
        players=[f['players'][side] for f,side in zip(frames,sides)]
        legal=[list(map(int,np.flatnonzero(encode(p)[1]))) for p in players]
        require(all(len(a)==7 for a in legal),'Unexpected legal branch count')
        scores=[{} for _ in chunk]
        for j in range(7):
            fresh=pool.requests('start',chunk)
            require(all(f['players'][side]==p for f,side,p in zip(fresh,sides,players)),'Probe prefix drift')
            results=pool.requests('decide',[{'action':a[j]} for a in legal])
            for i,r in enumerate(results):
                require(r['diagnostic_ended'] and r['reward']==retained_hp(players[i],r['frame']['players'][sides[i]]),'Probe reward reconstruction mismatch')
                scores[i][str(legal[i][j])]=r['reward']
        for spec,p,score,frame in zip(chunk,players,scores,frames):
            target=target_action(p,spec['context']);gap=score[str(target)]-max(v for k,v in score.items() if int(k)!=target)
            require(gap>0,'Fresh unique optimum failed')
            records.append({'spec':spec,'player':p,'scores':score,'target':target,'gap':gap,
                            'team_keys':[t['team_key'] for t in frame['provenance']['teams']]})
    return {'status':'passed','contexts':len(records),'branches':7*len(records),
            'minimum_gap':min(r['gap'] for r in records),'records':records}


def summarize(parameters,records,oracle):
    require(len(records)==len(oracle['records'])==512,'Incomplete evaluation')
    cells={};inputs=set();probs=[]
    for offset in range(0,len(records),8):
        chunk=records[offset:offset+8];encoded=[encode(r['player']) for r in chunk]
        logits,_=forward(parameters,jnp.asarray(np.stack([x for x,_ in encoded])))
        probabilities,_=distribution(logits,jnp.asarray(np.stack([m for _,m in encoded])))
        probs.extend(np.asarray(probabilities))
    for r,o,probability in zip(records,oracle['records'],probs):
        require(r['spec']==o['spec'] and r['player']==o['player'],'Evaluation/probe state mismatch')
        require(r['reward']==o['scores'][str(r['action'])],'Evaluation branch reward mismatch')
        target=o['target'];x,mask=encode(r['player']);inputs.add(fingerprint({'features':x,'mask':mask}))
        require(np.isfinite(probability).all() and np.all(probability>=0) and np.all(probability[~mask]==0) and abs(float(probability.sum())-1)<5e-6,'Invalid evaluation probability')
        key=f"{r['spec']['panel']}/{r['spec']['context']}"
        cells.setdefault(key,[]).append([float(probability[target]),int(probability.argmax())==target,r['action']==target,r['reward'],(r['value']-r['reward'])**2])
    names=['target_mass','greedy_accuracy','sampled_accuracy','mean_reward','local_value_mse']
    return {'cells':{k:{'n':len(v),**{name:float(np.mean(np.asarray(v)[:,i])) for i,name in enumerate(names)}} for k,v in cells.items()}}


def input_inventory(oracle,plan):
    prior=json.loads((ROOT/'artifacts/reviews/f6-decision-canary/mechanics-probe.json').read_text())['records']
    def key(r):
        x,m=encode(r['player']);return fingerprint({'features':x,'mask':m})
    pre={key(r) for r in prior if r['spec']['panel']=='train' and r['spec']['episode']<8}
    current={panel:{key(r) for r in oracle['records'] if r['spec']['panel']==panel} for panel in ['train','variant']}
    return {'supervised_training_unique_inputs':len(pre),'fresh_evaluation_unique_inputs':{k:len(v) for k,v in current.items()},
            'overlap_with_supervised_training':{k:len(v&pre) for k,v in current.items()},
            'same_templates_and_known_variant_species':True,'fresh_coordinates_not_guaranteed_unseen_inputs':True}
