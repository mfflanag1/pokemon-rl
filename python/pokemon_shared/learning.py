"""Architecture dispatch; identical finite-word NLL, Adam and sampled examples."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize as flat_initialize,forward as flat_forward,distribution
from pokemon_fit.learning import checked_step as flat_step,predictions as flat_predictions,validate_batch
from pokemon_fixed.math import adam_step
from pokemon_full_train.learning import derived_seed
from pokemon_fit.data import fingerprint
from pokemon_imitation.data import validate,summarize
from pokemon_balanced.sampling import classify
from pokemon_balanced.learning import NAMES
from pokemon_full.features import encode
from . import network
FLAT_SHAPES={'w':(928,64),'b':(64,),'pw':(64,14),'pb':(14,),'vw':(64,),'vb':()}

def shapes(arm):
    if arm not in ['flat','shared']:raise ValueError('Unknown architecture')
    return FLAT_SHAPES if arm=='flat' else network.SHAPES

def initial(seed,arm):
    shapes(arm);p=flat_initialize(seed) if arm=='flat' else network.initialize(seed)
    z=jax.tree.map(jnp.zeros_like,p);opt=(z,z,jnp.int32(0));check_state(p,opt,arm)
    return p,opt

def check_state(p,opt,arm):
    expected=shapes(arm)
    for group in [p,opt[0],opt[1]]:
        if set(group)!=set(expected):raise ValueError('Parameter inventory mismatch')
        for k,shape in expected.items():
            a=np.asarray(group[k])
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid parameter/moment')
    step=np.asarray(opt[2])
    if step.shape!=() or step.dtype!=np.int32 or not 0<=int(step)<2**31:raise ValueError('Invalid optimizer counter')
    if any(np.any(np.asarray(x)<0) for x in opt[1].values()):raise ValueError('Negative second moment')

def value_keys(arm):return ['vw','vb'] if arm=='flat' else ['value_w','value_b']

def shared_objective(p,batch):
    logits,_=network.forward(p,batch['features']);_,logs=distribution(logits,batch['mask'])
    return -jnp.mean(logs[jnp.arange(len(logits)),batch['targets']])

@jax.jit
def shared_update(p,m,v,step,batch):
    loss,grads=jax.value_and_grad(shared_objective)(p,batch)
    p,m,v,step,norm=adam_step(p,grads,m,v,step,lr=.0003,max_norm=1.)
    return p,(m,v,step),{'pre_update_nll':loss,'gradient_norm':norm}

@jax.jit
def shared_predictions(p,x,mask):
    logits,_=network.forward(p,x);return distribution(logits,mask)

def selected_batch(sampler,old,seed,index):
    validate_batch(old)
    data,ids=sampler.sample(derived_seed('f16-balanced-sample',seed,index))
    oi=np.random.default_rng(derived_seed('f15-old-sample',seed,index)).integers(0,128,64)
    batch={k:np.concatenate([data[k],old[k][oi]]) for k in data};validate_batch(batch)
    return batch,{'teacher_indices':ids.tolist(),'old_indices':oi.tolist(),'batch_fingerprint':fingerprint(batch)}

def step(p,opt,batch,arm):
    check_state(p,opt,arm);validate_batch(batch);before={k:np.asarray(p[k]).copy() for k in value_keys(arm)};n=int(opt[2])
    b={k:jnp.asarray(v) for k,v in batch.items()}
    if arm=='flat':p,opt,metrics=flat_step(p,opt,b)
    else:
        p,opt,metrics=shared_update(p,*opt,b);metrics={k:float(v) for k,v in metrics.items()}
    check_state(p,opt,arm)
    if int(opt[2])!=n+1 or not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Update counter/metrics mismatch')
    if any(not np.array_equal(before[k],np.asarray(p[k])) for k in before):raise ValueError('Unused value-head changed')
    return p,opt,metrics

def predictions(p,features,mask,arm):
    shapes(arm);out=[];fn=flat_predictions if arm=='flat' else shared_predictions
    if len(features)==0 or features.shape!=(len(mask),928) or mask.shape!=(len(features),14):raise ValueError('Prediction schema')
    if features.dtype!=np.float32 or mask.dtype!=np.bool_ or not np.isfinite(features).all() or not mask.any(1).all():raise ValueError('Prediction dtype/mask')
    for first in range(0,len(features),128):
        x=features[first:first+128];m=mask[first:first+128];n=len(x)
        if n<128:x=np.concatenate([x,np.repeat(x[:1],128-n,0)]);m=np.concatenate([m,np.repeat(m[:1],128-n,0)])
        pr,_=fn(p,x,m);out.append(np.asarray(pr)[:n])
    probs=np.concatenate(out)
    if not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs[~mask]!=0) or np.max(abs(probs.sum(1)-1))>5e-6:raise ValueError('Invalid probabilities')
    return probs

def probe(p,data,arm):
    validate(data);pr=predictions(p,data['features'],data['mask'],arm);labels=classify(data);families={}
    for i,name in enumerate(NAMES):
        ids=labels==i
        if not ids.any():raise ValueError('Missing action family')
        families[name]=summarize(pr[ids],data['mask'][ids],data['targets'][ids],data['game_index'][ids])
    return pr,{'total':summarize(pr,data['mask'],data['targets'],data['game_index']),'families':families}

def tactical(p,oracle,arm):
    records=oracle['records']
    if len(records)!=512:raise ValueError('Tactical inventory')
    encoded=[encode(r['player']) for r in records];x=np.stack([a for a,_ in encoded]);mask=np.stack([m for _,m in encoded]);pr=predictions(p,x,mask,arm);cells={};items=[]
    for r,a in zip(records,pr):
        item={'target_mass':float(a[r['target']]),'greedy_correct':int(a.argmax())==r['target'],'probabilities':a.tolist()}
        items.append(item);key=f"{r['spec']['panel']}/{r['spec']['context']}";cells.setdefault(key,[]).append(item)
    if set(cells)!={f'{p}/{c}' for p in ['train','variant'] for c in range(16)} or any(len(v)!=16 for v in cells.values()):raise ValueError('Tactical cell count')
    return {'predictions':items,'cells':{k:{'n':len(v),'target_mass':float(np.mean([x['target_mass'] for x in v])),'greedy_accuracy':float(np.mean([x['greedy_correct'] for x in v]))} for k,v in cells.items()}}

def mastery(t,thresholds):
    checks={k:v['greedy_accuracy']>=thresholds['minimum_greedy_accuracy'] and v['target_mass']>=thresholds['minimum_context_mass'] for k,v in t['cells'].items()}
    if len(checks)!=32:raise ValueError('Mastery cell count')
    return {'passed':all(checks.values()),'cells':checks}
