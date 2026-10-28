"""Add a zero-initialized semantic input block; keep sealed action law and Adam."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import distribution,finite_parameters
from pokemon_fixed.math import adam_step
from .semantic import EXTRA

def extend(p):return {**p,'ew':jnp.zeros((EXTRA,64),jnp.float32)}
def forward(p,x,e):
    h=jnp.tanh(x@p['w']+p['b']+e@p['ew'])
    return h@p['pw']+p['pb'],jax.nn.sigmoid(h@p['vw']+p['vb'])
@jax.jit
def predictions(p,x,e,mask):
    logits,_=forward(p,x,e);return distribution(logits,mask)[0]
def objective(p,b):
    logits,_=forward(p,b['features'],b['extra']);_,logs=distribution(logits,b['mask'])
    return -jnp.mean(logs[jnp.arange(128),b['targets']])
@jax.jit
def update(p,m,v,step,b):
    loss,g=jax.value_and_grad(objective)(p,b)
    p,m,v,step,norm=adam_step(p,g,m,v,step,lr=.0003,max_norm=1.)
    return p,(m,v,step),{'pre_update_nll':loss,'gradient_norm':norm}
def checked(p,opt,b):
    if set(b)!={'features','extra','mask','targets'}:raise ValueError('Unexpected model fields')
    from pokemon_fit.learning import validate_batch
    validate_batch({k:b[k] for k in ['features','mask','targets']})
    e=np.asarray(b['extra'])
    if e.shape!=(128,EXTRA) or e.dtype!=np.float32 or not np.isfinite(e).all():raise ValueError('Bad semantic batch')
    p,opt,m=update(p,*opt,{k:jnp.asarray(v) for k,v in b.items()});finite_parameters(p);finite_parameters(opt[0]);finite_parameters(opt[1])
    metrics={k:float(v) for k,v in m.items()}
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite update')
    return p,opt,metrics


def load_checkpoint(directory,expected_input_hash):
    """Read all seven parameter/moment blocks; reject missing/extra state."""
    from pathlib import Path
    import json
    from pokemon_full_train.storage import digest
    from pokemon_full.policy import initialize
    directory=Path(directory);pointer=json.loads((directory/'latest.json').read_text())
    if Path(pointer['file']).name!=pointer['file']:raise ValueError('Invalid checkpoint filename')
    path=directory/pointer['file']
    if digest(path)!=pointer['sha256']:raise ValueError('Checkpoint digest mismatch')
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(data['metadata'].tobytes())
        if meta['input_hash']!=expected_input_hash:raise ValueError('Checkpoint input binding')
        expected=extend(initialize(meta['seed']))
        if set(data.files)!={'metadata'}|{prefix+'__'+k for prefix in ['p','m','v'] for k in expected}:raise ValueError('Checkpoint parameter inventory')
        states=[]
        for prefix in ['p','m','v']:
            state={}
            for key,template in expected.items():
                a=data[prefix+'__'+key]
                if a.shape!=template.shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Checkpoint shape/dtype/finite')
                state[key]=jnp.asarray(a)
            states.append(state)
    if type(meta['optimizer_step'])is not int or not 0<=meta['optimizer_step']<2**31:raise ValueError('Optimizer counter')
    return states[0],(states[1],states[2],jnp.asarray(meta['optimizer_step'],jnp.int32)),meta
