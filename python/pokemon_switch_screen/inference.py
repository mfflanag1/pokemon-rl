"""Fixed-shape actor inference using only one receiver's protocol per row."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.checkpoint import digest
from pokemon_switch_engine import engine,policy,views
from pokemon_switch_engine.scalar import require,SPECIES
from pokemon_switch_engine.damage import damage
from .provenance import PLAN

ROWS=64


def prefix(final,length):
    require(type(length) is int and 0<length<=len(final['protocol']),'Protocol prefix length')
    lines=final['protocol'][:length];count=turn=0;request=None
    for line in lines:
        if line.startswith('|request|'):count+=1;request=json.loads(line[9:])
        if line.startswith('|turn|'):turn=int(line[6:])
    return {'side':final['side'],'request_id':count,'turn':turn,'request':request,'protocol':lines}


def greedy(view):
    """Expected damage uses public active species and exact own HP; no private inputs."""
    mask=view['mask'];moves=[i for i in range(2) if mask[i]]
    if not moves:return next((i for i in range(2,4) if mask[i]),-1)
    own=view['own'][view['context'][2]];other=view['opponent'][view['context'][3]]
    require(other[0]>0,'Visible target species')
    def expected(m):
        return sum(23*damage(SPECIES[own[0]-1],SPECIES[other[0]-1],own[1],m,False,r)+
                   damage(SPECIES[own[0]-1],SPECIES[other[0]-1],own[1],m,True,r) for r in range(16))
    return max(moves,key=lambda m:(expected(m),-m))


@jax.jit
def address_keys(seed,opponent,mode,addresses):
    master=jax.random.key(PLAN['action_key_seed']);master=jax.random.fold_in(master,mode)
    master=jax.random.fold_in(jax.random.fold_in(master,seed),opponent)
    def one(row):
        k=master
        for i in range(4):k=jax.random.fold_in(k,row[i])
        return k
    return jax.vmap(one)(addresses)


@jax.jit
def random_actions(obs,keys):
    words=jax.vmap(lambda k:jax.random.bits(jax.random.fold_in(k,0x52414e44),(3,2),dtype=jnp.uint32))(keys)
    return policy.select(jnp.zeros((ROWS,4),jnp.float32),obs.mask,words)


def decide(parameters,initial,frames,seed,opponent,mode):
    require(opponent in PLAN['opponents'] and mode in ('smoke','development') and 0<len(frames)<=32,'Inference batch')
    shapes=[(2,9),(2,2),(2,4),(5,),(32,9),(4,)]
    arrays=[np.zeros((ROWS,*s),bool if i==5 else np.int32) for i,s in enumerate(shapes)]
    addresses=np.zeros((ROWS,4),np.uint32);reduced=[]
    for i,f in enumerate(frames):
        require(f['seat'] in (0,1),'Candidate seat');pair=[]
        for side,v in enumerate(f['views']):
            require(v['side']==f'p{side+1}','Receiving channel order');r=views.encode(v);pair.append(r)
            require(r['context'][1] in (0,1,2) and r['context'][0]<=12,'Live within admitted horizon')
            for k,name in enumerate(engine.PlayerView._fields):arrays[k][2*i+side]=r[name]
            addresses[2*i+side]=[f['pair'],f['seat'],f['group'],int(side!=f['seat'])]
        reduced.append(pair)
    obs=policy.encode(engine.PlayerView(*[jnp.asarray(a) for a in arrays]))
    keys=address_keys(jnp.uint32(seed),jnp.uint32(PLAN['opponents'].index(opponent)),jnp.uint32(mode=='smoke'),jnp.asarray(addresses))
    ca,cl,cv=map(np.asarray,policy.act(parameters,obs,keys))
    if opponent=='frozen-initial':oa=np.asarray(policy.act(initial,obs,keys)[0])
    elif opponent=='random':oa=np.asarray(random_actions(obs,keys))
    else:oa=np.array([greedy(v) for pair in reduced for v in pair]+[-1]*(ROWS-2*len(frames)))
    require(np.isfinite(cl).all() and np.isfinite(cv).all(),'Finite inference')
    result=[]
    for i,f in enumerate(frames):
        index=2*i+f['seat'];actions=[int(ca[2*i+s] if s==f['seat'] else oa[2*i+s]) for s in (0,1)]
        for side,a in enumerate(actions):
            m=reduced[i][side]['mask'];require(0<=a<4 and m[a] if any(m) else a==-1,'Legal action/wait')
        live=any(reduced[i][f['seat']]['mask'])
        result.append({'group':f['group'],'turn':f['views'][0]['turn'],'requests':[v['request_id'] for v in f['views']],
            'view_lengths':[len(v['protocol']) for v in f['views']],'view_hashes':[digest(v) for v in f['views']],
            'actions':actions,'candidate_value':float(cv[index]) if live else None,'candidate_logp':float(cl[index]) if live else None})
    return result
