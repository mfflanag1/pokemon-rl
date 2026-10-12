"""One fixed-shape, observation-only actor path for live evaluation and tape audit."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.policy import action_kernel,predict_kernel
from pokemon_jax.engine import PlayerView
from pokemon_jax.views import reduce_view
from pokemon_jax_learner.model import encode
from pokemon_learner.core import require

ROWS=80
OPPONENTS=('random','damage-greedy','frozen-initial')


def request_prefix(final,number):
    """Recover an original live request from its own final protocol, never another channel."""
    require(type(number) is int and number>0,'Request index')
    count=0;turn=0
    for i,line in enumerate(final['protocol']):
        if line.startswith('|turn|'):turn=int(line[6:])
        if line.startswith('|request|'):
            count+=1
            if count==number:
                return {'side':final['side'],'request_id':count,'turn':turn,
                        'request':json.loads(line[9:]),'protocol':final['protocol'][:i+1]}
    raise ValueError('Missing original request')


@jax.jit
def keys_for(seed,opponent,pairs,seats,turns,roles):
    master=jax.random.fold_in(jax.random.key(1296646193,impl='threefry2x32'),seed)
    master=jax.random.fold_in(master,opponent)
    def key(pair,seat,turn,role):
        k=jax.random.fold_in(master,pair)
        for x in (seat,turn,role):k=jax.random.fold_in(k,x)
        return k
    return jax.vmap(key)(pairs,seats,turns,roles)


def decide(parameters,initial,frames,seed,opponent):
    """frames: at most 40 {pair,seat,views}; metadata only addresses independent RNG.

    Neither policy receives that metadata, targets, simulator state or the other
    player's observation. The inherited model is row-independent.
    """
    require(opponent in OPPONENTS and 0<len(frames)<=40,'Evaluation batch')
    headers=np.zeros((ROWS,11),np.int32);headers[:,1]=1
    histories=np.full((ROWS,8,6),-1,np.int32)
    addresses=np.zeros((ROWS,4),np.uint32)
    for i,f in enumerate(frames):
        require(f['seat'] in (0,1),'Candidate seat')
        for side,view in enumerate(f['views']):
            require(view['side']==f'p{side+1}','Player channel order')
            reduced=reduce_view(view);h=reduced['header'];require(h[1:3]==[0,0],'Actor must be live')
            headers[2*i+side]=h;histories[2*i+side]=reduced['history']
            addresses[2*i+side]=[f['pair'],f['seat'],h[0],int(side!=f['seat'])]
    obs=encode(PlayerView(jnp.asarray(headers),jnp.asarray(histories)))
    keys=keys_for(jnp.uint32(seed),jnp.uint32(OPPONENTS.index(opponent)),*[jnp.asarray(addresses[:,k]) for k in range(4)])
    ca,cl,cv=map(np.asarray,action_kernel(parameters,obs,keys))
    cp=np.asarray(predict_kernel(parameters,obs)[0])
    if opponent=='frozen-initial':oa=np.asarray(action_kernel(initial,obs,keys)[0])
    elif opponent=='damage-greedy':oa=np.ones(ROWS,dtype=np.int32)
    else:
        oa=np.asarray(jax.vmap(lambda k:jax.random.bits(jax.random.fold_in(k,0x52414e44),(),dtype=jnp.uint32))(keys)&jnp.uint32(1))
    require(np.isfinite(cl).all() and np.isfinite(cv).all() and np.isfinite(cp).all(),'Finite inference')
    result=[]
    for i,f in enumerate(frames):
        seat=f['seat'];index=2*i+seat
        actions=[int(ca[2*i+s] if s==seat else oa[2*i+s]) for s in (0,1)]
        require(all(a in (0,1) for a in actions),'Illegal evaluation action')
        result.append({'turn':int(headers[index,0]),'requests':[v['request_id'] for v in f['views']],
                       'actions':actions,'value':float(cv[index]),'quick_attack_probability':float(cp[index,1]),
                       'behavior_logp':float(cl[index])})
    return result
