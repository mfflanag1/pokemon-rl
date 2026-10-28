"""Public feature/action permutations with explicit new-slot to old-slot maps."""
import numpy as np
from pokemon_full.policy import distribution
from .network import forward

def perm(p,n):
    a=np.asarray(p)
    if a.shape!=(n,) or a.dtype.kind not in 'iu' or set(a.tolist())!=set(range(n)):raise ValueError('Invalid permutation')
    return a

def remap(x,mask,targets,own,opponent,moves):
    own,opponent,moves=perm(own,6),perm(opponent,6),perm(moves,4)
    if x.shape!=(len(x),928) or x.dtype!=np.float32 or mask.shape!=(len(x),14) or mask.dtype!=np.bool_ or targets.shape!=(len(x),) or targets.dtype!=np.int32:raise ValueError('Remapping schema')
    if not np.isfinite(x).all() or np.any(targets<0) or np.any(targets>=14) or not mask[np.arange(len(x)),targets].all():raise ValueError('Remapping input validity')
    y=x.copy();mon=x[:,:768].reshape(-1,12,64);y[:,:384]=mon[:,own].reshape(-1,384);y[:,384:768]=mon[:,6+opponent].reshape(-1,384)
    y[:,768:864]=x[:,768:864].reshape(-1,4,24)[:,moves].reshape(-1,96)
    actions=np.concatenate([moves,4+moves,8+own]);m=mask[:,actions].copy();t=np.argsort(actions)[targets].astype(np.int32)
    return y,m,t,actions

def verify(p,data):
    from pokemon_balanced.sampling import classify
    labels=classify(data);ids=np.concatenate([np.flatnonzero(labels==i)[:8] for i in range(4)])
    x=data['features'][ids];m=data['mask'][ids];t=data['targets'][ids]
    y,n,u,a=remap(x,m,t,[3,0,5,1,4,2],[2,5,1,4,0,3],[2,0,3,1])
    l,v=map(np.asarray,forward(p,x));k,w=map(np.asarray,forward(p,y));pr=np.asarray(distribution(l,m)[0]);qr=np.asarray(distribution(k,n)[0])
    ld=float(np.max(abs(k-l[:,a])));pd=float(np.max(abs(qr-pr[:,a])));vd=float(np.max(abs(w-v)))
    if max(ld,pd,vd)>1e-5 or not n[np.arange(len(u)),u].all():raise ValueError('Permutation/law equivariance failure')
    return {'rows':len(ids),'max_logit_error':ld,'max_probability_error':pd,'max_value_error':vd,'targets_legal':True,'tolerance':1e-5}
