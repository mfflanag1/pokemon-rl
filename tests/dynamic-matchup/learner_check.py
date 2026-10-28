import copy,tempfile,unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full import policy
from pokemon_full_train import learning as sealed
from pokemon_matchup import network,learning
from pokemon_dynamic_matchup.common import initial,read,ROOT,PLAN,save,load,same_state
from pokemon_dynamic_matchup.features import FEATURES

def zeros(p):
    z=jax.tree.map(jnp.zeros_like,p);return z,z,jnp.int32(0)

def batch(p,n=16,real=True):
    rng=np.random.default_rng(61);x=rng.normal(0,.2,(n,FEATURES)).astype(np.float32)
    if not real:x[:,928:]=0
    mask=rng.random((n,14))>.4;mask[:,0]=True
    actions=np.array([rng.choice(np.flatnonzero(m)) for m in mask],np.int32)
    _,logs=policy.distribution(network.forward(p,jnp.asarray(x))[0],jnp.asarray(mask))
    ratio=np.resize(np.array([1.4,.6,1.05,.95],np.float32),n)
    return dict(features=x,mask=mask,actions=actions,
        old_logp=np.asarray(logs)[np.arange(n),actions]-np.log(ratio),
        advantages=np.resize(np.array([.8,-.7,-.4,.6],np.float32),n),
        returns=rng.uniform(.05,.95,n).astype(np.float32),valid=np.ones(n,bool))

def equal(a,b,exact=True):
    aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b);assert ta==tb
    for x,y in zip(aa,bb):
        if exact:np.testing.assert_array_equal(x,y)
        else:np.testing.assert_allclose(x,y,rtol=3e-6,atol=3e-7)

class Learner(unittest.TestCase):
    def test_all_three_function_preserving_initials(self):
        for seed in [7103,8209,9311]:
            p,o,_=initial(None,seed,'zero');q,qo,_=initial(None,seed,'matchup');same_state(p,o,q,qo)
            self.assertEqual(sum(v.size for v in p.values()),60806);self.assertEqual(int(o[2]),0)
            self.assertTrue(all(np.all(np.asarray(v)==0) for g in o[:2] for v in g.values()))
            base={k:v for k,v in p.items() if k in network.FLAT_SHAPES}
            x=batch(p,128)['features'];equal(policy.forward(base,x[:,:928]),network.forward(p,x))

    def test_zero_residual_objective_and_base_gradients(self):
        p,_=network.extend(policy.initialize(8));b=batch(p,real=False)
        base={k:v for k,v in p.items() if k in network.FLAT_SHAPES};old={**b,'features':b['features'][:,:928]}
        loss,m=learning.objective(p,b);l,n=sealed.objective(base,old)
        equal((loss,{k:m[k] for k in n}),(l,n))
        g=jax.grad(lambda q:learning.objective(q,b)[0])(p)
        equal({k:g[k] for k in base},jax.grad(lambda q:sealed.objective(q,old)[0])(base))
        for k in ['matchup_policy','matchup_value']:np.testing.assert_array_equal(g[k],0)

    def test_numpy_loss_and_adapter_gradients(self):
        p,_=network.extend(policy.initialize(8));p={**p,'matchup_policy':jnp.linspace(-.1,.1,25),'matchup_value':jnp.linspace(-.02,.02,350)}
        b=batch(p);logits,values=network.forward(p,b['features']);pr,logs=map(np.asarray,policy.distribution(logits,b['mask']))
        ratio=np.exp(logs[np.arange(16),b['actions']]-b['old_logp']);adv=b['advantages']
        actor=-np.minimum(ratio*adv,np.clip(ratio,.8,1.2)*adv).mean()
        value=.5*np.mean((np.asarray(values)-b['returns'])**2);entropy=-np.sum(pr*logs,axis=1).mean()
        loss,m=learning.objective(p,b)
        self.assertAlmostEqual(float(loss),float(actor+.5*value-.01*entropy),places=6)
        g=jax.grad(lambda q:learning.objective(q,b)[0])(p);rng=np.random.default_rng(4)
        for key in ['matchup_policy','matchup_value','w','vw']:
            d=rng.normal(size=p[key].shape).astype(np.float32);d/=np.linalg.norm(d);eps=.003
            f=lambda z:learning.objective({**p,key:z},b)[0]
            finite=float((f(p[key]+eps*d)-f(p[key]-eps*d))/(2*eps));analytic=float(jnp.sum(g[key]*d))
            np.testing.assert_allclose(finite,analytic,rtol=.08,atol=3e-5)
        self.assertGreater(float(jnp.linalg.norm(g['matchup_policy'])),0)
        self.assertGreater(float(jnp.linalg.norm(g['matchup_value'])),0)
        for key in ['old_logp','advantages','returns']:
            np.testing.assert_array_equal(jax.grad(lambda z:learning.objective(p,{**b,key:z})[0])(b[key]),0)

    def test_two_epoch_optimizer_and_checkpoint_recovery(self):
        p,o=network.extend(policy.initialize(5));b=batch(p,1024);del b['valid']
        q,qo,metrics=learning.optimize_batch(p,o,b,17,'matchup')
        self.assertEqual(len(metrics),8);self.assertEqual(int(qo[2]),8)
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);save(d,p,o,'h',5,'matchup',0,0);r,ro=load(d,'h',5,'matchup',0,0)
            r,ro,other=learning.optimize_batch(r,ro,b,17,'matchup');same_state(q,qo,r,ro);self.assertEqual(metrics,other)
        # A nonzero feature block must make the residual train.
        self.assertGreater(float(jnp.linalg.norm(q['matchup_policy'])),0)
        self.assertGreater(float(jnp.linalg.norm(q['matchup_value'])),0)

    def test_zero_control_adapter_stays_zero(self):
        p,o=network.extend(policy.initialize(5));b=batch(p,1024,False);del b['valid']
        q,qo,_=learning.optimize_batch(p,o,b,19,'zero')
        for group in [q,*qo[:2]]:
            for key in ['matchup_policy','matchup_value']:np.testing.assert_array_equal(group[key],0)
        self.assertFalse(np.array_equal(q['pw'],p['pw']))
        b['features'][0,-1]=1
        with self.assertRaises(ValueError):learning.optimize_batch(p,o,b,19,'zero')

    def test_invalid_padding_and_frozen_targets(self):
        p,o=network.extend(policy.initialize(3));clean=batch(p,4);padded={k:np.concatenate([v,v]) for k,v in clean.items()}
        padded['valid'][4:]=False
        for k in ['features','old_logp','advantages','returns']:padded[k][4:]=np.nan
        padded['actions'][4:]=-999;padded['mask'][4:]=False
        equal(learning.objective(p,clean),learning.objective(p,padded),False)
        equal(jax.grad(lambda q:learning.objective(q,clean)[0])(p),jax.grad(lambda q:learning.objective(q,padded)[0])(p),False)
        learning.update(p,o,padded)
        empty={**padded,'valid':np.zeros(8,bool)};loss,m=learning.objective(p,empty)
        self.assertEqual(float(loss),0);self.assertTrue(all(float(v)==0 for v in m.values()))

    def test_guards(self):
        p,o=network.extend(policy.initialize(4));b=batch(p)
        for bad in [{**b,'features':np.zeros((16,928),np.float32)},{**b,'actions':np.full(16,14,np.int32)},
                    {**b,'returns':np.full(16,1.5,np.float32)},{**b,'extra':1},{**b,'mask':np.zeros((16,14),bool)}]:
            with self.assertRaises(ValueError):learning.update(p,o,bad)
        for opt in [(o[0],o[1],np.int32(2**31-1)),(o[0],o[1],np.int64(0)),
                    (o[0],{**o[1],'vb':jnp.float32(-1)},o[2])]:
            with self.assertRaises(ValueError):learning.update(p,opt,b)

if __name__=='__main__':unittest.main()
