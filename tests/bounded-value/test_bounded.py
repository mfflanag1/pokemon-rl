import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_bounded import policy, learner, checkpoint
from pokemon_switch_engine import policy as old_policy, learner as old_learner
from pokemon_learner.checkpoint import digest, wire


def model_fixture():
    rng=np.random.default_rng(9143);p=policy.initialize(431)
    p={k:np.asarray(v).copy() for k,v in p.items()}
    p['vw']=rng.normal(0,.12,32).astype(np.float32);p['vb']=np.asarray(.1,np.float32)
    x=rng.normal(0,.3,(32,323)).astype(np.float32)
    mask=rng.random((32,4))>.35;mask[:,0]=True;mask[-4:]=False
    valid=mask.any(1);actions=np.where(valid,0,-1).astype(np.int32)
    logits,_=policy.forward(p,policy.ActorInput(jnp.asarray(x),jnp.asarray(mask)))
    old=np.asarray(old_policy.distribution(logits,jnp.asarray(mask))[1])[:,0]+np.linspace(-.5,.5,32,dtype=np.float32)
    adv=rng.normal(size=32).astype(np.float32);returns=rng.random(32).astype(np.float32)
    zeros=np.zeros(32,np.float32)
    buffer=learner.Buffer(x,mask,actions,old,zeros,valid,zeros,np.zeros(32,bool),
                          np.full(32,-1,np.int32),zeros,adv,returns)
    return {k:jnp.asarray(v) for k,v in p.items()},jax.tree.map(jnp.asarray,buffer)


class BoundedTests(unittest.TestCase):
    def test_initial_actor_exact_and_value_half_all_masks(self):
        p=policy.initialize(431);m=jnp.array([[bool(i&(1<<a)) for a in range(4)] for i in range(16)])
        obs=policy.ActorInput(jnp.zeros((16,323)),m);keys=jax.random.split(jax.random.key(97),16)
        old=old_policy.act(p,obs,keys);new=policy.act(p,obs,keys)
        np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
        np.testing.assert_array_equal(new[2],np.where(np.asarray(m.any(1)),.5,0))
        np.testing.assert_array_equal(policy.forward(p,obs)[0],old_policy.forward(p,obs)[0])
        with self.assertRaises(ValueError):policy.forward(p,{'features':obs.features,'mask':obs.mask})

    def test_extreme_actor_sampler_inheritance(self):
        p=policy.initialize(431);p={**p,'pb':jnp.array([1000.,-1000.,-1000.,1000.])}
        m=jnp.array([[bool(i&(1<<a)) for a in range(4)] for i in range(16)])
        obs=policy.ActorInput(jnp.ones((16,323)),m);keys=jax.random.split(jax.random.key(97),16)
        a=old_policy.act(p,obs,keys);b=policy.act(p,obs,keys)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        for row,action in enumerate(np.asarray(b[0])):
            self.assertTrue(action==-1 if not m[row].any() else m[row,action])

    def test_saturated_value_heads_finite_bounded_and_gradients(self):
        p=policy.initialize(431);obs=policy.ActorInput(jnp.ones((4,323)),jnp.ones((4,4),bool))
        for b in (-1000.,-20.,0.,20.,1000.):
            q={**p,'vb':jnp.asarray(b,jnp.float32)}
            value=policy.forward(q,obs)[1];grad=jax.grad(lambda z:policy.forward({**q,'vb':z},obs)[1].sum())(q['vb'])
            self.assertTrue(np.isfinite(value).all() and np.isfinite(grad))
            self.assertTrue(np.all((np.asarray(value)>=0)&(np.asarray(value)<=1)))
        self.assertEqual(float(jax.grad(lambda z:policy.forward({**p,'vb':z},obs)[1].sum())(jnp.float32(0))),1.)

    def test_full_network_numpy_gradient_parity(self):
        from pokemon_bounded import reference
        p,b=model_fixture();params={k:np.asarray(v) for k,v in p.items()}
        expected,grad=reference.loss_grad(params,np.asarray(b.features),np.asarray(b.mask),np.asarray(b.actions),
                                         np.asarray(b.old_logp),np.asarray(b.advantages),np.asarray(b.returns),np.asarray(b.valid))
        (value,metrics),actual=jax.value_and_grad(policy.objective,has_aux=True)(p,b)
        self.assertAlmostEqual(float(value),expected['loss'],delta=2e-6)
        for key in actual:np.testing.assert_allclose(actual[key],grad[key],rtol=1e-4,atol=2e-6,err_msg=key)
        logits,values=reference.forward(params,np.asarray(b.features))
        other=policy.forward(p,policy.ActorInput(b.features,b.mask))
        np.testing.assert_allclose(other[0],logits,atol=2e-6);np.testing.assert_allclose(other[1],values,atol=2e-6)

    def test_nan_padding_zero_batch_and_detached_targets(self):
        p,b=model_fixture();fn=jax.value_and_grad(policy.objective,has_aux=True)
        expected=fn(p,b)
        dirty=b._replace(features=jnp.where(b.valid[:,None],b.features,jnp.nan),
                         old_logp=jnp.where(b.valid,b.old_logp,jnp.nan),
                         advantages=jnp.where(b.valid,b.advantages,jnp.nan),returns=jnp.where(b.valid,b.returns,jnp.nan))
        actual=fn(p,dirty)
        for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(x,y)
        empty=dirty._replace(valid=jnp.zeros_like(b.valid))
        result=fn(p,empty)
        for x in jax.tree.leaves(result):np.testing.assert_array_equal(x,jnp.zeros_like(x))
        for field in ('old_logp','advantages','returns'):
            grad=jax.grad(lambda x:policy.objective(p,b._replace(**{field:x}))[0])(getattr(b,field))
            np.testing.assert_array_equal(grad,jnp.zeros_like(grad))

    def test_target_range_gaps_terminal_and_cutoff_grid(self):
        rng=np.random.default_rng(861);valid=jnp.asarray(rng.random((14,64))>.4)
        valid=valid.at[0].set(True)
        values=jnp.asarray(rng.choice([0.,np.nextafter(np.float32(0),np.float32(1)),.5,np.nextafter(np.float32(1),np.float32(0)),1.],(14,64)),jnp.float32)
        bootstrap=jnp.asarray(rng.random(64),jnp.float32)
        winners=jnp.asarray(rng.choice([-1,0,1],32),jnp.int32)
        reward,terminal,nxt,boot,adv,ret=old_learner.targets(values,valid,winners,bootstrap)
        self.assertTrue(np.all(np.asarray(ret)>=-2e-6) and np.all(np.asarray(ret)<=1+2e-6))
        self.assertEqual(int(reward.sum()),int((np.asarray(winners)!=-1).sum()))
        self.assertEqual(int(terminal.sum()),2*int((np.asarray(winners)!=-1).sum()))
        # Boundedness does not remove an unsupported constant cutoff fixed point.
        all_valid=jnp.ones((3,2),bool);v=jnp.full((3,2),.9,jnp.float32)
        fixed=old_learner.targets(v,all_valid,jnp.array([-1]),jnp.array([.9,.9]))
        np.testing.assert_allclose(fixed[-2],0,atol=1e-7)
        np.testing.assert_allclose(fixed[-1],.9,atol=1e-7)

    def test_collection_actor_and_environment_parity_at_same_parameters(self):
        p=policy.initialize(431)
        original=old_learner.collect(p,jnp.uint32(0),jnp.uint32(431))
        new=learner.collect(p,jnp.uint32(0),jnp.uint32(431));learner.validate_buffer(*new)
        for key in ('features','mask','actions','old_logp','valid','rewards','terminal','next_index'):
            np.testing.assert_array_equal(getattr(original[0],key),getattr(new[0],key),err_msg=key)
        self.assertEqual(checkpoint.fingerprint(original[1]),checkpoint.fingerprint(new[1]))
        live=np.asarray(new[0].valid);np.testing.assert_array_equal(np.asarray(new[0].values)[live],.5)
        self.assertTrue(np.any(np.asarray(new[0].bootstrap)==.5))

    def test_checkpoint_identity_corruption_and_next_update(self):
        state=learner.start(431);admitted=checkpoint.admission()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';checkpoint.save(path,state,431,admitted)
            original=learner.transaction(state,431)
            resumed=learner.transaction(checkpoint.load(path,431,admitted),431)
            self.assertEqual(checkpoint.fingerprint(original),checkpoint.fingerprint(resumed))
            with self.assertRaises(FileExistsError):checkpoint.save(path,state,431,admitted)
            raw=path.read_bytes();doc=json.loads(raw)
            for field in ('format','seed','admission_hash','counter','variance'):
                changed=copy.deepcopy(doc);body=changed['body']
                if field=='format':body['format']='switch-screen-checkpoint-1'
                elif field=='seed':body['seed']=641
                elif field=='admission_hash':body['admission_hash']='0'*64
                elif field=='counter':body['counters'][2]=1
                else:body['tables'][2]['vb']['hex']=np.array(-1,np.float32).tobytes().hex()
                changed['sha256']=digest(body);path.write_bytes(wire(changed))
                with self.assertRaises(ValueError):checkpoint.load(path,431,admitted)
            path.write_bytes(raw[:-5])
            with self.assertRaises((ValueError,json.JSONDecodeError)):checkpoint.load(path,431,admitted)


if __name__=='__main__':unittest.main()
