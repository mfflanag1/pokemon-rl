import copy
from fractions import Fraction
import json
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner.core import ppo_loss_grad
from pokemon_learner.checkpoint import digest,wire
from pokemon_switch_engine import policy,learner,checkpoint


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=learner.start(101)
        cls.result=checkpoint.transaction(cls.state,101)

    def test_exact_conditional_leaf_support_all_masks(self):
        masks=np.array([[bool(i&(1<<a)) for a in range(4)] for i in range(16)])
        for base in [[.7,-.2,.3,1.1],[1000,-1000,1000,-1000],[-1000,-1000,1000,1000]]:
            logits=jnp.tile(jnp.array(base,jnp.float32),(16,1));mask=jnp.array(masks)
            _,_,rare_action,rare,nodemasks=policy.nodes(logits,mask)
            rare=np.asarray(rare).reshape(16,3);ra=np.asarray(rare_action).reshape(16,3);nm=np.asarray(nodemasks)
            probs,logs=map(np.asarray,policy.distribution(logits,mask))
            for row in range(16):
                exact=[]
                for a in range(4):
                    mass=Fraction(1);words=np.full((16,3,2),2**32-1,np.uint32)
                    for node,target in [(0,a//2),(1+a//2,a%2)]:
                        r=Fraction(float(rare[row,node]));threshold=int(r*2**64)
                        if nm[row,node].all():
                            mass*=r if target==ra[row,node] else 1-r
                            w=threshold-1 if target==ra[row,node] else threshold
                            words[row,node]=[w>>32,w&0xffffffff]
                        elif not nm[row,node,target]:mass=Fraction(0)
                    exact.append(mass)
                    if masks[row,a]:
                        self.assertEqual(int(policy.select(logits,mask,jnp.array(words))[row]),a)
                        self.assertAlmostEqual(float(np.exp(float(logs[row,a])))/float(mass),1,places=5)
                    else:self.assertEqual(mass,0);self.assertEqual(probs[row,a],0)
                self.assertEqual(sum(exact),int(masks[row].any()))
            self.assertEqual(int(policy.select(logits,mask,jnp.zeros((16,3,2),jnp.uint32))[0]),-1)

    def test_numpy_categorical_ppo_gradient_parity(self):
        rng=np.random.default_rng(234);n=64
        logits=rng.normal(size=(n,4)).astype(np.float32);mask=rng.random((n,4))>.35;mask[:,0]=True
        actions=np.argmax(mask,axis=1).astype(np.int32);values=rng.normal(size=n).astype(np.float32)
        old=np.asarray(policy.distribution(jnp.array(logits),jnp.array(mask))[1])[np.arange(n),actions]+rng.uniform(-.5,.5,n).astype(np.float32)
        advantages=rng.normal(size=n).astype(np.float32);returns=rng.normal(size=n).astype(np.float32)
        expected,gl,gv=ppo_loss_grad(logits,values,mask,actions,old,advantages,returns,entropy_coef=.01)
        fn=lambda l,v:policy.loss(l,v,jnp.array(mask),jnp.array(actions),jnp.array(old),jnp.array(advantages),jnp.array(returns),jnp.ones(n,bool))
        (loss,metrics),grad=jax.value_and_grad(fn,(0,1),has_aux=True)(jnp.array(logits),jnp.array(values))
        self.assertAlmostEqual(float(loss),expected['loss'],places=6)
        np.testing.assert_allclose(grad[0],gl,rtol=5e-5,atol=3e-7);np.testing.assert_allclose(grad[1],gv,rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(np.asarray(grad[0])[~mask],0)

    def test_extreme_and_padding_finite_zero_gradients(self):
        mask=jnp.array([[1,0,1,0],[0,1,0,0],[0,0,0,0]],bool);valid=jnp.array([1,1,0],bool)
        logits=jnp.array([[1000,-1000,-1000,1000],[1000,1000,-1000,1000],[jnp.nan]*4])
        def loss(l):return policy.loss(l,jnp.array([0.,0.,jnp.nan]),mask,jnp.array([2,1,-1]),jnp.array([-16.635532,0.,jnp.nan]),jnp.array([1.,1.,jnp.nan]),jnp.array([1.,0.,jnp.nan]),valid)[0]
        value,grad=jax.value_and_grad(loss)(logits)
        self.assertTrue(np.isfinite(value));self.assertTrue(np.isfinite(grad).all())
        np.testing.assert_array_equal(grad,0)
        with self.assertRaises(ValueError):policy.forward(self.state.parameters,{'features':jnp.zeros((1,323)),'mask':jnp.ones((1,4),bool)})

    def test_three_seed_categorical_bandit_canary(self):
        # A four-action numerical learning canary, not Pokemon strength evaluation.
        @jax.jit
        def step(logits,key):
            mask=jnp.ones((128,4),bool);rows=jnp.tile(logits,(128,1));words=jax.random.bits(key,(128,3,2),dtype=jnp.uint32)
            actions=policy.select(rows,mask,words);_,logs=policy.distribution(rows,mask)
            old=logs[jnp.arange(128),actions];reward=(actions==3).astype(jnp.float32)
            def objective(l):return policy.loss(jnp.tile(l,(128,1)),jnp.zeros(128),mask,actions,old,reward-.25,reward,jnp.ones(128,bool),entropy_coef=0)[0]
            return logits-.3*jax.grad(objective)(logits)
        for seed in (101,211,307):
            logits=jnp.zeros(4)
            for i in range(128):logits=step(logits,jax.random.fold_in(jax.random.key(seed),i))
            self.assertGreater(float(policy.distribution(logits[None,:],jnp.ones((1,4),bool))[0][0,3]),.9)

    def test_collection_linkage_and_original_next_update(self):
        candidate,buffer,final,_=self.result;learner.validate_buffer(buffer,final)
        self.assertGreater(int(buffer.terminal.sum()),0)
        np.testing.assert_array_equal(buffer.bootstrap,0)  # Initial value head is exactly zero.
        # Checkpoint followed by the ORIGINAL next update, then fresh load/re-execution.
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';checkpoint.save(path,candidate,101)
            original=checkpoint.transaction(candidate,101)
            resumed=checkpoint.transaction(checkpoint.load(path,101),101)
            self.assertEqual(checkpoint.fingerprint(original),checkpoint.fingerprint(resumed))
            self.assertGreater(int((original[1].bootstrap!=0).sum()),0)
            self.assertEqual(int(original[0].step),4);self.assertEqual(int(original[0].next_game),64)

    def test_transaction_failure_preserves_original(self):
        before=checkpoint.fingerprint(self.state)
        for stage in ('collect','update'):
            with self.assertRaises(RuntimeError):checkpoint.transaction(self.state,101,fault=stage)
            self.assertEqual(checkpoint.fingerprint(self.state),before)

    def test_checkpoint_admission_corruption_and_atomicity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.json';checkpoint.save(path,self.result[0],101);raw=path.read_bytes();doc=json.loads(raw)
            with self.assertRaises(RuntimeError):checkpoint.save(path,self.state,101,interrupt=True)
            self.assertEqual(path.read_bytes(),raw);self.assertTrue(path.with_name(path.name+'.partial').exists())
            with self.assertRaisesRegex(ValueError,'preserve'):checkpoint.save(path,self.state,101)
            self.assertEqual(path.read_bytes(),raw)
            for mutation in ('checksum','sources','counter','runtime','dtype','seed','phase','negative_variance'):
                obj=copy.deepcopy(doc);b=obj['body']
                if mutation=='checksum':obj['sha256']='0'*64
                elif mutation=='sources':b['sources']={}
                elif mutation=='counter':b['counters'][2]+=1
                elif mutation=='runtime':b['runtime']['jax']='wrong'
                elif mutation=='dtype':b['tables'][0]['w']['dtype']='<f8'
                elif mutation=='seed':b['seed']+=1
                elif mutation=='phase':b['phase']='collecting'
                else:b['tables'][2]['vb']['hex']=np.array(-1,np.float32).tobytes().hex()
                if mutation!='checksum':obj['sha256']=digest(b)
                path.write_bytes(wire(obj))
                with self.assertRaises((ValueError,TypeError)):checkpoint.load(path,101)
            path.write_text('{"body":0,"body":1,"sha256":""}')
            with self.assertRaises(ValueError):checkpoint.load(path,101)

    def test_explicit_wait_gap_and_cutoff_targets(self):
        values=jnp.array([[.1,.2],[.3,.4],[.5,.6]],jnp.float32)
        valid=jnp.array([[1,1],[0,1],[1,0]],bool)
        rewards,terminal,nxt,boot,adv,ret=learner.targets(values,valid,jnp.array([-1]),jnp.array([.7,.8]))
        np.testing.assert_array_equal(nxt,[[4,3],[-1,-1],[-1,-1]])
        np.testing.assert_allclose(boot,[[0,0],[0,.8],[.7,0]],atol=1e-7)
        np.testing.assert_allclose(adv,[[.59,.58],[0,.4],[.2,0]],atol=1e-6)
        self.assertEqual(int(terminal.sum()),0);self.assertEqual(float(rewards.sum()),0)


if __name__=='__main__':unittest.main()
