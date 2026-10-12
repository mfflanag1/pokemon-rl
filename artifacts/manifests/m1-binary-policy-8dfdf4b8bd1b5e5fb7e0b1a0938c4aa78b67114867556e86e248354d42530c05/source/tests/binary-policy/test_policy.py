import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.policy import (parts, distribution_kernel, threshold_words,
    select_words, action_kernel, act, FLOOR, DOMAIN)
from pokemon_binary.loss import masked_loss
from pokemon_jax_learner.model import ActorInput, initialize


class PolicyTests(unittest.TestCase):
    def test_integer_threshold_boundaries(self):
        rng = np.random.default_rng(410)
        # Every admitted exponent, significand edges and deterministic random significands.
        bits = np.concatenate([((e << 23) | np.r_[0,1,2,0x3fffff,0x7ffffe,0x7fffff,
            rng.integers(0,2**23,400,dtype=np.uint32)]).astype(np.uint32) for e in range(103,126)])
        rare = np.r_[bits.view(np.float32), np.float32(.5)].astype(np.float32)
        thresholds = np.asarray(jax.jit(threshold_words)(rare))
        words, expected, masses = [], [], []
        for r, (hi, lo) in zip(rare, thresholds):
            numerator, denominator = float(r).as_integer_ratio()
            integer, remainder = divmod(numerator * 2**64, denominator)
            self.assertEqual(remainder, 0)
            self.assertEqual((int(hi)<<32) | int(lo), integer)
            for w in (0, integer-1, integer, integer+1, 2**64-1):
                words.append([w>>32,w&0xffffffff]); expected.append(w < integer); masses.append(r)
        n = len(words)
        actual = select_words(jnp.array(masses), jnp.ones(n,jnp.int32),
                              jnp.ones((n,2),bool), jnp.array(words,jnp.uint32))
        np.testing.assert_array_equal(actual, np.asarray(expected,np.int32))

    def test_mirrored_tails_and_masks(self):
        logits = np.array([[-27.038734,0],[0,-27.038734],[0,0],[1e30,-1e30],
                           [100,-100],[-100,100],[0,0]], np.float32)
        mask = np.array([[1,1],[1,1],[1,1],[1,1],[0,1],[1,0],[0,0]],bool)
        probs, logs, rare_action, rare = map(np.asarray, parts(logits, mask))
        self.assertEqual(rare[0],FLOOR);self.assertEqual(rare[1],FLOOR)
        np.testing.assert_array_equal(probs[0],probs[1][::-1])
        np.testing.assert_array_equal(logs[0],logs[1][::-1])
        self.assertLess(logs[0,1], 0)
        self.assertAlmostEqual(float(logs[0,1]),float(np.log1p(-FLOOR)),places=13)
        np.testing.assert_array_equal(probs[4:],[[0,1],[1,0],[0,0]])
        np.testing.assert_array_equal(logs[4:],0)
        words = jnp.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],jnp.uint32)
        np.testing.assert_array_equal(select_words(rare,rare_action,mask,words),[0,1,1,1,1,0,-1])
        words = jnp.full((7,2),jnp.uint32(0xffffffff))
        np.testing.assert_array_equal(select_words(rare,rare_action,mask,words),[1,0,0,0,1,0,-1])

    def test_selected_log_gradient_and_floor(self):
        for d in (-8.,-1.,0.,1.,8.,-27.,27.):
            x = jnp.array([[d,0]],jnp.float32)
            for a in (0,1):
                g = np.asarray(jax.grad(lambda l:distribution_kernel(l,jnp.ones((1,2),bool))[1][0,a])(x))[0]
                r = float(parts(x,jnp.ones((1,2),bool))[3][0])
                p0 = r if d < 0 else 1-r
                expected = np.array([1-p0,-(1-p0)]) if a == 0 else np.array([-p0,p0])
                if abs(d)>20: expected[:]=0
                np.testing.assert_allclose(g,expected,atol=1e-7,rtol=3e-6)
        for mask in ([[1,0]],[[0,1]],[[0,0]]):
            g = jax.grad(lambda l:jnp.sum(distribution_kernel(l,jnp.array(mask,bool))[1]))(jnp.zeros((1,2)))
            np.testing.assert_array_equal(g,0)

    def test_keyed_actor_is_word_selector_and_reorders(self):
        n=64;p=initialize(9)
        obs=ActorInput(jnp.arange(n*59,dtype=jnp.float32).reshape(n,59)/500,
                       jnp.ones((n,2),bool))
        keys=jax.random.split(jax.random.key(42),n)
        result=act(p,obs,keys)
        logits=obs.features@p['policy_w']+p['policy_b']
        _,logs,ra,r=parts(logits,obs.mask)
        words=jax.vmap(lambda key:jax.random.bits(jax.random.fold_in(key,DOMAIN),(2,),dtype=jnp.uint32))(keys)
        expected=select_words(r,ra,obs.mask,words)
        np.testing.assert_array_equal(result[0],expected)
        np.testing.assert_array_equal(result[1],logs[jnp.arange(n),expected])
        order=np.random.default_rng(9).permutation(n)
        perm=action_kernel(p,ActorInput(obs.features[order],obs.mask[order]),keys[order])
        for a,b in zip(result,perm):np.testing.assert_array_equal(np.asarray(a)[order],b)
        chunks=[action_kernel(p,ActorInput(obs.features[s],obs.mask[s]),keys[s]) for s in (slice(0,17),slice(17,None))]
        for i in range(3):np.testing.assert_array_equal(result[i],np.concatenate([c[i] for c in chunks]))
        with self.assertRaises(ValueError):act(p,{'features':obs.features},keys)
        with self.assertRaises(ValueError):act(p,obs,keys[:3])

    def test_loss_ratio_padding_and_forced_rows(self):
        logits=jnp.array([[0.,0.],[27.,0.],[3.,-2.],[jnp.nan,jnp.nan]])
        mask=jnp.array([[1,1],[1,1],[0,1],[0,0]],bool)
        valid=jnp.array([1,1,1,0],bool);a=jnp.array([1,1,1,-1])
        old=distribution_kernel(logits[:3],mask[:3])[1][jnp.arange(3),a[:3]]
        old=jnp.r_[old,jnp.nan]
        args=(jnp.array([0.,1.,2.,jnp.nan]),mask,a,old,jnp.array([1.,-1.,1.,jnp.nan]),
              jnp.array([1.,0.,3.,jnp.nan]),valid)
        (loss,metrics),g=jax.value_and_grad(masked_loss,has_aux=True)(logits,*args)
        self.assertTrue(np.isfinite(loss));self.assertEqual(float(metrics['max_ratio_error']),0.)
        np.testing.assert_array_equal(g[2:],0)
        empty_args=(*args[:-1],jnp.zeros(4,bool))
        (loss,m),g=jax.value_and_grad(masked_loss,has_aux=True)(logits,*empty_args)
        self.assertEqual(float(loss),0.);np.testing.assert_array_equal(g,0)
        self.assertTrue(all(float(v)==0 for v in m.values()))
        # Finite differences exercise clipped PPO and entropy together away from kinks.
        x=np.array([[.7,-.1],[-.4,.9]],np.float32)
        ma=jnp.ones((2,2),bool);ac=jnp.array([0,1]);v=jnp.zeros(2)
        old=distribution_kernel(jnp.zeros((2,2)),ma)[1][jnp.arange(2),ac]
        args=(v,ma,ac,old,jnp.array([1.,-1.]),v,jnp.ones(2,bool))
        gradient=np.asarray(jax.grad(masked_loss,has_aux=True)(jnp.array(x),*args)[0])
        fd=np.zeros_like(x)
        for i,j in np.ndindex(x.shape):
            plus=x.copy();minus=x.copy();plus[i,j]+=.001;minus[i,j]-=.001
            fd[i,j]=(float(masked_loss(plus,*args)[0])-float(masked_loss(minus,*args)[0]))/.002
        np.testing.assert_allclose(gradient,fd,atol=4e-5,rtol=3e-3)

if __name__=='__main__':unittest.main()
