import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax_learner import core as reference
from pokemon_fixed.math import masked_loss, adam_step, bias_correction


class MathTests(unittest.TestCase):
    def test_padded_loss_gradients_match_compacted_reference(self):
        rng = np.random.default_rng(84)
        for count in (1, 7, 16):
            valid = np.arange(16) < count
            rng.shuffle(valid)
            logits = rng.normal(size=(16, 2)).astype('float32')
            mask = rng.random((16, 2)) > .5
            mask[:, 0] = True
            actions = np.zeros(16, np.int32)
            values, old, adv, ret = [rng.normal(size=16).astype('float32') for _ in range(4)]
            args = (logits, values, mask, actions, old, adv, ret)
            ref, gl, gv = reference.ppo_loss_grad(*[a[valid] for a in args], entropy_coef=.01)
            # Padding may contain poisoned transport sentinels; it must be sanitized first.
            for a in (logits, values, old, adv, ret): a[~valid] = np.nan
            actions[~valid] = -1
            mask[~valid] = False
            (_, metrics), grads = jax.jit(jax.value_and_grad(masked_loss, argnums=(0, 1), has_aux=True))(*args, valid)
            for k in ref: np.testing.assert_allclose(metrics[k], ref[k], atol=3e-6, rtol=3e-5)
            for g, expected in zip(grads, (gl, gv)):
                np.testing.assert_allclose(np.asarray(g)[valid], expected, atol=3e-6, rtol=3e-5)
                np.testing.assert_array_equal(np.asarray(g)[~valid], 0)

    def test_empty_loss_and_gradients_zero(self):
        logits = jnp.full((8,2), jnp.nan)
        values = jnp.full(8,jnp.nan)
        args = (logits, values, jnp.zeros((8,2),bool), jnp.full(8,-1,jnp.int32), values, values, values, jnp.zeros(8,bool))
        (_, metrics), grads = jax.jit(jax.value_and_grad(masked_loss,argnums=(0,1),has_aux=True))(*args)
        for value in metrics.values(): self.assertEqual(float(value),0)
        for value in grads: np.testing.assert_array_equal(value,0)

    def test_adam_accumulation_and_late_bias(self):
        rng=np.random.default_rng(9)
        p={'w':jnp.asarray(rng.normal(size=(7,2)).astype('float32')),'b':jnp.zeros(2)}
        refp=dict(p); ref=reference.Adam(refp,lr=.003)
        m=jax.tree.map(jnp.zeros_like,p); v=jax.tree.map(jnp.zeros_like,p); step=jnp.int32(0)
        fn=jax.jit(adam_step)
        for i in range(256):
            g={k:jnp.asarray(rng.normal(size=x.shape).astype('float32')) for k,x in p.items()}
            p,m,v,step,norm=fn(p,g,m,v,step)
            refnorm=ref.apply(refp,g)
            for x,y in ((p,refp),(m,ref.m),(v,ref.v)):
                for k in x: np.testing.assert_allclose(x[k],y[k],atol=3e-6,rtol=3e-5)
            self.assertEqual(int(step),i+1)
            np.testing.assert_allclose(norm,refnorm,atol=3e-6,rtol=3e-5)
        for i in (1,2,100,10000,17320,20000,20001,10000000):
            got=jax.jit(bias_correction)(jnp.int32(i))
            np.testing.assert_array_equal(np.asarray(got),np.array([1-.9**i,1-.999**i],np.float32))


if __name__ == '__main__': unittest.main()
