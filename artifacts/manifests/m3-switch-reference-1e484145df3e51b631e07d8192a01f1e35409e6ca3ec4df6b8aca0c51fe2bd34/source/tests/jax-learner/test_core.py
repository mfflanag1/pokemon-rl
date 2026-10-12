import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_learner import core as ref
from pokemon_jax_learner import core as port


class CoreParity(unittest.TestCase):
    def args(self):
        logits=np.array([[.2,-.1,.4],[.5,.2,-.3],[99,0,0]],np.float32)
        mask=np.array([[1,1,0],[1,1,1],[0,1,0]],bool);a=np.array([0,2,1],np.int32)
        old=ref.distribution(logits,mask)[1][np.arange(3),a]
        return logits,np.array([.2,.8,.1],np.float32),mask,a,old,np.array([.7,-.4,3],np.float32),np.array([1,.1,.4],np.float32)

    def test_distribution_and_autodiff(self):
        args=self.args();a=ref.ppo_loss_grad(*args,entropy_coef=.13);b=port.ppo_loss_grad(*args,entropy_coef=.13)
        for k in a[0]:self.assertAlmostEqual(a[0][k],b[0][k],delta=3e-6)
        for x,y in zip(a[1:],b[1:]):np.testing.assert_allclose(x,y,rtol=3e-5,atol=3e-6)
        self.assertTrue(np.all(np.asarray(b[1])[~args[2]]==0))
        loss=lambda l:port.loss_kernel(l,*args[1:],.2,.5,.13)[0]
        np.testing.assert_allclose(jax.grad(loss)(jnp.asarray(args[0])),a[1],rtol=3e-5,atol=3e-6)

    def test_gae_interleaved_terminal_and_cut(self):
        args=([0,0,1],[.2,.3,.4],[2,-1,-1],[0,.8,0],np.array([False,False,True]),[10,11,10])
        for x,y in zip(ref.linked_gae(*args,gamma=1,lam=.5),port.linked_gae(*args,gamma=1,lam=.5)):
            np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)

    def test_ppo_clipped_signs_and_zero_advantage(self):
        args=list(self.args());args[4]=np.asarray(args[4]-np.log(np.array([2,.5,1],np.float32)),np.float32)
        for adv in ([2,-3,0],[-2,3,0]):
            args[5]=np.array(adv,np.float32)
            a=ref.ppo_loss_grad(*args);b=port.ppo_loss_grad(*args)
            np.testing.assert_allclose(a[1],b[1],atol=3e-6,rtol=3e-5)

    def test_adam_accumulation_and_clipping(self):
        a={'w':np.array([1,2,-3],np.float32)};b={k:jnp.asarray(v) for k,v in a.items()}
        ar=ref.Adam(a,lr=.003,max_norm=.5);br=port.Adam(b,lr=.003,max_norm=.5)
        for i in range(40):
            g={'w':np.array([i%3-.5,2-i%4,.03],np.float32)}
            self.assertAlmostEqual(ar.apply(a,g),br.apply(b,g),delta=3e-6)
            for x,y in [(a['w'],b['w']),(ar.m['w'],br.m['w']),(ar.v['w'],br.v['w'])]:np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)
        self.assertEqual(br.step,40)

    def test_validation_and_no_partial_optimizer_mutation(self):
        args=list(self.args());args[3]=np.array([2,2,1])
        with self.assertRaises(ValueError):port.ppo_loss_grad(*args)
        with self.assertRaises(ValueError):port.distribution(np.zeros((1,2),np.float32),np.zeros((1,2),bool))
        with self.assertRaises(ValueError):port.distribution(np.zeros((1,2),np.float64),np.ones((1,2),bool))
        with self.assertRaises(ValueError):port.linked_gae([0,1],[0,0],[1,-1],[0,0],np.array([False,True]),[0,1])
        p={'w':jnp.ones(2,jnp.float32)};opt=port.Adam(p)
        with self.assertRaises(ValueError):opt.apply(p,{'w':jnp.array([float('nan'),0])})
        self.assertEqual(opt.step,0);np.testing.assert_array_equal(p['w'],[1,1])

    def test_exact_clip_boundaries_keep_unclipped_derivative(self):
        logits=np.zeros((2,2),np.float32);mask=np.ones((2,2),bool);actions=np.zeros(2,np.int32)
        # ratio 1.5 at positive A upper bound, .5 at negative A lower bound.
        old=np.array([np.log(1/3),0],np.float32);advantages=np.array([1,-1],np.float32)
        args=(logits,np.zeros(2,np.float32),mask,actions,old,advantages,np.zeros(2,np.float32))
        metrics,expected,_=ref.ppo_loss_grad(*args,clip=.5)
        _,actual,_=port.ppo_loss_grad(*args,clip=.5)
        self.assertGreater(float(np.linalg.norm(expected)),0)
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)
        np.testing.assert_allclose(actual,[[-.375,.375],[.125,-.125]],atol=3e-6,rtol=0)

    def test_no_gradient_through_behavior_or_targets(self):
        args=self.args()
        def loss(old,adv,returns):return port.loss_kernel(*args[:4],old,adv,returns,.2,.5,.13)[0]
        gradients=jax.grad(loss,argnums=(0,1,2))(*[jnp.asarray(a) for a in args[4:]])
        for gradient in gradients:np.testing.assert_array_equal(gradient,np.zeros(3,np.float32))

    def test_central_difference_of_autodiff_and_shared_linear_model(self):
        from pokemon_jax_learner.model import ActorInput, initialize, forward, model_loss_grad
        rng=np.random.default_rng(123);parameters=initialize(3)
        features=rng.normal(0,.1,(7,59)).astype(np.float32);mask=np.ones((7,2),bool);actions=np.arange(7,dtype=np.int32)%2
        logits,values=forward(parameters,ActorInput(jnp.asarray(features),jnp.asarray(mask)))
        old=ref.distribution(np.asarray(logits),mask)[1][np.arange(7),actions]
        adv=rng.normal(size=7).astype(np.float32);returns=rng.normal(size=7).astype(np.float32)
        _,gl,gv=ref.ppo_loss_grad(np.asarray(logits),np.asarray(values),mask,actions,old,adv,returns,entropy_coef=.13)
        (_,metrics),actual=model_loss_grad(parameters,jnp.asarray(features),jnp.asarray(mask),jnp.asarray(actions),jnp.asarray(old),jnp.asarray(adv),jnp.asarray(returns),.2,.5,.13)
        expected={'policy_w':features.T@gl,'policy_b':gl.sum(axis=0),'value_w':features.T@gv,'value_b':gv.sum()}
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=3e-6,rtol=3e-5)
        args=list(self.args());_,gl,gv=port.ppo_loss_grad(*args,entropy_coef=.13)
        for i,gradient in [(0,gl),(1,gv)]:
            for pos in np.ndindex(args[i].shape):
                original=args[i][pos].copy();eps=np.float32(.002)
                args[i][pos]=original+eps;plus=port.ppo_loss_grad(*args,entropy_coef=.13)[0]['loss']
                args[i][pos]=original-eps;minus=port.ppo_loss_grad(*args,entropy_coef=.13)[0]['loss']
                args[i][pos]=original
                self.assertAlmostEqual((plus-minus)/(2*eps),float(gradient[pos]),delta=7e-5)
