import unittest
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize,distribution
from pokemon_matchup.network import extend as old_extend
from pokemon_terminal_matchup.actor import infer as old_infer
from pokemon_nonlinear_policy.network import extend,forward
from pokemon_nonlinear_policy.actor import infer,batched_actor
class Actor(unittest.TestCase):
    def test_seeded_initial_actor_matches_predecessor_including_wait_padding(self):
        old,_=old_extend(initialize(29));p,_=extend(old,29)
        rng=np.random.default_rng(7);x=rng.normal(0,.2,(16,1278)).astype(np.float32)
        mask=rng.random((16,14))>.7;mask[:8,0]=True;mask[8:]=False
        seeds=np.arange(16,dtype=np.uint32);counter=seeds*3
        expected=old_infer(old,x,mask,seeds,counter)
        for arm in ['control','nonlinear']:
            actual=infer(p,x,mask,seeds,counter,arm)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
            actions,logp,values=map(np.asarray,actual)
            np.testing.assert_array_equal(actions[8:],-1);np.testing.assert_array_equal(logp[8:],0)
            self.assertTrue(np.all(mask[np.arange(8),actions[:8]]))
            self.assertTrue(np.isfinite(values).all())
    def test_nonzero_residual_changes_policy_but_not_fixed_parameter_value(self):
        old,_=old_extend(initialize(29));p,_=extend(old,29)
        p={**p,'residual_out':jnp.linspace(-.6,.6,32)}
        x=np.random.default_rng(7).normal(0,.5,(16,1278)).astype(np.float32)
        mask=np.ones((16,14),bool);mask[:,4:8]=False
        a,v=forward(p,x,'control');b,w=forward(p,x,'nonlinear')
        np.testing.assert_array_equal(v,w);self.assertGreater(float(np.linalg.norm(a-b)),.01)
        pa,la=distribution(a,mask);pb,lb=distribution(b,mask)
        self.assertGreater(float(np.linalg.norm(pa-pb)),.001)
        seeds=np.arange(16,dtype=np.uint32)
        actions,logs,vals=map(np.asarray,infer(p,x,mask,seeds,seeds,'nonlinear'))
        np.testing.assert_allclose(logs,np.asarray(lb)[np.arange(16),actions],atol=2e-6)
        self.assertTrue(np.all(mask[np.arange(16),actions]));np.testing.assert_allclose(vals,w,atol=1e-7)
        np.testing.assert_array_equal(np.asarray(pb)[:,4:8],0)
    def test_input_api_rejects_unadmitted_arm_and_invalid_capacity(self):
        old,_=old_extend(initialize(29));p,_=extend(old,29)
        for arm,players,seeds,counters,capacity in [('unknown',[{}],[1],[0],1),('control',[],[],[],1),('control',[{}],[True],[0],1),('nonlinear',[{}],[1],[-1],1),('control',[{}],[1],[0],0)]:
            with self.assertRaises(ValueError):batched_actor(p,players,seeds,counters,capacity,arm,{})
if __name__=='__main__':unittest.main()
