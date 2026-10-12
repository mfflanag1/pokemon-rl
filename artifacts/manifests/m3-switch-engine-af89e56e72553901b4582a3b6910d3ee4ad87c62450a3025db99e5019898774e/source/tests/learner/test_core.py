import unittest
import numpy as np
from pokemon_learner.core import F32, linked_gae, distribution, ppo_loss_grad, Adam


class MathTest(unittest.TestCase):
    def test_hand_gae_interleaved_players_terminal_and_cut(self):
        # p1: V=.2 -> .4 -> terminal reward 1. p2: V=.3 -> cut at own V=.8.
        a,r=linked_gae([0,0,1],[.2,.3,.4],[2,-1,-1],[0,.8,0],np.array([False,False,True]),[10,11,10],gamma=1,lam=.5)
        np.testing.assert_allclose(a,[.5,.5,.6],rtol=0,atol=1e-5)
        np.testing.assert_allclose(r,[.7,.8,1],rtol=0,atol=1e-5)
        # Separate discounted three-step hand result: deltas .16, .24, .6;
        # recurrence multiplier .45 => A=.3895,.51,.6; R=.5895,.91,1.2.
        a,r=linked_gae([0,.1,1.2],[.2,.4,.6],[1,2,-1],[0,0,0],np.array([False,False,True]),[1,1,1],gamma=.9,lam=.5)
        np.testing.assert_allclose(a,[.3895,.51,.6],rtol=0,atol=1e-5)
        np.testing.assert_allclose(r,[.5895,.91,1.2],rtol=0,atol=1e-5)

    def test_bad_linkage_and_terminal_bootstrap_reject(self):
        with self.assertRaisesRegex(ValueError,'Cross-player'):
            linked_gae([0,1],[0,0],[1,-1],[0,0],np.array([False,True]),[0,1])
        with self.assertRaisesRegex(ValueError,'Terminal'):
            linked_gae([1],[0],[-1],[9],np.array([True]),[0])
        with self.assertRaisesRegex(ValueError,'forward'):
            linked_gae([0],[0],[0],[0],np.array([False]),[0])

    def args(self, logits, mask=None, actions=None, adv=None, old=None, entropy=0):
        logits=np.array(logits,dtype=F32);n=len(logits)
        mask=np.ones_like(logits,dtype=bool) if mask is None else np.array(mask,dtype=bool)
        actions=np.zeros(n,dtype=np.int64) if actions is None else np.array(actions)
        logp=distribution(logits,mask)[1]
        return (logits,np.zeros(n,dtype=F32),mask,actions,
                logp[np.arange(n),actions] if old is None else np.array(old,dtype=F32),
                np.ones(n,dtype=F32) if adv is None else np.array(adv,dtype=F32),np.zeros(n,dtype=F32))

    def test_same_policy_ratios_and_hand_log_ratio(self):
        args=self.args([[0,0,0,0],[1,-1,3,0]])
        metrics,*_=ppo_loss_grad(*args)
        self.assertLessEqual(metrics['max_ratio_error'],1e-5)
        metrics,*_=ppo_loss_grad(*self.args([[0,0]],old=[np.log(.25)]))
        self.assertAlmostEqual(metrics['max_ratio_error'],1,delta=1e-5)
        self.assertAlmostEqual(metrics['actor_loss'],-1.2,delta=1e-5)

    def test_single_legal_action_and_multi_candidate_forced_decision(self):
        args=self.args([[3,99,-4,5]],mask=[[False,False,True,False]],actions=[2],adv=[3])
        metrics,gl,gv=ppo_loss_grad(*args,entropy_coef=.2)
        self.assertLessEqual(abs(metrics['entropy']),1e-7)
        self.assertLessEqual(float(np.max(np.abs(gl))),1e-7)
        self.assertAlmostEqual(metrics['actor_loss'],-3,delta=1e-5)
        args=self.args([[0,0,2,1]],mask=[[True,True,False,False]],actions=[0])
        _,gl,_=ppo_loss_grad(*args)
        self.assertGreater(float(np.linalg.norm(gl)),.1)
        self.assertEqual(float(gl[0,2]),0.)

    def test_clipping_both_advantage_signs(self):
        # Positive A, ratio=2 clips; negative A, ratio=.5 clips.
        args=self.args([[0,0],[0,0]],old=[np.log(.25),np.log(1.)],adv=[2,-3])
        metrics,gl,_=ppo_loss_grad(*args)
        self.assertAlmostEqual(metrics['actor_loss'],0,delta=1e-5)
        np.testing.assert_array_equal(gl,np.zeros_like(gl))
        # Opposite sides must continue to have gradients.
        args=self.args([[0,0],[0,0]],old=[np.log(.25),np.log(1.)],adv=[-2,3])
        _,gl,_=ppo_loss_grad(*args)
        self.assertGreater(float(np.linalg.norm(gl)),.1)

    def test_analytic_gradients_against_central_difference(self):
        args=list(self.args([[.2,-.1,.4],[.5,.2,-.3]],adv=[.7,-.4]))
        args[1][:]=[.2,.8];args[-1][:]=[1,.1]
        metrics,gl,gv=ppo_loss_grad(*args,entropy_coef=.13)
        for index,expected in [(0,gl),(1,gv)]:
            for pos in np.ndindex(args[index].shape):
                original=args[index][pos].copy();epsilon=F32(.002)
                args[index][pos]=original+epsilon;plus=ppo_loss_grad(*args,entropy_coef=.13)[0]['loss']
                args[index][pos]=original-epsilon;minus=ppo_loss_grad(*args,entropy_coef=.13)[0]['loss']
                args[index][pos]=original
                self.assertAlmostEqual((plus-minus)/(2*epsilon),float(expected[pos]),delta=5e-5)

    def test_masked_probabilities_and_nonfinite_rejection(self):
        mask=np.array([[True,False,True]])
        p,_=distribution(np.array([[0,1e20,0]],dtype=F32),mask)
        np.testing.assert_array_equal(p,[[.5,0,.5]])
        with self.assertRaisesRegex(ValueError,'Empty'):distribution(np.zeros((1,3),dtype=F32),~np.ones((1,3),dtype=bool))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):distribution(np.array([[np.nan,0,0]],dtype=F32),mask)
        args=list(self.args([[0,0]],mask=[[True,False]]));args[3]=np.array([1])
        with self.assertRaisesRegex(ValueError,'Known-forbidden'):ppo_loss_grad(*args)

    def test_adam_first_step_and_nonfinite(self):
        parameters={'w':np.array([1.,2.],dtype=F32)};adam=Adam(parameters,lr=.03,max_norm=10.)
        adam.apply(parameters,{'w':np.array([2.,-3.],dtype=F32)})
        np.testing.assert_allclose(parameters['w'],[.97,2.03],atol=1e-6,rtol=0)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):adam.apply(parameters,{'w':np.array([np.inf,0],dtype=F32)})
