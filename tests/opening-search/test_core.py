import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
from pokemon_opening_search.core import root_distribution,sample,joint,load_frozen,predictions,COMMAND
from pokemon_opening_search.campaign import scientific,audit_summary
from pokemon_worker_watch.transport import Worker


class RootTests(unittest.TestCase):
    def test_exact_fixed_opponent_backups_and_mixed_root(self):
        # Twenty fixed opponent mixtures, 2x2 and RPS. This checks the declared
        # regularized response, not equilibrium solving or exploitability.
        for seed in range(20):
            rng=np.random.default_rng(seed)
            for payoff in (np.asarray([[1.,0.],[0.,1.]]),
                           np.asarray([[.5,0,1],[1,.5,0],[0,1,.5]])):
                n=len(payoff);opponent=rng.dirichlet(np.ones(n));q=payoff@opponent
                prior=np.zeros(14);prior[:n]=1/n;scores=np.zeros(14);scores[:n]=q
                mask=np.arange(14)<n;out=root_distribution(prior,scores,mask,.1)
                exact=np.exp((q-q.max())/.1);exact/=exact.sum()
                np.testing.assert_allclose(out[:n],exact,rtol=1e-12,atol=1e-12)
                self.assertEqual(out[n:].sum(),0);self.assertTrue((out[:n]>0).all())

    def test_equal_backups_preserve_prior(self):
        p=np.zeros(14);p[[0,4,9]]=[.1,.3,.6]
        np.testing.assert_allclose(root_distribution(p,np.ones(14)*.5,p>0,.1),p)

    def test_illegal_scores_and_mass_rejected(self):
        p=np.zeros(14);p[0]=1;mask=p>0
        for value in (float('nan'),float('inf'),-1,2):
            q=np.zeros(14);q[0]=value
            with self.assertRaises(ValueError):root_distribution(p,q,mask,.1)
        with self.assertRaises(ValueError):root_distribution(np.ones(14),np.zeros(14),mask,.1)

    def test_root_sampling_repeat_and_legality(self):
        p=np.zeros(14);p[[3,7,12]]=[.2,.5,.3]
        for seed in range(200):
            self.assertEqual(sample(p,seed),sample(p,seed));self.assertIn(sample(p,seed),(3,7,12))

    def test_pivot_and_first_turn_not_leaf(self):
        frame={'players':{s:{'phase':'move','turn':1,'mask':[True]*14} for s in ('p1','p2')}}
        self.assertFalse(joint(frame))
        for p in frame['players'].values():p['turn']=2
        self.assertTrue(joint(frame));frame['players']['p1']['phase']='replace';self.assertFalse(joint(frame))

    def test_replay_removes_timing_only(self):
        self.assertEqual(scientific({'seconds':1,'x':[{'seconds':2,'v':3}]}),{'x':[{'v':3}]})
        self.assertNotEqual(scientific({'action':1}),scientific({'action':2}))

    def test_waiting_and_terminal_masks_have_finite_zero_distribution(self):
        p,catalog,_=load_frozen(7103)
        with tempfile.TemporaryDirectory(prefix='f33-empty-mask-') as directory:
            worker=Worker(Path(directory),command=COMMAND)
            try:
                frame=worker.request('start',run_seed=1027263300,episode_index=0,split='development',
                    swap=False,tactical_side=None)
                waiting=copy.deepcopy(frame['players']['p1']);waiting.update(mask=[False]*14,phase='wait',active_moves=[])
                probabilities,value=predictions(p,[waiting],catalog)[0]
                np.testing.assert_array_equal(probabilities,np.zeros(14));self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value,0);self.assertLessEqual(value,1)
            finally:worker.close()

if __name__=='__main__':unittest.main()
