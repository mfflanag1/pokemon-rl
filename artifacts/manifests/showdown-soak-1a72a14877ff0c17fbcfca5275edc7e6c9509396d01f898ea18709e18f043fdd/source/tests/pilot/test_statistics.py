import unittest
import numpy as np
from pokemon_pilot.statistics import paired_summary, calibration


class StatisticsTests(unittest.TestCase):
    def test_pair_is_unit_and_seats_are_not_independent(self):
        # Opposite seats cancel exactly within every pair; game resampling would vary.
        x=np.tile([0.,1.],(20,1))
        r=paired_summary(x,seed=7,replicates=512,alpha=.05)
        self.assertEqual((r['mean'],r['lower'],r['upper']),(.5,.5,.5))
        self.assertEqual((r['pairs'],r['games']),(20,40))
        self.assertEqual(r['wins'],20);self.assertEqual(r['losses'],20)

    def test_paired_differences_keep_common_variation(self):
        x=np.array([[0,0],[1,1],[.5,.5]]*10,dtype=float)
        r=paired_summary(x,baseline=x.copy(),seed=3,replicates=512,alpha=.05)
        self.assertLess(r['lower'],r['upper'])
        self.assertEqual(r['improvement'],{'mean':0.,'lower':0.,'upper':0.})
        self.assertEqual(r,paired_summary(x,baseline=x,seed=3,replicates=512,alpha=.05))

    def test_extremes_and_draw_scoring(self):
        r=paired_summary(np.ones((12,2)),baseline=np.zeros((12,2)),seed=1,replicates=64,alpha=.01)
        self.assertEqual(r['improvement'],{'mean':1.,'lower':1.,'upper':1.})
        d=paired_summary(np.full((12,2),.5),seed=2,replicates=64,alpha=.05)
        self.assertEqual(d['draws'],24);self.assertEqual(d['mean'],.5)

    def test_reject_malformed_inputs(self):
        x=np.zeros((4,2))
        for bad in [[],[0,1],np.zeros((1,2)),np.zeros((3,3)),np.full((4,2),np.nan),np.full((4,2),.1),np.ones((4,2),dtype=bool)]:
            with self.subTest(bad=str(bad)),self.assertRaises(ValueError): paired_summary(bad,seed=1,replicates=64,alpha=.05)
        for kw in [{'seed':-1},{'seed':True},{'replicates':0},{'replicates':True},{'alpha':0},{'alpha':1},{'alpha':float('nan')},{'baseline':np.zeros((3,2))}]:
            args={'seed':1,'replicates':64,'alpha':.05};args.update(kw)
            with self.subTest(kw=kw),self.assertRaises(ValueError):paired_summary(x,**args)

    def test_calibration_raw_and_clipped_are_distinct(self):
        r=calibration([-1.,0.,.5,1.,2.],[0.,0.,1.,1.,1.])
        self.assertEqual(r['count'],5);self.assertAlmostEqual(r['raw_mse'],.45)
        self.assertAlmostEqual(r['clipped_brier'],.05);self.assertAlmostEqual(r['out_of_range_rate'],.4)
        self.assertEqual(len(r['bins']),10);self.assertEqual(sum(x['count'] for x in r['bins']),5)
        self.assertEqual(r['bins'][0]['count'],2);self.assertEqual(r['bins'][9]['count'],2)
        self.assertEqual(r['bins'][1]['mean_prediction'],None)
        for v,y in [([],[]),([float('inf')],[0]),([0],[.1]),([0],[0,1])]:
            with self.assertRaises(ValueError):calibration(v,y)

    def test_unrepresentable_finite_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            paired_summary(np.zeros((2,2)),seed=1,replicates=4,alpha=10**1000)
        with self.assertRaises(ValueError):calibration([1e308],[0])


if __name__=='__main__':unittest.main()
