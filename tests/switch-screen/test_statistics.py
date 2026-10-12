import json
import math
import unittest
import numpy as np
from pokemon_switch_screen.statistics import summarize


class Statistics(unittest.TestCase):
    def test_cutoffs_are_bounds(self):
        r=summarize([[1,-1],[0,1]], [[-1,0],[1,0]])
        self.assertEqual((r['wins'],r['losses'],r['cutoffs']),(2,1,1))
        self.assertEqual((r['outcome_lower'],r['outcome_upper']),(0.5,0.75))
        self.assertEqual((r['baseline_lower'],r['baseline_upper']),(0.25,0.5))
        self.assertEqual((r['secured_win_change'],r['worst_case_improvement'],r['best_case_improvement']),(0.25,0,0.5))
        json.dumps(r,allow_nan=False)

    def test_no_false_improvement_from_censoring(self):
        r=summarize(np.ones((576,2),int),np.full((576,2),-1))
        self.assertEqual(r['secured_win_change'],1)
        self.assertEqual(r['worst_case_improvement'],0)
        self.assertLess(r['adjusted_improvement_lower'],0)

    def test_hand_calculated_pair_radius(self):
        n=576;r=summarize(np.ones((n,2),int),np.zeros((n,2),int))
        radius=math.sqrt(math.log(120)/(2*n))
        self.assertAlmostEqual(r['adjusted_win_lower'],1-radius)
        self.assertAlmostEqual(r['adjusted_improvement_lower'],1-2*radius)
        self.assertEqual((r['pairs'],r['games']),(n,2*n))

    def test_symmetry_and_extremes(self):
        a=np.array([[1,-1],[0,1],[-1,0]])
        r=summarize(a,a);s=summarize(a[:,::-1],a[:,::-1])
        self.assertEqual(r,s);self.assertEqual(r['secured_win_change'],0)
        self.assertLess(r['worst_case_improvement'],0)
        json.dumps(summarize(a,a,alpha=5e-324,family_size=10000),allow_nan=False)

    def test_reject_inputs(self):
        good=np.zeros((2,2),int)
        for bad in ([],[1,0],[[1,0]],[[2,0],[0,1]],good.astype(float),good.astype(bool),good.astype(complex),good.astype(object),np.zeros((100001,2),int)):
            with self.subTest(bad=str(bad)[:40]),self.assertRaises(ValueError):summarize(bad,good)
        for key,values in [('alpha',[True,0,1,-1,float('nan'),float('inf'),10**1000,'0.05']),('family_size',[True,0,-1,1.0,10001,'6'])]:
            for value in values:
                with self.subTest(key=key,value=str(value)[:30]),self.assertRaises(ValueError):summarize(good,good,**{key:value})

if __name__=='__main__':unittest.main()
