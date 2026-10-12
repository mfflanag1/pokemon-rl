import unittest
import numpy as np
from pokemon_jax.statistics import law_tests


class StatisticalGates(unittest.TestCase):
    def test_family_and_missing_strata(self):
        rows=law_tests(np.zeros((32,2),bool),np.full((32,2),85),np.zeros(32,int),np.ones(32,int),np.zeros(32,bool))
        self.assertEqual(len(rows),592)
        self.assertTrue(all(r['p_value']==1 for r in rows if r['n']==0))
        self.assertTrue(any(r['p_value']<.01/1184 for r in rows))

    def test_cross_stream_corruption_rejected(self):
        # Marginal rolls are perfectly balanced; coupled actors violate independence.
        roll=np.tile(np.arange(85,101),64)
        rolls=np.stack([roll,roll],axis=1)
        rows=law_tests(np.zeros((1024,2),bool),rolls,np.arange(1024)%2,np.arange(1024)%4+1)
        equality=next(r for r in rows if r['name']=='roll-equal')
        self.assertLess(equality['p_value'],.01/1184)
