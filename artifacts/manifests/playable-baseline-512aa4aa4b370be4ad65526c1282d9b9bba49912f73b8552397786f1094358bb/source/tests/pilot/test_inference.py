import copy
import json
from pathlib import Path
import unittest
import numpy as np
from pokemon_pilot.inference import decide,request_prefix
from pokemon_pilot.provenance import mechanics_seed,roster
from pokemon_jax.views import reduce_view
from pokemon_jax_learner.model import initialize

ROOT=Path(__file__).resolve().parents[2]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=json.loads((ROOT/'tests/showdown-chance/fixtures/player-case.json').read_text())
        cls.parameters=initialize(101)

    def test_prefix_reconstructs_all_original_live_views(self):
        final=self.case['turns'][-1]['views']
        live=[self.case['initial']]+[t['views'] for t in self.case['turns'][:-1]]
        for pair in live:
            for side,v in enumerate(pair):
                self.assertEqual(reduce_view(v),reduce_view(request_prefix(final[side],v['request_id'])))
        with self.assertRaises(ValueError):request_prefix(final[0],99)

    def test_row_independence_and_same_observation_same_law(self):
        a={'pair':0,'seat':0,'views':self.case['initial']}
        b={'pair':1,'seat':1,'views':self.case['turns'][0]['views']}
        x=decide(self.parameters,self.parameters,[a],101,'random')[0]
        y=decide(self.parameters,self.parameters,[b,a],101,'random')[1]
        self.assertEqual(x,y)
        changed=copy.deepcopy(a);changed['pair']=999
        z=decide(self.parameters,self.parameters,[changed],101,'random')[0]
        self.assertEqual(x['value'],z['value']);self.assertEqual(x['quick_attack_probability'],z['quick_attack_probability'])

    def test_greedy_priority_tie_break_and_terminal_rejection(self):
        f={'pair':1,'seat':0,'views':self.case['initial']}
        r=decide(self.parameters,self.parameters,[f],101,'damage-greedy')[0]
        self.assertEqual(r['actions'][1],1)
        f['views']=self.case['turns'][-1]['views']
        with self.assertRaises(ValueError):decide(self.parameters,self.parameters,[f],101,'random')

    def test_smoke_and_development_seed_domains_are_disjoint(self):
        p=json.loads((ROOT/'contracts/pilot-v1/plan.json').read_text())
        self.assertNotEqual(mechanics_seed(p,101,'random',0,'smoke'),mechanics_seed(p,101,'random',0,'development'))
        self.assertEqual(roster(p,True),([101],[0,2],4))
        self.assertEqual(roster(p,False),([101,211,307],[0,256,2048],1000))


if __name__=='__main__':unittest.main()
