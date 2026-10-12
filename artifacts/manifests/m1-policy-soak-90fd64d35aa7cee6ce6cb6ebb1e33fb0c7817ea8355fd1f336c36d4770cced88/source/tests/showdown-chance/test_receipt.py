"""Reject missing parity strata and promotion claims at the completion gate."""
import copy
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]

class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.gate=runpy.run_path(str(ROOT/'scripts/seal-showdown-chance.py'))

    def test_player_evidence_accounting(self):
        index={'shards':[{'first':100*i,'sha256':str(i),'count':100} for i in range(100)]}
        rows=[{'first':s['first'],'capture_sha256':s['sha256'],'battles':100,'turns':300,'events':500,
               'player_view_comparisons_including_padding':1800,'pending_player_comparisons':600,'disagreements':0}
              for s in index['shards']]
        player={k:sum(r[k] for r in rows) for k in self.gate['TOTAL_KEYS']}
        player.update(status='passed',fresh_distinct_seeds=10000,disjoint_from_prior_reference_seeds=True,
                      raw_chance_checks={'crit':50000,'roll':50000,'tie':30000})
        check=self.gate['validate_player'];check(player,rows,index)
        for field,value in [('fresh_distinct_seeds',9999),('disjoint_from_prior_reference_seeds',False),
                            ('disagreements',1),('pending_player_comparisons',0),('turns',30001)]:
            with self.assertRaises(ValueError):check(dict(player,**{field:value}),rows,index)
        with self.assertRaises(ValueError):check(player,rows[:-1],index)
        for field,value in [('first',100),('capture_sha256','bad'),('events',499),
                            ('disagreements',1),('player_view_comparisons_including_padding',1600)]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):check(player,bad,index)
        bad=copy.deepcopy(player);bad['raw_chance_checks']['roll']-=1
        with self.assertRaises(ValueError):check(bad,rows,index)

    def test_scope_and_test_summary_gate(self):
        good={'status':'passed','full_g2':'open','full_g3':'open','prime_used':False,
              'two_hour_qualification':False,'final_reservation_unchanged':True,
              'integration':{'status':'passed','two_hour_qualification':False,'traces':{'collect':1,'update':1,'cycle':1}}}
        check=self.gate['validate_scope'];check(good)
        for k,v in [('full_g2','passed'),('full_g3','passed'),('two_hour_qualification',True),
                    ('prime_used',True),('final_reservation_unchanged',False)]:
            with self.assertRaises(ValueError):check(dict(good,**{k:v}))
        bad=copy.deepcopy(good);bad['integration']['traces']['cycle']=2
        with self.assertRaises(ValueError):check(bad)
        for prefix in ('#','ℹ'):
            self.assertEqual(self.gate['node_pass_count'](f'{prefix} pass 65\n{prefix} fail 0\n'),65)
            with self.assertRaises(ValueError):self.gate['node_pass_count'](f'{prefix} pass 65\n{prefix} fail 1\n')

if __name__=='__main__':unittest.main()
