import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from pokemon_switch_engine import learner
from pokemon_switch_screen.campaign import Reference,evaluate_cell,audit_cell,verify_record,cell_name
from pokemon_switch_screen.inference import decide,prefix,greedy
from pokemon_switch_screen.provenance import spec,roster


class Campaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.out=Path(cls.temp.name);cls.p=learner.start(1009).parameters
        w=Reference(cls.out/'node.log')
        try:evaluate_cell(cls.out,w,cls.p,cls.p,1009,0,'random',4,'smoke');w.finish()
        except BaseException:w.abort();raise
        with gzip.open(cls.out/cell_name(1009,0,'random'),'rt') as f:cls.records=[json.loads(l) for l in f]
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_real_reference_tape_replay(self):
        scores,d=audit_cell(self.out,self.p,self.p,1009,0,'random',4,'smoke')
        self.assertEqual(scores.shape,(4,2));self.assertTrue(d['all_actions_and_values_replayed'])
        self.assertGreater(d['candidate_decisions'],0)

    def test_corrupted_outcome_and_action_rejected(self):
        r=self.records[0]
        for kind in ('winner','actions','hash','groups','seed'):
            bad=copy.deepcopy(r)
            if kind=='winner':bad['winner']='p2' if r['winner']=='p1' else 'p1'
            if kind=='actions':bad['decisions'][0]['actions']=[-1,-1]
            if kind=='hash':bad['decisions'][0]['view_hashes'][0]='bad'
            if kind=='groups':bad['groups']+=1
            if kind=='seed':bad['spec']['seed']='sodium,'+'0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):verify_record(bad,1009,0,'random',0,'smoke')

    def test_actor_row_independence(self):
        frames=[]
        for r in self.records[:2]:
            _,fs=verify_record(r,1009,0,'random',2*r['pair']+r['seat'],'smoke');frames.append(fs[0])
        a=decide(self.p,self.p,[frames[0]],1009,'random','smoke')[0]
        changed=copy.deepcopy(frames[0]);changed['views'][1]=frames[1]['views'][1]
        b=decide(self.p,self.p,[changed],1009,'random','smoke')[0]
        self.assertEqual((a['actions'][0],a['candidate_value'],a['candidate_logp']),
                         (b['actions'][0],b['candidate_value'],b['candidate_logp']))

    def test_full_grid_seat_swap_and_disjoint_namespaces(self):
        grid=set()
        for pair in range(576):
            a=spec(1009,'random',pair,0,'development');b=spec(1009,'random',pair,1,'development')
            self.assertEqual(a['seed'],b['seed']);self.assertEqual(a['teams'],b['teams'][::-1]);grid.add(tuple(a['teams']))
        self.assertEqual(len(grid),576)
        self.assertNotEqual(spec(1009,'random',0,0,'smoke')['seed'],spec(1009,'random',0,0,'development')['seed'])
        self.assertEqual(roster(False),([1009,2027,3037],[0,2048,8192],576))

    def test_greedy_forced_switch_and_public_type(self):
        from pokemon_switch_engine.scalar import initial,view
        v=view(initial([0,8]),0);self.assertEqual(greedy(v),0)
        v['mask']=[False,False,False,True];self.assertEqual(greedy(v),3)
        v['mask']=[False]*4;self.assertEqual(greedy(v),-1)

if __name__=='__main__':unittest.main()
