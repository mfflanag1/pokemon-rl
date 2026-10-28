import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_learning_curve.common import PLAN,read,validate_plan,require_baseline_meta,probe_roles
from pokemon_diversity.analysis import gate

class Core(unittest.TestCase):
    def test_fixed_plan_and_budget(self):
        p=read(PLAN);validate_plan(p)
        self.assertEqual((p['final_step']-p['start_step'])*3,p['expected_new_updates'])
        self.assertEqual(p['expected_new_updates']*128,p['expected_new_presentations'])
        for key,value in [('start_step',0),('final_step',16384),('endpoints',[2048,8192]),('new_training_games',1),('probe_games',257),('checkpoint_steps_per_seed',[2048,4096,8192]),('prime_used',True)]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plan(q)
    def test_baseline_identity_rejects_reset_or_wrong_endpoint(self):
        m={'seed':7103,'cohorts':2048,'cursors':{'bc':2048,'arm':'expanded'},'optimizer_step':2048};o=({}, {}, np.int32(2048))
        require_baseline_meta(m,o,7103)
        with self.assertRaises(ValueError):require_baseline_meta(m,({}, {}, np.int32(0)),7103)
        for key,value in [('seed',8209),('cohorts',8192),('cursors',{'bc':2048,'arm':'original'}),('optimizer_step',0)]:
            q=copy.deepcopy(m);q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):require_baseline_meta(q,o,7103)
    def test_fresh_team_roles(self):
        a={'train'};b={'f15','f16','f18'};c={'fresh'}
        self.assertEqual(probe_roles(a,b,c)['disallowed_overlap'],0)
        for s in a|b:
            with self.assertRaises(ValueError):probe_roles(a,b,{s})
        self.assertEqual(a,{'train'});self.assertEqual(b,{'f15','f16','f18'})
    def test_primary_gate_does_not_select_intermediate(self):
        p=read(PLAN);families=['normal_move','tera_move','voluntary_switch','forced_switch']
        def endpoint(switch):return {'probe':{'families':{f:{'game_accuracy':switch if f=='voluntary_switch' else .5} for f in families}},'retention':{'passed':True}}
        points={'2048':endpoint(.1),'4096':endpoint(.9),'8192':endpoint(.11)}
        result=gate({'original':points['2048'],'expanded':points['8192']},p)
        self.assertFalse(result['passed'])
        points['8192']=endpoint(.4);self.assertTrue(gate({'original':points['2048'],'expanded':points['8192']},p)['passed'])
if __name__=='__main__':unittest.main()
