import copy,hashlib,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_diversity.collection import combine,validate_metadata
from pokemon_diversity.analysis import paired_accuracy,gate,team_admission
from pokemon_diversity.common import read,PLAN,validate_plan
from pokemon_balanced.learning import support

def dataset(games):
    n=games*4;x=np.zeros((n,928),np.float32);target=np.tile(np.array([0,4,8,9],np.int32),games)
    x[:,866]=1;x[3::4,866]=0;x[3::4,867]=1
    mask=np.ones((n,14),bool);ids=np.repeat(np.arange(games,dtype=np.int32),4)
    d=dict(features=x,mask=mask,targets=target,game_index=ids)
    rows=[dict(game=int(ids[i]),phase='replace' if i%4==3 else 'move',target=int(target[i]),own_species='A',foe_species='B',destination_species='C' if target[i]>=8 else '',feature_hash=hashlib.sha256(x[i].tobytes()+mask[i].tobytes()).hexdigest()) for i in range(n)]
    return d,rows

class Core(unittest.TestCase):
    def test_union_offsets_and_inputs_unchanged(self):
        a,ar=dataset(512);b,br=dataset(1536)
        c,cr=combine(a,b,ar,br)
        np.testing.assert_array_equal(np.unique(c['game_index']),np.arange(2048))
        self.assertEqual(cr[2048]['game'],512);self.assertEqual(br[0]['game'],0)
        self.assertEqual(int(b['game_index'][0]),0)
        np.testing.assert_array_equal(c['targets'][:2048],a['targets'])
        b['game_index'][0]=1536
        with self.assertRaises(ValueError):combine(a,b,ar,br)
    def test_metadata_linkage_rejects_corruption(self):
        d,r=dataset(2);validate_metadata(d,r)
        for key,value in [('target',1),('game',99),('feature_hash','0'*64),('phase','replace')]:
            q=copy.deepcopy(r);q[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_metadata(d,q)
        with self.assertRaises(ValueError):validate_metadata(d,r[:-1])
    def test_team_roles(self):
        good=team_admission({'a'},{'b'},{'c'},{'d'})
        self.assertEqual(good['disallowed_overlap'],0)
        for args in [({'a'},{'a'},{'c'},{'d'}),({'a'},{'b'},{'a'},{'d'}),({'a'},{'b'},{'d'},{'d'}),({'a'},{'d'},{'c'},{'d'})]:
            with self.assertRaises(ValueError):team_admission(*args)
    def test_plan_scope(self):
        plan=read(PLAN);validate_plan(plan)
        for key,value in [('probe_games',257),('bc_steps',2049),('public_server_used',True),('seeds',[7103])]:
            p=copy.deepcopy(plan);p[key]=value
            with self.assertRaises(ValueError):validate_plan(p)
    def test_paired_cluster_unit_and_bounds(self):
        d,_=dataset(3)
        # Duplicate all rows of game0: decisions get more weight, games do not.
        ids=np.concatenate([np.arange(12),np.arange(4),np.arange(4)])
        d={k:v[ids] for k,v in d.items()};n=len(ids)
        before=np.full((n,14),.01);after=before.copy()
        for i in range(n):
            t=d['targets'][i];wrong=(t+1)%14
            before[i,wrong]=.87;after[i,t if d['game_index'][i]==0 else wrong]=.87
        s={'seed':7,'resamples':100,'alpha':.05,'contrasts':12}
        r=paired_accuracy(before,after,d,s,7103)
        for v in r.values():
            self.assertEqual(v['games'],3);self.assertAlmostEqual(v['paired_game_accuracy_difference'],1/3)
            self.assertEqual(v['per_game_differences'],[1.,0.,0.]);self.assertEqual(v['hoeffding_interval'],[-1.,1.])
        z=paired_accuracy(before,before,d,s,7103)
        self.assertTrue(all(x['approximate_adjusted_bootstrap_interval']==[0.,0.] for x in z.values()))
    def test_probe_support_fail_closed(self):
        d,_=dataset(2)
        p={'coverage_thresholds':{key:0 for key in ['minimum_tera_rows','minimum_tera_games','minimum_voluntary_switch_rows','minimum_voluntary_switch_games']}}
        support(d,p)
        for key in p['coverage_thresholds']:
            q=copy.deepcopy(p);q['coverage_thresholds'][key]=3
            with self.subTest(key=key),self.assertRaises(ValueError):support(d,q)
    def test_descriptive_gate_all_conditions(self):
        p=read(PLAN);names=['normal_move','tera_move','voluntary_switch','forced_switch']
        base={n:{'game_accuracy':.2} for n in names};end=copy.deepcopy(base);end['voluntary_switch']['game_accuracy']=.4
        a={'original':{'probe':{'families':base}},'expanded':{'probe':{'families':end},'retention':{'passed':True}}}
        self.assertTrue(gate(a,p)['passed'])
        a['expanded']['probe']['families']['tera_move']['game_accuracy']=.1
        self.assertFalse(gate(a,p)['passed'])
        a['expanded']['probe']['families']['tera_move']['game_accuracy']=.2
        a['expanded']['retention']['passed']=False
        self.assertFalse(gate(a,p)['passed'])
if __name__=='__main__':unittest.main()
