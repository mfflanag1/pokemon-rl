import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_shared.common import read,PLAN,validate_plan,gate,probe_roles
from pokemon_shared.learning import initial,predictions
from pokemon_shared.symmetry import remap
from pokemon_full.features import encode
from pokemon_full_train.storage import ROOT

class Scope(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=read(PLAN)
  cls.oracle=read(ROOT/'artifacts/runs/f18-data-coverage-v1/inputs/tactical-oracle.json')
  with np.load(ROOT/'artifacts/runs/f18-data-coverage-v1/inputs/old-supervised.npz',allow_pickle=False) as z:cls.old={k:z[k] for k in z.files}
 def test_fixed_plan_and_arithmetic(self):
  validate_plan(self.plan);self.assertEqual(3*2*8192,self.plan['expected_updates']);self.assertEqual(49152*128,self.plan['expected_presentations']);self.assertEqual(self.plan['teacher_presentations']+self.plan['rehearsal_presentations'],6291456)
  for k,v in [('bc_steps',8193),('arms',['shared']),('probe_seed',1),('public_server_used',True),('endpoints',[0,2048,4096])]:
   p=copy.deepcopy(self.plan);p[k]=v
   with self.assertRaises(ValueError):validate_plan(p)
 def test_remap_all_actions_round_trip(self):
  x=self.old['features'][:14].copy();m=np.ones((14,14),bool);t=np.arange(14,dtype=np.int32);o=[2,0,4,1,5,3];f=[5,1,3,0,4,2];mvs=[3,0,2,1]
  y,n,u,a=remap(x,m,t,o,f,mvs);self.assertFalse(np.array_equal(x,y));self.assertTrue(n[np.arange(14),u].all());np.testing.assert_array_equal(a[u],t)
  z,q,v,_=remap(y,n,u,np.argsort(o),np.argsort(f),np.argsort(mvs));np.testing.assert_array_equal(z,x);np.testing.assert_array_equal(q,m);np.testing.assert_array_equal(v,t)
  with self.assertRaises(ValueError):remap(x,m,t,[0]*6,f,mvs)
 def test_shared_legal_distribution_remaps(self):
  x,m,t=[self.old[k] for k in ['features','mask','targets']];y,n,u,a=remap(x,m,t,[5,0,3,1,2,4],[2,4,0,5,1,3],[2,0,3,1]);p,_=initial(7103,'shared')
  pr=predictions(p,x,m,'shared');qr=predictions(p,y,n,'shared');self.assertGreater(float(np.ptp(pr)),.01);np.testing.assert_allclose(qr,pr[:,a],atol=1e-5,rtol=0);self.assertTrue(n[np.arange(len(u)),u].all())
 def test_flat_is_noninvariant_positive_control(self):
  x,m,t=[self.old[k] for k in ['features','mask','targets']];y,n,u,a=remap(x,m,t,[5,0,3,1,2,4],[2,4,0,5,1,3],[2,0,3,1]);p,_=initial(7103,'flat')
  self.assertGreater(float(np.max(abs(predictions(p,y,n,'flat')-predictions(p,x,m,'flat')[:,a]))),1e-5)
 def test_private_fields_rejected_by_actor_projection(self):
  p=copy.deepcopy(self.oracle['records'][0]['player']);encode(p)
  for q in [dict(p,private_state={'opponent_item':'secret'}),dict(p,tactical_action=9)]:
   with self.assertRaises(ValueError):encode(q)
  q=copy.deepcopy(p);foe=next(v for v in q['opponent'] if v);foe['hidden_stats']={'spe':999}
  with self.assertRaises(ValueError):encode(q)
 def test_identity_metadata_not_used_as_features(self):
  p=copy.deepcopy(self.oracle['records'][0]['player']);x,m=encode(p);p['request_id']='different';p['side']='p2' if p['side']=='p1' else 'p1'
  for mon in p['own']+p['opponent']:
   if mon:mon['ident']='unrelated-private-looking-id'
  y,n=encode(p);np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(m,n)
 def test_final_shared_mastery_scope_and_no_endpoint_selection(self):
  def endpoint(acc,mastered=True):return {'probe':{'families':{k:{'game_accuracy':acc} for k in ['normal_move','tera_move','voluntary_switch','forced_switch']}},'mastery':{'passed':mastered}}
  arms={'flat':{'endpoints':{'8192':endpoint(.1,False)}},'shared':{'endpoints':{'2048':endpoint(1),'8192':endpoint(.4)}}}
  self.assertTrue(gate(arms,self.plan)['passed']);arms['shared']['endpoints']['8192']=endpoint(.1);self.assertFalse(gate(arms,self.plan)['passed']);arms['shared']['endpoints']['8192']=endpoint(.4,False);self.assertFalse(gate(arms,self.plan)['passed'])
 def test_fresh_probe_disjoint_including_f19(self):
  train={'train'};past={'f15','f16','f18','f19'}
  self.assertEqual(probe_roles(train,past,{'new'})['disallowed_overlap'],0)
  for key in train|past:
   with self.assertRaises(ValueError):probe_roles(train,past,{key})
if __name__=='__main__':unittest.main()
