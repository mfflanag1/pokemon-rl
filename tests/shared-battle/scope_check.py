import copy,unittest
from unittest.mock import patch
from pokemon_shared_battle.common import *
from pokemon_full_learning.statistics import paired_bounds
from pokemon_full_train.campaign import score_interval

def games(reward,terminal=True):
 result=[]
 for i in range(128):
  for swap in [False,True]:
   side='p2' if swap else 'p1'
   result.append({'episode':i,'swap':swap,'learner_side':side,'opponent':'tactical',
    'provenance':{'teams':[{'team_key':str(i),'partition':'development'}]},
    'decisions':1,'trace':[],'value_mean':.5,'terminal_value_mse':.25,
    'outcome':{'kind':'terminal' if terminal else 'administrative_cutoff','rewards':{side:reward,('p1' if swap else 'p2'):1-reward}}})
 return result

class Scope(unittest.TestCase):
 def test_fixed_plan_budget(self):
  p=read(PLAN);validate_plan(p);self.assertEqual(3*3*3*128*2,p['expected_games'])
  for k,v in [('arms',['shared']),('optimizer_updates',1),('candidate_step',2048),('evaluation_seed',1),('expected_games',1),('prime_used',True)]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):validate_plan(q)
 def test_all_nine_sealed_checkpoints(self):
  p=read(PLAN)
  for seed in p['seeds']:
   for arm in p['arms']:
    params,opt,identity=load_policy(p,seed,arm)
    self.assertEqual(int(opt[2]),1024 if arm=='reference' else 8192)
    self.assertEqual(identity['parameter_hash'],parameter_hash(params))
 def test_wrong_checkpoint_identity_rejected(self):
  p=read(PLAN)
  with self.assertRaises(ValueError):load_policy(p,7103,'best')
  with self.assertRaises(ValueError):load_policy(p,1,'flat')
  original=load_strict_checkpoint
  def changed(*a):
   p,o,m=original(*a);return p,o,dict(m,seed=1)
  with patch('pokemon_shared_battle.common.load_strict_checkpoint',changed):
   with self.assertRaises(ValueError):load_policy(p,7103,'reference')
 def test_same_schedule_all_policy_arms(self):
  p=read(PLAN);jobs=evaluation_specs(p,7103,'tactical',0,4)
  self.assertEqual(len(jobs),8)
  for i in range(4):
   self.assertFalse(jobs[2*i]['swap']);self.assertTrue(jobs[2*i+1]['swap'])
   self.assertEqual(jobs[2*i]['episode'],i);self.assertEqual(jobs[2*i+1]['episode'],i)
  self.assertEqual(jobs,evaluation_specs(p,7103,'tactical',0,4))
 def test_fresh_role_guard(self):
  g=games(1)
  self.assertEqual(len(role_check(g,{'train'},{'past'})),128)
  with self.assertRaises(ValueError):role_check(g,{'0'},set())
  with self.assertRaises(ValueError):role_check(g,set(),{'127'})
 def test_pair_integrity(self):
  a=games(0);b=games(1);lo,hi=paired_bounds(a,b)
  self.assertTrue((lo==1).all() and (hi==1).all())
  for key,value in [('episode',99),('learner_side','p2'),('swap',True),('provenance',{})]:
   q=copy.deepcopy(b);q[0][key]=value
   with self.assertRaises(ValueError):paired_bounds(a,q)
 def test_cutoffs_not_draws(self):
  self.assertEqual(score_interval(games(0,False)[0]),(0.,1.))
  lo,hi=paired_bounds(games(0,False),games(1));self.assertTrue((lo==0).all() and (hi==1).all())
 def test_all_contrasts_and_no_promotion(self):
  p=read(PLAN);panels={a:{o:games(1 if a=='shared' else 0) for o in p['opponents']} for a in p['arms']}
  for panel in panels.values():
   for opp,gs in panel.items():
    for g in gs:g['opponent']=opp
  r=analysis(panels,{a:{'passed':False} for a in p['arms']},p)
  self.assertTrue(r['battle_benefit']);self.assertEqual(len(r['comparisons']),3)
  self.assertEqual(sum(len(v) for v in r['comparisons'].values()),9);self.assertFalse(r['promotion_authorized'])
if __name__=='__main__':unittest.main()
