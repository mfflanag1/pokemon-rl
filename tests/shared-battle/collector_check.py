import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_shared.learning import initial
from pokemon_full_train.storage import canonical
from pokemon_transfer_v2.collector import Pool as Original
from pokemon_shared_battle.collector import Pool
from pokemon_shared_battle.common import check_collection

def specs(opponent='tactical',max_turns=120):
 return [dict(seed=1025262199,episode=i,max_turns=max_turns,split='development',swap=swap,opponent=opponent,
    learner_side='p2' if swap else 'p1',actor_seed=1025262198) for i in range(2) for swap in [False,True]]

class Collector(unittest.TestCase):
 def test_flat_complete_games_match_sealed_collector(self):
  p,_=initial(7103,'flat');jobs=specs()
  with tempfile.TemporaryDirectory() as directory:
   with Original(Path(directory)/'old',2) as pool:a,_=pool.collect(p,jobs)
   with Pool(Path(directory)/'new',2,'flat') as pool:b,_=pool.collect(p,jobs)
  check_collection(b,0,2,'tactical')
  for (ar,ag),(br,bg) in zip(a,b):
   self.assertFalse(ar or br);other=dict(bg);other.pop('actor_trace');self.assertEqual(canonical(ag),canonical(other))
  self.assertTrue(any(g['outcome']['kind']=='terminal' for _,g in b))
 def test_shared_worker_order_and_cutoffs(self):
  p,_=initial(8209,'shared');jobs=specs('base_power',2)
  with tempfile.TemporaryDirectory() as directory:
   with Pool(Path(directory)/'one',1,'shared') as pool:a,_=pool.collect(p,jobs)
   with Pool(Path(directory)/'two',2,'shared') as pool:b,_=pool.collect(p,jobs)
  check_collection(a,0,2,'base_power');check_collection(b,0,2,'base_power')
  for (_,ag),(_,bg) in zip(a,b):
   self.assertEqual(ag['outcome'],bg['outcome']);self.assertEqual(ag['trace'],bg['trace'])
   for x,y in zip(ag['actor_trace'],bg['actor_trace']):
    self.assertEqual(x['action'],y['action']);self.assertEqual(x['feature_sha256'],y['feature_sha256'])
    self.assertAlmostEqual(x['log_probability'],y['log_probability'],places=5)
  self.assertTrue(all(g['outcome']['kind']=='administrative_cutoff' for _,g in b))
 def test_no_training_or_invalid_record(self):
  p,_=initial(9311,'shared')
  with tempfile.TemporaryDirectory() as directory:
   with Pool(Path(directory)/'pool',1,'shared') as pool:
    with self.assertRaises(ValueError):pool.collect(p,[dict(seed=1,episode=0,max_turns=2,split='train')])
    records,_=pool.collect(p,specs('random',2))
  check_collection(records,0,2,'random')
  q=copy.deepcopy(records);q[0][1]['actor_trace'][0]['action']=-1
  with self.assertRaises(ValueError):check_collection(q,0,2,'random')
 def test_known_trapped_pair_matches_with_feedback(self):
  import json
  from pokemon_transfer_v2.common import initialize,read
  from pokemon_full_train.storage import ROOT
  from pokemon_full_learning.campaign import evaluation_specs
  plan=read(ROOT/'contracts/full-transfer-v2/plan.json');p,_=initialize(7103,plan)
  jobs=evaluation_specs(plan,7103,'random',4,4)
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)
   with Original(path/'old',8) as pool:a,_=pool.collect(p,jobs)
   with Pool(path/'new',8,'flat') as pool:b,_=pool.collect(p,jobs)
   records=[json.loads(line) for f in (path/'new').glob('feedback-*.jsonl') for line in f.read_text().splitlines()]
  check_collection(b,4,4,'random');self.assertEqual(sum(g['unavailable_choices'] for _,g in b),2)
  self.assertEqual(len(records),2)
  for (_,ag),(_,bg) in zip(a,b):
   other=dict(bg);other.pop('actor_trace');self.assertEqual(canonical(ag),canonical(other))
  for row in records:
   side=row['submitted']['side'];self.assertIn('trapped',row['submitted']['feedback'][0])
   self.assertTrue(row['before']['players'][side]['mask'][10]);self.assertFalse(row['after']['players'][side]['mask'][10])
if __name__=='__main__':unittest.main()
