import copy,unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import select,distribution
from pokemon_full_train.storage import ROOT
from pokemon_shared.common import read
from pokemon_shared.learning import initial
from pokemon_shared.network import forward
from pokemon_throughput.collector import batched_actor as original
from pokemon_shared_battle.actor import batched_actor

class Actor(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.players=[r['player'] for r in read(ROOT/'artifacts/runs/f20-shared-policy-v1/inputs/tactical-oracle.json')['records'][:4]]
 def test_flat_unchanged(self):
  p,_=initial(7103,'flat');a=batched_actor(p,self.players,[1,2,3,4],[0,1,2,3],8,'flat');b=original(p,self.players,[1,2,3,4],[0,1,2,3],8)
  for x,y in zip(a,b):
   for u,v in zip(x,y):np.testing.assert_array_equal(u,v)
 def test_shared_direct_word_law(self):
  p,_=initial(7103,'shared');s=[3,7,11,19];c=[0,1,2,3]
  actual=batched_actor(p,self.players,s,c,4,'shared')
  x=np.stack([encode(a)[0] for a in self.players]);m=np.stack([encode(a)[1] for a in self.players])
  keys=[jax.random.fold_in(jax.random.key(seed),counter) for seed,counter in zip(s,c)]
  words=jnp.stack([jax.random.bits(k,(15,2),dtype=jnp.uint32) for k in keys])
  logits,values=forward(p,x);actions=select(logits,m,words);_,logs=distribution(logits,m)
  for i,row in enumerate(actual):
   self.assertEqual(row[2],int(actions[i]));self.assertAlmostEqual(row[3],float(logs[i,actions[i]]),places=6);self.assertEqual(row[4],float(values[i]))
 def test_wait_and_singleton(self):
  a=copy.deepcopy(self.players[0]);a['mask']=[False]*14;a['phase']='wait'
  b=copy.deepcopy(self.players[1]);b['mask']=[i==9 for i in range(14)]
  p,_=initial(8209,'shared');rows=batched_actor(p,[a,b],[1,2],[3,4],4,'shared')
  self.assertEqual(rows[0][2:4],(-1,0.));self.assertEqual(rows[1][2:4],(9,0.))
 def test_padding_and_reorder(self):
  p,_=initial(9311,'shared');a=batched_actor(p,self.players,[1,2,3,4],[8,4,6,2],8,'shared')
  b=batched_actor(p,list(reversed(self.players)),[4,3,2,1],[2,6,4,8],8,'shared')
  for x,y in zip(a,reversed(b)):
   for u,v in zip(x,y):np.testing.assert_array_equal(u,v)
 def test_hidden_and_advice_rejected(self):
  p,_=initial(7103,'shared')
  for key in ['private_state','tactical_action']:
   player=dict(self.players[0],**{key:9})
   with self.assertRaises(ValueError):batched_actor(p,[player],[1],[0],2,'shared')
  player=copy.deepcopy(self.players[0]);foe=next(m for m in player['opponent'] if m);foe['hidden_stats']={'spe':999}
  with self.assertRaises(ValueError):batched_actor(p,[player],[1],[0],2,'shared')
 def test_actor_no_mutation(self):
  p,_=initial(7103,'shared');before={k:np.array(v) for k,v in p.items()};players=copy.deepcopy(self.players)
  batched_actor(p,players,[1,2,3,4],[0]*4,8,'shared');self.assertEqual(players,self.players)
  for k in p:np.testing.assert_array_equal(p[k],before[k])
 def test_bad_batch_and_architecture(self):
  p,_=initial(7103,'shared')
  for players,seeds,counts,cap,arm in [([],[],[],8,'shared'),(self.players,[1],[],8,'shared'),(self.players,[1]*4,[0]*4,2,'shared'),(self.players,[1]*4,[0]*4,8,'other')]:
   with self.assertRaises(ValueError):batched_actor(p,players,seeds,counts,cap,arm)
 def test_common_random_words_with_identical_logits(self):
  fp,_=initial(7103,'flat');sp,_=initial(7103,'shared')
  fp=jax.tree.map(jnp.zeros_like,fp);sp=jax.tree.map(jnp.zeros_like,sp)
  players=self.players*32;seeds=list(range(128));counts=list(range(128,256))
  flat=batched_actor(fp,players,seeds,counts,128,'flat')
  shared=batched_actor(sp,players,seeds,counts,128,'shared')
  changed=batched_actor(sp,players,seeds,[v+1 for v in counts],128,'shared')
  for a,b in zip(flat,shared):self.assertEqual(a[2:],b[2:])
  self.assertGreater(sum(a[2]!=b[2] for a,b in zip(shared,changed)),10)
 def test_uint32_range_admitted_before_cast(self):
  for arm in ['flat','shared']:
   p,_=initial(7103,arm)
   for bad in [-1,2**32,True,1.5]:
    with self.assertRaises(ValueError):batched_actor(p,self.players,[bad]*4,[0]*4,4,arm)
    with self.assertRaises(ValueError):batched_actor(p,self.players,[1]*4,[bad]*4,4,arm)
if __name__=='__main__':unittest.main()
