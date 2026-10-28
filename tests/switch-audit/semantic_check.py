import copy,json,tempfile,unittest,subprocess,hashlib
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from pokemon_full_train.storage import ROOT,save_checkpoint
from pokemon_full.features import encode
from pokemon_full.policy import forward as old_forward
from pokemon_fit.learning import checked_step
from pokemon_balanced.common import initialize
from pokemon_switch_audit.semantic import extras,EXTRA,current_types
from pokemon_switch_audit.learner import extend,forward,checked,load_checkpoint
from pokemon_switch_audit.canary import prepare,read,PLAN,batch,analyze
class SemanticTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=read(PLAN);cls.data,cls.oracle,_=prepare(cls.plan);cls.catalog=read(ROOT/cls.plan['catalog']);cls.player=cls.oracle[0]['player']
 def state(self):
  p,_=initialize(7103);p=extend(p);o=({k:jnp.zeros_like(v) for k,v in p.items()},{k:jnp.zeros_like(v) for k,v in p.items()},jnp.asarray(0,jnp.int32));return p,o
 def test_semantic_types_matchups_and_shape(self):
  p=copy.deepcopy(self.player);p['own'][0]['species']='Charizard';p['own'][0]['tera']=None;p['own'][0]['types']=None
  p['own'][0]['moves']=['flamethrower'];p['phase']='replace';p['active_moves']=[]
  foe=next(m for m in p['opponent'] if m and m['active']);foe['species']='Venusaur';foe['types']=None;foe['tera']=None
  x=extras(p,self.catalog);self.assertEqual(x.shape,(EXTRA,));self.assertEqual(x.dtype,np.float32)
  self.assertEqual(x[1],1);self.assertEqual(x[9],1) # Fire/Flying
  self.assertEqual(x[19+12],1.) # Rock damage4x normalized by4
  self.assertEqual(x[38+24],.5) # Flamethrower2x normalized by4
 def test_catalog_regeneration_and_edge_types(self):
  raw=subprocess.check_output(['node','packages/switch-audit/catalog.mjs'],cwd=ROOT)
  self.assertEqual(hashlib.sha256(raw).hexdigest(),self.plan['catalog_sha256'])
  self.assertEqual(self.catalog['species']['florgesorange']['types'],['Fairy'])
  self.assertTrue(all(v==1 for v in self.catalog['effectiveness']['Stellar'].values()))
  self.assertTrue(all(v['???']==1 for v in self.catalog['effectiveness'].values()))
  p=copy.deepcopy(self.player);next(m for m in p['opponent'] if m and m['active'])['types']=['???']
  self.assertTrue(np.isfinite(extras(p,self.catalog)).all())
 def test_public_boundary_and_irrelevant_identity(self):
  p=copy.deepcopy(self.player);a=extras(p,self.catalog);p['request_id']+=100;p['own'][0]['ident']='Changed';np.testing.assert_array_equal(a,extras(p,self.catalog))
  bad=copy.deepcopy(p);next(m for m in bad['opponent'] if m)['stats']={'atk':999}
  with self.assertRaises(ValueError):extras(bad,self.catalog)
  with self.assertRaises(ValueError):extras({**p,'teacher_action':9},self.catalog)
 def test_slot_permutation_equivariance(self):
  p=copy.deepcopy(self.player);a=extras(p,self.catalog);p['own'][1],p['own'][2]=p['own'][2],p['own'][1];b=extras(p,self.catalog)
  np.testing.assert_array_equal(a[138:276],b[276:414]);np.testing.assert_array_equal(a[276:414],b[138:276]);np.testing.assert_array_equal(a[-19:],b[-19:])
 def test_zero_initialized_function_identity(self):
  p,_=initialize(7103);q=extend(p);d,e=self.data['probe'];x=jnp.asarray(d['features'][:16]);z=jnp.asarray(e[:16])
  for a,b in zip(old_forward(p,x),forward(q,x,z)):np.testing.assert_array_equal(a,b)
 def test_zero_arm_sealed_update_parity(self):
  p,o=initialize(7103);q,qo=self.state();b,_=batch(self.data['train'],self.data['old'],7103,0,'zero')
  a,ao,am=checked_step(p,o,{k:jnp.asarray(b[k]) for k in ['features','mask','targets']});q,qo,qm=checked(q,qo,b)
  for k in a:np.testing.assert_allclose(a[k],q[k],rtol=0,atol=1e-7)
  self.assertTrue(np.all(np.asarray(q['ew'])==0));self.assertAlmostEqual(am['pre_update_nll'],qm['pre_update_nll'],places=6)
 def test_same_rows_and_nonvacuous_semantic_gradient(self):
  p,o=self.state();a,sa=batch(self.data['train'],self.data['old'],7103,0,'zero');b,sb=batch(self.data['train'],self.data['old'],7103,0,'semantic')
  for k in ['features','mask','targets']:np.testing.assert_array_equal(a[k],b[k])
  for k in ['teacher_indices','old_indices']:self.assertEqual(sa[k],sb[k])
  q,_,_=checked(p,o,b);self.assertGreater(np.linalg.norm(np.asarray(q['ew'])),0)
  for k in ['vw','vb']:np.testing.assert_array_equal(p[k],q[k])
 def test_checkpoint_original_next_update(self):
  p,o=self.state();b,_=batch(self.data['train'],self.data['old'],7103,0,'semantic');p,o,_=checked(p,o,b)
  with tempfile.TemporaryDirectory() as tmp:
   save_checkpoint(tmp,p,o,{'input_hash':'test','seed':7103,'cohorts':1,'cursors':{'bc':1}});q,qo,_=load_checkpoint(tmp,'test')
   b,_=batch(self.data['train'],self.data['old'],7103,1,'semantic');p,o,m=checked(p,o,b);q,qo,n=checked(q,qo,b)
   self.assertEqual(m,n)
   for a,b in [(p,q),(o[0],qo[0]),(o[1],qo[1])]:
    for k in a:np.testing.assert_array_equal(a[k],b[k])
   pointer=json.loads((Path(tmp)/'latest.json').read_text());path=Path(tmp)/pointer['file']
   with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files if k!='p__ew'}
   with open(path,'wb') as f:np.savez_compressed(f,**arrays)
   pointer['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(Path(tmp)/'latest.json').write_text(json.dumps(pointer))
   with self.assertRaisesRegex(ValueError,'parameter inventory'):load_checkpoint(tmp,'test')
 def test_registered_gate_and_invalid_extra(self):
  fam=['normal_move','tera_move','forced_switch','voluntary_switch'];arms={a:{'probe':{'families':{f:{'game_accuracy':.4 if f!='voluntary_switch' or a=='semantic' else .1} for f in fam}},'retention':{'passed':True}} for a in ['zero','semantic']}
  self.assertTrue(analyze(arms,self.plan['coverage_thresholds'])['passed']);arms['semantic']['probe']['families']['normal_move']['game_accuracy']=.2;self.assertFalse(analyze(arms,self.plan['coverage_thresholds'])['passed'])
  p,o=self.state();b,_=batch(self.data['train'],self.data['old'],7103,0,'semantic');b['extra']=b['extra'][:,:-1]
  with self.assertRaises(ValueError):checked(p,o,b)
if __name__=='__main__':unittest.main()
