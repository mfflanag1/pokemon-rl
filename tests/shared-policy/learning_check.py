import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_shared.learning import initial,step,selected_batch,check_state,shared_objective,value_keys
from pokemon_shared.common import save,load,same_state
from pokemon_full_train.storage import ROOT,atomic_json,digest
from pokemon_diversity.sampling import PreparedSampler
from pokemon_diversity.learning import bc_step
from pokemon_full.learner import parameter_hash

class Learning(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  base=ROOT/'artifacts/runs/f18-data-coverage-v1/inputs'
  with np.load(base/'expanded-teacher.npz',allow_pickle=False) as z:cls.data={k:z[k] for k in z.files}
  with np.load(base/'old-supervised.npz',allow_pickle=False) as z:cls.old={k:z[k] for k in z.files}
  cls.sampler=PreparedSampler(cls.data);cls.batch,cls.selection=selected_batch(cls.sampler,cls.old,7103,0)
 def test_flat_exact_sealed_updates(self):
  p,o=initial(7103,'flat')
  for i in range(3):
   batch,sel=selected_batch(self.sampler,self.old,7103,i);q,v,item=bc_step(p,o,self.sampler,self.old,7103,i)
   p,o,m=step(p,o,batch,'flat');same_state(p,o,q,v);self.assertEqual(m,item['metrics']);self.assertEqual(sel,{k:item[k] for k in sel})
 def test_shared_step_improves_same_batch_likelihood(self):
  p,o=initial(8209,'shared');b={k:jnp.asarray(v) for k,v in self.batch.items()};a=float(shared_objective(p,b))
  q,v,_=step(p,o,self.batch,'shared');z=float(shared_objective(q,b));self.assertLess(z,a);self.assertNotEqual(parameter_hash(p),parameter_hash(q));self.assertEqual(int(v[2]),1)
 def test_shared_gradient_directional_difference(self):
  p,_=initial(9311,'shared');b={k:jnp.asarray(v) for k,v in self.batch.items()};g=jax.grad(shared_objective)(p,b);norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in g.values())))
  self.assertGreater(norm,1e-4);d={k:v/norm for k,v in g.items()};eps=.01
  a={k:p[k]+eps*d[k] for k in p};c={k:p[k]-eps*d[k] for k in p}
  finite=(float(shared_objective(a,b))-float(shared_objective(c,b)))/(2*eps)
  self.assertAlmostEqual(finite,norm,delta=max(.002,.02*norm))
 def test_value_weights_unchanged_both_arms(self):
  for arm in ['flat','shared']:
   p,o=initial(17,arm);before={k:np.asarray(p[k]).copy() for k in value_keys(arm)};p,o,_=step(p,o,self.batch,arm)
   for k,v in before.items():np.testing.assert_array_equal(p[k],v);np.testing.assert_array_equal(o[0][k],v);np.testing.assert_array_equal(o[1][k],v)
 def test_original_next_update_both_architectures(self):
  for arm in ['flat','shared']:
   p,o=initial(17,arm);p,o,_=step(p,o,self.batch,arm)
   with tempfile.TemporaryDirectory() as t:
    save(Path(t),p,o,'test',17,arm,1);q,v=load(Path(t),'test',17,arm,1);same_state(p,o,q,v)
    a,ao,am=step(p,o,self.batch,arm);b,bo,bm=step(q,v,self.batch,arm);same_state(a,ao,b,bo);self.assertEqual(am,bm)
 def test_checkpoint_identity_inventory_and_finite_rejections(self):
  p,o=initial(17,'shared')
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);save(d,p,o,'test',17,'shared',0)
   for args in [('bad',17,'shared',0),('test',18,'shared',0),('test',17,'flat',0),('test',17,'shared',1)]:
    with self.assertRaises(ValueError):load(d,*args)
   pointer=json.loads((d/'latest.json').read_text());file=d/pointer['file']
   with np.load(file,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
   for alteration in ['extra','missing','nan']:
    a=dict(arrays)
    if alteration=='extra':a['p__unregistered']=np.zeros(1,np.float32)
    elif alteration=='missing':del a['m__mon_w']
    else:a['p__mon_w']=a['p__mon_w'].copy();a['p__mon_w'][0,0]=np.nan
    with file.open('wb') as f:np.savez_compressed(f,**a)
    atomic_json(d/'latest.json',{'file':file.name,'sha256':digest(file)})
    with self.assertRaises(ValueError):load(d,'test',17,'shared',0)
 def test_invalid_states_rejected(self):
  p,o=initial(17,'shared')
  for q,v in [(dict(p,extra=jnp.zeros(1)),o),(dict(p,mon_w=np.zeros((64,32),np.float64)),o),(p,(o[0],o[1],np.int64(0))),(p,(o[0],{**o[1],'mon_w':-jnp.ones((64,32))},o[2]))]:
   with self.assertRaises(ValueError):check_state(q,v,'shared')
 def test_invalid_training_batch_rejected(self):
  p,o=initial(17,'shared')
  for b in [dict(self.batch,private=np.zeros(128)),dict(self.batch,targets=np.full(128,14,np.int32)),dict(self.batch,mask=np.zeros((128,14),bool))]:
   with self.assertRaises(ValueError):step(p,o,b,'shared')
if __name__=='__main__':unittest.main()
