import copy
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize,forward
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_fit.data import prepare,load_data
from pokemon_fit.learning import validate_batch,model_batch,objective,checked_step
from pokemon_fit.audit import equal_tree,checkpoint


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/'contracts/decision-fit-v1/plan.json').read_text())
        cls.records=json.loads((ROOT/cls.plan['dataset']).read_text())['records']
        cls.splits,cls.inventory=load_data(cls.plan)

    def test_complete_dataset_and_stable_order(self):
        other,inventory=prepare(list(reversed(self.records)),self.plan)
        self.assertEqual(inventory,{k:v for k,v in self.inventory.items() if k not in ['dataset_sha256','predecessor_receipt_sha256']})
        for key,split in other.items():
            self.assertEqual(len(split['targets']),self.plan['splits'][key]['rows'])
            for field in split:np.testing.assert_array_equal(split[field],self.splits[key][field])
        self.assertEqual(len(set(map(tuple,self.inventory['fit_train']['coordinates'])) & set(map(tuple,self.inventory['probe_same']['coordinates']))),0)

    def test_oracle_corruptions_rejected(self):
        for kind in ['duplicate','seat','label','branch','nonfinite','tie','extra_player']:
            rows=copy.deepcopy(self.records);row=rows[0]
            if kind=='duplicate':rows[1]=copy.deepcopy(row)
            if kind=='seat':row['player']['side']='p2'
            if kind=='label':row['target']=(row['target']+1)%14
            if kind=='branch':row['scores'].pop(next(iter(row['scores'])))
            if kind=='nonfinite':row['scores'][str(row['target'])]=float('nan')
            if kind=='tie':row['scores']={k:1. for k in row['scores']}
            if kind=='extra_player':row['player']['oracle_target']=row['target']
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):prepare(rows,self.plan)

    def test_loss_receives_only_three_fields_and_fit_rows(self):
        batch=model_batch(self.splits['fit_train']);self.assertEqual(set(batch),{'features','mask','targets'})
        for field in batch:np.testing.assert_array_equal(batch[field],self.splits['fit_train'][field])
        for kind in ['metadata','label','illegal','mask','shape','nan']:
            b={k:np.asarray(v).copy() for k,v in batch.items()}
            if kind=='metadata':b['context']=np.zeros(128)
            if kind=='label':b['targets'][0]=-1
            if kind=='illegal':b['mask'][0,b['targets'][0]]=False
            if kind=='mask':b['mask'][0]=False
            if kind=='shape':b['features']=b['features'][:127]
            if kind=='nan':b['features'][0,0]=np.nan
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_batch(b)

    def test_gradient_direction_against_masked_softmax(self):
        p=initialize(7103);b=model_batch(self.splits['fit_train'])
        def reference(params):
            logits,_=forward(params,b['features'])
            logs=jax.nn.log_softmax(jnp.where(b['mask'],logits,-jnp.inf))
            return -logs[jnp.arange(128),b['targets']].mean()
        # At initial logits the unchanged dyadic law agrees with softmax within FP32 tolerance.
        self.assertAlmostEqual(float(objective(p,b)),float(reference(p)),places=5)
        actual=jax.grad(objective)(p,b);expected=jax.grad(reference)(p)
        for key in p:np.testing.assert_allclose(actual[key],expected[key],atol=2e-6,rtol=2e-5)
        np.testing.assert_array_equal(actual['vw'],np.zeros_like(actual['vw']));self.assertEqual(float(actual['vb']),0.)
        z=jax.tree.map(jnp.zeros_like,p);after,_,_=checked_step(p,(z,z,jnp.int32(0)),b)
        self.assertLess(float(objective(after,b)),float(objective(p,b)))

    def test_original_next_update_checkpoint_and_corruption(self):
        p=initialize(7103);z=jax.tree.map(jnp.zeros_like,p);opt=(z,z,jnp.int32(0));b=model_batch(self.splits['fit_train'])
        p,opt,_=checked_step(p,opt,b)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pointer=save_checkpoint(root,p,opt,{'seed':7103,'input_hash':'unit','cohorts':1})
            loaded,loaded_opt=checkpoint(root,p,opt,7103,1,'unit')
            a,ao,am=checked_step(p,opt,b);c,co,cm=checked_step(loaded,loaded_opt,b)
            equal_tree((a,ao),(c,co));self.assertEqual(am,cm)
            with self.assertRaises(ValueError):checkpoint(root,p,opt,7103,2,'unit')
            with self.assertRaises(ValueError):load_checkpoint(root,'different')
            with open(root/pointer['file'],'ab') as out:out.write(b'corruption')
            with self.assertRaises(ValueError):load_checkpoint(root,'unit')


if __name__=='__main__':unittest.main()
