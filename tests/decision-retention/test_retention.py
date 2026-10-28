import copy
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full_train.storage import ROOT,save_checkpoint
from pokemon_full_train.learning import optimize,derived_seed,prepare_batch
from pokemon_decision.collect import Pool
from pokemon_fit.audit import equal_tree
from pokemon_retention.common import initial_state,training_specs,eval_specs,pair,probe,summarize,input_inventory
from pokemon_retention.audit import check_checkpoint

PLAN=json.loads((ROOT/'contracts/decision-retention-v1/plan.json').read_text())

class Retention(unittest.TestCase):
    def test_initializers_reset_all_optimizer_state_and_keep_exact_weights(self):
        for seed in PLAN['seeds']:
            fresh,fo=initial_state(seed,'fresh',PLAN);warm,wo=initial_state(seed,'supervised',PLAN)
            equal_tree(fresh,initialize(seed));self.assertFalse(np.array_equal(fresh['w'],warm['w']))
            for p,opt in [(fresh,fo),(warm,wo)]:
                self.assertEqual(int(opt[2]),0)
                for state in opt[:2]:self.assertTrue(all(np.all(np.asarray(v)==0) for v in state.values()))
                self.assertEqual(float(p['vb']),0);self.assertTrue(np.all(np.asarray(p['vw'])==0))
        with self.assertRaises(ValueError):initial_state(7103,'chosen',PLAN)

    def test_fresh_coordinates_and_matching_arms(self):
        evaluation=eval_specs(PLAN);self.assertEqual(len(evaluation),512)
        self.assertNotIn(PLAN['evaluation_seed'],[8622026,8622027])
        for seed in PLAN['seeds']:
            specs=training_specs(seed,0,PLAN);self.assertEqual(len(specs),128)
            self.assertNotIn(specs[0]['run_seed'],[8622026,8622027,PLAN['evaluation_seed'],derived_seed('f6-training',seed)])
            for c in range(16):self.assertEqual(sum(s['context']==c for s in specs),8)
            self.assertTrue({s['episode'] for s in specs}.isdisjoint({s['episode'] for s in training_specs(seed,1,PLAN)}))

    def test_real_paired_collector_and_original_next_update(self):
        with tempfile.TemporaryDirectory() as tmp,Pool(Path(tmp)/'workers',2) as pool:
            p,opt=initial_state(7103,'fresh',PLAN);w,_=initial_state(7103,'supervised',PLAN)
            specs=training_specs(9913,0,PLAN)[:16]
            rows,a=pool.collect(p,specs,993);_,b=pool.collect(w,specs,993);pair(a,b)
            self.assertTrue(all(r['natural_outcome'] is None for r in a+b))
            self.assertTrue(all('target' not in row and 'context' not in row for row in rows))
            np.testing.assert_allclose(prepare_batch(rows)['returns'],[r['reward'] for r in a],atol=1e-7)
            altered=copy.deepcopy(b);altered[0]['player']['turn']+=1
            with self.assertRaises(ValueError):pair(a,altered)
            root=Path(tmp)/'checkpoint';save_checkpoint(root/'initial',p,opt,{'seed':7103,'arm':'fresh','cohorts':0,'input_hash':'test'})
            lp,lo=check_checkpoint(root,'initial',p,opt,7103,'fresh',0,'test')
            q,qo,qm=optimize(p,opt,rows,12);z,zo,zm=optimize(lp,lo,rows,12)
            equal_tree((q,qo),(z,zo));self.assertEqual(qm,zm);self.assertEqual(int(qo[2]),2)
            with self.assertRaises(ValueError):check_checkpoint(root,'initial',p,opt,7103,'supervised',0,'test')

    def test_probe_and_evaluation_labels_reject_mismatched_outcomes(self):
        oracle=json.loads((ROOT/'artifacts/reviews/f6-decision-canary/mechanics-probe.json').read_text())
        records=[{'spec':r['spec'],'player':r['player'],'action':r['target'],'reward':r['scores'][str(r['target'])],'value':.5} for r in oracle['records']]
        p,_=initial_state(7103,'supervised',PLAN);s=summarize(p,records,oracle)
        self.assertEqual(len(s['cells']),32);self.assertTrue(all(c['n']==16 and c['sampled_accuracy']==1 for c in s['cells'].values()))
        bad=copy.deepcopy(records);bad[0]['reward']=.123
        with self.assertRaises(ValueError):summarize(p,bad,oracle)
        bad=copy.deepcopy(records);bad[0]['spec']['run_seed']=0
        with self.assertRaises(ValueError):summarize(p,bad,oracle)
        inv=input_inventory(oracle,PLAN);self.assertEqual(inv['supervised_training_unique_inputs'],82)
        self.assertEqual(inv['overlap_with_supervised_training']['train'],82);self.assertEqual(inv['overlap_with_supervised_training']['variant'],0)
        with tempfile.TemporaryDirectory() as tmp,Pool(Path(tmp),2) as pool:
            sample=probe(pool,[{'panel':'train','context':c,'episode':0,'run_seed':9915} for c in [0,8]])
            self.assertEqual(sample['branches'],14);self.assertGreater(sample['minimum_gap'],0)

if __name__=='__main__':unittest.main()
