import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from pokemon_balanced.common import PLAN,read,admit,initialize,uniform_control,validate_plan,analysis
from pokemon_balanced.learning import bc_step,probe,support,NAMES
from pokemon_balanced.sampling import classify,select
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_full_train.learning import derived_seed
from pokemon_fit.learning import checked_step
from pokemon_full.learner import parameter_hash

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read(PLAN);cls.old,cls.oracle,cls.data,_,_=admit(cls.plan)
    def test_exact_kernel_and_control_rehearsal(self):
        p,opt=initialize(7103);new,ids=select(self.data,derived_seed('f16-balanced-sample',7103,0))
        old_ids=np.random.default_rng(derived_seed('f15-old-sample',7103,0)).integers(0,128,64)
        batch={k:jnp.asarray(np.concatenate([new[k],self.old[k][old_ids]])) for k in new}
        expected,eo,em=checked_step(p,opt,batch);actual,ao,item=bc_step(p,opt,self.data,self.old,7103,0)
        self.assertEqual(parameter_hash(expected),parameter_hash(actual));self.assertEqual(em,item['metrics'])
        self.assertEqual(item['teacher_indices'],ids.tolist());self.assertEqual(item['old_indices'],read(ROOT/'artifacts/runs/f15-full-imitation-v1/7103/bc/update-0000.json')['old_indices'])
        for key in ['vw','vb']:np.testing.assert_array_equal(actual[key],p[key])
        for a,b in zip(eo[:2],ao[:2]):
            for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_original_next_update_checkpoint(self):
        p,opt=initialize(7103);p,opt,_=bc_step(p,opt,self.data,self.old,7103,0)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp,p,opt,{'seed':7103,'input_hash':'engineering','cohorts':1,'cursors':{'bc':1}})
            restored,ro,_=load_checkpoint(tmp,'engineering')
            a,ao,am=bc_step(p,opt,self.data,self.old,7103,1);b,bo,bm=bc_step(restored,ro,self.data,self.old,7103,1)
            self.assertEqual(parameter_hash(a),parameter_hash(b));self.assertEqual(am,bm);self.assertEqual(int(ao[2]),2)
            for x,y in zip(ao[:2],bo[:2]):
                for k in x:np.testing.assert_array_equal(x[k],y[k])
    def test_probe_partition_metrics_and_no_mutation(self):
        labels=classify(self.data);ids=np.concatenate([np.flatnonzero(labels==i)[:35] for i in range(4)])
        small={k:v[ids] for k,v in self.data.items()};p,_=initialize(7103);before=parameter_hash(p)
        probabilities,metrics=probe(p,small)
        self.assertTrue(all({'game_accuracy','game_target_mass'}<=set(m) for m in metrics['families'].values()))
        self.assertEqual(parameter_hash(p),before);self.assertEqual(probabilities.shape,(140,14));self.assertEqual(sum(v['rows'] for v in metrics['families'].values()),140)
        self.assertTrue(np.all(probabilities[~small['mask']]==0))
        for i,name in enumerate(NAMES):
            selected=classify(small)==i
            accuracy=np.mean(probabilities[selected].argmax(1)==small['targets'][selected])
            self.assertAlmostEqual(metrics['families'][name]['decision_accuracy'],accuracy)
    def test_predecessor_control_and_fixed_plan(self):
        validate_plan(self.plan);self.assertEqual(len(self.data['targets']),26278)
        for seed in self.plan['seeds']:
            p,opt=uniform_control(seed,self.plan)
            self.assertEqual(parameter_hash(p),read(ROOT/f'artifacts/runs/f15-full-imitation-v1/{seed}/imitation/evaluation.json')['parameter_hash']);self.assertEqual(int(opt[2]),2048)
        for key,value in [('bc_steps',2049),('teacher_quota',[16,16,16,16]),('prime_used',True)]:
            bad={**self.plan,key:value}
            with self.assertRaises(ValueError):validate_plan(bad)
    def test_coverage_and_strength_are_separate(self):
        plan={**self.plan,'statistics':{**self.plan['statistics'],'resamples':256}}
        def games(win,opponent):
            return [{'episode':i,'swap':sw,'learner_side':'p2' if sw else 'p1','opponent':opponent,'provenance':{'pair':i},'outcome':{'kind':'terminal','rewards':{'p1':1-win if sw else win,'p2':win if sw else 1-win}},'decisions':1,'terminal_value_mse':.25,'value_mean':.5} for i in range(128) for sw in [False,True]]
        panels={e:{o:games(1 if e=='balanced' or o!='tactical' else 0,o) for o in plan['opponents']} for e in plan['endpoints']}
        cells={f'{a}/{i}':{'n':16,'target_mass':.9,'greedy_accuracy':1.} for a in ['train','variant'] for i in range(16)}
        tactical={e:{'cells':copy.deepcopy(cells)} for e in plan['endpoints']}
        probes={e:{'families':{n:{'game_accuracy':.4,'game_target_mass':.4} for n in NAMES}} for e in plan['endpoints']}
        probes['uniform']['families']['tera_move']['game_accuracy']=.1
        result=analysis(panels,tactical,probes,plan);self.assertTrue(result['coverage_screen']);self.assertTrue(result['strength_gate'])
        low=copy.deepcopy(probes);low['balanced']['families']['tera_move']['game_accuracy']=.19
        result=analysis(panels,tactical,low,plan);self.assertFalse(result['coverage_screen']);self.assertTrue(result['strength_gate'])
        equal=copy.deepcopy(panels);equal['balanced']=copy.deepcopy(panels['uniform'])
        result=analysis(equal,tactical,probes,plan);self.assertTrue(result['coverage_screen']);self.assertFalse(result['strength_gate'])
        regression=copy.deepcopy(panels);regression['balanced']['random']=games(0,'random')
        self.assertFalse(analysis(regression,tactical,probes,plan)['strength_gate'])
    def test_support_and_illegal_inputs_fail_before_update(self):
        measured=support(self.data,self.plan);self.assertEqual(set(measured),set(NAMES))
        impossible=copy.deepcopy(self.plan);impossible['coverage_thresholds']['minimum_tera_rows']=10**9
        with self.assertRaises(ValueError):support(self.data,impossible)
        bad={k:v.copy() for k,v in self.data.items()};bad['targets'][0]=14;p,opt=initialize(7103);before=parameter_hash(p)
        with self.assertRaises(ValueError):bc_step(p,opt,bad,self.old,7103,0)
        self.assertEqual(parameter_hash(p),before);self.assertEqual(int(opt[2]),0)
if __name__=='__main__':unittest.main()
