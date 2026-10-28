import copy
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_credit.common import initialize,read,specs
from pokemon_credit.learning import optimize as sealed_optimize
from pokemon_entropy.learning import optimize,no_entropy_objective
from pokemon_full_train.learning import objective as ppo_objective,derived_seed
from pokemon_fit.learning import objective as fit_objective
from pokemon_fit.data import fingerprint
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_full.learner import parameter_hash
from pokemon_full.policy import forward,distribution
from pokemon_transfer.common import tactical,retention
from pokemon_entropy.common import PLAN,validate_plan,admit

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.opt=initialize(7103)
        with np.load(ROOT/'artifacts/runs/f10-long-transfer-v1/inputs/supervised.npz') as d:cls.sup={k:d[k] for k in d.files}
        x=np.tile(cls.sup['features'],(2,1));m=np.tile(cls.sup['mask'],(2,1));a=np.tile(cls.sup['targets'],2)
        _,logs=distribution(forward(cls.p,x)[0],m)
        cls.batch={'features':x,'mask':m,'actions':a,'old_logp':np.asarray(logs)[np.arange(256),a],
            'advantages':np.linspace(-.8,.8,256,dtype=np.float32),'returns':np.linspace(0,1,256,dtype=np.float32)}

    def test_exact_control_kernel_parity(self):
        a,oa,ma=optimize(self.p,self.opt,self.batch,self.sup,.01);b,ob,mb=sealed_optimize(self.p,self.opt,self.batch,self.sup)
        self.assertEqual(parameter_hash(a),parameter_hash(b));self.assertEqual(ma,mb)
        for x,y in [(oa[0],ob[0]),(oa[1],ob[1])]:
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        self.assertEqual(int(oa[2]),2)

    def test_no_entropy_is_only_loss_gradient_change(self):
        part={k:jnp.asarray(v) for k,v in self.batch.items()};part['valid']=jnp.ones(256,bool);sup={k:jnp.asarray(v) for k,v in self.sup.items()}
        def reference(p):
            loss,m=ppo_objective(p,part)
            return loss+jnp.float32(.01)*m['entropy']+jnp.float32(.1)*fit_objective(p,sup)
        value,metrics=no_entropy_objective(self.p,part,sup)
        np.testing.assert_allclose(value,reference(self.p),rtol=1e-6,atol=1e-7)
        a=jax.grad(lambda p:no_entropy_objective(p,part,sup)[0])(self.p);b=jax.grad(reference)(self.p)
        for k in a:np.testing.assert_allclose(a[k],b[k],rtol=3e-5,atol=3e-7)
        self.assertGreater(float(metrics['entropy']),0)
        q,_,_=optimize(self.p,self.opt,self.batch,self.sup,0.);c,_,_=optimize(self.p,self.opt,self.batch,self.sup,.01)
        self.assertNotEqual(parameter_hash(q),parameter_hash(c))

    def test_checkpoint_next_update_for_both_arms(self):
        for coefficient in [.01,0.]:
            q,opt,_=optimize(self.p,self.opt,self.batch,self.sup,coefficient)
            with tempfile.TemporaryDirectory() as d:
                save_checkpoint(Path(d),q,opt,{'seed':7103,'input_hash':'f14-engineering','cohorts':1})
                rp,ro,_=load_checkpoint(Path(d),'f14-engineering')
                a,oa,ma=optimize(q,opt,self.batch,self.sup,coefficient);b,ob,mb=optimize(rp,ro,self.batch,self.sup,coefficient)
                self.assertEqual(parameter_hash(a),parameter_hash(b));self.assertEqual(ma,mb)
                for x,y in [(oa[0],ob[0]),(oa[1],ob[1])]:
                    for k in x:np.testing.assert_array_equal(x[k],y[k])

    def test_invalid_inputs_and_immutable_evaluation(self):
        original=fingerprint(self.batch);before=parameter_hash(self.p)
        for c in [-.01,.001,True,float('nan')]:
            with self.assertRaises(ValueError):optimize(self.p,self.opt,self.batch,self.sup,c)
        with self.assertRaises(ValueError):optimize(self.p,self.opt,{**self.batch,'probe_labels':np.ones(256)},self.sup,0.)
        bad={k:v.copy() for k,v in self.batch.items()};bad['mask'][0]=False
        with self.assertRaises(ValueError):optimize(self.p,self.opt,bad,self.sup,0.)
        optimize(self.p,self.opt,self.batch,self.sup,0.)
        self.assertEqual(original,fingerprint(self.batch));self.assertEqual(before,parameter_hash(self.p))

    def test_bank_roles_fresh_keys_and_admission(self):
        plan=read(PLAN);validate_plan(plan);bank,sup,oracle=admit(plan);seed=derived_seed('f14-continuation',7103)
        fit=specs(bank,plan,seed,0,32,'fit');probe=specs(bank,plan,seed,0,96,'probe');old=specs(bank,plan,7103,0,96,'probe')
        self.assertEqual(len({s['scenario_hash'] for s in probe}),24);self.assertFalse({s['scenario_hash'] for s in fit}&{s['scenario_hash'] for s in probe})
        self.assertNotEqual(probe[0]['actor_seed'],old[0]['actor_seed'])
        self.assertEqual([s['scenario_hash'] for s in probe],[s['scenario_hash'] for s in old])
        bad=copy.deepcopy(plan);bad['target_lambda']=1.
        with self.assertRaises(ValueError):validate_plan(bad)
        self.assertEqual(len(oracle['records']),512);self.assertEqual(sup['features'].shape,(128,928))

    def test_tactical_retention_boundaries(self):
        settings=read(PLAN)['tactical_thresholds'];initial={'cells':{f'{p}/{i}':{'n':16,'target_mass':.75,'greedy_accuracy':1.} for p in ['train','variant'] for i in range(16)}}
        final=copy.deepcopy(initial);self.assertTrue(retention(initial,final,settings)['passed'])
        final['cells']['variant/7']['target_mass']=.69;self.assertFalse(retention(initial,final,settings)['passed'])
        final=copy.deepcopy(initial);final['cells']['train/2']['greedy_accuracy']=.94;self.assertFalse(retention(initial,final,settings)['passed'])

if __name__=='__main__':unittest.main()
