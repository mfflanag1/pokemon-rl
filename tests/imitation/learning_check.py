import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pokemon_imitation.common import PLAN,read,validate_plan,admit,initialize
from pokemon_imitation.learning import bc_step,reset_optimizer,teacher_probe
from pokemon_imitation.collection import specs
from pokemon_fit.learning import checked_step
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_full.learner import parameter_hash
from pokemon_fit.data import fingerprint

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.opt=initialize(7103)
        with np.load(ROOT/'artifacts/runs/f10-long-transfer-v1/inputs/supervised.npz') as d:cls.old={k:d[k].copy() for k in d.files}
        cls.data={**cls.old,'game_index':np.repeat(np.arange(8,dtype=np.int32),16)}
    def test_exact_mixed_update(self):
        p,opt,item=bc_step(self.p,self.opt,self.data,self.old,171,0)
        batch={k:np.concatenate([self.old[k][item['teacher_indices']],self.old[k][item['old_indices']]]) for k in self.old}
        q,qo,m=checked_step(self.p,self.opt,batch)
        self.assertEqual(parameter_hash(p),parameter_hash(q));self.assertEqual(item['metrics'],m);self.assertEqual(item['batch_fingerprint'],fingerprint(batch));self.assertEqual(int(opt[2]),1)
        for k in ['vw','vb']:np.testing.assert_array_equal(p[k],self.p[k])
    def test_reset_and_original_next_update(self):
        p,opt,_=bc_step(self.p,self.opt,self.data,self.old,172,0)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp),p,opt,{'seed':7103,'input_hash':'f15-engineering','cohorts':1})
            q,qo,_=load_checkpoint(Path(tmp),'f15-engineering')
            a,ao,am=bc_step(p,opt,self.data,self.old,172,1);b,bo,bm=bc_step(q,qo,self.data,self.old,172,1)
            self.assertEqual(parameter_hash(a),parameter_hash(b));self.assertEqual(am,bm)
            for i in [0,1]:
                for k in ao[i]:np.testing.assert_array_equal(ao[i][k],bo[i][k])
        reset=reset_optimizer(p);self.assertEqual(int(reset[2]),0)
        self.assertTrue(all(np.all(np.asarray(v)==0) for group in reset[:2] for v in group.values()))
    def test_probe_padding_and_no_mutation(self):
        d={k:np.concatenate([v,v[:9]]) for k,v in self.data.items()};before=fingerprint(d);h=parameter_hash(self.p)
        probabilities,result=teacher_probe(self.p,d)
        self.assertEqual(probabilities.shape,(137,14));self.assertEqual(result['rows'],137);self.assertEqual(result['games'],8)
        self.assertEqual(before,fingerprint(d));self.assertEqual(h,parameter_hash(self.p));self.assertTrue(np.all(probabilities[~d['mask']]==0))
    def test_invalid_and_evaluation_fields(self):
        bad=copy.deepcopy(self.data);bad['targets'][0]=14
        with self.assertRaises(ValueError):bc_step(self.p,self.opt,bad,self.old,1,0)
        bad={**self.data,'outcomes':np.zeros(128)}
        with self.assertRaises(ValueError):bc_step(self.p,self.opt,bad,self.old,1,0)
        with self.assertRaises(ValueError):teacher_probe(self.p,{**self.data,'targets':self.data['targets'].astype(np.int64)})
    def test_admission_and_frozen_budget(self):
        plan=read(PLAN);validate_plan(plan);old,oracle=admit(plan)
        self.assertEqual(fingerprint(old),fingerprint(self.old));self.assertEqual(len(oracle['records']),512)
        for key,value in [('bc_steps',2049),('evaluation_seed',0),('final_reserved_data_used',True)]:
            p=copy.deepcopy(plan);p[key]=value
            with self.assertRaises(ValueError):validate_plan(p)
    def test_balanced_sides_and_fixed_coordinates(self):
        plan=read(PLAN);jobs=specs(plan,7103,8)
        self.assertEqual([s['episode'] for s in jobs],list(range(8,16)))
        self.assertEqual([s['learner_side'] for s in jobs],['p1','p2']*4)
        self.assertTrue(all(s['split']=='train' and s['opponent']=='tactical' and s['swap'] is False for s in jobs))
        self.assertEqual(jobs,specs(plan,7103,8));self.assertNotEqual(jobs[0]['seed'],specs(plan,8209,8)[0]['seed'])
if __name__=='__main__':unittest.main()
