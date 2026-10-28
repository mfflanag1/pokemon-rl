import tempfile,sys,unittest
from pathlib import Path
import numpy as np
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_learning_curve.common import baseline,PLAN,read
from pokemon_diversity.common import load_strict_checkpoint
from pokemon_full_train.storage import ROOT,save_checkpoint
from pokemon_diversity.learning import bc_step
from pokemon_balanced.learning import bc_step as legacy
from pokemon_diversity.sampling import PreparedSampler
from pokemon_diversity.analysis import paired_accuracy

class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read(PLAN);base=ROOT/cls.plan['predecessor_run']/'inputs'
        with np.load(base/'expanded-teacher.npz',allow_pickle=False) as z:cls.data={k:z[k] for k in z.files}
        with np.load(base/'old-supervised.npz',allow_pickle=False) as z:cls.old={k:z[k] for k in z.files}
        cls.sampler=PreparedSampler(cls.data)
    def equal(self,a,b):
        for x,y in [(a[0],b[0]),(a[1][0],b[1][0]),(a[1][1],b[1][1])]:
            self.assertEqual(set(x),set(y))
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        self.assertEqual(int(a[1][2]),int(b[1][2]))
    def test_first_continued_update_matches_legacy(self):
        p,o,_=baseline(7103,self.plan)
        a=bc_step(p,o,self.sampler,self.old,7103,2048);b=legacy(p,o,self.data,self.old,7103,2048)
        self.equal(a,b);self.assertEqual(a[2],b[2]);self.assertEqual(int(a[1][2]),2049)
    def test_optimizer_not_reset(self):
        p,o,_=baseline(8209,self.plan)
        self.assertTrue(any(np.any(np.asarray(v)!=0) for v in o[0].values()))
        self.assertTrue(any(np.any(np.asarray(v)!=0) for v in o[1].values()))
        zeros={k:jnp.zeros_like(v) for k,v in p.items()}
        a=bc_step(p,o,self.sampler,self.old,8209,2048)
        b=bc_step(p,(zeros,zeros,jnp.int32(0)),self.sampler,self.old,8209,2048)
        self.assertEqual(a[2]['batch_fingerprint'],b[2]['batch_fingerprint'])
        self.assertTrue(any(not np.array_equal(a[0][k],b[0][k]) for k in p))
    def test_sampling_index_not_reset(self):
        p,o,_=baseline(9311,self.plan)
        a=bc_step(p,o,self.sampler,self.old,9311,2048)
        b=bc_step(p,o,self.sampler,self.old,9311,0)
        self.assertNotEqual(a[2]['teacher_indices'],b[2]['teacher_indices']);self.assertNotEqual(a[2]['old_indices'],b[2]['old_indices'])
    def test_original_next_update_recovery(self):
        p,o,_=baseline(7103,self.plan);p,o,_=bc_step(p,o,self.sampler,self.old,7103,2048)
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);save_checkpoint(directory,p,o,{'seed':7103,'input_hash':'f19-test','cohorts':2049,'cursors':{'bc':2049}})
            q,v,meta=load_strict_checkpoint(directory,'f19-test')
            self.assertEqual(meta['optimizer_step'],2049)
            a=bc_step(p,o,self.sampler,self.old,7103,2049);b=bc_step(q,v,self.sampler,self.old,7103,2049)
            self.equal(a,b);self.assertEqual(a[2],b[2])
if __name__=='__main__':unittest.main()
