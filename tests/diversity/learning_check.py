import tempfile,sys,unittest,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_balanced.common import initialize
from pokemon_balanced.learning import bc_step as legacy
from pokemon_diversity.learning import bc_step
from pokemon_diversity.sampling import PreparedSampler
from pokemon_diversity.common import compare_control,read,PLAN,load_strict_checkpoint

class Learning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'artifacts/runs/f16-balanced-imitation-v1/inputs'
        with np.load(base/'teacher.npz',allow_pickle=False) as z:cls.data={k:z[k] for k in z.files}
        with np.load(base/'old-supervised.npz',allow_pickle=False) as z:cls.old={k:z[k] for k in z.files}
        cls.sampler=PreparedSampler(cls.data)
    def equal(self,a,b):
        for x,y in [(a[0],b[0]),(a[1][0],b[1][0]),(a[1][1],b[1][1])]:
            self.assertEqual(set(x),set(y))
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        self.assertEqual(int(a[1][2]),int(b[1][2]))
    def test_eight_updates_identical_to_sealed(self):
        p,o=initialize(7103);q,v=initialize(7103)
        for step in range(8):
            p,o,a=bc_step(p,o,self.sampler,self.old,7103,step)
            q,v,b=legacy(q,v,self.data,self.old,7103,step)
            self.equal((p,o),(q,v));self.assertEqual(a,b)
    def test_original_next_update_checkpoint(self):
        p,o=initialize(8209);p,o,_=bc_step(p,o,self.sampler,self.old,8209,0)
        with tempfile.TemporaryDirectory() as td:
            save_checkpoint(td,p,o,{'seed':8209,'input_hash':'test','cohorts':1,'cursors':{'bc':1,'arm':'original'}})
            q,v,_=load_checkpoint(td,'test')
            a=bc_step(p,o,self.sampler,self.old,8209,1);b=bc_step(q,v,self.sampler,self.old,8209,1)
            self.equal(a,b);self.assertEqual(a[2],b[2])
    def test_strict_checkpoint_inventory(self):
        p,o=initialize(7103)
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);save_checkpoint(directory,p,o,{'seed':7103,'input_hash':'strict','cohorts':0,'cursors':{}})
            load_strict_checkpoint(directory,'strict')
            pointer=json.loads((directory/'latest.json').read_text());path=directory/pointer['file']
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            arrays['unexpected_moment']=np.zeros(1,np.float32)
            with open(path,'wb') as f:np.savez_compressed(f,**arrays)
            pointer['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(directory/'latest.json').write_text(json.dumps(pointer))
            with self.assertRaises(ValueError):load_strict_checkpoint(directory,'strict')
    def test_control_guard_rejects_wrong_state(self):
        p,o=initialize(9311)
        with self.assertRaises(ValueError):compare_control(p,o,9311,read(PLAN))
if __name__=='__main__':unittest.main()
