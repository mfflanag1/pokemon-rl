import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pokemon_full.policy import initialize,distribution
from pokemon_matchup.network import extend,forward
from pokemon_matchup.learning import optimize_batch as predecessor
from pokemon_batch_collection.learning import optimize_batch
from pokemon_batch_collection.common import save,load,same_state

def data(n,p):
    rng=np.random.default_rng(441)
    x=rng.normal(0,.03,(n,1278)).astype(np.float32);mask=np.ones((n,14),bool)
    logits,v=forward(p,x);_,logs=distribution(logits,mask);a=np.arange(n,dtype=np.int32)%14
    return dict(features=x,mask=mask,actions=a,old_logp=np.asarray(logs)[np.arange(n),a],
        returns=rng.uniform(0,1,n).astype(np.float32),advantages=rng.normal(0,.2,n).astype(np.float32))
class Learning(unittest.TestCase):
    def test_small_bitwise_matches_sealed_optimizer(self):
        p,o=extend(initialize(41));b=data(1024,p)
        q,qo,m=predecessor(p,o,b,93,'matchup');r,ro,n=optimize_batch(p,o,b,93,'small')
        same_state(q,qo,r,ro);self.assertEqual(m,n)
    def test_large_each_row_twice_and_frozen_targets(self):
        p,o=extend(initialize(42));b=data(8192,p);b['features'][:,0]=np.arange(8192)
        ids=[]
        def capture(p,o,part):
            i=np.asarray(part['features'])[:,0].astype(int);ids.extend(i.tolist())
            for k in ['old_logp','advantages','returns']:np.testing.assert_array_equal(np.asarray(part[k]),b[k][i])
            return p,(o[0],o[1],o[2]+1),{}
        with patch('pokemon_batch_collection.learning.update',capture):_,o,m=optimize_batch(p,o,b,94,'large')
        self.assertEqual(int(o[2]),64);self.assertEqual(len(m),64)
        for epoch in [ids[:8192],ids[8192:]]:self.assertEqual(sorted(epoch),list(range(8192)))
        self.assertNotEqual(ids[:8192],ids[8192:])
    def test_large_original_next_update_checkpoint_recovery(self):
        p,o=extend(initialize(43));b=data(8192,p)
        p,o,_=optimize_batch(p,o,b,95,'large')
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);save(d,p,o,'hash',43,'large',1,208)
            q,qo=load(d,'hash',43,'large',1,208)
            r,ro,m=optimize_batch(p,o,b,96,'large');s,so,n=optimize_batch(q,qo,b,96,'large')
            same_state(r,ro,s,so);self.assertEqual(m,n);self.assertEqual(int(ro[2]),128)
            with self.assertRaises(ValueError):load(d,'hash',43,'small',1,208)
            with self.assertRaises(ValueError):save(d,p,o,'hash',43,'large',2,208)
    def test_wrong_rows_invalid_values_and_counter_fail(self):
        p,o=extend(initialize(44));b=data(1024,p)
        with self.assertRaises(ValueError):optimize_batch(p,o,b,1,'large')
        with self.assertRaises(ValueError):optimize_batch(p,o,b,1,'best')
        b['advantages'][0]=np.nan
        with self.assertRaises(ValueError):optimize_batch(p,o,b,1,'small')
        b=data(1024,p)
        with self.assertRaises(ValueError):optimize_batch(p,(o[0],o[1],np.int32(2**31-8)),b,1,'small')
    def test_midpoint_metadata_same_rows_steps_different_refresh_count(self):
        from pokemon_batch_collection.common import metadata
        a=metadata('h',7103,'small',64,100);b=metadata('h',7103,'large',8,100)
        self.assertEqual(a['selected_rows'],b['selected_rows']);self.assertEqual(a['optimizer_step'],b['optimizer_step'])
        self.assertNotEqual(a['cohorts'],b['cohorts'])
if __name__=='__main__':unittest.main()
