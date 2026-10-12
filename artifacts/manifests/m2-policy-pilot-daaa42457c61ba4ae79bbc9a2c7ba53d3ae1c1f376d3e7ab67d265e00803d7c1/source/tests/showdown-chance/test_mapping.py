import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_showdown.mapping import map_words, kernel
ROOT=Path(__file__).resolve().parents[2]

class MappingTests(unittest.TestCase):
    def test_pinned_node_oracle(self):
        rows=json.loads((ROOT/'tests/showdown-chance/fixtures/word-oracle.json').read_text())['rows']
        result=map_words(np.array([r['word'] for r in rows],np.uint32))
        self.assertEqual(set(result),{'bucket24','crit','roll','tie0','tie1'})
        for k in ('bucket24','crit','roll'):np.testing.assert_array_equal(result[k],[r[k] for r in rows])
        for seat in (0,1):np.testing.assert_array_equal(result[f'tie{seat}'],[r['ties'][seat] for r in rows])

    def test_large_integer_reference_and_batch_order(self):
        words=np.random.default_rng(910).integers(0,2**32,100000,dtype=np.uint32)
        words=np.r_[words,np.uint32(178956970),np.uint32(178956971)]
        result=kernel(jnp.array(words));expected=np.array([(int(w)*24)//2**32 for w in words],np.int32)
        np.testing.assert_array_equal(result['bucket24'],expected)
        np.testing.assert_array_equal(result['crit'],expected==0)
        np.testing.assert_array_equal(result['roll'],100-(words>>28))
        np.testing.assert_array_equal(result['tie0'],words>>31)
        np.testing.assert_array_equal(result['tie1'],1-(words>>31))
        order=np.random.default_rng(2).permutation(len(words));permuted=kernel(jnp.array(words[order]))
        for k,v in result.items():np.testing.assert_array_equal(np.asarray(v)[order],permuted[k])
        self.assertEqual(result['crit'].dtype,jnp.bool_)
        for k in ('bucket24','roll','tie0','tie1'):self.assertEqual(result[k].dtype,jnp.int32)

    def test_checked_inputs(self):
        for bad in (np.array([],np.uint32),np.array([-1]),np.array([0.],np.float32),np.zeros((1,2),np.uint32),np.array([2**32],np.uint64)):
            with self.assertRaises((TypeError,ValueError)):map_words(bad)
if __name__=='__main__':unittest.main()
