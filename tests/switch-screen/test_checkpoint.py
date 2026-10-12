import copy
from pathlib import Path
import tempfile
import unittest
import jax.numpy as jnp
from pokemon_learner.checkpoint import digest,wire
from pokemon_switch_engine import learner,checkpoint as original
from pokemon_switch_screen import checkpoint as cp
from pokemon_switch_screen.provenance import read


class Checkpoint(unittest.TestCase):
    def test_unchanged_kernels_and_exact_replay(self):
        s=learner.start(1009);result=cp.transaction(s,1009)
        self.assertEqual(cp.result_fingerprints(result),cp.result_fingerprints(original.transaction(s,1009)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';cp.save(p,result[0],1009,{'test':1})
            loaded=cp.load(p,1009,{'test':1})
            self.assertEqual(cp.result_fingerprints(cp.transaction(loaded,1009)),cp.result_fingerprints(cp.transaction(result[0],1009)))
            with self.assertRaises(ValueError):cp.save(p,result[0],1009,{'test':1})

    def test_binding_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';cp.save(p,learner.start(1009),1009,{'test':1});doc=read(p)
            for seed,binding in [(2027,{'test':1}),(1009,{'test':2})]:
                with self.assertRaises(ValueError):cp.load(p,seed,binding)
            for field,value in [('live_games',True),('cutoff_games_reset',False),('counters',[0,True,0]),('tables',[])]:
                broken=copy.deepcopy(doc);broken['body'][field]=value;broken['sha256']=digest(broken['body']);p.write_bytes(wire(broken))
                with self.assertRaises(ValueError):cp.load(p,1009,{'test':1})

    def test_seed_and_counter_limits(self):
        s=learner.start(1009)
        for seed in [True,101,-1]:
            with self.assertRaises(ValueError):cp.transaction(s,seed)
        with self.assertRaises(ValueError):cp.validate(s._replace(version=jnp.int32(1)))
        with self.assertRaises(ValueError):cp.transaction(s._replace(version=jnp.int32(8193),step=jnp.int32(16386),next_game=jnp.uint32(262176)),1009)

if __name__=='__main__':unittest.main()
