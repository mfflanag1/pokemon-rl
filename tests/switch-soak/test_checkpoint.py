import copy
import json
from pathlib import Path
import tempfile
import unittest
import jax.numpy as jnp
from pokemon_learner.checkpoint import digest,wire
from pokemon_switch_engine import learner,checkpoint as sealed
from pokemon_switch_soak import checkpoint as cp
from pokemon_switch_soak.provenance import PLAN,read
from pokemon_soak.campaign import check_gap


class Checkpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=learner.start(101);cls.bind={'mode':'smoke','source-test':'fixed'}

    def test_extended_counter_boundary(self):
        state=self.state._replace(step=jnp.int32(130),version=jnp.int32(65),next_game=jnp.uint32(2080))
        cp.validate(state)
        with self.assertRaises(ValueError):sealed.validate(state)
        next_state=cp.transaction(state)[0];self.assertEqual(cp.counters(next_state),dict(optimizer_step=132,updates=66,next_game=2112))

    def test_counter_rejections_and_cap(self):
        for changes in [dict(step=jnp.int32(1)),dict(version=jnp.int32(-1)),dict(next_game=jnp.int32(0))]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):cp.validate(self.state._replace(**changes))
        n=PLAN['maximum_updates'];state=self.state._replace(step=jnp.int32(2*n),version=jnp.int32(n),next_game=jnp.uint32(32*n))
        cp.validate(state)
        with self.assertRaises(ValueError):cp.transaction(state)

    def test_exact_restore_and_seed_binding(self):
        state=cp.transaction(self.state)[0]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'one.json';cp.save(path,state,self.bind);restored=cp.load(path,self.bind)
            self.assertEqual(cp.fingerprint(restored),cp.fingerprint(state))
            self.assertEqual(cp.result_fingerprints(cp.transaction(restored)),cp.result_fingerprints(cp.transaction(state)))
            with self.assertRaises(ValueError):cp.load(path,{**self.bind,'mode':'qualification'})
            with self.assertRaises(ValueError):cp.save(path,state,self.bind)

    def test_rehashed_corruptions(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'one.json';cp.save(path,self.state,self.bind);original=read(path)
            for corruption in ('seed','counter','nan','extra','phase'):
                doc=copy.deepcopy(original);b=doc['body']
                if corruption=='seed':b['seed']=211
                if corruption=='counter':b['counters']['optimizer_step']=True
                if corruption=='nan':b['tables'][0]['vb']['hex']='0000c07f'
                if corruption=='extra':b['extra']=1
                if corruption=='phase':b['environment']['live_games']=1
                doc['sha256']=digest(b);path.write_bytes(wire(doc))
                with self.subTest(corruption=corruption),self.assertRaises(ValueError):cp.load(path,self.bind)

    def test_interrupted_write_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'one.json'
            with self.assertRaises(RuntimeError):cp.save(path,self.state,self.bind,interrupt=True)
            self.assertFalse(path.exists());before=path.with_suffix('.json.partial').read_bytes()
            with self.assertRaises(ValueError):cp.save(path,self.state,self.bind)
            self.assertEqual(before,path.with_suffix('.json.partial').read_bytes())

    def test_transaction_discard(self):
        original=cp.fingerprint(self.state)
        for stage in ('collect','update'):
            with self.assertRaises(RuntimeError):cp.transaction(self.state,fault=stage)
            self.assertEqual(cp.fingerprint(self.state),original)

    def test_sealed_kernel_identity(self):
        self.assertEqual(cp.result_fingerprints(cp.transaction(self.state)),cp.result_fingerprints(sealed.transaction(self.state,101)))

    def test_clock_pause_and_discontinuity(self):
        self.assertEqual(check_gap((100,1000),(101,1001),30),1)
        for now in ((101,1040),(140,1001),(101,950),(99,1001)):
            with self.assertRaises(ValueError):check_gap((100,1000),now,30)

    def test_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'one.json';path.write_text('{"body":{},"body":{}}')
            with self.assertRaises(ValueError):cp.load(path,self.bind)


if __name__=='__main__':unittest.main()
