import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.loop import Executables, start, check_report, objective
from pokemon_binary.policy import act, predict_kernel
from pokemon_binary.checkpoint import Runner, save, load, state
from pokemon_jax_learner.model import ActorInput
from pokemon_jax_learner.collector import collect as reference_collect
from pokemon_learner.checkpoint import digest


def exact(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe=Executables();cls.exe.diagnostics()

    def test_collector_matches_compact_host_control(self):
        for seed in (11,29,47):
            l=start(seed)
            # Also reproduce the constructed historical rare-tail case, both ways.
            for bias in (np.array([0,0],np.float32),np.array([0,-27.038734],np.float32),np.array([-27.038734,0],np.float32)):
                p=dict(l.parameters,policy_w=jnp.zeros((59,2)),policy_b=jnp.array(bias))
                b,s=self.exe.compiled['collect'](p,l.next_game,jnp.uint32(seed));valid=np.asarray(b.valid)
                ref=reference_collect(p,0,np.arange(64,dtype=np.uint32),environment_seed=seed,action_seed=seed,actor_fn=act)
                for name in b.actor._fields:
                    a=np.asarray(getattr(b.actor,name))[valid];other=getattr(ref.actor,name)
                    if a.dtype.kind=='f':np.testing.assert_allclose(a,other,atol=3e-6,rtol=3e-5)
                    else:np.testing.assert_array_equal(a,other)
                for name in ('rewards','terminal','advantages','returns'):
                    np.testing.assert_allclose(np.asarray(getattr(b.targets,name))[valid],getattr(ref.targets,name),atol=3e-6,rtol=3e-5)
                to_compact=np.full(512,-1,np.int32);to_compact[valid]=np.arange(valid.sum())
                nxt=np.asarray(b.targets.next_index)[valid]
                np.testing.assert_array_equal(np.where(nxt<0,-1,to_compact[np.maximum(nxt,0)]),ref.targets.next_index)
                self.assertTrue(np.all(np.asarray(s.winner)>=0))
                _,m=objective(p,b.actor,b.targets,b.valid,self.exe.plan)
                self.assertLessEqual(float(m['max_ratio_error']),1e-5)

    def test_fused_split_update_and_padding(self):
        l=start(31)
        for _ in range(8):
            b,_=self.exe.compiled['collect'](l.parameters,l.next_game,jnp.uint32(31))
            split,m=self.exe.compiled['update'](l,b,jnp.uint32(31))
            fused,r=self.exe(l,31);check_report(r)
            for x,y in zip(jax.tree.leaves(split),jax.tree.leaves(fused)):
                np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)
            poisoned=b._replace(actor=b.actor._replace(features=jnp.where(b.valid[:,None],b.actor.features,jnp.nan)))
            clean=self.exe.compiled['update'](l,b,jnp.uint32(31))
            dirty=self.exe.compiled['update'](l,poisoned,jnp.uint32(31));exact(clean,dirty)
            empty=b._replace(valid=jnp.zeros(512,bool))
            new,metrics=self.exe.compiled['update'](l,empty,jnp.uint32(31))
            exact((new.parameters,new.m,new.v,new.step),(l.parameters,l.m,l.v,l.step))
            self.assertFalse(np.any(metrics['optimizer_applied']))
            l=fused
        self.assertEqual(self.exe.traces,dict(collect=1,update=1,cycle=1))

    def test_checkpoint_identity_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';r=Runner(42,self.exe)
            for _ in range(3):r.run()
            save(path,r);s=state(r);restored=load(path,self.exe)
            exact(r.run(),restored.run());self.assertEqual(state(r),state(restored))
            for fault in ('after_collect','after_update'):
                broken=load(path,self.exe)
                with self.assertRaises(RuntimeError):broken.run(fault=fault)
                with self.assertRaises(ValueError):save(Path(tmp)/'broken.json',broken)
                again=load(path,self.exe);expected=load(path,self.exe)
                exact(again.run(),expected.run());self.assertEqual(state(again),state(expected))
            original=path.read_bytes();doc=json.loads(original)
            for key,value in (('format','fixed-loop-checkpoint-1'),('random_streams','old sampler'),('seed',-1)):
                changed=json.loads(original);changed['body'][key]=value;changed['sha256']=digest(changed['body']);path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):load(path,self.exe)
            path.write_bytes(original)
            with self.assertRaises(RuntimeError):save(path,r,interrupt_before_replace=True)
            self.assertEqual(path.read_bytes(),original)

if __name__=='__main__':unittest.main()
