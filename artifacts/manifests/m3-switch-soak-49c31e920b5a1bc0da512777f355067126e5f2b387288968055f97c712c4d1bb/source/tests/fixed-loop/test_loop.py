import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import validate_state
from pokemon_jax_learner.collector import collect as reference_collect
from pokemon_jax_learner import core as reference_core
from pokemon_jax_learner.model import ActorInput, predict_kernel, model_loss_grad
from pokemon_fixed.loop import Executables, start, permutation, objective, check_report
from pokemon_fixed.checkpoint import Runner, save, load, state
from pokemon_learner.checkpoint import digest


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = Executables()
        cls.ex.diagnostics()

    def assertTreeClose(self, a, b, exact=False):
        self.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            if exact: np.testing.assert_array_equal(x,y)
            else: np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)

    def test_collection_matches_sealed_collector_and_own_linkage(self):
        for seed in (11,29,47):
            learner=start(seed)
            for offset in (0,64,999):
                b,s=self.ex.compiled['collect'](learner.parameters,jnp.uint32(offset),jnp.uint32(seed))
                validate_state(s)
                self.assertTrue(np.all(np.asarray(s.winner)>=0))
                ref=reference_collect(learner.parameters,0,np.arange(offset,offset+64,dtype=np.uint32),
                                      environment_seed=seed,action_seed=seed)
                valid=np.asarray(b.valid)
                for name in b.actor._fields:
                    got=np.asarray(getattr(b.actor,name))[valid]
                    expected=getattr(ref.actor,name)
                    np.testing.assert_allclose(got,expected,atol=3e-6,rtol=3e-5)
                for name in ('rewards','terminal','advantages','returns'):
                    np.testing.assert_allclose(np.asarray(getattr(b.targets,name))[valid],getattr(ref.targets,name),atol=3e-6,rtol=3e-5)
                compact_index=np.cumsum(valid)-1
                nxt=np.asarray(b.targets.next_index)[valid]
                np.testing.assert_array_equal(np.where(nxt>=0,compact_index[np.maximum(nxt,0)],-1),ref.targets.next_index)
                self.assertEqual(int(np.asarray(b.targets.rewards).sum()),64)
                self.assertTrue(np.all(np.asarray(b.actor.features)[valid,1:3]==0))

    def test_padded_update_matches_sealed_math_with_same_new_shuffle(self):
        learner=start(19)
        b,_=self.ex.compiled['collect'](learner.parameters,learner.next_game,jnp.uint32(19))
        got,metrics=self.ex.compiled['update'](learner,b,jnp.uint32(19))
        parameters=dict(learner.parameters); opt=reference_core.Adam(parameters,lr=.003)
        for epoch in range(2):
            order=np.asarray(permutation(jnp.uint32(19),learner.version,jnp.int32(epoch),b.valid))
            # Same new full-capacity shuffle, then truly compact reference minibatches.
            for mb,ix in enumerate(order.reshape(4,128)):
                ix=ix[np.asarray(b.valid)[ix]]
                self.assertEqual(int(metrics['valid_count'][epoch,mb]),len(ix))
                if not len(ix): continue
                (_,m),g=model_loss_grad(parameters,b.actor.features[ix],b.actor.mask[ix],b.actor.actions[ix],
                    b.actor.old_logp[ix],b.targets.advantages[ix],b.targets.returns[ix],.2,.5,.01)
                opt.apply(parameters,g)
                for k in m: np.testing.assert_allclose(metrics[k][epoch,mb],m[k],atol=3e-6,rtol=3e-5)
        self.assertTreeClose(got.parameters,parameters)
        self.assertTreeClose(got.m,opt.m);self.assertTreeClose(got.v,opt.v)
        self.assertEqual(int(got.step),opt.step)

    def test_fused_and_separate_paths_and_stable_tracing(self):
        learner=start(9); traces=dict(self.ex.traces)
        for _ in range(8):
            b,_=self.ex.compiled['collect'](learner.parameters,learner.next_game,jnp.uint32(9))
            separate,_=self.ex.compiled['update'](learner,b,jnp.uint32(9))
            fused,r=self.ex(learner,9);check_report(r)
            self.assertTreeClose(fused,separate)
            learner=fused
        self.assertEqual(self.ex.traces,traces)
        with self.assertRaises((TypeError,ValueError)):
            self.ex.compiled['cycle'](learner,jnp.zeros(2,jnp.uint32))
        self.assertEqual(self.ex.traces,traces)

    def test_padding_features_cannot_affect_parameter_gradients(self):
        l=start(7);b,_=self.ex.compiled['collect'](l.parameters,l.next_game,jnp.uint32(7))
        fn=jax.jit(jax.value_and_grad(lambda pars,a: objective(pars,a,b.targets,b.valid,self.ex.plan),has_aux=True))
        reference=fn(l.parameters,b.actor)
        poisoned=b.actor._replace(features=jnp.where(b.valid[:,None],b.actor.features,jnp.nan))
        self.assertTreeClose(reference,fn(l.parameters,poisoned),exact=True)

    def test_all_empty_update_skips_adam_and_single_legal_action(self):
        l=start(5);b,_=self.ex.compiled['collect'](l.parameters,l.next_game,jnp.uint32(5))
        empty=b._replace(valid=jnp.zeros(512,bool))
        got,metrics=self.ex.compiled['update'](l,empty,jnp.uint32(5))
        for field in ('parameters','m','v','step'): self.assertTreeClose(getattr(l,field),getattr(got,field),exact=True)
        self.assertTrue(np.all(np.asarray(metrics['optimizer_applied'])==False))
        obs=ActorInput(b.actor.features,jnp.tile(jnp.array([True,False]),(512,1)))
        probs,_,_=predict_kernel(l.parameters,obs)
        np.testing.assert_array_equal(probs[:,1],0)

    def test_disk_resume_and_failed_cycle_roll_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            runner=Runner(29,self.ex);runner.run();save(path,runner)
            baseline=state(runner)
            restored=load(path,self.ex)
            self.assertTreeClose(runner.run(),restored.run(),exact=True)
            self.assertEqual(state(runner),state(restored))
            for fault in ('after_collect','after_update'):
                crashed=load(path,self.ex)
                with self.assertRaises(RuntimeError): crashed.run(fault=fault)
                with self.assertRaises(ValueError): save(Path(tmp)/'bad.json',crashed)
                self.assertEqual(state(load(path,self.ex)),baseline)
                replay=load(path,self.ex);replay.run()
                self.assertEqual(state(replay),state(runner))

    def test_checkpoint_tamper_and_interrupted_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json';runner=Runner(41,self.ex);runner.run();save(path,runner)
            original=path.read_bytes();before=state(runner)
            runner.run()
            with self.assertRaises(RuntimeError): save(path,runner,interrupt_before_replace=True)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(state(load(path,self.ex)),before)
            with self.assertRaises(FileExistsError): save(path,runner)
            for key,value in [('next_game',7),('decisions',-2),('step',1),('mode','live')]:
                doc=json.loads(original);doc['body'][key]=value;doc['sha256']=digest(doc['body'])
                bad=Path(tmp)/'bad.json';bad.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): load(bad,self.ex)
            doc=json.loads(original);doc['body']['provenance']['runtime']['backend']='gpu';doc['sha256']=digest(doc['body'])
            bad.write_text(json.dumps(doc))
            with self.assertRaises(ValueError): load(bad,self.ex)


if __name__=='__main__':unittest.main()
