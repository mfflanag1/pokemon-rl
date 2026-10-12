"""Counter-wrap and failed-publication probes, not independent training episodes."""
import unittest
import jax.numpy as jnp
import numpy as np
from pokemon_fixed.loop import Executables, start, check_report
from pokemon_fixed.checkpoint import Runner
from pokemon_jax_learner.collector import collect as reference_collect


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex=Executables();cls.ex.diagnostics()

    def test_largest_seed_and_unsigned_game_ids_match_reference(self):
        seed=2**32-1;learner=start(seed);offset=3_199_999_936
        buffer,_=self.ex.compiled['collect'](learner.parameters,jnp.uint32(offset),jnp.uint32(seed))
        ref=reference_collect(learner.parameters,0,np.arange(offset,offset+64,dtype=np.uint32),
                              environment_seed=seed,action_seed=seed)
        valid=np.asarray(buffer.valid)
        for name in ('game_id','side','turn','actions','mask'):
            np.testing.assert_array_equal(np.asarray(getattr(buffer.actor,name))[valid],getattr(ref.actor,name))
        for name in ('old_logp','old_values','features'):
            np.testing.assert_allclose(np.asarray(getattr(buffer.actor,name))[valid],getattr(ref.actor,name),atol=3e-6,rtol=3e-5)

    def test_last_admitted_update_does_not_wrap_and_next_is_rejected(self):
        # Synthetic late counter state, with fresh finite weights/moments. This probes
        # addressing and integer arithmetic; it is not a claimed 50-million-update run.
        runner=Runner(73,self.ex);limit=self.ex.plan['maximum_updates'];version=limit-1
        runner.learner=runner.learner._replace(version=jnp.int32(version),
            next_game=jnp.uint32(version*64),step=jnp.int32(version*8))
        runner.decisions=version*400;runner.executed_moves=version*350
        report=runner.run();check_report(report)
        self.assertEqual(int(runner.learner.version),limit)
        self.assertEqual(int(runner.learner.next_game),limit*64)
        self.assertLessEqual(int(runner.learner.step),limit*8)
        self.assertGreater(runner.decisions,2**32)
        before=runner.learner
        with self.assertRaises(ValueError):runner.run()
        self.assertIs(runner.learner,before)
        self.assertEqual(runner.phase,'idle')

    def test_actual_nonfinite_state_fails_without_committing(self):
        runner=Runner(11,self.ex)
        bad=dict(runner.learner.parameters);bad['policy_b']=jnp.array([jnp.nan,0],jnp.float32)
        runner.learner=runner.learner._replace(parameters=bad);before=runner.learner
        with self.assertRaises(ValueError):runner.run()
        self.assertIs(runner.learner,before)
        self.assertEqual(runner.decisions,0)
        self.assertEqual(int(runner.learner.version),0)
        with self.assertRaises(ValueError):runner.run()


if __name__=='__main__':unittest.main()
