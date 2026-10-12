import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pokemon_learner.trainer import Trainer
from pokemon_learner.trainer import shared_outcome_rewards
from pokemon_learner.checkpoint import state, save, load, digest

ROOT=Path(__file__).resolve().parents[2]
PLAN=json.loads((ROOT/'contracts/learner-v1/canaries.json').read_text())


class ResumeTest(unittest.TestCase):
    def test_shared_terminal_reward_perspectives_and_frozen_trainer_ratios(self):
        scores=shared_outcome_rewards([[0,1],[1,2],[0,2],[1,1]],[0,2])
        np.testing.assert_array_equal(scores,[[1,0],[0,1],[.5,.5],[.5,.5]])
        for task in ['bandit','delayed']:
            t=Trainer(task,11,PLAN)
            self.assertLessEqual(t.update(t.collect())[0]['max_ratio_error'],1e-5)

    def test_boundary_next_samples_and_updates_bit_identical_both_tasks(self):
        for task in ['bandit','delayed']:
            t=Trainer(task,11,PLAN)
            for _ in range(3):t.update(t.collect())
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'checkpoint.json';save(p,t);restored=load(p,expected_plan=PLAN)
                self.assertEqual(state(t),state(restored))
                for _ in range(3):
                    a,b=t.collect(),restored.collect()
                    for key in ['states','mask','actions','old_logp','old_values','advantages','returns','next_index']:
                        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
                    self.assertEqual(t.update(a),restored.update(b))
                    self.assertEqual(state(t),state(restored))

    def test_collection_and_update_crash_rollback_accounting(self):
        t=Trainer('delayed',29,PLAN);t.update(t.collect())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json';save(p,t);committed=state(t)
            expected=load(p,expected_plan=PLAN);buffer=expected.collect();expected.update(buffer)
            during_collection=load(p,expected_plan=PLAN)
            with self.assertRaisesRegex(RuntimeError,'collection crash'):during_collection.collect(crash_before_linkage=True)
            self.assertEqual(during_collection.phase,'collecting')
            self.assertEqual(during_collection.decisions-committed['counters']['decisions'],len(buffer.states))
            with self.assertRaisesRegex(ValueError,'completed-update'):save(Path(d)/'partial.json',during_collection)
            recovered=load(p,expected_plan=PLAN);recovered.update(recovered.collect())
            self.assertEqual(state(expected),state(recovered))
            crashed=load(p,expected_plan=PLAN);discard=crashed.collect()
            with self.assertRaisesRegex(ValueError,'completed-update'):save(Path(d)/'bad.json',crashed)
            lost_samples=crashed.decisions-committed['counters']['decisions']
            self.assertEqual(lost_samples,len(discard.states))
            recovered=load(p,expected_plan=PLAN);replayed=recovered.collect()
            np.testing.assert_array_equal(discard.actions,replayed.actions);recovered.update(replayed)
            self.assertEqual(state(expected),state(recovered))
            crashed=load(p,expected_plan=PLAN);discard=crashed.collect()
            with self.assertRaisesRegex(RuntimeError,'update crash'):crashed.update(discard,crash_after_minibatches=1)
            self.assertEqual(crashed.optimizer.step-committed['optimizer']['step'],1)
            recovered=load(p,expected_plan=PLAN);recovered.update(recovered.collect())
            self.assertEqual(state(expected),state(recovered))

    def test_atomic_partial_checksum_runtime_and_stale_buffer(self):
        t=Trainer('bandit',47,PLAN)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json';save(p,t);before=p.read_bytes()
            b=t.collect();t.update(b)
            with self.assertRaisesRegex(ValueError,'Stale'):t.update(b)
            with self.assertRaisesRegex(RuntimeError,'write crash'):save(p,t,interrupt_before_replace=True)
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(load(p,expected_plan=PLAN).updates,0)
            self.assertTrue(p.with_name(p.name+'.partial').exists())
            with self.assertRaisesRegex(ValueError,'Interrupted checkpoint'):save(p,t)
            # Preserve the interrupted write, then resume checkpoint publication.
            p.with_name(p.name+'.partial').rename(Path(d)/'interrupted-write.json')
            save(p,t);self.assertEqual(load(p,expected_plan=PLAN).updates,1)
            doc=json.loads(before);doc['body']['counters']['decisions']=1;p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'checksum'):load(p,expected_plan=PLAN)
            doc=json.loads(before);doc['body']['provenance']['runtime']['numpy']='wrong';doc['sha256']=digest(doc['body']);p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'drift'):load(p,expected_plan=PLAN)

    def test_toy_hidden_reward_has_no_policy_input_path(self):
        t=Trainer('delayed',11,PLAN)
        before=t.probabilities();t.plan['delayed']['rewarding_actions']=[2,0]
        # Compare logits/probabilities at fixed visible state; changing hidden target
        # changes scoring only, never the actor's distribution.
        from pokemon_learner.core import distribution
        mask=np.tile(PLAN['delayed']['root_mask'],(2,1))
        p,_=distribution(t.parameters['logits'][[0,1]],mask)
        self.assertEqual(before,[float(p[0,0]),float(p[1,2])])
