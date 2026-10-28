import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import copy
import tempfile
import unittest
import numpy as np
from pokemon_live_train import learner as l, checkpoint as cp, evidence as e, evaluate


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=l.start(1103)
        for _ in range(11):cls.state=l.transaction(cls.state,1103)[0]

    def test_recomputed_checksum_cannot_hide_counter_or_state_corruption(self):
        body=cp.body(self.state,{'binding':1})
        variants=[]
        b=copy.deepcopy(body);b['counters'][0]+=1;variants.append(b)
        b=copy.deepcopy(body);b['counters'][3]+=32;variants.append(b)
        b=copy.deepcopy(body);b['ids'][0]+=1;variants.append(b)
        b=copy.deepcopy(body);b['environment']['turn']['hex']='ff'*len(bytes.fromhex(b['environment']['turn']['hex']));variants.append(b)
        b=copy.deepcopy(body);b['format']='live-learner-checkpoint-1';variants.append(b)
        with tempfile.TemporaryDirectory() as folder:
            for i,b in enumerate(variants):
                path=Path(folder)/str(i);e.once(path,{'body':b,'sha256':e.digest(b)})
                with self.assertRaises(ValueError):cp.load(path,1103,{'binding':1})

    def test_context_counter_indices_match_exact_encoder(self):
        features=np.asarray(l.observe(self.state.environment).features)
        np.testing.assert_array_equal(features[:,30]*128,np.repeat(np.asarray(self.state.environment.turn),2))
        np.testing.assert_array_equal(features[:,34]*128,np.repeat(np.asarray(self.state.environment.count),2))

    def test_scalar_audit_rejects_corrupt_evaluation_action_and_outcome(self):
        from pokemon_live import policy
        import jax.numpy as jnp
        teams,ids=evaluate.schedule(1103,0,0,0)
        final,tape=evaluate.game_batch(policy.initialize(1103),policy.initialize(6601),teams,ids,np.uint32(8820),np.int32(0),np.int32(0))
        corrupted=(tape[0].at[0,0,0].set(-1),*tape[1:])
        with self.assertRaises(ValueError):evaluate.audit_tape(final,corrupted,teams,ids,8820)
        with self.assertRaises(ValueError):evaluate.audit_tape(final._replace(hp=final.hp.at[0,0,0].add(1)),tape,teams,ids,8820)

    def test_gate_missing_cells_and_one_seed_regression_not_hidden(self):
        wins=np.ones((256,2),np.int32);losses=np.zeros_like(wins)
        good=evaluate.comparison(wins,losses);bad=evaluate.comparison(losses,wins)
        cells={name:good for name in l.PLAN['screen']['opponents']};cells['damage']=bad
        self.assertFalse(evaluate.gate(cells));self.assertEqual(evaluate.interpretation(cells),'failed-noisy-point-regression-guard')
        cells['damage']=good;cells['random']=evaluate.comparison(wins,wins)
        self.assertEqual(evaluate.interpretation(cells),'inconclusive-at-preregistered-budget')


if __name__=='__main__':unittest.main()
