import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_live import learner as original, policy
from pokemon_continuation import engine
from pokemon_live_train import learner as l, checkpoint as cp, evidence as e, evaluate


class TrainingTests(unittest.TestCase):
    def test_first_ten_windows_equal_sealed_m6_all_seeds(self):
        for seed in original.PLAN['seeds']:
            a=original.start(seed);b=l.start(seed)
            for _ in range(10):
                x=original.transaction(a,seed);y=l.transaction(b,seed)
                for one,two in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(one,two)
                a=x[0];b=y[0]
            c=l.transaction(b,seed)[0];self.assertEqual(int(c.windows),11)
            with self.assertRaises(ValueError):original.transaction(a,seed)

    def test_long_counters_checkpoint_next_transaction(self):
        s=l.start(4019)
        for _ in range(12):s=l.transaction(s,4019)[0]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.json';binding={'test':True};cp.save(path,s,binding)
            loaded=cp.load(path,4019,binding)
            self.assertEqual(e.fingerprints(l.transaction(s,4019)),e.fingerprints(l.transaction(loaded,4019)))
            with self.assertRaises(ValueError):cp.load(path,4019,{'test':False})
            with self.assertRaises(FileExistsError):cp.save(path,s,binding)
        for changed in (s._replace(step=s.step+1),s._replace(next_game=s.next_game+32)):
            with self.assertRaises(ValueError):cp.validate(changed)

    def test_counts_close_across_multiple_cohorts(self):
        s=l.start(4019);initial=e.counters(s);initial['next_game']=0;t=e.Totals()
        for _ in range(15):
            r=l.transaction(s,4019);t.add(e.window_summary(s,r));s=r[0]
        e.closed_counts(t,initial,e.counters(s));self.assertEqual(t.counts['games'],96)
        bad=t.document();bad['cutoff']+=1
        with self.assertRaises(ValueError):e.closed_counts(bad,initial,e.counters(s))

    def test_frozen_opponents_scalar_game_replay(self):
        p=policy.initialize(1103);f=policy.initialize(6601)
        for opponent in range(3):
            for seat in (0,1):
                teams,ids=evaluate.schedule(1103,opponent,0,seat)
                result=evaluate.game_batch(p,f,teams,ids,np.uint32(8820),np.int32(seat),np.int32(opponent))
                evaluate.audit_tape(*result,teams,ids,8820)
                self.assertGreater(evaluate.diagnostics(*result,seat)['decisions'],0)
                self.assertEqual(cp.fingerprint(result),cp.fingerprint(evaluate.game_batch(p,f,teams,ids,np.uint32(8820),np.int32(seat),np.int32(opponent))))

    def test_damage_opponent_hidden_boundary(self):
        s=l.start(1103).environment
        changed=s._replace(swift=s.swift.at[:,1,:].set(1-s.swift[:,1,:]),pp=s.pp.at[:,1,:,:].add(-1))
        np.testing.assert_array_equal(evaluate.damage_actions(engine.view_kernel(s,0)),evaluate.damage_actions(engine.view_kernel(changed,0)))
        with self.assertRaises(ValueError):evaluate.damage_actions(s)

    def test_cutoffs_cannot_improve_gate(self):
        init=np.zeros((256,2),np.int32);cutoff=np.full_like(init,-1)
        cell=evaluate.comparison(cutoff,init)
        self.assertLessEqual(cell['adjusted_improvement_lower'],0)
        self.assertFalse(evaluate.gate({k:cell for k in l.PLAN['screen']['opponents']}))
        win=evaluate.comparison(np.ones_like(init),init)
        self.assertTrue(evaluate.gate({k:win for k in l.PLAN['screen']['opponents']}))
        with self.assertRaises(ValueError):evaluate.gate({'random':win})


if __name__=='__main__':unittest.main()
