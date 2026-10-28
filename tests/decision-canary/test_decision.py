import copy
import json
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import initialize, forward, distribution
from pokemon_full_train.learning import prepare_batch, optimize, derived_seed
from pokemon_decision.collect import Pool, target_action, retained_hp, specs_for, evaluation_specs
from pokemon_decision.campaign import learning_gate, summarize
from pokemon_decision.audit import verify_seed_summary


class Decisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.pool=Pool(Path(cls.tmp.name),2)
        cls.p=initialize(9931)
        cls.rows,cls.records=cls.pool.collect(cls.p,specs_for(9931,0,16),17)
    @classmethod
    def tearDownClass(cls): cls.pool.close();cls.tmp.cleanup()

    def test_reward_and_bandit_target_not_natural_terminal(self):
        batch=prepare_batch(self.rows)
        np.testing.assert_allclose(batch['returns'],[r['reward'] for r in self.records],atol=1e-7)
        self.assertTrue(all(r['natural_outcome'] is None for r in self.records))
        self.assertTrue(all(row['next_index']==-1 and row['bootstrap']==0 for row in self.rows))
        for r in self.records:self.assertEqual(r['reward'],retained_hp(r['player'],r['after']))

    def test_actor_logps_bind_sampled_actions_and_only_projected_inputs(self):
        for row,r in zip(self.rows,self.records):
            x,mask=encode(r['player']);logits,value=forward(self.p,jnp.asarray(x[None]))
            _,logs=distribution(logits,jnp.asarray(mask[None]))
            self.assertAlmostEqual(row['old_logp'],float(logs[0,row['action']]),places=5)
            self.assertAlmostEqual(row['value'],float(value[0]),places=6)
            leaked={**r['player'],'context':r['spec']['context']}
            with self.assertRaises(ValueError):encode(leaked)
            changed=copy.deepcopy(r);changed['provenance']['private_target']=3
            np.testing.assert_array_equal(encode(changed['player'])[0],x)

    def test_conflicting_targets_are_not_aliased_by_encoder(self):
        for a in self.records:
            for b in self.records:
                if target_action(a['player'],a['spec']['context']) != target_action(b['player'],b['spec']['context']):
                    self.assertFalse(np.array_equal(encode(a['player'])[0],encode(b['player'])[0]))

    def test_exact_replay_independent_of_pool_assignment(self):
        with tempfile.TemporaryDirectory() as tmp, Pool(Path(tmp),1) as p:
            rows,records=p.collect(self.p,specs_for(9931,0,16),17)
        self.assertEqual(records,self.records)
        for a,b in zip(rows,self.rows):
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_unchanged_ppo_two_valid_padded_steps(self):
        z=jax.tree.map(jnp.zeros_like,self.p)
        p,opt,metrics=optimize(self.p,(z,z,jnp.int32(0)),self.rows,3)
        self.assertEqual(int(opt[2]),2);self.assertEqual(len(metrics),2)
        self.assertTrue(any(not np.array_equal(p[k],self.p[k]) for k in p))
        self.assertTrue(all(np.isfinite(v) for m in metrics for v in m.values()))

    def test_fainted_exposed_member_and_forced_replacement_continue(self):
        spec={'context':8,'panel':'train','episode':0,'run_seed':912}
        before=self.pool.workers[0].request('start',**spec)
        result=self.pool.workers[0].request('decide',action=0)
        exposed=next(p for p in result['frame']['players']['p1']['own'] if p['active'])
        self.assertIn('fnt',exposed['condition'])
        self.assertEqual(retained_hp(before['players']['p1'],result['frame']['players']['p1']),0)
        self.assertEqual(result['reward'],0)
        tail=self.pool.workers[0].request('continue')
        self.assertIn(tail['frame']['outcome']['kind'],['terminal','administrative_cutoff'])

    def test_training_and_variant_coordinates_disjoint_and_balanced(self):
        train=specs_for(7103,0)
        self.assertEqual(len(train),128)
        for c in range(16):self.assertEqual(sum(s['context']==c for s in train),8)
        self.assertNotEqual(train[0]['run_seed'],specs_for(8209,0)[0]['run_seed'])
        self.assertNotEqual(train[0]['run_seed'],evaluation_specs(8622026,8)[0]['run_seed'])
        self.assertEqual(len(evaluation_specs(8622026,8)),256)

    def test_gate_requires_each_context_not_just_mean(self):
        initial={'x':{'optimal_mass':.1}}
        final={'x':{'optimal_mass':.9,'cells':{'0':{'optimal_mass':.59}}}}
        plan={'minimum_probability_gain':.15,'minimum_final_optimal_mass':.6,'minimum_final_per_context':.6}
        self.assertFalse(learning_gate(initial,final,plan)['passed'])
        final['x']['cells']['0']['optimal_mass']=.8
        self.assertTrue(learning_gate(initial,final,plan)['passed'])

    def test_audit_rejects_unlinked_report_counts_and_checkpoint_pointers(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);pointer={'file':'checkpoint.npz','sha256':'abc'}
            for name in ['initial','final']:
                (base/name).mkdir();(base/name/'latest.json').write_text(json.dumps(pointer))
            summary={'optimizer_steps':128,'continuation_outcomes':{'terminal':1},
                'initial_checkpoint':pointer,'final_checkpoint':pointer}
            tails=[{'continuation':{'frame':{'outcome':{'kind':'terminal'}}}}]
            verify_seed_summary(base,summary,128,tails)
            for key,bad in [('optimizer_steps',127),('continuation_outcomes',{'administrative_cutoff':1}),
                            ('initial_checkpoint',{}),('final_checkpoint',{})]:
                with self.assertRaises(ValueError):verify_seed_summary(base,{**summary,key:bad},128,tails)

if __name__=='__main__': unittest.main()
