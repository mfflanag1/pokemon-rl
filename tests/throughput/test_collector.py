import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import initialize
from pokemon_full_train.learning import collect, actor, derived_seed, prepare_batch
from pokemon_full_train.transport import Worker
from pokemon_throughput.collector import Pool, Episode, batched_actor, merge_rows
from pokemon_throughput.benchmark import compare_results, compare_update

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('f1',ROOT/'tests/full-showdown/test_learner.py')
f1=importlib.util.module_from_spec(spec);spec.loader.exec_module(f1)


class ActorTests(unittest.TestCase):
    def test_batch_permutation_padding_and_player_isolation(self):
        p=initialize(73);p['vw']=jnp.linspace(-.1,.1,64)
        players=[f1.player() for _ in range(3)]
        players[1]['mask']=[False]*14
        players[2]['mask']=[False]*14;players[2]['mask'][13]=True
        seeds=[31,32,33];counts=[7,8,9]
        a=batched_actor(p,players,seeds,counts,8)
        b=batched_actor(p,players[::-1],seeds[::-1],counts[::-1],8)[::-1]
        for i in range(3):
            c=actor(p,players[i],jax.random.fold_in(jax.random.key(seeds[i]),counts[i]))
            for x,y in [(a[i],b[i]),(a[i],c)]:
                np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
                self.assertEqual(x[2],y[2]);np.testing.assert_allclose(x[3:],y[3:],atol=3e-6)
        players[0]['hidden_state']='forbidden'
        with self.assertRaisesRegex(ValueError,'projection'):
            batched_actor(p,players,seeds,counts,8)

    def test_final_role_refused_before_requests(self):
        with self.assertRaises(ValueError): Episode(seed=1,episode=0,max_turns=2,split='final')


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=initialize(73);cls.p['vw']=jnp.linspace(-.04,.04,64)
        cls.jobs=[dict(seed=73,episode=i,max_turns=12) for i in range(3)]
        cls.jobs += [dict(seed=8412026,episode=0,max_turns=12,split='development',opponent=o,
                         swap=s,learner_side='p2' if s else 'p1') for o in ['random','base_power'] for s in [False,True]]

    def test_matched_serial_parallel_and_update_with_worker_delays(self):
        with tempfile.TemporaryDirectory() as d:
            with Worker(Path(d)/'serial') as w:
                serial=[collect(w,self.p,**job) for job in self.jobs]
            original=Worker.request
            def delayed(worker,op,**kwargs):
                if op=='step' and worker.sequence%3==0: time.sleep(.002)
                return original(worker,op,**kwargs)
            with patch.object(Worker,'request',delayed),Pool(Path(d)/'parallel',2) as pool:
                parallel,profile=pool.collect(self.p,self.jobs)
            compare_results(serial,parallel)
            self.assertGreater(profile['padded_actor_rows'],0)
            self.assertTrue(all(not rows for rows,_ in parallel[3:]))
            rows=merge_rows(parallel)
            prepare_batch(rows)
            for i,r in enumerate(rows):
                if r['next_index']>=0:
                    self.assertGreater(r['next_index'],i)
                    self.assertEqual(rows[r['next_index']]['trajectory'],r['trajectory'])
            compare_update(self.p,serial,parallel)

    def test_failure_aborts_entire_pool_and_no_partial_result(self):
        original=Worker.request
        def fail(worker,op,**kwargs):
            if op=='step':
                # Real worker error exercises its evidence and child cleanup.
                return original(worker,'unknown-operation')
            return original(worker,op,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            pool=Pool(d,2)
            with patch.object(Worker,'request',fail),self.assertRaises(RuntimeError):
                pool.collect(self.p,self.jobs[:2])
            self.assertTrue(pool.closed)
            self.assertTrue(all(w.process.poll() is not None for w in pool.workers))
            self.assertTrue(list(Path(d).rglob('failure-*.json')))

    def test_rejected_choice_is_retained_and_same_side_linked(self):
        import subprocess,json
        tape=json.loads(subprocess.check_output(['node','tests/full-showdown/trapped-fixture.mjs'],cwd=ROOT,text=True))
        ep=Episode(seed=73,episode=0,max_turns=2)
        ep.receive('start',copy.deepcopy(tape['initial']))
        choices=iter([9,0,0,-1,0,0])
        for step in range(3):
            outputs=[]
            for side in ['p1','p2']:
                x,mask=encode(ep.frame['players'][side]);a=next(choices)
                outputs.append((x,mask,a,-float(np.log(max(mask.sum(),1))),.5))
            op,payload=ep.advance(outputs)
            if step<2:
                self.assertEqual(payload,tape['commands'][step])
                ep.receive(op,copy.deepcopy(tape[['rejected','continued'][step]]))
            else:
                self.assertEqual(op,'cutoff');ep.receive(op,copy.deepcopy(tape['cutoff']))
        rows,report=ep.result
        self.assertEqual(report['unavailable_choices'],1)
        self.assertEqual([r['action'] for r in rows],[9,0,0])
        self.assertEqual([r['next_index'] for r in rows],[2,-1,-1])
        np.testing.assert_allclose(prepare_batch(rows)['returns'],.5,atol=1e-7)


if __name__=='__main__': unittest.main()
