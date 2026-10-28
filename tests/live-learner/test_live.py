import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
import json
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import engine, carry
from pokemon_switch_engine import learner as old, policy as old_policy
from pokemon_learner.checkpoint import digest, wire
from pokemon_live import learner as l, policy as p, checkpoint as cp, audit, reference


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = l.start(431); cls.results = []; state = cls.initial
        for _ in range(6):
            r = l.transaction(state, 431); cls.results.append(r); state = r[0]

    def test_all_windows_scalar_observation_and_state_parity(self):
        previous = self.initial; totals = dict(post12_decisions=0, history_over32_decisions=0, forced_waits=0)
        for i, result in enumerate(self.results):
            candidate, buffer, final, _ = result
            before = previous.environment if i % 5 else engine.initial_kernel(l.teams(candidate.ids, candidate.seed))
            result = audit.collection(before, candidate.ids, 431, l.PLAN['boundaries'][i % 5], buffer, final)
            for k in totals: totals[k] += result[k]
            previous = candidate
        self.assertTrue(all(x > 0 for x in totals.values()), totals)

    def test_encoder_order_dtype_shape_and_private_boundary(self):
        view = engine.view_kernel(self.initial.environment, 0); encoded = p.encode(view)
        expected = np.concatenate([np.asarray(x).reshape(l.BATCH, -1) for x in view[:-1]], 1).astype(np.float32) / 128
        np.testing.assert_array_equal(expected, encoded.features)
        self.assertEqual(encoded.features.shape, (32, 1187))
        np.testing.assert_array_equal(encoded.features[:, -9:], -np.ones((32, 9), np.float32) / 128)
        for value in (self.initial.environment, view._asdict(), tuple(view)):
            with self.assertRaises(ValueError): p.encode(value)
        with self.assertRaises(ValueError): p.forward(self.initial.parameters, old_policy.ActorInput(encoded.features, encoded.mask))
        with self.assertRaises(ValueError): p.encode(view._replace(history=view.history[:, :32]))

    def test_nonvacuous_hidden_set_and_public_sensitivity(self):
        s = self.initial.environment; changed = s._replace(swift=s.swift.at[:, 1, :].set(1-s.swift[:, 1, :]),
            pp=s.pp.at[:, 1, :, :].add(-1), hp=s.hp.at[:, 1, 1].add(-7),
            species=s.species.at[:, 1, 1].set((s.species[:, 1, 1]+1)%3))
        self.assertNotEqual(cp.fingerprint(s), cp.fingerprint(changed))
        a = p.encode(engine.view_kernel(s, 0)); b = p.encode(engine.view_kernel(changed, 0))
        self.assertEqual(cp.fingerprint(a), cp.fingerprint(b))
        params = dict(self.initial.parameters); params['vw'] = jnp.ones(32) * .01
        self.assertEqual(cp.fingerprint(p.forward(params, a)), cp.fingerprint(p.forward(params, b)))
        public = p.encode(engine.view_kernel(s._replace(hp=s.hp.at[:, 0, 0].add(-10)), 0))
        self.assertNotEqual(cp.fingerprint(a), cp.fingerprint(public))
        self.assertNotEqual(cp.fingerprint(p.forward(params, a)), cp.fingerprint(p.forward(params, public)))
        # Repeat in the opposite seat with its opponent's hidden PP/swift changed.
        other = s._replace(swift=s.swift.at[:, 0, :].set(1-s.swift[:, 0, :]), pp=s.pp.at[:, 0, :, :].add(-1))
        self.assertEqual(cp.fingerprint(p.encode(engine.view_kernel(s, 1))), cp.fingerprint(p.encode(engine.view_kernel(other, 1))))

    def test_late_history_changes_actor_and_value(self):
        view = engine.view_kernel(self.results[2][0].environment, 0)
        self.assertGreater(int(view.context[:, 4].max()), 32)
        changed = view._replace(history=view.history.at[:, 50, 7].add(17))
        params = dict(self.initial.parameters); params['vw'] = jnp.ones(32) * .01
        original = p.forward(params, p.encode(view)); modified = p.forward(params, p.encode(changed))
        self.assertFalse(np.array_equal(original[0], modified[0]))
        self.assertFalse(np.array_equal(original[1], modified[1]))

    def test_batch_permutation_preserves_addresses_and_returns(self):
        state = self.results[0][0]; order = np.arange(31,-1,-1)
        permuted = jax.tree.map(lambda x: x[order], state.environment)
        a, af = l.collect(state.parameters, state.environment, state.ids, state.seed, jnp.int32(25))
        b, bf = l.collect(state.parameters, permuted, state.ids[order], state.seed, jnp.int32(25))
        for name in a._fields:
            if name == 'next_index': continue
            av = np.asarray(getattr(a,name)); bv = np.asarray(getattr(b,name))
            np.testing.assert_array_equal(av.reshape(l.GROUPS,32,2,*av.shape[1:])[:,order], bv.reshape(l.GROUPS,32,2,*bv.shape[1:]), err_msg=name)
        self.assertEqual(cp.fingerprint(jax.tree.map(lambda x:x[order],af)), cp.fingerprint(bf))

    def test_empty_checkpoint_next_update_and_refill(self):
        state = self.results[0][0]; env = state.environment
        hp = np.asarray(env.hp).copy(); hp[:,1] = 0; hp[:,0] = np.maximum(hp[:,0],1)
        state = state._replace(environment=env._replace(hp=jnp.array(hp),winner=jnp.zeros(32,jnp.int32),phase=jnp.full(32,2,jnp.int32)))
        original_optimizer = cp.fingerprint(state[:4])
        for _ in range(4):
            state = l.transaction(state,431)[0]
            self.assertEqual(cp.fingerprint(state[:4]),original_optimizer)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'empty.json'; cp.save(path,state,{'unit':1}); loaded=cp.load(path,431,{'unit':1})
            self.assertEqual(cp.fingerprint(l.transaction(state,431)),cp.fingerprint(l.transaction(loaded,431)))
        refilled=l.transaction(state,431)[0]
        self.assertEqual(int(refilled.next_game),64); self.assertEqual(int(refilled.updates),int(state.updates)+1)

    def test_cohort_refill_and_single_terminal_reward(self):
        total = 0; previous = self.initial
        for i, (candidate, buffer, final, _) in enumerate(self.results):
            np.testing.assert_array_equal(candidate.ids, np.arange(32, dtype=np.uint32) + (32 if i == 5 else 0))
            self.assertEqual(int(candidate.next_game), 64 if i == 5 else 32)
            if i < 5: total += int(buffer.rewards.sum())
            previous = candidate
        self.assertEqual(total, int((self.results[4][2].winner != -1).sum()))
        self.assertGreater(int(self.results[4][1].terminal.sum()) + int(self.results[3][1].terminal.sum()) + int(self.results[2][1].terminal.sum()) + int(self.results[1][1].terminal.sum()), 0)

    def test_each_boundary_exact_next_update_checkpoint(self):
        admitted = {'test_identity': 'unit-test'}
        for s in [self.initial] + [r[0] for r in self.results[:5]]:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'cp.json'; cp.save(path, s, admitted); loaded = cp.load(path, 431, admitted)
                self.assertEqual(cp.fingerprint(loaded), cp.fingerprint(s))
                self.assertEqual(cp.fingerprint(l.transaction(s, 431)), cp.fingerprint(l.transaction(loaded, 431)))
                with self.assertRaises(FileExistsError): cp.save(path, s, admitted)

    def test_empty_window_does_not_decay_adam_or_duplicate_rewards(self):
        s = self.results[0][0]; final = self.results[0][2]
        ended = np.flatnonzero(np.asarray(final.winner) != -1)
        self.assertGreater(len(ended), 0)
        # Synthetic validator-admitted terminal cohort tests the empty branch with
        # nonzero Adam moments. Reachability is covered by the scalar replay test.
        env = s.environment
        hp = np.asarray(env.hp).copy(); hp[:, 1] = 0; hp[:, 0] = np.maximum(hp[:, 0], 1)
        env = env._replace(hp=jnp.asarray(hp), winner=jnp.zeros(32, jnp.int32), phase=jnp.full(32, 2, jnp.int32))
        s = s._replace(environment=env); cp.validate(s)
        result = l.transaction(s, 431); after, buffer, _, metrics = result
        self.assertFalse(bool(buffer.valid.any())); self.assertEqual(float(buffer.rewards.sum()), 0)
        self.assertEqual(cp.fingerprint(s[:4]), cp.fingerprint(after[:4]))
        self.assertEqual(int(after.updates), int(s.updates)); self.assertEqual(int(after.windows), int(s.windows) + 1)
        self.assertEqual(cp.fingerprint(env), cp.fingerprint(after.environment))
        self.assertTrue(all(np.all(np.asarray(x) == 0) for x in metrics.values()))

    def test_rollback_and_seed_rejection(self):
        s = self.results[0][0]; before = cp.fingerprint(s)
        for fault in ('collect', 'update'):
            with self.assertRaises(RuntimeError): l.transaction(s, 431, fault=fault)
            self.assertEqual(before, cp.fingerprint(s))
        with self.assertRaises(ValueError): l.transaction(s, 641)

    def test_endpoint_is_audit_only_and_not_a_production_checkpoint(self):
        state=self.results[-1][0]
        while int(state.windows)<10: state=l.transaction(state,431)[0]
        with self.assertRaises(ValueError):l.transaction(state,431)
        endpoint=l.transaction(state,431,audit_only=True)[0]
        self.assertEqual(int(endpoint.windows),11)
        with self.assertRaises(ValueError):cp.validate(endpoint)
        with self.assertRaises(ValueError):l.transaction(endpoint,431,audit_only=True)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):cp.save(Path(td)/'bad.json',endpoint,{'unit':1})

    def test_checkpoint_negatives_rehashed(self):
        s = self.results[0][0]; admitted = {'identity': 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'good.json'; cp.save(path, s, admitted); doc = json.loads(path.read_text())
            mutations = [lambda b: b.update(format='bounded-value-checkpoint-1'),
                lambda b: b.update(partial_buffer=[]), lambda b: b['counters'].__setitem__(1, 0),
                lambda b: b['counters'].__setitem__(0, 3), lambda b: b['ids'].__setitem__(0, 1),
                lambda b: b['environment']['history'].update(shape=[32, 32, 9]),
                lambda b: b['counters'].__setitem__(4, 641), lambda b: b.update(admission_hash='wrong')]
            for i, mutate in enumerate(mutations):
                copy = json.loads(wire(doc)); mutate(copy['body']); copy['sha256'] = digest(copy['body'])
                bad = Path(td) / f'bad-{i}.json'; bad.write_bytes(wire(copy))
                with self.assertRaises((ValueError, TypeError)): cp.load(bad, 431, admitted)
            path.write_bytes(path.read_bytes()[:100])
            with self.assertRaises(ValueError): cp.load(path, 431, admitted)

    def test_live_administrative_cutoff_bootstraps_and_then_refills(self):
        state = self.initial
        params = dict(state.parameters); params['pb'] = jnp.array([-100.,-100.,100.,100.],jnp.float32)
        env = state.environment
        for boundary in (13,25,37,49):
            buffer, final = l.collect(params,env,state.ids,state.seed,jnp.int32(boundary))
            l.validate_buffer(buffer,env,final,boundary,params); env = final
        self.assertTrue(np.all(np.asarray(env.winner)==-1))
        state = state._replace(parameters=params,environment=env,windows=jnp.int32(4))
        result = l.transaction(state,431); after,buffer,final,_ = result
        self.assertTrue(np.all(np.asarray(final.turn)==61)); self.assertTrue(np.all(np.asarray(final.phase)==0))
        self.assertTrue(np.all(np.asarray(final.winner)==-1)); self.assertEqual(int(buffer.terminal.sum()),0)
        last = np.asarray(buffer.valid)&(np.asarray(buffer.next_index)==-1)
        self.assertEqual(int(last.sum()),64); np.testing.assert_array_equal(np.asarray(buffer.bootstrap)[last],np.full(64,.5))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cutoff.json';cp.save(path,after,{'unit':1});loaded=cp.load(path,431,{'unit':1})
            self.assertEqual(cp.fingerprint(l.transaction(after,431)),cp.fingerprint(l.transaction(loaded,431)))
        refilled = l.transaction(after,431)[0]
        self.assertEqual(int(refilled.next_game),64); np.testing.assert_array_equal(refilled.ids,np.arange(32,64,dtype=np.uint32))
        self.assertLessEqual(int(refilled.environment.turn.max()),13)

    def test_wait_gap_gae_hand_calculation(self):
        values = jnp.array([[.2,.3],[0,.4],[.5,0]], jnp.float32)
        valid = jnp.array([[1,1],[0,1],[1,0]], bool)
        reward, terminal, nxt, boot, advantages, returns = old.targets(values, valid, jnp.array([0]), jnp.zeros(2))
        np.testing.assert_array_equal(nxt, [[4,3],[-1,-1],[-1,-1]])
        np.testing.assert_array_equal(reward, [[0,0],[0,0],[1,0]])
        np.testing.assert_allclose(advantages, [[.775,-.28],[0,-.4],[.5,0]], atol=1e-7)
        np.testing.assert_allclose(returns, [[.975,.02],[0,0],[1,0]], atol=1e-7)
        np.testing.assert_array_equal(boot, np.zeros((3,2)))

    def test_numpy_network_loss_and_gradient_parity(self):
        rng = np.random.default_rng(193); params = {k: np.asarray(rng.normal(0,.05,shape),np.float32) for k,shape in p.SHAPES.items()}
        x = rng.normal(0,.1,(11,1187)).astype(np.float32); mask = rng.random((11,4))>.4; mask[:,0] = True
        valid = np.ones(11,bool); valid[-2:] = False; actions = np.zeros(11,np.int32)
        logits, values = reference.forward(params,x); oldlog = np.asarray(old_policy.distribution(jnp.array(logits),jnp.array(mask))[1][:,0]) + np.linspace(-.4,.4,11,dtype=np.float32)
        adv = np.linspace(-.7,.6,11,dtype=np.float32); ret = np.linspace(0,1,11,dtype=np.float32)
        metrics, grad = reference.loss_grad(params,x,mask,actions,oldlog,adv,ret,valid)
        x[~valid] = np.nan; oldlog[~valid] = np.nan; adv[~valid] = np.nan; ret[~valid] = np.nan; actions[~valid] = -1; mask[~valid] = False
        buffer = l.Buffer(jnp.array(x),jnp.array(mask),jnp.array(actions),jnp.array(oldlog),jnp.array(values),jnp.array(valid),
            jnp.zeros(11),jnp.zeros(11,bool),jnp.full(11,-1,jnp.int32),jnp.zeros(11),jnp.array(adv),jnp.array(ret))
        (_, jm), jg = jax.value_and_grad(p.objective,has_aux=True)(jax.tree.map(jnp.asarray,params),buffer)
        for k in jm: self.assertAlmostEqual(float(jm[k]),metrics[k],delta=2e-6,msg=k)
        for k in jg: np.testing.assert_allclose(jg[k],grad[k],atol=2e-6,rtol=5e-5,err_msg=k)
        self.assertGreater(np.linalg.norm(grad['w']),1e-4); self.assertGreater(np.linalg.norm(grad['vw']),1e-4)


if __name__ == '__main__': unittest.main()
