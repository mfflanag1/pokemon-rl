import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import State, Chance, initial, step, step_kernel, damage, view_kernel, actor_probe
from pokemon_jax.chance import sample, accepted24, finish24, LIMIT24
from pokemon_slice.engine import damage as cpu_damage


class Engine(unittest.TestCase):
    def test_all_damage(self):
        for crit in (False,True):
            for roll in range(85,101):
                self.assertEqual(int(damage(crit,roll)),cpu_damage(crit,roll))

    def test_priority_ko_cancel_and_padding(self):
        s=initial(2);chance=Chance(jnp.ones((2,2),bool),jnp.full((2,2),100),jnp.array([0,1]))
        s,events=step(s,[[0,1],[0,0]],chance)
        self.assertEqual(np.asarray(events)[:,:,0].tolist(),[[1,0],[1,0]])
        s,events=step(s,[[0,1],[0,0]],chance)
        self.assertEqual(s.winner.tolist(),[1,1]);self.assertEqual(s.count.tolist(),[3,3])
        self.assertEqual(events[:,1].tolist(),[[-1]*5]*2)
        self.assertEqual(view_kernel(s,0).header[:,9:].tolist(),[[0,0],[0,0]])
        again,_=step(s,[[1,0],[1,1]],chance)
        for x,y in zip(s,again):np.testing.assert_array_equal(x,y)

    def test_cancel_does_not_reveal_choice(self):
        c=Chance(jnp.ones((1,2),bool),jnp.full((1,2),100),jnp.array([0]))
        s,_=step(initial(1),[[0,0]],c)
        # P1 KOs first either way; P2's alternate queued choice stays unexecuted.
        a,_=step(s,[[1,0]],c);b,_=step(s,[[1,1]],c)
        for side in (0,1):
            for x,y in zip(view_kernel(a,side),view_kernel(b,side)):np.testing.assert_array_equal(x,y)
            for x,y in zip(actor_probe(view_kernel(a,side)),actor_probe(view_kernel(b,side))):np.testing.assert_array_equal(x,y)
        self.assertEqual(view_kernel(a,0).header[0,7:9].tolist(),[1,0])

    def test_invalid_boundary(self):
        s=initial(1);c=sample(1,[0],[1])
        for acts in ([[2,0]],[[-1,0]],[[True,False]],[[0.0,1.0]]):
            with self.assertRaises(ValueError):step(s,acts,c)
        with self.assertRaises(ValueError):initial(1,{'id':'different'})
        with self.assertRaises(ValueError):step(s,[[0,1]],c._replace(roll=jnp.array([[101,90]])))
        with self.assertRaises(ValueError):step(s._replace(hp=jnp.array([[1,105]],jnp.int32)),[[0,1]],c)
        with self.assertRaises(ValueError):actor_probe(s)

    def test_batch_order_partition_and_seed(self):
        ids=np.arange(64);turns=ids%4+1
        a=sample(19,ids,turns);b=sample(19,ids[::-1],turns[::-1])
        halves=[sample(19,ids[k:k+32],turns[k:k+32]) for k in (0,32)]
        for i in range(3):
            np.testing.assert_array_equal(a[i],b[i][::-1]);np.testing.assert_array_equal(a[i],np.concatenate([h[i] for h in halves]))
        self.assertFalse(np.array_equal(a.roll,sample(20,ids,turns).roll))
        with self.assertRaises(ValueError):sample(1,[0,0],[1,1])

    def test_rejection_boundaries_and_branch(self):
        self.assertTrue(bool(accepted24(jnp.uint32(LIMIT24-1))))
        self.assertFalse(bool(accepted24(jnp.uint32(LIMIT24))))
        self.assertFalse(bool(accepted24(jnp.uint32(2**32-1))))
        key=jax.random.key(5,impl='threefry2x32')
        for word in (0,23,LIMIT24-1):self.assertEqual(int(finish24(key,jnp.uint32(word))),word%24)
        _,draw=jax.random.split(key)
        expected=int(jax.random.bits(draw,dtype=jnp.uint32))%24
        self.assertEqual(int(finish24(key,jnp.uint32(LIMIT24))),expected)

    def test_event_streams_are_pure(self):
        ids=np.arange(32);turns=np.ones(32,int)
        before=sample(1,ids,turns)
        sample(1,ids,turns+2);sample(999,ids,turns)
        after=sample(1,ids,turns)
        for x,y in zip(before,after):np.testing.assert_array_equal(x,y)

    def test_unobserved_damage_roll_collision(self):
        # Different underlying chance outcomes can yield the same public history.
        s=initial(1)
        a=Chance(jnp.zeros((1,2),bool),jnp.full((1,2),85),jnp.array([0]))
        b=a._replace(roll=jnp.full((1,2),86))
        sa,ea=step(s,[[0,0]],a);sb,eb=step(s,[[0,0]],b)
        self.assertFalse(np.array_equal(ea,eb))
        for side in (0,1):
            va,vb=view_kernel(sa,side),view_kernel(sb,side)
            for x,y in zip(va,vb):np.testing.assert_array_equal(x,y)
            for x,y in zip(actor_probe(va),actor_probe(vb)):np.testing.assert_array_equal(x,y)

    def test_canceled_chance_is_not_visible(self):
        c=Chance(jnp.ones((1,2),bool),jnp.full((1,2),100),jnp.array([0]))
        s,_=step(initial(1),[[0,0]],c)
        a,_=step(s,[[1,0]],c)
        c=c._replace(crit=jnp.array([[True,False]]),roll=jnp.array([[100,85]]))
        b,_=step(s,[[1,0]],c)
        for side in (0,1):
            for x,y in zip(view_kernel(a,side),view_kernel(b,side)):np.testing.assert_array_equal(x,y)

    def test_compiled_rollout_all_terminal(self):
        from pokemon_jax.campaign import rollout
        from pokemon_jax.engine import validate_state
        ids=jnp.arange(1024,dtype=jnp.uint32)
        s=jax.jit(rollout)(jax.random.key(7,impl='threefry2x32'),ids)
        validate_state(s)
        self.assertTrue(np.all(np.asarray(s.winner)>=0))

    def test_exact_ko_threshold(self):
        c=Chance(jnp.zeros((1,2),bool),jnp.full((1,2),95),jnp.array([0]))
        s=initial(1)
        for _ in range(2):s,_=step(s,[[0,0]],c)
        self.assertEqual(s.hp.tolist(),[[33,33]])
        s,events=step(s,[[0,0]],c._replace(roll=jnp.full((1,2),85)))
        self.assertEqual(s.hp.tolist(),[[33,0]]);self.assertEqual(s.turn.tolist(),[3])
        self.assertEqual(s.pp.tolist(),[[[53,48],[54,48]]])
        self.assertEqual(events[:,1].tolist(),[[-1]*5])

    def test_mixed_terminal_batch_isolation(self):
        c=Chance(jnp.ones((1,2),bool),jnp.full((1,2),100),jnp.array([0]))
        dead=initial(1)
        for _ in range(2):dead,_=step(dead,[[0,0]],c)
        live=initial(1)
        mixed=jax.tree.map(lambda x,y:jnp.concatenate([x,y]),dead,live)
        cc=jax.tree.map(lambda x:jnp.concatenate([x,x]),c)
        result,_=step(mixed,[[1,1],[0,1]],cc)
        solo,_=step(live,[[0,1]],c)
        for original,batched,single in zip(dead,result,solo):
            np.testing.assert_array_equal(original[0],batched[0]);np.testing.assert_array_equal(single[0],batched[1])

    def test_reject_redistributed_history_turns(self):
        from pokemon_jax.engine import validate_state
        c=Chance(jnp.zeros((1,2),bool),jnp.full((1,2),85),jnp.array([0]))
        s=initial(1)
        for _ in range(2):s,_=step(s,[[0,0]],c)
        malformed=s._replace(history=s.history.at[0,1,0].set(2))
        with self.assertRaises(ValueError):validate_state(malformed)
        with self.assertRaises(ValueError):step(malformed,[[0,0]],c)
