"""Binary policy with versioned Showdown chance maps; fixed shapes and observation boundary retained."""
import json
import jax
import jax.numpy as jnp
from pokemon_jax.engine import initial, step_kernel, view_kernel, PlayerView
from pokemon_jax.chance import event_key
from pokemon_jax_learner.model import ActorInput
from pokemon_binary.policy import action_kernel
from pokemon_binary.loop import (ROOT, Learner, ActorTape, Targets, Buffer, start, encode,
                                 gae, update, finite_tree, check_report)
from pokemon_learner.core import require
from .chance import sample_kernel


def plan():
    p=json.loads((ROOT/'contracts/showdown-chance-v1/loop.json').read_text())
    require(jax.default_backend()=='cpu' and not jax.config.x64_enabled,'CPU FP32 only')
    require((p['games_per_batch'],p['turns'],p['seats'],p['capacity'],p['features'],p['actions'],
             p['epochs'],p['minibatch'])==(64,4,2,512,59,2,2,128),'Fixed shape plan')
    require(p['maximum_updates']*64<2**32 and p['maximum_updates']*8<2**31,'Counter bounds')
    return p


def collect(parameters, next_game, seed, p):
    ids = next_game + jnp.arange(64, dtype=jnp.uint32)
    master = jax.random.key(seed, impl='threefry2x32')
    env_key = jax.random.fold_in(master, 0x454e56)
    actor_key = jax.random.fold_in(master, 0x414354)
    # initial() performs profile admission once at trace time, not per environment step.
    state = initial(64)
    def turn(state, _):
        valid = jnp.repeat(state.winner == -1, 2)
        views = [view_kernel(state, side) for side in (0, 1)]
        view = PlayerView(*[jnp.stack((views[0][k], views[1][k]), axis=1)
                            .reshape((-1, *views[0][k].shape[1:])) for k in (0, 1)])
        obs = encode(view)
        keys = jax.vmap(lambda game, t: jnp.stack([event_key(actor_key, game, t, side)
                    for side in (0, 1)]))(ids, state.turn.astype(jnp.uint32)).reshape(-1)
        actions, old_logp, values = action_kernel(parameters, obs, keys)
        chance = sample_kernel(env_key, ids, state.turn.astype(jnp.uint32))
        after, _ = step_kernel(state, jnp.where(valid, actions, 0).reshape(64, 2), chance)
        ended = (state.winner == -1) & (after.winner != -1)
        rewards = (ended[:, None] & (after.winner[:, None] == jnp.arange(2))).astype(jnp.float32).reshape(-1)
        actor = ActorTape(jnp.where(valid[:, None], obs.features, 0), obs.mask, actions, old_logp, values,
                          jnp.repeat(ids, 2), jnp.tile(jnp.arange(2, dtype=jnp.int32), 64),
                          jnp.where(valid, jnp.repeat(state.turn, 2), 0))
        return after, (actor, rewards, jnp.repeat(ended, 2), valid)
    state, (actor, rewards, terminal, valid) = jax.lax.scan(turn, state, xs=None, length=4)
    actor = ActorTape(*[x.reshape((-1, *x.shape[2:])) for x in actor])
    rewards, terminal, valid = [x.reshape(-1) for x in (rewards, terminal, valid)]
    advantages, returns = gae(rewards, actor.old_values, terminal, valid, p['gamma'], p['gae_lambda'])
    nxt = jnp.where(valid & ~terminal, jnp.arange(512, dtype=jnp.int32) + 128, -1)
    return Buffer(actor, Targets(rewards, terminal, nxt, advantages, returns), valid), state


def cycle(learner, seed, p):
    buffer, terminal_state = collect(learner.parameters, learner.next_game, seed, p)
    learner, metrics = update(learner, buffer, seed, p)
    a, t, valid = buffer
    legal = jnp.all(jnp.where(valid, (a.actions >= 0) & (a.actions < 2)
                    & jnp.take_along_axis(a.mask, jnp.clip(a.actions, 0, 1)[:, None], axis=1)[:, 0], a.actions == -1))
    # Both player decisions are retained on a KO turn even when the second move cancels.
    report = {'metrics': metrics, 'decisions': jnp.sum(valid),
              'games': jnp.int32(64), 'executed_moves': jnp.sum(terminal_state.count),
              'terminal_games': jnp.sum(terminal_state.winner >= 0),
              'terminal_scores': jnp.sum(t.rewards), 'legal': legal,
              'actor_has_no_outcome': jnp.all(jnp.where(valid[:, None], a.features[:, 1:3] == 0, True)),
              'finite': finite_tree((learner, buffer, metrics)),
              'minimum_behavior_probability': jnp.min(jnp.where(valid, jnp.exp(a.old_logp), 1)),
              'max_link': jnp.max(t.next_index)}
    return learner, report


class Executables:
    """Ahead-of-time compiled functions: wrong shapes/dtypes fail, never retrace.

    Trace counters are independent of JAX's private cache API. They count Python
    tracing of these entry points. The compiled callables cannot compile a new shape.
    """
    def __init__(self, p=None):
        self.plan = plan() if p is None else p
        self.traces = {'collect': 0, 'update': 0, 'cycle': 0}
        self.compiled = {}
        sample = start(0)
        specs = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), sample)
        seed = jax.ShapeDtypeStruct((), jnp.uint32)
        def trace(name, fn):
            def wrapped(*args):
                self.traces[name] += 1
                return fn(*args, self.plan)
            return jax.jit(wrapped)
        self.functions = {name: trace(name, fn) for name, fn in
                          [('collect', collect), ('update', update), ('cycle', cycle)]}
        self.lowered = {}
        self.lowered['cycle'] = self.functions['cycle'].lower(specs, seed)
        self.compiled['cycle'] = self.lowered['cycle'].compile()

    def diagnostics(self):
        """Compile separate phase executables for profiling and differential tests."""
        sample = start(0)
        seed = jnp.uint32(0)
        for name, args in [('collect', (sample.parameters, sample.next_game, seed))]:
            self.lowered[name] = self.functions[name].lower(*args)
            self.compiled[name] = self.lowered[name].compile()
        buffer, _ = self.compiled['collect'](sample.parameters, sample.next_game, seed)
        self.lowered['update'] = self.functions['update'].lower(sample, buffer, seed)
        self.compiled['update'] = self.lowered['update'].compile()

    def __call__(self, learner, seed):
        return self.compiled['cycle'](learner, jnp.asarray(seed, dtype=jnp.uint32))


