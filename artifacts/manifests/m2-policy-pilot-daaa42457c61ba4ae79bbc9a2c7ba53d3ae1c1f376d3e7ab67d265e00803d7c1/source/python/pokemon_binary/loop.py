"""Versioned binary-policy cycle, derived from the sealed fixed-loop implementation.

Only PlayerView-derived ActorInput enters the frozen actor. The engine and targets
remain outside that API. Time/game/seat padding is explicit; no host compaction,
host linkage, Python minibatch loop or variable-shape numerical call is required.
"""
from typing import NamedTuple
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import initial, step_kernel, view_kernel, PlayerView
from pokemon_jax.chance import event_key, sample_kernel
from pokemon_jax_learner.model import ActorInput, forward, initialize
from pokemon_learner.core import require
from pokemon_fixed.math import adam_step
from .loss import masked_loss
from .policy import action_kernel

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / 'contracts/binary-policy-v1/plan.json'


class Learner(NamedTuple):
    parameters: dict
    m: dict
    v: dict
    step: jax.Array
    version: jax.Array
    next_game: jax.Array


class ActorTape(NamedTuple):
    features: jax.Array
    mask: jax.Array
    actions: jax.Array
    old_logp: jax.Array
    old_values: jax.Array
    game_id: jax.Array
    side: jax.Array
    turn: jax.Array


class Targets(NamedTuple):
    rewards: jax.Array
    terminal: jax.Array
    next_index: jax.Array
    advantages: jax.Array
    returns: jax.Array


class Buffer(NamedTuple):
    actor: ActorTape
    targets: Targets
    valid: jax.Array


def plan():
    p = json.loads(PLAN_PATH.read_text())
    require(jax.default_backend() == 'cpu' and not jax.config.x64_enabled, 'CPU FP32 only')
    require((p['games_per_batch'], p['turns'], p['seats'], p['capacity'], p['features'], p['actions'],
             p['epochs'], p['minibatch']) == (64, 4, 2, 512, 59, 2, 2, 128), 'Fixed plan shape')
    require(p['maximum_updates'] * p['capacity'] > 0 and p['maximum_updates'] * 8 < 2**31
            and p['maximum_updates'] * 64 < 2**32, 'Counter bounds')
    return p


def start(seed):
    parameters = initialize(seed)
    return Learner(parameters, jax.tree.map(jnp.zeros_like, parameters),
                   jax.tree.map(jnp.zeros_like, parameters), jnp.int32(0), jnp.int32(0), jnp.uint32(0))


def finite_tree(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def encode(view):
    """Pure fixed-shape equivalent of the sealed checked PlayerView encoder."""
    n = view.header.shape[0]
    return ActorInput(jnp.concatenate((view.header, view.history.reshape(n, 48)), axis=1)
                      .astype(jnp.float32) / jnp.float32(105), view.header[:, 9:].astype(jnp.bool_))


def gae(rewards, values, terminal, valid, gamma, lam):
    """Reverse scan along time, never across games or players; whole episodes only."""
    shape = (4, 128)
    def one(carry, row):
        next_value, next_adv = carry
        r, v, ended, live = row
        vnext = jnp.where(ended, 0, next_value)
        anext = jnp.where(ended, 0, next_adv)
        advantage = jnp.where(live, r + jnp.float32(gamma) * vnext - v
                              + jnp.float32(gamma * lam) * anext, 0)
        return (jnp.where(live, v, 0), advantage), advantage
    _, advantages = jax.lax.scan(one, (jnp.zeros(128), jnp.zeros(128)),
        tuple(x.reshape(shape) for x in (rewards, values, terminal, valid)), reverse=True)
    advantages = advantages.reshape(-1)
    return advantages, jnp.where(valid, advantages + values, 0)


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


def permutation(seed, version, epoch, valid):
    """Full-capacity permutation then stable valid-first partition, all shapes fixed."""
    key = jax.random.fold_in(jax.random.key(seed, impl='threefry2x32'), 0x534846)
    key = jax.random.fold_in(jax.random.fold_in(key, version), epoch)
    order = jax.random.permutation(key, 512)
    return order[jnp.argsort(~valid[order], stable=True)]


def objective(parameters, actor, targets, valid, p):
    # Sanitization also precedes the model: NaN padding cannot poison weight gradients.
    obs = ActorInput(jnp.where(valid[:, None], actor.features, 0), actor.mask)
    logits, values = forward(parameters, obs)
    return masked_loss(logits, values, actor.mask, actor.actions, actor.old_logp,
                       targets.advantages, targets.returns, valid,
                       p['ppo_clip'], p['value_coefficient'], p['entropy_coefficient'])


def update(learner, buffer, seed, p):
    def epoch(carry, epoch_id):
        order = permutation(seed, learner.version, epoch_id, buffer.valid).reshape(4, 128)
        def minibatch(carry, indices):
            parameters, m, v, step = carry
            sub = jax.tree.map(lambda x: x[indices], buffer)
            (_, metrics), gradients = jax.value_and_grad(lambda pars: objective(
                pars, sub.actor, sub.targets, sub.valid, p), has_aux=True)(parameters)
            count = jnp.sum(sub.valid)
            def apply(_): return adam_step(parameters, gradients, m, v, step,
                                           p['learning_rate'], p['max_gradient_norm'])
            newp, newm, newv, newstep, norm = jax.lax.cond(count > 0, apply,
                lambda _: (parameters, m, v, step, jnp.float32(0)), operand=None)
            metrics = dict(metrics, valid_count=count, gradient_norm=norm,
                           finite=finite_tree((metrics, gradients, newp, newm, newv)),
                           optimizer_applied=count > 0)
            return (newp, newm, newv, newstep), metrics
        return jax.lax.scan(minibatch, carry, order)
    (parameters, m, v, step), metrics = jax.lax.scan(epoch,
        (learner.parameters, learner.m, learner.v, learner.step), jnp.arange(2, dtype=jnp.int32))
    return Learner(parameters, m, v, step, learner.version + 1, learner.next_game + jnp.uint32(64)), metrics


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


def check_report(report):
    """Host boundary: do not publish a new learner/checkpoint after a failed cycle."""
    r = jax.device_get(report)
    require(bool(r['finite']) and bool(np.all(r['metrics']['finite'])), 'Nonfinite cycle')
    require(bool(r['legal']) and bool(r['actor_has_no_outcome']), 'Actor information/legality')
    require(int(r['terminal_games']) == 64 and float(r['terminal_scores']) == 64, 'Incomplete whole episodes')
    # A maximum hit is 58 HP, below 105: every admitted game needs at least two turns.
    require(256 <= int(r['decisions']) <= 512 and int(r['max_link']) < 512, 'Capacity/linkage')
    require(np.all(np.isfinite([np.asarray(x).astype(float).sum() for x in jax.tree.leaves(r)])), 'Nonfinite diagnostics')
    return r
