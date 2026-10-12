"""Synchronous whole-episode CPU canaries with explicit behavior versions."""
from dataclasses import dataclass
import copy
import numpy as np
from .core import F32, Adam, require, finite, distribution, linked_gae, ppo_loss_grad


@dataclass(frozen=True)
class Buffer:
    version: int
    states: np.ndarray
    mask: np.ndarray
    actions: np.ndarray
    old_logp: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    rewards: np.ndarray
    next_index: np.ndarray
    trajectory: np.ndarray


def sample(logits, mask, rng):
    p, logp = distribution(logits, mask)
    cdf = np.cumsum(p.astype(np.float64), axis=1)
    cdf[:, -1] = 1.
    actions = (rng.random(len(p))[:, None] >= cdf).sum(axis=1).astype(np.int64)
    require(np.all(mask[np.arange(len(p)), actions]), 'Sampled known-forbidden action')
    return actions, logp[np.arange(len(p)), actions].copy()


def shared_outcome_rewards(choices, targets):
    """Sole correct player wins; both correct or both wrong draw. Own score 1/.5/0."""
    correct = np.asarray(choices) == np.asarray(targets)
    require(correct.ndim == 2 and correct.shape[1] == 2, 'Shared outcome shape')
    scores = np.full(correct.shape, F32(.5), dtype=F32)
    different = correct[:, 0] != correct[:, 1]
    scores[different] = correct[different].astype(F32)
    return scores


class Trainer:
    def __init__(self, task, seed, plan):
        require(task in {'bandit', 'delayed'} and seed in plan['seeds'], 'Unknown canary task/seed')
        self.task, self.seed, self.plan = task, seed, copy.deepcopy(plan)
        children = np.random.SeedSequence(seed).spawn(3)
        initial = np.random.Generator(np.random.PCG64(children[0]))
        self.sample_rng = np.random.Generator(np.random.PCG64(children[1]))
        self.shuffle_rng = np.random.Generator(np.random.PCG64(children[2]))
        self.parameters = {'logits': initial.normal(0, .01, (6, 4)).astype(F32), 'values': np.zeros(6, dtype=F32)}
        self.optimizer = Adam(self.parameters, lr=plan['learning_rate'], max_norm=plan['max_gradient_norm'])
        self.version = self.updates = self.decisions = self.episodes = 0
        self.phase = 'idle'

    def collect(self, *, crash_before_linkage=False):
        require(self.phase == 'idle', 'Collect only at a completed-update boundary')
        self.phase = 'collecting'
        size = self.plan[self.task]['episodes_per_update']
        # Frozen copies: no update or environment truth is consulted during collection.
        frozen = {k: v.copy() for k, v in self.parameters.items()}
        if self.task == 'bandit':
            states = np.zeros(size, dtype=np.int64)
            mask = np.ones((size, 4), dtype=bool)
            actions, old_logp = sample(frozen['logits'][states], mask, self.sample_rng)
            rewards = (actions == self.plan['bandit']['rewarding_action']).astype(F32)
            nxt = np.full(size, -1, dtype=np.int64)
            terminal = np.ones(size, dtype=bool)
            trajectory = np.arange(size)
        else:
            # Interleaved decision tape: p1 has two forced decisions while p2 waits.
            # Each episode contains two separate perspective trajectories.
            states = np.tile(np.array([0, 1, 2, 3, 4, 5]), size)
            mask = np.zeros((len(states), 4), dtype=bool)
            root = states < 2
            mask[root] = self.plan['delayed']['root_mask']
            mask[~root, self.plan['delayed']['forced_action']] = True
            actions, old_logp = sample(frozen['logits'][states], mask, self.sample_rng)
            rewards = np.zeros((size, 6), dtype=F32)
            chosen = actions.reshape(size, 6)
            # A single terminal comparison determines both players' own scores.
            scores = shared_outcome_rewards(chosen[:, :2], self.plan['delayed']['rewarding_actions'])
            rewards[:, 3] = scores[:, 0]; rewards[:, 5] = scores[:, 1]
            rewards = rewards.ravel()
            nxt = np.tile(np.array([2, 4, 3, -1, 5, -1]), (size, 1))
            nxt = np.where(nxt >= 0, nxt + 6*np.arange(size)[:, None], -1).ravel()
            terminal = nxt == -1
            trajectory = (2*np.arange(size)[:, None] + np.array([0, 1, 0, 0, 1, 1])).ravel()
        self.decisions += len(states); self.episodes += size
        if crash_before_linkage: raise RuntimeError('Injected collection crash')
        values = frozen['values'][states].copy()
        advantages, returns = linked_gae(rewards, values, nxt, np.zeros(len(states), dtype=F32), terminal, trajectory,
                                         gamma=self.plan['gamma'], lam=self.plan['lambda'])
        self.phase = 'collected'
        arrays = [states, mask, actions, old_logp, values, advantages, returns, rewards, nxt, trajectory]
        for a in arrays: a.flags.writeable = False
        return Buffer(self.version, *arrays)

    def update(self, buffer, *, crash_after_minibatches=None):
        require(self.phase == 'collected' and buffer.version == self.version, 'Stale/reused behavior buffer')
        self.phase = 'updating'
        diagnostics, minibatches = [], 0
        for _ in range(self.plan['epochs']):
            order = self.shuffle_rng.permutation(len(buffer.states))
            for start in range(0, len(order), self.plan['minibatch_size']):
                ix = order[start:start+self.plan['minibatch_size']]; states = buffer.states[ix]
                metrics, gl, gv = ppo_loss_grad(self.parameters['logits'][states], self.parameters['values'][states],
                    buffer.mask[ix], buffer.actions[ix], buffer.old_logp[ix], buffer.advantages[ix], buffer.returns[ix],
                    clip=self.plan['clip'], value_coef=self.plan['value_coefficient'], entropy_coef=self.plan['entropy_coefficient'])
                gradients = {k: np.zeros_like(v) for k, v in self.parameters.items()}
                np.add.at(gradients['logits'], states, gl); np.add.at(gradients['values'], states, gv)
                metrics['gradient_norm_before_clip'] = self.optimizer.apply(self.parameters, gradients)
                diagnostics.append(metrics); minibatches += 1
                if crash_after_minibatches == minibatches: raise RuntimeError('Injected update crash')
        self.updates += 1; self.version += 1; self.phase = 'idle'
        finite(*self.parameters.values())
        return diagnostics

    def probabilities(self):
        if self.task == 'bandit':
            p, _ = distribution(self.parameters['logits'][[0]], np.ones((1, 4), dtype=bool))
            return [float(p[0, self.plan['bandit']['rewarding_action']])]
        p, _ = distribution(self.parameters['logits'][[0, 1]], np.tile(self.plan['delayed']['root_mask'], (2, 1)))
        require(np.all(p[:, 3] == 0), 'Forbidden probability is nonzero')
        return [float(p[i, a]) for i, a in enumerate(self.plan['delayed']['rewarding_actions'])]
