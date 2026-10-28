"""Concurrent CPU simulators with fixed-shape, player-only JAX inference.

Policy weights remain fixed throughout collect(). Results are returned in input
order; worker assignment never contributes to RNG keys or trajectory identities.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode, FEATURES
from pokemon_full.policy import act
from pokemon_full.learner import finish_rows
from pokemon_full_train.learning import SIDES, baseline, derived_seed
from pokemon_full_train.transport import Worker


@jax.jit
def infer(parameters, features, masks, seeds, counters):
    roots = jax.vmap(jax.random.key)(seeds)
    keys = jax.vmap(jax.random.fold_in)(roots, counters)
    return act(parameters, features, masks, keys)


def batched_actor(parameters, players, seeds, counters, capacity):
    """The only observation interface: projected players, never coordinator frames."""
    n = len(players)
    if not 0 < n <= capacity or len(seeds) != n or len(counters) != n:
        raise ValueError('Invalid actor batch')
    features = np.zeros((capacity, FEATURES), np.float32)
    masks = np.zeros((capacity, 14), bool)
    root_seeds = np.zeros(capacity, np.uint32)
    counts = np.zeros(capacity, np.uint32)
    for i, player in enumerate(players):
        features[i], masks[i] = encode(player)
        root_seeds[i], counts[i] = seeds[i], counters[i]
    actions, logs, values = map(np.asarray, infer(parameters, features, masks, root_seeds, counts))
    if not np.isfinite(logs).all() or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid batched actor output')
    for i in range(n):
        if masks[i].any():
            if not 0 <= actions[i] < 14 or not masks[i, actions[i]]:
                raise ValueError('Illegal batched actor action')
        elif actions[i] != -1 or logs[i] != 0:
            raise ValueError('Invalid waiting actor output')
    return [(features[i].copy(), masks[i].copy(), int(actions[i]), float(logs[i]), float(values[i]))
            for i in range(n)]


class Episode:
    def __init__(self, *, seed, episode, max_turns, split='train', swap=False,
                 opponent=None, learner_side='p1', actor_seed=None):
        if split not in ['train', 'development'] or (split == 'development' and opponent is None):
            raise ValueError('Invalid collection role')
        if split == 'train' and (opponent is not None or swap):
            raise ValueError('Training profile is official self-play')
        if learner_side not in SIDES or opponent not in [None, 'random', 'base_power']:
            raise ValueError('Invalid opponent/seat')
        if type(max_turns) is not int or not 2 <= max_turns <= 240:
            raise ValueError('Invalid turn boundary')
        self.start = dict(run_seed=seed, episode_index=episode, split=split, swap=swap)
        self.episode, self.split, self.swap = episode, split, swap
        self.max_turns, self.opponent, self.learner_side = max_turns, opponent, learner_side
        actor_seed = seed if actor_seed is None else actor_seed
        self.seeds = [derived_seed('actor', actor_seed, episode, swap, s) for s in SIDES]
        self.rngs = [np.random.default_rng(derived_seed('opponent', actor_seed, episode, swap, s)) for s in SIDES]
        self.counts, self.last = [0, 0], {s: None for s in SIDES}
        self.rows, self.trace, self.values, self.phases = [], [], [], Counter()
        self.decisions = self.rejections = 0
        self.frame = self.provenance = self.result = None
        self.bootstrap = None

    def receive(self, op, frame):
        self.frame = frame
        if op == 'start':
            self.provenance = frame['provenance']
            if any(t['partition'] != self.split for t in self.provenance['teams']):
                raise ValueError('Worker partition mismatch')
        elif op == 'step':
            self.rejections += sum(not r['accepted'] for r in frame['submitted'])
        elif op == 'cutoff':
            if frame['outcome']['kind'] != 'administrative_cutoff':
                raise ValueError('Wrong cutoff boundary')
            self.finish(self.bootstrap)

    def finish(self, values):
        outcome = self.frame['outcome']
        finish_rows(self.rows, self.last, outcome, values)
        terminal = outcome['kind'] == 'terminal'
        self.result = (self.rows, {'episode': self.episode, 'split': self.split, 'swap': self.swap,
            'learner_side': self.learner_side, 'opponent': self.opponent, 'provenance': self.provenance,
            'outcome': outcome, 'decisions': self.decisions, 'phases': dict(self.phases),
            'unavailable_choices': self.rejections, 'trace': self.trace,
            'value_mean': float(np.mean([v for _, v in self.values])) if self.values else None,
            'terminal_value_mse': float(np.mean([(v-outcome['rewards'][s])**2 for s,v in self.values]))
                if terminal and self.values else None})

    def advance(self, outputs):
        if self.frame['outcome'] is not None:
            self.finish([o[4] for o in outputs])
            return None
        joint = all(self.frame['players'][s]['phase'] == 'move' and outputs[i][1].any()
                    for i,s in enumerate(SIDES))
        if joint and self.frame['players']['p1']['turn'] >= self.max_turns:
            self.bootstrap = [o[4] for o in outputs]
            return 'cutoff', {}
        if self.decisions >= 10000:
            raise RuntimeError('Episode decision cap: no incomplete trajectory enters an update')
        submission, ids = {}, {}
        for i, side in enumerate(SIDES):
            x, mask, action, logp, value = outputs[i]
            if not mask.any():
                continue
            controlled = self.opponent is None or side == self.learner_side
            if not controlled:
                action = baseline(self.frame['players'][side], self.opponent, self.rngs[i])
            if self.split == 'train':
                index = len(self.rows)
                if self.last[side] is not None:
                    self.rows[self.last[side]]['next_index'] = index
                self.rows.append({'features':x, 'mask':mask, 'action':action, 'old_logp':logp,
                    'value':value, 'reward':0., 'next_index':-1, 'bootstrap':0.,
                    'terminal':False, 'trajectory':i})
                self.last[side] = index
            if controlled:
                self.values.append((side,value))
            player = self.frame['players'][side]
            self.trace.append([side,player['request_id'],action,value if controlled else None,
                               hashlib.sha256(x.tobytes()).hexdigest()])
            submission[side], ids[side] = action, player['request_id']
            self.counts[i] += 1
            self.decisions += 1
            self.phases[player['phase']] += 1
        if not submission:
            raise RuntimeError('Nonterminal frame has no actionable player')
        return 'step', dict(actions=submission, request_ids=ids)


class Pool:
    def __init__(self, evidence, workers):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Expected 1..16 workers')
        self.workers, self.closed = [], False
        self.executor = ThreadPoolExecutor(max_workers=workers)
        try:
            for i in range(workers):
                self.workers.append(Worker(Path(evidence)/f'worker-{i}'))
        except BaseException:
            self.close()
            raise

    def collect(self, parameters, specs):
        if self.closed or not specs:
            raise ValueError('Closed pool or empty collection')
        # Validate all jobs before starting any simulator request.
        episodes = [Episode(**spec) for spec in specs]
        slots, next_job, results = {}, 0, [None]*len(specs)
        profile = Counter()
        started = time.perf_counter()
        try:
            while next_job < len(episodes) or slots:
                commands = []
                for slot in range(len(self.workers)):
                    if slot not in slots and next_job < len(episodes):
                        slots[slot] = next_job
                        commands.append((slot, 'start', episodes[next_job].start))
                        next_job += 1
                self._requests(commands, slots, episodes, profile)
                players, seeds, counters = [], [], []
                order = sorted(slots)
                for slot in order:
                    ep = episodes[slots[slot]]
                    players.extend(ep.frame['players'][s] for s in SIDES)
                    seeds.extend(ep.seeds); counters.extend(ep.counts)
                t0 = time.perf_counter()
                outputs = batched_actor(parameters, players, seeds, counters, 2*len(self.workers))
                profile['encode_infer_sync_seconds'] += time.perf_counter()-t0
                profile['inference_calls'] += 1
                profile['active_actor_rows'] += len(players)
                profile['padded_actor_rows'] += 2*len(self.workers)-len(players)
                commands = []
                t0 = time.perf_counter()
                for i, slot in enumerate(order):
                    ep = episodes[slots[slot]]
                    command = ep.advance(outputs[2*i:2*i+2])
                    if command:
                        commands.append((slot, *command))
                profile['coordinator_seconds'] += time.perf_counter()-t0
                self._requests(commands, slots, episodes, profile)
                for slot in order:
                    index = slots[slot]
                    if episodes[index].result is not None:
                        results[index] = episodes[index].result
                        del slots[slot]
            profile['collection_seconds'] = time.perf_counter()-started
            return results, dict(profile)
        except BaseException:
            # Never return a partial cohort. Bounded requests retain their own failure evidence.
            self.close()
            raise

    def _requests(self, commands, slots, episodes, profile):
        if not commands:
            return
        t0 = time.perf_counter()
        futures = [(slot,op,self.executor.submit(self.workers[slot].request,op,**payload))
                   for slot,op,payload in commands]
        for slot,op,future in futures:
            episodes[slots[slot]].receive(op,future.result())
        profile['worker_barrier_seconds'] += time.perf_counter()-t0
        profile['worker_requests'] += len(commands)

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            for worker in self.workers:
                worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def merge_rows(results):
    """Episode-major order, independent of worker completion order."""
    rows = []
    for ordinal, (new, _) in enumerate(results):
        offset = len(rows)
        for row in new:
            rows.append({**row, 'next_index': row['next_index']+offset if row['next_index'] >= 0 else -1,
                         'trajectory': 2*ordinal+row['trajectory']})
    return rows
