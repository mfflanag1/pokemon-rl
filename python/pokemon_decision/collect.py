from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pokemon_full.features import encode, hp_fraction
from pokemon_full_train.learning import derived_seed
from pokemon_throughput.collector import batched_actor
from .transport import Worker


def retained_hp(before, after):
    exposed = [p for p in after['own'] if p['active']]
    if len(exposed) != 1:
        raise ValueError('Expected one exposed member')
    old = next(p for p in before['own'] if p['ident'] == exposed[0]['ident'])
    prior = hp_fraction(old['condition'])[0]
    if prior <= 0:
        raise ValueError('Exposed member was already fainted')
    return min(1., max(0., hp_fraction(exposed[0]['condition'])[0] / prior))


def target_action(player, context):
    """Evaluation-only semantic expectation; never sent to actor or PPO."""
    if context < 8:
        target = 'Dugtrio' if context % 8 < 4 else 'Vaporeon'
        return 8 + next(i for i, p in enumerate(player['own']) if p['species'] == target)
    return 4 if context % 8 < 4 else 0


def diagnostic_row(player, output, reward):
    x, mask, action, logp, value = output
    return {'features': x, 'mask': mask, 'action': action, 'old_logp': logp,
        'value': value, 'reward': reward, 'next_index': -1, 'bootstrap': 0.,
        'terminal': True, 'trajectory': 0}


class Pool:
    def __init__(self, evidence, count=8):
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError('Expected 1..8 workers')
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=count)
        try:
            for i in range(count):
                self.workers.append(Worker(evidence / f'worker-{i}'))
        except BaseException:
            self.close()
            raise

    def requests(self, op, payloads):
        if not 0 < len(payloads) <= len(self.workers):
            raise ValueError('Invalid request batch')
        futures = [self.executor.submit(w.request, op, **p) for w, p in zip(self.workers, payloads)]
        return [f.result() for f in futures]

    def collect(self, parameters, specs, actor_seed, continuation=False):
        if not specs:
            raise ValueError('Empty diagnostic cohort')
        rows, records = [], []
        for offset in range(0, len(specs), len(self.workers)):
            chunk = specs[offset:offset+len(self.workers)]
            frames = self.requests('start', chunk)
            sides = ['p2' if spec['context'] % 2 else 'p1' for spec in chunk]
            players = [f['players'][s] for f, s in zip(frames, sides)]
            seeds = [derived_seed('f6-actor', actor_seed, spec) for spec in chunk]
            # Fixed inference shape also during one-worker replay. Changing the
            # GEMM batch shape can change FP32 log probabilities by a few ULPs.
            outputs = batched_actor(parameters, players, seeds, [0]*len(chunk), 8)
            results = self.requests('decide', [{'action': o[2]} for o in outputs])
            continued = self.requests('continue', [{} for _ in chunk]) if continuation else [None]*len(chunk)
            for spec, side, before, player, output, result, tail, seed in zip(chunk, sides, frames, players, outputs, results, continued, seeds):
                after = result['frame']['players'][side]
                reward = retained_hp(player, after)
                if reward != result['reward'] or not result['diagnostic_ended']:
                    raise ValueError('Reward reconstruction mismatch')
                # True here terminates the one-decision diagnostic objective only.
                # It never labels the underlying battle a natural terminal.
                rows.append(diagnostic_row(player, output, reward))
                records.append({'spec': spec, 'actor_seed': seed, 'side': side, 'player': player,
                    'after': after, 'action': output[2], 'logp': output[3], 'value': output[4], 'reward': reward,
                    'provenance': before['provenance'], 'natural_outcome': result['frame']['outcome'],
                    'continuation': tail})
        return rows, records

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for w in self.workers:
            w.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def specs_for(seed, update, n=128):
    return [{'context': i % 16, 'panel': 'train', 'episode': update*n+i,
             'run_seed': derived_seed('f6-training', seed)} for i in range(n)]


def evaluation_specs(seed, repeats):
    return [{'context': c, 'panel': panel, 'episode': k, 'run_seed': seed}
            for panel in ['train', 'variant'] for k in range(repeats) for c in range(16)]
