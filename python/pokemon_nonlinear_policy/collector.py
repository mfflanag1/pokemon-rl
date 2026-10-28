"""Full-game training and separate development self-play, with explicit architecture."""
from collections import Counter
import hashlib
import time
from pokemon_opponent_mix.collector import AuditedEpisode
from pokemon_full_learning.collector import Pool as BasePool
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from pokemon_opponent_mix.transport import Worker
from pokemon_full_train.learning import SIDES
from .actor import batched_actor
from pokemon_terminal_matchup.features import validate_catalog


class Pool(BasePool):
    def __init__(self, evidence, workers, architecture, catalog):
        if architecture not in ['control', 'nonlinear']:
            raise ValueError('Unknown architecture')
        self.evidence = Path(evidence)
        self.architecture = architecture
        validate_catalog(catalog)
        self.catalog = catalog
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Expected 1..16 workers')
        self.workers, self.closed, self.feedback_logs = [], False, []
        self.executor = ThreadPoolExecutor(max_workers=workers)
        try:
            for i in range(workers):
                worker = Worker(Path(evidence)/f'worker-{i}')
                self.workers.append(worker)
                log = open(Path(evidence)/f'feedback-{i}.jsonl', 'x')
                self.feedback_logs.append(log)
                state = {'last':None, 'start':None}
                original = worker.request
                def request(op, _original=original, _state=state, _log=log, **payload):
                    before = _state['last']; frame = _original(op, **payload)
                    if op == 'start': _state['start'] = payload
                    for submitted in frame.get('submitted', []):
                        if not submitted['accepted']:
                            _log.write(json.dumps({'start':_state['start'], 'command':{'op':op, **payload},
                                'submitted':submitted, 'before':before, 'after':frame}, allow_nan=False)+'\n')
                            _log.flush()
                    _state['last'] = frame
                    return frame
                worker.request = request
        except BaseException:
            self.close()
            raise

    def close(self):
        try: super().close()
        finally:
            for log in self.feedback_logs:
                if not log.closed: log.close()

    def collect(self, parameters, specs):
        if self.closed or not specs:
            raise ValueError('Closed pool or empty collection')
        episodes = [AuditedEpisode(**spec) for spec in specs]
        if len({ep.split for ep in episodes}) != 1:
            raise ValueError('Mixed training/development collection')
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
                    seeds.extend(ep.seeds)
                    counters.extend(ep.counts)
                t0 = time.perf_counter()
                outputs = batched_actor(parameters, players, seeds, counters,
                                        2*len(self.workers), self.architecture, self.catalog)
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
        except BaseException as exc:
            # Coordinator-only failure evidence; never part of an actor input.
            path = self.evidence/'collector-failure.json'
            try:
                if not path.exists():
                    with path.open('x') as file:
                        json.dump({'error':repr(exc), 'specs':specs, 'slots':slots,
                            'last_frames':[worker.last for worker in self.workers]},file,allow_nan=False)
            finally:
                self.close()
            raise
