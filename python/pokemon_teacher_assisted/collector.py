"""F35 collection with frozen structured actors and explicit rejection attribution.

The qualified episode/GAE path still receives the unchanged v2 player view.
Only the actor sees the corresponding side's observable history. Policies remain
fixed throughout a collection call, including the archived opponent.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from pokemon_full_learning.collector import Pool as BasePool
from pokemon_full_train.learning import SIDES
from pokemon_nonlinear_policy.actor import batched_actor as frozen_actor
from pokemon_opponent_mix.collector import AuditedEpisode
from pokemon_worker_watch.collector import Pool as ObservedPool
from pokemon_worker_watch.transport import Worker as ObservedWorker
from pokemon_structured_policy.actor import batched_actor


class Worker(ObservedWorker):
    def __init__(self, evidence):
        super().__init__(evidence, command=['node', 'packages/teacher-assisted/worker.mjs'])

    def request(self, op, **payload):
        frame = super().request(op, **payload)
        if op in ('start', 'step', 'cutoff'):
            histories = frame.get('observable_history')
            if not isinstance(histories, dict) or set(histories) != set(SIDES):
                self.close()
                raise ValueError('Missing player-local observable history')
            if any(histories[s].get('schema_version') != 'structured-public-history-v1' for s in SIDES):
                self.close()
                raise ValueError('Unknown observable history schema')
        return frame


class Episode(AuditedEpisode):
    """Add attribution without changing inherited row/reward/GAE semantics."""
    def __init__(self, **spec):
        super().__init__(**spec)
        self.pending_attempts = {}
        self.rejected_attempts = []
        self.accepted_decisions = self.accepted_controlled_decisions = 0

    def advance(self, outputs):
        before_rows = len(self.rows)
        command = super().advance(outputs)
        if command is not None and command[0] == 'step':
            next_row = before_rows
            self.pending_attempts = {}
            for side, action in command[1]['actions'].items():
                controlled = self.opponent is None or side == self.learner_side
                row = next_row if controlled and self.split == 'train' else None
                if row is not None:
                    next_row += 1
                self.pending_attempts[side] = {'side': side, 'controlled': controlled,
                    'action': action, 'request_id': command[1]['request_ids'][side],
                    'actor_counter': self.counts[SIDES.index(side)] - 1,
                    'learner_row_index': row}
        return command

    def receive(self, op, frame):
        if op == 'step':
            seen = set()
            for submitted in frame['submitted']:
                side = submitted['side']
                if side not in self.pending_attempts or side in seen:
                    raise ValueError('Submission/attempt attribution mismatch')
                seen.add(side)
                attempt = self.pending_attempts[side]
                if submitted['action'] != attempt['action'] or type(submitted['accepted']) is not bool:
                    raise ValueError('Submission action/acceptance mismatch')
                if submitted['accepted']:
                    self.accepted_decisions += 1
                    self.accepted_controlled_decisions += int(attempt['controlled'])
                else:
                    reasons = submitted.get('feedback')
                    if not isinstance(reasons, list) or not reasons or any(
                            not isinstance(x, str) or not x.startswith('[Unavailable choice]') for x in reasons):
                        raise ValueError('Unknown rejected-action feedback')
                    after = frame['players'][side]
                    if not any(after['mask']) or after['request_id'] <= attempt['request_id']:
                        raise ValueError('Rejected action lacks refreshed actionable request')
                    self.rejected_attempts.append({**attempt, 'feedback': reasons,
                        'refreshed_request_id': after['request_id'],
                        'refreshed_mask': after['mask'], 'selected_for_update': None})
            if seen != set(self.pending_attempts):
                raise ValueError('Missing submission result')
            self.pending_attempts = {}
        return super().receive(op, frame)

    def finish(self, values):
        super().finish(values)
        game = self.result[1]
        if self.rejections != len(self.rejected_attempts):
            raise ValueError('Rejection accounting mismatch')
        if self.accepted_decisions + self.rejections != self.decisions:
            raise ValueError('Accepted/rejected decision accounting mismatch')
        game.update(rejected_attempts=self.rejected_attempts,
            accepted_decisions=self.accepted_decisions,
            accepted_controlled_decisions=self.accepted_controlled_decisions,
            eligible_learner_rows=len(self.rows))


class BaselineEpisode(Episode):
    def __init__(self, **spec):
        if spec.get('opponent') != 'baseline-v1' or spec.get('split') != 'development':
            raise ValueError('Baseline is excluded from F35 training')
        super().__init__(**{**spec, 'opponent': 'tactical'})
        self.start['tactical_side'] = None
        self.start['baseline_side'] = 'p2' if self.learner_side == 'p1' else 'p1'

    def advance(self, outputs):
        self.frame = {**self.frame, 'tactical_actions': self.frame['baseline_actions']}
        return super().advance(outputs)

    def finish(self, values):
        super().finish(values)
        self.result[1]['opponent'] = 'baseline-v1'


class ArchivedEpisode(Episode):
    """Use the episode's external-opponent action seam for a frozen learned host.

The inherited tactical seam selects only the opposing action and excludes that
side from learner rows. No tactical advice is requested from Node for these games.
The actual opponent identity is retained in every completed game record.
"""
    def __init__(self, **spec):
        if spec.get('opponent') != 'frozen_f32' or spec.get('split') != 'train':
            raise ValueError('Archived opponent is a training-only stream')
        super().__init__(**{**spec, 'opponent': 'tactical'})
        self.start['tactical_side'] = None
        self.archived_action = None

    def advance(self, outputs):
        foe = 'p2' if self.learner_side == 'p1' else 'p1'
        if self.frame['outcome'] is None and any(self.frame['players'][foe]['mask']):
            if self.archived_action is None:
                raise ValueError('Missing frozen opponent action')
            # A coordinator-owned frame copy; the player inputs remain unchanged.
            self.frame = {**self.frame, 'tactical_actions': {foe: self.archived_action}}
        return super().advance(outputs)

    def finish(self, values):
        super().finish(values)
        self.result[1]['opponent'] = 'frozen_f32'


class Pool(ObservedPool):
    def __init__(self, evidence, workers, catalog, frozen_parameters=None):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Expected 1..16 workers')
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.catalog, self.frozen_parameters = catalog, frozen_parameters
        self.workers, self.feedback_logs, self.closed = [], [], False
        self.executor = ThreadPoolExecutor(max_workers=workers)
        try:
            self.workers = []
            for index in range(workers):
                self.workers.append(Worker(self.evidence / f'worker-{index}'))
        except BaseException:
            self.close()
            raise

    def collect(self, parameters, specs):
        if self.closed or not specs:
            raise ValueError('Closed pool or empty collection')
        episodes = [({'frozen_f32': ArchivedEpisode, 'baseline-v1': BaselineEpisode}.get(s.get('opponent'), Episode))(**s)
                    for s in specs]
        if len({ep.split for ep in episodes}) != 1:
            raise ValueError('Mixed training/development collection')
        if any(isinstance(ep, ArchivedEpisode) for ep in episodes) and self.frozen_parameters is None:
            raise ValueError('Archived opponent parameters required')
        slots, next_job, results = {}, 0, [None] * len(specs)
        profile, begun = Counter(), time.perf_counter()
        try:
            while next_job < len(episodes) or slots:
                commands = []
                for slot in range(len(self.workers)):
                    if slot not in slots and next_job < len(episodes):
                        slots[slot] = next_job
                        commands.append((slot, 'start', episodes[next_job].start))
                        next_job += 1
                self._requests(commands, slots, episodes, profile)
                players, histories, seeds, counters = [], [], [], []
                order = sorted(slots)
                for slot in order:
                    ep = episodes[slots[slot]]
                    players.extend(ep.frame['players'][s] for s in SIDES)
                    histories.extend(ep.frame['observable_history'][s] for s in SIDES)
                    seeds.extend(ep.seeds)
                    counters.extend(ep.counts)
                started = time.perf_counter()
                outputs = batched_actor(parameters, players, histories, seeds, counters,
                                        2 * len(self.workers), self.catalog)
                profile['encode_infer_sync_seconds'] += time.perf_counter() - started
                profile['inference_calls'] += 1
                profile['active_actor_rows'] += len(players)
                profile['padded_actor_rows'] += 2 * len(self.workers) - len(players)
                archived = [episodes[slots[slot]] for slot in order
                            if isinstance(episodes[slots[slot]], ArchivedEpisode)]
                if archived:
                    started = time.perf_counter()
                    indices = [1 if ep.learner_side == 'p1' else 0 for ep in archived]
                    old_outputs = frozen_actor(self.frozen_parameters,
                        [ep.frame['players'][SIDES[i]] for ep, i in zip(archived, indices)],
                        [ep.seeds[i] for ep, i in zip(archived, indices)],
                        [ep.counts[i] for ep, i in zip(archived, indices)],
                        len(self.workers), 'nonlinear', self.catalog)
                    for ep, out in zip(archived, old_outputs):
                        ep.archived_action = out[2]
                    profile['archived_opponent_infer_seconds'] += time.perf_counter() - started
                commands = []
                started = time.perf_counter()
                for index, slot in enumerate(order):
                    ep = episodes[slots[slot]]
                    command = ep.advance(outputs[2 * index:2 * index + 2])
                    if command:
                        commands.append((slot, *command))
                profile['coordinator_seconds'] += time.perf_counter() - started
                self._requests(commands, slots, episodes, profile)
                for slot in order:
                    index = slots[slot]
                    if episodes[index].result is not None:
                        results[index] = episodes[index].result
                        del slots[slot]
            profile['collection_seconds'] = time.perf_counter() - begun
            return results, dict(profile)
        except BaseException as exc:
            try:
                path = self.evidence / f'collection-failure-{time.time_ns()}.json'
                path.write_text(json.dumps({'error': repr(exc), 'specs': specs,
                    'slots': slots, 'last_frames': [w.last for w in self.workers],
                    'inflight_episodes': [{'index': index, 'start': episodes[index].start,
                        'decisions': episodes[index].decisions, 'actor_counters': episodes[index].counts,
                        'trace': episodes[index].trace, 'rejected_attempts': episodes[index].rejected_attempts,
                        'pending_attempts': episodes[index].pending_attempts} for index in slots.values()]}, allow_nan=False))
            except BaseException as evidence_error:
                exc.add_note('Failure evidence: ' + repr(evidence_error))
            self.close()
            raise
