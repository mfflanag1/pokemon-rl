"""Complete evaluation games, retaining every attempted action and rejection."""
from collections import Counter
import hashlib
import time
from pokemon_full_learning.collector import Episode
from pokemon_transfer_v2.collector import Pool as FeedbackPool
from pokemon_full_train.learning import SIDES
from .actor import batched_actor

class AuditedEpisode(Episode):
    def __init__(self,**spec):
        super().__init__(**spec)
        self.actor_trace=[]

    def advance(self,outputs):
        command=super().advance(outputs)
        if command is not None and command[0]=='step':
            i=SIDES.index(self.learner_side);x,mask,action,logp,value=outputs[i]
            if mask.any():
                self.actor_trace.append({'request_id':self.frame['players'][self.learner_side]['request_id'],
                    'action':action,'log_probability':logp,'value':value,
                    'feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'mask':mask.tolist()})
        return command

    def finish(self,values):
        super().finish(values)
        self.result[1]['actor_trace']=self.actor_trace

class Pool(FeedbackPool):
    def __init__(self,evidence,workers,architecture):
        if architecture not in ['flat','shared']:raise ValueError('Unknown architecture')
        self.architecture=architecture
        super().__init__(evidence,workers)

    def collect(self, parameters, specs):
        if any(s.get('split')!='development' or s.get('opponent') is None for s in specs):
            raise ValueError('Frozen evaluation accepts development opponents only')
        if self.closed or not specs:
            raise ValueError('Closed pool or empty collection')
        # Validate all jobs before starting any simulator request.
        episodes = [AuditedEpisode(**spec) for spec in specs]
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
                outputs = batched_actor(parameters, players, seeds, counters, 2*len(self.workers), self.architecture)
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

