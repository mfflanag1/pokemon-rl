"""Add public decision annotations without changing the sealed actor or learner rows."""
from pokemon_curriculum.collector import Pool as BasePool,Episode as BaseEpisode
from pokemon_full.features import encode
from pokemon_throughput.collector import batched_actor
from collections import Counter
import time

class Episode(BaseEpisode):
    def advance(self,outputs):
        player=self.frame['players'][self.learner_side]
        phase=player['phase'];turn=player['turn'];before=len(self.rows)
        command=super().advance(outputs)
        if len(self.rows)>before:
            row=self.rows[-1]
            row['phase']=phase;row['turn']=turn
            row['voluntary_switch']=phase=='move' and row['action']>=8
            row['public_advice']=self.frame['tactical_actions'][self.learner_side]
        return command

class Pool(BasePool):
    def collect(self,parameters,specs):
        if self.closed or not specs:raise ValueError('Closed pool or empty collection')
        episodes=[Episode(**spec) for spec in specs];slots={};next_job=0;results=[None]*len(specs);profile=Counter();begun=time.perf_counter()
        try:
            while next_job<len(episodes) or slots:
                commands=[]
                for slot in range(len(self.workers)):
                    if slot not in slots and next_job<len(episodes):
                        slots[slot]=next_job;commands.append((slot,'start',episodes[next_job].start));next_job+=1
                self._requests(commands,slots,episodes,profile)
                order=sorted(slots);players=[];seeds=[];counters=[]
                for slot in order:
                    ep=episodes[slots[slot]];players.extend(ep.frame['players'][s] for s in ['p1','p2']);seeds.extend(ep.seeds);counters.extend(ep.counts)
                outputs=batched_actor(parameters,players,seeds,counters,16);commands=[]
                for i,slot in enumerate(order):
                    command=episodes[slots[slot]].advance(outputs[2*i:2*i+2])
                    if command:commands.append((slot,*command))
                self._requests(commands,slots,episodes,profile)
                for slot in order:
                    index=slots[slot]
                    if episodes[index].result is not None:results[index]=episodes[index].result;del slots[slot]
            profile['collection_seconds']=time.perf_counter()-begun
            return results,dict(profile)
        except BaseException:self.close();raise
