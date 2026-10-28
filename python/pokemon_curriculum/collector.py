"""Complete on-policy suffixes; replayed prefix actions never become learner rows."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import time
from pokemon_battle_audit.transport import Worker as AuditWorker
from pokemon_battle_audit.core import reset_scenario, fingerprint
from pokemon_mixture.collector import Episode as BaseEpisode, Pool as BasePool
from pokemon_throughput.collector import batched_actor


class Worker(AuditWorker):
    def __init__(self,evidence,bank):
        self.bank=bank
        super().__init__(evidence)
        self.feedback=open(Path(evidence)/'feedback.jsonl','x')
        self.previous=None

    def request(self,op,**payload):
        if op=='start':
            allowed={'run_seed','episode_index','split','swap','tactical_side','scenario_hash'}
            if set(payload)-allowed: raise ValueError('Unknown collector start fields')
            selected=payload.pop('scenario_hash',None)
            advisor=payload.pop('tactical_side',None)
            if selected is not None:
                if payload['split']!='train' or payload['swap'] is not False or selected not in self.bank:
                    raise ValueError('Prefix reset requires an admitted train scenario')
                scenario=self.bank[selected]
                if advisor!=('p2' if scenario['learner_side']=='p1' else 'p1'):
                    raise ValueError('Scenario learner/opponent seat mismatch')
                frame,counts=reset_scenario(SimpleNamespace(request=super().request),scenario)
                prefix={'scenario_hash':selected,'counts':counts,'decisions':sum(counts),
                        'target_frame_hash':fingerprint(frame['players']),'simulator_start':dict(scenario['start'])}
            else:
                frame=super().request(op,**payload)
                prefix={'scenario_hash':None,'counts':[0,0],'decisions':0,'target_frame_hash':fingerprint(frame['players']),'simulator_start':dict(payload)}
            frame={**frame,'prefix':prefix}
        else:
            frame=super().request(op,**payload)
        for submitted in frame.get('submitted',[]) if op=='step' else []:
            if not submitted['accepted']:
                self.feedback.write(json.dumps({'command':payload,'submitted':submitted,'before':self.previous,'after':frame},allow_nan=False)+'\n')
                self.feedback.flush()
        self.previous=frame
        return frame

    def close(self):
        super().close()
        if hasattr(self,'feedback') and not self.feedback.closed:self.feedback.close()


class Episode(BaseEpisode):
    def __init__(self,*,scenario_hash=None,**spec):
        super().__init__(**spec)
        if scenario_hash is not None:
            if self.split!='train' or self.opponent!='tactical': raise ValueError('Prefix role must be train against tactical')
            self.start['scenario_hash']=scenario_hash
        self.scenario_hash=scenario_hash;self.prefix=None;self.diagnostics=Counter()

    def receive(self,op,frame):
        super().receive(op,frame)
        if op=='start':
            self.prefix=frame['prefix']
            if self.prefix['scenario_hash']!=self.scenario_hash: raise ValueError('Wrong prefix identity')
            self.counts=list(self.prefix['counts'])
            if self.rows or self.trace or any(x is not None for x in self.last.values()): raise ValueError('Prefix contaminated learner rows')
        elif op=='step':
            d=frame['diagnostics'].get(self.learner_side)
            if d:
                self.diagnostics['learner_decisions']+=1
                self.diagnostics['heuristic_disagreements']+=int(d['disagreement'])
                self.diagnostics.update(d['tags'])

    def advance(self,outputs):
        # This public-state observation happens before the action; it never
        # enters the policy, loss, selection rule or reward.
        candidate=None
        if not self.frame['outcome']:
            i=0 if self.learner_side=='p1' else 1
            player=self.frame['players'][self.learner_side];action=outputs[i][2]
            if 0<=action<8 and action%4<len(player['active_moves']) and player['active_moves'][action%4]['id']=='recover':
                own=next((m for m in player['own'] if m.get('active')),None)
                hp=own['condition'].split(' ')[0] if own else ''
                if '/' in hp and hp.split('/')[0]==hp.split('/')[1]:
                    candidate=action
        command=super().advance(outputs)
        if candidate is not None and command and command[0]=='step' and self.learner_side in command[1]['actions']:
            self.diagnostics['full_hp_recover']+=1
            self.diagnostics['full_hp_recover_tera']+=int(candidate>=4)
        return command

    def finish(self,values):
        super().finish(values)
        rows,game=self.result
        game.update(prefix=self.prefix,diagnostics=dict(self.diagnostics))
        # Base receive(cutoff) can finish before advance; counters only include
        # suffix decisions. The prefix is reset cost, never training history.
        if self.split=='train' and len(rows)!=game['learner_decisions']: raise ValueError('Wrong on-policy learner row count')


class Pool(BasePool):
    def __init__(self,evidence,workers,bank):
        if type(workers) is not int or workers!=8: raise ValueError('F12 fixed actor capacity requires eight workers')
        self.workers=[];self.closed=False;self.executor=ThreadPoolExecutor(max_workers=workers)
        try:
            for i in range(workers):self.workers.append(Worker(Path(evidence)/str(i),bank))
        except BaseException:
            self.close();raise

    def collect(self,parameters,specs):
        if self.closed or not specs:raise ValueError('Closed pool or empty cohort')
        episodes=[Episode(**spec) for spec in specs]
        slots={};next_job=0;results=[None]*len(specs);profile=Counter();begun=time.perf_counter()
        try:
            while next_job<len(episodes) or slots:
                commands=[]
                for slot in range(len(self.workers)):
                    if slot not in slots and next_job<len(episodes):
                        slots[slot]=next_job;commands.append((slot,'start',episodes[next_job].start));next_job+=1
                self._requests(commands,slots,episodes,profile)
                order=sorted(slots);players=[];seeds=[];counters=[]
                for slot in order:
                    ep=episodes[slots[slot]]
                    players.extend(ep.frame['players'][s] for s in ['p1','p2']);seeds.extend(ep.seeds);counters.extend(ep.counts)
                t=time.perf_counter();outputs=batched_actor(parameters,players,seeds,counters,16)
                profile['encode_infer_seconds']+=time.perf_counter()-t
                commands=[]
                for i,slot in enumerate(order):
                    command=episodes[slots[slot]].advance(outputs[2*i:2*i+2])
                    if command:commands.append((slot,*command))
                self._requests(commands,slots,episodes,profile)
                for slot in order:
                    index=slots[slot]
                    if episodes[index].result is not None:
                        results[index]=episodes[index].result;del slots[slot]
            profile['collection_seconds']=time.perf_counter()-begun
            profile['prefix_replay_decisions']=sum(g['prefix']['decisions'] for _,g in results)
            return results,dict(profile)
        except BaseException:
            self.close();raise
