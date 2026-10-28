"""Whole teacher games; only accepted, player-visible decisions become BC rows."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from pokemon_battle_audit.transport import Worker
from pokemon_battle_audit.core import fingerprint,pending,cutoff_boundary,record_step
from pokemon_full.features import encode
from pokemon_full_train.storage import atomic_json
from pokemon_credit.campaign import record
from pokemon_entropy.campaign import array_record
from .data import validate

def collect_game(worker,seed,episode,split,max_turns):
    if split not in ['train','development']:raise ValueError('Teacher role refused')
    start={'run_seed':seed,'episode_index':episode,'split':split,'swap':False}
    frame=worker.request('start',**start);provenance=frame['provenance'];rows=[];identities=[];steps=[];attempts=0;rejected=0
    if any(t['partition']!=split for t in provenance['teams']):raise ValueError('Teacher team partition')
    while not frame['outcome']:
        if cutoff_boundary(frame,max_turns):frame=worker.request('cutoff');break
        if attempts>=10000:raise ValueError('Teacher decision cap')
        sides=pending(frame);command={'actions':{s:frame['tactical_actions'][s] for s in sides},'request_ids':{s:frame['players'][s]['request_id'] for s in sides}}
        encoded={s:encode(frame['players'][s]) for s in sides}
        for side,(x,m) in encoded.items():
            a=command['actions'][side]
            if type(a) is not int or not 0<=a<14 or not m[a]:raise ValueError('Illegal teacher target')
        after=worker.request('step',**command);steps.append(record_step(frame,command,{s:None for s in sides},after))
        if [s['side'] for s in after['submitted']]!=sides:raise ValueError('Teacher submission identity')
        for submitted in after['submitted']:
            side=submitted['side'];attempts+=1
            if not submitted['accepted']:rejected+=1;continue
            x,m=encoded[side];a=command['actions'][side]
            rows.append((x,m,a));identities.append({'episode':episode,'side':side,'request_id':frame['players'][side]['request_id'],'action':a,'phase':frame['players'][side]['phase']})
        frame=after
    if not rows:raise ValueError('Empty teacher game')
    dataset={'features':np.stack([x for x,_,_ in rows]),'mask':np.stack([m for _,m,_ in rows]),'targets':np.asarray([a for _,_,a in rows],np.int32),'game_index':np.full(len(rows),episode,np.int32)}
    validate(dataset)
    game={'episode':episode,'start':start,'provenance':provenance,'outcome':frame['outcome'],'decisions':attempts,'accepted_rows':len(rows),'rejected_choices':rejected,'row_identity':identities,'steps':steps}
    if len(rows)+rejected!=attempts:raise ValueError('Teacher row linkage')
    return dataset,game

def generate(seed,count,split,plan,evidence,original=None):
    evidence.mkdir(parents=True,exist_ok=False)
    def work(slot):
        outputs=[]
        with Worker(evidence/f'worker-{slot}') as worker:
            for episode in range(slot,count,plan['workers']):
                data,g=collect_game(worker,seed,episode,split,plan['max_turns'])
                record(evidence/f'game-{episode:04d}.json',g,original/f'game-{episode:04d}.json' if original else None)
                outputs.append((episode,data,g))
        return outputs
    with ThreadPoolExecutor(max_workers=plan['workers']) as executor:
        outputs=sorted([item for batch in executor.map(work,range(min(count,plan['workers']))) for item in batch],key=lambda a:a[0])
    if [o[0] for o in outputs]!=list(range(count)):raise ValueError('Teacher missing or duplicate game')
    data={k:np.concatenate([o[1][k] for o in outputs]) for k in outputs[0][1]};validate(data)
    games=[o[2] for o in outputs];h=array_record(evidence/'dataset.npz',data,original/'dataset.npz' if original else None)
    inventory={'split':split,'seed':seed,'games':count,'rows':len(data['targets']),'fingerprint':h,'terminals':sum(g['outcome']['kind']=='terminal' for g in games),
        'cutoffs':sum(g['outcome']['kind']=='administrative_cutoff' for g in games),'decisions':sum(g['decisions'] for g in games),'rejected_choices':sum(g['rejected_choices'] for g in games),
        'phase_counts':dict(Counter(i['phase'] for g in games for i in g['row_identity'])),'action_counts':{str(k):v for k,v in Counter(map(int,data['targets'])).items()}}
    record(evidence/'inventory.json',inventory,original/'inventory.json' if original else None)
    return data,games,inventory
