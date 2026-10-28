"""Full official train battles against a frozen public-view tactical opponent."""
import numpy as np
from pokemon_full_train.learning import derived_seed,prepare_batch
from pokemon_throughput.collector import merge_rows
from pokemon_fit.data import fingerprint

def specs(plan,seed,first):
    return [dict(seed=derived_seed('f15-rl-games',seed),episode=i,max_turns=plan['max_turns'],split='train',swap=False,opponent='tactical',learner_side='p1' if i%2==0 else 'p2',actor_seed=derived_seed('f15-rl-actor',seed)) for i in range(first,first+plan['chunk_games'])]

def collect_update(pool,p,plan,seed,index,cursor):
    collection=[];first=cursor
    while sum(len(rows) for rows,_ in collection)<plan['selected_rows']:
        if cursor-first>=plan['maximum_games_per_cohort']:raise ValueError('Fixed RL collection cap exhausted')
        chunk,_=pool.collect(p,specs(plan,seed,cursor))
        for i,(rows,g) in enumerate(chunk):
            if (g['episode'],g['split'],g['opponent'],g['learner_side'])!=(cursor+i,'train','tactical','p1' if (cursor+i)%2==0 else 'p2'):raise ValueError('RL identity')
            if len(rows)!=g['learner_rows'] or len(rows)!=g['learner_decisions'] or any(t['partition']!='train' for t in g['provenance']['teams']):raise ValueError('RL learner role')
        collection.extend(chunk);cursor+=plan['chunk_games']
    batch=prepare_batch(merge_rows(collection));n=len(batch['actions'])
    ids=np.random.default_rng(derived_seed('f15-rl-selection',seed,index)).choice(n,1024,replace=False)
    selected={k:v[ids] for k,v in batch.items()}
    return selected,{'games':[g for _,g in collection],'cursor_before':first,'cursor_after':cursor,'collected_rows':n,'selected_indices':ids.tolist(),'complete_batch_fingerprint':fingerprint(batch),'selected_batch_fingerprint':fingerprint(selected)}
