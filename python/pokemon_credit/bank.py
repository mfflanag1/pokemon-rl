"""Public-state selected continuation bank; disjoint teacher games for fit/probe."""
from collections import Counter
from pokemon_battle_audit.core import fingerprint,feature_hash,record_step,pending,cutoff_boundary
from pokemon_battle_audit.transport import Worker
from pokemon_full_train.storage import atomic_json

def alive(mon):
    return bool(mon) and 'fnt' not in mon.get('condition','') and mon.get('condition','').split(' ')[0]!='0'

def material(player):
    own=sum(alive(m) for m in player['own'])
    fainted=sum(bool(m) and not alive(m) for m in player['opponent'])
    return own,6-fainted

def eligible(frame,side,family):
    if frame['outcome'] or frame['players'][side]['turn']<8:return False
    if any(frame['players'][s]['phase']!='move' or not any(frame['players'][s]['mask']) for s in ['p1','p2']):return False
    own,foe=material(frame['players'][side])
    return own>=foe and ((family=='midgame' and 2<=foe<=4 and 3<=own<=6) or (family=='endgame' and 1<=foe<=2 and 1<=own<=4))

def generate(plan,evidence):
    scenarios=[];games=[];counts=Counter()
    with Worker(evidence/'worker') as worker:
        for episode in range(plan['teacher_games']):
            family=plan['families'][episode%2];role=plan['roles'][(episode//2)%2];side=['p1','p2'][(episode//4)%2]
            start={'run_seed':plan['teacher_seed'],'episode_index':episode,'split':'train','swap':False}
            frame=worker.request('start',**start);prefix=[];counters=[0,0];selected=None
            while not frame['outcome']:
                if selected is None and counts[family,role]<plan['states_per_family_role'] and not cutoff_boundary(frame,plan['max_turns']) and eligible(frame,side,family):
                    own,foe=material(frame['players'][side])
                    selected={'schema_version':'f11-train-scenario-v1','split':'train','seed':plan['teacher_seed'],
                        'learner_side':side,'start':start,'prefix':list(prefix),'counters':list(counters),'actor_seeds':[],
                        'provenance':frame['provenance'],'target_frame_hash':fingerprint(frame['players']),
                        'target_feature_hash':feature_hash(frame['players'][side]),'families':[family],
                        'origin':{'role':role,'teacher_episode':episode,'teacher_policy':'tactical-v1','turn':frame['players'][side]['turn'],'own_remaining':own,'foe_public_remaining':foe}}
                    selected['content_hash']=fingerprint(selected);scenarios.append(selected);counts[family,role]+=1
                if cutoff_boundary(frame,plan['max_turns']):frame=worker.request('cutoff');break
                if sum(counters)>=10000:raise ValueError('Teacher decision cap')
                command={'actions':{s:frame['tactical_actions'][s] for s in pending(frame)},'request_ids':{s:frame['players'][s]['request_id'] for s in pending(frame)}}
                after=worker.request('step',**command);prefix.append(record_step(frame,command,{s:None for s in command['actions']},after))
                for i,s in enumerate(['p1','p2']):counters[i]+=int(s in command['actions'])
                frame=after
            games.append({'episode':episode,'family':family,'role':role,'learner_side':side,'outcome':frame['outcome'],'decisions':sum(counters),'selected':selected['content_hash'] if selected else None,'steps':prefix})
    atomic_json(evidence/'teacher-games.json',games);atomic_json(evidence/'scenarios.json',scenarios)
    if any(counts[f,r]!=plan['states_per_family_role'] for f in plan['families'] for r in plan['roles']):raise ValueError('Fixed teacher budget failed bank quotas; no extension')
    return scenarios,games
