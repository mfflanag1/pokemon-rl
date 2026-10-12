"""Independent replay of codec, player boundary and fixture-accounting evidence."""
import gzip
import json
from pathlib import Path
from pokemon_reference.contract import hash_json
from .contract import mask,command,PROFILE,require
from .views import actor


def read_bundle(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def check_channel(view):
    other='p2' if view['side']=='p1' else 'p1'
    for line in view['protocol']:
        p=line.split('|')
        if len(p)>3 and p[1] in ('switch','-damage','-heal') and p[2].startswith(other+'a:'):
            hp=p[4] if p[1]=='switch' else p[3]
            require(hp=='0 fnt' or hp.endswith('/100'),'Opponent HP leak')
        require(not any(line.startswith('|'+k+'|') for k in ('-heal','-status','-boost','-unboost','-terastallize','error','split')),'Out-of-scope protocol')


def check_view(view):
    check_channel(view)
    return actor(view)


def validate(data):
    require(data['profile']==PROFILE and data['format']=='switch-reference-fixtures-1','Profile/version')
    roster={r['id']:r for r in data['manifest']['roster']}
    expected=[]
    for a in PROFILE['species']:
        for b in PROFILE['species']:
            if a['id']==b['id']:continue
            for x in PROFILE['variants']:
                for y in PROFILE['variants']:
                    expected.append({'id':f"{a['id']}-{x}_{b['id']}-{y}",'species':[a['id'],b['id']],'variants':[x,y]})
    require(data['manifest']['roster']==expected,'Exact profile roster')
    require(len(data['cases'])==72,'Fixture count')
    golden=[];decisions=0;statuses={'terminal':0,'truncated':0};seen=set()
    coverage=dict.fromkeys(('terminal','truncated','replacement','wait','voluntary_switch','critical_hits','super_effective','resisted','low_hp_ability_attacks','speed_tie_turns','masked_hp'),0)
    for c in data['cases']:
        key=(c['spec']['teams'][0],c['style']);require(key not in seen,'Repeated fixture');seen.add(key)
        require(c['style'] in ('attack','mixed','switch') and all(t in roster for t in c['spec']['teams']),'Fixture identity')
        snaps=c['snapshots'];bygroup={};decisions+=len(c['tape'])
        require([s['frame']['group'] for s in snaps]==list(range(len(snaps))),'Frame linkage')
        for i,s in enumerate(snaps):
            f=s['frame'];require(len(f['views'])==2 and [v['side'] for v in f['views']]==['p1','p2'],'Frame seats')
            for v in f['views']:check_channel(v)
            if f['phase']=='replacement':coverage['replacement']+=1
            if f['phase']!='terminal':
                for v in f['views']:
                    coverage['wait']+=int(v['request'].get('wait',False))
                    m=mask(v['request']);golden.append({'view':v,'mask':m,'commands':[command(v['request'],j) if m[j] else None for j in range(4)]})
            bygroup[i]=f
            if i+1<len(snaps):
                active=[next((m for m in side['pokemon'] if m['active']),None) for side in s['audit']['sides']]
                if f['phase']=='joint_move' and active[0]['stats']['spe']==active[1]['stats']['spe']:coverage['speed_tie_turns']+=1
                for seat,mon in enumerate(active):
                    if not mon:continue
                    lines=snaps[i+1]['frame']['views'][seat]['protocol'][len(f['views'][seat]['protocol']):]
                    signature=next(p['moves'][0] for p in PROFILE['species'] if p['id']==mon['species'])
                    prefix=f"|move|p{seat+1}a: {mon['name']}|{signature}|"
                    if 0<3*mon['hp']<=mon['maxhp'] and any(line.startswith(prefix) for line in lines):coverage['low_hp_ability_attacks']+=1
        submitted={}
        for d in c['tape']:
            f=bygroup[d['group']];seat=['p1','p2'].index(d['side']);v=f['views'][seat]
            require(seat in f['requested'] and d['view_hash']==hash_json(v) and d['request_id']==v['request_id'],'Submission identity')
            require(d['command']==command(v['request'],d['action']),'Submission codec')
            submitted.setdefault(d['group'],[]).append(seat)
            if d['action']>=2 and f['phase']=='joint_move':coverage['voluntary_switch']+=1
        for i in range(len(snaps)-1):require(sorted(submitted.get(i,[]))==sorted(bygroup[i]['requested']),'Decision linkage')
        end=snaps[-1]['frame'];out=c['outcome'];statuses[out['status']]+=1;coverage[out['status']]+=1
        for line in end['views'][0]['protocol']:
            for token,key in (('-crit','critical_hits'),('-supereffective','super_effective'),('-resisted','resisted')):
                if line.startswith('|'+token+'|'):coverage[key]+=1
            if (line.startswith('|switch|p2a:') or line.startswith('|-damage|p2a:')) and '/100' in line:coverage['masked_hp']+=1
        require(out['turn']==end['turn'] and out['winner']==end['winner'],'Outcome linkage')
        if out['status']=='truncated':require(end['phase']=='joint_move' and end['turn']==13 and out['winner'] is None,'Truncation is not draw')
        else:
            require(end['phase']=='terminal','Terminal phase')
            marker='|tie|' if out['winner'] is None else '|win|P'+out['winner'][1]
            require(all(marker in v['protocol'] for v in end['views']),'Public winner')
    require(golden==data['golden'],'Complete golden-view roster')
    fingerprints=[]
    for row in golden:
        a=check_view(row['view']);require(a.policy_inputs()==a.value_inputs(),'Policy/value boundary');fingerprints.append(a.feature_id)
    require(coverage==data['coverage'] and all(n>0 for n in coverage.values()),'Independently counted coverage')
    info=data['information'];initial={};after={};own={};comparison_actors=0
    require(len(info['comparisons'])==1152,'Counterfactual roster size');pairs=set()
    for c in info['comparisons']:
        observer=c['observer'];r=roster[c['opponent']];key=(observer,c['own'],c['opponent'])
        require(key not in pairs and observer in (0,1) and c['own'] in roster,'Counterfactual identity');pairs.add(key)
        ik='/'.join(map(str,[observer,c['own'],r['species'][0]]));ak=ik+'/'+r['variants'][0]
        require(c['initial_key']==ik and c['after_key']==ak,'Counterfactual grouping')
        actual={}
        for label,seat in (('initial',observer),('after',observer),('opponent_own',1-observer)):
            view=c[label+'_view'];require(view['side']==f'p{seat+1}' and hash_json(view)==c[label+'_hash'],'Counterfactual view/hash binding')
            actual[label]=check_view(view).feature_id;comparison_actors+=1
        initial.setdefault(ik,[]).append((c['initial_hash'],actual['initial']))
        after.setdefault(ak,[]).append((c['after_hash'],actual['after']))
        own.setdefault((observer,c['own'],*r['species'],r['variants'][1]),[]).append((c['opponent_own_hash'],actual['opponent_own']))
    require(len(initial)==144 and all(len(v)==8 and len(set(v))==1 for v in initial.values()),'Hidden initial invariance')
    require(len(after)==288 and all(len(v)==4 and len(set(v))==1 for v in after.values()),'Unrevealed reserve after-turn invariance')
    require(len(own)==576 and all(len(v)==2 and v[0][0]!=v[1][0] and v[0][1]!=v[1][1] for v in own.values()),'Own variant positive control')
    reveals={}
    for p in info['probes']:
        observer=p['observer'];r=roster[p['spec']['teams'][1-observer]]
        actors=[check_view(p[k]['view']) for k in ('initial','after','reveal')]
        for k in ('initial','after','reveal'):
            row=p[k];require(mask(row['view']['request'])==row['mask'],'Probe mask')
            require([command(row['view']['request'],i) if row['mask'][i] else None for i in range(4)]==row['commands'],'Probe command')
        require(actors[1].feature_id!=actors[2].feature_id,'Reveal updates input')
        reveals.setdefault((observer,r['species'][0],*r['variants']),[]).append((actors[1].feature_id,actors[2].feature_id))
    require(len(reveals)==24 and all(len(v)==2 and v[0][0]==v[1][0] and v[0][1]!=v[1][1] for v in reveals.values()),'Different reserve reveals')
    damage=data['damage'];require(len(damage)==18,'Conditional damage roster');damage_keys=set();n=0
    for row in damage:
        key=(row['attacker'],row['defender'],row['seat']);require(key not in damage_keys,'Duplicate damage key');damage_keys.add(key)
        require(row['hp']==[row['maxhp']//3-1,row['maxhp']//3,row['maxhp']//3+1],'Ability HP boundary')
        v=row['values'];require(len(v)==3 and all(len(a)==2 and all(len(b)==2 and all(len(c)==16 for c in b) for b in a) for a in v),'Damage dimensions')
        for critical in range(2):
            for roll in range(16):
                require(v[0][0][critical][roll]==v[1][0][critical][roll]>v[2][0][critical][roll],'Ability effect')
                require(v[0][1][critical][roll]==v[1][1][critical][roll]==v[2][1][critical][roll],'Normal move control')
        for hp in v:
            for move in hp:
                require(all(type(d) is int and d>0 for crit in move for d in crit),'Integer damage')
                require(all(move[1][i]>move[0][i] for i in range(16)),'Critical control')
        require(row['calls']==192,'One draw per damage call');n+=row['calls']
    return {'status':'passed','fixture_hash':hash_json(data),'games':72,'outcomes':statuses,'decisions':decisions,
            'codec_views':len(golden),'actor_views':len(fingerprints)+len(info['probes'])*3+comparison_actors,
            'actor_fingerprints_hash':hash_json(fingerprints),'hidden_initial_equal':sum(len(v)-1 for v in initial.values()),'hidden_post_equal':sum(len(v)-1 for v in after.values()),
            'own_variant_positive_controls':len(own),'different_reserve_reveals':len(reveals),'coverage':coverage,'conditional_damage_values':n,
            'full_g2':'open','full_g3':'open','jax_port_implemented':False,'training_performed':False,
            'prime_used':False,'final_evaluation_used':False}
