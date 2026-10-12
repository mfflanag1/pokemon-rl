"""Independent scalar/JAX checks against passive real-Showdown evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from . import scalar, engine
from .views import encode
from pokemon_learner.checkpoint import digest

ROOT=Path(__file__).resolve().parents[2]


def read(path): return json.loads(Path(path).read_text())
def packed(path): return json.loads(gzip.decompress(Path(path).read_bytes()))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, label):
    if actual != expected:
        keys = [k for k in expected if actual.get(k) != expected[k]] if isinstance(expected,dict) else []
        raise AssertionError(f'{label}: {keys}; actual={actual!r}; expected={expected!r}')


def check_batch(cases):
    ss=[scalar.initial(c['teams']) for c in cases]; js=engine.initial(np.array([c['teams'] for c in cases],np.int32))
    counts={'games':len(cases),'frames':0,'player_views':0,'decisions':0,'replacement_frames':0,'truncated':0,'terminal':0,'events':0,'chance_draws':0}
    for group in range(max(len(c['steps']) for c in cases)+1):
        jviews=[engine.view_kernel(js,side) for side in range(2)]
        actions=[];crit=[];roll=[];tie=[]
        for i,c in enumerate(cases):
            if group<=len(c['steps']):
                frame=c['initial'] if group==0 else c['steps'][group-1];reference=frame['state']
                compare(ss[i],reference,f'Scalar state game {c["id"]} group {group}')
                compare(engine.unpack(js,i),reference,f'JAX state game {c["id"]} group {group}')
                for side in range(2):
                    actual=encode(frame['views'][side]);expected=scalar.view(ss[i],side)
                    compare(actual,expected,f'Reference view game {c["id"]} group {group} side {side}')
                    jview={k:np.asarray(v[i]).tolist() for k,v in jviews[side]._asdict().items()}
                    compare(jview,actual,f'JAX view game {c["id"]} group {group} side {side}')
                    counts['player_views']+=1
                counts['frames']+=1;counts['replacement_frames']+=int(reference['phase']==1)
            if group<len(c['steps']):
                row=c['steps'][group];chance=row['chance'];actions.append(row['actions'])
                crit.append(chance['crit']);roll.append(chance['roll']);tie.append(chance['tie'])
                before=ss[i];ss[i],events=scalar.step(before,row['actions'],chance)
                compare(events,row['events'],f'Executed events {c["id"]}/{group}')
                move_sides=[e[1] for e in events if e[2]==1]
                for role in ('crit','roll'):
                    ds=[d for d in row['draws'] if d['role']==role]
                    compare(sorted(d['side'] for d in ds),sorted(move_sides),'Draw coverage')
                    for d in ds: compare(chance[role][d['side']],d['result'],'Conditional chance binding')
                for d in row['draws']:
                    scalar.require(d['role'] in ('crit','roll','tie'),'Chance role')
                ties=[d for d in row['draws'] if d['role']=='tie']
                speeds=[scalar.stats(scalar.SPECIES[before['species'][p][before['active'][p]]],bool(before['swift'][p][before['active'][p]]))[5] for p in (0,1)]
                tied=before['phase']==0 and (row['actions'][0]>=2)==(row['actions'][1]>=2) and speeds[0]==speeds[1]
                compare(bool(ties),tied,'Tie draw coverage')
                if ties:
                    scalar.require(all(d['events_before']==0 for d in ties),'Tie after execution')
                    # Gen9 re-sorts move ties before first execution; the final shuffle wins.
                    compare(chance['tie'],ties[-1]['result'],'Effective tie outcome binding')
                counts['decisions']+=sum(a>=0 for a in row['actions']);counts['events']+=len(events)
                for d in row['draws']:
                    value=d['word'];scalar.require(type(value) is int and 0<=value<2**32,'Raw word')
                    expected=(value*24//2**32==0) if d['role']=='crit' else value//2**28 if d['role']=='roll' else d['order'][value//2**31]
                    compare(d['result'],expected,'Chance mapping');counts['chance_draws']+=1
            else:actions.append([-1,-1]);crit.append([False,False]);roll.append([0,0]);tie.append(0)
        if group<max(len(c['steps']) for c in cases):
            js,_=engine.step(js,np.array(actions,np.int32),engine.Chance(jnp.array(crit,bool),jnp.array(roll,jnp.int32),jnp.array(tie,jnp.int32)))
    for c in cases:counts[c['outcome']]+=1
    return counts


def information_checks(bundle):
    data=packed(bundle);initial={};after={};own={};reveals={};n=0
    count=len(data['information']['comparisons']);scalar.require(count==1152,'Information fixture coverage')
    for c in data['information']['comparisons']:
        values={k:encode(c[k+'_view']) for k in ('initial','after','opponent_own')};n+=3
        for key,table,view in [('initial_key',initial,values['initial']),('after_key',after,values['after'])]:
            if c[key] in table:compare(view,table[c[key]],'Hidden-state invariance')
            else:table[c[key]]=view
        fields=c['opponent'].split('_');key=(c['observer'],c['own'],fields[0].split('-')[0],fields[1])
        own.setdefault(key,[]).append(values['opponent_own'])
    scalar.require(all(len(v)==2 and v[0]!=v[1] for v in own.values()),'Own speed positive controls')
    for c in data['information']['probes']:
        a,b=encode(c['after']['view']),encode(c['reveal']['view']);n+=2
        scalar.require(a!=b,'Reveal positive control')
        fields=c['spec']['teams'][1-c['observer']].split('_')
        key=(c['observer'],fields[0],fields[1].split('-')[1]);reveals.setdefault(key,[]).append((a,b))
    scalar.require(all(len(v)==2 and v[0][0]==v[1][0] and v[0][1]!=v[1][1] for v in reveals.values()),'Distinct reserve reveals')
    return {'encoded_views':n,'hidden_initial_equal':count-len(initial),'hidden_after_equal':count-len(after),
            'own_speed_positive':len(own),'reserve_reveal_pairs':len(reveals)}


def check_identity(c,plan):
    i=c['id'];teams=[i%24,i//24%24]
    namespace=plan['plan']['namespace'];suffix='smoke' if plan['smoke'] else 'fresh'
    compare(c['teams'],teams,'Team schedule')
    compare(c['spec'],{'teams':[scalar.ROSTER_IDS[t] for t in teams],
        'seed':'sodium,'+digest(['pokemon-reference-r1',f'{namespace}/{suffix}/{i}'])},'Fresh seed/spec schedule')
    for group,row in enumerate(c['steps']):
        state=c['initial']['state'] if group==0 else c['steps'][group-1]['state']
        actions=[]
        for side,mask in enumerate(scalar.masks(state)):
            legal=[a for a,ok in enumerate(mask) if ok];style=i%5
            if not legal:a=-1
            elif style==0:a=0 if 0 in legal else legal[0]
            elif style==1:a=1 if 1 in legal else legal[0]
            elif style==2:a=next((a for a in legal if a>=2),legal[0])
            else:a=legal[int(digest([namespace,i,group,side])[:8],16)%len(legal)]
            actions.append(a)
        compare(row['actions'],actions,'Fresh action schedule')


def audit(directory,*,write=True):
    directory=Path(directory);report=read(directory/'report.json');plan=read(directory/'plan.json')
    scalar.require(report['status']=='complete' and report['plan_sha256']==sha(directory/'plan.json'),'Capture binding')
    scalar.require(report['count']==plan['count'] and report['legacy']==72,'Capture coverage')
    compare(plan['plan'],read(ROOT/'contracts/switch-engine-v1/plan.json'),'Source plan')
    scalar.require(type(plan['smoke']) is bool and plan['count']==(24 if plan['smoke'] else plan['plan']['fresh_games']), 'Planned count')
    scalar.require(report['passivity']==min(plan['count'],plan['plan']['passivity_games']), 'Passivity coverage')
    compare(sha(directory/'legacy.json.gz'),report['legacy_sha256'],'Legacy binding')
    legacy_cases=packed(directory/'legacy.json.gz')
    sealed=packed(ROOT/'artifacts/runs/m3-switch-reference-v1/original/fixtures.json.gz')['cases']
    compare([c['id'] for c in legacy_cases],list(range(72)),'Sealed legacy grid')
    for c,old in zip(legacy_cases,sealed):
        compare(c['spec'],old['spec'],'Sealed legacy spec')
        frames=[c['initial'],*c['steps']];compare(len(frames),len(old['snapshots']),'Sealed legacy length')
        for frame,original in zip(frames,old['snapshots']):
            compare(frame['views'],original['frame']['views'],'Sealed legacy views')
            compare(frame['audit'],original['audit'],'Sealed legacy audit')
        for group,row in enumerate(c['steps']):
            expected=[next((d['action'] for d in old['tape'] if d['group']==group and d['side']==f'p{side+1}'),-1) for side in (0,1)]
            compare(row['actions'],expected,'Sealed legacy actions')
    legacy=check_batch(legacy_cases);total=None;seen=0
    for shard in report['shards']:
        compare(shard['file'],f'cases-{seen//plan["plan"]["shard_size"]:04d}.json.gz','Shard filename')
        scalar.require(shard['first']==seen and sha(directory/shard['file'])==shard['sha256'],'Shard identity/hash')
        cases=packed(directory/shard['file']);scalar.require(len(cases)==shard['count'],'Shard length')
        scalar.require([c['id'] for c in cases]==list(range(seen,seen+len(cases))),'Case grid')
        for c in cases:check_identity(c,plan)
        counts=check_batch(cases);total=counts if total is None else {k:total[k]+v for k,v in counts.items()};seen+=len(cases)
    scalar.require(seen==report['count'],'Complete reference campaign')
    information=information_checks(ROOT/'artifacts/runs/m3-switch-reference-v1/original/fixtures.json.gz')
    result={'status':'pass','legacy':legacy,'fresh':total,'information':information,'report_sha256':sha(directory/'report.json')}
    if write:(directory/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();print(json.dumps(audit(args.directory)))
