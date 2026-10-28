"""Complete 60-turn reference comparison and serialized continuation audits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from . import scalar, engine, carry
import jax
from pokemon_switch_engine.checkpoint import fingerprint
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


def check_identity(c):
    plan=carry.PLAN;i=c['id'];teams=[i//6,(i//6*7+9)%24]
    compare(c['teams'],teams,'Team schedule')
    compare(c['spec'],{'teams':[scalar.ROSTER_IDS[t] for t in teams],
        'seed':'sodium,'+digest(['pokemon-reference-r1',f'{plan["namespace"]}/fresh/{i}'])},'Seed schedule')
    for group,row in enumerate(c['steps']):
        state=c['initial']['state'] if group==0 else c['steps'][group-1]['state'];expected=[]
        for side,m in enumerate(scalar.masks(state)):
            legal=[a for a,ok in enumerate(m) if ok];style=i%6
            if not legal:a=-1
            elif style in (0,1):a=style if style in legal else legal[0]
            elif style==2 or style==3 and state['turn']<=24 or style==4 and state['turn']%3==1:a=next((a for a in legal if a>=2),legal[0])
            elif style in (3,4):a=0 if 0 in legal else legal[0]
            else:a=legal[int(digest([plan['namespace'],i,group,side])[:8],16)%len(legal)]
            expected.append(a)
        compare(row['actions'],expected,'Fixed action schedule')
    final=c['steps'][-1]['state']
    outcome='terminal' if final['winner']!=-1 else 'truncated'
    compare(c['outcome'],outcome,'Actual outcome')
    scalar.require(outcome=='terminal' or final['turn']==61 and final['phase']==0,'Complete administrative horizon')


def check_windows(cases):
    # One real reference trace per lane, carried through all fresh boundaries.
    admitted=carry.admission();saved=0;next_checks=0;replacement_steps=0;noops=0
    for c in cases:
        state=engine.initial(np.array([c['teams']],np.int32));index=0;ids=np.array([c['id']],np.uint32)
        for boundary in carry.BOUNDARIES:
            while not bool(np.asarray(carry.stopped(state,boundary))[0]):
                row=c['steps'][index];ch=row['chance']
                cc=engine.Chance(jnp.array([ch['crit']],bool),jnp.array([ch['roll']],jnp.int32),jnp.array([ch['tie']],jnp.int32))
                replacement_steps+=int(state.phase[0]==1)
                state,events=carry.step(state,np.array([row['actions']],np.int32),cc,boundary);index+=1
                compare(engine.unpack(state,0),row['state'],'Windowed reference state')
            # Complete wire round-trip binds game ID and source/runtime; no game reset.
            raw=carry.encode(state,ids,317,boundary,admitted)
            restored=carry.decode(raw,ids,317,boundary,admitted)
            compare(fingerprint(restored),fingerprint(state),'Serialized boundary state');saved+=1
            # Stopped lane cannot advance or consume an effective group address.
            cc=engine.Chance(jnp.zeros((1,2),bool),jnp.zeros((1,2),jnp.int32),jnp.zeros(1,jnp.int32))
            wait,events=carry.step(restored,np.array([[-1,-1]],np.int32),cc,boundary)
            compare(fingerprint(wait),fingerprint(state),'Boundary no-op');compare(np.asarray(events).tolist(),[[[-1]*9]*2],'No-op events');noops+=1
            if index<len(c['steps']):
                row=c['steps'][index];ch=row['chance'];cc=engine.Chance(jnp.array([ch['crit']],bool),jnp.array([ch['roll']],jnp.int32),jnp.array([ch['tie']],jnp.int32))
                a=engine.step(state,np.array([row['actions']],np.int32),cc)
                b=engine.step(restored,np.array([row['actions']],np.int32),cc)
                compare(fingerprint(a),fingerprint(b),'Original next simulator step');next_checks+=1
            state=restored
            if int(state.winner[0])!=-1:break
        compare(index,len(c['steps']),'Complete window trace')
    compare(carry.admission(),admitted,'Window audit admission')
    return {'serialized_boundaries':saved,'original_next_steps':next_checks,'replacement_steps':replacement_steps,'frozen_boundary_noops':noops}


def audit(directory):
    out=Path(directory);report=read(out/'report.json');plan=read(out/'plan.json')
    compare(plan['plan'],carry.PLAN,'Continuation plan')
    scalar.require(report['status']=='complete' and report['plan_sha256']==sha(out/'plan.json'),'Plan binding')
    scalar.require(report['file']=='cases.json.gz' and report['sha256']==sha(out/report['file']),'Tape binding')
    cases=packed(out/report['file']);compare([c['id'] for c in cases],list(range(carry.PLAN['fresh_cases'])),'Complete roster')
    compare(report['games'],len(cases),'Reported games');compare(report['unpatched_replays'],len(cases),'Replay roster')
    for c in cases:check_identity(c)
    counts=check_batch(cases);windows=check_windows(cases)
    coverage={k:0 for k in ['terminal','truncated','late_terminal','replacement','late_replacement','window_replacement','history_over32',
                            'boundary13','boundary25','boundary37','boundary49','boundary61','max_events','post12_moves']}
    for c in cases:
        coverage[c['outcome']]+=1;coverage['late_terminal']+=int(c['outcome']=='terminal' and c['steps'][-1]['state']['turn']>12)
        for row in c['steps']:
            s=row['state'];coverage['max_events']=max(coverage['max_events'],len(s['history']))
            coverage['history_over32']+=int(len(s['history'])>32)
            if s['phase']==1:
                coverage['replacement']+=1;coverage['late_replacement']+=int(s['turn']>12);coverage['window_replacement']+=int(s['turn'] in (12,24,36,48))
            if s['phase']==0 and s['turn'] in carry.BOUNDARIES:coverage['boundary'+str(s['turn'])]+=1
            coverage['post12_moves']+=sum(e[0]>12 and e[2]==1 for e in row['events'])
    scalar.require(all(v>0 for v in coverage.values()),'Nonvacuous fixture coverage')
    compare(report['coverage'],coverage,'Reported fixture coverage')
    return {'status':'passed','parity':counts,'continuation':windows,'coverage':coverage,'reference_report_sha256':sha(out/'report.json'),
            'learning_updates':0,'full_g2':'open','full_g3':'open','full_g4':'open','strength':'unqualified'}


if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1])))
