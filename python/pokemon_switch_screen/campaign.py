"""Fixed-budget fresh training and complete capped Showdown development tapes."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_soak.campaign import write,utc
from pokemon_switch_engine import learner,views
from pokemon_switch_engine.scalar import require,ROSTER,ROSTER_IDS
from pokemon_pilot.campaign import Reference as BaseReference
from . import checkpoint as cp
from .provenance import ROOT,PLAN,admission,read,roster,spec
from .inference import decide,prefix
from .statistics import summarize


class Reference(BaseReference):
    def __init__(self,log):
        self.log=Path(log).open('xb')
        self.process=subprocess.Popen(['node','packages/switch-screen/worker.mjs'],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
        hello=self.receive();require(hello['type']=='ready','Reference handshake');self.reference=hello['reference']


def done(frame):return frame['phase']=='terminal' or frame['truncated']
def cell_name(seed,update,opponent):return f'matches-{seed}-{update}-{opponent}.jsonl.gz'
def checkpoint_name(seed,update):return f'checkpoint-{seed}-{update}.json'


def train(out,admitted,smoke):
    seeds,updates,_=roster(smoke);records=[]
    for seed in seeds:
        state=learner.start(seed);checkpoints=[];decisions=terminal=0;beg=time.perf_counter()
        for u in updates:
            while int(state.version)<u:
                state,buffer,final,_=cp.transaction(state,seed)
                decisions+=int(np.asarray(buffer.valid).sum());terminal+=int((np.asarray(final.winner)!=-1).sum())
            path=out/checkpoint_name(seed,u);cp.save(path,state,seed,admitted)
            nxt=cp.result_fingerprints(cp.transaction(state,seed))
            require(nxt==cp.result_fingerprints(cp.transaction(cp.load(path,seed,admitted),seed)),'Original next-update replay')
            checkpoints.append({'update':u,'file':path.name,'sha256':file_hash(path),'state':cp.fingerprint(state),
                                'next':nxt,'decisions':decisions,'games':32*u,'terminal':terminal,'truncated':32*u-terminal})
            write(out/'progress.json',{'phase':'training','seed':seed,'updates':u,'decisions':decisions})
            print(json.dumps({'phase':'training','seed':seed,'updates':u,'decisions':decisions}),flush=True)
        records.append({'seed':seed,'checkpoints':checkpoints,'elapsed_seconds_including_replay':time.perf_counter()-beg})
    return records


def evaluate_cell(out,worker,parameters,initial,seed,update,opponent,pairs,mode):
    path=out/cell_name(seed,update,opponent);beg=time.perf_counter()
    with path.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as tape:
        for first in range(0,2*pairs,32):
            ids=list(range(first,min(first+32,2*pairs)))
            records={i:{'seed':seed,'update':update,'opponent':opponent,'pair':i//2,'seat':i%2,
                        'spec':spec(seed,opponent,i//2,i%2,mode),'decisions':[]} for i in ids}
            frame=worker.call({'op':'start','games':[{'id':i,**records[i]['spec']} for i in ids]})
            for _ in range(learner.GROUPS):
                require(frame['type']=='batch' and [f['id'] for f in frame['games']]==ids,'Reference roster')
                live=[f for f in frame['games'] if not done(f)]
                if not live:break
                inputs=[{**records[f['id']],'views':f['views'],'group':f['group']} for f in live]
                actions=decide(parameters,initial,inputs,seed,opponent,mode)
                for f,c in zip(live,actions):records[f['id']]['decisions'].append(c)
                frame=worker.call({'op':'step','decisions':[{'id':f['id'],**c} for f,c in zip(live,actions)]})
            require([f['id'] for f in frame['games']]==ids and all(done(f) for f in frame['games']),'Incomplete capped game')
            for f in frame['games']:
                r=records[f['id']];r.update(winner=f['winner'],truncated=f['truncated'],final_views=f['views'],groups=f['group'])
                tape.write(wire(r)+b'\n')
    return {'file':path.name,'sha256':file_hash(path),'games':2*pairs,'elapsed_seconds':time.perf_counter()-beg}


def verify_record(r,seed,update,opponent,index,mode):
    require((r['seed'],r['update'],r['opponent'],r['pair'],r['seat'])==(seed,update,opponent,index//2,index%2),'Match identity/order')
    require(r['spec']==spec(seed,opponent,index//2,index%2,mode),'Match chance/team identity')
    final=r['final_views'];require(len(final)==2 and [v['side'] for v in final]==['p1','p2'],'Receiving channels')
    end=[views.encode(v) for v in final]
    winners=[[l for l in v['protocol'] if l.startswith('|win|') or l=='|tie|'] for v in final]
    if r['truncated'] is True:
        require(r['winner'] is None and winners==[[],[]] and all(v['context'][:2]==[13,0] for v in end),'Explicit nonterminal cutoff')
        outcome=-1
    else:
        require(r['truncated'] is False and r['winner'] in ('p1','p2'),'Terminal outcome')
        require(winners==[[f"|win|P{r['winner'][1]}"]]*2 and all(v['context'][1]==3 for v in end),'Terminal protocol linkage')
        winner=int(r['winner'][1])-1
        require(all(m[1]==0 for m in end[1-winner]['own']) and any(m[1]>0 for m in end[winner]['own']),'Terminal HP linkage')
        outcome=int(int(r['winner'][1])-1==r['seat'])
    ds=r['decisions'];require(1<=len(ds)<=learner.GROUPS and r['groups']==len(ds) and [d['group'] for d in ds]==list(range(len(ds))),'Request-group coverage')
    frames=[]
    for group,d in enumerate(ds):
        before=[prefix(v,n) for v,n in zip(final,d['view_lengths'])]
        require(len(before)==2 and [digest(v) for v in before]==d['view_hashes'],'Original view binding')
        require(d['turn']==before[0]['turn']==before[1]['turn'] and d['requests']==[v['request_id'] for v in before],'Request identity')
        b=[views.encode(v) for v in before]
        if group==0:
            require(all(v['context'][:4]==[1,0,0,0] and v['context'][4]==0 for v in b),'Fresh initial request')
            for side in (0,1):
                row=ROSTER[ROSTER_IDS.index(r['spec']['teams'][side])]
                require([m[0]-1 for m in b[side]['own']]==row[0] and
                        [m[7] for m in b[side]['own']]==[([65,63,80][s]+32*v) for s,v in zip(*row)],'Own team/stat identity')
        after=final if group==len(ds)-1 else [prefix(v,n) for v,n in zip(final,ds[group+1]['view_lengths'])]
        a=[views.encode(v) for v in after]
        require(all(len(after[s]['protocol'])>len(before[s]['protocol']) for s in (0,1)),'Advancing player histories')
        h0=b[0]['context'][4];h1=a[0]['context'][4];events=a[0]['history'][h0:h1]
        require(a[0]['history'][:h0]==b[0]['history'][:h0] and 1<=len(events)<=2 and len({e[1] for e in events})==len(events),'Executed group coverage')
        for e in events:
            action=d['actions'][e[1]]
            require(e[0]==d['turn'] and (e[2]==0 and action==e[3]+2 or e[2]==1 and action==e[5]),'Submitted/executed action linkage')
        for side,action in enumerate(d['actions']):
            if action>=0 and side not in {e[1] for e in events}:
                require(action<2 and a[side]['own'][b[side]['context'][2]][1]==0,'Only faint-cancelled move may be absent')
        frames.append({**r,'group':group,'views':before})
    return outcome,frames


def audit_cell(out,parameters,initial,seed,update,opponent,pairs,mode):
    scores=np.empty((pairs,2),np.int32);count=decisions=0;actions=[0]*4;values=[];terminal_values=[];terminal_outcomes=[]
    def consume(batch):
        nonlocal decisions
        for group in range(learner.GROUPS):
            selected=[(r,fs) for r,fs in batch if len(fs)>group]
            if not selected:break
            actual=decide(parameters,initial,[fs[group] for _,fs in selected],seed,opponent,mode)
            for (r,_),a in zip(selected,actual):
                require(a==r['decisions'][group],'Action/logp/value replay mismatch')
                action=a['actions'][r['seat']]
                if action>=0:
                    decisions+=1;actions[action]+=1;values.append(a['candidate_value'])
                    score=scores[r['pair'],r['seat']]
                    if score>=0:terminal_values.append(a['candidate_value']);terminal_outcomes.append(int(score))
    with gzip.open(out/cell_name(seed,update,opponent),'rt') as f:
        batch=[]
        for line in f:
            require(count<2*pairs,'Extra reference game');r=json.loads(line)
            score,frames=verify_record(r,seed,update,opponent,count,mode)
            scores[count//2,count%2]=score;count+=1;batch.append((r,frames))
            if len(batch)==32:consume(batch);batch=[]
        if batch:consume(batch)
    require(count==2*pairs,'Missing reference game')
    # Values at cutoffs are not outcome labels; terminal-only errors remain diagnostic.
    return scores,{'candidate_decisions':decisions,'actions':actions,'value_min':min(values),'value_max':max(values),
                  'terminal_decision_value_mse':float(np.mean((np.array(terminal_values)-terminal_outcomes)**2)) if terminal_values else None,
                  'terminal_value_samples':len(terminal_values),'all_actions_and_values_replayed':True}


def analyze(out,admitted,smoke=False):
    seeds,updates,pairs=roster(smoke);cells=[];primary=[];mode='smoke' if smoke else 'development'
    for seed in seeds:
        initial=cp.load(out/checkpoint_name(seed,0),seed,admitted).parameters;baselines={}
        for u in updates:
            parameters=cp.load(out/checkpoint_name(seed,u),seed,admitted).parameters
            for opponent in PLAN['opponents']:
                scores,diagnostics=audit_cell(out,parameters,initial,seed,u,opponent,pairs,mode)
                if u==0:baselines[opponent]=scores.copy()
                summary=summarize(scores,baselines[opponent],alpha=PLAN['family_alpha'],family_size=PLAN['primary_family_size'])
                cells.append({'seed':seed,'update':u,'opponent':opponent,**summary,**diagnostics})
                if u==updates[-1] and opponent=='random':
                    passed=summary['outcome_lower']>=PLAN['minimum_secured_win_fraction'] and summary['adjusted_win_lower']>PLAN['minimum_secured_win_lower_bound'] and summary['adjusted_improvement_lower']>PLAN['minimum_worst_case_improvement_lower_bound']
                    primary.append({'seed':seed,**summary,'passes_screen':not smoke and passed})
                print(json.dumps({'phase':'audit','seed':seed,'update':u,'opponent':opponent}),flush=True)
    return {'cells':cells,'primary':primary,'all_seed_screen_passed':not smoke and all(p['passes_screen'] for p in primary),
            'games':len(cells)*2*pairs,'candidate_decisions':sum(c['candidate_decisions'] for c in cells),
            'terminal':sum(c['wins']+c['losses'] for c in cells),'truncated':sum(c['cutoffs'] for c in cells)}


def main(directory,smoke=False):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run path');out.mkdir(exist_ok=False)
    r={'status':'running','mode':'smoke' if smoke else 'development','started_at':utc(),
       'full_g2':'open','full_g3':'open','full_g4':'open','prime_used':False,'final_evaluation_used':False,
       'public_server_used':False,'public_ladder_strength':'unqualified'}
    write(out/'report.json',r);worker=None
    try:
        a=admission();write(out/'preregistration.json',a);r['admission_hash']=digest(a)
        r['preproduction_freeze_sha256']=None
        if not smoke:
            freeze=ROOT/'artifacts/reviews/m3-switch-screen/preregistration.json';f=read(freeze)
            require(f['admission']==a and f['frozen_at']<r['started_at'],'Preregister before any production training/evaluation')
            r['preproduction_freeze_sha256']=file_hash(freeze)
        write(out/'report.json',r)
        r['training']=train(out,a,smoke);write(out/'training.json',r['training'])
        worker=Reference(out/'reference.stderr.log');write(out/'reference.json',worker.reference);files=[]
        seeds,updates,pairs=roster(smoke)
        for seed in seeds:
            initial=cp.load(out/checkpoint_name(seed,0),seed,a).parameters
            for u in updates:
                parameters=cp.load(out/checkpoint_name(seed,u),seed,a).parameters
                for opponent in PLAN['opponents']:
                    files.append(evaluate_cell(out,worker,parameters,initial,seed,u,opponent,pairs,r['mode']))
                    write(out/'matches-index.json',files)
                    progress={'phase':'evaluation','seed':seed,'update':u,'opponent':opponent,'games':2*pairs}
                    write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        worker.finish();worker=None;r['analysis']=analyze(out,a,smoke)
        require(admission()==a,'Source/runtime/reserved-data drift');r['final_reservation_unchanged']=True;r['status']='passed'
    except BaseException as e:
        r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if worker is not None:
            try:worker.abort()
            except Exception as e:r['cleanup_error']=repr(e)
        r['ended_at']=utc();r['content_hash']=digest(r);write(out/'report.json',r)
    print(json.dumps({'status':r['status'],'games':r['analysis']['games'],'screen_passed':r['analysis']['all_seed_screen_passed']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--smoke',action='store_true');args=p.parse_args();main(args.directory,args.smoke)
