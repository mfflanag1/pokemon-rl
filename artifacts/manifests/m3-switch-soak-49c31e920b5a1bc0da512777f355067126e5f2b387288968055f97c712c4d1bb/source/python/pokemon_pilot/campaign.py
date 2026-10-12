"""Fixed-budget training and real-Showdown development evaluation; never final data."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import time
import traceback
import jax
import numpy as np
from pokemon_showdown.loop import Executables
from pokemon_showdown.checkpoint import Runner,save,load
from pokemon_soak.campaign import write,read,utc,fingerprint,check_diagnostics
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_jax.views import reduce_view
from .provenance import ROOT,admission,mechanics_seed,roster
from .inference import decide,request_prefix
from .statistics import paired_summary,calibration


class Reference:
    def __init__(self,log):
        self.log=log.open('xb')
        self.process=subprocess.Popen(['node','packages/pilot/worker.mjs'],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
        hello=self.receive();require(hello['type']=='ready','Reference handshake');self.reference=hello['reference']
    def receive(self):
        line=self.process.stdout.readline()
        require(bool(line),'Showdown worker exited; inspect reference.stderr.log')
        return json.loads(line)
    def call(self,command):
        self.process.stdin.write(wire(command).decode()+'\n');self.process.stdin.flush();return self.receive()
    def finish(self):
        end=self.call({'op':'finish'});require(end=={'type':'finished','reference':self.reference},'Reference drift')
        self.process.stdin.close()  # Deliver EOF so Node's stdin handle cannot retain the worker.
        require(self.process.wait(timeout=30)==0,'Reference worker failure');self.process.stdout.close();self.log.close()
    def abort(self):
        if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=30)
        self.process.stdin.close();self.process.stdout.close()
        self.log.close()


def train(out,exe,seeds,updates):
    result=[]
    for seed in seeds:
        runner=Runner(seed,exe);beg=time.perf_counter();checkpoints=[]
        for update in updates:
            while int(runner.learner.version)<update:check_diagnostics(runner.run())
            filename=f'checkpoint-{seed}-{update}.json';save(out/filename,runner)
            restored=load(out/filename,exe)
            left,metrics=exe(runner.learner,seed);other=restored.run()
            require(fingerprint(left)==fingerprint(restored.learner) and fingerprint(metrics)==fingerprint(other),'Training boundary replay')
            checkpoints.append({'update':update,'file':filename,'sha256':file_hash(out/filename),
                'decisions':runner.decisions,'games':int(runner.learner.next_game),
                'next_learner_fingerprint':fingerprint(left),'next_metrics_fingerprint':fingerprint(metrics)})
        result.append({'seed':seed,'checkpoints':checkpoints,'elapsed_seconds_including_replay':time.perf_counter()-beg})
        print(json.dumps({'phase':'training','seed':seed,'updates':updates[-1],'decisions':runner.decisions}),flush=True)
    return result


def cell_name(seed,update,opponent):return f'matches-{seed}-{update}-{opponent}.jsonl.gz'


def evaluate_cell(out,worker,parameters,initial,plan,seed,update,opponent,pairs,mode):
    path=out/cell_name(seed,update,opponent);beg=time.perf_counter()
    with path.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as tape:
        for first in range(0,2*pairs,plan['evaluation_batch_games']):
            ids=list(range(first,min(first+plan['evaluation_batch_games'],2*pairs)))
            records={i:{'seed':seed,'update':update,'opponent':opponent,'pair':i//2,'seat':i%2,
                        'mechanics_seed':mechanics_seed(plan,seed,opponent,i//2,mode),'decisions':[]} for i in ids}
            frame=worker.call({'op':'start','games':[{'id':i,'seed':records[i]['mechanics_seed']} for i in ids]})
            for _ in range(4):
                require(frame['type']=='batch' and [f['id'] for f in frame['games']]==ids,'Reference roster')
                live=[f for f in frame['games'] if not f['ended']]
                if not live:break
                inputs=[{**records[f['id']],'views':f['views']} for f in live]
                choices=decide(parameters,initial,inputs,seed,opponent)
                for f,c in zip(live,choices):records[f['id']]['decisions'].append(c)
                frame=worker.call({'op':'step','decisions':[{'id':f['id'],**c} for f,c in zip(live,choices)]})
            require(frame['type']=='batch' and [f['id'] for f in frame['games']]==ids,'Final reference roster')
            for f in frame['games']:
                require(f['ended'] and f['winner'] in (0,1),'Unfinished reference game')
                r=records[f['id']];r.update(winner=f['winner'],final_views=f['views'])
                tape.write(wire(r)+b'\n')
    return {'file':path.name,'sha256':file_hash(path),'games':2*pairs,'elapsed_seconds':time.perf_counter()-beg}


def verify_record(record,plan,seed,update,opponent,index,mode):
    require((record['seed'],record['update'],record['opponent'],record['pair'],record['seat'])==
            (seed,update,opponent,index//2,index%2),'Match roster/order')
    require(record['mechanics_seed']==mechanics_seed(plan,seed,opponent,index//2,mode),'Match chance seed')
    views=record['final_views'];require(len(views)==2,'Final player channels')
    reduced=[reduce_view(v) for v in views]
    winner=record['winner'];require(type(winner) is int and winner in (0,1),'Winner')
    for side,r in enumerate(reduced):
        require(views[side]['side']==f'p{side+1}' and r['header'][1:3]==[1,1 if side==winner else 2],'Public terminal outcome')
    decisions=record['decisions'];require(2<=len(decisions)<=4,'Battle length')
    require(all(r['header'][0]==len(decisions) for r in reduced),'Terminal turn/decision count')
    require([d['turn'] for d in decisions]==list(range(1,len(decisions)+1)),'Decision turn sequence')
    # Publicly executed actions must match the submitted tape, including KO cancellation.
    hits=[r for r in reduced[0]['history'] if r[0]>=0]
    for turn,actor,move,crit,damage,hp in hits:
        require(move==decisions[turn-1]['actions'][actor],'Submitted/executed action mismatch')
    for turn in range(1,len(decisions)+1):
        actors=[r[1] for r in hits if r[0]==turn]
        require(len(actors) in (1,2) and len(set(actors))==len(actors),'Public executed-turn coverage')
    return float(winner==record['seat'])


def audit_cell(out,parameters,initial,plan,seed,update,opponent,pairs,mode):
    scores=np.empty((pairs,2));values=[];outcomes=[];quick=[];count=decisions_count=0
    def consume(batch):
        nonlocal decisions_count
        for turn in range(4):
            live=[r for r in batch if len(r['decisions'])>turn]
            if not live:break
            frames=[{**r,'views':[request_prefix(v,r['decisions'][turn]['requests'][side]) for side,v in enumerate(r['final_views'])]} for r in live]
            actual=decide(parameters,initial,frames,seed,opponent)
            for r,a in zip(live,actual):
                require(a==r['decisions'][turn],'Persisted action/logp/value replay mismatch')
                values.append(a['value']);outcomes.append(scores[r['pair'],r['seat']]);quick.append(a['quick_attack_probability'])
                decisions_count+=1
    with gzip.open(out/cell_name(seed,update,opponent),'rt') as stream:
        batch=[]
        for line in stream:
            require(count<2*pairs,'Extra reference game');r=json.loads(line)
            score=verify_record(r,plan,seed,update,opponent,count,mode)
            scores[count//2,count%2]=score;count+=1;batch.append(r)
            if len(batch)==40:consume(batch);batch=[]
        if batch:consume(batch)
    require(count==2*pairs,'Missing reference game')
    return scores,{'critic':calibration(values,outcomes),'candidate_decisions':decisions_count,
                   'mean_quick_attack_probability':float(np.mean(quick)),'all_actions_and_values_replayed':True}


def analyze(out,exe,plan,smoke=False):
    seeds,updates,pairs=roster(plan,smoke);cells=[];primary=[];mode='smoke' if smoke else 'development'
    for seed in seeds:
        initial=load(out/f'checkpoint-{seed}-0.json',exe).learner.parameters
        baselines={}
        for update in updates:
            parameters=load(out/f'checkpoint-{seed}-{update}.json',exe).learner.parameters
            for opponent in plan['opponents']:
                scores,diagnostics=audit_cell(out,parameters,initial,plan,seed,update,opponent,pairs,mode)
                if update==0:baselines[opponent]=scores.copy()
                summary=paired_summary(scores,baseline=baselines[opponent],seed=plan['statistics_seed'],
                                        replicates=plan['bootstrap_replicates'],alpha=plan['family_alpha'])
                cells.append({'seed':seed,'update':update,'opponent':opponent,**summary,**diagnostics})
                if update==updates[-1] and opponent=='random':
                    adjusted=paired_summary(scores,baseline=baselines[opponent],seed=plan['statistics_seed'],
                        replicates=plan['bootstrap_replicates'],alpha=plan['family_alpha']/plan['primary_family_size'])
                    passed=adjusted['mean']>=.6 and adjusted['lower']>.55 and adjusted['improvement']['lower']>0
                    primary.append({'seed':seed,**adjusted,'passes_development_screen':passed})
                print(json.dumps({'phase':'audit','seed':seed,'update':update,'opponent':opponent}),flush=True)
    return {'cells':cells,'primary':primary,'all_seed_screen_passed':not smoke and all(x['passes_development_screen'] for x in primary),
            'games':len(cells)*2*pairs,'candidate_decisions':sum(c['candidate_decisions'] for c in cells),
            'interpretation':'Development screen only. Percentile-bootstrap intervals are approximate, including the Bonferroni adjustment; no exact coverage guarantee. Calibration pools repeated decisions descriptively, not as independent samples. Full G2/G3 remain open.'}


def main(directory,smoke=False):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run path');out.mkdir(exist_ok=False)
    r={'status':'running','mode':'smoke' if smoke else 'development','started_at':utc(),
       'full_g2':'open','full_g3':'open','prime_used':False,'final_evaluation_used':False,'strategic_strength_claim':False}
    write(out/'report.json',r);worker=None
    try:
        a=admission();write(out/'preregistration.json',a);r['admission_hash']=digest(a)
        plan=a['plan'];seeds,updates,pairs=roster(plan,smoke);exe=Executables()
        r['training']=train(out,exe,seeds,updates);write(out/'training.json',r['training'])
        worker=Reference(out/'reference.stderr.log');write(out/'reference.json',worker.reference);files=[]
        for seed in seeds:
            initial=load(out/f'checkpoint-{seed}-0.json',exe).learner.parameters
            for update in updates:
                parameters=load(out/f'checkpoint-{seed}-{update}.json',exe).learner.parameters
                for opponent in plan['opponents']:
                    files.append(evaluate_cell(out,worker,parameters,initial,plan,seed,update,opponent,pairs,r['mode']))
                    write(out/'matches-index.json',files)
                    print(json.dumps({'phase':'evaluation','seed':seed,'update':update,'opponent':opponent,'games':2*pairs}),flush=True)
        worker.finish();worker=None
        r['analysis']=analyze(out,exe,plan,smoke)
        require(admission()==a,'Pilot source/runtime/reservation drift');r['final_reservation_unchanged']=True;r['status']='passed'
    except BaseException as e:
        r.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if worker is not None:
            try:worker.abort()
            except Exception as cleanup:r['cleanup_error']=repr(cleanup)
        r['ended_at']=utc();r['content_hash']=digest(r);write(out/'report.json',r)
    print(json.dumps({'status':r['status'],'games':r['analysis']['games'],'all_seed_screen_passed':r['analysis']['all_seed_screen_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();main(args.directory,args.smoke)
