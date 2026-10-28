"""Fixed exploratory semantic-input canary with exhaustive deterministic replay."""
import argparse,json,hashlib,time,resource
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from pokemon_full_train.storage import ROOT,digest,canonical,atomic_json,freeze,admission,save_checkpoint,runtime
from pokemon_full_train.learning import derived_seed
from pokemon_full.features import encode
from pokemon_full.learner import parameter_hash
from pokemon_balanced.common import initialize
from pokemon_balanced.sampling import select,classify
from pokemon_balanced.learning import NAMES
from pokemon_imitation.data import summarize
from pokemon_fit.data import fingerprint
from pokemon_credit.campaign import record,same
from pokemon_entropy.campaign import array_record
from pokemon_transfer.common import retention
from .semantic import extras,EXTRA
from .learner import extend,checked,predictions,load_checkpoint
PLAN=ROOT/'contracts/switch-audit-v1/plan.json'
def read(p):return json.loads(p.read_text())
def admit(plan):
    r=read(ROOT/plan['predecessor_receipt']);body=dict(r);h=body.pop('content_hash')
    if digest(ROOT/plan['predecessor_receipt'])!=plan['predecessor_sha256'] or hashlib.sha256(canonical(body)).hexdigest()!=h:raise ValueError('F16 receipt changed')
    if (ROOT/plan['predecessor_receipt']).read_bytes()!=(ROOT/'artifacts/manifests'/('f16-balanced-imitation-'+h)/'receipt.json').read_bytes():raise ValueError('F16 archive changed')
    if read(ROOT/r['run']/'inputs/manifest.json')['runtime']!=runtime():raise ValueError('Predecessor runtime drift')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F16 source drift')
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F16 evidence drift '+rel)
    for key in ['audit_report','catalog']:
        if digest(ROOT/plan[key])!=plan[key+'_sha256']:raise ValueError('Changed '+key)
    audit=ROOT/'artifacts/runs/f17-switch-audit-v1'
    for rel,sha in read(audit/'inputs.json').items():
        if digest(ROOT/rel)!=sha:raise ValueError('Audit input drift')
    for rel,sha in read(audit/'source-binding.json')['sources'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('Audit source drift')
    if read(audit/'source-binding.json')['runtime']!=runtime():raise ValueError('Audit runtime drift')
    return r

def prepare(plan):
    catalog=read(ROOT/plan['catalog']);audit=ROOT/'artifacts/runs/f17-switch-audit-v1';base=ROOT/'artifacts/runs/f16-balanced-imitation-v1';out={};inputs={}
    for split,source,count in [('train',base/'inputs/teacher.npz',512),('probe',base/'probe/dataset.npz',128)]:
        with np.load(source,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        rows=[]
        for i in range(count):
            path=audit/split/f'game-{i:04d}.json';inputs[str(path.relative_to(ROOT))]=digest(path);rows.extend(read(path))
        if len(rows)!=len(d['targets']):raise ValueError('Missing semantic rows')
        e=[]
        for i,r in enumerate(rows):
            x,m=encode(r['player'])
            if not np.array_equal(x,d['features'][i]) or not np.array_equal(m,d['mask'][i]) or r['metadata']['target']!=int(d['targets'][i]) or r['metadata']['game']!=int(d['game_index'][i]):raise ValueError('Semantic row linkage')
            e.append(extras(r['player'],catalog))
        out[split]=(d,np.stack(e))
    oracle=read(base/'inputs/tactical-oracle.json');records=oracle['records'];
    if len(records)!=512:raise ValueError('Expected512 old oracle records')
    xs=[];ms=[];es=[];targets=[]
    for r in records:
        x,m=encode(r['player']);e=extras(r['player'],catalog);xs.append(x);ms.append(m);es.append(e);targets.append(r['target'])
    od={'features':np.stack(xs),'mask':np.stack(ms),'targets':np.asarray(targets,np.int32),'game_index':np.arange(512,dtype=np.int32)};out['oracle']=(od,np.stack(es))
    with np.load(base/'inputs/old-supervised.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    if len(old['targets'])!=128:raise ValueError('Expected128 old fit rows')
    # Rehearsal comes from the F6/F7 fit partition, not the later F10 tactical probe.
    from pokemon_fit.data import load_data
    fit_plan=read(ROOT/'contracts/decision-fit-v1/plan.json');fit_splits,_=load_data(fit_plan)
    if any(not np.array_equal(old[k],fit_splits['fit_train'][k]) for k in old):raise ValueError('Old fit partition differs')
    raw_path=ROOT/fit_plan['dataset'];inputs[str(raw_path.relative_to(ROOT))]=digest(raw_path)
    fit_records=sorted([r for r in read(raw_path)['records'] if r['spec']['panel']=='train' and 0<=r['spec']['episode']<8],key=lambda r:(r['spec']['panel'],r['spec']['episode'],r['spec']['context']))
    if len(fit_records)!=128:raise ValueError('Original rehearsal row count')
    old_extra=[]
    for i,r in enumerate(fit_records):
        x,m=encode(r['player'])
        if not np.array_equal(x,old['features'][i]) or not np.array_equal(m,old['mask'][i]) or r['target']!=int(old['targets'][i]):raise ValueError('Original rehearsal row order')
        old_extra.append(extras(r['player'],catalog))
    out['old']=(old,np.stack(old_extra))
    return out,records,inputs

def batch(data,old,seed,step,arm):
    d,e=data;o,oe=old
    b,ids=select(d,derived_seed('f16-balanced-sample',seed,step));oi=np.random.default_rng(derived_seed('f15-old-sample',seed,step)).integers(0,128,64)
    result={k:np.concatenate([b[k],o[k][oi]]) for k in b};extra=np.concatenate([e[ids],oe[oi]])
    result['extra']=extra if arm=='semantic' else np.zeros_like(extra)
    return result,{'teacher_indices':ids.tolist(),'old_indices':oi.tolist(),'batch_fingerprint':fingerprint(result)}

def probabilities(p,d,e,arm):
    parts=[]
    for start in range(0,len(e),256):
        n=min(256,len(e)-start);x=d['features'][start:start+n];m=d['mask'][start:start+n];z=e[start:start+n]
        # Pad by repeating a legal existing row; discarded outputs never affect metrics.
        pad=lambda a:np.concatenate([a,np.repeat(a[-1:],256-n,axis=0)])
        parts.append(np.asarray(predictions(p,jnp.asarray(pad(x)),jnp.asarray(pad(z) if arm=='semantic' else np.zeros((256,EXTRA),np.float32)),jnp.asarray(pad(m))))[:n])
    return np.concatenate(parts)
def metrics(pr,d):
    labels=classify(d)
    return {'total':summarize(pr,d['mask'],d['targets'],d['game_index']), 'families':{name:summarize(pr[labels==i],d['mask'][labels==i],d['targets'][labels==i],d['game_index'][labels==i]) for i,name in enumerate(NAMES)}}
def tactical(pr,records):
    groups={}
    for p,r in zip(pr,records):groups.setdefault(f"{r['spec']['panel']}/{r['spec']['context']}",[]).append((float(p[r['target']]),int(p.argmax())==r['target']))
    return {'cells':{k:{'n':len(rows),'target_mass':float(np.mean([r[0] for r in rows])),'greedy_accuracy':float(np.mean([r[1] for r in rows]))} for k,rows in groups.items()}}
def analyze(a,settings):
    z,s=a['zero']['probe']['families'],a['semantic']['probe']['families']
    checks={'voluntary_accuracy':s['voluntary_switch']['game_accuracy']>=settings['voluntary_min_accuracy'],
      'voluntary_gain':s['voluntary_switch']['game_accuracy']-z['voluntary_switch']['game_accuracy']>=settings['voluntary_min_gain'],
      **{name+'_preserved':z[name]['game_accuracy']-s[name]['game_accuracy']<=settings['maximum_other_accuracy_drop'] for name in ['normal_move','tera_move','forced_switch']},'known_contexts_retained':a['semantic']['retention']['passed']}
    return {'checks':checks,'passed':all(checks.values())}

def run(path,original=None):
    plan=read(PLAN)
    if plan['seeds']!=[7103,8209,9311] or plan['arms']!=['zero','semantic'] or plan['steps']!=2048 or plan['expected_updates']!=12288 or plan['new_games']!=0:raise ValueError('Canary scope drift')
    admit(plan);data,oracle,inputs=prepare(plan)
    if original:admission(original,read(original/'inputs/manifest.json'))
    path.mkdir(parents=True,exist_ok=False);manifest=freeze(path,plan);h=digest(path/'inputs/manifest.json');atomic_json(path/'audit-inputs.json',inputs)
    if original:same(manifest,read(original/'inputs/manifest.json'),'manifest');same(inputs,read(original/'audit-inputs.json'),'audit inputs')
    for name,(d,e) in data.items():array_record(path/'inputs'/f'{name}.npz',{**d,'extra':e},original/'inputs'/f'{name}.npz' if original else None)
    start=time.perf_counter();updates=0;recoveries=0;comparisons=0;results={};value_heads_unchanged=True
    ts=read(ROOT/'contracts/balanced-imitation-v1/plan.json')['tactical_thresholds']
    for seed in plan['seeds']:
        initial,_=initialize(seed);ip=extend(initial);initial_t=tactical(probabilities(ip,*data['oracle'],'semantic'),oracle);arms={}
        for arm in plan['arms']:
            p=extend(initial);opt=({k:jnp.zeros_like(v) for k,v in p.items()},{k:jnp.zeros_like(v) for k,v in p.items()},jnp.asarray(0,jnp.int32));base=path/str(seed)/arm;obase=original/str(seed)/arm if original else None
            def checkpoint(label,p,opt,step,recover=False):
                nonlocal recoveries,comparisons
                meta={'seed':seed,'input_hash':h,'cohorts':step,'cursors':{'bc':step,'arm':arm}}
                save_checkpoint(base/label,p,opt,meta)
                if original:
                    q,qo,qm=load_checkpoint(obase/label,h)
                    if qm['seed']!=seed or qm['cohorts']!=step or qm['cursors']!=meta['cursors'] or int(qo[2])!=int(opt[2]):raise ValueError('Checkpoint metadata differs')
                    for a,b in [(p,q),(opt[0],qo[0]),(opt[1],qo[1])]:
                        if set(a)!=set(b) or any(not np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a):raise ValueError('Checkpoint state differs')
                    comparisons+=1
                    if recover:p,opt=q,qo;recoveries+=1
                return p,opt
            checkpoint('initial',p,opt,0)
            for step in range(plan['steps']):
                if step==plan['steps']-1:p,opt=checkpoint('before-last',p,opt,step,True)
                b,selection=batch(data['train'],data['old'],seed,step,arm);before=parameter_hash(p);p,opt,m=checked(p,opt,b)
                value_heads_unchanged=value_heads_unchanged and all(np.array_equal(np.asarray(p[k]),np.asarray(initial[k])) for k in ['vw','vb'])
                if int(opt[2])!=step+1 or not value_heads_unchanged:raise ValueError('Optimizer/value-head drift')
                record(base/'updates'/f'{step:04d}.json',{**selection,'before':before,'after':parameter_hash(p),'optimizer_step':int(opt[2]),'metrics':m},obase/'updates'/f'{step:04d}.json' if original else None);updates+=1
            checkpoint('final',p,opt,plan['steps']);summaries={}
            for name in ['train','probe','oracle']:
                d,e=data[name];pr=probabilities(p,d,e,arm);array_record(base/(name+'-predictions.npz'),{'probabilities':pr},obase/(name+'-predictions.npz') if original else None)
                if name=='oracle':summaries['retention']=retention(initial_t,tactical(pr,oracle),ts)
                else:summaries[name]=metrics(pr,d)
            summaries['parameter_hash']=parameter_hash(p);arms[arm]=summaries
            record(base/'summary.json',summaries,obase/'summary.json' if original else None);atomic_json(path/'progress.json',{'seed':seed,'arm':arm,'updates':updates,'elapsed_seconds':time.perf_counter()-start})
        results[str(seed)]={'arms':arms,'gate':analyze(arms,plan['coverage_thresholds'])}
    if updates!=plan['expected_updates']:raise ValueError('Update total mismatch')
    scientific={'status':'passed','plan':plan,'input_hash':h,'updates':updates,'presentations':updates*128,'new_games':0,'rows':{name:len(d['targets']) for name,(d,e) in data.items()},'seeds':results,'all_seed_gate':all(r['gate']['passed'] for r in results.values()),'value_head_weights_unchanged':value_heads_unchanged}
    record(path/'scientific.json',scientific,original/'scientific.json' if original else None)
    admission(path,manifest);admit(plan)
    if any(digest(ROOT/p)!=sha for p,sha in inputs.items()):raise ValueError('Audit row drift')
    report={**scientific,'elapsed_seconds':time.perf_counter()-start,'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2};atomic_json(path/'report.json',report)
    if original:
        if recoveries!=6 or comparisons!=18:raise ValueError('Missing checkpoint recovery')
        atomic_json(path/'verification.json',{'status':'passed','report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),'updates':updates,'presentations':updates*128,'checkpoint_comparisons':comparisons,'original_next_update_recoveries':recoveries,'all_seed_gate':report['all_seed_gate']})
    print(json.dumps({'status':'passed','elapsed_seconds':report['elapsed_seconds'],'all_seed_gate':report['all_seed_gate']}),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--original',type=Path);args=a.parse_args();run(args.run.resolve(),args.original.resolve() if args.original else None)
