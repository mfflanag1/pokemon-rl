"""Admission, replay audit and immutable receipt for the M3 engineering smoke."""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch.provenance import CURRENT
from pokemon_switch_engine import checkpoint,learner,parity
from pokemon_switch_engine.campaign import record
from pokemon_switch_engine.scalar import require

previous=runpy.run_path(str(ROOT/'scripts/seal-showdown-soak.py'))
read,write_once=[previous[k] for k in ('read','write_once')]
REVIEW=ROOT/'artifacts/reviews/m3-switch-engine'


def admission():
    checks=0
    for name,key in [('showdown-soak','predecessor_receipt'),('m3-switch-reference','reference_receipt')]:
        path=ROOT/f'artifacts/reports/{name}.json';prior=read(path);h=learner.PLAN[key]
        require(prior['content_hash']==h==digest({k:v for k,v in prior.items() if k!='content_hash'}),'Predecessor receipt')
        archive=ROOT/'artifacts/manifests'/f'{name}-{h}'
        require((archive/'receipt.json').read_bytes()==path.read_bytes(),'Predecessor archive receipt')
        for kind,field in [('source','source_files'),('evidence','evidence_files')]:
            for name,expected in prior[field].items():
                require(file_hash(archive/kind/name)==expected,'Archived predecessor drift: '+name);checks+=1
                if name not in CURRENT:require(file_hash(ROOT/name)==expected,'Live predecessor drift: '+name);checks+=1
    inherited=previous['admission']()
    return {'format':'switch-engine-admission-1','predecessor_hash_checks':checks,
        'inherited_runtime':inherited['inherited_runtime'],'runtime':checkpoint.runtime(),'source_files':checkpoint.sources()}


def audit_learning(directory):
    folder=directory/'learner';report=read(folder/'report.json');require(report['status']=='complete' and report['plan']==learner.PLAN,'Learner completion/plan')
    for key in ('public_server_used','final_evaluation_used','strength_evaluated','prime_used'):require(report[key] is False,'Scope')
    require(report['parity_sha256']==file_hash(directory/'reference/parity.json'),'Parity before learner binding')
    require([r['seed'] for r in report['seeds']]==learner.CONFIG['seeds'],'Seed grid')
    totals={'updates':0,'games':0,'decisions':0,'terminal':0,'truncated':0,'original_next_update':0,'rollback':0,'checkpoints':0}
    for run in report['seeds']:
        seed=run['seed'];require(run['file']==f'seed-{seed}/report.json' and file_hash(folder/run['file'])==run['sha256'],'Seed evidence')
        data=read(folder/run['file']);base=folder/f'seed-{seed}';rows=data['records']
        require(data['seed']==seed and len(rows)==64,'Fixed update budget')
        prior=checkpoint.fingerprint(learner.start(seed))
        for i,row in enumerate(rows,1):
            require(row['update']==i and row['optimizer_step']==2*i and row['next_game']==32*i and row['before']==prior,'Update counter/fingerprint chain')
            require(type(row['decisions']) is int and 0<row['decisions']<=896 and sum(row['actions'])==row['decisions'],'Decision accounting')
            require(row['terminal']+row['truncated']==32 and min(row['terminal'],row['truncated'],*row['actions'])>=0,'Game accounting')
            metrics=row['metrics'];require(set(metrics)=={'loss','actor_loss','value_loss','entropy','clip_fraction','approx_kl','gradient_norm'},'Metrics schema')
            require(all(np.asarray(v).shape==(2,) and np.isfinite(v).all() for v in metrics.values()),'Finite epoch metrics')
            require(checkpoint.fingerprint({k:np.asarray(v,np.float32) for k,v in metrics.items()})==row['metrics_sha256'],'Metrics fingerprint')
            prior=row['after'];totals['updates']+=1;totals['games']+=32
            for key in ('decisions','terminal','truncated'):totals[key]+=row[key]
        require([c['update'] for c in data['checkpoints']]==[0,32,64],'Checkpoint grid')
        for c in data['checkpoints']:
            i=c['update'];require(c['file']==f'checkpoint-{i:03d}.json' and file_hash(base/c['file'])==c['sha256'],'Checkpoint file')
            state=checkpoint.load(base/c['file'],seed)
            require(int(state.version)==i and checkpoint.fingerprint(state)==c['state']==(rows[i]['before'] if i<64 else rows[-1]['after']),'Checkpoint boundary')
            totals['checkpoints']+=1
            if i<64:
                repeated=record(state,checkpoint.transaction(state,seed));require(repeated==rows[i],'Original-next-update fresh replay')
                require(data['original_next_update'][i//32]=={'checkpoint_update':i,'original_next_update':rows[i],'status':'exact'},'Saved replay audit')
                totals['original_next_update']+=1
        require(len(data['original_next_update'])==2 and len(data['rollback'])==2,'Audit count')
        for r,stage in zip(data['rollback'],('collect','update')):
            require(r=={'update':32,'stage':stage,'state':rows[32]['before']},'Rollback record')
            state=checkpoint.load(base/'checkpoint-032.json',seed);h=checkpoint.fingerprint(state)
            try:checkpoint.transaction(state,seed,fault=stage)
            except RuntimeError as error:require('Injected' in str(error),'Rollback error')
            else:raise ValueError('Expected rollback failure')
            require(checkpoint.fingerprint(state)==h,'Rollback mutation');totals['rollback']+=1
    require(report['updates']==totals['updates']==192 and report['training_games']==totals['games']==6144
            and report['decisions']==totals['decisions'] and report['original_next_update_audits']==totals['original_next_update']==6
            and report['rollback_audits']==totals['rollback']==6 and report['checkpoint_load_audits']==totals['checkpoints']==9,'Learner totals')
    return totals


def seal(directory):
    directory=Path(directory).resolve();require(directory.is_relative_to(ROOT/'artifacts/runs'),'Run path')
    now=admission();require(not (directory/'failure.json').exists(),'Failed run cannot seal')
    registration=read(directory/'preregistration.json');report=read(directory/'report.json')
    require(registration=={'plan':learner.PLAN,'admission':now} and report['admission']==now and report['status']=='complete','Frozen source/runtime')
    require(report['parity_sha256']==file_hash(directory/'reference/parity.json') and report['learner_sha256']==file_hash(directory/'learner/report.json'),'Report binding')
    parity_report=parity.audit(directory/'reference',write=False)
    require(read(directory/'reference/plan.json')['smoke'] is False and parity_report==read(directory/'reference/parity.json') and parity_report['fresh']['games']==10000,'Fresh complete parity')
    oracle=json.loads(subprocess.check_output(['node','packages/switch-engine/audit.mjs',str(directory/'reference')],cwd=ROOT))
    require(oracle==read(directory/'reference/oracle-audit.json') and oracle['status']=='pass' and oracle['unpatched_games']==24,'Fresh oracle/passivity audit')
    totals=audit_learning(directory)
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] and v['source_files']==now['source_files']
            and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Source-bound regression')
    node=re.search(r'ℹ pass (\d+)',log);suites=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node and int(node[1])==237 and suites==[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17] and log.count('\nOK\n')==len(suites),'Regression coverage')
    for name in ('mechanics-review.md','learner-review.md','campaign-review.md','review-dispositions.md','corrected-verification.md'):
        require((REVIEW/name).is_file(),'Missing review: '+name)
    sources={**now['source_files'],**{str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]}}
    evidence={f'artifacts/reports/{name}.json':file_hash(ROOT/f'artifacts/reports/{name}.json') for name in ('showdown-soak','m3-switch-reference')}
    for folder in (directory,REVIEW):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file()})
    receipt={'format':'switch-engine-receipt-1','status':'complete-restricted-engineering-smoke','run':str(directory.relative_to(ROOT)),
        'predecessor':learner.PLAN['predecessor_receipt'],'reference_receipt':learner.PLAN['reference_receipt'],'parity':parity_report,'learner':totals,
        'tests':{'node':int(node[1]),'python':sum(suites)},'source_files':sources,'evidence_files':evidence,
        'full_g2':'open','full_g3':'open','full_g4':'open','public_ladder_strength':'unqualified','sustained_qualification':'pending',
        'final_evaluation_used':False,'prime_used':False}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m3-switch-engine-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for name,h in files.items():
            require(file_hash(ROOT/name)==h,'Live drift');write_once(archive/kind/name,(ROOT/name).read_bytes());require(file_hash(archive/kind/name)==h,'Archive drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m3-switch-engine.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'hash':receipt['content_hash'],'parity':parity_report['fresh'],'learner':totals,'tests':receipt['tests']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?');p.add_argument('--admission',action='store_true');args=p.parse_args()
    if args.admission:print(json.dumps(admission(),sort_keys=True))
    elif args.directory:seal(args.directory)
    else:p.error('Provide directory or --admission')
